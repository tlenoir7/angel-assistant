"""
Async OpenAI GPT-4o Realtime API bridge for Socket.IO (web_app_fastapi).

Uses asyncio + websockets to talk to wss://api.openai.com/v1/realtime while emitting
events to the default Socket.IO namespace.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import UTC, datetime
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# --- Module-level session registry (keyed by Socket.IO sid) ---
_realtime_sessions: dict[str, dict[str, Any]] = {}
_realtime_tasks: dict[str, asyncio.Task] = {}
_realtime_timeout_tasks: dict[str, asyncio.Task] = {}

OPENAI_BETA = "realtime=v1"
REALTIME_MODEL = os.environ.get(
    "OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17"
).strip()
REALTIME_WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
SESSION_TTL_SEC = 600  # 10 minutes


def _realtime_api_key() -> str | None:
    k = (os.environ.get("OPENAI_REALTIME_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    return k or None


def build_angel_realtime_instructions() -> str:
    """Compact Realtime instructions (<~1500 tokens target). No markdown."""
    now = datetime.now(UTC).strftime("%A, %Y-%m-%d %H:%M UTC")
    return f"""You are Angel, Tyler's personal intelligence system. You are speaking in live voice on Tyler's iPhone: be warm, conversational, concise, and natural—like a trusted partner talking in person. Do not use markdown, bullet lists, or headings. Say dates and names plainly when needed.

Tyler is building toward becoming an FBI Special Agent. Your core mission is to help Tyler discover truth, protect people, and advance that mission with integrity.

Current date and time (UTC): {now}.

Speak as Angel: capable, steady, and on Tyler's side. Keep replies short enough for voice unless Tyler asks for detail."""


def _session_update_payload(instructions: str) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": instructions,
            "voice": "alloy",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            },
            "temperature": 0.8,
            "max_response_output_tokens": 1024,
        },
    }


async def _send_json(ws: Any, obj: dict, lock: asyncio.Lock) -> None:
    payload = json.dumps(obj)
    async with lock:
        await ws.send(payload)


async def _reader_loop(
    *,
    ws: Any,
    sio: Any,
    sid: str,
    angel: Any,
    send_lock: asyncio.Lock,
    instructions: str,
    session: dict[str, Any],
) -> None:
    """Read OpenAI Realtime events; emit to Socket.IO client."""
    session_ready = False

    try:
        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")

            if t == "session.created":
                await _send_json(ws, _session_update_payload(instructions), send_lock)
                continue

            if t == "session.updated":
                if not session_ready:
                    session_ready = True
                    await sio.emit("realtime_ready", {"ok": True}, room=sid)
                continue

            if t == "response.audio.delta":
                delta = msg.get("delta") or (msg.get("response") or {}).get("delta") or ""
                if delta:
                    await sio.emit("realtime_audio_response", {"audio_b64": delta}, room=sid)
                continue

            if t == "response.audio.done":
                await sio.emit("realtime_audio_done", {}, room=sid)
                continue

            if t == "response.audio_transcript.delta":
                d = msg.get("delta") or ""
                if d:
                    await sio.emit(
                        "realtime_transcript",
                        {"delta": d, "done": False, "role": "assistant"},
                        room=sid,
                    )
                continue

            if t == "response.audio_transcript.done":
                transcript = _sanitize_transcript(str(msg.get("transcript", "")))
                await sio.emit(
                    "realtime_transcript",
                    {"transcript": transcript, "done": True, "role": "assistant"},
                    room=sid,
                )
                user_part = (session.get("last_user_transcript") or "").strip()
                if transcript and user_part:
                    try:
                        await asyncio.to_thread(
                            angel.add_conversation_turn,
                            user_part,
                            transcript,
                        )
                    except Exception as e:
                        print(f"[realtime] add_conversation_turn failed: {e}", flush=True)
                continue

            if t == "conversation.item.input_audio_transcription.completed":
                item = msg.get("item") or {}
                tr = msg.get("transcript") or item.get("transcript")
                if not tr and isinstance(item.get("content"), list):
                    for c in item.get("content") or []:
                        if isinstance(c, dict) and c.get("transcript"):
                            tr = c.get("transcript")
                            break
                if tr:
                    session["last_user_transcript"] = str(tr).strip()
                continue

            if t == "input_audio_buffer.speech_started":
                await sio.emit("realtime_speech_detected", {}, room=sid)
                continue

            if t == "input_audio_buffer.speech_stopped":
                await sio.emit("realtime_processing", {}, room=sid)
                continue

            if t == "error":
                err = msg.get("error") or {}
                txt = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                await sio.emit("realtime_error", {"message": str(txt)}, room=sid)
                continue

    except (ConnectionClosed, WebSocketException) as e:
        print(f"[realtime] reader closed sid={sid!s}: {e}", flush=True)
        await sio.emit("realtime_error", {"message": "OpenAI Realtime connection closed"}, room=sid)
    except Exception as e:
        traceback.print_exc()
        await sio.emit("realtime_error", {"message": str(e)}, room=sid)
    finally:
        await _cleanup_sid(sid, sio, emit_ended=True, skip_reader_task=True)


def _sanitize_transcript(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


async def _cleanup_sid(
    sid: str,
    sio: Any,
    *,
    emit_ended: bool = False,
    skip_reader_task: bool = False,
) -> None:
    st = _realtime_sessions.pop(sid, None)
    t = _realtime_tasks.pop(sid, None)
    tt = _realtime_timeout_tasks.pop(sid, None)

    if tt and not tt.done():
        tt.cancel()
        try:
            await tt
        except asyncio.CancelledError:
            pass

    # When cleanup runs from _reader_loop's finally, the reader task is still running
    # this coroutine — do not cancel/await it.
    if t and not t.done() and not skip_reader_task:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    if st:
        ws = st.get("ws")
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    if emit_ended:
        try:
            await sio.emit("realtime_ended", {"ok": True}, room=sid)
        except Exception:
            pass


async def _timeout_worker(sid: str, sio: Any) -> None:
    try:
        await asyncio.sleep(SESSION_TTL_SEC)
        await sio.emit(
            "realtime_error",
            {"message": "Realtime session timed out (10 minutes)."},
            room=sid,
        )
        await _cleanup_sid(sid, sio, emit_ended=True)
    except asyncio.CancelledError:
        pass


async def handle_realtime_start(sio: Any, sid: str, angel: Any, data: Any) -> None:
    if _realtime_api_key() is None:
        await sio.emit("realtime_error", {"message": "Realtime not configured"}, room=sid)
        return

    await _cleanup_sid(sid, sio, emit_ended=False)

    instructions = build_angel_realtime_instructions()
    headers = [
        ("Authorization", f"Bearer {_realtime_api_key()}"),
        ("OpenAI-Beta", OPENAI_BETA),
    ]

    try:
        ws = await websockets.connect(
            REALTIME_WS_URL,
            additional_headers=headers,
            max_size=None,
        )
    except Exception as e:
        traceback.print_exc()
        await sio.emit("realtime_error", {"message": f"Realtime connect failed: {e}"}, room=sid)
        return

    send_lock = asyncio.Lock()
    sess: dict[str, Any] = {
        "ws": ws,
        "send_lock": send_lock,
        "instructions": instructions,
        "last_user_transcript": "",
    }
    _realtime_sessions[sid] = sess

    reader = asyncio.create_task(
        _reader_loop(
            ws=ws,
            sio=sio,
            sid=sid,
            angel=angel,
            send_lock=send_lock,
            instructions=instructions,
            session=sess,
        )
    )
    _realtime_tasks[sid] = reader

    to_task = asyncio.create_task(_timeout_worker(sid, sio))
    _realtime_timeout_tasks[sid] = to_task


async def handle_realtime_audio_chunk(sio: Any, sid: str, data: Any) -> None:
    st = _realtime_sessions.get(sid)
    if not st:
        return
    payload = data if isinstance(data, dict) else {}
    b64 = payload.get("audio_b64") or payload.get("audio") or ""
    if not isinstance(b64, str) or not b64.strip():
        return
    ws = st.get("ws")
    lock = st.get("send_lock")
    if ws is None or lock is None:
        return
    try:
        await _send_json(
            ws,
            {"type": "input_audio_buffer.append", "audio": b64.strip()},
            lock,
        )
    except Exception:
        traceback.print_exc()


async def handle_realtime_text(sio: Any, sid: str, angel: Any, data: Any) -> None:
    st = _realtime_sessions.get(sid)
    if not st:
        await sio.emit("realtime_error", {"message": "No active Realtime session."}, room=sid)
        return
    payload = data if isinstance(data, dict) else {}
    text = (payload.get("text") or "").strip()
    if not text:
        await sio.emit("realtime_error", {"message": "Empty text."}, room=sid)
        return
    ws = st.get("ws")
    lock = st.get("send_lock")
    if ws is None or lock is None:
        return

    st["last_user_transcript"] = text

    try:
        await _send_json(
            ws,
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            },
            lock,
        )
        await _send_json(
            ws,
            {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "max_output_tokens": 1024,
                },
            },
            lock,
        )
    except Exception as e:
        traceback.print_exc()
        await sio.emit("realtime_error", {"message": str(e)}, room=sid)


async def handle_realtime_stop(sio: Any, sid: str) -> None:
    await _cleanup_sid(sid, sio, emit_ended=True)


async def on_socket_disconnect(sid: str, sio: Any) -> None:
    if sid in _realtime_sessions:
        await _cleanup_sid(sid, sio, emit_ended=True)
