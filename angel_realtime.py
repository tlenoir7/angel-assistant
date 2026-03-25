"""
GPT-4o Realtime API client for Angel's voice pipeline.
Maintains a persistent WebSocket connection for the entire Angel session.
"""

import base64
import io
import json
import os
import threading
import time
import wave
import struct
from collections.abc import Callable

import websockets
from websockets.sync.client import connect as ws_connect

try:
    import audioop
except ImportError:
    audioop = None

OPENAI_BETA_HEADER = "realtime=v1"


def _realtime_verbose() -> bool:
    return (os.getenv("ANGEL_VERBOSE_STDIO") or "").strip().lower() in ("1", "true", "yes")


def _realtime_ws_url() -> str:
    model = (os.environ.get("OPENAI_REALTIME_MODEL") or "gpt-realtime").strip()
    return f"wss://api.openai.com/v1/realtime?model={model}"


def _wav_to_pcm16_24k(wav_bytes: bytes) -> bytes:
    """Convert WAV audio bytes to exactly PCM 16-bit signed little-endian 24000 Hz mono."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wav:
        nchannels, sampwidth, framerate, nframes, *_ = wav.getparams()
        frames = wav.readframes(nframes)
    if sampwidth != 2:
        raise ValueError("Only 16-bit WAV is supported (sampwidth=2)")
    if nchannels == 2:
        if audioop is not None:
            frames = audioop.tomono(frames, 2, 0.5, 0.5)
        else:
            count = len(frames) // 2
            samples = list(struct.unpack(f"<{count}h", frames))
            samples = [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)]
            frames = struct.pack(f"<{len(samples)}h", *samples)
        nchannels = 1
    if framerate != 24000:
        if audioop is not None:
            frames, _ = audioop.ratecv(frames, 2, nchannels, framerate, 24000)
        else:
            count = len(frames) // 2
            samples = list(struct.unpack(f"<{count}h", frames))
            new_len = int(len(samples) * 24000 / framerate)
            new_samples = []
            for i in range(new_len):
                pos = i * framerate / 24000
                i0 = min(int(pos), len(samples) - 1)
                i1 = min(i0 + 1, len(samples) - 1)
                frac = pos - int(pos)
                s = int(samples[i0] * (1 - frac) + samples[i1] * frac)
                new_samples.append(max(-32768, min(32767, s)))
            frames = struct.pack(f"<{len(new_samples)}h", *new_samples)
    return frames


def _get_headers() -> dict:
    # Prefer OPENAI_REALTIME_API_KEY for Realtime; fall back to OPENAI_API_KEY.
    api_key = os.environ.get("OPENAI_REALTIME_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_REALTIME_API_KEY (or OPENAI_API_KEY) environment variable is not set"
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": OPENAI_BETA_HEADER,
    }


class AngelRealtimeSession:
    """
    Persistent Realtime API session. One websocket stays open for the entire Angel session.

    Two usage modes:
    - **Blocking turn** (desktop GUI): ``send_audio()`` appends, commits, receives until done.
    - **Streaming proxy**: start a receiver thread with ``start_receiver_thread``, then use
      ``append_input_audio_base64`` / ``commit_input_buffer`` / ``create_audio_response``.
    """

    def __init__(self) -> None:
        self._ws = None
        self._send_lock = threading.Lock()
        self._receiver_stop = threading.Event()
        self._receiver_thread: threading.Thread | None = None

    def _send_json(self, obj: dict) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime session not connected")
        payload = json.dumps(obj)
        with self._send_lock:
            self._ws.send(payload)

    def connect(self, system_prompt: str) -> None:
        """
        Open the websocket, send session.update with the system prompt once,
        wait for session.updated confirmation, then return.
        """
        if self._ws is not None:
            self.disconnect()
        self._receiver_stop.clear()
        self._ws = ws_connect(_realtime_ws_url(), additional_headers=_get_headers())
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": system_prompt,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": None,
            },
        }
        self._send_json(session_config)
        while True:
            msg = json.loads(self._ws.recv())
            if msg.get("type") == "session.updated":
                break
            if msg.get("type") == "error":
                self._ws.close()
                self._ws = None
                raise RuntimeError(msg.get("error", {}).get("message", "Session update failed"))

    def start_receiver_thread(self, on_message: Callable[[dict], None]) -> None:
        """
        Background thread: forward every server JSON event to ``on_message``.
        Used by the Railway Socket.IO proxy; do not use together with ``send_audio`` on the same session.
        """
        self.stop_receiver_thread()
        self._receiver_stop.clear()

        def _runner() -> None:
            while not self._receiver_stop.is_set() and self._ws is not None:
                try:
                    raw = self._ws.recv()
                except Exception:
                    if not self._receiver_stop.is_set():
                        try:
                            on_message({"type": "_realtime_socket_closed"})
                        except Exception:
                            pass
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                try:
                    on_message(msg)
                except Exception:
                    pass

        self._receiver_thread = threading.Thread(target=_runner, daemon=True)
        self._receiver_thread.start()

    def stop_receiver_thread(self) -> None:
        self._receiver_stop.set()
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=3.0)
            self._receiver_thread = None

    def append_input_audio_base64(self, audio_b64: str) -> None:
        """Append one PCM16 little-endian mono 24 kHz chunk (base64) to the input buffer."""
        self._send_json({"type": "input_audio_buffer.append", "audio": audio_b64})

    def append_input_pcm16_24k(self, pcm_bytes: bytes, chunk_size: int = 8192) -> None:
        """Split raw PCM16 LE mono 24 kHz bytes into append events (no artificial delays)."""
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            b64 = base64.standard_b64encode(chunk).decode("ascii")
            self.append_input_audio_base64(b64)

    def append_input_wav_bytes(self, wav_bytes: bytes, chunk_size: int = 8192) -> None:
        """Decode a WAV blob to PCM16 24kHz mono and append to the input buffer."""
        pcm_bytes = _wav_to_pcm16_24k(wav_bytes)
        self.append_input_pcm16_24k(pcm_bytes, chunk_size=chunk_size)

    def commit_input_buffer(self) -> None:
        self._send_json({"type": "input_audio_buffer.commit"})

    def create_audio_response(self, max_output_tokens: int = 4096) -> None:
        self._send_json(
            {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    "max_output_tokens": max_output_tokens,
                },
            }
        )

    def send_audio(self, wav_bytes: bytes) -> tuple[str, bytes]:
        """
        Convert WAV to PCM16 24kHz, send via input_audio_buffer.append, commit,
        send response.create, collect audio deltas and transcript until response.done.
        Returns (transcript, raw_pcm_bytes).
        """
        if self._ws is None:
            raise RuntimeError("Realtime session not connected")

        # Drain any pending server messages so we start with a clear buffer
        drain_types: list[str] = []
        while True:
            try:
                raw = self._ws.recv(timeout=0.01)
                msg = json.loads(raw)
                drain_types.append(str(msg.get("type")))
            except TimeoutError:
                break
        if drain_types:
            tail = drain_types[:10]
            more = f" (+{len(drain_types) - 10} more)" if len(drain_types) > 10 else ""
            print(f"[Realtime] drain: n={len(drain_types)} types={tail}{more}", flush=True)

        pcm_bytes = _wav_to_pcm16_24k(wav_bytes)

        MIN_PCM_BYTES = 2400  # 100ms at 24kHz 16-bit mono (24000 * 0.1 * 2)
        if not pcm_bytes or len(pcm_bytes) < MIN_PCM_BYTES:
            raise ValueError(
                f"Input audio too short: got {len(pcm_bytes)} PCM bytes "
                f"(need at least {MIN_PCM_BYTES} bytes, ~100ms at 24kHz). "
                "Cannot commit empty or very short buffer."
            )

        chunk_size = 4096
        total_sent = 0
        n_chunks = 0
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            n = len(chunk)
            if _realtime_verbose():
                print(f"[Realtime] send_audio chunk n={n}")
            chunk_b64 = base64.standard_b64encode(chunk).decode("ascii")
            self._send_json({"type": "input_audio_buffer.append", "audio": chunk_b64})
            total_sent += n
            n_chunks += 1
            time.sleep(0.1)
        print(
            f"[Realtime] send_audio: wav_in={len(wav_bytes)} pcm={len(pcm_bytes)} "
            f"chunks={n_chunks} bytes_sent={total_sent}",
            flush=True,
        )

        time.sleep(0.5)
        self._send_json({"type": "input_audio_buffer.commit"})
        time.sleep(0.3)
        self._send_json(
            {
                "type": "response.create",
                "response": {"modalities": ["audio", "text"], "max_output_tokens": 500},
            }
        )

        transcript_parts: list[str] = []
        audio_chunks: list[bytes] = []
        evt_counts: dict[str, int] = {}
        while True:
            raw = self._ws.recv()
            msg = json.loads(raw)
            t = msg.get("type")
            if isinstance(t, str):
                evt_counts[t] = evt_counts.get(t, 0) + 1

            # Audio: response.audio.delta (current API event name)
            if t == "response.audio.delta":
                delta_b64 = msg.get("delta", "")
                if delta_b64:
                    audio_chunks.append(base64.standard_b64decode(delta_b64))
            elif t == "response.audio_transcript.delta":
                transcript_parts.append(msg.get("delta", ""))
            elif t == "response.audio_transcript.done":
                transcript_parts.append(msg.get("transcript", ""))
            elif t == "response.error":
                err_blob = json.dumps(msg, ensure_ascii=False)
                tail = err_blob if _realtime_verbose() else (err_blob[:400] + ("…" if len(err_blob) > 400 else ""))
                print(f"[Realtime] response.error: {tail}", flush=True)
            elif t == "response.done":
                status = (msg.get("response") or {}).get("status", "")
                if status not in ("completed", "incomplete"):
                    raise RuntimeError(f"Response ended with status: {status}")
                break
            elif t == "error":
                raise RuntimeError(msg.get("error", {}).get("message", "Realtime API error"))

        transcript = "".join(transcript_parts).strip()
        audio_bytes = b"".join(audio_chunks)
        keys = sorted(evt_counts.keys())
        summary = ",".join(f"{k}:{evt_counts[k]}" for k in keys[:12])
        if len(keys) > 12:
            summary += f",…(+{len(keys) - 12} types)"
        print(
            f"[Realtime] response stream: events={sum(evt_counts.values())} types=[{summary}]",
            flush=True,
        )
        return transcript, audio_bytes

    def disconnect(self) -> None:
        """Close the websocket connection."""
        self.stop_receiver_thread()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
