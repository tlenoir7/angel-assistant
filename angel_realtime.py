"""
GPT-4o Realtime API client for Angel's voice pipeline.
Maintains a persistent WebSocket connection for the entire Angel session.
"""

import base64
import io
import json
import os
import time
import wave
import struct

import websockets
from websockets.sync.client import connect as ws_connect

try:
    import audioop
except ImportError:
    audioop = None

REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
OPENAI_BETA_HEADER = "realtime=v1"


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
    api_key = os.environ.get("OPENAI_REALTIME_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_REALTIME_API_KEY or OPENAI_API_KEY environment variable is not set")
    return {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": OPENAI_BETA_HEADER,
    }


class AngelRealtimeSession:
    """
    Persistent Realtime API session. One websocket stays open for the entire Angel session.
    """

    def __init__(self) -> None:
        self._ws = None

    def connect(self, system_prompt: str) -> None:
        """
        Open the websocket, send session.update with the system prompt once,
        wait for session.updated confirmation, then return.
        """
        if self._ws is not None:
            self.disconnect()
        self._ws = ws_connect(REALTIME_URL, additional_headers=_get_headers())
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
        self._ws.send(json.dumps(session_config))
        while True:
            msg = json.loads(self._ws.recv())
            if msg.get("type") == "session.updated":
                break
            if msg.get("type") == "error":
                self._ws.close()
                self._ws = None
                raise RuntimeError(msg.get("error", {}).get("message", "Session update failed"))

    def send_audio(self, wav_bytes: bytes) -> tuple[str, bytes]:
        """
        Convert WAV to PCM16 24kHz, send via input_audio_buffer.append, commit,
        send response.create, collect audio deltas and transcript until response.done.
        Returns (transcript, raw_pcm_bytes).
        """
        if self._ws is None:
            raise RuntimeError("Realtime session not connected")

        # Drain any pending server messages so we start with a clear buffer
        while True:
            try:
                raw = self._ws.recv(timeout=0.01)
                msg = json.loads(raw)
                print(f"[Realtime] drain: unexpected message type = {msg.get('type')!r}", msg)
            except TimeoutError:
                break

        print(f"[Realtime] send_audio: wav_bytes length = {len(wav_bytes)}")
        pcm_bytes = _wav_to_pcm16_24k(wav_bytes)
        print(f"[Realtime] send_audio: pcm_bytes length after conversion = {len(pcm_bytes)}")

        MIN_PCM_BYTES = 2400  # 100ms at 24kHz 16-bit mono (24000 * 0.1 * 2)
        if not pcm_bytes or len(pcm_bytes) < MIN_PCM_BYTES:
            raise ValueError(
                f"Input audio too short: got {len(pcm_bytes)} PCM bytes "
                f"(need at least {MIN_PCM_BYTES} bytes, ~100ms at 24kHz). "
                "Cannot commit empty or very short buffer."
            )

        chunk_size = 4096
        total_sent = 0
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            n = len(chunk)
            print(f"[Realtime] send_audio: sending chunk size = {n}")
            chunk_b64 = base64.standard_b64encode(chunk).decode("ascii")
            self._ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk_b64}))
            total_sent += n
            time.sleep(0.1)
        print(f"[Realtime] send_audio: total bytes sent before commit = {total_sent}")

        time.sleep(0.5)
        self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        time.sleep(0.3)
        self._ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"], "max_output_tokens": 500},
        }))

        transcript_parts: list[str] = []
        audio_chunks: list[bytes] = []
        while True:
            raw = self._ws.recv()
            msg = json.loads(raw)
            t = msg.get("type")
            print(f"[Realtime] response event: type={t!r}")

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
                print("[Realtime] response.error event:", json.dumps(msg, indent=2))
            elif t == "response.done":
                status = (msg.get("response") or {}).get("status", "")
                if status not in ("completed", "incomplete"):
                    raise RuntimeError(f"Response ended with status: {status}")
                break
            elif t == "error":
                raise RuntimeError(msg.get("error", {}).get("message", "Realtime API error"))

        transcript = "".join(transcript_parts).strip()
        audio_bytes = b"".join(audio_chunks)
        return transcript, audio_bytes

    def disconnect(self) -> None:
        """Close the websocket connection."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
