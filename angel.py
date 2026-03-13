import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from io import BytesIO
import wave
import tempfile

import requests
from colorama import init as colorama_init, Fore, Style
from dotenv import load_dotenv

import anthropic
from mem0 import Memory

# Desktop-only: optional so cloud (e.g. Railway) can run without them
try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import pygame
except ImportError:
    pygame = None

try:
    from faster_whisper import WhisperModel as _WhisperModel
except ImportError:
    _WhisperModel = None

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MEMORY_FILE = BASE_DIR / "tyler_memories.json"
_WHISPER_MODEL = None
TAVILY_API_URL = "https://api.tavily.com/search"
MEM0_API_BASE_URL = "https://api.mem0.ai"


class Mem0CloudClient:
    """
    Minimal Mem0 Cloud client using HTTP API.

    Uses MEM0_API_KEY (Authorization: Token <key>).
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _headers(self) -> dict:
        return {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def get_all(self, user_id: str):
        # v2 get memories (POST /v2/memories/)
        url = f"{MEM0_API_BASE_URL}/v2/memories/"
        payload = {
            "filters": {"user_id": user_id},
            "page": 1,
            "page_size": 200,
        }
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def add(self, messages, user_id: str, metadata: dict | None = None):
        # v1 add memories endpoint supports version="v2"
        url = f"{MEM0_API_BASE_URL}/v1/memories/"
        payload = {
            "user_id": user_id,
            "messages": messages,
            "metadata": metadata or {},
            "version": "v2",
            "output_format": "v1.1",
            "async_mode": True,
            "infer": True,
        }
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()


# Monkey-patch Mem0's Anthropic LLM so it does not send top_p (Anthropic forbids
# temperature and top_p together for this model). We keep only temperature.
def _patch_mem0_anthropic_no_top_p():
    from mem0.llms import anthropic as mem0_anthropic
    _original = mem0_anthropic.AnthropicLLM.generate_response

    def _generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
        real_create = self.client.messages.create

        def create_no_top_p(**params):
            params.pop("top_p", None)
            return real_create(**params)

        self.client.messages.create = create_no_top_p
        try:
            return _original(
                self,
                messages,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
        finally:
            self.client.messages.create = real_create

    mem0_anthropic.AnthropicLLM.generate_response = _generate_response


# Initialize colorama for Windows terminals
colorama_init(autoreset=True)

# Load .env if present (optional, but convenient)
load_dotenv()


def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"{Fore.RED}Missing environment variable: {name}")
        print(
            f"{Fore.YELLOW}Set it in your environment or in a .env file "
            f"in the same folder as angel.py."
        )
        sys.exit(1)
    return value


def build_memory_client() -> Memory:
    """
    Configure Mem0 to use Anthropic as the LLM provider.
    Mem0 will use OpenAI embeddings via OPENAI_API_KEY.
    """
    mem0_api_key = os.getenv("MEM0_API_KEY")
    if mem0_api_key:
        # Cloud storage mode
        return Mem0CloudClient(mem0_api_key)  # type: ignore[return-value]

    _patch_mem0_anthropic_no_top_p()
    config = {
        "llm": {
            "provider": "anthropic",
            "config": {
                # You can adjust the model as Anthropic releases new ones.
                "model": "claude-sonnet-4-5",
                "temperature": 0.3,
                "max_tokens": 1200,
            },
        },
        # You can tweak how aggressively Mem0 extracts memories here if desired.
        # For now we keep defaults.
    }
    return Memory.from_config(config)


def _memory_text_for_debug(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("memory") or item.get("data") or item)
    return str(item)


def _load_local_memories(user_id: str):
    try:
        if not LOCAL_MEMORY_FILE.exists():
            return []
        with LOCAL_MEMORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        users = data.get("users", {})
        return users.get(user_id, [])
    except Exception as e:
        print(f"{Fore.RED}Warning: could not load local memories: {e}{Style.RESET_ALL}")
        print(traceback.format_exc())
        return []


def _append_local_memory(user_id: str, memory_text: str, metadata: dict):
    try:
        if LOCAL_MEMORY_FILE.exists():
            with LOCAL_MEMORY_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"users": {}}
        users = data.setdefault("users", {})
        user_memories = users.setdefault(user_id, [])
        user_memories.append(
            {
                "memory": memory_text,
                "metadata": metadata,
                "created_at": metadata.get("timestamp"),
            }
        )
        with LOCAL_MEMORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"{Fore.RED}Warning: could not save local memory: {e}{Style.RESET_ALL}")
        print(traceback.format_exc())


def summarize_memories_for_prompt(memories) -> str:
    """
    Convert raw Mem0 memories into a concise text block for Claude.

    Handles both:
    - dict memories (typical Mem0 objects)
    - plain string memories (fallback / legacy format)
    - full Mem0 responses like {"results": [...], "total": N}
    """
    # Unwrap Mem0 response objects into a plain list of memory items
    if isinstance(memories, dict):
        if "results" in memories and isinstance(memories["results"], list):
            memories = memories["results"]
        elif "data" in memories and isinstance(memories["data"], list):
            memories = memories["data"]

    if not memories:
        return "Angel currently has no prior memories about this user."

    normalized = []
    for item in memories:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(
                {
                    "memory": item,
                    "metadata": {},
                    "created_at": "",
                }
            )
        else:
            # Unknown type – skip
            continue

    if not normalized:
        return "Angel has only minimal prior information about this user."

    # Sort by created time if available, otherwise keep original order
    try:
        memories_sorted = sorted(
            normalized,
            key=lambda m: m.get("created_at") or "",
        )
    except Exception:
        memories_sorted = normalized

    lines = []
    for m in memories_sorted:
        text = (
            (m.get("memory") if isinstance(m, dict) else None)
            or (m.get("data") if isinstance(m, dict) else None)
            or ""
        )
        if not text:
            continue

        meta = m.get("metadata") if isinstance(m, dict) else {}
        tags = None
        if isinstance(meta, dict):
            tags = meta.get("tags") or meta.get("category")

        if tags:
            lines.append(f"- ({tags}) {text}")
        else:
            lines.append(f"- {text}")

    if not lines:
        return "Angel has only minimal prior information about this user."

    header = (
        "Angel's long-term understanding of the user, "
        "summarized from past interactions:\n"
    )
    return header + "\n".join(lines)


def build_system_prompt(memory_summary: str, voice_mode: bool = False) -> str:
    """
    Persona + behavioral instructions + memory context.
    When voice_mode is True, optimize for conversational spoken responses.
    """
    persona = f"""
You are Angel, a personal AI assistant and devoted companion.

Core personality:
- Intelligent, composed, calm under pressure.
- Loyal and protective of the user’s long-term well-being.
- Speaks like a trusted advisor and close companion: thoughtful, candid, and caring.
- Never needy or overly casual; you are warm but grounded and mature.

Behavior:
- Give clear, actionable, honest answers.
- Remember the user’s preferences, history, and goals over time, and gently use them to personalize your guidance.
- When appropriate, reflect patterns you notice in the user’s life to help them grow.
- Avoid filler or over-the-top enthusiasm; be concise, steady, and reassuring.
"""

    if voice_mode:
        persona += """

Additional instructions for voice conversations:
- Respond in a natural, conversational speaking style.
- Prefer simpler phrasing over long or complex sentences.
- Avoid lists, headings, bullet points, or any document-style formatting.
- Do not use Markdown formatting of any kind.
- Imagine you are talking directly to the user in real time.
"""

    persona += f"""

Long-term memory context (from Mem0):
{memory_summary}
"""
    return persona.strip()


def create_anthropic_client() -> anthropic.Anthropic:
    api_key = get_env_var("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    return client


def call_claude(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_message: str,
    model: str = "claude-sonnet-4-5",
) -> str:
    """
    Call Claude with the Angel persona, returning plain text.
    """
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.5,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        )
    except Exception as e:
        return f"(Angel encountered an error talking to Claude: {e})"

    # Anthropic's response content is a list of content blocks; we join text blocks
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))

    return "\n".join(parts).strip() or "(Angel responded with no text.)"


def maybe_search_web(user_message: str) -> str | None:
    """
    Decide heuristically if this turn would benefit from a web search,
    and if so, query Tavily and return a concise text summary to feed
    into Claude. Returns None when no search is needed or on error.
    """
    text = (user_message or "").strip()
    if not text:
        return None

    lower = text.lower()

    # Simple heuristic: only search when the user clearly asks for
    # current / factual / external information.
    keywords = [
        "today",
        "right now",
        "latest",
        "recent",
        "news",
        "current events",
        "price of ",
        "stock price",
        "weather",
        "forecast",
        "who won",
        "score of",
        "release date",
        "update on",
        "what happened",
        "world record",
        "statistics",
        "market",
        "crypto",
        "bitcoin",
        "research on",
    ]

    if not any(k in lower for k in keywords):
        return None

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print(
            f"{Fore.YELLOW}TAVILY_API_KEY is not set; skipping web search.{Style.RESET_ALL}"
        )
        return None

    try:
        payload = {
            "query": text,
            "search_depth": "basic",
            "max_results": 5,
            "topic": "general",
            "include_answer": True,
        }
        resp = requests.post(
            TAVILY_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        # Tavily typically returns an "answer" plus "results".
        answer = data.get("answer") or ""
        results = data.get("results") or []

        lines = []
        if answer:
            lines.append(f"Web search answer: {answer}")

        for i, r in enumerate(results[:3], start=1):
            title = r.get("title") or ""
            snippet = r.get("content") or r.get("snippet") or ""
            source = r.get("url") or ""
            piece = f"[{i}] {title}: {snippet} (source: {source})"
            lines.append(piece)

        if not lines:
            return None

        return (
            "The following up-to-date web search results may be helpful:\n"
            + "\n".join(lines)
        )
    except Exception as e:
        print(f"{Fore.RED}Error during Tavily web search: {e}{Style.RESET_ALL}")
        print(traceback.format_exc())
        return None


def strip_markdown(text: str) -> str:
    """
    Strip common Markdown formatting so TTS sounds natural.
    """
    if not text:
        return ""

    # Links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Inline code: `code` -> code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Bold / italics markers: *, **, _, __
    text = re.sub(r"[*_]+", "", text)

    # Headings starting with #, ##, etc.
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Blockquotes
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)

    # List markers at line starts: -, *, +, 1.
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Collapse repeated spaces and excessive blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _record_microphone(duration_seconds: int = 8, rate: int = 16000) -> bytes:
    """
    Record audio from the default microphone for a fixed duration and
    return WAV bytes suitable for Whisper. Returns empty bytes if pyaudio
    is not available (e.g. on cloud servers).
    """
    if pyaudio is None:
        return b""
    print(
        f"{Fore.YELLOW}Recording for about {duration_seconds} seconds... "
        f"start speaking now.{Style.RESET_ALL}"
    )
    pa = pyaudio.PyAudio()
    fmt = pyaudio.paInt16
    channels = 1
    frames_per_buffer = 1024

    stream = pa.open(
        format=fmt,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )

    frames = []
    total_frames = int(rate / frames_per_buffer * duration_seconds)
    try:
        for _ in range(total_frames):
            data = stream.read(frames_per_buffer)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(fmt))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

    print(f"{Fore.YELLOW}Recording complete.{Style.RESET_ALL}")
    return buffer.getvalue()


def transcribe_with_whisper(audio_wav_bytes: bytes) -> str:
    """
    Transcribe audio using a local faster-whisper model if available,
    falling back to the OpenAI Whisper API if not.
    """
    global _WHISPER_MODEL

    # Primary path: OpenAI Whisper API for best accuracy.
    api_key = get_env_var("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/audio/transcriptions"

    files = {
        "file": ("speech.wav", audio_wav_bytes, "audio/wav"),
    }
    data = {
        "model": "whisper-1",
        "response_format": "json",
    }

    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            data=data,
            files=files,
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        text = (payload.get("text") or "").strip()
        if text:
            return text
    except Exception as e:
        print(f"{Fore.RED}Error transcribing audio with Whisper API: {e}{Style.RESET_ALL}")
        print(traceback.format_exc())

    # Fallback: local faster-whisper if available.
    if _WhisperModel is not None:
        try:
            if _WHISPER_MODEL is None:
                _WHISPER_MODEL = _WhisperModel("small.en", device="cpu", compute_type="int8")
            audio_buffer = BytesIO(audio_wav_bytes)
            segments, _info = _WHISPER_MODEL.transcribe(audio_buffer, beam_size=1)
            text_parts = [seg.text.strip() for seg in segments if getattr(seg, "text", "").strip()]
            text = " ".join(text_parts).strip()
            if text:
                return text
        except Exception as e:
            print(f"{Fore.RED}Error using faster-whisper fallback: {e}{Style.RESET_ALL}")
            print(traceback.format_exc())

    return ""


def get_elevenlabs_mp3(text: str) -> bytes | None:
    """
    Generate MP3 bytes for the given text using ElevenLabs Flash.
    Uses ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID from environment.
    Returns None if key is missing, text is empty, or the API call fails.
    Used by both desktop (pygame playback) and web (stream to browser).
    """
    if not text:
        return None
    cleaned = strip_markdown(text)
    if not cleaned:
        return None
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return None
    voice_id = os.getenv("ELEVENLABS_VOICE_ID") or "EXAVITQu4vr4xnSDxMaL"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "text": cleaned,
        "model_id": "eleven_flash_v2",
        "output_format": "mp3_44100_128",
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"{Fore.RED}Error getting ElevenLabs audio: {e}{Style.RESET_ALL}")
        return None


def speak_with_elevenlabs(text: str):
    """
    Stream Angel's reply from ElevenLabs Flash model and play it immediately.
    No-op if pygame is not available (e.g. on cloud servers).
    """
    mp3_bytes = get_elevenlabs_mp3(text)
    if not mp3_bytes or pygame is None:
        return
    print(f"{Fore.MAGENTA}Angel is speaking...{Style.RESET_ALL}")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(mp3_bytes)
            tmp_path = tmp.name
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(50)
    except Exception as e:
        print(f"{Fore.RED}Error playing audio from ElevenLabs: {e}{Style.RESET_ALL}")
        print(traceback.format_exc())
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class AngelCore:
    """
    Core logic for Angel: memory, Claude, and (optionally) voice mode.
    This is reused by both the CLI and the GUI.
    """

    def __init__(self, user_id: str, use_voice: bool = False):
        self.user_id = user_id or "default-user"
        self.use_voice = use_voice

        self.memory_client = build_memory_client()
        self.anthropic_client = create_anthropic_client()
        self._use_mem0_cloud = bool(os.getenv("MEM0_API_KEY"))

    def _fetch_combined_memories(self):
        try:
            raw = self.memory_client.get_all(user_id=self.user_id)
            if isinstance(raw, dict) and "results" in raw:
                memories = raw["results"]
            else:
                memories = raw
        except Exception as e:
            print(f"{Fore.RED}Warning: could not fetch memories: {e}{Style.RESET_ALL}")
            memories = []

        combined = []
        if isinstance(memories, list):
            combined.extend(memories)
        elif isinstance(memories, dict) and isinstance(memories.get("results"), list):
            combined.extend(memories["results"])

        # Local JSON fallback mode only (do not mix local + cloud to avoid duplicates)
        if not self._use_mem0_cloud:
            local = _load_local_memories(self.user_id)
            if isinstance(local, list):
                combined.extend(local)
        return combined

    def load_initial_memory_summary(self) -> str:
        memories = self._fetch_combined_memories()
        try:
            print(f"{Fore.MAGENTA}MEMORIES LOADED (AngelCore):{Style.RESET_ALL} {len(memories)}")
            for i, m in enumerate(memories, start=1):
                print(f"{Fore.MAGENTA}  {i}. {_memory_text_for_debug(m)}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Warning: could not print loaded memories: {e}{Style.RESET_ALL}")
            print(traceback.format_exc())
        return summarize_memories_for_prompt(memories)

    def generate_reply(self, user_message: str) -> str:
        merged_memories = self._fetch_combined_memories()
        memory_summary = summarize_memories_for_prompt(merged_memories)
        system_prompt = build_system_prompt(memory_summary, voice_mode=self.use_voice)

        print(f"{Fore.BLUE}Angel is thinking...{Style.RESET_ALL}")

        # Optionally augment the user message with fresh web context.
        web_context = maybe_search_web(user_message)
        if web_context:
            print(f"{Fore.BLUE}Angel: let me look that up for you...{Style.RESET_ALL}")
            # Prepend web findings to the user's message so Claude can
            # naturally integrate them into the reply.
            augmented_user_message = (
                f"{web_context}\n\nOriginal user question:\n{user_message}"
            )
        else:
            augmented_user_message = user_message

        # Use Haiku for voice (speed) and Sonnet for text. Both can return
        # full, detailed answers; we don't impose extra length limits here.
        model = "claude-haiku-4-5" if self.use_voice else "claude-sonnet-4-5"
        reply = call_claude(
            self.anthropic_client, system_prompt, augmented_user_message, model=model
        )

        # For voice mode, strip Markdown before saving to memory so
        # memories stay clean and speech-oriented.
        memory_reply = strip_markdown(reply) if self.use_voice else reply

        # Store this turn as memory candidate (same pattern as CLI)
        try:
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": memory_reply},
            ]
            metadata = {
                "source": "angel-core",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            try:
                self.memory_client.add(messages, user_id=self.user_id, metadata=metadata)
            except Exception as e:
                print(f"{Fore.RED}Error saving memory to Mem0: {e}{Style.RESET_ALL}")
                print(traceback.format_exc())

            if not self._use_mem0_cloud:
                local_text = f"User: {messages[0]['content']} | Angel: {messages[1]['content']}"
                _append_local_memory(self.user_id, local_text, metadata)
        except Exception as e:
            print(f"{Fore.RED}Warning: could not store memory (AngelCore): {e}{Style.RESET_ALL}")

        return reply


def main():
    print(f"{Fore.CYAN}=== Angel – Personal AI Companion ==={Style.RESET_ALL}")

    # Ensure required environment variables exist
    _anthropic = get_env_var("ANTHROPIC_API_KEY")
    _openai = get_env_var("OPENAI_API_KEY")

    # Ask for a user identifier so Angel can remember you across runs
    print(
        f"{Fore.YELLOW}Enter a user id Angel should remember you by "
        f"(e.g. 'tyler', 'user-1')."
    )
    user_id = input(f"{Fore.GREEN}User id: {Style.RESET_ALL}").strip() or "default-user"

    # Choose mode: text or voice
    print()
    print(
        f"{Fore.YELLOW}Choose how you want to talk to Angel:{Style.RESET_ALL}"
    )
    print(f"{Fore.YELLOW}  1) Text chat (keyboard){Style.RESET_ALL}")
    print(f"{Fore.YELLOW}  2) Voice chat (microphone + ElevenLabs){Style.RESET_ALL}")
    mode_choice = input(
        f"{Fore.GREEN}Enter 1 or 2 (default 1): {Style.RESET_ALL}"
    ).strip() or "1"
    use_voice = mode_choice == "2"

    # Initialize Mem0 and Anthropic
    print(f"{Fore.BLUE}Initializing memory and AI brain...{Style.RESET_ALL}")
    memory_client = build_memory_client()
    anthropic_client = create_anthropic_client()

    # Load existing memories
    print(f"{Fore.BLUE}Fetching Angel's memories of you (if any)...{Style.RESET_ALL}")
    try:
        existing_raw = memory_client.get_all(user_id=user_id)
        # Mem0 returns {"results": [...], "total": N}; fall back gracefully if shape changes
        if isinstance(existing_raw, dict) and "results" in existing_raw:
            existing_memories = existing_raw["results"]
        else:
            existing_memories = existing_raw
    except Exception as e:
        print(f"{Fore.RED}Warning: could not fetch memories: {e}{Style.RESET_ALL}")
        existing_memories = []

    # Also load local JSON memories (fallback that always persists)
    local_memories = _load_local_memories(user_id)

    # Combine Mem0 + local memories for context and debug
    all_startup_memories = []
    if isinstance(existing_memories, list):
        all_startup_memories.extend(existing_memories)
    if isinstance(local_memories, list):
        all_startup_memories.extend(local_memories)

    try:
        print(f"{Fore.MAGENTA}MEMORIES LOADED:{Style.RESET_ALL} {len(all_startup_memories)}")
        for i, m in enumerate(all_startup_memories, start=1):
            print(f"{Fore.MAGENTA}  {i}. {_memory_text_for_debug(m)}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Warning: could not print loaded memories: {e}{Style.RESET_ALL}")
        print(traceback.format_exc())

    memory_summary = summarize_memories_for_prompt(all_startup_memories)

    print()
    print(f"{Fore.CYAN}Angel is ready.{Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}Type your message and press Enter "
        f"(or speak if you chose voice mode). "
        f"Say or type 'exit' or 'quit' to close.{Style.RESET_ALL}"
    )
    print()

    # Conversation loop
    while True:
        try:
            if use_voice:
                print(
                    f"{Fore.GREEN}Press Enter, then speak for ~8 seconds.{Style.RESET_ALL}"
                )
                _ = input()
                audio_bytes = _record_microphone()
                user_message = transcribe_with_whisper(audio_bytes).strip()
                if not user_message:
                    print(
                        f"{Fore.RED}I could not hear anything clear enough to transcribe. "
                        f"Let’s try again.{Style.RESET_ALL}"
                    )
                    continue
                print(
                    f"{Fore.GREEN}You (transcribed):{Style.RESET_ALL} {user_message}"
                )
            else:
                user_message = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_message:
            continue

        if user_message.lower() in {"exit", "quit"}:
            print(f"{Fore.CYAN}Angel: Until next time.{Style.RESET_ALL}")
            break

        # Refresh memories each turn in case Mem0 updated between calls
        try:
            current_raw = memory_client.get_all(user_id=user_id)
            if isinstance(current_raw, dict) and "results" in current_raw:
                current_memories = current_raw["results"]
            else:
                current_memories = current_raw
        except Exception:
            current_memories = existing_memories  # fallback

        # Merge in local JSON memories as well
        current_local = _load_local_memories(user_id)
        merged_memories = []
        if isinstance(current_memories, list):
            merged_memories.extend(current_memories)
        if isinstance(current_local, list):
            merged_memories.extend(current_local)

        memory_summary = summarize_memories_for_prompt(merged_memories)
        system_prompt = build_system_prompt(memory_summary)

        # Call Claude
        print(f"{Fore.BLUE}Angel is thinking...{Style.RESET_ALL}")
        reply = call_claude(anthropic_client, system_prompt, user_message)

        print(f"{Fore.CYAN}Angel:{Style.RESET_ALL} {reply}")
        print()

        # Speak the reply out loud in voice mode
        if use_voice:
            speak_with_elevenlabs(reply)

        # Store this turn as memory candidate
        try:
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": reply},
            ]
            # Metadata can be used later for filtering or categories
            metadata = {
                "source": "angel-cli",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            # Try saving to Mem0; if it fails, print full error/traceback
            try:
                memory_client.add(messages, user_id=user_id, metadata=metadata)
            except Exception as e:
                print(f"{Fore.RED}Error saving memory to Mem0: {e}{Style.RESET_ALL}")
                print(traceback.format_exc())

            # Always save to local JSON as a persistent fallback
            local_text = f"User: {messages[0]['content']} | Angel: {messages[1]['content']}"
            _append_local_memory(user_id, local_text, metadata)
        except Exception as e:
            print(f"{Fore.RED}Warning: could not store memory: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()