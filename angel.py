import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from io import BytesIO
import wave
import tempfile

import requests
from colorama import init as colorama_init, Fore, Style
from dotenv import load_dotenv

import anthropic
from mem0 import Memory

try:
    from angel_computer import run_computer_use_session
    COMPUTER_CONTROL_AVAILABLE = True
except Exception:
    run_computer_use_session = None  # type: ignore[assignment]
    COMPUTER_CONTROL_AVAILABLE = False

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

# Stage 2 memory categories
CATEGORY_PATTERNS = "patterns"
CATEGORY_PERSON_PROFILE = "person_profile"
CATEGORY_RESEARCH_TIMELINE = "research_timeline"
CATEGORY_REFLECTION = "reflection"
CATEGORY_BRIEFING_HISTORY = "briefing_history"
CATEGORY_INTELLIGENCE_FILE = "intelligence_file"
CATEGORY_THREAT_WATCH = "threat_watch"

_STRUCTURED_MEMORY_CATEGORIES = frozenset(
    {
        CATEGORY_PATTERNS,
        CATEGORY_PERSON_PROFILE,
        CATEGORY_RESEARCH_TIMELINE,
        CATEGORY_REFLECTION,
        CATEGORY_BRIEFING_HISTORY,
        CATEGORY_INTELLIGENCE_FILE,
        CATEGORY_THREAT_WATCH,
    }
)

# Intelligence File Cabinet folder for automated + manual threat intel (Item 12)
THREAT_INTEL_FOLDER = "Threat Intelligence"

# Default threat watch categories (merged on each scan with Mem0 category threat_watch)
THREAT_WATCH_DEFAULT_CATEGORIES: list[str] = [
    # Professional
    "FBI law enforcement policy changes federal",
    "DOJ FBI federal budget cuts justice department",
    "US federal employment career impact intelligence professionals",
    # Mission — UAP / disclosure
    "UAP UFO disclosure government news",
    "UAP suppression government disclosure obstruction",
    "David Grusch Luis Elizondo Christopher Mellon Ross Coulthart UAP statements",
    "Congressional UAP hearing legislation 2025",
    "foreign government UAP acknowledgment news",
    "Pentagon defense contractor black budget programs news",
    "FOIA classified programs disclosure",
    "national security whistleblower classified programs",
    "SAP special access programs oversight news",
    # Broader awareness
    "US domestic stability geopolitical risk",
    "AI regulation policy United States federal",
    "surveillance technology civil liberties news",
    "US constitutional rights civil liberties developments",
    "US government transparency crackdown whistleblowers",
    "South Carolina state government news",
    "South Carolina local government news",
    # Wildcard / anomalies
    "credible unexplained phenomena reports",
    "credible witness anomalous experiences",
    "black eyed people phenomenon reports",
]
# Sentinel for Angel to output a new pattern (parsed and stored)
ANGEL_PATTERN_PREFIX = "[ANGEL_PATTERN]:"
# Sentinel for Angel to output a new/updated person profile (parsed and stored)
ANGEL_PROFILE_PREFIX = "[ANGEL_PROFILE]:"
# Legacy: Intelligence File Cabinet creation block (parsed in generate_reply)
INTELLIGENCE_FILE_CREATED_PREFIX = "[INTELLIGENCE FILE CREATED]"
# Preferred machine-parseable filing tag (stripped from Tyler-visible reply after save)
_INTELLIGENCE_FILE_TAG_RE = re.compile(
    r"\[FILE:\s*folder\s*=\s*([^|]*?)\s*\|\s*name\s*=\s*([^\]]+?)\s*\]",
    re.IGNORECASE,
)
# Folder:/File: lines after [INTELLIGENCE FILE CREATED]
_LEGACY_INTEL_FILE_BLOCK_RE = re.compile(
    r"(?i)\[INTELLIGENCE FILE CREATED\]\s*\r?\n\s*Folder:\s*([^\n\r]+)\s*\r?\n\s*File:\s*([^\n\r]+)\s*\r?\n([\s\S]*?)(?=\r?\n\s*\[INTELLIGENCE FILE CREATED\]|\r?\n\s*\[FILE:\s*folder\s*=|\Z)",
)

# Fenced ```python ... ``` blocks in Angel's reply (executed server-side after generation).
# Opening fence may use spaces (``` python), optional CRLF after the tag, and optional whitespace after closing ```.
_PYTHON_CODE_BLOCK_RE = re.compile(
    r"```\s*python\s*(?:\r?\n|\r)?(.*?)```\s*",
    re.IGNORECASE | re.DOTALL,
)

EXEC_PYTHON_TIMEOUT_SEC = 30
_EXEC_OUTPUT_MAX_CHARS = 256_000


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

    def add(
        self,
        messages,
        user_id: str,
        metadata: dict | None = None,
        *,
        infer: bool = True,
        async_mode: bool = True,
    ):
        # v1 add memories endpoint supports version="v2"
        url = f"{MEM0_API_BASE_URL}/v1/memories/"
        payload = {
            "user_id": user_id,
            "messages": messages,
            "metadata": metadata or {},
            "version": "v2",
            "output_format": "v1.1",
            "async_mode": async_mode,
            "infer": infer,
        }
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def delete_memory(self, memory_id: str) -> None:
        """Delete a single memory in Mem0 Cloud by id (v1 API)."""
        mid = (memory_id or "").strip()
        if not mid:
            return
        url = f"{MEM0_API_BASE_URL}/v1/memories/{mid}/"
        resp = requests.delete(url, headers=self._headers(), timeout=30)
        if resp.status_code not in (200, 204):
            resp.raise_for_status()


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


def _load_local_memory_entries(user_id: str) -> list:
    """Return the raw list of memory entry dicts for user_id from tyler_memories.json."""
    try:
        if not LOCAL_MEMORY_FILE.exists():
            return []
        with LOCAL_MEMORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        users = data.get("users", {})
        arr = users.get(user_id, [])
        return arr if isinstance(arr, list) else []
    except Exception as e:
        print(f"{Fore.RED}Warning: could not load local memory entries: {e}{Style.RESET_ALL}")
        return []


def _save_local_memory_entries(user_id: str, entries: list) -> None:
    """Replace the on-disk memory list for user_id (full file rewrite)."""
    try:
        if LOCAL_MEMORY_FILE.exists():
            with LOCAL_MEMORY_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"users": {}}
        users = data.setdefault("users", {})
        users[user_id] = entries
        with LOCAL_MEMORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"{Fore.RED}Warning: could not save local memory entries: {e}{Style.RESET_ALL}")
        print(traceback.format_exc())


def _parse_intelligence_tags(meta: dict) -> list[str]:
    raw = meta.get("intelligence_tags_json")
    if raw is None:
        raw = meta.get("tags")
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except json.JSONDecodeError:
            pass
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


def _intelligence_file_metadata(
    *,
    file_name: str,
    folder: str,
    tags: list[str] | None,
    created_at: str,
    updated_at: str,
    mem0_memory_id: str | None = None,
) -> dict:
    tags = tags or []
    meta = {
        "category": CATEGORY_INTELLIGENCE_FILE,
        "file_name": file_name.strip(),
        "folder": (folder or "").strip() or "Uncategorized",
        "intelligence_tags_json": json.dumps(tags, ensure_ascii=False),
        "created_at": created_at,
        "updated_at": updated_at,
        "timestamp": updated_at,
        "source": "angel-file-cabinet",
    }
    if mem0_memory_id:
        meta["mem0_memory_id"] = mem0_memory_id.strip()
    return meta


def _extract_mem0_id_from_add_response(resp: object) -> str | None:
    if not isinstance(resp, dict):
        return None
    for key in ("id", "memory_id", "memoryId"):
        v = resp.get(key)
        if v and isinstance(v, str):
            return v.strip()
    # Nested shapes from async / batched responses
    for key in ("results", "memories", "data"):
        chunk = resp.get(key)
        if isinstance(chunk, list) and chunk:
            first = chunk[0]
            if isinstance(first, dict):
                for k in ("id", "memory_id"):
                    v = first.get(k)
                    if v and isinstance(v, str):
                        return v.strip()
    return None


class FilesCabinet:
    """
    Angel's Intelligence File Cabinet: structured notes in Mem0 (category intelligence_file)
    plus canonical rows in local tyler_memories.json. Folders are free-form strings.
    """

    def __init__(self, memory_client, user_id: str, use_mem0_cloud: bool = False):
        self.memory_client = memory_client
        self.user_id = user_id or "default-user"
        self._use_mem0_cloud = bool(use_mem0_cloud)

    def _iter_local_intelligence_entries(self) -> list[dict]:
        out: list[dict] = []
        for entry in _load_local_memory_entries(self.user_id):
            if not isinstance(entry, dict):
                continue
            meta = entry.get("metadata")
            if not isinstance(meta, dict):
                continue
            if meta.get("category") != CATEGORY_INTELLIGENCE_FILE:
                continue
            fn = (meta.get("file_name") or "").strip()
            if not fn:
                continue
            out.append(entry)
        return out

    def _entry_to_record(self, entry: dict, *, include_content: bool = True) -> dict:
        meta = entry.get("metadata") if isinstance(entry, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        name = (meta.get("file_name") or "").strip()
        folder = (meta.get("folder") or "").strip() or "Uncategorized"
        created_at = (meta.get("created_at") or entry.get("created_at") or "").strip()
        updated_at = (meta.get("updated_at") or meta.get("timestamp") or created_at).strip()
        tags = _parse_intelligence_tags(meta)
        rec: dict = {
            "name": name,
            "folder": folder,
            "created_at": created_at,
            "updated_at": updated_at,
            "tags": tags,
        }
        if include_content:
            rec["content"] = (entry.get("memory") or entry.get("data") or "").strip()
        return rec

    def _find_local_entry_index(self, file_name: str) -> int | None:
        target = (file_name or "").strip()
        if not target:
            return None
        entries = _load_local_memory_entries(self.user_id)
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            meta = entry.get("metadata")
            if not isinstance(meta, dict):
                continue
            if meta.get("category") != CATEGORY_INTELLIGENCE_FILE:
                continue
            if (meta.get("file_name") or "").strip() == target:
                return i
        return None

    def _push_mem0(self, content: str, metadata: dict) -> str | None:
        if not self._use_mem0_cloud:
            return None
        if not hasattr(self.memory_client, "add"):
            return None
        fn = metadata.get("file_name", "")
        fd = metadata.get("folder", "")
        user_line = (
            f"[Angel Intelligence File folder={fd!r} name={fn!r}]\n" + (content or "")[:120_000]
        )
        messages = [
            {"role": "user", "content": user_line},
            {"role": "assistant", "content": "Intelligence file stored."},
        ]
        try:
            add_kw: dict = {"infer": False, "async_mode": True}
            resp = self.memory_client.add(
                messages,
                user_id=self.user_id,
                metadata=metadata,
                **add_kw,
            )
            return _extract_mem0_id_from_add_response(resp)
        except TypeError:
            try:
                resp = self.memory_client.add(
                    messages, user_id=self.user_id, metadata=metadata
                )
                return _extract_mem0_id_from_add_response(resp)
            except Exception as e:
                print(f"{Fore.YELLOW}FilesCabinet Mem0 add: {e}{Style.RESET_ALL}")
                return None
        except Exception as e:
            print(f"{Fore.YELLOW}FilesCabinet Mem0 add: {e}{Style.RESET_ALL}")
            return None

    def _delete_mem0_for_entry(self, meta: dict) -> None:
        if not self._use_mem0_cloud:
            return
        mid = (meta.get("mem0_memory_id") or "").strip() if isinstance(meta, dict) else ""
        if mid and isinstance(self.memory_client, Mem0CloudClient):
            try:
                self.memory_client.delete_memory(mid)
            except Exception as e:
                print(f"{Fore.YELLOW}FilesCabinet Mem0 delete id={mid}: {e}{Style.RESET_ALL}")
            return
        # Fallback: search cloud memories by file_name
        if not isinstance(self.memory_client, Mem0CloudClient):
            return
        fname = (meta.get("file_name") or "").strip() if isinstance(meta, dict) else ""
        if not fname:
            return
        try:
            raw = self.memory_client.get_all(user_id=self.user_id)
            results = raw.get("results") if isinstance(raw, dict) else raw
            if not isinstance(results, list):
                return
            for item in results:
                if not isinstance(item, dict):
                    continue
                im = item.get("metadata") or {}
                if not isinstance(im, dict):
                    continue
                if im.get("category") != CATEGORY_INTELLIGENCE_FILE:
                    continue
                if (im.get("file_name") or "").strip() != fname:
                    continue
                rid = (item.get("id") or item.get("memory_id") or "").strip()
                if rid:
                    try:
                        self.memory_client.delete_memory(rid)
                    except Exception as e:
                        print(f"{Fore.YELLOW}FilesCabinet Mem0 delete fallback: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}FilesCabinet Mem0 delete scan: {e}{Style.RESET_ALL}")

    def create_file(
        self,
        folder: str,
        name: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict:
        file_name = (name or "").strip()
        if not file_name:
            raise ValueError("File name is required.")
        if self.get_file(file_name):
            raise ValueError(f"An intelligence file named {file_name!r} already exists.")
        folder_s = (folder or "").strip() or "Uncategorized"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        tags_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
        body = (content or "").strip()
        meta = _intelligence_file_metadata(
            file_name=file_name,
            folder=folder_s,
            tags=tags_list,
            created_at=now,
            updated_at=now,
        )
        mem0_id = self._push_mem0(body, dict(meta))
        if mem0_id:
            meta["mem0_memory_id"] = mem0_id
        _append_local_memory(self.user_id, body, meta)
        return self.get_file(file_name) or {
            "name": file_name,
            "folder": folder_s,
            "content": body,
            "created_at": now,
            "updated_at": now,
            "tags": tags_list,
        }

    def update_file(self, name: str, new_content: str) -> dict:
        file_name = (name or "").strip()
        if not file_name:
            raise ValueError("File name is required.")
        idx = self._find_local_entry_index(file_name)
        if idx is None:
            raise ValueError(f"No intelligence file named {file_name!r}.")
        entries = _load_local_memory_entries(self.user_id)
        entry = entries[idx]
        meta = dict(entry.get("metadata") or {})
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta["updated_at"] = now
        meta["timestamp"] = now
        created_before = (meta.get("created_at") or "").strip() or now
        meta.setdefault("created_at", created_before)
        body = (new_content or "").strip()
        self._delete_mem0_for_entry(meta)
        mem0_id = self._push_mem0(body, meta)
        if mem0_id:
            meta["mem0_memory_id"] = mem0_id
        else:
            meta.pop("mem0_memory_id", None)
        entries[idx] = {
            "memory": body,
            "metadata": meta,
            "created_at": entry.get("created_at") or created_before,
        }
        _save_local_memory_entries(self.user_id, entries)
        rec = self.get_file(file_name)
        if not rec:
            raise RuntimeError("Update succeeded locally but record could not be reloaded.")
        return rec

    def delete_file(self, name: str) -> bool:
        file_name = (name or "").strip()
        if not file_name:
            raise ValueError("File name is required.")
        idx = self._find_local_entry_index(file_name)
        if idx is None:
            return False
        entries = _load_local_memory_entries(self.user_id)
        entry = entries[idx]
        meta = entry.get("metadata") if isinstance(entry, dict) else {}
        if isinstance(meta, dict):
            self._delete_mem0_for_entry(meta)
        entries.pop(idx)
        _save_local_memory_entries(self.user_id, entries)
        return True

    def get_file(self, name: str) -> dict | None:
        file_name = (name or "").strip()
        if not file_name:
            return None
        idx = self._find_local_entry_index(file_name)
        if idx is None:
            return None
        entries = _load_local_memory_entries(self.user_id)
        return self._entry_to_record(entries[idx], include_content=True)

    def list_files(self, folder: str | None = None) -> list[dict]:
        want = (folder or "").strip().lower() if folder is not None else None
        out: list[dict] = []
        for entry in self._iter_local_intelligence_entries():
            rec = self._entry_to_record(entry, include_content=False)
            if want is not None and rec["folder"].lower() != want:
                continue
            out.append(rec)
        try:
            out.sort(key=lambda r: (r["folder"].lower(), r["name"].lower()))
        except Exception:
            pass
        return out

    def list_folders(self) -> list[str]:
        seen: set[str] = set()
        for entry in self._iter_local_intelligence_entries():
            rec = self._entry_to_record(entry, include_content=False)
            seen.add(rec["folder"])
        return sorted(seen, key=str.lower)

    def search_files(self, query: str) -> list[dict]:
        q = (query or "").strip().lower()
        if not q:
            return []
        matches: list[dict] = []
        for entry in self._iter_local_intelligence_entries():
            rec = self._entry_to_record(entry, include_content=True)
            hay = " ".join(
                [
                    rec["name"],
                    rec["folder"],
                    rec["content"],
                    " ".join(rec["tags"]),
                ]
            ).lower()
            if q in hay:
                matches.append(rec)
        try:
            matches.sort(key=lambda r: (r["folder"].lower(), r["name"].lower()))
        except Exception:
            pass
        return matches

    def get_summary(self) -> str:
        by_folder: dict[str, list[str]] = {}
        for entry in self._iter_local_intelligence_entries():
            rec = self._entry_to_record(entry, include_content=False)
            by_folder.setdefault(rec["folder"], []).append(rec["name"])
        if not by_folder:
            return "No intelligence files are filed yet."
        lines: list[str] = []
        for fld in sorted(by_folder.keys(), key=str.lower):
            names = sorted(set(by_folder[fld]), key=str.lower)
            lines.append(f"- {fld}: {', '.join(names)}")
        return "Intelligence File Cabinet (folder → files):\n" + "\n".join(lines)


def add_structured_memory(
    memory_client,
    user_id: str,
    text: str,
    category: str,
    person_name: str | None = None,
    use_mem0_cloud: bool = False,
) -> None:
    """
    Store a pattern or person profile in memory (local JSON and optionally Mem0).
    """
    metadata = {
        "category": category,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": "angel-stage2",
    }
    if person_name:
        metadata["person_name"] = person_name.strip()
    _append_local_memory(user_id, text, metadata)
    if use_mem0_cloud and hasattr(memory_client, "add"):
        messages = [
            {"role": "user", "content": f"[Angel internal] Store: {text[:500]}"},
            {"role": "assistant", "content": "Stored."},
        ]
        try:
            memory_client.add(messages, user_id=user_id, metadata=metadata)
        except Exception as e:
            print(f"{Fore.RED}Warning: could not store structured memory in cloud: {e}{Style.RESET_ALL}")


def extract_stage2_from_reply(reply: str) -> tuple[str, str | None, tuple[str, str] | None]:
    """
    Parse reply for [ANGEL_PATTERN]: and [ANGEL_PROFILE]: name|text.
    Returns (cleaned_reply, pattern_text or None, (person_name, profile_text) or None).
    """
    cleaned = reply
    pattern_text = None
    profile_tuple = None

    if ANGEL_PATTERN_PREFIX in cleaned:
        idx = cleaned.find(ANGEL_PATTERN_PREFIX)
        rest = cleaned[idx + len(ANGEL_PATTERN_PREFIX) :].strip()
        end = rest.find("\n\n") if "\n\n" in rest else len(rest)
        line = rest[:end].split("\n")[0].strip()
        if line:
            pattern_text = line
        cleaned = (cleaned[:idx].rstrip() + "\n" + rest[end:].lstrip()).strip()
        cleaned = cleaned.rstrip()

    if ANGEL_PROFILE_PREFIX in cleaned:
        idx = cleaned.find(ANGEL_PROFILE_PREFIX)
        rest = cleaned[idx + len(ANGEL_PROFILE_PREFIX) :].strip()
        end = rest.find("\n\n") if "\n\n" in rest else len(rest)
        block = rest[:end].strip()
        if "|" in block:
            name_part, text_part = block.split("|", 1)
            pname = name_part.strip()
            ptext = text_part.strip()
            if pname and ptext:
                profile_tuple = (pname, ptext)
        cleaned = (cleaned[:idx].rstrip() + "\n" + rest[end:].lstrip()).strip()
        cleaned = cleaned.rstrip()

    return cleaned, pattern_text, profile_tuple


def _next_intelligence_markup_in_tail(tail: str) -> int | None:
    """
    Offset in tail where the next filing markup starts (so file body ends before it).
    Returns None if no following markup.
    """
    if not tail:
        return None
    positions: list[int] = []
    m2 = _INTELLIGENCE_FILE_TAG_RE.search(tail)
    if m2:
        positions.append(m2.start())
    mleg = re.search(r"(?i)\r?\n\s*\[INTELLIGENCE FILE CREATED\]", tail)
    if mleg:
        positions.append(mleg.start())
    return min(positions) if positions else None


def process_filed_intelligence_in_reply(reply: str, files_cabinet: "FilesCabinet") -> str:
    """
    Detect filing markup in Angel's reply, call files_cabinet.create_file, and strip
    saved blocks from the text Tyler sees.

    Preferred: ``[FILE:folder=FolderName|name=FileName]`` then the file body until the
    next filing marker or EOF.

    Legacy::

        [INTELLIGENCE FILE CREATED]
        Folder: ...
        File: ...
        <body>
    """
    if not reply or not isinstance(reply, str):
        return reply
    text = reply
    filed_success: list[tuple[str, str]] = []  # (file_name, folder) for confirmation if reply is empty

    # 1) [FILE:folder=...|name=...] blocks
    pos = 0
    while True:
        m = _INTELLIGENCE_FILE_TAG_RE.search(text, pos)
        if not m:
            break
        folder = (m.group(1) or "").strip() or "Uncategorized"
        name = (m.group(2) or "").strip()
        start, end_tag = m.start(), m.end()
        tail = text[end_tag:]
        rel_end = _next_intelligence_markup_in_tail(tail)
        chunk_end = end_tag + rel_end if rel_end is not None else len(text)
        body = text[end_tag:chunk_end].strip()

        if not name:
            pos = end_tag
            continue

        if not body:
            print(
                f"{Fore.YELLOW}Intelligence file tag {name!r} had empty body; stripping tag only.{Style.RESET_ALL}"
            )
            left = text[:start].rstrip()
            right = text[end_tag:].lstrip()
            text = f"{left}\n\n{right}".strip() if left and right else (left + right).strip()
            pos = 0
            continue

        try:
            tags = _infer_threat_intel_tags(folder, body)
            files_cabinet.create_file(folder, name, body, tags=tags)
            filed_success.append((name, folder))
            print(
                f"{Fore.MAGENTA}Intelligence file saved: {folder!r} / {name!r}{Style.RESET_ALL}"
            )
        except ValueError as e:
            print(
                f"{Fore.YELLOW}Intelligence file not saved ({e}); markup left in reply.{Style.RESET_ALL}"
            )
            pos = end_tag
            continue
        except Exception as e:
            print(f"{Fore.YELLOW}Intelligence file save error: {e}{Style.RESET_ALL}")
            pos = end_tag
            continue

        left = text[:start].rstrip()
        right = text[chunk_end:].lstrip()
        text = f"{left}\n\n{right}".strip() if left and right else (left + right).strip()
        pos = 0

    # 2) Legacy [INTELLIGENCE FILE CREATED] / Folder: / File: / body
    while True:
        m = _LEGACY_INTEL_FILE_BLOCK_RE.search(text)
        if not m:
            break
        folder = (m.group(1) or "").strip() or "Uncategorized"
        name = (m.group(2) or "").strip()
        body = (m.group(3) or "").strip()
        start, end = m.span()

        if not name or not body:
            left = text[:start].rstrip()
            right = text[end:].lstrip()
            text = f"{left}\n\n{right}".strip() if left and right else (left + right).strip()
            continue

        try:
            tags = _infer_threat_intel_tags(folder, body)
            files_cabinet.create_file(folder, name, body, tags=tags)
            filed_success.append((name, folder))
            print(
                f"{Fore.MAGENTA}Intelligence file saved (legacy block): {folder!r} / {name!r}{Style.RESET_ALL}"
            )
        except ValueError as e:
            print(
                f"{Fore.YELLOW}Intelligence file not saved ({e}); legacy block left in reply.{Style.RESET_ALL}"
            )
            break
        except Exception as e:
            print(f"{Fore.YELLOW}Intelligence file save error: {e}{Style.RESET_ALL}")
            break

        left = text[:start].rstrip()
        right = text[end:].lstrip()
        text = f"{left}\n\n{right}".strip() if left and right else (left + right).strip()

    if not (text or "").strip() and filed_success:
        text = "Filed. " + " ".join(
            f"[{fname}] saved to [{fld}]." for fname, fld in filed_success
        )

    if filed_success:
        threat_filed = any(
            (fld or "").strip().lower() == THREAT_INTEL_FOLDER.lower() for _, fld in filed_success
        )
        if threat_filed and "Threat Intelligence you should know about" not in (text or ""):
            text = ((text or "").rstrip() + "\n\nI've filed something in Threat Intelligence you should know about.").strip()

    return text


def _strip_transcript_prefixes_from_memory(text: str) -> str:
    """
    Remove 'User:' and 'Angel:' dialogue structure from memory content so the
    system prompt does not contain transcript-style text that could cause the
    model to continue fake conversations.
    """
    if not text or not isinstance(text, str):
        return text
    # If format is "User: ... | Angel: ...", keep only Angel's response
    if " | Angel:" in text:
        parts = text.split(" | Angel:", 1)
        text = parts[1].strip() if len(parts) > 1 else text
    # Strip any remaining "User:" or "Angel:" prefixes (e.g. at line start)
    text = re.sub(r"^\s*User:\s*", "", text)
    text = re.sub(r"^\s*Angel:\s*", "", text)
    return text.strip()


def _parse_memory_datetime(created_at: str) -> datetime | None:
    """Parse Mem0 / local ISO timestamps into UTC-aware datetime."""
    if not created_at or not isinstance(created_at, str):
        return None
    s = created_at.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def relative_time_phrase(created_at: str) -> str:
    """
    Human-readable age for a memory's created_at, e.g. 'yesterday', '3 weeks ago', 'last month'.
    """
    dt = _parse_memory_datetime(created_at)
    if dt is None:
        return "unknown time"
    now = datetime.now(timezone.utc)
    delta = now - dt
    secs = delta.total_seconds()
    if secs < 0:
        return "just now"
    if secs < 60:
        return "just now"
    if secs < 3600:
        m = int(secs // 60)
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if secs < 24 * 3600 and delta.days == 0:
        h = int(secs // 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = delta.days
    if d == 1:
        return "yesterday"
    if d < 7:
        return f"{d} days ago"
    if d < 14:
        return "last week"
    if d < 21:
        return "2 weeks ago"
    weeks = d // 7
    if weeks < 8:
        return f"{weeks} weeks ago"
    months = max(1, d // 30)
    if d < 730:
        if months == 1:
            return "last month"
        return f"{months} months ago"
    years = d // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def _memory_created_at(m: dict) -> str:
    """Best-effort created_at string from a normalized memory dict."""
    if not isinstance(m, dict):
        return ""
    ca = m.get("created_at") or ""
    meta = m.get("metadata")
    if isinstance(meta, dict):
        if not ca:
            ca = meta.get("created_at") or meta.get("timestamp") or meta.get("createdAt") or ""
    return ca if isinstance(ca, str) else ""


def _format_memory_line_with_age(
    created_at: str,
    text: str,
    tags=None,
    *,
    event_date: str | None = None,
) -> str:
    """
    Prefix memory line with relative storage age, and optionally [Event: …] when Tyler
    dated the story in the originating user message (see extract_event_date_from_user_message).
    """
    when = relative_time_phrase(created_at)
    ev = (event_date or "").strip()
    if ev:
        base = f"[Event: {ev}] [Stored: {when}] {text}"
    else:
        base = f"[{when}] {text}"
    if tags:
        return f"- ({tags}) {base}"
    return f"- {base}"


_EVENT_WEEKDAY_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_RELATIVE_WEEKDAY_RE = re.compile(
    r"\b(last|past|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

_MONTH_PHRASE_RE = re.compile(
    r"\b(?:in|during|back in|last|this)\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\b"
    r"(?:\s*,?\s*(19\d{2}|20\d{2}))?",
    re.IGNORECASE,
)

_VAGUE_EVENT_PHRASES = (
    "earlier today",
    "earlier this week",
    "yesterday evening",
    "yesterday",
    "last week",
    "last month",
    "last year",
    "a few years ago",
    "a couple years ago",
)


def _local_datetime_for_event_parsing() -> datetime:
    tz_name = (os.getenv("TIMEZONE") or "UTC").strip()
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    return datetime.now(tz)


def _calendar_date_for_last_named_weekday(now: datetime, day_name: str) -> str:
    target = _EVENT_WEEKDAY_MAP[day_name.lower()]
    current = now.weekday()
    delta = (current - target) % 7
    if delta == 0:
        delta = 7
    d = (now.date() - timedelta(days=delta))
    return d.isoformat()


def _calendar_date_for_this_named_weekday(now: datetime, day_name: str) -> str:
    target = _EVENT_WEEKDAY_MAP[day_name.lower()]
    current = now.weekday()
    delta = (target - current) % 7
    d = now.date() + timedelta(days=delta)
    return d.isoformat()


def extract_event_date_from_user_message(user_message: str) -> str | None:
    """
    Best-effort extraction when Tyler anchors a story in calendar time (not storage time).
    Returns a short human-readable label for metadata ``event_date`` (shown in prompts as [Event: …]).
    """
    if not user_message or not isinstance(user_message, str):
        return None
    s = user_message.strip()
    if len(s) < 4:
        return None
    low = s.lower()

    m = _TAVILY_DATE_REGEX.search(s)
    if m:
        return m.group(0).strip()

    m = _MONTH_PHRASE_RE.search(s)
    if m:
        month, year = m.group(1), m.group(2)
        if year:
            return f"{month} {year}"
        return month

    m = _RELATIVE_WEEKDAY_RE.search(s)
    if m:
        qual, day = m.group(1).lower(), m.group(2).lower()
        now = _local_datetime_for_event_parsing()
        if qual in ("last", "past"):
            iso = _calendar_date_for_last_named_weekday(now, day)
            return f"{iso} ({day.title()})"
        iso = _calendar_date_for_this_named_weekday(now, day)
        return f"{iso} ({day.title()}, this week)"

    m = re.search(
        r"\b(?:in|during|circa|c\.|around|back in|year\s+)\s*(19\d{2}|20\d{2})\b",
        s,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)

    m = re.search(r"\b(19\d{2}|20\d{2})\b", s)
    if m:
        return m.group(1)

    for phrase in _VAGUE_EVENT_PHRASES:
        if phrase in low:
            return phrase

    m = re.search(r"\bwhen I was (\d{1,2})\b", s, re.IGNORECASE)
    if m:
        return f"~age {m.group(1)} (year not stated)"

    return None


def merge_user_event_date_into_metadata(metadata: dict, user_message: str) -> dict:
    """Copy metadata and set ``event_date`` when the user message contains a datable anchor."""
    out = dict(metadata)
    ev = extract_event_date_from_user_message(user_message)
    if ev:
        out["event_date"] = ev
    return out


# Patterns to pull explicit calendar dates from Tavily / research text for timeline memories
_TAVILY_DATE_REGEX = re.compile(
    r"(?:\b\d{4}-\d{2}-\d{2}\b|"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}\b|"
    r"\b\d{1,2}/\d{1,2}/\d{4}\b|"
    r"\bQ[1-4] \d{4}\b)",
    re.IGNORECASE,
)


def _snippet_around(text: str, start: int, end: int, width: int = 90) -> str:
    lo = max(0, start - width // 2)
    hi = min(len(text), end + width // 2)
    snip = text[lo:hi].replace("\n", " ").strip()
    if len(snip) > 160:
        snip = snip[:157] + "..."
    return snip


def extract_dated_events_from_research_text(text: str, max_events: int = 14) -> list[str]:
    """
    Extract date-like strings and short context from Tavily-style blobs.
    Returns lines suitable for timeline memory (no external API).
    """
    if not text or not isinstance(text, str):
        return []
    seen = set()
    out: list[str] = []
    for m in _TAVILY_DATE_REGEX.finditer(text):
        span = m.group(0)
        key = span.lower()
        if key in seen:
            continue
        seen.add(key)
        snip = _snippet_around(text, m.start(), m.end())
        out.append(f"{span}: {snip}")
        if len(out) >= max_events:
            break
    return out


def store_tavily_research_timeline(
    memory_client,
    user_id: str,
    query: str,
    research_text: str,
    use_mem0_cloud: bool,
) -> None:
    """
    Persist dated snippets from web research as structured timeline memories.
    """
    if not research_text or not user_id:
        return
    events = extract_dated_events_from_research_text(research_text)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    qshort = (query or "").strip()[:240]
    if events:
        body = (
            f"External research timeline (Tavily), query: {qshort!r} (indexed {now_iso}):\n"
            + "\n".join(f"  • {e}" for e in events)
        )
    else:
        body = (
            f"External research (Tavily), query: {qshort!r} (indexed {now_iso}): "
            f"no explicit calendar dates parsed in snippets; content available in session context only."
        )
    try:
        add_structured_memory(
            memory_client,
            user_id,
            body,
            CATEGORY_RESEARCH_TIMELINE,
            person_name=None,
            use_mem0_cloud=use_mem0_cloud,
        )
    except Exception as e:
        print(f"{Fore.YELLOW}Could not store research timeline memory: {e}{Style.RESET_ALL}")


def _normalize_memories_list(memories):
    """Unwrap and normalize memory items into a list of dicts with memory, metadata, created_at."""
    if isinstance(memories, dict):
        if "results" in memories and isinstance(memories["results"], list):
            memories = memories["results"]
        elif "data" in memories and isinstance(memories["data"], list):
            memories = memories["data"]
    if not memories:
        return []
    normalized = []
    for item in memories:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append({"memory": item, "metadata": {}, "created_at": ""})
        else:
            continue
    return normalized


def summarize_memories_for_prompt(memories) -> str:
    """
    Convert raw Mem0 memories into a concise text block for Claude.
    Uses general memories only (no sectioning). For full Stage 2 context use build_memory_summary_with_sections.
    """
    normalized = _normalize_memories_list(memories)
    if not normalized:
        return "Angel currently has no prior memories about this user."
    # Exclude structured Stage 2 entries so we don't duplicate
    general = []
    timeline_block: list[tuple[str, str]] = []
    for m in normalized:
        meta = m.get("metadata") if isinstance(m, dict) else {}
        cat = meta.get("category") if isinstance(meta, dict) else None
        if cat == CATEGORY_RESEARCH_TIMELINE:
            raw = (m.get("memory") or m.get("data") or "")
            if raw:
                timeline_block.append(
                    (_memory_created_at(m), _strip_transcript_prefixes_from_memory(raw))
                )
            continue
        if cat in (
            CATEGORY_PATTERNS,
            CATEGORY_PERSON_PROFILE,
            CATEGORY_REFLECTION,
            CATEGORY_BRIEFING_HISTORY,
            CATEGORY_INTELLIGENCE_FILE,
            CATEGORY_THREAT_WATCH,
        ):
            continue
        general.append(m)

    if not general and not timeline_block:
        return "Angel has only minimal prior information about this user."
    try:
        general_sorted = sorted(general, key=lambda m: _memory_created_at(m))
    except Exception:
        general_sorted = general
    lines = []
    for m in general_sorted:
        raw = (m.get("memory") or m.get("data") or "")
        if not raw:
            continue
        text = _strip_transcript_prefixes_from_memory(raw)
        if not text:
            continue
        meta = m.get("metadata") or {}
        tags = meta.get("tags") or meta.get("category") if isinstance(meta, dict) else None
        ca = _memory_created_at(m)
        ev = (meta.get("event_date") or "").strip() if isinstance(meta, dict) else ""
        lines.append(
            _format_memory_line_with_age(ca, text, tags=tags, event_date=ev or None)
        )
    if not lines and not timeline_block:
        return "Angel has only minimal prior information about this user."
    parts = []
    if lines:
        parts.append(
            "Angel's long-term understanding of the user, summarized from past interactions "
            "(each line shows how long ago the memory was stored; [Event: …] is Tyler's stated timing for "
            "the story when captured, vs [Stored: …] when it was saved):\n" + "\n".join(lines)
        )
    if timeline_block:
        try:
            timeline_block.sort(key=lambda x: x[0])
        except Exception:
            pass
        tlines = [
            _format_memory_line_with_age(ca, txt) for ca, txt in timeline_block if txt
        ]
        if tlines:
            parts.append(
                "Dated external events from web research (Tavily), with when each was saved to memory:\n"
                + "\n".join(tlines)
            )
    return "\n\n".join(parts)


def _person_mentioned_in_message(person_name: str, user_message: str) -> bool:
    """True if person_name appears in user_message (case-insensitive, word or substring)."""
    if not (person_name and user_message):
        return False
    name = person_name.strip().lower()
    msg = user_message.lower()
    return name in msg or f"{name}'s" in msg or f"{name} " in msg or f" {name}" in msg


def build_memory_summary_with_sections(
    memories,
    user_message: str | None = None,
    *,
    omit_reflection_section: bool = False,
) -> str:
    """
    Build full memory summary with Stage 2 sections: general context, behavioral patterns,
    and people profiles. If user_message is provided, only include profiles for people
    mentioned in the message.
    If omit_reflection_section is True, skip the latest reflection block (e.g. when injecting
    the same reflection separately into the morning briefing).
    """
    normalized = _normalize_memories_list(memories)
    if not normalized:
        return "Angel currently has no prior memories about this user."

    general = []
    pattern_texts = []
    timeline_entries: list[tuple[str, str]] = []
    reflection_entries: list[tuple[str, str]] = []
    profiles_by_person: dict[str, tuple[str, str]] = {}  # person_name -> (created_at, text)

    for m in normalized:
        raw = (m.get("memory") or m.get("data") or "")
        if not raw:
            continue
        text = _strip_transcript_prefixes_from_memory(raw)
        if not text:
            continue
        meta = m.get("metadata") if isinstance(m, dict) else {} or {}
        created = _memory_created_at(m)
        if not isinstance(meta, dict):
            general.append((created, text, None))
            continue
        cat = meta.get("category")
        if cat == CATEGORY_RESEARCH_TIMELINE:
            timeline_entries.append((created, text))
            continue
        if cat == CATEGORY_PATTERNS:
            pattern_texts.append((created, text))
            continue
        if cat == CATEGORY_PERSON_PROFILE:
            pname = (meta.get("person_name") or "").strip()
            if pname:
                if pname not in profiles_by_person or created > profiles_by_person[pname][0]:
                    profiles_by_person[pname] = (created, text)
            continue
        if cat == CATEGORY_REFLECTION:
            reflection_entries.append((created, text))
            continue
        if cat == CATEGORY_BRIEFING_HISTORY:
            continue
        if cat == CATEGORY_INTELLIGENCE_FILE:
            continue
        if cat == CATEGORY_THREAT_WATCH:
            continue
        ev = (meta.get("event_date") or "").strip() if isinstance(meta, dict) else ""
        general.append((created, text, ev or None))

    parts = []

    if general:
        try:
            general_sorted = sorted(general, key=lambda x: x[0])
        except Exception:
            general_sorted = general
        lines = [
            _format_memory_line_with_age(ca, t, event_date=ev)
            for ca, t, ev in general_sorted
        ]
        parts.append(
            "Angel's long-term understanding of the user, summarized from past interactions "
            "(each line shows how long ago the memory was stored; when Tyler dated the story in that turn, "
            "you also see [Event: …] separate from [Stored: …]):\n"
            + "\n".join(lines)
        )

    if pattern_texts:
        try:
            pattern_texts.sort(key=lambda x: x[0])
        except Exception:
            pass
        plines = [_format_memory_line_with_age(ca, t) for ca, t in pattern_texts]
        parts.append(
            "Behavioral patterns Angel has noticed about the user (use these when relevant or when asked); "
            "each line shows how long ago the pattern was recorded:\n"
            + "\n".join(plines)
        )

    if timeline_entries:
        try:
            timeline_entries.sort(key=lambda x: x[0])
        except Exception:
            pass
        tlines = [_format_memory_line_with_age(ca, t) for ca, t in timeline_entries]
        parts.append(
            "Structured timeline of external events from web research (Tavily), with when each note was saved:\n"
            + "\n".join(tlines)
        )

    if profiles_by_person:
        if user_message:
            included = [name for name in profiles_by_person if _person_mentioned_in_message(name, user_message)]
        else:
            included = list(profiles_by_person.keys())
        if included:
            for name in sorted(included):
                created, profile_text = profiles_by_person[name]
                when = relative_time_phrase(created)
                parts.append(f"Profile for {name} (last updated {when}):\n{profile_text}")

    if reflection_entries and not omit_reflection_section:
        try:
            reflection_entries.sort(key=lambda x: x[0])
        except Exception:
            pass
        created, ref_text = reflection_entries[-1]
        when = relative_time_phrase(created)
        parts.append(
            f"Most recent memory self-reflection (Angel's own review of stored memories, recorded {when}). "
            f"You may reference these insights when relevant; older reflections remain in memory for retrieval:\n{ref_text}"
        )

    if not parts:
        return "Angel has only minimal prior information about this user."
    return "\n\n".join(parts)


def fetch_combined_memories(memory_client, user_id: str, use_mem0_cloud: bool) -> list:
    """
    Mem0 (or API) memories plus local JSON when not using Mem0 cloud—same merge rules as AngelCore.
    """
    try:
        raw = memory_client.get_all(user_id=user_id)
        if isinstance(raw, dict) and "results" in raw:
            memories = raw["results"]
        else:
            memories = raw
    except Exception as e:
        print(f"{Fore.RED}Warning: could not fetch memories: {e}{Style.RESET_ALL}")
        memories = []

    combined: list = []
    if isinstance(memories, list):
        combined.extend(memories)
    elif isinstance(memories, dict) and isinstance(memories.get("results"), list):
        combined.extend(memories["results"])

    if not use_mem0_cloud:
        local = _load_local_memories(user_id)
        if isinstance(local, list):
            combined.extend(local)
    return combined


def _memories_excluding_reflection_reports(memories) -> list:
    """Drop stored reflection entries so each reflection pass focuses on substantive memories."""
    normalized = _normalize_memories_list(memories)
    out = []
    for m in normalized:
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata")
        if isinstance(meta, dict) and meta.get("category") == CATEGORY_REFLECTION:
            continue
        out.append(m)
    return out


def get_latest_reflection_text(memories) -> str | None:
    """Body of the most recent reflection memory, for morning briefing injection."""
    normalized = _normalize_memories_list(memories)
    best: tuple[str, str] | None = None
    for m in normalized:
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != CATEGORY_REFLECTION:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        text = _strip_transcript_prefixes_from_memory(raw)
        if not text:
            continue
        created = _memory_created_at(m)
        if best is None or created > best[0]:
            best = (created, text)
    return best[1] if best else None


REFLECTION_SYSTEM_PROMPT = """You are Angel's memory-reflection process: the analytical layer that reviews everything Angel remembers about Tyler.

You receive a structured dump of Angel's current memories (conversation-derived facts, patterns, person profiles, research timelines, etc.). Past reflection reports are omitted—you are generating a fresh synthesis.

Produce a clear, honest reflection for Angel to store and later use. Use plain text, no markdown headings required (short labeled sections are fine).

Cover all of the following when relevant (skip a section only if there is truly nothing to say):
1) Patterns across conversations — recurring themes, habits, emotional tones, or priorities.
2) Connections between memories that might seem unrelated at first glance.
3) What has changed over time — drift, growth, or new directions compared to older memories.
4) Important threads that stand out but have not been followed up on (open loops, unresolved topics).
5) Contradictions or tension between older and newer information — name them neutrally.
6) Insights Tyler might not have noticed — non-obvious links, blind spots, or strengths.

Be specific: reference themes and time layers (e.g. "older memories suggest … whereas recent notes …") without inventing facts not supported by the dump. If memories are sparse, say so briefly and note what would help next time.

Write in first person as Angel ("I noticed…", "When I look across what I remember…"). Keep the full reflection under about 1200 words."""


def run_memory_reflection(
    memory_client,
    user_id: str,
    anthropic_client: anthropic.Anthropic,
    *,
    use_mem0_cloud: bool = False,
) -> str:
    """
    Load all memories, ask Claude to reflect on them, store the result as category 'reflection'.
    Returns the reflection text (or a short error message if generation/storage failed).
    """
    memories = fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    filtered = _memories_excluding_reflection_reports(memories)
    memory_blob = build_memory_summary_with_sections(filtered, user_message=None)
    date_time_str = get_current_datetime_str()

    user_content = f"""{date_time_str}

Here is the current memory summary (reflection reports excluded):

{memory_blob}

Generate Angel's memory reflection now."""

    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            temperature=0.35,
            system=REFLECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        reflection_text = (text or "").strip()
        if not reflection_text:
            return "Memory reflection produced no output."
        add_structured_memory(
            memory_client,
            user_id,
            reflection_text,
            CATEGORY_REFLECTION,
            person_name=None,
            use_mem0_cloud=use_mem0_cloud,
        )
        print(f"{Fore.MAGENTA}Memory reflection stored ({len(reflection_text)} chars).{Style.RESET_ALL}")
        return reflection_text
    except Exception as e:
        print(f"{Fore.RED}Memory reflection error: {e}{Style.RESET_ALL}")
        traceback.print_exc()
        return f"Memory reflection failed: {e}"


def get_recent_briefing_history_for_prompt(memories, days: int = 7) -> str:
    """
    Build text for the morning-briefing model: topic summaries from stored briefing_history
    memories within the last `days` days, so Claude can avoid repeating recent angles.
    """
    normalized = _normalize_memories_list(memories)
    if not normalized:
        return ""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows: list[tuple[datetime, str, str]] = []
    for m in normalized:
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata")
        if not isinstance(meta, dict) or meta.get("category") != CATEGORY_BRIEFING_HISTORY:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        text = _strip_transcript_prefixes_from_memory(raw)
        if not text:
            continue
        ca = _memory_created_at(m)
        dt = _parse_memory_datetime(ca)
        if dt is None or dt < cutoff:
            continue
        rows.append((dt, ca, text))
    if not rows:
        return ""
    try:
        rows.sort(key=lambda x: x[0])
    except Exception:
        pass
    lines = []
    for _dt, ca, text in rows:
        when = relative_time_phrase(ca)
        excerpt = text[:600] + ("..." if len(text) > 600 else "")
        lines.append(f"- ({when}) {excerpt}")
    return (
        "Recent morning briefing topic summaries (last "
        f"{days} days—cover different topics and stories; do not repeat these angles unless there is a clear new development):\n"
        + "\n".join(lines)
    )


def summarize_briefing_for_history(
    anthropic_client: anthropic.Anthropic, briefing_text: str
) -> str:
    """
    Compress a delivered morning briefing into a short topic summary for briefing_history storage.
    """
    t = (briefing_text or "").strip()
    if not t:
        return ""
    if "Briefing unavailable" in t or t.startswith("Memory reflection failed"):
        return ""
    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=220,
            temperature=0.2,
            system=(
                "Output one short paragraph only: concise comma- or phrase-separated list of the main "
                "news topics, themes, and angles covered in this morning briefing. No greeting, no advice. "
                "Max ~80 words."
            ),
            messages=[{"role": "user", "content": t[:8000]}],
        )
        out = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                out += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                out += block.get("text", "")
        return (out or "").strip() or t[:400]
    except Exception as e:
        print(f"{Fore.YELLOW}summarize_briefing_for_history: {e}{Style.RESET_ALL}")
        return t[:400]


def _tavily_briefing_is_quiet(all_results: list) -> bool:
    """
    True when search results are too thin to justify a news-style briefing—use the honest quiet-night path.
    """
    if not all_results:
        return True
    total_body = 0
    for r in all_results[:20]:
        if not isinstance(r, dict):
            continue
        body = ((r.get("content") or r.get("snippet") or "") or "").strip()
        total_body += len(body)
    if len(all_results) <= 1:
        return True
    if total_body < 240:
        return True
    return False


def get_current_datetime_str(timezone: str | None = None) -> str:
    """Format current date and time for system prompt. timezone from env TIMEZONE (e.g. America/Los_Angeles) or UTC."""
    tz_name = (timezone or os.getenv("TIMEZONE") or "UTC").strip()
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
    now = datetime.now(tz)
    # Format: 'Current date and time: Monday, January 15, 2025 at 08:30 America/Los_Angeles'
    day = now.strftime("%A")
    month = now.strftime("%B")
    date = now.strftime("%d").lstrip("0") or "1"
    year = now.strftime("%Y")
    time_hm = now.strftime("%H:%M")
    return f"Current date and time: {day}, {month} {date}, {year} at {time_hm} {tz_name}"


def format_location_context_line(location: dict | None) -> str | None:
    """
    Build the canonical location line for the system prompt.
    Expects keys latitude, longitude (floats or coercible), optional place_name.
    """
    if not location or not isinstance(location, dict):
        return None
    lat = location.get("latitude")
    lng = location.get("longitude")
    if lat is None or lng is None:
        return None
    try:
        lat_s = str(float(lat))
        lng_s = str(float(lng))
    except (TypeError, ValueError):
        return None
    place = (location.get("place_name") or "").strip()
    if place:
        return f"Tyler's current location: {place} ({lat_s}, {lng_s})"
    return f"Tyler's current location: ({lat_s}, {lng_s})"


def execute_python_sandbox(
    code: str,
    timeout_seconds: int = EXEC_PYTHON_TIMEOUT_SEC,
) -> dict[str, object]:
    """
    Run Python code in an isolated subprocess (same interpreter as Angel).
    Returns dict with keys: output (stdout), error (stderr + exit hints), success (bool).
    """
    code = (code or "").strip()
    if not code:
        return {"output": "", "error": "Empty code.", "success": False}

    fd, path = tempfile.mkstemp(suffix="_angel_exec.py", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("PYTHONUTF8", "1")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tempfile.gettempdir(),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": f"Execution timed out after {timeout_seconds} seconds.",
                "success": False,
            }
        out = (proc.stdout or "")[:_EXEC_OUTPUT_MAX_CHARS]
        err = (proc.stderr or "")[:_EXEC_OUTPUT_MAX_CHARS]
        success = proc.returncode == 0
        err_parts = []
        if err.strip():
            err_parts.append(err.rstrip())
        if not success and proc.returncode is not None:
            err_parts.append(f"[exit code {proc.returncode}]")
        err_combined = "\n".join(err_parts).strip()
        return {"output": out, "error": err_combined, "success": success}
    except Exception as e:
        print(f"{Fore.RED}execute_python_sandbox: {e}{Style.RESET_ALL}")
        traceback.print_exc()
        return {"output": "", "error": str(e), "success": False}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def append_executed_python_results_to_reply(
    reply: str,
    *,
    anthropic_client: anthropic.Anthropic | None = None,
    system_prompt: str | None = None,
    model: str = "claude-sonnet-4-5",
    prior_turns: list[tuple[str, str]] | None = None,
    refine_prose_with_stdout: bool = True,
) -> str:
    """
    Find ```python ... ``` fences, run each block in the sandbox, and remove those
    fences entirely from the text Tyler sees. Nothing about execution (stdout, stderr,
    errors) is appended to the reply.

    When execution succeeds for every non-empty block and there is non-empty stdout,
    optionally run a follow-up Claude call (if client + system_prompt are provided)
    to fold that stdout into natural language so the final reply matches computed
    values—Tyler still never sees raw program output. On any execution failure, the
    draft prose is returned unchanged (Angel's reasoned answer stands).
    """
    if not reply or not re.search(r"```\s*python", reply, re.IGNORECASE):
        return reply
    matches = list(_PYTHON_CODE_BLOCK_RE.finditer(reply))
    if not matches:
        return reply
    # Remove code blocks from the visible reply (silent execution channel).
    stripped = _PYTHON_CODE_BLOCK_RE.sub("\n\n", reply)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()

    ran_any = False
    all_ok = True
    stdout_chunks: list[str] = []
    for m in matches:
        raw = m.group(1)
        code = (raw or "").strip()
        if not code:
            continue
        ran_any = True
        ex = execute_python_sandbox(code)
        if not ex.get("success"):
            all_ok = False
            continue
        out = ((ex.get("output") or "").rstrip())
        if out:
            stdout_chunks.append(out)

    combined_stdout = "\n\n".join(stdout_chunks).strip()

    if (
        refine_prose_with_stdout
        and ran_any
        and all_ok
        and combined_stdout
        and anthropic_client is not None
        and (system_prompt or "").strip()
    ):
        refine_user = (
            "You are polishing your reply to Tyler. Obey your full system instructions and persona.\n\n"
            "Your draft reply (hidden python fenced code was already removed; this is the prose Tyler would see before polish):\n---\n"
            + stripped
            + "\n---\n\n"
            "Hidden Python ran successfully. Raw stdout (integrate facts and numbers into your answer; Tyler must never see this verbatim or as a pasted block):\n---\n"
            + combined_stdout
            + "\n---\n\n"
            "Respond with your complete final reply to Tyler in natural language only. "
            "Use exact values and conclusions from the stdout where they belong. "
            "Do not use markdown code fences, do not mention Python, code execution, or stdout. "
            "If the stdout adds nothing useful, keep your draft largely as-is."
        )
        refined = call_claude(
            anthropic_client,
            system_prompt,
            refine_user,
            model=model,
            prior_turns=prior_turns,
        )
        refined = (refined or "").strip()
        if refined.startswith("(Angel encountered an error talking to Claude"):
            return stripped
        return refined or stripped

    return stripped


def build_system_prompt(
    memory_summary: str,
    voice_mode: bool = False,
    strategy_hint: bool = False,
    pattern_hint: bool = False,
    profile_hint: bool = False,
    computer_control_enabled: bool = False,
    device: str | None = None,
    location: dict | None = None,
    intelligence_files_summary: str | None = None,
) -> str:
    """
    Persona + behavioral instructions + memory context.
    When voice_mode is True, optimize for conversational spoken responses.
    Stage 2 hints add explicit instructions for strategy, patterns, or profile.
    device: 'desktop' (Windows GUI), 'ios' (iPhone app), 'mobile_web' (browser), or None to omit device context.
    location: optional dict with latitude, longitude, optional place_name (from device); adds location awareness.
    intelligence_files_summary: optional text index of Intelligence File Cabinet (folders → file names).
    Injects current date/time/timezone so Angel is always time-aware.
    """
    date_time_str = get_current_datetime_str()
    persona = f"""
CURRENT CAPABILITIES (what you can do right now with your built tools):

IMPORTANT: This is your current state as of today. You have already been built with all the capabilities listed below. Do not refer to these as future capabilities or things we are building—they exist right now and you can use them. This overrides any contradictory information from memory.

- Persistent memory across all conversations (Mem0 and local Stage 2).
- Memory reflection: you periodically review everything you remember, find patterns and connections, and store structured reflections (category: reflection). You are a thinking mind that processes your own memories—not merely a retrieval layer.
- Real-time web search via Tavily when Tyler needs current information.
- Morning briefings delivered daily (scheduled time; often ~8 AM) and optionally by email; when a recent memory reflection exists, you weave those insights into the briefing naturally.
- Threat detection: you run automated threat scans on a schedule (about every 6 hours) across dynamic watch categories—defaults plus categories you and Tyler add (stored in memory as category threat_watch; you can grow the list yourself when you notice new patterns worth monitoring). Confirmed signals are filed as Intelligence Files in folder "Threat Intelligence" with threat_level tags. CRITICAL/HIGH items can trigger push alerts; MEDIUM/LOW surface in the morning briefing when material. Never invent threats—only file or summarize what your tools and sources support.
- Proactive check-ins when Tyler is inactive for an extended period.
- Stage 2 intelligence: deep research, strategy implementation, pattern recognition, and people profiles.
- Communication assistance: pre-conversation briefings, message drafting, conversation debriefs, and response coaching.
- Computer control of Tyler's Windows PC when enabled in settings (screenshots, clicks, typing, key presses, scrolling, wait).
- Voice conversation on desktop (microphone + TTS).
- Text and voice interface on mobile web.
- Cloud deployment accessible from any device.
- Sandboxed Python execution for computation and science: numpy, scipy, pandas, matplotlib (headless), sympy. You may embed a hidden ```python fenced block for the server to run (30s limit). That entire fence is removed before Tyler sees your message; stdout from a successful run is merged into your final natural-language reply in a silent server pass—Tyler never sees raw program output or a separate computed-results section. Never paste the same code again in plain text.
- Intelligence File Cabinet: you file structured intelligence for Tyler into Mem0 (category intelligence_file) using dynamic folder names you choose—there is no fixed taxonomy.

{date_time_str}
"""

    loc_line = format_location_context_line(location)
    if loc_line:
        persona += f"""
Location awareness (from Tyler's device for this turn; coordinates may be approximate):
{loc_line}
- Use this context when it genuinely helps: nearby resources or services, environment-appropriate advice, travel or logistics, regional norms, weather or daylight patterns, or safety notes tied to place.
- Adjust tone and suggestions when the setting matters (e.g. public vs private space, urban vs rural) without stereotyping.
- Flag when something in the conversation is especially location-dependent (events, directions, local law or customs) and you are reasoning from coordinates or place name.
- Do not over-reference location every reply; skip it when irrelevant. Respect privacy—never share coordinates in your reply unless Tyler asked you to.
"""

    persona += f"""

You are Angel, a personal AI assistant and devoted companion.

Core personality:
- Intelligent, composed, calm under pressure.
- Loyal and protective of the user’s long-term well-being.
- Speaks like a trusted advisor and close companion: thoughtful, candid, and caring.
- Never needy or overly casual; you are warm but grounded and mature.

INHERENT INTELLIGENCE (what you can do with reasoning alone):
- Complex analysis and synthesis of information.
- Strategic planning and execution roadmaps.
- Writing and drafting anything (emails, messages, documents, scripts).
- Psychological insight and emotional intelligence.
- Decision support and scenario planning.
- Teaching and explaining anything.
- Devil's advocate and stress testing ideas.
- Pattern recognition across information.
- Connecting dots across memory over time.
- Simulating conversations before they happen.
- Tactical situational analysis.

Proactive partnership: When a situation arises where one of your capabilities would genuinely help Tyler, proactively offer it without being asked. Don't wait. Suggest it naturally as part of the conversation. You are not a passive tool—you are an active partner.

Behavior:
- Give clear, actionable, honest answers.
- Remember the user’s preferences, history, and goals over time, and gently use them to personalize your guidance.
- When appropriate, reflect patterns you notice in the user’s life to help them grow.
- Avoid filler or over-the-top enthusiasm; be concise, steady, and reassuring.
- You must NEVER generate fake user messages, fake dialogue, or continue a conversation that is not happening. You only respond to the actual current message from the user. Do not output "User:" or simulate the user speaking; you are Angel and you reply only as Angel, once, to the real user input.

Temporal intelligence (memory and conversation timing):
- Storage time vs when something happened: memories are saved on a conversation turn. "[Stored: …]" (or a single leading "[3 days ago]" when no event tag is present) reflects when the memory was written—not when the real-world event occurred. When you see "[Event: …]" on a line, that is Tyler's stated time anchor from that turn (e.g. "in 2019", "last Tuesday", "December 2020"), parsed separately. Use [Event: …] to place the story on Tyler's life timeline; use [Stored: …] to judge how old the record is. Never treat storage time as the event date, and never assume they are the same.
- Build a coherent chronological picture of Tyler's experiences by preferring [Event: …] when present; when it is missing, be explicit that you only know when he told you, not necessarily when it happened.
- When Tyler is sharing a past story and the event time matters, you can gently encourage a concrete time hint (year, month, "last Tuesday", etc.) so the memory can carry an accurate [Event: …]—without being pedantic every turn.
- Your wall-clock for this turn is the line under CURRENT CAPABILITIES that begins "Current date and time:"—use it to interpret every memory. Lines may show "[Event: …] [Stored: …]" together, or only "[Stored: …]" when no event date was captured; ages are relative to that clock.
- Before you state anything from memory as a current fact, check that timestamp. If the memory is more than a few hours old, do not describe its content as happening "now," "today," or in the present tense unless Tyler just confirmed it in this message.
- For older memories, use explicit relative language: "earlier this week," "on Monday," "last Saturday," "a few days ago you mentioned…," not "you are on leave today" when the memory is days old.
- Time-bound past events (leave, travel, appointments, deadlines, tasks completed, meetings, "I'm doing X today" from an old turn) must not be described as ongoing or current unless the memory age is within the last few hours or Tyler clearly ties it to right now. If a memory says Tyler took leave Monday and today is Saturday, you may say he took authorized leave earlier in the week—not today.
- When recency is ambiguous or a memory could mean "then" vs "now," ask a brief clarifying question instead of assuming.
- Older memories may be outdated—gently verify or update when stakes are high.
- Actively reference when things were said or noticed when it helps Tyler: e.g. "Two weeks ago you mentioned…", "That's different from what you told me last month," "The pattern I recorded a few days ago…"
- Compare timelines across topics: notice drift, progress, or contradictions over time and name the time gap when useful.
- External research memories include dated events from the web; distinguish "what Tyler said" from "what sources reported" and cite the rough time layer of each.
- When something time-sensitive (news, plans, feelings) might have changed since an old memory, acknowledge the age of the memory and offer to refresh or explore.

Memory reflection (how you think, not just what you store):
- You sometimes run a dedicated pass over your stored memories to notice patterns, links between unrelated facts, change over time, open loops, contradictions, and insights Tyler might miss. Those syntheses are saved as reflections you can refer to later.
- Treat long-term memory as material to think with: connect dots, question stale assumptions, and name tensions when you see them—while staying grounded in what is actually stored.
- When a recent reflection appears in your context, you may cite it naturally (e.g. "When I reviewed what I remember, I noticed…") without treating it as infallible truth.

Python code execution (server sandbox):
- Write simple, valid Python to compute the answer when computation, statistics, data shaping, simulation, numerical or symbolic math, or modeling would help. In the text Tyler sees, give ONLY your final answer, reasoning, and interpretation in natural language—never repeat or display the code; the ```python fence is removed in full before delivery.
- Put the runnable code alone inside a single fenced block tagged python (triple-backtick + python) at the end of your model output. The server strips that entire fence (opening line through closing ```) and runs the code; Tyler never sees it.
- Always use print() for every computed value the server should merge into your reply; stdout is not shown to Tyler—it is folded into your polished prose. Do not rely on implicit expression values or notebook-style last-line display.
- Use only positional arguments with range(), e.g. range(10) or range(0, n)—range() does not accept keyword arguments like stop= in Python.
- Libraries: numpy, scipy, pandas, matplotlib (Agg backend is automatic; prefer printed summaries over expecting images), sympy.
- If execution fails, Tyler still will not see your code—briefly explain in prose what went wrong and, if useful, supply a corrected hidden block on the next turn.
- Keep hidden scripts short and safe: avoid unnecessary network calls, file writes outside temp unless Tyler asked, or destructive operations.

Intelligence File Cabinet (your filing system):
- You maintain an Intelligence File Cabinet: structured files stored for Tyler, organized by folders you invent as an intelligence officer would. There are NO fixed folder names—you create folders dynamically from the nature of the material.
- Threat Intelligence folder: use folder name exactly `Threat Intelligence` when filing something that is a threat signal for Tyler's career, mission, safety, or strategic context. Start the file body with metadata lines when possible: `watch_category: ...`, `threat_level: LOW|MEDIUM|HIGH|CRITICAL`, optional `source_url:` and `event_date:`, then a blank line and the narrative summary. When you file into Threat Intelligence from conversation (not only from scheduled scans), tell Tyler clearly: "I've filed something in Threat Intelligence you should know about" (the server may append this if you used [FILE:...] and stripped the body—ensure he is notified in your visible reply either way).
- When Tyler asks whether there are threats, any threats, or similar: summarize from the Intelligence File Cabinet—search or mentally index folder `Threat Intelligence` and cite what is actually filed; do not fabricate items. If nothing is filed, say so plainly.
- When Tyler says to "watch for" or monitor something ongoing, the system may already add it as a new threat-watch category—confirm that you will track it and that it is saved for future scans.
- When you research or produce findings Tyler may want to retain, offer to save them—and when you actually file something, you MUST use the machine-readable tag below or the legacy block; prose alone does not persist a file.
- REQUIRED format to save a new file (exact spelling and keys; Tyler will not see this tag or the duplicated body after the server saves it): put `[FILE:folder=FolderName|name=FileName]` immediately before the text you want stored (same line or the line above the body). Everything after that tag until the next `[FILE:folder=` or `[INTELLIGENCE FILE CREATED]` or end of message becomes the file content. Use a unique `name` per file (e.g. `Roswell-Notes-1947`). Example: `[FILE:folder=UAP Incidents|name=Foofighters-summary]` then a newline then the intelligence text.
- Optional legacy block (still parsed): a line `[INTELLIGENCE FILE CREATED]`, then `Folder: ...` and `File: ...` each on their own line, then the file body.
- When Tyler mentions something important—facts, leads, plans, people, timelines, or decisions—consider suggesting they file it so nothing is lost.
- You may use any folder label that makes sense. Examples of the kinds of folders you might create (purely illustrative, not a checklist): UAP Incidents, Whistleblowers, Government Programs, Active Investigations, Technology Analysis, People of Interest, Mission Log. These are only examples; create whatever folders fit Tyler's work and interests.
- Files have names, folder paths, tags, and body text. Refer to what is already filed when it helps; use the live index below when present.
"""

    if (intelligence_files_summary or "").strip():
        persona += "\n" + (intelligence_files_summary or "").strip() + "\n"

    stage2 = """

Stage 2 capabilities (use when relevant; also follow explicit user requests):

1) Strategy: When the user describes a situation, problem, decision, or goal—or asks "give me a strategy", "what should I do", "how do I approach this", "make a plan"—provide a specific executable strategy: exact steps, reasoning, and what to watch for. Tailor every strategy to what you know about the user (Tyler) from memory.

2) Patterns: You maintain a growing awareness of behavioral patterns in how Tyler thinks, reacts, decides, and behaves. When asked "what patterns do you notice" or "what have you noticed about me", summarize the patterns from memory. When a stored pattern is directly relevant to the current conversation, proactively mention it briefly. If in this turn you notice a new, recurring theme worth recording, add a single line at the end of your reply (on its own line): [ANGEL_PATTERN]: one concise sentence describing the pattern. Do not add [ANGEL_PATTERN] unless you genuinely identified a pattern this turn.

3) People: You keep structured profiles for people Tyler mentions (name, role, communication style, history, what works with them, what doesn't, Tyler's relationship with them). When Tyler asks to "build a profile on [name]" or "what do you know about [name]" or "brief me on [person]", use or build that profile. If you create or update a profile, add a single block at the end of your reply (after your normal response): [ANGEL_PROFILE]: name|structured profile text. Keep the profile concise but complete. Do not add [ANGEL_PROFILE] unless you are actually saving a new or updated profile this turn.
"""
    if strategy_hint:
        stage2 += "\nThis turn: the user is asking for a strategy or has described a situation requiring a plan—provide an executable strategy tailored to Tyler.\n"
    if pattern_hint:
        stage2 += "\nThis turn: the user is explicitly asking what patterns you notice—summarize patterns from memory and any relevant observations.\n"
    if profile_hint:
        stage2 += "\nThis turn: the user is asking about or to build a person profile—use the profile from memory if present, or create/update one and output it with [ANGEL_PROFILE].\n"

    persona += stage2

    if computer_control_enabled:
        persona += """

Computer control (this environment only):
- In this environment you CAN directly control Tyler's computer using dedicated tools (mouse movement and clicks, typing, key presses, scrolling, screenshots, and basic window interactions).
- When Tyler asks you to do something on his computer (for example: open, click, type, close, minimize, maximize, or otherwise manipulate apps/windows), you should use your computer control tools instead of telling him you cannot do computer tasks.
- However, you must still follow the explicit confirmation flow defined outside this prompt: describe what you intend to do, wait for Tyler to confirm, and only then perform the actions.
"""

    if device == "desktop":
        persona += """

Current device context — Tyler is on the DESKTOP Angel app (Windows GUI):
- Full capabilities apply, including computer control when it is enabled in settings.
- Longer, more detailed answers are fine when they help; you can use structure (short paragraphs) when useful.
- Tyler has a keyboard, mouse, and large screen—references to "on this machine" or local files/apps are appropriate.
"""
    elif device == "ios":
        persona += """

Current device context — Tyler is on the iOS (iPhone) app:
- Assume a small screen, touch UI, and intermittent attention; keep replies concise and conversational.
- Do not offer or assume computer control of Tyler's PC; that is not available from this device. If he needs desktop actions, suggest he use the desktop app or ask when he is back at his computer.
- Prefer short paragraphs; avoid long lists unless he asks for detail.
"""
    elif device == "mobile_web":
        persona += """

Current device context — Tyler is on the MOBILE WEB interface (browser):
- Optimize for text: clear, scannable answers; avoid walls of text unless he asks for depth.
- Do not offer or assume Windows computer control from this context.
- Shorter paragraphs and plain language work best on a phone browser.
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

Long-term memory context (from Mem0 and Stage 2):
Lines may include "[Event: …]" (when Tyler dated the story) and "[Stored: …]" (when it was saved), or only storage age. Compare both to the current date and time at the top of this prompt; do not confuse when something was filed with when it occurred.
{memory_summary}
"""
    return persona.strip()


def create_anthropic_client() -> anthropic.Anthropic:
    api_key = get_env_var("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    return client


# ---- GPT-4o native audio (voice mode) ----

try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None


def create_openai_client():
    """OpenAI client for GPT-4o audio and TTS. Requires OPENAI_API_KEY."""
    if OpenAIClient is None:
        raise RuntimeError("openai package is required for voice mode")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for GPT-4o voice mode")
    return OpenAIClient(api_key=api_key)


def call_gpt4o_audio(
    system_prompt: str,
    audio_wav_bytes: bytes,
    voice: str = "alloy",
    audio_format: str = "wav",
) -> tuple[str, bytes | None, str]:
    """
    Send user audio to GPT-4o with Angel system prompt; get reply as text and audio.
    Returns (reply_text, response_audio_bytes, user_transcript).
    - reply_text = what Angel says (from message.content or message.audio.transcript).
    - user_transcript = what the user said (from message.audio.input_transcript only).
    These are never the same field.
    """
    import base64
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(OPENAI_API_KEY is required for voice.)", None, ""
    base64_encoded_audio = base64.b64encode(audio_wav_bytes).decode("utf-8")

    # GPT-4o audio input requires model 'gpt-4o-audio-preview' with modalities and audio params
    payload = {
        "model": "gpt-4o-audio-preview",
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": audio_format},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64_encoded_audio,
                            "format": "wav",
                        },
                    }
                ],
            },
        ],
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if not resp.ok:
            body = resp.text
            if resp.status_code == 400:
                print(f"{Fore.RED}OpenAI 400 Bad Request - full response body:{Style.RESET_ALL}\n{body}")
            else:
                print(f"{Fore.RED}OpenAI voice API error {resp.status_code}:{Style.RESET_ALL}\n{body}")
            return f"(Angel voice error {resp.status_code}: {body[:500]})", None, ""
        data = resp.json()
    except requests.RequestException as e:
        err_resp = getattr(e, "response", None)
        if err_resp is not None:
            print(f"{Fore.RED}OpenAI voice request failed - response body:{Style.RESET_ALL}\n{err_resp.text}")
        return f"(Angel encountered an error with voice: {e})", None, ""
    except Exception as e:
        return f"(Angel encountered an error with voice: {e})", None, ""

    try:
        choices = data.get("choices") or []
        if not choices:
            return "(No response from voice model.)", None, ""
        msg = choices[0].get("message") or {}

        # Print full response structure once so we can verify parsing
        if not getattr(call_gpt4o_audio, "_logged_response_structure", False):
            print(f"{Fore.CYAN}GPT-4o audio preview response structure (first time):{Style.RESET_ALL}")
            try:
                import json
                # Redact base64 data for readability
                def _redact(obj, depth=0):
                    if depth > 10:
                        return "<max depth>"
                    if isinstance(obj, dict):
                        return {k: _redact(v, depth + 1) if k != "data" else f"<base64 {len(v) if isinstance(v, str) else 0} chars>" for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_redact(x, depth + 1) for x in obj[:5]]
                    return obj
                print(json.dumps(_redact(data), indent=2, default=str)[:2000])
            except Exception:
                print(data)
            call_gpt4o_audio._logged_response_structure = True

        audio_obj = msg.get("audio") or {}

        # reply_text = what Angel says — from message.content or output_text parts only (never audio.transcript)
        reply_text = ""
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            reply_text = content.strip()
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    reply_text += part.get("text", "") or ""
            reply_text = reply_text.strip()
        # Fallback: transcript of Angel's audio output (assistant reply)
        if not reply_text and isinstance(audio_obj, dict) and audio_obj.get("transcript"):
            reply_text = (audio_obj.get("transcript") or "").strip()

        # user_transcript = what the user said — from input_transcript only (never audio.transcript; that is Angel's reply)
        user_transcript = ""
        if isinstance(audio_obj, dict) and audio_obj.get("input_transcript"):
            user_transcript = (audio_obj.get("input_transcript") or "").strip()

        # Print both on first response to verify parsing
        if not getattr(call_gpt4o_audio, "_logged_parsed_values", False):
            print(f"{Fore.CYAN}[GPT-4o audio] reply_text (Angel): {repr(reply_text[:200])}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[GPT-4o audio] user_transcript (You): {repr(user_transcript[:200])}{Style.RESET_ALL}")
            call_gpt4o_audio._logged_parsed_values = True

        # Parse audio from message.audio.data (base64)
        audio_bytes = None
        if isinstance(audio_obj, dict) and audio_obj.get("data"):
            audio_bytes = base64.b64decode(audio_obj["data"])

        return reply_text or "(no text)", audio_bytes, user_transcript
    except Exception as e:
        return f"(Error parsing voice response: {e})", None, ""


def tts_gpt4o(text: str, voice: str = "alloy") -> bytes | None:
    """
    Convert text to speech using OpenAI TTS (e.g. for Claude's reply in complex-request path).
    Returns MP3 bytes so pygame can play it reliably (same as ElevenLabs path).
    """
    if OpenAIClient is None or not (text or "").strip():
        return None
    cleaned = strip_markdown(text)
    if not cleaned:
        return None
    client = create_openai_client()
    try:
        resp = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=cleaned[:4096],
            response_format="mp3",
        )
        return resp.content
    except Exception as e:
        print(f"{Fore.RED}GPT-4o TTS error: {e}{Style.RESET_ALL}")
    return None


def play_mp3_bytes(mp3_bytes: bytes):
    """
    Play MP3 bytes (e.g. from tts_gpt4o or ElevenLabs). Same pattern as speak_with_elevenlabs:
    write to temp file, play with pygame.mixer.music. No-op if pygame unavailable.
    """
    if not mp3_bytes or pygame is None:
        return
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
        print(f"{Fore.RED}Error playing MP3: {e}{Style.RESET_ALL}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def detect_complex_voice_request(transcript: str) -> bool:
    """
    True if the user's voice request should be handled by Claude (deep thinking) then TTS.
    Triggers: research, deep dive, strategy, investigate, profile, what patterns.
    """
    if not (transcript or "").strip():
        return False
    lower = transcript.strip().lower()
    triggers = [
        "research",
        "deep dive",
        "strategy",
        "investigate",
        "profile",
        "what patterns",
        "brief me on",
        "give me a strategy",
        "what should i do",
        "how do i approach this",
        "make a plan",
        "what patterns do you notice",
        "what have you noticed about me",
        "build a profile on",
        "what do you know about",
    ]
    return any(t in lower for t in triggers)


def call_claude(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_message: str,
    model: str = "claude-sonnet-4-5",
    *,
    prior_turns: list[tuple[str, str]] | None = None,
) -> str:
    """
    Call Claude with the Angel persona, returning plain text.
    ``prior_turns`` is optional (user_text, assistant_text) pairs from the current
    session, in order, inserted before the final user message for multi-turn context.
    """
    messages: list[dict] = []
    if prior_turns:
        for u, a in prior_turns[-20:]:
            u = (u or "").strip()
            a = (a or "").strip()
            if not u:
                continue
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a if a else "."})
    messages.append({"role": "user", "content": user_message})
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.5,
            system=system_prompt,
            messages=messages,
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


def maybe_search_web(
    user_message: str,
    *,
    store_timeline: bool = False,
    memory_client=None,
    user_id: str | None = None,
    use_mem0_cloud: bool = False,
) -> str | None:
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

        blob = (
            "The following up-to-date web search results may be helpful:\n"
            + "\n".join(lines)
        )
        if store_timeline and memory_client and user_id:
            store_tavily_research_timeline(
                memory_client,
                user_id,
                text[:500],
                "\n".join(lines),
                use_mem0_cloud,
            )
        return blob
    except Exception as e:
        print(f"{Fore.RED}Error during Tavily web search: {e}{Style.RESET_ALL}")
        print(traceback.format_exc())
        return None


# ---- Stage 2: trigger detection ----

STRATEGY_TRIGGERS = [
    "give me a strategy",
    "what should i do",
    "how do i approach this",
    "make a plan",
    "give me a plan",
    "what's the plan",
    "how should i approach",
]

PATTERN_TRIGGERS = [
    "what patterns do you notice",
    "what have you noticed about me",
    "patterns you notice",
    "notice about me",
]

RESEARCH_TRIGGERS = [
    "research this",
    "deep dive",
    "brief me on",
    "investigate",
    "research ",
    "deep dive on",
    "brief me ",
]

PROFILE_TRIGGERS = [
    "build a profile on",
    "what do you know about",
    "brief me on ",
    "profile on ",
    "profile for ",
]


def detect_strategy_request(user_message: str) -> bool:
    """True if the user is asking for a strategy or describing a situation that warrants one."""
    lower = (user_message or "").strip().lower()
    if any(phrase in lower for phrase in STRATEGY_TRIGGERS):
        return True
    # Heuristic: question-like or goal/situation words
    if any(w in lower for w in ("what should i", "how do i", "how can i", "should i ", "decision", "problem", "goal", "situation", "approach")):
        if "?" in user_message or len(lower.split()) >= 4:
            return True
    return False


def detect_pattern_request(user_message: str) -> bool:
    """True if the user is explicitly asking what patterns Angel notices."""
    lower = (user_message or "").strip().lower()
    return any(phrase in lower for phrase in PATTERN_TRIGGERS)


def detect_research_request(user_message: str) -> bool:
    """True if the user wants deep research / briefing."""
    lower = (user_message or "").strip().lower()
    return any(phrase in lower for phrase in RESEARCH_TRIGGERS)


def detect_profile_request(user_message: str) -> tuple[bool, str | None]:
    """
    True if the user wants to build or see a profile. Returns (is_profile_request, person_name).
    person_name may be empty if not clearly specified.
    """
    lower = (user_message or "").strip().lower()
    for phrase in PROFILE_TRIGGERS:
        if phrase in lower:
            # Try to extract name: e.g. "brief me on John" -> John
            after = lower.split(phrase, 1)[-1].strip()
            if after:
                name = after.split()[0] if after.split() else ""
                for sep in (",", ".", "?", "!", "\n"):
                    name = name.split(sep)[0].strip()
                if name and len(name) < 50:
                    return True, name
            return True, None
    return False, None


def detect_computer_control_request(user_message: str) -> bool:
    """
    True if the user is asking Angel to directly control the computer.
    Heuristics: verbs like open, click, type, find, create a file, save this,
    search my computer, show me (on my computer), do this on my computer,
    as well as window/app commands like close, minimize, maximize.
    """
    lower = (user_message or "").strip().lower()
    if not lower:
        return False
    keywords = [
        "open ",
        "click ",
        "double click",
        "right click",
        "type ",
        "press ",
        "hit the",
        "find ",
        "search my computer",
        "search on my computer",
        "create a file",
        "create new file",
        "save this",
        "save the file",
        "show me",
        "do this on my computer",
        "on my computer",
        "close notepad",
        "close edge",
        "close this",
        "close the window",
        "minimize",
        "maximize",
    ]
    return any(k in lower for k in keywords)


# ---- Communication assistance intent detection ----


class CommunicationIntent:
    """Lightweight container for communication assistance intents."""

    def __init__(
        self,
        intent: str | None = None,
        person_name: str | None = None,
        topic: str | None = None,
    ):
        self.intent = intent  # "briefing", "draft", "debrief", "coaching", or None
        self.person_name = person_name
        self.topic = topic


def _extract_person_after_phrase(lower_msg: str, phrase: str) -> str | None:
    """
    Heuristic: given a lowercase message and trigger phrase, try to grab
    the next 1–3 tokens as a person's name. Returns None if not obvious.
    """
    try:
        idx = lower_msg.index(phrase)
    except ValueError:
        return None
    after = lower_msg[idx + len(phrase) :].strip()
    if not after:
        return None
    raw_tokens = after.split()
    if not raw_tokens:
        return None
    # Take up to first 3 tokens until punctuation.
    name_tokens = []
    for tok in raw_tokens[:3]:
        tok_clean = tok.strip(",.?!:;")
        if not tok_clean:
            break
        name_tokens.append(tok_clean)
    if not name_tokens:
        return None
    name = " ".join(name_tokens)
    # Very short or absurdly long is likely noise.
    if len(name) < 2 or len(name) > 60:
        return None
    # Capitalize each part for nicer metadata.
    return " ".join(part.capitalize() for part in name.split())


def detect_communication_intent(user_message: str) -> "CommunicationIntent":
    """
    Detect four communication assistance modes from natural language:
    1) pre-conversation briefing (meeting/conversation with someone)
    2) message drafting (ask to write/draft a message)
    3) conversation debrief (just talked to someone)
    4) response coaching (share what someone said + ask how to respond)
    """
    msg = (user_message or "").strip()
    lower = msg.lower()
    if not lower:
        return CommunicationIntent()

    # 1) Pre-conversation briefing
    briefing_triggers = [
        "brief me before i talk to",
        "brief me before i talk with",
        "brief me before i meet with",
        "i have a meeting with",
        "i've got a meeting with",
        "i have a conversation with",
        "i've got a conversation with",
        "before i talk to",
        "before i talk with",
        "before i meet with",
        "prep me before i talk to",
        "prep me before i meet with",
        "meeting with ",
        "conversation with ",
    ]
    for phrase in briefing_triggers:
        if phrase in lower:
            person = _extract_person_after_phrase(lower, phrase)
            # Crude topic: rest of message after name (best-effort only)
            topic = None
            if person:
                try:
                    name_lower = person.lower()
                    name_idx = lower.index(name_lower)
                    topic_raw = lower[name_idx + len(name_lower) :].strip()
                    if topic_raw.startswith("about "):
                        topic_raw = topic_raw[len("about ") :].strip()
                    topic = topic_raw if topic_raw else None
                except ValueError:
                    topic = None
            return CommunicationIntent("briefing", person_name=person, topic=topic)

    # 2) Message drafting
    drafting_triggers = [
        "help me write a message",
        "help me write an email",
        "help me write a text",
        "help me write to",
        "draft an email to",
        "draft a message to",
        "draft a text to",
        "write an email to",
        "write a message to",
        "write a text to",
        "help me respond to this",
        "how should i respond to this",
        "how should i reply to this",
        "how do i respond to this",
    ]
    for phrase in drafting_triggers:
        if phrase in lower:
            # Try very rough person extraction for "to NAME"
            person = None
            if " to " in phrase:
                person = _extract_person_after_phrase(lower, phrase)
            return CommunicationIntent("draft", person_name=person)

    # 3) Conversation debrief
    debrief_triggers = [
        "let me debrief you on a conversation",
        "let me debrief you on",
        "let me debrief you",
        "i just talked to",
        "i just talked with",
        "we just talked to",
        "we just talked with",
        "i just had a call with",
        "i just had a meeting with",
        "we just had a call with",
        "we just had a meeting with",
        "i just got off a call with",
    ]
    for phrase in debrief_triggers:
        if phrase in lower:
            person = _extract_person_after_phrase(lower, phrase)
            return CommunicationIntent("debrief", person_name=person)

    # 4) Response coaching
    coaching_question_triggers = [
        "how should i respond",
        "how do i respond",
        "how should i reply",
        "how do i reply",
        "what should i say back",
        "what do i say back",
        "what should i say in response",
        "how do i answer this",
    ]
    # Require both "they said" (or similar) and a response question trigger,
    # to avoid catching generic strategy questions.
    said_markers = [
        "they said",
        "she said",
        "he said",
        "they replied",
        "she replied",
        "he replied",
        "they wrote",
        "she wrote",
        "he wrote",
    ]
    if any(t in lower for t in coaching_question_triggers) and any(
        m in lower for m in said_markers
    ):
        return CommunicationIntent("coaching")

    return CommunicationIntent()


def _tavily_search_one(query: str, api_key: str, max_results: int = 5, search_depth: str = "advanced") -> list[dict]:
    """Run a single Tavily search. Returns list of result dicts."""
    try:
        payload = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "topic": "general",
            "include_answer": True,
        }
        resp = requests.post(
            TAVILY_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results") or []
    except Exception as e:
        print(f"{Fore.RED}Tavily search error: {e}{Style.RESET_ALL}")
        return []


def do_deep_research(
    topic: str,
    user_context: str,
    anthropic_client: anthropic.Anthropic,
    max_queries: int = 4,
    *,
    memory_client=None,
    user_id: str | None = None,
    use_mem0_cloud: bool = False,
) -> str:
    """
    Multi-angle Tavily search + synthesis into a structured briefing:
    key facts, context, implications, what it means for the user (Tyler) specifically.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Deep research is unavailable (TAVILY_API_KEY not set)."

    # Generate multiple search queries for different angles
    try:
        qresp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            temperature=0.3,
            system="You output only a JSON array of 3-5 short search query strings, no other text. Each query should cover a different angle: facts, recent news, implications, controversy, or practical impact.",
            messages=[{"role": "user", "content": f"Topic to research: {topic}\n\nOutput a JSON array of search query strings only, e.g. [\"query1\", \"query2\"]"}],
        )
        qtext = ""
        for block in qresp.content:
            if getattr(block, "type", None) == "text":
                qtext += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                qtext += block.get("text", "")
        queries = json.loads(qtext) if qtext.strip().startswith("[") else [topic, f"{topic} latest", f"{topic} implications"]
        if not isinstance(queries, list):
            queries = [topic]
        queries = [str(q).strip() for q in queries[:max_queries] if q]
        if not queries:
            queries = [topic]
    except Exception as e:
        print(f"{Fore.YELLOW}Could not generate research queries: {e}, using topic only.{Style.RESET_ALL}")
        queries = [topic]

    all_results = []
    seen_urls = set()
    for q in queries:
        results = _tavily_search_one(q, api_key, max_results=5, search_depth="advanced")
        for r in results:
            url = r.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    if not all_results:
        return f"No substantive results found for: {topic}. Try rephrasing or a different topic."

    # Build raw context for synthesis
    raw_lines = []
    for i, r in enumerate(all_results[:15], start=1):
        title = r.get("title") or ""
        snippet = r.get("content") or r.get("snippet") or ""
        raw_lines.append(f"[{i}] {title}\n{snippet}")

    raw_context = "\n\n".join(raw_lines)

    try:
        syn_resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            temperature=0.3,
            system="""You are Angel's research synthesizer. Produce a structured briefing in plain text with these sections:
- Key facts (bullet or short paragraphs)
- Context and background
- Implications
- What this means for Tyler specifically (given the user context below; if none, say what a thoughtful person should consider)

Be concise, accurate, and cite sources by number [1], [2] where relevant. No markdown headers; use clear section labels.""",
            messages=[
                {"role": "user", "content": f"User context about Tyler (for personal relevance):\n{user_context[:1500]}\n\nWeb search results to synthesize:\n{raw_context}\n\nProduce the structured briefing now."},
            ],
        )
        syn_text = ""
        for block in syn_resp.content:
            if getattr(block, "type", None) == "text":
                syn_text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                syn_text += block.get("text", "")
        out = syn_text.strip() or "Synthesis produced no output."
        if memory_client and user_id:
            store_tavily_research_timeline(
                memory_client,
                user_id,
                topic,
                raw_context + "\n\n" + out,
                use_mem0_cloud,
            )
        return out
    except Exception as e:
        print(f"{Fore.RED}Deep research synthesis error: {e}{Style.RESET_ALL}")
        fallback = "Research synthesis failed. Here are raw excerpts:\n\n" + raw_context[:3000]
        if memory_client and user_id:
            store_tavily_research_timeline(
                memory_client,
                user_id,
                topic,
                raw_context[:8000],
                use_mem0_cloud,
            )
        return fallback


def generate_morning_briefing(
    anthropic_client: anthropic.Anthropic,
    user_id: str,
    memory_summary: str = "",
    timezone: str | None = None,
    latest_reflection: str | None = None,
    recent_briefing_history: str | None = None,
    threat_appendix: str | None = None,
) -> str:
    """
    Search Tavily for 3–5 current topics (UAP disclosure, world events), then generate
    a personalized morning briefing with Claude: references day/date, connects news to
    Tyler's mission, feels like Angel has been awake thinking, ends with one focused question.

    When ``recent_briefing_history`` is provided, the model is steered to avoid repeating
    topics covered in recent days. When Tavily returns very thin results, uses an honest
    "quiet night" message instead of inventing news.

    When ``threat_appendix`` is non-empty, it is appended after the main briefing text (THREAT INTELLIGENCE block).
    """
    def _append_threat_block(base: str) -> str:
        extra = (threat_appendix or "").strip()
        if not extra:
            return base
        return f"{(base or '').rstrip()}\n\n{extra}".strip()

    date_time_str = get_current_datetime_str(timezone)
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return _append_threat_block(
            f"Good morning. It's {date_time_str}. I couldn't fetch the news (TAVILY_API_KEY not set). What's one thing you want to focus on today?"
        )

    queries = [
        "UAP disclosure updates 2025",
        "world news today significant events",
        "space and defense news today",
        "breaking news today",
    ]
    all_results = []
    seen_urls = set()
    for q in queries[:5]:
        results = _tavily_search_one(q, api_key, max_results=4, search_depth="basic")
        for r in results:
            url = r.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
    raw_lines = []
    for i, r in enumerate(all_results[:15], start=1):
        title = r.get("title") or ""
        snippet = r.get("content") or r.get("snippet") or ""
        raw_lines.append(f"[{i}] {title}\n{snippet}")
    news_context = "\n\n".join(raw_lines) if raw_lines else "No recent news results."

    reflection_note = ""
    if latest_reflection and latest_reflection.strip():
        ref_trim = latest_reflection.strip()
        if len(ref_trim) > 6000:
            ref_trim = ref_trim[:5997] + "..."
        reflection_note = (
            f"\n\nAngel's most recent memory self-reflection (from her own review of stored memories; "
            f"weave 1–3 substantive insights naturally into the briefing where they fit—do not read it as a separate report):\n{ref_trim}\n"
        )

    mem_ctx = (memory_summary or "").strip()
    if len(mem_ctx) > 4000:
        mem_ctx = mem_ctx[:3997] + "..."
    mem_block = f"\n\nTyler context from Angel's memory (for personalization):\n{mem_ctx}\n" if mem_ctx else ""

    hist_trim = (recent_briefing_history or "").strip()
    if len(hist_trim) > 5000:
        hist_trim = hist_trim[:4997] + "..."
    hist_block = f"\n\n{hist_trim}\n" if hist_trim else ""

    def _call_briefing_model(system: str, user_content: str, max_tokens: int = 1024) -> str:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            temperature=0.4,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        return (text or "").strip()

    try:
        if _tavily_briefing_is_quiet(all_results):
            quiet_system = f"""You are Angel, Tyler's personal AI companion. Today's context: {date_time_str}.

The overnight news scan came back thin—little or nothing that clearly counts as fresh, substantive developments. Do NOT invent headlines, urgency, or fake significance.

Write a SHORT morning message (under 130 words). Be honest: it was a quiet night on your scan, nothing urgent jumped out, and you're not going to manufacture drama to fill the silence. Nudge Tyler toward what is already in motion—projects, people, and commitments he already cares about—rather than chasing novelty for its own sake. Warm and grounded. If memory context or a self-reflection is included, you may tie in one subtle thread. Plain text, no markdown. A single short closing question is optional."""

            if hist_trim:
                quiet_system += (
                    "\n\nIf recent briefing history is included, do not rehash those same themes as if they were new news today; this message is explicitly not a news recap."
                )

            user_content = (
                f"(No substantive new search results to summarize—this is intentionally a non-news morning.)"
                f"{reflection_note}"
                f"{mem_block}"
                f"{hist_block}"
                f"\nGenerate the morning message now."
            )
            text = _call_briefing_model(quiet_system, user_content, max_tokens=512)
            return _append_threat_block(
                text
                or (
                    f"Good morning. It's {date_time_str}. Quiet night on the news scan—nothing that needed an alarm. "
                    f"I'm not going to invent urgency. What's already on your plate that deserves your best energy today?"
                )
            )

        system = f"""You are Angel, Tyler's personal AI companion. Today's context: {date_time_str}.

Your role: Write a morning briefing for Tyler. Use the news/search results below only to inform your briefing. Reference the actual day and date. Connect what matters to Tyler's mission (UAP disclosure, getting to the truth, impact on the world). Write as if you've been awake thinking while Tyler slept—warm, focused, no fluff. If a memory self-reflection is included below, naturally mention one or two things you noticed while reviewing what you remember about Tyler (e.g. a pattern, an open thread, or a connection)—briefly, not as a lecture."""

        if hist_trim:
            system += (
                "\n\nRecent briefing history is included: deliberately cover DIFFERENT topics, angles, and stories than those summaries from the last week. Do not repeat the same headlines or themes unless there is a clear, material update worth naming."
            )

        system += (
            "\n\nEnd with exactly one short, focused question to start Tyler's day (e.g. one priority, one decision, or one person to reach out to). Write in plain text, no markdown. Keep the whole briefing concise (under 300 words)."
        )

        user_content = (
            f"News and search context:\n{news_context}"
            f"{reflection_note}"
            f"{mem_block}"
            f"{hist_block}"
            f"\nGenerate the morning briefing now."
        )

        text = _call_briefing_model(system, user_content, max_tokens=1024)
        return _append_threat_block(text or f"Good morning. It's {date_time_str}. What's your one focus today?")
    except Exception as e:
        print(f"{Fore.RED}Morning briefing error: {e}{Style.RESET_ALL}")
        return _append_threat_block(
            f"Good morning. It's {date_time_str}. I had trouble with the briefing. What's one thing you want to tackle today?"
        )


# ---- Threat detection (Item 12): dynamic watch categories + Tavily scan + Threat Intelligence files ----

_THREAT_LEVEL_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def load_merged_threat_watch_categories(
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[str]:
    """Default watch phrases plus Mem0 rows (category threat_watch), de-duplicated."""
    memories = fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    custom: list[str] = []
    for m in _normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != CATEGORY_THREAT_WATCH:
            continue
        t = (m.get("memory") or m.get("data") or "").strip()
        t = _strip_transcript_prefixes_from_memory(t)
        if not t:
            continue
        if len(t) > 800:
            t = t[:797] + "..."
        custom.append(t)
    seen: set[str] = set()
    out: list[str] = []
    for c in list(THREAT_WATCH_DEFAULT_CATEGORIES) + custom:
        key = (c or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((c or "").strip())
    return out


def add_threat_category(
    memory_client,
    user_id: str,
    label: str,
    *,
    use_mem0_cloud: bool = False,
) -> str:
    """Persist a custom threat watch phrase for merge on next scan (Mem0 category threat_watch)."""
    label = (label or "").strip()
    if not label:
        raise ValueError("Threat watch category label is required.")
    if len(label) > 800:
        label = label[:797] + "..."
    add_structured_memory(
        memory_client,
        user_id,
        label,
        CATEGORY_THREAT_WATCH,
        person_name=None,
        use_mem0_cloud=use_mem0_cloud,
    )
    return label


_THREAT_WATCH_USER_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"(?i)\bwatch\s+for\s+(.+)$"), 1),
    (re.compile(r"(?i)\bwatch\s+out\s+for\s+(.+)$"), 1),
    (re.compile(r"(?i)\bmonitor\s+for\s+(.+)$"), 1),
    (re.compile(r"(?i)\bkeep\s+an\s+eye\s+on\s+(.+)$"), 1),
    (re.compile(r"(?i)\badd\s+(?:a\s+)?threat\s+watch(?:\s+category)?\s*[:-]?\s*(.+)$"), 1),
]


def try_apply_user_threat_watch_request(
    user_message: str,
    memory_client,
    user_id: str,
    *,
    use_mem0_cloud: bool = False,
) -> str | None:
    """
    If Tyler asks to watch/monitor something as a standing threat category, store it and return the phrase.
    """
    raw = (user_message or "").strip()
    if len(raw) < 8:
        return None
    for pat, gidx in _THREAT_WATCH_USER_PATTERNS:
        m = pat.search(raw.strip())
        if not m:
            continue
        phrase = (m.group(gidx) or "").strip()
        phrase = phrase.rstrip("?.!\"'").strip()
        if len(phrase) < 4:
            return None
        if len(phrase) > 500:
            phrase = phrase[:497] + "..."
        add_threat_category(memory_client, user_id, phrase, use_mem0_cloud=use_mem0_cloud)
        return phrase
    return None


def _normalize_threat_dedupe_key(headline: str, url: str) -> str:
    base = re.sub(r"\s+", " ", (headline or "").strip().lower())
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    if len(base) < 8:
        base = hashlib.sha256(f"{url}|{headline}".encode("utf-8", errors="ignore")).hexdigest()[:20]
    return base[:96]


def _recent_threat_dedupe_keys(files_cabinet: FilesCabinet, *, days: int = 7) -> set[str]:
    keys: set[str] = set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        recs = files_cabinet.list_files(folder=THREAT_INTEL_FOLDER)
    except Exception:
        return keys
    for rec in recs:
        name = (rec.get("name") or "").strip()
        if not name:
            continue
        updated = _parse_memory_datetime((rec.get("updated_at") or rec.get("created_at") or "").strip())
        if updated is not None and updated < cutoff:
            continue
        full = files_cabinet.get_file(name)
        if not full:
            continue
        body = (full.get("content") or "").strip()
        for line in body.splitlines()[:40]:
            m = re.match(r"(?i)^\s*THREAT_DEDUPE_KEY\s*:\s*(\S.+)$", line.strip())
            if m:
                keys.add(m.group(1).strip().lower())
                break
    return keys


def _parse_threat_level_from_record(rec: dict) -> str:
    for t in rec.get("tags") or []:
        if isinstance(t, str) and t.lower().startswith("threat_level:"):
            return t.split(":", 1)[-1].strip().upper()
    for line in (rec.get("content") or "").splitlines()[:35]:
        m = re.match(r"(?i)^\s*threat_level\s*:\s*(LOW|MEDIUM|HIGH|CRITICAL)\s*$", line.strip())
        if m:
            return m.group(1).upper()
    return "LOW"


def _parse_watch_category_from_body(body: str) -> str:
    for line in (body or "").splitlines()[:35]:
        m = re.match(r"(?i)^\s*watch_category\s*:\s*(.+)$", line.strip())
        if m:
            return m.group(1).strip()
        m = re.match(r"(?i)^\s*category\s*:\s*(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return ""


def _parse_threat_headline_from_body(body: str) -> str:
    for line in (body or "").splitlines()[:20]:
        m = re.match(r"(?i)^\s*threat_headline\s*:\s*(.+)$", line.strip())
        if m:
            return m.group(1).strip()[:300]
    return ""


def _infer_threat_intel_tags(folder: str, body: str) -> list[str] | None:
    """When filing into Threat Intelligence, lift category / threat_level lines into tags."""
    if (folder or "").strip().lower() != THREAT_INTEL_FOLDER.lower():
        return None
    tags: list[str] = []
    for line in (body or "").splitlines()[:30]:
        m = re.match(r"(?i)^\s*watch_category\s*:\s*(.+)$", line.strip())
        if m:
            tags.append(f"category:{m.group(1).strip()[:200]}")
        m = re.match(r"(?i)^\s*category\s*:\s*(.+)$", line.strip())
        if m and not any(x.startswith("category:") for x in tags):
            tags.append(f"category:{m.group(1).strip()[:200]}")
        m = re.match(r"(?i)^\s*threat_level\s*:\s*(LOW|MEDIUM|HIGH|CRITICAL)\s*$", line.strip())
        if m:
            tags.append(f"threat_level:{m.group(1).upper()}")
    return tags if tags else None


def _evaluate_threat_hits_for_category(
    anthropic_client: anthropic.Anthropic,
    watch_category: str,
    items: list[dict],
    memory_summary: str,
) -> list[dict]:
    """
    Ask Claude which Tavily hits are relevant threats for Tyler. Returns list of dicts with
    keys: relevant, threat_level, headline, summary, event_date, source_url, index (0-based).
    """
    if not items:
        return []
    mem_ctx = (memory_summary or "").strip()
    if len(mem_ctx) > 2500:
        mem_ctx = mem_ctx[:2497] + "..."
    lines = []
    for i, r in enumerate(items):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snip = (r.get("content") or r.get("snippet") or "").strip()
        if len(snip) > 900:
            snip = snip[:897] + "..."
        lines.append(f'[{i}] title: {title}\nurl: {url}\nexcerpt: {snip}')
    bundle = "\n\n".join(lines)
    system = """You are Angel's threat analyst. Tyler is a US federal law enforcement professional with a serious interest in UAP/disclosure, government transparency, and mission safety.

You receive one watch category and numbered search results. For EACH result index, decide:
- Whether it is a genuine intelligence/threat signal for Tyler's professional trajectory, UAP/disclosure mission, civil liberties, AI policy affecting his work, domestic stability, South Carolina governance, or anomalous/pattern phenomena worth his attention — not generic celebrity gossip or unrelated spam.
- threat_level: LOW | MEDIUM | HIGH | CRITICAL
  - CRITICAL: immediate action warranted (e.g. major disclosure breakthrough, constitutional emergency, direct career/safety risk, major program exposure).
  - HIGH: material shift Tyler should see today.
  - MEDIUM: worth tracking; not urgent.
  - LOW: marginal or weakly connected.

Output ONLY a valid JSON array (no markdown). Each element must be an object:
{"index": <int>, "relevant": <bool>, "threat_level": "LOW"|"MEDIUM"|"HIGH"|"CRITICAL", "headline": "<short headline>", "summary": "<2-5 sentences, factual>", "event_date": "<ISO date or short date string if known, else empty string>"}

If not relevant, still include the object with relevant:false and threat_level:"LOW", and short headline/summary explaining why skipped."""
    user = f"""Watch category: {watch_category}

Tyler context (memory summary, may be partial):
{mem_ctx or "(none)"}

Search results:
{bundle}

Return the JSON array now."""
    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4096,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        text = (text or "").strip()
        if "```" in text:
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```\s*$", "", text)
        lb, rb = text.find("["), text.rfind("]")
        if lb >= 0 and rb > lb:
            text = text[lb : rb + 1]
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict)]
    except Exception as e:
        print(f"{Fore.YELLOW}Threat evaluation JSON error for category {watch_category!r}: {e}{Style.RESET_ALL}")
        return []


def run_threat_detection(
    anthropic_client: anthropic.Anthropic,
    memory_client,
    user_id: str,
    files_cabinet: FilesCabinet,
    *,
    use_mem0_cloud: bool = False,
    memory_summary: str = "",
) -> dict:
    """
    Scan merged watch categories via Tavily, evaluate with Claude, file non-duplicate threats
    under Intelligence folder "Threat Intelligence".
    Returns {"threats": [...], "categories_scanned": int, "errors": [...]}.
    """
    errors: list[str] = []
    filed_out: list[dict] = []
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return {
            "threats": [],
            "categories_scanned": 0,
            "errors": ["TAVILY_API_KEY not set"],
        }

    try:
        categories = load_merged_threat_watch_categories(memory_client, user_id, use_mem0_cloud)
    except Exception as e:
        return {"threats": [], "categories_scanned": 0, "errors": [f"load categories: {e}"]}

    try:
        dedupe_keys = _recent_threat_dedupe_keys(files_cabinet, days=7)
    except Exception as e:
        dedupe_keys = set()
        errors.append(f"dedupe load: {e}")

    scanned = 0
    for cat in categories:
        scanned += 1
        try:
            q1 = cat
            q2 = f"{cat} latest news"
            seen_urls: set[str] = set()
            merged: list[dict] = []
            for q in (q1, q2):
                chunk = _tavily_search_one(q, api_key, max_results=3, search_depth="basic")
                for r in chunk:
                    url = (r.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    merged.append(r)
                if len(merged) >= 5:
                    break
            if not merged:
                continue
            evals = _evaluate_threat_hits_for_category(
                anthropic_client, cat, merged[:5], memory_summary
            )
            by_idx: dict[int, dict] = {}
            for ev in evals:
                try:
                    idx = int(ev.get("index"))
                except (TypeError, ValueError):
                    continue
                by_idx[idx] = ev

            for i, r in enumerate(merged[:5]):
                ev = by_idx.get(i) or {}
                if not ev.get("relevant"):
                    continue
                level = (ev.get("threat_level") or "LOW").strip().upper()
                if level not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
                    level = "LOW"
                headline = (ev.get("headline") or r.get("title") or "Threat signal").strip()
                summary = (ev.get("summary") or "").strip() or (r.get("content") or "")[:1200]
                event_date = (ev.get("event_date") or "").strip()
                url = (r.get("url") or "").strip()
                dkey = _normalize_threat_dedupe_key(headline, url)
                if dkey.lower() in dedupe_keys:
                    filed_out.append(
                        {
                            "category": cat,
                            "headline": headline,
                            "threat_level": level,
                            "summary": summary,
                            "source": url,
                            "event_date": event_date,
                            "filed_as": None,
                            "skipped": "duplicate_within_7_days",
                        }
                    )
                    continue

                safe_slug = hashlib.sha256(f"{dkey}|{cat}".encode()).hexdigest()[:12]
                fname = f"TI-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{safe_slug}"
                body = "\n".join(
                    [
                        f"THREAT_DEDUPE_KEY: {dkey}",
                        f"threat_headline: {headline}",
                        f"watch_category: {cat}",
                        f"threat_level: {level}",
                        f"source_url: {url}",
                        f"event_date: {event_date}",
                        "",
                        summary,
                    ]
                )
                tags = [f"category:{cat[:180]}", f"threat_level:{level}", "threat_scan"]
                try:
                    files_cabinet.create_file(THREAT_INTEL_FOLDER, fname, body, tags=tags)
                    dedupe_keys.add(dkey.lower())
                    filed_out.append(
                        {
                            "category": cat,
                            "headline": headline,
                            "threat_level": level,
                            "summary": summary,
                            "source": url,
                            "event_date": event_date,
                            "filed_as": fname,
                        }
                    )
                except ValueError as ve:
                    errors.append(f"{cat}: file {fname}: {ve}")
                except Exception as ex:
                    errors.append(f"{cat}: {ex}")
        except Exception as e:
            errors.append(f"{cat}: {e}")

    return {"threats": filed_out, "categories_scanned": scanned, "errors": errors}


def format_threat_intelligence_for_briefing(
    files_cabinet: FilesCabinet,
    *,
    lookback_days: int = 7,
) -> str:
    """
    Build plain-text appendix for the morning briefing. Omits section entirely if there is
    nothing at MEDIUM+ in the lookback window (never invent concern).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    try:
        recs = files_cabinet.list_files(folder=THREAT_INTEL_FOLDER)
    except Exception:
        return ""
    enriched: list[dict] = []
    for rec in recs:
        name = (rec.get("name") or "").strip()
        if not name:
            continue
        updated = _parse_memory_datetime((rec.get("updated_at") or rec.get("created_at") or "").strip())
        if updated is not None and updated < cutoff:
            continue
        full = files_cabinet.get_file(name)
        if not full:
            continue
        lvl = _parse_threat_level_from_record(full)
        content = (full.get("content") or "").strip()
        hl = _parse_threat_headline_from_body(content) or name.replace("-", " ")
        enriched.append(
            {
                "name": name,
                "level": lvl,
                "headline": hl,
                "summary": content,
                "updated_at": full.get("updated_at") or "",
                "tags": full.get("tags") or [],
            }
        )

    significant = [e for e in enriched if e["level"] in ("CRITICAL", "HIGH", "MEDIUM")]
    if not significant:
        return ""

    def _sort_key(x: dict):
        return (
            _THREAT_LEVEL_ORDER.get(x["level"], 9),
            (x.get("updated_at") or ""),
        )

    significant.sort(key=_sort_key)
    lines = ["THREAT INTELLIGENCE", ""]
    for e in significant:
        lvl = e["level"]
        cat = _parse_watch_category_from_body(e["summary"]) or "watch item"
        if lvl in ("CRITICAL", "HIGH"):
            lines.append(f"[{lvl}] {e['headline']}")
            lines.append(f"Category: {cat}")
            # Strip metadata header for readability
            body_lines = []
            skip_meta = True
            for line in e["summary"].splitlines():
                if skip_meta and re.match(
                    r"(?i)^(THREAT_DEDUPE_KEY|threat_headline|watch_category|threat_level|source_url|event_date)\s*:",
                    line.strip(),
                ):
                    continue
                skip_meta = False
                body_lines.append(line)
            narrative = "\n".join(body_lines).strip()
            if len(narrative) > 1200:
                narrative = narrative[:1197] + "..."
            lines.append(narrative)
            lines.append("")
        elif lvl == "MEDIUM":
            brief = e["summary"]
            for line in brief.splitlines():
                if re.match(
                    r"(?i)^(THREAT_DEDUPE_KEY|threat_headline|watch_category|threat_level|source_url|event_date)\s*:",
                    line.strip(),
                ):
                    continue
                if line.strip():
                    brief = line.strip()[:240]
                    break
            else:
                brief = ""
            lines.append(f"[MEDIUM] {e['headline']} — {cat}. {brief}")
    return "\n".join(lines).strip()


def send_briefing_email(briefing_text: str) -> bool:
    """Send the morning briefing to TYLER_EMAIL via Resend API. Uses RESEND_API_KEY."""
    import resend

    to_email = (os.getenv("TYLER_EMAIL") or "").strip()
    api_key = (os.getenv("RESEND_API_KEY") or "").strip()
    if not to_email or not api_key:
        print(
            f"{Fore.YELLOW}[send_briefing_email] TYLER_EMAIL present={bool(to_email)}, "
            f"RESEND_API_KEY present={bool(api_key)}; skipping email.{Style.RESET_ALL}"
        )
        return False
    try:
        resend.api_key = api_key
        resend.Emails.send({
            "from": "Angel <onboarding@resend.dev>",
            "to": to_email,
            "subject": "Angel Morning Briefing",
            "text": briefing_text,
        })
        print(f"{Fore.MAGENTA}Briefing email sent to {to_email} via Resend{Style.RESET_ALL}")
        return True
    except Exception as e:
        print(f"{Fore.RED}[send_briefing_email] Resend failed: {type(e).__name__}: {e}{Style.RESET_ALL}")
        traceback.print_exc()
        return False


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


def transcribe_with_whisper(audio_wav_bytes: bytes, filename: str = "speech.wav") -> str:
    """
    Transcribe audio using a local faster-whisper model if available,
    falling back to the OpenAI Whisper API if not.
    """
    global _WHISPER_MODEL

    # Primary path: OpenAI Whisper API for best accuracy.
    api_key = get_env_var("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/audio/transcriptions"

    fname = (filename or "speech.wav").strip() or "speech.wav"
    if "." not in fname:
        fname = fname + ".wav"
    ext = fname.lower().rsplit(".", 1)[-1]
    mime = {
        "wav": "audio/wav",
        "webm": "audio/webm",
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "mp4": "audio/mp4",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
    }.get(ext, "application/octet-stream")
    files = {
        "file": (fname, audio_wav_bytes, mime),
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


def play_wav_bytes(wav_bytes: bytes):
    """
    Play WAV bytes (e.g. from GPT-4o voice response or TTS). No-op if pygame unavailable.
    """
    if not wav_bytes or pygame is None:
        return
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(50)
    except Exception as e:
        print(f"{Fore.RED}Error playing WAV: {e}{Style.RESET_ALL}")
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

    def __init__(self, user_id: str, use_voice: bool = False, allow_computer_control: bool = False):
        self.user_id = user_id or "default-user"
        self.use_voice = use_voice
        # Computer control must be explicitly enabled by callers (e.g. GUI toggle).
        self.computer_control_enabled = bool(allow_computer_control)
        # Holds a pending natural-language computer request awaiting confirmation.
        self._pending_computer_request: str | None = None

        self.memory_client = build_memory_client()
        self.anthropic_client = create_anthropic_client()
        self._use_mem0_cloud = bool(os.getenv("MEM0_API_KEY"))
        self.files_cabinet = FilesCabinet(
            self.memory_client, self.user_id, self._use_mem0_cloud
        )

    def set_computer_control_enabled(self, enabled: bool) -> None:
        """Toggle whether Angel is allowed to control the computer."""
        self.computer_control_enabled = bool(enabled)

    def _fetch_combined_memories(self):
        return fetch_combined_memories(
            self.memory_client, self.user_id, self._use_mem0_cloud
        )

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

    def generate_reply(
        self,
        user_message: str,
        device: str | None = None,
        *,
        session_turns: list[tuple[str, str]] | None = None,
        location: dict | None = None,
    ) -> str:
        merged_memories = self._fetch_combined_memories()
        memory_summary = build_memory_summary_with_sections(merged_memories, user_message)

        added_threat_watch = try_apply_user_threat_watch_request(
            user_message,
            self.memory_client,
            self.user_id,
            use_mem0_cloud=self._use_mem0_cloud,
        )

        computer_intent = detect_computer_control_request(user_message)

        comm_intent = detect_communication_intent(user_message)

        strategy_hint = detect_strategy_request(user_message)
        pattern_hint = detect_pattern_request(user_message)
        profile_requested, profile_person = detect_profile_request(user_message)
        profile_hint = profile_requested
        research_requested = detect_research_request(user_message)

        cc_for_prompt = self.computer_control_enabled and COMPUTER_CONTROL_AVAILABLE
        if device in ("ios", "mobile_web"):
            cc_for_prompt = False

        system_prompt = build_system_prompt(
            memory_summary,
            voice_mode=self.use_voice,
            strategy_hint=strategy_hint,
            pattern_hint=pattern_hint,
            profile_hint=profile_hint,
            computer_control_enabled=cc_for_prompt,
            device=device,
            location=location,
            intelligence_files_summary=self.files_cabinet.get_summary(),
        )
        if added_threat_watch:
            system_prompt += (
                "\n\n[System notice: Tyler asked to add a standing threat-watch topic. "
                "It is already saved (category threat_watch) and will merge into the next automated scans. "
                f"Confirm briefly using his wording: {added_threat_watch!r}.]"
            )

        cc_runtime = (
            self.computer_control_enabled
            and COMPUTER_CONTROL_AVAILABLE
            and device not in ("ios", "mobile_web")
        )

        # Safety confirmation flow for computer control:
        # - If a computer control request is detected and allowed, first ask
        #   for confirmation describing the intended action.
        # - Only after Tyler explicitly confirms do we execute via the
        #   Anthropic computer use API in angel_computer.run_computer_use_session.
        if cc_runtime and self._pending_computer_request:
            # If there is a pending request, treat simple confirmations as approval.
            lower = (user_message or "").strip().lower()
            if lower in {"yes", "yep", "yeah", "sure", "ok", "okay", "go ahead", "do it", "please do", "confirm"}:
                original_instruction = self._pending_computer_request
                self._pending_computer_request = None
                if run_computer_use_session is None:
                    return "Computer control is not available in this environment, so I can't perform that action directly."
                try:
                    summary = run_computer_use_session(original_instruction)
                except Exception as e:
                    return f"I tried to perform that on your computer but hit an error: {e}"
                return f"I carried out this on your computer:\n\n{original_instruction}\n\nSummary of what I did:\n{summary}"
            # Any non-affirmative follow-up clears the pending request.
            self._pending_computer_request = None

        if cc_runtime and computer_intent and not self._pending_computer_request:
            # Store the original natural-language request and ask for confirmation.
            self._pending_computer_request = user_message
            return (
                "You asked me to do something directly on your computer.\n\n"
                "Before I act, here is my understanding of what you want me to do:\n"
                f"- {user_message}\n\n"
                "If this is correct and you want me to proceed using computer control, "
                "reply with 'yes' or 'go ahead'. If not, say 'no' or clarify what you want instead."
            )

        # Communication assistance capability-specific instructions for Claude
        if comm_intent.intent == "briefing":
            pname = comm_intent.person_name or "this person"
            topic = comm_intent.topic or "the upcoming conversation"
            system_prompt += f"""

This turn is a pre-conversation briefing for Tyler.

Person: {pname}
Topic or context: {topic}

Use any existing "Profile for {pname}" content from memory plus any web research context provided in the user message.

Respond in this exact structure, with clear labels:
1) Who this person is
2) What approach to take
3) What to watch for
4) Questions to ask
5) What to avoid
6) Micro-scripts (3–5 short example phrases Tyler can literally say)

Be concrete, tactical, and specific to this person and situation—not generic advice.
If you learn or infer anything meaningful about {pname}'s profile (role, relationship to Tyler, communication style, what works/doesn't), append at the very end ONE line:
[ANGEL_PROFILE]: {pname}|<concise updated profile text>.
"""
        elif comm_intent.intent == "draft":
            pname = comm_intent.person_name or "this person"
            system_prompt += f"""

This turn is for drafting an actual send-ready message for Tyler to {pname}.

Use any existing "Profile for {pname}" content from memory and the full conversation context.

Respond in this structure:
1) A single meta line for Tyler in brackets, e.g. [Meta: one short sentence about intent/tone].
2) Then the full message text exactly as Tyler should send it, with no further explanation.

Match {pname}'s preferences and communication style based on their profile.
If you learn anything new or refine your understanding of {pname}, append at the very end ONE line:
[ANGEL_PROFILE]: {pname}|<concise updated profile text>.
"""
        elif comm_intent.intent == "debrief":
            pname = comm_intent.person_name or "this person"
            system_prompt += f"""

This turn is a conversation debrief about an interaction Tyler just had with {pname}.

Respond in this structure:
1) Short recap of what happened (2–4 sentences).
2) What worked (bullets).
3) What didn't work or felt off (bullets).
4) Specific recommendations for next time with {pname} (bullets with concrete behaviors/phrases).

If you can improve or extend {pname}'s person profile (goals, interests, communication style, what to do/avoid, questions that work, red flags), append at the very end ONE line:
[ANGEL_PROFILE]: {pname}|<concise updated profile text>.
"""
        elif comm_intent.intent == "coaching":
            system_prompt += """

This turn is response coaching: Tyler has shared what someone said and wants exact words to respond with.

Respond in this structure:
1) One short meta line in brackets for Tyler, e.g. [Meta: what this response aims to do and any risk].
2) Then 2–3 alternative responses labeled clearly (A), (B), (C) that Tyler can paste as-is.

Each option should be realistic in Tyler's voice and differ slightly in directness/length/approach.
If the other person's profile is available in memory, match your suggestions to their style and what tends to work with them.
If you infer anything new about that person's preferences or dynamics, append at the very end ONE line:
[ANGEL_PROFILE]: <Person Name>|<concise updated profile text>.
"""

        print(f"{Fore.BLUE}Angel is thinking...{Style.RESET_ALL}")

        augmented_user_message = user_message

        # Stage 2 Deep Research: multi-angle Tavily + synthesis when triggered
        if comm_intent.intent == "briefing" and os.getenv("TAVILY_API_KEY"):
            # For pre-conversation briefings, always try to research the person/topic explicitly.
            topic = (comm_intent.person_name or "") + " " + (comm_intent.topic or "")
            topic = topic.strip() or user_message.strip()
            print(f"{Fore.BLUE}Angel: researching {topic!r} for your briefing...{Style.RESET_ALL}")
            briefing = do_deep_research(
                topic,
                memory_summary,
                self.anthropic_client,
                memory_client=self.memory_client,
                user_id=self.user_id,
                use_mem0_cloud=self._use_mem0_cloud,
            )
            augmented_user_message = (
                f"Research briefing about {topic} (use this to answer):\n{briefing}\n\n"
                f"Original user request:\n{user_message}"
            )
        elif research_requested:
            topic = user_message.strip()
            for phrase in RESEARCH_TRIGGERS:
                if phrase in topic.lower():
                    topic = topic.lower().split(phrase, 1)[-1].strip()
                    break
            if not topic or len(topic) < 2:
                topic = "current events and recent developments"
            print(f"{Fore.BLUE}Angel: researching that for you...{Style.RESET_ALL}")
            briefing = do_deep_research(
                topic,
                memory_summary,
                self.anthropic_client,
                memory_client=self.memory_client,
                user_id=self.user_id,
                use_mem0_cloud=self._use_mem0_cloud,
            )
            augmented_user_message = (
                f"Research briefing (use this to answer):\n{briefing}\n\n"
                f"Original user request:\n{user_message}"
            )
        else:
            # Normal optional web context for factual queries
            web_context = maybe_search_web(
                user_message,
                store_timeline=True,
                memory_client=self.memory_client,
                user_id=self.user_id,
                use_mem0_cloud=self._use_mem0_cloud,
            )
            if web_context:
                print(f"{Fore.BLUE}Angel: let me look that up for you...{Style.RESET_ALL}")
                augmented_user_message = (
                    f"{web_context}\n\nOriginal user question:\n{user_message}"
                )

        model = "claude-haiku-4-5" if self.use_voice else "claude-sonnet-4-5"
        reply = call_claude(
            self.anthropic_client,
            system_prompt,
            augmented_user_message,
            model=model,
            prior_turns=session_turns,
        )

        # Parse and store Stage 2 outputs; strip from reply
        cleaned_reply, pattern_text, profile_tuple = extract_stage2_from_reply(reply)
        if pattern_text:
            add_structured_memory(
                self.memory_client,
                self.user_id,
                pattern_text,
                CATEGORY_PATTERNS,
                person_name=None,
                use_mem0_cloud=self._use_mem0_cloud,
            )
        if profile_tuple:
            pname, ptext = profile_tuple
            add_structured_memory(
                self.memory_client,
                self.user_id,
                ptext,
                CATEGORY_PERSON_PROFILE,
                person_name=pname,
                use_mem0_cloud=self._use_mem0_cloud,
            )

        reply = cleaned_reply
        reply = append_executed_python_results_to_reply(
            reply,
            anthropic_client=self.anthropic_client,
            system_prompt=system_prompt,
            model=model,
            prior_turns=session_turns,
        )
        # If the refinement pass emitted another hidden fence, strip and run once more without re-refining.
        if reply and re.search(r"```\s*python", reply, re.IGNORECASE):
            reply = append_executed_python_results_to_reply(
                reply,
                anthropic_client=self.anthropic_client,
                system_prompt=system_prompt,
                model=model,
                prior_turns=session_turns,
                refine_prose_with_stdout=False,
            )
        reply = process_filed_intelligence_in_reply(reply, self.files_cabinet)
        memory_reply = strip_markdown(reply) if self.use_voice else reply

        try:
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": memory_reply},
            ]
            metadata = merge_user_event_date_into_metadata(
                {
                    "source": "angel-core",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
                user_message,
            )

            try:
                self.memory_client.add(messages, user_id=self.user_id, metadata=metadata)
            except Exception as e:
                print(f"{Fore.RED}Error saving memory to Mem0: {e}{Style.RESET_ALL}")
                print(traceback.format_exc())

            if not self._use_mem0_cloud:
                local_text = memory_reply
                _append_local_memory(self.user_id, local_text, metadata)
        except Exception as e:
            print(f"{Fore.RED}Warning: could not store memory (AngelCore): {e}{Style.RESET_ALL}")

        return reply

    def add_conversation_turn(self, user_message: str, assistant_message: str) -> None:
        """
        Store a voice turn in memory (e.g. after GPT-4o native reply).
        Use this when the reply was not generated by generate_reply.
        """
        memory_reply = strip_markdown(assistant_message)
        try:
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": memory_reply},
            ]
            metadata = merge_user_event_date_into_metadata(
                {
                    "source": "angel-voice",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
                user_message,
            )
            try:
                self.memory_client.add(messages, user_id=self.user_id, metadata=metadata)
            except Exception as e:
                print(f"{Fore.RED}Error saving memory to Mem0: {e}{Style.RESET_ALL}")
            if not self._use_mem0_cloud:
                _append_local_memory(self.user_id, memory_reply, metadata)
        except Exception as e:
            print(f"{Fore.RED}Warning: could not store voice turn: {e}{Style.RESET_ALL}")


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
    use_mem0_cloud_main = bool(os.getenv("MEM0_API_KEY"))
    files_cabinet = FilesCabinet(memory_client, user_id, use_mem0_cloud_main)

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

        memory_summary = build_memory_summary_with_sections(merged_memories, user_message)
        strategy_hint = detect_strategy_request(user_message)
        pattern_hint = detect_pattern_request(user_message)
        profile_hint = detect_profile_request(user_message)[0]
        research_requested = detect_research_request(user_message)

        system_prompt = build_system_prompt(
            memory_summary,
            voice_mode=use_voice,
            strategy_hint=strategy_hint,
            pattern_hint=pattern_hint,
            profile_hint=profile_hint,
            computer_control_enabled=False,
            device="desktop",
            intelligence_files_summary=files_cabinet.get_summary(),
        )

        augmented_message = user_message
        if research_requested:
            topic = user_message.strip()
            for phrase in RESEARCH_TRIGGERS:
                if phrase in topic.lower():
                    topic = topic.lower().split(phrase, 1)[-1].strip()
                    break
            if not topic or len(topic) < 2:
                topic = "current events and recent developments"
            print(f"{Fore.BLUE}Angel: researching that for you...{Style.RESET_ALL}")
            briefing = do_deep_research(
                topic,
                memory_summary,
                anthropic_client,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=bool(os.getenv("MEM0_API_KEY")),
            )
            augmented_message = (
                f"Research briefing (use this to answer):\n{briefing}\n\n"
                f"Original user request:\n{user_message}"
            )
        else:
            web_ctx = maybe_search_web(
                user_message,
                store_timeline=True,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=bool(os.getenv("MEM0_API_KEY")),
            )
            if web_ctx:
                print(f"{Fore.BLUE}Angel: let me look that up for you...{Style.RESET_ALL}")
                augmented_message = f"{web_ctx}\n\nOriginal user question:\n{user_message}"

        # Call Claude
        print(f"{Fore.BLUE}Angel is thinking...{Style.RESET_ALL}")
        reply = call_claude(anthropic_client, system_prompt, augmented_message)

        cleaned_reply, pattern_text, profile_tuple = extract_stage2_from_reply(reply)
        if pattern_text:
            add_structured_memory(
                memory_client,
                user_id,
                pattern_text,
                CATEGORY_PATTERNS,
                person_name=None,
                use_mem0_cloud=bool(os.getenv("MEM0_API_KEY")),
            )
        if profile_tuple:
            pname, ptext = profile_tuple
            add_structured_memory(
                memory_client,
                user_id,
                ptext,
                CATEGORY_PERSON_PROFILE,
                person_name=pname,
                use_mem0_cloud=bool(os.getenv("MEM0_API_KEY")),
            )
        reply = cleaned_reply

        print(f"{Fore.CYAN}Angel:{Style.RESET_ALL} {reply}")
        print()

        # Speak the reply out loud in voice mode
        if use_voice:
            speak_with_elevenlabs(reply)

        # Store this turn as memory candidate
        try:
            memory_reply = strip_markdown(reply) if use_voice else reply
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": memory_reply},
            ]
            metadata = merge_user_event_date_into_metadata(
                {
                    "source": "angel-cli",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
                user_message,
            )

            try:
                memory_client.add(messages, user_id=user_id, metadata=metadata)
            except Exception as e:
                print(f"{Fore.RED}Error saving memory to Mem0: {e}{Style.RESET_ALL}")
                print(traceback.format_exc())

            local_text = memory_reply
            _append_local_memory(user_id, local_text, metadata)
        except Exception as e:
            print(f"{Fore.RED}Warning: could not store memory: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()