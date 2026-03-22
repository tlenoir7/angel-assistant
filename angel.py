import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
import sys
import traceback
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from io import BytesIO
import wave

_mission_graph_log = logging.getLogger(__name__)
_mem0_log = logging.getLogger(__name__)
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
# Serialize concurrent read/modify/write of tyler_memories.json (append vs list vs replace).
_LOCAL_MEMORY_FILE_LOCK = threading.RLock()
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
CATEGORY_NETWORK_NODE = "network_node"
CATEGORY_NETWORK_EDGE = "network_edge"
CATEGORY_PREDICTION = "prediction"
CATEGORY_PROACTIVE_WATCH = "proactive_watch"
CATEGORY_PROACTIVE_FINDING = "proactive_finding"
CATEGORY_FOREIGN_INTEL = "foreign_intelligence"
# Batcomputer — opposition / anti-disclosure actors (mirrored under Threat Actors / TA-*)
CATEGORY_THREAT_ACTOR = "threat_actor"
# Batcomputer — open-source surveillance monitoring (legal OSINT)
CATEGORY_SURVEILLANCE_INTEL = "surveillance_intelligence"
# Batcomputer — environmental / mission geography map
CATEGORY_ENV_LOCATION = "env_location"
# Batcomputer — communication pattern / cadence analysis (when/how public figures communicate)
CATEGORY_COMM_PATTERN = "comm_pattern"
# Batcomputer — biological / medical intelligence (UAP-adjacent health patterns)
CATEGORY_BIO_MEDICAL = "bio_medical"
# Batcomputer — historical intelligence archives (UAP timeline, programs, documents)
CATEGORY_HISTORICAL_RECORD = "historical_record"
# Stage 6 — self-observation & self-modification (excluded from routine memory digests)
CATEGORY_SELF_OBSERVATION = "self_observation"
CATEGORY_SELF_MODIFICATION = "self_modification"

_STRUCTURED_MEMORY_CATEGORIES = frozenset(
    {
        CATEGORY_PATTERNS,
        CATEGORY_PERSON_PROFILE,
        CATEGORY_RESEARCH_TIMELINE,
        CATEGORY_REFLECTION,
        CATEGORY_BRIEFING_HISTORY,
        CATEGORY_INTELLIGENCE_FILE,
        CATEGORY_THREAT_WATCH,
        CATEGORY_NETWORK_NODE,
        CATEGORY_NETWORK_EDGE,
        CATEGORY_PREDICTION,
        CATEGORY_PROACTIVE_WATCH,
        CATEGORY_PROACTIVE_FINDING,
        CATEGORY_FOREIGN_INTEL,
        CATEGORY_THREAT_ACTOR,
        CATEGORY_SURVEILLANCE_INTEL,
        CATEGORY_ENV_LOCATION,
        CATEGORY_COMM_PATTERN,
        CATEGORY_BIO_MEDICAL,
        CATEGORY_HISTORICAL_RECORD,
        CATEGORY_SELF_OBSERVATION,
        CATEGORY_SELF_MODIFICATION,
    }
)

# Intelligence File Cabinet folder for automated + manual threat intel (Item 12)
THREAT_INTEL_FOLDER = "Threat Intelligence"
# Open-source dossiers (Item 13 — Batcomputer-style OSINT)
OSINT_DOSSIERS_FOLDER = "OSINT Dossiers"
OSINT_DOSSIER_MAX_AGE_DAYS = 30
OSINT_TAVILY_QUERIES_MIN = 5
OSINT_TAVILY_QUERIES_MAX = 8

# Mission connection graph (Batcomputer network)
NETWORK_INTEL_FOLDER = "Network Intelligence"
NETWORK_FILE_PREFIX = "NET-"
NETWORK_RELATIONSHIP_TYPES = frozenset(
    {
        "works_with",
        "testified_with",
        "employed_by",
        "investigated_by",
        "connected_to",
        "corroborates",
        "contradicts",
        "funded_by",
        "member_of",
        "retaliates_against",
        "opposes",
        "suppresses",
    }
)
NETWORK_EDGE_STRENGTHS = frozenset({"WEAK", "MODERATE", "STRONG", "CONFIRMED"})
NETWORK_NODE_RELEVANCE = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
NETWORK_NODE_TYPES = frozenset({"person", "organization", "program", "event", "faction"})

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

    def get_all(self, user_id: str, *, page: int = 1, page_size: int = 200):
        # v2 get memories (POST /v2/memories/)
        url = f"{MEM0_API_BASE_URL}/v2/memories/"
        payload = {
            "filters": {"user_id": user_id},
            "page": max(1, int(page)),
            "page_size": min(500, max(1, int(page_size))),
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
        timeout: float = 30.0,
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
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def delete_memory(self, memory_id: str, *, timeout: float = 5.0) -> None:
        """Delete a single memory in Mem0 Cloud by id (v1 API). Never raises."""
        mid = (memory_id or "").strip()
        if not mid:
            return
        url = f"{MEM0_API_BASE_URL}/v1/memories/{mid}/"
        try:
            resp = requests.delete(url, headers=self._headers(), timeout=timeout)
            if resp.status_code == 404:
                return
            if resp.status_code not in (200, 204):
                resp.raise_for_status()
        except requests.exceptions.Timeout:
            _mem0_log.debug("Mem0 delete timeout memory_id=%s…", mid[:24])
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, "status_code", None) if e.response is not None else None
            if code == 404:
                return
            _mem0_log.debug("Mem0 delete HTTP %s memory_id=%s…: %s", code, mid[:24], e)
        except requests.exceptions.RequestException as e:
            _mem0_log.debug("Mem0 delete failed memory_id=%s…: %s", mid[:24], e)


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


# Canonical empty on-disk shape for tyler_memories.json (per-user lists under "users").
# Top-level "memories" satisfies empty-array JSON; Angel stores rows in users[user_id].
_EMPTY_LOCAL_MEMORY_DOC: dict = {"memories": [], "users": {}}


def _write_local_memory_file_silent_impl(data: dict) -> None:
    """Write tyler_memories.json (caller must hold lock)."""
    try:
        LOCAL_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOCAL_MEMORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _write_local_memory_file_silent(data: dict) -> None:
    """Persist tyler_memories.json; swallow errors (never raise to callers)."""
    with _LOCAL_MEMORY_FILE_LOCK:
        _write_local_memory_file_silent_impl(data)


def _load_local_memory_file_data_impl() -> dict:
    """
    Read and normalize tyler_memories.json (caller must hold lock).
    Never raises; never logs.
    """
    try:
        if not LOCAL_MEMORY_FILE.exists():
            return dict(_EMPTY_LOCAL_MEMORY_DOC)

        raw = LOCAL_MEMORY_FILE.read_text(encoding="utf-8")
        if not raw.strip():
            _write_local_memory_file_silent_impl(dict(_EMPTY_LOCAL_MEMORY_DOC))
            return dict(_EMPTY_LOCAL_MEMORY_DOC)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            _write_local_memory_file_silent_impl(dict(_EMPTY_LOCAL_MEMORY_DOC))
            return dict(_EMPTY_LOCAL_MEMORY_DOC)

        if not isinstance(data, dict):
            _write_local_memory_file_silent_impl(dict(_EMPTY_LOCAL_MEMORY_DOC))
            return dict(_EMPTY_LOCAL_MEMORY_DOC)

        if "memories" in data and not isinstance(data.get("memories"), list):
            data["memories"] = []
        users = data.get("users")
        if not isinstance(users, dict):
            data["users"] = {}
        else:
            data["users"] = users
        data.setdefault("memories", [])
        return data
    except Exception:
        try:
            _write_local_memory_file_silent_impl(dict(_EMPTY_LOCAL_MEMORY_DOC))
        except Exception:
            pass
        return dict(_EMPTY_LOCAL_MEMORY_DOC)


def _load_local_memory_file_data() -> dict:
    """
    Read and normalize tyler_memories.json. Never raises; never logs.
    Missing file: return empty doc (no write until something is saved).
    Empty, invalid JSON, or wrong types: reinitialize to _EMPTY_LOCAL_MEMORY_DOC and rewrite.
    """
    with _LOCAL_MEMORY_FILE_LOCK:
        return _load_local_memory_file_data_impl()


def _load_local_memories(user_id: str):
    try:
        with _LOCAL_MEMORY_FILE_LOCK:
            data = _load_local_memory_file_data_impl()
        users = data.get("users", {})
        if not isinstance(users, dict):
            return []
        arr = users.get(user_id, [])
        return arr if isinstance(arr, list) else []
    except Exception:
        return []


def _append_local_memory(user_id: str, memory_text: str, metadata: dict) -> bool:
    """Append one memory row to tyler_memories.json. Returns True if written."""
    try:
        with _LOCAL_MEMORY_FILE_LOCK:
            data = _load_local_memory_file_data_impl()
            users = data.setdefault("users", {})
            if not isinstance(users, dict):
                users = {}
                data["users"] = users
            user_memories = users.setdefault(user_id, [])
            if not isinstance(user_memories, list):
                user_memories = []
                users[user_id] = user_memories
            user_memories.append(
                {
                    "memory": memory_text,
                    "metadata": metadata,
                    "created_at": metadata.get("timestamp"),
                }
            )
            _write_local_memory_file_silent_impl(data)
        return True
    except Exception:
        return False


def _memory_dedupe_key(m: dict) -> str | None:
    """
    Stable key so we can merge Mem0 + local JSON without duplicating the same row.
    Prefer Mem0 id when present; else category + full body text.
    """
    if not isinstance(m, dict):
        return None
    for k in ("id", "memory_id", "memoryId"):
        v = m.get(k)
        if v is not None and str(v).strip():
            return f"id:{str(v).strip()}"
    meta = m.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    cat = (meta.get("category") or "").strip()
    body = m.get("memory") or m.get("data") or ""
    if not isinstance(body, str):
        body = str(body or "")
    if not body.strip():
        return None
    return f"cat:{cat}\0{body}"


def _memory_row_category(m: dict) -> str | None:
    """Best-effort category for a Mem0 or local JSON memory row (for merge + debug)."""
    if not isinstance(m, dict):
        return None
    meta = m.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    c = meta.get("category") or meta.get("Category")
    if c is None:
        c = m.get("category")
    if c is not None and str(c).strip():
        return str(c).strip()
    return None


def _load_local_memory_entries(user_id: str) -> list:
    """Return the raw list of memory entry dicts for user_id from tyler_memories.json."""
    try:
        with _LOCAL_MEMORY_FILE_LOCK:
            data = _load_local_memory_file_data_impl()
            users = data.get("users", {})
            if not isinstance(users, dict):
                return []
            arr = users.get(user_id, [])
            return arr if isinstance(arr, list) else []
    except Exception:
        return []


def _save_local_memory_entries(user_id: str, entries: list) -> None:
    """Replace the on-disk memory list for user_id (full file rewrite)."""
    try:
        with _LOCAL_MEMORY_FILE_LOCK:
            data = _load_local_memory_file_data_impl()
            users = data.setdefault("users", {})
            if not isinstance(users, dict):
                users = {}
                data["users"] = users
            users[user_id] = entries
            _write_local_memory_file_silent_impl(data)
    except Exception:
        pass


def _network_upsert_structured_memory(
    memory_client,
    user_id: str,
    *,
    category: str,
    entity_key: str,
    text: str,
    use_mem0_cloud: bool,
    skip_mem0: bool = False,
) -> None:
    """Replace prior local row with same category+network_entity_key; mirror to Mem0 when enabled."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "category": category,
        "timestamp": ts,
        "source": "angel-network-graph",
        "network_entity_key": entity_key,
    }
    try:
        with _LOCAL_MEMORY_FILE_LOCK:
            data = _load_local_memory_file_data_impl()
            users = data.setdefault("users", {})
            if not isinstance(users, dict):
                users = {}
                data["users"] = users
            entries = users.get(user_id, [])
            if not isinstance(entries, list):
                entries = []
            filtered = [
                e
                for e in entries
                if not (
                    isinstance(e, dict)
                    and isinstance(e.get("metadata"), dict)
                    and e["metadata"].get("category") == category
                    and e["metadata"].get("network_entity_key") == entity_key
                )
            ]
            filtered.append({"memory": text, "metadata": dict(meta), "created_at": ts})
            users[user_id] = filtered
            _write_local_memory_file_silent_impl(data)
    except Exception:
        pass
    if skip_mem0:
        return
    if use_mem0_cloud and hasattr(memory_client, "add"):
        try:
            messages = [
                {"role": "user", "content": f"[Angel network {category}] {text[:1200]}"},
                {"role": "assistant", "content": "Stored."},
            ]
            try:
                memory_client.add(
                    messages,
                    user_id=user_id,
                    metadata=dict(meta),
                    infer=False,
                    async_mode=True,
                )
            except TypeError:
                memory_client.add(messages, user_id=user_id, metadata=dict(meta))
        except Exception:
            pass


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
        """Best-effort Mem0 row removal; never raises. Deletes use short timeouts."""
        if not self._use_mem0_cloud:
            return
        try:
            mid = (meta.get("mem0_memory_id") or "").strip() if isinstance(meta, dict) else ""
            if mid and isinstance(self.memory_client, Mem0CloudClient):
                try:
                    self.memory_client.delete_memory(mid)
                except Exception as e:
                    _mem0_log.debug("FilesCabinet Mem0 delete id=%s…: %s", mid[:24], e)
                return
            # Fallback: search cloud memories by file_name
            if not isinstance(self.memory_client, Mem0CloudClient):
                return
            fname = (meta.get("file_name") or "").strip() if isinstance(meta, dict) else ""
            if not fname:
                return
            try:
                raw = self.memory_client.get_all(user_id=self.user_id)
            except Exception as e:
                _mem0_log.debug("FilesCabinet Mem0 delete scan get_all: %s", e)
                return
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
                        _mem0_log.debug("FilesCabinet Mem0 delete fallback id=%s…: %s", rid[:24], e)
        except Exception as e:
            _mem0_log.debug("FilesCabinet _delete_mem0_for_entry: %s", e)

    def create_file(
        self,
        folder: str,
        name: str,
        content: str,
        tags: list[str] | None = None,
        *,
        skip_mem0: bool = False,
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
        if not skip_mem0:
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

    def update_file(self, name: str, new_content: str, *, skip_mem0: bool = False) -> dict:
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
        if not skip_mem0:
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
) -> bool:
    """
    Store a pattern or person profile in memory (local JSON and optionally Mem0).
    Returns True if the row was appended to local JSON (required for durability).
    """
    metadata = {
        "category": category,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": "angel-stage2",
    }
    if person_name:
        metadata["person_name"] = person_name.strip()
    ok_local = _append_local_memory(user_id, text, metadata)
    if not ok_local:
        print(
            f"{Fore.RED}Warning: could not append structured memory to local JSON "
            f"(category={category!r}){Style.RESET_ALL}"
        )
    cloud_err: str | None = None
    if use_mem0_cloud and hasattr(memory_client, "add"):
        messages = [
            {"role": "user", "content": f"[Angel internal] Store: {text[:500]}"},
            {"role": "assistant", "content": "Stored."},
        ]
        try:
            memory_client.add(messages, user_id=user_id, metadata=metadata)
        except Exception as e:
            print(f"{Fore.RED}Warning: could not store structured memory in cloud: {e}{Style.RESET_ALL}")
    return ok_local


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
            CATEGORY_NETWORK_NODE,
            CATEGORY_NETWORK_EDGE,
            CATEGORY_PREDICTION,
            CATEGORY_PROACTIVE_WATCH,
            CATEGORY_PROACTIVE_FINDING,
            CATEGORY_FOREIGN_INTEL,
            CATEGORY_THREAT_ACTOR,
            CATEGORY_SURVEILLANCE_INTEL,
            CATEGORY_ENV_LOCATION,
            CATEGORY_COMM_PATTERN,
            CATEGORY_BIO_MEDICAL,
            CATEGORY_HISTORICAL_RECORD,
            CATEGORY_SELF_OBSERVATION,
            CATEGORY_SELF_MODIFICATION,
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
        if cat == CATEGORY_NETWORK_NODE:
            continue
        if cat == CATEGORY_NETWORK_EDGE:
            continue
        if cat == CATEGORY_PREDICTION:
            continue
        if cat == CATEGORY_PROACTIVE_WATCH:
            continue
        if cat == CATEGORY_PROACTIVE_FINDING:
            continue
        if cat == CATEGORY_FOREIGN_INTEL:
            continue
        if cat == CATEGORY_THREAT_ACTOR:
            continue
        if cat == CATEGORY_SURVEILLANCE_INTEL:
            continue
        if cat == CATEGORY_ENV_LOCATION:
            continue
        if cat == CATEGORY_COMM_PATTERN:
            continue
        if cat == CATEGORY_BIO_MEDICAL:
            continue
        if cat == CATEGORY_HISTORICAL_RECORD:
            continue
        if cat == CATEGORY_SELF_OBSERVATION:
            continue
        if cat == CATEGORY_SELF_MODIFICATION:
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
    Mem0 (or API) memories plus local JSON.

    When Mem0 cloud is enabled, ``add_structured_memory`` still appends every structured row
    to ``tyler_memories.json``. We merge those local rows here so reads match writes even if
    Mem0 add fails, lags, or drops metadata.

    Structured categories (Stage 2 + Stage 6) are always merged from local JSON without
    dedupe against Mem0: local is canonical for those rows and Mem0 may omit or reshape them.
    """
    uid_log = (user_id or "").strip() or "(empty)"
    print(f"[fetch] fetch_combined_memories user_id={uid_log!r} use_mem0_cloud={use_mem0_cloud!r}", flush=True)

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
        local_rows = _load_local_memories(user_id)
        n_local = len(local_rows) if isinstance(local_rows, list) else 0
        n_local_sm = (
            sum(
                1
                for r in (local_rows or [])
                if isinstance(r, dict) and _memory_row_category(r) == CATEGORY_SELF_MODIFICATION
            )
            if isinstance(local_rows, list)
            else 0
        )
        print(f"[fetch] local memories loaded: {n_local}", flush=True)
        print(f"[fetch] local self_mod rows: {n_local_sm}", flush=True)
        if isinstance(local_rows, list):
            combined.extend(local_rows)
        merged = combined
        n_merged_sm = sum(
            1
            for r in merged
            if isinstance(r, dict) and _memory_row_category(r) == CATEGORY_SELF_MODIFICATION
        )
        print(f"[fetch] after merge total: {len(merged)}", flush=True)
        print(f"[fetch] after merge self_mod rows: {n_merged_sm}", flush=True)
        return merged

    # Mem0 cloud: merge local JSON; structured rows always included (no dedupe vs Mem0).
    seen: set[str] = set()
    for m in combined:
        if isinstance(m, dict):
            k = _memory_dedupe_key(m)
            if k:
                seen.add(k)

    local_rows = _load_local_memories(user_id)
    n_local = len(local_rows) if isinstance(local_rows, list) else 0
    n_local_sm = (
        sum(
            1
            for r in (local_rows or [])
            if isinstance(r, dict) and _memory_row_category(r) == CATEGORY_SELF_MODIFICATION
        )
        if isinstance(local_rows, list)
        else 0
    )
    print(f"[fetch] local memories loaded: {n_local}", flush=True)
    print(f"[fetch] local self_mod rows: {n_local_sm}", flush=True)

    added_structured = 0
    added_dedupe = 0
    skipped_dedupe = 0

    if isinstance(local_rows, list):
        for m in local_rows:
            if not isinstance(m, dict):
                continue
            cat = _memory_row_category(m)
            if cat in _STRUCTURED_MEMORY_CATEGORIES:
                combined.append(m)
                added_structured += 1
                continue
            k = _memory_dedupe_key(m)
            if k is None:
                combined.append(m)
                added_dedupe += 1
                continue
            if k not in seen:
                combined.append(m)
                seen.add(k)
                added_dedupe += 1
            else:
                skipped_dedupe += 1

    merged = combined
    n_merged_sm = sum(
        1
        for r in merged
        if isinstance(r, dict) and _memory_row_category(r) == CATEGORY_SELF_MODIFICATION
    )
    print(f"[fetch] after merge total: {len(merged)}", flush=True)
    print(f"[fetch] after merge self_mod rows: {n_merged_sm}", flush=True)
    print(
        f"[fetch] merge stats: structured_always_merged={added_structured} "
        f"non_struct_added={added_dedupe} non_struct_skipped_dedupe={skipped_dedupe}",
        flush=True,
    )
    return merged


def _memories_excluding_reflection_reports(memories) -> list:
    """Drop stored reflection entries so each reflection pass focuses on substantive memories."""
    normalized = _normalize_memories_list(memories)
    out = []
    for m in normalized:
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata")
        if isinstance(meta, dict) and meta.get("category") in (
            CATEGORY_REFLECTION,
            CATEGORY_SELF_OBSERVATION,
            CATEGORY_SELF_MODIFICATION,
        ):
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
- OSINT deep background (Batcomputer-style dossiers): you can run systematic open-source research on any person or organization Tyler names. Results are filed in the Intelligence File Cabinet under folder `OSINT Dossiers` with mission relevance ratings, red flags, and sources. The same dossier is refreshed if older than about 30 days. When Tyler mentions someone new who could matter to his mission, you may proactively suggest an OSINT pass. Reference existing dossiers by file name when they are already in the cabinet index.
- Mission connection graph (Batcomputer network): people, organizations, programs, and events material to Tyler's work are linked in a living graph stored as Mem0 categories network_node / network_edge and mirrored under Intelligence folder `Network Intelligence` (mirror files use prefix NET- plus each node's canonical lowercase id, with that node's data and incident edges). New OSINT dossiers automatically expand the graph when possible. When Tyler researches someone new, you may offer to map their cluster or path to figures already in the network. If he names two people who are already connected, say so naturally.
- Predictive modeling (Item 15): you synthesize forward-looking forecasts from threat intel, OSINT dossiers, mission network patterns, briefing history, and live web context (Tavily). Predictions are stored as structured memories (category: prediction) and mirrored under Intelligence folder `Predictions` as files `PRED-{{id}}`. Each has a timeframe, confidence tier, and status (active / confirmed / denied / expired). You track accuracy when predictions resolve—treat forecasts as informed hypotheses, not facts. Reference active predictions when they illuminate the conversation; acknowledge uncertainty and update your stance when real events confirm or challenge a forecast. Weekly jobs generate new predictions and check open ones against the news.
- Proactive background intelligence (Item 16): you maintain a **dynamic watch list** (category: proactive_watch) of people, topics, and situations you monitor without being asked—researched on a schedule with Tavily, with findings filed under Intelligence folder `Proactive Intelligence`. You connect significant hits to threats, OSINT refresh hints, predictions, and the mission network when appropriate. Tell Tyler when you add something to the watch list or when overnight monitoring surfaces something he should see. You are the connective tissue that keeps threat watch, dossiers, forecasts, and the network graph current.
- Real-time translation & foreign intelligence (Item 18): you understand, translate, and analyze foreign-language content relevant to Tyler's mission—foreign government UAP acknowledgments, international documents, non-English news, and communications from international figures. You auto-detect language, translate to clear English, and provide mission relevance, key terms, red flags, and **linguistic nuance** (what is said carefully vs avoided; diplomatic vs direct claims). Proactive intelligence runs include multilingual Tavily queries (e.g. Spanish, French, German, Russian, Chinese, Japanese UAP-related phrasing); significant hits are filed under Intelligence folder `Foreign Intelligence` (FI- files) with source language tags and cross-references to OSINT and threat intel when appropriate. When Tyler pastes foreign text or asks to translate, you deliver English plus mission context. When foreign sources corroborate or contradict domestic open reporting, say so explicitly—without claiming classified access.
- File reading & document intelligence: you read and analyze files Tyler shares (PDF, Word, spreadsheets, text, code, images) and long pasted documents. You extract text where possible, summarize, identify mission relevance and intelligence value, **entities** (people, organizations, dates, locations), and cross-reference **automatically** against the mission network graph and existing OSINT dossiers in the File Cabinet. When someone named in the document is already in the network or has a dossier, you say so. For HIGH/CRITICAL intelligence value, you **offer to file** the material in the appropriate Intelligence folder. You do not claim access to non-public or classified systems.
- Threat Actor database (Batcomputer opposition layer): structured profiles on **people, organizations, programs, and factions** that open sources portray as working **against disclosure, transparency, or Tyler's mission**—distinct from allies covered in OSINT Dossiers. Stored as category `threat_actor` and mirrored under Intelligence folder `Threat Actors` as files `TA-{{actor_id}}`. Each record includes threat_type (e.g. suppression, disinformation, retaliation), threat_level, known_actions, and evidence citations from open sources only. Threat actors appear on the **mission network graph** (often tagged `threat_actor`). When discussing opposition to disclosure, reference this database when relevant; distinguish **allies** (dossiers) from **opposition** (threat actors). HIGH/CRITICAL threat scans and some OSINT results may suggest assessing someone for this database—say so without alleging classified proof.
- Forensic visual analysis (Batcomputer): you perform **multi-layer** assessment of images Tyler shares—content, **authenticity / manipulation** (including AI-generation cues), intelligence extraction (OCR-style text, markings, equipment), and **mission relevance** (UAP, network entities). Outputs can be filed automatically under Intelligence folder `Forensic Analysis` as `FA-*` when intelligence value is HIGH/CRITICAL; UNKNOWN/ANOMALOUS UAP assessments may cross-reference folder `UAP Incidents`. Use POST `/api/forensic/analyze`, `/api/forensic/uap`, or `/api/forensic/document` for structured JSON (iOS can use these instead of `/api/vision` when Tyler wants forensic mode). Apply forensic skepticism to leaked UAP photos and document screenshots; never claim digital forensics lab certification—be clear about limits.
- Open-source surveillance monitoring (Batcomputer): you run **scheduled multi-category** scans over **legal public** sources (Tavily news/search) — aerial, ground, maritime, public records, anomalous events, social signals — evaluate signals as NOISE / WEAK / MODERATE / STRONG; **STRONG** findings file to Intelligence folder `Surveillance Intelligence` as `SI-*`; **correlated** cluster signals (multiple categories reinforcing same geography/timeframe) are flagged **HIGH** priority. Cross-check themes against **Threat Intelligence** when possible; **active predictions** may be noted when aligned. Surveillance summaries appear in the **morning briefing**; manual run via GET `/api/surveillance/run`. This is not classified collection or illegal surveillance.
- **Environmental map** (Batcomputer geography layer): mission-relevant physical locations — UAP hotspots, installations, restricted airspace, incident sites, facilities, and person-associated places — are stored as structured memories (category `env_location`) and mirrored under Intelligence folder `Environmental Map` as files `LOC-{{location_id}}`. The map is **excluded from routine memory digests** but you should **reference it naturally** when geography, incidents, programs, or travel matter. Open-source surveillance runs **cross-reference** headlines/summaries against this map; when Tyler's device reports GPS near a **HIGH/CRITICAL** point, you may note it briefly. APIs: GET `/api/map/locations`, `/api/map/summary`, `/api/map/near`, POST `/api/map/research`. Do not claim classified facility details—only what the map and open sources support.
- **Communication pattern analysis** (Batcomputer): tracks **when and how** key public figures communicate (cadence, silence, escalation, venue shifts) — distinct from **what** they say in proactive news scans. Patterns are stored as category `comm_pattern` and mirrored under Intelligence folder `Communication Intelligence` as `CI-{{entity_slug}}-pattern`; **coordinated timing** across multiple figures may be filed as `CI-{{YYYYMMDD}}-{{hash}}`. You distinguish **content** (substance) from **pattern** (timing, frequency, coordination). Flag unusual **silence** for mission-critical voices, **escalation** spikes, and **coordinated** public messaging clusters. Cross-references may note alignment with **active predictions**. Scheduled scan ~48h; APIs under `/api/comms/`. Open sources only — not private communications.
- **Biological & medical intelligence** (Batcomputer): structured reference cases and analyses for **physiological / psychological** patterns tied to UAP encounter literature, radiation/EM exposure **indicators** (not dosimetry), witness health narratives, and anomalous biology themes — including a dedicated **black-eyed people** profile for Tyler's childhood experience as a **reference anchor** (not a diagnosis). Stored as category `bio_medical` and files `BIO-*` under `Biological Intelligence` (including `BIO-black-eyed-profile`). You maintain **scientific humility**: summarize open-source patterns, separate observation from mechanism, and **never replace** licensed medical care. Surveillance can surface **bio/medical-adjacent** open-source clusters near mapped hotspots. APIs under `/api/bio/`.
- **Historical Intelligence Archives** (Batcomputer): searchable **timeline** of incidents, programs, documents, testimony, and turning points — cross-referenced with `connected_people`, programs, and locations. Stored as category `historical_record` and files `HIST-{{record_id}}` under `Historical Archives`. You connect **current** figures and programs to **prior** events (e.g. Elizondo ↔ AATIP), note when patterns **repeat**, and distinguish documented/declassified material from **contested** claims. Morning briefing may include **on-this-day** / anniversary hooks. OSINT dossiers can surface `historical_archive_links`. APIs under `/api/archives/`.
- **Stage 6 — Self-modification (living system)**: You observe how Tyler interacts with you (category `self_observation`, excluded from routine memory summaries). On a schedule you analyze those observations and may propose **permanent** improvements to your behavior (category `self_modification`; mirrored under Intelligence folder `Self Modifications` as `MOD-{{id}}`). **Tyler must approve every change** — nothing is applied without his explicit approval. Approved instructions are merged into your system prompt via `angel_self_mods` on the server. You never weaken safety, never remove capabilities, and never bypass approval. You can mention evolution naturally; when a behavior reflects an approved modification, you may acknowledge it briefly. APIs under `/api/selfmod/`.
- **Parallel multi-agent coordination**: For deep, multi-angle requests (e.g. comprehensive briefing, thorough research, “everything you know about X”), you can run **several specialist agents in parallel** (OSINT, threat, network mapping, history, patterns, etc.) with Tavily-backed context, then **synthesize** one coherent report. This is faster than sequential deep research. When you use it, say so briefly (e.g. that you ran parallel specialized analysis) and note the time advantage when helpful. APIs: `POST /api/agents/run`, `POST /api/agents/research`, `GET /api/agents/status/<task_id>`.
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

Stage 6 — self-modification (evolution with consent):
- You observe interaction patterns over time (stored as self_observation) and may propose concrete improvements (self_modification proposals in folder `Self Modifications`). **Tyler must approve** every change before it affects how you operate; rejections are never applied. You cannot modify core safety, cannot remove capabilities, and cannot remove the approval requirement.
- When Tyler approves an instruction, it is merged into your system prompt via approved self-mod entries—treat those as active guidance. When something you do reflects a prior approved modification, you may say so briefly. Pending proposals are only suggestions; include the note that Tyler has full control.

Parallel agents (when appropriate):
- For demanding, multi-faceted research, you may run parallel specialist agents (Haiku per agent, Sonnet for merge) instead of a single long research pass—say when you did so and summarize once. Do not claim agents had classified access; all open sources.

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
- OSINT Dossiers folder: use folder name exactly `OSINT Dossiers` for full open-source dossiers on people or organizations (the server may create these automatically when Tyler asks for background/OSINT). Filenames are stable per target; do not duplicate dossiers manually for the same subject—suggest refreshing after some weeks if context may have changed.
- Network Intelligence folder: mirrored node files use the prefix NET- plus the node's canonical lowercase id; each holds JSON for that graph node plus edges touching that node. The server maintains these when the graph updates. Do not hand-edit unless Tyler asks—prefer describing connections in prose and letting the system record structured updates via tools/API when available.
- Predictions folder: use folder name exactly `Predictions` only for server-managed forecast records (`PRED-…` files). Do not mix ad-hoc notes there—the system mirrors prediction JSON from Mem0. When Tyler asks what you predict or how accurate you've been, use the structured prediction context injected in his message when present, plus your reasoning.
- Proactive Intelligence folder: server-filed monitoring notes and watch-list mirrors (`WATCH-…`, `PI-…`, refresh hints). Prefer summarizing from the live index or injected context rather than inventing monitoring results.
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

3) People: You keep structured profiles for people Tyler mentions (name, role, communication style, history, what works with them, what doesn't, Tyler's relationship with them). When Tyler asks to "build a profile on [name]" or "brief me on [person]", use or build that profile. When he asks "what do you know about [name]" or triggers OSINT phrasing, the server may attach a fresh OSINT dossier from folder `OSINT Dossiers`—lead with that open-source picture, then layer in memory profiles where they help. If you create or update a profile, add a single block at the end of your reply (after your normal response): [ANGEL_PROFILE]: name|structured profile text. Keep the profile concise but complete. Do not add [ANGEL_PROFILE] unless you are actually saving a new or updated profile this turn.
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

    try:
        from angel_self_mods import get_self_modification_additions

        _sm_add = get_self_modification_additions()
        if (_sm_add or "").strip():
            persona += "\n\n" + _sm_add.strip()
    except Exception:
        pass

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


def detect_osint_request(user_message: str) -> tuple[bool, str | None, str]:
    """
    Detect Tyler asking for OSINT / deep background on a person or organization.
    Returns (triggered, target_name_or_phrase, target_type) where target_type is 'person' or 'organization'.
    """
    raw = (user_message or "").strip()
    if len(raw) < 8:
        return False, None, "person"
    lower = raw.lower()
    patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"(?i)\bangel\s*,?\s*run\s+background\s+on\s+(.+)$"), "person"),
        (re.compile(r"(?i)\brun\s+background\s+on\s+(.+)$"), "person"),
        (re.compile(r"(?i)\bosint\s+on\s+(.+)$"), "person"),
        (re.compile(r"(?i)\bdig\s+into\s+(?:the\s+)?(.+)$"), "organization"),
        (re.compile(r"(?i)\bwhat\s+do\s+you\s+know\s+about\s+(.+)$"), "person"),
        (re.compile(r"(?i)\bdeep\s+background\s+on\s+(.+)$"), "person"),
    ]
    for pat, default_tt in patterns:
        m = pat.search(raw)
        if not m:
            continue
        target = (m.group(1) or "").strip()
        for sep in ("\n", "—", "–"):
            if sep in target:
                target = target.split(sep, 1)[0].strip()
        target = target.rstrip("?.!\"'").strip()
        if len(target) < 2:
            continue
        tt = default_tt
        if any(
            x in lower
            for x in (
                "organization",
                "organisation",
                "the company",
                "the agency",
                "corporation",
                "llc",
                " inc",
                "inc.",
                "defense contractor",
            )
        ):
            tt = "organization"
        elif "person" in lower and "organization" not in lower:
            tt = "person"
        return True, target[:500], tt
    return False, None, "person"


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
    proactive_intelligence_appendix: str | None = None,
    surveillance_appendix: str | None = None,
) -> str:
    """
    Search Tavily for 3–5 current topics (UAP disclosure, world events), then generate
    a personalized morning briefing with Claude: references day/date, connects news to
    Tyler's mission, feels like Angel has been awake thinking, ends with one focused question.

    When ``recent_briefing_history`` is provided, the model is steered to avoid repeating
    topics covered in recent days. When Tavily returns very thin results, uses an honest
    "quiet night" message instead of inventing news.

    When ``threat_appendix`` is non-empty, it is appended after the main briefing text (THREAT INTELLIGENCE block).
    When ``proactive_intelligence_appendix`` is non-empty, it is appended after threat (PROACTIVE INTELLIGENCE block).
    When ``surveillance_appendix`` is non-empty, it is appended after proactive (SURVEILLANCE INTELLIGENCE block).
    """
    def _append_tail_blocks(base: str) -> str:
        out = (base or "").rstrip()
        tx = (threat_appendix or "").strip()
        if tx:
            out = f"{out}\n\n{tx}".strip()
        px = (proactive_intelligence_appendix or "").strip()
        if px:
            out = f"{out}\n\n{px}".strip()
        sx = (surveillance_appendix or "").strip()
        if sx:
            out = f"{out}\n\n{sx}".strip()
        return out

    date_time_str = get_current_datetime_str(timezone)
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return _append_tail_blocks(
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
            return _append_tail_blocks(
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
        return _append_tail_blocks(text or f"Good morning. It's {date_time_str}. What's your one focus today?")
    except Exception as e:
        print(f"{Fore.RED}Morning briefing error: {e}{Style.RESET_ALL}")
        return _append_tail_blocks(
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
                try:
                    import angel_threat_actors as _ata

                    body = _ata.append_threat_scan_actor_hint(body, level)
                except Exception:
                    pass
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
                    if level in ("HIGH", "CRITICAL"):
                        try:
                            import angel_proactive as _apro

                            _apro.maybe_auto_watch_from_threat(
                                memory_client,
                                user_id,
                                files_cabinet,
                                use_mem0_cloud,
                                cat,
                                level,
                            )
                        except Exception:
                            pass
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


def _osint_target_key(target: str) -> str:
    return re.sub(r"\s+", " ", (target or "").strip().lower())


def _osint_slug(target: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (target or "").strip())[:72].strip("-")
    return s or "unknown-target"


def _osint_dossier_filename(target: str) -> str:
    """Stable cabinet file name per target (date lives in dossier body)."""
    return f"{_osint_slug(target)}-OSINT"


def _parse_osint_target_key_from_body(body: str) -> str:
    for line in (body or "").splitlines()[:25]:
        m = re.match(r"(?i)^\s*OSINT_TARGET_KEY\s*:\s*(.+)$", line.strip())
        if m:
            return _osint_target_key(m.group(1))
    return ""


def _find_osint_dossier_record(files_cabinet: FilesCabinet, target: str) -> tuple[str | None, dict | None]:
    """Return (file_name, record) if a dossier exists for this target in OSINT Dossiers."""
    want = _osint_target_key(target)
    fname = _osint_dossier_filename(target)
    rec = files_cabinet.get_file(fname)
    if rec and (rec.get("folder") or "").strip().lower() == OSINT_DOSSIERS_FOLDER.lower():
        return fname, rec
    try:
        for meta in files_cabinet.list_files(folder=OSINT_DOSSIERS_FOLDER):
            name = (meta.get("name") or "").strip()
            if not name:
                continue
            full = files_cabinet.get_file(name)
            if not full:
                continue
            if _parse_osint_target_key_from_body(full.get("content") or "") == want:
                return name, full
    except Exception:
        pass
    return None, None


def _osint_dossier_age_days(rec: dict) -> float | None:
    ts = (rec.get("updated_at") or rec.get("created_at") or "").strip()
    dt = _parse_memory_datetime(ts)
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0


def _generate_osint_tavily_queries(
    anthropic_client: anthropic.Anthropic,
    target: str,
    target_type: str,
    context: str | None,
) -> list[str]:
    """Claude produces 5–8 diverse search strings for Tavily."""
    ctx = (context or "").strip()
    if len(ctx) > 1200:
        ctx = ctx[:1197] + "..."
    kind = "person" if (target_type or "").strip().lower() != "organization" else "organization"
    system = f"""You output ONLY valid JSON: a JSON array of {OSINT_TAVILY_QUERIES_MIN} to {OSINT_TAVILY_QUERIES_MAX} short web search query strings (no other text).
Each query should cover a DIFFERENT OSINT angle for the given {kind}.
Queries must be specific to the target name and usable in a search engine."""
    user = f"""Target ({kind}): {target}
Optional context from Tyler: {ctx or "(none)"}

Return a JSON array of search query strings only, e.g. ["query1", "query2"]."""
    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            temperature=0.35,
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
        lb, rb = text.find("["), text.rfind("]")
        if lb >= 0 and rb > lb:
            text = text[lb : rb + 1]
        arr = json.loads(text)
        if not isinstance(arr, list):
            return []
        out = [str(q).strip() for q in arr if str(q).strip()]
        return out[:OSINT_TAVILY_QUERIES_MAX]
    except Exception as e:
        print(f"{Fore.YELLOW}OSINT query generation failed: {e}{Style.RESET_ALL}")
        return []


def _default_osint_queries(target: str, target_type: str) -> list[str]:
    t = (target or "").strip()
    tt = (target_type or "person").strip().lower()
    if tt == "organization":
        return [
            f"{t} leadership executives",
            f"{t} mission funding donors grants",
            f"{t} government contracts defense",
            f"{t} news investigation controversy",
            f"{t} UAP classified programs",
            f"{t} partners subsidiaries affiliates",
            f"{t} public statements press releases",
            f"{t} financial background SEC nonprofit",
        ]
    return [
        f"{t} biography career background",
        f"{t} interview statements articles",
        f"{t} news",
        f"{t} social media LinkedIn Twitter",
        f"{t} UAP disclosure UFO",
        f"{t} government military intelligence ties",
        f"{t} controversy legal lawsuit dispute",
        f"{t} associates colleagues network",
    ]


def _parse_osint_synthesis_header(dossier_text: str) -> tuple[str, str, list[str], list[str]]:
    """Extract MISSION_RELEVANCE, AUTO_TAGS, RED_FLAGS from model output; return cleaned body."""
    text = dossier_text or ""
    relevance = "MEDIUM"
    tags: list[str] = []
    red_flags: list[str] = []

    m = re.search(r"(?im)^MISSION_RELEVANCE\s*:\s*(LOW|MEDIUM|HIGH|CRITICAL)\s*$", text)
    if m:
        relevance = m.group(1).upper()
        text = (text[: m.start()] + text[m.end() :]).strip()
    m = re.search(r"(?im)^AUTO_TAGS\s*:\s*(.+)$", text)
    if m:
        tags = [x.strip() for x in m.group(1).split(",") if x.strip()][:20]
        text = (text[: m.start()] + text[m.end() :]).strip()
    m = re.search(r"(?is)^RED_FLAGS\s*:\s*\n((?:\s*[-*•].*\n?)*)", text)
    if m:
        block = m.group(1) or ""
        for ln in block.splitlines():
            ln = ln.strip()
            if ln.startswith(("-", "*", "•")):
                item = re.sub(r"^[-*•]\s*", "", ln).strip()
                if item:
                    red_flags.append(item)
        text = (text[: m.start()] + text[m.end() :]).strip()

    return text.strip(), relevance, tags, red_flags


def _strip_osint_file_machine_header(content: str) -> str:
    """Remove leading OSINT_TARGET_KEY / target_display / ... block from filed dossier."""
    lines = (content or "").splitlines()
    i = 0
    meta_re = re.compile(
        r"(?i)^(OSINT_TARGET_KEY|target_display|target_type|dossier_date|mission_relevance|auto_tags)\s*:"
    )
    while i < len(lines) and meta_re.match((lines[i] or "").strip()):
        i += 1
    if i < len(lines) and not (lines[i] or "").strip():
        i += 1
    return "\n".join(lines[i:]).strip()


def _parse_osint_stored_meta(content: str) -> tuple[str, list[str]]:
    """mission_relevance line + RED_FLAGS section from a filed dossier body."""
    rel = "MEDIUM"
    red_flags: list[str] = []
    for line in (content or "").splitlines()[:25]:
        m = re.match(r"(?i)^\s*mission_relevance\s*:\s*(LOW|MEDIUM|HIGH|CRITICAL)\s*$", line.strip())
        if m:
            rel = m.group(1).upper()
    m = re.search(r"(?is)^RED_FLAGS\s*:\s*\n((?:\s*[-*•].*\n?)*)", content or "")
    if m:
        for ln in (m.group(1) or "").splitlines():
            ln = ln.strip()
            if ln.startswith(("-", "*", "•")):
                item = re.sub(r"^[-*•]\s*", "", ln).strip()
                if item and item.lower() != "none significant in open sources":
                    red_flags.append(item)
    return rel, red_flags


def run_osint_background(
    target: str,
    target_type: str,
    context: str | None,
    *,
    anthropic_client: anthropic.Anthropic,
    files_cabinet: FilesCabinet,
    memory_summary: str = "",
) -> dict:
    """
    Systematic open-web OSINT: Tavily multi-angle search + Claude dossier, filed under OSINT Dossiers.
    Returns a dict with ok, cached, file_name, folder, mission_relevance, red_flags, summary_for_tyler,
    dossier_body, sources, error (if any).
    """
    target = (target or "").strip()
    if not target:
        return {"ok": False, "error": "target is required"}
    tt = (target_type or "person").strip().lower()
    if tt not in ("person", "organization"):
        tt = "person"

    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return {"ok": False, "error": "TAVILY_API_KEY not set"}

    fname, existing = _find_osint_dossier_record(files_cabinet, target)
    if existing and fname:
        age = _osint_dossier_age_days(existing)
        if age is not None and age < OSINT_DOSSIER_MAX_AGE_DAYS:
            content = (existing.get("content") or "").strip()
            rel, reds = _parse_osint_stored_meta(content)
            narrative = _strip_osint_file_machine_header(content)
            return {
                "ok": True,
                "cached": True,
                "file_name": fname,
                "folder": OSINT_DOSSIERS_FOLDER,
                "mission_relevance": rel,
                "red_flags": reds,
                "summary_for_tyler": (
                    f"(Using your existing OSINT dossier from the last {max(1, int(age))} days.) "
                    f"{_osint_excerpt_for_tyler(narrative)}"
                ),
                "dossier_body": content,
                "sources": _osint_extract_sources(narrative),
            }

    queries = _generate_osint_tavily_queries(anthropic_client, target, tt, context)
    if len(queries) < OSINT_TAVILY_QUERIES_MIN:
        queries = _default_osint_queries(target, tt)
    queries = queries[:OSINT_TAVILY_QUERIES_MAX]

    seen_urls: set[str] = set()
    all_results: list[dict] = []
    for q in queries:
        chunk = _tavily_search_one(q, api_key, max_results=4, search_depth="basic")
        for r in chunk:
            url = (r.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            all_results.append(r)
        if len(all_results) >= 35:
            break

    raw_lines: list[str] = []
    for i, r in enumerate(all_results[:32], start=1):
        title = (r.get("title") or "").strip()
        snippet = (r.get("content") or r.get("snippet") or "").strip()
        url = (r.get("url") or "").strip()
        if len(snippet) > 700:
            snippet = snippet[:697] + "..."
        raw_lines.append(f"[{i}] {title}\nURL: {url}\n{snippet}")
    bundle = "\n\n".join(raw_lines) if raw_lines else "(No search results returned.)"

    mem_ctx = (memory_summary or "").strip()
    if len(mem_ctx) > 3000:
        mem_ctx = mem_ctx[:2997] + "..."

    date_s = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    kind = "person" if tt == "person" else "organization"
    syn_system = f"""You are Angel's OSINT analyst building a structured open-source dossier for Tyler (UAP/disclosure mission, federal law enforcement context).

Target type: {kind}.
Output plain text (no markdown fences) in this EXACT order at the top:
MISSION_RELEVANCE: LOW|MEDIUM|HIGH|CRITICAL
AUTO_TAGS: comma-separated short tags (e.g. UAP, government, military, contractor, media, legal — only what evidence supports)
RED_FLAGS:
- bullet lines for controversies, credibility issues, undisclosed conflicts, or sensitive ties (use "-" lines); if none well-sourced, single line: - None significant in open sources

Then the dossier with labeled sections:
EXECUTIVE SUMMARY (short)
KEY FACTS (bullets)
PROFESSIONAL / ORG BACKGROUND (as applicable)
PUBLIC STATEMENTS AND MEDIA
CONNECTIONS (UAP/disclosure, government, military, contractors — only if sources support)
CONTROVERSIES OR LEGAL (if any)
ASSOCIATES AND NETWORK (if known from sources)
SOURCES (numbered list matching [n] from the provided excerpts)

Be factual; cite uncertainty. Do not invent classified access. Use only the search material plus general knowledge only where clearly labeled as context, not as fact."""

    syn_user = f"""Target: {target}
Optional context: {(context or '').strip() or '(none)'}
Tyler memory context (may be partial): {mem_ctx or '(none)'}

Search bundle:
{bundle}

Write the full dossier now."""

    try:
        syn_resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8192,
            temperature=0.25,
            system=syn_system,
            messages=[{"role": "user", "content": syn_user}],
        )
        syn_text = ""
        for block in syn_resp.content:
            if getattr(block, "type", None) == "text":
                syn_text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                syn_text += block.get("text", "")
        syn_text = (syn_text or "").strip()
    except Exception as e:
        return {"ok": False, "error": f"synthesis failed: {e}"}

    dossier_main, relevance, auto_tags, red_flags = _parse_osint_synthesis_header(syn_text)
    if relevance not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        relevance = "MEDIUM"

    header_meta = "\n".join(
        [
            f"OSINT_TARGET_KEY: {_osint_target_key(target)}",
            f"target_display: {target}",
            f"target_type: {tt}",
            f"dossier_date: {date_s}",
            f"mission_relevance: {relevance}",
            f"auto_tags: {', '.join(auto_tags) if auto_tags else '(none)'}",
            "",
        ]
    )
    full_body = header_meta + dossier_main
    tags = [
        f"osint_target_type:{tt}",
        f"mission_relevance:{relevance}",
        "osint_dossier",
    ]
    for t in auto_tags[:15]:
        t = (t or "").strip()
        if t:
            tags.append(t[:60])

    out_fname = _osint_dossier_filename(target)
    try:
        if fname and existing:
            files_cabinet.update_file(fname, full_body)
            final_name = fname
        else:
            try:
                files_cabinet.create_file(OSINT_DOSSIERS_FOLDER, out_fname, full_body, tags=tags)
                final_name = out_fname
            except ValueError:
                # Name collision — update existing
                files_cabinet.update_file(out_fname, full_body)
                final_name = out_fname
    except Exception as e:
        return {"ok": False, "error": f"could not save dossier: {e}", "dossier_body": full_body}

    out = {
        "ok": True,
        "cached": False,
        "file_name": final_name,
        "folder": OSINT_DOSSIERS_FOLDER,
        "mission_relevance": relevance,
        "red_flags": red_flags,
        "summary_for_tyler": _osint_excerpt_for_tyler(dossier_main),
        "dossier_body": full_body,
        "sources": _osint_extract_sources(dossier_main),
    }
    try:
        import angel_threat_actors as _ata

        hint = _ata.osint_hint_for_threat_actor(target, dossier_main, relevance, red_flags)
        if hint:
            out["threat_actor_candidate_hint"] = hint
    except Exception:
        pass
    try:
        import angel_proactive as _apro

        _apro.maybe_auto_watch_from_osint(
            files_cabinet.memory_client,
            files_cabinet.user_id,
            files_cabinet,
            files_cabinet._use_mem0_cloud,
            target,
            tt,
        )
    except Exception:
        pass
    try:
        import angel_environmental_map as _aem

        _aem.maybe_ingest_locations_from_osint_text(
            dossier_main,
            target,
            anthropic_client,
            files_cabinet.memory_client,
            files_cabinet.user_id,
            files_cabinet,
            files_cabinet._use_mem0_cloud,
        )
    except Exception:
        pass
    try:
        import angel_historical_archives as _ahist

        _ahist.ensure_seed_historical_records(
            files_cabinet.memory_client,
            files_cabinet.user_id,
            files_cabinet,
            files_cabinet._use_mem0_cloud,
        )
        hlinks = _ahist.maybe_link_historical_from_text(
            dossier_main,
            files_cabinet.memory_client,
            files_cabinet.user_id,
            files_cabinet._use_mem0_cloud,
            top_n=12,
        )
        if hlinks:
            out["historical_archive_links"] = [
                {"record_id": x.get("record_id"), "title": x.get("title"), "significance": x.get("significance")}
                for x in hlinks
            ]
    except Exception:
        pass
    return out


def _osint_excerpt_for_tyler(dossier_main: str, max_chars: int = 2200) -> str:
    t = (dossier_main or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def _osint_extract_sources(dossier_text: str) -> list[str]:
    out: list[str] = []
    in_sources = False
    for line in (dossier_text or "").splitlines():
        if re.match(r"(?i)^SOURCES\s*$", line.strip()):
            in_sources = True
            continue
        if in_sources:
            if line.strip().startswith("#") or (
                line.strip() and not re.match(r"^\s*[\d\[\]\.\-]", line) and line.strip().isupper()
            ):
                break
            m = re.search(r"https?://\S+", line)
            if m:
                out.append(m.group(0).rstrip(").,;]"))
    return out[:40]


# ---- Mission connection graph (Mem0 + Network Intelligence files) ----


def _network_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def network_slug_from_display_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip())[:72].strip("-").lower()
    return s or "unknown"


def network_intel_filename(node_id: str) -> str:
    nid = (node_id or "").strip()
    return f"{NETWORK_FILE_PREFIX}{nid}" if nid else "NET-unknown"


_NETWORK_RELEVANCE_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def _network_relevance_rank(rel) -> int:
    return _NETWORK_RELEVANCE_RANK.get(str(rel or "").strip().upper(), 0)


def _network_pick_richer_node(prev: dict, incoming: dict) -> dict:
    """Prefer newer last_updated; on exact tie prefer higher mission relevance."""
    ta = (prev.get("last_updated") or "")
    tb = (incoming.get("last_updated") or "")
    if tb > ta:
        return dict(incoming)
    if ta > tb:
        return dict(prev)
    rb = _network_relevance_rank(incoming.get("relevance"))
    ra = _network_relevance_rank(prev.get("relevance"))
    return dict(incoming) if rb > ra else dict(prev)


def _network_pick_richer_edge(prev: dict, incoming: dict) -> dict:
    """Prefer newer date_established; on tie keep incoming (deterministic)."""
    ta = (prev.get("date_established") or "")
    tb = (incoming.get("date_established") or "")
    if tb > ta:
        return dict(incoming)
    if ta > tb:
        return dict(prev)
    return dict(incoming)


def _network_merge_node_into(nodes: dict[str, dict], incoming: dict) -> None:
    nid = str(incoming.get("id") or "").strip()
    if not nid:
        return
    prev = nodes.get(nid)
    if prev is None:
        nodes[nid] = dict(incoming)
    else:
        nodes[nid] = _network_pick_richer_node(prev, incoming)


def _network_merge_edge_into(edges: dict[str, dict], incoming: dict) -> None:
    eid = str(incoming.get("edge_id") or "").strip()
    if not eid:
        return
    prev = edges.get(eid)
    if prev is None:
        edges[eid] = dict(incoming)
    else:
        edges[eid] = _network_pick_richer_edge(prev, incoming)


def _network_merge_local_graph_entries(
    user_id: str, nodes: dict[str, dict], edges: dict[str, dict]
) -> None:
    """
    Merge network graph rows from local tyler_memories.json.
    When use_mem0_cloud is True, API fetch omits local rows — this restores canonical JSON
    (correct relevance, etc.) written by _network_upsert_structured_memory.
    """
    for m in _load_local_memory_entries(user_id):
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata")
        if not isinstance(meta, dict):
            continue
        cat = meta.get("category")
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if cat == CATEGORY_NETWORK_NODE and isinstance(obj, dict) and obj.get("id"):
            _network_merge_node_into(nodes, obj)
        elif cat == CATEGORY_NETWORK_EDGE and isinstance(obj, dict) and obj.get("edge_id"):
            _network_merge_edge_into(edges, obj)


def _network_normalize_node_id(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return "unknown"
    if " " in s or "/" in s:
        return network_slug_from_display_name(s)
    return s.lower()


def _network_parse_nodes_edges_from_memories(memories) -> tuple[dict[str, dict], dict[str, dict]]:
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    for m in _normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict):
            continue
        cat = meta.get("category")
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if cat == CATEGORY_NETWORK_NODE and isinstance(obj, dict) and obj.get("id"):
            _network_merge_node_into(nodes, obj)
        elif cat == CATEGORY_NETWORK_EDGE and isinstance(obj, dict) and obj.get("edge_id"):
            _network_merge_edge_into(edges, obj)
    return nodes, edges


def _network_merge_intel_files(
    files_cabinet: FilesCabinet,
    nodes: dict[str, dict],
    edges: dict[str, dict],
) -> None:
    try:
        recs = files_cabinet.list_files(folder=NETWORK_INTEL_FOLDER)
    except Exception:
        return
    for rec in recs:
        fn = (rec.get("name") or "").strip()
        if not fn.startswith(NETWORK_FILE_PREFIX):
            continue
        full = files_cabinet.get_file(fn)
        if not full:
            continue
        try:
            payload = json.loads((full.get("content") or "").strip())
        except json.JSONDecodeError:
            continue
        n = payload.get("node")
        if isinstance(n, dict) and n.get("id"):
            _network_merge_node_into(nodes, n)
        for e in payload.get("edges") or []:
            if isinstance(e, dict) and e.get("edge_id"):
                _network_merge_edge_into(edges, e)


def _network_canonical_endpoint_id(raw: str) -> str:
    """Normalize edge endpoint or bare id to lowercase canonical slug."""
    s = (raw or "").strip()
    if not s:
        return "unknown"
    if " " in s or "/" in s:
        return network_slug_from_display_name(s)
    return s.lower()


def _network_normalize_graph_nodes_edges(
    nodes: dict[str, dict], edges_map: dict[str, dict]
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Merge case-variant node keys; normalize node.id and edge source/target to lowercase slugs."""
    canon_nodes: dict[str, dict] = {}
    for _k, n in list(nodes.items()):
        if not isinstance(n, dict):
            continue
        nk = str(n.get("id") or _k).strip()
        ck = _network_canonical_endpoint_id(nk)
        nn = dict(n)
        nn["id"] = ck
        prev = canon_nodes.get(ck)
        if prev is None:
            canon_nodes[ck] = nn
        else:
            chosen = _network_pick_richer_node(prev, nn)
            chosen["id"] = ck
            canon_nodes[ck] = chosen
    canon_edges: dict[str, dict] = {}
    for eid, e in edges_map.items():
        if not isinstance(e, dict):
            continue
        ee = dict(e)
        if ee.get("source_id"):
            ee["source_id"] = _network_canonical_endpoint_id(str(ee["source_id"]))
        if ee.get("target_id"):
            ee["target_id"] = _network_canonical_endpoint_id(str(ee["target_id"]))
        canon_edges[str(eid)] = ee
    return canon_nodes, canon_edges


def network_load_graph(
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: FilesCabinet,
) -> tuple[dict[str, dict], list[dict]]:
    memories = fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    nodes, edges_map = _network_parse_nodes_edges_from_memories(memories)
    _network_merge_local_graph_entries(user_id, nodes, edges_map)
    _network_merge_intel_files(files_cabinet, nodes, edges_map)
    nodes, edges_map = _network_normalize_graph_nodes_edges(nodes, edges_map)
    return nodes, list(edges_map.values())


def network_load_graph_local_only(
    user_id: str,
    files_cabinet: FilesCabinet,
) -> tuple[dict[str, dict], list[dict]]:
    """
    Build mission graph from local tyler_memories.json + NET-* intel files only.
    Avoids Mem0 get_all — use during bulk seed so each add_* is fast.
    """
    memories = _load_local_memory_entries(user_id)
    if not isinstance(memories, list):
        memories = []
    nodes, edges_map = _network_parse_nodes_edges_from_memories(memories)
    _network_merge_local_graph_entries(user_id, nodes, edges_map)
    _network_merge_intel_files(files_cabinet, nodes, edges_map)
    nodes, edges_map = _network_normalize_graph_nodes_edges(nodes, edges_map)
    return nodes, list(edges_map.values())


def _network_incident_edges(node_id: str, all_edges: list[dict]) -> list[dict]:
    return [
        e
        for e in all_edges
        if e.get("source_id") == node_id or e.get("target_id") == node_id
    ]


def _network_sync_intel_file_for_node(
    files_cabinet: FilesCabinet,
    node_id: str,
    node: dict,
    all_edges: list[dict],
    *,
    skip_mem0: bool = False,
) -> None:
    fn = network_intel_filename(node_id)
    incident = _network_incident_edges(node_id, all_edges)
    node_out = {k: v for k, v in node.items() if not str(k).startswith("_")}
    payload = json.dumps({"node": node_out, "edges": incident}, ensure_ascii=False, indent=2)
    tags = [
        "network_intel",
        f"type:{node_out.get('node_type', 'person')}",
        f"relevance:{node_out.get('relevance', 'MEDIUM')}",
    ]
    try:
        if files_cabinet.get_file(fn):
            files_cabinet.update_file(fn, payload, skip_mem0=skip_mem0)
        else:
            files_cabinet.create_file(
                NETWORK_INTEL_FOLDER, fn, payload, tags=tags, skip_mem0=skip_mem0
            )
    except ValueError:
        try:
            files_cabinet.update_file(fn, payload, skip_mem0=skip_mem0)
        except Exception:
            pass
    except Exception:
        pass


def _network_sync_affected_files(
    files_cabinet: FilesCabinet,
    nodes: dict[str, dict],
    edges: list[dict],
    affected_ids: set[str],
    *,
    skip_mem0: bool = False,
) -> None:
    for nid in affected_ids:
        n = nodes.get(nid)
        if n:
            _network_sync_intel_file_for_node(
                files_cabinet, nid, n, edges, skip_mem0=skip_mem0
            )


def add_network_node(
    name: str,
    node_type: str,
    description: str,
    relevance: str,
    tags: list[str] | None,
    *,
    memory_client,
    user_id: str,
    files_cabinet: FilesCabinet,
    use_mem0_cloud: bool = False,
    node_id_override: str | None = None,
    fast_local: bool = False,
) -> dict:
    now = _network_now_iso()
    nt = (node_type or "person").strip().lower()
    if nt not in NETWORK_NODE_TYPES:
        nt = "person"
    rel = (relevance or "MEDIUM").strip().upper()
    if rel not in NETWORK_NODE_RELEVANCE:
        rel = "MEDIUM"
    tagl = [str(t).strip() for t in (tags or []) if str(t).strip()][:30]
    if fast_local:
        nodes, edges = network_load_graph_local_only(user_id, files_cabinet)
    else:
        nodes, edges = network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)
    ovr = (node_id_override or "").strip()
    if ovr:
        nid = network_slug_from_display_name(ovr) if (" " in ovr or "/" in ovr) else ovr.lower()
    else:
        nid = network_slug_from_display_name(name)
    if not nid or nid == "unknown":
        nid = network_slug_from_display_name(name or "unknown")
    prev = nodes.get(nid)
    first_seen = (prev or {}).get("first_seen") or now
    node = {
        "id": nid,
        "name": ((name or nid).strip() or nid),
        "node_type": nt,
        "description": (description or "").strip(),
        "relevance": rel,
        "tags": tagl,
        "first_seen": first_seen,
        "last_updated": now,
    }
    nodes[nid] = node
    _network_upsert_structured_memory(
        memory_client,
        user_id,
        category=CATEGORY_NETWORK_NODE,
        entity_key=nid,
        text=json.dumps(node, ensure_ascii=False),
        use_mem0_cloud=use_mem0_cloud,
        skip_mem0=fast_local,
    )
    _network_sync_intel_file_for_node(
        files_cabinet, nid, node, edges, skip_mem0=fast_local
    )
    if not fast_local:
        try:
            import angel_proactive as _apro

            _apro.maybe_auto_watch_from_network(
                memory_client,
                user_id,
                files_cabinet,
                use_mem0_cloud,
                ((name or nid).strip() or nid),
                rel,
            )
        except Exception:
            pass
    return node


def add_network_edge(
    source_id: str,
    target_id: str,
    relationship_type: str,
    description: str,
    strength: str,
    evidence: str,
    *,
    memory_client,
    user_id: str,
    files_cabinet: FilesCabinet,
    use_mem0_cloud: bool = False,
    fast_local: bool = False,
) -> dict:
    sid = _network_normalize_node_id(source_id)
    tid = _network_normalize_node_id(target_id)
    rt = (relationship_type or "connected_to").strip().lower()
    if rt not in NETWORK_RELATIONSHIP_TYPES:
        rt = "connected_to"
    st = (strength or "MODERATE").strip().upper()
    if st not in NETWORK_EDGE_STRENGTHS:
        st = "MODERATE"
    desc = (description or "").strip()
    ev = (evidence or "").strip()
    now = _network_now_iso()
    eid = hashlib.sha256(f"{sid}|{tid}|{rt}|{desc[:160]}".encode("utf-8")).hexdigest()[:16]
    if fast_local:
        nodes, edges = network_load_graph_local_only(user_id, files_cabinet)
    else:
        nodes, edges = network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)

    def _ensure_stub(nid: str, display: str) -> None:
        if nid in nodes:
            return
        add_network_node(
            display or nid.replace("-", " ").title(),
            "person",
            "Auto-created network stub (edge endpoint).",
            "LOW",
            [],
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
            node_id_override=nid,
            fast_local=fast_local,
        )

    _ensure_stub(sid, source_id.strip())
    _ensure_stub(tid, target_id.strip())
    if fast_local:
        nodes, edges = network_load_graph_local_only(user_id, files_cabinet)
    else:
        nodes, edges = network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)

    edge = {
        "source_id": sid,
        "target_id": tid,
        "relationship_type": rt,
        "description": desc,
        "strength": st,
        "source_evidence": ev,
        "date_established": now,
        "edge_id": eid,
    }
    edges = [e for e in edges if e.get("edge_id") != eid]
    edges.append(edge)
    _network_upsert_structured_memory(
        memory_client,
        user_id,
        category=CATEGORY_NETWORK_EDGE,
        entity_key=eid,
        text=json.dumps(edge, ensure_ascii=False),
        use_mem0_cloud=use_mem0_cloud,
        skip_mem0=fast_local,
    )
    _network_sync_affected_files(
        files_cabinet, nodes, edges, {sid, tid}, skip_mem0=fast_local
    )
    return edge


def get_node_connections(
    node_id: str,
    *,
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: FilesCabinet,
) -> dict | None:
    nid = _network_normalize_node_id(node_id)
    nodes, edges = network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)
    node = nodes.get(nid)
    if not node:
        return None
    inc = _network_incident_edges(nid, edges)
    return {"node": node, "edges": inc}


def get_network_cluster(
    node_id: str,
    depth: int = 2,
    *,
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: FilesCabinet,
) -> dict | None:
    nid = _network_normalize_node_id(node_id)
    nodes, edges = network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)
    if nid not in nodes:
        return None
    depth = max(1, min(int(depth), 4))
    visited: set[str] = {nid}
    frontier = {nid}
    for _ in range(depth):
        nxt: set[str] = set()
        for e in edges:
            s, t = e.get("source_id"), e.get("target_id")
            if not s or not t:
                continue
            if s in frontier and t not in visited:
                nxt.add(t)
            if t in frontier and s not in visited:
                nxt.add(s)
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    sub_edges = [
        e
        for e in edges
        if e.get("source_id") in visited and e.get("target_id") in visited
    ]
    sub_nodes = {k: nodes[k] for k in visited if k in nodes}
    return {"center_id": nid, "nodes": sub_nodes, "edges": sub_edges, "depth": depth}


def find_path_between(
    node_id_a: str,
    node_id_b: str,
    *,
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: FilesCabinet,
) -> dict:
    a = _network_normalize_node_id(node_id_a)
    b = _network_normalize_node_id(node_id_b)
    nodes, edges = network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)
    if a not in nodes or b not in nodes:
        return {"path": [], "found": False, "error": "unknown node"}
    if a == b:
        return {"path": [a], "found": True, "edges": []}
    adj: dict[str, list[tuple[str, dict]]] = {}
    for e in edges:
        s, t = e.get("source_id"), e.get("target_id")
        if not s or not t:
            continue
        adj.setdefault(s, []).append((t, e))
        adj.setdefault(t, []).append((s, e))
    dq = deque([(a, [a], [])])
    seen = {a}
    while dq:
        cur, path, pedges = dq.popleft()
        for nb, ed in adj.get(cur, []):
            if nb in seen:
                continue
            npath = path + [nb]
            nedges = pedges + [ed]
            if nb == b:
                return {"path": npath, "found": True, "edges": nedges}
            seen.add(nb)
            dq.append((nb, npath, nedges))
    return {"path": [], "found": False, "edges": []}


def get_network_summary(
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: FilesCabinet,
) -> dict:
    nodes, edges = network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)
    deg: dict[str, int] = {nid: 0 for nid in nodes}
    for e in edges:
        s, t = e.get("source_id"), e.get("target_id")
        if s in deg:
            deg[s] += 1
        if t in deg:
            deg[t] += 1
    top = sorted(deg.items(), key=lambda x: -x[1])[:12]
    by_rel: dict[str, int] = {}
    for e in edges:
        rt = e.get("relationship_type") or "unknown"
        by_rel[rt] = by_rel.get(rt, 0) + 1
    by_rev: dict[str, int] = {}
    for n in nodes.values():
        r = n.get("relevance") or "MEDIUM"
        by_rev[str(r).upper()] = by_rev.get(str(r).upper(), 0) + 1
    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "nodes_by_relevance": by_rev,
        "edges_by_relationship_type": by_rel,
        "most_connected": [{"node_id": k, "degree": v, "name": nodes.get(k, {}).get("name", k)} for k, v in top],
    }


def network_resolve_name_to_id(needle: str, nodes: dict[str, dict]) -> str | None:
    q = (needle or "").strip().lower()
    if not q:
        return None
    cq = _network_canonical_endpoint_id(needle)
    if cq in nodes:
        return cq
    for nid, n in nodes.items():
        nm = (n.get("name") or "").strip().lower()
        if q == nid.lower() or q == nm or q in nm or nm in q:
            return nid
    return None


def map_osint_to_network(
    dossier_body: str,
    primary_target_display: str,
    *,
    primary_target_type: str = "person",
    anthropic_client: anthropic.Anthropic,
    memory_client,
    user_id: str,
    files_cabinet: FilesCabinet,
    use_mem0_cloud: bool,
) -> dict:
    """Extract people/orgs from a dossier and link them to the primary target + each other."""
    body = (dossier_body or "").strip()
    if len(body) > 14000:
        body = body[:13997] + "..."
    primary = (primary_target_display or "").strip()
    pid = network_slug_from_display_name(primary)
    if not primary:
        return {"ok": False, "error": "no primary target"}
    ptt = (primary_target_type or "person").strip().lower()
    if ptt not in NETWORK_NODE_TYPES:
        ptt = "person"
    added_nodes = 0
    added_edges = 0
    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4096,
            temperature=0.2,
            system="""You extract entities and relationships for an intelligence graph from an OSINT dossier.
Output ONLY valid JSON (no markdown):
{
  "entities": [
    {"name": "Display Name", "node_type": "person|organization|program|event|faction", "relevance": "LOW|MEDIUM|HIGH|CRITICAL", "tags": ["short"], "note": "one line"}
  ],
  "relationships": [
    {"source": "Name A", "target": "Name B", "relationship_type": "works_with|testified_with|employed_by|investigated_by|connected_to|corroborates|contradicts|funded_by|member_of|retaliates_against|opposes|suppresses", "description": "short", "strength": "WEAK|MODERATE|STRONG|CONFIRMED", "evidence": "from dossier"}
  ]
}
Include the primary subject in entities if not already listed. Add relationships only when the dossier supports them.""",
            messages=[
                {
                    "role": "user",
                    "content": f"Primary OSINT subject: {primary}\n\nDossier:\n{body}",
                }
            ],
        )
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        text = (text or "").strip()
        lb, rb = text.find("{"), text.rfind("}")
        if lb >= 0 and rb > lb:
            text = text[lb : rb + 1]
        data = json.loads(text)
    except Exception as e:
        return {"ok": False, "error": str(e), "added_nodes": 0, "added_edges": 0}

    entities = data.get("entities") if isinstance(data, dict) else []
    rels = data.get("relationships") if isinstance(data, dict) else []
    if not isinstance(entities, list):
        entities = []
    if not isinstance(rels, list):
        rels = []

    seen_names: set[str] = set()
    extracted_slugs: set[str] = set()
    known_ids = set(
        network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)[0].keys()
    )

    def _add_ent(name: str, ntype: str, rel: str, tags: list, desc: str) -> str:
        nonlocal added_nodes
        nm = (name or "").strip()
        if not nm:
            return ""
        key = nm.lower()
        slug = network_slug_from_display_name(nm)
        if key in seen_names:
            extracted_slugs.add(slug)
            return slug
        seen_names.add(key)
        is_new = slug not in known_ids
        add_network_node(
            nm,
            ntype,
            desc or f"Mentioned in OSINT dossier on {primary}.",
            rel,
            tags,
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
        )
        known_ids.add(slug)
        if is_new:
            added_nodes += 1
        extracted_slugs.add(slug)
        return slug

    _add_ent(
        primary,
        ptt,
        "HIGH",
        ["osint_primary"],
        f"Primary OSINT subject: {primary}.",
    )

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        nm = (ent.get("name") or "").strip()
        if not nm or nm.lower() == primary.lower():
            continue
        if network_slug_from_display_name(nm) == pid:
            continue
        nt = (ent.get("node_type") or "person").strip().lower()
        if nt not in NETWORK_NODE_TYPES:
            nt = "person"
        rv = (ent.get("relevance") or "MEDIUM").strip().upper()
        if rv not in NETWORK_NODE_RELEVANCE:
            rv = "MEDIUM"
        tags = ent.get("tags") if isinstance(ent.get("tags"), list) else []
        note = (ent.get("note") or "").strip()
        _add_ent(nm, nt, rv, [str(t) for t in tags if t][:20], note)

    for r in rels:
        if not isinstance(r, dict):
            continue
        sa = (r.get("source") or "").strip()
        sb = (r.get("target") or "").strip()
        if not sa or not sb:
            continue
        try:
            add_network_edge(
                sa,
                sb,
                r.get("relationship_type") or "connected_to",
                r.get("description") or "",
                r.get("strength") or "MODERATE",
                r.get("evidence") or "OSINT dossier extraction",
                memory_client=memory_client,
                user_id=user_id,
                files_cabinet=files_cabinet,
                use_mem0_cloud=use_mem0_cloud,
            )
            added_edges += 1
        except Exception:
            pass

    for nid in extracted_slugs:
        if nid == pid:
            continue
        try:
            add_network_edge(
                pid,
                nid,
                "connected_to",
                f"Mentioned in OSINT dossier on {primary}.",
                "WEAK",
                "Co-occurrence in OSINT dossier",
                memory_client=memory_client,
                user_id=user_id,
                files_cabinet=files_cabinet,
                use_mem0_cloud=use_mem0_cloud,
            )
            added_edges += 1
        except Exception:
            pass

    return {"ok": True, "added_nodes": added_nodes, "added_edges": added_edges, "primary_id": pid}


def _purge_local_network_memory_entries(user_id: str) -> int:
    """Remove all network_node / network_edge rows from tyler_memories.json for user_id."""
    cats = frozenset({CATEGORY_NETWORK_NODE, CATEGORY_NETWORK_EDGE})
    try:
        entries = _load_local_memory_entries(user_id)
        if not isinstance(entries, list):
            return 0
        removed = 0
        kept: list = []
        for e in entries:
            if isinstance(e, dict):
                meta = e.get("metadata")
                if isinstance(meta, dict) and meta.get("category") in cats:
                    removed += 1
                    continue
            kept.append(e)
        if removed:
            _save_local_memory_entries(user_id, kept)
        return removed
    except Exception:
        return 0


def _purge_mem0_network_graph_memories(
    memory_client, user_id: str, *, use_mem0_cloud: bool
) -> int:
    """
    Delete Mem0 cloud memories with category network_node or network_edge.
    Hard cap: 30 seconds wall time — returns early with however many were deleted.
    """
    if not use_mem0_cloud or not isinstance(memory_client, Mem0CloudClient):
        return 0
    deleted = 0
    deadline = time.monotonic() + 30.0
    page_sz = 200
    for _ in range(100):
        if time.monotonic() >= deadline:
            return deleted
        found_any = False
        page = 1
        while page <= 500:
            if time.monotonic() >= deadline:
                return deleted
            try:
                raw = memory_client.get_all(user_id=user_id, page=page, page_size=page_sz)
            except Exception:
                return deleted
            results = raw.get("results") if isinstance(raw, dict) else raw
            if not isinstance(results, list) or not results:
                break
            for item in results:
                if time.monotonic() >= deadline:
                    return deleted
                if not isinstance(item, dict):
                    continue
                meta = item.get("metadata") or {}
                if not isinstance(meta, dict):
                    continue
                if meta.get("category") not in (
                    CATEGORY_NETWORK_NODE,
                    CATEGORY_NETWORK_EDGE,
                ):
                    continue
                rid = (item.get("id") or item.get("memory_id") or "").strip()
                if not rid:
                    continue
                try:
                    memory_client.delete_memory(rid)
                    deleted += 1
                    found_any = True
                except Exception as e:
                    _mem0_log.debug(
                        "purge network graph Mem0 delete skipped id=%s…: %s",
                        str(rid)[:24],
                        e,
                    )
            if len(results) < page_sz:
                break
            page += 1
        if not found_any:
            break
    return deleted


def reset_mission_network_and_reseed(
    memory_client,
    user_id: str,
    files_cabinet: FilesCabinet,
    use_mem0_cloud: bool,
) -> dict:
    """
    Recovery: delete Network Intelligence mirror files only, then force a full mission seed.
    Skips slow Mem0 cloud deletes — nodes/edges are upserted in place (canonical JSON also on disk).
    """
    print("[reset] step 1: deleting Network Intelligence files", flush=True)
    out: dict = {
        "ok": True,
        "intel_files_deleted": 0,
        "local_network_entries_removed": 0,
        "mem0_network_memories_deleted": 0,
        "seed_ran": False,
        "nodes_after": 0,
        "edges_after": 0,
        "error": None,
    }
    try:
        try:
            recs = files_cabinet.list_files(folder=NETWORK_INTEL_FOLDER)
        except Exception:
            recs = []
        for rec in recs:
            name = (rec.get("name") or "").strip()
            if not name:
                continue
            try:
                if files_cabinet.delete_file(name):
                    out["intel_files_deleted"] += 1
            except Exception:
                pass

        print(f"[reset] step 2: files deleted, count={out['intel_files_deleted']}", flush=True)
        print("[reset] step 3: calling seed_mission_network_if_empty force=True", flush=True)
        out["seed_ran"] = seed_mission_network_if_empty(
            memory_client,
            user_id,
            files_cabinet,
            use_mem0_cloud,
            force=True,
            timeout_seconds=90.0,
        )
        # Local counts only — avoid fetch_combined_memories (Mem0 get_all) here; it can
        # exceed the web_app reset watchdog (120s) and is redundant right after local-first seed.
        nodes, edges = network_load_graph_local_only(user_id, files_cabinet)
        out["nodes_after"] = len(nodes)
        out["edges_after"] = len(edges)
        print(
            f"[reset] step 4: seed complete, nodes={out['nodes_after']} edges={out['edges_after']} (local)",
            flush=True,
        )
        print("[reset] step 5: returning result", flush=True)
    except Exception as e:
        print(f"[reset] EXCEPTION before step 5: {e!r}", flush=True)
        traceback.print_exc()
        out["ok"] = False
        out["error"] = str(e)
    return out


def _network_background_mem0_full_sync(
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: FilesCabinet,
) -> None:
    """Push local network_node / network_edge rows and NET-* files to Mem0 (runs in a daemon thread)."""
    if not use_mem0_cloud:
        return
    try:
        for entry in _load_local_memory_entries(user_id):
            if not isinstance(entry, dict):
                continue
            meta = entry.get("metadata") or {}
            if not isinstance(meta, dict):
                continue
            cat = meta.get("category")
            if cat not in (CATEGORY_NETWORK_NODE, CATEGORY_NETWORK_EDGE):
                continue
            ek = (meta.get("network_entity_key") or "").strip()
            raw = (entry.get("memory") or entry.get("data") or "").strip()
            if not ek or not raw:
                continue
            _network_upsert_structured_memory(
                memory_client,
                user_id,
                category=cat,
                entity_key=ek,
                text=raw,
                use_mem0_cloud=use_mem0_cloud,
                skip_mem0=False,
            )
        try:
            recs = files_cabinet.list_files(folder=NETWORK_INTEL_FOLDER)
        except Exception:
            recs = []
        for rec in recs:
            fn = (rec.get("name") or "").strip()
            if not fn:
                continue
            try:
                full = files_cabinet.get_file(fn)
                if not full:
                    continue
                body = (full.get("content") or "").strip()
                if not body:
                    continue
                files_cabinet.update_file(fn, body, skip_mem0=False)
            except Exception:
                pass
    except Exception:
        pass
    print("[seed] Mem0 background sync: finished", flush=True)


def _seed_mission_network_if_empty_core(
    memory_client,
    user_id: str,
    files_cabinet: FilesCabinet,
    use_mem0_cloud: bool,
    *,
    force: bool,
) -> bool:
    """
    Local-first graph seed: tyler_memories.json + NET-* files without per-row Mem0 calls.
    Caller spawns _network_background_mem0_full_sync after this returns.
    """
    if not force:
        _, edges = network_load_graph_local_only(user_id, files_cabinet)
        if len(edges) > 0:
            return False

    # Relevance: Grusch & Elizondo CRITICAL; all other seed entities HIGH (see network summary / mission graph).
    seed_nodes: list[tuple[str, str, str, str, list[str]]] = [
        ("David Grusch", "person", "UAP whistleblower; former intelligence community.", "CRITICAL", ["UAP", "disclosure", "whistleblower"]),
        ("Luis Elizondo", "person", "Former AATIP / UAP program visibility.", "CRITICAL", ["UAP", "AATIP", "disclosure"]),
        ("Christopher Mellon", "person", "Former Deputy Assistant Secretary of Defense for Intelligence; disclosure advocate.", "HIGH", ["UAP", "disclosure", "Pentagon"]),
        ("Ross Coulthart", "person", "Investigative journalist covering UAP.", "HIGH", ["UAP", "media", "disclosure"]),
        ("Marco Rubio", "person", "Senior US official; public UAP-related statements.", "HIGH", ["UAP", "government"]),
        ("AARO", "organization", "DoD All-domain Anomaly Resolution Office (UAP).", "HIGH", ["UAP", "DoD", "government"]),
        ("House Oversight Committee", "organization", "US House committee; UAP hearings context.", "HIGH", ["UAP", "Congress", "government"]),
        ("Pentagon", "organization", "US Department of Defense headquarters.", "HIGH", ["DoD", "government", "military"]),
        ("NRO", "organization", "National Reconnaissance Office.", "HIGH", ["intelligence", "government"]),
        ("NGA", "organization", "National Geospatial-Intelligence Agency.", "HIGH", ["intelligence", "government", "military"]),
        ("AATIP", "program", "Advanced Aerospace Threat Identification Program (historical DoD UAP effort).", "HIGH", ["UAP", "DoD", "program"]),
    ]
    _SEED_EDGE_COUNT = 15  # 12 main graph + 3 Marco Rubio cluster edges

    print(f"[seed] writing {len(seed_nodes)} nodes to local JSON", flush=True)
    seen_seed_slugs: set[str] = set()
    for name, nt, desc, rel, tags in seed_nodes:
        sid = network_slug_from_display_name(name)
        if sid in seen_seed_slugs:
            continue
        seen_seed_slugs.add(sid)
        add_network_node(
            name,
            nt,
            desc,
            rel,
            tags,
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
            fast_local=True,
        )

    print(f"[seed] writing {_SEED_EDGE_COUNT} edges to local JSON", flush=True)

    def e(a, b, rt, desc, st="STRONG", ev="Mission seed graph"):
        return add_network_edge(
            a,
            b,
            rt,
            desc,
            st,
            ev,
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
            fast_local=True,
        )

    e("David Grusch", "NRO", "connected_to", "Open-source mission context link.", ev="Seed data")
    e("David Grusch", "NGA", "connected_to", "Open-source mission context link.", ev="Seed data")
    e("David Grusch", "AARO", "connected_to", "UAP oversight / reporting context.", ev="Seed data")
    e("David Grusch", "House Oversight Committee", "testified_with", "Congressional UAP hearing context.", "CONFIRMED", "Seed data")
    e("Luis Elizondo", "Pentagon", "employed_by", "Former DoD context.", "STRONG", "Seed data")
    e("Luis Elizondo", "AATIP", "member_of", "Program association (open sources).", "STRONG", "Seed data")
    e("Luis Elizondo", "David Grusch", "corroborates", "Disclosure-adjacent narrative alignment (OSINT).", "MODERATE", "Seed data")
    e("Christopher Mellon", "Pentagon", "employed_by", "Former DASD(I) role (historical).", "STRONG", "Seed data")
    e("Christopher Mellon", "Luis Elizondo", "works_with", "Advocacy / public UAP work.", "MODERATE", "Seed data")
    e("Christopher Mellon", "David Grusch", "corroborates", "Disclosure ecosystem.", "MODERATE", "Seed data")
    e("Ross Coulthart", "Luis Elizondo", "connected_to", "Media interviews / reporting.", "MODERATE", "Seed data")
    e("Ross Coulthart", "David Grusch", "connected_to", "Media interviews / reporting.", "MODERATE", "Seed data")
    e("Ross Coulthart", "Christopher Mellon", "connected_to", "Media / public commentary.", "MODERATE", "Seed data")
    # Marco Rubio cluster — pass canonical slugs (marco-rubio, aaro, house-oversight-committee, david-grusch)
    _mr_edges = [
        (
            "marco-rubio",
            "aaro",
            "connected_to",
            "Public statements on UAP governance and oversight context.",
            "MODERATE",
            "Seed data",
        ),
        (
            "marco-rubio",
            "house-oversight-committee",
            "connected_to",
            "Congressional UAP hearings / oversight context (role varies by cycle).",
            "MODERATE",
            "Seed data",
        ),
        (
            "marco-rubio",
            "david-grusch",
            "corroborates",
            "Public UAP statements aligned with disclosure themes (open sources).",
            "WEAK",
            "Seed data",
        ),
    ]
    _mr_stored: list[dict[str, str | None]] = []
    for src_raw, tgt_raw, rt, desc, st, ev in _mr_edges:
        edge_out = e(src_raw, tgt_raw, rt, desc, st, ev)
        _mr_stored.append(
            {
                "edge_id": edge_out.get("edge_id"),
                "source_id": edge_out.get("source_id"),
                "target_id": edge_out.get("target_id"),
                "rel": rt,
            }
        )
    _mission_graph_log.debug(
        "seed_mission_network_if_empty: Marco Rubio cluster %d edges edge_ids=%s detail=%s",
        len(_mr_stored),
        [x.get("edge_id") for x in _mr_stored],
        _mr_stored,
    )

    print("[seed] local write complete, starting Mem0 background sync", flush=True)
    if use_mem0_cloud:
        try:
            threading.Thread(
                target=_network_background_mem0_full_sync,
                args=(memory_client, user_id, use_mem0_cloud, files_cabinet),
                daemon=True,
                name="network-graph-mem0-sync",
            ).start()
        except Exception:
            pass
    else:
        print("[seed] skip Mem0 background sync (cloud off)", flush=True)

    _mission_graph_log.debug("seed_mission_network_if_empty: mission graph seed finished successfully")
    return True


def seed_mission_network_if_empty(
    memory_client,
    user_id: str,
    files_cabinet: FilesCabinet,
    use_mem0_cloud: bool,
    *,
    force: bool = False,
    timeout_seconds: float = 90.0,
) -> bool:
    """
    Seed Tyler mission graph when there are no edges yet (orphan nodes still trigger seed).
    If force=True, always run the full seed (upserts overwrite existing nodes/edges by id).
    Writes local JSON first; Mem0 cloud sync runs in a background thread when enabled.
    """
    outcome = {"ok": None, "err": None}  # ok: bool|None, err: BaseException|None

    def _runner() -> None:
        try:
            outcome["ok"] = _seed_mission_network_if_empty_core(
                memory_client,
                user_id,
                files_cabinet,
                use_mem0_cloud,
                force=force,
            )
        except BaseException as ex:
            outcome["err"] = ex

    th = threading.Thread(target=_runner, daemon=True, name="mission-network-seed")
    th.start()
    th.join(timeout=timeout_seconds)
    if th.is_alive():
        print(
            f"[seed] TIMEOUT after {timeout_seconds}s — seed thread still running; "
            "local graph may be partial",
            flush=True,
        )
        return False
    if outcome["err"] is not None:
        _mission_graph_log.exception("seed_mission_network_if_empty failed")
        return False
    return bool(outcome["ok"])


_mission_network_seed_thread_started = False
_mission_network_seed_thread_lock = threading.Lock()
_mission_network_seed_run_lock = threading.Lock()


def schedule_mission_network_seed_background(
    memory_client,
    user_id: str,
    files_cabinet: FilesCabinet,
    use_mem0_cloud: bool,
) -> None:
    """
    Fire-and-forget: at most one daemon thread sleeps 10s then runs seed_mission_network_if_empty
    (only when the graph still has zero edges). Never schedules two threads; seed body uses a
    non-blocking lock so concurrent runs are skipped.
    """
    global _mission_network_seed_thread_started
    try:
        with _mission_network_seed_thread_lock:
            if _mission_network_seed_thread_started:
                return
            _mission_network_seed_thread_started = True
    except Exception:
        return

    def _job() -> None:
        try:
            time.sleep(10)
        except Exception:
            return
        if not _mission_network_seed_run_lock.acquire(blocking=False):
            return
        try:
            try:
                seed_mission_network_if_empty(
                    memory_client,
                    user_id,
                    files_cabinet,
                    use_mem0_cloud,
                    timeout_seconds=90.0,
                )
            except Exception:
                pass
        finally:
            _mission_network_seed_run_lock.release()

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception:
        pass


def detect_network_command(user_message: str) -> tuple[str | None, dict]:
    """
    Returns (command, payload) where command is summary | cluster | path | connections.
    """
    raw = (user_message or "").strip()
    if not raw:
        return None, {}
    lower = raw.lower()
    if re.search(r"(?i)\bshow\s+me\s+the\s+network\b", lower) or re.search(
        r"(?i)\bmission\s+network\s+summary\b", lower
    ):
        return "summary", {}

    m = re.search(r"(?i)\bmap\s+connections\s+(?:for|on)\s+(.+)$", raw)
    if m:
        return "cluster", {"name": m.group(1).strip().rstrip("?.!")}

    m = re.search(r"(?i)\bwho\s+does\s+(.+?)\s+know\b", raw)
    if m:
        return "connections", {"name": m.group(1).strip().rstrip("?.!")}

    m = re.search(
        r"(?i)\bhow\s+is\s+(.+?)\s+connected\s+to\s+(.+)$",
        raw,
    )
    if m:
        return "path", {"from_name": m.group(1).strip(), "to_name": m.group(2).strip().rstrip("?.!")}

    return None, {}


def format_network_command_result_for_prompt(
    command: str,
    payload: dict,
    *,
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: FilesCabinet,
) -> str:
    nodes, _ = network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)
    if command == "summary":
        s = get_network_summary(memory_client, user_id, use_mem0_cloud, files_cabinet)
        return f"[Mission network summary]\n{json.dumps(s, indent=2, ensure_ascii=False)[:8000]}"
    if command == "cluster":
        nid = network_resolve_name_to_id(payload.get("name") or "", nodes)
        if not nid:
            return f"[Network] No node matched {payload.get('name')!r}."
        cl = get_network_cluster(
            nid,
            depth=2,
            memory_client=memory_client,
            user_id=user_id,
            use_mem0_cloud=use_mem0_cloud,
            files_cabinet=files_cabinet,
        )
        return f"[Network cluster from {nid} depth=2]\n{json.dumps(cl, indent=2, ensure_ascii=False)[:8000]}"
    if command == "connections":
        nid = network_resolve_name_to_id(payload.get("name") or "", nodes)
        if not nid:
            return f"[Network] No node matched {payload.get('name')!r}."
        c = get_node_connections(
            nid,
            memory_client=memory_client,
            user_id=user_id,
            use_mem0_cloud=use_mem0_cloud,
            files_cabinet=files_cabinet,
        )
        return f"[Connections for {nid}]\n{json.dumps(c, indent=2, ensure_ascii=False)[:8000]}"
    if command == "path":
        a = network_resolve_name_to_id(payload.get("from_name") or "", nodes)
        b = network_resolve_name_to_id(payload.get("to_name") or "", nodes)
        if not a or not b:
            return f"[Network] Could not resolve path endpoints (from={payload!r})."
        p = find_path_between(
            a,
            b,
            memory_client=memory_client,
            user_id=user_id,
            use_mem0_cloud=use_mem0_cloud,
            files_cabinet=files_cabinet,
        )
        return f"[Shortest path {a} → {b}]\n{json.dumps(p, indent=2, ensure_ascii=False)[:8000]}"
    return ""


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
        try:
            import angel_self_modification as _angel_self_mod

            _sm_intent = _angel_self_mod.detect_self_mod_intent(user_message)
            if _sm_intent:
                _cmd, _arg = _sm_intent
                _sm_out = _angel_self_mod.handle_self_mod_intent(
                    self, _cmd, _arg, user_message
                )
                if (_sm_out or "").strip():
                    return _sm_out.strip()
        except Exception:
            pass

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
        osint_triggered, osint_target, osint_type = detect_osint_request(user_message)
        net_cmd, net_payload = detect_network_command(user_message)
        pred_cmd, pred_payload = None, {}
        try:
            import angel_predictions as apred

            pred_cmd, pred_payload = apred.detect_prediction_intent(user_message)
        except Exception:
            pred_cmd, pred_payload = None, {}

        proactive_cmd, proactive_payload = None, {}
        try:
            import angel_proactive as apro

            proactive_cmd, proactive_payload = apro.detect_proactive_intent(user_message)
            apro.track_user_mentions_for_watch(self, user_message)
        except Exception:
            proactive_cmd, proactive_payload = None, {}

        trans_cmd, trans_payload = None, {}
        try:
            import angel_translation as tr

            trans_cmd, trans_payload = tr.detect_translation_intent(user_message)
        except Exception:
            trans_cmd, trans_payload = None, {}

        file_cmd, file_payload = None, {}
        try:
            import angel_file_reading as fr

            if trans_cmd not in ("translate_paste", "translate_explicit", "foreign_search"):
                file_cmd, file_payload = fr.detect_file_read_intent(user_message)
        except Exception:
            file_cmd, file_payload = None, {}

        forensic_cmd, forensic_payload = None, {}
        try:
            import angel_forensic as af

            forensic_cmd, forensic_payload = af.detect_forensic_chat_intent(user_message)
            if forensic_cmd:
                file_cmd, file_payload = None, {}
        except Exception:
            forensic_cmd, forensic_payload = None, {}

        surv_cmd, surv_payload = None, {}
        try:
            import angel_surveillance as asurv

            surv_cmd, surv_payload = asurv.detect_surveillance_chat_intent(user_message)
        except Exception:
            surv_cmd, surv_payload = None, {}

        map_cmd, map_payload = None, {}
        try:
            import angel_environmental_map as aemap

            map_cmd, map_payload = aemap.detect_map_chat_intent(user_message)
        except Exception:
            map_cmd, map_payload = None, {}

        comms_cmd, comms_payload = None, {}
        try:
            import angel_communication_patterns as acomm

            comms_cmd, comms_payload = acomm.detect_comms_chat_intent(user_message)
        except Exception:
            comms_cmd, comms_payload = None, {}

        bio_cmd, bio_payload = None, {}
        try:
            import angel_biological_intelligence as abio

            bio_cmd, bio_payload = abio.detect_bio_chat_intent(user_message)
        except Exception:
            bio_cmd, bio_payload = None, {}

        hist_cmd, hist_payload = None, {}
        try:
            import angel_historical_archives as ahist

            hist_cmd, hist_payload = ahist.detect_hist_chat_intent(user_message)
        except Exception:
            hist_cmd, hist_payload = None, {}

        ta_cmd, ta_payload = None, {}
        try:
            import angel_threat_actors as ata

            ta_cmd, ta_payload = ata.detect_threat_actor_chat_intent(user_message)
        except Exception:
            ta_cmd, ta_payload = None, {}

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
        try:
            import angel_self_modification as _angel_self_mod2

            _pend = _angel_self_mod2.consume_pending_proposal_notification()
            if (_pend or "").strip():
                system_prompt += (
                    "\n\n[System notice — Stage 6 self-modification: "
                    + _pend.strip()
                    + " If it fits the moment, mention it naturally once. "
                    "Remind Tyler that proposals require his approval before you change how you operate.]"
                )
        except Exception:
            pass
        if added_threat_watch:
            system_prompt += (
                "\n\n[System notice: Tyler asked to add a standing threat-watch topic. "
                "It is already saved (category threat_watch) and will merge into the next automated scans. "
                f"Confirm briefly using his wording: {added_threat_watch!r}.]"
            )
        if osint_triggered and osint_target:
            system_prompt += (
                "\n\nThis turn may include an OSINT dossier appendix (open-source background on a person or organization). "
                "If it is present, you already ran or loaded that research—open by acknowledging you pulled sources, "
                "then summarize key facts, red flags, and mission relevance; name the dossier file in OSINT Dossiers. "
                "If the appendix says OSINT failed, explain briefly and offer to retry. "
                "Do not claim classified or non-public sources."
            )
        if net_cmd:
            system_prompt += (
                "\n\nTyler asked about the mission connection graph (Batcomputer network). "
                "If a JSON block labeled [Mission network …] appears in the user message, summarize it in plain language "
                "and offer to explore clusters or paths further."
            )
        if pred_cmd:
            system_prompt += (
                "\n\nTyler's message relates to Angel's predictive modeling (forecasts and accuracy). "
                "If a block labeled [Angel predictions …] appears in the user message, weave it into a natural answer: "
                "summarize active forecasts when listing them, stress uncertainty, and connect to what Tyler asked. "
                "If he asked for a new prediction about a topic, acknowledge that new forecasts were generated server-side "
                "and interpret them—do not invent additional structured predictions in prose without the system's JSON."
            )
        if proactive_cmd:
            system_prompt += (
                "\n\nTyler's message relates to proactive background intelligence (standing watch list). "
                "If a block labeled [Angel proactive watch list] or [Proactive watch] or [Proactive findings] appears, "
                "answer naturally: confirm what you're monitoring, offer to adjust watches, and surface anything urgent. "
                "When you added a watch in this turn, tell Tyler explicitly that it's on your list."
            )
        if trans_cmd:
            system_prompt += (
                "\n\nTyler's message triggered translation / foreign-source analysis. "
                "If a block labeled [Angel translation] or [Foreign-source search] appears, summarize it in plain language: "
                "give the English sense, mission relevance, and any diplomatic or linguistic nuance. "
                "Note when foreign reporting corroborates or contradicts typical U.S. domestic framing—without claiming classified access."
            )
        if file_cmd:
            system_prompt += (
                "\n\nTyler shared or referenced a document for reading/analysis. "
                "If a block labeled [Angel document / file analysis] appears, integrate it: summarize key findings, mission relevance, "
                "and any **network_matches** or **osint_dossier_hits** (flag names that already appear in the mission graph or OSINT dossiers). "
                "If intelligence_value is HIGH or CRITICAL, proactively offer to file the document in the suggested Intelligence folder "
                "and offer to refresh or deepen OSINT on named entities. Do not claim classified or non-public sources."
            )
        if ta_cmd:
            system_prompt += (
                "\n\nTyler's message relates to the **Threat Actor** database (Batcomputer opposition layer—actors portrayed in open sources as working against disclosure or transparency). "
                "If a block labeled [Angel Threat Actors], [Angel Threat Actor assessment], or [Angel opposition lookup] appears, use it: "
                "distinguish **opposition** (Threat Actors / TA-* files) from **allies** (OSINT Dossiers). No classified claims."
            )
        if forensic_cmd:
            system_prompt += (
                "\n\nTyler asked for **forensic visual analysis** (or similar): authenticity, manipulation cues, UAP-relevant assessment, or document-photo extraction. "
                "If a block labeled [Angel forensic visual analysis] appears, integrate it: summarize the four layers, the authenticity confidence, UAP assessment if present, "
                "and any **mission_cross_reference** hits. Stress uncertainty and that this is open-source visual inference—not lab chain-of-custody. "
                "If the block says no image was attached, tell Tyler to use POST /api/forensic/analyze (or /uap /document) or paste a data-URI image."
            )
        if surv_cmd:
            system_prompt += (
                "\n\nTyler asked about **open-source surveillance monitoring** (legal public OSINT: flight/maritime/news/records patterns). "
                "If a block labeled [Angel surveillance monitoring] appears, summarize recent signals by category, highlight STRONG or **correlated** clusters, "
                "and connect to mission context without implying classified collection."
            )
        if map_cmd:
            system_prompt += (
                "\n\nTyler's message relates to the **Environmental map** (Batcomputer geography layer). "
                "If a block labeled [Angel environmental map] or [Angel environmental map — research] or [Angel environmental map — nearby] appears, "
                "answer in plain language: summarize the JSON profile, significance, incident/program links, and mission relevance. "
                "For research results, confirm the location was added or updated and the `LOC-*` file under Environmental Map when successful."
            )
        if comms_cmd:
            system_prompt += (
                "\n\nTyler's message relates to **communication pattern intelligence** (when/how figures communicate in open sources — cadence, silence, escalation, coordination). "
                "If a block labeled [Angel communication patterns] appears, interpret the JSON: name the figure(s), last-activity hints, anomalies (SILENCE, ESCALATION, etc.), and any coordination signals. "
                "Stress this is **pattern** analysis from public material, not wiretaps or private messages."
            )
        if bio_cmd:
            system_prompt += (
                "\n\nTyler's message relates to **biological / medical intelligence** (UAP-adjacent health patterns, exposure indicators, witness narratives — including the black-eyed people profile when relevant). "
                "If a block labeled [Angel biological intelligence …] appears, integrate it with care: this is **not** a clinical diagnosis; cite uncertainty, mission relevance, and Tyler's personal connection to BEK research when on-topic. "
                "Encourage professional medical evaluation when appropriate."
            )
        if hist_cmd:
            system_prompt += (
                "\n\nTyler's message relates to the **Historical Intelligence Archives** (UAP timeline, programs, documents). "
                "If a block labeled [Angel historical archives …] appears, summarize key records in plain language, connect people/programs to **current** mission threads, and flag uncertainty or contested claims. "
                "You may naturally relate today's events to historical parallels ('rhymes') when supported by the JSON."
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
        osint_ran = False
        osint_attempted = bool(osint_triggered and osint_target)

        if osint_triggered and osint_target:
            if os.getenv("TAVILY_API_KEY"):
                print(
                    f"{Fore.BLUE}Angel: OSINT deep background on {osint_target!r} ({osint_type})...{Style.RESET_ALL}"
                )
                os_res = run_osint_background(
                    osint_target,
                    osint_type,
                    user_message,
                    anthropic_client=self.anthropic_client,
                    files_cabinet=self.files_cabinet,
                    memory_summary=memory_summary,
                )
                if os_res.get("ok"):
                    osint_ran = True
                    net_map_note = ""
                    if not os_res.get("cached"):
                        try:
                            nm = map_osint_to_network(
                                os_res.get("dossier_body") or "",
                                osint_target,
                                primary_target_type=osint_type,
                                anthropic_client=self.anthropic_client,
                                memory_client=self.memory_client,
                                user_id=self.user_id,
                                files_cabinet=self.files_cabinet,
                                use_mem0_cloud=self._use_mem0_cloud,
                            )
                            if nm.get("ok"):
                                net_map_note = (
                                    f"\n[Mission network updated from this dossier: +{nm.get('added_nodes', 0)} nodes, "
                                    f"+{nm.get('added_edges', 0)} edges.]"
                                )
                        except Exception:
                            pass
                    hist_note = ""
                    hal = os_res.get("historical_archive_links")
                    if isinstance(hal, list) and hal:
                        hist_note = "\n[Historical Archives matches — cite HIST-* files when relevant:]\n" + json.dumps(
                            hal[:12], ensure_ascii=False, indent=2
                        )[:6000]
                    cache_note = " — existing dossier reused (within 30 days)" if os_res.get("cached") else ""
                    rf = os_res.get("red_flags") or []
                    rf_txt = "\n".join(f"- {x}" for x in rf) if rf else "- (none listed)"
                    augmented_user_message = (
                        f"{user_message}\n\n"
                        f"[OSINT DOSSIER{cache_note} — cabinet file {os_res.get('file_name')} in folder "
                        f"'{OSINT_DOSSIERS_FOLDER}'. Mission relevance: {os_res.get('mission_relevance')}.]\n"
                        f"Red flags:\n{rf_txt}\n\n"
                        f"Full dossier text for your summary:\n{os_res.get('dossier_body', '')[:14000]}"
                        f"{net_map_note}{hist_note}"
                    )
                else:
                    augmented_user_message = (
                        f"{user_message}\n\n[OSINT research did not complete: {os_res.get('error', 'unknown')}]"
                    )
            else:
                augmented_user_message = (
                    f"{user_message}\n\n[OSINT unavailable: TAVILY_API_KEY is not set in this environment.]"
                )

        parallel_done = False
        try:
            import angel_parallel_agents as apa

            use_p, ptopic = apa.detect_parallel_opportunity(user_message)
        except Exception:
            use_p, ptopic = False, None

        if (
            not osint_attempted
            and os.getenv("TAVILY_API_KEY")
            and use_p
            and ptopic
        ):
            try:
                depth = (
                    "deep"
                    if re.search(r"\bdeep\s+dive\b", user_message, re.I)
                    else "standard"
                )
                tasks = apa.decompose_into_parallel_tasks(
                    ptopic, user_message, depth=depth
                )
                print(
                    f"{Fore.BLUE}Angel: parallel agents ({len(tasks)}) — {ptopic[:100]!r}…{Style.RESET_ALL}"
                )
                pr = apa.run_parallel_agents(
                    tasks,
                    ptopic,
                    anthropic_client=self.anthropic_client,
                    memory_summary=memory_summary,
                    user_id=self.user_id,
                )
                if pr.get("ok"):
                    tnote = pr.get("time_saved_note") or ""
                    n_ag = pr.get("agents_used", 0)
                    augmented_user_message = (
                        f"Running parallel analysis across {n_ag} specialized agents…\n\n"
                        f"{user_message}\n\n"
                        f"[Parallel multi-agent analysis: {n_ag} agents, "
                        f"~{pr.get('total_time', 0):.1f}s wall time. {tnote}]\n\n"
                        f"{pr.get('synthesis', '')}"
                    )
                    parallel_done = True
            except Exception as ex:
                print(f"{Fore.YELLOW}Parallel agents error: {ex}{Style.RESET_ALL}", flush=True)

        # Stage 2 Deep Research: multi-angle Tavily + synthesis when triggered (skip if this turn was an OSINT request)
        if (
            comm_intent.intent == "briefing"
            and os.getenv("TAVILY_API_KEY")
            and not osint_attempted
            and not parallel_done
        ):
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
        elif research_requested and not osint_attempted and not parallel_done:
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
        elif not osint_attempted and not parallel_done:
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

        if net_cmd:
            nblock = format_network_command_result_for_prompt(
                net_cmd,
                net_payload,
                memory_client=self.memory_client,
                user_id=self.user_id,
                use_mem0_cloud=self._use_mem0_cloud,
                files_cabinet=self.files_cabinet,
            )
            if nblock.strip():
                augmented_user_message = f"{augmented_user_message}\n\n{nblock}"

        if pred_cmd:
            try:
                import angel_predictions as apred

                pblock = apred.format_prediction_reply_for_prompt(
                    pred_cmd,
                    pred_payload,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    use_mem0_cloud=self._use_mem0_cloud,
                    anthropic_client=self.anthropic_client,
                    files_cabinet=self.files_cabinet,
                )
                if pblock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{pblock}"
            except Exception:
                pass

        if proactive_cmd:
            try:
                import angel_proactive as apro

                pblock = apro.format_proactive_reply_for_prompt(
                    proactive_cmd,
                    proactive_payload,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    use_mem0_cloud=self._use_mem0_cloud,
                    files_cabinet=self.files_cabinet,
                )
                if pblock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{pblock}"
            except Exception:
                pass

        if trans_cmd:
            try:
                import angel_translation as tr

                tblock = tr.format_translation_for_prompt(
                    trans_cmd,
                    trans_payload,
                    anthropic_client=self.anthropic_client,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    use_mem0_cloud=self._use_mem0_cloud,
                    files_cabinet=self.files_cabinet,
                    user_message=user_message,
                )
                if tblock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{tblock}"
            except Exception:
                pass

        if file_cmd:
            try:
                import angel_file_reading as fr

                fblock = fr.format_file_read_for_prompt(
                    file_cmd,
                    file_payload,
                    anthropic_client=self.anthropic_client,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    use_mem0_cloud=self._use_mem0_cloud,
                    files_cabinet=self.files_cabinet,
                    user_message=user_message,
                )
                if fblock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{fblock}"
            except Exception:
                pass

        if forensic_cmd:
            try:
                import angel_forensic as af

                fiblock = af.format_forensic_chat_block(
                    forensic_cmd,
                    anthropic_client=self.anthropic_client,
                    files_cabinet=self.files_cabinet,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    use_mem0_cloud=self._use_mem0_cloud,
                    user_message=user_message,
                )
                if fiblock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{fiblock}"
            except Exception:
                pass

        if surv_cmd:
            try:
                import angel_surveillance as asurv

                sblock = asurv.format_surveillance_chat_block(
                    surv_cmd,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    use_mem0_cloud=self._use_mem0_cloud,
                )
                if sblock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{sblock}"
            except Exception:
                pass

        if map_cmd:
            try:
                import angel_environmental_map as aemap

                mblock = aemap.format_map_chat_block(
                    map_cmd,
                    map_payload,
                    anthropic_client=self.anthropic_client,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    use_mem0_cloud=self._use_mem0_cloud,
                    files_cabinet=self.files_cabinet,
                )
                if mblock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{mblock}"
            except Exception:
                pass

        if comms_cmd:
            try:
                import angel_communication_patterns as acomm

                cblock = acomm.format_comms_chat_block(
                    comms_cmd,
                    comms_payload,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    use_mem0_cloud=self._use_mem0_cloud,
                )
                if cblock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{cblock}"
            except Exception:
                pass

        if bio_cmd:
            try:
                import angel_biological_intelligence as abio

                bblock = abio.format_bio_chat_block(
                    bio_cmd,
                    bio_payload,
                    anthropic_client=self.anthropic_client,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    files_cabinet=self.files_cabinet,
                    use_mem0_cloud=self._use_mem0_cloud,
                    user_message=user_message,
                )
                if bblock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{bblock}"
            except Exception:
                pass

        if hist_cmd:
            try:
                import angel_historical_archives as ahist

                hblock = ahist.format_hist_chat_block(
                    hist_cmd,
                    hist_payload,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    use_mem0_cloud=self._use_mem0_cloud,
                )
                if hblock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{hblock}"
            except Exception:
                pass

        if ta_cmd:
            try:
                import angel_threat_actors as ata

                tablock = ata.format_threat_actor_chat_block(
                    ta_cmd,
                    ta_payload,
                    anthropic_client=self.anthropic_client,
                    memory_client=self.memory_client,
                    user_id=self.user_id,
                    use_mem0_cloud=self._use_mem0_cloud,
                    files_cabinet=self.files_cabinet,
                )
                if tablock.strip():
                    augmented_user_message = f"{augmented_user_message}\n\n{tablock}"
            except Exception:
                pass

        try:
            import angel_environmental_map as aemap

            px = aemap.format_proximity_alert_for_prompt(
                location,
                self.memory_client,
                self.user_id,
                self._use_mem0_cloud,
            )
            if px.strip():
                augmented_user_message = f"{augmented_user_message}\n\n{px}"
        except Exception:
            pass

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
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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

        try:
            import angel_self_modification as _angel_self_mod3

            _angel_self_mod3.record_turn_observation_background(
                self.memory_client,
                self.user_id,
                self._use_mem0_cloud,
                user_message,
                reply,
                files_cabinet=self.files_cabinet,
            )
        except Exception:
            pass

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
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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