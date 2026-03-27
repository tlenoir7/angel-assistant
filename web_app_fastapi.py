import base64
import json
import logging
import re
import os
import shutil
import threading
import time
from io import BytesIO
from pathlib import Path
import traceback
from datetime import UTC, datetime, timedelta, timezone
from contextlib import asynccontextmanager

import asyncio
import anthropic as _anthropic
import concurrent.futures as _cf
import functools
import requests
from flask import Flask, Response, jsonify, render_template_string, request, send_file
from apscheduler.schedulers.background import BackgroundScheduler
import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware
from urllib.parse import parse_qs

import angel_predictions
import angel_proactive
import angel_translation
import angel_file_reading
import angel_threat_actors
import angel_forensic
import angel_vision_forensic
import angel_surveillance
import angel_environmental_map
import angel_communication_patterns
import angel_biological_intelligence
import angel_historical_archives
import angel_chemistry
import angel_medical
import angel_research
import angel_physics
import angel_cad
import angel_ironman
import angel_capability_graph

# AngelCore includes Stage 2: strategy, patterns, deep research, people profiles
from angel import (
    AngelCore,
    COMPUTER_CONTROL_AVAILABLE,
    tts_gpt4o,
    transcribe_with_whisper,
    generate_morning_briefing,
    send_briefing_email,
    create_anthropic_client,
    build_memory_summary_with_sections,
    build_system_prompt,
    run_memory_reflection,
    get_latest_reflection_text,
    get_recent_briefing_history_for_prompt,
    summarize_briefing_for_history,
    add_structured_memory,
    CATEGORY_BRIEFING_HISTORY,
    strip_markdown,
    execute_python_sandbox,
    THREAT_WATCH_DEFAULT_CATEGORIES,
    THREAT_INTEL_FOLDER,
    load_merged_threat_watch_categories,
    add_threat_category,
    run_threat_detection,
    format_threat_intelligence_for_briefing,
    OSINT_DOSSIERS_FOLDER,
    run_osint_background,
    NETWORK_INTEL_FOLDER,
    add_network_node,
    add_network_edge,
    get_node_connections,
    get_network_cluster,
    find_path_between,
    get_network_summary,
    map_osint_to_network,
    schedule_mission_network_seed_background,
    reset_mission_network_and_reseed,
    network_load_graph,
    is_trusted_operator,
)

try:
    from angel_realtime import AngelRealtimeSession
except Exception:
    AngelRealtimeSession = None  # type: ignore[misc, assignment]

# Module-level storage for morning briefing and check-in
app: Flask | None = None  # set by build_asgi_app() to the Flask HTTP app

morning_briefing = None
briefing_generated_at = None
check_in_message = None
check_in_generated_at = None
last_activity_at = time.time()
angel = None

# Async Socket.IO server created in build_asgi_app (python-socketio AsyncServer)
sio_server: socketio.AsyncServer | None = None
# sid -> {"device": str, "turns": list[tuple[str, str]]} for multi-turn Claude context
SOCKET_SESSIONS: dict[str, dict] = {}
# sid -> {"session": AngelRealtimeSession} for OpenAI Realtime proxy (namespace /realtime)
REALTIME_PROXY_BY_SID: dict[str, dict] = {}

# Async mission network reset (Mem0 purge can exceed HTTP timeouts on Railway)
_network_reset_lock = threading.Lock()
_network_reset_status: dict = {
    "in_progress": False,
    "last_success": None,  # bool | None — None if no reset has finished yet
    "last_nodes_after": None,
    "last_edges_after": None,
    "last_reset_at": None,  # ISO 8601 UTC when last run finished
    "last_error": None,
}

# Isolated thread pool for /api/files/read so worker threads can time out without blocking the WSGI worker forever.
_executor = _cf.ThreadPoolExecutor(max_workers=4)


def _do_file_read(file_b64, file_name, file_type, context):
    # Network diagnostic
    import requests as _req

    try:
        r = _req.get("https://api.anthropic.com", timeout=5)
        print(f"[files-thread] anthropic.com reachable: {r.status_code}", flush=True)
    except Exception as e:
        print(f"[files-thread] anthropic.com UNREACHABLE: {e}", flush=True)

    # Create fresh client for thread safety
    fresh_client = _anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "")
    )
    return angel_file_reading.read_and_analyze_file(
        file_b64,
        file_name,
        file_type,
        context,
        fresh_client,
        memory_client=angel.memory_client,
        user_id=angel.user_id,
        files_cabinet=angel.files_cabinet,
        use_mem0_cloud=angel._use_mem0_cloud,
        model="claude-haiku-4-5",
    )


_log = logging.getLogger(__name__)


def _request_json_keys_for_log() -> object:
    """Uses get_json(silent=True) — avoids strict request.json parsing side effects."""
    try:
        data = request.get_json(silent=True)
        return list(data.keys()) if isinstance(data, dict) else "no json"
    except Exception:
        return "no json"


def _log_startup_seed_summary(name: str, r: object) -> None:
    """One short stdout line per seed job — avoids huge dicts (Railway 500 logs/sec)."""
    if not isinstance(r, dict):
        print(f"[web_app] {name}: {r!r}", flush=True)
        return
    if name == "Initial predictions seed":
        saved = r.get("saved")
        n = len(saved) if isinstance(saved, list) else None
        ids: list[str] = []
        if isinstance(saved, list):
            for item in saved[:12]:
                if isinstance(item, dict):
                    pid = item.get("prediction_id")
                    if pid:
                        ids.append(str(pid)[:10])
        print(
            f"[web_app] {name}: ok={r.get('ok')} count={r.get('count')} "
            f"seeded={r.get('seeded')} skipped_dup={r.get('skipped_duplicates')} "
            f"saved_n={n} prediction_id_prefixes={ids}",
            flush=True,
        )
        return
    parts: list[str] = []
    for k in (
        "ok",
        "seeded",
        "added",
        "total",
        "total_cases",
        "count",
        "ensured",
        "skipped_duplicates",
        "force_reseed",
        "reason",
        "error",
    ):
        if k in r:
            parts.append(f"{k}={r[k]!r}")
    print(f"[web_app] {name}: " + " ".join(parts), flush=True)


_RESET_WATCHDOG_SEC = 120.0


def _network_reset_timeout_handler() -> None:
    """If reset_mission_network_and_reseed hangs past the watchdog, unblock API clients."""
    print(
        "[network_reset_worker] TIMER %.0fs elapsed — forcing in_progress=False (reset may still be running)"
        % _RESET_WATCHDOG_SEC,
        flush=True,
    )
    try:
        with _network_reset_lock:
            if _network_reset_status.get("in_progress"):
                _network_reset_status["in_progress"] = False
                _network_reset_status["last_success"] = False
                _network_reset_status["last_reset_at"] = (
                    datetime.now(UTC).isoformat().replace("+00:00", "Z")
                )
                prev_err = _network_reset_status.get("last_error")
                _network_reset_status["last_error"] = prev_err or (
                    "reset timed out after %.0fs (watchdog)" % _RESET_WATCHDOG_SEC
                )
    except Exception as ex:
        print(f"[network_reset_worker] timeout handler error: {ex!r}", flush=True)
    _log.debug(
        "network reset watchdog: %.0fs timeout, in_progress cleared",
        _RESET_WATCHDOG_SEC,
    )


def _network_reset_worker(
    memory_client,
    user_id: str,
    files_cabinet,
    use_mem0_cloud: bool,
) -> None:
    """Runs in a daemon thread; never raises."""
    _log.debug("[network_reset_worker] thread START")
    summary: dict | None = None
    err: str | None = None
    reset_timer = threading.Timer(_RESET_WATCHDOG_SEC, _network_reset_timeout_handler)
    reset_timer.daemon = True
    reset_timer.start()
    try:
        summary = reset_mission_network_and_reseed(
            memory_client,
            user_id,
            files_cabinet,
            use_mem0_cloud,
        )
        print(
            "[network_reset_worker] reset_mission_network_and_reseed RETURNED "
            f"ok={summary.get('ok') if isinstance(summary, dict) else None!r} "
            f"nodes_after={summary.get('nodes_after') if isinstance(summary, dict) else None} "
            f"edges_after={summary.get('edges_after') if isinstance(summary, dict) else None}",
            flush=True,
        )
    except Exception as ex:
        err = str(ex)
        print(f"[network_reset_worker] reset_mission_network_and_reseed RAISED: {err!r}", flush=True)
        traceback.print_exc()
    finally:
        try:
            reset_timer.cancel()
        except Exception:
            pass
        _log.debug("[network_reset_worker] finally block ENTER")
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        _log.debug(
            "network reset worker: thread completing (summary=%s err=%s)",
            "ok" if isinstance(summary, dict) and summary.get("ok") else None,
            err,
        )
        try:
            with _network_reset_lock:
                _network_reset_status["in_progress"] = False
                _log.debug(
                    "[network_reset_worker] set in_progress=False last_reset_at=%s summary_type=%s",
                    now,
                    type(summary).__name__,
                )
                _network_reset_status["last_reset_at"] = now
                if isinstance(summary, dict):
                    _network_reset_status["last_success"] = bool(summary.get("ok"))
                    _network_reset_status["last_nodes_after"] = summary.get("nodes_after")
                    _network_reset_status["last_edges_after"] = summary.get("edges_after")
                    _network_reset_status["last_error"] = summary.get("error")
                else:
                    _network_reset_status["last_success"] = False
                    _network_reset_status["last_nodes_after"] = None
                    _network_reset_status["last_edges_after"] = None
                    _network_reset_status["last_error"] = err or "reset thread failed"
        except Exception as ex:
            print(f"[network_reset_worker] EXCEPTION updating status dict: {ex!r}", flush=True)
            _log.exception("network reset worker: failed to update status dict: %s", ex)
            try:
                with _network_reset_lock:
                    _network_reset_status["in_progress"] = False
                    print("[network_reset_worker] set in_progress=False (fallback after status dict error)", flush=True)
                    _network_reset_status["last_error"] = (
                        _network_reset_status.get("last_error") or str(ex)
                    )
            except Exception:
                print("[network_reset_worker] EXCEPTION in fallback in_progress=False", flush=True)
                _log.exception("network reset worker: could not force in_progress=False")
        else:
            _log.debug("network reset worker: status updated; in_progress=False at %s", now)


# Expo push: in-memory + push_tokens.json (same directory as this module)
PUSH_TOKENS_PATH = Path(__file__).resolve().parent / "push_tokens.json"
EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
expo_push_tokens: list[str] = []

# Item 12 threat detection: scheduler handle for delayed HIGH pushes; buffer of headlines
angel_scheduler: BackgroundScheduler | None = None
HIGH_THREAT_PUSH_BUFFER: list[str] = []


_VALID_CLIENT_DEVICES = frozenset({"ios", "desktop", "mobile_web"})


def _parse_location_dict(loc) -> dict | None:
    """Normalize location object: latitude/longitude (or lat/lng), optional place_name."""
    if not isinstance(loc, dict):
        return None
    lat = loc.get("latitude", loc.get("lat"))
    lng = loc.get("longitude", loc.get("lng"))
    if lat is None or lng is None:
        return None
    try:
        lat_f = float(lat)
        lng_f = float(lng)
    except (TypeError, ValueError):
        return None
    out: dict = {"latitude": lat_f, "longitude": lng_f}
    place = (loc.get("place_name") or loc.get("place") or loc.get("name") or "").strip()
    if place:
        out["place_name"] = place
    return out


def normalize_location(loc) -> dict | None:
    """
    Public name for the same rules as HTTP JSON ``location`` (and Socket.IO payload ``location``).
    Accepts a dict with latitude/longitude or lat/lng, optional place_name / place / name.
    """
    return _parse_location_dict(loc)


def _parse_location_from_json_body(data: dict | None) -> dict | None:
    """Read optional top-level ``location`` key from a JSON body."""
    if not isinstance(data, dict):
        return None
    return normalize_location(data.get("location"))


def _vision_device_from_body(device_raw: str) -> str:
    d = (device_raw or "").strip().lower()
    if d in _VALID_CLIENT_DEVICES:
        return d
    return "mobile_web"


def _normalize_jpeg_base64(image_field: str) -> str:
    """Strip data-URL wrapper and whitespace from a base64 JPEG payload."""
    s = (image_field or "").strip()
    if not s:
        return ""
    if s.startswith("data:") and "base64," in s:
        s = s.split("base64,", 1)[1].strip()
    return "".join(s.split())


def _request_device() -> str:
    """
    Identify client: ios (native app), desktop (Windows GUI), mobile_web (browser).
    Prefer X-Angel-Device or Device header, then JSON body or form field 'device'.
    Defaults to mobile_web when unset (hosted web UI).
    """
    for key in ("X-Angel-Device", "Device"):
        v = (request.headers.get(key) or "").strip().lower()
        if v in _VALID_CLIENT_DEVICES:
            return v
    data = request.get_json(silent=True) or {}
    v = (data.get("device") or "").strip().lower()
    if v in _VALID_CLIENT_DEVICES:
        return v
    v = (request.form.get("device") or "").strip().lower()
    if v in _VALID_CLIENT_DEVICES:
        return v
    return "mobile_web"


def _sanitize_text(s: str) -> str:
    """
    Strip/replace any invalid Unicode (including surrogate characters)
    so Flask/Werkzeug can safely encode responses.
    """
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def _load_expo_push_tokens_from_disk() -> None:
    """Populate module-level expo_push_tokens from push_tokens.json if present."""
    global expo_push_tokens
    try:
        if not PUSH_TOKENS_PATH.is_file():
            expo_push_tokens = []
            return
        raw = PUSH_TOKENS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        tokens: list[str] = []
        if isinstance(data, list):
            tokens = [str(t).strip() for t in data if str(t).strip()]
        elif isinstance(data, dict):
            arr = data.get("tokens") or data.get("expo_push_tokens")
            if isinstance(arr, list):
                tokens = [str(t).strip() for t in arr if str(t).strip()]
        # de-dupe, preserve order
        seen = set()
        out: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                out.append(t)
        expo_push_tokens = out
        print(f"[web_app] Loaded {len(expo_push_tokens)} Expo push token(s) from disk.", flush=True)
    except Exception as e:
        print(f"[web_app] Could not load push_tokens.json: {e}", flush=True)
        expo_push_tokens = []


def _save_expo_push_tokens_to_disk() -> None:
    """Persist expo_push_tokens to push_tokens.json."""
    try:
        payload = {"tokens": list(expo_push_tokens)}
        PUSH_TOKENS_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[web_app] Could not save push_tokens.json: {e}", flush=True)


def register_expo_push_token(token: str) -> bool:
    """Add token to module list and disk; returns True if token was new."""
    global expo_push_tokens
    t = (token or "").strip()
    if not t:
        return False
    if t not in expo_push_tokens:
        expo_push_tokens.append(t)
        _save_expo_push_tokens_to_disk()
        return True
    return False


def send_expo_push_notifications(
    title: str,
    body: str,
    *,
    tokens: list[str] | None = None,
) -> dict:
    """
    POST to Expo push API. ``tokens`` defaults to module-level expo_push_tokens.
    Returns a small result dict for logging / API responses.
    """
    use = tokens if tokens is not None else expo_push_tokens
    use = [str(t).strip() for t in use if str(t).strip()]
    if not use:
        return {"ok": False, "error": "no_tokens", "status": None, "expo_response": None}

    payload = [
        {
            "to": t,
            "title": (title or "")[:200],
            "body": (body or "")[:400],
            "sound": "default",
        }
        for t in use
    ]
    try:
        resp = requests.post(
            EXPO_PUSH_URL,
            json=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        text_preview = (resp.text or "")[:2000]
        try:
            expo_json = resp.json()
        except Exception:
            expo_json = None
        ok = resp.status_code == 200
        if not ok:
            print(
                f"[web_app] Expo push HTTP {resp.status_code}: {text_preview}",
                flush=True,
            )
        return {
            "ok": ok,
            "status": resp.status_code,
            "expo_response": expo_json,
            "recipients": len(use),
        }
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e), "status": None, "expo_response": None}


def _flush_high_threat_push() -> None:
    """Send one batched push for HIGH threats (scheduled ~1 hour after scan)."""
    global HIGH_THREAT_PUSH_BUFFER
    if not HIGH_THREAT_PUSH_BUFFER:
        return
    headlines = HIGH_THREAT_PUSH_BUFFER[:8]
    HIGH_THREAT_PUSH_BUFFER = []
    body = "; ".join(h for h in headlines if h)
    if not body:
        return
    send_expo_push_notifications("Angel: elevated threat intel", body[:200])


def _run_surveillance_job() -> None:
    """Every 8 hours: open-source surveillance OSINT scan (legal public sources)."""
    global angel
    if angel is None:
        return
    try:
        r = angel_surveillance.run_osint_surveillance(
            angel.anthropic_client,
            angel.memory_client,
            angel.user_id,
            angel.files_cabinet,
            angel._use_mem0_cloud,
        )
        print(
            f"[web_app] Surveillance: categories={r.get('categories_scanned')} "
            f"signals={len(r.get('signals') or [])} filed_si={len(r.get('filed_si') or [])} "
            f"correlated={bool(r.get('correlated'))}",
            flush=True,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[web_app] Surveillance job failed: {e}", flush=True)


def _run_threat_detection_job() -> None:
    """Scheduled Tavily + Claude threat scan; CRITICAL push now, HIGH push batched within ~1 hour."""
    global angel, HIGH_THREAT_PUSH_BUFFER, angel_scheduler
    if angel is None:
        return
    try:
        client = create_anthropic_client()
        memories = angel._fetch_combined_memories()
        memory_summary = build_memory_summary_with_sections(
            memories, None, omit_reflection_section=True
        )
        result = run_threat_detection(
            client,
            angel.memory_client,
            angel.user_id,
            angel.files_cabinet,
            use_mem0_cloud=angel._use_mem0_cloud,
            memory_summary=memory_summary,
        )
        for t in result.get("threats") or []:
            if not isinstance(t, dict):
                continue
            if t.get("skipped"):
                continue
            lvl = (t.get("threat_level") or "").strip().upper()
            headline = (t.get("headline") or "").strip() or "Threat intel"
            if not t.get("filed_as"):
                continue
            if lvl == "CRITICAL":
                send_expo_push_notifications("Angel: CRITICAL threat intel", headline[:200])
            elif lvl == "HIGH":
                HIGH_THREAT_PUSH_BUFFER.append(headline)
        if HIGH_THREAT_PUSH_BUFFER and angel_scheduler is not None:
            try:
                angel_scheduler.add_job(
                    _flush_high_threat_push,
                    "date",
                    run_date=datetime.now(timezone.utc) + timedelta(hours=1),
                    id="angel_high_threat_push_flush",
                    replace_existing=True,
                )
            except Exception as e:
                print(f"[web_app] Could not schedule HIGH threat push: {e}", flush=True)
        print(
            f"[web_app] Threat detection: categories={result.get('categories_scanned')}, "
            f"items={len(result.get('threats') or [])}, errors={len(result.get('errors') or [])}",
            flush=True,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[web_app] Threat detection job failed: {e}", flush=True)


def _run_morning_briefing_job():
    global morning_briefing, briefing_generated_at
    try:
        user_id = os.getenv("ANGEL_USER_ID", "railway-user")
        client = create_anthropic_client()
        memories = angel._fetch_combined_memories()
        memory_summary = build_memory_summary_with_sections(
            memories, None, omit_reflection_section=True
        )
        latest_reflection = get_latest_reflection_text(memories)
        recent_briefing_history = get_recent_briefing_history_for_prompt(memories, days=7)
        tz = os.getenv("TIMEZONE", "America/Los_Angeles")
        threat_appendix = format_threat_intelligence_for_briefing(angel.files_cabinet)
        pro_run = angel_proactive.run_proactive_intelligence(
            client,
            angel.memory_client,
            user_id,
            angel.files_cabinet,
            angel._use_mem0_cloud,
        )
        proactive_appendix = angel_proactive.format_proactive_intelligence_for_briefing(
            angel.memory_client,
            user_id,
            angel._use_mem0_cloud,
            last_run_summary=pro_run,
        )
        surv_run = angel_surveillance.run_osint_surveillance(
            client,
            angel.memory_client,
            user_id,
            angel.files_cabinet,
            angel._use_mem0_cloud,
        )
        surveillance_appendix = angel_surveillance.format_surveillance_for_briefing(
            angel.memory_client,
            user_id,
            angel._use_mem0_cloud,
            last_run=surv_run,
        )
        morning_briefing = generate_morning_briefing(
            client,
            user_id,
            memory_summary,
            timezone=tz,
            latest_reflection=latest_reflection,
            recent_briefing_history=recent_briefing_history or None,
            threat_appendix=threat_appendix or None,
            proactive_intelligence_appendix=proactive_appendix or None,
            surveillance_appendix=surveillance_appendix or None,
        )
        try:
            angel_environmental_map.ensure_seed_locations(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            extra_blocks: list[str] = []
            lat_s = (os.getenv("BRIEFING_LOCATION_LAT") or "").strip()
            lon_s = (os.getenv("BRIEFING_LOCATION_LON") or "").strip()
            if lat_s and lon_s:
                tn = angel_environmental_map.format_briefing_travel_note_if_env(
                    float(lat_s),
                    float(lon_s),
                    angel.memory_client,
                    user_id,
                    angel._use_mem0_cloud,
                )
                if tn:
                    extra_blocks.append(tn)
            summ = angel_environmental_map.get_location_summary(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            if summ and summ.get("total", 0):
                compact = {
                    "total": summ["total"],
                    "by_significance": summ.get("by_significance"),
                    "sample_locations": [
                        x.get("name") for x in (summ.get("locations") or [])[:12]
                    ],
                }
                extra_blocks.append(
                    "ENVIRONMENTAL MAP (summary)\n"
                    + json.dumps(compact, ensure_ascii=False, indent=2)[:3000]
                )
            if (
                extra_blocks
                and morning_briefing
                and "Briefing unavailable" not in morning_briefing
            ):
                morning_briefing = (morning_briefing or "").rstrip() + "\n\n" + "\n\n".join(
                    extra_blocks
                )
        except Exception as e:
            print(f"[web_app] Environmental map briefing appendix: {e}", flush=True)

        try:
            comm_brief = angel_communication_patterns.format_communication_patterns_for_briefing(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            if (
                comm_brief
                and morning_briefing
                and "Briefing unavailable" not in morning_briefing
            ):
                morning_briefing = (morning_briefing or "").rstrip() + "\n\n" + comm_brief
        except Exception as e:
            print(f"[web_app] Communication patterns briefing appendix: {e}", flush=True)

        try:
            hist_brief = angel_historical_archives.format_on_this_day_and_anniversaries(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            if (
                hist_brief
                and morning_briefing
                and "Briefing unavailable" not in morning_briefing
            ):
                morning_briefing = (morning_briefing or "").rstrip() + "\n\n" + hist_brief
        except Exception as e:
            print(f"[web_app] Historical archives briefing appendix: {e}", flush=True)

        briefing_generated_at = time.time()
        send_briefing_email(morning_briefing)
        if morning_briefing and "Briefing unavailable" not in morning_briefing:
            preview = _sanitize_text((morning_briefing or "")[:100])
            send_expo_push_notifications("Angel Morning Briefing", preview)
            topics = summarize_briefing_for_history(client, morning_briefing)
            if topics:
                add_structured_memory(
                    angel.memory_client,
                    user_id,
                    topics,
                    CATEGORY_BRIEFING_HISTORY,
                    person_name=None,
                    use_mem0_cloud=angel._use_mem0_cloud,
                )
    except Exception as e:
        traceback.print_exc()
        morning_briefing = f"Briefing unavailable: {e}"
        briefing_generated_at = time.time()


def _run_weekly_reflection_job():
    """Sunday 6 AM (TIMEZONE): Angel reviews stored memories and saves a reflection."""
    try:
        user_id = os.getenv("ANGEL_USER_ID", "railway-user")
        client = create_anthropic_client()
        text = run_memory_reflection(
            angel.memory_client,
            user_id,
            client,
            use_mem0_cloud=angel._use_mem0_cloud,
        )
        print(f"[web_app] Weekly memory reflection completed: {len(text)} chars", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"[web_app] Weekly memory reflection failed: {e}", flush=True)


def _run_weekly_predictions_generate_job():
    """Sunday 7 AM Eastern: generate new mission forecasts (Item 15)."""
    global angel
    if angel is None:
        return
    try:
        r = angel_predictions.generate_predictions(
            angel.anthropic_client,
            angel.memory_client,
            angel.user_id,
            angel.files_cabinet,
            angel._use_mem0_cloud,
        )
        print(
            f"[web_app] Weekly predictions generate: count={r.get('count')} skipped_dup={r.get('skipped_duplicates')}",
            flush=True,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[web_app] Weekly predictions generate failed: {e}", flush=True)


def _run_weekly_predictions_check_job():
    """Wednesday 7 AM Eastern: Tavily + Haiku reality check on active predictions."""
    global angel
    if angel is None:
        return
    try:
        r = angel_predictions.check_predictions_against_reality(
            angel.anthropic_client,
            angel.memory_client,
            angel.user_id,
            angel.files_cabinet,
            angel._use_mem0_cloud,
        )
        print(
            f"[web_app] Predictions reality check: checked={r.get('checked')} auto_resolved={r.get('auto_resolved')}",
            flush=True,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[web_app] Predictions reality check failed: {e}", flush=True)


def _schedule_predictions_initial_seed() -> None:
    """First deploy: after a short delay, seed predictions if none exist."""

    def _job() -> None:
        try:
            time.sleep(20)
        except Exception:
            return
        global angel
        if angel is None:
            return
        try:
            r = angel_predictions.seed_initial_predictions_if_needed(
                angel.anthropic_client,
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            _log_startup_seed_summary("Initial predictions seed", r)
        except Exception:
            traceback.print_exc()

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception:
        pass


def _schedule_proactive_watch_seed() -> None:
    """Seed default proactive watch list if empty."""

    def _job() -> None:
        try:
            time.sleep(26)
        except Exception:
            return
        global angel
        if angel is None:
            return
        try:
            r = angel_proactive.seed_proactive_watch_if_empty(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            _log_startup_seed_summary("Proactive watch seed", r)
        except Exception:
            traceback.print_exc()

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception:
        pass


def _schedule_foreign_watch_seed() -> None:
    """Item 18: ensure foreign-intelligence watch items exist (deduped by label)."""

    def _job() -> None:
        try:
            time.sleep(32)
        except Exception:
            return
        global angel
        if angel is None:
            return
        try:
            r = angel_translation.ensure_foreign_watch_items(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            _log_startup_seed_summary("Foreign watch seed", r)
        except Exception:
            traceback.print_exc()

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception:
        pass


def _schedule_threat_actor_seed() -> None:
    """Batcomputer: seed default Threat Actor (opposition) profiles if DB empty."""

    def _job() -> None:
        try:
            time.sleep(36)
        except Exception:
            return
        global angel
        if angel is None:
            return
        try:
            r = angel_threat_actors.ensure_threat_actor_seeds(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
                force_reseed=False,
            )
            print(f"[web_app] Threat actor seed: {r}", flush=True)
        except Exception:
            traceback.print_exc()

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception:
        pass


def _schedule_environmental_map_seed() -> None:
    """Batcomputer: seed mission geography (UAP hotspots, installations, etc.) if missing."""

    def _job() -> None:
        try:
            time.sleep(40)
        except Exception:
            return
        global angel
        if angel is None:
            return
        try:
            r = angel_environmental_map.ensure_seed_locations(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            _log_startup_seed_summary("Environmental map seed", r)
        except Exception:
            traceback.print_exc()

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception:
        pass


def _schedule_biological_intelligence_seed() -> None:
    """Batcomputer: seed UAP medical reference cases + black-eyed profile."""

    def _job() -> None:
        try:
            time.sleep(44)
        except Exception:
            return
        global angel
        if angel is None:
            return
        try:
            r = angel_biological_intelligence.ensure_seed_bio_cases(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            _log_startup_seed_summary("Biological intelligence seed", r)
        except Exception:
            traceback.print_exc()

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception:
        pass


def _schedule_historical_archives_seed() -> None:
    """Batcomputer: seed Historical Intelligence Archives (timeline records)."""

    def _job() -> None:
        try:
            time.sleep(48)
        except Exception:
            return
        global angel
        if angel is None:
            return
        try:
            r = angel_historical_archives.ensure_seed_historical_records(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            _log_startup_seed_summary("Historical archives seed", r)
        except Exception:
            traceback.print_exc()

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception:
        pass


def _schedule_self_modification_seed() -> None:
    """Stage 6: seed 2–3 insightful self-mod proposals from existing memory (first run)."""

    def _job() -> None:
        try:
            time.sleep(52)
        except Exception:
            return
        global angel
        if angel is None:
            return
        try:
            import angel_self_modification as asm

            r = asm.seed_initial_self_modification(
                angel.anthropic_client,
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            _log_startup_seed_summary("Self modification seed", r)
        except Exception:
            traceback.print_exc()

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception:
        pass


def _run_weekly_self_modification_job() -> None:
    """Sunday 6:30 AM (TIMEZONE): analyze self_observation entries and propose modifications."""
    global angel
    if angel is None:
        return
    try:
        import angel_self_modification as asm

        r = asm.run_self_modification_analysis(
            angel.anthropic_client,
            angel.memory_client,
            angel.user_id,
            angel.files_cabinet,
            angel._use_mem0_cloud,
        )
        print(
            f"[web_app] Weekly self-modification analysis: ok={r.get('ok')} count={r.get('count')}",
            flush=True,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[web_app] Weekly self-modification analysis failed: {e}", flush=True)


def _run_communication_patterns_job() -> None:
    """Every 48h: open-source communication cadence / coordination analysis for watched figures."""

    global angel
    if angel is None:
        return
    try:
        user_id = os.getenv("ANGEL_USER_ID", "railway-user")
        r = angel_communication_patterns.run_scheduled_analysis(
            angel.anthropic_client,
            angel.memory_client,
            user_id,
            angel.files_cabinet,
            angel._use_mem0_cloud,
        )
        print(
            f"[web_app] Communication patterns: ok={r.get('ok')} "
            f"entities={len(r.get('entities_analyzed') or [])} "
            f"coordination={len(r.get('coordination_signals') or [])}",
            flush=True,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[web_app] Communication patterns job failed: {e}", flush=True)


def _run_proactive_intelligence_job():
    """Every 4 hours: background Tavily monitoring for watch list (Item 16)."""
    global angel
    if angel is None:
        return
    try:
        r = angel_proactive.run_proactive_intelligence(
            angel.anthropic_client,
            angel.memory_client,
            angel.user_id,
            angel.files_cabinet,
            angel._use_mem0_cloud,
        )
        print(
            f"[web_app] Proactive intelligence: checked={r.get('checked')} significant={r.get('significant')}",
            flush=True,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[web_app] Proactive intelligence job failed: {e}", flush=True)


def _run_check_in_job():
    """If no activity for 4+ hours, generate a short check-in message (once per idle period)."""
    global check_in_message, check_in_generated_at, last_activity_at
    if time.time() - last_activity_at < 4 * 3600:
        return
    if check_in_message and check_in_generated_at and check_in_generated_at > last_activity_at:
        return
    try:
        from angel import get_current_datetime_str
        client = create_anthropic_client()
        dt = get_current_datetime_str()
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=256,
            temperature=0.5,
            system="You are Angel, Tyler's personal AI. Write one short, warm check-in message (1-2 sentences) as if Tyler hasn't been in touch for a few hours. No questions required. Be concise.",
            messages=[{"role": "user", "content": f"Current context: {dt}. Generate a brief check-in."}],
        )
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        check_in_message = (text or "Just checking in.").strip()
        check_in_generated_at = time.time()
    except Exception as e:
        traceback.print_exc()
        check_in_message = "Thinking of you. Reach out when you're ready."
        check_in_generated_at = time.time()


def create_app() -> Flask:
    global angel, angel_scheduler
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "angel-dev-secret-set-in-production")

    user_id = os.getenv("ANGEL_USER_ID", "railway-user")
    angel = AngelCore(user_id=user_id, use_voice=True)
    # Warm up memories once on startup
    angel.load_initial_memory_summary()
    schedule_mission_network_seed_background(
        angel.memory_client,
        angel.user_id,
        angel.files_cabinet,
        angel._use_mem0_cloud,
    )
    _schedule_predictions_initial_seed()
    _schedule_proactive_watch_seed()
    _schedule_foreign_watch_seed()
    _schedule_threat_actor_seed()
    _schedule_environmental_map_seed()
    _schedule_biological_intelligence_seed()
    _schedule_historical_archives_seed()
    _schedule_self_modification_seed()
    _load_expo_push_tokens_from_disk()

    # Log briefing email env at startup for debugging
    tyler_email = (os.getenv("TYLER_EMAIL") or "").strip()
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD") or ""
    print(
        f"[web_app] TYLER_EMAIL present={bool(tyler_email)}, "
        f"GMAIL_APP_PASSWORD present={bool(gmail_pass)}",
        flush=True,
    )

    try:
        env_names = sorted(os.environ.keys())
        print(f"[web_app] Environment variable name count: {len(env_names)}", flush=True)
    except Exception as e:
        print(f"[web_app] Failed to count environment variable names: {e}", flush=True)

    # Morning briefing: run at BRIEFING_TIME (default 08:00)
    briefing_time = os.getenv("BRIEFING_TIME", "08:00").strip()
    try:
        hour, minute = map(int, briefing_time.split(":")[:2])
    except Exception:
        hour, minute = 8, 0
    sched_tz = os.getenv("TIMEZONE", "America/Los_Angeles").strip()
    scheduler = BackgroundScheduler()
    angel_scheduler = scheduler
    if os.environ.get("DISABLE_BRIEFING", "").lower() == "true":
        pass  # briefing disabled
    else:
        if hour > 23 or minute > 59:
            _log.warning(
                "Invalid BRIEFING_TIME=%r (hour=%s minute=%s); skipping morning briefing cron job",
                briefing_time,
                hour,
                minute,
            )
        else:
            scheduler.add_job(_run_morning_briefing_job, "cron", hour=hour, minute=minute)
    scheduler.add_job(_run_proactive_intelligence_job, "interval", hours=4)
    scheduler.add_job(
        _run_weekly_reflection_job,
        "cron",
        day_of_week="sun",
        hour=6,
        minute=0,
        timezone=sched_tz,
    )
    scheduler.add_job(
        _run_weekly_self_modification_job,
        "cron",
        day_of_week="sun",
        hour=6,
        minute=30,
        timezone=sched_tz,
    )
    scheduler.add_job(
        _run_weekly_predictions_generate_job,
        "cron",
        day_of_week="sun",
        hour=7,
        minute=0,
        timezone="America/New_York",
    )
    scheduler.add_job(
        _run_weekly_predictions_check_job,
        "cron",
        day_of_week="wed",
        hour=7,
        minute=0,
        timezone="America/New_York",
    )
    scheduler.add_job(_run_check_in_job, "interval", minutes=15)
    scheduler.add_job(_run_threat_detection_job, "interval", hours=6)
    scheduler.add_job(_run_surveillance_job, "interval", hours=8)
    scheduler.add_job(_run_communication_patterns_job, "interval", hours=48)
    scheduler.start()

    def _session_turns_for(sid: str) -> list[tuple[str, str]]:
        sess = SOCKET_SESSIONS.get(sid)
        if not sess:
            return []
        return list(sess.get("turns") or [])

    def _append_turn(sid: str, user_text: str, assistant_text: str) -> None:
        sess = SOCKET_SESSIONS.get(sid)
        if not sess:
            return
        turns: list = sess["turns"]
        turns.append((user_text, assistant_text))
        while len(turns) > 25:
            turns.pop(0)

    # --- Socket.IO: registered on AsyncServer in build_asgi_app() ---

    def _build_openai_realtime_system_prompt() -> str:
        memories = angel._fetch_combined_memories()
        memory_summary = build_memory_summary_with_sections(memories, None)
        return build_system_prompt(
            memory_summary,
            voice_mode=True,
            strategy_hint=False,
            pattern_hint=False,
            profile_hint=False,
            computer_control_enabled=False,
            device="ios",
            intelligence_files_summary=angel.files_cabinet.get_summary(),
        )

    def _forward_openai_realtime_to_client(sio: socketio.AsyncServer, sid: str, msg: dict) -> None:
        from asgiref.sync import async_to_sync

        async def _emit(event: str, data: dict) -> None:
            await sio.emit(event, data, room=sid, namespace="/realtime")

        emit_sync = async_to_sync(_emit)
        t = msg.get("type")
        try:
            if t == "_realtime_socket_closed":
                emit_sync("realtime_error", {"message": "OpenAI Realtime connection closed"})
                return
            if t == "response.audio.delta":
                d = msg.get("delta", "")
                if d:
                    emit_sync("realtime_response_audio", {"delta": d})
                return
            if t == "response.audio_transcript.delta":
                d = msg.get("delta", "")
                if d:
                    emit_sync("realtime_transcript", {"delta": d, "done": False})
                return
            if t == "response.audio_transcript.done":
                emit_sync(
                    "realtime_transcript",
                    {
                        "transcript": _sanitize_text(str(msg.get("transcript", ""))),
                        "done": True,
                    },
                )
                return
            if t == "response.text.delta":
                d = msg.get("delta", "")
                if d:
                    emit_sync(
                        "realtime_transcript",
                        {"delta": d, "done": False, "channel": "text"},
                    )
                return
            if t == "response.done":
                emit_sync("realtime_response_done", {})
                return
            if t == "error":
                err = msg.get("error") or {}
                msg_txt = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                emit_sync("realtime_error", {"message": _sanitize_text(msg_txt)})
                return
            if t == "response.error":
                emit_sync("realtime_error", {"message": _sanitize_text(str(msg))})
        except Exception as e:
            print(f"[realtime] forward error: {e}", flush=True)
            traceback.print_exc()

    def register_async_socketio(sio: socketio.AsyncServer) -> None:
        """Bind AsyncServer handlers (called from build_asgi_app)."""

        @sio.on("connect")
        async def _ws_connect(sid, environ, auth=None):
            global last_activity_at
            last_activity_at = time.time()
            device = "ios"
            if isinstance(auth, dict):
                d = (auth.get("device") or "").strip().lower()
                if d in _VALID_CLIENT_DEVICES:
                    device = d
            try:
                qs = environ.get("QUERY_STRING") or ""
                qd = (parse_qs(qs).get("device") or [""])[0].strip().lower()
                if qd in _VALID_CLIENT_DEVICES:
                    device = qd
            except Exception:
                pass
            SOCKET_SESSIONS[sid] = {"device": device, "turns": []}
            print(f"[socket] connect sid={sid!s} device={device}", flush=True)
            await sio.emit("connected", {"ok": True}, room=sid)

        @sio.on("disconnect")
        async def _ws_disconnect(sid):
            SOCKET_SESSIONS.pop(sid, None)
            try:
                import angel_realtime_server as _ars

                await _ars.on_socket_disconnect(sid, sio)
            except Exception:
                traceback.print_exc()

        @sio.on("realtime_start")
        async def _evt_realtime_start(sid, data=None):
            try:
                import angel_realtime_server as _ars

                await _ars.handle_realtime_start(sio, sid, angel, data)
            except Exception as e:
                traceback.print_exc()
                await sio.emit("realtime_error", {"message": str(e)}, room=sid)

        @sio.on("realtime_audio_chunk")
        async def _evt_realtime_audio_chunk(sid, data):
            try:
                import angel_realtime_server as _ars

                await _ars.handle_realtime_audio_chunk(sio, sid, data)
            except Exception:
                traceback.print_exc()

        @sio.on("realtime_text")
        async def _evt_realtime_text(sid, data):
            try:
                import angel_realtime_server as _ars

                await _ars.handle_realtime_text(sio, sid, angel, data)
            except Exception as e:
                traceback.print_exc()
                await sio.emit("realtime_error", {"message": str(e)}, room=sid)

        @sio.on("realtime_stop")
        async def _evt_realtime_stop(sid, data=None):
            try:
                import angel_realtime_server as _ars

                await _ars.handle_realtime_stop(sio, sid)
            except Exception as e:
                traceback.print_exc()
                await sio.emit("realtime_error", {"message": str(e)}, room=sid)

        @sio.on("user_text")
        async def _ws_user_text(sid, data):
            global last_activity_at, check_in_message, check_in_generated_at
            last_activity_at = time.time()
            check_in_message = None
            check_in_generated_at = None
            sess = SOCKET_SESSIONS.get(sid)
            if not sess:
                await sio.emit("angel_error", {"message": "No session; reconnect."}, room=sid)
                return
            payload = data if isinstance(data, dict) else {}
            if any(
                k in payload for k in ("file_content", "file_name", "attachment", "base64")
            ):
                _log.info(
                    "socket user_text received attachment-like payload keys=%s (attachments should use /api/files/read)",
                    list(payload.keys()),
                )
            text = (payload.get("message") or payload.get("text") or "").strip()
            if not text:
                await sio.emit("angel_error", {"message": "Empty message."}, room=sid)
                return
            location = normalize_location(payload.get("location"))
            print(f"[socket] location received: {location!r}", flush=True)
            await sio.emit("angel_thinking", {}, room=sid)
            try:
                turns = _session_turns_for(sid)
                loop = asyncio.get_running_loop()
                stream_state: dict[str, object] = {"chunks": [], "tokens": 0}

                def _flush_stream_chunks() -> None:
                    chunks = stream_state.get("chunks")
                    if not isinstance(chunks, list) or not chunks:
                        return
                    out = "".join(str(c) for c in chunks if c)
                    stream_state["chunks"] = []
                    stream_state["tokens"] = 0
                    if not out:
                        return
                    try:
                        asyncio.run_coroutine_threadsafe(
                            sio.emit(
                                "angel_chunk",
                                {"chunk": _sanitize_text(out)},
                                room=sid,
                            ),
                            loop,
                        )
                    except Exception:
                        pass

                def _stream_callback(chunk: str) -> None:
                    if not isinstance(chunk, str) or not chunk:
                        return
                    chunks = stream_state.get("chunks")
                    if not isinstance(chunks, list):
                        chunks = []
                        stream_state["chunks"] = chunks
                    chunks.append(chunk)
                    tok_inc = max(1, len(re.findall(r"\S+", chunk)))
                    stream_state["tokens"] = int(stream_state.get("tokens") or 0) + tok_inc
                    if int(stream_state["tokens"]) >= 16:
                        _flush_stream_chunks()

                reply = await asyncio.to_thread(
                    angel.generate_reply,
                    text,
                    device=sess["device"],
                    session_turns=turns or None,
                    location=location,
                    stream_callback=_stream_callback,
                )
                _flush_stream_chunks()
                clean_a = strip_markdown(reply) if angel.use_voice else reply
                _append_turn(sid, text, clean_a)
                await sio.emit(
                    "angel_reply_complete",
                    {"reply": _sanitize_text(reply)},
                    room=sid,
                )
            except Exception as e:
                traceback.print_exc()
                await sio.emit("angel_error", {"message": str(e)}, room=sid)

        @sio.on("user_audio")
        async def _ws_user_audio(sid, data):
            global last_activity_at, check_in_message, check_in_generated_at
            last_activity_at = time.time()
            check_in_message = None
            check_in_generated_at = None
            sess = SOCKET_SESSIONS.get(sid)
            if not sess:
                await sio.emit("angel_error", {"message": "No session; reconnect."}, room=sid)
                return
            payload = data if isinstance(data, dict) else {}
            b64 = payload.get("audio_base64") or payload.get("audio") or ""
            if not b64:
                await sio.emit("angel_error", {"message": "Missing audio_base64."}, room=sid)
                return
            await sio.emit("angel_thinking", {}, room=sid)
            try:
                raw = base64.b64decode(b64)
            except Exception as e:
                await sio.emit("angel_error", {"message": f"Invalid base64 audio: {e}"}, room=sid)
                return
            filename = (payload.get("filename") or "recording.m4a").strip() or "recording.m4a"
            try:
                transcript = await asyncio.to_thread(
                    functools.partial(transcribe_with_whisper, raw, filename=filename)
                )
                transcript = (transcript or "").strip()
            except Exception as e:
                traceback.print_exc()
                await sio.emit("angel_error", {"message": f"Transcription failed: {e}"}, room=sid)
                return
            await sio.emit("angel_transcript", {"transcript": _sanitize_text(transcript)}, room=sid)
            if not transcript:
                await sio.emit(
                    "angel_response",
                    {"reply": "I couldn't make out what you said."},
                    room=sid,
                )
                return
            location = normalize_location(payload.get("location"))
            print(f"[socket] location received: {location!r}", flush=True)
            try:
                turns = _session_turns_for(sid)
                reply = await asyncio.to_thread(
                    angel.generate_reply,
                    transcript,
                    device=sess["device"],
                    session_turns=turns or None,
                    location=location,
                )
                clean_a = strip_markdown(reply) if angel.use_voice else reply
                _append_turn(sid, transcript, clean_a)
                await sio.emit("angel_response", {"reply": _sanitize_text(reply)}, room=sid)
            except Exception as e:
                traceback.print_exc()
                await sio.emit("angel_error", {"message": str(e)}, room=sid)

        @sio.on("connect", namespace="/realtime")
        async def _realtime_ns_connect(sid, environ, auth=None):
            global last_activity_at
            last_activity_at = time.time()
            if AngelRealtimeSession is None:
                print("[realtime] AngelRealtimeSession unavailable (import failed)", flush=True)
                return False
            if not (os.getenv("OPENAI_REALTIME_API_KEY") or os.getenv("OPENAI_API_KEY")):
                print("[realtime] Missing OPENAI_REALTIME_API_KEY / OPENAI_API_KEY", flush=True)
                return False
            try:
                system_prompt = _build_openai_realtime_system_prompt()
                rt = AngelRealtimeSession()
                rt.connect(system_prompt)
            except Exception as e:
                traceback.print_exc()
                print(f"[realtime] connect failed: {e}", flush=True)
                return False
            REALTIME_PROXY_BY_SID[sid] = {"session": rt}
            rt.start_receiver_thread(
                lambda m, sid=sid: _forward_openai_realtime_to_client(sio, sid, m)
            )
            print(f"[realtime] OpenAI Realtime proxy started sid={sid!s}", flush=True)
            return True

        @sio.on("disconnect", namespace="/realtime")
        async def _realtime_ns_disconnect(sid):
            entry = REALTIME_PROXY_BY_SID.pop(sid, None)
            if entry and entry.get("session"):
                try:
                    entry["session"].disconnect()
                except Exception:
                    pass
            print(f"[realtime] disconnected sid={sid!s}", flush=True)

        @sio.on("realtime_audio", namespace="/realtime")
        async def _realtime_ns_audio(sid, data):
            global last_activity_at
            last_activity_at = time.time()
            entry = REALTIME_PROXY_BY_SID.get(sid)
            if not entry:
                await sio.emit(
                    "realtime_error",
                    {"message": "No Realtime session; reconnect."},
                    namespace="/realtime",
                    room=sid,
                )
                return
            payload = data if isinstance(data, dict) else {}
            b64 = payload.get("audio") or payload.get("audio_base64") or ""
            if not isinstance(b64, str) or not b64.strip():
                await sio.emit(
                    "realtime_error",
                    {"message": "Missing audio (base64)."},
                    namespace="/realtime",
                    room=sid,
                )
                return
            fmt = (payload.get("format") or "pcm16_24000").strip().lower()
            rt = entry["session"]
            try:
                if fmt in ("pcm16", "pcm16_24000", "pcm_s16le_24000", "raw"):
                    raw = base64.b64decode(b64.strip())
                    rt.append_input_pcm16_24k(raw)
                elif fmt in ("wav", "audio/wav", "wave"):
                    raw = base64.b64decode(b64.strip())
                    rt.append_input_wav_bytes(raw)
                else:
                    rt.append_input_audio_base64(b64.strip())
            except Exception as e:
                traceback.print_exc()
                await sio.emit(
                    "realtime_error",
                    {"message": str(e)},
                    namespace="/realtime",
                    room=sid,
                )

        @sio.on("realtime_commit", namespace="/realtime")
        async def _realtime_ns_commit(sid, data=None):
            global last_activity_at
            last_activity_at = time.time()
            entry = REALTIME_PROXY_BY_SID.get(sid)
            if not entry:
                await sio.emit(
                    "realtime_error",
                    {"message": "No Realtime session; reconnect."},
                    namespace="/realtime",
                    room=sid,
                )
                return
            rt = entry["session"]
            try:
                rt.commit_input_buffer()
                rt.create_audio_response()
            except Exception as e:
                traceback.print_exc()
                await sio.emit(
                    "realtime_error",
                    {"message": str(e)},
                    namespace="/realtime",
                    room=sid,
                )

    app.register_async_socketio = register_async_socketio

    INDEX_HTML = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Angel – Mobile</title>
      <style>
        body {
          margin: 0;
          padding: 0;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background-color: #121212;
          color: #ffffff;
          display: flex;
          flex-direction: column;
          height: 100vh;
        }
        header {
          background-color: #1e1e1e;
          padding: 12px 16px;
          display: flex;
          flex-direction: column;
        }
        header h1 {
          margin: 0;
          font-size: 1.4rem;
        }
        header span {
          margin-top: 4px;
          color: #aaaaaa;
          font-size: 0.8rem;
        }
        #chat {
          flex: 1;
          overflow-y: auto;
          padding: 12px 16px;
          box-sizing: border-box;
        }
        .msg {
          margin-bottom: 10px;
        }
        .from-user {
          color: #64b5f6;
        }
        .from-angel {
          color: #4caf50;
        }
        .bubble {
          padding: 8px 12px;
          border-radius: 10px;
          background-color: #1e1e1e;
          display: inline-block;
          max-width: 90%;
          word-wrap: break-word;
        }
        footer {
          padding: 8px 12px;
          background-color: #121212;
          border-top: 1px solid #333333;
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        #status {
          font-size: 0.8rem;
          color: #aaaaaa;
        }
        #input-row {
          display: flex;
          gap: 6px;
        }
        #text-input {
          flex: 1;
          padding: 8px 10px;
          border-radius: 6px;
          border: none;
          background-color: #1e1e1e;
          color: #ffffff;
          font-size: 0.9rem;
        }
        #send-btn, #voice-btn {
          padding: 8px 10px;
          border-radius: 6px;
          border: none;
          background-color: #2e2e2e;
          color: #ffffff;
          font-size: 0.9rem;
        }
        #send-btn:active, #voice-btn:active {
          background-color: #3e3e3e;
        }
        #voice-toggle.on { background-color: #2e7d32; }
        #voice-toggle.off { background-color: #424242; }
        .briefing-block {
          margin-bottom: 16px;
          padding: 14px 16px;
          background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
          border-radius: 12px;
          border-left: 4px solid #4fc3f7;
          color: #e3f2fd;
        }
        .briefing-block h3 {
          margin: 0 0 8px 0;
          font-size: 0.95rem;
          color: #90caf9;
        }
        .briefing-block p {
          margin: 0;
          font-size: 0.9rem;
          line-height: 1.5;
          white-space: pre-wrap;
        }
        .check-in-block {
          margin-bottom: 12px;
          padding: 10px 14px;
          background-color: #1b5e20;
          border-radius: 10px;
          color: #c8e6c9;
          font-size: 0.9rem;
        }
      </style>
    </head>
    <body>
      <header>
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
          <div>
            <h1>Angel</h1>
            <span>Personal AI Companion – Mobile</span>
          </div>
          <button id="voice-toggle" type="button" style="padding: 6px 12px; border-radius: 6px; border: none; background: #2e2e2e; color: #fff; font-size: 0.85rem; cursor: pointer; white-space: nowrap;">
            &#128266; Voice Mode
          </button>
        </div>
      </header>
      <main id="chat"><div id="briefing-container"></div><div id="check-in-container"></div></main>
      <footer>
        <div id="status">Idle</div>
        <div id="input-row">
          <input id="text-input" type="text" placeholder="Type a message..." />
          <button id="send-btn" type="button">Send</button>
          <button id="voice-btn" type="button">🎤 Hold to speak</button>
        </div>
      </footer>
      <audio id="tts-audio" style="display:none;" preload="auto"></audio>
      <script>
        document.addEventListener("DOMContentLoaded", function() {
        const chat = document.getElementById("chat");
        const statusEl = document.getElementById("status");
        const textInput = document.getElementById("text-input");
        const sendBtn = document.getElementById("send-btn");
        const voiceBtn = document.getElementById("voice-btn");
        const voiceToggle = document.getElementById("voice-toggle");
        const ttsAudio = document.getElementById("tts-audio");
        const briefingContainer = document.getElementById("briefing-container");
        const checkInContainer = document.getElementById("check-in-container");

        console.log("[Angel] DOM elements:", { sendBtn: sendBtn != null, voiceToggle: voiceToggle != null, voiceBtn: voiceBtn != null, textInput: textInput != null });

        var inputRow = document.getElementById("input-row");
        if (inputRow) inputRow.style.display = "flex";
        if (textInput) textInput.style.visibility = "visible";
        if (sendBtn) sendBtn.style.visibility = "visible";

        let voiceMode = false;

        function isToday(ts) {
          if (ts == null || ts === undefined) return false;
          const d = new Date(ts * 1000);
          const today = new Date();
          return d.getDate() === today.getDate() && d.getMonth() === today.getMonth() && d.getFullYear() === today.getFullYear();
        }
        function isWithinLast24Hours(ts) {
          if (ts == null || ts === undefined) return false;
          return (Date.now() / 1000) - ts < 24 * 3600;
        }

        async function loadBriefingAndCheckIn() {
          try {
            console.log("[loadBriefingAndCheckIn] Fetching /api/briefing and /api/check_in...");
            const [briefRes, checkRes] = await Promise.all([fetch("/api/briefing"), fetch("/api/check_in")]);
            const briefingData = briefRes.ok ? await briefRes.json() : {};
            const checkInData = checkRes.ok ? await checkRes.json() : {};
            console.log("[loadBriefingAndCheckIn] /api/briefing response:", { ok: briefRes.ok, status: briefRes.status, briefing: briefingData.briefing ? "(present)" : null, generated_at: briefingData.generated_at });
            console.log("[loadBriefingAndCheckIn] /api/check_in response:", { ok: checkRes.ok, message: checkInData.message ? "(present)" : null, generated_at: checkInData.generated_at });
            const showBriefing = briefingData.briefing && (isToday(briefingData.generated_at) || isWithinLast24Hours(briefingData.generated_at));
            console.log("[loadBriefingAndCheckIn] isToday=" + isToday(briefingData.generated_at) + ", within24h=" + isWithinLast24Hours(briefingData.generated_at) + ", showBriefing=" + showBriefing + ", generated_at=" + briefingData.generated_at);
            if (briefingContainer) {
              if (showBriefing) {
                const escaped = briefingData.briefing.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\\n/g, "<br>");
                briefingContainer.innerHTML = '<div class="briefing-block"><h3>☀️ Morning briefing</h3><p>' + escaped + '</p></div>';
              } else {
                briefingContainer.innerHTML = '';
              }
            }
            if (checkInContainer) {
              if (checkInData.message && checkInData.generated_at) {
                const escapedCheckIn = checkInData.message.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                checkInContainer.innerHTML = '<div class="check-in-block">' + escapedCheckIn + '</div>';
              } else {
                checkInContainer.innerHTML = '';
              }
            }
          } catch (e) {
            console.error("Load briefing/check-in:", e);
          }
        }
        loadBriefingAndCheckIn();

        function updateVoiceToggleLabel() {
          if (!voiceToggle) return;
          voiceToggle.textContent = voiceMode ? "\uD83D\uDD0A Voice Mode" : "\uD83D\uDD07 Text Mode";
          voiceToggle.classList.toggle("on", voiceMode);
          voiceToggle.classList.toggle("off", !voiceMode);
        }

        if (voiceToggle) {
          voiceToggle.type = "button";
          voiceToggle.addEventListener("click", function(e) {
            e.preventDefault();
            voiceMode = !voiceMode;
            updateVoiceToggleLabel();
          });
          updateVoiceToggleLabel();
        }

        async function playTts(text) {
          if (!text || !voiceMode || !ttsAudio) return;
          try {
            const resp = await fetch("/api/tts", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text: text }),
            });
            if (!resp.ok) {
              console.error("TTS request failed with status", resp.status);
              // Fallback to Web Speech API if available
              if ("speechSynthesis" in window) {
                const u = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.speak(u);
              }
              return;
            }
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            if (ttsAudio) {
              ttsAudio.src = url;
              ttsAudio.onended = () => { URL.revokeObjectURL(url); };
            }
            try {
              if (ttsAudio) await ttsAudio.play();
              console.log("TTS audio playback started successfully.");
            } catch (playErr) {
              console.error("TTS audio playback failed:", playErr);
              // Fallback to Web Speech API if audio playback fails
              if ("speechSynthesis" in window) {
                const u = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.speak(u);
              }
            }
          } catch (e) {
            console.error("TTS request or playback error:", e);
            if ("speechSynthesis" in window) {
              const u = new SpeechSynthesisUtterance(text);
              window.speechSynthesis.speak(u);
            }
          }
        }

        function appendMessage(sender, text) {
          if (!chat) return;
          const div = document.createElement("div");
          div.className = "msg " + (sender === "You" ? "from-user" : "from-angel");
          const bubble = document.createElement("div");
          bubble.className = "bubble";
          bubble.textContent = sender + ": " + text;
          div.appendChild(bubble);
          chat.appendChild(div);
          chat.scrollTop = chat.scrollHeight;
        }

        async function sendText() {
          if (!textInput) return;
          const msg = textInput.value.trim();
          if (!msg) return;
          textInput.value = "";
          appendMessage("You", msg);
          if (statusEl) statusEl.textContent = "Angel is thinking...";
          try {
            const resp = await fetch("/api/message", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message: msg, device: "mobile_web" }),
            });
            const data = await resp.json().catch(function() { return {}; });
            if (!resp.ok) {
              appendMessage("Angel", data.error || "Something went wrong. Please try again.");
              return;
            }
            if (data.reply != null) {
              appendMessage("Angel", data.reply);
              await playTts(data.reply);
            } else {
              appendMessage("Angel", "I didn't get a reply. Please try again.");
            }
          } catch (e) {
            console.error("sendText error:", e);
            appendMessage("Angel", "I ran into an error processing that.");
          } finally {
            if (statusEl) statusEl.textContent = "Idle";
          }
        }

        if (sendBtn) {
          sendBtn.type = "button";
          sendBtn.addEventListener("click", function(e) {
            e.preventDefault();
            sendText();
          });
        }
        if (textInput) {
          textInput.addEventListener("keydown", function(e) {
            if (e.key === "Enter") {
              e.preventDefault();
              sendText();
            }
          });
        }

        // Voice input using MediaRecorder
        let mediaRecorder = null;
        let chunks = [];

        async function initMedia() {
          if (mediaRecorder) return;
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = (e) => {
              if (e.data.size > 0) {
                chunks.push(e.data);
              }
            };
            mediaRecorder.onstop = async () => {
              const blob = new Blob(chunks, { type: "audio/webm" });
              chunks = [];
              if (statusEl) statusEl.textContent = "Transcribing and thinking...";
              appendMessage("You", "(voice message)");
              const formData = new FormData();
              formData.append("audio", blob, "audio.webm");
              formData.append("device", "mobile_web");
              try {
                const resp = await fetch("/api/voice", {
                  method: "POST",
                  body: formData,
                });
                const data = await resp.json();
                if (data.transcript) {
                  appendMessage("You", data.transcript);
                }
                appendMessage("Angel", data.reply);
                await playTts(data.reply);
              } catch (e) {
                appendMessage("Angel", "I couldn't process that voice message.");
              } finally {
                if (statusEl) statusEl.textContent = "Idle";
              }
            };
          } catch (e) {
            alert("Microphone access denied or unavailable.");
          }
        }

        if (voiceBtn) {
          voiceBtn.addEventListener("mousedown", async () => {
            await initMedia();
            if (!mediaRecorder) return;
            chunks = [];
            mediaRecorder.start();
            if (statusEl) statusEl.textContent = "Listening (hold button)...";
          });
          voiceBtn.addEventListener("mouseup", () => {
            if (mediaRecorder && mediaRecorder.state === "recording") {
              mediaRecorder.stop();
            }
          });
          voiceBtn.addEventListener("touchstart", async (e) => {
            e.preventDefault();
            await initMedia();
            if (!mediaRecorder) return;
            chunks = [];
            mediaRecorder.start();
            if (statusEl) statusEl.textContent = "Listening (hold button)...";
          });
          voiceBtn.addEventListener("touchend", (e) => {
            e.preventDefault();
            if (mediaRecorder && mediaRecorder.state === "recording") {
              mediaRecorder.stop();
            }
          });
        }

        });
      </script>
    </body>
    </html>
    """

    @app.route("/", methods=["GET"])
    def index():
        # Sanitize the HTML template itself in case any stray surrogates
        # make their way into it (e.g. from string concatenation).
        safe_html = _sanitize_text(INDEX_HTML)
        return render_template_string(safe_html)

    @app.route("/api/message", methods=["POST"])
    def api_message():
        global last_activity_at, check_in_message, check_in_generated_at
        last_activity_at = time.time()
        check_in_message = None
        check_in_generated_at = None
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400
        device = _request_device()
        location = _parse_location_from_json_body(data)
        reply = angel.generate_reply(message, device=device, location=location)
        reply = _sanitize_text(reply)
        return jsonify({"reply": reply})

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        """Alias for /api/message (mobile / OpenAI-style clients)."""
        return api_message()

    @app.route("/api/admin/seed-memories", methods=["GET"])
    def api_admin_seed_memories():
        bundled = Path("/app/tyler_memories.json")
        volume = Path("/app/data/tyler_memories.json")

        def _file_status(p: Path) -> dict:
            return {
                "path": str(p),
                "exists": p.exists(),
                "size_bytes": (p.stat().st_size if p.exists() else 0),
            }

        def _count_memories(p: Path) -> int:
            if not p.exists():
                return 0
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    return len(raw)
                if isinstance(raw, dict):
                    users = raw.get("users")
                    if isinstance(users, dict):
                        return sum(len(v) for v in users.values() if isinstance(v, list))
                    memories = raw.get("memories")
                    if isinstance(memories, list):
                        return len(memories)
                return 0
            except Exception:
                return 0

        copied = False
        copied_memory_count = 0
        reason = "not_needed"

        if bundled.exists():
            should_copy = (not volume.exists()) or (
                bundled.stat().st_size > volume.stat().st_size
            )
            if should_copy:
                volume.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(bundled, volume)
                copied = True
                copied_memory_count = _count_memories(volume)
                reason = "copied"
            else:
                reason = "volume_up_to_date"
        else:
            reason = "bundled_missing"

        return jsonify(
            {
                "ok": True,
                "copied": copied,
                "copied_memory_count": copied_memory_count,
                "reason": reason,
                "bundled": {**_file_status(bundled), "memory_count": _count_memories(bundled)},
                "volume": {**_file_status(volume), "memory_count": _count_memories(volume)},
            }
        )

    @app.route("/api/voice", methods=["POST"])
    def api_voice():
        global last_activity_at, check_in_message, check_in_generated_at
        last_activity_at = time.time()
        check_in_message = None
        check_in_generated_at = None
        if "audio" not in request.files:
            return jsonify({"error": "Missing audio file"}), 400
        file = request.files["audio"]
        audio_bytes = file.read()
        # Convert to WAV bytes if needed; many Whisper models accept
        # common browser formats, but to be safe we pass raw bytes and
        # rely on transcribe_with_whisper's behavior.
        transcript = transcribe_with_whisper(audio_bytes).strip()
        if not transcript:
            reply = "I couldn't clearly hear what you said."
            return jsonify({"transcript": "", "reply": reply})
        device = _request_device()
        location = None
        raw_loc = request.form.get("location")
        if raw_loc:
            try:
                location = normalize_location(json.loads(raw_loc))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"[api_voice] Ignoring invalid location JSON: {e}", flush=True)
        reply = angel.generate_reply(transcript, device=device, location=location)
        return jsonify(
            {
                "transcript": _sanitize_text(transcript),
                "reply": _sanitize_text(reply),
            }
        )

    @app.route("/api/execute", methods=["POST"])
    def api_execute():
        """
        Run Python in a sandboxed subprocess (30s timeout).
        JSON body: code (str), context (optional description for logging).
        """
        data = request.get_json(silent=True) or {}
        code = (data.get("code") or "").strip()
        context = (data.get("context") or "").strip()
        if not code:
            print("[api_execute] error: missing code", flush=True)
            return jsonify({"success": False, "output": "", "error": "Missing 'code'."}), 400
        try:
            if context:
                print(f"[api_execute] context: {context[:800]!r}", flush=True)
            result = execute_python_sandbox(code)
            return jsonify(
                {
                    "success": bool(result.get("success")),
                    "output": _sanitize_text(str(result.get("output") or "")),
                    "error": _sanitize_text(str(result.get("error") or "")),
                }
            )
        except Exception as e:
            print(f"[api_execute] unexpected error: {e}", flush=True)
            traceback.print_exc()
            return jsonify(
                {
                    "success": False,
                    "output": "",
                    "error": _sanitize_text(str(e)),
                }
            ), 500

    @app.route("/api/vision", methods=["POST"])
    def api_vision():
        """
        Camera vision: JPEG (base64) + question, using Claude vision with Angel's full system prompt and memory.
        JSON body: image (base64 JPEG), question (str), device (ios | desktop | mobile_web).
        For routed forensic JSON (classify + pipelines + auto-file Visual Intelligence), use POST /api/vision/forensic.
        For multi-layer FA-* forensics or UAP/document modes, use POST /api/forensic/analyze (or /uap /document).
        """
        global last_activity_at, check_in_message, check_in_generated_at
        last_activity_at = time.time()
        check_in_message = None
        check_in_generated_at = None

        data = request.get_json(silent=True) or {}
        image_b64 = _normalize_jpeg_base64(str(data.get("image") or ""))
        question = (data.get("question") or "").strip()
        device = _vision_device_from_body(str(data.get("device") or ""))

        if not image_b64:
            print("[api_vision] error: missing or empty image", flush=True)
            return jsonify({"error": "Missing or empty 'image' (base64 JPEG)."}), 400
        if not question:
            print("[api_vision] error: missing question", flush=True)
            return jsonify({"error": "Missing or empty 'question'."}), 400

        try:
            try:
                raw_bytes = base64.b64decode(image_b64, validate=True)
            except TypeError:
                raw_bytes = base64.b64decode(image_b64)
        except Exception as e:
            print(f"[api_vision] error: invalid base64 image: {e}", flush=True)
            traceback.print_exc()
            return jsonify({"error": "Invalid base64 image data."}), 400

        if len(raw_bytes) < 10 or not raw_bytes.startswith(b"\xff\xd8\xff"):
            print("[api_vision] error: payload does not look like JPEG", flush=True)
            return jsonify({"error": "Image must be base64-encoded JPEG."}), 400

        try:
            memories = angel._fetch_combined_memories()
            memory_summary = build_memory_summary_with_sections(memories, question)
            cc_for_prompt = (
                angel.computer_control_enabled
                and COMPUTER_CONTROL_AVAILABLE
                and device not in ("ios", "mobile_web")
            )
            system_prompt = build_system_prompt(
                memory_summary,
                voice_mode=False,
                strategy_hint=False,
                pattern_hint=False,
                profile_hint=False,
                computer_control_enabled=cc_for_prompt,
                device=device,
                intelligence_files_summary=angel.files_cabinet.get_summary(),
            )
            client = create_anthropic_client()
            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": question,
                },
            ]
            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                temperature=0.4,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            parts: list[str] = []
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            reply = "\n".join(parts).strip() or "(No text in response.)"
            return jsonify({"reply": _sanitize_text(reply)})
        except Exception as e:
            print(f"[api_vision] Claude vision error: {e}", flush=True)
            traceback.print_exc()
            return jsonify({"error": "Vision analysis failed.", "details": str(e)}), 500

    @app.route("/api/vision/forensic", methods=["POST"])
    def api_vision_forensic():
        """
        Computer vision on demand: one Claude call — classify + routed forensic JSON.
        Body: { image_base64, context?, tyler_location?, file_name?, skip_autofile?, skip_network_apply? }
        Also accepts image (alias) for image_base64.
        """
        if angel is None:
            return jsonify({"ok": False, "error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        image_b64 = str(
            data.get("image_base64") or data.get("image_b64") or data.get("image") or ""
        ).strip()
        ctx = (data.get("context") or data.get("question") or "").strip()
        tyler_loc = (data.get("tyler_location") or "").strip()
        fn = (data.get("file_name") or data.get("name") or "photo.jpg").strip()
        skip_af = bool(data.get("skip_autofile", False))
        skip_net = bool(data.get("skip_network_apply", False))
        if not image_b64:
            return jsonify({"ok": False, "error": "Missing image_base64 (or image)."}), 400
        try:
            memories = angel._fetch_combined_memories()
            mem_excerpt = build_memory_summary_with_sections(memories, ctx or "visual forensic")
            if len(mem_excerpt) > 12000:
                mem_excerpt = mem_excerpt[:11997] + "..."
            r = angel_vision_forensic.analyze_image_forensic(
                image_b64,
                ctx,
                tyler_loc or None,
                angel.anthropic_client,
                angel.files_cabinet,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                file_name=fn or "photo.jpg",
                intelligence_files_summary=angel.files_cabinet.get_summary(),
                memory_summary=mem_excerpt,
                skip_autofile=skip_af,
                skip_network_apply=skip_net,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/vision/forensic/file", methods=["POST"])
    def api_vision_forensic_file():
        """
        Save a forensic result to Visual Intelligence when not auto-filed.
        Body: { image_base64, forensic_json, file_name? } — forensic_json is the prior /api/vision/forensic object (or inner fields).
        """
        if angel is None:
            return jsonify({"ok": False, "error": "Angel not initialized"}), 503
        _log.info(
            "vision/forensic/file called - keys: %s", _request_json_keys_for_log()
        )
        data = request.get_json(silent=True) or {}
        image_b64 = str(
            data.get("image_base64") or data.get("image_b64") or data.get("image") or ""
        ).strip()
        fj = data.get("forensic_json") or data.get("forensic")
        if isinstance(fj, str):
            try:
                fj = json.loads(fj)
            except Exception:
                fj = None
        fn = (data.get("file_name") or "photo.jpg").strip()
        if not image_b64 or not isinstance(fj, dict):
            return jsonify(
                {"ok": False, "error": "Missing image_base64 or forensic_json object."}
            ), 400
        try:
            r = angel_vision_forensic.file_visual_intel_manual(
                fj, image_b64, angel.files_cabinet, file_name=fn or "photo.jpg"
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Forensic visual analysis (Batcomputer): authenticity, UAP imagery, document photos ---
    # iPhone / iOS: keep using /api/vision for default chat+camera; call /api/forensic/* when Tyler
    # enables "forensic mode" or needs manipulation/authenticity assessment (see system prompt).

    @app.route("/api/forensic/analyze", methods=["POST"])
    def api_forensic_analyze():
        """Full four-layer forensic JSON. Body: { image_b64, file_name?, context? }."""
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        image_b64 = data.get("image_b64") or data.get("image") or ""
        fn = (data.get("file_name") or data.get("name") or "image.jpg").strip()
        ctx = (data.get("context") or "").strip()
        try:
            r = angel_forensic.forensic_analyze_image(
                str(image_b64),
                fn,
                ctx,
                angel.anthropic_client,
                angel.files_cabinet,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/forensic/uap", methods=["POST"])
    def api_forensic_uap():
        """UAP-focused forensic pass. Body: { image_b64, file_name?, context? }."""
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        image_b64 = data.get("image_b64") or data.get("image") or ""
        fn = (data.get("file_name") or data.get("name") or "uap.jpg").strip()
        ctx = (data.get("context") or "").strip()
        try:
            r = angel_forensic.forensic_analyze_uap(
                str(image_b64),
                fn,
                ctx,
                angel.anthropic_client,
                angel.files_cabinet,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/forensic/document", methods=["POST"])
    def api_forensic_document():
        """Document-image forensics. Body: { image_b64, file_name?, context? }."""
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        image_b64 = data.get("image_b64") or data.get("image") or ""
        fn = (data.get("file_name") or data.get("name") or "document.jpg").strip()
        ctx = (data.get("context") or "").strip()
        try:
            r = angel_forensic.forensic_analyze_document(
                str(image_b64),
                fn,
                ctx,
                angel.anthropic_client,
                angel.files_cabinet,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Open-source surveillance monitoring (Batcomputer; legal OSINT) ---

    @app.route("/api/surveillance/run", methods=["GET"])
    def api_surveillance_run():
        """Manual trigger: full multi-category Tavily surveillance scan."""
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            r = angel_surveillance.run_osint_surveillance(
                angel.anthropic_client,
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/surveillance/findings", methods=["GET"])
    def api_surveillance_findings():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            lim = request.args.get("limit", "40")
            try:
                n = max(1, min(100, int(lim)))
            except ValueError:
                n = 40
            rows = angel_surveillance.fetch_recent_surveillance_findings(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
                limit=n,
            )
            return jsonify({"ok": True, "findings": rows, "count": len(rows)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/surveillance/signals", methods=["GET"])
    def api_surveillance_signals():
        """Active signals grouped by surveillance category."""
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            by_cat = angel_surveillance.get_signals_by_category(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "by_category": by_cat})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Intelligence File Cabinet (dynamic folders; Mem0 category intelligence_file) ---
    @app.route("/api/files/create", methods=["POST"])
    def api_files_create():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        _log.info("files/create called - keys: %s", _request_json_keys_for_log())
        data = request.get_json(silent=True) or {}
        folder = data.get("folder", "")
        name = (data.get("name") or "").strip()
        content = data.get("content", "")
        tags = data.get("tags")
        if tags is not None and not isinstance(tags, list):
            tags = None
        try:
            rec = angel.files_cabinet.create_file(
                folder, name, content, tags=tags
            )
            return jsonify({"ok": True, "file": rec})
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/files/ping", methods=["POST"])
    def api_files_ping():
        return jsonify({"ok": True, "message": "files ping works"})

    @app.route("/api/files/read", methods=["POST"])
    def api_files_read():
        """
        Read and analyze an uploaded file: multipart form (field `file`, optional `context`, `file_type`)
        or JSON body { file_content (base64), file_name, context, file_type }.
        """
        print("[api/files/read] handler entered", flush=True)
        _log.info(
            "files/read: handler entered (before body) method=%s content_type=%s",
            request.method,
            request.content_type or "(none)",
        )
        print("[files/read] 1 - checking angel", flush=True)
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        print("[files/read] 2 - angel ok, getting json", flush=True)
        data = request.get_json(silent=True) or {}
        print("[files/read] 3 - got json keys:", list(data.keys()), flush=True)

        context = ""
        file_name = "upload.bin"
        file_type = ""
        file_b64 = ""

        ct = (request.content_type or "").lower()
        print("[files/read] 4 - content_type:", ct[:120] if ct else "(empty)", flush=True)
        if "multipart/form-data" in ct:
            print("[files/read] 5 - multipart branch", flush=True)
            f = request.files.get("file")
            print("[files/read] 6 - files.get('file'):", f is not None, flush=True)
            if f and f.filename:
                print("[files/read] 7 - reading upload bytes…", flush=True)
                try:
                    raw = f.read()
                    print("[files/read] 8 - read raw len:", len(raw), flush=True)
                    file_b64 = base64.b64encode(raw).decode("ascii")
                    file_name = f.filename or "upload.bin"
                    print("[files/read] 9 - base64 encoded, name:", file_name, flush=True)
                except Exception as e:
                    print("[files/read] 8-ERR - read failed:", e, flush=True)
                    return jsonify({"ok": False, "error": f"Could not read upload: {e}"}), 400
            context = (request.form.get("context") or "").strip()
            file_type = (request.form.get("file_type") or request.form.get("mime") or "").strip()
            print("[files/read] 10 - multipart context/file_type set", flush=True)
        else:
            print("[files/read] 11 - JSON/non-multipart branch (reuse data from step 2)", flush=True)
            file_b64 = (
                data.get("file_content")
                or data.get("file_data")
                or data.get("file_b64")
                or data.get("image_base64")
                or data.get("content")
                or ""
            ).strip()
            print("[files/read] 12 - got file_b64 length:", len(file_b64), flush=True)
            file_name = (
                data.get("file_name")
                or data.get("filename")
                or data.get("name")
                or "uploaded_file"
            ).strip() or "uploaded_file"
            context = (data.get("context") or "").strip()
            file_type = (data.get("file_type") or data.get("mime") or "").strip()
            print("[files/read] 13 - meta file_name/context/file_type ok", flush=True)

        print("[files/read] 14 - final file_b64 length:", len(file_b64), flush=True)
        if not file_b64:
            print("[files/read] 15 - empty file_b64, returning 400", flush=True)
            return jsonify({
                "ok": False,
                "error": "No file content. Use multipart form field 'file' or JSON 'file_content' (base64).",
            }), 400

        try:
            print("[files/read] 16 - before read_and_analyze_file (thread pool)", flush=True)
            _log.info(
                "files/read before angel_file_reading file=%s type=%s b64_len=%s context_len=%s content_type=%s",
                file_name,
                file_type or "(none)",
                len(file_b64 or ""),
                len(context or ""),
                request.content_type or "(none)",
            )
            print("[files/read] submitted to thread pool", flush=True)
            future = _executor.submit(_do_file_read, file_b64, file_name, file_type, context)
            try:
                r = future.result(timeout=50)
            except _cf.TimeoutError:
                _log.error("files/read thread pool timed out after 50s file=%s", file_name)
                return jsonify({"ok": False, "error": "File analysis timed out"}), 504
            except Exception as e:
                _log.exception("files/read thread pool error file=%s", file_name)
                return jsonify({"ok": False, "error": str(e)}), 500
            print("[files/read] got result from thread pool", flush=True)
            print("[files/read] 17 - after read_and_analyze_file ok=", r.get("ok"), flush=True)
            _log.info(
                "files/read after angel_file_reading ok=%s extraction_method=%s file_type=%s error=%s",
                r.get("ok"),
                r.get("extraction_method"),
                r.get("file_type_detected"),
                r.get("error"),
            )
            if r.get("ok"):
                print("[files/read] 18 - adding filing_offer / suggested_folder", flush=True)
                r["filing_offer"] = angel_file_reading.filing_suggestion_line(
                    str(r.get("intelligence_value") or "MEDIUM")
                )
                r["suggested_folder"] = angel_file_reading.suggest_filing_folder(r)
            print("[files/read] 19 - returning jsonify", flush=True)
            return jsonify(r)
        except Exception as e:
            print("[files/read] ERR - exception:", e, flush=True)
            _log.exception("files/read endpoint crashed before response")
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/files/list", methods=["GET"])
    def api_files_list():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        _log.info("files/list called")
        raw_folder = request.args.get("folder")
        folder_filter = None
        if raw_folder is not None and str(raw_folder).strip():
            folder_filter = str(raw_folder).strip()
        files = angel.files_cabinet.list_files(folder=folder_filter)
        return jsonify({"ok": True, "files": files})

    @app.route("/api/files/get", methods=["GET"])
    def api_files_get():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        _log.info("files/get called name=%s", (request.args.get("name") or "").strip())
        name = (request.args.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Missing query parameter 'name'."}), 400
        rec = angel.files_cabinet.get_file(name)
        if not rec:
            return jsonify({"ok": False, "error": f"No file named {name!r}."}), 404
        return jsonify({"ok": True, "file": rec})

    @app.route("/api/files/update", methods=["POST"])
    def api_files_update():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        _log.info("files/update called - keys: %s", _request_json_keys_for_log())
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        content = data.get("content", "")
        try:
            rec = angel.files_cabinet.update_file(name, content)
            return jsonify({"ok": True, "file": rec})
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/files/search", methods=["POST"])
    def api_files_search():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        _log.info("files/search called - keys: %s", _request_json_keys_for_log())
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"ok": False, "error": "Missing or empty 'query'."}), 400
        matches = angel.files_cabinet.search_files(query)
        return jsonify({"ok": True, "files": matches})

    @app.route("/api/files/summary", methods=["GET"])
    def api_files_summary():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        _log.info("files/summary called")
        text = angel.files_cabinet.get_summary()
        return jsonify({"ok": True, "summary": text})

    @app.route("/api/files/delete", methods=["POST"])
    def api_files_delete():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        _log.info("files/delete called - keys: %s", _request_json_keys_for_log())
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Missing or empty 'name'."}), 400
        try:
            deleted = angel.files_cabinet.delete_file(name)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        if not deleted:
            return jsonify({"ok": False, "error": f"No file named {name!r}."}), 404
        return jsonify({"ok": True, "deleted": name})

    @app.route("/api/files/folders", methods=["GET"])
    def api_files_folders():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        _log.info("files/folders called")
        folders = angel.files_cabinet.list_folders()
        return jsonify({"ok": True, "folders": folders})

    # --- Item 12: threat detection (watch categories + Threat Intelligence files) ---
    # Default watch queries are defined in angel.THREAT_WATCH_DEFAULT_CATEGORIES (imported above).

    @app.route("/api/threats/scan", methods=["GET"])
    def api_threats_scan():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            client = create_anthropic_client()
            memories = angel._fetch_combined_memories()
            memory_summary = build_memory_summary_with_sections(
                memories, None, omit_reflection_section=True
            )
            result = run_threat_detection(
                client,
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                use_mem0_cloud=angel._use_mem0_cloud,
                memory_summary=memory_summary,
            )
            for t in result.get("threats") or []:
                if isinstance(t, dict) and t.get("summary"):
                    t["summary"] = _sanitize_text(str(t["summary"]))
            return jsonify({"ok": True, **result})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/threats/list", methods=["GET"])
    def api_threats_list():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            recs = angel.files_cabinet.list_files(folder=THREAT_INTEL_FOLDER)
            out = []
            for r in recs:
                name = (r.get("name") or "").strip()
                if not name:
                    continue
                full = angel.files_cabinet.get_file(name)
                if full:
                    out.append(full)
            return jsonify({"ok": True, "folder": THREAT_INTEL_FOLDER, "files": out})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/threats/category/add", methods=["POST"])
    def api_threats_category_add():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        label = (data.get("category") or data.get("label") or "").strip()
        if not label:
            return jsonify({"ok": False, "error": "JSON body needs 'category' or 'label'."}), 400
        try:
            add_threat_category(
                angel.memory_client,
                angel.user_id,
                label,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            merged = load_merged_threat_watch_categories(
                angel.memory_client, angel.user_id, angel._use_mem0_cloud
            )
            return jsonify({"ok": True, "added": label, "categories": merged})
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/threats/categories", methods=["GET"])
    def api_threats_categories():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            merged = load_merged_threat_watch_categories(
                angel.memory_client, angel.user_id, angel._use_mem0_cloud
            )
            return jsonify(
                {
                    "ok": True,
                    "categories": merged,
                    "default_count": len(THREAT_WATCH_DEFAULT_CATEGORIES),
                }
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Threat Actor database (Batcomputer opposition network) ---

    @app.route("/api/threat-actors", methods=["GET"])
    def api_threat_actors_list():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            angel_threat_actors.maybe_ensure_threat_actor_seeds(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            tl = (request.args.get("threat_level") or "").strip().upper()
            level = tl if tl in ("LOW", "MEDIUM", "HIGH", "CRITICAL") else None
            rows = angel_threat_actors.list_threat_actors(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
                threat_level=level,
            )
            return jsonify({"ok": True, "actors": rows, "count": len(rows)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/threat-actors/reseed", methods=["GET"])
    def api_threat_actors_reseed():
        """Force upsert of all default threat-actor seeds (manual recovery)."""
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            r = angel_threat_actors.ensure_threat_actor_seeds(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
                force_reseed=True,
            )
            rows = angel_threat_actors.list_threat_actors(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, **r, "actors": rows, "count": len(rows)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/threat-actors/summary", methods=["GET"])
    def api_threat_actors_summary():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            angel_threat_actors.maybe_ensure_threat_actor_seeds(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            s = angel_threat_actors.get_threat_actor_summary(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, **s})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/threat-actors/search", methods=["GET"])
    def api_threat_actors_search():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        q = (request.args.get("q") or "").strip()
        if not q:
            return jsonify({"ok": False, "error": "Missing query parameter q"}), 400
        try:
            angel_threat_actors.maybe_ensure_threat_actor_seeds(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            rows = angel_threat_actors.search_threat_actors(
                q,
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "actors": rows, "count": len(rows)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/threat-actors/<path:actor_id>", methods=["GET"])
    def api_threat_actors_get(actor_id):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            angel_threat_actors.maybe_ensure_threat_actor_seeds(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            a = angel_threat_actors.get_threat_actor(
                actor_id,
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
            )
            if not a:
                return jsonify({"ok": False, "error": "not found"}), 404
            return jsonify({"ok": True, "actor": a})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/threat-actors/add", methods=["POST"])
    def api_threat_actors_add():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "name required"}), 400
        try:
            actor = angel_threat_actors.add_threat_actor(
                name,
                str(data.get("actor_type") or "organization"),
                str(data.get("role") or ""),
                str(data.get("affiliation") or ""),
                str(data.get("threat_type") or "unknown"),
                str(data.get("threat_level") or "MEDIUM"),
                data.get("known_actions") if isinstance(data.get("known_actions"), list) else [],
                data.get("evidence") if isinstance(data.get("evidence"), list) else [],
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                files_cabinet=angel.files_cabinet,
                use_mem0_cloud=angel._use_mem0_cloud,
                notes=str(data.get("notes") or ""),
                status=str(data.get("status") or "active"),
                actor_id=(data.get("actor_id") or "").strip() or None,
                sync_network=bool(data.get("sync_network", True)),
            )
            return jsonify({"ok": True, "actor": actor})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/threat-actors/assess", methods=["POST"])
    def api_threat_actors_assess():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "name required"}), 400
        context = str(data.get("context") or "")
        try:
            r = angel_threat_actors.assess_new_threat_actor(
                angel.anthropic_client,
                name,
                context,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                files_cabinet=angel.files_cabinet,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Item 13: OSINT deep background dossiers ---

    @app.route("/api/osint/research", methods=["POST"])
    def api_osint_research():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        target = (data.get("target") or "").strip()
        tt = (data.get("target_type") or "person").strip().lower()
        if tt not in ("person", "organization"):
            tt = "person"
        context = data.get("context")
        if context is not None and not isinstance(context, str):
            context = str(context)
        if not target:
            return jsonify({"ok": False, "error": "JSON body must include non-empty 'target'."}), 400
        try:
            client = create_anthropic_client()
            memories = angel._fetch_combined_memories()
            memory_summary = build_memory_summary_with_sections(
                memories, None, omit_reflection_section=True
            )
            result = run_osint_background(
                target,
                tt,
                context,
                anthropic_client=client,
                files_cabinet=angel.files_cabinet,
                memory_summary=memory_summary,
            )
            if result.get("ok") and not result.get("cached") and result.get("dossier_body"):
                try:
                    nm = map_osint_to_network(
                        str(result["dossier_body"]),
                        target,
                        primary_target_type=tt,
                        anthropic_client=client,
                        memory_client=angel.memory_client,
                        user_id=angel.user_id,
                        files_cabinet=angel.files_cabinet,
                        use_mem0_cloud=angel._use_mem0_cloud,
                    )
                    result = dict(result)
                    result["network_mapping"] = nm
                except Exception as nex:
                    result = dict(result)
                    result["network_mapping_error"] = str(nex)
            if result.get("dossier_body"):
                result = dict(result)
                result["dossier_body"] = _sanitize_text(str(result["dossier_body"]))
            if result.get("summary_for_tyler"):
                result["summary_for_tyler"] = _sanitize_text(str(result["summary_for_tyler"]))
            code = 200 if result.get("ok") else 502
            return jsonify({"ok": bool(result.get("ok")), **result}), code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/osint/dossiers", methods=["GET"])
    def api_osint_dossiers():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            recs = angel.files_cabinet.list_files(folder=OSINT_DOSSIERS_FOLDER)
            return jsonify({"ok": True, "folder": OSINT_DOSSIERS_FOLDER, "dossiers": recs})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/osint/dossier/<path:name>", methods=["GET"])
    def api_osint_dossier_get(name):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        fname = (name or "").strip()
        if not fname:
            return jsonify({"ok": False, "error": "Missing dossier name."}), 400
        try:
            rec = angel.files_cabinet.get_file(fname)
            if not rec:
                return jsonify({"ok": False, "error": f"No file named {fname!r}."}), 404
            if (rec.get("folder") or "").strip().lower() != OSINT_DOSSIERS_FOLDER.lower():
                return jsonify({"ok": False, "error": "Not an OSINT dossier file."}), 404
            return jsonify({"ok": True, "dossier": rec})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Mission connection graph (network_node / network_edge + Network Intelligence files) ---

    @app.route("/api/network/summary", methods=["GET"])
    def api_network_summary():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            s = get_network_summary(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
                angel.files_cabinet,
            )
            return jsonify({"ok": True, **s})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/network/nodes", methods=["GET"])
    def api_network_nodes():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            nodes, _ = network_load_graph(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
                angel.files_cabinet,
            )
            return jsonify({"ok": True, "nodes": list(nodes.values())})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/network/node/<path:node_id>", methods=["GET"])
    def api_network_node_get(node_id):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            data = get_node_connections(
                node_id,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            if not data:
                return jsonify({"ok": False, "error": f"Unknown node {node_id!r}."}), 404
            return jsonify({"ok": True, **data})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/network/cluster/<path:node_id>", methods=["GET"])
    def api_network_cluster_get(node_id):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        depth = request.args.get("depth", "2")
        try:
            d = int(depth)
        except ValueError:
            d = 2
        try:
            cl = get_network_cluster(
                node_id,
                depth=d,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            if not cl:
                return jsonify({"ok": False, "error": f"Unknown node {node_id!r}."}), 404
            return jsonify({"ok": True, "cluster": cl})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/network/path", methods=["GET"])
    def api_network_path():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        a = (request.args.get("from") or request.args.get("from_id") or "").strip()
        b = (request.args.get("to") or request.args.get("to_id") or "").strip()
        if not a or not b:
            return jsonify({"ok": False, "error": "Query params 'from' and 'to' (node ids) required."}), 400
        try:
            p = find_path_between(
                a,
                b,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify({"ok": True, **p})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/network/node/add", methods=["POST"])
    def api_network_node_add():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "JSON 'name' required."}), 400
        nt = (data.get("node_type") or "person").strip().lower()
        desc = (data.get("description") or "").strip()
        rel = (data.get("relevance") or "MEDIUM").strip().upper()
        tags = data.get("tags")
        if tags is not None and not isinstance(tags, list):
            tags = [str(tags)]
        try:
            nid_raw = (data.get("id") or data.get("node_id") or "").strip() or None
            node = add_network_node(
                name,
                nt,
                desc,
                rel,
                tags,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                files_cabinet=angel.files_cabinet,
                use_mem0_cloud=angel._use_mem0_cloud,
                node_id_override=nid_raw,
            )
            return jsonify({"ok": True, "node": node})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/network/edge/add", methods=["POST"])
    def api_network_edge_add():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        src = (data.get("source_id") or data.get("source") or "").strip()
        tgt = (data.get("target_id") or data.get("target") or "").strip()
        if not src or not tgt:
            return jsonify({"ok": False, "error": "source_id and target_id required."}), 400
        try:
            edge = add_network_edge(
                src,
                tgt,
                data.get("relationship_type") or "connected_to",
                data.get("description") or "",
                data.get("strength") or "MODERATE",
                data.get("source_evidence") or data.get("evidence") or "",
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                files_cabinet=angel.files_cabinet,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "edge": edge})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/network/reset", methods=["GET"])
    def api_network_reset():
        """
        Destructive recovery: starts background wipe + re-seed (returns immediately).
        Poll GET /api/network/reset/status or /api/network/summary for completion.
        """
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        with _network_reset_lock:
            if _network_reset_status.get("in_progress"):
                return jsonify({
                    "ok": True,
                    "status": "reset_already_running",
                    "message": (
                        "A network reset is already in progress. "
                        "Poll /api/network/reset/status."
                    ),
                })
            _network_reset_status["in_progress"] = True
        try:
            threading.Thread(
                target=_network_reset_worker,
                args=(
                    angel.memory_client,
                    angel.user_id,
                    angel.files_cabinet,
                    angel._use_mem0_cloud,
                ),
                daemon=True,
            ).start()
        except Exception as e:
            traceback.print_exc()
            with _network_reset_lock:
                _network_reset_status["in_progress"] = False
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({
            "ok": True,
            "status": "reset_started",
            "message": (
                "Network reset running in background. "
                "Check /api/network/summary in 60 seconds."
            ),
        })

    @app.route("/api/network/reset/status", methods=["GET"])
    def api_network_reset_status():
        """State of the async network reset (in progress + last finished run)."""
        with _network_reset_lock:
            st = dict(_network_reset_status)
        return jsonify({
            "ok": True,
            "reset_in_progress": bool(st.get("in_progress")),
            "last_reset_completed_successfully": st.get("last_success"),
            "last_reset_nodes_after": st.get("last_nodes_after"),
            "last_reset_edges_after": st.get("last_edges_after"),
            "last_reset_at": st.get("last_reset_at"),
            "last_reset_error": st.get("last_error"),
        })

    # --- Item 15: Predictive modeling (forecasts + accuracy tracking) ---

    @app.route("/api/predictions", methods=["GET"])
    def api_predictions_active():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            rows = angel_predictions.get_active_predictions(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "predictions": rows, "count": len(rows)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/predictions/all", methods=["GET"])
    def api_predictions_all():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            by_id = angel_predictions.fetch_all_predictions(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
            )
            rows = sorted(
                by_id.values(),
                key=lambda p: (p.get("updated_at") or p.get("created_at") or ""),
                reverse=True,
            )
            return jsonify({"ok": True, "predictions": rows, "count": len(rows)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/predictions/accuracy", methods=["GET"])
    def api_predictions_accuracy():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            acc = angel_predictions.get_prediction_accuracy(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, **acc})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/predictions/generate", methods=["POST"])
    def api_predictions_generate():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        focus = (data.get("focus_topic") or data.get("topic") or "").strip() or None
        try:
            r = angel_predictions.generate_predictions(
                angel.anthropic_client,
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
                focus_topic=focus,
            )
            return jsonify({"ok": True, **r})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/predictions/resolve", methods=["POST"])
    def api_predictions_resolve():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        pid = (data.get("prediction_id") or data.get("id") or "").strip()
        outcome = (data.get("outcome") or data.get("outcome_notes") or "").strip()
        if not pid:
            return jsonify({"ok": False, "error": "prediction_id required"}), 400
        accurate = bool(data.get("accurate"))
        try:
            updated = angel_predictions.resolve_prediction(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
                prediction_id=pid,
                outcome=outcome or ("Confirmed" if accurate else "Denied"),
                accurate=accurate,
            )
            if not updated:
                return jsonify({"ok": False, "error": "prediction not found"}), 404
            return jsonify({"ok": True, "prediction": updated})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/predictions/check", methods=["GET"])
    def api_predictions_check():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            r = angel_predictions.check_predictions_against_reality(
                angel.anthropic_client,
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, **r})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Item 16: Proactive background intelligence ---

    @app.route("/api/proactive/watchlist", methods=["GET"])
    def api_proactive_watchlist():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            by_id = angel_proactive.fetch_all_watches(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
            )
            rows = sorted(by_id.values(), key=lambda x: (x.get("label") or "").lower())
            return jsonify({"ok": True, "watches": rows, "count": len(rows)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/proactive/watch/add", methods=["POST"])
    def api_proactive_watch_add():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        label = (data.get("label") or "").strip()
        if not label:
            return jsonify({"ok": False, "error": "label required"}), 400
        try:
            w = angel_proactive.add_watch_item(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
                label=label,
                watch_type=str(data.get("watch_type") or "topic"),
                priority=str(data.get("priority") or "MEDIUM"),
                check_frequency=str(data.get("check_frequency") or "weekly"),
                auto_added=False,
            )
            return jsonify({"ok": True, "watch": w})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/proactive/watch/<watch_id>", methods=["DELETE"])
    def api_proactive_watch_delete(watch_id):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            w = angel_proactive.deactivate_watch(
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
                watch_id,
            )
            if not w:
                return jsonify({"ok": False, "error": "watch not found"}), 404
            return jsonify({"ok": True, "watch": w})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/proactive/run", methods=["GET"])
    def api_proactive_run():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            r = angel_proactive.run_proactive_intelligence(
                angel.anthropic_client,
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, **r})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/proactive/findings", methods=["GET"])
    def api_proactive_findings():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            lim = request.args.get("limit", "30")
            try:
                n = max(1, min(80, int(lim)))
            except ValueError:
                n = 30
            rows = angel_proactive.fetch_recent_findings(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
                limit=n,
            )
            return jsonify({"ok": True, "findings": rows, "count": len(rows)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Item 18: Real-time translation & foreign intelligence ---

    @app.route("/api/translate", methods=["POST"])
    def api_translate():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        content = data.get("content") or ""
        context = (data.get("context") or "").strip()
        try:
            r = angel_translation.translate_and_analyze(
                angel.anthropic_client,
                content,
                context,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/translate/search", methods=["POST"])
    def api_translate_search():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        topic = (data.get("topic") or "").strip()
        if not topic:
            return jsonify({"ok": False, "error": "topic required"}), 400
        langs = data.get("languages")
        if langs is not None and not isinstance(langs, list):
            langs = None
        context = (data.get("context") or "").strip()
        try:
            r = angel_translation.search_foreign_sources_and_translate(
                angel.anthropic_client,
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
                topic=topic,
                languages=langs,
                context=context,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/translate/document", methods=["POST"])
    def api_translate_document():
        """Full foreign document: translate, analyze, file under Foreign Intelligence, expand network graph."""
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        content = data.get("content") or ""
        context = (data.get("context") or "").strip()
        source_label = (data.get("file_name") or data.get("source_label") or "document_upload").strip()
        try:
            r = angel_translation.translate_document_and_file(
                angel.anthropic_client,
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
                content,
                context,
                source_label=source_label[:200],
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/foreign/findings", methods=["GET"])
    def api_foreign_findings():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            lim = request.args.get("limit", "30")
            try:
                n = max(1, min(80, int(lim)))
            except ValueError:
                n = 30
            rows = angel_translation.fetch_recent_foreign_findings(
                angel.memory_client,
                angel.user_id,
                angel._use_mem0_cloud,
                limit=n,
            )
            return jsonify({"ok": True, "findings": rows, "count": len(rows)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Environmental map (Batcomputer geography layer) ---
    @app.route("/api/map/locations", methods=["GET"])
    def api_map_locations():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            lt = (request.args.get("location_type") or "").strip() or None
            sig = (request.args.get("significance") or "").strip() or None
            angel_environmental_map.ensure_seed_locations(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            locs = angel_environmental_map.list_locations(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
                location_type=lt,
                significance=sig,
            )
            return jsonify({"ok": True, "locations": locs})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/map/locations/<path:location_id>", methods=["GET"])
    def api_map_location_one(location_id):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_environmental_map.ensure_seed_locations(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            loc = angel_environmental_map.get_location(
                location_id,
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            if not loc:
                return jsonify({"ok": False, "error": "not found"}), 404
            return jsonify({"ok": True, "location": loc})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/map/summary", methods=["GET"])
    def api_map_summary():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_environmental_map.ensure_seed_locations(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            s = angel_environmental_map.get_location_summary(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "summary": s})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/map/region/<path:region>", methods=["GET"])
    def api_map_region(region):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_environmental_map.ensure_seed_locations(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            rows = angel_environmental_map.get_locations_by_region(
                region,
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "region": region, "locations": rows})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/map/location/add", methods=["POST"])
    def api_map_location_add():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        location_type = (data.get("location_type") or "").strip()
        region = (data.get("region") or "").strip()
        description = (data.get("description") or "").strip()
        significance = (data.get("significance") or "MEDIUM").strip()
        if not name or not location_type or not region or not description:
            return jsonify(
                {
                    "ok": False,
                    "error": "name, location_type, region, and description are required",
                }
            ), 400
        coords = data.get("coordinates")
        if coords is not None and not isinstance(coords, dict):
            coords = None
        ce = data.get("connected_entities")
        if ce is not None and not isinstance(ce, list):
            ce = []
        ki = data.get("known_incidents")
        if ki is not None and not isinstance(ki, list):
            ki = []
        tags = data.get("tags")
        if tags is not None and not isinstance(tags, list):
            tags = []
        lid = (data.get("location_id") or "").strip() or None
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            loc = angel_environmental_map.add_location(
                name,
                location_type,
                coords,
                region,
                description,
                significance,
                ce,
                ki,
                memory_client=angel.memory_client,
                user_id=user_id,
                files_cabinet=angel.files_cabinet,
                use_mem0_cloud=angel._use_mem0_cloud,
                location_id=lid,
                tags=tags,
                active_monitoring=bool(data.get("active_monitoring", True)),
            )
            return jsonify({"ok": True, "location": loc})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/map/research", methods=["POST"])
    def api_map_research():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        context = (data.get("context") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "name required"}), 400
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            r = angel_environmental_map.research_location(
                name,
                context,
                angel.anthropic_client,
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            code = 200 if r.get("ok") else 400
            return jsonify(r), code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/map/near", methods=["GET"])
    def api_map_near():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            lat_s = (request.args.get("lat") or "").strip()
            lon_s = (request.args.get("lon") or "").strip()
            rad_s = (request.args.get("radius") or "75").strip()
            if not lat_s or not lon_s:
                return jsonify({"ok": False, "error": "lat and lon required"}), 400
            lat = float(lat_s)
            lon = float(lon_s)
            radius = float(rad_s)
            angel_environmental_map.ensure_seed_locations(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            rows = angel_environmental_map.get_locations_near_coordinates(
                lat,
                lon,
                radius,
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify(
                {
                    "ok": True,
                    "lat": lat,
                    "lon": lon,
                    "radius_miles": radius,
                    "locations": rows,
                }
            )
        except ValueError:
            return jsonify({"ok": False, "error": "invalid lat, lon, or radius"}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Communication pattern analysis (cadence / coordination) ---
    @app.route("/api/comms/patterns", methods=["GET"])
    def api_comms_patterns():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            rows = angel_communication_patterns.list_entity_patterns(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "patterns": rows})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/comms/run", methods=["GET"])
    def api_comms_run():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            r = angel_communication_patterns.run_scheduled_analysis(
                angel.anthropic_client,
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            code = 200 if r.get("ok") else 400
            return jsonify(r), code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/comms/entity/<path:name>", methods=["GET"])
    def api_comms_entity(name):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            p = angel_communication_patterns.get_pattern_for_entity_name(
                name,
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            if not p:
                return jsonify({"ok": False, "error": "not found"}), 404
            return jsonify({"ok": True, "pattern": p})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/comms/anomalies", methods=["GET"])
    def api_comms_anomalies():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            rows = angel_communication_patterns.list_current_anomalies(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "anomalies": rows})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/comms/coordination", methods=["GET"])
    def api_comms_coordination():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            rows = angel_communication_patterns.list_coordination_signals(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "coordination_signals": rows})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Biological / medical intelligence ---
    @app.route("/api/bio/analyze", methods=["POST"])
    def api_bio_analyze():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        content = (data.get("content") or "").strip()
        context = (data.get("context") or "").strip()
        source = (data.get("source") or "").strip()
        if not content:
            return jsonify({"ok": False, "error": "content required"}), 400
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            r = angel_biological_intelligence.analyze_biological_report(
                content,
                context,
                source,
                angel.anthropic_client,
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            code = 200 if r.get("ok") else 400
            return jsonify(r), code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/bio/cases", methods=["GET"])
    def api_bio_cases():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_biological_intelligence.ensure_seed_bio_cases(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            rows = angel_biological_intelligence.list_known_cases(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "cases": rows})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/bio/patterns", methods=["GET"])
    def api_bio_patterns():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_biological_intelligence.ensure_seed_bio_cases(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            agg = angel_biological_intelligence.aggregate_medical_patterns(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "patterns": agg})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/bio/case/add", methods=["POST"])
    def api_bio_case_add():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "JSON body required"}), 400
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            r = angel_biological_intelligence.add_bio_case(
                data,
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            code = 200 if r.get("ok") else 400
            return jsonify(r), code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/bio/profile/black-eyed", methods=["GET"])
    def api_bio_profile_black_eyed():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            angel_biological_intelligence.ensure_seed_bio_cases(
                angel.memory_client,
                os.getenv("ANGEL_USER_ID", "railway-user"),
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            p = angel_biological_intelligence.get_black_eyed_profile()
            return jsonify({"ok": True, "profile": p})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Chemistry & materials intelligence (PubChem, NIST, Materials Project) ---
    @app.route("/api/chemistry/status", methods=["GET"])
    def api_chemistry_status():
        try:
            return jsonify({"ok": True, **angel_chemistry.chemistry_status()})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/chemistry/compound", methods=["POST"])
    def api_chemistry_compound():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        q = (data.get("query") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not q:
            return jsonify({"ok": False, "error": "query required"}), 400
        try:
            r = angel_chemistry.compound_api_payload(
                q,
                ctx,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/chemistry/material", methods=["POST"])
    def api_chemistry_material():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        q = (data.get("query") or "").strip()
        use_case = (data.get("use_case") or "").strip()
        if not q:
            return jsonify({"ok": False, "error": "query required"}), 400
        try:
            r = angel_chemistry.material_api_payload(
                q,
                use_case,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/chemistry/synthesis", methods=["POST"])
    def api_chemistry_synthesis():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        target = (data.get("target") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not target:
            return jsonify({"ok": False, "error": "target required"}), 400
        try:
            r = angel_chemistry.analyze_synthesis_route(
                target,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/chemistry/design", methods=["POST"])
    def api_chemistry_design():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        req = data.get("requirements")
        if not isinstance(req, dict):
            return jsonify({"ok": False, "error": "requirements object required"}), 400
        try:
            r = angel_chemistry.design_material_for_requirements(
                req,
                anthropic_client=angel.anthropic_client,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Medical Intelligence Core (PubMed, openFDA, MedlinePlus, ClinicalTrials.gov) ---
    @app.route("/api/medical/status", methods=["GET"])
    def api_medical_status():
        try:
            return jsonify({"ok": True, **angel_medical.medical_databases_status()})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/condition", methods=["POST"])
    def api_medical_condition():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        cond = (data.get("condition") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not cond:
            return jsonify({"ok": False, "error": "condition required"}), 400
        try:
            r = angel_medical.analyze_condition(
                cond,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/drug", methods=["POST"])
    def api_medical_drug():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        drug = (data.get("drug") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not drug:
            return jsonify({"ok": False, "error": "drug required"}), 400
        try:
            r = angel_medical.analyze_drug(
                drug,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/literature", methods=["POST"])
    def api_medical_literature():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        q = (data.get("query") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not q:
            return jsonify({"ok": False, "error": "query required"}), 400
        try:
            r = angel_medical.search_medical_literature(
                q,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/trials", methods=["POST"])
    def api_medical_trials():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        cond = (data.get("condition") or "").strip()
        intr = (data.get("intervention") or "").strip() or None
        if not cond:
            return jsonify({"ok": False, "error": "condition required"}), 400
        try:
            r = angel_medical.search_trials(
                cond,
                intervention=intr,
                status="RECRUITING",
                max_results=15,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, **r})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/biological-threat", methods=["POST"])
    def api_medical_biological_threat():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        agent = (data.get("agent") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not agent:
            return jsonify({"ok": False, "error": "agent required"}), 400
        try:
            r = angel_medical.research_biological_agent(
                agent,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/biomedical-research", methods=["POST"])
    def api_medical_biomedical_research():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        target = (data.get("target") or "").strip()
        ttype = (data.get("target_type") or "condition").strip()
        ctx = (data.get("context") or "").strip()
        if not target:
            return jsonify({"ok": False, "error": "target required"}), 400
        try:
            r = angel_medical.run_biomedical_research(
                target,
                ttype,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/gene", methods=["POST"])
    def api_medical_gene():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        gene = (data.get("gene") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not gene:
            return jsonify({"ok": False, "error": "gene required"}), 400
        try:
            r = angel_medical.run_biomedical_research(
                gene,
                "gene",
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/protein", methods=["POST"])
    def api_medical_protein():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        protein = (data.get("protein") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not protein:
            return jsonify({"ok": False, "error": "protein required"}), 400
        try:
            r = angel_medical.run_biomedical_research(
                protein,
                "protein",
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/drug-target", methods=["POST"])
    def api_medical_drug_target():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        drug = (data.get("drug") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not drug:
            return jsonify({"ok": False, "error": "drug required"}), 400
        try:
            r = angel_medical.research_drug_target(
                drug,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/biological-agent", methods=["POST"])
    def api_medical_biological_agent():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        agent = (data.get("agent") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not agent:
            return jsonify({"ok": False, "error": "agent required"}), 400
        try:
            r = angel_medical.research_biological_agent(
                agent,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/design-treatment", methods=["POST"])
    def api_medical_design_treatment():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        cond = (data.get("condition") or "").strip()
        ctx = (data.get("context") or "").strip()
        cons = data.get("constraints") if isinstance(data.get("constraints"), dict) else {}
        if not cond:
            return jsonify({"ok": False, "error": "condition required"}), 400
        try:
            r = angel_medical.design_theoretical_treatment(
                cond,
                cons,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/optimize-combination", methods=["POST"])
    def api_medical_optimize_combination():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        raw = data.get("compounds")
        comps: list[str] = []
        if isinstance(raw, list):
            comps = [str(x).strip() for x in raw if str(x).strip()]
        ctx = (data.get("context") or "").strip()
        cond = (data.get("condition") or "").strip()
        if len(comps) < 2:
            return jsonify({"ok": False, "error": "compounds array with at least two names required"}), 400
        try:
            r = angel_medical.optimize_combination(
                comps,
                cond,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/repurposing", methods=["POST"])
    def api_medical_repurposing():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        cond = (data.get("condition") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not cond:
            return jsonify({"ok": False, "error": "condition required"}), 400
        try:
            r = angel_medical.research_repurposing_opportunities(
                cond,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/exotic-treatments", methods=["POST"])
    def api_medical_exotic_treatments():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        cond = (data.get("condition") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not cond:
            return jsonify({"ok": False, "error": "condition required"}), 400
        try:
            r = angel_medical.research_exotic_treatments(
                cond,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/uap-medical", methods=["POST"])
    def api_medical_uap_medical():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        prof = (data.get("symptom_profile") or "").strip()
        ctx = (data.get("context") or "").strip()
        if len(prof) < 8:
            return jsonify({"ok": False, "error": "symptom_profile required (min length)"}), 400
        try:
            r = angel_medical.research_uap_medical_effects(
                prof,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/medical/health-profile", methods=["POST"])
    def api_medical_health_profile():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        hd = data.get("health_data")
        if not isinstance(hd, dict):
            upd = data.get("updates")
            hd = upd if isinstance(upd, dict) else None
        if not isinstance(hd, dict):
            return jsonify({"ok": False, "error": "health_data or updates object required"}), 400
        try:
            r = angel_medical.update_health_profile(
                hd,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/health/profile", methods=["GET"])
    def api_health_profile():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            r = angel_medical.get_health_profile(
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/health/update", methods=["POST"])
    def api_health_update():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        upd = data.get("updates")
        if not isinstance(upd, dict):
            return jsonify({"ok": False, "error": "updates object required"}), 400
        try:
            r = angel_medical.update_health_profile(
                upd,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/health/assess", methods=["POST"])
    def api_health_assess():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        ctx = (data.get("context") or "").strip()
        try:
            r = angel_medical.get_personalized_health_assessment(
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/health/interactions", methods=["POST"])
    def api_health_interactions():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        meds = data.get("medications") or []
        sups = data.get("supplements") or []
        ctx = (data.get("context") or "").strip()
        if not isinstance(meds, list):
            meds = []
        if not isinstance(sups, list):
            sups = []
        try:
            r = angel_medical.check_drug_interactions(
                meds,
                sups,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/health/recommendations", methods=["POST"])
    def api_health_recommendations():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        goal = (data.get("goal") or "").strip()
        ctx = (data.get("context") or "").strip()
        try:
            r = angel_medical.get_personalized_recommendations(
                goal,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/health/status", methods=["GET"])
    def api_health_status():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            r = angel_medical.get_profile_completeness(
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/health/wearable", methods=["POST"])
    def api_health_wearable():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        device = (data.get("device") or "").strip()
        metrics = data.get("metrics")
        if not device:
            return jsonify({"ok": False, "error": "device required"}), 400
        if not isinstance(metrics, dict):
            return jsonify({"ok": False, "error": "metrics object required"}), 400
        try:
            r = angel_medical.update_wearable_data(
                device,
                metrics,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Theoretical research agent (ArXiv, NASA NTRS, DARPA/DTIC, patents) ---
    @app.route("/api/research/query", methods=["POST"])
    def api_research_query():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        context = (data.get("context") or "").strip()
        rtypes = data.get("research_types")
        if not query:
            return jsonify({"ok": False, "error": "query required"}), 400
        if rtypes is not None and not isinstance(rtypes, list):
            rtypes = None
        try:
            out = angel_research.run_research_agent(
                query,
                context or query,
                rtypes,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify({"ok": True, **out})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/research/paper", methods=["POST"])
    def api_research_paper():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        aid = (data.get("arxiv_id") or "").strip()
        if not aid:
            return jsonify({"ok": False, "error": "arxiv_id required"}), 400
        try:
            r = angel_research.get_arxiv_paper(
                aid,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify({"ok": bool(r.get("ok")), **r})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/research/patent", methods=["POST"])
    def api_research_patent():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        pnum = (data.get("patent_number") or "").strip()
        pq = (data.get("query") or "").strip()
        try:
            if pnum:
                r = angel_research.get_patent_detail(pnum)
                return jsonify({"ok": bool(r.get("ok")), **r})
            if pq:
                r = angel_research.search_patents(pq, max_results=10)
                return jsonify({"ok": bool(r.get("ok")), **r})
            return jsonify({"ok": False, "error": "patent_number or query required"}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/research/status", methods=["GET"])
    def api_research_status():
        try:
            return jsonify({"ok": True, **angel_research.research_status()})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Physics simulation engine ---
    @app.route("/api/physics/simulate", methods=["POST"])
    def api_physics_simulate():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        st = (data.get("simulation_type") or "").strip()
        params = data.get("params") if isinstance(data.get("params"), dict) else {}
        ctx = (data.get("context") or "").strip()
        if not st:
            return jsonify({"ok": False, "error": "simulation_type required"}), 400
        try:
            out = angel_physics.run_physics_simulation(
                st,
                params,
                ctx,
                anthropic_client=angel.anthropic_client,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify({"ok": True, **out})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/physics/extract-params", methods=["POST"])
    def api_physics_extract_params():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        msg = (data.get("message") or "").strip()
        if not msg:
            return jsonify({"ok": False, "error": "message required"}), 400
        try:
            out = angel_physics.extract_simulation_params(msg, angel.anthropic_client)
            return jsonify({"ok": True, **out})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/physics/natural", methods=["POST"])
    def api_physics_natural():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        msg = (data.get("message") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not msg:
            return jsonify({"ok": False, "error": "message required"}), 400
        try:
            out = angel_physics.run_physics_natural(
                msg,
                ctx or msg,
                anthropic_client=angel.anthropic_client,
                files_cabinet=angel.files_cabinet,
            )
            code = 200 if out.get("ok") else 400
            return jsonify(out), code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/physics/status", methods=["GET"])
    def api_physics_status():
        try:
            return jsonify({"ok": True, **angel_physics.physics_library_status()})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- CAD generation (cadquery primary; FreeCAD optional for primitives) ---
    @app.route("/api/cad/generate", methods=["POST"])
    def api_cad_generate():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        shape = (data.get("shape") or "").strip().lower()
        params = data.get("params") if isinstance(data.get("params"), dict) else {}
        ctx = (data.get("context") or "").strip()
        dn = (data.get("design_name") or "").strip() or f"cad_{datetime.now(timezone.utc).strftime('%H%M%S')}"
        if not shape:
            return jsonify({"ok": False, "error": "shape required"}), 400
        try:
            safe_dn = re.sub(r"[^a-zA-Z0-9._-]+", "_", dn)[:80] or "cad_design"
            gen = angel_cad.generate_shape(
                shape,
                params,
                session_id=angel.user_id,
                design_name=safe_dn,
                context=ctx,
            )
            if not gen.get("ok"):
                return jsonify({"ok": False, **gen}), 400
            rationale = {"note": "direct API generate", "context": ctx}
            brief_path = Path(gen["design_dir"]) / f"{safe_dn}_design_brief.json"
            try:
                brief_path.write_text(
                    json.dumps({"shape": shape, "params": params, "rationale": rationale}, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            out = {
                "ok": True,
                "design_name": safe_dn,
                "shape": shape,
                "files_generated": {k: v for k, v in (gen.get("paths") or {}).items() if v},
                "design_rationale": json.dumps(rationale),
                "backend": gen.get("backend"),
                "design_dir": gen.get("design_dir"),
            }
            base = str(request.host_url).rstrip("/")
            return jsonify(angel_cad.enrich_with_download_urls(out, base))
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/cad/from-brief", methods=["POST"])
    def api_cad_from_brief():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        brief = data.get("brief")
        if brief is None and isinstance(data.get("design_brief"), str):
            brief = data.get("design_brief")
        ctx = (data.get("context") or "").strip()
        phys = data.get("physics_constraints") if isinstance(data.get("physics_constraints"), dict) else {}
        if isinstance(brief, dict):
            pass
        elif not (isinstance(brief, str) and brief.strip()):
            return jsonify({"ok": False, "error": "brief (string or object) required"}), 400
        try:
            out = angel_cad.generate_from_brief(
                brief if isinstance(brief, (str, dict)) else str(brief),
                ctx,
                session_id=angel.user_id,
                anthropic_client=angel.anthropic_client,
                physics_constraints=phys,
                design_name=(data.get("design_name") or "").strip() or None,
                files_cabinet=angel.files_cabinet,
            )
            base = str(request.host_url).rstrip("/")
            code = 200 if out.get("ok") else 400
            return jsonify(angel_cad.enrich_with_download_urls(out, base)), code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/cad/download/<design_name>/<filename>", methods=["GET"])
    def api_cad_download(design_name, filename):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        p = angel_cad.resolve_download_path(angel.user_id, design_name, filename)
        if p is None:
            return jsonify({"ok": False, "error": "file not found"}), 404
        return send_file(str(p), as_attachment=True, download_name=filename)

    @app.route("/api/cad/mesh/<design_name>/<filename>", methods=["POST"])
    def api_cad_mesh_convert(design_name, filename):
        """Convert an existing STL for a design into mesh JSON (one-off or cache refresh)."""
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            out = angel_cad.convert_stl_filename_to_mesh_json(angel.user_id, design_name, filename)
            code = 200 if out.get("ok") else 400
            return jsonify(out), code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e), "design_name": design_name}), 500

    @app.route("/api/cad/mesh-json/<design_name>", methods=["GET"])
    def api_cad_mesh_json(design_name):
        """Return cached mesh JSON for a design (generate once on demand)."""
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            out = angel_cad.get_or_create_mesh_json(angel.user_id, design_name)
            code = 200 if out.get("ok") else 400
            return jsonify(out), code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e), "design_name": design_name}), 500

    @app.route("/api/cad/thumbnail/<design_name>", methods=["GET"])
    def api_cad_thumbnail(design_name):
        """Generate (and cache) a simple 3-view STL thumbnail PNG for a design."""
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            png, err = angel_cad.get_or_create_thumbnail_png_bytes(angel.user_id, design_name)
            if not png:
                return jsonify({"ok": False, "error": err or "thumbnail failed", "design_name": design_name}), 400
            return send_file(
                BytesIO(png),
                mimetype="image/png",
                as_attachment=False,
                download_name=f"{design_name}.png",
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e), "design_name": design_name}), 500

    @app.route("/api/cad/list", methods=["GET"])
    def api_cad_list():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            rows = angel_cad.list_designs(angel.user_id)
            return jsonify({"ok": True, "designs": rows})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/cad/status", methods=["GET"])
    def api_cad_status():
        try:
            return jsonify({"ok": True, **angel_cad.get_cad_status()})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Theoretical Suit Design Targets (Iron Man / Batman Beyond) ---
    @app.route("/api/ironman/status", methods=["GET"])
    def api_ironman_status():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            rows = []
            for dom in angel_ironman.ALL_DOMAINS:
                tgt = angel_ironman.get_domain_target(
                    dom,
                    memory_client=angel.memory_client,
                    user_id=angel.user_id,
                    use_mem0_cloud=angel._use_mem0_cloud,
                ).get("target") or {}
                cb = tgt.get("current_best") if isinstance(tgt.get("current_best"), dict) else {}
                rows.append(
                    {
                        "domain": dom,
                        "design_philosophy": tgt.get("design_philosophy"),
                        "target_name": tgt.get("target_name"),
                        "trl": cb.get("trl"),
                        "gap_ratio": cb.get("gap_ratio"),
                        "mission_relevance": tgt.get("mission_relevance"),
                        "last_researched": tgt.get("last_researched"),
                    }
                )
            return jsonify({"ok": True, "domains": rows})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/ironman/domain", methods=["POST"])
    def api_ironman_domain():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        dom = (data.get("domain") or "").strip()
        ctx = (data.get("context") or "").strip()
        if not dom:
            return jsonify({"ok": False, "error": "domain required"}), 400
        try:
            r = angel_ironman.research_domain(
                dom,
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/ironman/assess", methods=["POST"])
    def api_ironman_assess():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        design = (data.get("design") or "both").strip().lower()
        ctx = (data.get("context") or "").strip()
        try:
            if design == "iron_man":
                r = angel_ironman.run_full_ironman_assessment(
                    ctx,
                    anthropic_client=angel.anthropic_client,
                    memory_client=angel.memory_client,
                    user_id=angel.user_id,
                    use_mem0_cloud=angel._use_mem0_cloud,
                    files_cabinet=angel.files_cabinet,
                )
                return jsonify(r)
            if design == "batman_beyond":
                r = angel_ironman.run_full_batman_beyond_assessment(
                    ctx,
                    anthropic_client=angel.anthropic_client,
                    memory_client=angel.memory_client,
                    user_id=angel.user_id,
                    use_mem0_cloud=angel._use_mem0_cloud,
                    files_cabinet=angel.files_cabinet,
                )
                return jsonify(r)
            r = angel_ironman.run_full_suit_assessment(
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
                files_cabinet=angel.files_cabinet,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/ironman/convergence", methods=["POST"])
    def api_ironman_convergence():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        ctx = (data.get("context") or "").strip()
        try:
            r = angel_ironman.analyze_suit_convergence(
                ctx,
                anthropic_client=angel.anthropic_client,
                memory_client=angel.memory_client,
                user_id=angel.user_id,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(r)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/ironman/cad", methods=["POST"])
    def api_ironman_cad():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        design = (data.get("design") or "").strip().lower()
        dom = (data.get("domain") or "").strip()
        specs = data.get("specs") or {}
        if design not in ("iron_man", "batman_beyond"):
            return jsonify({"ok": False, "error": "design must be iron_man or batman_beyond"}), 400
        if not dom:
            return jsonify({"ok": False, "error": "domain required"}), 400
        if not isinstance(specs, dict):
            return jsonify({"ok": False, "error": "specs object required"}), 400
        try:
            r = angel_ironman.generate_suit_cad_component(
                design, dom, specs, user_id=angel.user_id
            )
            return jsonify({"ok": True, "design": design, "domain": dom, "cad": r})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Capability Graph ---
    @app.route("/api/capabilities/graph", methods=["GET"])
    def api_capabilities_graph():
        try:
            return jsonify(angel_capability_graph.get_graph_as_json())
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/capabilities/chain", methods=["POST"])
    def api_capabilities_chain():
        data = request.get_json(silent=True) or {}
        cap = (data.get("capability") or "").strip()
        if not cap:
            return jsonify({"ok": False, "error": "capability required"}), 400
        try:
            return jsonify(angel_capability_graph.get_capability_chain(cap))
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/capabilities/analyze", methods=["POST"])
    def api_capabilities_analyze():
        global angel
        data = request.get_json(silent=True) or {}
        msg = (data.get("message") or "").strip()
        ctx = (data.get("context") or "").strip()
        text = (msg + "\n\n" + ctx).strip()
        if not text:
            return jsonify({"ok": False, "error": "message or context required"}), 400
        try:
            if angel is not None:
                out = angel_capability_graph.find_optimal_combination(
                    text, anthropic_client=angel.anthropic_client
                )
            else:
                out = angel_capability_graph.find_optimal_combination(text, anthropic_client=None)
            rec = angel_capability_graph.recognize_capability_combinations(text)
            sug = angel_capability_graph.suggest_capability_combinations(text)
            return jsonify({"ok": True, "recognition": rec, "optimal": out, "suggestion": sug})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/capabilities/combinations", methods=["GET"])
    def api_capabilities_combinations():
        try:
            return jsonify({"ok": True, "combinations": angel_capability_graph.KNOWN_HIGH_VALUE_COMBINATIONS})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    # --- Historical Intelligence Archives ---
    @app.route("/api/archives/timeline", methods=["GET"])
    def api_archives_timeline():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_historical_archives.ensure_seed_historical_records(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            tl = angel_historical_archives.get_timeline(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "timeline": tl})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/archives/summary", methods=["GET"])
    def api_archives_summary():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_historical_archives.ensure_seed_historical_records(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            s = angel_historical_archives.get_archive_summary(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "summary": s})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/archives/search", methods=["GET"])
    def api_archives_search():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        q = (request.args.get("q") or "").strip()
        if not q:
            return jsonify({"ok": False, "error": "q required"}), 400
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_historical_archives.ensure_seed_historical_records(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            rows = angel_historical_archives.search_archives(
                q,
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "query": q, "records": rows})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/archives/person/<path:name>", methods=["GET"])
    def api_archives_person(name):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_historical_archives.ensure_seed_historical_records(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            rows = angel_historical_archives.get_records_by_person(
                name,
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            return jsonify({"ok": True, "person": name, "records": rows})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/archives/add", methods=["POST"])
    def api_archives_add():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "JSON body required"}), 400
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            rec = angel_historical_archives.add_historical_record(
                (data.get("title") or "").strip(),
                (data.get("record_type") or "incident").strip(),
                (data.get("date") or "").strip(),
                (data.get("location") or "").strip(),
                (data.get("description") or "").strip(),
                (data.get("significance") or "MEDIUM").strip(),
                data.get("connected_people") if isinstance(data.get("connected_people"), list) else [],
                (data.get("evidence_quality") or "documented").strip(),
                (data.get("current_relevance") or "").strip(),
                data.get("sources") if isinstance(data.get("sources"), list) else [],
                data.get("tags") if isinstance(data.get("tags"), list) else [],
                memory_client=angel.memory_client,
                user_id=user_id,
                files_cabinet=angel.files_cabinet,
                use_mem0_cloud=angel._use_mem0_cloud,
                record_id=(data.get("record_id") or "").strip() or None,
                date_precision=(data.get("date_precision") or "approximate").strip(),
                connected_programs=data.get("connected_programs")
                if isinstance(data.get("connected_programs"), list)
                else None,
                connected_locations=data.get("connected_locations")
                if isinstance(data.get("connected_locations"), list)
                else None,
            )
            return jsonify({"ok": True, "record": rec})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/archives/research", methods=["POST"])
    def api_archives_research():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        context = (data.get("context") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "name required"}), 400
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            r = angel_historical_archives.research_historical_event(
                name,
                context,
                angel.anthropic_client,
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            code = 200 if r.get("ok") else 400
            return jsonify(r), code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/archives/<path:record_id>", methods=["GET"])
    def api_archives_one(record_id):
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_historical_archives.ensure_seed_historical_records(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            rec = angel_historical_archives.get_record(
                record_id,
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            if not rec:
                return jsonify({"ok": False, "error": "not found"}), 404
            return jsonify({"ok": True, "record": rec})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/archives", methods=["GET"])
    def api_archives_list():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            angel_historical_archives.ensure_seed_historical_records(
                angel.memory_client,
                user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            rt = (request.args.get("type") or "").strip() or None
            sig = (request.args.get("significance") or "").strip().upper() or None
            sy = (request.args.get("start_year") or "").strip()
            ey = (request.args.get("end_year") or "").strip()
            rows = angel_historical_archives.list_all_records(
                angel.memory_client,
                user_id,
                angel._use_mem0_cloud,
            )
            if rt:
                rows = [r for r in rows if (r.get("record_type") or "").lower() == rt.lower()]
            if sig:
                rows = [r for r in rows if (r.get("significance") or "").upper() == sig]
            if sy and ey:
                try:
                    syy = int(sy)
                    eyy = int(ey)
                    rows = [
                        r
                        for r in rows
                        if (y := angel_historical_archives._extract_year(r.get("date") or ""))
                        is not None
                        and syy <= y <= eyy
                    ]
                except ValueError:
                    pass
            try:
                rows.sort(key=angel_historical_archives._sort_key_date)
            except Exception:
                pass
            return jsonify({"ok": True, "records": rows, "count": len(rows)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/briefing", methods=["GET"])
    def api_briefing():
        global morning_briefing, briefing_generated_at
        return jsonify({
            "briefing": morning_briefing,
            "generated_at": briefing_generated_at,
        })

    @app.route("/api/trigger_briefing", methods=["GET"])
    def api_trigger_briefing():
        """Testing only: run the morning briefing job immediately (generate, store, send email)."""
        global morning_briefing, briefing_generated_at
        try:
            _run_morning_briefing_job()
            return jsonify({
                "status": "ok",
                "briefing": morning_briefing,
                "generated_at": briefing_generated_at,
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/register_push_token", methods=["POST"])
    def api_register_push_token():
        """Register an Expo push token (JSON body: token or expoPushToken)."""
        data = request.get_json(silent=True) or {}
        token = (
            data.get("token")
            or data.get("expoPushToken")
            or data.get("push_token")
            or ""
        )
        token = str(token).strip()
        if not token:
            return jsonify({"status": "error", "error": "missing token"}), 400
        is_new = register_expo_push_token(token)
        return jsonify(
            {
                "status": "ok",
                "registered": True,
                "new_token": is_new,
                "total_tokens": len(expo_push_tokens),
            }
        )

    @app.route("/api/test_push", methods=["GET"])
    def api_test_push():
        """Send a test push immediately to all registered Expo tokens."""
        if not expo_push_tokens:
            return jsonify(
                {
                    "status": "error",
                    "error": "no push tokens registered",
                    "hint": "POST /api/register_push_token first",
                }
            ), 400
        result = send_expo_push_notifications(
            "Angel test",
            "Push notifications are working.",
        )
        code = 200 if result.get("ok") else 502
        return jsonify({"status": "ok" if result.get("ok") else "error", **result}), code

    @app.route("/api/reflect", methods=["GET"])
    def api_reflect():
        """Manual trigger: run memory reflection now and store as category 'reflection'."""
        try:
            user_id = os.getenv("ANGEL_USER_ID", "railway-user")
            client = create_anthropic_client()
            text = run_memory_reflection(
                angel.memory_client,
                user_id,
                client,
                use_mem0_cloud=angel._use_mem0_cloud,
            )
            return jsonify(
                {
                    "status": "ok",
                    "reflection": _sanitize_text(text),
                    "chars": len(text),
                }
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/selfmod/observations", methods=["GET"])
    def api_selfmod_observations():
        """Recent self_observation entries (Stage 6)."""
        global angel
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            import angel_self_modification as asm

            limit = int(request.args.get("limit", 50))
            obs = asm.api_list_observations(
                angel.memory_client, angel.user_id, angel._use_mem0_cloud, limit=limit
            )
            return jsonify({"status": "ok", "observations": obs})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/selfmod/proposals", methods=["GET"])
    def api_selfmod_proposals():
        """All self_modification records (latest per id)."""
        global angel
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            import angel_self_modification as asm

            props = asm.api_list_proposals(
                angel.memory_client, angel.user_id, angel._use_mem0_cloud
            )
            return jsonify({"status": "ok", "proposals": props})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/selfmod/applied", methods=["GET"])
    def api_selfmod_applied():
        """Tyler-approved modifications applied via angel_self_mods_data.json."""
        try:
            from angel_self_mods import list_applied_records

            return jsonify({"status": "ok", "applied": list_applied_records()})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/selfmod/approve", methods=["POST"])
    def api_selfmod_approve():
        global angel
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        mid = (data.get("modification_id") or "").strip()
        if not mid:
            return jsonify({"error": "missing modification_id"}), 400
        try:
            import angel_self_modification as asm

            text = asm.handle_self_mod_intent(angel, "approve", mid, f"approve {mid}")
            return jsonify({"status": "ok", "message": _sanitize_text(text)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/selfmod/reject", methods=["POST"])
    def api_selfmod_reject():
        global angel
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        mid = (data.get("modification_id") or "").strip()
        if not mid:
            return jsonify({"error": "missing modification_id"}), 400
        try:
            import angel_self_modification as asm

            text = asm.handle_self_mod_intent(angel, "reject", mid, f"reject {mid}")
            return jsonify({"status": "ok", "message": _sanitize_text(text)})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/selfmod/revert", methods=["POST"])
    def api_selfmod_revert():
        """Revert an applied modification (same logic as chat: revert [id or title])."""
        global angel
        if angel is None:
            return jsonify({"ok": False, "error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        mid = (data.get("modification_id") or "").strip()
        if not mid:
            return jsonify({"ok": False, "error": "missing modification_id"}), 400
        try:
            import angel_self_modification as asm

            r = asm.revert_self_modification(angel, mid)
            if r.get("ok"):
                return jsonify(
                    {
                        "ok": True,
                        "modification_id": r.get("modification_id"),
                        "message": _sanitize_text(str(r.get("message") or "")),
                    }
                )
            return jsonify({"ok": False, "error": _sanitize_text(str(r.get("error") or ""))}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": _sanitize_text(str(e))}), 500

    @app.route("/api/selfmod/generate", methods=["POST"])
    def api_selfmod_generate():
        global angel
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        try:
            import angel_self_modification as asm

            r = asm.run_self_modification_analysis(
                angel.anthropic_client,
                angel.memory_client,
                angel.user_id,
                angel.files_cabinet,
                angel._use_mem0_cloud,
            )
            return jsonify({"status": "ok" if r.get("ok") else "error", **r})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/agents/run", methods=["POST"])
    def api_agents_run():
        """Run up to 5 parallel specialist agents + coordinator synthesis."""
        global angel
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        tasks_raw = data.get("tasks")
        context = (data.get("context") or "").strip()
        depth = (data.get("depth") or "standard").strip().lower()
        if depth not in ("standard", "deep"):
            depth = "standard"
        if not isinstance(tasks_raw, list) or not tasks_raw:
            return jsonify({"error": "tasks (non-empty list) required"}), 400
        try:
            import angel_parallel_agents as apa

            r = apa.run_parallel_agents(
                tasks_raw[:5],
                context or "Manual parallel run",
                anthropic_client=angel.anthropic_client,
                memory_summary=None,
                memory_client=angel.memory_client,
                use_mem0_cloud=angel._use_mem0_cloud,
                user_id=angel.user_id,
                depth=depth,
                trusted_operator=is_trusted_operator(angel.user_id),
            )
            out = dict(r)
            if out.get("synthesis"):
                out["synthesis"] = _sanitize_text(str(out["synthesis"]))
            return jsonify({"status": "ok", **out})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/agents/research", methods=["POST"])
    def api_agents_research():
        """Decompose topic into parallel tasks and run."""
        global angel
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
        data = request.get_json(silent=True) or {}
        topic = (data.get("topic") or "").strip()
        depth = (data.get("depth") or "standard").strip().lower()
        if depth not in ("standard", "deep"):
            depth = "standard"
        if not topic:
            return jsonify({"error": "topic required"}), 400
        try:
            import angel_parallel_agents as apa

            r = apa.run_research_decomposed(
                topic,
                depth=depth,
                anthropic_client=angel.anthropic_client,
                user_id=angel.user_id,
                user_message=topic,
                memory_summary=None,
                memory_client=angel.memory_client,
                use_mem0_cloud=angel._use_mem0_cloud,
                trusted_operator=is_trusted_operator(angel.user_id),
            )
            out = dict(r)
            if out.get("synthesis"):
                out["synthesis"] = _sanitize_text(str(out["synthesis"]))
            return jsonify({"status": "ok", **out})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/agents/status/<path:task_id>", methods=["GET"])
    def api_agents_status(task_id: str):
        try:
            import angel_parallel_agents as apa

            st = apa.get_task_status(task_id)
            if st is None:
                return jsonify({"error": "not found"}), 404
            return jsonify({"status": "ok", "task": st})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/api/check_in", methods=["GET"])
    def api_check_in():
        global check_in_message, check_in_generated_at
        return jsonify({
            "message": check_in_message,
            "generated_at": check_in_generated_at,
        })

    @app.route("/api/status", methods=["GET"])
    def api_status():
        tz_env = os.getenv("TIMEZONE", "America/Los_Angeles")
        try:
            try:
                from zoneinfo import ZoneInfo

                tzinfo = ZoneInfo(tz_env)
            except Exception:
                tzinfo = None

            now = datetime.now(tzinfo) if tzinfo is not None else datetime.now(UTC)
            current_time = now.isoformat()
        except Exception:
            current_time = None

        tyler_email = os.getenv("TYLER_EMAIL")
        gmail_pass = os.getenv("GMAIL_APP_PASSWORD")

        return jsonify(
            {
                "TYLER_EMAIL_set": bool(tyler_email),
                "GMAIL_APP_PASSWORD_set": bool(gmail_pass),
                "MEM0_API_KEY_set": bool(os.getenv("MEM0_API_KEY")),
                "TAVILY_API_KEY_set": bool(os.getenv("TAVILY_API_KEY")),
                "timezone": tz_env,
                "current_time": current_time,
            }
        )

    @app.route("/api/tts", methods=["POST"])
    def api_tts():
        try:
            data = request.get_json(silent=True) or {}
            text = (data.get("text") or "").strip()

            api_key_present = bool(os.getenv("OPENAI_API_KEY"))

            if not text:
                return jsonify({"error": "Missing or empty text"}), 400

            if not api_key_present:
                return jsonify(
                    {
                        "error": "OPENAI_API_KEY is not set in the environment; TTS is unavailable."
                    }
                ), 503

            mp3_bytes = tts_gpt4o(text, voice="alloy")
            if not mp3_bytes:
                return jsonify(
                    {
                        "error": (
                            "TTS generation failed or returned empty audio. "
                            "Check server logs for details."
                        )
                    }
                ), 503

            resp = Response(mp3_bytes, mimetype="audio/mpeg")
            resp.headers["Content-Type"] = "audio/mpeg"
            resp.headers["Cache-Control"] = "no-store"
            return resp
        except Exception as e:
            # Log full traceback to server logs so Railway shows the real cause.
            print(f"[api_tts] Exception: {e}", flush=True)
            traceback.print_exc()
            return jsonify({"error": f"TTS error: {str(e)}"}), 500

    return app


def build_asgi_app():
    """Flask (HTTP) + Async Socket.IO; uvicorn entry: web_app_fastapi:asgi_app"""
    global sio_server, app
    flask_app = create_app()
    app = flask_app
    _sock_dbg = os.getenv("SOCKETIO_DEBUG", "").lower() in ("1", "true", "yes")
    sio = socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins=os.getenv("SOCKETIO_CORS", "*"),
        logger=_sock_dbg,
        engineio_logger=_sock_dbg,
    )
    sio_server = sio
    flask_app.register_async_socketio(sio)

    @asynccontextmanager
    async def _fastapi_lifespan(fa: FastAPI):
        print("[web_app_fastapi] ASGI startup (Flask + Async Socket.IO mounted)", flush=True)
        yield
        print("[web_app_fastapi] ASGI shutdown", flush=True)

    fa = FastAPI(title="Angel API", version="1.0", lifespan=_fastapi_lifespan)
    fa.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    fa.mount("/", WSGIMiddleware(flask_app))
    return socketio.ASGIApp(sio, other_asgi_app=fa)


asgi_app = build_asgi_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(asgi_app, host="0.0.0.0", port=port, workers=1)

