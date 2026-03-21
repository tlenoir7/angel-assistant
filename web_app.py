import base64
import json
import os
import threading
import time
from io import BytesIO
from pathlib import Path
import traceback
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, Response, jsonify, render_template_string, request
from flask_socketio import SocketIO, emit
from apscheduler.schedulers.background import BackgroundScheduler

import angel_predictions
import angel_proactive

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
)

try:
    from angel_realtime import AngelRealtimeSession
except Exception:
    AngelRealtimeSession = None  # type: ignore[misc, assignment]

# Module-level storage for morning briefing and check-in
morning_briefing = None
briefing_generated_at = None
check_in_message = None
check_in_generated_at = None
last_activity_at = time.time()
angel = None

# Flask-SocketIO (initialized in create_app; async_mode threading + gunicorn --threads — see Procfile)
socketio: SocketIO | None = None
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


def _network_reset_worker(
    memory_client,
    user_id: str,
    files_cabinet,
    use_mem0_cloud: bool,
) -> None:
    """Runs in a daemon thread; never raises."""
    summary: dict | None = None
    err: str | None = None
    try:
        summary = reset_mission_network_and_reseed(
            memory_client,
            user_id,
            files_cabinet,
            use_mem0_cloud,
        )
    except Exception as ex:
        err = str(ex)
        traceback.print_exc()
    finally:
        try:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            with _network_reset_lock:
                _network_reset_status["in_progress"] = False
                _network_reset_status["last_reset_at"] = now
                if summary is not None:
                    _network_reset_status["last_success"] = bool(summary.get("ok"))
                    _network_reset_status["last_nodes_after"] = summary.get("nodes_after")
                    _network_reset_status["last_edges_after"] = summary.get("edges_after")
                    _network_reset_status["last_error"] = summary.get("error")
                else:
                    _network_reset_status["last_success"] = False
                    _network_reset_status["last_nodes_after"] = None
                    _network_reset_status["last_edges_after"] = None
                    _network_reset_status["last_error"] = err or "reset thread failed"
        except Exception:
            pass


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
        morning_briefing = generate_morning_briefing(
            client,
            user_id,
            memory_summary,
            timezone=tz,
            latest_reflection=latest_reflection,
            recent_briefing_history=recent_briefing_history or None,
            threat_appendix=threat_appendix or None,
            proactive_intelligence_appendix=proactive_appendix or None,
        )
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
            print(f"[web_app] Initial predictions seed: {r}", flush=True)
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
            print(f"[web_app] Proactive watch seed: {r}", flush=True)
        except Exception:
            traceback.print_exc()

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception:
        pass


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
    global angel, socketio, angel_scheduler
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
    _load_expo_push_tokens_from_disk()

    # Log briefing email env at startup for debugging
    tyler_email = (os.getenv("TYLER_EMAIL") or "").strip()
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD") or ""
    print(
        f"[web_app] TYLER_EMAIL present={bool(tyler_email)}, "
        f"GMAIL_APP_PASSWORD present={bool(gmail_pass)}",
        flush=True,
    )

    # Log all environment variable NAMES (not values) available to the process
    try:
        env_names = sorted(os.environ.keys())
        print(
            "[web_app] Environment variable names available to process:",
            ", ".join(env_names),
            flush=True,
        )
    except Exception as e:
        print(f"[web_app] Failed to list environment variable names: {e}", flush=True)

    # Morning briefing: run at BRIEFING_TIME (default 08:00)
    briefing_time = os.getenv("BRIEFING_TIME", "08:00").strip()
    try:
        hour, minute = map(int, briefing_time.split(":")[:2])
    except Exception:
        hour, minute = 8, 0
    sched_tz = os.getenv("TIMEZONE", "America/Los_Angeles").strip()
    scheduler = BackgroundScheduler()
    angel_scheduler = scheduler
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
    scheduler.start()

    # --- WebSocket (Socket.IO) for persistent iOS / low-latency clients ---
    socketio = SocketIO(
        app,
        cors_allowed_origins=os.getenv("SOCKETIO_CORS", "*"),
        async_mode="threading",
        logger=os.getenv("SOCKETIO_DEBUG", "").lower() in ("1", "true", "yes"),
        engineio_logger=os.getenv("SOCKETIO_DEBUG", "").lower() in ("1", "true", "yes"),
    )

    @socketio.on("connect")
    def _ws_connect(auth):
        global last_activity_at
        last_activity_at = time.time()
        device = "ios"
        if isinstance(auth, dict):
            d = (auth.get("device") or "").strip().lower()
            if d in _VALID_CLIENT_DEVICES:
                device = d
        qd = (request.args.get("device") or "").strip().lower()
        if qd in _VALID_CLIENT_DEVICES:
            device = qd
        sid = request.sid
        SOCKET_SESSIONS[sid] = {"device": device, "turns": []}
        print(f"[socket] connect sid={sid!s} device={device}", flush=True)

    @socketio.on("disconnect")
    def _ws_disconnect():
        SOCKET_SESSIONS.pop(request.sid, None)

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

    @socketio.on("user_text")
    def _ws_user_text(data):
        global last_activity_at, check_in_message, check_in_generated_at
        last_activity_at = time.time()
        check_in_message = None
        check_in_generated_at = None
        sid = request.sid
        sess = SOCKET_SESSIONS.get(sid)
        if not sess:
            emit("angel_error", {"message": "No session; reconnect."})
            return
        payload = data if isinstance(data, dict) else {}
        text = (payload.get("message") or payload.get("text") or "").strip()
        if not text:
            emit("angel_error", {"message": "Empty message."})
            return
        location = normalize_location(payload.get("location"))
        print(f"[socket] location received: {location!r}", flush=True)
        emit("angel_thinking", {})
        try:
            turns = _session_turns_for(sid)
            reply = angel.generate_reply(
                text,
                device=sess["device"],
                session_turns=turns or None,
                location=location,
            )
            clean_a = strip_markdown(reply) if angel.use_voice else reply
            _append_turn(sid, text, clean_a)
            emit("angel_response", {"reply": _sanitize_text(reply)})
        except Exception as e:
            traceback.print_exc()
            emit("angel_error", {"message": str(e)})

    @socketio.on("user_audio")
    def _ws_user_audio(data):
        global last_activity_at, check_in_message, check_in_generated_at
        last_activity_at = time.time()
        check_in_message = None
        check_in_generated_at = None
        sid = request.sid
        sess = SOCKET_SESSIONS.get(sid)
        if not sess:
            emit("angel_error", {"message": "No session; reconnect."})
            return
        payload = data if isinstance(data, dict) else {}
        b64 = payload.get("audio_base64") or payload.get("audio") or ""
        if not b64:
            emit("angel_error", {"message": "Missing audio_base64."})
            return
        emit("angel_thinking", {})
        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            emit("angel_error", {"message": f"Invalid base64 audio: {e}"})
            return
        filename = (payload.get("filename") or "recording.m4a").strip() or "recording.m4a"
        try:
            transcript = transcribe_with_whisper(raw, filename=filename).strip()
        except Exception as e:
            traceback.print_exc()
            emit("angel_error", {"message": f"Transcription failed: {e}"})
            return
        emit("angel_transcript", {"transcript": _sanitize_text(transcript)})
        if not transcript:
            emit("angel_response", {"reply": "I couldn't make out what you said."})
            return
        location = normalize_location(payload.get("location"))
        print(f"[socket] location received: {location!r}", flush=True)
        try:
            turns = _session_turns_for(sid)
            reply = angel.generate_reply(
                transcript,
                device=sess["device"],
                session_turns=turns or None,
                location=location,
            )
            clean_a = strip_markdown(reply) if angel.use_voice else reply
            _append_turn(sid, transcript, clean_a)
            emit("angel_response", {"reply": _sanitize_text(reply)})
        except Exception as e:
            traceback.print_exc()
            emit("angel_error", {"message": str(e)})

    # --- Socket.IO namespace /realtime: proxy to OpenAI GPT-4o Realtime API (iPhone, etc.) ---
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

    def _forward_openai_realtime_to_client(sid: str, msg: dict) -> None:
        t = msg.get("type")
        try:
            with app.app_context():
                if t == "_realtime_socket_closed":
                    socketio.emit(
                        "realtime_error",
                        {"message": "OpenAI Realtime connection closed"},
                        room=sid,
                        namespace="/realtime",
                    )
                    return
                if t == "response.audio.delta":
                    d = msg.get("delta", "")
                    if d:
                        socketio.emit(
                            "realtime_response_audio",
                            {"delta": d},
                            room=sid,
                            namespace="/realtime",
                        )
                elif t == "response.audio_transcript.delta":
                    d = msg.get("delta", "")
                    if d:
                        socketio.emit(
                            "realtime_transcript",
                            {"delta": d, "done": False},
                            room=sid,
                            namespace="/realtime",
                        )
                elif t == "response.audio_transcript.done":
                    socketio.emit(
                        "realtime_transcript",
                        {
                            "transcript": _sanitize_text(str(msg.get("transcript", ""))),
                            "done": True,
                        },
                        room=sid,
                        namespace="/realtime",
                    )
                elif t == "response.text.delta":
                    d = msg.get("delta", "")
                    if d:
                        socketio.emit(
                            "realtime_transcript",
                            {"delta": d, "done": False, "channel": "text"},
                            room=sid,
                            namespace="/realtime",
                        )
                elif t == "response.done":
                    socketio.emit(
                        "realtime_response_done",
                        {},
                        room=sid,
                        namespace="/realtime",
                    )
                elif t == "error":
                    err = msg.get("error") or {}
                    msg_txt = (
                        err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    )
                    socketio.emit(
                        "realtime_error",
                        {"message": _sanitize_text(msg_txt)},
                        room=sid,
                        namespace="/realtime",
                    )
                elif t == "response.error":
                    socketio.emit(
                        "realtime_error",
                        {"message": _sanitize_text(str(msg))},
                        room=sid,
                        namespace="/realtime",
                    )
        except Exception as e:
            print(f"[realtime] forward error: {e}", flush=True)
            traceback.print_exc()

    @socketio.on("connect", namespace="/realtime")
    def _realtime_ns_connect(auth):
        global last_activity_at
        last_activity_at = time.time()
        if AngelRealtimeSession is None:
            print("[realtime] AngelRealtimeSession unavailable (import failed)", flush=True)
            return False
        if not (os.getenv("OPENAI_REALTIME_API_KEY") or os.getenv("OPENAI_API_KEY")):
            print("[realtime] Missing OPENAI_REALTIME_API_KEY / OPENAI_API_KEY", flush=True)
            return False
        sid = request.sid
        try:
            system_prompt = _build_openai_realtime_system_prompt()
            rt = AngelRealtimeSession()
            rt.connect(system_prompt)
        except Exception as e:
            traceback.print_exc()
            print(f"[realtime] connect failed: {e}", flush=True)
            return False
        REALTIME_PROXY_BY_SID[sid] = {"session": rt}
        rt.start_receiver_thread(lambda m, sid=sid: _forward_openai_realtime_to_client(sid, m))
        print(f"[realtime] OpenAI Realtime proxy started sid={sid!s}", flush=True)
        return True

    @socketio.on("disconnect", namespace="/realtime")
    def _realtime_ns_disconnect():
        sid = request.sid
        entry = REALTIME_PROXY_BY_SID.pop(sid, None)
        if entry and entry.get("session"):
            try:
                entry["session"].disconnect()
            except Exception:
                pass
        print(f"[realtime] disconnected sid={sid!s}", flush=True)

    @socketio.on("realtime_audio", namespace="/realtime")
    def _realtime_ns_audio(data):
        global last_activity_at
        last_activity_at = time.time()
        sid = request.sid
        entry = REALTIME_PROXY_BY_SID.get(sid)
        if not entry:
            emit(
                "realtime_error",
                {"message": "No Realtime session; reconnect."},
                namespace="/realtime",
            )
            return
        payload = data if isinstance(data, dict) else {}
        b64 = payload.get("audio") or payload.get("audio_base64") or ""
        if not isinstance(b64, str) or not b64.strip():
            emit("realtime_error", {"message": "Missing audio (base64)."}, namespace="/realtime")
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
                # Pass through as standard base64 PCM16 chunks (OpenAI wire format)
                rt.append_input_audio_base64(b64.strip())
        except Exception as e:
            traceback.print_exc()
            emit("realtime_error", {"message": str(e)}, namespace="/realtime")

    @socketio.on("realtime_commit", namespace="/realtime")
    def _realtime_ns_commit(data=None):
        global last_activity_at
        last_activity_at = time.time()
        sid = request.sid
        entry = REALTIME_PROXY_BY_SID.get(sid)
        if not entry:
            emit(
                "realtime_error",
                {"message": "No Realtime session; reconnect."},
                namespace="/realtime",
            )
            return
        rt = entry["session"]
        try:
            rt.commit_input_buffer()
            rt.create_audio_response()
        except Exception as e:
            traceback.print_exc()
            emit("realtime_error", {"message": str(e)}, namespace="/realtime")

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

    # --- Intelligence File Cabinet (dynamic folders; Mem0 category intelligence_file) ---
    @app.route("/api/files/create", methods=["POST"])
    def api_files_create():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
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

    @app.route("/api/files/list", methods=["GET"])
    def api_files_list():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
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
        text = angel.files_cabinet.get_summary()
        return jsonify({"ok": True, "summary": text})

    @app.route("/api/files/delete", methods=["POST"])
    def api_files_delete():
        if angel is None:
            return jsonify({"error": "Angel not initialized"}), 503
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

            now = datetime.now(tzinfo) if tzinfo is not None else datetime.utcnow()
            current_time = now.isoformat()
        except Exception:
            current_time = None

        tyler_email = os.getenv("TYLER_EMAIL")
        gmail_pass = os.getenv("GMAIL_APP_PASSWORD")
        print(
            "[api_status] Env check | "
            f"TYLER_EMAIL_is_none={tyler_email is None} | "
            f"GMAIL_APP_PASSWORD_is_none={gmail_pass is None}",
            flush=True,
        )

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
            text_len = len(text)

            api_key_present = bool(os.getenv("OPENAI_API_KEY"))

            # Diagnostic logging so Railway shows what's going on.
            print(
                f"[api_tts] called | api_key_present={api_key_present} | text_len={text_len}",
                flush=True,
            )

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


app = create_app()


if __name__ == "__main__":
    # Local `python web_app.py` (production uses gunicorn; see Procfile).
    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        debug=False,
        use_reloader=False,
    )

