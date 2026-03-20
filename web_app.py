import json
import os
import time
from io import BytesIO
from pathlib import Path
import traceback
from datetime import datetime

import requests
from flask import Flask, Response, jsonify, render_template_string, request
from apscheduler.schedulers.background import BackgroundScheduler

# AngelCore includes Stage 2: strategy, patterns, deep research, people profiles
from angel import (
    AngelCore,
    tts_gpt4o,
    transcribe_with_whisper,
    generate_morning_briefing,
    send_briefing_email,
    create_anthropic_client,
    build_memory_summary_with_sections,
    run_memory_reflection,
    get_latest_reflection_text,
    get_recent_briefing_history_for_prompt,
    summarize_briefing_for_history,
    add_structured_memory,
    CATEGORY_BRIEFING_HISTORY,
)

# Module-level storage for morning briefing and check-in
morning_briefing = None
briefing_generated_at = None
check_in_message = None
check_in_generated_at = None
last_activity_at = time.time()
angel = None

# Expo push: in-memory + push_tokens.json (same directory as this module)
PUSH_TOKENS_PATH = Path(__file__).resolve().parent / "push_tokens.json"
EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
expo_push_tokens: list[str] = []


_VALID_CLIENT_DEVICES = frozenset({"ios", "desktop", "mobile_web"})


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
        morning_briefing = generate_morning_briefing(
            client,
            user_id,
            memory_summary,
            timezone=tz,
            latest_reflection=latest_reflection,
            recent_briefing_history=recent_briefing_history or None,
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
    global angel
    app = Flask(__name__)

    user_id = os.getenv("ANGEL_USER_ID", "railway-user")
    angel = AngelCore(user_id=user_id, use_voice=True)
    # Warm up memories once on startup
    angel.load_initial_memory_summary()
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
    scheduler.add_job(_run_morning_briefing_job, "cron", hour=hour, minute=minute)
    scheduler.add_job(
        _run_weekly_reflection_job,
        "cron",
        day_of_week="sun",
        hour=6,
        minute=0,
        timezone=sched_tz,
    )
    scheduler.add_job(_run_check_in_job, "interval", minutes=15)
    scheduler.start()

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
        reply = angel.generate_reply(message, device=device)
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
        reply = angel.generate_reply(transcript, device=device)
        return jsonify(
            {
                "transcript": _sanitize_text(transcript),
                "reply": _sanitize_text(reply),
            }
        )

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
    # For local testing; Railway will use the Procfile/gunicorn command.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)

