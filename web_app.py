import os
import time
from io import BytesIO
import traceback

from flask import Flask, Response, jsonify, render_template_string, request
from apscheduler.schedulers.background import BackgroundScheduler

# AngelCore includes Stage 2: strategy, patterns, deep research, people profiles
from angel import (
    AngelCore,
    get_elevenlabs_mp3,
    transcribe_with_whisper,
    generate_morning_briefing,
    send_briefing_email,
    create_anthropic_client,
    build_memory_summary_with_sections,
)

# Module-level storage for morning briefing and check-in
morning_briefing = None
briefing_generated_at = None
check_in_message = None
check_in_generated_at = None
last_activity_at = time.time()
angel = None


def _sanitize_text(s: str) -> str:
    """
    Strip/replace any invalid Unicode (including surrogate characters)
    so Flask/Werkzeug can safely encode responses.
    """
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def _run_morning_briefing_job():
    global morning_briefing, briefing_generated_at
    try:
        user_id = os.getenv("ANGEL_USER_ID", "railway-user")
        client = create_anthropic_client()
        memories = angel._fetch_combined_memories()
        memory_summary = build_memory_summary_with_sections(memories, None)
        tz = os.getenv("TIMEZONE", "America/Los_Angeles")
        morning_briefing = generate_morning_briefing(client, user_id, memory_summary, timezone=tz)
        briefing_generated_at = time.time()
        send_briefing_email(morning_briefing)
    except Exception as e:
        traceback.print_exc()
        morning_briefing = f"Briefing unavailable: {e}"
        briefing_generated_at = time.time()


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

    # Morning briefing: run at BRIEFING_TIME (default 08:00)
    briefing_time = os.getenv("BRIEFING_TIME", "08:00").strip()
    try:
        hour, minute = map(int, briefing_time.split(":")[:2])
    except Exception:
        hour, minute = 8, 0
    scheduler = BackgroundScheduler()
    scheduler.add_job(_run_morning_briefing_job, "cron", hour=hour, minute=minute)
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
          <button id="send-btn">Send</button>
          <button id="voice-btn">🎤 Hold to speak</button>
        </div>
      </footer>
      <audio id="tts-audio" style="display:none;" preload="auto"></audio>
      <script>
        const chat = document.getElementById("chat");
        const statusEl = document.getElementById("status");
        const textInput = document.getElementById("text-input");
        const sendBtn = document.getElementById("send-btn");
        const voiceBtn = document.getElementById("voice-btn");
        const voiceToggle = document.getElementById("voice-toggle");
        const ttsAudio = document.getElementById("tts-audio");
        const briefingContainer = document.getElementById("briefing-container");
        const checkInContainer = document.getElementById("check-in-container");

        let voiceMode = true;

        function isToday(ts) {
          if (!ts) return false;
          const d = new Date(ts * 1000);
          const today = new Date();
          return d.getDate() === today.getDate() && d.getMonth() === today.getMonth() && d.getFullYear() === today.getFullYear();
        }

        async function loadBriefingAndCheckIn() {
          try {
            const [briefRes, checkRes] = await Promise.all([fetch("/api/briefing"), fetch("/api/check_in")]);
            const briefingData = briefRes.ok ? await briefRes.json() : {};
            const checkInData = checkRes.ok ? await checkRes.json() : {};
            if (briefingData.briefing && isToday(briefingData.generated_at)) {
              briefingContainer.innerHTML = '<div class="briefing-block"><h3>☀️ Morning briefing</h3><p>' + briefingData.briefing.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>") + '</p></div>';
            } else {
              briefingContainer.innerHTML = '';
            }
            if (checkInData.message && checkInData.generated_at) {
              checkInContainer.innerHTML = '<div class="check-in-block">' + checkInData.message.replace(/</g, "&lt;").replace(/>/g, "&gt;") + '</div>';
            } else {
              checkInContainer.innerHTML = '';
            }
          } catch (e) {
            console.error("Load briefing/check-in:", e);
          }
        }
        loadBriefingAndCheckIn();

        function updateVoiceToggleLabel() {
          voiceToggle.textContent = voiceMode ? "\uD83D\uDD0A Voice Mode" : "\uD83D\uDD07 Text Mode";
          voiceToggle.classList.toggle("on", voiceMode);
          voiceToggle.classList.toggle("off", !voiceMode);
        }

        voiceToggle.addEventListener("click", () => {
          voiceMode = !voiceMode;
          updateVoiceToggleLabel();
        });
        updateVoiceToggleLabel();

        async function playTts(text) {
          if (!text || !voiceMode) return;
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
            ttsAudio.src = url;
            ttsAudio.onended = () => {
              URL.revokeObjectURL(url);
            };
            try {
              await ttsAudio.play();
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
          const msg = textInput.value.trim();
          if (!msg) return;
          textInput.value = "";
          appendMessage("You", msg);
          statusEl.textContent = "Angel is thinking...";
          try {
            const resp = await fetch("/api/message", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message: msg }),
            });
            const data = await resp.json();
            appendMessage("Angel", data.reply);
            await playTts(data.reply);
          } catch (e) {
            appendMessage("Angel", "I ran into an error processing that.");
          } finally {
            statusEl.textContent = "Idle";
          }
        }

        sendBtn.addEventListener("click", sendText);
        textInput.addEventListener("keydown", (e) => {
          if (e.key === "Enter") {
            sendText();
          }
        });

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
              statusEl.textContent = "Transcribing and thinking...";
              appendMessage("You", "(voice message)");
              const formData = new FormData();
              formData.append("audio", blob, "audio.webm");
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
                statusEl.textContent = "Idle";
              }
            };
          } catch (e) {
            alert("Microphone access denied or unavailable.");
          }
        }

        voiceBtn.addEventListener("mousedown", async () => {
          await initMedia();
          if (!mediaRecorder) return;
          chunks = [];
          mediaRecorder.start();
          statusEl.textContent = "Listening (hold button)...";
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
          statusEl.textContent = "Listening (hold button)...";
        });

        voiceBtn.addEventListener("touchend", (e) => {
          e.preventDefault();
          if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
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
        reply = angel.generate_reply(message)
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
        reply = angel.generate_reply(transcript)
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

    @app.route("/api/check_in", methods=["GET"])
    def api_check_in():
        global check_in_message, check_in_generated_at
        return jsonify({
            "message": check_in_message,
            "generated_at": check_in_generated_at,
        })

    @app.route("/api/tts", methods=["POST"])
    def api_tts():
        try:
            data = request.get_json(silent=True) or {}
            text = (data.get("text") or "").strip()
            text_len = len(text)

            api_key_present = bool(os.getenv("ELEVENLABS_API_KEY"))
            voice_id = os.getenv("ELEVENLABS_VOICE_ID") or "EXAVITQu4vr4xnSDxMaL"

            # Diagnostic logging so Railway shows what's going on.
            print(
                f"[api_tts] called | api_key_present={api_key_present} "
                f"| voice_id={voice_id} | text_len={text_len}",
                flush=True,
            )

            if not text:
                return jsonify({"error": "Missing or empty text"}), 400

            if not api_key_present:
                return jsonify(
                    {
                        "error": "ELEVENLABS_API_KEY is not set in the environment; TTS is unavailable."
                    }
                ), 503

            mp3_bytes = get_elevenlabs_mp3(text)
            if not mp3_bytes:
                return jsonify(
                    {
                        "error": (
                            "TTS generation failed or returned empty audio. "
                            "Check server logs for details (api key, voice id, response)."
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

