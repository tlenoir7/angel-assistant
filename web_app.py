import os
from io import BytesIO
import traceback

from flask import Flask, Response, jsonify, render_template_string, request

from angel import AngelCore, get_elevenlabs_mp3, transcribe_with_whisper


def _sanitize_text(s: str) -> str:
    """
    Strip/replace any invalid Unicode (including surrogate characters)
    so Flask/Werkzeug can safely encode responses.
    """
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def create_app() -> Flask:
    app = Flask(__name__)

    user_id = os.getenv("ANGEL_USER_ID", "railway-user")
    angel = AngelCore(user_id=user_id, use_voice=True)
    # Warm up memories once on startup
    angel.load_initial_memory_summary()

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
      <main id="chat"></main>
      <footer>
        <div id="status">Idle</div>
        <div id="input-row">
          <input id="text-input" type="text" placeholder="Type a message..." />
          <button id="send-btn">Send</button>
          <button id="voice-btn">🎤 Hold to speak</button>
        </div>
      </footer>
      <script>
        const chat = document.getElementById("chat");
        const statusEl = document.getElementById("status");
        const textInput = document.getElementById("text-input");
        const sendBtn = document.getElementById("send-btn");
        const voiceBtn = document.getElementById("voice-btn");
        const voiceToggle = document.getElementById("voice-toggle");

        let voiceMode = true;

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
            if (!resp.ok) return;
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audio.onended = () => URL.revokeObjectURL(url);
            await audio.play();
          } catch (e) {}
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
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400
        reply = angel.generate_reply(message)
        reply = _sanitize_text(reply)
        return jsonify({"reply": reply})

    @app.route("/api/voice", methods=["POST"])
    def api_voice():
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

    @app.route("/api/tts", methods=["POST"])
    def api_tts():
        try:
            data = request.get_json(silent=True) or {}
            text = (data.get("text") or "").strip()
            if not text:
                return jsonify({"error": "Missing or empty text"}), 400
            mp3_bytes = get_elevenlabs_mp3(text)
            if not mp3_bytes:
                return jsonify(
                    {"error": "TTS unavailable (check ELEVENLABS_API_KEY and logs)"}
                ), 503
            return Response(mp3_bytes, mimetype="audio/mpeg")
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

