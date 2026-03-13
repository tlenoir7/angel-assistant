import threading
import time
from collections import deque
from io import BytesIO
import math
import os
import sys

import tkinter as tk
from tkinter import scrolledtext, ttk

import pyaudio
import pygame
import webrtcvad
import wave
import pystray
from PIL import Image, ImageDraw

from angel import AngelCore, transcribe_with_whisper, speak_with_elevenlabs


class AngelApp:
    def __init__(self, root: tk.Tk, user_id: str):
        self.root = root
        self.root.title("Angel")
        self.root.configure(bg="#121212")
        self.root.minsize(640, 480)

        self.core = AngelCore(user_id=user_id, use_voice=True)

        # Conversation and audio state
        self.listening = False  # used for manual override
        self.running = True  # master flag for background listener
        self.current_generation = 0  # increment per user utterance
        # Whether Angel is actively listening & responding or paused
        self.active = False  # starts paused/standby
        # Pause background VAD only during manual override
        self.suspend_listener = False
        # Configurable RMS energy threshold to ignore quiet background noise
        # on 16-bit PCM (-32768..32767). You can tweak this if Angel misses you
        # or triggers too often on noise.
        # 800 is a middle ground between the original 500 (too sensitive)
        # and 1500 (too strict for your setup).
        self.energy_threshold = 800.0

        # System tray icon (initialized in main)
        self.tray_icon: pystray.Icon | None = None

        self._build_ui()

        # Preload memories for context (optional but nice for logs)
        self.core.load_initial_memory_summary()

        # Start continuous background listener using VAD (in standby/paused mode)
        threading.Thread(target=self._listening_loop, daemon=True).start()
        # Initial status: Paused / standby
        self.set_status("Status: Paused (standby)")

    def _build_ui(self):
        # Configure styles for dark theme
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Angel.TFrame",
            background="#121212",
        )
        style.configure(
            "AngelHeader.TLabel",
            background="#1E1E1E",
            foreground="#FFFFFF",
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "AngelSubheader.TLabel",
            background="#1E1E1E",
            foreground="#BBBBBB",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Angel.TButton",
            background="#2E2E2E",
            foreground="#FFFFFF",
            font=("Segoe UI", 10),
            padding=5,
        )
        style.map(
            "Angel.TButton",
            background=[("active", "#3E3E3E")],
        )

        # Main container
        main_frame = ttk.Frame(self.root, style="Angel.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header bar
        header = tk.Frame(main_frame, bg="#1E1E1E", height=48)
        header.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(
            header,
            text="Angel",
            bg="#1E1E1E",
            fg="#FFFFFF",
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(side=tk.LEFT, padx=12, pady=8)

        subtitle = tk.Label(
            header,
            text="Personal AI Companion",
            bg="#1E1E1E",
            fg="#AAAAAA",
            font=("Segoe UI", 9),
        )
        subtitle.pack(side=tk.LEFT, padx=8, pady=8)

        settings_btn = ttk.Button(
            header,
            text="Settings",
            command=self.open_settings,
            style="Angel.TButton",
        )
        settings_btn.pack(side=tk.RIGHT, padx=12, pady=8)

        minimize_btn = ttk.Button(
            header,
            text="Minimize to tray",
            command=self.minimize_to_tray,
            style="Angel.TButton",
        )
        minimize_btn.pack(side=tk.RIGHT, padx=8, pady=8)

        # Conversation area
        convo_frame = ttk.Frame(main_frame, style="Angel.TFrame")
        convo_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        self.chat = scrolledtext.ScrolledText(
            convo_frame,
            wrap=tk.WORD,
            bg="#181818",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
            relief=tk.FLAT,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.config(state=tk.DISABLED)

        # Bottom bar
        bottom = tk.Frame(main_frame, bg="#121212", height=60)
        bottom.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 8))

        self.status_label = tk.Label(
            bottom,
            text="Status: Paused (standby)",
            bg="#121212",
            fg="#AAAAAA",
            font=("Segoe UI", 9),
        )
        self.status_label.pack(side=tk.LEFT, padx=8)

        self.mic_button = ttk.Button(
            bottom,
            text="🎤 Speak",
            command=self.on_mic_pressed,
            style="Angel.TButton",
        )
        self.mic_button.pack(side=tk.RIGHT, padx=8)

    def append_message(self, sender: str, text: str):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{sender}: ", "sender")
        self.chat.insert(tk.END, text + "\n\n")
        self.chat.tag_config("sender", foreground="#4CAF50" if sender == "Angel" else "#64B5F6")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def set_status(self, text: str):
        self.status_label.config(text=text)

    def on_mic_pressed(self):
        """
        Manual override: temporarily suspends the background listener and
        captures a single utterance. Useful if always-on listening missed you.
        """
        if not self.active:
            # Do nothing when Angel is paused
            self.set_status("Status: Paused – tap Start from the tray to activate.")
            return
        if self.listening:
            return
        self.listening = True
        self.suspend_listener = True
        self.mic_button.config(text="Listening (override)...", state=tk.DISABLED)
        self.set_status("Override: listening for your voice...")
        threading.Thread(target=self._capture_and_respond_once, daemon=True).start()

    # ---- Audio + VAD ----

    def _listening_loop(self):
        """
        Always-on background loop: continuously captures utterances via VAD.
        """
        while self.running:
            if self.suspend_listener or not self.active:
                time.sleep(0.1)
                continue
            wav_bytes = self._record_utterance_with_vad()
            if not wav_bytes:
                continue
            self._start_generation(wav_bytes)

    def _start_generation(self, wav_bytes: bytes):
        self.current_generation += 1
        gen_id = self.current_generation
        threading.Thread(
            target=self._process_utterance, args=(gen_id, wav_bytes), daemon=True
        ).start()

    def _capture_and_respond_once(self):
        """
        Single-utterance capture for manual override; uses same processing path
        as the continuous listener.
        """
        if not self.active:
            return
        try:
            wav_bytes = self._record_utterance_with_vad()
            if wav_bytes:
                self._start_generation(wav_bytes)
        finally:
            self.suspend_listener = False
            self.listening = False
            self.root.after(
                0, self.mic_button.config, {"text": "🎤 Speak", "state": tk.NORMAL}
            )
            self.root.after(0, self.set_status, "Listening...")

    def _process_utterance(self, gen_id: int, wav_bytes: bytes):
        """
        Turn a captured utterance into a transcript, get Angel's reply,
        and speak it. Only the latest generation id will actually speak.
        """
        if not self.active:
            return

        transcript = transcribe_with_whisper(wav_bytes).strip()
        if not transcript:
            return

        # Immediately indicate that Angel heard you and start thinking,
        # without waiting on other processing.
        self.root.after(0, self.set_status, "Status: Thinking (heard you)...")

        # Start Claude call right away for minimal latency.
        reply = self.core.generate_reply(transcript)

        # If a newer utterance has started since we began, skip speaking this one.
        if gen_id != self.current_generation:
            return

        # Append messages to the UI after the reply is ready.
        self.root.after(0, self.append_message, "You", transcript)
        self.root.after(0, self.append_message, "Angel", reply)

        # Play TTS (blocks this worker thread but not the UI)
        self.root.after(0, self.set_status, "Status: Active – Angel is speaking...")
        speak_with_elevenlabs(reply)

        # Only set status to Listening if nothing else has started
        if gen_id == self.current_generation:
            self.root.after(0, self.set_status, "Status: Active (listening)")

    def _record_utterance_with_vad(self) -> bytes | None:
        """
        Use WebRTC VAD to capture a single spoken utterance with natural
        start/stop detection. Returns WAV bytes or None.
        """
        rate = 16000
        channels = 1
        format_ = pyaudio.paInt16
        frame_duration_ms = 20  # 20 ms frames
        frame_size = int(rate * frame_duration_ms / 1000)  # samples per frame
        bytes_per_sample = 2  # 16-bit
        frame_bytes = frame_size * bytes_per_sample

        vad = webrtcvad.Vad(2)  # 0-3, higher = more aggressive

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=format_,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=frame_size,
        )

        ring_buffer = deque(maxlen=int(1.0 * 1000 / frame_duration_ms))  # 1 second
        triggered = False
        voiced_frames = []
        num_unvoiced = 0
        interrupt_voiced_count = 0  # consecutive voiced frames used to trigger interruption

        try:
            start_time = time.time()
            max_total_seconds = 20  # safety guard
            while True:
                frame = stream.read(frame_size, exception_on_overflow=False)
                if len(frame) < frame_bytes:
                    continue

                # Compute RMS energy and only run VAD if the frame is
                # energetic enough. This helps ignore background noise.
                # 16-bit PCM little-endian samples.
                if len(frame) == frame_bytes:
                    samples = [
                        int.from_bytes(frame[i : i + 2], byteorder="little", signed=True)
                        for i in range(0, frame_bytes, 2)
                    ]
                    if samples:
                        mean_sq = sum(s * s for s in samples) / len(samples)
                        rms = math.sqrt(mean_sq)
                    else:
                        rms = 0.0
                else:
                    rms = 0.0

                if rms < self.energy_threshold:
                    is_speech = False
                    interrupt_voiced_count = 0
                else:
                    is_speech = vad.is_speech(frame, rate)

                    # If Angel is currently speaking and we detect at least
                    # 3 consecutive voiced frames above the energy threshold,
                    # stop playback so the user can interrupt and start a new turn.
                    if is_speech:
                        interrupt_voiced_count += 1
                        try:
                            if (
                                interrupt_voiced_count >= 3
                                and pygame.mixer.get_init()
                                and pygame.mixer.music.get_busy()
                            ):
                                pygame.mixer.music.stop()
                        except Exception:
                            pass
                    else:
                        interrupt_voiced_count = 0

                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in ring_buffer if speech])

                    # Start when we see enough voiced frames
                    if num_voiced > 0.6 * ring_buffer.maxlen:
                        triggered = True
                        for f, s in ring_buffer:
                            voiced_frames.append(f)
                        ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    if not is_speech:
                        num_unvoiced += 1
                    else:
                        num_unvoiced = 0

                    # Consider utterance ended after ~0.5s of silence
                    if num_unvoiced > int(0.5 * 1000 / frame_duration_ms):
                        break

                if time.time() - start_time > max_total_seconds:
                    break

            if not voiced_frames:
                return None

            buffer = BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(pa.get_sample_size(format_))
                wf.setframerate(rate)
                wf.writeframes(b"".join(voiced_frames))

            return buffer.getvalue()
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    # ---- Settings ----

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Angel Settings")
        win.configure(bg="#1E1E1E")

        label = tk.Label(
            win,
            text="Settings",
            bg="#1E1E1E",
            fg="#FFFFFF",
            font=("Segoe UI", 12, "bold"),
        )
        label.pack(padx=12, pady=(12, 4), anchor="w")

        mode_label = tk.Label(
            win,
            text="Mode: Voice (default for this app)",
            bg="#1E1E1E",
            fg="#CCCCCC",
            font=("Segoe UI", 9),
        )
        mode_label.pack(padx=12, pady=4, anchor="w")

        info_label = tk.Label(
            win,
            text="Angel remembers you via the user id and Mem0.\n"
            "Advanced settings like voice ID can still be configured via .env.",
            bg="#1E1E1E",
            fg="#AAAAAA",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
        )
        info_label.pack(padx=12, pady=(4, 12), anchor="w")

    # ---- Window + tray helpers ----

    def minimize_to_tray(self):
        """Hide the main window but keep Angel running in the tray."""
        self.root.withdraw()

    def show_window(self):
        """Show and focus the main window."""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def start_angel(self):
        """Activate Angel: start listening and responding."""
        self.active = True
        self.set_status("Status: Active (listening)")

    def pause_angel(self):
        """Pause Angel: stop listening and new activity, keep app running."""
        self.active = False
        self.set_status("Status: Paused")

    def shutdown(self):
        """Completely stop Angel and close the app."""
        self.running = False
        try:
            if self.tray_icon is not None:
                self.tray_icon.stop()
        except Exception:
            pass
        self.root.after(0, self.root.destroy)


def _create_tray_icon(app: AngelApp, root: tk.Tk) -> pystray.Icon:
    # Simple dark circular icon with an 'A'
    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((4, 4, size - 4, size - 4), fill=(30, 30, 30, 255))
    draw.text((size // 3, size // 4), "A", fill=(255, 255, 255, 255))

    def do_start(_):
        root.after(0, app.start_angel)

    def do_pause(_):
        root.after(0, app.pause_angel)

    def do_open(_):
        root.after(0, app.show_window)

    def do_restart(_):
        # Simple restart: re-exec the current process.
        app.running = False
        try:
            if app.tray_icon is not None:
                app.tray_icon.stop()
        except Exception:
            pass
        root.after(0, root.destroy)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def do_quit(_):
        app.shutdown()

    menu = pystray.Menu(
        pystray.MenuItem("Start Angel", do_start),
        pystray.MenuItem("Pause Angel", do_pause),
        pystray.MenuItem("Open Window", do_open),
        pystray.MenuItem("Restart", do_restart),
        pystray.MenuItem("Quit", do_quit),
    )

    icon = pystray.Icon("Angel", image, "Angel", menu)
    return icon


def main():
    # Prefer a fixed default when no interactive console is available so
    # startup shortcuts and pythonw.exe launches don't block on input.
    default_user = "tyler"
    user_id = default_user

    try:
        if sys.stdin is not None and sys.stdin.isatty():
            entered = input(
                f"Enter user id for Angel (press Enter for '{default_user}'): "
            ).strip()
            if entered:
                user_id = entered
    except (EOFError, OSError):
        # Non-interactive environment; keep default_user
        user_id = default_user

    root = tk.Tk()
    app = AngelApp(root, user_id=user_id)
    root.protocol("WM_DELETE_WINDOW", app.minimize_to_tray)

    # Create system tray icon with controls
    icon = _create_tray_icon(app, root)
    app.tray_icon = icon

    threading.Thread(target=icon.run, daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    main()

