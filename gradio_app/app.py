import os
import json
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile
from datetime import datetime

BACKEND_URL  = os.getenv("BACKEND_URL", "http://localhost:8000")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")


# ── History ───────────────────────────────────────────────────────────────────

def load_history() -> list:
    if os.path.isfile(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_history(history: list) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Gagal simpan history: {e}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def voice_chat(audio, history_state):
    if audio is None:
        return None, "Tidak ada audio.", "Menunggu input...", history_state

    sr, audio_data = audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    try:
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{BACKEND_URL}/voice-chat",
                files={"file": ("voice.wav", f, "audio/wav")},
                timeout=120,
            )

        if response.status_code == 200:
            out_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
            with open(out_path, "wb") as f:
                f.write(response.content)

            transcript = response.headers.get("X-Transcript",    "")
            resp_text  = response.headers.get("X-Response-Text", "")

            entry = {
                "no"        : len(history_state) + 1,
                "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "transcript": transcript,
                "response"  : resp_text,
            }
            history_state = history_state + [entry]
            save_history(history_state)

            info = ""
            if transcript:
                info += f"Transkripsi:\n{transcript}\n\n"
            if resp_text:
                info += f"Respons LLM:\n{resp_text}"
            if not info:
                info = "Berhasil diproses."

            return out_path, info, "Selesai", history_state

        else:
            try:
                detail = response.json().get("detail", response.text[:200])
            except Exception:
                detail = response.text[:200]
            return None, f"Error {response.status_code}:\n{detail}", "Gagal", history_state

    except requests.exceptions.ConnectionError:
        return None, "Tidak bisa konek ke backend.\nPastikan uvicorn berjalan di port 8000.", "Koneksi gagal", history_state
    except requests.exceptions.Timeout:
        return None, "Timeout.", "Timeout", history_state
    except Exception as e:
        return None, f"Error: {e}", "Error", history_state
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def reset_audio():
    """Reset rekaman audio, info box, dan status."""
    return None, "", "Menunggu input..."


def reset_history(history_state):
    """Panggil endpoint reset-history di backend, lalu kosongkan history lokal."""
    try:
        response = requests.delete(f"{BACKEND_URL}/reset-history", timeout=10)
        if response.status_code == 200:
            save_history([])
            return [], "Riwayat berhasil direset."
        else:
            return history_state, f"Gagal reset history: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return history_state, "Tidak bisa konek ke backend."
    except Exception as e:
        return history_state, f"Error: {e}"


# ── UI ────────────────────────────────────────────────────────────────────────

initial_history = load_history()

with gr.Blocks(title="🎙️ Voice Chatbot") as demo:
    gr.Markdown(
        """
        # 🎙️ Voice Chatbot - Code-Switching ID·EN·AR
        **Whisper STT** → **Gemini LLM** → **Coqui TTS**
        """
    )

    history_state = gr.State(initial_history)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",
                label="Rekam Pertanyaan",
            )
            with gr.Row():
                submit_btn       = gr.Button("Kirim", variant="primary")
                reset_audio_btn  = gr.Button("Reset Rekaman", variant="secondary")
                reset_history_btn = gr.Button("Reset History", variant="stop")
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                value="Menunggu input...",
            )
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                type="filepath",
                label="Balasan Suara",
                autoplay=True,
            )
            info_box = gr.Textbox(
                label="Transkripsi & Respons",
                lines=5,
                interactive=False,
                placeholder="Transkripsi dan respons LLM akan muncul di sini...",
            )

    submit_btn.click(
        fn=voice_chat,
        inputs=[audio_input, history_state],
        outputs=[audio_output, info_box, status_box, history_state],
    )

    # reset audio juga bersihkan info_box dan status_box
    reset_audio_btn.click(
        fn=reset_audio,
        inputs=None,
        outputs=[audio_input, info_box, status_box],
    )

    # tombol reset history memanggil backend DELETE /reset-history
    reset_history_btn.click(
        fn=reset_history,
        inputs=[history_state],
        outputs=[history_state, status_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)