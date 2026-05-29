import os
import uuid
import tempfile
import whisper

# Pilihan model: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"
# Sesuaikan dengan kemampuan perangkat:
#   - CPU terbatas  → "base" atau "small"
#   - CPU memadai   → "medium" atau "large-v3-turbo"
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")
# Di file .env diset: WHISPER_MODEL=medium  (atau large-v3-turbo jika perangkat kuat)

# Load model sekali saat modul diimport (agar tidak reload setiap request)
print(f"[STT] Memuat model Whisper: {WHISPER_MODEL_NAME} ...")
_model = whisper.load_model(WHISPER_MODEL_NAME)
print(f"[STT] Model Whisper siap.")


def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    """
    Transkrip file audio menggunakan openai-whisper (Python library).
    Mendukung ujaran code-switching Bahasa Indonesia, Inggris, dan Arab.

    Args:
        file_bytes (bytes): Isi file audio dalam bentuk bytes.
        file_ext (str): Ekstensi file audio, default ".wav".

    Returns:
        str: Teks hasil transkripsi (bisa campuran ID/EN/AR),
             atau pesan error jika gagal.
    """
    # ── FIX #2 ───────────────────────────────────────────────────────────────
    # initial_prompt membantu Whisper "tahu" bahwa audio ini code-switching.
    # Prompt ini tidak muncul di output, hanya jadi konteks internal model.
    # Referensi: https://github.com/openai/whisper#python-usage
    # ─────────────────────────────────────────────────────────────────────────
    MULTILINGUAL_PROMPT = (
        "Percakapan ini mengandung campuran Bahasa Indonesia, Bahasa Inggris, "
        "dan Bahasa Arab. Transkripsi semua kata sesuai bahasa aslinya. "
        "This conversation contains Indonesian, English, and Arabic code-switching."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Simpan bytes ke file temporer
        audio_path = os.path.join(tmpdir, f"{uuid.uuid4()}{file_ext}")
        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        try:
            result = _model.transcribe(
                audio_path,
                language=None,                   # auto-detect, bukan hardcode "id"
                task="transcribe",               # pertahankan bahasa asli (bukan translate)
                initial_prompt=MULTILINGUAL_PROMPT,  # bantu Whisper kenali code-switching
                fp16=False,                     
            )
            
            text = result["text"].strip()

            # Log bahasa yang terdeteksi oleh Whisper (berguna untuk debugging)
            detected_lang = result.get("language", "unknown")
            print(f"[STT] Bahasa terdeteksi Whisper: {detected_lang}")
            print(f"[STT] Transkripsi: {text}")

            return text

        except Exception as e:
            print(f"[STT ERROR] {e}")
            return f"[ERROR] Whisper failed: {e}"