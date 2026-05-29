import os
import uuid
import tempfile
import whisper

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")

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
                language=None,                  
                task="transcribe",               
                initial_prompt=MULTILINGUAL_PROMPT,  
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