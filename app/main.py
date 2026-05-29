"""
main.py — Entry point backend FastAPI untuk Voice Chatbot.
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech
from app.utils import normalize_text, clean_llm_response, truncate_text, cleanup_file, logger
import app.llm as llm_module

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

app = FastAPI(
    title="Voice Chatbot API",
    description="Pipeline STT → LLM → TTS untuk chatbot berbasis suara Bahasa Indonesia.",
    version="1.0.0",
)


def _safe_header(text: str, max_chars: int = 500) -> str:
    """Encode teks agar aman untuk HTTP header (latin-1 only)."""
    truncated = text[:max_chars]
    return truncated.encode("latin-1", errors="ignore").decode("latin-1")


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok", "message": "Voice Chatbot API is running."})


@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    tts_output_path = None

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="File audio kosong atau tidak valid.")

        original_filename = file.filename or "audio.wav"
        file_ext = os.path.splitext(original_filename)[-1].lower() or ".wav"
        logger.info(f"[/voice-chat] Menerima file: {original_filename} ({len(file_bytes)} bytes)")

        # STT
        logger.info("[/voice-chat] Memulai transkripsi STT...")
        raw_transcript = transcribe_speech_to_text(file_bytes, file_ext=file_ext)
        if raw_transcript.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=f"STT error: {raw_transcript}")
        logger.info(f"[/voice-chat] Transkripsi: {raw_transcript}")

        # Normalisasi
        normalized_transcript = normalize_text(raw_transcript)
        if not normalized_transcript:
            raise HTTPException(status_code=422, detail="Transkripsi kosong setelah normalisasi.")

        # LLM
        logger.info("[/voice-chat] Mengirim ke LLM...")
        llm_raw_response = generate_response(normalized_transcript)
        if llm_raw_response.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=f"LLM error: {llm_raw_response}")
        logger.info(f"[/voice-chat] Respons LLM: {llm_raw_response}")

        # Bersihkan teks
        clean_response = clean_llm_response(llm_raw_response)
        tts_input = truncate_text(clean_response, max_chars=500)

        # TTS
        logger.info("[/voice-chat] Memulai sintesis TTS...")
        tts_output_path = transcribe_text_to_speech(tts_input)
        if tts_output_path.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=f"TTS error: {tts_output_path}")
        logger.info(f"[/voice-chat] Audio output: {tts_output_path}")

        return FileResponse(
            path=tts_output_path,
            media_type="audio/wav",
            filename="response.wav",
            headers={
                "X-Transcript":                  _safe_header(raw_transcript),
                "X-Response-Text":               _safe_header(clean_response),
                "Access-Control-Expose-Headers": "X-Transcript, X-Response-Text",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[/voice-chat] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.delete("/reset-history")
async def reset_history():
    try:
        cleanup_file(CHAT_HISTORY_FILE)
        llm_module.chat = llm_module.load_chat_history()
        logger.info("[/reset-history] Riwayat chat berhasil direset.")
        return JSONResponse(content={"status": "ok", "message": "Riwayat chat berhasil direset."})
    except Exception as e:
        logger.exception(f"[/reset-history] Gagal reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))