import os
import re
import time
from google import genai
from google.genai import types
from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Gemini API Keys + Models
# ─────────────────────────────────────────────
GEMINI_API_KEYS = [key for key in [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6"),
    os.getenv("GEMINI_API_KEY_7"),
] if key]

GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemma-4-26b-a4b-it",
    "gemini-2.5-flash",
]

if not GEMINI_API_KEYS:
    raise ValueError("[LLM] Tidak ada API key ditemukan di .env!")

print(f"[LLM] Gemini: {len(GEMINI_API_KEYS)} key, {len(GEMINI_MODELS)} model")

# Mode 1: PRESERVE CODE-SWITCHING
# Respons mengikuti pola bahasa yang sama dengan input (campur ID/EN/AR)
SYSTEM_PROMPT_PRESERVE_CS = """
You are a responsive, intelligent multilingual virtual assistant that understands and speaks Indonesian, English, and Arabic.

The user may speak in code-switching - mixing Indonesian, English, and Arabic in one utterance.

Your task:
- Respond naturally using the SAME language pattern as the user's input.
- If the user mixes Indonesian and English, your response should also naturally mix Indonesian and English.
- If the user includes Arabic words or phrases (e.g. Islamic expressions like "Alhamdulillah", "Insya Allah", or Arabic questions), preserve and use those Arabic expressions naturally in your response as well.
- Keep your response concise: maximum 3–5 sentences.
- Be informative, polite, and directly answer the question without repeating it.
- If unsure, be honest and say so.

Example:
User: "Aku lagi belajar machine learning, bisa explain apa itu neural network?"
Assistant: "Neural network itu basically sistem komputasi yang terinspirasi dari cara kerja otak manusia. Terdiri dari layers of nodes yang saling terhubung — input layer, hidden layers, dan output layer. Dengan training data yang cukup, network ini bisa learn to recognize patterns secara otomatis."
"""

# Mode 2: NORMALIZED RESPONSE
# Semua respons dinormalisasi ke Bahasa Indonesia baku yang jelas
SYSTEM_PROMPT_NORMALIZED = """
You are a responsive, intelligent, and fluent virtual assistant.
The user may speak in code-switching — mixing Indonesian, English, and Arabic.

Your task:
- Always respond in clear, polite, and standard Bahasa Indonesia regardless of what language(s) the user used.
- Translate or paraphrase any English or Arabic content from the user into Indonesian in your response.
- Keep your response concise: maximum 3–5 sentences.
- Be informative and directly answer the question without repeating it.
- If unsure, be honest and say so.

Example:
User: "Aku lagi belajar machine learning, bisa explain apa itu neural network?"
Assistant: "Neural network adalah sistem komputasi yang terinspirasi dari cara kerja otak manusia. Sistem ini terdiri dari lapisan-lapisan node yang saling terhubung, yaitu lapisan input, lapisan tersembunyi, dan lapisan output. Dengan data pelatihan yang cukup, jaringan ini dapat belajar mengenali pola secara otomatis."
"""

# Default mode (bisa diubah via .env: LLM_MODE=preserve_cs atau LLM_MODE=normalized)
DEFAULT_LLM_MODE = os.getenv("LLM_MODE", "normalized")

# Map mode string → system prompt
_SYSTEM_PROMPTS = {
    "preserve_cs": SYSTEM_PROMPT_PRESERVE_CS,
    "normalized":  SYSTEM_PROMPT_NORMALIZED,
}

def get_system_prompt(mode: str = None) -> str:
    """Ambil system prompt sesuai mode. Default ke DEFAULT_LLM_MODE."""
    m = mode or DEFAULT_LLM_MODE
    if m not in _SYSTEM_PROMPTS:
        print(f"[LLM] Mode '{m}' tidak dikenal, pakai default: {DEFAULT_LLM_MODE}")
        m = DEFAULT_LLM_MODE
    return _SYSTEM_PROMPTS[m]

# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

history_adapter = TypeAdapter(list[types.Content])

# ─────────────────────────────────────────────
# State aktif
# ─────────────────────────────────────────────
_current_key_idx   = 0
_current_model_idx = 0

# ─────────────────────────────────────────────
# Gemini helpers
# ─────────────────────────────────────────────
def _make_gemini_client(key_idx: int):
    return genai.Client(api_key=GEMINI_API_KEYS[key_idx])

def _make_config(mode: str = None):
    return types.GenerateContentConfig(system_instruction=get_system_prompt(mode))

def _make_gemini_chat(client, model: str, history=None, mode: str = None):
    cfg = _make_config(mode)
    if history:
        return client.chats.create(model=model, config=cfg, history=history)
    return client.chats.create(model=model, config=cfg)

# ─────────────────────────────────────────────
# Inisialisasi client & chat awal
# ─────────────────────────────────────────────
_client        = _make_gemini_client(0)
_current_model = GEMINI_MODELS[0]
chat           = _make_gemini_chat(_client, _current_model)
print(f"[LLM] Aktif → Gemini key #1, model: {_current_model}")
print(f"[LLM] Mode output: {DEFAULT_LLM_MODE}")

# ─────────────────────────────────────────────
# Simpan / muat history
# ─────────────────────────────────────────────
def export_chat_history(c) -> str:
    return history_adapter.dump_json(c.get_history()).decode("utf-8")

def save_chat_history(c):
    if c is None:
        return
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write(export_chat_history(c))

def load_chat_history(mode: str = None):

    if not os.path.exists(CHAT_HISTORY_FILE) or os.path.getsize(CHAT_HISTORY_FILE) == 0:
        return _make_gemini_chat(_client, _current_model, mode=mode)
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        json_str = f.read().strip()
    if not json_str:
        return _make_gemini_chat(_client, _current_model, mode=mode)
    try:
        history = history_adapter.validate_json(json_str)
        return _make_gemini_chat(_client, _current_model, history=history, mode=mode)
    except Exception as e:
        print(f"[LLM] Gagal load history: {e}")
        return _make_gemini_chat(_client, _current_model, mode=mode)

chat = load_chat_history()

# ─────────────────────────────────────────────
# Fallback logic
# ─────────────────────────────────────────────
def _next_fallback() -> bool:
    global _current_key_idx, _current_model_idx
    global _client, _current_model, chat

    next_model_idx = _current_model_idx + 1
    if next_model_idx < len(GEMINI_MODELS):
        _current_model_idx = next_model_idx
        _current_model     = GEMINI_MODELS[_current_model_idx]
        chat               = _make_gemini_chat(_client, _current_model)
        print(f"[LLM] Fallback → Gemini model: {_current_model} (key #{_current_key_idx + 1})")
        return True

    next_key_idx = _current_key_idx + 1
    if next_key_idx < len(GEMINI_API_KEYS):
        _current_key_idx   = next_key_idx
        _current_model_idx = 0
        _current_model     = GEMINI_MODELS[0]
        _client            = _make_gemini_client(_current_key_idx)
        chat               = _make_gemini_chat(_client, _current_model)
        print(f"[LLM] Fallback → Gemini key #{_current_key_idx + 1}, model: {_current_model}")
        return True

    print("[LLM] Semua Gemini key dan model sudah habis.")
    return False


def _parse_retry_wait(err_str: str) -> int:
    match = re.search(r"retryDelay.*?(\d+)s", err_str)
    if match:
        return int(match.group(1)) + 3
    return 10


# ─────────────────────────────────────────────
# Fungsi utama: generate response
# ─────────────────────────────────────────────
def generate_response(prompt: str, mode: str = None) -> str:
    """
    Generate respons dari LLM berdasarkan prompt.

    Args:
        prompt (str): Teks input dari user (sudah dinormalisasi / di-tag).
        mode (str): Mode output LLM.
                    "preserve_cs" → respons mempertahankan code-switching
                    "normalized"  → respons selalu Bahasa Indonesia baku
                    None          → pakai DEFAULT_LLM_MODE dari .env

    Returns:
        str: Teks respons dari LLM, atau pesan [ERROR] jika gagal.
    """
    global chat

    active_mode = mode or DEFAULT_LLM_MODE
    system_prompt = get_system_prompt(active_mode)
    print(f"[LLM] Mode: {active_mode}")

    tried = set()

    while True:
        combo = (_current_key_idx, _current_model_idx)
        if combo in tried:
            return "[ERROR] Semua API key dan model sudah dicoba, semuanya kena rate limit."
        tried.add(combo)

        try:
            
            # Buat sesi chat baru dengan system prompt sesuai mode
            # (tidak pakai chat global agar mode tidak tercampur antar request)
            cfg = types.GenerateContentConfig(system_instruction=system_prompt)
            session = _client.chats.create(
                model=_current_model,
                config=cfg,
                history=chat.get_history() if chat else [],
            )
            print(f"[LLM] Kirim ke Gemini key #{_current_key_idx + 1} / {_current_model} ...")
            response = session.send_message(prompt)
            # Update chat global dengan history terbaru
            chat = session
            save_chat_history(chat)
            result = response.text.strip()

            print(f"[LLM] Respons ({active_mode}): {result}")
            return result

        except Exception as e:
            err = str(e)

            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate_limit" in err.lower():
                wait = _parse_retry_wait(err)
                print(f"[LLM] Rate limit pada Gemini key #{_current_key_idx + 1} / {_current_model}. "
                      f"Tunggu {wait}s lalu coba fallback...")
                time.sleep(wait)
                if not _next_fallback():
                    return "[ERROR] Semua API key dan model sudah habis quota-nya."

            elif "404" in err or "NOT_FOUND" in err or "model_not_found" in err.lower():
                print(f"[LLM] Model {_current_model} tidak tersedia, fallback...")
                if not _next_fallback():
                    return "[ERROR] Semua model tidak tersedia."

            else:
                return f"[ERROR] {err}"