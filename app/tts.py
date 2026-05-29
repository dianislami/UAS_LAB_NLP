import os
import re
import uuid
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COQUI_DIR = os.path.join(BASE_DIR, "coqui_utils")

COQUI_MODEL_PATH  = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")
COQUI_SPEAKER     = "wibowo"


# ─────────────────────────────────────────────
# Singleton TTS instance — load sekali, pakai terus
# ─────────────────────────────────────────────
_tts_instance = None

def _get_tts():
    global _tts_instance
    if _tts_instance is None:
        print("[TTS] Loading model... (hanya sekali)")
        from TTS.api import TTS as CoquiTTS
        _tts_instance = CoquiTTS(
            model_path=COQUI_MODEL_PATH,
            config_path=COQUI_CONFIG_PATH,
            gpu=False,
        )
        print("[TTS] Model siap.")
    return _tts_instance


# ─────────────────────────────────────────────
# Deteksi Arab
# ─────────────────────────────────────────────
_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)


def _has_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


# ─────────────────────────────────────────────
# Phoneme conversion — Bahasa Indonesia
# ─────────────────────────────────────────────
def _id_text_to_phoneme(text: str) -> str:
    text = text.lower()

    # Digraf dulu — urutan penting
    text = text.replace("ng", "ŋ")
    text = text.replace("ny", "ɲ")
    text = text.replace("sy", "ʃ")
    text = text.replace("kh", "x")
    text = text.replace("gh", "ɡ")
    
    # Konsonan
    text = text.replace("c", "tʃ")
    text = text.replace("j", "dʒ")
    text = text.replace("q", "k")
    text = text.replace("v", "f")
    text = text.replace("z", "s")

    # g — ˈɡ di awal kata, ɡ di tengah
    text = re.sub(r'(?<![a-zəɔɛɪʊŋɲʃɡʒ])g', 'ˈɡ', text)
    text = text.replace("g", "ɡ")

    # y — jj di awal kata, j di tengah/akhir
    text = re.sub(r'(?<![a-zəɔɛɪʊŋɲʃɡʒ])y', 'jj', text)
    text = text.replace("y", "j")

    # Vokal
    text = text.replace("e", "ə")
    text = text.replace("o", "ɔ")

    # Bersihkan karakter non-vocab
    text = (text
            .replace('\u201c', '').replace('\u201d', '')
            .replace('\u2018', '').replace('\u2019', ''))
    text = text.replace('-', ' ').replace('_', ' ')
    text = re.sub(r' +', ' ', text).strip()

    return text


# ─────────────────────────────────────────────
# Transliterasi Arab → Latin (approx)
# ─────────────────────────────────────────────
def _arabic_to_latin_approx(text: str) -> str:
    ar_to_latin = {
        'ا': 'a', 'أ': 'a', 'إ': 'i', 'آ': 'aa',
        'ب': 'b', 'ت': 't', 'ث': 'ts', 'ج': 'j',
        'ح': 'h', 'خ': 'kh', 'د': 'd', 'ذ': 'dz',
        'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'sy',
        'ص': 's', 'ض': 'd', 'ط': 't', 'ظ': 'z',
        'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
        'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n',
        'ه': 'h', 'و': 'w', 'ي': 'y', 'ى': 'a',
        'ة': 'h', 'ء': '', 'ئ': 'i', 'ؤ': 'u',
        '\u064B': '', '\u064C': '', '\u064D': '',
        '\u064E': 'a', '\u064F': 'u', '\u0650': 'i',
        '\u0651': '', '\u0652': '',
    }
    result = ""
    for char in text:
        result += ar_to_latin.get(char, char if char.isascii() else '')
    return result.strip()


# ─────────────────────────────────────────────
# Segmentasi HANYA jika ada Arab
# ─────────────────────────────────────────────
def prepare_tts_text(text: str) -> str:
    if not _has_arabic(text):
        result = _id_text_to_phoneme(text)
        print(f"[TTS] Teks setelah prepare (no Arabic): {result[:100]}")
        return result
    ...
    print(f"[TTS] Teks mengandung Arab, segmentasi Arab/non-Arab...")
    parts = []
    remaining = text

    while remaining:
        ar_match = _ARABIC_RE.search(remaining)
        if ar_match:
            before = remaining[:ar_match.start()]
            if before.strip():
                parts.append(_id_text_to_phoneme(before))

            arab_latin = _arabic_to_latin_approx(ar_match.group())
            if arab_latin:
                parts.append(_id_text_to_phoneme(arab_latin))

            remaining = remaining[ar_match.end():]
        else:
            if remaining.strip():
                parts.append(_id_text_to_phoneme(remaining))
            break

    result = " ".join(parts)
    result = re.sub(r' +', ' ', result).strip()
    print(f"[TTS] Teks setelah prepare (with Arabic): {result[:100]}")
    return result


# ─────────────────────────────────────────────
# Public function — dipanggil dari luar
# ─────────────────────────────────────────────
def transcribe_text_to_speech(text: str) -> str:
    prepared = prepare_tts_text(text)

    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")

    tts = _get_tts()
    tts.tts_to_file(
        text=prepared,
        speaker=COQUI_SPEAKER,
        file_path=output_path,
    )
    print(f"  TTS → selesai ({os.path.basename(output_path)})")
    return output_path