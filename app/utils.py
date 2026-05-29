import os
import re
import uuid
import tempfile
import logging
from pathlib import Path

# ─────────────────────────────────────────────
# Konfigurasi logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("voice_chatbot")


# ─────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────

def save_upload_to_temp(file_bytes: bytes, suffix: str = ".wav") -> str:
    """
    Menyimpan bytes audio yang diunggah ke file temporer.

    Args:
        file_bytes (bytes): Isi file audio.
        suffix (str): Ekstensi file, default ".wav".

    Returns:
        str: Path absolut ke file temporer yang dibuat.
    """
    tmp_dir = tempfile.gettempdir()
    filename = f"upload_{uuid.uuid4()}{suffix}"
    path = os.path.join(tmp_dir, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Audio disimpan sementara di: {path}")
    return path


def cleanup_file(path: str) -> None:
    """
    Menghapus file temporer setelah diproses.

    Args:
        path (str): Path ke file yang akan dihapus.
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info(f"File temporer dihapus: {path}")
    except OSError as e:
        logger.warning(f"Gagal menghapus file {path}: {e}")


def is_valid_audio_file(path: str) -> bool:
    """
    Memeriksa apakah file audio ada dan tidak kosong.

    Args:
        path (str): Path ke file audio.

    Returns:
        bool: True jika valid, False jika tidak.
    """
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


# ─────────────────────────────────────────────
# Teks utilities
# ─────────────────────────────────────────────

_ALLOWED_CHARS_PATTERN = re.compile(
    r"[^\x20-\x7E"           # ASCII printable (Latin, angka, tanda baca)
    r"\u00C0-\u024F"         # Latin Extended (diakritik, aksen)
    r"\u0080-\u00BF"         # Latin-1 Supplement
    r"\u0600-\u06FF"         # Arab dasar
    r"\u0750-\u077F"         # Arab Supplement
    r"\u08A0-\u08FF"         # Arab Extended-A
    r"\uFB50-\uFDFF"         # Arab Presentation Forms-A
    r"\uFE70-\uFEFF"         # Arab Presentation Forms-B
    r"\n]",
    re.UNICODE
)


def normalize_text(text: str) -> str:
    """
    Normalisasi teks hasil transkripsi sebelum dikirim ke LLM.

    Langkah-langkah:
    - Hapus karakter kontrol non-printable, TAPI pertahankan karakter Arab
    - Ganti newline dengan spasi
    - Hapus spasi berlebih

    FIX #9: regex sebelumnya membuang semua karakter Arab (U+0600-U+06FF).
    Sekarang karakter Arab, Latin Extended, dan ASCII printable semua dipertahankan.

    Args:
        text (str): Teks mentah dari STT (bisa mengandung ID/EN/AR).

    Returns:
        str: Teks yang sudah dinormalisasi, karakter Arab tetap utuh.
    """
    if not text:
        return ""

    # Hapus karakter non-printable; Arab, Latin, ASCII dipertahankan
    text = _ALLOWED_CHARS_PATTERN.sub("", text)

    # Ganti baris baru dengan spasi
    text = text.replace("\n", " ")

    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─────────────────────────────────────────────
# Language Tagging untuk Code-Switching
# ─────────────────────────────────────────────

# Pola deteksi bahasa berdasarkan karakter
_ARABIC_PATTERN  = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+")
_ENGLISH_PATTERN = re.compile(r"\b[a-zA-Z]+\b")

# Kata-kata umum Bahasa Indonesia yang sering muncul (untuk disambiguasi dari EN)
_INDONESIAN_WORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu",
    "ada", "tidak", "bisa", "akan", "sudah", "saya", "aku", "kamu", "dia",
    "kita", "mereka", "kami", "adalah", "pada", "juga", "lebih", "sangat",
    "karena", "agar", "supaya", "kalau", "jika", "maka", "tapi", "tetapi",
    "atau", "hanya", "saja", "sudah", "belum", "lagi", "masih", "punya",
    "boleh", "harus", "perlu", "ingin", "mau", "sedang", "sudah", "pernah",
    "baru", "lalu", "kemudian", "setelah", "sebelum", "ketika", "saat",
    "apa", "siapa", "dimana", "kapan", "bagaimana", "mengapa", "berapa",
}


def detect_languages(text: str) -> dict:
    """
    Deteksi bahasa yang ada dalam teks code-switching ID-EN-AR.

    Args:
        text (str): Teks transkripsi yang mungkin mengandung campuran bahasa.

    Returns:
        dict: {
            "has_arabic": bool,
            "has_english": bool,
            "has_indonesian": bool,
            "languages": list[str],   # e.g. ["ID", "EN", "AR"]
            "arabic_segments": list[str],
            "english_segments": list[str],
        }
    """
    languages = []

    # Cek Arab
    arabic_matches = _ARABIC_PATTERN.findall(text)
    has_arabic = len(arabic_matches) > 0
    if has_arabic:
        languages.append("AR")

    # Cek kata Latin — pisahkan EN vs ID
    latin_words = _ENGLISH_PATTERN.findall(text)
    latin_lower = [w.lower() for w in latin_words]

    # Anggap ada ID jika ada kata Indonesia yang dikenali, atau mayoritas kata Latin ada di kamus ID
    id_hits = sum(1 for w in latin_lower if w in _INDONESIAN_WORDS)
    non_id_latin = [w for w in latin_lower if w not in _INDONESIAN_WORDS and len(w) > 1]

    has_indonesian = id_hits > 0 or (len(latin_words) > 0 and not has_arabic and len(non_id_latin) < len(latin_lower) * 0.8)
    has_english    = len(non_id_latin) > 0

    if has_indonesian:
        languages.append("ID")
    if has_english:
        languages.append("EN")

    # Jika tidak ada yang terdeteksi tapi ada teks Latin, default ke ID
    if not languages and len(latin_words) > 0:
        languages = ["ID"]
        has_indonesian = True

    return {
        "has_arabic":        has_arabic,
        "has_english":       has_english,
        "has_indonesian":    has_indonesian,
        "languages":         languages,
        "arabic_segments":   arabic_matches,
        "english_segments":  non_id_latin,
    }


def tag_languages(text: str) -> str:
    """
    Tambahkan tag bahasa ke teks untuk memperjelas konteks ke LLM.
    Format: [AR: <kata Arab>] dan [EN: <kata Inggris>] ditandai inline.

    Contoh input : "Hari ini saya belajar machine learning dan الذكاء الاصطناعي"
    Contoh output: "Hari ini saya belajar [EN: machine learning] dan [AR: الذكاء الاصطناعي]"

    Args:
        text (str): Teks asli code-switching.

    Returns:
        str: Teks dengan tag bahasa inline.
    """
    # Tag segmen Arab
    tagged = _ARABIC_PATTERN.sub(lambda m: f"[AR: {m.group()}]", text)

    # Tag kata Inggris (yang bukan kata Indonesia)
    def _tag_en(m):
        word = m.group()
        if word.lower() not in _INDONESIAN_WORDS and len(word) > 1:
            return f"[EN: {word}]"
        return word

    tagged = _ENGLISH_PATTERN.sub(_tag_en, tagged)
    return tagged


def build_tagged_prompt(raw_transcript: str) -> str:
    """
    Buat prompt lengkap dengan konteks language tagging untuk dikirim ke LLM.
    Menggabungkan deteksi bahasa + tagging menjadi prompt yang informatif.

    Args:
        raw_transcript (str): Teks transkripsi mentah dari STT.

    Returns:
        str: Prompt siap kirim ke LLM dengan konteks code-switching.
    """
    normalized = normalize_text(raw_transcript)
    lang_info  = detect_languages(normalized)
    langs      = lang_info["languages"]

    # Kalau pure monolingual, tidak perlu tagging
    if len(langs) <= 1:
        return normalized

    # Tambah konteks ke LLM
    lang_label = "+".join(langs)  # e.g. "ID+EN+AR"
    tagged_text = tag_languages(normalized)

    prompt = (
        f"[Konteks: ujaran code-switching {lang_label}]\n"
        f"{tagged_text}"
    )
    logger.info(f"[build_tagged_prompt] Bahasa terdeteksi: {langs}")
    return prompt


def clean_llm_response(text: str) -> str:
    """
    Membersihkan output LLM sebelum dikirim ke TTS.

    - Hapus markdown seperti **bold**, _italic_, ``` kode ```
    - Hapus karakter tanda baca berlebih
    - Pastikan tidak ada newline

    Args:
        text (str): Teks respons dari LLM.

    Returns:
        str: Teks bersih siap untuk TTS.
    """
    if not text:
        return ""

    # Hapus blok kode markdown
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Hapus bold/italic markdown
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)

    # Hapus header markdown
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Hapus bullet points
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)

    # Ganti newline dengan spasi
    text = text.replace("\n", " ")

    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    logger.info(f"[clean_llm_response] Output: {text[:80]}...")
    return text


def truncate_text(text: str, max_chars: int = 500) -> str:
    """
    Memotong teks agar tidak terlalu panjang untuk TTS.
    Pemotongan dilakukan pada batas kalimat terdekat.

    Args:
        text (str): Teks yang akan dipotong.
        max_chars (int): Batas maksimum karakter.

    Returns:
        str: Teks yang sudah dipotong.
    """
    if len(text) <= max_chars:
        return text

    # Cari posisi akhir kalimat sebelum batas
    cutoff = text[:max_chars]
    last_period = max(cutoff.rfind("."), cutoff.rfind("!"), cutoff.rfind("?"))

    if last_period > 0:
        return text[: last_period + 1]

    return cutoff.strip() + "..."