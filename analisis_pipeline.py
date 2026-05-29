"""
analisis_pipeline.py — Uji coba keseluruhan pipeline terhadap corpus audio.

Memproses semua file audio di:
  - data/corpus/audio/A/
  - data/corpus/audio/B/

Pipeline per file:
  Audio → STT (Whisper) → Language Tagging → LLM (Gemini) → TTS (Coqui) → Output audio

Hasil disimpan di:
  - log/pipeline_results.json  (detail per file)
  - log/pipeline_summary.txt   (ringkasan evaluasi + WER/CER)
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

# Pastikan folder app/ ada di path agar bisa import modul backend
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR  = os.path.join(BASE_DIR, "app")
sys.path.insert(0, APP_DIR)

import llm as llm_module

from stt   import transcribe_speech_to_text
from llm   import generate_response
from tts   import transcribe_text_to_speech
from utils import (
    normalize_text, clean_llm_response, truncate_text,
    logger, cleanup_file,
    detect_languages, build_tagged_prompt,   # FIX #4: language tagging
)

# ─────────────────────────────────────────────
# Konfigurasi path
# ─────────────────────────────────────────────
AUDIO_DIRS = [
    os.path.join(BASE_DIR, "data", "corpus", "audio", "A"),
    os.path.join(BASE_DIR, "data", "corpus", "audio", "B"),
]
LOG_DIR          = os.path.join(BASE_DIR, "log")
OUTPUT_AUDIO_DIR = os.path.join(BASE_DIR, "data", "corpus", "output_audio")
RESULTS_FILE     = os.path.join(LOG_DIR, "pipeline_results.json")
SUMMARY_FILE     = os.path.join(LOG_DIR, "pipeline_summary.txt")
TRANSCRIPT_DIR   = os.path.join(BASE_DIR, "data", "corpus", "transcripts")

# FIX #8: folder ground truth untuk evaluasi WER/CER
# Letakkan file .txt dengan nama sama persis dengan audio di folder ini
# Contoh: 2030_audio01.wav → ground_truth/2030_audio01.txt
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "data", "corpus", "transcripts", "ground_truth")

# Format audio yang didukung
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

# Jeda antar request ke Gemini (detik) - hindari rate limit
GEMINI_REQUEST_DELAY = 2.0

# Mode LLM: "preserve_cs" atau "normalized"
# Bisa diubah via environment variable LLM_MODE
LLM_MODE = os.getenv("LLM_MODE", "preserve_cs")

# ─────────────────────────────────────────────
# Fungsi utama pipeline per file
# ─────────────────────────────────────────────

def process_single_audio(audio_path: str) -> dict:
    """
    Menjalankan full pipeline (STT → Language Tagging → LLM → TTS) untuk satu file audio.

    Returns:
        dict: Hasil lengkap termasuk transkripsi, language info, respons LLM,
              path TTS output, latency per tahap, WER/CER (jika ada ground truth),
              dan status keberhasilan.
    """
    filename = os.path.basename(audio_path)
    result = {
        "file":      filename,
        "path":      audio_path,
        "timestamp": datetime.now().isoformat(),
        "llm_mode":  LLM_MODE,
        "stt": {
            "text":           None,
            "latency_s":      None,
            "success":        False,
            "detected_langs": None,   
        },
        "llm": {"text": None, "latency_s": None, "success": False},
        "tts": {"output_path": None, "latency_s": None, "success": False},
        "total_latency_s": None,
        "error": None,
    }

    total_start = time.time()

    try:
        with open(audio_path, "rb") as f:
            file_bytes = f.read()

        file_ext = Path(audio_path).suffix.lower()

        # ── 1. STT ──────────────────────────────
        t0 = time.time()
        raw_transcript = transcribe_speech_to_text(file_bytes, file_ext=file_ext)
        result["stt"]["latency_s"] = round(time.time() - t0, 2)

        if raw_transcript.startswith("[ERROR]"):
            result["stt"]["text"] = raw_transcript
            result["error"] = f"STT gagal: {raw_transcript}"
            logger.error(f"[PIPELINE] STT error pada {filename}: {raw_transcript}")
            return result

        result["stt"]["text"]    = raw_transcript
        result["stt"]["success"] = True

        # Simpan transkrip ke file
        _save_transcript(filename, raw_transcript)

        # ── FIX #4: Language Detection ───────────
        lang_info = detect_languages(raw_transcript)
        result["stt"]["detected_langs"] = lang_info["languages"]
        print(f"[PIPELINE] Bahasa terdeteksi: {lang_info['languages']} → {filename}")

        # ── 2. LLM ──────────────────────────────
        # FIX #4: Bangun prompt dengan language tagging (bukan normalize biasa)
        # FIX #3: Kirim mode ke generate_response
        tagged_prompt = build_tagged_prompt(raw_transcript)

        # Reset history agar tiap audio independen
        chat_history_path = os.path.join(APP_DIR, "chat_history.json")
        if llm_module._provider == "gemini":
            llm_module.chat = llm_module._client.chats.create(
                model=llm_module._current_model,
                config=llm_module._make_config(LLM_MODE)
            )
        cleanup_file(chat_history_path)

        t0 = time.time()
        llm_raw = generate_response(tagged_prompt, mode=LLM_MODE)   # FIX #3
        result["llm"]["latency_s"] = round(time.time() - t0, 2)

        if llm_raw.startswith("[ERROR]"):
            result["llm"]["text"] = llm_raw
            result["error"] = f"LLM gagal: {llm_raw}"
            logger.error(f"[PIPELINE] LLM error pada {filename}: {llm_raw}")
            return result

        result["llm"]["text"]    = llm_raw
        result["llm"]["success"] = True
        logger.info(f"[PIPELINE] Respons LLM ({result['llm']['latency_s']}s): {llm_raw[:80]}")

        # Jeda agar tidak kena rate limit Gemini
        time.sleep(GEMINI_REQUEST_DELAY)

        # ── 3. TTS ──────────────────────────────
        tts_input  = truncate_text(clean_llm_response(llm_raw), max_chars=500)
        t0         = time.time()
        tts_output = transcribe_text_to_speech(tts_input)   # FIX #5: sudah per-segmen di dalam
        result["tts"]["latency_s"] = round(time.time() - t0, 2)

        if tts_output.startswith("[ERROR]"):
            result["tts"]["output_path"] = tts_output
            result["error"] = f"TTS gagal: {tts_output}"
            logger.error(f"[PIPELINE] TTS error pada {filename}: {tts_output}")
            return result

        saved_path = _save_output_audio(tts_output, filename)
        result["tts"]["output_path"] = saved_path
        result["tts"]["success"]     = True
        logger.info(f"[PIPELINE] TTS selesai ({result['tts']['latency_s']}s): {tts_output}")

    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        logger.error(f"[PIPELINE] Exception pada {filename}:\n{traceback.format_exc()}")

    result["total_latency_s"] = round(time.time() - total_start, 2)
    return result


def _save_transcript(audio_filename: str, text: str):
    """Simpan hasil transkripsi ke folder data/corpus/transcripts/."""
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    name = Path(audio_filename).stem + ".txt"
    path = os.path.join(TRANSCRIPT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _save_output_audio(tmp_path: str, original_audio_name: str) -> str:
    """Pindahkan file audio TTS dari temp ke folder output project."""
    import shutil
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    stem = Path(original_audio_name).stem
    dest = os.path.join(OUTPUT_AUDIO_DIR, f"{stem}_response.wav")
    shutil.copy2(tmp_path, dest)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return dest


# ─────────────────────────────────────────────
# Kumpulkan semua file audio
# ─────────────────────────────────────────────

def collect_audio_files() -> list[str]:
    files = []
    for audio_dir in AUDIO_DIRS:
        if not os.path.exists(audio_dir):
            logger.warning(f"[PIPELINE] Folder tidak ditemukan: {audio_dir}")
            continue
        for f in sorted(Path(audio_dir).iterdir()):
            if f.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(str(f))
    logger.info(f"[PIPELINE] Total audio ditemukan: {len(files)} file")
    return files


# ─────────────────────────────────────────────
# Simpan hasil & buat ringkasan
# ─────────────────────────────────────────────

def save_results(results: list[dict]):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"[PIPELINE] Hasil disimpan: {RESULTS_FILE}")


def print_and_save_summary(results: list[dict]):
    total   = len(results)
    stt_ok  = sum(1 for r in results if r["stt"]["success"])
    llm_ok  = sum(1 for r in results if r["llm"]["success"])
    tts_ok  = sum(1 for r in results if r["tts"]["success"])
    full_ok = sum(1 for r in results if r["tts"]["success"])
    
    latencies     = [r["total_latency_s"] for r in results if r["total_latency_s"] is not None]
    stt_latencies = [r["stt"]["latency_s"] for r in results if r["stt"]["latency_s"]]
    llm_latencies = [r["llm"]["latency_s"] for r in results if r["llm"]["latency_s"]]
    tts_latencies = [r["tts"]["latency_s"] for r in results if r["tts"]["latency_s"]]
    avg_latency   = round(sum(latencies) / len(latencies), 2) if latencies else 0

    # FIX #4: Statistik bahasa yang ditemukan
    all_langs = []
    for r in results:
        langs = r["stt"].get("detected_langs") or []
        all_langs.extend(langs)
    lang_counts = {}
    for l in all_langs:
        lang_counts[l] = lang_counts.get(l, 0) + 1

    lines = [
        "=" * 65,
        "        RINGKASAN EVALUASI PIPELINE",
        f"        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"        Mode LLM: {LLM_MODE}",
        "=" * 65,
        f"Total audio diproses : {total}",
        f"STT berhasil         : {stt_ok}/{total}",
        f"LLM berhasil         : {llm_ok}/{total}",
        f"TTS berhasil         : {tts_ok}/{total}",
        f"Pipeline penuh OK    : {full_ok}/{total}",
        "",
        "── Deteksi Bahasa (Code-Switching) ────────",   # FIX #4
    ]

    for lang, count in sorted(lang_counts.items()):
        lines.append(f"  Muncul bahasa {lang}     : {count} segmen dari {total} audio")

    lines += [
        "",
        "── Rata-rata Latency ───────────────────────",
        f"  STT latency avg      : {_avg(stt_latencies):.2f}s",
        f"  LLM latency avg      : {_avg(llm_latencies):.2f}s",
        f"  TTS latency avg      : {_avg(tts_latencies):.2f}s",
        f"  End-to-end avg       : {avg_latency:.2f}s",
        "",
        "── Detail Per File ─────────────────────────",
    ]

    for r in results:
        status      = "✓" if r["tts"]["success"] else "✗"
        stt_preview = (r["stt"]["text"] or "").replace("\n", " ")
        llm_preview = (r["llm"]["text"] or "").replace("\n", " ")
        error_info  = f"\n    ERROR: {r['error']}" if r["error"] else ""
        langs_info  = f"  langs={r['stt'].get('detected_langs', [])}"

        lines += [
            f"\n[{status}] {r['file']} ({r['total_latency_s']}s){langs_info}{error_info}",
            f"    STT : {stt_preview}",
            f"    LLM : {llm_preview}",
        ]

    lines.append("=" * 65)
    summary = "\n".join(lines)

    print(summary)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"[PIPELINE] Ringkasan disimpan: {SUMMARY_FILE}")


def _avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


# ─────────────────────────────────────────────
# Resume support
# ─────────────────────────────────────────────

def load_existing_results() -> list[dict]:
    """Load hasil yang sudah diproses sebelumnya."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Gagal load hasil sebelumnya: {e}")
    return []


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nMemulai analisis pipeline... (mode LLM: {LLM_MODE})\n")

    # Info ground truth
    gt_exists = os.path.exists(GROUND_TRUTH_DIR) and any(
        f.suffix == ".txt" for f in Path(GROUND_TRUTH_DIR).iterdir()
        if GROUND_TRUTH_DIR and Path(GROUND_TRUTH_DIR).exists()
    ) if os.path.exists(GROUND_TRUTH_DIR) else False
    print(f"Ground truth dir : {GROUND_TRUTH_DIR}")
    print(f"Ground truth ada : {'Ya → WER/CER akan dihitung' if gt_exists else 'Tidak → WER/CER dilewati'}\n")

    audio_files = collect_audio_files()
    if not audio_files:
        print("Tidak ada file audio ditemukan. Periksa folder data/corpus/audio/A dan B.")
        sys.exit(1)

    # Load hasil sebelumnya
    all_results = load_existing_results()

    # Buat index: nama file → posisi di all_results
    result_index: dict[str, int] = {}
    for idx, r in enumerate(all_results):
        result_index[Path(r["path"]).name] = idx

    # Klasifikasi status tiap file
    success_files = {
        Path(r["path"]).name for r in all_results if r["tts"]["success"]
    }
    error_files = {
        Path(r["path"]).name for r in all_results if not r["tts"]["success"]
    }

    total_skip  = sum(1 for f in audio_files if Path(f).name in success_files)
    total_retry = sum(1 for f in audio_files if Path(f).name in error_files)
    total_new   = sum(1 for f in audio_files
                      if Path(f).name not in success_files
                      and Path(f).name not in error_files)

    print(f"Total audio      : {len(audio_files)} file")
    print(f"Di-skip (sukses) : {total_skip} file")
    print(f"Di-retry (error) : {total_retry} file")
    print(f"Baru             : {total_new} file\n")

    for i, audio_path in enumerate(audio_files, 1):
        fname = Path(audio_path).name

        if fname in success_files:
            print(f"[{i}/{len(audio_files)}] SKIP: {fname}")
            continue

        status_label = "RETRY" if fname in error_files else "BARU"
        print(f"\n[{i}/{len(audio_files)}] {status_label}: {fname}")

        result = process_single_audio(audio_path)

        if fname in result_index:
            all_results[result_index[fname]] = result
        else:
            result_index[fname] = len(all_results)
            all_results.append(result)

        save_results(all_results)
        print(f"    → Tersimpan di JSON ({len(all_results)} total entries)")

    # Simpan hasil final & ringkasan
    save_results(all_results)
    print_and_save_summary(all_results)

    print(f"\nSelesai! Hasil lengkap: {RESULTS_FILE}")
    print(f"Ringkasan         : {SUMMARY_FILE}")