import re
import json
import unicodedata
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Ground Truth per nomor soal (1–20)
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    1:  "aku mau book flight ke Jeddah minggu depan bisa bantu schedule",
    2:  "aku butuh travel umrah simple tapi include Madinah visit",
    3:  "can you help aku arrange transport dari Jeddah ke Madinah tomorrow",
    4:  "explain step by step cara apply visa Saudi dengan benar",
    5:  "ya akhi uridu book flight ila Jeddah al usbual qadim hal bisa bantu ajida afdal schedule wa rihlatan mubashirah",
    6:  "uridu arrange transport min Jeddah ila Madinah ghadan",
    7:  "book flight ke Jeddah lalu lanjut ke Madinah schedule terbaik kapan",
    8:  "uridu schedule trip min Jeddah ila Makkah bukra sabah",
    9:  "mumkin book transport min Makkah ila Madinah untuk besok",
    10: "apa perbedaan umrah dan hajj secara detail dalam Islam",
    11: "kenapa fasting di ramadan itu wajib bagi Muslim",
    12: "bagaimana proses visa Saudi untuk umrah dari Indonesia sekarang",
    13: "jelaskan step by step cara booking flight ke Jeddah secara online",
    14: "how to prepare dokumen umrah dari Indonesia dengan benar",
    15: "tolong buat checklist persiapan umrah termasuk barang wajib dibawa",
    16: "guide aku cara pilih hotel di Makkah dekat Haram dengan budget terbatas",
    17: "menurut kamu belajar bahasa Arab itu susah gak untuk pemula",
    18: "i feel overwhelmed dengan persiapan umrah ada tips sederhana",
    19: "ahyanan saya bingung mulai dari mana untuk umrah",
    20: "translate ke english aku mau pergi ke Makkah minggu depan",
}

# Kategori intent per soal
INTENT_CATEGORY = {
    1: "Commands", 2: "Commands", 3: "Commands", 4: "Instruction",
    5: "Commands", 6: "Commands", 7: "Commands", 8: "Commands",
    9: "Commands", 10: "Info-Seeking", 11: "Info-Seeking",
    12: "Info-Seeking", 13: "Instruction", 14: "Instruction",
    15: "Instruction", 16: "Instruction", 17: "Conversational",
    18: "Conversational", 19: "Conversational", 20: "Transformation",
}

# Kelompok bahasa per soal
LANG_GROUP = {
    1: "ID", 2: "ID", 3: "CS_EN_ID", 4: "CS_EN_ID",
    5: "CS_AR_ID", 6: "AR", 7: "ID", 8: "CS_AR_ID",
    9: "CS_AR_ID", 10: "ID", 11: "CS_EN_ID", 12: "ID",
    13: "ID", 14: "CS_EN_ID", 15: "ID", 16: "ID",
    17: "ID", 18: "CS_EN_ID", 19: "CS_AR_ID", 20: "ID",
}


# ---------------------------------------------------------------------------
# Fungsi normalisasi teks
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, strip tanda baca dasar, normalkan whitespace."""
    text = text.lower().strip()
    # Hapus karakter non-alphanumeric kecuali apostrof dan spasi
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    # Normalkan whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# WER menggunakan dynamic programming (edit distance pada level kata)
# ---------------------------------------------------------------------------
def compute_wer(reference: str, hypothesis: str) -> dict:
    """
    Hitung WER antara reference dan hypothesis.
    Returns dict dengan S (substitusi), D (deletion), I (insertion),
    N (jumlah kata ref), WER.
    """
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()

    n = len(ref_words)
    m = len(hyp_words)

    # dp[i][j] = edit distance ref[0:i] vs hyp[0:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # deletion
                                   dp[i][j - 1],      # insertion
                                   dp[i - 1][j - 1])  # substitution

    edit_dist = dp[n][m]
    wer = edit_dist / n if n > 0 else 0.0

    return {
        "edit_distance": edit_dist,
        "ref_word_count": n,
        "hyp_word_count": m,
        "wer": round(wer, 4),
        "wer_pct": round(wer * 100, 2),
    }


# ---------------------------------------------------------------------------
# CER menggunakan dynamic programming (edit distance pada level karakter)
# ---------------------------------------------------------------------------
def compute_cer(reference: str, hypothesis: str) -> dict:
    """
    Hitung CER antara reference dan hypothesis.
    Returns dict dengan edit distance karakter, panjang ref, CER.
    """
    ref_chars = list(normalize(reference))
    hyp_chars = list(normalize(hypothesis))

    n = len(ref_chars)
    m = len(hyp_chars)

    if n == 0:
        return {"edit_distance_char": m, "ref_char_count": 0,
                "hyp_char_count": m, "cer": 0.0, "cer_pct": 0.0}

    # Optimasi memori: gunakan dua baris saja
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (m + 1)

    edit_dist = prev[m]
    cer = edit_dist / n

    return {
        "edit_distance_char": edit_dist,
        "ref_char_count": n,
        "hyp_char_count": m,
        "cer": round(cer, 4),
        "cer_pct": round(cer * 100, 2),
    }


# ---------------------------------------------------------------------------
# Ekstrak nomor soal dari nama file
# ---------------------------------------------------------------------------
def extract_audio_number(filename: str) -> int | None:
    """
    Ekstrak nomor soal (1-20) dari nama file.
    Mendukung: audio1, audio01, Audio01, audio1(1), audio1.m4a, dll.
    """
    # Normalkan: lowercase, hapus ekstensi, hapus (1) duplikat
    name = filename.lower()
    name = re.sub(r'\.(m4a|wav)$', '', name)  # hapus .wav
    name = re.sub(r'\.m4a$', '', name)         # hapus .m4a tersisa
    name = re.sub(r'\(\d+\)$', '', name)        # hapus (1), (2), dst
    name = name.strip()

    # Cari pola audioNN di akhir
    m = re.search(r'audio0*(\d+)$', name)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 20:
            return num
    return None


# ---------------------------------------------------------------------------
# Parse pipeline_summary.txt
# ---------------------------------------------------------------------------
def parse_pipeline_summary(filepath: str) -> list[dict]:
    """
    Parse semua entri dari pipeline_summary.txt.
    Returns list of dicts: {filename, npm, audio_num, stt, llm}
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Pattern: [✓] FILENAME (XXs) ... STT : ... LLM : ...
    pattern = r'\[✓\] (\S+\.wav) \([\d.]+s\)[^\n]*\n\s+STT : (.+?)\n\s+LLM : (.+?)(?=\n\n\[|$)'
    entries_raw = re.findall(pattern, content, re.DOTALL)

    entries = []
    for filename, stt, llm in entries_raw:
        filename = filename.strip()
        stt = stt.strip()
        llm = llm.strip()

        # Ekstrak NPM (4 digit di awal)
        npm_match = re.match(r'(\d{4})_', filename)
        npm = npm_match.group(1) if npm_match else "unknown"

        # Ekstrak nomor soal
        audio_num = extract_audio_number(filename)

        entries.append({
            "filename": filename,
            "npm": npm,
            "audio_num": audio_num,
            "stt": stt,
            "llm": llm,
        })

    return entries


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(summary_path: str,
                 output_json: str = "wer_cer_results.json",
                 output_txt: str = "wer_cer_summary.txt"):

    print(f"Membaca: {summary_path}")
    entries = parse_pipeline_summary(summary_path)
    print(f"Total entri terparsing: {len(entries)}")

    results = []
    skipped = []

    # Hitung WER & CER per entri
    for entry in entries:
        num = entry["audio_num"]
        if num is None or num not in GROUND_TRUTH:
            skipped.append(entry["filename"])
            continue

        ref = GROUND_TRUTH[num]
        hyp = entry["stt"]

        wer_result = compute_wer(ref, hyp)
        cer_result = compute_cer(ref, hyp)

        result = {
            "filename": entry["filename"],
            "npm": entry["npm"],
            "audio_num": num,
            "intent_category": INTENT_CATEGORY.get(num, "unknown"),
            "lang_group": LANG_GROUP.get(num, "unknown"),
            "ground_truth": ref,
            "stt_output": hyp,
            "wer": wer_result["wer"],
            "wer_pct": wer_result["wer_pct"],
            "cer": cer_result["cer"],
            "cer_pct": cer_result["cer_pct"],
            "wer_edit_distance": wer_result["edit_distance"],
            "cer_edit_distance": cer_result["edit_distance_char"],
            "ref_word_count": wer_result["ref_word_count"],
            "ref_char_count": cer_result["ref_char_count"],
        }
        results.append(result)

    print(f"Entri berhasil dihitung: {len(results)}")
    print(f"Entri dilewati (nomor soal tidak dikenali): {len(skipped)}")

    # ------------------------------------------------------------------
    # Simpan JSON
    # ------------------------------------------------------------------
    output = {
        "meta": {
            "total_entries": len(results),
            "total_skipped": len(skipped),
            "skipped_files": skipped,
            "ground_truth": GROUND_TRUTH,
        },
        "per_file": results,
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"JSON tersimpan: {output_json}")

    # ------------------------------------------------------------------
    # Buat ringkasan statistik
    # ------------------------------------------------------------------
    lines = []

    def hline(char="─", w=78):
        lines.append(char * w)

    def sec(title):
        hline("═")
        lines.append(f"  {title}")
        hline("═")

    lines.append("LAPORAN WER & CER - UAS Praktikum NLP 2025/2026 Genap")
    lines.append(f"Total file dianalisis : {len(results)}")
    lines.append(f"Total file dilewati   : {len(skipped)}")
    lines.append("")

    # ── Statistik Global ──────────────────────────────────────────────
    sec("STATISTIK GLOBAL")
    all_wer = [r["wer_pct"] for r in results]
    all_cer = [r["cer_pct"] for r in results]

    def stats(values, label):
        avg = sum(values) / len(values)
        mn  = min(values)
        mx  = max(values)
        med = sorted(values)[len(values) // 2]
        good_50 = sum(1 for v in values if v < 50)
        good_25 = sum(1 for v in values if v < 25)
        lines.append(f"{label}")
        lines.append(f"  Rata-rata : {avg:.2f}%")
        lines.append(f"  Median    : {med:.2f}%")
        lines.append(f"  Min       : {mn:.2f}%")
        lines.append(f"  Max       : {mx:.2f}%")
        lines.append(f"  Good <50% : {good_50}/{len(values)} ({good_50/len(values)*100:.1f}%)")
        lines.append(f"  Good <25% : {good_25}/{len(values)} ({good_25/len(values)*100:.1f}%)")
        lines.append("")

    stats(all_wer, "WER (Word Error Rate)")
    stats(all_cer, "CER (Character Error Rate)")

    # ── Per Nomor Soal ────────────────────────────────────────────────
    sec("WER & CER PER NOMOR SOAL")
    by_soal = defaultdict(list)
    for r in results:
        by_soal[r["audio_num"]].append(r)

    header = f"{'Q':>3} | {'N':>4} | {'Lang':>10} | {'Intent':>16} | {'WER avg':>8} | {'WER min':>7} | {'WER max':>7} | {'CER avg':>8} | {'CER min':>7} | {'CER max':>7}"
    lines.append(header)
    hline()

    for num in range(1, 21):
        rows = by_soal.get(num, [])
        if not rows:
            continue
        wers = [r["wer_pct"] for r in rows]
        cers = [r["cer_pct"] for r in rows]
        lang = LANG_GROUP.get(num, "?")
        intent = INTENT_CATEGORY.get(num, "?")
        lines.append(
            f"{num:>3} | {len(rows):>4} | {lang:>10} | {intent:>16} | "
            f"{sum(wers)/len(wers):>7.2f}% | {min(wers):>6.2f}% | {max(wers):>6.2f}% | "
            f"{sum(cers)/len(cers):>7.2f}% | {min(cers):>6.2f}% | {max(cers):>6.2f}%"
        )

    lines.append("")

    # ── Per Kelompok Bahasa ───────────────────────────────────────────
    sec("WER & CER PER KELOMPOK BAHASA")
    by_lang = defaultdict(list)
    for r in results:
        by_lang[r["lang_group"]].append(r)

    lang_order = ["ID", "CS_EN_ID", "CS_AR_ID", "AR"]
    header2 = f"{'Bahasa':>12} | {'N':>4} | {'WER avg':>8} | {'WER med':>8} | {'WER min':>7} | {'WER max':>7} | {'CER avg':>8} | {'CER med':>8}"
    lines.append(header2)
    hline()
    for lang in lang_order:
        rows = by_lang.get(lang, [])
        if not rows:
            continue
        wers = sorted([r["wer_pct"] for r in rows])
        cers = sorted([r["cer_pct"] for r in rows])
        w_med = wers[len(wers) // 2]
        c_med = cers[len(cers) // 2]
        lines.append(
            f"{lang:>12} | {len(rows):>4} | {sum(wers)/len(wers):>7.2f}% | {w_med:>7.2f}% | "
            f"{min(wers):>6.2f}% | {max(wers):>6.2f}% | "
            f"{sum(cers)/len(cers):>7.2f}% | {c_med:>7.2f}%"
        )
    lines.append("")

    # ── Per Kategori Intent ───────────────────────────────────────────
    sec("WER & CER PER KATEGORI INTENT")
    by_intent = defaultdict(list)
    for r in results:
        by_intent[r["intent_category"]].append(r)

    header3 = f"{'Kategori':>16} | {'N':>4} | {'WER avg':>8} | {'WER min':>7} | {'WER max':>7} | {'CER avg':>8} | {'CER min':>7} | {'CER max':>7}"
    lines.append(header3)
    hline()
    for cat in ["Commands", "Info-Seeking", "Instruction", "Conversational", "Transformation"]:
        rows = by_intent.get(cat, [])
        if not rows:
            continue
        wers = [r["wer_pct"] for r in rows]
        cers = [r["cer_pct"] for r in rows]
        lines.append(
            f"{cat:>16} | {len(rows):>4} | {sum(wers)/len(wers):>7.2f}% | {min(wers):>6.2f}% | {max(wers):>6.2f}% | "
            f"{sum(cers)/len(cers):>7.2f}% | {min(cers):>6.2f}% | {max(cers):>6.2f}%"
        )
    lines.append("")

    # ── Per Mahasiswa (NPM) ───────────────────────────────────────────
    sec("WER & CER PER MAHASISWA (NPM)")
    by_npm = defaultdict(list)
    for r in results:
        by_npm[r["npm"]].append(r)

    header4 = f"{'NPM':>6} | {'N':>3} | {'WER avg':>8} | {'WER min':>7} | {'WER max':>7} | {'CER avg':>8} | {'CER min':>7} | {'CER max':>7}"
    lines.append(header4)
    hline()
    npm_stats = []
    for npm, rows in by_npm.items():
        wers = [r["wer_pct"] for r in rows]
        cers = [r["cer_pct"] for r in rows]
        npm_stats.append((npm, len(rows), sum(wers)/len(wers), min(wers), max(wers),
                          sum(cers)/len(cers), min(cers), max(cers)))

    # Sort by WER avg ascending (terbaik duluan)
    npm_stats.sort(key=lambda x: x[2])
    for npm, n, w_avg, w_min, w_max, c_avg, c_min, c_max in npm_stats:
        lines.append(
            f"{npm:>6} | {n:>3} | {w_avg:>7.2f}% | {w_min:>6.2f}% | {w_max:>6.2f}% | "
            f"{c_avg:>7.2f}% | {c_min:>6.2f}% | {c_max:>6.2f}%"
        )
    lines.append("")

    # ── Top 10 Terbaik & Terburuk per File ───────────────────────────
    sec("TOP 10 FILE — WER TERBAIK (TERENDAH)")
    sorted_best = sorted(results, key=lambda r: r["wer_pct"])[:10]
    for i, r in enumerate(sorted_best, 1):
        lines.append(f"  {i:>2}. [{r['filename']}]  WER={r['wer_pct']:.2f}%  CER={r['cer_pct']:.2f}%")
        lines.append(f"      STT : {r['stt_output'][:80]}")
        lines.append(f"      REF : {r['ground_truth'][:80]}")
        lines.append("")

    sec("TOP 10 FILE — WER TERBURUK (TERTINGGI)")
    sorted_worst = sorted(results, key=lambda r: r["wer_pct"], reverse=True)[:10]
    for i, r in enumerate(sorted_worst, 1):
        lines.append(f"  {i:>2}. [{r['filename']}]  WER={r['wer_pct']:.2f}%  CER={r['cer_pct']:.2f}%")
        lines.append(f"      STT : {r['stt_output'][:80]}")
        lines.append(f"      REF : {r['ground_truth'][:80]}")
        lines.append("")

    # Tulis ke file
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Summary tersimpan: {output_txt}")

    # Cetak ringkasan ke console
    print("\n" + "="*50)
    print("RINGKASAN CEPAT")
    print("="*50)
    print(f"Total file   : {len(results)}")
    print(f"WER rata-rata: {sum(all_wer)/len(all_wer):.2f}%")
    print(f"CER rata-rata: {sum(all_cer)/len(all_cer):.2f}%")
    print(f"WER < 50%    : {sum(1 for v in all_wer if v < 50)}/{len(all_wer)} ({sum(1 for v in all_wer if v < 50)/len(all_wer)*100:.1f}%)")
    print(f"WER < 25%    : {sum(1 for v in all_wer if v < 25)}/{len(all_wer)} ({sum(1 for v in all_wer if v < 25)/len(all_wer)*100:.1f}%)")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os

    # Cari pipeline_summary.txt — bisa dipass sebagai argumen atau di direktori yang sama
    if len(sys.argv) > 1:
        summary_path = sys.argv[1]
    else:
        # Coba beberapa lokasi umum
        candidates = [
            "pipeline_summary.txt",
            "log/pipeline_summary.txt",
            os.path.join(os.path.dirname(__file__), "pipeline_summary.txt"),
        ]
        summary_path = None
        for c in candidates:
            if os.path.exists(c):
                summary_path = c
                break

        if summary_path is None:
            print("ERROR: pipeline_summary.txt tidak ditemukan.")
            print("Penggunaan: python wer_cer_analysis.py [path/ke/pipeline_summary.txt]")
            sys.exit(1)

    run_analysis(
        summary_path=summary_path,
        output_json="wer_cer_results.json",
        output_txt="wer_cer_summary.txt",
    )