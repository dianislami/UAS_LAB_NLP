# 🎙️ Voice Chatbot - UAS PRAKTIKUM NLP 2025/2026

## Nama: Dian Islami
## NPM: 2308107010048

Sistem *speech-to-speech* end-to-end yang mendukung ujaran *code-switching* Bahasa Indonesia, Inggris, dan Arab. Pipeline: **Whisper STT → Gemini LLM → Coqui TTS**.

---

## Struktur Folder

```
voice_chatbot_project/
├── app/
│   ├── main.py             ← FastAPI backend (entry point server)
│   ├── stt.py              ← Speech-to-Text (openai-whisper)
│   ├── llm.py              ← LLM (Google Gemini API)
│   ├── tts.py              ← Text-to-Speech (Coqui TTS)
│   ├── utils.py            ← Fungsi pembantu (normalisasi, cleanup, dll)
│   └── coqui_utils/
│       ├── checkpoint_1260000-inference.pth
│       ├── config.json
│       └── speakers.pth
├── gradio_app/
│   └── app.py              ← Frontend Gradio
├── data/
│   └── corpus/
│       ├── audio/
│       │   ├── A/          ← Audio kelas A (.wav)
│       │   └── B/          ← Audio kelas B (.wav)
│       └── transcripts/    ← Hasil transkripsi
│       └── output_audio/   ← Output audio TTS per audio
├── log/
│   ├── pipeline_results.json   ← Hasil detail per file
│   └── pipeline_summary.txt    ← Ringkasan evaluasi pipeline
├── analisis_pipeline.py    ← Script batch processing semua audio
├── .env                    ← API key
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Awal (Lakukan Sekali)

### 1. Buat Virtual Environment

```bash
python3 -m venv env

# Linux / macOS
source env/bin/activate

# Windows
env\Scripts\activate
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

> **Catatan penting:** Setelah install coqui-tts, sesuaikan versi transformers:
> ```bash
> pip install transformers==5.0
> ```

### 3. Buat File `.env`

Buat file `.env` di root proyek:
```
GEMINI_API_KEY=AIza...isi_api_key_kamu_disini...
WHISPER_MODEL=large-v3-turbo
```

Dapatkan API key Gemini di: https://aistudio.google.com

### 4. Siapkan Model Coqui TTS

Download tiga file dari [indonesian-tts v1.2](https://github.com/Wikidepia/indonesian-tts/releases/tag/v1.2):
- `checkpoint_1260000-inference.pth`
- `config.json`
- `speakers.pth`

Letakkan di `app/coqui_utils/`.

---

## Cara Menjalankan

### Mode 1: Chatbot Real-Time (Mic → Suara)

Jalankan **dua terminal secara bersamaan**:

**Terminal 1: Backend:**
```bash
# Dari root folder proyek
source env/bin/activate       # (Linux/macOS)
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2: Frontend:**
```bash
source env/bin/activate       # (Linux/macOS)
cd gradio_app
python app.py
```

Buka browser: **http://localhost:7860**

**Cara pakai:**
1. Klik ikon mikrofon dan rekam pertanyaan
2. Klik **"Kirim"**
3. Tunggu proses pipeline
4. Respons suara diputar otomatis, teks transkripsi & respons muncul di panel kanan

---

### Mode 2 - Batch Processing Audio (data/corpus/audio/A dan B)

Pastikan file audio sudah ada di folder yang benar:
```
data/corpus/audio/A/xxxx_audio01.wav
data/corpus/audio/A/xxxx_audio02.wav
data/corpus/audio/B/yyyy_audio01.wav
...
```

Jalankan dari root folder:
```bash
source env/bin/activate
python analisis_pipeline.py
```

Hasil akan tersimpan di:
- `log/pipeline_results.json` - detail lengkap setiap file
- `log/pipeline_summary.txt` - ringkasan evaluasi
- `data/corpus/transcripts/` - hasil transkripsi per file

---

## Pilihan Model Whisper

Edit `.env` sesuai kemampuan perangkat:

| Model | VRAM/RAM | Akurasi |
|-------|----------|---------|
| `tiny` | ~273 MB | Rendah |
| `base` | ~388 MB | Cukup |
| `small` | ~852 MB | Sedang |
| `medium` | ~2.1 GB | Baik |
| `large-v3` | ~3.9 GB | Terbaik |

```
WHISPER_MODEL=large-v3-turbo
```

---

## Troubleshooting

**Error: `ModuleNotFoundError: No module named 'whisper'`**
```bash
pip install openai-whisper
```

**Error: `GEMINI_API_KEY not found`**
- Pastikan file `.env` ada di root proyek
- Pastikan nama variabel: `GEMINI_API_KEY=...`

**Error TTS: `transformers` version conflict**
```bash
pip install transformers==5.0
```

**Gemini rate limit (429)**
- Cek RPM/RPD model di Google AI Studio
- Naikkan nilai `GEMINI_REQUEST_DELAY` di `analisis_pipeline.py` (default: 2 detik)

**Audio tidak terbaca di pipeline batch**
- Pastikan format `.wav` dan ada di folder `data/corpus/audio/A` atau `B`
- Nama file harus sesuai konvensi: `{id}_{utteranceid}.wav`

---
