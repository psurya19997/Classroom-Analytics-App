# 🎓 Classroom Insight

**AI-powered engagement analysis from any Zoom or YouTube classroom recording.**

Upload a class video, and this pipeline triangulates **face, voice, and words** across three independent modalities into per-child engagement metrics, teacher-behavior flags, and a full multimodal transcript.

🔗 **Live demo:** https://classroom-analytics-app.streamlit.app/

---

## 📸 Dashboard

> Add a screenshot here — see the *"How to capture the screenshot"* section below.

![Classroom Insight Dashboard](docs/dashboard.png)

---

## Why this exists

Teachers and school leaders don't have data on how engaged students actually are during class — just gut feel. This project turns any recorded class into structured, per-child engagement evidence in one click.

## What it produces

- **Pedagogical baseline metrics** — Talk Rate, Dialogue Frequency, Student Agency, Question Rates
- **Holistic emotion timeline** — per-child, per-minute, triangulated from face + voice + words
- **Visual engagement tracking** — attendance vs cameras-on over time
- **Teacher incident log** — Shouting / Praising / Demotivating flags, linked to the target child
- **Full transcript** — every utterance mapped to the visible roster

## Architecture

```
📹  Video URL (YouTube / Drive)
     ▼
🎬  FFmpeg → 16 kHz WAV + JPG frames (1 per 20–60 s)
     ▼
🧠  Gemini 1.5 Flash — 2 structured (Pydantic) calls
     ├─ Vision → attendance, per-child emotion, posture, hand-raised
     └─ Audio  → transcript + diarization + tone/text emotion + teacher flags
     ▼
🗄  SQLite (one .db per video, 5 normalized tables)
     ▼
📊  Python triangulation → highest-confidence modality wins per (child, time-bucket)
     ▼
🖥  Streamlit dashboard
```

Three modalities are kept **independent** — Gemini reports emotion + a confidence score for each source, and Python picks the winner per bucket. This is the difference between "gut-feel averaging" and "trust the strongest signal."

## Tech Stack

`Python 3.10+` · `Streamlit` · `Gemini 1.5 Flash` (via `google-genai`) · `SQLite` · `FFmpeg` · `Pydantic` · `Plotly` · `yt-dlp` · `gdown`

## Run locally

**Prerequisites:** Python 3.10+, [FFmpeg on PATH](https://ffmpeg.org/download.html), a Gemini API key.

```bash
git clone <this-repo>
cd Classroom-Analytics-App
pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key-here" > .env
streamlit run app.py
```

Open http://localhost:8501, paste any YouTube or Google Drive video URL, click **Analyze Video**.

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Add `packages.txt` at repo root containing `ffmpeg`
3. On [share.streamlit.io](https://share.streamlit.io) → **Settings → Secrets** add `GEMINI_API_KEY = "..."`
4. Set **Sharing → Public** so recruiters can view without login

## Cost

Roughly **$0.005–$0.02 per 10-minute video** on Gemini 1.5 Flash. Re-analyzing the same URL hits the SQLite cache — zero cost.

## Data Model (5 tables per video)

| Table | Grain | Purpose |
|---|---|---|
| `video_metadata` | 1 row / video | Dedup key + file properties |
| `child_registry` | N rows / video | Frozen per-video identity, name, role, appearance |
| `visual_frame_log` | 1 row / minute | Class-level counts (attendance, cameras-on, teacher-visible) |
| `child_frame_emotion` | children × minutes | Per-child visual emotion + confidence + posture + hand-raised |
| `script_storage` | 1 row / utterance | Transcript + audio/text emotion + teacher-behavior flags + speaker/target FKs |

See `doc/project_architecture.md` for the full schema, subjective-column tagging, and design rationale.

---

## How to capture the screenshot

1. Open the deployed app and click **Load** on any sample analysis
2. Scroll to include: title/hero → metric cards → emotion pie + timeline → teacher incident cards
3. Take a full-page screenshot (Chrome: DevTools → ⋮ → *Capture full size screenshot*)
4. Save as `docs/dashboard.png` and commit

## LinkedIn / portfolio caption (template)

> I built **Classroom Insight** — an end-to-end multimodal pipeline that turns any Zoom or YouTube class recording into per-child engagement analytics. Under the hood: FFmpeg for audio/frame extraction, Gemini 1.5 Flash with structured (Pydantic) outputs for vision + audio, SQLite for per-video storage, and confidence-weighted triangulation across face / voice / words so no single noisy modality dominates.
>
> Live demo (public): https://classroom-analytics-app.streamlit.app/
> Code: <your GitHub URL>

---

## Author

**Surya Prakash** · Data Analytics & AI
[LinkedIn](https://www.linkedin.com/in/surya-prakash-a8464420b/) · [Email](mailto:suryap19997@gmail.com)
