# 🎓 Classroom Insight — From Video to Evidence

**AI-powered engagement analytics for classroom recordings.**
Upload a Zoom / YouTube / Google Drive class video → get per-child engagement metrics, teacher-behavior flags, and a full multimodal transcript in one click.

🔗 **Live demo:** https://classroom-analytics-app.streamlit.app/
📄 **Pilot study (PDF):** [`Classroom-Analysis/From Video to Evidence_ An Automated Pipeline for Evaluating Classroom Engagement in a Pilot Study.pdf`](Classroom-Analysis/From%20Video%20to%20Evidence_%20An%20Automated%20Pipeline%20for%20Evaluating%20Classroom%20Engagement%20in%20a%20Pilot%20Study.pdf)

---

## 📦 Repository Layout

This repo contains **two related projects** — the research pipeline that proved the idea, and the production Streamlit app that turned it into a one-click tool.

| Folder | What it is | Runtime |
|---|---|---|
| [`Classroom-Analytics-App/`](Classroom-Analytics-App/) | **Production app** — Streamlit dashboard, live pipeline, SQLite cache, deployed on Streamlit Cloud | Python 3.10+ · Streamlit · Gemini Flash |
| [`Classroom-Analysis/`](Classroom-Analysis/) | **Research pipeline** — Colab notebook, PRD, pilot-study PDF comparing Advance vs. Traditional teaching modules | Google Colab · Gemini 1.5 Pro · DistilBERT |

---

## 🧠 What it does

Teachers and school leaders don't have real data on how engaged students actually are during class — just gut feel. This project turns any recorded class into structured, per-child engagement evidence.

**Outputs:**

- **Pedagogical baseline metrics** — Talk Rate, Dialogue Frequency, Student Agency, Question Rates (student & teacher)
- **Holistic emotion timeline** — per-child, per-minute, triangulated from face + voice + words
- **Visual engagement tracking** — attendance vs. cameras-on over time
- **Teacher incident log** — Shouting / Praising / Demotivating flags, linked to the target child
- **Full multimodal transcript** — every utterance mapped to the visible roster

---

## ⚙️ Architecture

```
📹  Video URL (YouTube / Google Drive)
     │
     ▼
🎬  FFmpeg
     ├─ 16 kHz mono WAV  (audio track)
     └─ 1 JPG frame every 20–60 s  (dynamic bucket)
     │
     ▼
🧠  Gemini Flash — 2 structured (Pydantic) calls
     ├─ Vision → attendance, per-child emotion, posture, hand-raised
     └─ Audio  → transcription, diarization, tone emotion, text emotion,
                 teacher-behavior flags, targeted-child names
     │
     ▼
🗄  SQLite  (one .db per video, 5 normalized tables)
     ├─ video_metadata
     ├─ child_registry           (frozen per-video identity + roster)
     ├─ visual_frame_log         (class-level counts per minute)
     ├─ child_frame_emotion      (per-child × per-frame visual signal)
     └─ script_storage           (per-utterance audio + text + FKs)
     │
     ▼
📊  Triangulation (Python)
     Highest-confidence modality wins per (child, time-bucket).
     Face / Voice / Words are kept independent so no noisy signal dominates.
     │
     ▼
🖥  Streamlit dashboard
```

The three modalities are kept **independent** — Gemini reports an emotion + confidence score per source, and Python picks the winner per bucket. That's the difference between "gut-feel averaging" and "trust the strongest signal."

---

## 📊 Engagement Metrics

| Metric | Meaning |
|---|---|
| **Student Talk Rate** | Student words per minute — participation intensity |
| **Dialogue Frequency** | Speaker changes per minute — classroom interactivity |
| **Student Agency** | % of turns initiated by students — learner autonomy |
| **Student Question Rate** | Student questions per minute — inquiry-driven engagement |
| **Teacher Question Rate** | Teacher questions per minute — instructional prompting |

### Pilot Study Results (Advance vs. Traditional modules)

| Metric | Traditional | Advance | Δ |
|---|---|---|---|
| Student Talk Rate | 4.08 | 18.02 | **+342 %** |
| Dialogue Frequency | 1.49 | 6.49 | **+336 %** |
| Student Agency | 0.58 | 1.42 | **+145 %** |
| Student Question Rate | 0.25 | 1.04 | **+316 %** |
| Teacher Question Rate | 2.43 | 6.75 | **+178 %** |

Cohen's d > 0.8 across all metrics; Mann–Whitney U significant at p < 0.10 (one-sided). Full write-up in the [pilot study PDF](Classroom-Analysis/).

---

## 🗄 Data Model (5 tables per video)

| Table | Grain | Purpose |
|---|---|---|
| `video_metadata` | 1 row / video | Dedup key + file properties |
| `child_registry` | N rows / video | Frozen per-video identity, name, role, appearance |
| `visual_frame_log` | 1 row / minute | Class-level counts (attendance, cameras-on, teacher-visible) |
| `child_frame_emotion` | children × minutes | Per-child visual emotion + confidence + posture + hand-raised |
| `script_storage` | 1 row / utterance | Transcript + audio/text emotion + teacher-behavior flags + speaker/target FKs |

See [`Classroom-Analytics-App/doc/`](Classroom-Analytics-App/doc) for the full schema and design rationale.

---

## 🧰 Tech Stack

**App:** `Python 3.10+` · `Streamlit` · `Gemini Flash` (via `google-genai`) · `SQLite` · `FFmpeg` · `Pydantic` · `Plotly` · `yt-dlp` · `gdown`
**Research pipeline:** `Google Colab` · `Gemini 1.5 Pro` · `DistilBERT` (fine-tuned for question detection) · `Pandas` · `NumPy` · `SciPy` · `Seaborn` · `Matplotlib`

---

## 🚀 Run the app locally

**Prerequisites:** Python 3.10+, [FFmpeg on PATH](https://ffmpeg.org/download.html), a Gemini API key.

```bash
git clone https://github.com/psurya19997/Classroom-Analytics-App.git
cd Classroom-Analytics-App
pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key-here" > .env
streamlit run app.py
```

Open http://localhost:8501, paste any YouTube or Google Drive video URL, click **Analyze Video**.

### Deploy to Streamlit Cloud

1. Push to GitHub
2. Keep `packages.txt` (contains `ffmpeg`) at the app root
3. On [share.streamlit.io](https://share.streamlit.io) → **Settings → Secrets** add `GEMINI_API_KEY = "..."`
4. Set **Sharing → Public**

### Cost

Roughly **$0.005 – $0.02 per 10-minute video** on Gemini Flash. Re-analysing the same URL hits the SQLite cache — zero cost.

---

## 🔬 Run the research pipeline

Open [`Classroom-Analysis/New_Classroom_Content_Analysis_End_to_End_Pipeline.ipynb`](Classroom-Analysis/New_Classroom_Content_Analysis_End_to_End_Pipeline.ipynb) in Google Colab. Runtime → GPU recommended for the DistilBERT question-detection step.

---

## 🗺 Roadmap

- Expand dataset across multiple teachers, subjects, and grade levels
- Add gaze & gesture tracking as a 4th modality
- Cohort-level comparison view in the dashboard (multiple videos side-by-side)
- Exportable per-child PDF report for parent-teacher meetings

---

## 🪪 Author

**Surya Prakash** — Data Analytics & AI

- 📧 [suryap19997@gmail.com](mailto:suryap19997@gmail.com)
- 🌐 [LinkedIn](https://www.linkedin.com/in/surya-prakash-a8464420b/)
- 💻 [GitHub — Classroom-Analytics-App](https://github.com/psurya19997/Classroom-Analytics-App)

---

## 🏷 Tags

`Data Analytics` · `AI in Education` · `Multimodal AI` · `NLP` · `Speech Processing` · `Streamlit` · `Gemini` · `Python` · `Educational Data Science`
