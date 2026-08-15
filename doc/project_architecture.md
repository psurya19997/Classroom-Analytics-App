# Classroom Analytics App: Project Architecture & Documentation

## 1. Project Purpose
The goal of this project is to create a state-of-the-art, multimodal (Audio + Video) analytics engine for online classrooms (e.g., Zoom, Google Meet). 

Unlike standard tools that only transcribe text, this application triangulates data across three independent modalities (Visual, Acoustic, and Semantic) to objectively track:
- **Student Engagement:** (e.g., Camera-on ratios, distraction counts, hand-raising).
- **Pedagogical Dynamics:** (e.g., Student Agency, Talk Rates, Dialogue Frequency).
- **Teacher Behavior:** (e.g., Detecting demotivation, shouting, praising, and tracking specific targeted children).
- **Holistic Emotion:** Tracking child-wise emotional states over the duration of the class by mathematically combining facial expressions, tone of voice, and spoken words.

---

## 2. What We Have Implemented So Far (Phase 1)
We have successfully built the foundation of the pipeline and the Streamlit dashboard:
- **Audio Extraction:** Built a pipeline that accepts YouTube or Google Drive links, downloads the video via `yt-dlp`/`gdown`, and uses `ffmpeg` to strip and compress the audio to a lightweight 16kHz `.wav` format.
- **AI Pivot (Crash Resolution):** We successfully pivoted away from a broken local PyTorch environment (due to an unfixable Windows `WinError 1114` C++ crash) and securely routed our AI processing through the Gemini 1.5 Flash Cloud API.
- **Baseline Metrics:** The Python backend currently calculates:
  - `Student_Talk_Rate`
  - `Dialogue_Frequency`
  - `Student_Question_Rate`
  - `Teacher_Question_Rate`
  - `Student_Agency` (The ratio of Student vs. Teacher conversational initiations after a 3-second gap).
- **Dashboard UI:** A Streamlit frontend that displays these metrics, plots an overall emotional pie chart, and renders a sortable raw transcript.

---

## 3. The Data Infrastructure (The Master Architecture)

**Storage:** One **SQLite** file per video (`cache/{video_id}.db`). No server, no account — a plain file, queried with SQL from Python.

**Scope:** All identifiers are stable **within a single video only**. No cross-video identity.

**Convention:** All primary/foreign key IDs are **INTEGER**.

**Crucial Research Principle:** We do *not* force Gemini to guess a holistic emotion upfront. Instead, we keep the raw modalities (Text, Audio, Image) completely isolated. Gemini reports emotion + a **Confidence Score** (0.0 – 1.0) for each modality independently. Python then triangulates the final Holistic Emotion using those confidences.

**Legend:** 🟢 Objective · 🟡 Semi-subjective · 🔴 Subjective · **[G]** = costs Gemini tokens.

---

### Table 1: `video_metadata` — 1 row per video
Pure objective properties. Prevents duplicate processing.

| Column | Type | Source | Note |
|---|---|---|---|
| video_id | INTEGER PK | autoincrement | 🟢 |
| url_hash | TEXT UNIQUE | hash of URL | 🟢 (dedup lookup key) |
| original_url | TEXT | user input | 🟢 |
| source_platform | TEXT | URL parse | 🟢 |
| video_title | TEXT | yt-dlp meta | 🟢 |
| upload_date | DATE | yt-dlp meta | 🟢 |
| processed_date | DATETIME | `datetime.now()` | 🟢 |
| total_duration_sec | REAL | ffprobe | 🟢 |
| file_size_mb | REAL | os.stat | 🟢 |
| resolution | TEXT | ffprobe | 🟢 |
| frames_analyzed | INTEGER | count | 🟢 |

**Gemini cost: 0.**

---

### Table 2: `child_registry` — N rows per video
One row per distinct person detected in the video. Populated on first-seen basis and **frozen** — consistency is prioritized over freshness so downstream joins with `script_storage` stay stable.

| Column | Type | Source | Note |
|---|---|---|---|
| child_id | INTEGER PK | autoincrement per video | 🟢 |
| video_id | INTEGER FK | | 🟢 |
| name | TEXT | **[G]** vision (frozen at first sighting) | 🟢 objective — visible Zoom/tile label, else fallback `Kid_{child_id}` |
| name_source | TEXT | derived | 🟢 `"visible"` or `"fallback"` |
| appearance | TEXT | **[G]** vision | 🔴 e.g., "boy in red hoodie" — for disambiguation only |
| role | TEXT | **[G]** vision | 🟡 `student` / `teacher` / `unknown` |
| first_seen_sec | REAL | **[G]** vision | 🟢 |
| total_frames_present | INTEGER | computed post-pipeline | 🟢 |

**Sync rule (critical):** `name` is the canonical join key between visual and audio pipelines. When audio-side extraction fills `speaker_child_id` or `targeted_child_id`, it fuzzy-matches the spoken name (e.g. "Aarav") against `child_registry.name` (e.g. "Aarav K.") to resolve the numeric `child_id`. Because `name` is frozen at first sighting, this mapping never drifts mid-video.

**Known limitations documented:**
- **OCR misreads** of Zoom name tiles → mitigated by reconciling labels at end-of-pass (most common label wins per `child_id`).
- **Non-name labels** (`"iPhone"`, `"Guest"`) → fall back to `Kid_{child_id}`, flagged via `name_source = "fallback"`.
- **Mid-video rename** by the participant → ignored; frozen name persists (v1 tradeoff).
- **Duplicate first names** (two Aaravs) → `targeted_child_id` set to `NULL` when ambiguous; ambiguity surfaced to dashboard.
- **Teacher tile** may create a `child_registry` row → filtered via `role = "teacher"`.

---

### Table 3: `visual_frame_log` — 1 row per frame (class-level)
Time-series visual context extracted via `ffmpeg` (e.g., 1 frame per minute).

| Column | Type | Source | Note |
|---|---|---|---|
| video_id | INTEGER FK | | 🟢 |
| timestamp_sec | INTEGER PK | frame time | 🟢 |
| attendance_count | INTEGER | **[G]** vision | 🟢 total tiles present |
| camera_on_count | INTEGER | **[G]** vision | 🟢 tiles with visible face |
| kids_not_visible_count | INTEGER | derived: `attendance_count − camera_on_count` | 🟢 kids present but camera off |
| teacher_visible | BOOLEAN | **[G]** vision | 🟢 |

*Removed:* `screen_share_active` (out of scope), `black_screen_count` (renamed & redefined above).

---

### Table 4: `child_frame_emotion` — children × frames
One row per `(timestamp, child_id)`. This is the per-child visual signal.

| Column | Type | Source | Note |
|---|---|---|---|
| video_id | INTEGER FK | | 🟢 |
| timestamp_sec | INTEGER | | 🟢 |
| child_id | INTEGER FK | | 🟢 |
| emotion_visual | TEXT | **[G]** vision | 🔴 (Joyful / Bored / Frustrated / …) |
| confidence_visual | REAL | **[G]** vision | 🟡 (0.0–1.0) |
| facial_expression | TEXT | **[G]** vision | 🟡 (Smiling / Frowning) |
| posture | TEXT | **[G]** vision | 🟡 (Slouching / Leaning-in) |
| hand_raised | BOOLEAN | **[G]** vision | 🟢 |
| distracted | BOOLEAN | **[G]** vision | 🟡 (gaze interpretation) |

PK: `(video_id, timestamp_sec, child_id)`.

---

### Table 5: `script_storage` — 1 row per utterance
Chronological audio/text log. FKs into `child_registry` for per-child accounting.

| Column | Type | Source | Note |
|---|---|---|---|
| utterance_id | INTEGER PK | autoincrement | 🟢 |
| video_id | INTEGER FK | | 🟢 |
| start_sec | REAL | **[G]** audio | 🟢 |
| end_sec | REAL | **[G]** audio | 🟢 |
| speaker_norm | TEXT | **[G]** audio | 🟡 `Student` / `Teacher` |
| speaker_child_id | INTEGER FK → child_registry.child_id | **[G]** audio + fuzzy-match on `name` | 🟡 *who is talking* (nullable if unknown) |
| utterance | TEXT | **[G]** audio | 🟢 |
| is_question | BOOLEAN | **[G]** audio | 🟡 |
| speaking_rate | REAL | derived: `word_count / (end_sec − start_sec)` | 🟢 words/sec — Python, not Gemini |
| emotion_audio | TEXT | **[G]** audio | 🔴 |
| confidence_audio | REAL | **[G]** audio | 🟡 |
| emotion_text | TEXT | **[G]** audio | 🔴 |
| confidence_text | REAL | **[G]** audio | 🟡 |
| keywords | TEXT (JSON) | **[G]** audio | 🟡 |
| teacher_behavior_flag | TEXT | **[G]** audio | 🔴 (`Shouting` / `Praising` / `Demotivating` / `None`) |
| targeted_child_id | INTEGER FK → child_registry.child_id | **[G]** audio + fuzzy-match on `name` | 🔴 *who teacher is addressing* (nullable if class-wide or ambiguous) |

*Removed:* `pitch` (redundant with `emotion_audio`, no meaningful baseline), `volume` (mic-gain-dependent, unreliable). `speaking_rate` moved from Gemini-sourced to Python-derived.

**How `speaker_child_id` and `targeted_child_id` behave:**
| Scenario | speaker_norm | speaker_child_id | targeted_child_id |
|---|---|---|---|
| Teacher lectures whole class | Teacher | NULL | NULL |
| Teacher calls on Aarav | Teacher | NULL | 3 (Aarav) |
| Aarav answers | Student | 3 | NULL |
| Aarav asks the teacher | Student | 3 | NULL |
| Two students overlap | Student | dominant child_id | NULL |

Both columns are best-effort in v1 — expect frequent NULLs when Gemini cannot confidently attribute a voice or resolve a name.

---

## 4. Holistic Metric Computation Engine
The final step of the architecture occurs in the Python codebase, *not* in the AI prompt.

`metrics.py` reads `visual_frame_log`, `child_frame_emotion`, and `script_storage` via SQL joins, and uses the **Confidence Scores** to mathematically triangulate the true state of the classroom.

**Time-bucket contract (Dynamic):** The triangulation time buckets and frame extraction cadence scale dynamically based on the total duration of the video to ensure maximum precision without overwhelming API limits:
- **< 10 minutes:** 20-second buckets (1 frame extracted every 20s).
- **10 to 29 minutes:** 30-second buckets (1 frame extracted every 30s).
- **≥ 30 minutes:** 60-second buckets (1 frame extracted every 60s).

`script_storage` rows are mathematically aggregated into these specific buckets before joining with the visual data.

Example: if `confidence_visual` is low due to poor camera lighting but `confidence_text` is 0.99, the algorithm heavily weights the semantic signal to compute the final `holistic_emotion` for that minute-per-child.

---

## 5. Dashboard Wiring (Streamlit)

```
User pastes URL
      │
      ▼
pipeline.py
  • Download audio + extract frames (ffmpeg)
  • 1 Gemini audio call  → script_storage
  • N Gemini vision calls → visual_frame_log + child_frame_emotion + child_registry
  • Write to cache/{video_id}.db  (SQLite)
      │
      ▼
metrics.py
  • SQL joins across tables
  • Confidence-weighted triangulation → holistic emotion per child per minute
  • Returns dict of final metrics
      │
      ▼
app.py (Streamlit)
  • @st.cache_data keyed on video_id → instant re-load
  • Renders metric cards, per-child timelines, transcript, teacher-incident log
```

| UI section | Table(s) | Query |
|---|---|---|
| Metric cards (Talk Rate, Agency, …) | `script_storage` | GROUP BY speaker |
| Emotion pie chart | `script_storage` + `child_frame_emotion` | triangulate + count |
| Per-child emotion timeline | `child_frame_emotion` JOIN `child_registry` | time-series per `child_id` |
| Teacher incidents log | `script_storage WHERE teacher_behavior_flag <> 'None'` | filter + JOIN on `targeted_child_id` |
| Attendance over time | `visual_frame_log` | time-series |
| Raw transcript | `script_storage` JOIN `child_registry` | ORDER BY `start_sec` |

**Cache key = `video_id`** (looked up via `url_hash`). Re-analyzing the same URL reads the `.db` instantly — zero Gemini cost.
