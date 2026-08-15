import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os

from pipeline import run_pipeline, get_video_id
import metrics
import db

load_dotenv()

st.set_page_config(page_title="Classroom Insight", page_icon="🎓", layout="wide")

st.title("🎓 Classroom Insight — See What Your Class Really Feels")
st.markdown(
    "<p style='font-size: 1.1rem; color: #888; margin-top: -10px;'>"
    "AI-powered engagement analysis from any Zoom or YouTube classroom recording."
    "</p>",
    unsafe_allow_html=True,
)

# --- Hero: problem, approach, tech stack ---
hero_left, hero_right = st.columns([3, 2])
with hero_left:
    st.markdown(
        """
        **The problem.** Teachers and school leaders don't have data on how engaged students actually are during class — just gut feel.

        **The approach.** Upload a recorded class, and this pipeline extracts audio + video frames, runs them through Gemini for transcription, diarization, emotion detection, and behavior tagging, then triangulates the signals across three modalities (face, voice, words) into per-child engagement metrics you can act on.
        """
    )
with hero_right:
    st.markdown("**Tech Stack**")
    st.markdown(
        """
        <div style='line-height:2.2'>
            <span style='background:#1f77b4;color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:0.85rem'>Python</span>
            <span style='background:#ff7f0e;color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:0.85rem'>Streamlit</span>
            <span style='background:#2ca02c;color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:0.85rem'>Gemini 1.5 Flash</span>
            <span style='background:#d62728;color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:0.85rem'>SQLite</span>
            <span style='background:#9467bd;color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:0.85rem'>FFmpeg</span>
            <span style='background:#8c564b;color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:0.85rem'>Pydantic</span>
            <span style='background:#17becf;color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:0.85rem'>Plotly</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("🛠 How it works (technical architecture)", expanded=False):
    st.markdown(
        """
```
📹  Video URL (YouTube / Google Drive)
     │
     ▼
🎬  FFmpeg
     ├─ 16 kHz mono WAV (audio)
     └─ 1 JPG frame every 20–60 s (dynamic bucket)
     │
     ▼
🧠  Gemini 1.5 Flash — 2 structured (Pydantic) calls
     ├─ Vision → attendance, per-child emotion, posture, hand-raised
     └─ Audio  → transcription, diarization, tone emotion, text emotion,
                 teacher-behavior flags, targeted-child names
     │
     ▼
🗄  SQLite  (one .db per video, five normalized tables)
     ├─ video_metadata
     ├─ child_registry           (frozen per-video identity + roster)
     ├─ visual_frame_log         (class-level counts per minute)
     ├─ child_frame_emotion      (per-child × per-frame visual signal)
     └─ script_storage           (per-utterance audio + text + FKs)
     │
     ▼
📊  Triangulation (Python)
     Highest-confidence modality wins per (child, time-bucket).
     Face / Voice / Words are kept independent — the winner is picked
     after the fact, so a low-light frame doesn't poison a clear voice.
     │
     ▼
🖥  Streamlit Dashboard (this page)
```

**Why three independent modalities.** Emotion detection from any single source is unreliable — a bored face can hide an engaged mind, and enthusiastic words can mask a flat tone. By keeping face, voice, and words separate and asking Gemini for a confidence score on each, the pipeline trusts the strongest signal per moment instead of averaging noise.

**Why SQLite per-video.** No server to run, no cross-video schema drift, and every analysis is a portable file you can open in DB Browser for SQLite.
        """
    )

st.markdown("---")

def _fmt_duration(sec):
    if not sec or sec <= 0:
        return "—"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_ts(iso):
    if not iso:
        return "—"
    return iso.replace("T", " ")[:16]


# --- Sidebar: Sample analyses first (recruiter-friendly) ---
with st.sidebar.expander("▶️ Try a Sample Analysis", expanded=True):
    st.caption("Instant — no API key needed. Click Load on any example below.")
    history = db.list_analyses()
    if not history:
        st.caption("_No sample analyses cached yet. Run one from Configuration below._")
    else:
        for row in history:
            name = row["video_title"] or "(untitled)"
            url = row["original_url"] or ""
            with st.container(border=True):
                st.markdown(f"**{name}**")
                if url:
                    st.markdown(f"[🔗 open source]({url})")
                cols = st.columns(2)
                cols[0].caption(f"🕐 {_fmt_ts(row['processed_date'])}")
                cols[1].caption(f"⏱ {_fmt_duration(row['total_duration_sec'])}")
                if st.button("Load", key=f"load_{row['video_id']}", use_container_width=True):
                    st.session_state['video_id'] = row['video_id']
                    st.session_state['is_demo'] = False
                    st.rerun()

st.sidebar.markdown("---")

# --- Sidebar: Run a new analysis ---
st.sidebar.header("⚙️ Analyze Your Own Video")
user_api_key = st.sidebar.text_input(
    "Gemini API Key (optional)",
    value="",
    type="password",
    placeholder="Leave blank to use server key",
    help="Overrides the server-side key for this run only. Never stored.",
)
video_url = st.sidebar.text_input("Video URL (YouTube or Google Drive)")
analyze_button = st.sidebar.button("Analyze Video", type="primary", use_container_width=True)


def _resolve_api_key(user_key: str):
    """Priority: UI-supplied key > .env key. Returns (key, source)."""
    if user_key and user_key.strip():
        return user_key.strip(), "user"
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key, "env"
    return None, None


def _is_auth_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(tok in msg for tok in (
        "api key not valid", "api_key_invalid", "invalid api key",
        "permission_denied", "unauthenticated", "401", "403",
    ))


if analyze_button:
    api_key, key_source = _resolve_api_key(user_api_key)

    if not api_key:
        st.sidebar.error("No API key available. Enter one in the sidebar or set GEMINI_API_KEY in the server's .env file.")
    elif not video_url:
        st.sidebar.error("Please provide a Video URL.")
    else:
        with st.spinner("Processing Multimodal Data..."):
            import time
            progress_bar = st.progress(0)
            status_container = st.empty()
            step_history = []  # list of (msg, elapsed_sec)
            start_time = time.time()
            last_step_start = {"t": start_time, "msg": None}

            def update_progress(msg, pct):
                now = time.time()
                if last_step_start["msg"] is not None and last_step_start["msg"] != msg:
                    step_history.append((last_step_start["msg"], now - last_step_start["t"]))
                    last_step_start["t"] = now
                last_step_start["msg"] = msg
                progress_bar.progress(pct)

                total_elapsed = now - start_time
                lines = [f"✅ {m}  <span style='color:#888'>({e:.1f}s)</span>" for m, e in step_history]
                if pct < 100:
                    lines.append(f"⏳ **{msg}**  <span style='color:#888'>· {total_elapsed:.1f}s total</span>")
                else:
                    lines.append(f"✅ **{msg}**  <span style='color:#888'>· {total_elapsed:.1f}s total</span>")
                status_container.markdown("<br>".join(lines), unsafe_allow_html=True)

            try:
                result = run_pipeline(video_url, api_key, progress_callback=update_progress)
                st.session_state['video_id'] = result['video_id']
                st.session_state['is_demo'] = False
                st.success(f"Processing Complete! Status: {result['status']}")
            except Exception as e:
                if _is_auth_error(e):
                    if key_source == "user":
                        st.error(f"The API key you entered was rejected by Gemini. Please check it and try again.\n\nDetails: {e}")
                    else:
                        st.error(f"The server's configured Gemini API key is invalid or missing permissions. Contact the administrator, or enter your own key in the sidebar to unblock.\n\nDetails: {e}")
                else:
                    st.error(f"Pipeline Error ({key_source} key): {e}")

# Auto-load newest cached demo on first visit
if 'video_id' not in st.session_state:
    _history = db.list_analyses()
    if _history:
        st.session_state['video_id'] = _history[0]['video_id']
        st.session_state['is_demo'] = True

# Dashboard Rendering
if 'video_id' in st.session_state:
    vid = st.session_state['video_id']
    if st.session_state.get('is_demo'):
        st.info("👋 You're viewing a **sample analysis** so you can explore the dashboard instantly. Paste a video URL in the sidebar to run your own.")
    st.markdown("---")
    
    # 1. Baseline Metrics
    st.subheader("Pedagogical Baseline Metrics")
    m = metrics.get_baseline_metrics(vid)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Talk Rate", f"{m['student_talk_rate']} wpm", help="Student words per minute")
    col2.metric("Dialogue Freq", f"{m['dialogue_frequency']} /min", help="Speaker switches per minute")
    col3.metric("Student Agency", f"{m['student_agency']}", help="Student vs Teacher initiations")
    col4.metric("Student Q Rate", f"{m['student_question_rate']} /min")
    col5.metric("Teacher Q Rate", f"{m['teacher_question_rate']} /min")
    st.caption("💡 *Higher Talk Rate + Dialogue Frequency signals an active, dialogic classroom. Student Agency above 1.0 means students initiated more than the teacher.*")
    
    st.markdown("---")
    
    # 2. Multimodal Emotion Analysis
    st.subheader("Holistic Emotion Triangulation")
    st.markdown("These emotions are mathematically triangulated from **Facial Expressions + Voice Tone + Semantic Words** using confidence weighting.")
    
    emo_df = metrics.get_child_emotion_timeline(vid)
    
    if not emo_df.empty:
        colA, colB = st.columns([1, 2])
        
        with colA:
            # Overall Pie Chart
            pie_data = emo_df['holistic_emotion'].value_counts().reset_index()
            pie_data.columns = ['Emotion', 'Count']
            fig_pie = px.pie(pie_data, names='Emotion', values='Count', title="Overall Class Emotion", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with colB:
            # Child-wise Timeline Scatter/Line Plot
            fig_line = px.scatter(
                emo_df, 
                x="time_min", 
                y="holistic_emotion", 
                color="child_name",
                title="Child-Wise Emotional Timeline",
                labels={"time_min": "Time (Minutes)", "holistic_emotion": "Triangulated Emotion"}
            )
            # Add subtle connecting lines
            fig_line.update_traces(mode='lines+markers', opacity=0.8)
            st.plotly_chart(fig_line, use_container_width=True)
        st.caption("💡 *Each dot is one child's emotion at one time bucket. Emotions come from the modality (face / voice / text) with the highest confidence — so a bored face during an enthusiastic answer weights the answer, not the face.*")
    else:
        st.info("No emotion data available to triangulate.")
        
    st.markdown("---")
    
    # 3. Visual Attendance Tracking
    st.subheader("Visual Engagement Tracking")
    att_df = metrics.get_attendance_timeline(vid)
    if not att_df.empty:
        fig_att = px.line(
            att_df, 
            x="time_min", 
            y=["attendance_count", "camera_on_count"],
            title="Attendance & Cameras On Over Time",
            labels={"time_min": "Time (Minutes)", "value": "Count", "variable": "Metric"}
        )
        st.plotly_chart(fig_att, use_container_width=True)
        st.caption("💡 *A gap between Attendance and Cameras-On usually means students are present but disengaged — worth flagging to the teacher.*")
    else:
        st.info("No visual attendance data available.")
        
    st.markdown("---")
    
    # 4. Teacher Behavior Log
    st.subheader("Teacher Incident Log")
    inc_df = metrics.get_teacher_incidents(vid)
    if not inc_df.empty:
        flag_style = {
            "Shouting":     ("#3a0a0a", "#ff6b6b", "🔴"),
            "Demotivating": ("#3a2a0a", "#ffb86b", "🟡"),
            "Praising":     ("#0a3a1a", "#6bff9e", "🟢"),
        }
        for _, r in inc_df.iterrows():
            flag = r["teacher_behavior_flag"]
            bg, fg, icon = flag_style.get(flag, ("#2a2a2a", "#dddddd", "⚪"))
            ts = f"{int(r['start_sec']//60):02d}:{int(r['start_sec']%60):02d}"
            target = r["targeted_child"] if pd.notna(r["targeted_child"]) else "class"
            st.markdown(
                f"""
                <div style='background:{bg};border-left:4px solid {fg};padding:12px 16px;border-radius:6px;margin-bottom:8px;'>
                    <div style='color:{fg};font-weight:600;font-size:0.9rem;'>{icon} {flag} · <span style='color:#aaa;font-weight:400'>{ts} → {target}</span></div>
                    <div style='color:#ddd;margin-top:6px;font-style:italic;'>“{r['utterance']}”</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("No negative teacher behaviors flagged in this session.")
        
    st.markdown("---")
    
    # 5. Chat-style Transcript
    st.subheader("Full Class Transcript")
    st.markdown("Every utterance mapped to its speaker (fuzzy-matched to the visible roster where possible) and text-based emotion.")
    with st.expander(f"View full transcript", expanded=False):
        trans_df = metrics.get_raw_transcript(vid)
        if trans_df.empty:
            st.info("No transcript rows available.")
        else:
            for _, r in trans_df.iterrows():
                is_teacher = str(r["speaker_display"]).lower().startswith("teacher") or str(r["speaker_display"]).lower() == "teacher"
                align = "flex-end" if is_teacher else "flex-start"
                bubble_bg = "#264653" if is_teacher else "#2a3d2a"
                accent = "#f4a261" if is_teacher else "#a8dadc"
                q_marker = " ❓" if r["is_question"] else ""
                emo = f" · <span style='color:#bbb'>{r['emotion_text']}</span>" if pd.notna(r["emotion_text"]) else ""
                st.markdown(
                    f"""
                    <div style='display:flex;justify-content:{align};margin-bottom:8px;'>
                        <div style='background:{bubble_bg};padding:10px 14px;border-radius:12px;max-width:75%;border-left:3px solid {accent};'>
                            <div style='font-size:0.75rem;color:{accent};font-weight:600;'>
                                {r['speaker_display']} · {r['time_range']}{q_marker}{emo}
                            </div>
                            <div style='color:#eee;margin-top:4px;'>{r['utterance']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:#888;padding:20px 0;font-size:0.9rem;'>
        Built by <strong>Surya Prakash</strong> · Data Analytics & AI<br>
        <a href='https://github.com/psurya19997/Classroom-Analytics-App' style='color:#4da6ff;text-decoration:none;margin:0 8px;'>🐙 GitHub</a>
        ·
        <a href='https://www.linkedin.com/in/surya-prakash-a8464420b/' style='color:#4da6ff;text-decoration:none;margin:0 8px;'>💼 LinkedIn</a>
        ·
        <a href='mailto:suryap19997@gmail.com' style='color:#4da6ff;text-decoration:none;margin:0 8px;'>✉️ Email</a>
        <br><br>
        <span style='color:#666;font-size:0.8rem;'>Source: view the code on GitHub. Feedback welcome.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
