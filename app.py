import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os

from pipeline import run_pipeline, get_video_id
import metrics
import db

load_dotenv()

st.set_page_config(page_title="Classroom Analytics (Multimodal)", layout="wide")

st.title("🎓 Classroom Analytics Engine (Multimodal SQLite)")
st.markdown("Analyze online classroom engagement, attendance, and emotion via **Audio & Video Triangulation**.")

# Sidebar for API Key and URL
st.sidebar.header("Configuration")
user_api_key = st.sidebar.text_input(
    "Gemini API Key (optional)",
    value="",
    type="password",
    placeholder="Leave blank to use server key",
    help="Overrides the server-side key for this run only. Never stored.",
)
video_url = st.sidebar.text_input("Video URL (YouTube or Google Drive)")
analyze_button = st.sidebar.button("Analyze Video")


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


with st.sidebar.expander(f"📜 History and Examples", expanded=True):
    history = db.list_analyses()
    if not history:
        st.caption("No videos analyzed yet.")
    else:
        st.caption(f"{len(history)} video(s) in cache")
        for row in history:
            name = row["video_title"] or "(untitled)"
            url = row["original_url"] or ""
            with st.container(border=True):
                st.markdown(f"**{name}**")
                if url:
                    st.markdown(f"[🔗 open link]({url})")
                cols = st.columns(2)
                cols[0].caption(f"🕐 {_fmt_ts(row['processed_date'])}")
                cols[1].caption(f"⏱ {_fmt_duration(row['total_duration_sec'])}")
                if st.button("Load", key=f"load_{row['video_id']}", use_container_width=True):
                    st.session_state['video_id'] = row['video_id']
                    st.rerun()


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
        with st.spinner("Processing Multimodal Data (This may take a few minutes)..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(msg, pct):
                status_text.text(msg)
                progress_bar.progress(pct)

            try:
                result = run_pipeline(video_url, api_key, progress_callback=update_progress)
                st.session_state['video_id'] = result['video_id']
                st.success(f"Processing Complete! Status: {result['status']}")
            except Exception as e:
                if _is_auth_error(e):
                    if key_source == "user":
                        st.error(f"The API key you entered was rejected by Gemini. Please check it and try again.\n\nDetails: {e}")
                    else:
                        st.error(f"The server's configured Gemini API key is invalid or missing permissions. Contact the administrator, or enter your own key in the sidebar to unblock.\n\nDetails: {e}")
                else:
                    st.error(f"Pipeline Error ({key_source} key): {e}")

# Dashboard Rendering
if 'video_id' in st.session_state:
    vid = st.session_state['video_id']
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
    else:
        st.info("No visual attendance data available.")
        
    st.markdown("---")
    
    # 4. Teacher Behavior Log
    st.subheader("Teacher Incident Log")
    inc_df = metrics.get_teacher_incidents(vid)
    if not inc_df.empty:
        st.dataframe(inc_df, use_container_width=True)
    else:
        st.success("No negative teacher behaviors flagged in this session.")
        
    st.markdown("---")
    
    # 5. Raw Transcript
    st.subheader("Raw Transcript (Multimodal)")
    st.markdown("Transcript mapped to identified children and text-based emotion.")
    trans_df = metrics.get_raw_transcript(vid)
    st.dataframe(trans_df, use_container_width=True)
