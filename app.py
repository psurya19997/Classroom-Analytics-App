import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os

from pipeline import run_pipeline, get_video_id
import metrics

load_dotenv()

st.set_page_config(page_title="Classroom Analytics (Multimodal)", layout="wide")

st.title("🎓 Classroom Analytics Engine (Multimodal SQLite)")
st.markdown("Analyze online classroom engagement, attendance, and emotion via **Audio & Video Triangulation**.")

# Sidebar for API Key and URL
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
video_url = st.sidebar.text_input("Video URL (YouTube or Google Drive)")
analyze_button = st.sidebar.button("Analyze Video")

if analyze_button:
    if not api_key:
        st.sidebar.error("Please provide a Gemini API Key.")
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
                st.error(f"Pipeline Error: {str(e)}")

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
