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
PAPER_URL = "https://github.com/psurya19997/Classroom-Analytics-App/blob/main/docs/paper.pdf"

st.markdown(
    "<p style='font-size: 1.1rem; color: #aaa; margin-top: -6px; line-height:1.55;'>"
    "Turn any recorded online class into per-child engagement analytics — "
    "attendance, curiosity, participation, and teacher-behavior signals, in one click. "
    f"<a href='{PAPER_URL}' target='_blank' style='color:#4da6ff;text-decoration:none;white-space:nowrap;'>"
    "📄 Read the pilot study →</a>"
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
    _badge = "background:{c};color:white;padding:5px 12px;border-radius:14px;font-size:0.85rem;white-space:nowrap;font-weight:500;"
    _tech = [
        ("Python", "#1f77b4"), ("Streamlit", "#ff7f0e"), ("Gemini 1.5 Flash", "#2ca02c"),
        ("SQLite", "#d62728"), ("FFmpeg", "#9467bd"), ("Plotly", "#17becf"),
    ]
    stack_html = (
        "<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
        + "".join(f"<span style='{_badge.format(c=c)}'>{name}</span>" for name, c in _tech)
        + "</div>"
    )
    st.markdown(stack_html, unsafe_allow_html=True)

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

    # --- Video header banner (with inline rename) ---
    header = metrics.get_video_header(vid)
    if header:
        h_title = header.get("video_title") or "(untitled recording)"
        h_url = header.get("original_url") or ""
        h_dur = header.get("total_duration_sec") or 0
        h_frames = header.get("frames_analyzed") or 0
        dur_str = _fmt_duration(h_dur)
        link_html = f"<a href='{h_url}' target='_blank' style='color:#4da6ff;text-decoration:none;'>🔗 open source</a>" if h_url else ""

        edit_key = f"edit_title_{vid}"
        if st.session_state.get(edit_key, False):
            # Edit mode
            with st.container(border=True):
                st.markdown("**✏️ Rename this video**")
                new_title = st.text_input(
                    "New title",
                    value=h_title,
                    key=f"input_{vid}",
                    max_chars=100,
                    label_visibility="collapsed",
                )
                bc1, bc2, _ = st.columns([1, 1, 6])
                if bc1.button("💾 Save", key=f"save_{vid}", type="primary", use_container_width=True):
                    clean = (new_title or "").strip()[:100]
                    if clean:
                        db.update_video_title(vid, clean)
                        st.session_state[edit_key] = False
                        st.rerun()
                    else:
                        st.warning("Title can't be empty.")
                if bc2.button("Cancel", key=f"cancel_{vid}", use_container_width=True):
                    st.session_state[edit_key] = False
                    st.rerun()
        else:
            # Normal view: banner + tiny edit button
            hcol, ecol = st.columns([12, 1])
            with hcol:
                header_html = (
                    f"<div style='background:#1a2332;border-left:4px solid #4da6ff;padding:14px 18px;border-radius:6px;margin:12px 0 20px 0;'>"
                    f"<div style='font-size:1.1rem;font-weight:600;color:#fff;'>🎬 {h_title}</div>"
                    f"<div style='color:#aaa;font-size:0.9rem;margin-top:6px;'>"
                    f"⏱ {dur_str} &nbsp;·&nbsp; 🖼 {h_frames} frames analyzed &nbsp;·&nbsp; {link_html}"
                    f"</div>"
                    f"</div>"
                )
                st.markdown(header_html, unsafe_allow_html=True)
            with ecol:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                if st.button("✏️", key=f"edit_{vid}", help="Rename this video"):
                    st.session_state[edit_key] = True
                    st.rerun()

    st.markdown("---")

    # 1. Class at a Glance
    st.subheader("📊 Class at a Glance")
    m = metrics.get_baseline_metrics(vid)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Talk Rate", f"{m['student_talk_rate']} wpm",
        help="**Student words per minute.**\n\nHow much students speak overall. Higher = more student voice in the room. Research benchmark: dialogic classrooms typically exceed 12 wpm."
    )
    col2.metric(
        "Dialogue Freq", f"{m['dialogue_frequency']} /min",
        help="**Speaker switches per minute.**\n\nCounts how often the conversation moves between teacher and students. Higher = more back-and-forth interactivity vs. a monologue."
    )
    col3.metric(
        "Student Agency", f"{m['student_agency']}",
        help="**Ratio of student-initiated turns to teacher-initiated turns** (after a 3-second gap).\n\nAbove 1.0 = students drove the conversation. Below 1.0 = teacher-led."
    )
    col4.metric(
        "Student Q Rate", f"{m['student_question_rate']} /min",
        help="**Student questions per minute.**\n\nA direct proxy for curiosity and inquiry-based engagement. High values = students are actively probing, not just receiving."
    )
    col5.metric(
        "Teacher Q Rate", f"{m['teacher_question_rate']} /min",
        help="**Teacher questions per minute.**\n\nMeasures how often the teacher invites student thinking (open questions) vs. lecturing."
    )
    st.caption("💡 *Higher Talk Rate + Dialogue Frequency signals an active, dialogic classroom. Student Agency above 1.0 means students initiated more than the teacher.*")
    
    st.markdown("---")
    
    # 2. Per-Child Scorecard
    st.subheader("👤 Per-Child Scorecard")
    st.markdown("Every student's behavior derived from **face + voice + words**. Focus on **curiosity** and **questions asked** — the strongest markers of inquiry-based learning.")

    child_df = metrics.get_child_stats(vid)
    if child_df.empty:
        st.info("No per-child data available.")
    else:
        names = child_df["name"].tolist()
        selected = st.selectbox(
            "Select a student",
            options=names,
            index=0,
            help="Sorted by attendance %. First student is shown by default.",
        )
        row = child_df[child_df["name"] == selected].iloc[0]

        # Attendance color
        att = row["attendance_pct"]
        att_color = "#4ade80" if att >= 80 else "#fbbf24" if att >= 50 else "#f87171"

        # Highlight if dominant emotion is Curious/positive
        dom = row["dominant_emotion"]
        is_curious = str(dom).strip().capitalize() in ("Curious", "Attentive", "Joyful")
        emo_bg = "#0a3a1a" if is_curious else "#2a2a2a"
        emo_accent = "#6bff9e" if is_curious else "#dddddd"
        emo_icon = "✨" if is_curious else "🎭"

        # Consensus badge — how much face/voice/text agreed on the dominant emotion
        cons = metrics._consensus_level(row.get("consensus_ratio"))
        if cons:
            cons_icon, cons_label, cons_color = cons
            consensus_html = (
                f"<span title='{cons_label} — average agreement across face, voice and text signals' "
                f"style='color:{cons_color};font-size:0.9rem;margin-left:12px;'>"
                f"{cons_icon} {cons_label}</span>"
            )
        else:
            consensus_html = ""

        # Distraction color
        distr = row["distracted_pct"]
        distr_color = "#f87171" if distr > 30 else "#fbbf24" if distr > 15 else "#dddddd"

        card_html = (
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div>"
            f"<div style='font-size:1.5rem;font-weight:700;color:#fff;'>👤 {row['name']}</div>"
            f"<div style='color:#8b949e;font-size:0.9rem;font-style:italic;margin-top:2px;'>{row['appearance']}</div>"
            f"</div>"
            f"<div style='background:{att_color};color:#0a0a0a;padding:8px 16px;border-radius:20px;font-weight:700;font-size:1.1rem;'>{att}% attendance</div>"
            f"</div>"
            f"<div style='background:{emo_bg};border-left:4px solid {emo_accent};padding:10px 14px;border-radius:6px;margin-top:16px;'>"
            f"<span style='color:{emo_accent};font-weight:600;'>{emo_icon} Mostly {dom}</span>"
            f"{consensus_html}"
            f"</div>"
            f"</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # HIGHLIGHTED metrics — Questions asked featured
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        q_html = (
            f"<div style='background:linear-gradient(135deg,#1e3a8a,#3b82f6);padding:20px;border-radius:12px;text-align:center;'>"
            f"<div style='font-size:0.9rem;color:#dbeafe;text-transform:uppercase;letter-spacing:1px;'>❓ Questions Asked</div>"
            f"<div style='font-size:3rem;font-weight:800;color:#fff;line-height:1.2;margin-top:6px;'>{row['questions_asked']}</div>"
            f"<div style='color:#dbeafe;font-size:0.8rem;'>markers of curiosity</div>"
            f"</div>"
        )
        h1.markdown(q_html, unsafe_allow_html=True)
        s_html = (
            f"<div style='background:linear-gradient(135deg,#065f46,#10b981);padding:20px;border-radius:12px;text-align:center;'>"
            f"<div style='font-size:0.9rem;color:#d1fae5;text-transform:uppercase;letter-spacing:1px;'>💬 Spoke</div>"
            f"<div style='font-size:3rem;font-weight:800;color:#fff;line-height:1.2;margin-top:6px;'>{row['spoke_times']}</div>"
            f"<div style='color:#d1fae5;font-size:0.8rem;'>{int(row['speaking_sec'])}s total</div>"
            f"</div>"
        )
        h2.markdown(s_html, unsafe_allow_html=True)

        # Secondary metrics
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        s1.metric("✋ Hand raised", int(row["hand_raised"]))
        s2.metric("👁 Distracted", f"{distr}%")
        s3.metric("Teacher feedback", f"🟢 {int(row['praised'])}  ·  🔴 {int(row['shouted'])}")

        st.caption("💡 *High questions + Curious dominant emotion = strong inquiry-based engagement.*")

    st.markdown("---")

    # 2b. Engagement Heatmap
    st.subheader("🔥 Engagement Heatmap")
    st.markdown("Rows = students (sorted most engaged at the top). Columns = time. Green = engaged / curious. Red = bored / distracted. Hover for details.")

    score_pivot, hover_pivot = metrics.get_engagement_heatmap(vid)
    if score_pivot.empty:
        st.info("Not enough visual data to build a heatmap.")
    else:
        import plotly.graph_objects as go
        fig_hm = go.Figure(data=go.Heatmap(
            z=score_pivot.values,
            x=list(score_pivot.columns),
            y=list(score_pivot.index),
            text=hover_pivot.values,
            hovertemplate="%{text}<extra></extra>",
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            colorbar=dict(title="Engagement", tickvals=[0, 0.5, 1], ticktext=["Low", "Neutral", "High"]),
        ))
        fig_hm.update_layout(
            xaxis=dict(title="Time (MM:SS)", side="bottom"),
            yaxis=dict(title="", autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=40),
            height=max(280, 40 + 28 * len(score_pivot.index)),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("💡 *A vertical band of red across all students = something the teacher did lost the room. Isolated red rows = specific students needing check-in.*")

# --- Footer ---
st.markdown("---")
footer_html = (
    "<div style='text-align:center;color:#888;padding:20px 0;font-size:0.9rem;'>"
    "Built by <strong>Surya Prakash</strong> · Data Analytics & AI<br>"
    "<a href='https://github.com/psurya19997/Classroom-Analytics-App' style='color:#4da6ff;text-decoration:none;margin:0 8px;'>🐙 GitHub</a>"
    " · "
    "<a href='https://www.linkedin.com/in/surya-prakash-a8464420b/' style='color:#4da6ff;text-decoration:none;margin:0 8px;'>💼 LinkedIn</a>"
    " · "
    "<a href='mailto:suryap19997@gmail.com' style='color:#4da6ff;text-decoration:none;margin:0 8px;'>✉️ Email</a>"
    "<br><br>"
    "<span style='color:#666;font-size:0.8rem;'>Source: view the code on GitHub. Feedback welcome.</span>"
    "</div>"
)
st.markdown(footer_html, unsafe_allow_html=True)
