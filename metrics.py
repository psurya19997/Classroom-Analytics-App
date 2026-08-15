import pandas as pd
import db

def get_total_duration(video_id):
    conn = db.get_connection(video_id)
    df = pd.read_sql_query("SELECT total_duration_sec FROM video_metadata LIMIT 1", conn)
    conn.close()
    if not df.empty:
        return df.iloc[0]['total_duration_sec']
    return 1.0

def get_bucket_sec(duration_sec):
    if duration_sec < 600:
        return 20
    elif duration_sec < 1800:
        return 30
    else:
        return 60

def get_baseline_metrics(video_id):
    conn = db.get_connection(video_id)
    duration_min = max(0.1, get_total_duration(video_id) / 60.0)
    
    script_df = pd.read_sql_query("SELECT speaker_norm, is_question, utterance FROM script_storage ORDER BY start_sec", conn)
    
    student_words = 0
    dialogue_changes = 0
    student_questions = 0
    teacher_questions = 0
    student_init = 0
    teacher_init = 0
    
    prev_spk = None
    for _, row in script_df.iterrows():
        spk = row['speaker_norm']
        text = row['utterance'] or ""
        
        if spk == 'Student':
            student_words += len(text.split())
            if row['is_question']:
                student_questions += 1
        elif spk == 'Teacher':
            if row['is_question']:
                teacher_questions += 1
                
        if prev_spk and spk != prev_spk:
            dialogue_changes += 1
        prev_spk = spk

    script_full = pd.read_sql_query("SELECT speaker_norm, start_sec, end_sec FROM script_storage ORDER BY start_sec", conn)
    conn.close()
    
    for i in range(1, len(script_full)):
        row = script_full.iloc[i]
        prev_row = script_full.iloc[i-1]
        
        if row['speaker_norm'] != prev_row['speaker_norm']:
            gap = row['start_sec'] - prev_row['end_sec']
            if gap >= 3.0:
                if row['speaker_norm'] == 'Student':
                    student_init += 1
                else:
                    teacher_init += 1

    student_agency = round(student_init / teacher_init, 2) if teacher_init > 0 else 0
    
    return {
        "student_talk_rate": round(student_words / duration_min, 2),
        "dialogue_frequency": round(dialogue_changes / duration_min, 2),
        "student_question_rate": round(student_questions / duration_min, 2),
        "teacher_question_rate": round(teacher_questions / duration_min, 2),
        "student_agency": student_agency
    }

def get_child_emotion_timeline(video_id):
    duration_sec = get_total_duration(video_id)
    b_sec = get_bucket_sec(duration_sec)
    conn = db.get_connection(video_id)
    
    v_df = pd.read_sql_query('''
        SELECT c.timestamp_sec as bucket, r.name as child_name, c.emotion_visual, c.confidence_visual
        FROM child_frame_emotion c
        JOIN child_registry r ON c.child_id = r.child_id
        WHERE r.role = 'student'
    ''', conn)

    a_df = pd.read_sql_query('''
        SELECT CAST(s.start_sec / ? AS INTEGER) * ? as bucket,
               r.name as child_name,
               s.emotion_audio, s.confidence_audio,
               s.emotion_text, s.confidence_text
        FROM script_storage s
        JOIN child_registry r ON s.speaker_child_id = r.child_id
        WHERE s.speaker_norm = 'Student' AND r.role = 'student'
    ''', conn, params=(b_sec, b_sec))
    conn.close()
    
    if v_df.empty and a_df.empty:
        return pd.DataFrame(columns=['bucket', 'child_name', 'holistic_emotion', 'confidence', 'time_min'])
        
    merged = pd.merge(v_df, a_df, on=['bucket', 'child_name'], how='outer')
    
    results = []
    for _, row in merged.iterrows():
        candidates = []
        if pd.notna(row.get('emotion_visual')) and pd.notna(row.get('confidence_visual')):
            candidates.append((row['emotion_visual'], row['confidence_visual']))
        if pd.notna(row.get('emotion_audio')) and pd.notna(row.get('confidence_audio')):
            candidates.append((row['emotion_audio'], row['confidence_audio']))
        if pd.notna(row.get('emotion_text')) and pd.notna(row.get('confidence_text')):
            candidates.append((row['emotion_text'], row['confidence_text']))
            
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_emotion, best_conf = candidates[0]
            results.append({
                'bucket': row['bucket'],
                'child_name': row['child_name'],
                'holistic_emotion': best_emotion,
                'confidence': best_conf
            })
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df['time_min'] = res_df['bucket'] / 60.0
    return res_df

def get_teacher_incidents(video_id):
    conn = db.get_connection(video_id)
    df = pd.read_sql_query('''
        SELECT s.start_sec, s.teacher_behavior_flag, r.name as targeted_child, s.utterance
        FROM script_storage s
        LEFT JOIN child_registry r ON s.targeted_child_id = r.child_id
        WHERE s.teacher_behavior_flag != 'None' AND s.teacher_behavior_flag IS NOT NULL
        ORDER BY s.start_sec
    ''', conn)
    conn.close()
    return df

def get_attendance_timeline(video_id):
    conn = db.get_connection(video_id)
    df = pd.read_sql_query('''
        SELECT timestamp_sec / 60.0 as time_min, attendance_count, camera_on_count
        FROM visual_frame_log
        ORDER BY timestamp_sec
    ''', conn)
    conn.close()
    return df

def get_video_header(video_id):
    conn = db.get_connection(video_id)
    df = pd.read_sql_query(
        "SELECT video_title, original_url, total_duration_sec, frames_analyzed, processed_date FROM video_metadata LIMIT 1",
        conn,
    )
    conn.close()
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def get_child_stats(video_id):
    """One row per student with all scorecard fields."""
    conn = db.get_connection(video_id)

    meta = pd.read_sql_query("SELECT frames_analyzed FROM video_metadata LIMIT 1", conn)
    total_frames = int(meta.iloc[0]["frames_analyzed"]) if not meta.empty and meta.iloc[0]["frames_analyzed"] else 1

    registry = pd.read_sql_query(
        "SELECT child_id, name, appearance, role, total_frames_present, first_seen_sec FROM child_registry WHERE role = 'student'",
        conn,
    )
    if registry.empty:
        conn.close()
        return pd.DataFrame()

    cfe = pd.read_sql_query(
        "SELECT child_id, emotion_visual, confidence_visual, hand_raised, distracted FROM child_frame_emotion",
        conn,
    )
    scripts = pd.read_sql_query(
        "SELECT speaker_child_id, targeted_child_id, is_question, teacher_behavior_flag, start_sec, end_sec, utterance FROM script_storage",
        conn,
    )
    conn.close()

    rows = []
    for _, r in registry.iterrows():
        cid = r["child_id"]
        my_frames = cfe[cfe["child_id"] == cid]
        my_utts = scripts[scripts["speaker_child_id"] == cid]
        addressed = scripts[scripts["targeted_child_id"] == cid]

        attendance_pct = round(100.0 * (r["total_frames_present"] or 0) / total_frames, 1)

        if not my_frames.empty:
            weighted = my_frames.groupby("emotion_visual")["confidence_visual"].sum().sort_values(ascending=False)
            dominant_emotion = weighted.index[0]
            hand_raised_count = int(my_frames["hand_raised"].sum())
            distracted_pct = round(100.0 * my_frames["distracted"].sum() / len(my_frames), 1)
        else:
            dominant_emotion = "—"
            hand_raised_count = 0
            distracted_pct = 0.0

        spoke_times = len(my_utts)
        questions_asked = int(my_utts["is_question"].sum()) if not my_utts.empty else 0
        speaking_sec = float((my_utts["end_sec"] - my_utts["start_sec"]).sum()) if not my_utts.empty else 0.0
        praised = int(((addressed["teacher_behavior_flag"] == "Praising")).sum()) if not addressed.empty else 0
        shouted = int(((addressed["teacher_behavior_flag"] == "Shouting")).sum()) if not addressed.empty else 0

        rows.append({
            "child_id": cid,
            "name": r["name"],
            "appearance": r["appearance"] or "",
            "attendance_pct": attendance_pct,
            "dominant_emotion": dominant_emotion,
            "questions_asked": questions_asked,
            "spoke_times": spoke_times,
            "hand_raised": hand_raised_count,
            "distracted_pct": distracted_pct,
            "speaking_sec": speaking_sec,
            "praised": praised,
            "shouted": shouted,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("attendance_pct", ascending=False).reset_index(drop=True)


ENGAGEMENT_SCORE = {
    "Attentive": 1.0, "Curious": 1.0, "Joyful": 0.95, "Happy": 0.9,
    "Engaged": 0.9, "Interested": 0.85,
    "Neutral": 0.5, "Calm": 0.55,
    "Confused": 0.3, "Frustrated": 0.15,
    "Bored": 0.15, "Distracted": 0.1, "Sleepy": 0.05,
}


def _emotion_to_score(e):
    if not e:
        return 0.5
    return ENGAGEMENT_SCORE.get(str(e).strip().capitalize(), 0.5)


def get_engagement_heatmap(video_id):
    """Returns (matrix DataFrame [students × time_buckets], hover_text DataFrame same shape)."""
    conn = db.get_connection(video_id)
    df = pd.read_sql_query(
        """
        SELECT c.timestamp_sec, r.name, c.emotion_visual, c.confidence_visual,
               c.hand_raised, c.distracted, c.facial_expression, c.posture
        FROM child_frame_emotion c
        JOIN child_registry r ON c.child_id = r.child_id
        WHERE r.role = 'student'
        ORDER BY r.name, c.timestamp_sec
        """,
        conn,
    )
    conn.close()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["engagement"] = df["emotion_visual"].apply(_emotion_to_score)
    df["time_label"] = df["timestamp_sec"].apply(lambda s: f"{int(s//60):02d}:{int(s%60):02d}")

    score_pivot = df.pivot_table(index="name", columns="time_label", values="engagement", aggfunc="mean")

    df["hover"] = df.apply(
        lambda r: (
            f"<b>{r['name']}</b> @ {r['time_label']}<br>"
            f"Emotion: {r['emotion_visual']} (conf {r['confidence_visual']:.2f})<br>"
            f"Face: {r['facial_expression']} · Posture: {r['posture']}<br>"
            f"{'✋ hand raised · ' if r['hand_raised'] else ''}"
            f"{'👁 distracted' if r['distracted'] else ''}"
        ),
        axis=1,
    )
    hover_pivot = df.pivot_table(index="name", columns="time_label", values="hover", aggfunc="first")

    order = score_pivot.mean(axis=1).sort_values(ascending=False).index
    return score_pivot.loc[order], hover_pivot.loc[order]


def get_raw_transcript(video_id):
    conn = db.get_connection(video_id)
    df = pd.read_sql_query('''
        SELECT s.start_sec, s.end_sec, s.speaker_norm, r.name as specific_name, s.utterance, s.is_question, s.emotion_text
        FROM script_storage s
        LEFT JOIN child_registry r ON s.speaker_child_id = r.child_id
        ORDER BY s.start_sec
    ''', conn)
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=['time_range', 'speaker_display', 'utterance', 'emotion_text', 'is_question'])
        
    df['speaker_display'] = df.apply(lambda row: row['specific_name'] if pd.notna(row['specific_name']) else row['speaker_norm'], axis=1)
    df['time_range'] = df.apply(lambda row: f"{int(row['start_sec']//60)}:{int(row['start_sec']%60):02d} - {int(row['end_sec']//60)}:{int(row['end_sec']%60):02d}", axis=1)
    
    return df[['time_range', 'speaker_display', 'utterance', 'emotion_text', 'is_question']]
