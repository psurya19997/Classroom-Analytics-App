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


def _triangulate_per_bucket(video_id):
    """
    Pure weighted-sum triangulation per (student, time-bucket).
    For each bucket, sum confidence per emotion across all available modalities
    (face + voice + text). Emotion with highest total wins.

    Columns:
      child_id, name, bucket, best_emotion, best_score,
      agreement_count       - how many modalities picked the winning emotion (1..3)
      modalities_present    - how many modalities had data in this bucket (1..3)
      facial_expression, posture, hand_raised, distracted   - visual context for hover
    """
    duration_sec = get_total_duration(video_id)
    b_sec = get_bucket_sec(duration_sec)

    conn = db.get_connection(video_id)
    v_df = pd.read_sql_query(
        """
        SELECT r.child_id, r.name, c.timestamp_sec as bucket,
               c.emotion_visual, c.confidence_visual,
               c.facial_expression, c.posture, c.hand_raised, c.distracted
        FROM child_frame_emotion c
        JOIN child_registry r ON c.child_id = r.child_id
        WHERE r.role = 'student'
        """,
        conn,
    )
    a_df = pd.read_sql_query(
        f"""
        SELECT r.child_id, r.name,
               CAST(s.start_sec / {b_sec} AS INTEGER) * {b_sec} as bucket,
               s.emotion_audio, s.confidence_audio,
               s.emotion_text,  s.confidence_text
        FROM script_storage s
        JOIN child_registry r ON s.speaker_child_id = r.child_id
        WHERE s.speaker_norm = 'Student' AND r.role = 'student'
        """,
        conn,
    )
    conn.close()

    # If multiple utterances in a bucket, keep the one with highest audio confidence
    if not a_df.empty:
        a_df = a_df.sort_values("confidence_audio", ascending=False).drop_duplicates(
            subset=["child_id", "name", "bucket"], keep="first"
        )

    if v_df.empty and a_df.empty:
        return pd.DataFrame(columns=[
            "child_id", "name", "bucket", "best_emotion", "best_score",
            "agreement_count", "modalities_present",
            "facial_expression", "posture", "hand_raised", "distracted",
        ])

    merged = pd.merge(v_df, a_df, on=["child_id", "name", "bucket"], how="outer")

    def _pick(row):
        # Collect (emotion, confidence) pairs from whichever modalities are present
        votes = []
        if pd.notna(row.get("emotion_visual")) and pd.notna(row.get("confidence_visual")):
            votes.append((str(row["emotion_visual"]).strip().capitalize(), float(row["confidence_visual"])))
        if pd.notna(row.get("emotion_audio")) and pd.notna(row.get("confidence_audio")):
            votes.append((str(row["emotion_audio"]).strip().capitalize(), float(row["confidence_audio"])))
        if pd.notna(row.get("emotion_text")) and pd.notna(row.get("confidence_text")):
            votes.append((str(row["emotion_text"]).strip().capitalize(), float(row["confidence_text"])))
        n_present = len(votes)
        if n_present == 0:
            return pd.Series({
                "best_emotion": None, "best_score": None,
                "agreement_count": 0, "modalities_present": 0,
            })
        # Pure weighted sum: sum confidence per emotion across modalities
        totals = {}
        for emo, conf in votes:
            totals[emo] = totals.get(emo, 0.0) + conf
        best_emotion = max(totals, key=totals.get)
        best_score = totals[best_emotion]
        agree = sum(1 for e, _ in votes if e == best_emotion)
        return pd.Series({
            "best_emotion": best_emotion,
            "best_score": best_score,
            "agreement_count": agree,
            "modalities_present": n_present,
        })

    picks = merged.apply(_pick, axis=1)
    out = pd.concat([merged, picks], axis=1)
    return out[[
        "child_id", "name", "bucket", "best_emotion", "best_score",
        "agreement_count", "modalities_present",
        "facial_expression", "posture", "hand_raised", "distracted",
    ]]


def _consensus_level(agreement_ratio):
    """Map avg agreement ratio in [0,1] to a badge. 1.0 = all modalities agreed."""
    if agreement_ratio is None:
        return None
    if agreement_ratio >= 0.85:
        return ("🟢", "High consensus", "#22c55e")
    if agreement_ratio >= 0.55:
        return ("🟡", "Mixed consensus", "#eab308")
    return ("🔴", "Low consensus", "#ef4444")


def get_child_stats(video_id):
    """One row per student with all scorecard fields. Dominant emotion is triangulated across all 3 modalities."""
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
        "SELECT child_id, hand_raised, distracted FROM child_frame_emotion",
        conn,
    )
    scripts = pd.read_sql_query(
        "SELECT speaker_child_id, targeted_child_id, is_question, teacher_behavior_flag, start_sec, end_sec, utterance FROM script_storage",
        conn,
    )
    conn.close()

    # Triangulated per-bucket emotions across all 3 modalities
    tri = _triangulate_per_bucket(video_id)

    rows = []
    for _, r in registry.iterrows():
        cid = r["child_id"]
        my_frames = cfe[cfe["child_id"] == cid]
        my_utts = scripts[scripts["speaker_child_id"] == cid]
        addressed = scripts[scripts["targeted_child_id"] == cid]
        my_tri = tri[tri["child_id"] == cid] if not tri.empty else tri

        attendance_pct = round(100.0 * (r["total_frames_present"] or 0) / total_frames, 1)

        # Dominant emotion via score-weighted vote across all buckets
        if not my_tri.empty and my_tri["best_emotion"].notna().any():
            valid = my_tri.dropna(subset=["best_emotion"])
            weighted = valid.groupby("best_emotion")["best_score"].sum().sort_values(ascending=False)
            dominant_emotion = weighted.index[0]
            # Consensus for this student: average (agreement_count / modalities_present) across buckets
            ratios = valid["agreement_count"] / valid["modalities_present"].replace(0, 1)
            consensus_ratio = float(ratios.mean())
        else:
            dominant_emotion = "—"
            consensus_ratio = None

        if not my_frames.empty:
            hand_raised_count = int(my_frames["hand_raised"].sum())
            distracted_pct = round(100.0 * my_frames["distracted"].sum() / len(my_frames), 1)
        else:
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
            "consensus_ratio": consensus_ratio,  # 0..1 avg modality agreement
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
    """Returns (matrix DataFrame [students × time_buckets], hover_text DataFrame same shape).
    Cells reflect the emotion picked by pure weighted-sum vote across face / voice / text."""
    tri = _triangulate_per_bucket(video_id)
    if tri.empty:
        return pd.DataFrame(), pd.DataFrame()

    tri = tri.dropna(subset=["best_emotion"]).copy()
    if tri.empty:
        return pd.DataFrame(), pd.DataFrame()

    tri["engagement"] = tri["best_emotion"].apply(_emotion_to_score)
    tri["time_label"] = tri["bucket"].apply(lambda s: f"{int(s//60):02d}:{int(s%60):02d}")

    def _hover(r):
        face_ctx = ""
        if pd.notna(r.get("facial_expression")) or pd.notna(r.get("posture")):
            face_ctx = (
                f"Face: {r.get('facial_expression') or '—'} · "
                f"Posture: {r.get('posture') or '—'}<br>"
            )
        flags = ""
        if r.get("hand_raised"): flags += "✋ hand raised · "
        if r.get("distracted"):  flags += "👁 distracted"
        agree = int(r.get("agreement_count") or 0)
        present = int(r.get("modalities_present") or 0)
        if present == 0:
            consensus = ""
        elif agree == present and present >= 2:
            consensus = f"🟢 all {present} sources agreed<br>"
        elif agree >= 2:
            consensus = f"🟡 {agree} of {present} sources agreed<br>"
        else:
            consensus = f"🔴 sources disagreed ({present} available)<br>"
        return (
            f"<b>{r['name']}</b> @ {r['time_label']}<br>"
            f"<b>{r['best_emotion']}</b><br>"
            f"{consensus}"
            f"{face_ctx}"
            f"{flags}"
        )

    tri["hover"] = tri.apply(_hover, axis=1)

    score_pivot = tri.pivot_table(index="name", columns="time_label", values="engagement", aggfunc="mean")
    hover_pivot = tri.pivot_table(index="name", columns="time_label", values="hover", aggfunc="first")

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
