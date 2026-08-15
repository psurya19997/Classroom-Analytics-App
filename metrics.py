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
    ''', conn)
    
    a_df = pd.read_sql_query('''
        SELECT CAST(s.start_sec / ? AS INTEGER) * ? as bucket, 
               r.name as child_name, 
               s.emotion_audio, s.confidence_audio, 
               s.emotion_text, s.confidence_text
        FROM script_storage s
        JOIN child_registry r ON s.speaker_child_id = r.child_id
        WHERE s.speaker_norm = 'Student'
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

def get_raw_transcript(video_id):
    conn = db.get_connection(video_id)
    df = pd.read_sql_query('''
        SELECT s.start_sec, s.end_sec, s.speaker_norm, r.name as specific_name, s.utterance, s.is_question, s.emotion_text
        FROM script_storage s
        LEFT JOIN child_registry r ON s.speaker_child_id = r.child_id
        ORDER BY s.start_sec
    ''', conn)
    conn.close()
    
    df['speaker_display'] = df.apply(lambda row: row['specific_name'] if pd.notna(row['specific_name']) else row['speaker_norm'], axis=1)
    df['time_range'] = df.apply(lambda row: f"{int(row['start_sec']//60)}:{int(row['start_sec']%60):02d} - {int(row['end_sec']//60)}:{int(row['end_sec']%60):02d}", axis=1)
    
    return df[['time_range', 'speaker_display', 'utterance', 'emotion_text', 'is_question']]
