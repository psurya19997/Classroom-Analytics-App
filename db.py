import sqlite3
import os

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_db_path(video_id):
    return os.path.join(CACHE_DIR, f"{video_id}.db")

def init_db(video_id):
    """Initializes the database schema for a given video."""
    db_path = get_db_path(video_id)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Table 1: video_metadata
    c.execute('''
        CREATE TABLE IF NOT EXISTS video_metadata (
            video_id INTEGER PRIMARY KEY AUTOINCREMENT,
            url_hash TEXT UNIQUE,
            original_url TEXT,
            source_platform TEXT,
            video_title TEXT,
            upload_date DATE,
            processed_date DATETIME,
            total_duration_sec REAL,
            file_size_mb REAL,
            resolution TEXT,
            frames_analyzed INTEGER
        )
    ''')

    # Table 2: child_registry
    c.execute('''
        CREATE TABLE IF NOT EXISTS child_registry (
            child_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            name TEXT,
            name_source TEXT,
            appearance TEXT,
            role TEXT,
            first_seen_sec REAL,
            total_frames_present INTEGER,
            FOREIGN KEY (video_id) REFERENCES video_metadata (video_id)
        )
    ''')

    # Table 3: visual_frame_log
    c.execute('''
        CREATE TABLE IF NOT EXISTS visual_frame_log (
            video_id INTEGER,
            timestamp_sec INTEGER,
            attendance_count INTEGER,
            camera_on_count INTEGER,
            kids_not_visible_count INTEGER,
            teacher_visible BOOLEAN,
            PRIMARY KEY (video_id, timestamp_sec),
            FOREIGN KEY (video_id) REFERENCES video_metadata (video_id)
        )
    ''')

    # Table 4: child_frame_emotion
    c.execute('''
        CREATE TABLE IF NOT EXISTS child_frame_emotion (
            video_id INTEGER,
            timestamp_sec INTEGER,
            child_id INTEGER,
            emotion_visual TEXT,
            confidence_visual REAL,
            facial_expression TEXT,
            posture TEXT,
            hand_raised BOOLEAN,
            distracted BOOLEAN,
            PRIMARY KEY (video_id, timestamp_sec, child_id),
            FOREIGN KEY (video_id) REFERENCES video_metadata (video_id),
            FOREIGN KEY (child_id) REFERENCES child_registry (child_id)
        )
    ''')

    # Table 5: script_storage
    c.execute('''
        CREATE TABLE IF NOT EXISTS script_storage (
            utterance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            start_sec REAL,
            end_sec REAL,
            speaker_norm TEXT,
            speaker_child_id INTEGER,
            utterance TEXT,
            is_question BOOLEAN,
            speaking_rate REAL,
            emotion_audio TEXT,
            confidence_audio REAL,
            emotion_text TEXT,
            confidence_text REAL,
            keywords TEXT,
            teacher_behavior_flag TEXT,
            targeted_child_id INTEGER,
            FOREIGN KEY (video_id) REFERENCES video_metadata (video_id),
            FOREIGN KEY (speaker_child_id) REFERENCES child_registry (child_id),
            FOREIGN KEY (targeted_child_id) REFERENCES child_registry (child_id)
        )
    ''')

    conn.commit()
    conn.close()

def get_connection(video_id):
    """Returns a connection to the video's database."""
    return sqlite3.connect(get_db_path(video_id))
