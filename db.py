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
            frames_analyzed INTEGER,
            model_vision TEXT,
            model_audio TEXT,
            status TEXT,
            error_message TEXT
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
            speaker_tag TEXT,
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

    # Table 6: pipeline_run_log — one row per stage/batch for debugging
    c.execute('''
        CREATE TABLE IF NOT EXISTS pipeline_run_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            stage TEXT,
            batch_index INTEGER,
            started_at DATETIME,
            duration_ms INTEGER,
            status TEXT,
            finish_reason TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            error TEXT,
            FOREIGN KEY (video_id) REFERENCES video_metadata (video_id)
        )
    ''')

    # Table 7: speaker_identity_map
    c.execute('''
        CREATE TABLE IF NOT EXISTS speaker_identity_map (
            video_id INTEGER,
            speaker_tag TEXT,
            child_id INTEGER,
            vote_result TEXT,
            diarization_split BOOLEAN DEFAULT 0,
            PRIMARY KEY (video_id, speaker_tag),
            FOREIGN KEY (video_id) REFERENCES video_metadata (video_id),
            FOREIGN KEY (child_id) REFERENCES child_registry (child_id)
        )
    ''')

    _migrate_schema(c)

    conn.commit()
    conn.close()


def _migrate_schema(c):
    """Idempotently add columns/tables to older DBs so they keep loading."""
    c.execute("PRAGMA table_info(video_metadata)")
    existing = {row[1] for row in c.fetchall()}
    for col, decl in [
        ("model_vision", "TEXT"),
        ("model_audio", "TEXT"),
        ("status", "TEXT"),
        ("error_message", "TEXT"),
    ]:
        if col not in existing:
            c.execute(f"ALTER TABLE video_metadata ADD COLUMN {col} {decl}")

    # Add speaker_tag to script_storage if missing
    c.execute("PRAGMA table_info(script_storage)")
    existing_script = {row[1] for row in c.fetchall()}
    if "speaker_tag" not in existing_script:
        c.execute("ALTER TABLE script_storage ADD COLUMN speaker_tag TEXT")

    # Ensure speaker_identity_map exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS speaker_identity_map (
            video_id INTEGER,
            speaker_tag TEXT,
            child_id INTEGER,
            vote_result TEXT,
            diarization_split BOOLEAN DEFAULT 0,
            PRIMARY KEY (video_id, speaker_tag),
            FOREIGN KEY (video_id) REFERENCES video_metadata (video_id),
            FOREIGN KEY (child_id) REFERENCES child_registry (child_id)
        )
    ''')


def migrate_existing_db(video_id):
    """Run migrations on an already-existing cache DB (no-op if up to date)."""
    db_path = get_db_path(video_id)
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # ensure the run-log table exists even for old DBs
    c.execute('''
        CREATE TABLE IF NOT EXISTS pipeline_run_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            stage TEXT,
            batch_index INTEGER,
            started_at DATETIME,
            duration_ms INTEGER,
            status TEXT,
            finish_reason TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            error TEXT
        )
    ''')
    _migrate_schema(c)
    conn.commit()
    conn.close()

def get_connection(video_id):
    """Returns a connection to the video's database, migrating schema if needed."""
    migrate_existing_db(video_id)
    return sqlite3.connect(get_db_path(video_id))


def update_video_title(video_id, new_title):
    """Set video_metadata.video_title on the given video's DB."""
    conn = sqlite3.connect(get_db_path(video_id))
    conn.execute("UPDATE video_metadata SET video_title = ?", (new_title,))
    conn.commit()
    conn.close()


def list_analyses():
    """Walk cache/ and return one row per analyzed video, newest first."""
    rows = []
    if not os.path.isdir(CACHE_DIR):
        return rows
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".db"):
            continue
        video_id = fname[:-3]
        db_path = os.path.join(CACHE_DIR, fname)
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT url_hash, original_url, video_title, processed_date, total_duration_sec
                FROM video_metadata LIMIT 1
            """)
            r = cur.fetchone()
            conn.close()
            if r:
                rows.append({
                    "video_id": video_id,
                    "url_hash": r[0],
                    "original_url": r[1],
                    "video_title": r[2],
                    "processed_date": r[3],
                    "total_duration_sec": r[4],
                })
        except sqlite3.Error:
            continue
    rows.sort(key=lambda x: x["processed_date"] or "", reverse=True)
    return rows
