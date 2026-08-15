import os
import json
import subprocess
import hashlib
from datetime import datetime
import glob
import difflib

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

import db

CACHE_DIR = "cache"
TMP_DIR = "tmp"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

class ScriptRow(BaseModel):
    start_sec: float
    end_sec: float
    speaker_norm: str = Field(description="'Student' or 'Teacher'")
    speaker_name: str | None = Field(description="If known from context/vision. Nullable.")
    utterance: str
    is_question: bool
    emotion_audio: str = Field(description=(
        "Emotion based ONLY on voice tone/prosody — IGNORE the words. Pick exactly one of: "
        "Attentive (active listening backchannels, engaged pauses); "
        "Curious (upward inflection, question-like tone); "
        "Joyful (bright, energetic delivery, laughter); "
        "Neutral (flat, unremarkable delivery); "
        "Confused (hesitant, unfinished phrases, uptalk); "
        "Bored (monotone, low energy, slow pace); "
        "Frustrated (tense, sharp, loud, terse)."
    ))
    confidence_audio: float = Field(description="0.0 to 1.0")
    emotion_text: str = Field(description=(
        "Emotion based ONLY on the words spoken — IGNORE the tone. Pick exactly one of: "
        "Attentive (on-topic response, engages with the lesson content); "
        "Curious (contains a question, 'why', 'how', 'what if'); "
        "Joyful (positive words, exclamations of enjoyment); "
        "Neutral (factual, matter-of-fact statement); "
        "Confused ('I don't get it', 'wait', 'huh', 'sorry?'); "
        "Bored (short disengaged replies, off-topic muttering); "
        "Frustrated (negative words, complaints, 'this is stupid')."
    ))
    confidence_text: float = Field(description="0.0 to 1.0")
    keywords: list[str]
    teacher_behavior_flag: str = Field(description="'Shouting', 'Praising', 'Demotivating', or 'None'")
    targeted_child_name: str | None = Field(description="Child teacher is addressing, if any.")

class AudioResponse(BaseModel):
    script: list[ScriptRow]

class ChildEmotionRow(BaseModel):
    name: str = Field(description="Display name read from video feed, or 'Kid_X'")
    role: str = Field(description="'student' or 'teacher'")
    appearance: str = Field(description="Visual description for disambiguation")
    emotion_visual: str = Field(description=(
        "Emotion based ONLY on face and posture — IGNORE audio if any. Pick exactly one of: "
        "Attentive (gaze on screen/notes, upright or forward posture); "
        "Curious (leaning in, brow raise, head tilt with focus); "
        "Joyful (smiling, laughing, animated expression); "
        "Neutral (blank but present, no strong signal); "
        "Confused (furrowed brow, mouth agape, tilted head without focus); "
        "Bored (slouched, drooping eyes, head resting on hand); "
        "Frustrated (jaw tension, tight mouth, glaring)."
    ))
    confidence_visual: float
    facial_expression: str
    posture: str
    hand_raised: bool
    distracted: bool

class FrameLog(BaseModel):
    attendance_count: int
    camera_on_count: int
    teacher_visible: bool
    child_emotions: list[ChildEmotionRow]

class VisionResponse(BaseModel):
    frames: list[FrameLog]

def get_video_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]

def get_duration(file_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def download_video(url, video_id):
    video_path = os.path.join(TMP_DIR, f"{video_id}.mp4")
    title = None

    if os.path.exists(video_path):
        return video_path, title

    if "youtube.com" in url or "youtu.be" in url:
        from yt_dlp import YoutubeDL
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': video_path,
            'quiet': True
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title') if info else None
    elif "drive.google.com" in url:
        import gdown
        import re
        # Accept /file/d/<ID>/..., open?id=<ID>, or uc?id=<ID>
        m = re.search(r'/file/d/([^/]+)', url) or re.search(r'[?&]id=([^&]+)', url)
        if m:
            direct_url = f"https://drive.google.com/uc?id={m.group(1)}"
        else:
            direct_url = url
        try:
            gdown.download(direct_url, video_path, quiet=False, fuzzy=True)
        except TypeError:
            # Older gdown (<4.4) has no fuzzy kwarg
            gdown.download(direct_url, video_path, quiet=False)
    else:
        raise ValueError("Unsupported URL format")

    return video_path, title

def extract_audio(video_path, video_id):
    wav_path = os.path.join(TMP_DIR, f"{video_id}_16k.wav")
    if not os.path.exists(wav_path):
        cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', wav_path]
        subprocess.run(cmd, check=True)
    return wav_path

def extract_frames(video_path, video_id, duration_sec):
    frames_dir = os.path.join(TMP_DIR, f"{video_id}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    if duration_sec < 600:
        bucket_sec = 20
    elif duration_sec < 1800:
        bucket_sec = 30
    else:
        bucket_sec = 60

    frame_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vf', f'fps=1/{bucket_sec}', frame_pattern]
    subprocess.run(cmd, check=True)
    return frames_dir, bucket_sec

def fuzzy_match(name, choices):
    if not name or not choices:
        return None
    matches = difflib.get_close_matches(name, choices, n=1, cutoff=0.6)
    return matches[0] if matches else None

def run_pipeline(url, api_key, progress_callback=None):
    video_id = get_video_id(url)
    db_path = db.get_db_path(video_id)
    
    if os.path.exists(db_path):
        if progress_callback: progress_callback("Loading from SQLite Cache...", 100)
        return {"status": "cached", "video_id": video_id}

    if progress_callback: progress_callback("Downloading video...", 10)
    video_path, video_title = download_video(url, video_id)
    duration_sec = get_duration(video_path)
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    
    if progress_callback: progress_callback("Extracting Audio & Frames...", 30)
    audio_path = extract_audio(video_path, video_id)
    frames_dir, bucket_sec = extract_frames(video_path, video_id, duration_sec)
    
    db.init_db(video_id)
    conn = db.get_connection(video_id)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO video_metadata (url_hash, original_url, video_title, processed_date, total_duration_sec, file_size_mb)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (video_id, url, video_title, datetime.now().isoformat(), duration_sec, file_size_mb))
    metadata_id = c.lastrowid
    
    client = genai.Client(api_key=api_key)

    # 1. Vision Processing (run FIRST so we can inject known names into the audio prompt)
    if progress_callback: progress_callback("Running Gemini Vision Analysis on Frames...", 50)
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    uploaded_frames = []
    for f in frame_files:
        uploaded_frames.append(client.files.upload(file=f))

    vision_prompt = f"""
    These are sequentially extracted frames from a classroom video (1 frame every {bucket_sec} seconds).
    You MUST analyze attendance and list the emotions of EVERY child visible, even if they are in tiny thumbnail boxes on the edge of the screen!
    For each child you spot, list their facial expressions and posture.
    Judge each child's `emotion_visual` STRICTLY from face and posture — do not infer from any audio context.
    Use the exact 7-emotion vocabulary defined in the schema.
    Read their display names if visible, otherwise describe them (e.g. 'Kid with red shirt').
    Do NOT return an empty frames array. You must provide a log for every single frame provided!
    """
    vision_resp = client.models.generate_content(
        model='gemini-flash-latest',
        contents=uploaded_frames + [vision_prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=VisionResponse)
    )
    vision_data = json.loads(vision_resp.text)

    # Build participant hint from vision output for the audio call
    seen_names = {}  # name -> role
    for f_log in vision_data.get("frames", []):
        for child in f_log.get("child_emotions", []):
            nm = child.get("name")
            if nm and nm not in seen_names:
                seen_names[nm] = child.get("role", "student")
    if seen_names:
        participant_hint = "\n".join(f"- {n} ({r})" for n, r in seen_names.items())
        roster_block = f"\nKnown participants visible in this class:\n{participant_hint}\n\nWhen attributing an utterance, set `speaker_name` and `targeted_child_name` to one of the names above ONLY if the audio makes it reasonably clear (e.g., someone addresses them by name, or an earlier line established the speaker). Otherwise leave them null."
    else:
        roster_block = ""

    # 2. Audio Processing
    if progress_callback: progress_callback("Running Gemini Audio/Text Analysis...", 70)
    audio_file = client.files.upload(file=audio_path)
    audio_prompt = f"""
    Transcribe and diarize this classroom audio. You MUST return a script array, even if it's just one person talking.
    For each utterance, extract two INDEPENDENT emotional judgments:
    - `emotion_audio`: judged ONLY from voice tone/prosody — do NOT let the words influence you.
    - `emotion_text`: judged ONLY from the words said — do NOT let the tone influence you.
    They should sometimes disagree on the same utterance; that's expected and desirable.
    Provide confidence scores (0.0 to 1.0) for both. Use the exact 7-emotion vocabulary defined in the schema.
    Flag teacher behaviors (Shouting, Praising, Demotivating, or None). If unsure, use 'None'.
    If a child's name isn't spoken, leave it null. Do not return an empty array if speech exists!
    {roster_block}
    """
    audio_resp = client.models.generate_content(
        model='gemini-flash-latest',
        contents=[audio_file, audio_prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=AudioResponse)
    )
    script_data = json.loads(audio_resp.text)
    
    # 3. Database Insertion
    if progress_callback: progress_callback("Saving to SQLite Database...", 90)
    
    child_map = {} # Maps string name -> child_id
    
    for i, f_log in enumerate(vision_data.get("frames", [])):
        t_sec = (i + 1) * bucket_sec
        c.execute('''
            INSERT INTO visual_frame_log (video_id, timestamp_sec, attendance_count, camera_on_count, kids_not_visible_count, teacher_visible)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (metadata_id, t_sec, f_log["attendance_count"], f_log["camera_on_count"], max(0, f_log["attendance_count"] - f_log["camera_on_count"]), f_log["teacher_visible"]))
        
        for child in f_log.get("child_emotions", []):
            name = child["name"]
            if name not in child_map:
                c.execute('''
                    INSERT INTO child_registry (video_id, name, name_source, role, appearance, first_seen_sec)
                    VALUES (?, ?, 'visible', ?, ?, ?)
                ''', (metadata_id, name, child["role"], child["appearance"], t_sec))
                child_map[name] = c.lastrowid
            
            c_id = child_map[name]
            c.execute('''
                INSERT INTO child_frame_emotion (video_id, timestamp_sec, child_id, emotion_visual, confidence_visual, facial_expression, posture, hand_raised, distracted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (metadata_id, t_sec, c_id, child["emotion_visual"], child["confidence_visual"], child["facial_expression"], child["posture"], child["hand_raised"], child["distracted"]))
            
    # Post-pipeline aggregate for child_registry
    c.execute('''
        UPDATE child_registry
        SET total_frames_present = (
            SELECT COUNT(*) FROM child_frame_emotion WHERE child_frame_emotion.child_id = child_registry.child_id
        )
    ''')
            
    known_names = list(child_map.keys())
            
    for row in script_data.get("script", []):
        matched_spk = fuzzy_match(row["speaker_name"], known_names)
        spk_id = child_map[matched_spk] if matched_spk else None
        
        matched_tgt = fuzzy_match(row["targeted_child_name"], known_names)
        tgt_id = child_map[matched_tgt] if matched_tgt else None
        
        duration = max(0.1, row["end_sec"] - row["start_sec"])
        word_count = len(row["utterance"].split())
        speaking_rate = round(word_count / duration, 2)
        
        c.execute('''
            INSERT INTO script_storage (video_id, start_sec, end_sec, speaker_norm, speaker_child_id, utterance, is_question, speaking_rate, emotion_audio, confidence_audio, emotion_text, confidence_text, keywords, teacher_behavior_flag, targeted_child_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (metadata_id, row["start_sec"], row["end_sec"], row["speaker_norm"], spk_id, row["utterance"], row["is_question"], speaking_rate, row["emotion_audio"], row["confidence_audio"], row["emotion_text"], row["confidence_text"], json.dumps(row["keywords"]), row["teacher_behavior_flag"], tgt_id))
        
    c.execute('UPDATE video_metadata SET frames_analyzed = ? WHERE video_id = ?', (len(frame_files), metadata_id))
    
    conn.commit()
    conn.close()
    
    if progress_callback: progress_callback("Done!", 100)
    return {"status": "success", "video_id": video_id}
