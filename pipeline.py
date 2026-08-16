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

VISION_BATCH_SIZE = 10  # frames per Gemini vision call; keeps output well under token cap
VISION_MODEL = 'gemini-flash-latest'
AUDIO_MODEL = 'gemini-flash-latest'

class ScriptRow(BaseModel):
    start_sec: float
    end_sec: float
    speaker_norm: str = Field(description="'Student' or 'Teacher'")
    speaker_tag: str = Field(description="Stable identifier for the voice, e.g., 'Speaker_1', 'Teacher'.")
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

class ActiveSpeakerBatchResponse(BaseModel):
    speaker_names: list[str | None]

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

def _norm_name(n):
    return " ".join(n.strip().casefold().split()) if n else ""

def _is_teacher_tag(tag):
    return _norm_name(tag) in {"teacher", "instructor", "ma'am", "sir", "maam"}

def fuzzy_match(name, choices, cutoff=0.85):
    """Case/whitespace-insensitive fuzzy match. Returns the ORIGINAL (display) choice string, or None."""
    if not name or not choices:
        return None
    norm_to_original = {}
    for c in choices:
        norm = _norm_name(c)
        if norm and norm not in norm_to_original:
            norm_to_original[norm] = c
    matches = difflib.get_close_matches(_norm_name(name), list(norm_to_original.keys()), n=1, cutoff=cutoff)
    return norm_to_original[matches[0]] if matches else None

def run_pipeline(url, api_key, progress_callback=None):
    video_id = get_video_id(url)
    db_path = db.get_db_path(video_id)

    if os.path.exists(db_path):
        if progress_callback: progress_callback("Loading from SQLite Cache...", 100)
        return {"status": "cached", "video_id": video_id}

    try:
        return _run_pipeline_inner(url, video_id, api_key, progress_callback)
    except Exception as e:
        # Cleanup: remove the partial cache DB so the next run re-processes
        # instead of short-circuiting on os.path.exists(db_path).
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except OSError:
                pass
        raise


def _run_pipeline_inner(url, video_id, api_key, progress_callback):
    db_path = db.get_db_path(video_id)

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
        INSERT INTO video_metadata (url_hash, original_url, video_title, processed_date, total_duration_sec, file_size_mb, model_vision, model_audio, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
    ''', (video_id, url, video_title, datetime.now().isoformat(), duration_sec, file_size_mb, VISION_MODEL, AUDIO_MODEL))
    metadata_id = c.lastrowid
    
    client = genai.Client(api_key=api_key)

    # 1. Vision Processing (run FIRST so we can inject known names into the audio prompt)
    if progress_callback: progress_callback("Running Gemini Vision Analysis on Frames...", 50)
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    uploaded_frames = [client.files.upload(file=f) for f in frame_files]

    vision_prompt = f"""
    These are sequentially extracted frames from a classroom video (1 frame every {bucket_sec} seconds).
    You MUST analyze attendance and list the emotions of EVERY child visible, even if they are in tiny thumbnail boxes on the edge of the screen!
    For each child you spot, list their facial expressions and posture.
    Judge each child's `emotion_visual` STRICTLY from face and posture — do not infer from any audio context.
    Use the exact 7-emotion vocabulary defined in the schema.
    Read their display names if visible, otherwise describe them (e.g. 'Kid with red shirt').
    Do NOT return an empty frames array. You must provide a log for every single frame provided!
    """

    total_batches = max(1, (len(uploaded_frames) + VISION_BATCH_SIZE - 1) // VISION_BATCH_SIZE)
    all_vision_frames = []
    for batch_idx in range(total_batches):
        lo = batch_idx * VISION_BATCH_SIZE
        hi = min(lo + VISION_BATCH_SIZE, len(uploaded_frames))
        batch = uploaded_frames[lo:hi]
        if progress_callback:
            pct = 50 + int(20 * (batch_idx + 1) / total_batches)
            progress_callback(f"Vision batch {batch_idx + 1}/{total_batches} (frames {lo + 1}-{hi})...", pct)

        vision_resp = client.models.generate_content(
            model=VISION_MODEL,
            contents=batch + [vision_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VisionResponse,
            ),
        )

        finish = vision_resp.candidates[0].finish_reason
        finish_name = getattr(finish, "name", str(finish))
        if finish_name != "STOP":
            raise RuntimeError(
                f"Vision batch {batch_idx + 1}/{total_batches} (frames {lo + 1}-{hi}) "
                f"did not finish cleanly: finish_reason={finish_name}. "
                "Lower VISION_BATCH_SIZE or set max_output_tokens."
            )

        batch_frames = json.loads(vision_resp.text).get("frames", [])
        if len(batch_frames) != len(batch):
            raise RuntimeError(
                f"Vision batch {batch_idx + 1}/{total_batches} returned {len(batch_frames)} "
                f"frames for {len(batch)} uploaded (truncation suspected)."
            )
        all_vision_frames.extend(batch_frames)

    if len(all_vision_frames) != len(frame_files):
        raise RuntimeError(
            f"Vision produced {len(all_vision_frames)} frames but {len(frame_files)} were uploaded."
        )
    vision_data = {"frames": all_vision_frames}

    # Build participant hint from vision output for the audio call
    seen_names = {}  # name -> role
    for f_log in vision_data.get("frames", []):
        for child in f_log.get("child_emotions", []):
            nm = child.get("name")
            if nm and nm not in seen_names:
                seen_names[nm] = child.get("role", "student")
    if seen_names:
        participant_hint = "\n".join(f"- {n} ({r})" for n, r in seen_names.items())
        roster_block = f"\nKnown participants visible in this class:\n{participant_hint}\n\nSet `targeted_child_name` to one of the names above ONLY if they are addressed directly. DO NOT guess names for `speaker_tag` — always use a stable tag like 'Speaker_1'."
    else:
        roster_block = ""

    # 2. Audio Processing
    if progress_callback: progress_callback("Running Gemini Audio/Text Analysis...", 70)
    audio_file = client.files.upload(file=audio_path)
    audio_prompt = f"""
    Transcribe and diarize this classroom audio. You MUST return a script array, even if it's just one person talking.
    For each utterance, assign a stable `speaker_tag` (e.g., 'Speaker_1', 'Teacher'). Do not try to guess real names for the tags.
    For each utterance, extract two INDEPENDENT emotional judgments:
    - `emotion_audio`: judged ONLY from voice tone/prosody — do NOT let the words influence you.
    - `emotion_text`: judged ONLY from the words said — do NOT let the tone influence you.
    They should sometimes disagree on the same utterance; that's expected and desirable.
    Provide confidence scores (0.0 to 1.0) for both. Use the exact 7-emotion vocabulary defined in the schema.
    Flag teacher behaviors (Shouting, Praising, Demotivating, or None). If unsure, use 'None'.
    If a targeted child's name isn't spoken, leave it null. Do not return an empty array if speech exists!
    {roster_block}
    """
    audio_resp = client.models.generate_content(
        model=AUDIO_MODEL,
        contents=[audio_file, audio_prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=AudioResponse)
    )
    script_data = json.loads(audio_resp.text)
    
    # 3. Database Insertion
    if progress_callback: progress_callback("Saving to SQLite Database...", 90)
    
    child_map = {}        # raw display name -> child_id (supports every alias Gemini emits)
    canonical_ids = {}    # normalized name -> child_id (the actual dedup key)

    for i, f_log in enumerate(vision_data.get("frames", [])):
        t_sec = (i + 1) * bucket_sec
        c.execute('''
            INSERT INTO visual_frame_log (video_id, timestamp_sec, attendance_count, camera_on_count, kids_not_visible_count, teacher_visible)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (metadata_id, t_sec, f_log["attendance_count"], f_log["camera_on_count"], max(0, f_log["attendance_count"] - f_log["camera_on_count"]), f_log["teacher_visible"]))

        for child in f_log.get("child_emotions", []):
            name = child["name"] or ""
            n_norm = _norm_name(name)
            if not n_norm:
                # Unnamed detection — can't attribute to any specific kid. Class-level
                # attendance_count above still records it; we just skip the per-child row
                # so nameless detections don't collapse into one phantom student.
                continue
            if n_norm not in canonical_ids:
                c.execute('''
                    INSERT INTO child_registry (video_id, name, name_source, role, appearance, first_seen_sec)
                    VALUES (?, ?, 'visible', ?, ?, ?)
                ''', (metadata_id, name, child["role"], child["appearance"], t_sec))
                canonical_ids[n_norm] = c.lastrowid
            child_map[name] = canonical_ids[n_norm]  # every raw variant points at the same id

            c_id = canonical_ids[n_norm]
            c.execute('''
                INSERT INTO child_frame_emotion (video_id, timestamp_sec, child_id, emotion_visual, confidence_visual, facial_expression, posture, hand_raised, distracted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (metadata_id, t_sec, c_id, child["emotion_visual"], child["confidence_visual"], child["facial_expression"], child["posture"], child["hand_raised"], child["distracted"]))
            
    known_names = list(child_map.keys())

    # 3.5 Targeted Identity Resolution
    if progress_callback: progress_callback("Resolving Speaker Identities...", 85)
    
    unique_tags = list({row["speaker_tag"] for row in script_data.get("script", [])})
    targeted_tags = [t for t in unique_tags if not _is_teacher_tag(t)]
    teacher_tags = [t for t in unique_tags if _is_teacher_tag(t)]
    
    samples_per_tag = {tag: [] for tag in targeted_tags}
    
    tag_utterances = {tag: [] for tag in targeted_tags}
    for row in script_data.get("script", []):
        tag = row["speaker_tag"]
        if tag in tag_utterances:
            tag_utterances[tag].append(row)
            
    # Stratified Sampling
    for tag, utts in tag_utterances.items():
        utts.sort(key=lambda x: x["start_sec"])
        if len(utts) <= 3:
            samples_per_tag[tag] = utts
        else:
            thirds = [
                utts[:len(utts)//3],
                utts[len(utts)//3:2*len(utts)//3],
                utts[2*len(utts)//3:]
            ]
            for chunk in thirds:
                if chunk:
                    longest = max(chunk, key=lambda x: x["end_sec"] - x["start_sec"])
                    samples_per_tag[tag].append(longest)

    targeted_frames = []
    for tag, samples in samples_per_tag.items():
        for i, s in enumerate(samples):
            mid_sec = s["start_sec"] + (s["end_sec"] - s["start_sec"]) / 2.0
            out_path = os.path.join(TMP_DIR, f"{video_id}_speaker_{tag}_{i}.jpg")
            subprocess.run(['ffmpeg', '-y', '-i', video_path, '-ss', str(mid_sec), '-vframes', '1', '-q:v', '2', out_path], capture_output=True, check=False)
            if os.path.exists(out_path):
                targeted_frames.append({"tag": tag, "path": out_path})
                
    tag_votes = {tag: [] for tag in targeted_tags}
    
    if targeted_frames:
        uploaded_target_frames = [client.files.upload(file=f["path"]) for f in targeted_frames]
        
        total_micro_batches = max(1, (len(targeted_frames) + VISION_BATCH_SIZE - 1) // VISION_BATCH_SIZE)
        for batch_idx in range(total_micro_batches):
            lo = batch_idx * VISION_BATCH_SIZE
            hi = min(lo + VISION_BATCH_SIZE, len(targeted_frames))
            batch_files = uploaded_target_frames[lo:hi]
            batch_frames_info = targeted_frames[lo:hi]
            
            micro_prompt = f"There are {len(batch_files)} frames extracted from a video conference. For each frame in order, read the UI highlight (e.g. green border) and provide the display name of the active speaker. If no one is highlighted, or it's a screen share, return null. The response array MUST have exactly {len(batch_files)} elements."
            
            micro_resp = client.models.generate_content(
                model=VISION_MODEL,
                contents=batch_files + [micro_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=ActiveSpeakerBatchResponse)
            )
            finish = micro_resp.candidates[0].finish_reason
            finish_name = getattr(finish, "name", str(finish))
            if finish_name != "STOP":
                raise RuntimeError(
                    f"Micro-vision batch {batch_idx + 1}/{total_micro_batches} "
                    f"did not finish cleanly: finish_reason={finish_name}."
                )

            names_array = json.loads(micro_resp.text).get("speaker_names", [])
            if len(names_array) != len(batch_files):
                raise RuntimeError(
                    f"Micro-vision batch {batch_idx + 1}/{total_micro_batches} returned "
                    f"{len(names_array)} names for {len(batch_files)} frames (truncation suspected)."
                )

            for i, name in enumerate(names_array):
                t_tag = batch_frames_info[i]["tag"]
                if name:
                    tag_votes[t_tag].append(name)

    resolved_map = {} # tag -> child_id
    
    for tag in teacher_tags:
        c.execute("SELECT child_id FROM child_registry WHERE video_id = ? AND role = 'teacher' LIMIT 1", (metadata_id,))
        res = c.fetchone()
        if res:
            t_id = res[0]
        else:
            c.execute('''
                INSERT INTO child_registry (video_id, name, name_source, role, appearance, first_seen_sec)
                VALUES (?, ?, 'audio_unresolved', 'teacher', 'Audio only teacher', 0.0)
            ''', (metadata_id, tag))
            t_id = c.lastrowid
            child_map[tag] = t_id
            known_names.append(tag)
        resolved_map[tag] = t_id
        c.execute('''
            INSERT INTO speaker_identity_map (video_id, speaker_tag, child_id, vote_result)
            VALUES (?, ?, ?, 'teacher_bypass')
        ''', (metadata_id, tag, t_id))
    
    for tag in targeted_tags:
        votes = tag_votes[tag]
        # Resolve each vote to a child_id first so aliases of the same kid merge in the count.
        valid_ids = []
        for n in votes:
            matched = fuzzy_match(n, known_names)
            if matched:
                valid_ids.append(child_map[matched])

        child_id = None
        vote_str = "unresolved"

        if valid_ids:
            from collections import Counter
            id_counts = Counter(valid_ids)
            best_id, count = id_counts.most_common(1)[0]
            if count >= 2 or (len(votes) < 3 and count == len(votes) and count > 0):
                child_id = best_id
                vote_str = f"{count}/{len(votes)}"
        
        if not child_id:
            c.execute('''
                INSERT INTO child_registry (video_id, name, name_source, role, appearance, first_seen_sec)
                VALUES (?, ?, 'audio_unresolved', 'student', 'Audio only participant', 0.0)
            ''', (metadata_id, tag))
            child_id = c.lastrowid
            child_map[tag] = child_id
            known_names.append(tag)
            
        resolved_map[tag] = child_id
        c.execute('''
            INSERT INTO speaker_identity_map (video_id, speaker_tag, child_id, vote_result)
            VALUES (?, ?, ?, ?)
        ''', (metadata_id, tag, child_id, vote_str))

    from collections import Counter
    child_id_counts = Counter(resolved_map.values())
    for cid, cnt in child_id_counts.items():
        if cnt > 1:
            c.execute("UPDATE speaker_identity_map SET diarization_split = 1 WHERE video_id = ? AND child_id = ?", (metadata_id, cid))

    assert all(resolved_map.get(t) is not None for t in unique_tags), "Not all tags resolved to a child_id"

    for row in script_data.get("script", []):
        tag = row["speaker_tag"]
        spk_id = resolved_map.get(tag)
        
        matched_tgt = fuzzy_match(row["targeted_child_name"], known_names)
        tgt_id = child_map[matched_tgt] if matched_tgt else None
        
        duration = max(0.1, row["end_sec"] - row["start_sec"])
        word_count = len(row["utterance"].split())
        speaking_rate = round(word_count / duration, 2)
        
        c.execute('''
            INSERT INTO script_storage (video_id, start_sec, end_sec, speaker_norm, speaker_tag, speaker_child_id, utterance, is_question, speaking_rate, emotion_audio, confidence_audio, emotion_text, confidence_text, keywords, teacher_behavior_flag, targeted_child_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (metadata_id, row["start_sec"], row["end_sec"], row["speaker_norm"], tag, spk_id, row["utterance"], row["is_question"], speaking_rate, row["emotion_audio"], row["confidence_audio"], row["emotion_text"], row["confidence_text"], json.dumps(row["keywords"]), row["teacher_behavior_flag"], tgt_id))
        
    c.execute('''
        UPDATE child_registry
        SET total_frames_present = (
            SELECT COUNT(*) FROM child_frame_emotion WHERE child_frame_emotion.child_id = child_registry.child_id
        )
    ''')
        
    c.execute('UPDATE video_metadata SET frames_analyzed = ?, status = ? WHERE video_id = ?', (len(all_vision_frames), 'success', metadata_id))

    conn.commit()
    conn.close()

    if progress_callback: progress_callback("Done!", 100)
    return {"status": "success", "video_id": video_id}
