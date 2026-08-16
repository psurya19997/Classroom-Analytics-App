# Speaker Identity Resolution Plan

We need to implement a robust 3-vote majority algorithm to reliably map audio voices to on-screen student names. This plan addresses the temporal mismatch (by extracting targeted frames at the exact midpoint of utterances) and seamlessly integrates unresolved speakers into the dashboard.

## Handling Unresolved Speakers in the Dashboard

To fulfill your request that unresolved speakers (like "Speaker_3") appear in the Student Scorecard and the Engagement Heatmap, we will insert them into the `child_registry` as a "pseudo-student". 

By doing this, the Streamlit dashboard doesn't need to be rewritten for SQL joins. The queries will naturally pick up "Speaker_3".

**Crucial Constraints & UI Fixes:**
- **Role:** Pseudo-students MUST be inserted with `role='student'` (along with `name_source='audio_unresolved'`), otherwise dashboard queries (`WHERE r.role = 'student'`) will silently filter them out.
- **Attendance Display:** Because pseudo-students are never visible, their `attendance_pct` calculates to 0%. To prevent this from looking like a data bug to teachers, the Streamlit UI (`app.py`) will be updated to display "— (audio only)" instead of "0% on-camera" for these specific records.
- **Targeted Child Name:** The `targeted_child_name` field will NOT be removed from the `AudioResponse` schema. Gemini can still hear names being addressed (e.g., "Good job, Manisha!"), and this is strictly required for the `praised` and `shouted` metrics. This will continue to resolve against `child_registry`.

## Proposed Changes

---

### Database Schema Updates
We will idempotently update the schema to support the identity map and audio tags.

#### [MODIFY] db.py
- Update `_migrate_schema` to create the new `speaker_identity_map` table:
  ```sql
  CREATE TABLE IF NOT EXISTS speaker_identity_map (
      video_id INTEGER,
      speaker_tag TEXT,
      child_id INTEGER,
      vote_result TEXT,
      diarization_split BOOLEAN DEFAULT FALSE,
      PRIMARY KEY (video_id, speaker_tag)
  )
  ```
- Update `_migrate_schema` to idempotently add the `speaker_tag` column to the `script_storage` table.

---

### Pipeline Orchestration
We will introduce the Targeted Identity Resolution step immediately following the audio analysis.

#### [MODIFY] pipeline.py
- **Audio Schema Update**: Update `ScriptRow` to replace `speaker_name` with a mandatory `speaker_tag` field. **Keep `targeted_child_name` intact.** Adjust the prompt to enforce strict diarization tagging (e.g., `Speaker_1`, `Teacher`).
- **Targeted Frame Extraction**:
  - After the audio script is generated, parse the unique `speaker_tags`.
  - For each tag, sample up to 3 utterances (preferring those >2 seconds).
  - Calculate the exact midpoint of each utterance (`start_sec + duration/2`).
  - Execute a fast `ffmpeg` subprocess to extract just these specific frames.
- **Micro-Vision Call**:
  - Create a lightweight `ActiveSpeakerResponse` Pydantic schema.
  - Batch upload the targeted frames to Gemini Vision with a prompt asking *only* for the name of the active, highlighted speaker.
- **Voting & Resolution Logic**:
  - Tally the responses for each `speaker_tag`.
  - If a 2/3 majority is reached, map it to the known `child_id`.
  - **If unresolved (tie or nulls)**: Insert a new row into `child_registry` with `name = speaker_tag`, `name_source = 'audio_unresolved'`, and `role = 'student'`. Map it to this new `child_id`.
  - Save the vote metadata to `speaker_identity_map`.
  - Write the final `script_storage` rows, ensuring every utterance has both its raw `speaker_tag` and a valid `speaker_child_id`.

---

### Dashboard Adjustments
#### [MODIFY] metrics.py & app.py
- In `metrics.py` -> `get_child_stats`, ensure `name_source` is selected from `child_registry` and passed through to the DataFrame.
- In `app.py` -> Scorecard rendering, if `name_source == 'audio_unresolved'`, override the attendance badge to show `"Audio only"` instead of `0%`.
