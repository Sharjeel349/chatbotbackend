from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
import uuid
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import pyttsx3
import requests
from pydub import AudioSegment

from FYP2.services.roman_urdu_service import (
    clean_and_normalize_roman_urdu,
    contains_urdu_script,
    transliterate_urdu_to_roman,
)
from FYP2.services.advisor_graph import AdvisorState, advisor_graph


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

AUDIO_DIR = BASE_DIR / "static" / "audio"
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import torch

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "large-v3-turbo" if torch.cuda.is_available() else "base",
)
STT_BACKEND = os.getenv("STT_BACKEND", "faster-whisper")
MAX_HISTORY_MESSAGES = 3

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL_DEFAULT = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_MODEL_ENGLISH = os.getenv("OLLAMA_MODEL_ENGLISH", OLLAMA_MODEL_DEFAULT)
OLLAMA_MODEL_ROMAN_URDU = os.getenv("OLLAMA_MODEL_ROMAN_URDU", OLLAMA_MODEL_DEFAULT)


@lru_cache(maxsize=1)
def get_available_ollama_models() -> set[str]:
    try:
        res = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if res.status_code == 200:
            models = res.json().get("models", [])
            names = set()
            for m in models:
                n = m.get("name", "")
                names.add(n)
                names.add(n.split(":")[0])
            return names
    except Exception:
        pass
    return set()


def resolve_ollama_model(preferred: str, fallback: str = "qwen2.5:7b-instruct") -> str:
    available = get_available_ollama_models()
    if not available:
        return preferred
    if preferred in available or preferred.split(":")[0] in available:
        return preferred
    return fallback


_stt_lock = threading.Lock()

URDU_LANGUAGE_CODES = {"ur", "hi", "pa"}


def _event(event_type: str, **payload: Any) -> dict[str, Any]:
    return {"type": event_type, **payload}


def encode_stream_event(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode("utf-8")


ROMAN_URDU_KEYWORDS = {
    "mera", "meri", "mere", "apna", "apni", "apne", "aap", "ka", "ki", "ke",
    "ko", "se", "mein", "par", "hai", "hain", "ho", "hu", "hoon", "tha", "thi",
    "the", "kya", "kia", "kaisa", "kaisi", "kaise", "kitna", "kitni", "kitne",
    "batao", "bataen", "bataey", "batai", "mujhe", "sunao", "karna", "karne",
    "bhi", "nahi", "nahin", "sir", "madam", "pas", "pass", "fail"
}

def is_hallucination(text: str) -> bool:
    """Check if Whisper hallucinated repetitive words (e.g. 'ayk ayk ayk...' or 'ایک ایک ایک...')."""
    words = text.strip().split()
    if len(words) >= 4:
        from collections import Counter

        top_word, count = Counter(w.casefold() for w in words).most_common(1)[0]
        if count / len(words) > 0.5:
            return True
        if re.search(r"\b(\w+)(?:\s+\1){3,}\b", text, re.IGNORECASE):
            return True
    return False


def convert_to_roman_urdu_offline(text: str) -> str:
    """Robust conversion of Urdu script to natural Roman Urdu using roman_urdu_service."""
    return transliterate_urdu_to_roman(text)


def response_language(language_code: str, transcript: str) -> str:
    if language_code.lower() in URDU_LANGUAGE_CODES or contains_urdu_script(transcript):
        return "roman_urdu"
    words = set(re.findall(r"[a-z]+", transcript.casefold()))
    if len(words & ROMAN_URDU_KEYWORDS) >= 1:
        return "roman_urdu"
    return "english"


@lru_cache(maxsize=1)
def get_stt_model():
    if STT_BACKEND == "faster-whisper":
        try:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model '%s' on device '%s' with '%s'",
                WHISPER_MODEL,
                DEVICE,
                COMPUTE_TYPE,
            )
            model = WhisperModel(
                WHISPER_MODEL,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=min(4, os.cpu_count() or 2) if DEVICE == "cpu" else 1,
                num_workers=1,
                download_root=str(MODEL_CACHE_DIR),
            )
            return "faster-whisper", model
        except Exception:
            logger.exception("faster-whisper failed; using OpenAI Whisper fallback")

    import whisper

    logger.info("Loading OpenAI Whisper model '%s'", WHISPER_MODEL)
    return "openai-whisper", whisper.load_model(WHISPER_MODEL)



def warm_up_models() -> None:
    """Load speech recognition during API startup instead of the first request."""
    get_stt_model()


def _transcribe(audio_path: Path) -> tuple[str, str]:
    with _stt_lock:
        backend, model = get_stt_model()
        initial_prompt = (
            "میرا سی جی پی اے کتنا ہے، رزلٹ، کورسز، فیس، سمسٹر، ایڈوائزر، امتحان، داخلہ، "
            "BIIT, Arid, CGPA, GPA, semester, course, prerequisite, drop, freeze, "
            "unfreeze, withdrawal, probation, cease, attendance, fee, challan, installment, exam, mid-term, final-term"
        )
        if backend == "faster-whisper":
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=1,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
                initial_prompt=initial_prompt,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            detected_lang = info.language or "en"
            if is_hallucination(text):
                logger.warning("Detected Whisper repetition hallucination in text: %s", text)
                text = ""
            return text, detected_lang

        result = model.transcribe(
            str(audio_path),
            fp16=False,
            temperature=0,
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            verbose=False,
        )
        text = result.get("text", "").strip()
        detected_lang = result.get("language", "en")
        if is_hallucination(text):
            logger.warning("Detected Whisper repetition hallucination in text: %s", text)
            text = ""
        return text, detected_lang


def _safe_rollback(db_conn) -> None:
    try:
        db_conn.rollback()
    except Exception:
        logger.exception("Database rollback failed")


def get_latest_emotion(db_conn) -> str:
    cur = db_conn.cursor()
    try:
        cur.execute(
            "SELECT mind_State FROM EEG_State ORDER BY eeg_state_id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row.get("mind_state", "Neutral") if row else "Neutral"
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not fetch the latest EEG state")
        return "Neutral"
    finally:
        cur.close()


def get_student_profile(student_id: str, db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT u.name, s.sid AS registration_no, s.cgpa, cs.semester,
                   cs.section_code, tu.name AS advisor_name,
                   t.designation AS advisor_designation
            FROM Student s
            JOIN Users u ON s.user_id = u.user_id
            JOIN Class_Section cs ON s.section_id = cs.section_id
            LEFT JOIN Teacher t ON cs.advisor_id = t.teacher_id
            LEFT JOIN Users tu ON t.user_id = tu.user_id
            WHERE s.sid = %s;
            """,
            (student_id,),
        )
        return cur.fetchone()
    finally:
        cur.close()


def get_failed_courses(student_id: str, db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT DISTINCT ON (c.course_id)
                   c.course_id, c.title, c.skill_category, co.session,
                   u_teacher.name AS teacher_name, e.grade
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            LEFT JOIN Teacher t ON co.teacher_id = t.teacher_id
            LEFT JOIN Users u_teacher ON t.user_id = u_teacher.user_id
            WHERE (e.grade = 'F' OR e.grade = 'W') AND e.student_id = %s
              AND c.course_id NOT IN (
                  SELECT co2.course_id
                  FROM Enrolment e2
                  JOIN Course_Offering co2 ON e2.offering_id = co2.offering_id
                  WHERE e2.student_id = %s
                    AND e2.grade IS NOT NULL
                    AND e2.grade NOT IN ('F', 'W', 'I')
              )
            ORDER BY c.course_id, co.session DESC;
            """,
            (student_id, student_id),
        )
        return cur.fetchall()
    finally:
        cur.close()


def get_transcript(student_id: str, db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT co.session, c.course_id, c.title, c.skill_category,
                   e.grade, c.credit_hours
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            WHERE e.student_id = %s AND e.grade IS NOT NULL
            ORDER BY co.session DESC, c.title ASC
            LIMIT 8;
            """,
            (student_id,),
        )
        return cur.fetchall()
    finally:
        cur.close()


def get_freeze_history(student_id: str, db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT COUNT(*) FILTER (WHERE status = 'Frozen') AS total_frozen,
                   COUNT(*) AS total_semesters_registered
            FROM Student_Session
            WHERE sid = %s;
            """,
            (student_id,),
        )
        return cur.fetchone()
    finally:
        cur.close()


def get_current_enrollments(student_id: str, db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT c.course_id, c.title, c.skill_category,
                   u.name AS teacher_name
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            JOIN Teacher t ON co.teacher_id = t.teacher_id
            JOIN Users u ON t.user_id = u.user_id
            WHERE e.student_id = %s AND e.grade IS NULL;
            """,
            (student_id,),
        )
        return cur.fetchall()
    finally:
        cur.close()


def get_course_prerequisites(db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT c.course_id, c.title AS course_title,
                   p.course_id AS prerequisite_id, p.title AS prerequisite_title,
                   cp.sos, cp.discipline
            FROM Course_Prerequisite cp
            JOIN Course c ON cp.course_id = c.course_id AND cp.sos = c.sos
            JOIN Course p ON cp.prerequisite_id = p.course_id AND cp.sos = p.sos
            ORDER BY cp.sos ASC, c.title ASC;
            """
        )
        return cur.fetchall()
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not fetch course prerequisites")
        return []
    finally:
        cur.close()


def get_teacher_strictness_stats(db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT u.name AS teacher_name,
                   COUNT(e.grade) AS total_graded,
                   COUNT(*) FILTER (WHERE e.grade = 'A') AS total_a,
                   COUNT(*) FILTER (WHERE e.grade = 'F') AS total_f
            FROM Teacher t
            JOIN Users u ON t.user_id = u.user_id
            JOIN Course_Offering co ON t.teacher_id = co.teacher_id
            JOIN Enrolment e ON co.offering_id = e.offering_id
            WHERE e.grade IS NOT NULL
            GROUP BY u.name
            ORDER BY total_f DESC
            LIMIT 30;
            """
        )
        return cur.fetchall()
    finally:
        cur.close()


def _row_val(row, key: str, idx: int = 0, default=None):
    if not row:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[idx]
    except (IndexError, TypeError, KeyError):
        return default


def get_completed_credit_hours(student_id: str, db_conn):
    """Calculates total credit hours earned and degree completion progress based on the student's Scheme of Study."""
    cur = db_conn.cursor()
    try:
        cur.execute("SELECT sos FROM Student WHERE sid = %s;", (student_id,))
        sos_row = cur.fetchone()
        sos = _row_val(sos_row, "sos", 0)

        cur.execute(
            """
            SELECT COALESCE(SUM(c.credit_hours), 0) AS completed_credits,
                   COUNT(DISTINCT c.course_id) AS total_passed_courses
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            WHERE e.student_id = %s 
              AND e.grade IS NOT NULL 
              AND e.grade <> 'F';
            """,
            (student_id,),
        )
        row = cur.fetchone() or {"completed_credits": 0, "total_passed_courses": 0}
        completed_credits = int(_row_val(row, "completed_credits", 0, 0))
        total_passed_courses = int(_row_val(row, "total_passed_courses", 1, 0))

        total_sos_credits = 0
        total_sos_courses = 0
        if sos:
            cur.execute(
                """
                SELECT COALESCE(SUM(credit_hours), 0) AS total_credits,
                       COUNT(*) AS total_courses
                FROM Course
                WHERE sos = %s;
                """,
                (sos,),
            )
            sos_stats = cur.fetchone()
            if sos_stats:
                total_sos_credits = int(_row_val(sos_stats, "total_credits", 0, 0))
                total_sos_courses = int(_row_val(sos_stats, "total_courses", 1, 0))

        remaining_credits = max(0, total_sos_credits - completed_credits) if total_sos_credits > 0 else 0
        pct = round((completed_credits / total_sos_credits * 100), 1) if total_sos_credits > 0 else 0.0

        return {
            "completed_credits": completed_credits,
            "total_passed_courses": total_passed_courses,
            "student_sos": sos,
            "total_degree_credits": total_sos_credits,
            "total_degree_courses": total_sos_courses,
            "remaining_credits_to_graduate": remaining_credits,
            "degree_progress": f"{pct}%",
        }
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not fetch completed credit hours")
        return {"completed_credits": 0, "total_passed_courses": 0}
    finally:
        cur.close()


def get_improvable_courses(student_id: str, db_conn):
    """Fetches courses with D or F grades that the student is eligible to repeat."""
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT c.course_id, c.title AS course_title, c.credit_hours, e.grade, e.attempt, co.session
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            WHERE e.student_id = %s 
              AND e.grade IN ('D', 'F')
            ORDER BY e.grade DESC, c.title ASC;
            """,
            (student_id,),
        )
        return cur.fetchall()
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not fetch improvable courses")
        return []
    finally:
        cur.close()


def get_current_teachers_and_courses(student_id: str, db_conn):
    """Fetches the student's current semester courses along with assigned teacher names and designations."""
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT c.course_id, c.title AS course_title, c.credit_hours,
                   u.name AS teacher_name, t.designation, t.department
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            JOIN Teacher t ON co.teacher_id = t.teacher_id
            JOIN Users u ON t.user_id = u.user_id
            WHERE e.student_id = %s AND e.grade IS NULL
            ORDER BY c.title ASC;
            """,
            (student_id,),
        )
        return cur.fetchall()
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not fetch current teachers and courses")
        return []
    finally:
        cur.close()


def check_student_prerequisite_status(student_id: str, db_conn):
    """Checks prerequisites specifically for the student's Scheme of Study and flags pass/fail/eligible status."""
    cur = db_conn.cursor()
    try:
        cur.execute("SELECT sos FROM Student WHERE sid = %s;", (student_id,))
        srow = cur.fetchone()
        sos = _row_val(srow, "sos", 0)

        cur.execute(
            """
            SELECT 
                target_c.course_id AS target_course_id,
                target_c.title AS target_course_title,
                prereq_c.course_id AS prereq_course_id,
                prereq_c.title AS prereq_course_title,
                cp.sos,
                cp.discipline,
                COALESCE((
                    SELECT e.grade 
                    FROM Enrolment e 
                    JOIN Course_Offering co ON e.offering_id = co.offering_id
                    WHERE co.course_id = prereq_c.course_id AND e.student_id = %s
                    ORDER BY CASE WHEN e.grade <> 'F' THEN 1 ELSE 2 END, e.enrolment_id DESC
                    LIMIT 1
                ), 'Not Taken') AS student_prereq_grade,
                CASE 
                    WHEN EXISTS (
                        SELECT 1 
                        FROM Enrolment e 
                        JOIN Course_Offering co ON e.offering_id = co.offering_id
                        WHERE co.course_id = prereq_c.course_id 
                          AND e.student_id = %s
                          AND e.grade IS NOT NULL 
                          AND e.grade <> 'F'
                    ) THEN 'Eligible'
                    ELSE 'Not Eligible'
                END AS eligibility_status
            FROM Course_Prerequisite cp
            JOIN Course target_c ON cp.course_id = target_c.course_id AND cp.sos = target_c.sos
            JOIN Course prereq_c ON cp.prerequisite_id = prereq_c.course_id AND cp.sos = prereq_c.sos
            WHERE (%s IS NULL OR cp.sos = %s)
            ORDER BY target_c.title ASC;
            """,
            (student_id, student_id, sos, sos),
        )
        return cur.fetchall()
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not check student prerequisite status")
        return []
    finally:
        cur.close()


def get_student_session_timeline(student_id: str, db_conn):
    """Fetches the student's complete session history (Active, Frozen, Completed)."""
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT session, status
            FROM Student_Session
            WHERE sid = %s
            ORDER BY session_id ASC;
            """,
            (student_id,),
        )
        return cur.fetchall()
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not fetch student session timeline")
        return []
    finally:
        cur.close()


def get_current_semester_load(student_id: str, db_conn):
    """Calculates total enrolled credit hours and course count in the ongoing semester."""
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT COALESCE(SUM(c.credit_hours), 0) AS total_enrolled_credits,
                   COUNT(DISTINCT c.course_id) AS total_enrolled_courses
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            WHERE e.student_id = %s AND e.grade IS NULL;
            """,
            (student_id,),
        )
        row = cur.fetchone()
        return row if row else {"total_enrolled_credits": 0, "total_enrolled_courses": 0}
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not fetch current semester load")
        return {"total_enrolled_credits": 0, "total_enrolled_courses": 0}
    finally:
        cur.close()


def get_student_attempt_warnings(student_id: str, db_conn) -> list[dict[str, Any]]:
    """Tracks student repeat course attempts from enrolment.attempt and flags academic repeat warnings."""
    cur = db_conn.cursor()
    try:
        query = """
            SELECT 
                co.course_id,
                c.title AS course_title,
                MAX(e.attempt) AS highest_attempt,
                COUNT(e.enrolment_id) AS total_enrolments,
                ARRAY_AGG(DISTINCT e.grade) FILTER (WHERE e.grade IS NOT NULL) AS recorded_grades,
                BOOL_OR(e.grade IS NOT NULL AND e.grade <> 'F') AS has_passed_course
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            WHERE e.student_id = %s
            GROUP BY co.course_id, c.title
            HAVING MAX(e.attempt) >= 2 OR COUNT(e.enrolment_id) >= 2
            ORDER BY MAX(e.attempt) DESC, c.title ASC;
        """
        cur.execute(query, (student_id,))
        rows = cur.fetchall()
        warnings = []
        for r in rows:
            attempt_num = int(_row_val(r, "highest_attempt", 2, 1))
            passed = bool(_row_val(r, "has_passed_course", 5, False))
            status = "Passed on Repeat" if passed else "Action Required: Currently Unresolved"
            urgency = "High Warning (3+ attempts)" if attempt_num >= 3 else "Repeat Warning (Attempt 2)"
            warnings.append({
                "course_id": _row_val(r, "course_id", 0),
                "course_title": _row_val(r, "course_title", 1),
                "attempts": attempt_num,
                "grades": _row_val(r, "recorded_grades", 4),
                "status": status,
                "warning_level": urgency,
            })
        return warnings
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not fetch student attempt warnings")
        return []
    finally:
        cur.close()


def get_next_semester_course_recommendations(student_id: str, db_conn) -> dict[str, Any]:
    """Recommends next semester subjects based on student's Scheme of Study, passed courses, and prerequisites."""
    cur = db_conn.cursor()
    try:
        # 1. Fetch student SOS, semester, and discipline
        cur.execute(
            """
            SELECT s.sid, s.sos, cs.semester, cs.discipline
            FROM Student s
            JOIN Class_Section cs ON s.section_id = cs.section_id
            WHERE s.sid = %s;
            """,
            (student_id,),
        )
        st_row = cur.fetchone()
        if not st_row:
            return {}
        sos = _row_val(st_row, "sos", 1)
        current_semester = _row_val(st_row, "semester", 2, "")
        discipline = _row_val(st_row, "discipline", 3, "")

        if not sos:
            return {"error": "Student Scheme of Study (SOS) not assigned."}

        # 2. Passed courses
        cur.execute(
            """
            SELECT DISTINCT co.course_id
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            WHERE e.student_id = %s AND e.grade IS NOT NULL AND e.grade <> 'F';
            """,
            (student_id,),
        )
        passed_courses = set(_row_val(r, "course_id", 0) for r in cur.fetchall())

        # 3. Currently registered courses (ongoing semester)
        cur.execute(
            """
            SELECT DISTINCT co.course_id
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            WHERE e.student_id = %s AND e.grade IS NULL;
            """,
            (student_id,),
        )
        currently_enrolled = set(_row_val(r, "course_id", 0) for r in cur.fetchall())

        # 4. Failed courses not yet cleared
        cur.execute(
            """
            SELECT DISTINCT co.course_id, c.title
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            WHERE e.student_id = %s AND e.grade = 'F'
              AND co.course_id NOT IN (
                  SELECT co2.course_id 
                  FROM Enrolment e2 
                  JOIN Course_Offering co2 ON e2.offering_id = co2.offering_id
                  WHERE e2.student_id = %s AND e2.grade IS NOT NULL AND e2.grade <> 'F'
              );
            """,
            (student_id, student_id),
        )
        unresolved_fails = [
            {"course_id": _row_val(r, "course_id", 0),
             "title": _row_val(r, "title", 1)}
            for r in cur.fetchall()
        ]

        # 5. All courses prescribed in student's SOS
        cur.execute(
            """
            SELECT course_id, title, credit_hours, course_type, skill_category
            FROM Course
            WHERE sos = %s
            ORDER BY course_id ASC;
            """,
            (sos,),
        )
        sos_courses = cur.fetchall()

        # 6. Prerequisites mapped for student's SOS
        cur.execute(
            """
            SELECT course_id, prerequisite_id
            FROM Course_Prerequisite
            WHERE sos = %s;
            """,
            (sos,),
        )
        prereq_map: dict[str, list[str]] = {}
        for r in cur.fetchall():
            cid = _row_val(r, "course_id", 0)
            pid = _row_val(r, "prerequisite_id", 1)
            prereq_map.setdefault(cid, []).append(pid)

        # 7. Categorize remaining courses
        remaining_courses = [
            c for c in sos_courses 
            if _row_val(c, "course_id", 0) not in passed_courses 
            and _row_val(c, "course_id", 0) not in currently_enrolled
        ]

        eligible_next = []
        locked_courses = []

        for c in remaining_courses:
            cid = _row_val(c, "course_id", 0)
            title = _row_val(c, "title", 1)
            cr = int(_row_val(c, "credit_hours", 2, 3))
            ctype = _row_val(c, "course_type", 3, "REQUIRED")

            reqs = prereq_map.get(cid, [])
            missing_prereqs = [r for r in reqs if r not in passed_courses]

            if not missing_prereqs:
                eligible_next.append({
                    "course_id": cid,
                    "title": title,
                    "credit_hours": cr,
                    "course_type": ctype,
                })
            else:
                locked_courses.append({
                    "course_id": cid,
                    "title": title,
                    "missing_prerequisites": missing_prereqs,
                })

        return {
            "student_sos": sos,
            "current_semester": current_semester,
            "discipline": discipline,
            "total_passed_courses": len(passed_courses),
            "unresolved_failed_courses_to_repeat": unresolved_fails,
            "eligible_courses_to_study_next": eligible_next[:10],
            "locked_courses_needing_prerequisites": locked_courses[:10],
        }
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not compute next semester course recommendations")
        return {}
    finally:
        cur.close()


def get_recent_chat_history(session_id: int, db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT sender, message_text
            FROM Chat_Message
            WHERE session_id = %s
              AND message_text NOT LIKE '%%AI model abhi available nahi hai%%'
              AND message_text NOT LIKE '%%The AI model is unavailable%%'
            ORDER BY timestamp DESC
            LIMIT %s;
            """,
            (session_id, MAX_HISTORY_MESSAGES),
        )
        return list(reversed(cur.fetchall()))
    finally:
        cur.close()


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _mentions(text: str, terms: set[str]) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in terms)


def build_student_context(
    query: str, session_id: int, student_id: str, db_conn
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "profile": get_student_profile(student_id, db_conn),
        "recent_chat": get_recent_chat_history(session_id, db_conn),
    }

    if _mentions(
        query,
        {
            "course",
            "subject",
            "programming",
            "coding",
            "enroll",
            "overload",
            "performance",
            "performing",
            "kaisa",
            "کورس",
            "سبجیکٹ",
            "پروگرامنگ",
            "drop",
            "dropped",
            "withdraw",
            "withdrawal",
            "fail",
            "failed",
            "ڈراپ",
            "فیل",
        },
    ):
        context["current_enrollments"] = get_current_enrollments(student_id, db_conn)
        failed_courses = get_failed_courses(student_id, db_conn)
        context["failed_courses"] = failed_courses
        context["dropped_courses"] = failed_courses
        context["transcript"] = get_transcript(student_id, db_conn)
        context["current_semester_load"] = get_current_semester_load(student_id, db_conn)

    # 1. Degree Progress & Completed Credit Hours
    if _mentions(
        query,
        {
            "credit", "credits", "credit hour", "credit hours", "completed", "passed",
            "progress", "degree", "fyp", "graduation", "pass kitne", "kitne pass",
            "کرڈٹ", "پاس", "ڈگری"
        },
    ):
        context["completed_credits_summary"] = get_completed_credit_hours(student_id, db_conn)

    # 2. Improvable / Repeat Courses (D or F grades)
    if _mentions(
        query,
        {
            "repeat", "improve", "improvement", "d grade", "f grade", "cgpa barhana",
            "dobara", "ریپیٹ", "امپروو", "دوبارہ"
        },
    ):
        context["improvable_courses"] = get_improvable_courses(student_id, db_conn)

    # 3. Semester Load & Overload Check
    if _mentions(
        query,
        {
            "workload", "load", "overload", "limit", "credits this semester", "kitne subjects",
            "load zyada", "work load", "ورک لوڈ"
        },
    ):
        context["current_semester_load"] = get_current_semester_load(student_id, db_conn)

    # 4. Teachers & Course Faculty
    if _mentions(
        query,
        {
            "teacher", "teachers", "instructor", "faculty", "sir", "madam", "parha raha",
            "parha rahi", "kon parha", "kaun parha", "استاد", "ٹیچر"
        },
    ):
        context["current_teachers"] = get_current_teachers_and_courses(student_id, db_conn)
        context["teacher_statistics"] = get_teacher_strictness_stats(db_conn)

    # 5. Session Timeline & Freeze History
    if _mentions(
        query,
        {
            "freeze", "semester", "gap", "history", "timeline", "session",
            "فریز", "سمسٹر", "for ease", "forease", "گیپ"
        },
    ):
        context["freeze_history"] = get_freeze_history(student_id, db_conn)
        context["session_timeline"] = get_student_session_timeline(student_id, db_conn)

    # 6. Prerequisites & Eligibility Check
    if _mentions(
        query,
        {
            "prerequisite", "prerequisites", "prereq", "prereqs", "pre-req", "pehle",
            "shart", "pre requisite", "eligible", "eligibility", "le sakta", "پہلے", "شرط"
        },
    ):
        context["student_prerequisite_eligibility"] = check_student_prerequisite_status(student_id, db_conn)
        context["course_prerequisites"] = get_course_prerequisites(db_conn)
        context["transcript"] = get_transcript(student_id, db_conn)
        failed_courses = get_failed_courses(student_id, db_conn)
        context["failed_courses"] = failed_courses

    # 7. Transcript & Grades
    if _mentions(
        query,
        {
            "grade", "transcript", "result", "marks", "performance", "performing",
            "گریڈ", "نتیجہ", "مارکس"
        },
    ):
        context["transcript"] = get_transcript(student_id, db_conn)

    # 8. Next Semester Courses & Roadmap (Based on Student SOS and Prerequisites)
    if _mentions(
        query,
        {
            "next", "study next", "future", "remaining", "roadmap", "what to take",
            "take next", "subjects next", "courses next", "kon se parhne", "agay",
            "aglay", "parhna", "parhoon", "bache huay", "next semester", "konsay",
            "subjects", "courses", "recommend", "suggestion", "advice",
            "آگے", "اگلے", "پڑھنا"
        },
    ):
        context["next_semester_course_recommendations"] = get_next_semester_course_recommendations(student_id, db_conn)
        if "completed_credits_summary" not in context:
            context["completed_credits_summary"] = get_completed_credit_hours(student_id, db_conn)

    # 9. Course Attempt & Repeat Warning Agent
    if _mentions(
        query,
        {
            "attempt", "attempts", "repeat", "warning", "fail", "failed", "danger",
            "risk", "probation", "expel", "max attempts", "dobara", "ریپیٹ", "وارننگ"
        },
    ):
        context["attempt_warnings"] = get_student_attempt_warnings(student_id, db_conn)

    return _plain(context)


@lru_cache(maxsize=1)
def _knowledge_sections() -> list[str]:
    sections: list[str] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        sections.extend(
            section.strip()
            for section in re.split(r"\n(?=## )", text)
            if section.strip()
        )
    return sections


_RETRIEVAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "have", "has", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or",
    "the", "to", "what", "when", "where", "which", "who", "why", "will", "with", "you",
    "kya", "hai", "hain", "ka", "ki", "ke", "ko", "se", "me", "mein", "par",
}


from FYP2.services.knowledge_service import retrieve_policy_knowledge


def retrieve_knowledge(query: str) -> str:
    """Retrieve policy knowledge using 100% offline Hybrid Search (ChromaDB + Lexical fallback)."""
    return retrieve_policy_knowledge(query)


SEMESTER_MIN_CGPA = {
    2: 1.00,
    3: 1.25,
    4: 1.50,
    5: 1.75,
    6: 2.00,
    7: 2.25,
    8: 2.50,
}

SEMESTER_WORD_MAP = {
    "1": 1, "1st": 1, "first": 1, "pehla": 1,
    "2": 2, "2nd": 2, "second": 2, "doosra": 2, "dusra": 2,
    "3": 3, "3rd": 3, "third": 3, "teesra": 3,
    "4": 4, "4th": 4, "fourth": 4, "chotha": 4,
    "5": 5, "5th": 5, "fifth": 5, "panchwan": 5,
    "6": 6, "6th": 6, "sixth": 6, "chattha": 6, "chatha": 6,
    "7": 7, "7th": 7, "seventh": 7, "saatwan": 7,
    "8": 8, "8th": 8, "eighth": 8, "aathwan": 8,
}

ACADEMIC_TERMS = {
    "course", "courses", "subject", "subjects", "semester", "semesters",
    "gpa", "cgpa", "exam", "exams", "examination", "test", "grade", "grades",
    "mark", "marks", "teacher", "teachers", "advisor", "advisors", "fee", "fees",
    "attendance", "freeze", "drop", "enroll", "enrollment", "admission",
    "degree", "transcript", "credit", "credits", "lecture", "class", "classes",
    "paper", "fail", "pass", "promote", "promotion", "repeat", "cease",
    "kors", "kursa", "hazri", "imtihan", "parhai", "dakhla", "qist", "challan",
}

IRRELEVANT_PATTERNS = [
    r"\bweather\b", r"\bmosam\b", r"\bmausam\b", r"\brain\b", r"\bbarish\b",
    r"\btemperature\b", r"\bforecast\b", r"\bsunny\b", r"\bcloudy\b",
    r"\bcricket\b", r"\bfootball\b", r"\bmatch\b", r"\bmovie\b", r"\bfilm\b",
    r"\bsong\b", r"\bgana\b", r"\bjoke\b", r"\blatifa\b", r"\brecipe\b",
    r"\bkhana\b", r"\bpolitics\b", r"\bsiasat\b", r"\bprime minister\b",
    r"\bpresident\b", r"\bnews\b", r"\bkhabar\b", r"\bkhabrain\b",
]


def is_irrelevant_query(text: str) -> bool:
    normalized = text.casefold().strip()
    words = set(re.findall(r"[a-z]+", normalized))
    # If the question contains academic terminology, do not flag as irrelevant
    if words & ACADEMIC_TERMS:
        return False
    return any(re.search(pat, normalized) for pat in IRRELEVANT_PATTERNS)





def build_prompt(
    user_text: str,
    language: str,
    emotion: str,
    context: dict[str, Any],
    knowledge: str,
) -> str:
    if language == "roman_urdu":
        language_rule = (
            "CRITICAL LANGUAGE RULE: You MUST reply in conversational Pakistani Roman Urdu using Latin/English alphabet ONLY. "
            "Do NOT reply in English. Do NOT add English translations in brackets or parentheses. "
            "(For example: 'Aap ki fees abhi tak jama nahi hui, lekin aap exam mein baith sakte hain.'). "
            "Never use Urdu or Arabic script. Keep technical university terms like CGPA, GPA, semester, course, fees, exam, and advisor in English."
        )
        reply_cue = "ADVISOR_REPLY (in Roman Urdu):"
    else:
        language_rule = "Reply only in clear, natural English."
        reply_cue = "ADVISOR_REPLY:"

    return f"""
You are a helpful university academic advisor.

LANGUAGE:
{language_rule}

ROLE & DOMAIN RESTRICTIONS (STRICT):
- You are STRICTLY a University Academic Advisor.
- You ONLY answer questions regarding university studies: degree rules, semester policies, courses, CGPA/GPA, exams, attendance, credit hours, and fees.
- If the student asks about anything outside university academics (such as weather, sports, general knowledge, movies, news, food/recipes, casual chitchat, or non-academic coding tasks):
  Do NOT answer the off-topic question. Politely refuse and state that you can only assist with university and academic matters.
  Example English refusal: "I am your university academic advisor and can only assist with academic matters, courses, CGPA, and university policies. Please ask a university-related question."
  Example Roman Urdu refusal: "Main aap ka university academic advisor hoon aur sirf academic mamlaat, courses, CGPA, aur university policies mein madad kar sakta hoon. Please apni parhai se mutaliq sawal poochein."

STYLE:
- Use no more than 5-10 short conversational sentences.
- Answer the student's exact question directly and first.
- Do NOT give unsolicited warnings or lectures about workload or grades unless the student specifically asks for evaluation.
- Never invent a student fact or university rule.

ACADEMIC ADVISING GUIDELINES:
- When advising what subjects to take or study next, check 'next_semester_course_recommendations' in STUDENT_CONTEXT_JSON.
- Always check if the student has 'unresolved_failed_courses_to_repeat'; advise them to prioritize repeating failed prerequisite courses first.
- Recommend courses listed in 'eligible_courses_to_study_next' (their prerequisites are already passed).
- If they ask about taking a course listed in 'locked_courses_needing_prerequisites', clearly explain what prerequisite course they must pass first.
- Mention their Scheme of Study (SOS) and degree completion progress when relevant.
- If 'attempt_warnings' shows 2 or more attempts on a course, counsel them gently regarding repeat course policies.

STUDENT_CONTEXT_JSON:
{json.dumps(context, ensure_ascii=False, default=str, separators=(",", ":"))}

RETRIEVED_UNIVERSITY_RULES:
{knowledge or "No matching rule was retrieved. Do not invent one."}

EEG_TONE_HINT:
{emotion}

STUDENT_MESSAGE:
{user_text}

{reply_cue}
""".strip()


def stream_llm_reply(prompt: str, model: str) -> Iterator[str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 120,
            "num_ctx": 1536,
            "num_thread": min(4, os.cpu_count() or 2),
        },
    }
    with requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=(5, 150),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break


def select_tts_voice(
    engine: pyttsx3.Engine,
    language: str = "english",
    detected_language: str = "en",
) -> str | None:
    """Select the most suitable voice for speech synthesis.

    If the user language is Urdu or Roman Urdu (even if the generated reply uses Latin/English alphabet),
    pyttsx3 is instructed to use an Urdu or Hindi voice (e.g., Lekha on macOS, Kalpana on Windows, or hi/ur)
    so that phonetics and pronunciation sound natural instead of being mispronounced with a Western English accent.
    """
    try:
        voices = engine.getProperty("voices")
    except Exception:
        return None

    if not voices:
        return None

    is_urdu_or_hindi = (
        language == "roman_urdu"
        or (bool(detected_language) and detected_language.lower() in URDU_LANGUAGE_CODES)
        or (bool(detected_language) and detected_language.lower() in ("ur", "hi", "urdu", "hindi", "pa"))
    )

    if is_urdu_or_hindi:
        # 1. Search for Urdu voice
        for v in voices:
            langs = [str(l).lower() for l in getattr(v, "languages", [])]
            vid = v.id.lower()
            vname = v.name.lower()
            if any("ur" in l for l in langs) or "urdu" in vid or "urdu" in vname:
                return v.id

        # 2. Search for Hindi voice (e.g. Lekha on macOS, Kalpana on Windows)
        for v in voices:
            langs = [str(l).lower() for l in getattr(v, "languages", [])]
            vid = v.id.lower()
            vname = v.name.lower()
            if any("hi" in l for l in langs) or "hindi" in vid or "hindi" in vname or "lekha" in vid or "kalpana" in vid:
                return v.id

        # 3. Fallback to Indian English (e.g. Rishi on macOS, en_IN)
        for v in voices:
            langs = [str(l).lower() for l in getattr(v, "languages", [])]
            vid = v.id.lower()
            vname = v.name.lower()
            if any("en_in" in l or "en-in" in l for l in langs) or "rishi" in vid or "en-in" in vid:
                return v.id

    # For English:
    for v in voices:
        langs = [str(l).lower() for l in getattr(v, "languages", [])]
        vname = v.name.lower()
        vid = v.id.lower()
        if any(l in ("en_us", "en-us", "en_gb", "en-gb") for l in langs) and any(
            n in vname or n in vid for n in ("samantha", "daniel", "karen", "alex", "ava", "rishi")
        ):
            return v.id

    return None


def synthesize_speech(
    text: str,
    output_path: Path,
    language: str = "english",
    detected_language: str = "en",
    preferred_voice: str | None = None,
) -> bool:
    try:
        from FYP2.services import tts_voice_service

        is_urdu_or_hindi = (
            language == "roman_urdu"
            or (bool(detected_language) and detected_language.lower() in URDU_LANGUAGE_CODES)
            or (bool(detected_language) and detected_language.lower() in ("ur", "hi", "urdu", "hindi", "pa"))
        )

        # If student has a preferred voice saved, use it! Otherwise pick best default
        if preferred_voice and preferred_voice.strip():
            chosen_voice = preferred_voice.strip()
        else:
            chosen_voice = "Lekha" if is_urdu_or_hindi else "Samantha"

        logger.info(
            "Synthesizing speech reply with voice='%s' (language='%s', detected='%s')",
            chosen_voice,
            language,
            detected_language,
        )

        # Synthesize sample or speech using robust tts_voice_service
        wav_path, _, _ = tts_voice_service.synthesize_sample(chosen_voice, text)
        if wav_path.exists():
            import shutil
            shutil.copyfile(wav_path, output_path)
            return output_path.exists()

        return False
    except Exception:
        logger.exception("Text-to-speech failed")
        return False


def save_voice_messages(
    session_id: int,
    user_text: str,
    bot_text: str,
    user_audio_url: str,
    bot_audio_url: str | None,
    emotion: str,
    db_conn,
) -> None:
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO Chat_Message
            (session_id, sender, message_text, voice_file, emotion)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (session_id, "User", user_text, user_audio_url, emotion),
        )
        cur.execute(
            """
            INSERT INTO Chat_Message
            (session_id, sender, message_text, voice_file, emotion)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (session_id, "Bot", bot_text, bot_audio_url, "Neutral"),
        )
        db_conn.commit()
    except Exception:
        _safe_rollback(db_conn)
        logger.exception("Could not save voice messages")
    finally:
        cur.close()


def stream_voice_chat(
    session_id: int, audio_file_bytes: bytes, db_conn, student_id: str
) -> Iterator[dict[str, Any]]:
    started = time.perf_counter()
    timings: dict[str, int] = {}
    unique_id = uuid.uuid4().hex
    user_audio_path = AUDIO_DIR / f"user_{unique_id}.wav"
    bot_audio_path = AUDIO_DIR / f"bot_{unique_id}.wav"
    user_audio_url = f"/static/audio/{user_audio_path.name}"
    bot_audio_url: str | None = f"/static/audio/{bot_audio_path.name}"

    try:
        stage = time.perf_counter()
        yield _event("status", message="Preparing your voice...")
        audio = AudioSegment.from_file(io.BytesIO(audio_file_bytes))
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(user_audio_path, format="wav")
        timings["audio_conversion"] = round((time.perf_counter() - stage) * 1000)

        stage = time.perf_counter()
        yield _event("status", message="Understanding your voice...")
        user_text, detected_language = _transcribe(user_audio_path)
        raw_user_text = user_text

        if contains_urdu_script(user_text) or detected_language in URDU_LANGUAGE_CODES or response_language(detected_language, user_text) == "roman_urdu":
            detected_language = "ur"

        timings["transcription"] = round((time.perf_counter() - stage) * 1000)

        if not user_text or is_hallucination(user_text):
            raw_user_text = "[No speech detected]"
            display_text = "[No speech detected]"
        else:
            display_text = clean_and_normalize_roman_urdu(raw_user_text) if contains_urdu_script(raw_user_text) else raw_user_text

        language = response_language(detected_language, display_text)
        yield _event(
            "transcript",
            text=display_text,
            detected_language=detected_language,
            response_language=language,
        )

        if raw_user_text == "[No speech detected]":
            reply = (
                "Mujhe aap ki awaaz saaf sunai nahi di, dobara bol dein."
                if language == "roman_urdu"
                else "I could not hear you clearly. Please try again."
            )
        else:
            stage = time.perf_counter()
            yield _event("status", message="Analyzing query with academic advisor graph...")
            emotion = get_latest_emotion(db_conn)

            graph_state: AdvisorState = {
                "query": raw_user_text,
                "language": language,
                "detected_language": detected_language,
                "emotion": emotion,
                "student_id": student_id,
                "session_id": session_id,
                "db_conn": db_conn,
            }

            reply = None
            prompt = None
            model = None

            for event in advisor_graph.stream(graph_state):
                if "handle_greeting" in event:
                    reply = event["handle_greeting"].get("response")
                    break
                elif "handle_irrelevant" in event:
                    reply = event["handle_irrelevant"].get("response")
                    break
                elif "retrieve_knowledge" in event:
                    retrieved_data = event["retrieve_knowledge"]
                    prompt = retrieved_data.get("prompt", "")
                    model = retrieved_data.get("model", OLLAMA_MODEL_ENGLISH)
                    timings["context"] = round((time.perf_counter() - stage) * 1000)
                    break

            yield _event("reply_start", response_language=language)
            if reply:
                yield _event("token", text=reply)
            elif prompt and model:
                stage = time.perf_counter()
                chunks: list[str] = []
                script_violation = False
                try:
                    for token in stream_llm_reply(prompt, model):
                        chunks.append(token)
                        if language == "roman_urdu" and contains_urdu_script(token):
                            script_violation = True
                        elif not script_violation:
                            yield _event("token", text=token)
                    reply = "".join(chunks).strip()
                    if language == "roman_urdu":
                        # Remove any English translation in parentheses throughout the reply
                        reply = re.sub(r"\s*\([A-Za-z0-9\s,.'’?!]+\)", "", reply).strip()
                except Exception:
                    logger.exception("Ollama generation failed")
                    reply = (
                        "AI model abhi available nahi hai, thori dair baad dobara koshish karein."
                        if language == "roman_urdu"
                        else "The AI model is unavailable. Please try again shortly."
                    )
                    yield _event("token", text=reply)
                timings["generation"] = round((time.perf_counter() - stage) * 1000)

            if not reply:
                reply = (
                    "Main is sawal ka jawab abhi tayar nahi kar saka."
                    if language == "roman_urdu"
                    else "I could not prepare an answer to that question."
                )

            if language == "roman_urdu" and contains_urdu_script(reply):
                logger.warning("Model returned Urdu script despite Roman Urdu instruction")
                reply = (
                    "Maazrat, main jawab Roman Urdu mein tayar nahi kar saka. "
                    "Meherbani karke dobara koshish karein."
                )

        emotion = locals().get("emotion", "Neutral")
        yield _event("reply_complete", text=reply)

        stage = time.perf_counter()
        yield _event("status", message="Preparing the audio reply...")

        # Fetch student's preferred voice from database
        preferred_voice = None
        if student_id and db_conn:
            try:
                cur = db_conn.cursor()
                cur.execute(
                    "SELECT preferred_voice FROM Student WHERE sid = %s OR CAST(user_id AS TEXT) = %s;",
                    (str(student_id), str(student_id)),
                )
                row = cur.fetchone()
                cur.close()
                if row and row.get("preferred_voice"):
                    preferred_voice = row["preferred_voice"]
            except Exception:
                _safe_rollback(db_conn)

        if not synthesize_speech(
            reply,
            bot_audio_path,
            language=language,
            detected_language=detected_language,
            preferred_voice=preferred_voice,
        ):
            bot_audio_url = None
        timings["speech"] = round((time.perf_counter() - stage) * 1000)

        save_voice_messages(
            session_id=session_id,
            user_text=user_text,
            bot_text=reply,
            user_audio_url=user_audio_url,
            bot_audio_url=bot_audio_url,
            emotion=emotion,
            db_conn=db_conn,
        )
        timings["total"] = round((time.perf_counter() - started) * 1000)

        yield _event(
            "complete",
            data={
                "user_text": user_text,
                "bot_text": reply,
                "user_audio_url": user_audio_url,
                "bot_audio_url": bot_audio_url,
                "user_emotion": emotion,
                "detected_language": detected_language,
                "response_language": language,
                "timing_ms": timings,
            },
        )
    except Exception:
        logger.exception("Voice processing failed")
        yield _event(
            "error",
            message="Voice processing failed. Please try again.",
        )


def process_voice_chat(
    session_id: int, audio_file_bytes: bytes, db_conn, student_id: str
) -> dict[str, Any]:
    """Compatibility wrapper for clients that still expect one JSON response."""
    for event in stream_voice_chat(session_id, audio_file_bytes, db_conn, student_id):
        if event["type"] == "error":
            raise RuntimeError(event["message"])
        if event["type"] == "complete":
            return event["data"]
    raise RuntimeError("Voice processing ended without a response")


def process_text_chat(
    session_id: int,
    message: str,
    db_conn,
    student_id: str | None = None,
    emotion: str | None = None,
    generate_audio: bool = False,
) -> dict[str, Any]:
    """Process a typed text message through the academic advisor graph."""
    started = time.perf_counter()
    timings: dict[str, int] = {}

    raw_user_text = message.strip()
    if not raw_user_text:
        raise ValueError("Message cannot be empty")

    # Look up student_id if not explicitly provided
    if not student_id and db_conn:
        try:
            cur = db_conn.cursor()
            cur.execute("SELECT student_id FROM Chat_Session WHERE session_id = %s;", (session_id,))
            session_row = cur.fetchone()
            cur.close()
            if session_row and session_row.get("student_id"):
                student_id = session_row.get("student_id")
        except Exception:
            _safe_rollback(db_conn)

    if not student_id:
        student_id = "2023-ARID-0274"

    # Language and script detection
    if contains_urdu_script(raw_user_text):
        detected_language = "ur"
        display_text = clean_and_normalize_roman_urdu(raw_user_text)
    else:
        lang_probe = response_language("en", raw_user_text)
        detected_language = "ur" if lang_probe == "roman_urdu" else "en"
        display_text = raw_user_text

    language = response_language(detected_language, display_text)

    # Determine emotion
    if not emotion or emotion.strip().upper() == "NEUTRAL":
        user_emotion = get_latest_emotion(db_conn)
    else:
        user_emotion = emotion.strip()

    stage = time.perf_counter()
    graph_state: AdvisorState = {
        "query": raw_user_text,
        "language": language,
        "detected_language": detected_language,
        "emotion": user_emotion,
        "student_id": student_id,
        "session_id": session_id,
        "db_conn": db_conn,
    }

    reply = None
    prompt = None
    model = None

    for event in advisor_graph.stream(graph_state):
        if "handle_greeting" in event:
            reply = event["handle_greeting"].get("response")
            break
        elif "handle_irrelevant" in event:
            reply = event["handle_irrelevant"].get("response")
            break
        elif "retrieve_knowledge" in event:
            retrieved_data = event["retrieve_knowledge"]
            prompt = retrieved_data.get("prompt", "")
            model = retrieved_data.get("model", OLLAMA_MODEL_ENGLISH)
            timings["context"] = round((time.perf_counter() - stage) * 1000)
            break

    if not reply and prompt and model:
        stage = time.perf_counter()
        chunks: list[str] = []
        try:
            for token in stream_llm_reply(prompt, model):
                chunks.append(token)
            reply = "".join(chunks).strip()
            if language == "roman_urdu":
                reply = re.sub(r"\s*\([A-Za-z0-9\s,.'’?!]+\)", "", reply).strip()
        except Exception:
            logger.exception("Ollama generation failed for text chat")
            reply = (
                "AI model abhi available nahi hai, thori dair baad dobara koshish karein."
                if language == "roman_urdu"
                else "The AI model is unavailable. Please try again shortly."
            )
        timings["generation"] = round((time.perf_counter() - stage) * 1000)

    if not reply:
        reply = (
            "Main is sawal ka jawab abhi tayar nahi kar saka."
            if language == "roman_urdu"
            else "I could not prepare an answer to that question."
        )

    if language == "roman_urdu" and contains_urdu_script(reply):
        logger.warning("Model returned Urdu script despite Roman Urdu instruction")
        reply = (
            "Maazrat, main jawab Roman Urdu mein tayar nahi kar saka. "
            "Meherbani karke dobara koshish karein."
        )

    bot_audio_url: str | None = None
    if generate_audio:
        stage = time.perf_counter()
        unique_id = uuid.uuid4().hex
        bot_audio_path = AUDIO_DIR / f"bot_{unique_id}.wav"
        preferred_voice = None
        if student_id and db_conn:
            try:
                cur = db_conn.cursor()
                cur.execute(
                    "SELECT preferred_voice FROM Student WHERE sid = %s OR CAST(user_id AS TEXT) = %s;",
                    (str(student_id), str(student_id)),
                )
                row = cur.fetchone()
                cur.close()
                if row and row.get("preferred_voice"):
                    preferred_voice = row["preferred_voice"]
            except Exception:
                _safe_rollback(db_conn)

        if synthesize_speech(
            reply,
            bot_audio_path,
            language=language,
            detected_language=detected_language,
            preferred_voice=preferred_voice,
        ):
            bot_audio_url = f"/static/audio/{bot_audio_path.name}"
        timings["speech"] = round((time.perf_counter() - stage) * 1000)

    save_voice_messages(
        session_id=session_id,
        user_text=display_text,
        bot_text=reply,
        user_audio_url=None,
        bot_audio_url=bot_audio_url,
        emotion=user_emotion,
        db_conn=db_conn,
    )
    timings["total"] = round((time.perf_counter() - started) * 1000)

    return {
        "user_text": display_text,
        "bot_text": reply,
        "user_audio_url": None,
        "bot_audio_url": bot_audio_url,
        "user_emotion": user_emotion,
        "detected_language": detected_language,
        "response_language": language,
        "timing_ms": timings,
    }

