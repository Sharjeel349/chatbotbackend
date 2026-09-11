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


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE_DIR / "static" / "audio"
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL_ENGLISH = os.getenv("OLLAMA_MODEL_ENGLISH", "gemma2:2b")
OLLAMA_MODEL_ROMAN_URDU = os.getenv("OLLAMA_MODEL_ROMAN_URDU", "mistral")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
STT_BACKEND = os.getenv("STT_BACKEND", "openai-whisper")
MAX_HISTORY_MESSAGES = 6

_stt_lock = threading.Lock()

URDU_LANGUAGE_CODES = {"ur", "hi", "pa"}
URDU_SCRIPT_RANGES = (
    ("\u0600", "\u06ff"),
    ("\u0750", "\u077f"),
    ("\u08a0", "\u08ff"),
)


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

URDU_WORD_MAP = {
    "میرا": "Mera", "میری": "Meri", "میرے": "Mere", "اپنا": "Apna", "اپنے": "Apne", "اپنی": "Apni",
    "کتنا": "kitna", "کتنی": "kitni", "کتنے": "kitne", "کیا": "kya", "کون": "kon", "کونسا": "konsa",
    "سی": "CGPA", "جی": "GPA", "پی": "P", "اے": "A",
    "سی جی پی اے": "CGPA", "جی پی اے": "GPA", "ہے": "hai", "ہیں": "hain", "ہو": "ho", "ہوں": "hoon",
    "تھا": "tha", "تھی": "thi", "تھے": "the",
    "کیسا": "kaisa", "کیسی": "kaisi", "کیسے": "kaise", "رزلٹ": "result", "مارکس": "marks",
    "سمسٹر": "semester", "کورس": "course", "کورسز": "courses", "پروگرامنگ": "programming",
    "گریڈ": "grade", "گریڈز": "grades", "فیل": "fail", "پاس": "pass", "ٹیچر": "teacher",
    "استاد": "teacher", "مشیر": "advisor", "ایڈوائزر": "advisor", "مجھے": "mujhe",
    "بتاو": "batao", "بتائیں": "batain", "نام": "naam", "تفصیل": "detail", "تفصیلات": "details",
    "معلومات": "information", "پرفارمنس": "performance", "کارکردگی": "performance",
    "ہم": "hum", "آپ": "Aap", "تم": "tum", "وہ": "woh", "سبجیکٹ": "subject", "سبجیکٹس": "subjects",
}

URDU_CHAR_MAP = {
    'ا': 'a', 'آ': 'aa', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's',
    'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd', 'ذ': 'z',
    'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'z', 'س': 's', 'ش': 'sh', 'ص': 's',
    'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
    'ک': 'k', 'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'o',
    'ہ': 'h', 'ھ': 'h', 'ء': 'a', 'ی': 'y', 'ے': 'ey', '۰': '0', '۱': '1',
    '۲': '2', '۳': '3', '۴': '4', '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
}


def contains_urdu_script(text: str) -> bool:
    return any(start <= char <= end for char in text for start, end in URDU_SCRIPT_RANGES)


def character_transliterate_urdu(word: str) -> str:
    return "".join(URDU_CHAR_MAP.get(char, char) for char in word)


def dictionary_transliterate_urdu(text: str) -> str:
    words = text.split()
    converted_words = []
    for word in words:
        clean = word.strip("،.؟!?")
        if clean in URDU_WORD_MAP:
            converted_words.append(URDU_WORD_MAP[clean])
        elif contains_urdu_script(clean):
            converted_words.append(character_transliterate_urdu(clean))
        else:
            converted_words.append(word)
    return " ".join(converted_words)


def convert_to_roman_urdu_offline(text: str) -> str:
    if not text or not contains_urdu_script(text):
        return text

    dict_converted = dictionary_transliterate_urdu(text)
    if not contains_urdu_script(dict_converted):
        return dict_converted

    prompt = (
        "Convert the following Urdu script text into simple natural Pakistani Roman Urdu using Latin alphabet only. "
        "Keep terms like CGPA, GPA, semester, course, programming, advisor in English script. "
        "Output ONLY the converted Roman Urdu sentence without any quote or explanation:\n\n"
        f"Urdu: {text}\nRoman Urdu:"
    )
    try:
        payload = {
            "model": OLLAMA_MODEL_ROMAN_URDU,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 40},
        }
        res = requests.post(OLLAMA_URL, json=payload, timeout=(3.0, 15.0))
        if res.status_code == 200:
            converted = res.json().get("response", "").strip().strip('"')
            if converted and not contains_urdu_script(converted):
                return converted
    except requests.exceptions.RequestException:
        logger.warning("Local Ollama Roman Urdu transliteration timed out or skipped; using dictionary fallback")

    return dict_converted


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

            logger.info("Loading faster-whisper model '%s' with INT8", WHISPER_MODEL)
            model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
                cpu_threads=min(4, os.cpu_count() or 2),
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
            "Mera kitna CGPA hai, mera result kaisa hai, programming courses performance, "
            "advisor name, fail courses, semester freeze."
        )
        if backend == "faster-whisper":
            try:
                segments, info = model.transcribe(
                    str(audio_path),
                    language="ur",
                    beam_size=1,
                    temperature=0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 400},
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
                if text:
                    return text, "ur"
            except Exception:
                logger.exception("faster-whisper urdu attempt failed")

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
            return text, info.language or "en"

        try:
            result = model.transcribe(
                str(audio_path),
                language="ur",
                fp16=False,
                temperature=0,
                beam_size=1,
                condition_on_previous_text=False,
                verbose=False,
            )
            text = result.get("text", "").strip()
            if text:
                return text, "ur"
        except Exception:
            logger.exception("openai-whisper urdu attempt failed")

        result = model.transcribe(
            str(audio_path),
            fp16=False,
            temperature=0,
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            verbose=False,
        )
        return result.get("text", "").strip(), result.get("language", "en")


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
            SELECT c.course_id, c.title, c.skill_category, co.session,
                   u_teacher.name AS teacher_name, e.grade
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id
            JOIN Teacher t ON co.teacher_id = t.teacher_id
            JOIN Users u_teacher ON t.user_id = u_teacher.user_id
            WHERE e.grade = 'F' AND e.student_id = %s;
            """,
            (student_id,),
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
            JOIN Course c ON co.course_id = c.course_id
            WHERE e.student_id = %s AND e.grade IS NOT NULL
            ORDER BY co.session DESC, c.title ASC
            LIMIT 30;
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
            JOIN Course c ON co.course_id = c.course_id
            JOIN Teacher t ON co.teacher_id = t.teacher_id
            JOIN Users u ON t.user_id = u.user_id
            WHERE e.student_id = %s AND e.grade IS NULL;
            """,
            (student_id,),
        )
        return cur.fetchall()
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


def get_recent_chat_history(session_id: int, db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT sender, message_text
            FROM Chat_Message
            WHERE session_id = %s
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
        },
    ):
        context["current_enrollments"] = get_current_enrollments(student_id, db_conn)
        context["failed_courses"] = get_failed_courses(student_id, db_conn)
        context["transcript"] = get_transcript(student_id, db_conn)

    if _mentions(query, {"freeze", "semester", "gap", "فریز", "سمسٹر"}):
        context["freeze_history"] = get_freeze_history(student_id, db_conn)

    if _mentions(
        query, {"teacher", "instructor", "sir", "madam", "استاد", "ٹیچر"}
    ):
        context["teacher_statistics"] = get_teacher_strictness_stats(db_conn)

    if _mentions(
        query, {"grade", "transcript", "result", "marks", "performance", "performing", "گریڈ", "نتیجہ"}
    ):
        context["transcript"] = get_transcript(student_id, db_conn)

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


def retrieve_knowledge(query: str) -> str:
    """Retrieve small local rules without loading another native ML runtime.

    The corpus is currently tiny, so lexical retrieval is faster and more
    reliable than loading Chroma plus a second embedding model on every API
    process. The separate index script remains available as the corpus grows.
    """
    normalized = query.casefold()
    expansions = {
        "سمسٹر": "semester",
        "فریز": "freeze",
        "کورس": "course",
        "سبجیکٹ": "course",
        "پروگرامنگ": "programming",
        "استاد": "teacher",
        "ٹیچر": "teacher",
        "گریڈ": "grade",
        "نتیجہ": "result",
        "سی جی پی اے": "cgpa",
    }
    for source, replacement in expansions.items():
        if source in normalized:
            normalized += f" {replacement}"

    query_terms = set(re.findall(r"[a-z0-9]+", normalized))
    if not query_terms:
        return ""

    ranked: list[tuple[int, str]] = []
    for section in _knowledge_sections():
        section_terms = set(re.findall(r"[a-z0-9]+", section.casefold()))
        score = len(query_terms & section_terms)
        if score:
            ranked.append((score, section))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(section for _, section in ranked[:3])


def direct_database_reply(
    query: str, language: str, context: dict[str, Any]
) -> str | None:
    profile = context.get("profile") or {}
    normalized = query.casefold()

    if any(
        term in normalized
        for term in ("cgpa", "gpa", "سی جی پی اے", "جی پی اے")
    ):
        cgpa = profile.get("cgpa")
        if cgpa is not None:
            return (
                f"Aap ka CGPA {cgpa} hai."
                if language == "roman_urdu"
                else f"Your CGPA is {cgpa}."
            )

    if any(
        term in normalized
        for term in ("advisor", "supervisor", "ایڈوائزر", "مشیر")
    ):
        advisor = profile.get("advisor_name")
        if advisor:
            return (
                f"Aap ke academic advisor {advisor} hain."
                if language == "roman_urdu"
                else f"Your academic advisor is {advisor}."
            )

    if (
        "semester" in normalized or "سمسٹر" in normalized
    ) and "freeze" not in normalized and "فریز" not in normalized:
        semester = profile.get("semester")
        if semester:
            return (
                f"Aap is waqt semester {semester} mein hain."
                if language == "roman_urdu"
                else f"You are currently in semester {semester}."
            )

    return None


def build_prompt(
    user_text: str,
    language: str,
    emotion: str,
    context: dict[str, Any],
    knowledge: str,
) -> str:
    if language == "roman_urdu":
        language_rule = (
            "Reply ONLY in natural Pakistani Roman Urdu using English/Latin letters. "
            "Never use Urdu or Arabic script. Keep common university words such as "
            "CGPA, course, semester, advisor and programming in English."
        )
    else:
        language_rule = "Reply only in clear, natural English."

    return f"""
You are a careful university academic advisor.

LANGUAGE:
{language_rule}

STYLE:
- Use no more than 3 short conversational sentences.
- Answer the student's exact question directly and first.
- Do NOT repeat identical sentences or generic templates from previous turns.
- If asked about course performance or grades, explicitly state specific course titles, grades, or pass/fail records from STUDENT_CONTEXT_JSON.
- Be empathetic but concise and direct.
- Never invent a student fact or university rule.
- If required information is missing, say what is missing.

ADVISING:
- Warn against programming overload when current or failed programming courses show risk.
- For teacher comparisons, use both failure and A-grade counts; do not label a teacher unfairly from a tiny sample.
- The degree limit in the available rules is 12 registered semesters.
- Treat the EEG state only as a soft tone hint, never as a diagnosis.

STUDENT_CONTEXT_JSON:
{json.dumps(context, ensure_ascii=False, default=str, separators=(",", ":"))}

RETRIEVED_UNIVERSITY_RULES:
{knowledge or "No matching rule was retrieved. Do not invent one."}

EEG_TONE_HINT:
{emotion}

STUDENT_MESSAGE:
{user_text}

ADVISOR_REPLY:
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
            "num_predict": 160,
            "num_ctx": 4096,
        },
    }
    with requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=(5, 90),
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


def synthesize_speech(text: str, output_path: Path) -> bool:
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        return output_path.exists()
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

        if contains_urdu_script(user_text) or detected_language in URDU_LANGUAGE_CODES:
            user_text = convert_to_roman_urdu_offline(user_text)
            detected_language = "ur"

        timings["transcription"] = round((time.perf_counter() - stage) * 1000)

        if not user_text:
            user_text = "[No speech detected]"
        language = response_language(detected_language, user_text)
        yield _event(
            "transcript",
            text=user_text,
            detected_language=detected_language,
            response_language=language,
        )

        if user_text == "[No speech detected]":
            reply = (
                "Mujhe aap ki awaaz saaf sunai nahi di, dobara bol dein."
                if language == "roman_urdu"
                else "I could not hear you clearly. Please try again."
            )
        else:
            stage = time.perf_counter()
            yield _event("status", message="Checking your academic information...")
            emotion = get_latest_emotion(db_conn)
            context = build_student_context(user_text, session_id, student_id, db_conn)
            knowledge = retrieve_knowledge(user_text)
            timings["context"] = round((time.perf_counter() - stage) * 1000)

            reply = direct_database_reply(user_text, language, context)
            yield _event("reply_start", response_language=language)
            if reply:
                yield _event("token", text=reply)
            else:
                prompt = build_prompt(user_text, language, emotion, context, knowledge)
                model = (
                    OLLAMA_MODEL_ROMAN_URDU
                    if language == "roman_urdu"
                    else OLLAMA_MODEL_ENGLISH
                )
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
        if not synthesize_speech(reply, bot_audio_path):
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
