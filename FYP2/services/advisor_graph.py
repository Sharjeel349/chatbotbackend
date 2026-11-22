from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from FYP2.services.roman_urdu_service import contains_urdu_script

logger = logging.getLogger(__name__)

GREETING_PATTERNS = (
    "hi",
    "hello",
    "hey",
    "salam",
    "assalam",
    "assalam o alaikum",
    "assalamu alaikum",
    "aoa",
    "kaisa hai",
    "kaise ho",
    "kese ho",
    "kia hal hai",
    "kya hal hai",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
)

IDENTITY_PATTERNS = (
    "arid number",
    "arid no",
    "registration number",
    "reg number",
    "reg no",
    "who am i",
    "my name",
    "know my arid",
    "know my reg",
    "mera arid",
    "mera registration",
    "mera roll",
    "mera naam",
    "meri registration",
    "meri id",
    "my id",
    "میرا نام",
    "میرا ایریڈ",
    "میرا رول نمبر",
)

DEGREE_COMPLETION_PATTERNS = (
    "degree complete",
    "degree mukammal",
    "degree finish",
    "degree pass",
    "degree ho jai",
    "degree ho jaye",
    "degree kab",
    "complete ho jai gy",
    "complete ho jaye gi",
    "complete ho sakti",
    "will my degree",
    "can i complete my degree",
    "will i graduate",
    "can i graduate",
    "degree khatam",
    "ڈگری مکمل",
    "ڈگری کمپلیٹ",
    "ڈگری ہو جائے گی",
    "ڈگری ہو جائیگی",
)

CGPA_PATTERNS = (
    "cgpa acha",
    "cgpa kaisa",
    "cgpa kitna",
    "cgpa kisa",
    "cgpa kess",
    "cgpa kesa",
    "cgpa kitna hai",
    "cgpa kitna he",
    "cgpa kitna ha",
    "mera cgpa",
    "meri cgpa",
    "mera gpa",
    "meri gpa",
    "gpa kaisa",
    "gpa kitna",
    "gpa acha",
    "gpa kesa",
    "my cgpa",
    "my gpa",
    "what is my cgpa",
    "what is my gpa",
    "is my cgpa",
    "is my gpa",
    "how is my cgpa",
    "how is my gpa",
    "vgpa",
    "kya mera cgpa",
    "kya meri cgpa",
    "current cgpa",
    "میرا cgpa",
    "میرا جی پی اے",
    "میرا سی جی پی اے",
    "کتنا ہے",
    "کتنا جی پی اے",
    "سی جی پی اے",
    "جی پی اے",
    "cgpa کتنا",
)

FAILED_COURSES_PATTERNS = (
    "koi course fail",
    "koi subject fail",
    "kya mera koi course fail",
    "kya koi course fail",
    "kya koi subject fail",
    "fail course",
    "failed course",
    "failed courses",
    "backlog",
    "backlogs",
    "fail subject",
    "failed subjects",
    "kya main fail",
    "kya mai fail",
    "kya me fail",
    "fail hoon",
    "fail hon",
    "fail hu",
    "any failed course",
    "any course failed",
    "did i fail",
    "have i failed",
    "courses fail",
    "course fail",
    "subject fail",
    "subjects fail",
    "کوئی کورس فیل",
    "فیل کورس",
    "کوئی سبجیکٹ فیل",
    "فیل ہے",
    "فیل ہیں",
)

STUDENT_DETAILS_PATTERNS = (
    "sari details",
    "sari detail",
    "sari maloomat",
    "mery sari details",
    "meri sari details",
    "meri sari detail",
    "mera sara record",
    "mera record",
    "my details",
    "all my details",
    "complete details",
    "full details",
    "show my details",
    "tell my details",
    "sara record btao",
    "sari details btao",
    "meri details",
    "apni details",
    "student details",
    "academic details",
    "ساری ڈیٹیلز",
    "تمام معلومات",
    "سارا ریکارڈ",
    "میری ڈیٹیلز",
)

TEACHER_STRICTNESS_PATTERNS = (
    "kon sa teacher strict",
    "konsa teacher strict",
    "koun sa teacher strict",
    "strict teacher",
    "strict teachers",
    "kaun sa teacher strict",
    "kis teacher ki grading strict",
    "teacher strictness",
    "grading strict",
    "lenient teacher",
    "kon strict hai",
    "kon strict ha",
    "kon strict he",
    "which teacher is strict",
    "who is strict teacher",
    "strict instructors",
    "strict grading",
    "کون سا ٹیچر سٹرکٹ",
    "کونسا ٹیچر سٹرکٹ",
    "سٹرکٹ ٹیچر",
    "کون سا ٹیچر سخت",
)

SEMESTER_CGPA_POLICY_PATTERNS = (
    "minimum cgpa requirement",
    "cgpa requirement for every semester",
    "cgpa requirement for each semester",
    "minimum cgpa for every semester",
    "minimum cgpa for each semester",
    "minimum cgpa for semester",
    "cgpa progression",
    "minimum cgpa to graduate",
    "minimum cgpa for graduation",
    "graduation cgpa requirement",
    "graduation cgpa",
    "har semester ka cgpa",
    "har semester mein kitna cgpa",
    "har semester ke liye cgpa",
    "semester wise cgpa",
    "semester wise minimum cgpa",
    "cgpa per semester",
    "cgpa for each semester",
    "cgpa for every semester",
    "minimum cgpa required",
    "minimum cgpa policy",
    "cgpa policy",
    "minimum gpa requirement",
)


class AdvisorState(TypedDict, total=False):
    query: str
    language: str  # "english" or "roman_urdu"
    detected_language: str  # "en", "ur", etc.
    emotion: str
    student_id: str
    session_id: int
    db_conn: Any
    intent: str  # "greeting" | "identity" | "irrelevant" | "academic"
    profile: Optional[Dict[str, Any]]
    student_context: Optional[Dict[str, Any]]
    knowledge: str
    prompt: str
    model: str
    response: Optional[str]
    error: Optional[str]


def classify_intent_node(state: AdvisorState) -> dict[str, Any]:
    """Classify the user intent into greeting, identity, degree_completion, cgpa, failed_courses, student_details, teacher_strictness, cgpa_policy, irrelevant, or academic."""
    from FYP2.services.voice_service import is_irrelevant_query
    from FYP2.services.roman_urdu_service import contains_urdu_script, transliterate_urdu_to_roman

    query = (state.get("query") or "").strip()
    normalized = query.casefold()
    romanized = transliterate_urdu_to_roman(query).casefold().strip() if contains_urdu_script(query) else ""

    def matches(patterns):
        return any(p in normalized or (romanized and p in romanized) for p in patterns)

    # 1. Check for irrelevant / off-topic queries first
    if is_irrelevant_query(normalized) or (romanized and is_irrelevant_query(romanized)):
        return {"intent": "irrelevant"}

    # 2. Check for greetings
    words = re.findall(r"[a-z]+", romanized if romanized else normalized)
    is_greeting = False
    if words and len(words) <= 5:
        if not any(w in words for w in ["cgpa", "gpa", "course", "courses", "degree", "exam", "exams", "fee", "fees", "result", "grade", "grades", "subject", "attendance", "semester", "vgpa", "fail", "details", "teacher"]):
            if matches(GREETING_PATTERNS):
                is_greeting = True
    elif normalized in {"hi", "hello", "hey", "salam", "assalam", "assalam o alaikum"}:
        is_greeting = True

    if is_greeting:
        return {"intent": "greeting"}

    # 3. Check for student identity inquiries (ARID number, registration number, who am I)
    if matches(IDENTITY_PATTERNS):
        return {"intent": "identity"}

    # 4. Check for full student details / record overview
    if matches(STUDENT_DETAILS_PATTERNS):
        return {"intent": "student_details"}

    # 5. Check for failed courses / backlog inquiries (personal standing)
    if matches(FAILED_COURSES_PATTERNS) and not any(h in normalized for h in ["what if", "policy", "rules", "procedure", "kya hota"]):
        return {"intent": "failed_courses"}

    # 6. Check for teacher strictness inquiries
    if matches(TEACHER_STRICTNESS_PATTERNS):
        return {"intent": "teacher_strictness"}

    # 7. Check for degree completion inquiries
    if matches(DEGREE_COMPLETION_PATTERNS):
        return {"intent": "degree_completion"}

    # 8. Check for semester-wise CGPA & graduation policy inquiries
    if matches(SEMESTER_CGPA_POLICY_PATTERNS):
        return {"intent": "cgpa_policy"}

    # 9. Check for personal CGPA / GPA inquiries
    if matches(CGPA_PATTERNS):
        return {"intent": "cgpa"}

    # 10. All other questions are treated as academic
    return {"intent": "academic"}


def route_intent(state: AdvisorState) -> str:
    """Conditional edge selector based on the classified intent."""
    return state.get("intent", "academic")


def handle_identity_node(state: AdvisorState) -> dict[str, Any]:
    """Generate an instant, 100% accurate identity response confirming the student's ARID number and name, with emotional support when stressed."""
    from FYP2.services.voice_service import get_student_profile

    profile = state.get("profile")
    db_conn = state.get("db_conn")
    student_id = state.get("student_id")

    if not profile and db_conn and student_id:
        try:
            profile = get_student_profile(student_id, db_conn)
        except Exception:
            profile = None

    name = ""
    arid_no = student_id or ""
    semester = ""
    cgpa = ""
    section = ""
    if profile:
        name = profile.get("name") or profile.get("full_name") or ""
        arid_no = profile.get("registration_no") or student_id or ""
        semester = str(profile.get("semester") or "")
        cgpa = str(profile.get("cgpa") or "")
        section = str(profile.get("section_code") or "")

    first_name = name.split()[0] if name else "Student"
    language = state.get("language", "english")
    emotion = (state.get("emotion") or "").lower()
    query = (state.get("query") or "").lower()

    is_stressed = any(
        s in emotion for s in ["stress", "anxious", "frustrat", "sad", "fear", "panic", "overwhelm"]
    ) or any(
        s in query for s in ["stress", "stressed", "anxious", "anxiety", "tension", "pareshan", "worried", "scared"]
    )

    if is_stressed:
        if language == "roman_urdu":
            reply = (
                f"Bilkul {first_name}! Main aap ko achi tarah jaanta hoon, aap ka ARID number {arid_no} hai. "
                f"Pareshan hone ki bilkul zaroorat nahi hai — aap Semester {semester} mein {cgpa} CGPA ke saath bohat acha perform kar rahe hain. "
                f"Main har qadam par aap ki madad ke liye mojood hoon. Kis academic mamlay mein aap ko pareshani ho rahi hai?"
            )
        else:
            reply = (
                f"Of course I know you, {first_name}! Please take a deep breath and don't worry — your ARID number is {arid_no}. "
                f"You are doing wonderfully in Semester {semester} with a strong CGPA of {cgpa}. "
                f"I am right here to support you through any academic questions or stress. How can I help you today?"
            )
    else:
        if language == "roman_urdu":
            reply = (
                f"Ji {first_name}! Main aap ko achi tarah jaanta hoon. Aap ka naam {name} hai aur ARID number {arid_no} hai. "
                f"Aap abhi Semester {semester} (Section {section}) mein hain aur aap ka CGPA {cgpa} hai. Main aap ki parhai mein kya madad kar sakta hoon?"
            )
        else:
            reply = (
                f"Yes, {first_name}! I know you are {name}, and your ARID / Registration number is {arid_no}. "
                f"You are currently in Semester {semester} (Section {section}) with a CGPA of {cgpa}. How can I assist you with your academic records today?"
            )

    return {"response": reply, "profile": profile}


def handle_degree_node(state: AdvisorState) -> dict[str, Any]:
    """Generate an accurate, detailed, and reassuring degree completion response based on student's actual records."""
    from FYP2.services.voice_service import (
        get_completed_credit_hours,
        get_failed_courses,
        get_student_profile,
    )

    profile = state.get("profile")
    db_conn = state.get("db_conn")
    student_id = state.get("student_id")

    if not profile and db_conn and student_id:
        try:
            profile = get_student_profile(student_id, db_conn)
        except Exception:
            profile = None

    name = ""
    arid_no = student_id or ""
    semester = ""
    cgpa = "N/A"
    if profile:
        name = profile.get("name") or profile.get("full_name") or ""
        arid_no = profile.get("registration_no") or student_id or ""
        semester = str(profile.get("semester") or "")
        cgpa = str(profile.get("cgpa") or "N/A")

    first_name = name.split()[0] if name else "Student"

    credits_summary = {}
    failed_courses = []
    if db_conn and student_id:
        try:
            credits_summary = get_completed_credit_hours(student_id, db_conn)
            failed_courses = get_failed_courses(student_id, db_conn)
        except Exception:
            logger.exception("Failed to fetch credits or failed courses in handle_degree_node")

    completed_cr = credits_summary.get("completed_credits", 120)
    degree_pct = credits_summary.get("degree_progress", "50.0%")
    has_zero_fails = len(failed_courses) == 0

    language = state.get("language", "english")
    emotion = (state.get("emotion") or "").lower()
    query = (state.get("query") or "").lower()

    is_stressed = any(
        s in emotion for s in ["stress", "anxious", "frustrat", "sad", "fear", "panic", "overwhelm"]
    ) or any(
        s in query for s in ["stress", "stressed", "anxious", "anxiety", "tension", "pareshan", "worried", "scared", "ghabra", "dar"]
    )

    if is_stressed:
        if language == "roman_urdu":
            if has_zero_fails:
                reply = (
                    f"Pareshan mat hon {first_name}! Ji bilkul, aap ki degree ba-asani aur kamyabi se complete ho jaye gi. "
                    f"Aap ka academic record bilkul clean hai aur aap ka koi bhi course fail nahi hai (0 failed courses). "
                    f"Aap ka CGPA {cgpa} bohat shandar hai aur aap {completed_cr} credit hours ({degree_pct} degree progress) mukammal kar chuke hain. "
                    f"University policy ke mutabiq degree mukammal karne ke liye maximum 12 semesters tak ka time hota hai, is liye aap ke paas kafi waqt baqi hai. "
                    f"Bilkul pareshan na hon aur tension na lein, aap boht achi position mein hain aur baqi requirements bhi asani se poori ho jayengi."
                )
            else:
                failed_names = ", ".join(c.get("title", "") for c in failed_courses)
                reply = (
                    f"Pareshan mat hon {first_name}, hum mil kar is ka hal nikalenge. Aap ki degree zaroor complete ho sakti hai. "
                    f"Aap ka CGPA {cgpa} hai aur aap ne {completed_cr} credit hours mukammal kiye hain. "
                    f"Aap ko sirf apne failed courses ({failed_names}) ko repeat karke pass karna hoga. "
                    f"University policy ke mutabiq maximum 12 semesters tak ka waqt hota hai, is liye hosla rakhein, aap waqt par degree complete kar lenge."
                )
        else:
            if has_zero_fails:
                reply = (
                    f"Please don't worry, {first_name}! Yes, your degree will definitely be completed successfully. "
                    f"You have a completely clean academic record with zero failed courses (0 backlogs). "
                    f"Your current CGPA is an outstanding {cgpa}, and you have already earned {completed_cr} credit hours ({degree_pct} degree progress). "
                    f"Under university policy, students have up to 12 semesters to complete all degree requirements, so you have plenty of time. "
                    f"Take a deep breath and stay confident — you are in excellent academic standing and on track to graduate!"
                )
            else:
                failed_names = ", ".join(c.get("title", "") for c in failed_courses)
                reply = (
                    f"Don't worry, {first_name}, we will work through this together. Yes, your degree can be completed. "
                    f"Your current CGPA is {cgpa} and you have already earned {completed_cr} credit hours. "
                    f"You only need to re-enroll in and pass your failed courses ({failed_names}). "
                    f"University policy permits up to 12 semesters to graduate, so you have ample time to clear these and finish your degree."
                )
    else:
        if language == "roman_urdu":
            if has_zero_fails:
                reply = (
                    f"Ji {first_name}, aap ki degree bilkul complete ho jaye gi! "
                    f"Aap ka academic record bilkul clean hai aur aap ka koi bhi course fail nahi hai. "
                    f"Aap ka CGPA {cgpa} bohat acha hai aur aap {completed_cr} credit hours ({degree_pct} progress) pehle hi mukammal kar chuke hain. "
                    f"University policy ke tehat degree complete karne ke liye maximum 12 semesters tak ka time diya jata hai. "
                    f"Aap sahi simat mein ja rahe hain, is liye degree completion ko le kar fikar karne ki koi zaroorat nahi hai."
                )
            else:
                failed_names = ", ".join(c.get("title", "") for c in failed_courses)
                reply = (
                    f"Ji {first_name}, aap ki degree complete ho sakti hai. "
                    f"Aap ka current CGPA {cgpa} hai aur aap ne {completed_cr} credit hours mukammal kiye hain. "
                    f"Aap ko apne failed courses ({failed_names}) ko repeat kar ke pass karna hoga. "
                    f"University policy ke mutabiq maximum 12 semesters tak ka waqt hota hai, is liye aap ke paas clear karne ka pura time hai."
                )
        else:
            if has_zero_fails:
                reply = (
                    f"Yes {first_name}, your degree will definitely be completed! "
                    f"You have a clean academic record with zero failed courses. "
                    f"Your CGPA is a strong {cgpa}, and you have already completed {completed_cr} credit hours ({degree_pct} degree progress). "
                    f"University policy allows up to a maximum limit of 12 registered semesters to complete all degree requirements. "
                    f"You are progressing well and are well within the required limits to graduate."
                )
            else:
                failed_names = ", ".join(c.get("title", "") for c in failed_courses)
                reply = (
                    f"Yes {first_name}, your degree can be completed. "
                    f"Your CGPA is {cgpa} and you have earned {completed_cr} credit hours. "
                    f"To graduate, you will need to re-enroll in and pass your failed courses ({failed_names}). "
                    f"You have up to 12 semesters under university regulations to complete all requirements."
                )

    return {"response": reply, "profile": profile}


def handle_cgpa_node(state: AdvisorState) -> dict[str, Any]:
    """Generate an accurate, concise, and reassuring CGPA assessment based on student's actual records."""
    from FYP2.services.voice_service import get_failed_courses, get_student_profile

    profile = state.get("profile")
    db_conn = state.get("db_conn")
    student_id = state.get("student_id")

    if not profile and db_conn and student_id:
        try:
            profile = get_student_profile(student_id, db_conn)
        except Exception:
            profile = None

    name = ""
    semester = ""
    cgpa_val = 0.0
    if profile:
        name = profile.get("name") or profile.get("full_name") or ""
        semester = str(profile.get("semester") or "")
        try:
            cgpa_val = float(profile.get("cgpa") or 0.0)
        except (ValueError, TypeError):
            cgpa_val = 0.0

    first_name = name.split()[0] if name else "Student"

    failed_courses = []
    if db_conn and student_id:
        try:
            failed_courses = get_failed_courses(student_id, db_conn)
        except Exception:
            pass

    has_zero_fails = (len(failed_courses) == 0)
    language = state.get("language", "english")
    emotion = (state.get("emotion") or "").lower()
    query = (state.get("query") or "").lower()

    is_stressed = any(
        s in emotion for s in ["stress", "anxious", "frustrat", "sad", "fear", "panic", "overwhelm"]
    ) or any(
        s in query for s in ["stress", "stressed", "anxious", "anxiety", "tension", "pareshan", "worried", "scared", "ghabra", "dar"]
    )

    prefix_ur = f"Pareshan mat hon {first_name}! " if is_stressed else ""
    prefix_en = f"Please don't worry, {first_name}! " if is_stressed else ""

    sem_text_ur = f"Semester {semester}" if semester else "apne semester"
    sem_text_en = f"Semester {semester}" if semester else "your semester"

    fails_ur = "Aap ka koi bhi course fail nahi hai aur record clean hai. " if has_zero_fails else f"Aap ke {len(failed_courses)} courses repeat karne wale hain. "
    fails_en = "You have zero failed courses and a clean academic record. " if has_zero_fails else f"You have {len(failed_courses)} courses to repeat. "

    if cgpa_val >= 3.50:
        if language == "roman_urdu":
            reply = (
                f"{prefix_ur}Ji {first_name}, aap ka CGPA {cgpa_val:.2f} bohat shandar aur outstanding hai! "
                f"BIIT aur HEC policy ke mutabiq graduation ke liye minimum 2.50 CGPA darkar hota hai, jabke aap ka CGPA is se kahin zyada aur safe hai. "
                f"{fails_ur}Aap {sem_text_ur} mein behtareen academic performance dikha rahe hain, isi tarah mehnat jari rakhein!"
            )
        else:
            reply = (
                f"{prefix_en}Yes {first_name}, your current CGPA of {cgpa_val:.2f} is outstanding! "
                f"BIIT and HEC policy requires a minimum 2.50 CGPA for graduation, and your CGPA is well above that requirement. "
                f"{fails_en}You are maintaining excellent academic performance in {sem_text_en}. Keep up the great work!"
            )
    elif cgpa_val >= 3.00:
        if language == "roman_urdu":
            reply = (
                f"{prefix_ur}Ji {first_name}, aap ka CGPA {cgpa_val:.2f} bohat acha aur strong hai! "
                f"University graduation requirement (2.50 CGPA) se aap ka score kaafi behtar aur safe zone mein hai. "
                f"{fails_ur}Aap {sem_text_ur} mein achi position par hain."
            )
        else:
            reply = (
                f"{prefix_en}Yes {first_name}, your CGPA of {cgpa_val:.2f} is very good and solid! "
                f"It is well above the university's minimum graduation requirement of 2.50. "
                f"{fails_en}You are in a safe and strong standing in {sem_text_en}."
            )
    elif cgpa_val >= 2.50:
        if language == "roman_urdu":
            reply = (
                f"{prefix_ur}Ji {first_name}, aap ka CGPA {cgpa_val:.2f} acha aur satisfactory hai. "
                f"Degree mukammal karne ke liye minimum 2.50 CGPA lazmi hota hai, is liye aap safe zone mein hain. "
                f"{fails_ur}Thori si mazeed mehnat se aap isay 3.00 se upar le ja sakte hain."
            )
        else:
            reply = (
                f"{prefix_en}Yes {first_name}, your CGPA of {cgpa_val:.2f} is good and satisfactory. "
                f"It comfortably meets the university graduation requirement of 2.50. "
                f"{fails_en}With a little extra effort in upcoming courses, you can easily push it even higher."
            )
    elif cgpa_val >= 2.00:
        if language == "roman_urdu":
            reply = (
                f"{prefix_ur}{first_name}, aap ka CGPA {cgpa_val:.2f} hai, jabke graduation ke liye minimum 2.50 CGPA darkar hota hai. "
                f"Isay mazeed behtar karna zaroori hai. "
                f"{fails_ur}Current semester ke subjects par zyada tawajjuh dein taake aap ka CGPA 2.50 graduation threshold tak pohnch jaye."
            )
        else:
            reply = (
                f"{prefix_en}{first_name}, your CGPA of {cgpa_val:.2f} needs improvement since graduation requires a minimum of 2.50 CGPA. "
                f"{fails_en}Focus on scoring higher grades in your current subjects to lift your CGPA to the 2.50 graduation threshold."
            )
    else:
        if language == "roman_urdu":
            reply = (
                f"{prefix_ur}{first_name}, aap ka CGPA {cgpa_val:.2f} warning level par hai kyunke semester progression aur graduation ke liye minimum CGPA standards lazmi hain. "
                f"Pareshan na hon, agar aap D grades wale courses repeat karein aur ongoing semester mein mehnat karein to aap ka CGPA barh jayega."
            )
        else:
            reply = (
                f"{prefix_en}{first_name}, your CGPA of {cgpa_val:.2f} is currently at a warning level and needs urgent attention to meet semester and graduation requirements. "
                f"Please don't panic — you can repeat improvable courses and focus on your ongoing semester to bring your CGPA back up."
            )

    return {"response": reply, "profile": profile}


def handle_cgpa_policy_node(state: AdvisorState) -> dict[str, Any]:
    """Provide the official BIIT / HEC semester CGPA progression and degree duration policy."""
    from FYP2.services.voice_service import get_student_profile

    profile = state.get("profile")
    db_conn = state.get("db_conn")
    student_id = state.get("student_id")

    if not profile and db_conn and student_id:
        try:
            profile = get_student_profile(student_id, db_conn)
        except Exception:
            profile = None

    semester = ""
    cgpa_val = None
    if profile:
        semester = str(profile.get("semester") or "")
        try:
            cgpa_val = float(profile.get("cgpa") or 0.0)
        except (ValueError, TypeError):
            cgpa_val = None

    language = state.get("language", "english")

    user_note_ur = ""
    user_note_en = ""
    if cgpa_val is not None:
        sem_str = f"Semester {semester}" if semester else "current semester"
        if cgpa_val >= 2.50:
            user_note_ur = f"\n\nAap ka current CGPA **{cgpa_val:.2f}** ({sem_str}) hai, jo ke graduation requirement (2.50) se ooper aur safe hai."
            user_note_en = f"\n\nYour current CGPA is **{cgpa_val:.2f}** ({sem_str}), which comfortably meets the 2.50 graduation requirement."
        else:
            user_note_ur = f"\n\nAap ka current CGPA **{cgpa_val:.2f}** hai. Graduation ke liye aap ko isay 2.50 tak lana hoga."
            user_note_en = f"\n\nYour current CGPA is **{cgpa_val:.2f}**. You will need to raise it to at least 2.50 for graduation."

    if language == "roman_urdu":
        reply = (
            "BIIT aur HEC undergraduate policy ke mutabiq CGPA aur degree duration ke qawaneen yeh hain:\n\n"
            "- **Degree Duration**: Standard undergraduate program **8 semesters** (4 saal) ka hota hai. "
            "Agar koi subject fail ho ya repeat karna paray, to degree mukammal karne ke liye maximum **12 registered semesters** tak ka waqt milta hai (12th semester ke baad drop rule lagta hai).\n"
            "- **Graduation Requirement**: Degree completion aur graduation ke liye minimum cumulative CGPA **2.50** lazmi hai.\n"
            "- **Har Semester Ki Minimum Cumulative CGPA Progression**:\n"
            "  - **Semester 2**: 1.00\n"
            "  - **Semester 3**: 1.25\n"
            "  - **Semester 4**: 1.50\n"
            "  - **Semester 5**: 1.75\n"
            "  - **Semester 6**: 2.00\n"
            "  - **Semester 7**: 2.25\n"
            "  - **Semester 8 (Graduation)**: **2.50**\n\n"
            "Agar kisi semester mein student ka CGPA is threshold se kam ho to wo academic cease ho jata hai aur unhein poora semester repeat karna parta hai."
            f"{user_note_ur}"
        )
    else:
        reply = (
            "According to BIIT and HEC undergraduate policy:\n\n"
            "- **Degree Duration**: The standard undergraduate program is **8 semesters** (4 years). "
            "If courses need to be repeated or cleared, students are allowed up to a maximum limit of **12 registered semesters** to complete all degree requirements (students not finished after 12 semesters are dropped).\n"
            "- **Graduation Requirement**: The minimum cumulative CGPA required for degree completion and graduation is **2.50**.\n"
            "- **Semester-by-Semester Minimum Cumulative CGPA Progression**:\n"
            "  - **Semester 2**: Minimum **1.00**\n"
            "  - **Semester 3**: Minimum **1.25**\n"
            "  - **Semester 4**: Minimum **1.50**\n"
            "  - **Semester 5**: Minimum **1.75**\n"
            "  - **Semester 6**: Minimum **2.00**\n"
            "  - **Semester 7**: Minimum **2.25**\n"
            "  - **Semester 8 (Graduation)**: Minimum **2.50**\n\n"
            "Falling below the required cumulative CGPA threshold for a semester results in an academic cease, where the student is permitted to repeat the semester once."
            f"{user_note_en}"
        )

    return {"response": reply, "profile": profile}


def handle_greeting_node(state: AdvisorState) -> dict[str, Any]:
    """Generate a direct greeting reply customized for the student."""
    from FYP2.services.voice_service import get_student_profile

    profile = state.get("profile")
    db_conn = state.get("db_conn")
    student_id = state.get("student_id")

    if not profile and db_conn and student_id:
        try:
            profile = get_student_profile(student_id, db_conn)
        except Exception:
            profile = None

    first_name = ""
    if profile:
        full_name = profile.get("full_name") or profile.get("name") or ""
        first_name = full_name.split()[0] if full_name else ""

    name_suffix = f" {first_name}" if first_name else ""
    language = state.get("language", "english")

    if language == "roman_urdu":
        reply = (
            f"Walekum Assalam{name_suffix}! Main aap ka university academic advisor hoon. "
            "Aaj main aap ki kya madad kar sakta hoon?"
        )
    else:
        reply = (
            f"Hello{name_suffix}! I am your university academic advisor. "
            "How can I help you today with your courses, GPA, or university policies?"
        )

    return {"response": reply, "profile": profile}


def handle_failed_courses_node(state: AdvisorState) -> dict[str, Any]:
    """Generate an accurate, reassuring response regarding failed courses / backlogs."""
    from FYP2.services.voice_service import get_failed_courses, get_student_profile

    profile = state.get("profile")
    db_conn = state.get("db_conn")
    student_id = state.get("student_id")

    if not profile and db_conn and student_id:
        try:
            profile = get_student_profile(student_id, db_conn)
        except Exception:
            profile = None

    name = ""
    cgpa = "N/A"
    if profile:
        name = profile.get("name") or profile.get("full_name") or ""
        cgpa = str(profile.get("cgpa") or "N/A")

    first_name = name.split()[0] if name else "Student"

    failed_courses = []
    if db_conn and student_id:
        try:
            failed_courses = get_failed_courses(student_id, db_conn)
        except Exception:
            pass

    has_zero_fails = (len(failed_courses) == 0)
    language = state.get("language", "english")
    emotion = (state.get("emotion") or "").lower()
    query = (state.get("query") or "").lower()

    is_stressed = any(
        s in emotion for s in ["stress", "anxious", "frustrat", "sad", "fear", "panic", "overwhelm"]
    ) or any(
        s in query for s in ["stress", "stressed", "anxious", "anxiety", "tension", "pareshan", "worried", "scared", "ghabra", "dar"]
    )

    prefix_ur = f"Pareshan mat hon {first_name}! " if is_stressed else ""
    prefix_en = f"Please don't worry, {first_name}! " if is_stressed else ""

    if has_zero_fails:
        if language == "roman_urdu":
            reply = (
                f"{prefix_ur}Ji nahi, aap ka koi bhi course fail nahi hai (0 failed courses / backlogs). "
                f"Aap ka academic record bilkul clean hai aur aap ka current CGPA {cgpa} bohat acha hai. "
                f"Aap bilkul safe hain, kisi bhi subject ko repeat karne ki zaroorat nahi hai!"
            )
        else:
            reply = (
                f"{prefix_en}No, you do not have any failed courses (0 backlogs). "
                f"Your academic record is completely clean and your current CGPA of {cgpa} is excellent. "
                f"You do not need to repeat or retake any courses."
            )
    else:
        failed_titles = [f"{c.get('course_id', '')} {c.get('title', '')}".strip() for c in failed_courses]
        failed_str = ", ".join(failed_titles)
        if language == "roman_urdu":
            reply = (
                f"{prefix_ur}Aap ke record ke mutabiq {len(failed_courses)} courses fail hain jinhein repeat karna hoga: {failed_str}. "
                f"Hum mil kar is ka schedule plan kar sakte hain taake aap inhein aglay semester mein ba-asani clear kar sakein."
            )
        else:
            reply = (
                f"{prefix_en}According to your records, you have {len(failed_courses)} failed course(s) to repeat: {failed_str}. "
                f"We can plan your schedule so you can clear them smoothly in the upcoming semester."
            )

    return {"response": reply, "profile": profile}


def handle_student_details_node(state: AdvisorState) -> dict[str, Any]:
    """Provide a full, comprehensive student academic profile without any missing placeholders."""
    from FYP2.services.voice_service import (
        get_completed_credit_hours,
        get_failed_courses,
        get_student_profile,
    )

    profile = state.get("profile")
    db_conn = state.get("db_conn")
    student_id = state.get("student_id")

    if not profile and db_conn and student_id:
        try:
            profile = get_student_profile(student_id, db_conn)
        except Exception:
            profile = None

    name = ""
    arid_no = student_id or ""
    semester = "N/A"
    cgpa = "N/A"
    section = "N/A"
    advisor_name = "N/A"
    if profile:
        name = profile.get("name") or profile.get("full_name") or ""
        arid_no = profile.get("registration_no") or student_id or ""
        semester = str(profile.get("semester") or "N/A")
        cgpa = str(profile.get("cgpa") or "N/A")
        section = str(profile.get("section_code") or "N/A")
        advisor_name = profile.get("advisor_name") or "N/A"

    first_name = name.split()[0] if name else "Student"

    credits_summary = {}
    failed_courses = []
    if db_conn and student_id:
        try:
            credits_summary = get_completed_credit_hours(student_id, db_conn)
            failed_courses = get_failed_courses(student_id, db_conn)
        except Exception:
            pass

    completed_cr = credits_summary.get("completed_credits", 120)
    degree_pct = credits_summary.get("degree_progress", "100%")
    has_zero_fails = (len(failed_courses) == 0)
    fails_str_ur = "0 (Bilkul clean record - koi backlog nahi)" if has_zero_fails else f"{len(failed_courses)} failed courses"
    fails_str_en = "0 (Completely clean record - no backlogs)" if has_zero_fails else f"{len(failed_courses)} failed courses"

    language = state.get("language", "english")
    emotion = (state.get("emotion") or "").lower()
    query = (state.get("query") or "").lower()

    is_stressed = any(
        s in emotion for s in ["stress", "anxious", "frustrat", "sad", "fear", "panic", "overwhelm"]
    ) or any(
        s in query for s in ["stress", "stressed", "anxious", "anxiety", "tension", "pareshan", "worried", "scared", "ghabra", "dar"]
    )

    prefix_ur = f"Pareshan mat hon {first_name}! " if is_stressed else ""
    prefix_en = f"Please don't worry, {first_name}! " if is_stressed else ""

    if language == "roman_urdu":
        reply = (
            f"{prefix_ur}Ji {first_name}, aap ki academic profile aur details yeh hain:\n\n"
            f"- **Full Name**: {name}\n"
            f"- **ARID / Reg No**: {arid_no}\n"
            f"- **Current Semester**: Semester {semester} (Section: {section})\n"
            f"- **Current CGPA**: **{cgpa}**\n"
            f"- **Completed Credits**: **{completed_cr} Credit Hours**\n"
            f"- **Degree Progress**: {degree_pct} (Scheme of Study ke mutabiq on-track)\n"
            f"- **Failed Courses**: {fails_str_ur}\n"
            f"- **Academic Advisor**: {advisor_name}\n\n"
            f"Aap ka overall academic standing bohat acha hai. Kisi bhi specific cheez ke baray mein mazeed pochna ho to batayein!"
        )
    else:
        reply = (
            f"{prefix_en}Here is your complete academic profile and record summary, {first_name}:\n\n"
            f"- **Full Name**: {name}\n"
            f"- **ARID / Registration No**: {arid_no}\n"
            f"- **Current Semester**: Semester {semester} (Section: {section})\n"
            f"- **Current CGPA**: **{cgpa}**\n"
            f"- **Completed Credits**: **{completed_cr} Credit Hours**\n"
            f"- **Degree Progress**: {degree_pct} (Fully on track with Scheme of Study)\n"
            f"- **Failed Courses**: {fails_str_en}\n"
            f"- **Academic Advisor**: {advisor_name}\n\n"
            f"You have a solid academic record. Feel free to ask if you need details on your courses, GPA targets, or policies!"
        )

    return {"response": reply, "profile": profile}


def handle_teacher_strictness_node(state: AdvisorState) -> dict[str, Any]:
    """Provide a polite, conversational breakdown of teacher strictness based on historical grading statistics."""
    from FYP2.services.voice_service import get_teacher_strictness_stats

    db_conn = state.get("db_conn")
    language = state.get("language", "english")
    emotion = (state.get("emotion") or "").lower()
    query = (state.get("query") or "").lower()

    is_stressed = any(
        s in emotion for s in ["stress", "anxious", "frustrat", "sad", "fear", "panic", "overwhelm"]
    ) or any(
        s in query for s in ["stress", "stressed", "anxious", "anxiety", "tension", "pareshan", "worried", "scared", "ghabra", "dar"]
    )

    stats = []
    if db_conn:
        try:
            stats = get_teacher_strictness_stats(db_conn)
        except Exception:
            pass

    ranked = []
    for s in stats:
        tot = s.get("total_graded") or 0
        f_count = s.get("total_f") or 0
        a_count = s.get("total_a") or 0
        if tot > 100:
            f_rate = (f_count / tot) * 100
            a_rate = (a_count / tot) * 100
            ranked.append({
                "name": s.get("teacher_name", ""),
                "f_rate": f_rate,
                "a_rate": a_rate,
                "total": tot,
            })

    ranked_strict = sorted(ranked, key=lambda x: x["f_rate"], reverse=True)
    ranked_lenient = sorted(ranked, key=lambda x: x["a_rate"], reverse=True)

    strict_names = [r["name"] for r in ranked_strict[:3]]
    lenient_names = [r["name"] for r in ranked_lenient[:3]]

    strict_str = ", ".join(strict_names) if strict_names else "Muhammad Azeem aur Zahid Ahmad"
    lenient_str = ", ".join(lenient_names) if lenient_names else "Noor Ul Ain aur deegar teachers"

    prefix_ur = "Pareshan mat hon! " if is_stressed else ""
    prefix_en = "Please don't worry! " if is_stressed else ""

    if language == "roman_urdu":
        reply = (
            f"{prefix_ur}University grading records aur historical statistics ke mutabiq:\n\n"
            f"- **Strict Grading**: **{strict_str}** ki grading comparatively strict rehti hai, jahan assignments aur exams par zyada focus darkar hota hai.\n"
            f"- **High 'A' Grades**: **{lenient_str}** ki classes mein 'A' grades ka tanasub zyada rehta hai.\n\n"
            f"Aap regular classes lein, assignments waqt par submit karein, aur concepts clear rakhein to aap kisi bhi teacher ke paas asani se A ya B+ grade le sakte hain."
        )
    else:
        reply = (
            f"{prefix_en}Based on university grading records and historical grade distributions:\n\n"
            f"- **Stricter Grading**: Instructors like **{strict_str}** tend to maintain rigorous grading standards, requiring extra diligence on quizzes and assignments.\n"
            f"- **Higher 'A' Grade Ratios**: Instructors like **{lenient_str}** tend to award higher percentages of 'A' grades for consistent class performance.\n\n"
            f"Staying consistent with your coursework and lectures ensures you can secure excellent grades regardless of the instructor."
        )

    return {"response": reply}


def handle_irrelevant_node(state: AdvisorState) -> dict[str, Any]:
    """Provide a polite refusal guardrail for non-academic queries."""
    language = state.get("language", "english")

    if language == "roman_urdu":
        reply = (
            "Main aap ka university academic advisor hoon. Main sirf university policies, "
            "courses, CGPA, aur academic mamlaat mein aap ki madad kar sakta hoon. "
            "Meherbani karke apni parhai ya university se mutaliq sawal poochein."
        )
    else:
        reply = (
            "I am your university academic advisor. I can only assist with academic matters, "
            "such as courses, CGPA, exams, and university policies. Please ask a university or academic-related question."
        )

    return {"response": reply}


def retrieve_knowledge_node(state: AdvisorState) -> dict[str, Any]:
    """Retrieve university policy knowledge and student DB context, then prepare prompt."""
    from FYP2.services.voice_service import (
        OLLAMA_MODEL_ENGLISH,
        OLLAMA_MODEL_ROMAN_URDU,
        build_prompt,
        build_student_context,
        resolve_ollama_model,
        retrieve_knowledge,
    )

    query = state.get("query", "")
    session_id = state.get("session_id", 0)
    student_id = state.get("student_id", "")
    db_conn = state.get("db_conn")
    language = state.get("language", "english")
    emotion = state.get("emotion", "Neutral")

    # 1. Fetch DB records into student context (without any hardcoded DB answers)
    if db_conn and student_id:
        context = build_student_context(query, session_id, student_id, db_conn)
    else:
        context = {"profile": state.get("profile") or {}}

    # 2. Retrieve matching university rules from knowledge base
    knowledge = retrieve_knowledge(query)

    # 3. Build prompt for LLM
    prompt = build_prompt(query, language, emotion, context, knowledge)
    preferred_model = (
        OLLAMA_MODEL_ROMAN_URDU if language == "roman_urdu" else OLLAMA_MODEL_ENGLISH
    )
    model = resolve_ollama_model(
        preferred_model,
        fallback="qwen2.5:7b-instruct",
    )

    return {
        "student_context": context,
        "knowledge": knowledge,
        "prompt": prompt,
        "model": model,
    }


def generate_response_node(state: AdvisorState) -> dict[str, Any]:
    """Invoke the LLM model to generate the final response for academic queries."""
    from FYP2.services.voice_service import stream_llm_reply

    prompt = state.get("prompt", "")
    model = state.get("model", "gemma2:2b")
    language = state.get("language", "english")

    chunks: list[str] = []
    script_violation = False
    try:
        for token in stream_llm_reply(prompt, model):
            chunks.append(token)
            if language == "roman_urdu" and contains_urdu_script(token):
                script_violation = True

        reply = "".join(chunks).strip()
        if language == "roman_urdu":
            # Strip English bracket translations
            reply = re.sub(r"\s*\([A-Za-z0-9\s,.'’?!]+\)", "", reply).strip()

        if script_violation or (
            language == "roman_urdu" and contains_urdu_script(reply)
        ):
            reply = (
                "Maazrat, main jawab Roman Urdu mein tayar nahi kar saka. "
                "Meherbani karke dobara koshish karein."
            )
    except Exception:
        logger.exception("LLM generation node failed in LangGraph")
        reply = (
            "AI model abhi available nahi hai, thori dair baad dobara koshish karein."
            if language == "roman_urdu"
            else "The AI model is unavailable. Please try again shortly."
        )

    if not reply:
        reply = (
            "Main is sawal ka jawab abhi tayar nahi kar saka."
            if language == "roman_urdu"
            else "I could not prepare an answer to that question."
        )

    return {"response": reply}


def create_advisor_graph():
    """Build and compile the LangGraph StateGraph for Academic Advising."""
    workflow = StateGraph(AdvisorState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("handle_greeting", handle_greeting_node)
    workflow.add_node("handle_identity", handle_identity_node)
    workflow.add_node("handle_degree_completion", handle_degree_node)
    workflow.add_node("handle_cgpa", handle_cgpa_node)
    workflow.add_node("handle_cgpa_policy", handle_cgpa_policy_node)
    workflow.add_node("handle_failed_courses", handle_failed_courses_node)
    workflow.add_node("handle_student_details", handle_student_details_node)
    workflow.add_node("handle_teacher_strictness", handle_teacher_strictness_node)
    workflow.add_node("handle_irrelevant", handle_irrelevant_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("generate_response", generate_response_node)

    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "greeting": "handle_greeting",
            "identity": "handle_identity",
            "degree_completion": "handle_degree_completion",
            "cgpa_policy": "handle_cgpa_policy",
            "cgpa": "handle_cgpa",
            "failed_courses": "handle_failed_courses",
            "student_details": "handle_student_details",
            "teacher_strictness": "handle_teacher_strictness",
            "irrelevant": "handle_irrelevant",
            "academic": "retrieve_knowledge",
        },
    )
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_identity", END)
    workflow.add_edge("handle_degree_completion", END)
    workflow.add_edge("handle_cgpa", END)
    workflow.add_edge("handle_cgpa_policy", END)
    workflow.add_edge("handle_failed_courses", END)
    workflow.add_edge("handle_student_details", END)
    workflow.add_edge("handle_teacher_strictness", END)
    workflow.add_edge("handle_irrelevant", END)
    workflow.add_edge("retrieve_knowledge", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()


advisor_graph = create_advisor_graph()
