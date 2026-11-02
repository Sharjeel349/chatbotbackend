from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SEMESTER_MIN_CGPA = {
    1: 1.00,
    2: 1.00,
    3: 1.25,
    4: 1.50,
    5: 1.75,
    6: 2.00,
    7: 2.25,
    8: 2.50,
}


def _safe_rollback(db_conn):
    try:
        db_conn.rollback()
    except Exception:
        pass


def _get_semester_number(semester_str: str) -> int:
    """Extract integer semester number from strings like '6th', 'Semester 6', '6'."""
    import re
    m = re.search(r"\d+", str(semester_str))
    return int(m.group()) if m else 1


# ==========================================
# 1. TEACHER PROFILE & ADVISORY OVERVIEW
# ==========================================

def get_teacher_profile(teacher_id: int, db_conn) -> Optional[Dict[str, Any]]:
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT t.teacher_id, t.user_id, u.name, u.email, t.department, t.designation
            FROM Teacher t
            JOIN Users u ON t.user_id = u.user_id
            WHERE t.teacher_id = %s;
            """,
            (teacher_id,),
        )
        teacher = cur.fetchone()
        if not teacher:
            return None

        # Fetch sections this teacher advises
        cur.execute(
            """
            SELECT section_code, semester, discipline
            FROM Class_Section
            WHERE advisor_id = %s
            ORDER BY section_code ASC;
            """,
            (teacher_id,),
        )
        sections = cur.fetchall()
        teacher["advised_sections"] = [f"{s['section_code']} ({s['semester']})" for s in sections]

        # Fetch total advisees count
        cur.execute(
            """
            SELECT COUNT(s.sid) AS total
            FROM Class_Section cs
            JOIN Student s ON cs.section_id = s.section_id
            WHERE cs.advisor_id = %s;
            """,
            (teacher_id,),
        )
        adv_cnt = cur.fetchone()
        teacher["total_advisees_count"] = adv_cnt["total"] if adv_cnt and adv_cnt.get("total") else 0
        return teacher
    finally:
        cur.close()


# ==========================================
# 2. ADVISEES LIST (ALL STUDENTS IN SECTIONS)
# ==========================================

def get_teacher_advisees(teacher_id: int, db_conn) -> List[Dict[str, Any]]:
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT 
                s.sid AS student_id,
                u.name,
                u.email,
                cs.section_code,
                cs.semester,
                s.cgpa,
                COALESCE((
                    SELECT COUNT(DISTINCT co_f.course_id) 
                    FROM Enrolment e_f 
                    JOIN Course_Offering co_f ON e_f.offering_id = co_f.offering_id
                    WHERE e_f.student_id = s.sid AND (e_f.grade = 'F' OR e_f.grade = 'W')
                      AND co_f.course_id NOT IN (
                          SELECT co_p.course_id
                          FROM Enrolment e_p
                          JOIN Course_Offering co_p ON e_p.offering_id = co_p.offering_id
                          WHERE e_p.student_id = s.sid
                            AND e_p.grade IS NOT NULL
                            AND e_p.grade NOT IN ('F', 'W', 'I')
                      )
                ), 0) AS failed_courses_count,
                COALESCE((
                    SELECT SUM(c.credit_hours)
                    FROM Enrolment e
                    JOIN Course_Offering co ON e.offering_id = co.offering_id
                    JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
                    WHERE e.student_id = s.sid AND e.grade IS NOT NULL AND e.grade <> 'F'
                ), 0) AS completed_credits
            FROM Class_Section cs
            JOIN Student s ON cs.section_id = s.section_id
            JOIN Users u ON s.user_id = u.user_id
            WHERE cs.advisor_id = %s
            ORDER BY s.cgpa ASC NULLS FIRST, u.name ASC;
            """,
            (teacher_id,),
        )
        rows = cur.fetchall()

        # Add computed academic status
        results = []
        for r in rows:
            sem_num = _get_semester_number(r.get("semester", "1"))
            min_cgpa = SEMESTER_MIN_CGPA.get(sem_num, 1.50)
            cgpa = float(r["cgpa"]) if r.get("cgpa") is not None else 0.0

            if cgpa < min_cgpa:
                status = "Probation Risk"
            elif r["failed_courses_count"] >= 2:
                status = "Academic Warning"
            else:
                status = "Good Standing"

            r["academic_status"] = status
            r["cgpa"] = cgpa
            results.append(r)

        return results
    finally:
        cur.close()


# ==========================================
# 3. AT-RISK STUDENTS RADAR
# ==========================================

def get_at_risk_students(teacher_id: int, db_conn) -> List[Dict[str, Any]]:
    advisees = get_teacher_advisees(teacher_id, db_conn)
    cur = db_conn.cursor()
    at_risk = []

    try:
        for student in advisees:
            sid = student["student_id"]
            sem_num = _get_semester_number(student["semester"])
            min_cgpa = SEMESTER_MIN_CGPA.get(sem_num, 1.50)
            cgpa = student["cgpa"]

            risk_factors = []
            if cgpa < min_cgpa:
                risk_factors.append(f"CGPA ({cgpa:.2f}) is below the required {min_cgpa:.2f} for Semester {sem_num}")

            if student["failed_courses_count"] > 0:
                risk_factors.append(f"Failed {student['failed_courses_count']} course(s)")

            # Fetch specific failed courses
            cur.execute(
                """
                SELECT DISTINCT c.title 
                FROM Enrolment e
                JOIN Course_Offering co ON e.offering_id = co.offering_id
                JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
                WHERE e.student_id = %s AND (e.grade = 'F' OR e.grade = 'W')
                  AND co.course_id NOT IN (
                      SELECT co2.course_id
                      FROM Enrolment e2
                      JOIN Course_Offering co2 ON e2.offering_id = co2.offering_id
                      WHERE e2.student_id = %s
                        AND e2.grade IS NOT NULL
                        AND e2.grade NOT IN ('F', 'W', 'I')
                  );
                """,
                (sid, sid),
            )
            failed_titles = [row["title"] for row in cur.fetchall()]

            if risk_factors:
                recommended_action = (
                    "Schedule mandatory 1-on-1 counseling and repeat failed prerequisite courses."
                    if student["failed_courses_count"] >= 2
                    else "Recommend academic coaching and workload balance."
                )

                at_risk.append({
                    "student_id": sid,
                    "name": student["name"],
                    "section_code": student["section_code"],
                    "semester": student["semester"],
                    "cgpa": cgpa,
                    "min_required_cgpa": min_cgpa,
                    "risk_factors": risk_factors,
                    "failed_courses": failed_titles,
                    "recommended_action": recommended_action,
                })

        return at_risk
    finally:
        cur.close()


# ==========================================
# 4. EEG & COGNITIVE STATE WELLBEING
# ==========================================

def get_eeg_cognitive_analytics(teacher_id: int, db_conn) -> Dict[str, Any]:
    cur = db_conn.cursor()
    try:
        # 1. Real EEG_State counts from DB
        cur.execute(
            """
            SELECT mind_state AS state, COUNT(*) as count
            FROM EEG_State
            GROUP BY mind_state;
            """
        )
        counts = cur.fetchall()

        # 2. Real chat_message emotion counts from DB
        cur.execute(
            """
            SELECT emotion AS state, COUNT(*) as count
            FROM chat_message
            WHERE emotion IS NOT NULL AND emotion <> ''
            GROUP BY emotion;
            """
        )
        msg_counts = cur.fetchall()

        raw_map = {}
        for c in counts:
            st = str(c["state"]).strip().upper()
            raw_map[st] = raw_map.get(st, 0) + c["count"]
        for mc in msg_counts:
            st = str(mc["state"]).strip().upper()
            raw_map[st] = raw_map.get(st, 0) + mc["count"]

        focused_cnt = raw_map.get("CALM", 0) + raw_map.get("FOCUSED", 0) + raw_map.get("ATTENTIVE", 0)
        neutral_cnt = raw_map.get("NEUTRAL", 0)
        confused_cnt = raw_map.get("CONFUSED", 0)
        stressed_cnt = raw_map.get("STRESSED", 0)

        total_normalized = focused_cnt + neutral_cnt + confused_cnt + stressed_cnt or 1
        percentages = {
            "Focused": round((focused_cnt / total_normalized) * 100, 1),
            "Neutral": round((neutral_cnt / total_normalized) * 100, 1),
            "Confused": round((confused_cnt / total_normalized) * 100, 1),
            "Stressed": round((stressed_cnt / total_normalized) * 100, 1),
        }
        dominant_state = max(percentages, key=percentages.get)

        # 3. Fetch real recent student telemetry records from DB
        cur.execute(
            """
            SELECT 
                u.name AS student_name,
                s.sid AS student_sid,
                COALESCE(cs.title, 'Academic Advising') AS course,
                cm.timestamp,
                COALESCE(cm.emotion, 'Neutral') AS state
            FROM chat_message cm
            JOIN chat_session cs ON cm.session_id = cs.session_id
            JOIN student s ON cs.student_id = s.sid
            JOIN users u ON s.user_id = u.user_id
            ORDER BY cm.timestamp DESC
            LIMIT 10;
            """
        )
        msg_rows = cur.fetchall()

        recent_records = []
        for r in msg_rows:
            st_clean = r["state"].capitalize()
            if st_clean.upper() == "CALM":
                st_clean = "Focused"
            intensity = 85 if st_clean == "Focused" else (75 if st_clean == "Stressed" else 65)
            
            diff = datetime.now() - r["timestamp"] if r["timestamp"] else None
            if diff:
                if diff.days == 0:
                    mins = int(diff.seconds / 60)
                    time_ago = f"{mins} mins ago" if mins < 60 else f"{int(mins/60)} hrs ago"
                elif diff.days == 1:
                    time_ago = "Yesterday"
                else:
                    time_ago = f"{diff.days} days ago"
            else:
                time_ago = "Recently"

            recent_records.append({
                "student_name": r["student_name"],
                "student_sid": r["student_sid"],
                "course": r["course"],
                "time_ago": time_ago,
                "state": st_clean,
                "intensity": intensity
            })

        insights = []
        if stressed_cnt > 0:
            insights.append(f"{stressed_cnt} recent sessions flagged with elevated stress/anxiety markers in academic chat telemetry.")
        if focused_cnt > 0:
            insights.append(f"{focused_cnt} learning sessions recorded in high focus and calm state.")
        if not insights:
            insights.append("Student cognitive engagement is currently stable and balanced across cohorts.")

        return {
            "total_evaluations": total_normalized,
            "dominant_state": dominant_state,
            "mind_state_percentages": percentages,
            "recent_eeg_records": recent_records,
            "insights": insights,
        }
    except Exception:
        _safe_rollback(db_conn)
        return {
            "total_evaluations": 0,
            "dominant_state": "Neutral",
            "mind_state_percentages": {"Neutral": 100.0},
            "recent_eeg_records": [],
            "insights": ["Cognitive telemetry is awaiting sensor feed."],
        }
    finally:
        cur.close()


# ==========================================
# 5. COURSES TAUGHT BY TEACHER (INSTRUCTOR)
# ==========================================

def get_teacher_courses(teacher_id: int, db_conn) -> List[Dict[str, Any]]:
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT 
                co.offering_id,
                c.course_id,
                c.title AS course_title,
                c.credit_hours,
                co.session,
                cs.section_code,
                COUNT(e.enrolment_id) AS enrolled_count
            FROM Course_Offering co
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            JOIN Class_Section cs ON co.section_id = cs.section_id
            LEFT JOIN Enrolment e ON co.offering_id = e.offering_id
            WHERE co.teacher_id = %s
            GROUP BY co.offering_id, c.course_id, c.title, c.credit_hours, co.session, cs.section_code
            ORDER BY co.session DESC, c.title ASC;
            """,
            (teacher_id,),
        )
        offerings = cur.fetchall()

        # Compute grade distribution per offering
        for off in offerings:
            cur.execute(
                """
                SELECT grade, COUNT(*) AS count
                FROM Enrolment
                WHERE offering_id = %s AND grade IS NOT NULL
                GROUP BY grade;
                """,
                (off["offering_id"],),
            )
            dist_rows = cur.fetchall()
            off["grade_distribution"] = {r["grade"]: r["count"] for r in dist_rows}

        return offerings
    finally:
        cur.close()


# ==========================================
# 6. STUDENT 360° BRIEF FOR ADVISOR
# ==========================================

def get_student_academic_brief(student_id: str, db_conn) -> Optional[Dict[str, Any]]:
    from FYP2.services.voice_service import (
        get_current_enrollments,
        get_failed_courses,
        get_student_profile,
        get_completed_credit_hours,
        check_student_prerequisite_status,
    )

    profile = get_student_profile(student_id, db_conn)
    if not profile:
        return None

    current_courses = get_current_enrollments(student_id, db_conn)
    failed = get_failed_courses(student_id, db_conn)
    credits_info = get_completed_credit_hours(student_id, db_conn)
    prereqs = check_student_prerequisite_status(student_id, db_conn)

    # Ineligible prerequisites
    blocked = [p for p in prereqs if p.get("eligibility_status") == "Not Eligible"]

    # Generate heuristic executive summary
    cgpa_val = float(profile["cgpa"]) if profile.get("cgpa") is not None else 0.0
    failed_names = ", ".join([f["title"] for f in failed]) if failed else "None"
    
    summary = (
        f"Student {profile['name']} ({profile['registration_no']}) is in semester {profile['semester']} "
        f"with a CGPA of {cgpa_val:.2f}. Total completed credits: {credits_info.get('completed_credits', 0)}. "
        f"Failed courses to repeat: {failed_names}. Currently taking {len(current_courses)} courses."
    )

    return {
        "student_id": profile["registration_no"],
        "name": profile["name"],
        "cgpa": cgpa_val,
        "semester": profile["semester"],
        "section_code": profile["section_code"],
        "completed_credits": credits_info.get("completed_credits", 0),
        "current_enrolled_courses": current_courses,
        "failed_courses": failed,
        "prerequisite_blockers": blocked,
        "ai_advisor_summary": summary,
    }


# ==========================================
# 7. ADVISORY NOTES
# ==========================================

def ensure_notes_table_exists(db_conn):
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS Advisor_Note (
                note_id SERIAL PRIMARY KEY,
                teacher_id INT NOT NULL,
                student_id VARCHAR(20) NOT NULL,
                note_text TEXT NOT NULL,
                action_plan TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        db_conn.commit()
    except Exception:
        _safe_rollback(db_conn)
    finally:
        cur.close()


def save_advisory_note(teacher_id: int, student_id: str, note_text: str, action_plan: Optional[str], db_conn) -> Dict[str, Any]:
    ensure_notes_table_exists(db_conn)
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO Advisor_Note (teacher_id, student_id, note_text, action_plan)
            VALUES (%s, %s, %s, %s)
            RETURNING note_id, teacher_id, student_id, note_text, action_plan, created_at;
            """,
            (teacher_id, student_id, note_text, action_plan),
        )
        note = cur.fetchone()
        db_conn.commit()

        # Automatically trigger an in-app notification for the student
        try:
            from FYP2.services import notification_service
            note_summary = note_text[:120] + "..." if len(note_text) > 120 else note_text
            msg = f"Your Academic Advisor logged a note: \"{note_summary}\""
            if action_plan:
                msg += f"\nAction Plan: {action_plan}"
            notification_service.create_notification(
                student_id=student_id,
                teacher_id=teacher_id,
                title="Advisor Logged Counseling Note",
                message=msg,
                notification_type="note",
                meeting_time=None,
                location=None,
                db_conn=db_conn,
            )
        except Exception as notif_err:
            logger.warning(f"Could not dispatch advising note notification: {notif_err}")

        return note
    finally:
        cur.close()


def get_advisory_notes(student_id: str, db_conn) -> List[Dict[str, Any]]:
    ensure_notes_table_exists(db_conn)
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT an.note_id, an.teacher_id, an.student_id, u.name AS teacher_name,
                   an.note_text, an.action_plan, an.created_at
            FROM Advisor_Note an
            JOIN Teacher t ON an.teacher_id = t.teacher_id
            JOIN Users u ON t.user_id = u.user_id
            WHERE an.student_id = %s
            ORDER BY an.created_at DESC;
            """,
            (student_id,),
        )
        return cur.fetchall()
    finally:
        cur.close()
