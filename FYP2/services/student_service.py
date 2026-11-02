from typing import Optional

def get_student_profile(student_id: str, db_conn):
    cur = db_conn.cursor()
    try:
        query = """
            SELECT 
                u.name, 
                s.sid AS registration_no, 
                s.cgpa,
                cs.semester, 
                cs.section_code,
                tu.name AS advisor_name, 
                t.designation AS advisor_designation
            FROM Student s
            JOIN Users u ON s.user_id = u.user_id
            JOIN Class_Section cs ON s.section_id = cs.section_id
            LEFT JOIN Teacher t ON cs.advisor_id = t.teacher_id
            LEFT JOIN Users tu ON t.user_id = tu.user_id
            WHERE s.sid = %s;
        """
        cur.execute(query, (student_id,))
        return cur.fetchone()
    finally:
        cur.close()

def get_offered_courses(student_id: str, session_name: Optional[str], db_conn):
    cur = db_conn.cursor()
    try:
        # If an explicit session_name is provided (and not the outdated 'Fall 2024' default)
        if session_name and session_name != "Fall 2024":
            query = """
                SELECT DISTINCT ON (c.course_id)
                    c.course_id, 
                    c.title, 
                    co.session, 
                    COALESCE(u_teacher.name, 'Faculty') AS teacher_name, 
                    cs.section_code
                FROM Course_Offering co
                JOIN Class_Section cs ON co.section_id = cs.section_id
                JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
                LEFT JOIN Teacher t ON co.teacher_id = t.teacher_id
                LEFT JOIN Users u_teacher ON t.user_id = u_teacher.user_id
                WHERE co.session = %s
                  AND co.section_id IN (
                      SELECT section_id FROM Student WHERE sid = %s
                  )
                ORDER BY c.course_id;
            """
            cur.execute(query, (session_name, student_id))
            results = cur.fetchall()
            if results:
                return results

        # 1. Primary: Show active/ongoing courses the student is currently enrolled in (grade is NULL)
        query = """
            SELECT DISTINCT ON (c.course_id)
                c.course_id,
                c.title,
                co.session,
                COALESCE(u_teacher.name, 'Faculty') AS teacher_name,
                cs.section_code
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Class_Section cs ON co.section_id = cs.section_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            LEFT JOIN Teacher t ON co.teacher_id = t.teacher_id
            LEFT JOIN Users u_teacher ON t.user_id = u_teacher.user_id
            WHERE e.student_id = %s AND e.grade IS NULL
            ORDER BY c.course_id, co.session DESC;
        """
        cur.execute(query, (student_id,))
        enrolled_current = cur.fetchall()
        if enrolled_current:
            return enrolled_current

        # 2. Fallback: Find courses offered for the student's section in the latest session available
        query = """
            SELECT DISTINCT ON (c.course_id)
                c.course_id, 
                c.title, 
                co.session, 
                COALESCE(u_teacher.name, 'Faculty') AS teacher_name, 
                cs.section_code
            FROM Student s
            JOIN Class_Section cs ON s.section_id = cs.section_id
            JOIN Course_Offering co ON co.section_id = cs.section_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            LEFT JOIN Teacher t ON co.teacher_id = t.teacher_id
            LEFT JOIN Users u_teacher ON t.user_id = u_teacher.user_id
            WHERE s.sid = %s
              AND co.session = (
                  SELECT session FROM Course_Offering ORDER BY session DESC LIMIT 1
              )
            ORDER BY c.course_id;
        """
        cur.execute(query, (student_id,))
        return cur.fetchall()
    finally:
        cur.close()

def get_dropped_courses(student_id: str, db_conn):
    cur = db_conn.cursor()
    try:
        query = """
            SELECT DISTINCT ON (c.course_id)
                c.course_id,
                c.title,
                co.session,
                COALESCE(u_teacher.name, 'Faculty') AS teacher_name,
                e.grade
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            LEFT JOIN Teacher t ON co.teacher_id = t.teacher_id
            LEFT JOIN Users u_teacher ON t.user_id = u_teacher.user_id
            WHERE e.student_id = %s AND (e.grade = 'F' OR e.grade = 'W')
              AND c.course_id NOT IN (
                  -- Exclude courses the student has passed / cleared in any subsequent (or other) semester
                  SELECT co2.course_id
                  FROM Enrolment e2
                  JOIN Course_Offering co2 ON e2.offering_id = co2.offering_id
                  WHERE e2.student_id = %s
                    AND e2.grade IS NOT NULL
                    AND e2.grade NOT IN ('F', 'W', 'I')
              )
            ORDER BY c.course_id, co.session DESC;
        """
        cur.execute(query, (student_id, student_id))
        return cur.fetchall()
    finally:
        cur.close()

def get_transcript(student_id: str, db_conn):
    cur = db_conn.cursor()
    try:
        query = """
            SELECT 
                co.session,
                c.course_id,
                c.title, 
                e.grade, 
                c.credit_hours
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id AND co.sos = c.sos
            WHERE e.student_id = %s AND e.grade IS NOT NULL
            ORDER BY co.session DESC, c.title ASC;
        """
        cur.execute(query, (student_id,))
        return cur.fetchall()
    finally:
        cur.close()


def update_student_voice_preference(student_id: str, preferred_voice: str, db_conn) -> dict:
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            UPDATE Student
            SET preferred_voice = %s
            WHERE sid = %s OR CAST(user_id AS TEXT) = %s;
            """,
            (preferred_voice, str(student_id), str(student_id)),
        )
        db_conn.commit()
        return {
            "status": "success",
            "student_id": student_id,
            "preferred_voice": preferred_voice,
            "message": "Voice preference updated successfully.",
        }
    except Exception as exc:
        try:
            db_conn.rollback()
        except Exception:
            pass
        err_msg = str(exc)
        if "preferred_voice" in err_msg and "does not exist" in err_msg:
            raise RuntimeError(
                "Column 'preferred_voice' not found in table 'Student'. "
                "Please run: ALTER TABLE Student ADD COLUMN preferred_voice VARCHAR(100) DEFAULT 'Lekha';"
            ) from exc
        raise
    finally:
        cur.close()


def get_student_voice_preference(student_id: str, db_conn) -> str:
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT preferred_voice
            FROM Student
            WHERE sid = %s OR CAST(user_id AS TEXT) = %s;
            """,
            (str(student_id), str(student_id)),
        )
        row = cur.fetchone()
        return row.get("preferred_voice", "Lekha") if row and row.get("preferred_voice") else "Lekha"
    except Exception:
        try:
            db_conn.rollback()
        except Exception:
            pass
        return "Lekha"
    finally:
        cur.close()