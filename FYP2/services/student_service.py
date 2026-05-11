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

def get_offered_courses(student_id: str, session_name: str, db_conn):
    cur = db_conn.cursor()
    try:
        query = """
            SELECT 
                c.course_id, 
                c.title, 
                co.session, 
                u_teacher.name AS teacher_name, 
                cs.section_code
            FROM Student s
            JOIN Class_Section cs ON s.section_id = cs.section_id
            JOIN Course_Offering co ON co.section_id = cs.section_id
            JOIN Course c ON co.course_id = c.course_id
            JOIN Teacher t ON co.teacher_id = t.teacher_id
            JOIN Users u_teacher ON t.user_id = u_teacher.user_id
            WHERE s.sid = %s AND co.session = %s;
        """
        cur.execute(query, (student_id, session_name))
        return cur.fetchall()
    finally:
        cur.close()

def get_dropped_courses(student_id: str, db_conn):
    cur = db_conn.cursor()
    try:
        query = """
            SELECT 
                c.course_id,
                c.title,
                co.session,
                u_teacher.name AS teacher_name,
                e.grade
            FROM Enrolment e
            JOIN Course_Offering co ON e.offering_id = co.offering_id
            JOIN Course c ON co.course_id = c.course_id
            JOIN Teacher t ON co.teacher_id = t.teacher_id
            JOIN Users u_teacher ON t.user_id = u_teacher.user_id
            WHERE e.grade = 'F' AND e.student_id = %s;
        """
        cur.execute(query, (student_id,))
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
            JOIN Course c ON co.course_id = c.course_id
            WHERE e.student_id = %s AND e.grade IS NOT NULL
            ORDER BY co.session DESC, c.title ASC;
        """
        cur.execute(query, (student_id,))
        return cur.fetchall()
    finally:
        cur.close()