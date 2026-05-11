from fastapi import FastAPI, HTTPException, Depends
from typing import List
import uvicorn

from db import get_connection
import models

app = FastAPI(title="University AI Advisor API")

def get_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


# -------------------------
# Root Endpoint
# -------------------------
@app.get("/")
def read_root():
    return {"message": "Academic API is running. Go to /docs to test endpoints."}


# -------------------------
# 1. Auth: Login
# -------------------------
@app.post("/api/auth/login", response_model=models.AuthResponse)
def login(credentials: models.LoginRequest, db=Depends(get_db)):
    cur = db.cursor()
    try:
        query = """
            SELECT u.user_id, u.name, u.email, u.role, s.sid AS student_id
            FROM Users u
            LEFT JOIN Student s ON u.user_id = s.user_id
            WHERE u.email = %s AND u.password = %s;
        """
        cur.execute(query, (credentials.email, credentials.password))
        row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        return {
            "user_id": row["user_id"],
            "student_id": row["student_id"],
            "name": row["name"],
            "role": row["role"],
            "message": "Login successful"
        }
    finally:
        cur.close()


# -------------------------
# 2. Student Profile
# -------------------------
@app.get("/api/student/{student_id}/profile", response_model=models.ProfileResponse)
def get_profile(student_id: str, db=Depends(get_db)):
    cur = db.cursor()
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
        row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Profile not found")
        return row
    finally:
        cur.close()


# -------------------------
# 3. Chat System
# -------------------------
@app.get("/api/student/{student_id}/chat-sessions", response_model=List[models.ChatSession])
def get_chat_history(student_id: str, db=Depends(get_db)):
    cur = db.cursor()
    try:
        query = """
            SELECT session_id, title, start_time
            FROM Chat_Session
            WHERE student_id = %s
            ORDER BY start_time DESC;
        """
        cur.execute(query, (student_id,))
        return cur.fetchall()
    finally:
        cur.close()


@app.get("/api/chat/{session_id}/messages", response_model=List[models.ChatMessage])
def get_chat_messages(session_id: int, db=Depends(get_db)):
    cur = db.cursor()
    try:
        query = """
            SELECT message_id, sender, message_text, timestamp
            FROM Chat_Message
            WHERE session_id = %s
            ORDER BY timestamp ASC;
        """
        cur.execute(query, (session_id,))
        return cur.fetchall()
    finally:
        cur.close()


# -------------------------
# 4. Academic Reports
# -------------------------
@app.get("/api/student/{student_id}/offered-courses", response_model=List[models.OfferedCourse])
def get_offered_courses(student_id: str, session_name: str = "Spring 2026", db=Depends(get_db)):
    cur = db.cursor()
    try:
        # Uses the new Course_Offering hub logic
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


@app.get("/api/student/{student_id}/dropped-courses", response_model=List[models.DroppedCourse])
def get_dropped_courses(student_id: str, db=Depends(get_db)):
    cur = db.cursor()
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


@app.get("/api/student/{student_id}/transcript", response_model=List[models.TranscriptCourse])
def get_transcript(student_id: str, db=Depends(get_db)):
    cur = db.cursor()
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



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)