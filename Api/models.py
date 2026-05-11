from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# --- Auth & Profile Models ---
class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    user_id: int
    student_id: Optional[str]
    name: str
    role: str
    message: str

class ProfileResponse(BaseModel):
    name: str
    registration_no: str
    cgpa: Optional[float]
    semester: str
    section_code: str
    advisor_name: Optional[str]
    advisor_designation: Optional[str]

# --- Chat Models ---
class ChatSession(BaseModel):
    session_id: int
    title: Optional[str]
    start_time: Optional[datetime]

class ChatMessage(BaseModel):
    message_id: int
    sender: str
    message_text: str
    timestamp: Optional[datetime]

# --- Academic Models ---
class OfferedCourse(BaseModel):
    course_id: str
    title: str
    session: str
    teacher_name: str
    section_code: str

class DroppedCourse(BaseModel):
    course_id: str
    title: str
    session: str
    teacher_name: str
    grade: str

class TranscriptCourse(BaseModel):
    session: str
    course_id: str
    title: str
    grade: Optional[str]
    credit_hours: int