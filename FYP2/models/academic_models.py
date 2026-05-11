from pydantic import BaseModel
from typing import Optional

class ProfileResponse(BaseModel):
    name: str
    registration_no: str
    cgpa: Optional[float]
    semester: str
    section_code: str
    advisor_name: Optional[str]
    advisor_designation: Optional[str]

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