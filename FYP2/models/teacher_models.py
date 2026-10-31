from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TeacherProfile(BaseModel):
    teacher_id: int
    user_id: int
    name: str
    email: str
    department: str
    designation: str
    advised_sections: List[str] = []
    total_advisees_count: Optional[int] = 0


class AdviseeItem(BaseModel):
    student_id: str
    name: str
    email: Optional[str] = None
    semester: str
    section_code: str
    cgpa: Optional[float] = None
    academic_status: str  # "Good Standing", "Academic Warning", "Probation Risk"
    failed_courses_count: int = 0
    completed_credits: int = 0
    latest_emotion: Optional[str] = "Neutral"


class AtRiskStudentItem(BaseModel):
    student_id: str
    name: str
    section_code: str
    semester: str
    cgpa: Optional[float] = None
    min_required_cgpa: float
    risk_factors: List[str]
    failed_courses: List[str] = []
    recommended_action: str


class TeacherCourseOffering(BaseModel):
    offering_id: int
    course_id: str
    course_title: str
    credit_hours: int
    session: str
    section_code: str
    enrolled_count: int
    grade_distribution: Dict[str, int] = {}


class CognitiveAnalyticsResponse(BaseModel):
    total_evaluations: int
    dominant_state: str
    mind_state_percentages: Dict[str, float]
    recent_eeg_records: List[Dict[str, Any]]
    insights: List[str]


class StudentAcademicBrief(BaseModel):
    student_id: str
    name: str
    cgpa: Optional[float]
    semester: str
    section_code: str
    completed_credits: int
    current_enrolled_courses: List[Dict[str, Any]]
    failed_courses: List[Dict[str, Any]]
    prerequisite_blockers: List[Dict[str, Any]]
    ai_advisor_summary: str


class AdvisoryNoteCreate(BaseModel):
    teacher_id: int
    student_id: str
    note_text: str
    action_plan: Optional[str] = None


class AdvisoryNoteResponse(BaseModel):
    note_id: int
    teacher_id: int
    student_id: str
    teacher_name: Optional[str] = None
    note_text: str
    action_plan: Optional[str] = None
    created_at: Optional[datetime] = None


class VoicePreferenceUpdateRequest(BaseModel):
    preferred_voice: str
