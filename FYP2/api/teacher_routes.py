import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from FYP2.core.database import get_db
from FYP2.models import teacher_models
from FYP2.services import teacher_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/teacher", tags=["Teacher & Academic Advisor"])


@router.get("/{teacher_id}/profile", response_model=teacher_models.TeacherProfile)
def get_teacher_profile(teacher_id: int, db=Depends(get_db)):
    """Fetch teacher profile along with assigned advisory class sections."""
    profile = teacher_service.get_teacher_profile(teacher_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Teacher not found")
    return profile


@router.get("/{teacher_id}/advisees", response_model=List[teacher_models.AdviseeItem])
def get_teacher_advisees(teacher_id: int, db=Depends(get_db)):
    """List all students across class sections advised by this teacher, sorted by academic standing."""
    return teacher_service.get_teacher_advisees(teacher_id, db)


@router.get("/{teacher_id}/at-risk-students", response_model=List[teacher_models.AtRiskStudentItem])
def get_at_risk_students(teacher_id: int, db=Depends(get_db)):
    """
    Early warning radar: returns students with CGPA below the university threshold
    or with multiple course failures.
    """
    return teacher_service.get_at_risk_students(teacher_id, db)


@router.get("/{teacher_id}/cognitive-analytics", response_model=teacher_models.CognitiveAnalyticsResponse)
def get_cognitive_analytics(teacher_id: int, db=Depends(get_db)):
    """
    Retrieve real-time and aggregated EEG mind-state analytics (Attentive, Confused, Stressed, Neutral).
    """
    return teacher_service.get_eeg_cognitive_analytics(teacher_id, db)


@router.get("/{teacher_id}/courses", response_model=List[teacher_models.TeacherCourseOffering])
def get_teacher_courses(teacher_id: int, db=Depends(get_db)):
    """List courses taught by this teacher with enrolment numbers and grade distributions."""
    return teacher_service.get_teacher_courses(teacher_id, db)


@router.get("/student/{student_id}/brief", response_model=teacher_models.StudentAcademicBrief)
def get_student_academic_brief(student_id: str, db=Depends(get_db)):
    """360-degree academic executive brief for a student with AI-generated advisor summary."""
    brief = teacher_service.get_student_academic_brief(student_id, db)
    if not brief:
        raise HTTPException(status_code=404, detail="Student not found")
    return brief


@router.post("/advisory-notes", response_model=teacher_models.AdvisoryNoteResponse)
def create_advisory_note(note: teacher_models.AdvisoryNoteCreate, db=Depends(get_db)):
    """Save an advisor 1-on-1 counseling note and action plan."""
    saved = teacher_service.save_advisory_note(
        teacher_id=note.teacher_id,
        student_id=note.student_id,
        note_text=note.note_text,
        action_plan=note.action_plan,
        db_conn=db,
    )
    return saved


@router.get("/student/{student_id}/advisory-notes", response_model=List[teacher_models.AdvisoryNoteResponse])
def get_student_advisory_notes(student_id: str, db=Depends(get_db)):
    """Fetch history of counseling notes recorded for this student."""
    return teacher_service.get_advisory_notes(student_id, db)
