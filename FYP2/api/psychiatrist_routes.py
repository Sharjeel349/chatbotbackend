import logging
from typing import List

from fastapi import APIRouter, Depends, Query

from FYP2.core.database import get_db
from FYP2.models import psychiatrist_models
from FYP2.services import psychiatrist_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/psychiatrist", tags=["Psychiatrist & Mental Health"])


@router.get("/dashboard", response_model=psychiatrist_models.PsychiatristDashboardSummary)
def get_psychiatrist_dashboard(db=Depends(get_db)):
    """
    Overview dashboard for the psychiatrist role.
    Returns summary KPIs and ONLY students who have recorded stress events.
    """
    return psychiatrist_service.get_stressed_students_dashboard(db)


@router.get("/stressed-students", response_model=List[psychiatrist_models.StressedStudentItem])
def get_stressed_students(db=Depends(get_db)):
    """List ONLY students with recorded stress in EEG or chat telemetry."""
    summary = psychiatrist_service.get_stressed_students_dashboard(db)
    return summary["students"]


@router.get(
    "/student/{student_id}/eeg-sessions",
    response_model=List[psychiatrist_models.StudentEEGSession],
)
def get_student_eeg_sessions(
    student_id: str,
    max_points: int = Query(180, ge=20, le=1000),
    db=Depends(get_db),
):
    """Fetch a student's EEG sessions with time-series points and 30s epoch states for graphing."""
    return psychiatrist_service.get_student_eeg_sessions(
        student_id=student_id,
        db_conn=db,
        max_points_per_session=max_points,
    )


@router.get(
    "/student/{student_id}/stressed-messages",
    response_model=List[psychiatrist_models.StressedMessageItem],
)
def get_student_stressed_messages(student_id: str, db=Depends(get_db)):
    """Fetch ONLY student-sent chat messages flagged with stress."""
    return psychiatrist_service.get_student_stressed_messages(student_id, db)
