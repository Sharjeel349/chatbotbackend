from fastapi import APIRouter, HTTPException, Depends
from typing import List
from FYP2.core.database import get_db
from FYP2.models import academic_models
from FYP2.services import student_service

router = APIRouter(prefix="/api/student", tags=["Student"])

@router.get("/{student_id}/profile", response_model=academic_models.ProfileResponse)
def get_profile(student_id: str, db=Depends(get_db)):
    row = student_service.get_student_profile(student_id, db)
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")
    return row

@router.get("/{student_id}/offered-courses", response_model=List[academic_models.OfferedCourse])
def get_offered_courses(student_id: str, session_name: str = "Fall 2024", db=Depends(get_db)):
    return student_service.get_offered_courses(student_id, session_name, db)

@router.get("/{student_id}/dropped-courses", response_model=List[academic_models.DroppedCourse])
def get_dropped_courses(student_id: str, db=Depends(get_db)):
    return student_service.get_dropped_courses(student_id, db)

@router.get("/{student_id}/transcript", response_model=List[academic_models.TranscriptCourse])
def get_transcript(student_id: str, db=Depends(get_db)):
    return student_service.get_transcript(student_id, db)