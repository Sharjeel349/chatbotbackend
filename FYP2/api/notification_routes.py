import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from FYP2.core.database import get_db
from FYP2.models import notification_models
from FYP2.services import notification_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


@router.post("", response_model=notification_models.NotificationItem)
def create_notification(
    payload: notification_models.NotificationCreateRequest,
    db=Depends(get_db),
):
    """
    Send an in-app notification to a specific student (by student_id)
    or broadcast to all students (student_id is null / omitted).
    Notification types: 'meeting', 'note', 'broadcast', 'general'.
    """
    try:
        created = notification_service.create_notification(
            student_id=payload.student_id,
            teacher_id=payload.teacher_id,
            teacher_name=payload.teacher_name,
            title=payload.title,
            message=payload.message,
            notification_type=payload.notification_type,
            meeting_time=payload.meeting_time,
            location=payload.location,
            db_conn=db,
        )
        return created
    except Exception as e:
        logger.error(f"Error creating notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to create notification")


@router.get("/student/{student_id}", response_model=List[notification_models.NotificationItem])
def get_student_notifications(
    student_id: str,
    unread_only: bool = Query(True, description="When True, only returns unread notifications"),
    db=Depends(get_db),
):
    """
    Retrieve notifications for a student (including broadcasts).
    By default, only unread notifications are returned.
    """
    return notification_service.get_student_notifications(
        student_id=student_id,
        unread_only=unread_only,
        db_conn=db,
    )


@router.get("/student/{student_id}/unread-count", response_model=notification_models.UnreadCountResponse)
def get_unread_count(student_id: str, db=Depends(get_db)):
    """
    Get the number of unread notifications for a student.
    Used for the notification bell badge dot.
    """
    count = notification_service.get_student_unread_count(student_id, db)
    return {"unread_count": count}


@router.post("/student/{student_id}/read/{notification_id}", response_model=notification_models.NotificationActionResponse)
def mark_notification_read(student_id: str, notification_id: int, db=Depends(get_db)):
    """
    Mark a notification as read for a specific student.
    Once read, it will not appear in the unread notification feed.
    """
    success = notification_service.mark_notification_as_read(student_id, notification_id, db)
    if not success:
        raise HTTPException(status_code=400, detail="Could not mark notification as read")
    return {"success": True, "message": "Notification marked as read"}


@router.post("/student/{student_id}/read-all", response_model=notification_models.NotificationActionResponse)
def mark_all_notifications_read(student_id: str, db=Depends(get_db)):
    """
    Mark all pending notifications as read for this student.
    """
    count = notification_service.mark_all_as_read(student_id, db)
    return {"success": True, "message": f"{count} notifications marked as read"}
