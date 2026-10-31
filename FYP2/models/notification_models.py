from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class NotificationCreateRequest(BaseModel):
    student_id: Optional[str] = None  # None / null means broadcast to ALL students
    teacher_id: Optional[int] = None
    teacher_name: Optional[str] = None
    title: str
    message: str
    notification_type: str = "general"  # 'meeting', 'note', 'broadcast', 'general'
    meeting_time: Optional[str] = None
    location: Optional[str] = None


class NotificationItem(BaseModel):
    notification_id: int
    student_id: Optional[str] = None
    teacher_id: Optional[int] = None
    teacher_name: Optional[str] = None
    title: str
    message: str
    notification_type: str
    meeting_time: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime
    is_read: bool = False


class UnreadCountResponse(BaseModel):
    unread_count: int


class NotificationActionResponse(BaseModel):
    success: bool
    message: str
