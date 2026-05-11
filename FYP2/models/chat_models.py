from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ChatSession(BaseModel):
    session_id: int
    title: Optional[str]
    start_time: Optional[datetime]

class ChatMessage(BaseModel):
    message_id: int
    sender: str
    message_text: str
    timestamp: Optional[datetime]
    emotion: str
    voice_file: Optional[str]

class SessionCreateRequest(BaseModel):
    student_id: str
    title: Optional[str] = "New Advisor Session"

class MessageCreateRequest(BaseModel):
    sender: str
    message_text: str
    voice_file: Optional[str] = None
    emotion: str