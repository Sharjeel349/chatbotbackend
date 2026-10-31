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

class TextChatRequest(BaseModel):
    message: Optional[str] = None
    text: Optional[str] = None
    message_text: Optional[str] = None
    student_id: Optional[str] = None
    emotion: Optional[str] = "Neutral"
    generate_audio: Optional[bool] = False

class TextChatResponse(BaseModel):
    user_text: str
    bot_text: str
    user_audio_url: Optional[str] = None
    bot_audio_url: Optional[str] = None
    user_emotion: str
    detected_language: Optional[str] = "en"
    response_language: Optional[str] = "english"
    timing_ms: Optional[dict] = None
