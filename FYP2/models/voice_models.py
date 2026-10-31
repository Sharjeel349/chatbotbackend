from typing import List, Optional
from pydantic import BaseModel


class VoiceItem(BaseModel):
    id: str
    name: str
    display_name: str
    language: str
    locale: str
    gender: str
    is_recommended: bool
    category: str
    sample_url: str


class VoiceListResponse(BaseModel):
    status: str = "success"
    total: int
    default_voice: str
    default_sample_text: str
    recommended_voices: List[VoiceItem]
    all_voices: List[VoiceItem]


class VoiceTestRequest(BaseModel):
    voice: str
    text: Optional[str] = None


class VoiceTestResponse(BaseModel):
    status: str = "success"
    voice: str
    text: str
    audio_url: str
    duration_ms: Optional[int] = None
