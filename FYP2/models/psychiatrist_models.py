from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class StressedStudentItem(BaseModel):
    student_id: str
    name: str
    email: Optional[str] = None
    semester: str
    section_code: str
    cgpa: Optional[float] = None
    stress_events_count: int = 0
    eeg_stress_count: int = 0
    chat_stress_count: int = 0
    peak_stress_level: str = "Stressed"
    latest_stress_state: str = "Stressed"
    latest_stress_at: Optional[datetime] = None
    latest_stress_time_ago: str = "Recently"
    eeg_sessions_count: int = 0


class PsychiatristDashboardSummary(BaseModel):
    total_stressed_students: int = 0
    severe_stress_students: int = 0
    moderate_stress_students: int = 0
    mild_stress_students: int = 0
    total_stress_events: int = 0
    students: List[StressedStudentItem] = []


class EEGDataPoint(BaseModel):
    eeg_data_id: int
    timestamp: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    alpha_power: float = 0.0
    beta_power: float = 0.0
    theta_power: float = 0.0
    ratio: float = 0.0
    mind_state: str = "Neutral"


class EEGEpochState(BaseModel):
    eeg_state_id: int
    mind_state: str
    timestamp: Optional[datetime] = None


class StudentEEGSession(BaseModel):
    session_id: int
    title: str
    start_time: Optional[datetime] = None
    total_points: int = 0
    stress_points: int = 0
    avg_ratio: float = 0.0
    peak_ratio: float = 0.0
    dominant_state: str = "Neutral"
    data_points: List[EEGDataPoint] = []
    epoch_states: List[EEGEpochState] = []


class StressedMessageItem(BaseModel):
    message_id: int
    session_id: int
    session_title: str
    sender: str
    message_text: str
    emotion: str
    timestamp: Optional[datetime] = None
    time_ago: str = "Recently"
