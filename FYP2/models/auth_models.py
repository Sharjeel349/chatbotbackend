from pydantic import BaseModel
from typing import Optional

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    user_id: int
    student_id: Optional[str] = None
    teacher_id: Optional[int] = None
    name: str
    role: str
    message: str