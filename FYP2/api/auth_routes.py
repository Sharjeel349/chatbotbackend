from fastapi import APIRouter, HTTPException, Depends
from FYP.core.database import get_db
from FYP.models import auth_models
from FYP.services import auth_service

router = APIRouter(prefix="/api/auth", tags=["Auth"])


@router.post("/login", response_model=auth_models.AuthResponse)
def login(credentials: auth_models.LoginRequest, db=Depends(get_db)):
    row = auth_service.authenticate_user(credentials.email, credentials.password, db)

    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "user_id": row["user_id"],
        "student_id": row["student_id"],
        "name": row["name"],
        "role": row["role"],
        "message": "Login successful"
    }


