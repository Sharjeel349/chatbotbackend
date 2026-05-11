from typing import List
from FYP.core.database import get_db
from FYP.models import chat_models
from FYP.services import chat_service, voice_service
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException,Form
from FYP.core.database import get_db


router = APIRouter(prefix="/api", tags=["Chat"])

@router.get("/student/{student_id}/chat-sessions", response_model=List[chat_models.ChatSession])
def get_chat_history(student_id: str, db=Depends(get_db)):
    return chat_service.get_chat_sessions(student_id, db)

@router.get("/chat/{session_id}/messages", response_model=List[chat_models.ChatMessage])
def get_chat_messages(session_id: int, db=Depends(get_db)):
    return chat_service.get_chat_messages(session_id, db)


@router.post("/chat-sessions", response_model=chat_models.ChatSession)
def create_new_session(request: chat_models.SessionCreateRequest, db=Depends(get_db)):
    try:
        new_session_data = chat_service.create_chat_session(
            student_id=request.student_id,
            title=request.title,
            db_conn=db
        )

        if not new_session_data:
            raise HTTPException(status_code=500, detail="Failed to create session in database.")

        return new_session_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/{session_id}/messages")
def create_message(session_id: int, request: chat_models.MessageCreateRequest, emotion: str, db=Depends(get_db)):
    try:
        msg = chat_service.save_chat_message(
            session_id=session_id,
            sender=request.sender,
            message_text=request.message_text,
            voice_file=request.voice_file,
            emotion=emotion,
            db_conn=db
        )
        return msg
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/voice")
async def handle_voice_message(
    session_id: int,
    audio_file: UploadFile = File(...),
    emotion: str = Form("NEUTRAL"),
    db=Depends(get_db)
):
    # Basic validation
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio format")

    # Read file into memory
    audio_bytes = await audio_file.read()

    # Pass to our service
    response = voice_service.process_voice_chat(
        session_id=session_id,
        audio_file_bytes=audio_bytes,
        emotion=emotion,
        db_conn=db
    )

    return response