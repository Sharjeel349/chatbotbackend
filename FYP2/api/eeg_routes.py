import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from FYP2.EEGHANDLING import eegsteam

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eeg", tags=["EEG Stream"])


class EEGStreamPayload(BaseModel):
    session_id: Optional[int] = None
    student_id: Optional[str] = None
    samples: List[List[float]]


class EEGStreamResponse(BaseModel):
    status: str = "ok"


@router.post("/stream", response_model=EEGStreamResponse)
def stream_eeg_data(payload: EEGStreamPayload):
    """
    Receives raw EEG microvolt sample buffers streamed from the iOS app,
    processes them via eegstream.py FFT, and writes to EEG_State and EEG_Live_Data.
    Does NOT return mindstate/metrics to iOS (as requested); iOS simply keeps the green icon connected.
    """
    if not payload.samples:
        raise HTTPException(status_code=400, detail="EEG samples cannot be empty")

    try:
        eegsteam.process_and_save_eeg_stream(
            buffer=payload.samples,
            session_id=payload.session_id,
        )
        return EEGStreamResponse(status="ok")
    except Exception as exc:
        logger.exception("Error processing EEG stream: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

