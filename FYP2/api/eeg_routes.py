import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from FYP2.EEGHANDLING import eegstream, updated_eeg_schema

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
    processes them via updated_eeg_schema.py FFT (using the updated 6-state logic),
    writes continuous points to eeg_live_data, and saves 30-second aggregated
    emotions to eeg_state.
    Does NOT return mindstate/metrics to iOS (as requested); iOS simply keeps the green icon connected.
    """
    if not payload.samples:
        raise HTTPException(status_code=400, detail="EEG samples cannot be empty")

    try:
        updated_eeg_schema.process_and_save_eeg_stream(
            buffer=payload.samples,
            session_id=payload.session_id,
        )
        return EEGStreamResponse(status="ok")
    except Exception as exc:
        logger.exception("Error processing EEG stream: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

