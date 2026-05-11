from fastapi import APIRouter

router = APIRouter(prefix="/eeg")

latest_state = {}

@router.post("/")
def receive_eeg(data: dict):
    global latest_state
    latest_state = data
    return {"message": "EEG received"}

@router.get("/")
def get_eeg():
    return latest_state