import logging
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Ensure the root project directory is on sys.path so FYP2 can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import uvicorn

from FYP2.api import auth_routes, chat_routes, notification_routes, student_routes, teacher_routes, voice_routes
from FYP2.services import tts_voice_service, voice_service


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(voice_service.warm_up_models)
    await run_in_threadpool(tts_voice_service.prewarm_default_voices)
    yield


app = FastAPI(title="University AI Advisor API", lifespan=lifespan)
app.mount(
    "/api/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

# Register routes
app.include_router(auth_routes.router)
app.include_router(student_routes.router)
app.include_router(teacher_routes.router)
app.include_router(chat_routes.router)
app.include_router(voice_routes.router)
app.include_router(notification_routes.router)


@app.get("/")
def read_root():
    return {"message": "Academic API is running. Go to /docs to test endpoints."}

if __name__ == "__main__":
    uvicorn.run("FYP2.main:app", host="0.0.0.0", port=8000, reload=True)
