import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi.concurrency import run_in_threadpool

from FYP2.api import auth_routes, chat_routes, student_routes
from FYP2.services import voice_service


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(voice_service.warm_up_models)
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
app.include_router(chat_routes.router)


@app.get("/")
def read_root():
    return {"message": "Academic API is running. Go to /docs to test endpoints."}

if __name__ == "__main__":
    uvicorn.run("FYP2.main:app", host="0.0.0.0", port=8000, reload=True)
