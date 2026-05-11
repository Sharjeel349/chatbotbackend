from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from api import auth_routes, student_routes, chat_routes

app = FastAPI(title="University AI Advisor API")
app.mount("/api/static", StaticFiles(directory="static"), name="static")

# Register routes
app.include_router(auth_routes.router)
app.include_router(student_routes.router)
app.include_router(chat_routes.router)


@app.get("/")
def read_root():
    return {"message": "Academic API is running. Go to /docs to test endpoints."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)