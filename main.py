from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import Base, engine
from app.auth.routes import router as auth_router
from app.ai.routes import router as ai_router

Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="AI Chat Assitance",
    description="JWT-secured AI chat backend using FastAPI, MySQL, and Ollama",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(ai_router)

# Status
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "AI Chat Assitance Backend is running"
    }