from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import time
import uuid

from app.core.database import get_db
from app.auth.deps import get_current_user
from app.models.user import User
from app.models.chat import ChatHistory
from app.ai.schemas import ChatRequest, ChatResponse
from app.core.config import settings
from app.ai.client import generate_ai, get_local_models, get_cloud_models

router = APIRouter(
    prefix="/api/ai",
    tags=["AI"],
    dependencies=[Depends(get_current_user)],
)


# CREATE / GENERATE CHAT
@router.post("/generate", response_model=ChatResponse)
def chat_with_ai(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = payload.session_id or str(uuid.uuid4())
    start_time = time.time()

    try:
        ai_response, tokens_used = generate_ai(
            payload.prompt,
            payload.model,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        model_name = payload.model or (
            settings["CLOUD_MODEL"]
            if settings["AI_PROVIDER"] == "cloud"
            else settings["DEFAULT_OLLAMA_MODEL"]
        )

        chat = ChatHistory(
            user_id=current_user.id,
            session_id=session_id,
            prompt=payload.prompt,
            response=ai_response,
            model_name=model_name,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            is_success=True,
        )

        db.add(chat)
        db.commit()
        db.refresh(chat)

        return ChatResponse(
            session_id=session_id,
            response=ai_response,
            model=model_name,
            latency_ms=latency_ms,
        )

    except Exception as e:
        db.add(
            ChatHistory(
                user_id=current_user.id,
                session_id=session_id,
                prompt=payload.prompt,
                response="",
                model_name=payload.model,
                is_success=False,
                error_message=str(e),
            )
        )
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI processing failed",
        )


# GET CHAT HISTORY (SESSION BASED)
@router.get("/history")
def get_chat_history(
    session_id: str | None = None,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == current_user.id)
    )

    if session_id:
        query = query.filter(ChatHistory.session_id == session_id)

    chats = (
        query.order_by(ChatHistory.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": c.id,
            "session_id": c.session_id,
            "prompt": c.prompt,
            "response": c.response,
            "model": c.model_name,
            "created_at": c.created_at,
        }
        for c in chats
    ]


# DELETE CHAT (FULL SESSION)
@router.delete("/history/{session_id}")
def delete_chat(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    deleted = (
        db.query(ChatHistory)
        .filter(
            ChatHistory.session_id == session_id,
            ChatHistory.user_id == current_user.id,
        )
        .delete()
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    db.commit()
    return {"message": "Chat deleted successfully"}


# LIST LOCAL MODELS
@router.get("/models/local")
def list_local_models():
    return {
        "provider": "local",
        "default": settings["DEFAULT_OLLAMA_MODEL"],
        "models": get_local_models(),
    }



# LIST CLOUD MODELS
@router.get("/models/cloud")
def list_cloud_models():
    return {
        "provider": "cloud",
        "default": settings["CLOUD_MODEL"],
        "models": get_cloud_models(),
    }