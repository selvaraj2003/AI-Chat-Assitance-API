import requests
from fastapi import HTTPException
from app.core.config import settings


def generate_ai(prompt: str, model: str | None):
    if settings["AI_PROVIDER"] == "cloud":
        return cloud_call(prompt, model)
    return ollama_call(prompt, model)


def ollama_call(prompt: str, model: str | None):
    r = requests.post(
        settings["OLLAMA_BASE_URL"],
        json={
            "model": model or settings["DEFAULT_OLLAMA_MODEL"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        },
        timeout=settings["OLLAMA_TIMEOUT"]
    )

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="Ollama error")

    d = r.json()
    return d["message"]["content"], d.get("eval_count")


def cloud_call(prompt: str, model: str | None):
    r = requests.post(
        settings["CLOUD_API_BASE_URL"],
        headers={
            "Authorization": f"Bearer {settings['CLOUD_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or settings["CLOUD_MODEL"],
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=settings["CLOUD_TIMEOUT"]
    )

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="Cloud AI error")

    d = r.json()
    return d["choices"][0]["message"]["content"], d["usage"]["total_tokens"]



def get_local_models():
    r = requests.get(
        settings["OLLAMA_BASE_URL"].replace("/api/chat", "/api/tags")
    )

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch local models")

    return [m["name"] for m in r.json()["models"]]


def get_cloud_models():
    r = requests.get(
        f"{settings['CLOUD_API_BASE_URL']}/api/tags"
    )

    if r.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch Ollama models"
        )
    return [m["name"] for m in r.json().get("models", [])]