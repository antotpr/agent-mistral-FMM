import os
import logging
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mistral Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

MISTRAL_API_URL = "https://api.mistral.ai/v1/conversations"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
AGENT_ID = "ag_019d86923391777ca69093336e269310"


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    email: Optional[str] = None  # conservé pour compatibilité, non utilisé


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str


@app.get("/")
def root():
    return {"message": "API Mistral Chat - backend OK", "docs": "/docs", "chat": "/chat"}


@app.get("/health")
def health():
    return {"status": "ok", "api_key_configured": bool(MISTRAL_API_KEY)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not MISTRAL_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="MISTRAL_API_KEY non configuree.",
        )

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Le message ne peut pas etre vide.")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Nouvelle conversation → POST /v1/conversations  (avec agent_id)
    # Continuation        → POST /v1/conversations/{id} (sans agent_id)
    if request.conversation_id:
        url = f"{MISTRAL_API_URL}/{request.conversation_id}"
        body = {"inputs": [{"role": "user", "content": request.message}]}
        logger.info("Continuation conversation %s", request.conversation_id)
    else:
        url = MISTRAL_API_URL
        body = {
            "agent_id": AGENT_ID,
            "inputs": [{"role": "user", "content": request.message}],
        }
        logger.info("Nouvelle conversation")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, headers=headers, json=body)
            logger.info("Reponse Mistral : %s", response.status_code)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("Erreur HTTP Mistral : %s - %s", e.response.status_code, e.response.text)
            raise HTTPException(
                status_code=502,
                detail=f"Erreur Mistral ({e.response.status_code}) : {e.response.text}",
            )
        except httpx.RequestError as e:
            logger.error("Erreur reseau : %s", e)
            raise HTTPException(status_code=502, detail=f"Impossible de joindre Mistral : {e}")

    data = response.json()

    conv_id = (
        data.get("conversation_id")
        or data.get("id")
        or request.conversation_id
        or ""
    )

    reply_text = _extract_reply(data)

    if not reply_text:
        logger.warning("Reponse vide de Mistral. Payload : %s", data)
        reply_text = "Desole, je n'ai pas pu generer de reponse."

    return ChatResponse(reply=reply_text, conversation_id=conv_id)


def _extract_reply(data: dict) -> str:
    outputs = data.get("outputs", [])
    for output in outputs:
        if output.get("role") != "assistant":
            continue
        content = output.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif "content" in block:
                        parts.append(str(block["content"]))
            text = "".join(parts).strip()
            if text:
                return text
    return ""
