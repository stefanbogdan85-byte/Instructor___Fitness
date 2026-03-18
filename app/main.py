from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# reduce TensorFlow log noise in server output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from src.services.fitness_assistant import FitnessAssistant

from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

@asynccontextmanager
async def lifespan(app: FastAPI):    
    load_dotenv()
    yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str

assistant_response = FitnessAssistant()

@app.get("/")
def root():
    return {"message": "Salutare! Fitness Assistant ruleaza."}


@app.post("/chat/")
async def chat(payload: ChatRequest):
    try:
        answer = await asyncio.wait_for(
            asyncio.to_thread(assistant_response.assistant_response, payload.message), timeout=45)
        return {"response": answer}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Raspunsul de chat a expirat")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)