from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pocket_tts import TTSModel
import soundfile as sf
import numpy as np
import os
import tempfile
from pathlib import Path

# -------------------------
# App setup
# -------------------------
app = FastAPI(
    title="Pocket TTS API",
    description="Production-grade Text-to-Speech API using Pocket-TTS",
    version="1.0.0"
)

# -------------------------
# Load model once (IMPORTANT)
# -------------------------
tts_model = TTSModel.load_model()

# -------------------------
# Routers
# -------------------------
tts_router = APIRouter(prefix="/tts", tags=["Text-to-Speech"])

# -------------------------
# Request Schemas
# -------------------------
class TTSRequest(BaseModel):
    id: str = Field(..., example="sample_001")
    text: str = Field(..., example="Hello, welcome to Pocket TTS")
    voice: str = Field(default="eponine", example="eponine")


class TTSSaveRequest(BaseModel):
    id: str = Field(..., example="welcome_001")
    text: str = Field(..., example="Welcome to Pocket TTS API")
    folder_name: str = Field(..., example="announcements")
    voice: str = Field(default="eponine", example="eponine")

# -------------------------
# Health check
# -------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# -------------------------
# TTS → stream MP3 (temp)
# -------------------------
@tts_router.post("")
def generate_tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        voice_state = tts_model.get_state_for_audio_prompt(req.voice)

        audio = tts_model.generate_audio(
            voice_state,
            f"...{req.text}."
        )

        audio_np = audio.cpu().numpy().astype(np.float32)

        tmp_dir = tempfile.gettempdir()
        mp3_path = os.path.join(tmp_dir, f"{req.id}.mp3")

        sf.write(
            mp3_path,
            audio_np,
            tts_model.sample_rate,
            format="MP3"
        )

        return FileResponse(
            mp3_path,
            media_type="audio/mpeg",
            filename=f"{req.id}_{req.voice}.mp3"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# TTS → save MP3 locally
# -------------------------
@tts_router.post("/save")
def generate_tts_and_save(req: TTSSaveRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        voice_state = tts_model.get_state_for_audio_prompt(req.voice)

        audio = tts_model.generate_audio(
            voice_state,
            f"...{req.text}."
        )

        audio_np = audio.cpu().numpy().astype(np.float32)

        base_dir = Path("output")
        save_dir = base_dir / req.folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        mp3_path = save_dir / f"{req.id}.mp3"

        sf.write(
            mp3_path,
            audio_np,
            tts_model.sample_rate,
            format="MP3"
        )

        return {
            "status": "success",
            "id": req.id,
            "voice": req.voice,
            "file_path": str(mp3_path.resolve())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Register router
# -------------------------
app.include_router(tts_router)
