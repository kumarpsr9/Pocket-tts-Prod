from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pocket_tts import TTSModel
import soundfile as sf
import numpy as np
import tempfile
from pathlib import Path
import torch
from typing import Dict

torch.inference_mode = torch.no_grad

# =====================================================
# App setup
# =====================================================
app = FastAPI(
    title="Pocket TTS API",
    description="CPU-optimized Text-to-Speech API using Pocket-TTS",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def boost_volume(audio_np: np.ndarray, gain: float = 2.0) -> np.ndarray:
    """
    Increase volume safely with clipping protection.
    gain=2.0 -> 200% volume (~ +6 dB)
    """
    #audio_np = audio_np * gain
    audio_np *= (0.99 / np.max(np.abs(audio_np)))
    audio_np = audio_np * gain
    return np.clip(audio_np, -1.0, 1.0)

# =====================================================
# Global model & cache
# =====================================================
tts_model: TTSModel | None = None
VOICE_CACHE: Dict[str, object] = {}

# =====================================================
# Startup: load model + embeddings
# =====================================================
@app.on_event("startup")
def warmup_model():
    global tts_model, VOICE_CACHE

    # -------- Load model once --------
    tts_model = TTSModel.load_model()

    # -------- Preload voice embeddings --------
    preload_voices = ["eponine"]

    for voice in preload_voices:
        VOICE_CACHE[voice] = tts_model.get_state_for_audio_prompt(voice)

    # -------- Warm-up inference (VERY IMPORTANT) --------
    with torch.inference_mode():
        tts_model.generate_audio(
            VOICE_CACHE["eponine"],
            "warmup"
        )

    print("✅ Pocket-TTS loaded | voices cached | CPU threads = 8")

# =====================================================
# Request schemas
# =====================================================
class TTSRequest(BaseModel):
    id: str = Field(..., example="sample_001")
    text: str = Field(..., example="Hello, welcome to Pocket TTS")
    voice: str = Field(default="eponine", example="eponine")


class TTSSaveRequest(BaseModel):
    id: str = Field(..., example="welcome_001")
    text: str = Field(..., example="Welcome to Pocket TTS API")
    folder_name: str = Field(..., example="announcements")
    voice: str = Field(default="eponine", example="eponine")

# =====================================================
# Health check
# =====================================================
@app.get("/")
def health():
    return {"status": "ok"}

# =====================================================
# Helper: get cached voice state
# =====================================================
def get_voice_state(voice: str):
    if voice not in VOICE_CACHE:
        VOICE_CACHE[voice] = tts_model.get_state_for_audio_prompt(voice)
    return VOICE_CACHE[voice]

# =====================================================
# TTS → stream MP3 (temp)
# =====================================================
@app.post("/tts")
def generate_tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        voice_state = get_voice_state(req.voice)

        with torch.inference_mode():
            audio = tts_model.generate_audio(
                voice_state,
                f"...{req.text}..."
            )

        audio_np = audio.cpu().numpy().astype(np.float32)

        mp3_path = Path(tempfile.gettempdir()) / f"{req.id}.mp3"

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

# =====================================================
# TTS → save MP3 locally
# =====================================================
@app.post("/save")
def generate_tts_and_save(req: TTSSaveRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        voice_state = get_voice_state(req.voice)

        with torch.inference_mode():
            audio = tts_model.generate_audio(
                voice_state,
                f"...{req.text}."
            )

        audio_np = audio.cpu().numpy().astype(np.float32)
        audio_np = boost_volume(audio_np, gain=2.0)

        save_dir = Path("output") / req.folder_name
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
            "file_path": mp3_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
