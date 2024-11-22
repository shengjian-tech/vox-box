import asyncio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse

from vox_box.backends.stt.base import STTBackend
from vox_box.backends.tts.base import TTSBackend
from vox_box.server.model import get_model_instance
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

executor = ThreadPoolExecutor()


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "alloy"
    response_format: str = "wav"
    speed: float = 1.0


@router.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    try:
        model_instance: TTSBackend = get_model_instance()
        if not isinstance(model_instance, TTSBackend):
            return HTTPException(
                status_code=400, detail="Model instance does not support speech API"
            )

        loop = asyncio.get_event_loop()
        audio_file = await loop.run_in_executor(
            executor,
            model_instance.speech,
            request.input,
            request.voice,
            request.speed,
            request.response_format,
        )

        media_type = get_media_type(request.response_format)
        return FileResponse(audio_file, media_type=media_type)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Failed to generate speech, {e}")


@router.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    try:
        form = await request.form()
        keys = form.keys()
        if "file" not in keys:
            return HTTPException(status_code=400, detail="Field file is required")

        audio_bytes = await form["file"].read()
        language = form.get("language")
        prompt = form.get("prompt")
        temperature = float(form.get("temperature", 0.2))
        timestamp_granularities = form.getlist("timestamp_granularities")
        response_format = form.get("response_format", "json")

        model_instance: STTBackend = get_model_instance()
        if not isinstance(model_instance, STTBackend):
            return HTTPException(
                status_code=400,
                detail="Model instance does not support transcriptions API",
            )

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            executor,
            model_instance.transcribe,
            audio_bytes,
            language,
            prompt,
            temperature,
            timestamp_granularities,
            response_format,
        )

        if response_format == "json":
            return {"text": data}
        elif response_format == "text":
            return data
        else:
            return data
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Failed to transcribe audio, {e}")


@router.get("/health")
async def health():
    model_instance = get_model_instance()
    if model_instance is None or (not model_instance.is_load()):
        return HTTPException(status_code=503, detail="Loading model")
    return {"status": "ok"}


@router.get("/v1/models")
async def get_model_list():
    model_instance = get_model_instance()
    if model_instance is None:
        return []
    return {"object": "list", "data": [model_instance.model_info()]}


@router.get("/v1/models/{model_id}")
async def get_model_info(model_id: str):
    model_instance = get_model_instance()
    if model_instance is None:
        return {}
    return model_instance.model_info()


@router.get("/v1/voices")
async def get_voice():
    model_instance = get_model_instance()
    if model_instance is None:
        return {}
    return {
        "model": model_instance.model_info().get("id", ""),
        "voices": model_instance.model_info().get("voices", []),
    }


def get_media_type(response_format) -> str:
    if response_format == "mp3":
        media_type = "audio/mpeg"
    elif response_format == "opus":
        media_type = "audio/ogg;codec=opus"
    elif response_format == "aac":
        media_type = "audio/aac"
    elif response_format == "flac":
        media_type = "audio/x-flac"
    elif response_format == "wav":
        media_type = "audio/wav"
    else:
        raise Exception(
            f"Invalid response_format: '{response_format}'", param="response_format"
        )

    return media_type
