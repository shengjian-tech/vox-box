import asyncio
import functools
from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse

from vox_box.backends.stt.base import STTBackend
from vox_box.backends.tts.base import TTSBackend
from vox_box.server.model import get_model_instance
from concurrent.futures import ThreadPoolExecutor

from fastapi import Form, UploadFile, File
from typing import Optional
import logging

router = APIRouter()

executor = ThreadPoolExecutor()

logger = logging.getLogger(__name__)

ALLOWED_SPEECH_OUTPUT_AUDIO_TYPES = {
    "mp3",
    "opus",
    "aac",
    "flac",
    "wav",
    "pcm",
}


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: str = "mp3"
    speed: float = 1.0


@router.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    try:
        if (
            request.response_format
            and request.response_format not in ALLOWED_SPEECH_OUTPUT_AUDIO_TYPES
        ):
            return HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {request.response_format}",
            )

        if request.speed < 0.25 or request.speed > 2:
            return HTTPException(
                status_code=400, detail="Speed must be between 0.25 and 2"
            )

        model_instance: TTSBackend = get_model_instance()
        if not isinstance(model_instance, TTSBackend):
            return HTTPException(
                status_code=400, detail="Model instance does not support speech API"
            )

        func = functools.partial(
            model_instance.speech,
            request.input,
            request.voice,
            request.speed,
            request.response_format,
        )

        loop = asyncio.get_event_loop()
        audio_file = await loop.run_in_executor(
            executor,
            func,
        )

        media_type = get_media_type(request.response_format)
        return FileResponse(audio_file, media_type=media_type)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Failed to generate speech, {e}")


@router.post("/v1/audio/copy")
async def copy(
    model: str = Form(...),
    input: str = Form(...),
    voice: str = Form(...),
    response_format: str = Form("mp3"),
    speed: float = Form(1.0),
    prompt_text: Optional[str] = Form(None),  # 新增参数
    prompt_wav: Optional[UploadFile] = File(None),  # 新增文件上传
):
    try:
        # 验证响应格式
        if response_format and response_format not in ALLOWED_SPEECH_OUTPUT_AUDIO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {response_format}",
            )

        # 验证语速
        if speed < 0.25 or speed > 2:
            raise HTTPException(
                status_code=400, detail="Speed must be between 0.25 and 2"
            )

        # 验证音频文件类型
        if prompt_wav:
            content_type = prompt_wav.content_type
            if content_type not in ALLOWED_TRANSCRIPTIONS_INPUT_AUDIO_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported prompt audio format: {content_type}",
                )

        # 获取TTS模型实例
        model_instance: TTSBackend = get_model_instance()
        if not isinstance(model_instance, TTSBackend):
            raise HTTPException(
                status_code=400, detail="Model instance does not support speech API"
            )

        # 准备额外参数
        kwargs = {}
        if prompt_text:
            kwargs["prompt_text"] = prompt_text
        if prompt_wav:
            # 读取上传的音频文件内容
            kwargs["prompt_wav"] = prompt_wav

        # 创建函数部分应用
        func = functools.partial(
            model_instance.speech,
            input,
            voice,
            speed,
            response_format,
            **kwargs
        )

        loop = asyncio.get_event_loop()
        audio_file = await loop.run_in_executor(
            executor,
            func,
        )

        media_type = get_media_type(response_format)
        return FileResponse(audio_file, media_type=media_type)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Failed to generate speech, {e}")

# ref: https://github.com/LMS-Community/slimserver/blob/public/10.0/types.conf
ALLOWED_TRANSCRIPTIONS_INPUT_AUDIO_FORMATS = {
    # flac
    "audio/flac",
    "audio/x-flac",
    # mp3
    "audio/mpeg",
    "audio/x-mpeg",
    "audio/mp3",
    "audio/mp3s",
    "audio/mpeg3",
    "audio/mpg",
    # mp4
    "audio/m4a",
    "audio/x-m4a",
    "audio/mp4",
    # mpeg
    "audio/mpga",
    # ogg
    "audio/ogg",
    "audio/x-ogg",
    # wav
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    # webm
    "video/webm",
    "audio/webm",
    # file
    "application/octet-stream",
}

ALLOWED_TRANSCRIPTIONS_OUTPUT_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}


@router.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    try:
        form = await request.form()
        keys = form.keys()
        if "file" not in keys:
            return HTTPException(status_code=400, detail="Field file is required")

        file: UploadFile = form[
            "file"
        ]  # flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm
        file_content_type = file.content_type
        if file_content_type not in ALLOWED_TRANSCRIPTIONS_INPUT_AUDIO_FORMATS:
            return HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_content_type}",
            )

        audio_bytes = await file.read()
        language = form.get("language")
        prompt = form.get("prompt")
        temperature = float(form.get("temperature", 0))
        if not (0 <= temperature <= 1):
            return HTTPException(
                status_code=400, detail="Temperature must be between 0 and 1"
            )

        timestamp_granularities = form.getlist("timestamp_granularities")
        response_format = form.get("response_format", "json")
        if response_format not in ALLOWED_TRANSCRIPTIONS_OUTPUT_FORMATS:
            return HTTPException(
                status_code=400, detail="Unsupported response_format: {response_format}"
            )

        model_instance: STTBackend = get_model_instance()
        if not isinstance(model_instance, STTBackend):
            return HTTPException(
                status_code=400,
                detail="Model instance does not support transcriptions API",
            )

        kwargs = {
            "content_type": file_content_type,
        }
        func = functools.partial(
            model_instance.transcribe,
            audio_bytes,
            language,
            prompt,
            temperature,
            timestamp_granularities,
            response_format,
            **kwargs,
        )

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            executor,
            func,
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


@router.get("/v1/languages")
async def get_languages():
    model_instance = get_model_instance()
    if model_instance is None:
        return {}
    return {
        "languages": model_instance.model_info().get("languages", []),
    }


@router.get("/v1/voices")
async def get_voice():
    model_instance = get_model_instance()
    if model_instance is None:
        return {}
    return {
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
    elif response_format == "pcm":
        media_type = "audio/pcm"
    else:
        raise Exception(
            f"Invalid response_format: '{response_format}'", param="response_format"
        )

    return media_type
