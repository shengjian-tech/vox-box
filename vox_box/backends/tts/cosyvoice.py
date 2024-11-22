import os
import sys
import wave
import numpy as np
import tempfile
from typing import Dict, List, Optional

from vox_box.backends.tts.base import TTSBackend
from vox_box.utils.log import log_method
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.utils.ffmpeg import convert
from vox_box.utils.model import create_model_dict

paths_to_insert = [
    os.path.join(os.path.dirname(__file__), "../../third_party/CosyVoice"),
    os.path.join(
        os.path.dirname(__file__), "../../third_party/CosyVoice/third_party/Matcha-TTS"
    ),
]


class CosyVoice(TTSBackend):
    def __init__(
        self,
        cfg: Config,
    ):
        self.model_load = False
        self._cfg = cfg
        self._voices = None
        self._model = None
        self._model_dict = {}

    def load(self):
        for path in paths_to_insert:
            sys.path.insert(0, path)

        from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceModel

        if self.model_load:
            return self

        self._model = CosyVoiceModel(self._cfg.model)
        self._voices = self._get_voices()

        self._model_dict = create_model_dict(
            self._cfg.model,
            task_type=TaskTypeEnum.TTS,
            backend_framework=BackendEnum.COSY_VOICE,
            voices=self._voices,
        )

        self.model_load = True
        return self

    def is_load(self) -> bool:
        return self.model_load

    def model_info(self) -> Dict:
        return self._model_dict

    @log_method
    def speech(
        self,
        input: str,
        voice: Optional[str] = "中文女",
        speed: float = 1,
        reponse_format: str = "mp3",
        **kwargs,
    ) -> str:
        if voice not in self._voices:
            raise ValueError(f"Voice {voice} not supported")

        model_output = self._model.inference_sft(input, voice, False, speed)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            wav_file_path = temp_file.name
            with wave.open(wav_file_path, "wb") as wf:
                wf.setnchannels(1)  # single track
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)  # Sample rate
                for i in model_output:
                    tts_audio = (
                        (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
                    )
                    wf.writeframes(tts_audio)

            with tempfile.NamedTemporaryFile(
                suffix=f".{reponse_format}", delete=False
            ) as output_temp_file:
                output_file_path = output_temp_file.name
                convert(wav_file_path, reponse_format, output_file_path, speed)
                return output_file_path

    def _get_required_resource(self) -> Dict:
        # TODO: not accurate
        Gib = 1024 * 1024 * 1024
        return {"cuda": {"vram": 16 * Gib}, "cpu": {"ram": 16 * Gib}}

    def _get_voices(self) -> List[str]:
        return self._model.list_avaliable_spks()
