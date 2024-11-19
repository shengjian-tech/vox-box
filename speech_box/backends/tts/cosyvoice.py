import os
import sys
import wave
import numpy as np

from speech_box.backends.tts.base import TTSBackend
from speech_box.utils.log import log_method

import tempfile
from typing import Dict, List, Optional
from hyperpyyaml import load_hyperpyyaml

from speech_box.config.config import Config
from speech_box.utils.ffmpeg import convert
from speech_box.utils.model import create_model_dict

sys.path.append("speech_box/third_party/CosyVoice/third_party/Matcha-TTS")
sys.path.append("speech_box/third_party/CosyVoice/")


class CosyVoice(TTSBackend):
    def __init__(
        self,
        cfg: Config,
    ):
        self.model_load = False
        self._cfg = cfg
        self._resource_required = None
        self._voices = None
        self._model = None
        self._model_dict = {}
        self._supported = self._supported()

    def load(self):
        from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceModel

        if not self._supported:
            return None

        if self.model_load:
            return self

        self._model = CosyVoiceModel(self._cfg.model)
        self._voices = self._get_voices()
        self._required_resource = self._get_required_resource()

        self._model_dict = create_model_dict(
            self._cfg.model,
            required_resource=self._required_resource,
            voices=self._voices,
        )

        self.model_load = True
        return self

    def supported(self) -> bool:
        return self._supported

    def _supported(self) -> bool:
        config_path = os.path.join(self._cfg.model, "cosyvoice.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_yaml = load_hyperpyyaml(f)
                if config_yaml is not None:
                    return True
        return False

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
