import json
import os
import tempfile
from typing import Dict, List, Optional

from speech_box.backends.tts.base import TTSBackend
from speech_box.config.config import Config
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write as write_wav

from speech_box.utils.ffmpeg import convert
from speech_box.utils.log import log_method
from speech_box.utils.model import create_model_dict


class Bark(TTSBackend):
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

        self._config_json = None
        self._speaker_json = None
        config_path = os.path.join(self._cfg.model, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self._config_json = json.load(f)

        speaker_path = os.path.join(self._cfg.model, "speaker_embeddings_path.json")
        if os.path.exists(speaker_path):
            with open(speaker_path, "r", encoding="utf-8") as f:
                self._speaker_json = json.load(f)

        self._supported = self._supported()

    def load(self):
        if not self._supported:
            return None

        if self.model_load:
            return self

        self._processor = AutoProcessor.from_pretrained(self._cfg.model)
        self._model = BarkModel.from_pretrained(self._cfg.model).to(self._cfg.device)
        self._model = self._model.to_bettertransformer()
        self._resource_required = self._get_required_resource()
        self._voices = self._get_voices()

        self._model_dict = create_model_dict(
            self._cfg.model,
            resource_required=self._resource_required,
            voices=self._voices,
        )
        self.model_load = True
        return self

    def supported(self) -> bool:
        return self._supported

    def _supported(self) -> bool:
        if self._config_json is not None:
            architectures = self._config_json.get("architectures")
            if architectures is not None and "BarkModel" in architectures:
                return True
        return False

    def model_info(self) -> Dict:
        return self._model_dict

    @log_method
    def speech(
        self,
        input: str,
        voice: Optional[str] = "v2/en_speaker_6",
        speed: float = 1,
        reponse_format: str = "mp3",
        **kwargs,
    ) -> str:
        if voice not in self._voices:
            raise ValueError(f"Voice {voice} not supported")

        inputs = self._processor(input, voice_preset=voice)
        audio_array = self._model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self._model.generation_config.sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            wav_file_path = temp_file.name
            write_wav(wav_file_path, rate=sample_rate, data=audio_array)

            with tempfile.NamedTemporaryFile(
                suffix=f".{reponse_format}", delete=False
            ) as output_temp_file:
                output_file_path = output_temp_file.name
                convert(wav_file_path, reponse_format, output_file_path, speed)
                return output_file_path

    def _get_required_resource(self) -> Dict:
        hidden_size = (
            self._config_json.get("coarse_acoustics_config", {}).get("hidden_size")
            or self._config_json.get("fine_acoustics_config", {}).get("hidden_size")
            or self._config_json.get("semantic_config", {}).get("hidden_size")
        )

        Gib = 1024 * 1024 * 1024
        # https://github.com/suno-ai/bark?tab=readme-ov-file#how-much-vram-do-i-need
        # TODO: ram is not accurate
        if hidden_size is not None and hidden_size == 768:
            # small model
            return {"cuda": {"vram": 2 * Gib}, "cpu": {"ram": 4 * Gib}}
        else:
            return {"cuda": {"vram": 12 * Gib}, "cpu": {"ram": 20 * Gib}}

    def _get_voices(self) -> List[str]:
        voices = []
        if self._speaker_json is not None:
            for key in self._speaker_json.keys():
                if key == "repo_or_path":
                    continue
                voices.append(key)

            return voices
