import json
import os
import tempfile
from typing import Dict, List, Optional

from vox_box.backends.tts.base import TTSBackend
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write as write_wav

from vox_box.utils.audio import convert
from vox_box.utils.log import log_method
from vox_box.utils.model import create_model_dict


class Bark(TTSBackend):
    def __init__(
        self,
        cfg: Config,
    ):
        self.model_load = False
        self._cfg = cfg
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

    def load(self):
        if self.model_load:
            return self

        self._processor = AutoProcessor.from_pretrained(self._cfg.model)
        self._model = BarkModel.from_pretrained(self._cfg.model).to(self._cfg.device)
        self._model = self._model.to_bettertransformer().to(self._cfg.device)
        self._voices = self._get_voices()

        self._model_dict = create_model_dict(
            self._cfg.model,
            task_type=TaskTypeEnum.TTS,
            backend_framework=BackendEnum.BARK,
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
        voice: Optional[str] = "v2/en_speaker_6",
        speed: float = 1,
        reponse_format: str = "mp3",
        **kwargs,
    ) -> str:
        if voice not in self._voices:
            raise ValueError(f"Voice {voice} not supported")

        inputs = self._processor(input, voice_preset=voice)
        inputs["history_prompt"] = inputs["history_prompt"].to(self._cfg.device)
        inputs.to(self._cfg.device)

        audio_array = self._model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self._model.generation_config.sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            wav_file_path = temp_file.name
            write_wav(wav_file_path, rate=sample_rate, data=audio_array)

            output_file_path = convert(wav_file_path, reponse_format, speed)
            return output_file_path

    def _get_voices(self) -> List[str]:
        voices_v1 = []
        voices_v2 = []
        if self._speaker_json is not None:
            for key in self._speaker_json.keys():
                if key == "repo_or_path":
                    continue
                if "v2" in key:
                    voices_v2.append(key)
                else:
                    voices_v1.append(key)

        voices = voices_v2 or voices_v1
        return sorted(voices)
