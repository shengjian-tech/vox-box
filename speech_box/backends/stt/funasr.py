import json
import logging
import os
from typing import Dict, List, Optional
import tempfile
from speech_box.backends.stt.base import STTBackend
from speech_box.config.config import Config
from speech_box.utils.log import log_method
from speech_box.utils.model import create_model_dict


logger = logging.getLogger(__name__)


class FunASR(STTBackend):
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

        self._configuration_json = None
        self._config_json = None
        configuration_path = os.path.join(self._cfg.model, "configuration.json")
        config_json_path = os.path.join(self._cfg.model, "config.json")
        if os.path.exists(configuration_path):
            with open(configuration_path, "r", encoding="utf-8") as f:
                self._configuration_json = json.load(f)

        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as f:
                self._config_json = json.load(f)

        self._supported = self._supported()

    def task_type():
        return "stt"

    def load(self):
        from funasr import AutoModel

        if not self._supported:
            return None

        if self.model_load:
            return self

        log_level = "INFO"
        if self._cfg.debug:
            log_level = "DEBUG"

        self._model = AutoModel(
            model=self._cfg.model,
            device=self._cfg.device,
            model_path=self._cfg.model,
            log_level=log_level,
            disable_update=True,
            punc_model=self._cfg.punc_model,
            vad_model=self._cfg.vad_model,
            spk_model=self._cfg.spk_model,
        )
        self._required_resource = self._get_required_resource()
        self._model_dict = create_model_dict(
            self._cfg.model, required_resource=self._required_resource
        )
        self._model_load = True

        return self

    def model_info(self) -> Dict:
        return self._model_dict

    def supported(self) -> bool:
        return self._supported

    def _supported(self) -> bool:
        # TODO: qwen audio is special
        if self._configuration_json is not None:
            task = self._configuration_json.get("task")
            if task is not None and task != "auto-speech-recognition":
                return False

            model_type = self._configuration_json.get("model", {}).get("type")
            if model_type is not None and model_type == "funasr":
                return True

        if self._config_json is not None:
            architectures = self._config_json.get("architectures")
            if (architectures is not None and "QWenLMHeadModel" in architectures) and (
                self._config_json.get("audio", {}).get("n_layer", 0) != 0
            ):
                return True

        return False

    @log_method
    def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        timestamp_granularities: Optional[List[str]] = None,
        response_format: str = "json",
        **kwargs
    ):
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        language = "auto" if language is None else language

        if timestamp_granularities is not None:
            logger.warning(
                "Param `timestamp_granularities` is not supported for FunASR"
            )

        if prompt is not None or temperature is not None:
            # https://arxiv.org/abs/2402.08846?spm=a2c6h.13066369.question.5.34f0ef7e3Rad2n&file=2402.08846
            # https://github.com/modelscope/FunASR/issues/1523
            # https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/qwen_audio/demo_chat.py#L6
            logger.info(
                "Params `prompt` and `temperature` are only supported for FunASR llm asr model, will ignore them if model isn't supported"
            )

        with tempfile.NamedTemporaryFile(buffering=0) as f:
            f.write(audio)
            res = self._model.generate(
                input=f.name,
                language=language,
                prompt=prompt,
                temperature=temperature,
                **kwargs
            )

            text = rich_transcription_postprocess(res[0]["text"])
            return text

    def _get_required_resource(self) -> Dict:
        # TODO: not accurate
        Gib = 1024 * 1024 * 1024
        return {"cuda": {"vram": 10 * Gib}, "cpu": {"ram": 10 * Gib}}
