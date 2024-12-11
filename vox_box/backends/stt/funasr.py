import json
import logging
import os
from typing import Dict, List, Optional
import tempfile
from vox_box.backends.stt.base import STTBackend
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.utils.audio import convert
from vox_box.utils.log import log_method
from vox_box.utils.model import create_model_dict


logger = logging.getLogger(__name__)


class FunASR(STTBackend):
    def __init__(
        self,
        cfg: Config,
    ):
        self.model_load = False
        self._cfg = cfg
        self._model = None
        self._model_dict = {}
        self._log_level = "INFO"
        if self._cfg.debug:
            self._log_level = "DEBUG"

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

    def load(self):
        from funasr import AutoModel

        if self.model_load:
            return self

        self._model = AutoModel(
            model=self._cfg.model,
            device=self._cfg.device,
            model_path=self._cfg.model,
            log_level=self._log_level,
            disable_update=True,
        )

        self._languages = self._get_languages()

        self._model_dict = create_model_dict(
            self._cfg.model,
            task_type=TaskTypeEnum.STT,
            backend_framework=BackendEnum.FUN_ASR,
            languages=self._languages,
        )
        self._model_load = True
        return self

    def is_load(self) -> bool:
        return self.model_load

    def model_info(self) -> Dict:
        return self._model_dict

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

        with tempfile.NamedTemporaryFile(buffering=0, delete=False) as f:
            f.write(audio)
            input_file = f.name

            content_type = kwargs.get("content_type")
            if content_type is not None and "webm" in content_type:
                input_file = convert(input_file, "wav", 1, "webm")

            res = self._model.generate(
                input=input_file,
                language=language,
                prompt=prompt,
                temperature=temperature,
                use_itn=True,
                log_level=self._log_level,
                **kwargs
            )

            text = rich_transcription_postprocess(res[0]["text"])
            return text

    def _get_languages(self) -> List[Dict]:
        return [
            {"auto": "auto"},
        ]
