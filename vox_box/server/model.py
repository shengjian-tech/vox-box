from typing import Union
from vox_box.backends.stt.base import STTBackend
from vox_box.backends.stt.faster_whisper import FasterWhisper
from vox_box.backends.stt.funasr import FunASR
from vox_box.backends.tts.bark import Bark
from vox_box.backends.tts.base import TTSBackend
from vox_box.backends.tts.cosyvoice import CosyVoice
from vox_box.config.config import BackendEnum, Config
from vox_box.downloader import downloaders
from vox_box.elstimator.estimate import estimate_model

_instance = None


class ModelInstance:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._backend_framework = None
        self._estimate = estimate_model(cfg)
        if (
            self._estimate is None
            or not self._estimate.get("supported", False)
            or self._estimate.get("backend_framework")
            not in BackendEnum.__members__.values()
        ):
            raise Exception("Model isn't supported")

        if self._cfg.model is None and (
            self._cfg.huggingface_repo_id is not None
            or self._cfg.model_scope_model_id is not None
        ):
            try:
                mode_path = downloaders.download_model(
                    huggingface_repo_id=self._cfg.huggingface_repo_id,
                    model_scope_model_id=self._cfg.model_scope_model_id,
                    cache_dir=self._cfg.cache_dir,
                )
                self._cfg.model = mode_path
            except Exception as e:
                raise Exception(f"Faild to download model, {e}")

        backend_framework_name = self._estimate.get("backend_framework")
        if backend_framework_name == BackendEnum.FASTER_WHISPER:
            self._backend_framework = FasterWhisper(cfg)
        elif backend_framework_name == BackendEnum.FUN_ASR:
            self._backend_framework = FunASR(cfg)
        elif backend_framework_name == BackendEnum.BARK:
            self._backend_framework = Bark(cfg)
        elif backend_framework_name == BackendEnum.COSY_VOICE:
            self._backend_framework = CosyVoice(cfg)

    def run(self):
        global _instance

        if _instance is None:
            try:
                _instance = self._backend_framework.load()
            except Exception as e:
                raise Exception(f"Faild to load model, {e}")
        return _instance


def get_model_instance() -> Union[TTSBackend, STTBackend]:
    global _instance
    return _instance
