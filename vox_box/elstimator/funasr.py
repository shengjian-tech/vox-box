import json
import logging
import os
from typing import Dict
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.downloader.downloaders import download_model
from vox_box.elstimator.base import Elstimator
from vox_box.utils.model import create_model_dict


logger = logging.getLogger(__name__)


class FunASR(Elstimator):
    def __init__(
        self,
        cfg: Config,
    ):
        self._cfg = cfg
        self._optional_files = ["configuration.json", "config.json"]

    def model_info(self) -> Dict:
        model = (
            self._cfg.model
            or self._cfg.huggingface_repo_id
            or self._cfg.model_scope_model_id
        )
        supported = self._supported()
        return create_model_dict(
            model,
            supported=supported,
            task_type=TaskTypeEnum.STT,
            backend_framework=BackendEnum.FUN_ASR,
        )

    def _supported(self) -> bool:
        if self._cfg.model is not None:
            return self._check_local_model(self._cfg.model)
        elif (
            self._cfg.huggingface_repo_id is not None
            or self._cfg.model_scope_model_id is not None
        ):
            return self._check_remote_model()

    def _check_local_model(self, base_dir: str) -> bool:
        configuration_json = None
        config_json = None

        configuration_path = os.path.join(base_dir, "configuration.json")
        config_json_path = os.path.join(base_dir, "config.json")

        if os.path.exists(configuration_path):
            with open(configuration_path, "r", encoding="utf-8") as f:
                configuration_json = json.load(f)

        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as f:
                config_json = json.load(f)

        if configuration_json is not None:
            task = configuration_json.get("task", "")
            model_type = configuration_json.get("model", {}).get("type", "")
            if task == "auto-speech-recognition" and model_type == "funasr":
                return True

        if config_json is not None:
            architectures = config_json.get("architectures")
            if (architectures is not None and "QWenLMHeadModel" in architectures) and (
                config_json.get("audio", {}).get("n_layer", 0) != 0
            ):
                return True

        return False

    def _check_remote_model(self) -> bool:
        downloaded_files = []
        for f in self._optional_files:
            try:
                download_file_path = download_model(
                    huggingface_repo_id=self._cfg.huggingface_repo_id,
                    huggingface_filename=f,
                    model_scope_model_id=self._cfg.model_scope_model_id,
                    model_scope_file_path=f,
                    cache_dir=self._cfg.cache_dir,
                )
            except Exception as e:
                logger.error(f"Failed to download {f} for model estimate, {e}")
                continue
            downloaded_files.append(download_file_path)

        if len(downloaded_files) != 0:
            base_dir = os.path.dirname(downloaded_files[0])
            return self._check_local_model(base_dir)

        return False
