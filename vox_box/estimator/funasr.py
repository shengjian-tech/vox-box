import json
import logging
import os
from typing import Dict, Tuple

import yaml
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.downloader.downloaders import download_file
from vox_box.estimator.base import Estimator
from vox_box.utils.model import create_model_dict


logger = logging.getLogger(__name__)


class FunASR(Estimator):
    def __init__(
        self,
        cfg: Config,
    ):
        self._cfg = cfg
        self._optional_files = ["configuration.json", "config.json", "config.yaml"]

    def model_info(self) -> Dict:
        model = (
            self._cfg.model
            or self._cfg.huggingface_repo_id
            or self._cfg.model_scope_model_id
        )
        supported, model_architecture = self._supported()
        return create_model_dict(
            model,
            supported=supported,
            task_type=TaskTypeEnum.STT,
            backend_framework=BackendEnum.FUN_ASR,
            model_architecture=model_architecture,
        )

    def _supported(self) -> Tuple[bool, str]:
        if self._cfg.model is not None:
            return self._check_local_model(self._cfg.model)
        elif (
            self._cfg.huggingface_repo_id is not None
            or self._cfg.model_scope_model_id is not None
        ):
            return self._check_remote_model()

    def _check_local_model(self, base_dir: str) -> Tuple[bool, str]:
        configuration_json = None
        config_json = None
        config_yaml = None

        configuration_path = os.path.join(base_dir, "configuration.json")
        config_json_path = os.path.join(base_dir, "config.json")
        config_yaml_path = os.path.join(base_dir, "config.yaml")

        supported = False
        model_architecture = ""

        if os.path.exists(configuration_path):
            with open(configuration_path, "r", encoding="utf-8") as f:
                configuration_json = json.load(f)

        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as f:
                config_json = json.load(f)

        if os.path.exists(config_yaml_path):
            with open(config_yaml_path, "r", encoding="utf-8") as f:
                config_yaml = yaml.safe_load(f)

        if configuration_json is not None:
            task = configuration_json.get("task", "")
            model_type = configuration_json.get("model", {}).get("type", "")
            if task == "auto-speech-recognition" and model_type == "funasr":
                supported = True
                if config_yaml is not None:
                    model_architecture = config_yaml.get("model")

        if config_json is not None:
            architectures = config_json.get("architectures")
            if (architectures is not None and "QWenLMHeadModel" in architectures) and (
                config_json.get("audio", {}).get("n_layer", 0) != 0
            ):
                supported = True
                model_architecture = "QwenAudio"

        return supported, model_architecture

    def _check_remote_model(self) -> Tuple[bool, str]:
        downloaded_files = []
        for f in self._optional_files:
            try:
                download_file_path = download_file(
                    huggingface_repo_id=self._cfg.huggingface_repo_id,
                    huggingface_filename=f,
                    model_scope_model_id=self._cfg.model_scope_model_id,
                    model_scope_file_path=f,
                    cache_dir=self._cfg.cache_dir,
                )
            except Exception as e:
                logger.debug(f"File {f} does not exist, {e}")
                continue
            downloaded_files.append(download_file_path)

        if len(downloaded_files) != 0:
            base_dir = os.path.dirname(downloaded_files[0])
            return self._check_local_model(base_dir)

        return False
