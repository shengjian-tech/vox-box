import json
import logging
import os
from typing import Dict
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.downloader.downloaders import download_model
from vox_box.elstimator.base import Elstimator
from vox_box.utils.model import create_model_dict

logger = logging.getLogger(__name__)


class Bark(Elstimator):
    def __init__(
        self,
        cfg: Config,
    ):
        self._cfg = cfg
        self._required_files = [
            "config.json",
            "speaker_embeddings_path.json",
        ]
        self._config_json = None
        self._speaker_json = None

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
            task_type=TaskTypeEnum.TTS,
            backend_framework=BackendEnum.BARK,
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
        if not all(
            os.path.exists(os.path.join(base_dir, file))
            for file in self._required_files
        ):
            return False

        supported = False
        config_path = os.path.join(base_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self._config_json = json.load(f)
            architectures = self._config_json.get("architectures")
            if architectures is not None and "BarkModel" in architectures:
                supported = True

        speaker_path = os.path.join(base_dir, "speaker_embeddings_path.json")
        with open(speaker_path, "r", encoding="utf-8") as f:
            self._speaker_json = json.load(f)

        return supported

    def _check_remote_model(self) -> bool:
        downloaded_files = []
        for f in self._required_files:
            try:
                downloaded_file_path = download_model(
                    huggingface_repo_id=self._cfg.huggingface_repo_id,
                    huggingface_filename=f,
                    model_scope_model_id=self._cfg.model_scope_model_id,
                    model_scope_file_path=f,
                    cache_dir=self._cfg.cache_dir,
                )
                downloaded_files.append(downloaded_file_path)
            except Exception as e:
                logger.error(f"Failed to download {f}, {e}")
                continue

        if len(downloaded_files) != 0:
            base_dir = os.path.dirname(downloaded_files[0])
            return self._check_local_model(base_dir)

        return False
