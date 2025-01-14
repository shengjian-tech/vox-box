import json
import logging
import os
from typing import Dict, List
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.downloader.downloaders import download_file
from vox_box.downloader.hub import match_files
from vox_box.estimator.base import Estimator
from vox_box.utils.model import create_model_dict
from faster_whisper.transcribe import WhisperModel

logger = logging.getLogger(__name__)


class FasterWhisper(Estimator):
    def __init__(
        self,
        cfg: Config,
    ):
        self._cfg = cfg
        self._required_files = [
            "model.bin",
            "tokenizer.json",
        ]

        self._optional_files = ["preprocessor_config.json"]

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
            backend_framework=BackendEnum.FASTER_WHISPER,
        )

    def _supported(self) -> bool:
        if self._cfg.model is not None:
            return self._check_local_model(self._cfg.model, self._required_files)
        elif (
            self._cfg.huggingface_repo_id is not None
            or self._cfg.model_scope_model_id is not None
        ):
            return self._check_remote_model()

    def _check_local_model(self, base_dir: str, required_files: List[str]) -> bool:
        if not all(
            os.path.exists(os.path.join(base_dir, file)) for file in required_files
        ):
            return False

        preprocessor_config_path = os.path.join(base_dir, "preprocessor_config.json")
        if os.path.exists(preprocessor_config_path):
            with open(preprocessor_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                if config.get("processor_class") == "WhisperProcessor":
                    return True
        try:
            model_path = base_dir
            if self._cfg.model is None:
                model_path = self._cfg.model

            WhisperModel(model_path)
            return True
        except Exception as e:
            logger.error(f"Failed to load model for estimating, {e}")
            return False

    def _check_remote_model(self) -> bool:  # noqa: C901
        # Huggingface
        if self._cfg.huggingface_repo_id is not None:
            arr = self._cfg.huggingface_repo_id.split("/")
            if len(arr) != 2:
                logger.error(
                    f"Invalid huggingface repo id: {self._cfg.huggingface_repo_id}"
                )
                return False

            if arr[0].lower() == "systran":
                return True

        # Model scope
        if self._cfg.model_scope_model_id is not None:
            arr = self._cfg.model_scope_model_id.split("/")
            if len(arr) != 2:
                logger.error(
                    f"Invalid model scope model id: {self._cfg.model_scope_model_id}"
                )
                return False

            if arr[0].lower() == "gpustack" and "whisper" in arr[1].lower():
                return True

        # Huggingface and Model scope
        try:
            matching_files = match_files(
                huggingface_repo_id=self._cfg.huggingface_repo_id,
                huggingface_filename="model.bin",
                model_scope_model_id=self._cfg.model_scope_model_id,
                model_scope_file_path="model.bin",
            )
            if "model.bin" not in matching_files:
                return False
        except Exception as e:
            logger.debug(f"File model.bin does not exist, {e}")
            return False

        downloaded_files = []
        download_files = ["tokenizer.json", "preprocessor_config.json"]
        for f in download_files:
            try:
                downloaded_file_path = download_file(
                    huggingface_repo_id=self._cfg.huggingface_repo_id,
                    huggingface_filename=f,
                    model_scope_model_id=self._cfg.model_scope_model_id,
                    model_scope_file_path=f,
                    cache_dir=self._cfg.cache_dir,
                )
            except Exception as e:
                logger.debug(f"File {f} does not exist, {e}")
                continue

            downloaded_files.append(downloaded_file_path)

        if len(download_files) != 0:
            base_dir = os.path.dirname(downloaded_files[0])
            return self._check_local_model(base_dir, ["tokenizer.json"])

        return False
