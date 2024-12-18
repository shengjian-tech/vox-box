import fnmatch
import logging
import os
from filelock import FileLock
from typing import Literal, Optional, Union
from pathlib import Path
from tqdm.contrib.concurrent import thread_map

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import (
    snapshot_download as modelscope_snapshot_download,
)
from modelscope.hub.utils.utils import model_id_to_group_owner_name
from vox_box.downloader.hub import (
    match_hugging_face_files,
    match_model_scope_file_paths,
)

logger = logging.getLogger(__name__)


def download_file(
    huggingface_repo_id: Optional[str] = None,
    huggingface_filename: Optional[str] = None,
    model_scope_model_id: Optional[str] = None,
    model_scope_file_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    huggingface_token: Optional[str] = None,
) -> str:
    result_path = None
    key = None

    if huggingface_repo_id is not None:
        key = (
            f"huggingface:{huggingface_repo_id}"
            if huggingface_filename is None
            else f"huggingface:{huggingface_repo_id}:{huggingface_filename}"
        )
        logger.debug(f"Downloading {key}")

        result_path = HfDownloader.download(
            repo_id=huggingface_repo_id,
            filename=huggingface_filename,
            token=huggingface_token,
            cache_dir=os.path.join(cache_dir, "huggingface"),
        )
    elif model_scope_model_id is not None:
        key = (
            f"modelscope:{model_scope_model_id}"
            if model_scope_file_path is None
            else f"modelscope:{model_scope_model_id}:{model_scope_file_path}"
        )
        logger.debug(f"Downloading {key}")

        result_path = ModelScopeDownloader.download(
            model_id=model_scope_model_id,
            file_path=model_scope_file_path,
            cache_dir=os.path.join(cache_dir, "model_scope"),
        )

    logger.debug(f"Downloaded {key}")
    return result_path


def get_file_size(
    huggingface_repo_id: Optional[str] = None,
    huggingface_filename: Optional[str] = None,
    model_scope_model_id: Optional[str] = None,
    model_scope_file_path: Optional[str] = None,
    huggingface_token: Optional[str] = None,
) -> str:
    if huggingface_repo_id is not None:
        return HfDownloader.get_file_size(
            repo_id=huggingface_repo_id,
            filename=huggingface_filename,
            token=huggingface_token,
        )
    elif model_scope_model_id is not None:
        return ModelScopeDownloader.get_file_size(
            model_id=model_scope_model_id,
            file_path=model_scope_file_path,
        )


class HfDownloader:
    _registry_url = "https://huggingface.co"

    @classmethod
    def get_file_size(cls, repo_id: str, filename: str, token: Optional[str]) -> int:
        api = HfApi(token=token)
        repo_info = api.repo_info(repo_id, files_metadata=True)
        total_size = sum(
            sibling.size
            for sibling in repo_info.siblings
            if fnmatch.fnmatch(sibling.rfilename, filename) and sibling.size is not None
        )

        return total_size

    @classmethod
    def download(
        cls,
        repo_id: str,
        filename: Optional[str],
        token: Optional[str] = None,
        local_dir: Optional[Union[str, os.PathLike[str]]] = None,
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        max_workers: int = 8,
    ) -> str:
        """Download a model from the Hugging Face Hub.

        Args:
            repo_id:
                The model repo id.
            filename:
                A filename or glob pattern to match the model file in the repo.
            token:
                The Hugging Face API token.
            local_dir:
                The local directory to save the model to.
            local_dir_use_symlinks:
                Whether to use symlinks when downloading the model.
            max_workers (`int`, *optional*):
                Number of concurrent threads to download files (1 thread = 1 file download).
                Defaults to 8.

        Returns:
            The path to the downloaded model.
        """

        if filename is not None:
            return cls.download_file(
                repo_id, filename, token, local_dir, local_dir_use_symlinks, cache_dir
            )

        return snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            cache_dir=cache_dir,
        )

    @classmethod
    def download_file(
        cls,
        repo_id: str,
        filename: Optional[str],
        token: Optional[str] = None,
        local_dir: Optional[Union[str, os.PathLike[str]]] = None,
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        max_workers: int = 8,
    ) -> str:
        """Download a model from the Hugging Face Hub.
        Args:
            repo_id: The model repo id.
            filename: A filename or glob pattern to match the model file in the repo.
            token: The Hugging Face API token.
            local_dir: The local directory to save the model to.
            local_dir_use_symlinks: Whether to use symlinks when downloading the model.
        Returns:
            The path to the downloaded model.
        """

        matching_files = match_hugging_face_files(repo_id, filename)

        if len(matching_files) == 0:
            raise ValueError(f"No file found in {repo_id} that match {filename}")

        subfolder, first_filename = (
            str(Path(matching_files[0]).parent),
            Path(matching_files[0]).name,
        )

        unfolder_matching_files = [Path(file).name for file in matching_files]

        def _inner_hf_hub_download(repo_file: str):
            return hf_hub_download(
                repo_id=repo_id,
                filename=repo_file,
                token=token,
                subfolder=subfolder,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                cache_dir=cache_dir,
            )

        thread_map(
            _inner_hf_hub_download,
            unfolder_matching_files,
            desc=f"Fetching {len(unfolder_matching_files)} files",
            max_workers=max_workers,
        )

        # Get local path of the model file.
        # For split files, get the first one. llama-box will handle the rest.
        if local_dir is None:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=first_filename,
                token=token,
                subfolder=subfolder,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        else:
            model_path = os.path.join(local_dir, first_filename)

        return model_path

    def __call__(self):
        return self.download()


class ModelScopeDownloader:
    @classmethod
    def get_file_size(
        cls,
        model_id: str,
        file_path: Optional[str],
    ) -> int:
        api = HubApi()
        repo_files = api.get_model_files(model_id, recursive=True)
        total_size = sum(
            sibling.get("Size")
            for sibling in repo_files
            if fnmatch.fnmatch(sibling.get("Path", ""), file_path) and "Size" in sibling
        )

        return total_size

    @classmethod
    def download(
        cls,
        model_id: str,
        file_path: Optional[str],
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
    ) -> str:
        """Download a model from Model Scope.

        Args:
            model_id:
                The model id.
            file_path:
                A filename or glob pattern to match the model file in the repo.
            cache_dir:
                The cache directory to save the model to.

        Returns:
            The path to the downloaded model.
        """

        group_or_owner, name = model_id_to_group_owner_name(model_id)
        name = name.replace(".", "___")
        lock_filename = os.path.join(cache_dir, group_or_owner, f"{name}.lock")

        with FileLock(lock_filename):
            if file_path is not None:
                matching_files = match_model_scope_file_paths(model_id, file_path)
                if len(matching_files) == 0:
                    raise ValueError(
                        f"No file found in {model_id} that match {file_path}"
                    )

                model_path = modelscope_snapshot_download(
                    model_id=model_id,
                    cache_dir=cache_dir,
                    allow_patterns=file_path,
                )
                return os.path.join(model_path, matching_files[0])

            return modelscope_snapshot_download(
                model_id=model_id,
                cache_dir=cache_dir,
            )
