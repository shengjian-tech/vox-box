import logging
from typing import List, Optional
from pathlib import Path
import fnmatch
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import validate_repo_id
from modelscope.hub.api import HubApi

logger = logging.getLogger(__name__)


def match_files(
    huggingface_repo_id: Optional[str] = None,
    huggingface_filename: Optional[str] = None,
    model_scope_model_id: Optional[str] = None,
    model_scope_file_path: Optional[str] = None,
) -> List[str]:
    if huggingface_repo_id is not None:
        return match_hugging_face_files(huggingface_repo_id, huggingface_filename)
    elif model_scope_model_id is not None:
        return match_model_scope_file_paths(model_scope_model_id, model_scope_file_path)


def match_hugging_face_files(repo_id: str, filename: str) -> List[str]:
    validate_repo_id(repo_id)

    hffs = HfFileSystem()

    files = [
        file["name"] if isinstance(file, dict) else file for file in hffs.ls(repo_id)
    ]

    file_list: List[str] = []
    for file in files:
        rel_path = Path(file).relative_to(repo_id)
        file_list.append(str(rel_path))

    matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore
    matching_files = sorted(matching_files)
    return matching_files


def match_model_scope_file_paths(model_id: str, file_path: str) -> List[str]:
    if "/" in file_path:
        root, _ = file_path.rsplit("/", 1)
    else:
        root = None

    api = HubApi()
    files = api.get_model_files(model_id, root=root)

    file_paths = [file["Path"] for file in files]
    matching_paths = [p for p in file_paths if fnmatch.fnmatch(p, file_path)]
    matching_paths = sorted(matching_paths)
    return matching_paths
