from typing import Dict, List
from vox_box.config.config import Config
from vox_box.elstimator.bark import Bark
from vox_box.elstimator.base import Elstimator
from vox_box.elstimator.cosyvoice import CosyVoice
from vox_box.elstimator.faster_whisper import FasterWhisper
from vox_box.elstimator.funasr import FunASR
from vox_box.utils.model import create_model_dict


def estimate_model(cfg: Config) -> Dict:
    elstimator: List[Elstimator] = [
        FasterWhisper(cfg),
        FunASR(cfg),
        CosyVoice(cfg),
        Bark(cfg),
    ]

    model = cfg.model or cfg.huggingface_repo_id or cfg.model_scope_model_id
    model_info = create_model_dict(
        model,
        supported=False,
    )
    for e in elstimator:
        model_info = e.model_info()
        if model_info["supported"]:
            return model_info
