from typing import Dict, List
from speech_box.config.config import Config
from speech_box.elstimator.bark import Bark
from speech_box.elstimator.base import Elstimator
from speech_box.elstimator.cosyvoice import CosyVoice
from speech_box.elstimator.faster_whisper import FasterWhisper
from speech_box.elstimator.funasr import FunASR
from speech_box.utils.model import create_model_dict


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
