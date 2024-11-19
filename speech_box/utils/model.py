import time
from typing import Dict


def create_model_dict(id: str, **kwargs) -> Dict:
    d = {
        "id": id,
        "object": "model",
        "created": int(time.time()),
        "owner": "speech-box",
    }

    for k, v in kwargs.items():
        d[k] = v

    return d
