import time
from typing import Dict


def create_model_dict(id: str, **kwargs) -> Dict:
    d = {
        "id": id,
        "object": "model",
        "created": int(time.time()),
        "owner": "vox-box",
        "backend": "vox-box",
    }

    for k, v in kwargs.items():
        if v is not None:
            d[k] = v

    return d
