from abc import ABC, abstractmethod
from typing import Dict
from vox_box.config.config import Config


class Estimator(ABC):
    def __init__(
        self,
        cfg: Config,
    ):
        pass

    @abstractmethod
    def model_info() -> Dict:
        pass
