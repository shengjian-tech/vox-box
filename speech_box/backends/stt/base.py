from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from speech_box.config.config import Config


class STTBackend(ABC):
    def __init__(
        self,
        cfg: Config,
    ):
        pass

    @abstractmethod
    def supported() -> bool:
        pass

    @abstractmethod
    def load() -> bool:
        pass

    @abstractmethod
    def model_info() -> Dict:
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.2,
        timestamp_granularities: Optional[List[str]] = None,
        response_format: str = "json",
        **kwargs
    ):
        pass
