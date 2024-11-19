from typing import Union
from speech_box.backends.stt.base import STTBackend
from speech_box.backends.stt.faster_whisper import FasterWhisper
from speech_box.backends.stt.funasr import FunASR
from speech_box.backends.tts.bark import Bark
from speech_box.backends.tts.base import TTSBackend
from speech_box.backends.tts.cosyvoice import CosyVoice
from speech_box.config.config import Config

_instance = None


class ModelInstance:
    def __init__(self, config: Config):
        self._config = config
        self._backends = [
            FunASR(config),
            FasterWhisper(config),
            Bark(config),
            CosyVoice(config),
        ]

    def run(self):
        global _instance

        if _instance is None:
            try:
                for b in self._backends:
                    if b.supported():
                        _instance = b.load()
                        break
            except Exception as e:
                raise Exception(f"Faild to load model, {e}")

            if _instance is None:
                raise Exception("Model isn't suppored")
        return _instance


def get_model_instance() -> Union[TTSBackend, STTBackend]:
    global _instance
    return _instance
