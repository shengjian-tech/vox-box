import dataclasses
import json
import logging
import os
import platform
from typing import Dict, List, Optional
import tempfile
import io
from speech_box.backends.stt.base import STTBackend
from speech_box.config.config import Config
from speech_box.utils.file import get_file_size_in_byte
from speech_box.utils.log import log_method
from speech_box.utils.model import create_model_dict
from faster_whisper.transcribe import WhisperModel

logger = logging.getLogger(__name__)


class FasterWhisper(STTBackend):
    def __init__(
        self,
        cfg: Config,
    ):
        self._cfg = cfg
        self.model_load = False
        self._cfg = cfg
        self._resource_required = None
        self._model = None
        self._model_dict = {}

        self._preprocessor_config_json = None
        preprocessor_config_path = os.path.join(
            self._cfg.model, "preprocessor_config.json"
        )
        if os.path.exists(preprocessor_config_path):
            with open(preprocessor_config_path, "r", encoding="utf-8") as f:
                self._preprocessor_config_json = json.load(f)

        self._supported = self._supported()

    def task_type():
        return "stt"

    def load(self):
        if self.model_load:
            return self

        cpu_threads = 0
        if self._cfg.device == "cpu":
            cpu_threads = 8

        compute_type = "default"
        if platform.system() == "Darwin":
            compute_type = "int8"

        self._model = WhisperModel(
            self._cfg.model,
            self._cfg.device,
            cpu_threads=cpu_threads,
            compute_type=compute_type,
        )

        resource_required = self._get_required_resource()

        self._model_dict = create_model_dict(
            self._cfg.model, resource_required=resource_required
        )
        self.model_load = True
        return self

    def model_info(self) -> Dict:
        return self._model_dict

    def supported(self) -> bool:
        return self._supported

    def _supported(self) -> bool:
        model_bin_path = os.path.join(self._cfg.model, "model.bin")
        if not os.path.exists(model_bin_path):
            return False

        tokenizer_path = os.path.join(self._cfg.model, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            return False

        if self._preprocessor_config_json is not None:
            processor_class = self._preprocessor_config_json.get("processor_class")
            if processor_class is not None and processor_class != "WhisperProcessor":
                return False

        try:
            self.load()
            return True
        except Exception as e:
            logger.error(f"Faild to load model, {e}")
            return False

    @log_method
    def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        timestamp_granularities: Optional[List[str]] = ["segment"],
        response_format: str = "json",
        **kwargs,
    ):

        if language == "auto":
            language = None
            # Accept:
            # af, am, ar, as, az, ba, be, bg, bn, bo, br, bs, ca, cs, cy, da, de,
            # el, en, es, et, eu, fa, fi, fo, fr, gl, gu, ha, haw, he, hi, hr, ht,
            # hu, hy, id, is, it, ja, jw, ka, kk, km, kn, ko, la, lb, ln, lo, lt,
            # lv, mg, mi, mk, ml, mn, mr, ms, mt, my, ne, nl, nn, no, oc, pa, pl,
            # ps, pt, ro, ru, sa, sd, si, sk, sl, sn, so, sq, sr, su, sv, sw, ta,
            # te, tg, th, tk, tl, tr, tt, uk, ur, uz, vi, yi, yo, zh, yue

        without_timestamps = True
        word_timestamps = False
        if response_format == "verbose_json" and timestamp_granularities is not None:
            without_timestamps = False
            if "word" in timestamp_granularities:
                word_timestamps = True

        with tempfile.NamedTemporaryFile(buffering=0) as f:
            f.write(audio)

            segs, info = self._model.transcribe(
                f.name,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
                without_timestamps=without_timestamps,
                word_timestamps=word_timestamps,
                **kwargs,
            )

            # The transcription will actually run here.
            timestamps = []
            text_buffer = io.StringIO()
            for seg in segs:
                text_buffer.write(seg.text)

                if not without_timestamps:

                    if word_timestamps:
                        for wd in seg.words:
                            timestamps.append(dataclasses.asdict(wd))
                    else:
                        timestamps.append(dataclasses.asdict(seg))

            text = text_buffer.getvalue()
            if without_timestamps:
                return text

            response = {
                "task": "transcribe",
                "language": info.language,
                "duration": info.duration,
                "text": text,
            }
            if word_timestamps:
                response["words"] = timestamps
            else:
                response["segments"] = timestamps

            return response

    def _get_required_resource(self) -> Dict:
        """
        File size from https://huggingface.co/Systran
        | large            | Size   | Size (MiB/GiB) |
        | ---------------- | ------ | -------------- |
        | tiny en          | 75MB   | 71.53 MiB      |
        | tiny             | 75MB   | 71.53 MiB      |
        | base en          | 145MB  | 138.67 MiB     |
        | base             | 145MB  | 138.67 MiB     |
        | distil small en  | 332MB  | 316.41 MiB     |
        | small en         | 484MB  | 461.91 MiB     |
        | small            | 484MB  | 461.91 MiB     |
        | distil medium en | 789MB  | 752.93 MiB     |
        | medium en        | 1.53G  | 1.42 GiB       |
        | medium           | 1.52G  | 1.41 GiB       |
        | distil large v2  | 1.51G  | 1.41 GiB       |
        | distil large v3  | 1.51G  | 1.41 GiB       |
        | large v3         | 3.09GB | 2.88 GiB       |
        | large v2         | 3.09GB | 2.88 GiB       |
        | large v1         | 3.09GB | 2.88 GiB       |

        Resource required from:
        https://github.com/openai/whisper?tab=readme-ov-file
        https://github.com/cinprens/Whisper-GUI/tree/main
        https://huggingface.co/Systran/faster-distil-whisper-large-v2

        | Size         | Parameters | English-only model | Multilingual model | Required VRAM | Required Ram | Relative speed |
        | ------------ | ---------- | ------------------ | ------------------ | ------------- | ------------ | -------------- |
        | tiny         | 39 M       | tiny.en            | tiny               | ~1 GB         | ~2 GB        | ~10x           |
        | base         | 74 M       | base.en            | base               | ~1 GB         | ~2 GB        | ~7x            |
        | small        | 244 M      | small.en           | small              | ~2 GB         | ~4 GB        | ~4x            |
        | medium       | 769 M      | medium.en          | medium             | ~5 GB         | ~8 GB        | ~2x            |
        | distil-large | 756 M      | N/A                | distil-large       |               |              |                |
        | large        | 1550 M     | N/A                | large              | ~10 GB        | ~16 GB       | 1x             |
        """
        Mib = 1024 * 1024
        Gib = 1024 * 1024 * 1024
        resource_requirements = {
            # tiny
            100 * Mib: {"cuda": {"vram": 1 * Gib}, "cpu": {"ram": 2 * Gib}},
            # base
            200 * Mib: {"cuda": {"vram": 1 * Gib}, "cpu": {"ram": 2 * Gib}},
            # small
            500 * Mib: {"cuda": {"vram": 2 * Gib}, "cpu": {"ram": 4 * Gib}},
            # medium
            1.6 * Gib: {"cuda": {"vram": 5 * Gib}, "cpu": {"ram": 8 * Gib}},
            # large
            3.1 * Gib: {"cuda": {"vram": 10 * Gib}, "cpu": {"ram": 16 * Gib}},
        }

        file_size_in_byte = get_file_size_in_byte(
            os.path.join(self._cfg.model, "model.bin")
        )
        for size, resource in resource_requirements.items():
            if file_size_in_byte <= size:
                return resource
