import os
import re
import sys
import wave
import numpy as np
import tempfile
import torch
from typing import Dict, List, Optional

from vox_box.backends.tts.base import TTSBackend
from vox_box.utils.log import log_method
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.utils.audio import convert
from vox_box.utils.model import create_model_dict
import logging

paths_to_insert = [
    os.path.join(os.path.dirname(__file__), "../../third_party/CosyVoice"),
    os.path.join(
        os.path.dirname(__file__), "../../third_party/CosyVoice/third_party/Matcha-TTS"
    ),
]

builtin_spk2info_path = os.path.join(os.path.dirname(__file__), "cosyvoice_spk2info.pt")

logger = logging.getLogger(__name__)

class CosyVoice(TTSBackend):
    def __init__(
        self,
        cfg: Config,
    ):
        self.model_load = False
        self.language_map = {
            "中文女": "Chinese Female",
            "中文男": "Chinese Male",
            "日语男": "Japanese Male",
            "粤语女": "Cantonese Female",
            "英文女": "English Female",
            "英文男": "English Male",
            "韩语女": "Korean Female",
        }
        self.reverse_language_map = {v: k for k, v in self.language_map.items()}
        self._cfg = cfg
        self._voices = None
        self._model = None
        self._model_dict = {}
        self._is_cosyvoice_v2 = False
        #add code
        # self._prompt_wav = "/agi/gpustack/xh.mp3"
        # self._prompt_text = "今天天氣真是太好了，陽光燦爛心情超級棒，但是朋友最近的感情問題也讓我心痛不已，好像世界末日一樣，真的好為他難過喔。"
        # self._prompt_speech_16k = None

        # 配置每个发音人的语音路径和提示语
        self.custom_voice_configs = {
            "zh_female_wanwanxiaohe": {
                "prompt_wav": "/agi/gpustack/xh.mp3",
                "prompt_text": "今天天氣真是太好了，陽光燦爛心情超級棒，但是朋友最近的感情問題也讓我心痛不已，好像世界末日一樣，真的好為他難過喔。",
            },
            "zh_female_gaolengyujie": {
                "prompt_wav": "/agi/gpustack/xh_red.mp3",
                "prompt_text": "亲爱的，我们到此为止吧，我倦了，不想再继续这场无聊的游戏了，你我之间，也许曾经有一些美好，但那也只是曾经罢了。",
            },
            "zh_male_yangguangqingnian": {
                "prompt_wav": "/agi/gpustack/xm.mp3",
                "prompt_text": "今天又是超棒的一天呀，阳光这么好，心情也跟着超级美丽呢，生活嘛，就该充满活力和欢笑呀，我呀，要像那灿烂的阳光一样，永远积极向上，去追寻自己的梦想，去体验各种好玩的事情，去认识更多有趣的人。",
            }
        }



        cosyvoice_yaml_path = os.path.join(self._cfg.model, "cosyvoice.yaml")
        if os.path.exists(cosyvoice_yaml_path):
            with open(cosyvoice_yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
                if re.search(r"Qwen2", content, re.IGNORECASE):
                    self._is_cosyvoice_v2 = True

    def load(self):
        for path in paths_to_insert:
            sys.path.insert(0, path)

        if self.model_load:
            return self

        #add code
        from cosyvoice.utils.file_utils import load_wav
        if self._is_cosyvoice_v2:
            from cosyvoice.cli.cosyvoice import CosyVoice2 as CosyVoiceModel2

            self._model = CosyVoiceModel2(self._cfg.model)

            # CosyVoice2 does not have builtin spk2info.pt
            if not self._model.frontend.spk2info:
                self._model.frontend.spk2info = torch.load(
                    builtin_spk2info_path, map_location=self._cfg.device
                )
        else:
            from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceModel

            self._model = CosyVoiceModel(self._cfg.model)

        self._voices = self._get_voices()
        self._model_dict = create_model_dict(
            self._cfg.model,
            task_type=TaskTypeEnum.TTS,
            backend_framework=BackendEnum.COSY_VOICE,
            voices=self._voices,
        )
        
        # 加载自定义音频
        # self._prompt_speech_16k = load_wav(self._prompt_wav, 16000)

        # 创建 custom_voice_prompts 字典用于保存加载后的数据
        self.custom_voice_prompts = {}
        # 加载每个发音人的音频和文本
        for speaker, config in self.custom_voice_configs.items():
            prompt_wav_path = config["prompt_wav"]
            
            if not os.path.exists(prompt_wav_path):
                print(f"警告：文件不存在 - {prompt_wav_path}")
                continue
            
            try:
                print("加载路径->", prompt_wav_path)
                prompt_speech_16k = load_wav(prompt_wav_path, 16000)
            except Exception as e:
                print(f"加载音频失败：{prompt_wav_path}，错误：{e}")
                continue

            self.custom_voice_prompts[speaker] = {
                "prompt_text": config["prompt_text"],
                "prompt_speech_16k": prompt_speech_16k
            }
        
        self.model_load = True
        return self

    def is_load(self) -> bool:
        return self.model_load

    def model_info(self) -> Dict:
        return self._model_dict

    @log_method
    def speech(
        self,
        input: str,
        voice: Optional[str] = "Chinese Female",
        speed: float = 1,
        reponse_format: str = "mp3",
        **kwargs,
    ) -> str:
        if voice not in self._voices:
            raise ValueError(f"Voice {voice} not supported")

        # 根据发言人选择调用方式
        if voice in self.language_map.values():
            original_voice = self._get_original_voice(voice)
            model_output = self._model.inference_sft(
            input, original_voice, stream=False, speed=speed
            )
        else:
            prompt_text  = self.custom_voice_prompts[voice]["prompt_text"]
            prompt_speech_16k = self.custom_voice_prompts[voice]["prompt_speech_16k"]
            print("prompt_text", self.custom_voice_prompts[voice]["prompt_text"])
            print("prompt_speech_16k", self.custom_voice_prompts[voice]["prompt_speech_16k"])
            model_output = self._model.inference_zero_shot(
                input, prompt_text, prompt_speech_16k, stream=False, speed=speed
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            wav_file_path = temp_file.name
            with wave.open(wav_file_path, "wb") as wf:
                wf.setnchannels(1)  # single track
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(22050)  # Sample rate
                #wf.setframerate(24000)  # Sample rate
                for i in model_output:
                    tts_audio = (
                        (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
                    )
                    wf.writeframes(tts_audio)

                output_file_path = convert(wav_file_path, reponse_format, speed)
                return output_file_path

    def _get_voices(self) -> List[str]:
        voices = self._model.list_available_spks()
        # 默认的音色
        arr1 = [self.language_map.get(voice, voice) for voice in voices]
        # 自定义的音色
        arr2 =  [key for key in self.custom_voice_configs]
        return arr2 + arr1
        # return [self.language_map.get(voice, voice) for voice in voices]

    def _get_original_voice(self, voice: str) -> str:
        return self.reverse_language_map.get(voice, voice)
