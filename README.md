# Vox Box

A text-to-speech and speech-to-text server compatible with the OpenAI API, powered by backend support from Whisper, FunASR, Bark, and CosyVoice.

## Installation

You can install the project using pip:

```bash
pip install vox-box
```

## Usage

```
vox-box start --model --huggingface-repo-id Systran/faster-whisper-small --data-dir ./cache/data-dir --host 0.0.0.0 --port 80
```

### Options
- -d, --debug: Enable debug mode.
- --host: Host to bind the server to. Default is 0.0.0.0.
- --port: Port to bind the server to. Default is 80.
- --model: model path.
- --device: Binding device, e.g., cuda:0. Default is cpu.
- --huggingface-repo-id: Huggingface repo id for the model.
- --model-scope-model-id: Model scope model id for the model.
- --data-dir: Directory to store downloaded model data. Default is OS specific.

## Supported Backends

The project supports the following backends:

- FunASR
- Faster-Whisper
- Bark
- CosyVoice

All models supported by these backends can be deployed with this project.

### Supported Models

- [FunASR](https://github.com/modelscope/FunASR?tab=readme-ov-file#model-zoo)
- [Faster-Whisper](https://huggingface.co/Systran)
- [Bark](https://huggingface.co/suno)
- [CosyVoice](https://modelscope.cn/collections/CosyVoice-1a4baea39a135)
