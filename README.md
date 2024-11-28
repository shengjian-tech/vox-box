# Vox Box

A text-to-speech and speech-to-text server compatible with the OpenAI API, powered by backend support from Whisper, FunASR, Bark, and CosyVoice.

## Requirements

- Python 3.10 or greater
- Support Nvidia GPU, requires the following NVIDIA libraries to be installed:
  - [cuBLAS for CUDA 12](https://developer.nvidia.com/cublas)
  - [cuDNN 9 for CUDA 12](https://developer.nvidia.com/cudnn)  

## Installation

You can install the project using pip:

```bash
pip install vox-box

# For MacOS, you need to manually install `openfst`, `pynini`, and `wetextprocessing` after installing `vox-box` to make `cosyvoice` work:
brew install openfst
export CPLUS_INCLUDE_PATH=$(brew --prefix openfst)/include
export LIBRARY_PATH=$(brew --prefix openfst)/lib
pip install pynini==2.1.6
pip install wetextprocessing==1.0.4.1
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

## Supported Models

| Model                           | Type           | Link                                                                                                                                                                                        |
| ------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Faster-whisper-large-v3         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3), [ModelScope](https://www.modelscope.cn/models/iic/Whisper-large-v3)                                                 |
| Faster-whisper-large-v2         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v2)                                                                                                                      |
| Faster-whisper-large-v1         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v1)                                                                                                                      |
| Whisper-large-v3-turbo          | speech-to-text | [ModelScope](https://www.modelscope.cn/models/iic/Whisper-large-v3-turbo)                                                                                                                   |
| Faster-whisper-medium           | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium)                                                                                                                        |
| Faster-whisper-medium.en        | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium.en)                                                                                                                     |
| Faster-whisper-small            | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small)                                                                                                                         |
| Faster-whisper-small.en         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small.en)                                                                                                                      |
| Faster-distil-whisper-large-v3  | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v3)                                                                                                               |
| Faster-distil-whisper-large-v2  | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v2)                                                                                                               |
| Faster-distil-whisper-medium.en | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-medium.en)                                                                                                              |
| Faster-whisper-tiny             | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny)                                                                                                                          |
| Faster-whisper-tiny.en          | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny.en)                                                                                                                       |
| Paraformer-zh                   | speech-to-text | [Hugging Face](https://huggingface.co/funasr/paraformer-zh), [ModelScope](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) |
| Paraformer-zh-streaming         | speech-to-text | [Hugging Face](https://huggingface.co/funasr/paraformer-zh-streaming), [ModelScope](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online)     |
| Paraformer-en                   | speech-to-text | [Hugging Face](https://huggingface.co/funasr/paraformer-en), [ModelScope](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020)           |
| Conformer-en                    | speech-to-text | [Hugging Face](https://huggingface.co/funasr/conformer-en), [Modelscope](https://modelscope.cn/models/iic/speech_conformer_asr-en-16k-vocab4199-pytorch)                                    |
| SenseVoiceSmall                 | speech-to-text | [Hugging Face](https://huggingface.co/FunAudioLLM/SenseVoiceSmall), [ModelScope](https://www.modelscope.cn/models/iic/SenseVoiceSmall)                                                      |
| Bark                            | text-to-speech | [Hugging Face](https://huggingface.co/suno/bark)                                                                                                                                            |
| Bark-small                      | text-to-speech | [Hugging Face](https://huggingface.co/suno/bark-small)                                                                                                                                      |
| CosyVoice-300M-Instruct         | text-to-speech | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M-Instruct), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-Instruct)                                          |
| CosyVoice-300M-SFT              | text-to-speech | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-SFT)                                                    |
| CosyVoice-300M                  | text-to-speech | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M)                                                            |
| CosyVoice-300M-25Hz             | text-to-speech | [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-25Hz)                                                                                                                          |

## Supported APIs

### Create speech 

**Endpoint**: `POST /v1/audio/speech`

Generates audio from the input text. Compatible with the [OpenAI audio/speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech).

**Example Request**:
```bash
curl http://localhost/v1/audio/speech \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice",
    "input": "Hello world",
    "voice": "English Female"
  }' \
  --output speech.mp3
```

**Response**:
The audio file content.

### Create transcription 

**Endpoint**: `POST /v1/audio/transcriptions`

Transcribes audio into the input language. Compatible with the [OpenAI audio/transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

**Example Request**:
```bash
curl https://localhost/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/file/audio.mp3" \
  -F model="whisper-large-v3"
```

**Response**:
```json
{
  "text": "Hello world."
}
```

### List Models

**Endpoint**: `GET /v1/models`

Returns the current running models.

### Get Model

**Endpoint**: `GET /v1/models/{model_id}`

Returns the current running model.

### Get Voices

**Endpoint**: `GET /v1/voices`

Returns the supported voice for current running model.

### Health Check

**Endpoint**: `GET /health`

Returns the heath check result of the Vox Box.
