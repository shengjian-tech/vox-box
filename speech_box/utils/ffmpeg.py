import subprocess


encoder_decoder_map = {
    "mp3": "libmp3lame",
    "opus": "libopus",
    "aac": "aac",
    "flac": "flac",
    "wav": "pcm_s16le",
    "pcm": "pcm_s16le",
}

output_format_map = {
    "mp3": "mp3",
    "opus": "ogg",
    "aac": "mp4",
    "flac": "flac",
    "wav": "wav",
    "pcm": "s16le",
}


def convert(
    input_file_path: str,
    response_format: str,
    converted_output_file_path: str,
    speed: float = 1,
):
    _command = [
        "ffmpeg",
        # Log level
        "-loglevel",
        "error",
        # Input file
        "-i",
        input_file_path,
        # Encoder/Decoder for audio streams
        "-c:a",
        encoder_decoder_map.get(response_format, "libmp3lame"),
        # Output format
        "-f",
        output_format_map.get(response_format, "mp3"),
        # Speed
        "-filter:a",
        f"atempo={speed}",
        # Output file overwrite without asking
        "-y",
        # Output file
        converted_output_file_path,
    ]

    result = None
    try:
        result = subprocess.run(
            _command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise Exception(
                f"Unexpected return code while execute command {_command}: {result.returncode}"
            )

    except Exception as e:
        raise Exception(
            f"Failed to execute {_command}: {e}"
            f", stdout: {result.stdout}, stderr: {result.stderr}"
            if result
            else ""
        )
