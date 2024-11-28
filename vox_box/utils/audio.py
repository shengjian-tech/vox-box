import shutil
import tempfile
import av


response_format_to_encoder_decoder_map = {
    "mp3": "libmp3lame",
    "opus": "libopus",
    "aac": "aac",
    "flac": "flac",
    "wav": "pcm_s16le",
    "pcm": "pcm_s16le",
}

response_format_to_suffix_map = {
    "mp3": ".mp3",
    "opus": ".ogg",
    "aac": ".aac",
    "flac": ".flac",
    "wav": ".wav",
    "pcm": ".pcm",
}


def convert(
    input_file_path: str,
    response_format: str,
    speed: float = 1,
) -> str:
    suffix = response_format_to_suffix_map.get(response_format)
    with tempfile.NamedTemporaryFile(
        suffix=f"{suffix}", delete=False
    ) as output_temp_file:

        try:
            output_file_path = output_temp_file.name
            if response_format == "wav" and speed == 1:
                shutil.copy(input_file_path, output_file_path)
                return output_file_path

            input_container = av.open(input_file_path)
            input_stream = input_container.streams.audio[0]
            if response_format == "pcm":
                convert_to_pcm(input_stream, output_file_path, speed)
            else:
                convert_to_format(
                    input_stream, output_file_path, response_format, speed
                )

            input_container.close()
            return output_file_path
        except Exception as e:
            raise Exception(
                f"Failed to convert audio to format {response_format}, speed: {speed}: {e}"
            )


def convert_to_pcm(input_stream, output_file_path: str, speed: float):
    # Bare PCM data should not have any container structure, need to ensure the output is purely raw audio data stream.
    with open(output_file_path, "wb") as output_file:
        resampler = av.AudioResampler(
            format="s16",  # 16-bit PCM
            layout=input_stream.layout,
            rate=int(input_stream.rate * speed),
        )

        for frame in input_stream.container.decode(input_stream):
            frame.pts = None  # Reset PTS to avoid issues with frame timing
            resampled_frames = resampler.resample(frame)
            for resampled_frame in resampled_frames:
                # convert the audio frame into a NumPy array. The array format is usually (samples, channels),
                # where 'samples' is the number of sample points per frame, and 'channels' is the number of channels (e.g., stereo has 2 channels, mono has 1).
                pcm_data = resampled_frame.to_ndarray()
                # convert the NumPy array into a byte stream, then written to the file to generate raw PCM data.
                output_file.write(pcm_data.tobytes())


def convert_to_format(
    input_stream, output_file_path: str, response_format: str, speed: float
):
    output_rate = int(input_stream.rate * speed)
    codec_name = response_format_to_encoder_decoder_map.get(response_format)
    codec = av.codec.Codec(codec_name, "w")
    codec_supported_rate = codec.audio_rates
    if codec_supported_rate:
        output_rate = min(codec_supported_rate, key=lambda x: abs(x - output_rate))

    output_container = av.open(output_file_path, mode="w")
    output_stream = output_container.add_stream(
        codec_name=response_format_to_encoder_decoder_map.get(response_format),
        rate=output_rate,
        channels=input_stream.channels,
    )

    resampler = av.AudioResampler(
        format=output_stream.format,
        layout=output_stream.layout,
        rate=output_stream.rate,
    )

    for frame in input_stream.container.decode(input_stream):
        # Reset PTS to avoid issues with frame timing
        frame.pts = None
        frames = resampler.resample(frame)
        for resampled_frame in frames:
            for packet in output_stream.encode(resampled_frame):
                output_container.mux(packet)

    # Flush encoder
    for packet in output_stream.encode():
        output_container.mux(packet)

    output_container.close()
