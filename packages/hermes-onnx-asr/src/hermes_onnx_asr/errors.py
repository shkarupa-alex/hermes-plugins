from __future__ import annotations


class OnnxAsrError(Exception):
    """A bounded, user-safe plugin error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


ERROR_MESSAGES = {
    "configuration_invalid": "ONNX ASR configuration is invalid.",
    "model_not_installed": "The selected speech recognition model is not installed.",
    "model_not_in_catalog": "The selected speech recognition model is not in the supported catalog.",
    "vad_not_installed": "The configured voice activity detector is not installed.",
    "ffmpeg_missing": "ffmpeg is required to convert this audio format.",
    "no_audio_stream": "The input does not contain a supported audio stream.",
    "decode_failed": "Audio conversion failed.",
    "insufficient_temp_space": "There is not enough temporary disk space for audio conversion.",
    "audio_too_long": "The recording exceeds the configured duration limit.",
    "no_speech_detected": "No speech detected.",
    "asr_queue_full": "The speech recognition queue is full. Try again later.",
    "model_switch_busy": "Cannot switch speech recognition models while work is queued.",
    "provider_shutting_down": "The speech recognition provider is shutting down.",
    "ffmpeg_timeout": "Audio conversion timed out.",
    "asr_timeout": "Speech recognition timed out.",
    "cpu_provider_violation": "A non-CPU ONNX execution provider was detected.",
    "model_load_failed": "The speech recognition model could not be loaded.",
    "transcription_failed": "Speech recognition failed.",
}


def safe_error(code: str) -> OnnxAsrError:
    return OnnxAsrError(code, ERROR_MESSAGES[code])
