from pathlib import Path

from src.transcription.transcriber import transcribe_audio


class LocalWhisperProvider:
    name = "Local Faster Whisper"

    def __init__(self, model: str, device: str, compute_type: str):
        self.model = model
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, audio_path: Path, *, language: str | None):
        return transcribe_audio(
            audio_path, model_size=self.model, device=self.device,
            compute_type=self.compute_type, language=language,
        )
