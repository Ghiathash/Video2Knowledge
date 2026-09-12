from dataclasses import dataclass
from enum import Enum


class ExecutionProfile(str, Enum):
    SMART = "smart"
    LOCAL = "local"
    CLOUD = "cloud"
    CUSTOM = "custom"


class ProviderKind(str, Enum):
    LOCAL_WHISPER = "local-whisper"
    GEMINI = "gemini"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class ProviderConfig:
    asr_provider: ProviderKind
    vision_provider: ProviderKind
    synthesis_provider: ProviderKind
    verification_provider: ProviderKind
    asr_model: str = "medium"
    vision_model: str | None = None
    synthesis_model: str | None = None
    verification_model: str | None = None
    device: str = "cuda"
    compute_type: str = "float16"

    def summary(self) -> dict[str, str]:
        labels = {
            ProviderKind.LOCAL_WHISPER: "Local (Faster Whisper)",
            ProviderKind.OLLAMA: "Local (Ollama)",
            ProviderKind.GEMINI: "Cloud (Gemini)",
        }
        return {
            "Speech Recognition": labels[self.asr_provider],
            "Visual Intelligence": labels[self.vision_provider],
            "Report Generation": labels[self.synthesis_provider],
            "Verification": labels[self.verification_provider],
        }
