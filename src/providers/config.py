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
    OPENAI_COMPATIBLE = "openai-compatible"


@dataclass(frozen=True)
class ProviderConfig:
    """Independent model/backend configuration for every pipeline role."""

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
    vision_base_url: str | None = None
    synthesis_base_url: str | None = None
    verification_base_url: str | None = None
    vision_api_key: str | None = None
    synthesis_api_key: str | None = None
    verification_api_key: str | None = None

    def summary(self) -> dict[str, str]:
        def label(kind: ProviderKind, model: str | None) -> str:
            backend = {
                ProviderKind.LOCAL_WHISPER: "Faster Whisper",
                ProviderKind.OLLAMA: "Ollama",
                ProviderKind.GEMINI: "Gemini",
                ProviderKind.OPENAI_COMPATIBLE: "OpenAI-compatible",
            }[kind]
            return f"{model} via {backend}" if model else backend

        return {
            "Speech Recognition": label(self.asr_provider, self.asr_model),
            "Visual Intelligence": label(self.vision_provider, self.vision_model),
            "Report Generation": label(self.synthesis_provider, self.synthesis_model),
            "Verification": label(self.verification_provider, self.verification_model),
        }
