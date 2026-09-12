from dataclasses import dataclass
import os

from src.providers.asr import LocalWhisperProvider
from src.providers.config import ExecutionProfile, ProviderConfig, ProviderKind
from src.providers.gemini import GeminiLLMProvider, GeminiVerificationProvider, GeminiVisionProvider
from src.providers.ollama import OllamaLLMProvider, OllamaVerificationProvider, OllamaVisionProvider
from src.providers.hardware import detect_hardware


@dataclass(frozen=True)
class ProviderBundle:
    config: ProviderConfig
    asr: object
    vision: object
    synthesis: object
    verification: object


def _gemini_ready(env: dict[str, str]) -> bool:
    return bool(env.get("GEMINI_API_KEY"))


def _local_ready(env: dict[str, str]) -> bool:
    return bool(env.get("LOCAL_VISION_MODEL") and env.get("LOCAL_LLM_MODEL"))


def resolve_profile(
    profile: ExecutionProfile,
    custom: ProviderConfig | None = None,
    env: dict[str, str] | None = None,
) -> ProviderConfig:
    env = dict(os.environ) if env is None else env
    asr_model = env.get("LOCAL_ASR_MODEL", "medium")
    hardware = detect_hardware()
    device = env.get("LOCAL_ASR_DEVICE", "cuda" if hardware.cuda_available else "cpu")
    compute = env.get("LOCAL_ASR_COMPUTE_TYPE", "float16" if device == "cuda" else "int8")
    if profile is ExecutionProfile.CUSTOM:
        if custom is None:
            raise ValueError("Advanced mode requires an explicit provider configuration.")
        selected = (custom.vision_provider, custom.synthesis_provider, custom.verification_provider)
        if ProviderKind.GEMINI in selected and not _gemini_ready(env):
            raise ValueError("Gemini is selected but GEMINI_API_KEY is not configured.")
        if custom.vision_provider is ProviderKind.OLLAMA and not custom.vision_model:
            raise ValueError("Local vision requires an installed Ollama vision model.")
        if custom.synthesis_provider is ProviderKind.OLLAMA and not custom.synthesis_model:
            raise ValueError("Local synthesis requires an installed Ollama language model.")
        if custom.verification_provider is ProviderKind.OLLAMA and not custom.verification_model:
            raise ValueError("Local verification requires an installed Ollama language model.")
        return custom
    if profile is ExecutionProfile.SMART:
        if _gemini_ready(env):
            profile = ExecutionProfile.CLOUD
        elif _local_ready(env):
            profile = ExecutionProfile.LOCAL
        else:
            raise ValueError("No cloud credentials or ready local vision/language models were found.")
    if profile is ExecutionProfile.CLOUD:
        if not _gemini_ready(env):
            raise ValueError("Cloud AI is not configured. Set GEMINI_API_KEY.")
        return ProviderConfig(
            ProviderKind.LOCAL_WHISPER, ProviderKind.GEMINI, ProviderKind.GEMINI, ProviderKind.GEMINI,
            asr_model, env.get("GEMINI_VISION_MODEL") or env.get("GEMINI_MODEL"),
            env.get("GEMINI_LLM_MODEL") or env.get("GEMINI_MODEL"),
            env.get("GEMINI_VERIFICATION_MODEL") or env.get("GEMINI_MODEL"), device, compute,
        )
    if not _local_ready(env):
        raise ValueError("Private / Offline mode requires LOCAL_VISION_MODEL and LOCAL_LLM_MODEL installed in Ollama.")
    return ProviderConfig(
        ProviderKind.LOCAL_WHISPER, ProviderKind.OLLAMA, ProviderKind.OLLAMA, ProviderKind.OLLAMA,
        asr_model, env["LOCAL_VISION_MODEL"], env["LOCAL_LLM_MODEL"],
        env.get("LOCAL_VERIFICATION_MODEL", env["LOCAL_LLM_MODEL"]), device, compute,
    )


def build_providers(config: ProviderConfig) -> ProviderBundle:
    if config.asr_provider is not ProviderKind.LOCAL_WHISPER:
        raise ValueError("Only Local Faster Whisper ASR is currently implemented.")
    asr = LocalWhisperProvider(config.asr_model, config.device, config.compute_type)
    if config.vision_provider is ProviderKind.GEMINI:
        vision = GeminiVisionProvider(config.vision_model)
    elif config.vision_provider is ProviderKind.OLLAMA:
        vision = OllamaVisionProvider(config.vision_model or "")
    else:
        raise ValueError("Unsupported vision provider.")
    synthesis = (
        GeminiLLMProvider(config.synthesis_model)
        if config.synthesis_provider is ProviderKind.GEMINI
        else OllamaLLMProvider(config.synthesis_model or "")
    )
    verification = (
        GeminiVerificationProvider(config.verification_model)
        if config.verification_provider is ProviderKind.GEMINI
        else OllamaVerificationProvider(config.verification_model or "")
    )
    return ProviderBundle(config, asr, vision, synthesis, verification)
