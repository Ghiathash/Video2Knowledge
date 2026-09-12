from dataclasses import dataclass
import os

from src.providers.asr import LocalWhisperProvider
from src.providers.config import ExecutionProfile, ProviderConfig, ProviderKind
from src.providers.gemini import GeminiLLMProvider, GeminiVerificationProvider, GeminiVisionProvider
from src.providers.hardware import detect_hardware
from src.providers.ollama import OllamaLLMProvider, OllamaVerificationProvider, OllamaVisionProvider
from src.providers.openai_compatible import OpenAICompatibleLLMProvider, OpenAICompatibleVerificationProvider, OpenAICompatibleVisionProvider


@dataclass(frozen=True)
class ProviderBundle:
    config: ProviderConfig
    asr: object
    vision: object
    synthesis: object
    verification: object


def _runtime(env: dict[str, str]) -> tuple[str, str, str]:
    hardware = detect_hardware()
    device = env.get("LOCAL_ASR_DEVICE", "cuda" if hardware.cuda_available else "cpu")
    compute = env.get("LOCAL_ASR_COMPUTE_TYPE", "float16" if device == "cuda" else "int8")
    return env.get("LOCAL_ASR_MODEL", "medium"), device, compute


def _validate(config: ProviderConfig, env: dict[str, str], *, private: bool = False) -> ProviderConfig:
    roles = (
        ("vision", config.vision_provider, config.vision_model, config.vision_base_url, config.vision_api_key),
        ("synthesis", config.synthesis_provider, config.synthesis_model, config.synthesis_base_url, config.synthesis_api_key),
        ("verification", config.verification_provider, config.verification_model, config.verification_base_url, config.verification_api_key),
    )
    for role, backend, model, base_url, api_key in roles:
        if private and backend is not ProviderKind.OLLAMA:
            raise ValueError("Private / Offline mode never permits cloud backends.")
        if not model:
            raise ValueError(f"A {role} model must be selected.")
        if backend is ProviderKind.GEMINI and not (api_key or env.get("GEMINI_API_KEY")):
            raise ValueError(f"The selected Gemini {role} model needs GEMINI_API_KEY or a session key.")
        if backend is ProviderKind.OPENAI_COMPATIBLE and not base_url:
            raise ValueError(f"The selected {role} model needs an OpenAI-compatible base URL.")
    return config


def _local_config(env: dict[str, str]) -> ProviderConfig:
    asr, device, compute = _runtime(env)
    vision = env.get("OLLAMA_VISION_MODEL") or env.get("LOCAL_VISION_MODEL")
    language = env.get("OLLAMA_LANGUAGE_MODEL") or env.get("LOCAL_LLM_MODEL")
    base = env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return ProviderConfig(
        ProviderKind.LOCAL_WHISPER, ProviderKind.OLLAMA, ProviderKind.OLLAMA, ProviderKind.OLLAMA,
        asr, vision, language, env.get("OLLAMA_VERIFICATION_MODEL") or env.get("LOCAL_VERIFICATION_MODEL") or language,
        device, compute, base, base, base,
    )


def _cloud_config(env: dict[str, str]) -> ProviderConfig | None:
    asr, device, compute = _runtime(env)
    if env.get("GEMINI_API_KEY"):
        default = env.get("GEMINI_MODEL", "gemini-2.5-flash")
        return ProviderConfig(ProviderKind.LOCAL_WHISPER, ProviderKind.GEMINI, ProviderKind.GEMINI, ProviderKind.GEMINI, asr, env.get("GEMINI_VISION_MODEL", default), env.get("GEMINI_LLM_MODEL", default), env.get("GEMINI_VERIFICATION_MODEL", default), device, compute)
    if env.get("OPENAI_COMPATIBLE_BASE_URL") and env.get("OPENAI_COMPATIBLE_MODEL"):
        url, model, key = env["OPENAI_COMPATIBLE_BASE_URL"], env["OPENAI_COMPATIBLE_MODEL"], env.get("OPENAI_COMPATIBLE_API_KEY")
        return ProviderConfig(ProviderKind.LOCAL_WHISPER, ProviderKind.OPENAI_COMPATIBLE, ProviderKind.OPENAI_COMPATIBLE, ProviderKind.OPENAI_COMPATIBLE, asr, env.get("OPENAI_COMPATIBLE_VISION_MODEL", model), env.get("OPENAI_COMPATIBLE_LANGUAGE_MODEL", model), env.get("OPENAI_COMPATIBLE_VERIFICATION_MODEL", model), device, compute, url, url, url, key, key, key)
    return None


def resolve_profile(profile: ExecutionProfile, custom: ProviderConfig | None = None, env: dict[str, str] | None = None) -> ProviderConfig:
    env = dict(os.environ) if env is None else env
    if custom is not None:
        return _validate(custom, env, private=profile is ExecutionProfile.LOCAL)
    if profile is ExecutionProfile.CUSTOM:
        raise ValueError("Advanced mode requires an explicit model configuration.")
    if profile is ExecutionProfile.LOCAL:
        try:
            return _validate(_local_config(env), env, private=True)
        except ValueError as error:
            raise ValueError("Private mode requires LOCAL_VISION_MODEL/OLLAMA_VISION_MODEL and LOCAL_LLM_MODEL/OLLAMA_LANGUAGE_MODEL.") from error
    cloud = _cloud_config(env)
    if profile is ExecutionProfile.CLOUD:
        if cloud is None:
            raise ValueError("No cloud model is configured. Set GEMINI_API_KEY or configure an OpenAI-compatible endpoint.")
        return _validate(cloud, env)
    if cloud is not None:
        return _validate(cloud, env)
    try:
        return _validate(_local_config(env), env, private=True)
    except ValueError as error:
        raise ValueError("No cloud credentials or ready local AI models were found. Configure a cloud model or local Ollama models.") from error


def build_providers(config: ProviderConfig) -> ProviderBundle:
    if config.asr_provider is not ProviderKind.LOCAL_WHISPER:
        raise ValueError("Only Local Faster Whisper ASR is currently implemented.")
    asr = LocalWhisperProvider(config.asr_model, config.device, config.compute_type)
    if config.vision_provider is ProviderKind.GEMINI:
        vision = GeminiVisionProvider(config.vision_model, config.vision_api_key)
    elif config.vision_provider is ProviderKind.OLLAMA:
        vision = OllamaVisionProvider(config.vision_model or "", config.vision_base_url)
    elif config.vision_provider is ProviderKind.OPENAI_COMPATIBLE:
        vision = OpenAICompatibleVisionProvider(config.vision_model or "", config.vision_base_url or "", config.vision_api_key)
    else:
        raise ValueError("Unsupported vision backend.")

    def language(kind, model, url, key, verification=False):
        if kind is ProviderKind.GEMINI:
            return GeminiVerificationProvider(model, key) if verification else GeminiLLMProvider(model, key)
        if kind is ProviderKind.OLLAMA:
            return OllamaVerificationProvider(model or "", url) if verification else OllamaLLMProvider(model or "", url)
        if kind is ProviderKind.OPENAI_COMPATIBLE:
            return OpenAICompatibleVerificationProvider(model or "", url or "", key) if verification else OpenAICompatibleLLMProvider(model or "", url or "", key)
        raise ValueError("Unsupported language backend.")

    return ProviderBundle(config, asr, vision, language(config.synthesis_provider, config.synthesis_model, config.synthesis_base_url, config.synthesis_api_key), language(config.verification_provider, config.verification_model, config.verification_base_url, config.verification_api_key, True))
