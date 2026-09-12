import pytest

from src.providers.config import ExecutionProfile, ProviderConfig, ProviderKind
from src.providers.factory import build_providers, resolve_profile
from src.providers.ollama import installed_ollama_models


def test_smart_prefers_hybrid_when_gemini_is_configured(monkeypatch):
    monkeypatch.setattr("src.providers.factory.detect_hardware", lambda: type("H", (), {"cuda_available": False})())
    config = resolve_profile(ExecutionProfile.SMART, env={"GEMINI_API_KEY": "test-only"})
    assert config.asr_provider is ProviderKind.LOCAL_WHISPER
    assert config.vision_provider is ProviderKind.GEMINI
    assert config.device == "cpu"


def test_smart_falls_back_to_local_models(monkeypatch):
    monkeypatch.setattr("src.providers.factory.detect_hardware", lambda: type("H", (), {"cuda_available": False})())
    config = resolve_profile(
        ExecutionProfile.SMART,
        env={"LOCAL_VISION_MODEL": "vision", "LOCAL_LLM_MODEL": "llm"},
    )
    assert config.vision_provider is ProviderKind.OLLAMA
    assert config.verification_provider is ProviderKind.OLLAMA


def test_smart_reports_missing_setup(monkeypatch):
    monkeypatch.setattr("src.providers.factory.detect_hardware", lambda: type("H", (), {"cuda_available": False})())
    with pytest.raises(ValueError, match="No cloud credentials"):
        resolve_profile(ExecutionProfile.SMART, env={})


def test_cloud_requires_credentials(monkeypatch):
    monkeypatch.setattr("src.providers.factory.detect_hardware", lambda: type("H", (), {"cuda_available": False})())
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        resolve_profile(ExecutionProfile.CLOUD, env={})


def test_local_requires_model_configuration(monkeypatch):
    monkeypatch.setattr("src.providers.factory.detect_hardware", lambda: type("H", (), {"cuda_available": False})())
    with pytest.raises(ValueError, match="LOCAL_VISION_MODEL"):
        resolve_profile(ExecutionProfile.LOCAL, env={})


def test_custom_rejects_gemini_without_credentials(monkeypatch):
    monkeypatch.setattr("src.providers.factory.detect_hardware", lambda: type("H", (), {"cuda_available": False})())
    custom = ProviderConfig(
        ProviderKind.LOCAL_WHISPER, ProviderKind.GEMINI,
        ProviderKind.OLLAMA, ProviderKind.OLLAMA,
        vision_model="gemini", synthesis_model="llm", verification_model="llm",
    )
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        resolve_profile(ExecutionProfile.CUSTOM, custom=custom, env={})


def test_provider_factory_builds_real_supported_bundle():
    config = ProviderConfig(
        ProviderKind.LOCAL_WHISPER, ProviderKind.OLLAMA,
        ProviderKind.OLLAMA, ProviderKind.OLLAMA,
        vision_model="vision", synthesis_model="llm", verification_model="llm",
        device="cpu", compute_type="int8",
    )
    bundle = build_providers(config)
    assert bundle.asr.name == "Local Faster Whisper"
    assert bundle.vision.name == "Local Ollama"


def test_ollama_model_status_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "src.providers.ollama.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    assert installed_ollama_models() == set()
