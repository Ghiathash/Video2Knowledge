from src.providers.hardware import HardwareInfo, detect_hardware


def test_hardware_detection_cpu_fallback(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("src.providers.hardware.subprocess.run", unavailable)
    assert detect_hardware() == HardwareInfo(cuda_available=False)


def test_hardware_detection_parses_nvidia_smi(monkeypatch):
    result = type("Result", (), {"stdout": "Example GPU, 8192\n"})()
    monkeypatch.setattr("src.providers.hardware.subprocess.run", lambda *args, **kwargs: result)
    assert detect_hardware() == HardwareInfo(True, "Example GPU", 8192)
