from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class HardwareInfo:
    cuda_available: bool
    gpu_name: str | None = None
    vram_mb: int | None = None


def detect_hardware() -> HardwareInfo:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        first = result.stdout.strip().splitlines()[0]
        name, memory = first.rsplit(",", 1)
        return HardwareInfo(True, name.strip(), int(memory.strip()))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return HardwareInfo(False)
