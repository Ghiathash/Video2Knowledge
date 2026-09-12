from pathlib import Path

from src.ingestion.video_loader import load_video
from src.preprocessing.audio import extract_audio
from src.preprocessing.frames import sample_frames


def preprocess_video(
    video_path: str | Path,
    output_dir: str | Path,
    frame_interval: float = 10.0,
):
    video = load_video(video_path)

    output_dir = Path(output_dir)

    audio_path = output_dir / "audio" / "audio.wav"
    frames_dir = output_dir / "frames" / "sampled"

    extracted_audio = extract_audio(
        video_path=video.path,
        output_path=audio_path,
    )

    frames = sample_frames(
        video_path=video.path,
        duration=video.duration,
        interval=frame_interval,
        output_dir=frames_dir,
    )

    return {
        "video": video,
        "audio_path": extracted_audio,
        "frames": frames,
    }
