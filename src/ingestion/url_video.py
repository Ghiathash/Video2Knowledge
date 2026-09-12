"""Safe public-video URL ingestion backed by yt-dlp's Python API."""

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import socket
from typing import Callable
from urllib.parse import urlparse

from src.ingestion.validator import SUPPORTED_VIDEO_EXTENSIONS


URLProgressCallback = Callable[[str, float | None, str], None]


class VideoURLIngestionError(RuntimeError):
    """A concise, user-facing URL ingestion failure."""


@dataclass(frozen=True)
class DownloadedVideo:
    path: Path
    source_url: str
    title: str
    duration: float | None
    source: str
    width: int | None = None
    height: int | None = None

    @property
    def resolution(self) -> str | None:
        return f"{self.width}x{self.height}" if self.width and self.height else None


def validate_public_video_url(url: str, *, resolver=None) -> str:
    value = url.strip()
    if not value:
        raise ValueError("Paste a video URL first.")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a valid public HTTP or HTTPS video URL.")
    if parsed.username or parsed.password:
        raise ValueError("Video URLs containing credentials are not allowed.")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Localhost and private-network video URLs are not allowed.")
    resolver = resolver or socket.getaddrinfo
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = {item[4][0].split("%", 1)[0] for item in resolver(hostname, port, type=socket.SOCK_STREAM)}
    except (OSError, socket.gaierror) as error:
        raise ValueError("The video URL host could not be resolved.") from error
    if not addresses:
        raise ValueError("The video URL host could not be resolved.")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Localhost and private-network video URLs are not allowed.")
    return value


def download_video_from_url(url: str, output_dir: str | Path, progress_callback: URLProgressCallback | None = None) -> DownloadedVideo:
    public_url = validate_public_video_url(url)
    input_dir = (Path(output_dir).resolve() / "input").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)

    def notify(stage: str, progress: float | None, message: str) -> None:
        if progress_callback:
            progress_callback(stage, progress, message)

    def hook(status: dict) -> None:
        state = status.get("status")
        if state == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            downloaded = status.get("downloaded_bytes", 0)
            ratio = min(downloaded / total, 1.0) if total else None
            notify("downloading", ratio, "Downloading video...")
        elif state == "finished":
            notify("preparing", 1.0, "Preparing media...")

    options = {
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(input_dir / "source_video.%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "writeinfojson": False,
        "writethumbnail": False,
        "progress_hooks": [hook],
    }
    notify("fetching", None, "Fetching video information...")
    try:
        from yt_dlp import YoutubeDL

        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(public_url, download=True)
    except Exception as error:
        raise VideoURLIngestionError("The video could not be downloaded. Check that the URL is public and supported.") from error

    candidates = [path for path in input_dir.glob("source_video.*") if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS]
    if not candidates:
        raise VideoURLIngestionError("The download finished but no supported video file was produced.")
    path = max(candidates, key=lambda item: (item.suffix.lower() == ".mp4", item.stat().st_size)).resolve()
    notify("complete", 1.0, "Video ready.")
    return DownloadedVideo(
        path=path, source_url=public_url, title=str(info.get("title") or "Untitled video"),
        duration=float(info["duration"]) if info.get("duration") is not None else None,
        source=urlparse(public_url).hostname or str(info.get("extractor_key") or "Unknown source"),
        width=int(info["width"]) if info.get("width") else None,
        height=int(info["height"]) if info.get("height") else None,
    )
