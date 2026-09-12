import sys
import types

import pytest

from main import parse_args
from src.application.service import prepare_url_source
from src.ingestion.url_video import VideoURLIngestionError, download_video_from_url, validate_public_video_url


def _public_resolver(*args, **kwargs):
    return [(2, 1, 6, "", ("93.184.216.34", 443))]


def test_url_validation_accepts_public_http_url():
    assert validate_public_video_url("https://video.example/watch/1", resolver=_public_resolver).startswith("https://")


@pytest.mark.parametrize("url", ["", "file:///tmp/video.mp4", "http://localhost/video", "http://127.0.0.1/video", "http://10.0.0.2/video"])
def test_url_validation_rejects_unsafe_sources(url):
    resolver = lambda *args, **kwargs: [(2, 1, 6, "", ("127.0.0.1", 80))]
    with pytest.raises(ValueError):
        validate_public_video_url(url, resolver=resolver)


def _install_fake_ytdlp(monkeypatch, *, fail=False):
    observed = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            observed.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            if fail:
                raise RuntimeError("download failed")
            path = observed["outtmpl"].replace("%(ext)s", "mp4")
            from pathlib import Path
            Path(path).write_bytes(b"video")
            observed["progress_hooks"][0]({"status": "downloading", "downloaded_bytes": 5, "total_bytes": 10})
            return {"title": "Lesson", "duration": 90, "width": 1280, "height": 720}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    monkeypatch.setattr("src.ingestion.url_video.socket.getaddrinfo", _public_resolver)
    return observed


def test_download_uses_single_video_and_returns_local_path(monkeypatch, tmp_path):
    observed = _install_fake_ytdlp(monkeypatch)
    progress = []
    result = download_video_from_url("https://video.example/lesson", tmp_path, lambda *item: progress.append(item))
    assert observed["noplaylist"] is True
    assert observed["merge_output_format"] == "mp4"
    assert result.path == (tmp_path / "input" / "source_video.mp4").resolve()
    assert result.path.is_file()
    assert any(item[0] == "downloading" for item in progress)


def test_failed_download_has_clean_error(monkeypatch, tmp_path):
    _install_fake_ytdlp(monkeypatch, fail=True)
    with pytest.raises(VideoURLIngestionError, match="could not be downloaded"):
        download_video_from_url("https://video.example/fail", tmp_path)


def test_service_prepares_isolated_url_run(monkeypatch, tmp_path):
    _install_fake_ytdlp(monkeypatch)
    run_id, run_dir, video = prepare_url_source("https://video.example/lesson", tmp_path)
    assert run_dir.name == run_id
    assert video.path.parent == run_dir / "input"


def test_cli_requires_exactly_one_input_source(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["main.py", "--input", "a.mp4", "--url", "https://video.example", "--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        parse_args()

    monkeypatch.setattr(sys, "argv", ["main.py", "--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        parse_args()
