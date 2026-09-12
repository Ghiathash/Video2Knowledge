from pathlib import Path

from src.application.service import (
    ProcessingResult, create_run_directory, run_video2knowledge,
    sanitize_filename, save_uploaded_file,
)
from src.providers.config import ExecutionProfile


def test_safe_filename_removes_path_traversal():
    assert sanitize_filename("../../unsafe video.mp4") == "unsafe_video.mp4"
    assert sanitize_filename("..\\..\\lecture.mov") == "lecture.mov"


def test_run_directories_are_isolated(tmp_path):
    first_id, first = create_run_directory(tmp_path)
    second_id, second = create_run_directory(tmp_path)
    assert first_id != second_id
    assert first != second
    assert first.parent == second.parent == tmp_path.resolve()


def test_upload_is_kept_inside_run(tmp_path):
    _, run = create_run_directory(tmp_path)
    uploaded = save_uploaded_file(run, "../../lesson.mp4", b"video")
    assert uploaded.parent == (run / "input").resolve()
    assert uploaded.read_bytes() == b"video"


def test_service_returns_structured_failure_for_missing_input(tmp_path):
    result = run_video2knowledge(
        tmp_path / "missing.mp4", tmp_path / "run",
        execution_profile=ExecutionProfile.SMART,
    )
    assert isinstance(result, ProcessingResult)
    assert result.status == "FAILED"
    assert result.pdf_path is None


def test_service_result_structure_with_mocked_pipeline(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test")
    pdf = tmp_path / "run" / "report.pdf"
    monkeypatch.setattr(
        "src.application.service.run_pipeline",
        lambda args, progress_callback=None: {
            "evaluation": {"status": "PASS"}, "output_dir": tmp_path / "run",
            "pdf_path": pdf, "elapsed_seconds": 1.0,
            "provider_summary": {"Speech Recognition": "Fake"},
            "sections_count": 2, "visual_candidates_count": 3,
            "knowledge_visuals_count": 1,
        },
    )
    result = run_video2knowledge(video, tmp_path / "run", execution_profile=ExecutionProfile.SMART)
    assert result.status == "PASS"
    assert result.sections_count == 2
    assert result.provider_summary == {"Speech Recognition": "Fake"}
