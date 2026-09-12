import importlib


def test_ui_import_does_not_run_pipeline(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Pipeline ran during UI import")

    monkeypatch.setattr("src.application.service.run_video2knowledge", fail_if_called)
    importlib.import_module("ui.streamlit_app")
    assert called is False
