import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import streamlit as st

from src.application.service import create_run_directory, prepare_url_source, run_video2knowledge, save_uploaded_file
from src.ingestion.validator import SUPPORTED_VIDEO_EXTENSIONS
from src.providers import ExecutionProfile, ProviderConfig, ProviderKind
from src.providers.factory import resolve_profile
from src.providers.hardware import detect_hardware
from src.providers.ollama import installed_ollama_models
from src.providers.openai_compatible import test_connection
from ui.components import header, load_report_details, model_status, section_title
from ui.styles import APP_CSS


DASH = "\u2014"
MODE_LABELS = {
    f"Smart {DASH} Recommended": ExecutionProfile.SMART,
    "Private / Offline": ExecutionProfile.LOCAL,
    "Cloud": ExecutionProfile.CLOUD,
    "Advanced": ExecutionProfile.CUSTOM,
}
MODE_HELP = {
    ExecutionProfile.SMART: "Uses the best models already configured, preferring cloud and then local Ollama.",
    ExecutionProfile.LOCAL: "Keeps AI processing local with Faster Whisper and Ollama.",
    ExecutionProfile.CLOUD: "Select hosted models independently for vision and language.",
    ExecutionProfile.CUSTOM: "Mix local and hosted models, endpoints, and processing controls.",
}

VISION_PRESETS = {
    "Gemini 2.5 Flash": (ProviderKind.GEMINI, "gemini-2.5-flash"),
    "Gemini custom model": (ProviderKind.GEMINI, ""),
    "Qwen2.5-VL via Ollama": (ProviderKind.OLLAMA, "qwen2.5vl:7b"),
    "LLaVA via Ollama": (ProviderKind.OLLAMA, "llava:7b"),
    "OpenAI-compatible model": (ProviderKind.OPENAI_COMPATIBLE, ""),
    "Custom model": (None, ""),
}
LANGUAGE_PRESETS = {
    "Gemini 2.5 Flash": (ProviderKind.GEMINI, "gemini-2.5-flash"),
    "Llama 3.2 via Ollama": (ProviderKind.OLLAMA, "llama3.2:3b"),
    "Qwen2.5 via Ollama": (ProviderKind.OLLAMA, "qwen2.5:7b"),
    "OpenAI-compatible model": (ProviderKind.OPENAI_COMPATIBLE, ""),
    "Custom model": (None, ""),
}


def _secret(name: str, label: str, key: str) -> str | None:
    configured = bool(os.getenv(name))
    value = st.text_input(label, type="password", key=key, placeholder="Configured in environment" if configured else "Session only")
    return value or os.getenv(name)


def _select_upload() -> None:
    st.session_state.active_source = "upload"


def _model_picker(role: str, presets: dict, allowed: set[ProviderKind] | None = None) -> tuple[ProviderKind, str, str | None, str | None]:
    visible = {name: value for name, value in presets.items() if value[0] is None or allowed is None or value[0] in allowed}
    choice = st.selectbox(f"{role} Model", list(visible), key=f"{role}_preset")
    backend, model = visible[choice]
    if backend is None:
        labels = {"Gemini": ProviderKind.GEMINI, "Ollama / Local": ProviderKind.OLLAMA, "OpenAI-compatible endpoint": ProviderKind.OPENAI_COMPATIBLE}
        labels = {name: kind for name, kind in labels.items() if allowed is None or kind in allowed}
        backend = labels[st.selectbox("Backend type", list(labels), key=f"{role}_backend")]
    env_model = {
        ProviderKind.GEMINI: "GEMINI_MODEL",
        ProviderKind.OLLAMA: "OLLAMA_VISION_MODEL" if role == "Vision" else "OLLAMA_LANGUAGE_MODEL",
        ProviderKind.OPENAI_COMPATIBLE: "OPENAI_COMPATIBLE_MODEL",
    }[backend]
    model = st.text_input("Model ID", value=model or os.getenv(env_model, ""), key=f"{role}_model_{choice}_{backend.value}").strip()
    if backend is ProviderKind.GEMINI:
        return backend, model, None, _secret("GEMINI_API_KEY", "Gemini API key", f"{role}_gemini_key")
    if backend is ProviderKind.OLLAMA:
        base = st.text_input("Ollama Base URL", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), key=f"{role}_ollama_url").strip()
        return backend, model, base, None
    base = st.text_input("Base URL", os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""), placeholder="https://host.example/v1", key=f"{role}_compatible_url").strip()
    key = _secret("OPENAI_COMPATIBLE_API_KEY", "API key (optional for local servers)", f"{role}_compatible_key")
    if st.button("Test connection", key=f"{role}_test", use_container_width=True):
        ok, message = test_connection(base, key) if base else (False, "Enter a base URL first.")
        (st.success if ok else st.error)(message)
    return backend, model, base, key


def _model_config(profile: ExecutionProfile) -> ProviderConfig | None:
    hardware = detect_hardware()
    device, compute = ("cuda", "float16") if hardware.cuda_available else ("cpu", "int8")
    if profile is ExecutionProfile.SMART:
        return None
    allowed = {ProviderKind.OLLAMA} if profile is ExecutionProfile.LOCAL else ({ProviderKind.GEMINI, ProviderKind.OPENAI_COMPATIBLE} if profile is ExecutionProfile.CLOUD else None)
    asr_model = st.selectbox("Speech Model", ["tiny", "base", "small", "medium", "large-v3"], index=3, format_func=lambda item: f"Faster Whisper {item.title()}")
    vision_kind, vision_model, vision_url, vision_key = _model_picker("Vision", VISION_PRESETS, allowed)
    synth_kind, synth_model, synth_url, synth_key = _model_picker("Synthesis", LANGUAGE_PRESETS, allowed)
    separate = st.toggle("Use a separate verification model", value=False)
    if separate:
        verify_kind, verify_model, verify_url, verify_key = _model_picker("Verification", LANGUAGE_PRESETS, allowed)
    else:
        verify_kind, verify_model, verify_url, verify_key = synth_kind, synth_model, synth_url, synth_key
    return ProviderConfig(
        ProviderKind.LOCAL_WHISPER, vision_kind, synth_kind, verify_kind,
        asr_model, vision_model, synth_model, verify_model, device, compute,
        vision_url, synth_url, verify_url, vision_key, synth_key, verify_key,
    )


def _render_result(result) -> None:
    details = load_report_details(result.output_dir)
    section_title("Complete", "Report ready", "Your grounded knowledge report passed the final pipeline checks.")
    columns = st.columns(5)
    values = [("Sections", result.sections_count or 0), ("Knowledge visuals", result.knowledge_visuals_count or 0), ("Candidates", result.visual_candidates_count or 0), ("Processing", f"{(result.elapsed_seconds or 0) / 60:.1f} min"), ("QA status", result.status)]
    for column, (label, value) in zip(columns, values, strict=True):
        column.metric(label, value)
    st.markdown("#### Processing configuration")
    for role, value in (result.provider_summary or {}).items():
        model_status(role, value, "Selected configuration", True)
    with st.expander("Report Overview"):
        st.write(details["overview"] or "Overview is available in the PDF report.")
    with st.expander("Detected Topics"):
        st.write("\n".join(f"{index}. {topic}" for index, topic in enumerate(details["topics"], 1)) or "No topic summary available.")
    with st.expander("Visual Knowledge"):
        st.write(f"{len(details['visuals'])} aligned visual evidence items.")
    with st.expander("Verification Summary"):
        verification = details["verification"]
        st.write(f"{verification.get('faithful', 0)} of {verification.get('checked', 0)} sections marked faithful by the verifier.")
    st.caption(f"Run ID: {result.run_id}  |  Output: {result.output_dir}")
    with Path(result.pdf_path).open("rb") as report:
        st.download_button("Download PDF", report, file_name=Path(result.pdf_path).name, mime="application/pdf", type="primary", use_container_width=True)
    if st.button("Process another video", use_container_width=True):
        st.session_state.pop("result", None)
        st.rerun()


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Video2Knowledge", page_icon="V2K", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    installed = installed_ollama_models()
    cloud_ready = bool(os.getenv("GEMINI_API_KEY") or (os.getenv("OPENAI_COMPATIBLE_BASE_URL") and os.getenv("OPENAI_COMPATIBLE_MODEL")))
    header([("Speech ready", True), ("Vision ready" if cloud_ready or installed else "Vision setup needed", bool(cloud_ready or installed)), ("Language ready" if cloud_ready or installed else "Language setup needed", bool(cloud_ready or installed))])
    if st.session_state.get("result"):
        _render_result(st.session_state.result)
        return

    left, right = st.columns([1.75, 1], gap="large")
    with left:
        with st.container(border=True):
            section_title("Input", "Choose your video", "Upload a local educational video. Files are copied into an isolated run directory.")
            extensions = sorted(ext.lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS)
            upload_tab, url_tab = st.tabs(["Upload Video", "Video URL"])
            with upload_tab:
                uploaded = st.file_uploader("Upload video", type=extensions, label_visibility="collapsed", on_change=_select_upload, key="uploaded_video")
                if uploaded:
                    st.caption(f"Selected: {uploaded.name}  |  {uploaded.size / (1024 * 1024):.1f} MB")
                else:
                    st.caption("MP4, MOV, MKV, WebM, or AVI up to 2 GB")
            with url_tab:
                video_url = st.text_input("Paste a video URL", placeholder="https://www.youtube.com/watch?v=...", key="video_url")
                st.caption("YouTube or another supported public video URL. Playlists and private-network URLs are blocked.")
                if st.button("Load Video", use_container_width=True, key="load_video_url"):
                    url_status = st.status("Loading video", expanded=True)
                    url_progress = st.progress(0, text="Fetching video information...")
                    last_url_stage = {"value": None}

                    def on_url_progress(stage: str, ratio: float | None, message: str) -> None:
                        values = {"fetching": 0.08, "downloading": ratio if ratio is not None else 0.35, "preparing": 0.9, "complete": 1.0}
                        url_progress.progress(float(values.get(stage, 0.1)), text=message)
                        if last_url_stage["value"] != stage:
                            url_status.update(label=message)
                            last_url_stage["value"] = stage

                    try:
                        output_base = Path(os.getenv("VIDEO2KNOWLEDGE_OUTPUT_DIR", "data/output"))
                        _, url_run_dir, downloaded = prepare_url_source(video_url, output_base, on_url_progress)
                        st.session_state.url_source = {"video": downloaded, "run_dir": url_run_dir}
                        st.session_state.active_source = "url"
                        url_status.update(label="Video ready", state="complete", expanded=False)
                        st.rerun()
                    except Exception as error:
                        url_progress.empty()
                        details = traceback.format_exc()
                        st.session_state.url_error_details = details.replace(video_url, "[video URL]") if video_url else details
                        url_status.update(label="Could not load video", state="error")
                        st.error(str(error))
                        with st.expander("Advanced: Download Logs"):
                            st.code(st.session_state.url_error_details)
                prepared_url = st.session_state.get("url_source")
                if prepared_url and prepared_url["video"].source_url == video_url.strip():
                    metadata = prepared_url["video"]
                    st.success("Video loaded and ready for processing.")
                    meta_columns = st.columns(3)
                    meta_columns[0].metric("Duration", f"{metadata.duration / 60:.1f} min" if metadata.duration else "Unknown")
                    meta_columns[1].metric("Source", metadata.source)
                    meta_columns[2].metric("Resolution", metadata.resolution or "Unknown")
                    st.markdown(f"**{metadata.title}**")
                    if metadata.duration and metadata.duration > 3600:
                        st.warning("Long videos may require significant processing time and model usage.")

        with st.container(border=True):
            section_title("Mode", "How should this run?", "Complexity stays hidden unless you choose Advanced.")
            label = st.segmented_control("Processing mode", list(MODE_LABELS), default=list(MODE_LABELS)[0], label_visibility="collapsed")
            profile = MODE_LABELS[label or list(MODE_LABELS)[0]]
            st.info(MODE_HELP[profile])

        with st.expander("Processing options", expanded=profile is ExecutionProfile.CUSTOM):
            max_visuals = st.slider("Maximum visuals to analyze", 5, 100, 40, 5)
            language = st.text_input("Transcript language", "en", help="Use 'auto' for automatic language detection.")

    with right:
        with st.container(border=True):
            section_title("AI Models", "Model configuration", "Choose models by pipeline role. Backend settings appear only when needed.")
            if profile is ExecutionProfile.SMART:
                try:
                    effective = resolve_profile(profile)
                    for role, value in effective.summary().items():
                        model_status(role, value, "Auto-selected", True)
                    custom = None
                except ValueError:
                    model_status("Speech", f"Faster Whisper {os.getenv('LOCAL_ASR_MODEL', 'medium').title()}", "Local", True)
                    model_status("Vision", "Not configured", "Smart selection", False)
                    model_status("Language", "Not configured", "Smart selection", False)
                    st.warning("Configure cloud models in the environment or choose Private, Cloud, or Advanced.")
                    custom = None
            else:
                custom = _model_config(profile)
        if profile in {ExecutionProfile.LOCAL, ExecutionProfile.CUSTOM}:
            with st.expander("Local model status"):
                if not installed:
                    st.warning("Ollama is not running or has no installed models.")
                    st.code("ollama pull <model>", language="text")
                else:
                    st.caption("Installed models: " + ", ".join(sorted(installed)))
                st.caption("Models are never downloaded automatically.")

        prepared_url = st.session_state.get("url_source")
        url_ready = bool(prepared_url and prepared_url["video"].source_url == st.session_state.get("video_url", "").strip())
        active_source = st.session_state.get("active_source", "upload")
        source_ready = uploaded is not None if active_source == "upload" else url_ready
        with st.container(key="generate_cta"):
            generate = st.button("Generate Report", type="primary", disabled=not source_ready, use_container_width=True)
            if not source_ready:
                st.caption("Upload a video or load a public URL to enable report generation.")

    if generate:
        try:
            resolve_profile(profile, custom=custom)
            if active_source == "url":
                run_dir = prepared_url["run_dir"]
                input_path = prepared_url["video"].path
            else:
                output_base = Path(os.getenv("VIDEO2KNOWLEDGE_OUTPUT_DIR", "data/output"))
                _, run_dir = create_run_directory(output_base)
                input_path = save_uploaded_file(run_dir, uploaded.name, uploaded.getvalue())
            started = time.perf_counter()
            progress = st.progress(0, text="Preparing secure run")
            status = st.status("Processing video", expanded=True)
            stage_copy = {1: "Validating the source video", 2: "Extracting audio and reference frames", 3: "Transcribing speech", 4: "Finding topic boundaries", 5: "Extracting visual candidates", 6: "Ranking educational frames", 7: "Understanding visual knowledge", 8: "Aligning transcript and visuals", 9: "Synthesizing and verifying the report", 10: "Rendering the PDF", 11: "Running final quality checks"}

            def on_progress(number: int, total: int, name: str):
                elapsed = time.perf_counter() - started
                progress.progress(number / total, text=f"{name}  {DASH}  {elapsed:.0f}s")
                status.write(stage_copy.get(number, name))

            result = run_video2knowledge(input_path, run_dir, execution_profile=profile, provider_config=custom, progress_callback=on_progress, max_visuals=max_visuals, language=language)
            if result.status == "PASS":
                status.update(label="Report complete", state="complete", expanded=False)
                st.session_state.result = result
                st.rerun()
            else:
                status.update(label="Processing failed", state="error")
                st.error(result.error or "The report could not be generated.")
                with st.expander("Advanced: Processing Logs"):
                    st.code(result.technical_details or "No diagnostic details available.")
        except Exception as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
