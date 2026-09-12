from pathlib import Path
import os

from dotenv import load_dotenv
import streamlit as st

from src.application.service import create_run_directory, run_video2knowledge, save_uploaded_file
from src.ingestion.validator import SUPPORTED_VIDEO_EXTENSIONS
from src.providers import ExecutionProfile, ProviderConfig, ProviderKind
from src.providers.factory import resolve_profile
from src.providers.hardware import detect_hardware
from src.providers.ollama import installed_ollama_models


MODE_LABELS = {
    "Smart — Recommended": ExecutionProfile.SMART,
    "Private / Offline": ExecutionProfile.LOCAL,
    "Cloud": ExecutionProfile.CLOUD,
    "Advanced": ExecutionProfile.CUSTOM,
}


def _custom_config() -> ProviderConfig:
    provider_labels = {"Gemini API": ProviderKind.GEMINI, "Local Ollama": ProviderKind.OLLAMA}
    vision = provider_labels[st.selectbox("Visual Understanding", provider_labels)]
    synthesis = provider_labels[st.selectbox("Report Synthesis", provider_labels)]
    verification = provider_labels[st.selectbox("Verification", provider_labels)]
    default_vision = os.getenv("LOCAL_VISION_MODEL", "") if vision is ProviderKind.OLLAMA else os.getenv("GEMINI_VISION_MODEL", "")
    default_llm = os.getenv("LOCAL_LLM_MODEL", "") if synthesis is ProviderKind.OLLAMA else os.getenv("GEMINI_LLM_MODEL", "")
    return ProviderConfig(
        ProviderKind.LOCAL_WHISPER, vision, synthesis, verification,
        st.text_input("Speech model", os.getenv("LOCAL_ASR_MODEL", "medium")),
        st.text_input("Vision model", default_vision),
        st.text_input("Synthesis model", default_llm),
        st.text_input("Verification model", os.getenv("LOCAL_VERIFICATION_MODEL", default_llm)),
        "cuda" if detect_hardware().cuda_available else "cpu",
        "float16" if detect_hardware().cuda_available else "int8",
    )


def _render_result(result) -> None:
    st.success("Your knowledge report is ready.")
    left, right = st.columns(2)
    left.metric("Sections", result.sections_count or 0)
    right.metric("Knowledge visuals", result.knowledge_visuals_count or 0)
    st.caption(f"Run ID: {result.run_id}")
    with st.expander("Processing configuration"):
        for stage, provider in (result.provider_summary or {}).items():
            st.write(f"**{stage}:** {provider}")
        st.write(f"Output: `{result.output_dir}`")
        if result.elapsed_seconds is not None:
            st.write(f"Duration: {result.elapsed_seconds / 60:.1f} minutes")
    with Path(result.pdf_path).open("rb") as report:
        st.download_button(
            "Download PDF", report, file_name=Path(result.pdf_path).name,
            mime="application/pdf", type="primary", use_container_width=True,
        )
    if st.button("New Video", use_container_width=True):
        st.session_state.pop("result", None)
        st.rerun()


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Video2Knowledge", page_icon="V2K", layout="centered")
    st.markdown("""
        <style>
        .block-container {max-width: 880px; padding-top: 4rem;}
        h1 {letter-spacing: -0.04em;}
        div[data-testid="stFileUploader"] {padding: 1rem; border-radius: 12px;}
        </style>
    """, unsafe_allow_html=True)
    st.title("Video2Knowledge")
    st.write("Turn educational videos into structured, visual knowledge reports.")

    if st.session_state.get("result"):
        _render_result(st.session_state.result)
        return

    source_tab, url_tab = st.tabs(["Upload Video", "Video URL"])
    with source_tab:
        extensions = sorted(ext.lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS)
        uploaded = st.file_uploader("Choose an educational video", type=extensions)
        st.caption("Supported: MP4, MOV, MKV, WebM, AVI. Maximum upload: 2 GB.")
    with url_tab:
        st.info("Video URL ingestion is not available in this version. Upload a local file instead.")

    label = st.radio("Processing Mode", MODE_LABELS, horizontal=True, index=0)
    profile = MODE_LABELS[label]
    descriptions = {
        ExecutionProfile.SMART: "Balanced quality, speed, privacy, and cost.",
        ExecutionProfile.LOCAL: "Runs AI processing locally after configured Ollama models are installed.",
        ExecutionProfile.CLOUD: "Uses Gemini for visual analysis, synthesis, and verification; ASR remains local.",
        ExecutionProfile.CUSTOM: "Choose a provider for each AI stage.",
    }
    st.caption(descriptions[profile])

    custom = None
    if profile is ExecutionProfile.CUSTOM:
        with st.expander("Advanced Settings", expanded=True):
            custom = _custom_config()
    elif profile is ExecutionProfile.LOCAL:
        with st.expander("Local AI model status"):
            installed = installed_ollama_models()
            vision_model = os.getenv("LOCAL_VISION_MODEL", "")
            language_model = os.getenv("LOCAL_LLM_MODEL", "")
            st.write(f"Speech model: `{os.getenv('LOCAL_ASR_MODEL', 'medium')}`")
            st.write(f"Vision model: `{vision_model or 'Not configured'}` — {'Ready' if vision_model in installed else 'Not installed'}")
            st.write(f"Language model: `{language_model or 'Not configured'}` — {'Ready' if language_model in installed else 'Not installed'}")
            st.caption("Install models explicitly with `ollama pull <model>`. No model is downloaded automatically.")

    with st.expander("Processing options"):
        max_visuals = st.slider("Maximum visuals to analyze", 5, 100, 40, 5)
        language = st.text_input("Language", "en")

    if st.button("Generate Report", type="primary", disabled=uploaded is None, use_container_width=True):
        try:
            resolve_profile(profile, custom=custom)
            output_base = Path(os.getenv("VIDEO2KNOWLEDGE_OUTPUT_DIR", "data/output"))
            run_id, run_dir = create_run_directory(output_base)
            input_path = save_uploaded_file(run_dir, uploaded.name, uploaded.getvalue())
            progress = st.progress(0, text="Preparing run")
            status = st.status("Processing video", expanded=True)

            def on_progress(number: int, total: int, name: str):
                progress.progress(number / total, text=name)
                status.write(name)

            result = run_video2knowledge(
                input_path, run_dir, execution_profile=profile,
                provider_config=custom, progress_callback=on_progress,
                max_visuals=max_visuals, language=language,
            )
            if result.status == "PASS":
                status.update(label="Report complete", state="complete", expanded=False)
                st.session_state.result = result
                st.rerun()
            else:
                status.update(label="Processing failed", state="error")
                st.error(result.error or "The report could not be generated.")
                with st.expander("Technical details"):
                    st.code(result.technical_details or "No diagnostic details available.")
        except Exception as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
