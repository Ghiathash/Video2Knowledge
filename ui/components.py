import html
import json
from pathlib import Path

import streamlit as st


def header(statuses: list[tuple[str, bool]]) -> None:
    chips = "".join(
        f'<span class="v2k-status"><span class="v2k-dot {"" if ready else "warn"}"></span>{html.escape(name)}</span>'
        for name, ready in statuses
    )
    st.markdown(
        '<div class="v2k-header"><div class="v2k-brand">Video2Knowledge</div>'
        '<div class="v2k-sub">Transform educational videos into grounded visual knowledge.</div>'
        f'<div style="margin-top:16px">{chips}</div></div>', unsafe_allow_html=True,
    )


def section_title(kicker: str, title: str, description: str = "") -> None:
    st.markdown(f'<div class="v2k-kicker">{kicker}</div><h3 style="margin:.1rem 0">{title}</h3><p style="color:#68758a;margin:.2rem 0 1rem">{description}</p>', unsafe_allow_html=True)


def model_status(role: str, model: str, backend: str, ready: bool) -> None:
    state = "Ready" if ready else "Setup required"
    st.markdown(f'<div class="model-row"><div class="model-role">{html.escape(role)}</div><div class="model-name">{html.escape(model or "Not configured")}</div><div class="model-meta">{html.escape(backend)} &middot; {state}</div></div>', unsafe_allow_html=True)


def load_report_details(output_dir: Path) -> dict:
    result: dict = {"topics": [], "overview": "", "visuals": [], "verification": {}}
    report = output_dir / "report" / "verified_report.json"
    aligned = output_dir / "alignment" / "aligned_sections.json"
    audit = output_dir / "report" / "verification_audit.json"
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        result["overview"] = data.get("overview", "")
        result["topics"] = [item.get("title", "Untitled") for item in data.get("sections", [])]
    except (OSError, json.JSONDecodeError):
        pass
    try:
        data = json.loads(aligned.read_text(encoding="utf-8"))
        result["visuals"] = [visual for item in data for visual in item.get("visuals", [])]
    except (OSError, json.JSONDecodeError):
        pass
    try:
        data = json.loads(audit.read_text(encoding="utf-8"))
        result["verification"] = {"checked": len(data), "faithful": sum(bool(item.get("is_faithful")) for item in data)}
    except (OSError, json.JSONDecodeError):
        pass
    return result
