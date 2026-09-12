import html
import json
import subprocess
from pathlib import Path

from PIL import Image as PILImage, ImageFilter

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_full_resolution_frame(
    video_path: Path,
    timestamp: float,
    output_dir: Path,
) -> Path:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"frame_{timestamp:.2f}"
        .replace(".", "_")
        + ".png"
    )

    output_path = output_dir / filename

    if output_path.exists():
        return output_path

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-compression_level",
        "2",
        str(output_path),
    ]

    subprocess.run(
        command,
        check=True,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Failed to extract frame at {timestamp}s"
        )

    return output_path


def _enhance_frame(
    image_path: Path,
) -> Path:

    enhanced_dir = (
        image_path.parent
        / "enhanced"
    )

    enhanced_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        enhanced_dir
        / image_path.name
    )

    if output_path.exists():
        return output_path

    with PILImage.open(
        image_path
    ) as img:

        img = img.convert("RGB")

        new_size = (
            img.width * 2,
            img.height * 2,
        )

        img = img.resize(
            new_size,
            PILImage.Resampling.LANCZOS,
        )

        img = img.filter(
            ImageFilter.UnsharpMask(
                radius=1.5,
                percent=120,
                threshold=3,
            )
        )

        img.save(
            output_path,
            "PNG",
            optimize=True,
        )

    return output_path


def _create_scaled_image(
    image_path: Path,
    max_width: float,
    max_height: float,
):

    image_path = _enhance_frame(
        image_path
    )

    with PILImage.open(image_path) as img:
        width, height = img.size

    scale = min(
        max_width / width,
        max_height / height,
    )

    return Image(
        str(image_path),
        width=width * scale,
        height=height * scale,
    )


def _find_visual(
    aligned_sections,
    timestamp,
    tolerance=0.5,
):

    for section in aligned_sections:
        for visual in section["visuals"]:

            if abs(
                float(visual["timestamp"])
                - float(timestamp)
            ) <= tolerance:

                return visual

    return None


def _add_page_number(canvas, document):

    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        9,
    )

    canvas.setFillColor(
        colors.grey
    )

    canvas.drawCentredString(
        A4[0] / 2,
        1.2 * cm,
        f"Page {canvas.getPageNumber()}",
    )

    canvas.restoreState()


def generate_pdf_report(
    verified_report_path,
    aligned_sections_path,
    output_path,
    source_video_path=None,
):

    report = _load_json(
        verified_report_path
    )

    aligned_sections = _load_json(
        aligned_sections_path
    )

    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = (
            PROJECT_ROOT / output_path
        )

    if source_video_path is None:
        raise ValueError("source_video_path is required")

    video_path = Path(source_video_path)

    if not video_path.is_absolute():
        video_path = (
            PROJECT_ROOT / video_path
        )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Source video not found: {video_path}"
        )

    report_frames_dir = (
        output_path.parent
        / "report_frames"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2.0 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading1"],
        fontSize=15,
        leading=20,
        spaceAfter=10,
    )

    section_style = ParagraphStyle(
        "SectionCustom",
        parent=styles["Heading1"],
        fontSize=18,
        leading=23,
        spaceAfter=6,
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["BodyText"],
        fontSize=11,
        leading=17,
        spaceAfter=14,
    )

    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=16,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=6,
    )

    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.darkgrey,
        spaceAfter=12,
    )

    story = []

    # Cover
    story.append(
        Spacer(1, 3 * cm)
    )

    story.append(
        Paragraph(
            html.escape(
                report["title"]
            ),
            title_style,
        )
    )

    story.append(
        Spacer(1, 0.8 * cm)
    )

    story.append(
        Paragraph(
            "Video2Knowledge",
            ParagraphStyle(
                "Subtitle",
                parent=styles["Normal"],
                alignment=TA_CENTER,
                fontSize=13,
                textColor=colors.grey,
            ),
        )
    )

    story.append(
        Spacer(1, 2 * cm)
    )

    story.append(
        Paragraph(
            "Overview",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            html.escape(
                report["overview"]
            ),
            normal_style,
        )
    )

    story.append(
        PageBreak()
    )

    inserted_visuals = 0

    for index, section in enumerate(
        report["sections"],
        start=1,
    ):

        story.append(
            Paragraph(
                f"{index}. "
                + html.escape(
                    section["title"]
                ),
                section_style,
            )
        )

        story.append(
            Paragraph(
                f"{_format_time(section['start'])}"
                f" - "
                f"{_format_time(section['end'])}",
                small_style,
            )
        )

        story.append(
            Paragraph(
                "Summary",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                html.escape(
                    section["summary"]
                ),
                normal_style,
            )
        )

        story.append(
            Paragraph(
                "Key Points",
                heading_style,
            )
        )

        for point in section["key_points"]:

            story.append(
                Paragraph(
                    "• "
                    + html.escape(point),
                    bullet_style,
                )
            )

        for timestamp in section[
            "visual_timestamps"
        ]:

            visual = _find_visual(
                aligned_sections,
                timestamp,
            )

            frame_path = (
                _extract_full_resolution_frame(
                    video_path,
                    float(timestamp),
                    report_frames_dir,
                )
            )

            story.append(
                Spacer(
                    1,
                    0.5 * cm,
                )
            )

            story.append(
                Paragraph(
                    "Relevant Visual",
                    heading_style,
                )
            )

            story.append(
                _create_scaled_image(
                    frame_path,
                    max_width=16.5 * cm,
                    max_height=10.5 * cm,
                )
            )

            caption = (
                f"Frame at "
                f"{_format_time(timestamp)}"
            )

            if visual:

                description = visual.get(
                    "description",
                    ""
                )

                if description:
                    caption += (
                        " - "
                        + description
                    )

            story.append(
                Paragraph(
                    html.escape(caption),
                    small_style,
                )
            )

            inserted_visuals += 1

        if index < len(report["sections"]):
            story.append(Spacer(1, 0.8 * cm))

    document.build(
        story,
        onFirstPage=_add_page_number,
        onLaterPages=_add_page_number,
    )

    print(
        f"Sections: {len(report['sections'])}"
    )

    print(
        f"Full-resolution visuals inserted: "
        f"{inserted_visuals}"
    )

    return output_path

