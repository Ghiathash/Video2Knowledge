import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path):
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate_report(
    verified_report_path,
    aligned_sections_path,
    verification_audit_path,
    pdf_path,
    output_path,
):
    report = _load_json(
        verified_report_path
    )

    aligned = _load_json(
        aligned_sections_path
    )

    audit = _load_json(
        verification_audit_path
    )

    pdf_path = Path(pdf_path)

    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sections = report.get(
        "sections",
        []
    )

    # -------------------------------------------------
    # Section checks
    # -------------------------------------------------

    empty_summaries = []
    empty_key_points = []
    invalid_time_ranges = []

    for index, section in enumerate(
        sections,
        start=1,
    ):
        if not section.get(
            "summary",
            ""
        ).strip():
            empty_summaries.append(
                index
            )

        if not section.get(
            "key_points"
        ):
            empty_key_points.append(
                index
            )

        if float(
            section["end"]
        ) <= float(
            section["start"]
        ):
            invalid_time_ranges.append(
                index
            )

    # -------------------------------------------------
    # Visual checks
    # -------------------------------------------------

    report_visuals = []

    for section in sections:
        report_visuals.extend(
            float(x)
            for x in section.get(
                "visual_timestamps",
                []
            )
        )

    aligned_visuals = []

    for section in aligned:
        for visual in section.get(
            "visuals",
            []
        ):
            aligned_visuals.append(
                float(
                    visual["timestamp"]
                )
            )

    missing_aligned_visuals = [
        timestamp
        for timestamp in report_visuals
        if not any(
            abs(
                timestamp - aligned_timestamp
            ) <= 0.5
            for aligned_timestamp
            in aligned_visuals
        )
    ]

    duplicate_visuals = sorted(
        {
            timestamp
            for timestamp in report_visuals
            if report_visuals.count(
                timestamp
            ) > 1
        }
    )

    # -------------------------------------------------
    # Verification audit
    # -------------------------------------------------

    corrected_sections = []

    for item in audit:
        if (
            "section_index" in item
            and not item.get(
                "is_faithful",
                True,
            )
        ):
            corrected_sections.append(
                item["section_index"]
            )

    # -------------------------------------------------
    # Coverage
    # -------------------------------------------------

    section_count = len(
        sections
    )

    aligned_section_count = len(
        aligned
    )

    section_coverage = (
        section_count
        / aligned_section_count
        if aligned_section_count
        else 0.0
    )

    visual_coverage = (
        (
            len(report_visuals)
            - len(
                missing_aligned_visuals
            )
        )
        / len(report_visuals)
        if report_visuals
        else 1.0
    )

    # -------------------------------------------------
    # Final status
    # -------------------------------------------------

    checks = {
        "pdf_exists": (
            pdf_path.exists()
        ),
        "has_sections": (
            section_count > 0
        ),
        "section_counts_match": (
            section_count
            == aligned_section_count
        ),
        "no_empty_summaries": (
            len(empty_summaries)
            == 0
        ),
        "no_empty_key_points": (
            len(empty_key_points)
            == 0
        ),
        "valid_time_ranges": (
            len(invalid_time_ranges)
            == 0
        ),
        "all_report_visuals_aligned": (
            len(
                missing_aligned_visuals
            )
            == 0
        ),
        "no_duplicate_visuals": (
            len(
                duplicate_visuals
            )
            == 0
        ),
    }

    passed = all(
        checks.values()
    )

    result = {
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "metrics": {
            "sections": (
                section_count
            ),
            "aligned_sections": (
                aligned_section_count
            ),
            "section_coverage": (
                round(
                    section_coverage,
                    3,
                )
            ),
            "report_visuals": (
                len(
                    report_visuals
                )
            ),
            "aligned_visuals": (
                len(
                    aligned_visuals
                )
            ),
            "visual_coverage": (
                round(
                    visual_coverage,
                    3,
                )
            ),
            "sections_corrected_by_verifier": (
                corrected_sections
            ),
        },
        "checks": checks,
        "problems": {
            "empty_summaries": (
                empty_summaries
            ),
            "empty_key_points": (
                empty_key_points
            ),
            "invalid_time_ranges": (
                invalid_time_ranges
            ),
            "missing_aligned_visuals": (
                missing_aligned_visuals
            ),
            "duplicate_visuals": (
                duplicate_visuals
            ),
        },
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return result
