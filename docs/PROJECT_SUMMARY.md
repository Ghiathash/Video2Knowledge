# Video2Knowledge — Project Summary

## Problem

Educational videos contain knowledge across speech and visuals, but most summarizers rely mainly on transcripts. Important diagrams, tables, equations, slides, code, whiteboards, and screen demonstrations can therefore disappear from the result.

## Solution

Video2Knowledge is a multimodal pipeline that combines ASR, topic segmentation, visual candidate generation, VLM analysis, transcript–visual alignment, grounded synthesis, and faithfulness verification to produce structured PDF knowledge reports.

## Engineering highlights

- Built a high-recall hybrid extractor that combines scene changes with transcript-guided visual cues, followed by fusion, near-duplicate removal, and ranking.
- Isolated synthesis context by section and grounded numerical claims in transcript or visible-text evidence.
- Added a dedicated verifier for unsupported claims and altered semantic relationships such as `AND` versus `OR`.
- Separated low-cost analysis images from full-resolution source frames used in the final PDF.
- Designed run-local artifacts and file checkpoints so long, costly workflows can resume without mixing outputs.

## Technical stack

Python, Faster Whisper, FFmpeg, OpenCV, NumPy, scikit-learn, Gemini via Google Gen AI SDK, Pydantic, Pillow, ReportLab, and pytest.

## Scope

The current evaluation validates pipeline integrity and report structure. Visual extraction is heuristic, topic segmentation uses a TF-IDF baseline, and output quality depends on source media and available ASR/VLM compute.
