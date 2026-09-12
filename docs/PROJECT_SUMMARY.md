# Video2Knowledge — Technical Project Summary

## Short portfolio description

Built a multimodal video-to-knowledge system that combines speech transcription, visual knowledge extraction, grounded LLM synthesis, faithfulness verification, resumable processing, pluggable local/cloud AI providers, and automated PDF report generation.

## Problem statement

Educational videos distribute knowledge across narration and visual material. Transcript-only tools miss diagrams, equations, code, tables, slides, whiteboards, and UI demonstrations; unconstrained generation can also produce plausible but unsupported claims.

## System goals

Video2Knowledge converts a local educational video into a structured, evidence-grounded PDF while keeping every run isolated and restartable. It provides a simple default workflow for non-technical users and explicit provider controls for technical users.

## Core architecture

The modular monolith has three layers:

1. A deterministic media and knowledge pipeline for ingestion, preprocessing, segmentation, candidate extraction, ranking, alignment, PDF rendering, and QA.
2. Provider protocols for ASR, visual intelligence, synthesis, and verification, with a central profile resolver/factory.
3. An application service consumed by both the backward-compatible CLI and Streamlit UI.

## Transcript pipeline

FFmpeg extracts mono 16 kHz audio. Faster Whisper runs locally on CPU or CUDA, producing timestamped segments. A TF-IDF lexical baseline compares neighboring context windows and combines similarity, transition phrases, continuation rules, and duration limits to create sections.

## Visual extraction pipeline

Scene-change detection supplies high-recall candidates, while transcript cues target explicitly referenced diagrams, tables, equations, screens, and related material. Candidate fusion combines evidence, image similarity removes near-duplicates, and a richness/change score ranks frames before costly visual inference. Only selected frames—not full videos—reach cloud vision APIs.

## Multimodal reasoning

Visual providers receive one candidate plus nearby transcript context. Accepted knowledge visuals are aligned to sections using timestamp containment and TF-IDF similarity near section boundaries. Synthesis is section-isolated to reduce cross-topic leakage.

## Provider abstraction and execution modes

Typed profiles resolve into a `ProviderConfig`, and a factory supplies implementations for the capabilities requested by the pipeline:

- Local ASR: Faster Whisper
- Cloud vision/synthesis/verification: Gemini
- Local vision/synthesis/verification: explicitly installed Ollama models

Smart mode chooses local media/ASR plus Gemini when credentials exist; otherwise it uses configured local models or returns a setup error. Cloud and custom profiles validate credentials and model selections early. Private mode requires no API key after local models are installed.

## Checkpointing and reliability

Expensive artifacts are stored under the current run directory. Transcription, segmentation, candidates, ranking, per-frame visual analysis, alignment, per-section synthesis, and per-section verification can resume. Fingerprints prevent incompatible synthesis/verification checkpoints from being reused. Generated report frames are also run-local.

Reliability work included handling container-duration/frame-tail mismatches, moving FFmpeg seeks before input decoding, rejecting unsupported numeric generation, preserving grounded drafts when verifier rewrites are unsafe, and preventing blank PDF pages.

## Testing

The suite covers legacy ingestion/preprocessing/transcription/segmentation behavior plus profile resolution, Smart fallback, credential validation, provider factories, hardware detection, safe filenames, upload containment, run isolation, service results, and side-effect-free UI imports. Cloud access is not required. CI generates deterministic media rather than committing large test videos.

## Deployment

The Streamlit server exposes the same application service as the CLI. A non-root Python 3.11 CPU Docker image includes FFmpeg and a Python-based healthcheck. Compose persists run outputs and model caches. An optional override requests NVIDIA GPUs and enables CUDA Faster Whisper; it does not affect standard CPU deployment.

## Security considerations

Secrets remain in server-side environment variables and `.env` is ignored. Upload filenames are sanitized, paths are resolved beneath unique run directories, subprocess calls use argument arrays, URL ingestion is disabled, and generated data/model caches are excluded from Git and Docker contexts.

## Technical challenges solved

- Preserving a validated pipeline while introducing dependency-injected providers
- Keeping provider complexity outside the normal user workflow
- Maintaining resumability across costly multimodal stages
- Separating compact analysis assets from presentation-quality source frames
- Enforcing numerical grounding and rejecting unsafe verifier changes
- Supporting Windows development and Linux CPU/GPU container configurations

## Lessons learned

Generating a plausible summary is not the same as generating a faithful one. Explicit evidence boundaries, numerical checks, intermediate artifacts, and a dedicated verifier make failures observable and recoverable. The project also reinforced that hybrid local/cloud design is a product decision as much as a model decision: privacy, hardware, latency, and configuration need clear user-facing behavior.

## Limitations

Visual candidate extraction remains heuristic, segmentation uses TF-IDF, Ollama output quality depends on user-selected models, URL ingestion is unavailable, and the evaluator is structural rather than a large human-reviewed benchmark.

## Roadmap

Add provider usage/cost metadata, robust URL ingestion, more providers including cloud ASR, visual-recall evaluation, richer report layouts, batch processing, and hosted-provider routing without exposing credentials to clients.
