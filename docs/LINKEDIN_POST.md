Educational videos do not store knowledge in one place.

Some of it is spoken. Some of it lives in diagrams, equations, code, tables, slides, whiteboards, and software demonstrations. That was the problem behind Video2Knowledge: a transcript alone is not enough to produce a useful study report.

I built a multimodal pipeline that combines local speech recognition, lexical transcript segmentation, hybrid visual candidate extraction, visual understanding, transcript–visual alignment, grounded synthesis, faithfulness verification, and PDF generation.

The architecture now supports a simple Smart mode as well as Private, Cloud, and Advanced configurations. Media processing, frame extraction, and PDF rendering stay local. AI-dependent stages use small provider interfaces, so the same pipeline can run with local Faster Whisper and configurable Ollama models, Gemini, or a hybrid of both.

Two engineering details mattered more than I initially expected:

- Long video workflows must be resumable. Transcription, frame analysis, synthesis, and verification use run-local checkpoints so a quota limit or connection failure does not discard completed work.
- Generating a plausible summary is not the same as generating a faithful one. I saw model output change semantic relationships such as AND/OR and introduce unsupported numbers, which motivated explicit numerical grounding and a verifier that cannot overwrite a safer draft with an unsupported rewrite.

I tested the pipeline on real educational videos covering animated neural-network explanations and programming/IDE content. The project also has offline unit tests, a Streamlit interface, CPU Docker packaging, an optional NVIDIA GPU configuration, and GitHub CI.

It is still an engineering portfolio project rather than a claim of perfect summarization: visual extraction is heuristic, local quality depends on the selected models, and the current evaluation is mainly structural and grounding-focused.

I would value technical feedback on multimodal evaluation, visual recall, and practical local/cloud routing.

#MultimodalAI #AIEngineering #MachineLearning #GenerativeAI #Python
