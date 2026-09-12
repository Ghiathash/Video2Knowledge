I built Video2Knowledge because educational videos contain important information not only in speech, but also in diagrams, tables, equations, slides, code, whiteboards, and UI demonstrations.

Transcript-only summarization can miss that second channel entirely. Video2Knowledge combines speech-to-text, lexical topic segmentation, hybrid visual candidate extraction, visual understanding, transcript–visual alignment, grounded synthesis, and faithfulness verification. The final output is a structured PDF that pairs grounded knowledge with relevant frames re-extracted from the source video.

This was also a practical learning project around ASR, NLP segmentation, multimodal systems, visual understanding, grounded generation, and hallucination/faithfulness verification.

The clearest lesson was: “Generating a plausible summary is not the same as generating a faithful one.”

During development, the system exposed cases where a model changed semantic relationships—for example, turning an AND relationship into OR. That motivated a dedicated verification layer which checks the draft against transcript and visible-text evidence before producing the report.

The visual extraction is still heuristic, and the current evaluation focuses on pipeline and structural quality rather than claiming benchmark-level performance. But it has been a useful exercise in designing an AI pipeline where evidence, intermediate artifacts, and failure modes stay visible.

I’d welcome technical feedback, especially on multimodal evaluation and visual-recall strategies.

#ArtificialIntelligence #GenerativeAI #MultimodalAI #MachineLearning #Python #AIEngineering
