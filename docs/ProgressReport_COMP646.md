# Multimodal Pedagogical Friction Detector: Progress Report

**Jingrui Wu · Keyuan Yan**  
Department of Computer Science, Rice University  
jw310@rice.edu · ky65@rice.edu  

**Spring 2026**

---

## Abstract

Effective teaching depends on noticing student confusion and adjusting instruction. We are building a system that detects *pedagogical friction*: moments where a low-quality instructional strategy coincides with visible student confusion. The pipeline takes a classroom or tutoring recording (video + audio), constructs a confusion timeline from facial affect (DeepFace), a strategy timeline from automatically transcribed speech (Whisper) and dialogue-level heuristics standing in for Edu-ConvoKit and Tutor CoPilot classifiers, aligns both streams on a 30-second grid, and applies a heuristic pre-filter before optional multimodal fusion with **Qwen2.5-VL** (replacing GPT-4o in our proposal) to produce a post-session **Teacher Friction Report**. We have implemented the end-to-end codebase, evaluated TIMSS-style public lessons locally, and run CPU smoke tests. Full VLM fusion and quantitative evaluation against human labels are in progress.

## 1. Introduction

Pedagogical friction bridges *what the teacher says* and *how students appear to respond*. Prior work such as Tutor CoPilot improves tutor strategies using text, and Edu-ConvoKit supports analysis of teacher discourse, but classroom video contains complementary evidence (facial affect, board work) that text-only tools miss. Our goal is to surface candidate friction intervals with timestamps, strategy context, and suggested higher-quality moves, grounded in accountable-talk style strategy taxonomies.

## 2. Related Work

- **Classroom discourse and feedback.** Talk Moves and related frameworks motivate labeling instructional moves (Suresh et al., AAAI 2021).
- **Tutor and dialogue modeling.** Tutor CoPilot and Edu-ConvoKit provide models and tooling we plan to integrate for production-quality strategy labels.
- **Multimodal reasoning.** Qwen2.5-VL supports image–text reasoning; we use it for fusion instead of GPT-4o for cost and reproducibility.
- **Affect from video.** Off-the-shelf emotion models (e.g., via DeepFace) offer a practical confusion proxy on CPU.

## 3. Methodology

**Vision stream:** Frames at configurable FPS (default 2.0; reduced for CPU tests); DeepFace emotion → per-frame confusion proxy; trailing 30 s mean → confusion timeline.

**Language stream:** Whisper (faster-whisper) transcription; utterances annotated with a *heuristic* layer for low-quality / high-pressure patterns, with hooks to replace by Edu-ConvoKit + Tutor CoPilot.

**Alignment and filtering:** 30 s bins; flag bins where confusion exceeds μ + zσ and strategy is low-quality or high-pressure.

**Fusion (optional):** Qwen2.5-VL receives frame crops + transcript excerpt + timeline summaries and outputs JSON (friction, rationale, alternative strategy).

**Stretch (not yet integrated end-to-end):** OCR on board crops.

**Figure (placeholder).** Replace with a PowerPoint/Keynote diagram exported to PDF (selectable text), matching the quality note in the sample ProgressReport.

**Table 1 — Preliminary plan**

| Setting | Precision | Recall | Notes |
|--------|-----------|--------|--------|
| Heuristic + fusion (full lesson) | *TBD* | *TBD* | Pending GPU / labels |
| Heuristic only (--skip-fusion) | — | — | Smoke tests on TIMSS clips |
| Human agreement (2 annotators) | *TBD* | — | Planned |

## 4. Experimental Settings

- **Data:** Public TIMSS Video lessons (YouTube mirrors); transcript archives from [timssvideo.com/resources](https://www.timssvideo.com/resources). Videos stay local (gitignored).
- **Software:** CPU path: `requirements-pipeline.txt`. Qwen: Python ≥ 3.10 and `requirements.txt`.
- **Code:** https://github.com/jw310-wjr/comp646-friction-detector  
- **Runtime:** Full lessons at 2 FPS mean many DeepFace calls; we support `--max-duration-sec` for trimmed runs.

## 5. Analysis and Discussion

### 5.1 Implementation status

Frame extraction, confusion aggregation, ASR, bin alignment, JSON/text reports, and GitHub release are complete. Short clips may yield zero candidates under conservative *z*; we will report tuning in the final report.

### 5.2 Fusion and compute

Qwen integration is in-repo but primary development machine ran Python 3.9 with `--skip-fusion`; fusion will be run on 3.10+ with GPU.

### 5.3 Limitations

Heuristic tags are not yet full Edu-ConvoKit / Tutor CoPilot checkpoints. Ground-truth friction labels are still to be collected.

## 6. Conclusion

We delivered a working prototype and public repository with TIMSS-aligned text resources. Next: VLM fusion at scale, classifier integration, human annotation, and metrics from §3.

## 7. Ethical and Societal Considerations

Classroom video involves minors and privacy: we use only publicly released TIMSS public-use materials and follow site terms. Affect inference can misread culture, disability, or camera angle; outputs are for teacher reflection, not high-stakes decisions. We recommend human review and clear disclaimers.

## Acknowledgments

We used AI coding assistants; all technical claims remain our responsibility. Thanks to TIMSS Video and TalkMoves for public materials.

## References

1. R. E. Wang et al. *Tutor CoPilot*. arXiv:2410.03017, 2025.  
2. R. E. Wang & D. Demszky. *Edu-ConvoKit*. NAACL Demo, 2024.  
3. A. Suresh et al. *TalkMoves Application*. AAAI, 2021.  
4. Qwen Team. *Qwen2.5-VL-7B-Instruct*. Hugging Face, 2025. https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct  

---

*PDF: compile `docs/ProgressReport_COMP646.tex` with pdflatex or paste this Markdown into Google Docs / Word and export PDF.*
