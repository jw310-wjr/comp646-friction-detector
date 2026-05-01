# Multimodal pedagogical friction detector (COMP 646)

> **Reproducibility note:** The accompanying paper reports results using `claude-sonnet-4-6` (Anthropic API) as the LLM fusion stage. To reproduce the exact paper results, set the environment variable `ANTHROPIC_API_KEY` before running `run_session.py`. All reported metrics (F1 = 1.00 on AU1; 7.7× precision gain across 53 lessons) were obtained with the Claude backend.

Pipeline: **video + audio** → DeepFace emotion timeline → Whisper ASR → heuristic dialogue strategy tags → LLM fusion → teacher friction report.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pipeline.txt
python scripts/run_session.py --video "./data/timss/your_lesson.mp4" --work-dir "./runs/demo" --skip-fusion
```

Use `--max-duration-sec 600` for a shorter CPU-friendly run on long lessons. VLM fusion needs Python **≥ 3.10** and `pip install -r requirements.txt` (see comments in that file).

## Push to your GitHub

Account: **[jw310-wjr](https://github.com/jw310-wjr)**. After pushing, the repo will be at `https://github.com/jw310-wjr/<repo-name>` (default: [`comp646-friction-detector`](https://github.com/jw310-wjr/comp646-friction-detector)).

1. Install and log in: `brew install gh` then `gh auth login` (use the same GitHub user as above).
2. From this directory: `./scripts/push_to_github.sh [repo-name] [public|private]`

Default repo name: `comp646-friction-detector`

## Progress report (CVPR-style template)

Course two-column template: `docs/ProgressReport_cvprformat.tex` + `docs/egbib.bib`. Copy **`cvpr.sty`** and **`ieee.bst`** from your class author kit into `docs/`, then see `docs/CVPR_COMPILE.txt`.

## Models and checkpoints

| Component | Checkpoint | Notes |
|---|---|---|
| Talk Move classifier | [`YaHi/teacher_electra_small`](https://huggingface.co/YaHi/teacher_electra_small) | Fine-tuned ELECTRA on TalkMoves corpus |
| ASR | `faster-whisper` `small` / `base` | Apache 2.0 |
| Facial affect | `DeepFace` (default backend) | MIT |
| LLM fusion | `claude-sonnet-4-6` | Anthropic API; set `ANTHROPIC_API_KEY` |

**TalkMoves corpus** — the `YaHi/teacher_electra_small` checkpoint was trained on the TalkMoves dataset (Suresh et al., AAAI 2021 Spring Symposium). If you use it in published work, cite that paper and check the dataset's terms of use at the original release.

## Data

Public TIMSS transcripts from [timssvideo.com/resources](https://www.timssvideo.com/resources) are included under `data/timss/`. **Lesson `.mp4` files are gitignored**—add videos locally after clone.

## Results

| Condition | Bins | Flagged | F1 |
|---|---|---|---|
| Heuristic (AU1) | 87 | 20 | 0.10 |
| ELECTRA (AU1) | 87 | 33 | 0.00 |
| Heuristic + Claude (AU1) | 87 | 1 | **1.00** |
| Heuristic + Claude (53 lessons) | 5113 | 211 | — |

Full experimental details in the paper (see Citation below).

## Citation

If you use this code or data, please cite:

```
Jingrui Wu, "Multimodal Pedagogical Friction Detector,"
COMP 646, Rice University, 2025.
```

## License

Code: [MIT License](LICENSE).
TIMSS transcript data: follow [TIMSSVIDEO](https://www.timssvideo.com/) terms of use.
