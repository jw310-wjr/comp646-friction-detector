# Multimodal pedagogical friction detector (COMP 646)

Pipeline: **video + audio** → DeepFace emotion timeline → Whisper ASR → heuristic dialogue strategy tags → optional **Qwen2.5-VL** fusion → teacher friction report.

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

## Data

Public TIMSS transcripts from [timssvideo.com/resources](https://www.timssvideo.com/resources) are included under `data/timss/`. **Lesson `.mp4` files are gitignored**—add videos locally after clone.

## License

TIMSS transcript data: follow [TIMSSVIDEO](https://www.timssvideo.com/) terms. Code in this repo: add your chosen license if distributing beyond coursework.
