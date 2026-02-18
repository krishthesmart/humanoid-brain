# Commercial Runbook (5 Task Robot Model)

This runbook is the exact operational path to a paid, commercial-ready release.

## 0) Scope
- Tasks: `cleaning`, `cooking`, `dishwashing`, `laundry`, `organizing`
- Model: `train_task_classifier.py` (MobileNetV3 small)
- Evaluator: `isaac_eval.py`

## 1) Environment Setup
```bash
cd /Users/arulmeiyappan/training_data
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Build Licensed Dataset Only
Do not use unlicensed/scraped data for commercial release.

1. Put licensed images here:
```text
licensed_dataset/
  images/
    cleaning/
    cooking/
    dishwashing/
    laundry/
    organizing/
```

2. Create dataset JSONL:
```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("licensed_dataset/images")
out = Path("licensed_dataset/dataset.jsonl")
tasks = ["cleaning","cooking","dishwashing","laundry","organizing"]

rows = []
for t in tasks:
    for p in sorted((root / t).glob("*")):
        if p.is_file():
            rows.append({"image": f"images/{t}/{p.name}", "task": t, "instruction": f"A person performing {t}"})

out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print("wrote", out, "rows=", len(rows))
PY
```

3. Fill license ledger:
- `data_licenses.csv`

## 3) Train (Validation/Test Split)
Use split mode (no `--train-all`) for publishable metrics:
```bash
caffeinate -dimsu python -u train_task_classifier.py \
  --dataset licensed_dataset/dataset.jsonl \
  --dataset-root licensed_dataset \
  --epochs 20 \
  --batch-size 16 \
  --num-workers 0 \
  --lr 1e-4 \
  --checkpoint-dir checkpoints_licensed \
  --output best_licensed.pt \
  --per-task-each-epoch
```

Resume if interrupted:
```bash
caffeinate -dimsu python -u train_task_classifier.py \
  --dataset licensed_dataset/dataset.jsonl \
  --dataset-root licensed_dataset \
  --epochs 20 \
  --batch-size 16 \
  --num-workers 0 \
  --lr 1e-4 \
  --checkpoint-dir checkpoints_licensed \
  --output best_licensed.pt \
  --per-task-each-epoch \
  --resume
```

## 4) Prepare Frozen External Eval Set
Create a fixed external set (not used in training), e.g.:
```text
isaac_eval/
  frames/
  labels.csv   # columns: image,task
```

`labels.csv` example:
```csv
image,task
frames/cleaning_0001.png,cleaning
frames/laundry_0001.png,laundry
```

## 5) External Accuracy (Release Gate)
```bash
python isaac_eval.py \
  --checkpoint best_licensed.pt \
  --labels-csv isaac_eval/labels.csv \
  --images-root isaac_eval
```

Release gate (recommended):
- Overall accuracy `>= 85%`
- Per-task accuracy `>= 85%` for all 5 tasks

## 6) Package Release Artifacts
Create a release folder:
```bash
mkdir -p release_v1
cp best_licensed.pt release_v1/
cp model_card.md release_v1/
cp acceptance_criteria.md release_v1/
cp pilot_sow_template.md release_v1/
```

Include:
- model file
- eval report output
- dataset license ledger
- scope/limitations doc

## 7) Pilot Commercial Rollout
Use `pilot_sow_template.md` and define:
- price (`$1,000/robot`)
- acceptance test
- deployment boundaries
- support/SLA

Only convert pilot to full production contract after acceptance criteria are met on customer scenes.
