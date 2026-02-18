# Model Card: 5-Task Robot Classifier

## Model
- Name: `best_licensed.pt`
- Architecture: `MobileNetV3-Small` classifier
- Tasks: `cleaning`, `cooking`, `dishwashing`, `laundry`, `organizing`

## Intended Use
- Classify household task from RGB frame for non-safety-critical robot workflows.

## Out of Scope
- Safety-critical decisions
- Medical/legal/security use
- Tasks outside the 5-class scope

## Training Data
- Source: Licensed/owned dataset only
- License ledger: `data_licenses.csv`
- Dataset version: `licensed_dataset_v1` (fill exact hash/version)

## Evaluation Protocol
- External set: `isaac_eval/labels.csv` + frames
- Command:
```bash
python isaac_eval.py --checkpoint best_licensed.pt --labels-csv isaac_eval/labels.csv --images-root isaac_eval
```

## Reported Metrics (fill before publish)
- Overall accuracy: `__%`
- Cleaning: `__%`
- Cooking: `__%`
- Dishwashing: `__%`
- Laundry: `__%`
- Organizing: `__%`

## Known Failure Modes
- Visual confusion between cleaning vs organizing in cluttered environments
- Domain shift from training camera to deployment camera
- Low light / motion blur

## Mitigations
- Confidence thresholding
- Human-in-the-loop fallback
- Periodic retraining with licensed edge cases

## Versioning
- Model version: `v1.0.0`
- Date: `YYYY-MM-DD`
- Owner: `Company/Team`
