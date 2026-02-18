# Acceptance Criteria (Commercial)

## Scope
- Model must classify 5 tasks:
  - `cleaning`
  - `cooking`
  - `dishwashing`
  - `laundry`
  - `organizing`

## Test Dataset
- Customer-approved, fixed evaluation set
- Balanced by task (recommended: >= 50 samples per task)
- Not used during training

## Pass/Fail Metrics
- Overall accuracy: `>= 85%`
- Per-task accuracy: `>= 85%` for each of the 5 tasks

## Test Command
```bash
python isaac_eval.py \
  --checkpoint best_licensed.pt \
  --labels-csv <eval_labels.csv> \
  --images-root <eval_root>
```

## Failure Handling
- If any task < 85%, model fails acceptance
- Vendor retrains and resubmits with changelog and new eval report

## Non-Functional Requirements
- Inference timeout (fill): `__ ms`
- Minimum confidence threshold (fill): `__`
- Logging enabled for predictions and confidence

## Sign-off
- Customer signer: `__________`
- Vendor signer: `__________`
- Date: `YYYY-MM-DD`
