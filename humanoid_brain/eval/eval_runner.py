"""Evaluation CLI for task classifier."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, List

import torch

from humanoid_brain.eval.dataset_loader import collect_class_names, create_dataloader, load_dataset
from humanoid_brain.models.task_classifier import TaskClassifier


def _format_confusion_matrix(classes: List[str], matrix: Dict[str, Dict[str, int]]) -> str:
    header = "true\\pred," + ",".join(classes)
    lines = [header]
    for true_label in classes:
        row = [true_label] + [str(matrix[true_label][pred]) for pred in classes]
        lines.append(",".join(row))
    return "\n".join(lines)


def run_eval(weights: str, dataset_jsonl: str, images_root: str, batch_size: int, device: str) -> None:
    dataset = load_dataset(dataset_jsonl=dataset_jsonl, images_root=images_root)
    dataloader = create_dataloader(dataset, batch_size=batch_size, num_workers=0)
    classes = collect_class_names(dataset)
    classifier = TaskClassifier(weights_path=weights, device=device)

    total = 0
    correct = 0
    per_task_total = Counter()
    per_task_correct = Counter()
    confusion = defaultdict(lambda: defaultdict(int))

    for x_batch, labels in dataloader:
        # Evaluate sample-wise because SDK predict() accepts one image at a time.
        for i in range(x_batch.shape[0]):
            image_tensor = x_batch[i]
            image = image_tensor.permute(1, 2, 0).cpu().numpy()
            pred = classifier.predict(image)
            y_true = labels[i]
            y_pred = pred["label"]
            if y_pred == "unknown":
                probs = pred["probs"]
                y_pred = max(probs, key=probs.get)

            total += 1
            per_task_total[y_true] += 1
            confusion[y_true][y_pred] += 1
            if y_true == y_pred:
                correct += 1
                per_task_correct[y_true] += 1

    overall_acc = (correct / total * 100.0) if total else 0.0
    print(f"Overall accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print("Per-task accuracy:")
    for cls in classes:
        t = per_task_total[cls]
        c = per_task_correct[cls]
        acc = (c / t * 100.0) if t else 0.0
        print(f"  {cls:12s} {acc:6.2f}% ({c}/{t})")

    print("\nConfusion matrix (CSV format):")
    print(_format_confusion_matrix(classes, confusion))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate humanoid task classifier.")
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--dataset-jsonl", required=True, type=str)
    parser.add_argument("--images-root", required=True, type=str)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    run_eval(
        weights=args.weights,
        dataset_jsonl=args.dataset_jsonl,
        images_root=args.images_root,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
