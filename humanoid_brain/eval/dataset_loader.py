"""Dataset loading utilities for classifier evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from humanoid_brain.config import DEFAULT_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


class ClassificationDataset(Dataset):
    """Image classification dataset from rows with image path + label."""

    def __init__(self, rows: List[Dict[str, str]], images_root: str, transform: Optional[transforms.Compose] = None):
        self.rows = rows
        self.images_root = Path(images_root)
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(DEFAULT_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_rel = row["image"]
        label = row["label"]
        image_path = (self.images_root / image_rel).resolve()
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image)
        return x, label


def _load_jsonl(dataset_jsonl: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(dataset_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append({"image": item["image"], "label": item.get("task", item.get("label"))})
    return rows


def _load_csv(dataset_csv: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(dataset_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for item in reader:
            rows.append({"image": item["image"], "label": item.get("task", item.get("label"))})
    return rows


def load_dataset(dataset_jsonl: Optional[str] = None, dataset_csv: Optional[str] = None, images_root: str = ".") -> ClassificationDataset:
    """Load JSONL or CSV classification dataset."""
    if not dataset_jsonl and not dataset_csv:
        raise ValueError("Provide dataset_jsonl or dataset_csv")
    rows = _load_jsonl(dataset_jsonl) if dataset_jsonl else _load_csv(dataset_csv)  # type: ignore[arg-type]
    if not rows:
        raise RuntimeError("Dataset is empty.")
    return ClassificationDataset(rows=rows, images_root=images_root)


def create_dataloader(dataset: ClassificationDataset, batch_size: int = 16, num_workers: int = 0) -> DataLoader:
    """Create dataloader for evaluation."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def collect_class_names(dataset: ClassificationDataset) -> List[str]:
    """Return sorted class names from dataset labels."""
    return sorted(set(row["label"] for row in dataset.rows))
