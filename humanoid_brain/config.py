"""Shared configuration for the humanoid brain package."""

from typing import List, Tuple

TASK_LABELS: List[str] = [
    "cleaning",
    "cooking",
    "dishwashing",
    "laundry",
    "organizing",
]

DEFAULT_DEVICE: str = "cpu"
DEFAULT_MIN_CONFIDENCE: float = 0.60
DEFAULT_INPUT_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
