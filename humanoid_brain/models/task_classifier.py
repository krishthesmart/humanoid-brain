"""Task classifier model wrapper."""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from humanoid_brain.config import (
    DEFAULT_DEVICE,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MIN_CONFIDENCE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TASK_LABELS,
)
from humanoid_brain.telemetry.events import ErrorEvent, TaskDecisionEvent
from humanoid_brain.telemetry.logger import TelemetryLogger


ImageLike = Union[np.ndarray, Image.Image]


class TaskClassifier:
    """Inference wrapper for 5-task MobileNetV3-Small classifier."""

    def __init__(
        self,
        weights_path: str,
        device: str = DEFAULT_DEVICE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        telemetry_logger: Optional[TelemetryLogger] = None,
    ):
        self.device = torch.device(device)
        self.min_confidence = min_confidence
        self.telemetry = telemetry_logger

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.class_names = checkpoint.get("classes", TASK_LABELS)
        self.model = self._build_model(num_classes=len(self.class_names))
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(DEFAULT_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @staticmethod
    def _build_model(num_classes: int) -> torch.nn.Module:
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def _to_pil(image: ImageLike) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be numpy array or PIL.Image")
        if image.dtype != np.uint8:
            clipped = np.clip(image, 0.0, 1.0) if image.max() <= 1.0 else np.clip(image, 0.0, 255.0)
            image = (clipped * 255.0).astype(np.uint8) if clipped.max() <= 1.0 else clipped.astype(np.uint8)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("image must be HWC with 3 or 4 channels")
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return Image.fromarray(image).convert("RGB")

    def predict(self, image: ImageLike) -> Dict[str, object]:
        """
        Predict task label and class probabilities.

        Returns:
          {
            "label": str,            # one of known classes or "unknown"
            "probs": Dict[str, float]
          }
        """
        try:
            pil = self._to_pil(image)
            x = self.transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs_tensor = torch.softmax(logits, dim=1)[0].cpu()
            probs = {name: float(probs_tensor[idx]) for idx, name in enumerate(self.class_names)}
            label = max(probs, key=probs.get)
            confidence = probs[label]
            if confidence < self.min_confidence:
                label = "unknown"

            if self.telemetry:
                self.telemetry.log_event(TaskDecisionEvent(label=label, probs=probs, confidence=confidence))
            return {"label": label, "probs": probs}
        except Exception as exc:
            if self.telemetry:
                self.telemetry.log_event(
                    ErrorEvent(source="TaskClassifier.predict", message=str(exc), details={"type": type(exc).__name__})
                )
            raise
