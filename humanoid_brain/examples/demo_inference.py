"""Demo script for non-ROS SDK inference."""

import argparse
import json

from PIL import Image

from humanoid_brain.sdk.inference_api import load_brain
from humanoid_brain.telemetry.logger import TelemetryLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-shot chore inference.")
    parser.add_argument("--weights", type=str, default="best_licensed_balanced.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--telemetry-jsonl", type=str, default=None)
    args = parser.parse_args()

    telemetry = TelemetryLogger(jsonl_path=args.telemetry_jsonl, to_stdout=True)
    brain = load_brain(
        weights_path=args.weights,
        device=args.device,
        min_confidence=args.min_confidence,
        telemetry_logger=telemetry,
    )

    image = Image.open(args.image).convert("RGB")
    decision = brain.decide(image=image, robot_state={}, env_state={})
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
