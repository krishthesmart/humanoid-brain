"""Structured telemetry logger."""

import json
from pathlib import Path
from typing import Optional

from .events import BaseEvent


class TelemetryLogger:
    """Logs telemetry events to stdout and/or JSONL."""

    def __init__(self, jsonl_path: Optional[str] = None, to_stdout: bool = True):
        self.to_stdout = to_stdout
        self.jsonl_path = jsonl_path
        self._jsonl_file = None

        if jsonl_path:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = path.open("a", encoding="utf-8")

    def log_event(self, event: BaseEvent) -> None:
        """Persist one event."""
        payload = event.to_dict()
        line = json.dumps(payload, ensure_ascii=True)
        if self.to_stdout:
            print(line)
        if self._jsonl_file:
            self._jsonl_file.write(line + "\n")
            self._jsonl_file.flush()

    def close(self) -> None:
        """Close underlying file handle."""
        if self._jsonl_file:
            self._jsonl_file.close()
            self._jsonl_file = None

    def __del__(self) -> None:
        self.close()
