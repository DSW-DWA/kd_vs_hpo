import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger("hpo.events")


def configure_console_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class ExperimentEventLog:
    """Append-only JSONL event log that is flushed after every event."""

    def __init__(self, path: Path, context: dict[str, Any] | None = None) -> None:
        configure_console_logging()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._context = context or {}

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **self._context,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()
        logger.info("%s | %s", event, json.dumps(fields, ensure_ascii=False, default=str))

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> "ExperimentEventLog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
