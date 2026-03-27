"""Shared timing utility for logging stage wall-clock times.

Appends one line per stage to a timing log file (default: output/timing.log).

Format:
    2026-03-26 04:04:44 | stage2_finetune | 2131.5s (35m 31s) | config: eps10, 974 steps
"""

import time
from datetime import datetime
from pathlib import Path

DEFAULT_TIMING_LOG = "output/timing.log"


class Timer:
    """Context manager that logs elapsed time to a shared timing log file.

    Usage:
        with Timer("stage2_finetune", notes="eps10, 974 steps"):
            train(...)

        # Or with a custom log path:
        with Timer("stage3_generate", log_file="output/timing.log", notes="1000 tables"):
            generate(...)
    """

    def __init__(self, stage_name: str, log_file: str = DEFAULT_TIMING_LOG, notes: str = ""):
        self.stage_name = stage_name
        self.log_file = log_file
        self.notes = notes
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        self._write_log(exc_type is not None)
        return False  # don't suppress exceptions

    def _write_log(self, failed: bool):
        mins, secs = divmod(self.elapsed, 60)
        hrs, mins = divmod(int(mins), 60)

        if hrs > 0:
            human = f"{hrs}h {int(mins)}m {secs:.0f}s"
        elif mins > 0:
            human = f"{int(mins)}m {secs:.0f}s"
        else:
            human = f"{secs:.1f}s"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "FAILED" if failed else "OK"
        parts = [timestamp, self.stage_name, f"{self.elapsed:.1f}s ({human})", status]
        if self.notes:
            parts.append(self.notes)
        line = " | ".join(parts)

        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
