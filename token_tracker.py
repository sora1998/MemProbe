"""
token_tracker.py — Shared token usage tracker for mem_bench.

Both config.py (agent/simulator calls) and A-mem's llm_controller.py
(memory evolution calls) import from here so all OpenAI spend is captured
in one place.

Log file: usage/<run_name>_<session_timestamp>.txt
The run_name is set by the runner via ``set_run_name(...)``; if not set
the file is named ``unnamed_<timestamp>.txt`` so files from different
runs never share a name.
"""

from __future__ import annotations

import datetime
import os
from typing import Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
USAGE_DIR = os.path.join(_HERE, "usage")

_SESSION_START: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_RUN_NAME: Optional[str] = None
_token_usage: Dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


def set_run_name(run_name: Optional[str]) -> None:
    """Tag this process's usage log with a human-readable run name."""
    global _RUN_NAME
    _RUN_NAME = run_name


def get_run_name() -> Optional[str]:
    return _RUN_NAME


def _session_log() -> str:
    os.makedirs(USAGE_DIR, exist_ok=True)
    name_part = _RUN_NAME if _RUN_NAME else "unnamed"
    return os.path.join(USAGE_DIR, f"{name_part}_{_SESSION_START}.txt")


def log_usage(model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    """Record one API call and append a line to the session log."""
    _token_usage["prompt_tokens"]     += prompt_tokens
    _token_usage["completion_tokens"] += completion_tokens
    _token_usage["total_tokens"]      += total_tokens

    run_tag = _RUN_NAME if _RUN_NAME else "unnamed"
    line = (
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"run={run_tag} "
        f"model={model} "
        f"prompt={prompt_tokens} "
        f"completion={completion_tokens} "
        f"total={total_tokens} "
        f"| cumulative={_token_usage['total_tokens']}"
    )
    with open(_session_log(), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_usage() -> Dict[str, int]:
    """Return accumulated token usage since last reset (or process start)."""
    return dict(_token_usage)


def reset_usage() -> None:
    """Reset counters (call between users if you want per-user stats)."""
    _token_usage["prompt_tokens"]     = 0
    _token_usage["completion_tokens"] = 0
    _token_usage["total_tokens"]      = 0
