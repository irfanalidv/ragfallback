"""Smoke-test the offline example scripts so the README recipes cannot rot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SCRIPTS = sorted(EXAMPLES.glob("offline_*.py"))


def test_offline_examples_exist() -> None:
    assert len(SCRIPTS) >= 4


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_offline_example_runs(script: Path) -> None:
    if script.name == "offline_hybrid_failover.py":
        pytest.importorskip("rank_bm25")
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
