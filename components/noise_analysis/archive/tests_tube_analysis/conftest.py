"""Test path setup for noise-analysis modules."""

from __future__ import annotations

import sys
from pathlib import Path

NOISE_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
QLBM_SOURCE_DIR = PROJECT_ROOT / "qlbm"

for path in (NOISE_ANALYSIS_DIR, QLBM_SOURCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

