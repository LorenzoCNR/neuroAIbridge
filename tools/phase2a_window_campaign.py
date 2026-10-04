"""Entrypoint for the versioned four-window Real Phase-2A campaign."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.phase2a_window_campaign import main  # noqa: E402

if __name__ == "__main__":
    main()
