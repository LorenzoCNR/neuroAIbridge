"""Usage: python tools/plot_frozen_audit.py --report PATH"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.frozen_report_plots import generate_frozen_figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a completed frozen raw/unit audit without re-evaluating models")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for name, path in generate_frozen_figures(args.report).items():
        print(f"{name}: {path}")
