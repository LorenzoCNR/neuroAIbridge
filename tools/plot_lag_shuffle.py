"""Usage: python tools/plot_lag_shuffle.py --report PATH"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.lag_shuffle_plots import plot_lag_shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot observed lag profiles against a trial-pairing shuffle null")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(plot_lag_shuffle(args.report))
