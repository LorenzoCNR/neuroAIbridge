"""Usage: python tools/evaluate_lag_shuffle.py --run PATH --output PATH [--branch BRANCH]"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.lag_shuffle import evaluate_lag_shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cross-population lag against a trial-pairing shuffle null")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=("held_out", "full_sample"), default="held_out")
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(evaluate_lag_shuffle(
        ROOT,
        args.run,
        args.output,
        branch=args.branch,
        n_permutations=args.n_permutations,
        seed=args.seed,
    ))
