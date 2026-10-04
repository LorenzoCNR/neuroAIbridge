"""Usage: python tools/evaluate_frozen.py --run PATH --output PATH [--branch BRANCH]"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neurobridge.experiments.frozen_sensitivity import evaluate_frozen_run, read_config, repeated_seed_plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and compare raw/unit frozen representations without training")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=("held_out", "full_sample"), default="held_out")
    parser.add_argument(
        "--record-checkpoint-mismatches",
        action="store_true",
        help="Record historical checkpoint replay differences while evaluating saved frozen embeddings.",
    )
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    result = evaluate_frozen_run(
        ROOT,
        args.run,
        args.output,
        branch=args.branch,
        record_checkpoint_mismatches=args.record_checkpoint_mismatches,
    )
    (args.output / "repeated_seed_plan.json").write_text(
        json.dumps(repeated_seed_plan(read_config(args.run)), indent=2), encoding="utf-8")
    print(result)
