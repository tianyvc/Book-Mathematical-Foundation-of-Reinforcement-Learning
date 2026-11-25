"""Launch a weekly experiment: python -m practice.run WEEK --seed 0 --steps 10000."""

import argparse
import importlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("week", type=int, choices=range(1, 13))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    output = args.output or Path("practice/results") / f"week{args.week:02}" / f"seed{args.seed}"
    module = importlib.import_module(f"practice.experiments.week{args.week:02}")
    module.run(seed=args.seed, steps=args.steps, output=output)


if __name__ == "__main__":
    main()
