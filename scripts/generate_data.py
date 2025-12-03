"""
CLI entry point for generating Knapsack datasets.

Usage:
    python -m scripts.generate_data --n_samples 50
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path to allow importing 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import (
    DEFAULT_NUM_RANDOM,
    DEFAULT_VALUE_RANGE,
    generate_easy_instances_bulk,
    generate_hard_instances_bulk,
    generate_random_instances_bulk,
    save_instance,
)

# Sizes: Small for validation, Large for scaling experiments
SIZES_TO_TEST = list(range(5, 100, 5)) + list(range(100, 1001, 100))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Knapsack Dataset")

    # Group 1: General Settings
    core_group = parser.add_argument_group("Core Settings")
    core_group.add_argument(
        "--out", type=str, default="data", help="Output root directory"
    )
    core_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Group 2: Generator Specifics
    gen_group = parser.add_argument_group("Generator Configuration")
    gen_group.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_NUM_RANDOM,
        help="Number of instances per size (for Random) or Max Subsamples (for Easy/Hard)",
    )
    gen_group.add_argument(
        "--max_val",
        type=int,
        default=DEFAULT_VALUE_RANGE,
        help="Max value/weight for random items",
    )

    args = parser.parse_args()

    # Initialize Logic
    root_dir = Path(args.out)
    rng = np.random.default_rng(args.seed)

    print(f"Generating datasets in '{root_dir}'...")
    print(
        f"Config: Seed={args.seed}, Samples={args.n_samples}, Range={args.max_val}"
    )

    for n in SIZES_TO_TEST:
        print(f"\n--- Processing Size N={n} ---")

        # 1. Easy Instances
        easy_out_dir = root_dir / "easy" / f"n_{n}"
        easy_gen = generate_easy_instances_bulk(n, max_instances=args.n_samples)
        
        for idx, (v, w, b) in enumerate(easy_gen):
            save_instance(easy_out_dir, f"instance_{idx:03d}", v, w, b)
        print(f"  > Saved {min(n, args.n_samples)} Easy instances to {easy_out_dir}")

        # 2. Hard Instances
        # Note: We limit N <= 50 for Hard because above this numerical precision
        # can become an issue (converting to a C long fails)
        hard_out_dir = root_dir / "hard" / f"n_{n}"
        if n <= 50:
            hard_gen = generate_hard_instances_bulk(n, max_instances=args.n_samples)
            
            for idx, (v, w, b) in enumerate(hard_gen):
                save_instance(hard_out_dir, f"instance_{idx:03d}", v, w, b)
            print(f"  > Saved {min(n, args.n_samples)} Hard instances to {hard_out_dir}")
        else:
            print(f"  > Saved 0 Hard instances to {hard_out_dir}")

        # 3. Random Instances
        random_out_dir = root_dir / "random" / f"n_{n}"
        random_gen = generate_random_instances_bulk(
            n_items=n, 
            rng=rng, 
            num_instances=args.n_samples, 
            value_range=args.max_val
        )
        for idx, (v, w, b) in enumerate(random_gen):
            save_instance(random_out_dir, f"instance_{idx:03d}", v, w, b)
        print(f"  > Saved {args.n_samples} Random instances to {random_out_dir}")

    print("\nGeneration complete.")


if __name__ == "__main__":
    main()
