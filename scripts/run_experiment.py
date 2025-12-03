"""
Execution script for benchmarking Knapsack solvers.

This script:
1. Scans the data directory for instance folders.
2. Runs algorithms incrementally.
3. SAVES results to CSV after every folder (N) to prevent data loss on crash.

Usage:
    python -m scripts.run_experiment --data_dir data/random --max_n 100
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path to allow importing 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms import (
    DPSolver,
    FRDPSolver,
    GreedySolver,
    RDPSolver,
)
from src.solver import KnapsackSolver, SolverResult


def setup_solvers() -> List[KnapsackSolver]:
    """Instantiates the list of solvers to participate in the benchmark."""
    return [
        # Optimal but slow
        DPSolver(),
        # Standard Rounding (May violate budget)
        RDPSolver(epsilon=1.0),
        RDPSolver(epsilon=0.5),
        RDPSolver(epsilon=0.25),
        # Feasible Rounding (Guaranteed feasibility)
        FRDPSolver(epsilon=1.0),
        FRDPSolver(epsilon=0.5),
        FRDPSolver(epsilon=0.25),
        # No formal guarantees, but fast
        GreedySolver(),
    ]


def run_solvers_on_instance(
    solvers: List[KnapsackSolver],
    values: np.ndarray,
    weights: np.ndarray,
    budget: int,
    n_items: int,
    dataset_name: str
) -> List[Dict[str, Any]]:
    """Runs all applicable solvers on a single problem instance."""
    results = []
    solver_outputs: Dict[str, SolverResult] = {}

    # 1. Execution Pass
    for solver in solvers:
        # Note: DP usually works fine unless budget is massive (Hard instances)
        if n_items >= 30 and solver.name == "Dynamic Programming" and dataset_name == 'hard':
            # Can be tuned based on user's RAM
            continue
        
        try:
            res = solver.solve(values, weights, budget)
            solver_outputs[solver.name] = res
                
        except MemoryError:
            # Expected for DP on Hard instances with huge budgets
            pass
        except Exception as e:
            print(f"[Warn] {solver.name} failed on N={n_items}: {e}")

    # 2. Recording Pass
    for name, res in solver_outputs.items():
        results.append({
            "algorithm": name,
            "time": res.runtime,
            "value": res.value,
            "weight": res.weight,
            "budget": res.budget,
            # We don't save the full solution vector to CSV to save space
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Knapsack Benchmark")
    parser.add_argument("--data_dir", type=str, default="data/random", 
                        help="Directory containing n_X folders")
    parser.add_argument("--out_dir", type=str, default="results", 
                        help="Where to save CSV logs")
    parser.add_argument("--max_n", type=int, default=1000, 
                        help="Skip folders with N > this value")
    args = parser.parse_args()

    # 1. Configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = data_path.name
    csv_path = out_dir / f"benchmark_{dataset_name}_{timestamp}.csv"

    solvers = setup_solvers()
    
    if not data_path.exists():
        print(f"Error: Data directory '{data_path}' not found.")
        return

    # Sort folders by integer N
    folders = sorted(
        data_path.glob("n_*"), 
        key=lambda x: int(x.name.split("_")[1])
    )
    
    print(f"--- Starting Experiment: {timestamp} ---")
    print(f"Dataset: {data_path}")
    print(f"Output: {csv_path}")

    # 2. Main Loop
    for folder in folders:
        try:
            n = int(folder.name.split('_')[1])
        except (IndexError, ValueError):
            continue 

        if n > args.max_n:
            continue
        
        # Local buffer for THIS folder only
        folder_records = []
        files = sorted(list(folder.glob("*.npz")))
        
        for file in tqdm(files, desc=f"Processing N={n}"):
            # Load
            data = np.load(file, allow_pickle=True)
            v = data['values']
            w = data['weights']
            
            # FIX: Force cast to Python int to avoid CSV '0' bug
            b = int(data['budget']) 
            
            # Run
            instance_results = run_solvers_on_instance(solvers, v, w, b, n, dataset_name)
            
            # Augment
            for res in instance_results:
                res["timestamp"] = timestamp
                res["dataset_path"] = str(data_path)
                res["n_items"] = n
                res["instance_id"] = file.stem
                folder_records.append(res)

        # 3. INCREMENTAL SAVE
        if folder_records:
            df_chunk = pd.DataFrame(folder_records)
            
            # If file doesn't exist, write header. If it does, append (no header).
            write_header = not csv_path.exists()
            df_chunk.to_csv(csv_path, mode='a', header=write_header, index=False)
            
            # Optional: Flush memory
            del df_chunk, folder_records

    print(f"\nExperiment Complete.")

if __name__ == "__main__":
    main()
