"""
Analytical Validation Script (Integration Test).

This script:
1. Scans the 'data/easy' and 'data/hard' directories.
2. Verifies instance structure (weights/values).
3. Runs Exact Solvers to ensure they match Analytical Truth.
4. Runs Approximate Solvers to ensure they satisfy epsilon bounds.

Usage:
    python -m scripts.validate_algos

WARNING:
    Should only validate small n instances to avoid huge runtimes
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import KnapsackSolver, SolverResult
from src.algorithms import DPSolver, RDPSolver, FRDPSolver

# --- 1. Instance Structure Checks ---

def check_easy_instance(values: np.ndarray, weights: np.ndarray, budget: int) -> bool:
    """Verifies that weights and values are all 1s."""
    return (
        np.all(values == 1) 
        and np.all(weights == 1) 
        and (1 <= budget <= len(values))
    )

def check_hard_instance(values: np.ndarray, weights: np.ndarray, budget: int) -> bool:
    """
    Verifies that values and weights are powers of two.
    """
    expected = 2 ** np.arange(len(values), dtype=object)
    valid_values = np.array_equal(values, expected)
    valid_weights = np.array_equal(weights, expected)
    
    # Check budget is a power of two
    valid_budget = (budget & (budget - 1)) == 0

    return valid_values and valid_weights and valid_budget

def check_instance(category: str, values: np.ndarray, weights: np.ndarray, budget: int) -> bool:
    """Dispatcher for instance structural checks."""
    if category == "easy":
        return check_easy_instance(values, weights, budget)
    elif category == "hard":
        return check_hard_instance(values, weights, budget)
    else:
        raise ValueError(f"Invalid Category: {category}")

# --- 2. Exact Solver Checks ---

def check_solve(result: SolverResult, budget: int) -> bool:
    """
    General logic for exact solver solutions.
    Optimal Value == Budget, Weight == Budget for both easy and hard instances
    """
    return (result.value == budget) and (result.weight == budget)

# --- 3. Approximate Solver Checks ---

def check_approx(result: SolverResult, budget: int, epsilon: float) -> bool:
    """
    General logic for approximation checks.
    Condition: 
    1. Value >= Optimal
    2. Weight <= Budget * (1 + epsilon)
    """
    return result.value >= budget and result.weight <= budget * (1.0 + epsilon)

def check_feasible(result: SolverResult, budget: int, epsilon: float) -> bool:
    """Check if solution is feasible for modified approximations"""
    return result.weight <= budget

# --- 4. Main Validation Loop ---

def _get_instance_files(base_dir: Path, n_max: int) -> List[Path]:
    """Helper to find valid files."""
    valid_files = []
    if not base_dir.exists(): return []
    
    for folder in sorted(base_dir.glob("n_*")):
        try:
            if int(folder.name.split("_")[1]) <= n_max:
                valid_files.extend(sorted(folder.glob("*.npz")))
        except ValueError: continue
    return valid_files

def validate(category: str, data_dir: Path, n_max: int = 20) -> List[Dict[str, Any]]:
    cat_dir = data_dir / category
    files = _get_instance_files(cat_dir, n_max)
    
    if not files:
        print(f"[Warn] No '{category}' files found (checked up to N={n_max})")
        return []

    results = []

    # Epsilons for approximation
    eps1 = 1.0
    eps2 = 0.5
    
    # Initialize Solvers
    dp_solver = DPSolver()
    rdp_solver1 = RDPSolver(epsilon=eps1) 
    rdp_solver2 = RDPSolver(epsilon=eps2)
    frdp_solver1 = FRDPSolver(epsilon=eps1)
    frdp_solver2 = FRDPSolver(epsilon=eps2)
    
    print(f"--- Validating {category.title()} Instances (Max N={n_max}) ---")

    for file in tqdm(files, desc=category):
        record = {
            "category": category,
            "filename": file.name,
            "n_items": 0,
            "status": "PASS",
            "error_msg": ""
        }
        
        try:
            data = np.load(file, allow_pickle=True)
            v = data['values']
            w = data['weights']
            b = int(data['budget'])
            record["n_items"] = len(v)

            # 1. Instance Check
            if not check_instance(category, v, w, b):
                raise ValueError("Instance Structure Corrupted")

            # 2. Exact Solver Check
            if not check_solve(dp_solver.solve(v, w, b),b):
                raise ValueError("Exact Solver Failed")

            # 3. Approx Solver Checks
            # Note: Approximations usually don't need 'category' logic, just bounds
            if not check_approx(rdp_solver1.solve(v, w, b), b, eps1):
                raise ValueError(f"Approx Solver Failed (eps={eps1})")
                
            if not check_approx(rdp_solver2.solve(v, w, b), b, eps2):
                raise ValueError(f"Approx Solver Failed (eps={eps2})")

            if not check_feasible(frdp_solver1.solve(v,w,b),b,eps1):
                raise ValueError(f"Feasible Solver Failed (eps={eps1})")

            if not check_feasible(frdp_solver2.solve(v,w,b),b,eps2):
                raise ValueError(f"Feasible Solver FAiled (eps={eps2})")
        
        except Exception as e:
            record["status"] = "FAIL"
            record["error_msg"] = str(e)

        results.append(record)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run Analytical Validation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--max_n", type=int, default=20, 
                        help="Only validate instances with N <= this value")
    parser.add_argument("--category", type=str, choices=["all", "easy", "hard"], 
                        default="all", help="Which category to validate")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    if args.category in ["all", "easy"]:
        all_results.extend(validate("easy", data_path, args.max_n))
        
    if args.category in ["all", "hard"]:
        all_results.extend(validate("hard", data_path, args.max_n))
        
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = out_path / "validation_report.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n--- Validation Summary ---")
        print(df.groupby(["category", "status"]).size())
        
        failures = df[df["status"] == "FAIL"]
        if not failures.empty:
            print(f"\n[ALERT] {len(failures)} failures detected. See {csv_path}")
            print(failures[["filename", "error_msg"]].head())
        else:
            print(f"\n[SUCCESS] All {len(df)} tests passed.")
    else:
        print("No tests run.")

if __name__ == "__main__":
    main()


