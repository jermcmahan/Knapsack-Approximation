"""
Implementation of Knapsack Optimization algorithms.

Includes:
- Dynamic Programming (Exact, Pseudo-Polynomial Time)
- Rounding Approximation (Bicriteria, Polynomial Time)
- Feasible Approximation (Heuristic, Polynomial Time)
- Greedy Heuristic (Heuristic, Log-Linear Time)
"""

from typing import Tuple

import numpy as np

from .solver import KnapsackSolver


class DPSolver(KnapsackSolver):
    """
    Standard Dynamic Programming implementation (O(N*W)).
    Uses vectorized NumPy operations for speed.
    WARNING: Will raise MemoryError on large budgets.
    """

    @property
    def name(self) -> str:
        return "Dynamic Programming"

    def _solve_impl(
        self, values: np.ndarray, weights: np.ndarray, budget: int
    ) -> Tuple[np.ndarray, int, int]:
        n = len(weights)

        # 1. Initialize Table
        # Use int64 to prevent overflow/precision issues with large integers
        try:
            table = np.zeros((n + 1, budget + 1), dtype=np.int64)
        except MemoryError as e:
            raise MemoryError(
                f"DP Table too large! Budget {budget} is too high for RAM."
            ) from e

        # 2. Fill Table (Vectorized)
        # We iterate backwards from N-1 down to 0
        for i in range(n - 1, -1, -1):
            w = weights[i]
            v = values[i]

            # Copy "Exclude" strategy from row below
            table[i, :] = table[i + 1, :]

            # Compute "Include" strategy where valid
            # Valid indices: w_idx + w <= budget
            if w <= budget:
                # table[i, j] = max(exclude, value + table[i+1, j+w])
                
                # Slices for valid capacities
                valid_range = budget - w + 1
                
                exclude_vals = table[i, :valid_range]
                include_vals = table[i + 1, w:] + v
                
                table[i, :valid_range] = np.maximum(exclude_vals, include_vals)

        # 3. Extract Optimal Value
        max_value = table[0, 0]
        solution = np.zeros(n, dtype=int)
        used_weight = 0

        # 4. Reconstruct Solution
        for i in range(n):
            if table[i, used_weight] != table[i + 1, used_weight]:
                solution[i] = 1
                used_weight += weights[i]

        return solution, max_value, used_weight


class RDPSolver(DPSolver):
    """
    Approximation that scales down weights to reduce DP table size.
    
    Logic:
        1. base = (epsilon * budget) / n
        2. w_new = floor(w_old / base)
        3. Solve DP on w_new
    
    Theoretical Guarantee:
        Weight <= Budget * (1 + epsilon)
        Value >= Optimal Value (of original problem)
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        return f"Approximation ($\\epsilon$={self.epsilon})"

    def _solve_impl(
        self, values: np.ndarray, weights: np.ndarray, budget: int
    ) -> Tuple[np.ndarray, int, int]:
        n = len(values)
        
        # Avoid division by zero
        if budget == 0 or self.epsilon == 0:
            return super()._solve_impl(values, weights, budget)

        # 1. Scale Weights
        # base factor k such that error is bounded
        base = (self.epsilon * budget) / n
        
        # Use floor to allow "fitting more" (violation allowed)
        scaled_weights = np.floor(weights / base).astype(np.int64)
        scaled_budget = int(np.floor(budget / base))

        # 2. Delegate to Parent (Vectorized DP)
        solution, value, _ = super()._solve_impl(values, scaled_weights, scaled_budget)

        # 3. Recalculate real weight using original array
        actual_weight = int(solution @ weights)

        return solution, value, actual_weight


class FRDPSolver(RDPSolver):
    """
    Feasible Rounded Approximation.
    Reduces the budget constraints to force the approximate solution 
    to respect the original budget.
    """

    @property
    def name(self) -> str:
        return f"Feasible Approx ($\\epsilon$={self.epsilon})"

    def _solve_impl(
        self, values: np.ndarray, weights: np.ndarray, budget: int
    ) -> Tuple[np.ndarray, int, int]:
        # Constrict budget to absorb the approximation error
        reduced_budget = int(budget / (1 + self.epsilon))
        
        return super()._solve_impl(values, weights, reduced_budget)


class GreedySolver(KnapsackSolver):
    """
    Heuristic solver based on Value/Weight ratio sorting.
    Fast (O(N log N)) but guarantees no optimality.
    """

    @property
    def name(self) -> str:
        return "Greedy Heuristic"

    def _solve_impl(
        self, values: np.ndarray, weights: np.ndarray, budget: int
    ) -> Tuple[np.ndarray, int, int]:
        n = len(values)
        
        # 1. Calculate Ratios
        # Add epsilon to weight to avoid division by zero
        ratios = values.astype(float) / (weights.astype(float) + 1e-9)

        # 2. Sort Descending
        sorted_indices = np.argsort(ratios)[::-1]

        current_val = 0
        current_weight = 0
        solution = np.zeros(n, dtype=int)

        # 3. Pick items
        for i in sorted_indices:
            v = values[i]
            w = weights[i]

            if current_weight + w <= budget:
                current_weight += w
                current_val += v
                solution[i] = 1

        return solution, current_val, current_weight
