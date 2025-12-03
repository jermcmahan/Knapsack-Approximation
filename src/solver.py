"""
Abstract base class for Knapsack Problem solvers.

This module defines the contract that all solver implementations must follow,
ensuring consistent timing and result reporting across experiments.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SolverResult:
    """
    Standardized result object returned by all solvers.

    Attributes:
        algorithm_name: The identifier of the algorithm used.
        solution: Binary vector (0/1) denoting selected items (dtype=int).
        value: The total value of selected items.
        weight: The accumulated weight of selected items.
        budget: The capacity constraint provided.
        runtime: Execution time in seconds.
    """
    algorithm_name: str
    solution: np.ndarray
    value: int
    weight: int
    budget: int
    runtime: float


class KnapsackSolver(ABC):
    """
    Abstract Base Class (ABC) for Knapsack solvers.

    Enforces a standard interface for solving, timing, and verifying results.
    Subclasses must implement `_solve_impl` and the `name` property.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The display name of the algorithm for plots and logs."""
        pass

    def solve(self, values: np.ndarray, weights: np.ndarray, budget: int) -> SolverResult:
        """
        Executes the solver on a specific instance and metrics.

        Args:
            values: Array of item values.
            weights: Array of item weights.
            budget: The maximum weight capacity.

        Returns:
            A SolverResult object containing the solution and performance metrics.

        Raises:
            ValueError: If the knapsack instance is invalid
        """
        # Instance Validation
        if len(values) != len(weights) or not isinstance(budget, (int, float)):
            raise ValueError(
                f"Invalid Knapsack Instance"
            )

        # Timing
        start_time = time.perf_counter()
        
        # Call implementation
        solution, value, weight = self._solve_impl(values, weights, budget)
        
        end_time = time.perf_counter()

        return SolverResult(
            algorithm_name=self.name,
            solution=solution,
            value=value,
            weight=weight,
            budget=budget,
            runtime=end_time - start_time,
        )

    @abstractmethod
    def _solve_impl(
        self, values: np.ndarray, weights: np.ndarray, budget: int
    ) -> Tuple[np.ndarray, int, int]:
        """
        Internal implementation of the solve algorithm.

        Args:
            values: Array of item values.
            weights: Array of item weights.
            budget: The maximum weight capacity.

        Returns:
            Tuple containing (solution_vector, achieved_value, total_weight).
        """
        pass
