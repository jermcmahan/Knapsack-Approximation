"""
Data generation logic for the Knapsack Problem.

This module handles the creation of synthetic datasets, including:
- Easy instances (Linear correlation)
- Hard instances (Powers of two / Subset Sum)
- Random instances (Uniform distribution)
"""

from pathlib import Path
from typing import Generator, Tuple

import numpy as np

# Default configuration constants
DEFAULT_NUM_RANDOM = 50
DEFAULT_VALUE_RANGE = 1000


def _subsample_budgets(all_budgets: np.ndarray, max_instances: int) -> np.ndarray:
    """Helper to subsample budgets if there are too many."""
    if len(all_budgets) > max_instances:
        # Pick 'max_instances' indices evenly spaced
        indices = np.linspace(0, len(all_budgets) - 1, max_instances, dtype=int)
        return all_budgets[indices]
    return all_budgets


def generate_easy_instances_bulk(
    n_items: int,
    max_instances: int = DEFAULT_NUM_RANDOM
) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """
    Generates 'Easy' instances (Parameter Sweep).

    Logic:
        Values  = [1, 1, ... 1]
        Weights = [1, 1, ... 1]
        Budget sweeps from 1 to N.

    Args:
        n_items: The number of items in the problem.
        max_instances: Maximum number of instances to yield (subsampling budgets).

    Yields:
        Tuple of (values, weights, budget).
    """
    # Use int32 for standard arithmetic
    values = np.ones(n_items, dtype=np.int32)
    weights = np.ones(n_items, dtype=np.int32)
    
    # Range of all possible budgets [1, 2, ... N]
    all_budgets = 1 + np.arange(n_items, dtype=np.int32)
    selected_budgets = _subsample_budgets(all_budgets, max_instances)

    for b in selected_budgets:
        yield values, weights, b


def generate_hard_instances_bulk(
    n_items: int,
    max_instances: int = DEFAULT_NUM_RANDOM
) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """
    Generates 'Hard' instances using Powers of Two.

    Logic:
        Values  = [1, 2, 4, ... 2^(N-1)]
        Weights = [1, 2, 4, ... 2^(N-1)]
        Budget sweeps through the powers of two.

    Args:
        n_items: Number of items.
        max_instances: Maximum number of instances to yield.

    Yields:
        Tuple of (values, weights, budget).
    """
    # CRITICAL: Use dtype=object to handle integers > 2^63 (N > 63)
    exponents = np.arange(n_items, dtype=object)
    powers_of_two = 2**exponents

    values = powers_of_two
    weights = powers_of_two
    
    # All possible budgets
    selected_budgets = _subsample_budgets(powers_of_two, max_instances)
    
    for b in selected_budgets:
        yield values, weights, b


def generate_random_instances_bulk(
    n_items: int,
    rng: np.random.Generator,
    num_instances: int = DEFAULT_NUM_RANDOM,
    value_range: int = DEFAULT_VALUE_RANGE,
) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """
    Generates uncorrelated random instances.

    Logic:
        Values  ~ Uniform(1, value_range)
        Weights ~ Uniform(1, value_range)
        Budget  = 50% of total weight.

    Args:
        n_items: Number of items.
        rng: A seeded numpy random generator.
        num_instances: How many distinct instances to generate.
        value_range: The maximum integer value for weights/values.

    Yields:
        Tuple of (values, weights, budget).
    """
    for _ in range(num_instances):
        values = rng.integers(1, value_range, size=n_items)
        weights = rng.integers(1, value_range, size=n_items)

        # Standard Capacity: 50% of total weight
        total_weight = np.sum(weights)
        budget = int(total_weight * 0.5)

        yield values, weights, budget


def save_instance(
    folder_path: Path, filename: str, values: np.ndarray, weights: np.ndarray, budget: int
) -> None:
    """
    Helper to save an instance to disk in compressed numpy format.

    Args:
        folder_path: The directory Path object.
        filename: The filename (without extension).
        values: Values array.
        weights: Weights array.
        budget: Budget integer.
    """
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path = folder_path / f"{filename}.npz"
    np.savez_compressed(file_path, values=values, weights=weights, budget=budget)
