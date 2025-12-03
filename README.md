# Knapsack Optimization Framework

A modular, benchmarking framework for the 0/1 Knapsack Problem. This repository implements a suite of exact and approximate algorithms, data generation pipelines, and statistical visualization tools to analyze algorithmic scalability and stability.

## 🌟 Key Features

* **Robust Data Generation:**
    * **Easy:** Linear correlation (Weight=1, Value=1) to see best-case performance.
    * **Hard:** Pathological instances (Powers of Two) to see worst-case performance.
    * **Random:** Uncorrelated instances for average-case performance.
* **Solver Suite:**
    * **Exact:** Dynamic Programming.
    * **Heuristic:** Greedy (value/density ratio).
    * **Approximation:** Rounded Dynamic Programming Bicriteria.
    * **Feasible Modification:** Modified Approximation to enforce Feasibility.
* **Analytics:**
    * Confidence Interval analysis for runtime analysis.
    * Min/Max analysis for constraint verification.
    * Approximation/Optimality value analysis.
* **Architecture:** Type-hinted Python 3.9+ with automated regression testing.

## 📂 Project Structure

```
Knapsack-Approximation/
├── data/                   # Generated datasets
│   ├── easy/
│   ├── hard/
│   └── random/
├── notebooks/
│   └── visualization.ipynb # Interactive plotting dashboard
├── results/                # Experiment logs and saved plots
│   ├── easy/
│   ├── hard/
│   ├── random/
│   └── plots/
├── scripts/                # CLI Entry points
│   ├── generate_data.py    # Dataset creation
│   ├── run_experiment.py   # Benchmark execution
│   └── validate_algos.py   # Mathematical correctness tests
├── src/                    # Core Library
│   ├── algorithms.py       # Solver implementations
│   ├── generator.py        # Data generation logic
│   ├── solver.py           # Abstract Base Class & Contracts
│   └── plotting.py         # Visualization library
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies.
```
Bash
git clone https://github.com/jermcmahan/Knapsack-Approximation.git
cd Knapsack-Approximation
pip install -r requirements.txt
```

### 2. Data Generation

Generate the synthetic datasets.

```
Bash
# Generate small validation sets and large random sets (up to N=1000)
python -m scripts.generate_data
```

### 3. Validation (Unit Testing)

Before running experiments, verify the solvers against analytical ground truths to ensure correctness.

```
Bash
python -m scripts.validate_analytical
Expected Output: [SUCCESS] All tests passed.
```

### 4. Run Benchmark

Run the solvers on the generated data. This script uses incremental saving to protect against crashes.

```
Bash
python -m scripts.run_experiment --data_dir data/random --out_dir results/random
```

Alternatively use the provided script "run_all.sh"

### 5. Visualization

Open notebooks/visualization.ipynb to generate the plots. The notebook will automatically find the latest CSV in results/ and produce:

* Runtime Analysis: Log-scale runtime with 95% Confidence Intervals.

* Constraint Analysis: Worst-case weight to budget comparison of each algorithm.

* Value Analysis: Comparison of achieved value to optimal or approximate lower bound.

## 🧠 Algorithms Implemented

| Algorithm                | Type       | Time Complexity   |
| :----------------------- | :--------- | :---------------- |
| Dynamic Programming      | Optimal    | Pseudo-Polynomial |
| Greedy Heuristic         | Heuristic  | Log-Linear        | 
| Rounded DP Approximation | Bicriteria | Polynomial        |
| Feasible Rounding        | Heuristic  | Polynomial        |

## 📊 Reproducibility
To reproduce the exact charts found in the report:

1. Run the full generation pipeline: python -m scripts.generate_data

2. Run the experiment suite: ./run_all.sh

3. Execute the cells in notebooks/visualization.ipynb

## 📜 Citation
If you use this code for your research, please cite:

```
[Your Name]. (2025). Knapsack Optimization Framework. 
GitHub Repository. https://github.com/...
```