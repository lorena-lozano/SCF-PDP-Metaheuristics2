# Socially Constrained Fairness - Pickup and Delivery Problem (SCF-PDP)
Authors: Lorena Lozano, Tina Taheri

Date: November 2025

This repository contains the implementation of advanced metaheuristics to solve the **Socially Constrained Fairness - Pickup and Delivery Problem (SCF-PDP)**. 

The SCF-PDP is a complex variant of the Vehicle Routing Problem (VRP) that not only considers traditional constraints like vehicle capacity and pickup-dropoff precedence, but also integrates **social fairness measures** to balance the workload, route duration, or service equity among users/vehicles.

## Features & Algorithms Implemented

This project includes multiple approaches to solve the problem, divided into classical optimization and advanced metaheuristics:

### 1. Constructive & Local Search Methods (A1)
- **Constructive algorithms**: Greedy and Randomized Greedy approaches ensuring capacity and precedence constraints.
- **Local Search Operators**: 
  - **N1**: Intra-route relocation.
  - **N2**: Inter-route relocation.
  - **N3**: Request swapping between routes.
- **Trajectory Metaheuristics**: GRASP, Variable Neighborhood Descent (VND), and Tabu Search.

### 2. Advanced Metaheuristics (A2)
- **Evolutionary Algorithm (EA)**: Utilizes Precedence-Preserving Order Crossover (PPOX) and capacity-aware mutations. Includes dynamic penalty mechanisms for infeasible solutions to guide the search toward feasibility.
- **Ant Colony Optimization (ACO)**: Implements pheromone-guided construction with precedence constraints. Features greedy seeding for large-scale instances (up to 10,000 requests) to accelerate convergence.

### 3. Fairness Analysis
Evaluates different fairness indices on the generated routes:
- **Jain's Fairness Index**
- **Max-Min Fairness**
- **Gini Coefficient**

The repository includes robust analytical scripts (`fairness_analysis_aco.py`, `fairness_comparison.py`) that generate comprehensive statistical summaries, convergence graphs, routing structure comparisons, and LaTeX tables.

## Full Report & Results
For a detailed explanation of the mathematical model, the implemented algorithms, parameter tuning, and a comprehensive analysis of the computational results, please refer to the **`SCF-PDP-Metaheuristics2`** document included in this repository.

## Repository Structure

- `SCFDP.py`: Core problem definition, constraints, and validation logic.
- `aco_alg.py` / `evolutionary_alg.py`: Core implementations of the ACO and EA metaheuristics.
- `experiment_runner.py` / `run_aco_simple.py`: Execution scripts for batch processing instances.
- `quick_compare_all_methods_v3.py`: Utility to quickly benchmark A1 vs A2 algorithms.
- `neighborhoods.py` / `local_search.py`: Neighborhood structures and local search framework.
- `a1_constructors_fixed.py` / `GRASP.py` / `VND.py` / `tabu_search.py`: Traditional optimization approaches.
- `fairness_*.py` / `greedy_random.py`: Analytics and visualization suites.

## Requirements

- Python 3.9+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

To install the required dependencies, run:
```bash
pip install pandas numpy matplotlib seaborn scipy
```

## How to Run

1. Run the Evolutionary Algorithm on a single instance:
```bash
python evolutionary_alg.py path/to/instance.txt output_solution.txt
```
2. Run full ACO experiments across all fairness measures:
```bash
python run_aco_simple.py
```
3. Run the fast algorithm comparison benchmark:
```bash
python quick_compare_all_methods_v3.py
```
4. Generate Fairness Analysis & Plots:
```bash
python fairness_analysis_aco.py
```
