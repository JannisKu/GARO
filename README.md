# Weber Facility Location Example

The folder Example contains the Julia code and LaTeX source for the facility location (Weber problem) figures in:

> Jannis Kurtz & Bart P.G. van Parys, *Globalized Adversarial Regret Optimization: Making Decisions beyond Classical Predictions* (2026).

It reproduces Figures 1 and 2 of the paper.

## Requirements

### Julia
- Julia 1.9+
- Packages: `JuMP`, `Clarabel`, `Polyhedra`, `CDDLib`, `JLD2`, `CSV`, `DataFrames`, `LinearAlgebra`

### LaTeX
- A standard TeX distribution (TeX Live, MiKTeX)
- Packages: `pgfplots` (>= 1.18), `tikz` with the `spy` library

## Reproducing the figures

### 1. Run the Julia code

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. weber.jl
```

This solves the facility location problem and writes 14 CSV files to `julia/csv/`:

| File | Contents |
|------|----------|
| `instance.csv` | Support points, nominal weights, and true weights |
| `ch-Xi.csv` | Vertices of the convex hull of the support |
| `min-ball-Xi.csv` | Circumcenter and circumradius of the support |
| `circle1.csv` | Circumscribed circle of the support (for plotting) |
| `mean.csv` | Nominal hub location |
| `M_r.csv` | Vertices of the set of attainable means |
| `min-ball-M_r.csv` | Chebyshev center and radius of the attainable means |
| `circle2.csv` | Circumscribed circle of the attainable means (for plotting) |
| `robust_mean.csv` | Wasserstein robust hub location |
| `sat_mean.csv` | Satisficing hub location |
| `glob_robust_mean.csv` | GARO hub location |
| `robust_hubs.csv` | Path of robust hub locations as the radius varies |
| `regret_hubs.csv` | Path of regret-optimal hub locations as the radius varies |
| `costs.csv` | Adversarial regret curves for all hub locations |

### 2. Compile the figures

```bash
cd ..
pdflatex figures.tex
```

This produces `figures.pdf` containing both figures.

## Problem instance

The problem data (support set, nominal distribution, Wasserstein radius, true distribution) is stored in `julia/problem_instance.jld2`.

## What the code computes

`weber.jl` implements four hub-location estimators for a distributionally robust facility location problem with a Wasserstein ambiguity set:

| Estimator | Description |
|-----------|-------------|
| `robust_mean` | Wasserstein robust mean: minimises worst-case expected cost |
| `regret_mean` | Regret-optimal mean: Chebyshev center of the set of attainable means |
| `sat_mean` | Satisficing mean (Long, 2023): minimises the slope of a linear cost guarantee |
| `glob_robust_mean` | GARO mean: minimises adversarial regret uniformly over all perturbation levels |

# Experiment: Knapsack Problem

The folder `Experiment` contains the Python code for the Knapsack experiments in Section 6 of the following paper:

> Jannis Kurtz & Bart P.G. van Parys, *Globalized Adversarial Regret Optimization: Making Decisions beyond Classical Predictions* (2026).

## Problem data

The problem data for each type of distribution can be found in the corresponding subfolders (Gaussian, Gaussian Inverse, Heavy Tail). The files corresponding to the problem instances are:

| File | Contents |
|------|----------|
| `data_train_i.csv` | List of training samples for data instance i |
| `data_test_i.csv` | List of test samples for data instance i |
| `Instances.csv` | Contains information about the knapsack instance in consecutive rows: (ID,Knapsack weights, Knapsack capacity, Sense of the knapsack constraint, Variable lower bounds, variable upper bounds, Flag integer variables, flag maximization problem)  |
| `boxplot_data.pkl` | Results for the out-of-sample performance. The file contains the following items: "label" (name of the method), "values" (list of out-of-sample objective values), "runtimes" (list of all runtimes)  |
| `guarantee_data.pkl` | Contains the values of the guarantee plots. The file contains the following items: "label" (name of the method), "gamma" (gamma values for x-axis), "y" (guarantee values for y-axis), "test_data" (gamma values for the test-samples) |


## Run the Python code

Execute the file RunKnapsack.py. The setup can be adjusted in the first lines:

| Variable | Explanation |
|------|----------|
| `num_points` | Number of samples which are then split in 80/20 training/test samples. |
| `dim` | Dimension of the knapsack problem. |
| `integer_vars` | Integer knapsack problem is used true/false |
| `maximization_prob` | Maximization problem is used true/false (should not be set to `true` without changing the constraint sense for the knapsack problem since minimum knapsack problem is considered) |
| `dataType` | String for data distribution |
| `num_data_instances` | Number of random sample sets executed |
| `num_opt_instances` | Number of random knapsack instances executed |
| setup_RO | List of values $\theta$ for RO|
| setup_RO_d | List of values $\theta$ for RO$_d$ |
| setup_REG | List of values $\theta$ for REG |
| setup_GARO | List of values $q$ for GARO |
| setup_Sat | List of values $\beta$ for SAT |

In the file Methods.py the solution methods for the optimization models can be found:

| Method | Explanation |
|------|----------|
| `minMaxEllipsoid` | RO |
| `runRobustOptimizationScenarios` | RO$_d$ |
| `runRobustSatisficing` | SAT |
| `runClassicalRegret` | REG |
| `runGARO` | GARO |


