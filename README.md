# Weber Facility Location Example

This folder contains the Julia code and LaTeX source for the facility location (Weber problem) figures in:

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
