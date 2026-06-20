# Hessian-augmented Supervised Learning for Hamilton–Jacobi–Bellman PDEs

Companion code for the paper *"Hessian-augmented Supervised Learning for
Hamilton–Jacobi–Bellman PDEs"* by M. Gómez-Aedo, B. Azmi, Y. Huang, D. Kalise,
and K. Kunisch.

The method approximates the value function of a deterministic optimal-control
problem in two stages: **generation**, which solves the Pontryagin Maximum
Principle from many initial conditions and obtains value, gradient and Hessian
data (the Hessian from a matrix Riccati equation along optimal trajectories);
and **regression**, a derivative-augmented weighted least-squares fit over
sparse-polynomial bases on hyperbolic-cross index sets. Feedback laws follow
analytically from the learned value function.

## Repository structure

```
generation/<experiment>/   produce the training/validation datasets
regression/<experiment>/    fit the value function, report validation errors
```

Each experiment folder is **self-contained** (it carries every function it
needs). The two stages communicate only through the `.mat` datasets that
generation writes into its own `data/` folder.

| Experiment        | Paper § | Dim n | Folder name      |
|-------------------|---------|-------|------------------|
| Analytical (f_AS) | 4.1     | 12    | `analytical_12d` |
| Van der Pol       | 4.2     | 2     | `vanderpol_2d`   |
| Rigid-body sat.   | 4.3     | 6     | `rigidbody_6d`   |
| Allen–Cahn        | 4.4     | 19    | `allencahn_19d`  |

## Requirements

- MATLAB (R2020a or newer).
- Statistics and Machine Learning Toolbox (`sobolset` for the scrambled Sobol
  sampling).
- Parallel Computing Toolbox — used by the 6-D and 19-D generation (`parfor`).
  Optional: without it, the loops run serially.

## How to run

For any experiment, run the generation once, then the regression:

```matlab
% 1) generation/<experiment>/
main_generate      % writes data/train_*.mat and data/val_*.mat

% 2) regression/<experiment>/
main_regress       % prints q, ||theta||, and L2 / H1 / H2 validation errors
```

The generated `.mat` datasets are included in the repository, so the regression
stage can be run directly without rerunning the (slow) PMP solves.

## Parameters (match the paper)

| Experiment     | Key settings                                                   |
|----------------|----------------------------------------------------------------|
| analytical_12d | n=12, f_AS on [-1,1]¹², HC(4) q=3482, λ=1e-8, γ₁=γ₂=1           |
| vanderpol_2d   | T=3, β=0.1, Ω=[-3,3]², HC(4) q=52                              |
| rigidbody_6d   | T=20, β=¼, α₃=½, Ω=[-π/3,π/3]³×[-π/4,π/4]³                     |
| allencahn_19d  | n=19, T=4, ν=0.1, β=0.01, Ω=[-1,1]¹⁹                            |

## Configuration

The `CONFIG` block at the top of each `main_regress.m` chooses:

- `basis_type` — `'HC'` (hyperbolic cross) or `'enriched'` (HC + total-degree).
- `hc_level`, `td_level` — basis levels.
- `ord` — regression order: `0` value only, `1` + gradient, `2` + Hessian.
- `lambda`, `gamma1`, `gamma2` — ridge parameter and gradient/Hessian weights.

The `CONFIG` block of each `main_generate.m` sets the number of training /
validation points and the random seeds.

## Dataset schema

Every `.mat` stores the same variables:

| Variable | Size            | Meaning                                       |
|----------|-----------------|-----------------------------------------------|
| `X`      | N × n           | sample points                                 |
| `V`      | N × 1           | value of the value function                   |
| `G`      | N × n           | gradient                                      |
| `H`      | N × n(n+1)/2    | Hessian, upper triangle [diag; off-diag i<j]  |
| `meta`   | struct          | dimension, domain, seed, problem metadata     |

## Notes

- Sampling uses scrambled Sobol points; training seed = 1, validation seed = 999.
  These differ from the exact datasets behind the published figures, which are
  Monte-Carlo averages, so single-run errors match the paper curves closely but
  not bit-for-bit.
- The Hessian ordering (diagonal first, then off-diagonal i<j) is shared by the
  generation extractors and the regression basis; the two must agree.
