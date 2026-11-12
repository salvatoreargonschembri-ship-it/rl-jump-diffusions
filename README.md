# Reinforcement Learning for Jump-Diffusions

Code for the MSc thesis **"Reinforcement Learning for Jump-Diffusions"**
Salvatore Schembri, Politecnico di Milano, A.Y. 2025/26.
Advisor: Prof. Daniele Marazzina.

The thesis presents continuous-time q-learning for jump-diffusions (Jia & Zhou, 2023; Gao, Li & Zhou, 2024) and extends the scalar mean-variance application of Gao, Li & Zhou in three directions: proportional transaction costs, CARA utility with transaction costs, and a multi-asset portfolio with correlated jump-diffusion dynamics.

## Contents

### Single asset (S&P 500, Chapter 6)

| Script | Purpose | Thesis output |
|---|---|---|
| `mv_RL.ipynb` | MV q-learning on real S&P 500 data and on BS / MJD simulators, and out-of-sample evaluation 2020-2023; saves `mv_results.npz` | Sections 6.6-6.7 and 6.9, Tables 6.1-6.2 |
| `mv_plot.ipynb` | Figures from `mv_results.npz` | Figures 6.1-6.3 and 6.6-6.7 |
| `mv_analysis.py` (and `.ipynb`) | Efficient frontier and log-log convergence analysis | Figures 6.4-6.5, Section 6.8 |
| `mv_backtest.py` | Stand-alone re-run of the training and of the out-of-sample backtest 2020-2023 (see note below) | Independent check of Section 6.9 |
| `mv_scalar_tc.py` | MV with proportional transaction costs, clipped-quadratic q-function, no-trade zone, four cost levels | Section 6.11, Table 6.3 |

### Multi asset (d = 5, Chapters 6-7 and Appendix B)

| Script | Purpose | Thesis output |
|---|---|---|
| `mv_multiasset_corr.py` | EXP2 (diagonal Σ_J), EXP1 (full Σ_J via Threshold Estimator), EXP3 (transaction costs); saves `mv_multiasset_results.npz` | Tables 7.2-7.3, Figures 7.1-7.4, Section 8.2 |
| `mv_extra_figures.py` | Correlation heatmap, Tikhonov sensitivity, OOS allocations, asset-level comparison (reads `mv_multiasset_results.npz`) | Figures 7.5-7.8 |
| `mv_multiasset_cara.py` | Multi-asset CARA q-learning (CARA-EXP1 / CARA-EXP2) | Section 6.12.3, Table 6.4, Figures 6.8-6.9 |
| `mv_cara_tc.py` | Multi-asset CARA with proportional transaction costs, four cost levels | Section 6.13, Table 6.5, Figures 6.10-6.11 |
| `mv_diagnostics.py` | MJD calibration diagnostics: QQ-plots, density fits, bootstrap confidence intervals | Appendix B.8, Table B.2, Figures B.1-B.3 |

`mv_results.npz` stores the trained single-asset parameters, so the Chapter 6 figure scripts can be re-run without repeating the training. The `figures/` folder contains the figures used in the thesis.

**Note on `mv_backtest.py`.** The single-asset numbers reported in the thesis (learned φ₁ = 1.224, Table 6.2 and Figures 6.6-6.7) are produced by `mv_RL.ipynb` and `mv_plot.ipynb`. `mv_backtest.py` repeats the same training with the same hyperparameters but consumes the random stream in a different order, so it converges to a nearby but not identical value (φ₁ ≈ 1.200, terminal wealth ≈ 1.207 for the learned policy). It also writes `convergenza_RL` and `rl_out_of_sample` to the working directory; the versions in `figures/` are those of the thesis.

**Terminology.** The reference strategy that keeps a constant dollar amount in the risky asset(s) is called *fixed-allocation benchmark* (label `Fixed allocation` in the figures). It is not a buy-and-hold strategy in the strict sense, which would keep the number of shares constant.

## Data

Prices are downloaded from Yahoo Finance at run time. Monthly log-returns are obtained by summing daily log-returns over windows of 21 trading days; the training set ends in December 2019 and the test set covers 2020-2023.

- **Single asset** (`^GSPC`): 2000-2023, 239 training months and 48 test months (the transaction-cost script `mv_scalar_tc.py` uses its own month-window alignment, giving 240 and 47).
- **Multi asset** (`^GSPC`, `AGG`, `VGK`, `GLD`, `QQQ`): the ETFs have staggered inception dates (AGG September 2003, GLD November 2004, VGK March 2005). Every multi-asset loader first trims the price matrix to the first date on which all five series are traded (10 March 2005) and only then forward fills short gaps; no backward filling is applied, so no artificial prices are created before a listing. This gives 177 training months (2005-2019) and 48 test months.
- `yfinance` returns the price columns in alphabetical ticker order; every loader explicitly realigns them to the declared `TICKERS` order before any computation.

## Implementation notes

- **Model-free updates.** In EXP1/EXP2 and CARA-EXP1/EXP2 the agent is trained with the same bootstrap simulator on the same returns; Σ_J enters only the analytical benchmarks. The two experiments therefore compare alternative benchmarks, not alternative learned policies.
- **Threshold Estimator.** The estimated Σ_J is the sample second-moment matrix of the thresholded daily returns and is positive semidefinite by construction. The diffusive covariance Σ = Σ_tot − Σ_J is obtained by subtraction and is projected onto the positive semidefinite cone.
- **MJD calibration.** Each asset is calibrated by maximum likelihood (L-BFGS-B, `maxiter=300`, `ftol=1e-8`, identical settings in all scripts). No ad hoc replacement of the estimated diffusive volatility is applied.
- **Transaction costs.** The out-of-sample rule applies the learned no-trade zone: the position is changed only when the target allocation differs from the previous one by more than φ₄; turnover uses the same rule. Maximum drawdown is computed with respect to the running maximum of the wealth path.
- Training runs use N = 20,000 q-learning iterations with fixed random seeds. Runtimes range from a few minutes to about an hour per script depending on hardware.

## Requirements

```
pip install -r requirements.txt
```

Python ≥ 3.9 with `numpy`, `scipy`, `pandas`, `matplotlib`, `yfinance`.

## How to run

1. `mv_RL.ipynb`, then `mv_plot.ipynb`, `mv_analysis.py`, `mv_scalar_tc.py` (Chapter 6, single asset). `mv_backtest.py` is optional: run it in a separate folder if you do not want it to overwrite `convergenza_RL` and `rl_out_of_sample`.
2. `mv_multiasset_corr.py`: produces `mv_multiasset_results.npz`, required by the next step.
3. `mv_extra_figures.py`.
4. `mv_multiasset_cara.py`, `mv_cara_tc.py`, `mv_diagnostics.py`, in any order.

Results may differ marginally from the thesis tables if the data provider revises historical prices.
