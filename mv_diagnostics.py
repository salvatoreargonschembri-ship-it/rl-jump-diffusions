"""
mv_diagnostics.py -- Calibration diagnostics for Appendix B
=============================================================
Produces three figures and one detailed parameter table for
Appendix B of the thesis:

  figure_app_qqplot.pdf/png      -- QQ-plots of MJD fit vs empirical returns
  figure_app_density.pdf/png     -- MJD density fit overlaid on histograms
  figure_app_mjd_params.pdf/png  -- MJD parameter estimates with 95% CI (bootstrap)

The figures document the quality of the MJD calibration on the
five-asset training set (2005-2019) used in Chapter 7.

Usage:
  python mv_diagnostics.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize
from scipy.stats import norm, poisson, probplot
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

mpl.rcParams.update({
    'font.family':    'serif',
    'font.size':      10,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
})

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ['^GSPC', 'AGG', 'VGK', 'GLD', 'QQQ']
NAMES   = ['S&P 500', 'Bonds', 'MSCI Europe', 'Gold', 'NASDAQ-100']
D       = len(TICKERS)
R_F     = 0.02
DT      = 1 / 12
TRAIN_END = '2019-12-31'
TEST_END  = '2023-12-31'


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_training_returns():
    import yfinance as yf
    import pandas as pd

    print("  Downloading data...")
    data = yf.download(TICKERS, start='2000-01-01', end=TEST_END, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        raw = data['Close']
    else:
        raw = data
    raw = raw[list(TICKERS)]          # FIX: yfinance ordina le colonne alfabeticamente;
                                     # riallineale all'ordine di TICKERS/NAMES
    # FIX: trova la prima data comune sui dati GREZZI, taglia, poi ffill.
    # Niente bfill: creava prezzi artificiali prima della quotazione degli ETF
    # (AGG 09/2003, GLD 11/2004, VGK 03/2005) e rendimenti zero fittizi.
    first_valid = raw.apply(lambda col: col.first_valid_index()).max()
    raw = raw.loc[first_valid:]
    raw = raw.ffill()

    log_daily = np.log(raw / raw.shift(1)).dropna()
    dpm = 21
    n_months = len(log_daily) // dpm
    ret_monthly = np.zeros((n_months, D))
    dates_monthly = []
    for i in range(n_months):
        block = log_daily.iloc[i*dpm:(i+1)*dpm]
        ret_monthly[i] = block.sum().values
        import pandas as pd
        dates_monthly.append(pd.Timestamp(block.index[-1]))

    cutoff = pd.Timestamp(TRAIN_END)
    mask_train = np.array([d <= cutoff for d in dates_monthly])
    ret_train = ret_monthly[mask_train]
    print(f"  Training returns: {ret_train.shape[0]} months x {D} assets")
    return ret_train


# ─────────────────────────────────────────────────────────────────────────────
# MJD CALIBRATION (single asset)
# ─────────────────────────────────────────────────────────────────────────────

def fit_mjd(ri):
    """Fit MJD parameters to a series of monthly log-returns ri."""
    def neg_ll(params):
        mu_i, sig_i, lam_i, m_i, delta_i = params
        if sig_i <= 0 or lam_i <= 0 or delta_i <= 0:
            return 1e10
        kappa = np.exp(m_i + 0.5*delta_i**2) - 1
        n_max = max(10, int(lam_i*DT*5 + 20))
        ns  = np.arange(0, n_max+1)
        pn  = poisson.pmf(ns, lam_i*DT)
        pdf = np.zeros(len(ri))
        for n, p_n in zip(ns, pn):
            if p_n < 1e-15:
                continue
            mu_n  = (mu_i - 0.5*sig_i**2 - lam_i*kappa)*DT + n*m_i
            var_n = sig_i**2*DT + n*delta_i**2
            pdf  += p_n * norm.pdf(ri, mu_n, np.sqrt(var_n))
        return -np.sum(np.log(np.maximum(pdf, 1e-300)))

    sig0 = np.std(ri) / np.sqrt(DT)
    mu0  = np.mean(ri) / DT + 0.5*sig0**2
    x0   = [mu0, sig0, 5.0, -0.01, 0.03]
    bds  = [(None,None),(1e-4,None),(1e-4,None),(None,None),(1e-4,None)]
    try:
        res = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bds,
                       options=dict(maxiter=300, ftol=1e-8))  # allineato a mv_multiasset_corr
        return res.x, res.fun
    except Exception:
        return np.array(x0), np.nan


def mjd_pdf(r_grid, params):
    """Evaluate the MJD mixture density on a grid."""
    mu_i, sig_i, lam_i, m_i, delta_i = params
    kappa = np.exp(m_i + 0.5*delta_i**2) - 1
    n_max = max(10, int(lam_i*DT*5 + 20))
    ns  = np.arange(0, n_max+1)
    pn  = poisson.pmf(ns, lam_i*DT)
    pdf = np.zeros(len(r_grid))
    for n, p_n in zip(ns, pn):
        if p_n < 1e-15:
            continue
        mu_n  = (mu_i - 0.5*sig_i**2 - lam_i*kappa)*DT + n*m_i
        var_n = sig_i**2*DT + n*delta_i**2
        pdf  += p_n * norm.pdf(r_grid, mu_n, np.sqrt(var_n))
    return pdf


def mjd_cdf(r_grid, params):
    """Evaluate the MJD CDF on a sorted grid (for QQ-plot)."""
    mu_i, sig_i, lam_i, m_i, delta_i = params
    kappa = np.exp(m_i + 0.5*delta_i**2) - 1
    n_max = max(10, int(lam_i*DT*5 + 20))
    ns  = np.arange(0, n_max+1)
    pn  = poisson.pmf(ns, lam_i*DT)
    cdf = np.zeros(len(r_grid))
    for n, p_n in zip(ns, pn):
        if p_n < 1e-15:
            continue
        mu_n  = (mu_i - 0.5*sig_i**2 - lam_i*kappa)*DT + n*m_i
        std_n = np.sqrt(sig_i**2*DT + n*delta_i**2)
        cdf  += p_n * norm.cdf(r_grid, mu_n, std_n)
    return cdf


def bs_pdf(r_grid, ri):
    """Gaussian (BS) density for comparison."""
    mu_bs  = np.mean(ri)
    std_bs = np.std(ri)
    return norm.pdf(r_grid, mu_bs, std_bs)


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVALS FOR MJD PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(ri, n_boot=200, ci=0.95):
    """Bootstrap 95% CI for MJD parameters via resampling."""
    n = len(ri)
    boot_params = np.zeros((n_boot, 5))
    for b in range(n_boot):
        ri_b = ri[np.random.randint(0, n, n)]
        params_b, _ = fit_mjd(ri_b)
        boot_params[b] = params_b
    alpha = (1 - ci) / 2
    lo = np.quantile(boot_params, alpha,   axis=0)
    hi = np.quantile(boot_params, 1-alpha, axis=0)
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — QQ-plots: MJD model quantiles vs empirical quantiles
# ─────────────────────────────────────────────────────────────────────────────

def figure_qqplot(ret_train, all_params):
    print("\n[Fig 1] QQ-plots...")
    fig, axes = plt.subplots(1, D, figsize=(15, 3.8))

    for i, (ax, name, params) in enumerate(zip(axes, NAMES, all_params)):
        ri   = ret_train[:, i]
        ri_s = np.sort(ri)
        n    = len(ri_s)
        # Empirical quantile levels
        probs = (np.arange(1, n+1) - 0.5) / n

        # MJD theoretical quantiles via numerical inversion of CDF
        r_fine = np.linspace(ri_s.min() - 0.05, ri_s.max() + 0.05, 2000)
        cdf_fine = mjd_cdf(r_fine, params)
        # Interpolate: for each prob, find r such that CDF(r) = prob
        mjd_q = np.interp(probs, cdf_fine, r_fine)

        ax.scatter(mjd_q, ri_s, s=12, color='steelblue', alpha=0.6,
                   label='MJD quantiles', zorder=3)

        # 45-degree reference line
        lo = min(mjd_q.min(), ri_s.min())
        hi = max(mjd_q.max(), ri_s.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='45-degree line')

        # BS quantiles for comparison
        mu_bs  = np.mean(ri)
        std_bs = np.std(ri)
        bs_q   = norm.ppf(probs, mu_bs, std_bs)
        ax.scatter(bs_q, ri_s, s=8, color='darkorange', alpha=0.35,
                   label='Normal quantiles', zorder=2)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Theoretical quantiles', fontsize=9)
        if i == 0:
            ax.set_ylabel('Empirical quantiles', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.suptitle('QQ-plots: MJD model vs empirical monthly returns (2005-2019)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig('figure_app_qqplot.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('figure_app_qqplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figure_app_qqplot.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Density fit: MJD and Normal overlaid on empirical histogram
# ─────────────────────────────────────────────────────────────────────────────

def figure_density(ret_train, all_params):
    print("\n[Fig 2] Density fits...")
    fig, axes = plt.subplots(1, D, figsize=(15, 3.8))

    for i, (ax, name, params) in enumerate(zip(axes, NAMES, all_params)):
        ri = ret_train[:, i]
        ax.hist(ri, bins=28, density=True, color='lightsteelblue',
                edgecolor='white', alpha=0.7, label='Empirical')

        r_grid = np.linspace(ri.min() - 0.03, ri.max() + 0.03, 500)
        ax.plot(r_grid, mjd_pdf(r_grid, params),
                color='steelblue', lw=2.0, label='MJD fit')
        ax.plot(r_grid, bs_pdf(r_grid, ri),
                color='darkorange', lw=1.5, ls='--', label='Normal fit')

        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Monthly log-return', fontsize=9)
        if i == 0:
            ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('MJD and Normal density fits on monthly returns (2005-2019)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig('figure_app_density.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('figure_app_density.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figure_app_density.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — MJD parameter estimates with bootstrap 95% CIs
# ─────────────────────────────────────────────────────────────────────────────

def figure_mjd_params(ret_train, all_params):
    print("\n[Fig 3] MJD parameter estimates with bootstrap CI...")
    print("  (Bootstrap resampling: 200 replicates per asset -- may take ~1 min)")

    param_names = [r'$\mu$', r'$\sigma$', r'$\lambda$', r'$m$', r'$\delta$']
    n_params = 5
    all_lo = np.zeros((D, n_params))
    all_hi = np.zeros((D, n_params))

    for i in range(D):
        print(f"    Bootstrap asset {i+1}/{D}: {NAMES[i]}...")
        lo, hi = bootstrap_ci(ret_train[:, i], n_boot=200)
        all_lo[i] = lo
        all_hi[i] = hi

    point_est = np.array(all_params)   # shape (D, 5)

    fig, axes = plt.subplots(1, n_params, figsize=(14, 4.5))
    colors = ['steelblue', 'darkorange', 'forestgreen', 'firebrick', 'purple']
    x = np.arange(D)

    for j, (ax, pname) in enumerate(zip(axes, param_names)):
        vals = point_est[:, j]
        lo_j = vals - all_lo[:, j]
        hi_j = all_hi[:, j] - vals

        ax.bar(x, vals, color=colors, alpha=0.75, width=0.6)
        ax.errorbar(x, vals, yerr=[lo_j, hi_j],
                    fmt='none', color='black', capsize=4, lw=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(['SP500', 'Bonds', 'MSCI', 'Gold', 'QQQ'],
                           rotation=30, ha='right', fontsize=8)
        ax.set_title(pname, fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        if j == 0:
            ax.set_ylabel('Estimated value', fontsize=9)

    fig.suptitle('MJD parameter estimates with bootstrap 95\\% confidence intervals\n'
                 'Training set 2005-2019',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('figure_app_mjd_params.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('figure_app_mjd_params.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figure_app_mjd_params.pdf/png")

    return point_est, all_lo, all_hi


# ─────────────────────────────────────────────────────────────────────────────
# PRINT DETAILED PARAMETER TABLE (for LaTeX)
# ─────────────────────────────────────────────────────────────────────────────

def print_parameter_table(all_params, all_lo, all_hi):
    print("\n" + "="*80)
    print("MJD PARAMETER TABLE (point estimates and 95% bootstrap CI)")
    print("="*80)
    hdr = f"{'Asset':<18} {'mu':>8} {'sigma':>8} {'lambda':>8} {'m':>8} {'delta':>8}"
    print(hdr)
    print("-"*80)
    for i, name in enumerate(NAMES):
        mu, sig, lam, m, delta = all_params[i]
        print(f"{name:<18} {mu:>8.4f} {sig:>8.4f} {lam:>8.2f} {m:>8.4f} {delta:>8.4f}")
        lo = all_lo[i]
        hi = all_hi[i]
        print(f"{'  [95% CI]':<18} "
              f"[{lo[0]:.3f},{hi[0]:.3f}] "
              f"[{lo[1]:.3f},{hi[1]:.3f}] "
              f"[{lo[2]:.2f},{hi[2]:.2f}] "
              f"[{lo[3]:.4f},{hi[3]:.4f}] "
              f"[{lo[4]:.4f},{hi[4]:.4f}]")
    print("="*80)

    # Also compute sigmaJ^2
    print("\nsigmaJ^2 = lambda*(exp(2m+2delta^2) - 2*exp(m+delta^2/2) + 1):")
    for i, name in enumerate(NAMES):
        mu, sig, lam, m, delta = all_params[i]
        sJ2 = lam*(np.exp(2*m+2*delta**2) - 2*np.exp(m+0.5*delta**2) + 1)
        print(f"  {name:<18}: sigmaJ^2 = {sJ2:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("\n[0] Loading training data (2005-2019)...")
    ret_train = load_training_returns()

    print("\n[1] Fitting MJD model to each asset...")
    all_params = []
    for i in range(D):
        params, ll = fit_mjd(ret_train[:, i])
        all_params.append(params)
        mu, sig, lam, m, delta = params
        sJ2 = lam*(np.exp(2*m+2*delta**2) - 2*np.exp(m+0.5*delta**2) + 1)
        print(f"  {NAMES[i]:<18}: mu={mu:.4f}  sigma={sig:.4f}  "
              f"lam={lam:.2f}  m={m:.4f}  delta={delta:.4f}  "
              f"sigmaJ2={sJ2:.6f}  loglik={-ll:.1f}")

    figure_qqplot(ret_train, all_params)
    figure_density(ret_train, all_params)
    point_est, all_lo, all_hi = figure_mjd_params(ret_train, all_params)
    print_parameter_table(point_est, all_lo, all_hi)

    print("\n" + "="*55)
    print("  All figures produced:")
    print("    figure_app_qqplot.pdf/png")
    print("    figure_app_density.pdf/png")
    print("    figure_app_mjd_params.pdf/png")
    print("="*55)