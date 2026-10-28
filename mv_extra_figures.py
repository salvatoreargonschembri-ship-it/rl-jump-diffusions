"""
mv_extra_figures.py — Additional figures for Chapter 7
==========================================================
Produce quattro figure supplementari per approfondire l'analisi
del Capitolo 7 della tesi (multi-asset mean-variance).

PREREQUISITE: run mv_multiasset_corr.py before this script.
The learned phi1 is read from mv_multiasset_results.npz (required).
The MJD calibration is identical to the one in mv_multiasset_corr.py
(per-asset MLE, L-BFGS-B, maxiter=300, ftol=1e-8); no ad hoc
replacement of the estimated diffusive volatility is applied.

Figures produced:
  figure_ma_corr.pdf/png              — empirical correlation heatmap
  figure_ma_tikhonov.pdf/png          — sensitivity of phi_1* to eta
  figure_ma_alloc_oos.pdf/png         — monthly OOS allocations
  figure_ma_benchmark_compare.pdf/png — BS vs MJD vs learned phi_1

Usage:
  python mv_extra_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from scipy.optimize import minimize
from scipy.stats import norm, poisson
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

mpl.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARAMETRI  (identici a mv_multiasset_corr.py)
# ─────────────────────────────────────────────────────────────────────────────

TICKERS      = ['^GSPC', 'AGG', 'VGK', 'GLD', 'QQQ']
NAMES        = ['S&P 500', 'Bonds', 'MSCI Europe', 'Gold', 'NASDAQ-100']
NAMES_SHORT  = ['SP500', 'Bonds', 'MSCI', 'Gold', 'QQQ']
D            = len(TICKERS)
R_F          = 0.02
DT           = 1 / 12
T            = 1.0
K            = 12
Z            = 1.4
X0           = 1.0
TRAIN_END    = '2019-12-31'
TEST_END     = '2023-12-31'
ETA_DEFAULT  = 0.10          # same as mv_multiasset_corr.py (eq. 7.15)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD LEARNED phi1 FROM mv_multiasset_results.npz (if available)
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_FILE = 'mv_multiasset_results.npz'

def load_learned_phi1():
    """
    Reads the learned phi1 saved by mv_multiasset_corr.py in
    mv_multiasset_results.npz. The file is required: run
    mv_multiasset_corr.py first.
    """
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(
            f"{RESULTS_FILE} not found. Run mv_multiasset_corr.py first.")
    data = np.load(RESULTS_FILE)
    phi1 = data['phi1_learned']
    print(f"  Loaded phi1_learned from {RESULTS_FILE}: {np.round(phi1, 3)}")
    return phi1


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CARICAMENTO DATI
# ─────────────────────────────────────────────────────────────────────────────

def load_multiasset():
    import yfinance as yf
    import pandas as pd
    print("  Downloading data...")
    data = yf.download(TICKERS, start='2000-01-01', end=TEST_END, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        raw = data['Close']
    else:
        raw = data
    raw = raw[list(TICKERS)]          # FIX: yfinance restituisce le colonne in ordine
                                      # alfabetico (AGG, GLD, QQQ, VGK, ^GSPC); qui le
                                      # riallinea all'ordine di TICKERS/NAMES
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
        dates_monthly.append(pd.Timestamp(block.index[-1]))
    dates_monthly = np.array(dates_monthly)
    cutoff = pd.Timestamp(TRAIN_END)
    mask_train = np.array([d <= cutoff for d in dates_monthly])
    mask_test  = ~mask_train
    return (ret_monthly[mask_train], ret_monthly[mask_test],
            dates_monthly[mask_train], dates_monthly[mask_test])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CALIBRAZIONE
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_bs(ret_monthly):
    mu    = ret_monthly.mean(axis=0)/DT + 0.5*(ret_monthly.std(axis=0)/np.sqrt(DT))**2
    Sigma = np.cov(ret_monthly.T)/DT
    rho   = mu - R_F
    return dict(mu=mu, Sigma=Sigma, rho=rho, SigmaJ=np.zeros((D,D)))


def calibrate_mjd(ret_monthly):
    bs = calibrate_bs(ret_monthly)
    mu_vec = bs['mu'].copy(); sigmaJ_sq = np.zeros(D); sigma_diff = np.zeros(D)
    for i in range(D):
        ri = ret_monthly[:,i]
        def neg_ll(params):
            mu_i,sig_i,lam_i,m_i,delta_i = params
            if sig_i<=0 or lam_i<=0 or delta_i<=0: return 1e10
            kappa = np.exp(m_i+0.5*delta_i**2)-1
            n_max = max(10,int(lam_i*DT*5+20))
            pn = poisson.pmf(np.arange(n_max+1), lam_i*DT)
            pdf = np.zeros(len(ri))
            for n,p_n in zip(range(n_max+1),pn):
                if p_n<1e-15: continue
                mn = (mu_i-0.5*sig_i**2-lam_i*kappa)*DT+n*m_i
                vn = sig_i**2*DT+n*delta_i**2
                pdf += p_n*norm.pdf(ri,mn,np.sqrt(vn))
            return -np.sum(np.log(np.maximum(pdf,1e-300)))
        sig0 = np.std(ri)/np.sqrt(DT)
        try:
            res = minimize(neg_ll, [mu_vec[i],sig0,5.,-0.01,0.03],
                           method='L-BFGS-B',
                           bounds=[(None,None),(1e-4,None),(1e-4,None),(None,None),(1e-4,None)],
                           options=dict(maxiter=300, ftol=1e-8))
            mu_i,sig_i,lam_i,m_i,delta_i = res.x
        except Exception:
            mu_i,sig_i,lam_i,m_i,delta_i = mu_vec[i],sig0,5.,-0.01,0.03
        sigma_diff[i] = sig_i
        sigmaJ_sq[i]  = lam_i*(np.exp(2*m_i+2*delta_i**2)
                                -2*np.exp(m_i+0.5*delta_i**2)+1)
        mu_vec[i] = mu_i
    corr  = np.corrcoef(ret_monthly.T)
    Sigma = np.outer(sigma_diff,sigma_diff)*corr
    Sigma = 0.5*(Sigma+Sigma.T)
    ev,evec = np.linalg.eigh(Sigma)
    Sigma = evec@np.diag(np.maximum(ev,1e-8))@evec.T
    return dict(mu=mu_vec, Sigma=Sigma, rho=mu_vec-R_F,
                SigmaJ=np.diag(sigmaJ_sq), sigma_diff=sigma_diff)


def phi1_star(params, eta=ETA_DEFAULT):
    """
    phi1* = (Sigma_eff + eps*I)^{-1} rho
    Tikhonov: eps = eta/D * trace(Sigma_eff)  (eq. 7.15 della tesi)
    """
    Sigma_eff = params['Sigma'] + params['SigmaJ']
    eps = eta / D * np.trace(Sigma_eff)   # corrected formula
    return np.linalg.solve(Sigma_eff + eps*np.eye(D), params['rho'])


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIGURA 1 — Heatmap correlazione empirica
# ─────────────────────────────────────────────────────────────────────────────

def figure_correlation(ret_train):
    print("\n[Fig 1] Correlation heatmap...")
    corr = np.corrcoef(ret_train.T)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='Correlation coefficient')
    ax.set_xticks(range(D)); ax.set_yticks(range(D))
    ax.set_xticklabels(NAMES, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(NAMES, fontsize=10)
    for i in range(D):
        for j in range(D):
            c = corr[i,j]
            ax.text(j, i, f'{c:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if abs(c)>0.5 else 'black',
                    fontweight='bold')
    ax.set_title('Empirical correlation matrix\nmonthly returns 2005-2019',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('figure_ma_corr.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('figure_ma_corr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  SP500-NASDAQ: {corr[0,4]:.3f}")
    print(f"  SP500-Bonds:  {corr[0,1]:.3f}")
    print(f"  SP500-Gold:   {corr[0,3]:.3f}")
    print("  Saved: figure_ma_corr.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIGURA 2 — Sensitività Tikhonov
# ─────────────────────────────────────────────────────────────────────────────

def figure_tikhonov_sensitivity(bs_params, mjd_params):
    print("\n[Fig 2] Tikhonov sensitivity...")
    eta_vals = np.logspace(-3, 0, 50)
    phi1_bs_grid  = np.array([phi1_star(bs_params,  eta=e) for e in eta_vals])
    phi1_mjd_grid = np.array([phi1_star(mjd_params, eta=e) for e in eta_vals])
    phi1_bs_def   = phi1_star(bs_params)
    phi1_mjd_def  = phi1_star(mjd_params)

    colors = ['steelblue','darkorange','forestgreen','firebrick','purple']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for i in range(D):
        axes[0].semilogx(eta_vals, phi1_bs_grid[:,i],
                         color=colors[i], lw=2, label=NAMES[i])
        axes[1].semilogx(eta_vals, phi1_mjd_grid[:,i],
                         color=colors[i], lw=2, label=NAMES[i])
    for ax, phi1_def, title in zip(
            axes, [phi1_bs_def, phi1_mjd_def], ['BS model', 'MJD model']):
        ax.axvline(ETA_DEFAULT, color='black', lw=1.5, ls='--',
                   label=f'$\\eta={ETA_DEFAULT}$ (default)')
        for i in range(D):
            ax.plot(ETA_DEFAULT, phi1_def[i], 'o', color=colors[i], ms=6, zorder=5)
        ax.set_xlabel(r'Regularization level $\eta$', fontsize=11)
        ax.set_ylabel(r'$\phi_{1,i}^*$', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')
    fig.suptitle(r'Sensitivity of $\boldsymbol{\phi}_1^*$ to Tikhonov $\eta$',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('figure_ma_tikhonov.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('figure_ma_tikhonov.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figure_ma_tikhonov.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIGURA 3 — Allocazioni OOS mensili
# ─────────────────────────────────────────────────────────────────────────────

def figure_allocations_oos(ret_test, dates_test,
                            phi1_learned, phi1_bs, phi1_mjd):
    print("\n[Fig 3] OOS allocations...")
    import pandas as pd
    omega   = Z
    n_test  = len(ret_test)
    wealth  = {k: np.zeros(n_test+1) for k in ('RL','BS','MJD')}
    alloc   = {k: np.zeros((n_test,D)) for k in ('RL','BS','MJD')}
    for k_dict in wealth: wealth[k_dict][0] = X0

    for k, r in enumerate(ret_test):
        t = (k % K)*DT; r_ex = np.exp(r)-1-R_F*DT
        for key, phi1 in [('RL',phi1_learned),('BS',phi1_bs),('MJD',phi1_mjd)]:
            a = -phi1*(wealth[key][k]-omega)
            alloc[key][k] = a
            wealth[key][k+1] = wealth[key][k] + np.dot(a, r_ex)

    colors = ['steelblue','darkorange','forestgreen','firebrick','purple']
    fig, axes = plt.subplots(3, 1, figsize=(13,11), sharex=True)
    for ax, key, title in zip(axes,('RL','BS','MJD'),
                               ('RL Policy (learned)','Benchmark BS','Benchmark MJD')):
        for i in range(D):
            ax.plot(dates_test, alloc[key][:,i],
                    color=colors[i], lw=1.5, label=NAMES[i])
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-30'),
                   alpha=0.12, color='red', label='COVID-19 crash')
        ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'),
                   alpha=0.08, color='orange', label='2022 bear market')
        ax.set_ylabel(r'Allocation $a_{i,t}$', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc='upper right', ncol=3)
        ax.grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes[-1].set_xlabel('Date', fontsize=11)
    fig.suptitle('Monthly OOS allocations 2020-2023', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('figure_ma_alloc_oos.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('figure_ma_alloc_oos.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figure_ma_alloc_oos.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FIGURA 4 — Confronto benchmark BS vs MJD vs learned
# ─────────────────────────────────────────────────────────────────────────────

def figure_benchmark_compare(bs_params, mjd_params, phi1_learned):
    print("\n[Fig 4] Benchmark comparison...")
    phi1_bs  = phi1_star(bs_params)
    phi1_mjd = phi1_star(mjd_params)
    x = np.arange(D); width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for bars, phi1, col, lbl in [
        (ax.bar(x-width, phi1_bs,      width, color='steelblue',  alpha=0.85),
         phi1_bs,  'steelblue',  r'$\boldsymbol{\phi}_1^*$ BS'),
        (ax.bar(x,       phi1_mjd,     width, color='darkorange', alpha=0.85),
         phi1_mjd, 'darkorange', r'$\boldsymbol{\phi}_1^*$ MJD'),
        (ax.bar(x+width, phi1_learned, width, color='forestgreen',alpha=0.85),
         phi1_learned,'forestgreen',r'$\boldsymbol{\phi}_1$ learned (RL)'),
    ]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}',
                        xy=(bar.get_x()+bar.get_width()/2, h),
                        xytext=(0,3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
        bars.set_label(lbl)

    ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=10)
    ax.set_ylabel(r'Value of $\phi_{1,i}$', fontsize=11)
    ax.set_title(r'$\boldsymbol{\phi}_1^*$ (BS, MJD) and learned $\boldsymbol{\phi}_1$ per asset',
                 fontsize=11)
    ax.legend(fontsize=10); ax.axhline(0, color='black', lw=0.8)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure_ma_benchmark_compare.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('figure_ma_benchmark_compare.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figure_ma_benchmark_compare.pdf/png")
    print("\n  % difference MJD vs BS per asset:")
    for i, name in enumerate(NAMES):
        if abs(phi1_bs[i]) > 1e-6:
            diff = (phi1_mjd[i]-phi1_bs[i])/abs(phi1_bs[i])*100
            print(f"    {name:<20}: {diff:+.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n[0] Loading learned phi1...")
    phi1_learned = load_learned_phi1()

    print("\n[0] Loading data and calibrating...")
    ret_train, ret_test, dates_train, dates_test = load_multiasset()
    bs_params  = calibrate_bs(ret_train)
    mjd_params = calibrate_mjd(ret_train)
    phi1_bs    = phi1_star(bs_params)
    phi1_mjd   = phi1_star(mjd_params)
    print(f"  phi_1* BS  = {np.round(phi1_bs,  3)}")
    print(f"  phi_1* MJD = {np.round(phi1_mjd, 3)}")
    print(f"  phi_1 RL   = {np.round(phi1_learned, 3)}")

    figure_correlation(ret_train)
    figure_tikhonov_sensitivity(bs_params, mjd_params)
    figure_allocations_oos(ret_test, dates_test,
                           phi1_learned, phi1_bs, phi1_mjd)
    figure_benchmark_compare(bs_params, mjd_params, phi1_learned)

    print("\n" + "="*55)
    print("  All figures produced:")
    for f in ['figure_ma_corr', 'figure_ma_tikhonov',
              'figure_ma_alloc_oos', 'figure_ma_benchmark_compare']:
        print(f"    {f}.pdf/png")
    print("="*55)
