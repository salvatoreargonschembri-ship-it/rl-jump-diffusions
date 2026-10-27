"""
mv_multiseed.py -- Error bars across training seeds (defense material)
======================================================================
The thesis reports a single training realization (seed 42). This script
retrains the EXP2 policy (diagonal, bootstrap simulator, N = 20000, same
algorithm and hyperparameters as the thesis) for S independent seeds and
reports mean and standard deviation of:

  - the learned policy vector phi1 (component by component),
  - the out-of-sample metrics 2020-2023 (X_T, return, vol, Sharpe, MaxDD),
  - rolling 24-month out-of-sample windows: in what fraction of windows
    does the RL policy beat the BS analytical benchmark on Sharpe?

Each seed's benchmark rows are evaluated with that seed's learned omega,
exactly as in the thesis methodology.

Usage (Anaconda Prompt, from the folder containing mv_data.npz):

  python mv_multiseed.py --seeds 20 --jobs 4
  python mv_multiseed.py --seeds 20 --jobs 1     # serial (works in Jupyter)

Without mv_data.npz the data is downloaded with yfinance (column fix
included). Results are saved incrementally after every completed seed
(mv_multiseed_results.npz), so an interrupted run keeps its progress.
"""

import argparse
import contextlib
import io
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize
from scipy.stats import norm, poisson

# =============================================================================
# THESIS CODE (verbatim from mv_multiasset_corr.py)
# =============================================================================

TICKERS    = ['^GSPC', 'AGG', 'VGK', 'GLD', 'QQQ']
NAMES      = ['S&P 500', 'Bonds', 'MSCI Europe (VGK)', 'Gold', 'NASDAQ-100']
D          = len(TICKERS)   # numero di asset

R_F        = 0.02
DT         = 1 / 12        # passo mensile
T          = 1.0
K          = 12             # mesi per traiettoria
THETA      = 0.1
Z          = 1.4            # target ricchezza
X0         = 1.0

N_TRAIN    = 20000
BATCH      = 32
ALPHA_PSI  = 0.001
ALPHA_OMEGA = 0.05
DECAY      = 0.51
GRAD_CLIP  = 5.0

# Adam per phi
ADAM_LR    = 0.001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS   = 1e-8

TRAIN_END  = '2019-12-31'
TEST_END   = '2023-12-31'

ETA_TIKHONOV = 0.10   # Tikhonov regularization level (eq. 7.15 della tesi)


def load_multiasset(tickers=TICKERS, train_end=TRAIN_END, test_end=TEST_END):
    """
    Scarica dati mensili per tutti gli asset e restituisce
    matrici di rendimenti log-mensili (train e test).
    """
    import yfinance as yf
    import pandas as pd

    print(f"  Scaricamento dati per {len(tickers)} asset...")
    data = yf.download(tickers, start='2000-01-01', end=test_end,
                      progress=False)
    # Handle both MultiIndex and flat columns
    if isinstance(data.columns, pd.MultiIndex):
        raw = data['Close']
    else:
        raw = data[['Close']].rename(columns={'Close': tickers[0]}) if len(tickers)==1 else data
    raw = raw[list(tickers)]          # FIX: yfinance ordina le colonne alfabeticamente;
                                     # riallineale all'ordine di tickers/NAMES
    # Forward fill brevi gap, poi tronca al primo giorno in cui
    # TUTTI i ticker hanno dati (evita di perdere anni per un ETF)
    raw = raw.ffill().bfill()
    # Trova la data in cui tutti i ticker sono disponibili
    first_valid = raw.apply(lambda col: col.first_valid_index()).max()
    raw = raw.loc[first_valid:]
    print(f"  Dati disponibili dal: {raw.index[0].strftime('%Y-%m-%d')}")

    # Rendimenti log-giornalieri
    log_daily = np.log(raw / raw.shift(1)).dropna()

    # Aggrega in mensili (21 giorni di trading)
    dpm = 21
    n_months = len(log_daily) // dpm
    ret_monthly = np.zeros((n_months, len(tickers)))
    dates_monthly = []
    for i in range(n_months):
        block = log_daily.iloc[i*dpm:(i+1)*dpm]
        ret_monthly[i] = block.sum().values
        dates_monthly.append(pd.Timestamp(block.index[-1]))
    dates_monthly = np.array(dates_monthly)

    # Split train/test
    cutoff = pd.Timestamp(train_end)
    mask_train = np.array([d <= cutoff for d in dates_monthly])
    mask_test  = ~mask_train

    ret_train   = ret_monthly[mask_train]
    ret_test    = ret_monthly[mask_test]
    dates_train = dates_monthly[mask_train]
    dates_test  = dates_monthly[mask_test]

    print(f"  Training: {len(ret_train)} mesi "
          f"({dates_train[0].strftime('%Y-%m')} -> "
          f"{dates_train[-1].strftime('%Y-%m')})")
    print(f"  Test:     {len(ret_test)} mesi "
          f"({dates_test[0].strftime('%Y-%m')} -> "
          f"{dates_test[-1].strftime('%Y-%m')})")
    return ret_train, ret_test, dates_train, dates_test


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CALIBRAZIONE PARAMETRI
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_bs_multi(ret_monthly, r_f=R_F):
    """
    Calibra parametri BS multi-asset dai rendimenti mensili.
    Restituisce mu, Sigma (matrice covarianza diffusiva).
    """
    # Stima annualizzata
    mu    = ret_monthly.mean(axis=0) / DT + 0.5 * (ret_monthly.std(axis=0) / np.sqrt(DT))**2
    Sigma = np.cov(ret_monthly.T) / DT   # matrice covarianza annualizzata
    rho   = mu - r_f                      # vettore excess return
    print(f"  BS mu:  {mu}")
    print(f"  BS rho: {rho}")
    return dict(mu=mu, Sigma=Sigma, rho=rho, SigmaJ=np.zeros((D,D)))


def calibrate_mjd_multi(ret_monthly, r_f=R_F):
    """
    Calibra parametri MJD per ogni asset indipendentemente,
    poi costruisce Sigma dalla componente diffusiva.
    """
    bs = calibrate_bs_multi(ret_monthly, r_f)
    mu_vec    = bs['mu'].copy()
    sigmaJ_sq = np.zeros(D)
    sigma_diff = np.zeros(D)

    for i in range(D):
        ri = ret_monthly[:, i]
        def neg_ll(params):
            mu_i, sig_i, lam_i, m_i, delta_i = params
            if sig_i <= 0 or lam_i <= 0 or delta_i <= 0:
                return 1e10
            kappa = np.exp(m_i + 0.5*delta_i**2) - 1
            n_max = max(10, int(lam_i*DT*5 + 20))
            ns    = np.arange(0, n_max+1)
            pn    = poisson.pmf(ns, lam_i*DT)
            pdf   = np.zeros(len(ri))
            for n, p_n in zip(ns, pn):
                if p_n < 1e-15: continue
                mu_n  = (mu_i - 0.5*sig_i**2 - lam_i*kappa)*DT + n*m_i
                var_n = sig_i**2*DT + n*delta_i**2
                pdf  += p_n * norm.pdf(ri, mu_n, np.sqrt(var_n))
            return -np.sum(np.log(np.maximum(pdf, 1e-300)))

        sig0 = np.std(ri) / np.sqrt(DT)
        x0   = [mu_vec[i], sig0, 5.0, -0.01, 0.03]
        bds  = [(None,None),(1e-4,None),(1e-4,None),(None,None),(1e-4,None)]
        try:
            res = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bds,
                           options=dict(maxiter=300, ftol=1e-8))
            mu_i, sig_i, lam_i, m_i, delta_i = res.x
        except Exception:
            mu_i, sig_i = mu_vec[i], sig0
            lam_i, m_i, delta_i = 5., -0.01, 0.03
        # Fallback: if MJD diffusive vol collapses to < 30% of BS vol,
        # the optimizer moved all variance into jumps — revert to BS vol
        sig_bs_i = np.std(ret_monthly[:, i]) / np.sqrt(DT)
        if sig_i < 0.30 * sig_bs_i:
            sig_i = sig_bs_i
            print(f"    [fallback to BS vol for {NAMES[i]}]")
        sigma_diff[i] = sig_i
        sigmaJ_sq[i]  = lam_i*(np.exp(2*m_i+2*delta_i**2)
                                - 2*np.exp(m_i+0.5*delta_i**2) + 1)
        mu_vec[i] = mu_i
        print(f"  MJD asset {NAMES[i]}: mu={mu_i:.4f}  sigma={sig_i:.4f}"
              f"  lam={lam_i:.2f}  sigmaJ2={sigmaJ_sq[i]:.6f}")

    # Diffusive covariance: use empirical correlations, MJD diffusive vols
    corr  = np.corrcoef(ret_monthly.T)
    Sigma = np.outer(sigma_diff, sigma_diff) * corr
    # SPD projection
    Sigma = 0.5*(Sigma + Sigma.T)
    ev, evec = np.linalg.eigh(Sigma)
    ev = np.maximum(ev, 1e-8)
    Sigma = evec @ np.diag(ev) @ evec.T
    SigmaJ = np.diag(sigmaJ_sq)
    rho   = mu_vec - r_f
    return dict(mu=mu_vec, Sigma=Sigma, rho=rho, SigmaJ=SigmaJ,
                sigma_diff=sigma_diff, sigmaJ_sq=sigmaJ_sq)


def phi1_star_multi(params):
    """
    phi1* = (Sigma + SigmaJ + eps*I)^{-1} rho
    Tikhonov regularization with eta=0.10 (Remark rem:tikhonov, eq. 7.15).
    """
    Sigma_eff = params['Sigma'] + params['SigmaJ']
    eta = ETA_TIKHONOV                              # 0.10
    eps = eta / D * np.trace(Sigma_eff)
    return np.linalg.solve(Sigma_eff + eps*np.eye(D), params['rho'])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SIMULATORI MULTI-ASSET
# ─────────────────────────────────────────────────────────────────────────────

class RealDataSimulatorMulti:
    """Campiona rendimenti mensili reali con reimmissione."""
    def __init__(self, ret_monthly):
        self.returns = ret_monthly  # shape (n_months, D)

    def simulate_trajectory(self, n=K):
        idx  = np.random.randint(0, len(self.returns), size=n)
        return self.returns[idx]   # shape (n, D)



class MVModelMulti:
    """
    Parametrizzazioni multi-asset (Capitolo 7):
      J^psi(t,x) = (x-omega)^2 * exp(-psi3*(T-t)) + psi2*(t^2-T^2) + psi1*(t-T)
      q^phi(t,x,a) = -exp(-phi_{d+1} - phi_{d+2}*(T-t))/2 * ||a + phi1*(x-omega)||^2
                     - d*theta/2 * [log(2*pi*theta) + phi_{d+1} + phi_{d+2}*(T-t)]
      pi^phi(.|t,x) = N(-phi1*(x-omega), theta*exp(phi_{d+1}+phi_{d+2}*(T-t))*I_d)
    """
    def __init__(self, d=D, psi3_init=0.):
        self.d     = d
        self.psi   = np.array([0., 0., psi3_init])          # (psi1, psi2, psi3)
        self.phi1  = np.zeros(d)                              # vettore actor
        self.phi2  = 0.                                       # log-var intercept
        self.phi3  = 0.                                       # log-var slope
        self.omega = Z

    def _logvar(self, t):
        return np.clip(self.phi2 + self.phi3*(T-t), -20, 20)

    def _c(self, t):
        return np.exp(-self._logvar(t))

    def J(self, t, x):
        p = self.psi; o = self.omega
        e = np.exp(np.clip(-p[2]*(T-t), -30, 30))
        return (x-o)**2*e + p[1]*(t**2-T**2) + p[0]*(t-T) - (o-Z)**2

    def grad_J(self, t, x):
        p = self.psi; o = self.omega
        e = np.exp(np.clip(-p[2]*(T-t), -30, 30))
        return np.array([t-T, t**2-T**2, -(T-t)*(x-o)**2*e])

    def q(self, t, x, a):
        """a: vettore R^d"""
        o   = self.omega
        c   = self._c(t)
        r   = a + self.phi1*(x-o)          # residuo vettoriale
        lv  = self._logvar(t)
        return (-c/2 * np.dot(r, r)
                - self.d*THETA/2*(np.log(2*np.pi*THETA) + lv))

    def grad_q(self, t, x, a):
        """Restituisce (grad_phi1, grad_phi2, grad_phi3)."""
        o   = self.omega
        c   = self._c(t)
        r   = a + self.phi1*(x-o)
        lv  = self._logvar(t)
        rr  = np.dot(r, r)
        g1  = c * r * (x-o)                          # shape (d,)
        g2  = 0.5*c*rr - self.d*THETA/2              # scalar
        g3  = (T-t)*(0.5*c*rr - self.d*THETA/2)     # scalar
        return g1, g2, g3

    def sample_action(self, t, x):
        """Campiona a ~ pi^phi(.|t,x)."""
        mu_a = -self.phi1 * (x - self.omega)
        std_a = np.sqrt(THETA * np.exp(np.clip(self._logvar(t), -20, 20)))
        std_a = max(std_a, 1e-6)
        return np.random.normal(mu_a, std_a)   # shape (d,)

    def mean_action(self, t, x):
        return -self.phi1 * (x - self.omega)   # shape (d,)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING OFFLINE MULTI-ASSET

# =============================================================================
# MULTI-SEED EXPERIMENT
# =============================================================================

OUT_NPZ = 'mv_multiseed_results.npz'
N_SEED_TRAIN = 20000          # thesis horizon
ROLL_WIN = 24                 # rolling OOS window (months)


def train_once(ret_train, psi3_init, seed, n_train=N_SEED_TRAIN):
    """One training run, identical to the thesis loop; returns phi1, omega."""
    np.random.seed(seed)
    sim   = RealDataSimulatorMulti(ret_train)
    model = MVModelMulti(d=D, psi3_init=psi3_init)
    m_phi1 = np.zeros(D); v_phi1 = np.zeros(D)
    m_phi2 = 0.; v_phi2 = 0.; m_phi3 = 0.; v_phi3 = 0.
    tail = []

    for j in range(1, n_train+1):
        lr    = j**(-DECAY)
        d_psi = np.zeros(3)
        d_phi1 = np.zeros(D); d_phi2 = 0.; d_phi3 = 0.
        d_omega = 0.
        for _ in range(BATCH):
            rets = sim.simulate_trajectory(K)
            xs   = np.zeros(K+1); xs[0] = X0
            acts = np.zeros((K, D))
            for k in range(K):
                acts[k] = model.sample_action(k*DT, xs[k])
                r_excess = np.exp(rets[k]) - 1 - R_F*DT
                xn = xs[k] + np.dot(acts[k], r_excess)
                xs[k+1] = xn if np.isfinite(xn) else xs[k]
            for k in range(K):
                td = (model.J((k+1)*DT, xs[k+1]) - model.J(k*DT, xs[k])
                      - model.q(k*DT, xs[k], acts[k])*DT)
                if np.isfinite(td):
                    d_psi  += model.grad_J(k*DT, xs[k]) * td / K
                    g1, g2, g3 = model.grad_q(k*DT, xs[k], acts[k])
                    d_phi1 += g1 * td / K
                    d_phi2 += g2 * td / K
                    d_phi3 += g3 * td / K
            d_omega += xs[K] - Z

        norm_psi = np.linalg.norm(d_psi)
        if norm_psi > GRAD_CLIP * BATCH:
            d_psi *= GRAD_CLIP * BATCH / norm_psi
        norm_phi = np.sqrt(np.dot(d_phi1, d_phi1) + d_phi2**2 + d_phi3**2)
        if norm_phi > GRAD_CLIP * BATCH:
            scale = GRAD_CLIP * BATCH / norm_phi
            d_phi1 *= scale; d_phi2 *= scale; d_phi3 *= scale

        model.psi += lr * ALPHA_PSI * d_psi / BATCH

        g1b = d_phi1/BATCH; g2b = d_phi2/BATCH; g3b = d_phi3/BATCH
        m_phi1 = ADAM_BETA1*m_phi1 + (1-ADAM_BETA1)*g1b
        v_phi1 = ADAM_BETA2*v_phi1 + (1-ADAM_BETA2)*g1b**2
        m_phi2 = ADAM_BETA1*m_phi2 + (1-ADAM_BETA1)*g2b
        v_phi2 = ADAM_BETA2*v_phi2 + (1-ADAM_BETA2)*g2b**2
        m_phi3 = ADAM_BETA1*m_phi3 + (1-ADAM_BETA1)*g3b
        v_phi3 = ADAM_BETA2*v_phi3 + (1-ADAM_BETA2)*g3b**2
        bc1 = 1 - ADAM_BETA1**j
        bc2 = 1 - ADAM_BETA2**j
        model.phi1 += ADAM_LR*(m_phi1/bc1)/(np.sqrt(v_phi1/bc2)+ADAM_EPS)
        model.phi2 += ADAM_LR*(m_phi2/bc1)/(np.sqrt(v_phi2/bc2)+ADAM_EPS)
        model.phi3 += ADAM_LR*(m_phi3/bc1)/(np.sqrt(v_phi3/bc2)+ADAM_EPS)
        model.phi2  = np.clip(model.phi2, -10, 10)
        model.phi3  = np.clip(model.phi3, -10, 10)

        if j % 10 == 0:
            model.omega -= lr * ALPHA_OMEGA * d_omega / (BATCH * 10)

        if j > n_train - 500:
            tail.append(model.phi1.copy())

    phi1_tail = np.mean(tail, axis=0)
    return model.phi1.copy(), phi1_tail, float(model.omega)


def wealth_path(phi1, omega, ret_seg):
    n = len(ret_seg)
    w = np.zeros(n+1); w[0] = X0
    for k, r in enumerate(ret_seg):
        r_excess = np.exp(r) - 1 - R_F*DT
        a = -phi1 * (w[k] - omega)
        w[k+1] = w[k] + np.dot(a, r_excess)
    return w


def wealth_path_bh(ret_seg):
    n = len(ret_seg)
    w = np.zeros(n+1); w[0] = X0
    a_bh = np.ones(D) * (Z - X0) / D
    for k, r in enumerate(ret_seg):
        r_excess = np.exp(r) - 1 - R_F*DT
        w[k+1] = w[k] + np.dot(a_bh, r_excess)
    return w


def stats(w):
    ret     = np.diff(np.log(np.maximum(w, 1e-8)))
    ann_ret = ret.mean() * 12
    ann_vol = ret.std()  * np.sqrt(12)
    sharpe  = (ann_ret - R_F) / ann_vol if ann_vol > 0 else np.nan
    dd      = np.min(w / np.maximum.accumulate(w)) - 1
    return w[-1], ann_ret, ann_vol, sharpe, dd


def rolling_sharpes(phi1, omega, ret_test, win=ROLL_WIN):
    out = []
    for s in range(len(ret_test) - win + 1):
        _, _, _, sh, _ = stats(wealth_path(phi1, omega, ret_test[s:s+win]))
        out.append(sh)
    return np.array(out)


def run_seed(args_tuple):
    """Worker: one seed end to end. Returns a dict of results."""
    seed, data_path = args_tuple
    d = np.load(data_path)
    ret_train, ret_test = d['ret_train'], d['ret_test']
    with contextlib.redirect_stdout(io.StringIO()):
        bs = calibrate_bs_multi(ret_train)
    psi3_init = float(np.mean(np.diag(bs['Sigma'])))
    phi1_bs = phi1_star_multi(bs)

    t0 = time.time()
    phi1_fin, phi1_tail, omega = train_once(ret_train, psi3_init, seed)

    xt, ar, av, sh, dd = stats(wealth_path(phi1_fin, omega, ret_test))
    xtb, arb, avb, shb, ddb = stats(wealth_path(phi1_bs, omega, ret_test))
    roll_rl = rolling_sharpes(phi1_fin, omega, ret_test)
    roll_bs = rolling_sharpes(phi1_bs, omega, ret_test)
    frac_beat = float(np.mean(roll_rl > roll_bs))

    return dict(seed=seed, phi1=phi1_fin, phi1_tail=phi1_tail, omega=omega,
                rl=(xt, ar, av, sh, dd), bs=(xtb, arb, avb, shb, ddb),
                roll_rl=roll_rl, roll_bs=roll_bs, frac_beat=frac_beat,
                minutes=(time.time()-t0)/60)


def summarize(results, phi1_bs, phi1_mjd, ret_test):
    results = sorted(results, key=lambda r: r['seed'])
    S = len(results)
    phi1_all  = np.array([r['phi1'] for r in results])
    omega_all = np.array([r['omega'] for r in results])
    rl_all    = np.array([r['rl'] for r in results])
    bs_all    = np.array([r['bs'] for r in results])
    frac_all  = np.array([r['frac_beat'] for r in results])

    xt_bh, ar_bh, av_bh, sh_bh, dd_bh = stats(wealth_path_bh(ret_test))

    print("\n" + "="*78)
    print(f"  MULTI-SEED SUMMARY ({S} seeds, N = {N_SEED_TRAIN}, "
          f"thesis methodology)")
    print("="*78)
    print("  Learned phi1 (mean +- std across seeds) vs benchmarks:")
    for i, nm in enumerate(NAMES):
        print(f"    {nm:<22} {phi1_all[:,i].mean():7.3f} +- "
              f"{phi1_all[:,i].std():5.3f}   "
              f"(BS* {phi1_bs[i]:6.3f}, MJD* {phi1_mjd[i]:6.3f})")
    print(f"    omega                  {omega_all.mean():7.3f} +- "
          f"{omega_all.std():5.3f}")

    def line(lbl, arr):
        m, s = arr.mean(axis=0), arr.std(axis=0)
        print(f"  {lbl:<14} XT {m[0]:.3f}+-{s[0]:.3f}   "
              f"Ret {m[1]*100:5.2f}+-{s[1]*100:4.2f}%   "
              f"Vol {m[2]*100:5.2f}%   Sharpe {m[3]:.2f}+-{s[3]:.2f}   "
              f"MaxDD {m[4]:.1%}")

    print("\n  Out-of-sample 2020-2023 (mean +- std across seeds):")
    line("RL", rl_all)
    line("Benchmark BS", bs_all)
    print(f"  {'Buy-and-hold':<14} XT {xt_bh:.3f}         "
          f"Ret {ar_bh*100:5.2f}%        Vol {av_bh*100:5.2f}%   "
          f"Sharpe {sh_bh:.2f}        MaxDD {dd_bh:.1%}")
    print(f"\n  RL beats BH on X_T in {np.mean(rl_all[:,0] > xt_bh):.0%} "
          f"of seeds; beats the BS benchmark in "
          f"{np.mean(rl_all[:,0] > bs_all[:,0]):.0%} of seeds")
    print(f"  Rolling {ROLL_WIN}-month windows: RL Sharpe beats BS benchmark "
          f"in {frac_all.mean():.0%} of windows (mean across seeds)")
    print("="*78)

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    x = np.arange(D)
    ax.errorbar(x, phi1_all.mean(axis=0), yerr=phi1_all.std(axis=0),
                fmt='o', color='steelblue', capsize=5, label='RL (mean +- sd)')
    ax.scatter(x+0.12, phi1_bs,  color='darkorange', marker='s',
               label=r'$\phi_1^{*}$ BS')
    ax.scatter(x+0.24, phi1_mjd, color='forestgreen', marker='^',
               label=r'$\phi_1^{*}$ MJD')
    ax.set_xticks(x); ax.set_xticklabels(['SP500','Bonds','MSCI','Gold','QQQ'],
                                         fontsize=8)
    ax.set_title(f'Learned policy across {S} seeds')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.boxplot([rl_all[:,0], bs_all[:,0]])
    ax.set_xticks([1, 2]); ax.set_xticklabels(['RL', 'BS bench'])
    ax.axhline(xt_bh, color='gray', ls=':', label='Buy-and-hold')
    ax.set_ylabel(r'$X_T$ (OOS 2020-2023)')
    ax.set_title('Terminal wealth across seeds')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    diff = np.concatenate([r['roll_rl'] - r['roll_bs'] for r in results])
    ax.hist(diff, bins=25, color='steelblue', alpha=0.8)
    ax.axvline(0, color='black', lw=1.)
    ax.set_xlabel(f'Sharpe(RL) - Sharpe(BS bench), rolling {ROLL_WIN}m windows')
    ax.set_title('Rolling-window Sharpe difference (all seeds pooled)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure_multiseed.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('figure_multiseed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figure_multiseed.pdf/png")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2)-1))
    p.add_argument('--data', type=str, default='mv_data.npz')
    args = p.parse_known_args()[0]

    if os.path.exists(args.data):
        d = np.load(args.data)
        ret_train, ret_test = d['ret_train'], d['ret_test']
        print(f"Data from {args.data}: train {ret_train.shape}, "
              f"test {ret_test.shape}")
    else:
        print("mv_data.npz not found, downloading...")
        ret_train, ret_test, _, _ = load_multiasset()
        np.savez(args.data, ret_train=ret_train, ret_test=ret_test)

    bs  = calibrate_bs_multi(ret_train)
    mjd = calibrate_mjd_multi(ret_train)
    phi1_bs, phi1_mjd = phi1_star_multi(bs), phi1_star_multi(mjd)
    print(f"phi1*_BS  = {np.round(phi1_bs, 3)}")

    seeds = list(range(1, args.seeds+1))
    tasks = [(s, args.data) for s in seeds]
    results = []
    t0 = time.time()
    print(f"\nTraining {len(seeds)} seeds, N = {N_SEED_TRAIN}, "
          f"jobs = {args.jobs}\n")

    def save_partial():
        np.savez(OUT_NPZ,
                 seeds=np.array([r['seed'] for r in results]),
                 phi1=np.array([r['phi1'] for r in results]),
                 phi1_tail=np.array([r['phi1_tail'] for r in results]),
                 omega=np.array([r['omega'] for r in results]),
                 rl=np.array([r['rl'] for r in results]),
                 bs=np.array([r['bs'] for r in results]),
                 frac_beat=np.array([r['frac_beat'] for r in results]),
                 phi1_bs=phi1_bs, phi1_mjd=phi1_mjd)

    if args.jobs == 1:
        for tpl in tasks:
            r = run_seed(tpl)
            results.append(r); save_partial()
            el = (time.time()-t0)/60
            eta = el/len(results)*(len(seeds)-len(results))
            print(f"  seed {r['seed']:>3}: XT={r['rl'][0]:.3f} "
                  f"Sharpe={r['rl'][3]:.2f} om={r['omega']:.3f} "
                  f"({r['minutes']:.1f} min)  [{len(results)}/{len(seeds)}, "
                  f"~{eta:.0f} min left]", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_seed, tpl): tpl[0] for tpl in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r); save_partial()
                el = (time.time()-t0)/60
                eta = el/len(results)*(len(seeds)-len(results))
                print(f"  seed {r['seed']:>3}: XT={r['rl'][0]:.3f} "
                      f"Sharpe={r['rl'][3]:.2f} om={r['omega']:.3f} "
                      f"({r['minutes']:.1f} min)  [{len(results)}/{len(seeds)},"
                      f" ~{eta:.0f} min left]", flush=True)

    summarize(results, phi1_bs, phi1_mjd, ret_test)
    print(f"\n  Saved: {OUT_NPZ}")
    print("Done.")