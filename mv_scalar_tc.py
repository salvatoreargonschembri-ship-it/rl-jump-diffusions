"""
mv_scalar_tc.py -- Scalar transaction costs experiment (Section 6.11, Table 6.3)
Versione script del notebook mv_multiasset_tc.ipynb con i fix richiesti:
  - regola OOS con no-trade zone (phi4) applicata;
  - Maximum Drawdown calcolato rispetto al running maximum;
  - turnover coerente con la regola NTZ.
Single asset (^GSPC): non toccato dal fix del loader multi-asset.
Usage: python mv_scalar_tc.py
"""
import matplotlib
matplotlib.use('Agg')

"""
Reinforcement Learning for Jump-Diffusions
Mean-Variance Portfolio Selection with Proportional Transaction Costs
=====================================================================
Section 6.10 of the thesis.

Experiment:
  - Trains the clipped-quadratic q-learning algorithm for
    c in {0, 0.001, 0.005, 0.01}
  - Uses real S&P 500 data (yfinance) with fallback to MJD simulation
  - Train: 2000-2019 (240 months), Test: 2020-2023 (48 months)
  - Produces 4 figures:
      figure_tc_convergence.png   -- phi1 and phi4 convergence
      figure_tc_phi4_scaling.png  -- phi4 vs c with c^{1/3} reference
      figure_tc_oos.png           -- OOS wealth + Sharpe/turnover
      figure_tc_turnover.png      -- turnover change vs c

Usage:
    pip install yfinance numpy matplotlib scipy
    python3 mv_transaction_costs.py

Reference:
    Gao, Li, Zhou (2024) arXiv:2405.16449
    Thesis Section 6.10
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from scipy.optimize import minimize
from scipy.stats import norm, poisson

# ---------------------------------------------------------------------------
# 1.  GLOBAL PARAMETERS
# ---------------------------------------------------------------------------

R_F        = 0.02
DT         = 1 / 12
T          = 1.0
K          = int(T / DT)   # 12
THETA      = 0.1
Z          = 1.4
X0         = 1.0

N_ITER     = 20_000   # training iterations per cost level
BATCH      = 64       # trajectories per iteration (fully vectorized)
DECAY      = 0.51     # l(j) = j^{-DECAY}
ALPHA_PSI  = 0.001
ALPHA_OMEGA= 0.05
ALPHA_PHI  = 0.02
GRAD_CLIP  = 5.0
OMEGA_FREQ = 10

COST_LEVELS = [0.0, 0.001, 0.005, 0.01]
COLORS      = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

OUT_DIR = "."   # directory for output figures


# ---------------------------------------------------------------------------
# 2.  DATA LOADING
# ---------------------------------------------------------------------------

def load_sp500(start='2000-01-01', end='2023-12-31'):
    """
    Load S&P 500 monthly log-returns.
    Falls back to synthetic MJD simulation if network is unavailable.
    """
    try:
        import yfinance as yf
        data   = yf.download('^GSPC', start=start, end=end, progress=False)
        closes = data['Close'].dropna().values.flatten()
        if len(closes) < 100:
            raise ValueError("Insufficient data returned")
        log_ret_daily = np.diff(np.log(closes))
        days_pm  = 21
        n_months = len(log_ret_daily) // days_pm
        monthly  = np.array([
            log_ret_daily[i*days_pm:(i+1)*days_pm].sum()
            for i in range(n_months)
        ])
        print(f"  S&P 500 live: {n_months} monthly returns ({start} -> {end})")
        return log_ret_daily, monthly
    except Exception as e:
        print(f"  yfinance unavailable ({e}). Using MJD fallback.")
        return _simulate_mjd()


def _simulate_mjd():
    """Synthetic MJD returns with thesis-calibrated parameters (Table 6.1)."""
    MU, SIG, LAM, MJ, DJ = 0.0807, 0.1309, 28.0, -0.0037, 0.0256
    KAP   = np.exp(MJ + 0.5*DJ**2) - 1
    rng   = np.random.RandomState(1)
    n_d   = 252 * 24           # 24 years of daily returns
    dt_d  = 1 / 252
    dW    = rng.randn(n_d) * np.sqrt(dt_d)
    r_d   = (MU - 0.5*SIG**2 - LAM*KAP)*dt_d + SIG*dW
    nj    = rng.poisson(LAM*dt_d, n_d)
    for i in range(n_d):
        if nj[i] > 0:
            r_d[i] += float(np.sum(rng.normal(MJ, DJ, nj[i])))
    days_pm  = 21
    n_months = n_d // days_pm
    monthly  = np.array([r_d[i*days_pm:(i+1)*days_pm].sum()
                         for i in range(n_months)])
    print(f"  Synthetic MJD: {n_months} monthly returns generated")
    return r_d, monthly


# ---------------------------------------------------------------------------
# 3.  CALIBRATION
# ---------------------------------------------------------------------------

def calibrate_bs(log_ret_daily, dt_d=1/252):
    sigma = np.std(log_ret_daily) / np.sqrt(dt_d)
    mu    = np.mean(log_ret_daily) / dt_d + 0.5*sigma**2
    rho   = (mu - R_F) / sigma
    print(f"  BS:  mu={mu:.4f}  sigma={sigma:.4f}  phi1*={rho*sigma/sigma**2:.4f}")
    return dict(mu=mu, sigma=sigma, rho=rho, sJ2=0.)


def calibrate_mjd(log_ret_daily, dt_d=1/252):
    def neg_ll(p):
        mu, sig, lam, m, d = p
        if sig <= 0 or lam <= 0 or d <= 0:
            return 1e10
        kap   = np.exp(m + 0.5*d**2) - 1
        n_max = max(10, int(lam*dt_d*5 + 20))
        ns    = np.arange(n_max+1)
        pn    = poisson.pmf(ns, lam*dt_d)
        pdf   = np.zeros(len(log_ret_daily))
        for n, p_n in zip(ns, pn):
            if p_n < 1e-15:
                continue
            mu_n  = (mu - 0.5*sig**2 - lam*kap)*dt_d + n*m
            var_n = sig**2*dt_d + n*d**2
            pdf  += p_n * norm.pdf(log_ret_daily, mu_n, np.sqrt(var_n))
        return -np.sum(np.log(np.maximum(pdf, 1e-300)))

    bs  = calibrate_bs(log_ret_daily, dt_d)
    x0  = [bs['mu'], bs['sigma'], 28., -0.004, 0.027]
    bds = [(None,None),(1e-4,None),(1e-4,None),(None,None),(1e-4,None)]
    res = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bds,
                   options=dict(maxiter=500, ftol=1e-9))
    mu, sig, lam, m, d = res.x
    rho   = (mu - R_F) / sig
    sJ2   = lam*(np.exp(2*m+2*d**2) - 2*np.exp(m+0.5*d**2) + 1)
    kap   = np.exp(m + 0.5*d**2) - 1
    phi1s = rho*sig/(sig**2+sJ2)
    print(f"  MJD: mu={mu:.4f}  sigma={sig:.4f}  lam={lam:.2f}  "
          f"sJ2={sJ2:.5f}  phi1*={phi1s:.4f}")
    return dict(mu=mu, sigma=sig, lam=lam, m=m, delta=d,
                rho=rho, sJ2=sJ2, kappa=kap)


def phi1_star(params):
    sig = params['sigma']
    rho = (params['mu'] - R_F) / sig
    return rho*sig / (sig**2 + params.get('sJ2', 0.))


# ---------------------------------------------------------------------------
# 4.  PARAMETRIC FAMILIES  (Section 6.10.4)
# ---------------------------------------------------------------------------
#
# Critic:  J^psi(t,x) = (x-om)^2 * exp(-psi[2]*(T-t))
#                      + psi[1]*(t^2-T^2) + psi[0]*(t-T) - (om-z)^2
#
# Actor:   q^phi(t,x,a,a_prev)
#        = -c(t)/2 * (a + phi[0]*(x-om))^2
#          - phi[3]*|a - a_prev|          <- NTZ term (new vs frictionless)
#          - theta/2 * [log(2*pi*theta) + phi[1] + phi[2]*(T-t)]
#
# where c(t) = exp(-phi[1] - phi[2]*(T-t))
#
# Parameters:
#   phi[0]  -> phi1* = rho*sigma/(sigma^2+sigmaJ^2)
#   phi[1], phi[2] -> variance of exploratory policy
#   phi[3]  -> NTZ half-width (learned; -> 0 as c -> 0)


def _c(phi, t):
    return np.exp(np.clip(-phi[1] - phi[2]*(T-t), -30, 30))


def J_fn(psi, om, t, x):
    e = np.exp(np.clip(-psi[2]*(T-t), -30, 30))
    return (x-om)**2*e + psi[1]*(t**2-T**2) + psi[0]*(t-T) - (om-Z)**2


def gJ_fn(psi, om, t, x):
    e = np.exp(np.clip(-psi[2]*(T-t), -30, 30))
    return np.array([t - T,
                     t**2 - T**2,
                     -(T-t)*(x-om)**2*e])


def q_fn(phi, om, t, x, a, ap):
    ct = _c(phi, t)
    r  = a + phi[0]*(x - om)
    return (-ct/2*r**2
            - phi[3]*np.abs(a - ap)
            - THETA/2*(np.log(2*np.pi*THETA) + phi[1] + phi[2]*(T-t)))


def gq_fn(phi, om, t, x, a, ap):
    ct = _c(phi, t)
    r  = a + phi[0]*(x - om)
    return np.array([ct*r*(x - om),
                     0.5*ct*r**2 - THETA/2,
                     (T-t)*(0.5*ct*r**2 - THETA/2),
                     -np.abs(a - ap)])


# ---------------------------------------------------------------------------
# 5.  ADAM OPTIMIZER
# ---------------------------------------------------------------------------

def adam_init():
    return {'m': np.zeros(4), 'v': np.zeros(4), 't': 0}


def adam_step(phi, grad, lr, state, b1=0.9, b2=0.999, eps=1e-8):
    state['t'] += 1
    state['m']  = b1*state['m'] + (1-b1)*grad
    state['v']  = b2*state['v'] + (1-b2)*grad**2
    mh  = state['m'] / (1 - b1**state['t'])
    vh  = state['v'] / (1 - b2**state['t'])
    # phi4 gets a 10x higher learning rate to compensate for its small gradient
    lr_vec      = np.array([lr, lr, lr, lr*10.])
    phi = phi + lr_vec * mh / (np.sqrt(vh) + eps)
    phi[1:3] = np.clip(phi[1:3], -10, 10)
    phi[3]   = max(0., phi[3])
    return phi, state


# ---------------------------------------------------------------------------
# 6.  WEALTH STEP
# ---------------------------------------------------------------------------

def step(x, a, ap, ret, cost):
    """One-step discounted wealth update including proportional TC."""
    return x + a*(np.exp(ret) - 1 - R_F*DT) - cost*np.abs(a - ap)


# ---------------------------------------------------------------------------
# 7.  OFFLINE Q-LEARNING  (fully vectorized over BATCH trajectories)
# ---------------------------------------------------------------------------

def train(train_ret, cost=0., seed=42, N=N_ITER, batch=BATCH,
          alpha_phi=ALPHA_PHI, verbose=True):
    """
    Returns phi, psi, omega, hist_phi1, hist_phi4.
    phi[0] converges to phi1* = rho*sigma/(sigma^2+sigmaJ^2).
    phi[3] is the learned NTZ width parameter.
    """
    np.random.seed(seed)
    phi    = np.zeros(4)
    psi    = np.zeros(3)
    omega  = Z
    adam_s = adam_init()
    h1     = np.zeros(N)
    h4     = np.zeros(N)

    t_grid   = np.arange(K) * DT          # (K,)
    tau_grid = T - t_grid                  # (K,)

    for j in range(1, N+1):
        lr = j**(-DECAY)

        # -- Sample all returns: (batch, K) --------------------------------
        idx     = np.random.randint(0, len(train_ret), (batch, K))
        ret_b   = train_ret[idx]
        noise_b = np.random.randn(batch, K)
        rnd_ntz = np.random.rand(batch, K)

        # -- Simulate trajectories ----------------------------------------
        xs   = np.zeros((batch, K+1)); xs[:, 0] = X0
        acts = np.zeros((batch, K))
        ap   = np.zeros(batch)

        for k in range(K):
            ct  = _c(phi, t_grid[k])
            mu  = -phi[0]*(xs[:, k] - omega)
            std = max(np.sqrt(THETA*ct), 1e-4)
            a   = mu + std*noise_b[:, k]

            # NTZ: with probability p_stay, hold current position
            ntz    = phi[3] / max(ct, 1e-8)
            inside = np.abs(a - ap) < ntz
            p_stay = np.minimum(1., np.exp(-np.abs(a - ap)/max(std, 1e-6)))
            stay   = inside & (rnd_ntz[:, k] < p_stay)
            a      = np.where(stay, ap, a)

            acts[:, k] = a
            xs[:, k+1] = step(xs[:, k], a, ap, ret_b[:, k], cost)
            xs[:, k+1] = np.where(np.isfinite(xs[:, k+1]),
                                   xs[:, k+1], xs[:, k])
            ap = a.copy()

        # -- Gradients (vectorized over batch and time) --------------------
        dp = np.zeros(3)
        dq = np.zeros(4)

        for k in range(K):
            t   = t_grid[k]
            tau = tau_grid[k]
            apk = acts[:, k-1] if k > 0 else np.zeros(batch)
            ct  = _c(phi, t)

            e_k  = np.exp(np.clip(-psi[2]*tau,      -30, 30))
            e_k1 = np.exp(np.clip(-psi[2]*(tau-DT), -30, 30))

            Jn = (xs[:, k+1]-omega)**2*e_k1 \
               + psi[1]*((k+1)**2*DT**2-T**2) \
               + psi[0]*((k+1)*DT-T) - (omega-Z)**2

            Jc = (xs[:, k]-omega)**2*e_k \
               + psi[1]*(k**2*DT**2-T**2) \
               + psi[0]*(k*DT-T) - (omega-Z)**2

            r_act = acts[:, k] + phi[0]*(xs[:, k] - omega)
            qc    = (-ct/2*r_act**2
                     - phi[3]*np.abs(acts[:, k]-apk)
                     - THETA/2*(np.log(2*np.pi*THETA)+phi[1]+phi[2]*tau))

            td = (Jn - Jc - qc*DT) / K   # shape: (batch,)

            # Critic gradient
            gJ_b = np.column_stack([
                np.full(batch, t - T),
                np.full(batch, t**2 - T**2),
                -tau*(xs[:, k]-omega)**2*e_k
            ])
            dp += (gJ_b * td[:, None]).mean(axis=0)

            # Actor gradient
            gq_b = np.column_stack([
                ct*r_act*(xs[:, k]-omega),
                0.5*ct*r_act**2 - THETA/2,
                tau*(0.5*ct*r_act**2 - THETA/2),
                -np.abs(acts[:, k]-apk)
            ])
            dq += (gq_b * td[:, None]).mean(axis=0)

        # -- Omega update --------------------------------------------------
        if j % OMEGA_FREQ == 0:
            omega -= lr * ALPHA_OMEGA * np.mean(xs[:, -1] - Z)

        # -- Gradient clipping ---------------------------------------------
        for arr in [dp, dq]:
            nm = np.linalg.norm(arr)
            if nm > GRAD_CLIP:
                arr *= GRAD_CLIP / nm

        # -- Parameter updates ---------------------------------------------
        psi           += lr * ALPHA_PSI * dp
        phi, adam_s    = adam_step(phi, dq, lr*alpha_phi, adam_s)
        h1[j-1], h4[j-1] = phi[0], phi[3]

        if verbose and j % max(1, N//4) == 0:
            print(f"    iter {j:6d}: phi1={phi[0]:.4f}  "
                  f"phi4={phi[3]:.5f}  omega={omega:.3f}")
            sys.stdout.flush()

    return phi, psi, omega, h1, h4


# ---------------------------------------------------------------------------
# 8.  EVALUATION
# ---------------------------------------------------------------------------

def evaluate_oos(phi, omega, oos_ret, cost=0., n_paths=2000, seed=1):
    """Deterministic mean-policy evaluation on held-out returns."""
    np.random.seed(seed)
    K_oos = len(oos_ret)
    W     = np.zeros((n_paths, K_oos+1)); W[:, 0] = X0
    ap    = np.zeros(n_paths)

    for k, r in enumerate(oos_ret):
        x         = W[:, k]
        a_target  = -phi[0]*(x - omega)
        # Regola con NTZ: si opera solo se il target dista piu' di phi4
        # dall'azione precedente, altrimenti si mantiene la posizione.
        a         = np.where(np.abs(a_target - ap) > max(phi[3], 0.), a_target, ap)
        W[:, k+1] = np.clip(step(x, a, ap, r, cost), -100, 1000)
        ap        = a.copy()

    XT  = W[:, -1]
    mu  = float(np.mean(XT))

    # Annualised metrics: compute per-path annualised log-return and vol
    # to avoid degenerate variance when paths are near-deterministic
    log_W = np.log(np.maximum(W, 1e-6))       # (n_paths, K_oos+1)
    monthly_ret = np.diff(log_W, axis=1)       # (n_paths, K_oos)
    ann_ret_paths = monthly_ret.mean(axis=1) * 12 * 100  # annualised %
    ann_vol_paths = monthly_ret.std(axis=1) * np.sqrt(12) * 100

    ann_ret = float(np.mean(ann_ret_paths))
    ann_vol = float(np.mean(ann_vol_paths))
    sharpe  = (ann_ret - R_F*100) / ann_vol if ann_vol > 0 else 0.
    peak    = np.maximum.accumulate(W, axis=1)
    maxdd   = float(np.mean(
        np.min(W / np.maximum(peak, 1e-6), axis=1) - 1
    )) * 100

    return dict(XT=mu, ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, maxdd=maxdd, wealth=W)


def compute_turnover(phi, omega, oos_ret, cost=0., n_paths=500, seed=2):
    np.random.seed(seed)
    K_oos = len(oos_ret)
    x     = np.full(n_paths, X0)
    ap    = np.zeros(n_paths)
    tv    = np.zeros(n_paths)

    for k, r in enumerate(oos_ret):
        a_target = -phi[0]*(x - omega)
        a   = np.where(np.abs(a_target - ap) > max(phi[3], 0.), a_target, ap)
        tv += np.abs(a - ap)
        x   = np.clip(step(x, a, ap, r, cost), -100, 1000)
        ap  = a.copy()

    return float(np.mean(tv)) / K_oos


# ---------------------------------------------------------------------------
# 9.  PLOTTING
# ---------------------------------------------------------------------------

def plot_convergence(histories, phi1s_mjd, smooth=500, out=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, c in enumerate(COST_LEVELS):
        h1, h4 = histories[c]
        s1 = np.convolve(h1, np.ones(smooth)/smooth, mode='valid')
        s4 = np.convolve(h4, np.ones(smooth)/smooth, mode='valid')
        axes[0].plot(s1, color=COLORS[i], label=f'c={c:.3f}', lw=1.5)
        axes[1].plot(s4, color=COLORS[i], label=f'c={c:.3f}', lw=1.5)
    axes[0].axhline(phi1s_mjd, color='gray', ls='--', lw=1.2,
                    label=f'$\\phi_1^*$ = {phi1s_mjd:.3f}')
    for ax, ylabel, title in zip(
        axes,
        ['$\\phi_1$', '$\\phi_4$ (NTZ width)'],
        ['Convergence of $\\phi_1$ by transaction cost level',
         'Convergence of $\\phi_4$']
    ):
        ax.set_xlabel('Iteration'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  {out}")
    plt.close()


def plot_phi4_scaling(results, out=None):
    phi4v = [results[c]['phi'][3] for c in COST_LEVELS]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(COST_LEVELS, phi4v, 'o-', color='steelblue', ms=9, lw=2)
    nz = [(c, v) for c, v in zip(COST_LEVELS, phi4v) if c > 0 and v > 1e-7]
    if nz:
        c0, v0 = nz[0]
        sc = v0 / (c0**(1/3))
        cx = np.linspace(0.0003, 0.012, 200)
        ax.plot(cx, sc*cx**(1/3), '--', color='gray', lw=1.2,
                label='$c^{1/3}$ reference')
        ax.legend(fontsize=9)
    ax.set_xlabel('Transaction cost $c$')
    ax.set_ylabel('Learned $\\phi_4$ (NTZ width parameter)')
    ax.set_title('NTZ width vs cost level')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  {out}")
    plt.close()


def plot_oos(oos_stats, out=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    K_oos  = oos_stats[COST_LEVELS[0]]['wealth'].shape[1] - 1
    months = np.arange(K_oos + 1)

    for i, c in enumerate(COST_LEVELS):
        W   = oos_stats[c]['wealth']
        med = np.median(W, axis=0)
        p25 = np.percentile(W, 25, axis=0)
        p75 = np.percentile(W, 75, axis=0)
        axes[0].plot(months, med, color=COLORS[i], label=f'c={c:.3f}', lw=1.8)
        axes[0].fill_between(months, p25, p75, color=COLORS[i], alpha=0.12)
    axes[0].axhline(Z, color='black', ls=':', lw=1, label=f'Target $z={Z}$')
    axes[0].set_xlabel('Month'); axes[0].set_ylabel('Wealth $X_t$')
    axes[0].set_title('OOS wealth trajectories 2020-2023 (median ± IQR)')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    ax2  = axes[1]; ax2b = ax2.twinx()
    xp   = np.arange(len(COST_LEVELS))
    sh   = [oos_stats[c]['sharpe']   for c in COST_LEVELS]
    tv   = [oos_stats[c]['turnover'] for c in COST_LEVELS]
    ax2.bar( xp - 0.2, sh, 0.35, color=COLORS, alpha=0.85, label='Sharpe ratio')
    ax2b.bar(xp + 0.2, tv, 0.35, color=COLORS, alpha=0.45, hatch='//',
             label='Monthly turnover')
    ax2.axhline(0, color='black', lw=0.8, ls='--')
    ax2.set_xticks(xp)
    ax2.set_xticklabels([f'c={c:.3f}' for c in COST_LEVELS], rotation=15)
    ax2.set_ylabel('Sharpe ratio')
    ax2b.set_ylabel('Monthly turnover $|\\Delta a|$')
    ax2.set_title('Sharpe ratio and turnover by cost level')
    ax2.legend(loc='upper left', fontsize=9)
    ax2b.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  {out}")
    plt.close()


def plot_turnover_change(oos_stats, out=None):
    tv0  = oos_stats[0.0]['turnover']
    pct  = [(oos_stats[c]['turnover']/max(tv0, 1e-10) - 1)*100
            for c in COST_LEVELS]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(len(COST_LEVELS)), pct, color=COLORS, alpha=0.85)
    ax.set_xticks(range(len(COST_LEVELS)))
    ax.set_xticklabels([f'c={c:.3f}' for c in COST_LEVELS])
    ax.set_ylabel('Turnover change vs $c=0$ (%)')
    ax.set_title('Impact of transaction costs on portfolio turnover')
    ax.axhline(0, color='black', lw=0.8)
    ax.grid(alpha=0.3, axis='y')
    for i, v in enumerate(pct):
        ax.text(i, v + (0.3 if v >= 0 else -1.0),
                f'{v:+.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 10.  MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("MV Portfolio with Proportional Transaction Costs")
    print("Section 6.10 — Thesis Numerical Experiment")
    print("=" * 65)

    # Data
    print("\n[1] Loading data...")
    log_ret_daily, monthly = load_sp500()
    train_ret = monthly[:240]   # 2000-2019
    test_ret  = monthly[240:]   # 2020-2023
    print(f"    Train: {len(train_ret)} months | Test: {len(test_ret)} months")

    # Calibration
    print("\n[2] Calibrating parameters (training set only)...")
    bs_params  = calibrate_bs(log_ret_daily[:5040])
    mjd_params = calibrate_mjd(log_ret_daily[:5040])
    phi1s_bs   = phi1_star(bs_params)
    phi1s_mjd  = phi1_star(mjd_params)
    print(f"    phi1* (BS)  = {phi1s_bs:.4f}")
    print(f"    phi1* (MJD) = {phi1s_mjd:.4f}")

    # Training
    print(f"\n[3] Training ({N_ITER} iterations x {len(COST_LEVELS)} cost levels)...")
    results   = {}
    histories = {}

    for cost in COST_LEVELS:
        t0 = time.time()
        print(f"\n  c = {cost:.3f}")
        phi, psi, omega, h1, h4 = train(
            train_ret, cost=cost, seed=42,
            N=N_ITER, batch=BATCH, alpha_phi=ALPHA_PHI
        )
        elapsed = time.time() - t0
        print(f"  Done ({elapsed:.0f}s): phi1={phi[0]:.4f}  "
              f"phi4={phi[3]:.5f}  omega={omega:.3f}")
        results[cost]   = dict(phi=phi, psi=psi, omega=omega)
        histories[cost] = (h1, h4)

    # Out-of-sample
    print("\n[4] Out-of-sample evaluation (2020-2023)...")
    oos_stats = {}
    print(f"\n  {'c':>7} | {'phi1':>6} | {'phi4':>7} | "
          f"{'Ann.Ret%':>9} | {'Sharpe':>7} | {'MaxDD%':>8} | "
          f"{'Turnover':>9} | {'XT':>6}")
    print("  " + "-"*75)

    for cost in COST_LEVELS:
        r     = results[cost]
        phi   = r['phi']; omega = r['omega']
        oos   = evaluate_oos(phi, omega, test_ret, cost=cost)
        tv    = compute_turnover(phi, omega, test_ret, cost=cost)
        oos['turnover'] = tv
        oos_stats[cost] = oos
        print(f"  {cost:>7.3f} | {phi[0]:>6.4f} | {phi[3]:>7.5f} | "
              f"{oos['ann_ret']:>8.2f}% | {oos['sharpe']:>7.3f} | "
              f"{oos['maxdd']:>7.2f}% | {tv:>9.5f} | {oos['XT']:>6.3f}")

    # Figures
    print("\n[5] Generating figures...")
    import os; os.makedirs(OUT_DIR, exist_ok=True)

    plot_convergence(
        histories, phi1s_mjd, smooth=max(100, N_ITER//40),
        out=f"{OUT_DIR}/figure_tc_convergence.png"
    )
    plot_phi4_scaling(
        results, out=f"{OUT_DIR}/figure_tc_phi4_scaling.png"
    )
    plot_oos(
        oos_stats, out=f"{OUT_DIR}/figure_tc_oos.png"
    )
    plot_turnover_change(
        oos_stats, out=f"{OUT_DIR}/figure_tc_turnover.png"
    )

    print("\nDone. Figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()