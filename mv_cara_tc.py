"""
mv_cara_tc.py — Multi-Asset CARA Portfolio Selection with Transaction Costs
============================================================================
Extends the CARA q-learning experiment of Section 6.11 (mv_multiasset_cara.py)
by adding proportional transaction costs c > 0, following the same clipped-
quadratic parametrization used for the MV case in Section 6.10.

Key difference from MV-TC:
  - CARA optimal allocation is wealth-INDEPENDENT: a* = Sigma_eff^{-1} rho / alpha
  - The NTZ is centred at a constant vector phi1 (not -phi1*(x-omega))
  - No omega parameter needed

Two experiments (parallel to CARA-EXP1 and CARA-EXP2):
  CARA-TC-EXP1: diagonal Sigma_J (MJD calibration), c = 0.001
  CARA-TC-EXP2: full Sigma_J (Threshold Estimator),  c = 0.001

Output:
  figure_cara_tc_convergence.pdf/png  — phi1 and phi4 convergence
  figure_cara_tc_oos.pdf/png          — OOS wealth paths 2020-2023

Usage:
  python mv_cara_tc.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize
from scipy.stats import norm, poisson
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

mpl.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GLOBAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TICKERS   = ['^GSPC', 'AGG', 'VGK', 'GLD', 'QQQ']
NAMES     = ['S&P 500', 'Bonds', 'MSCI Europe', 'Gold', 'NASDAQ-100']
D         = len(TICKERS)

R_F        = 0.02
DT         = 1 / 12
T          = 1.0
K          = 12
THETA      = 0.1
X0         = 1.0
ALPHA_CARA = 1.0

# Transaction cost levels to test
TC_LEVELS  = [0.000, 0.001, 0.005, 0.010]

N_TRAIN    = 20000
BATCH      = 32
DECAY      = 0.51
GRAD_CLIP  = 5.0
ETA_TIKHO  = 0.10

ADAM_LR    = 0.001
ADAM_B1    = 0.9
ADAM_B2    = 0.999
ADAM_EPS   = 1e-8
ADAM_LR_P4 = 0.0005   # smaller lr for phi4 (NTZ)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(start='2000-01-01', end='2023-12-31', train_end='2019-12-31'):
    try:
        import yfinance as yf
        import pandas as pd
        data = yf.download(TICKERS, start=start, end=end,
                           progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            raw = data['Close']
        else:
            raw = data
        raw = raw[list(TICKERS)]   # FIX: yfinance ordina le colonne alfabeticamente;
                                   # riallineale all'ordine di TICKERS/NAMES
        # Forward-fill then back-fill to handle assets with later start dates
        # FIX: trova la prima data comune sui dati GREZZI, taglia, poi ffill.
        # Niente bfill: creava prezzi artificiali prima della quotazione degli ETF
        # (AGG 09/2003, GLD 11/2004, VGK 03/2005) e rendimenti zero fittizi.
        first_valid = raw.apply(lambda col: col.first_valid_index()).max()
        raw = raw.loc[first_valid:]
        raw = raw.ffill()
        print(f"  Common start: {raw.index[0].strftime('%Y-%m-%d')}")
        log_daily = np.log(raw / raw.shift(1)).dropna()
        dpm      = 21
        n_months = len(log_daily) // dpm
        monthly  = np.zeros((n_months, D))
        dates_m  = []
        for i in range(n_months):
            block      = log_daily.iloc[i*dpm:(i+1)*dpm]
            monthly[i] = block.sum().values
            dates_m.append(pd.Timestamp(block.index[-1]))
        dates_m = np.array(dates_m)
        cutoff  = pd.Timestamp(train_end)
        mask_tr = np.array([d <= cutoff for d in dates_m])
        monthly_train = monthly[mask_tr]
        monthly_test  = monthly[~mask_tr]
        # Daily returns for training (for calibration)
        n_train_days  = mask_tr.sum() * dpm
        daily_train   = log_daily.values[:n_train_days]
        print(f"  Train: {len(monthly_train)} months | Test: {len(monthly_test)} months")
        return daily_train, monthly_train, monthly_test
    except Exception as e:
        print(f"  yfinance unavailable ({e}). Using synthetic fallback.")
        return _synthetic_data()


def _synthetic_data():
    rng = np.random.RandomState(1)
    MU  = np.array([0.033, 0.071, 0.080, 0.048, 0.053])
    SIG = np.array([0.165, 0.054, 0.200, 0.152, 0.227])
    LAM = np.array([5.00, 16.79, 3.17, 13.14, 10.12])
    MJ  = np.full(D, -0.0037)
    DJ  = np.full(D,  0.0256)
    CORR = np.array([
        [1.00, -0.20,  0.75,  0.05,  0.89],
        [-0.20, 1.00, -0.15,  0.02, -0.05],
        [0.75, -0.15,  1.00,  0.08,  0.70],
        [0.05,  0.02,  0.08,  1.00,  0.04],
        [0.89, -0.05,  0.70,  0.04,  1.00],
    ])
    SIGMA_COV = np.outer(SIG, SIG) * CORR
    n_days = 252 * 24
    dt_d   = 1/252
    L      = np.linalg.cholesky(SIGMA_COV * dt_d)
    drift  = (MU - 0.5*SIG**2) * dt_d
    daily  = drift[None, :] + rng.randn(n_days, D) @ L.T
    for i in range(D):
        kap = np.exp(MJ[i] + 0.5*DJ[i]**2) - 1
        nj  = rng.poisson(LAM[i]*dt_d, n_days)
        daily[:, i] -= LAM[i]*kap*dt_d
        for t in range(n_days):
            if nj[t] > 0:
                daily[t, i] += float(np.sum(rng.normal(MJ[i], DJ[i], nj[t])))
    days_pm  = 21
    n_months = n_days // days_pm
    monthly  = np.array([
        daily[i*days_pm:(i+1)*days_pm].sum(axis=0)
        for i in range(n_months)
    ])
    split = int(n_months * 240/288)
    return daily[:split*21], monthly[:split], monthly[split:]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_bs(daily, dt_d=1/252):
    sigma = np.std(daily, axis=0) / np.sqrt(dt_d)
    mu    = np.mean(daily, axis=0) / dt_d + 0.5*sigma**2
    Sigma = np.cov(daily.T) / dt_d
    rho   = mu - R_F
    return dict(mu=mu, sigma=sigma, Sigma=Sigma, rho=rho)


def calibrate_mjd_marginal(daily, dt_d=1/252):
    """Per-asset MJD MLE, returns diagonal SigmaJ."""
    from scipy.stats import norm as snorm
    sJ2 = np.zeros(D)
    for i in range(D):
        ret = daily[:, i]
        def neg_ll(p):
            mu, sig, lam, m, delta = p
            if sig <= 0 or lam <= 0 or delta <= 0: return 1e10
            kap   = np.exp(m + 0.5*delta**2) - 1
            nmax  = max(10, int(lam*dt_d*5 + 20))
            ns    = np.arange(nmax+1)
            pn    = poisson.pmf(ns, lam*dt_d)
            pdf   = np.zeros(len(ret))
            for n, p_n in zip(ns, pn):
                if p_n < 1e-15: continue
                mu_n  = (mu - 0.5*sig**2 - lam*kap)*dt_d + n*m
                var_n = sig**2*dt_d + n*delta**2
                pdf  += p_n * snorm.pdf(ret, mu_n, np.sqrt(var_n))
            return -np.sum(np.log(np.maximum(pdf, 1e-300)))
        s0  = np.std(ret)/np.sqrt(dt_d)
        m0  = np.mean(ret)/dt_d + 0.5*s0**2
        x0  = [m0, s0, 28., -0.004, 0.027]
        bds = [(None,None),(1e-4,None),(1e-4,None),(None,None),(1e-4,None)]
        res = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bds,
                       options=dict(maxiter=300, ftol=1e-9))
        mu_i, sig_i, lam_i, m_i, d_i = res.x
        sJ2[i] = lam_i*(np.exp(2*m_i+2*d_i**2) - 2*np.exp(m_i+0.5*d_i**2) + 1)
    return np.diag(sJ2)


def tikhonov_inverse(Sigma_eff, rho, eta=ETA_TIKHO):
    eps = eta / D * np.trace(Sigma_eff)
    return np.linalg.solve(Sigma_eff + eps*np.eye(D), rho)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PARAMETRIC MODEL — CARA with clipped-quadratic TC
# ─────────────────────────────────────────────────────────────────────────────

class CARAModelTC:
    """
    Multi-asset CARA q-learning with proportional transaction costs.

    Value function (exponential ansatz):
        J^psi(t,x) = -exp(-alpha*x * psi1*exp(-psi2*(T-t)) + psi3*(T-t))

    q-function (clipped-quadratic with per-asset NTZ):
        q^phi(t,x,a,a_prev) =
            -c(t)/2 * ||a - phi1||^2
            - sum_i phi4_i * |a_i - a_prev_i|
            - d*theta/2 * [log(2*pi*theta) + phi2 + phi3*(T-t)]

    Parameters:
        phi1  : (D,)  — learned optimal allocation (wealth-independent in CARA)
        phi2  : scalar — log-variance offset
        phi3  : scalar — log-variance time slope
        phi4  : (D,)  — per-asset NTZ half-width (>= 0)
        psi   : (3,)  — critic parameters
    """
    def __init__(self, d=D, psi3_init=0., alpha=ALPHA_CARA, c=0.001):
        self.d     = d
        self.alpha = alpha
        self.c_tc  = c           # proportional cost rate
        self.psi   = np.array([1., 0., psi3_init])
        self.phi1  = np.zeros(d)
        self.phi2  = 0.
        self.phi3  = 0.
        self.phi4  = np.zeros(d)  # NTZ half-widths

    def _logvar(self, t):
        return np.clip(self.phi2 + self.phi3*(T-t), -20, 20)

    def _c(self, t):
        return np.exp(-self._logvar(t))

    def J(self, t, x):
        psi1, psi2, psi3 = self.psi
        f_t = psi1 * np.exp(np.clip(-psi2*(T-t), -30, 30))
        return -np.exp(-self.alpha * x * f_t + psi3*(T-t))

    def grad_J(self, t, x):
        psi1, psi2, psi3 = self.psi
        e_t   = np.exp(np.clip(-psi2*(T-t), -30, 30))
        J_val = self.J(t, x)
        return np.array([
            J_val * (-self.alpha * x * e_t),
            J_val * ( self.alpha * x * psi1 * (T-t) * e_t),
            J_val * (T-t)
        ])

    def q(self, t, x, a, a_prev):
        c  = self._c(t)
        lv = self._logvar(t)
        r  = a - self.phi1
        ntz_cost = np.sum(self.phi4 * np.abs(a - a_prev))
        return (-c/2*np.dot(r,r)
                - ntz_cost
                - self.d*THETA/2*(np.log(2*np.pi*THETA)+lv))

    def grad_q(self, t, x, a, a_prev):
        c  = self._c(t)
        r  = a - self.phi1
        rr = np.dot(r, r)
        da = a - a_prev
        # grad w.r.t. phi1
        g_phi1 = c * r
        # grad w.r.t. phi2, phi3
        g_phi2 = 0.5*c*rr - self.d*THETA/2
        g_phi3 = (T-t)*g_phi2
        # grad w.r.t. phi4 (per-asset NTZ)
        g_phi4 = -np.abs(da)
        return g_phi1, g_phi2, g_phi3, g_phi4

    def sample_action(self, t, x, a_prev):
        """Sample from clipped Gaussian centred at phi1, respecting NTZ."""
        std_a = max(np.sqrt(THETA*np.exp(np.clip(self._logvar(t),-20,20))), 1e-6)
        a_raw = np.random.normal(self.phi1, std_a)
        # Apply NTZ: stay if within phi4 of previous action
        ntz   = np.maximum(self.phi4, 0.)
        a_clipped = np.where(
            np.abs(a_raw - a_prev) <= ntz,
            a_prev,
            a_raw
        )
        return a_clipped

    def mean_action(self, t, x, a_prev):
        """Deterministic mean action with NTZ clipping."""
        ntz = np.maximum(self.phi4, 0.)
        return np.where(
            np.abs(self.phi1 - a_prev) <= ntz,
            a_prev,
            self.phi1
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class BootstrapSimulator:
    """Bootstrap resampling from historical monthly returns."""
    def __init__(self, monthly_ret):
        self.ret = monthly_ret

    def simulate_trajectory(self, k):
        idx = np.random.randint(0, len(self.ret), size=k)
        return self.ret[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_cara_tc(sim, psi3_init, c_tc, label=''):
    """
    Offline q-learning for CARA with transaction costs.
    Uses Adam for phi1, phi2, phi3, phi4 and SGD for psi.
    """
    model = CARAModelTC(d=D, psi3_init=psi3_init, c=c_tc)

    # Adam state for phi parameters
    m1 = np.zeros(D); v1 = np.zeros(D)   # phi1
    m2 = 0.; v2 = 0.                      # phi2
    m3 = 0.; v3 = 0.                      # phi3
    m4 = np.zeros(D); v4 = np.zeros(D)   # phi4
    adam_t = 0

    hist_phi1 = np.zeros((N_TRAIN, D))
    hist_phi4 = np.zeros((N_TRAIN, D))

    for j in range(1, N_TRAIN+1):
        lr     = j**(-DECAY)
        d_psi  = np.zeros(3)
        d_phi1 = np.zeros(D); d_phi2 = 0.; d_phi3 = 0.; d_phi4 = np.zeros(D)

        for _ in range(BATCH):
            rets = sim.simulate_trajectory(K)
            xs   = np.zeros(K+1)
            acts = np.zeros((K, D))
            xs[0] = X0
            a_prev = model.phi1.copy()   # start from current mean

            for k in range(K):
                acts[k] = model.sample_action(k*DT, xs[k], a_prev)
                r_excess = np.exp(rets[k]) - 1 - R_F*DT
                # Wealth update including TC cost
                tc_cost  = c_tc * np.sum(np.abs(acts[k] - a_prev))
                xn = xs[k] + np.dot(acts[k], r_excess) - tc_cost
                xs[k+1] = xn if np.isfinite(xn) else xs[k]
                a_prev   = acts[k].copy()

            a_prev = model.phi1.copy()
            for k in range(K):
                td = (model.J((k+1)*DT, xs[k+1])
                      - model.J(k*DT, xs[k])
                      - model.q(k*DT, xs[k], acts[k], a_prev)*DT)
                if np.isfinite(td):
                    d_psi  += model.grad_J(k*DT, xs[k]) * td / K
                    gp1, gp2, gp3, gp4 = model.grad_q(
                        k*DT, xs[k], acts[k], a_prev)
                    d_phi1 += gp1 * td / K
                    d_phi2 += gp2 * td / K
                    d_phi3 += gp3 * td / K
                    d_phi4 += gp4 * td / K
                a_prev = acts[k].copy()

        # Clip gradients
        for g in [d_psi, d_phi1, d_phi4]:
            n = np.linalg.norm(g)
            if n > GRAD_CLIP * BATCH:
                g *= GRAD_CLIP * BATCH / n

        # SGD for psi (critic)
        model.psi += lr * 0.001 * d_psi / BATCH

        # Adam for phi1, phi2, phi3, phi4
        adam_t += 1
        bc1 = 1 - ADAM_B1**adam_t
        bc2 = 1 - ADAM_B2**adam_t

        def adam_update(param, grad, m, v, lr_scale=1.0):
            m[:] = ADAM_B1*m + (1-ADAM_B1)*grad
            v[:] = ADAM_B2*v + (1-ADAM_B2)*grad**2
            return param + lr_scale * ADAM_LR * (m/bc1) / (np.sqrt(v/bc2) + ADAM_EPS)

        def adam_update_scalar(param, grad, m, v, lr_scale=1.0):
            m_new = ADAM_B1*m + (1-ADAM_B1)*grad
            v_new = ADAM_B2*v + (1-ADAM_B2)*grad**2
            p_new = param + lr_scale * ADAM_LR * (m_new/bc1) / (np.sqrt(v_new/bc2) + ADAM_EPS)
            return p_new, m_new, v_new

        g1 = d_phi1 / BATCH
        g2 = d_phi2 / BATCH
        g3 = d_phi3 / BATCH
        g4 = d_phi4 / BATCH

        model.phi1 = adam_update(model.phi1, g1, m1, v1)
        model.phi2, m2, v2 = adam_update_scalar(model.phi2, g2, m2, v2)
        model.phi3, m3, v3 = adam_update_scalar(model.phi3, g3, m3, v3)
        # phi4 must stay >= 0
        phi4_new = adam_update(model.phi4, g4, m4, v4, lr_scale=0.5)
        model.phi4 = np.maximum(phi4_new, 0.)

        model.phi2 = np.clip(model.phi2, -10, 10)
        model.phi3 = np.clip(model.phi3, -10, 10)

        hist_phi1[j-1] = model.phi1
        hist_phi4[j-1] = model.phi4

        if j % 2000 == 0:
            phi1_str = ' '.join(f'{v:.3f}' for v in model.phi1)
            phi4_str = ' '.join(f'{v:.4f}' for v in model.phi4)
            print(f"  [{label}] iter {j:5d}/{N_TRAIN}  "
                  f"phi1=[{phi1_str}]  phi4=[{phi4_str}]")

    return model, hist_phi1, hist_phi4


# ─────────────────────────────────────────────────────────────────────────────
# 7.  OOS EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def oos_eval(model, monthly_test, c_tc):
    """Chronological OOS evaluation with TC deducted from wealth."""
    x = X0
    wealth = [x]
    turnover = []
    port_ret = []
    a_prev = model.phi1.copy()

    for k, r in enumerate(monthly_test):
        t = (k % K) * DT
        a = model.mean_action(t, x, a_prev)
        r_excess = np.exp(r) - 1 - R_F*DT
        tc_cost  = c_tc * np.sum(np.abs(a - a_prev))
        dx       = np.dot(a, r_excess) - tc_cost
        port_ret.append(dx / max(x, 1e-8))
        turnover.append(np.sum(np.abs(a - a_prev)))
        x        = x + dx
        wealth.append(x)
        a_prev   = a.copy()

    wealth   = np.array(wealth)
    port_ret = np.array(port_ret)
    ann_ret  = port_ret.mean() * 12
    ann_vol  = port_ret.std(ddof=1) * np.sqrt(12)
    sharpe   = (ann_ret - R_F) / ann_vol if ann_vol > 1e-8 else np.nan
    peak     = np.maximum.accumulate(wealth)
    maxdd    = float(np.min(wealth / peak) - 1)
    to_mean  = float(np.mean(turnover))

    return dict(XT=float(wealth[-1]), ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, maxdd=maxdd, turnover=to_mean, wealth=wealth)


def oos_buyhold(monthly_test, a_fixed=None, c_tc=0.):
    """Buy-and-hold with equal allocation and TC."""
    if a_fixed is None:
        a_fixed = np.full(D, 0.4/D)
    x = X0; wealth = [x]; port_ret = []; turnover = []
    a_prev = np.zeros(D)
    for k, r in enumerate(monthly_test):
        r_excess = np.exp(r) - 1 - R_F*DT
        tc_cost  = c_tc * np.sum(np.abs(a_fixed - a_prev))
        dx       = np.dot(a_fixed, r_excess) - tc_cost
        port_ret.append(dx / max(x, 1e-8))
        turnover.append(np.sum(np.abs(a_fixed - a_prev)))
        x = x + dx; wealth.append(x)
        a_prev = a_fixed.copy()
    wealth   = np.array(wealth)
    port_ret = np.array(port_ret)
    ann_ret  = port_ret.mean() * 12
    ann_vol  = port_ret.std(ddof=1) * np.sqrt(12)
    sharpe   = (ann_ret - R_F) / ann_vol if ann_vol > 1e-8 else np.nan
    peak     = np.maximum.accumulate(wealth)
    maxdd    = float(np.min(wealth / peak) - 1)
    return dict(XT=float(wealth[-1]), ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, maxdd=maxdd,
                turnover=float(np.mean(turnover)), wealth=wealth)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 65)
    print("  CARA Portfolio with Transaction Costs — Multi-Asset (d=5)")
    print("=" * 65)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1] Loading data (2000-2023)...")
    daily_train, monthly_train, monthly_test = load_data()

    # ── Calibration ──────────────────────────────────────────────────────────
    print("\n[2] Calibrating parameters...")
    bs   = calibrate_bs(daily_train)
    SigmaJ_diag = calibrate_mjd_marginal(daily_train)

    Sigma_eff_diag = bs['Sigma'] + SigmaJ_diag
    phi1_star_diag = tikhonov_inverse(Sigma_eff_diag, bs['rho']) / ALPHA_CARA

    # Threshold estimator for full SigmaJ (same as mv_multiasset_cara.py)
    try:
        dt_d   = 1/252
        thresh = 10 * np.std(daily_train, axis=0) * np.sqrt(dt_d * np.log(1/dt_d))
        mask   = np.all(np.abs(daily_train) > thresh[None, :], axis=1)
        SigmaJ_full = (daily_train[mask].T @ daily_train[mask]) / (dt_d * daily_train.shape[0])
        eigv, eigvec = np.linalg.eigh(SigmaJ_full)
        SigmaJ_full  = eigvec @ np.diag(np.maximum(eigv, 0)) @ eigvec.T
        Sigma_bs_adj = bs['Sigma'] - SigmaJ_full
        Sigma_eff_full = Sigma_bs_adj + SigmaJ_full
        phi1_star_full = tikhonov_inverse(Sigma_eff_full, bs['rho']) / ALPHA_CARA
    except Exception:
        print("  Threshold estimator failed, using diagonal as fallback")
        SigmaJ_full    = SigmaJ_diag
        phi1_star_full = phi1_star_diag

    psi3_init = float(bs['rho'] @ np.linalg.solve(Sigma_eff_diag, bs['rho']))

    print(f"\n  phi1* (diagonal SigmaJ):  "
          + ' '.join(f'{v:.3f}' for v in phi1_star_diag))
    print(f"  phi1* (full SigmaJ):      "
          + ' '.join(f'{v:.3f}' for v in phi1_star_full))

    # ── Training: sweep over TC levels ───────────────────────────────────────
    sim = BootstrapSimulator(monthly_train)
    results = {}

    for c_tc in TC_LEVELS:
        label = f'c={c_tc:.3f}'
        print(f"\n[3] Training CARA-TC ({label}, diagonal SigmaJ)...")
        model, h1, h4 = train_cara_tc(sim, psi3_init, c_tc, label=label)
        results[c_tc] = dict(model=model, hist_phi1=h1, hist_phi4=h4)
        print(f"  Converged phi1: " + ' '.join(f'{v:.3f}' for v in model.phi1))
        print(f"  Converged phi4: " + ' '.join(f'{v:.4f}' for v in model.phi4))

    # ── OOS Evaluation ───────────────────────────────────────────────────────
    print("\n[4] Out-of-sample evaluation 2020-2023...")
    print("\n" + "="*75)
    print("  CARA-TC OOS 2020-2023 (diagonal SigmaJ)")
    print("="*75)
    print(f"  {'c':>6} {'phi1[0]':>8} {'X_T':>6} {'Ret%':>7} "
          f"{'Vol%':>7} {'Sharpe':>7} {'MaxDD':>8} {'Turnover':>9}")
    print("  " + "-"*73)

    bh = oos_buyhold(monthly_test, c_tc=0.)
    print(f"  {'B&H':>6} {'--':>8} {bh['XT']:>6.3f} "
          f"{bh['ann_ret']*100:>7.2f} {bh['ann_vol']*100:>7.2f} "
          f"{bh['sharpe']:>7.3f} {bh['maxdd']*100:>7.2f}% "
          f"{bh['turnover']:>9.3f}")

    oos_results = {}
    for c_tc in TC_LEVELS:
        m = results[c_tc]['model']
        s = oos_eval(m, monthly_test, c_tc)
        oos_results[c_tc] = s
        print(f"  {c_tc:>6.3f} {m.phi1[0]:>8.3f} {s['XT']:>6.3f} "
              f"{s['ann_ret']*100:>7.2f} {s['ann_vol']*100:>7.2f} "
              f"{s['sharpe']:>7.3f} {s['maxdd']*100:>7.2f}% "
              f"{s['turnover']:>9.3f}")
    print("="*75)

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\n[5] Generating figures...")

    COLORS = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']

    # Figure 1: phi1[0] convergence for each TC level
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    iters = np.arange(1, N_TRAIN+1)
    w = 300

    ax = axes[0]
    for i, c_tc in enumerate(TC_LEVELS):
        h1 = results[c_tc]['hist_phi1'][:, 0]
        smoothed = np.convolve(h1, np.ones(w)/w, mode='valid')
        ax.plot(iters[w-1:], smoothed, color=COLORS[i], lw=2,
                label=f'c={c_tc:.3f}')
    ax.axhline(phi1_star_diag[0], color='black', lw=1.2, ls='--',
               label=f'Oracle phi1*[S&P]={phi1_star_diag[0]:.3f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\phi_{1,1}$ (S\&P 500 component)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, c_tc in enumerate(TC_LEVELS):
        h4 = results[c_tc]['hist_phi4'][:, 0]
        smoothed = np.convolve(h4, np.ones(w)/w, mode='valid')
        ax.plot(iters[w-1:], smoothed, color=COLORS[i], lw=2,
                label=f'c={c_tc:.3f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\phi_{4,1}$ (NTZ half-width, S\&P 500)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(f'figure_cara_tc_convergence.{ext}', dpi=150,
                    bbox_inches='tight')
    plt.close()
    print("  Saved: figure_cara_tc_convergence.pdf/png")

    # Figure 2: OOS wealth paths
    months = np.arange(len(monthly_test) + 1)
    year_ticks  = [0, 12, 24, 36, 48]
    year_labels = ['Jan 2020', 'Jan 2021', 'Jan 2022', 'Jan 2023', 'Jan 2024']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(months, bh['wealth'], color='gray', lw=1.5, ls=':',
            label='Buy-and-hold')
    for i, c_tc in enumerate(TC_LEVELS):
        s = oos_results[c_tc]
        ax.plot(months, s['wealth'], color=COLORS[i], lw=2.0,
                label=f'CARA-TC c={c_tc:.3f} (X_T={s["XT"]:.3f})')
    ax.axvspan(1, 4,   alpha=0.08, color='red')
    ax.axvspan(24, 36, alpha=0.08, color='blue')
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, fontsize=9)
    ax.set_xlabel('Month'); ax.set_ylabel(r'Wealth $X_t$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(f'figure_cara_tc_oos.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figure_cara_tc_oos.pdf/png")

    print("\nDone.")
