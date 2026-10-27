"""Reusable cross-sectional diagnostics; quote labels are not executable portfolios.

Labels: signal close t, raw adjusted open t+1 to t+1+h. Missing future
quotes remain missing; the signal universe never depends on future availability.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from ._validation import boolean_mask


def hac_mean(values, lag=5):
    """Bartlett Newey-West mean SE, preserving missing sessions on the time grid."""
    x = np.asarray(values, float)
    good = np.isfinite(x)
    n = int(good.sum())
    if n < 3:
        return dict(n=n, mean=np.nan, se=np.nan, t=np.nan, p=np.nan,
                    lower=np.nan, upper=np.nan, lag=lag)
    mu = float(x[good].mean())
    z = np.where(good, x-mu, 0.)
    meat = z @ z
    for k in range(1, min(lag, len(x)-1)+1):
        meat += 2*(1-k/(lag+1))*(z[k:] @ z[:-k])
    se = float(np.sqrt(max(meat, 0))/n)
    t = mu/se if se > 0 else np.nan
    return dict(n=n, mean=mu, se=se, t=t, p=float(2*norm.sf(abs(t))),
                lower=mu-1.96*se, upper=mu+1.96*se, lag=lag)


def bh_adjust(pvalues):
    p = np.asarray(pvalues, float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        q = p[order]*len(order)/np.arange(1, len(order)+1)
        out[order] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    return out


def forward_open_returns(open_adj, horizon=5, delay=0):
    if horizon < 1 or delay < 0:
        raise ValueError('horizon >= 1 and delay >= 0 required')
    a = np.asarray(open_adj, float)
    out = np.full(a.shape, np.nan)
    n = len(a)-1-delay-horizon
    if n > 0:
        entry = a[1+delay:1+delay+n]
        exit_ = a[1+delay+horizon:1+delay+horizon+n]
        with np.errstate(divide='ignore', invalid='ignore'):
            out[:n] = np.where((entry > 0) & (exit_ > 0), exit_/entry-1, np.nan)
    return out


def split_masks(dates, horizon, delay=0):
    dates = np.asarray(dates).astype('datetime64[D]')
    bounds = [('train', '2020-01-01', '2023-12-31'),
              ('validation', '2024-01-01', '2024-12-31'),
              ('test', '2025-01-01', '2099-12-31')]
    idx = np.arange(len(dates))
    exit_idx = idx+1+delay+horizon
    masks = {}
    for name, lo, hi in bounds:
        m = (dates >= np.datetime64(lo)) & (dates <= np.datetime64(hi))
        valid = exit_idx < len(dates)
        clipped = np.minimum(exit_idx, len(dates)-1)
        masks[name] = m & valid & (dates[clipped] <= np.datetime64(hi))
    return masks


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def daily_diagnostics(score, label, universe, min_stocks=100):
    """Five groups on the signal universe BEFORE observing label availability."""
    rows = []
    for t in range(len(score)):
        idx = np.flatnonzero(universe[t] & np.isfinite(score[t]))
        row = dict(t=t, universe_n=int(universe[t].sum()), signal_n=len(idx),
                   label_n=0, coverage=np.nan, ic=np.nan, rank_ic=np.nan,
                   spread=np.nan, monotonicity=np.nan)
        row.update({f'q{k}': np.nan for k in range(1, 6)})
        if len(idx) >= min_stocks:
            x, y = score[t, idx], label[t, idx]
            ok = np.isfinite(y)
            row['label_n'] = int(ok.sum())
            row['coverage'] = float(ok.mean())
            if ok.sum() >= min_stocks:
                row['ic'] = corr(x[ok], y[ok])
                row['rank_ic'] = corr(pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank())
            # Stable deterministic tie handling; no future quotes in group formation.
            groups = np.array_split(np.argsort(x, kind='stable'), 5)
            for k, g in enumerate(groups, 1):
                yy = y[g]
                row[f'q{k}'] = float(np.nanmean(yy)) if np.isfinite(yy).any() else np.nan
            row['spread'] = row['q5']-row['q1']
            row['monotonicity'] = corr(np.arange(1, 6), [row[f'q{k}'] for k in range(1,6)])
        rows.append(row)
    return pd.DataFrame(rows)


def residual_rank(score, controls, universe):
    """Same-date cross-sectional rank residual; no fitted future coefficients."""
    out = np.full(score.shape, np.nan)
    for t in range(len(score)):
        good = universe[t] & np.isfinite(score[t])
        for c in controls:
            good &= np.isfinite(c[t])
        idx = np.flatnonzero(good)
        if len(idx) < 100:
            continue
        y = pd.Series(score[t, idx]).rank(pct=True).to_numpy()
        X = np.column_stack([np.ones(len(idx))]+[
            pd.Series(c[t, idx]).rank(pct=True).to_numpy() for c in controls])
        out[t, idx] = y-X@np.linalg.lstsq(X, y, rcond=None)[0]
    return out


def market_regression(returns, benchmark, lag=5):
    """OLS daily intercept and Bartlett HAC; missing trading rows retain their lag."""
    y, b = np.asarray(returns,float), np.asarray(benchmark,float)
    if y.ndim != 1 or b.shape != y.shape:
        raise ValueError('returns and benchmark must be aligned one-dimensional arrays')
    if isinstance(lag,bool) or not np.isfinite(lag) or int(lag)!=lag or lag<0:
        raise ValueError('lag must be a nonnegative integer')
    lag=int(lag)
    ok = np.isfinite(y) & np.isfinite(b)
    X = np.column_stack([np.ones(int(ok.sum())), b[ok]])
    n, rank = int(ok.sum()), int(np.linalg.matrix_rank(X)) if ok.any() else 0
    if n <= 2 or rank < 2:
        return dict(alpha_daily=np.nan,alpha_annual=np.nan,beta=np.nan,
                    alpha_lower=np.nan,alpha_upper=np.nan,alpha_p=np.nan,n=n,rank=rank)
    coef = np.linalg.lstsq(X, y[ok], rcond=None)[0]
    z = np.zeros((len(y),2))
    z[ok] = X*(y[ok]-X@coef)[:, None]
    S = z.T@z
    for k in range(1, min(lag, len(y)-1)+1):
        g = z[k:].T@z[:-k]
        S += (1-k/(lag+1))*(g+g.T)
    bread = np.linalg.pinv(X.T@X)
    se = np.sqrt(np.maximum(np.diag(bread@S@bread), 0))
    return dict(alpha_daily=float(coef[0]), alpha_annual=float(coef[0]*242),
                beta=float(coef[1]), alpha_lower=float((coef[0]-1.96*se[0])*242),
                alpha_upper=float((coef[0]+1.96*se[0])*242),
                alpha_p=float(2*norm.sf(abs(coef[0]/se[0]))) if se[0]>0 else np.nan,
                n=n,rank=rank)


def evaluate_factor(ds, score, universe, horizons=(1, 5, 10, 20)):
    """Evaluate any aligned user-supplied score panel; higher always means better.

    Returns daily observations and purged train/validation/test scorecards.
    This evaluates quote prediction only; use qbt.run for constrained net returns.
    """
    score, universe = np.asarray(score,float), boolean_mask(universe, 'universe')
    if score.shape != (ds.T,ds.N) or universe.shape != score.shape:
        raise ValueError('score and universe must align with Dataset (T,N)')
    frames, stats = [], []
    price = np.asarray(ds.open,float)*np.asarray(ds.cumadj_ff,float)
    for h in horizons:
        frame=daily_diagnostics(score,forward_open_returns(price,h),universe)
        frame['date']=ds.dates.astype(str);frame['horizon']=h
        for name,mask in split_masks(ds.dates,h).items():
            stats.append(dict(split=name,horizon=h,**hac_mean(frame.loc[mask,'rank_ic'],max(h-1,5))))
        frames.append(frame)
    return pd.concat(frames,ignore_index=True),pd.DataFrame(stats)
