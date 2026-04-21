# -*- coding: utf-8 -*-
# cstk_fit.py
# Cell State Transition Kinetics Fitting

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares


# ---------- Core maths ----------

def positive_part(x):
    """Return max(x, 0) elementwise."""
    return np.maximum(x, 0.0)


def weibull_cdf(t, t0, lam, k):
    """
    Single-cell Weibull CDF with onset delay t0:
    F_cell(t) = 1 - exp(- ((t - t0)_+ / lam)^k )
    """
    t = np.asarray(t)
    tau = positive_part(t - t0)
    return 1.0 - np.exp(-(tau / lam) ** k)


def fraction_population(t, t0, lam, k, pi=1.0):
    """Population-level transitioned fraction: F(t) = pi * F_cell(t)."""
    return pi * weibull_cdf(t, t0, lam, k)


# ---------- Results container ----------

@dataclass
class FitResult:
    """Container for model fit outputs."""
    params: dict
    cov: Optional[np.ndarray]
    success: bool
    message: str
    method: str
    t_metrics: dict
    aic: Optional[float]
    bic: Optional[float]
    n: int
    r2: Optional[float]


# ---------- Utilities ----------

def _compute_t_metrics(t0, lam, k):
    """Return t10, t50, and t90 among responders."""
    def tp(p):
        return t0 + lam * (-np.log(1.0 - p)) ** (1.0 / k)

    return {"t10": tp(0.10), "t50": tp(0.50), "t90": tp(0.90)}


def _pack(params):
    """Pack parameter dict into an array."""
    return np.array([params[k] for k in params])


def _unpack(x, keys):
    """Unpack parameter array into a dict with named keys."""
    return {k: float(v) for k, v in zip(keys, x)}


def _bounds(keys):
    """Return parameter bounds for the fraction model."""
    lo, hi = [], []
    for key in keys:
        if key == "t0":
            # Allow slight negative t0 if baseline is pre-induction
            lo.append(-1e2)
            hi.append(1e6)
        elif key == "lam":
            lo.append(1e-6)
            hi.append(1e6)
        elif key == "k":
            lo.append(0.01)
            hi.append(100.0)
        elif key == "pi":
            lo.append(0.0)
            hi.append(1.0)
        else:
            lo.append(-np.inf)
            hi.append(np.inf)
    return (np.array(lo), np.array(hi))


def _init_guess(t, fix_pi=None, fix_t0=None):
    """Crude but stable initial guesses for the fraction model."""
    t = np.asarray(t)
    tmin, tmax = np.nanmin(t), np.nanmax(t)

    t0 = tmin if fix_t0 is None else float(fix_t0)
    lam = max((tmax - tmin) / 3.0, 1e-2)
    k = 1.2
    pi = 1.0 if fix_pi is None else float(fix_pi)

    return {"t0": t0, "lam": lam, "k": k, "pi": pi}


# ---------- Fitting (Gaussian least-squares only) ----------

def fit_weibull(
    t,
    y,
    yerr=None,                  # optional SD/SEM for weighting
    weights=None,               # alternative weights
    robust=True,                # use soft_l1 loss if True
    return_cov=True,
    fix_pi: Optional[float] = None,   # fix pi if provided
    fix_t0: Optional[float] = None    # fix t0 if provided
) -> FitResult:
    """
    Fit the Weibull-based fraction model to pre-normalized data in [0,1].

    Model
    -----
    y(t) ~ F(t) = pi * [1 - exp(-((t - t0)_+ / lam)^k)]

    Args
    ----
    t : array-like
        Time points.
    y : array-like
        Measurements intended to lie in [0,1].
    yerr : array-like, optional
        SD or SEM at each time point, used for weighting as 1 / yerr.
    weights : array-like, optional
        Alternative weights; ignored if yerr is provided.
    robust : bool
        If True, use soft_l1 loss; otherwise use linear loss.
    return_cov : bool
        If True, return a covariance estimate from the Jacobian.
    fix_pi : float or None
        If provided, fix pi to this value.
    fix_t0 : float or None
        If provided, fix t0 to this value.

    Returns
    -------
    FitResult
        Dataclass containing fitted parameters and summary statistics.
    """
    # Convert inputs to arrays
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    # Keep only finite values
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 3:
        raise ValueError("Need at least 3 finite (t, y) points to fit.")
    t = t[mask]
    y = y[mask]

    nobs = len(t)

    # Validate fixed parameters
    if fix_pi is not None:
        if not (0.0 <= float(fix_pi) <= 1.0):
            raise ValueError("fix_pi must be in [0,1].")
    if fix_t0 is not None:
        fix_t0 = float(fix_t0)

    # Build free-parameter list
    keys = ["t0", "lam", "k", "pi"]
    if fix_pi is not None:
        keys = [k for k in keys if k != "pi"]
    if fix_t0 is not None:
        keys = [k for k in keys if k != "t0"]

    # Initial guess and bounds
    guess = _init_guess(t, fix_pi=fix_pi, fix_t0=fix_t0)
    x0 = _pack({k: guess[k] for k in keys})
    bounds = _bounds(keys)

    # Set weights
    if yerr is not None:
        w = 1.0 / np.asarray(yerr, float)
    elif weights is not None:
        w = np.asarray(weights, float)
    else:
        w = np.ones_like(y)

    # Residual function
    def resid(x):
        p = _unpack(x, keys)

        # Add fixed values back into the parameter dict
        if fix_pi is not None:
            p["pi"] = float(fix_pi)
        if fix_t0 is not None:
            p["t0"] = float(fix_t0)

        mu = fraction_population(
            t,
            p.get("t0", guess["t0"]),
            p["lam"],
            p["k"],
            p.get("pi", 1.0),
        )
        return w * (y - mu)

    # Fit by nonlinear least squares
    loss = "soft_l1" if robust else "linear"
    res = least_squares(resid, x0, bounds=bounds, loss=loss)

    xhat = res.x
    success = res.success
    msg = res.message
    method = f"LSQ_{loss}"

    # Covariance estimate from Jacobian (weighted least-squares approximation)
    cov = None
    if return_cov and (res.jac is not None):
        try:
            J = res.jac
            dof = max(nobs - len(xhat), 1)
            s2 = np.sum(res.fun ** 2) / dof
            JTJ_inv = np.linalg.pinv(J.T @ J)
            cov = s2 * JTJ_inv
        except Exception:
            cov = None

    # AIC/BIC using Gaussian RSS surrogate
    rss = np.sum(res.fun ** 2)
    kpar = len(xhat)  # number of free parameters estimated
    aic = nobs * np.log(rss / nobs) + 2 * kpar
    bic = nobs * np.log(rss / nobs) + kpar * np.log(nobs)

    # R^2 on the unweighted fitted values
    params_tmp = _unpack(xhat, keys)
    if fix_pi is not None:
        params_tmp["pi"] = float(fix_pi)
    if fix_t0 is not None:
        params_tmp["t0"] = float(fix_t0)

    yhat = fraction_population(
        t,
        params_tmp.get("t0", guess["t0"]),
        params_tmp["lam"],
        params_tmp["k"],
        params_tmp.get("pi", 1.0),
    )

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.nanmean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    # Final fitted parameters
    params = _unpack(xhat, keys)
    if fix_pi is not None:
        params["pi"] = float(fix_pi)
    if fix_t0 is not None:
        params["t0"] = float(fix_t0)

    # Derived timing metrics
    tmetrics = _compute_t_metrics(params["t0"], params["lam"], params["k"])

    return FitResult(
        params=params,
        cov=cov,
        success=success,
        message=msg,
        method=method,
        t_metrics=tmetrics,
        aic=aic,
        bic=bic,
        n=nobs,
        r2=r2,
    )