# simulate_topology_models.py
#
# Simulate three minimal transition topologies, estimate empirical cumulative
# distribution and hazard functions, fit delayed Weibull curves, and save the
# resulting figures and summary files in a local "topologyModelFigures" folder.
#
# Previously, the dashed curves in the CDF and hazard figures plotted the
# analytic Markov solution (F_theory / haz_theory), NOT the fitted delayed
# Weibull curves, despite the figure legend claiming otherwise. The Weibull
# fit was computed (lam_hat, k_hat) and written to the summary, but never drawn.
#
# This version corrects that: the dashed curves are now the fitted Weibull CDF
# and Weibull hazard, evaluated at (lam_hat, k_hat). The solid curves remain the
# stochastic-simulation estimates (the ground truth), so the comparison shown is
# now genuinely "simulation vs Weibull fit".
#
# The exact analytic Markov solution is retained and can be overlaid as a thin
# dotted line by setting SHOW_EXACT_THEORY = True. Note that for the decreasing 
# hazard (k<1) topology, the exact first-passage-time distribution is a finite 
# sum of exponentials whose tail behaviour differs fundamentally from a Weibull; 
# the divergence becomes visible when the time axis is extended (see 
# EXTEND_DECREASING_TMAX below).
#
# The increasing-hazard analytic curve is now the EXACT hypoexponential
# (convolution of the four exponential steps), replacing the previous gamma
# moment-matched approximation.
# ---------------------------------------------------------------------------

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import gamma


# ---------------------------
# Revision switches
# ---------------------------
# Overlay the exact analytic Markov solution (thin dotted) alongside the Weibull
# fit. Off by default so the main-text figure shows simulation vs Weibull fit
# only (matching the two-curve legend). Turn on to generate the diagnostic /
# supplemental version that also shows the exact solution.
SHOW_EXACT_THEORY = False

# Optionally extend the time axis for the decreasing-hazard panel only, to make
# the long-time Weibull-vs-exact divergence visible. Set to None to keep the
# common range. Example: 60.0
EXTEND_DECREASING_TMAX = None


# ---------------------------
# Plot style
# ---------------------------
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1.0


# ---------------------------
# General helpers
# ---------------------------
def ensure_dir(path):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def lighten_color(rgb, factor=0.88):
    """Lighten an RGB colour by mixing it toward white."""
    rgb = np.array(rgb, dtype=float)
    return tuple(1.0 - factor * (1.0 - rgb))


def darken_color(rgb, factor=0.62):
    """Darken an RGB colour by scaling it toward black."""
    rgb = np.array(rgb, dtype=float)
    return tuple(factor * rgb)


def weibull_cdf(t, lam, k):
    """Standard Weibull cumulative distribution function."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = t >= 0
    out[mask] = 1.0 - np.exp(-((t[mask] / lam) ** k))
    return out


def weibull_hazard(t, lam, k):
    """Standard Weibull hazard function h(t) = (k/lam) (t/lam)^(k-1)."""
    t = np.asarray(t, dtype=float)
    out = np.full_like(t, np.nan, dtype=float)
    mask = t > 0
    out[mask] = (k / lam) * (t[mask] / lam) ** (k - 1.0)
    return out


def empirical_cdf(times, t_grid):
    """Compute the empirical CDF of first-passage times on a fixed grid."""
    times = np.asarray(times)
    return np.searchsorted(np.sort(times), t_grid, side="right") / len(times)


def empirical_cdf_with_ci(times, t_grid, alpha=0.05):
    """
    Compute the empirical CDF with pointwise normal-approximation confidence bands.
    """
    F_emp = empirical_cdf(times, t_grid)
    n = len(times)

    se = np.sqrt(F_emp * (1.0 - F_emp) / n)
    z = 1.96  # 95% interval

    F_lo = np.clip(F_emp - z * se, 0.0, 1.0)
    F_hi = np.clip(F_emp + z * se, 0.0, 1.0)

    return F_emp, F_lo, F_hi


def gaussian_kernel(size, sigma):
    """Return a normalized one-dimensional Gaussian kernel."""
    x = np.arange(size) - (size - 1) / 2
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= np.sum(g)
    return g


def smooth_1d(y, sigma=3):
    """Smooth a one-dimensional array by Gaussian convolution."""
    size = int(np.ceil(6 * sigma)) | 1
    kernel = gaussian_kernel(size, sigma)
    return np.convolve(y, kernel, mode="same")


def fit_weibull_to_cdf(times, t_grid=None):
    """Fit a Weibull CDF to an empirical CDF by nonlinear least squares."""
    times = np.asarray(times)
    if t_grid is None:
        t_grid = np.linspace(0, np.percentile(times, 99.5), 400)

    F_emp = empirical_cdf(times, t_grid)
    mask = (F_emp > 0.01) & (F_emp < 0.99)
    t_fit = t_grid[mask]
    F_fit = F_emp[mask]

    lam0 = np.median(times)
    k0 = 1.0

    popt, _ = curve_fit(
        weibull_cdf,
        t_fit,
        F_fit,
        p0=[lam0, k0],
        bounds=([1e-6, 1e-3], [np.inf, 20.0]),
        maxfev=30000,
    )
    return popt[0], popt[1]


def estimate_hazard_with_ci(times, t_max, n_bins=90, smooth_sigma=6.0, min_risk=5000):
    """
    Estimate a piecewise hazard function with smoothed confidence bands.

    The estimate is truncated once the risk set falls below min_risk.
    """
    times = np.asarray(times)
    edges = np.linspace(0, t_max, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dt = edges[1] - edges[0]

    counts, _ = np.histogram(times, bins=edges)

    n = len(times)
    risk = np.empty(n_bins, dtype=float)
    survived = n
    for i in range(n_bins):
        risk[i] = survived
        survived -= counts[i]

    with np.errstate(divide="ignore", invalid="ignore"):
        haz_raw = counts / (risk * dt)
        var_raw = counts / (risk**2 * dt**2)

    haz_raw[~np.isfinite(haz_raw)] = 0.0
    var_raw[~np.isfinite(var_raw)] = 0.0

    se_raw = np.sqrt(var_raw)
    lo_raw = np.clip(haz_raw - 1.96 * se_raw, 0, None)
    hi_raw = haz_raw + 1.96 * se_raw

    haz = smooth_1d(haz_raw, sigma=smooth_sigma)
    lo = smooth_1d(lo_raw, sigma=smooth_sigma)
    hi = smooth_1d(hi_raw, sigma=smooth_sigma)

    valid = risk >= min_risk
    if np.any(valid):
        last_valid = np.where(valid)[0][-1] + 1
    else:
        last_valid = len(centres)

    return {
        "t": centres[:last_valid],
        "haz": haz[:last_valid],
        "lo": lo[:last_valid],
        "hi": hi[:last_valid],
        "risk": risk[:last_valid],
        "counts": counts[:last_valid],
        "dt": dt,
    }


# ---------------------------
# Stochastic models
# ---------------------------
def simulate_constant_hazard(n, r=1.0, rng=None):
    """Simulate first-passage times for the one-step topology S -> T."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.exponential(scale=1.0 / r, size=n)


def simulate_increasing_hazard(n, rates=(0.7, 0.9, 1.1, 1.3), rng=None):
    """Simulate first-passage times for the sequential topology S0 -> S1 -> S2 -> S3 -> T."""
    if rng is None:
        rng = np.random.default_rng()
    x1 = rng.exponential(scale=1.0 / rates[0], size=n)
    x2 = rng.exponential(scale=1.0 / rates[1], size=n)
    x3 = rng.exponential(scale=1.0 / rates[2], size=n)
    x4 = rng.exponential(scale=1.0 / rates[3], size=n)
    return x1 + x2 + x3 + x4


def simulate_decreasing_hazard(n, r_transition=2.2, r_adapt=1.6, r_refractory=0.025, rng=None):
    """
    Simulate first-passage times for the competing-pathway topology
    S -> T, S -> R, R -> T.
    """
    if rng is None:
        rng = np.random.default_rng()

    t_to_transition = rng.exponential(scale=1.0 / r_transition, size=n)
    t_to_adapt = rng.exponential(scale=1.0 / r_adapt, size=n)

    adapted_first = t_to_adapt < t_to_transition
    times = np.empty(n, dtype=float)

    times[~adapted_first] = t_to_transition[~adapted_first]
    t_from_R = rng.exponential(scale=1.0 / r_refractory, size=np.sum(adapted_first))
    times[adapted_first] = t_to_adapt[adapted_first] + t_from_R

    return times


# ---------------------------
# Theory curves (exact analytic Markov solutions)
# ---------------------------
def theoretical_constant_cdf(t, r):
    """Exact CDF for the one-step topology."""
    return 1.0 - np.exp(-r * t)


def theoretical_constant_hazard(t, r):
    """Exact hazard for the one-step topology."""
    return np.full_like(t, r, dtype=float)


def _hypoexponential_coeffs(rates):
    """
    Partial-fraction coefficients for a hypoexponential (generalized Erlang)
    distribution with DISTINCT rates. Returns c_i such that
        S(t) = sum_i c_i exp(-rate_i t),  with sum_i c_i = 1.
    """
    rates = np.asarray(rates, dtype=float)
    n = len(rates)
    c = np.empty(n, dtype=float)
    for i in range(n):
        prod = 1.0
        for j in range(n):
            if j != i:
                prod *= rates[j] / (rates[j] - rates[i])
        c[i] = prod
    return rates, c


def theoretical_increasing_cdf(t, rates):
    """
    Exact CDF for the sequential topology (hypoexponential law), valid for
    distinct rates. Falls back to an Erlang if all rates are equal.
    """
    t = np.asarray(t, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)
    if np.allclose(rates_arr, rates_arr[0]):
        return gamma.cdf(t, a=len(rates_arr), scale=1.0 / rates_arr[0])

    r, c = _hypoexponential_coeffs(rates_arr)
    S = np.zeros_like(t)
    for ci, ri in zip(c, r):
        S += ci * np.exp(-ri * t)
    return np.clip(1.0 - S, 0.0, 1.0)


def theoretical_increasing_hazard(t, rates):
    """
    Exact hazard for the sequential topology (hypoexponential law), valid for
    distinct rates. Falls back to an Erlang if all rates are equal.
    """
    t = np.asarray(t, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)
    if np.allclose(rates_arr, rates_arr[0]):
        pdf = gamma.pdf(t, a=len(rates_arr), scale=1.0 / rates_arr[0])
        sf = gamma.sf(t, a=len(rates_arr), scale=1.0 / rates_arr[0])
    else:
        r, c = _hypoexponential_coeffs(rates_arr)
        sf = np.zeros_like(t)
        pdf = np.zeros_like(t)
        for ci, ri in zip(c, r):
            e = np.exp(-ri * t)
            sf += ci * e
            pdf += ci * ri * e

    with np.errstate(divide="ignore", invalid="ignore"):
        haz = pdf / sf
    haz[~np.isfinite(haz)] = np.nan
    return haz


def theoretical_decreasing_cdf(t, r_transition, r_adapt, r_refractory):
    """Exact CDF for the competing-pathway topology."""
    a = r_transition
    b = r_adapt
    c = r_refractory
    denom = (a + b) - c

    if np.isclose(denom, 0):
        F = 1.0 - np.exp(-(a + b) * t) * (1 + b * t)
        return np.clip(F, 0, 1)

    term1 = 1.0 - np.exp(-(a + b) * t)
    term2 = (b / denom) * (
        (1.0 - np.exp(-c * t)) - (c / (a + b)) * (1.0 - np.exp(-(a + b) * t))
    )
    F = (a / (a + b)) * term1 + term2
    return np.clip(F, 0, 1)


def theoretical_decreasing_hazard(t, r_transition, r_adapt, r_refractory):
    """Exact hazard for the competing-pathway topology."""
    a = r_transition
    b = r_adapt
    c = r_refractory
    denom = (a + b) - c

    if np.isclose(denom, 0):
        return np.full_like(t, np.nan, dtype=float)

    pdf = a * np.exp(-(a + b) * t) + (b * c / denom) * (
        np.exp(-c * t) - np.exp(-(a + b) * t)
    )
    cdf = theoretical_decreasing_cdf(t, a, b, c)
    sf = 1.0 - cdf

    with np.errstate(divide="ignore", invalid="ignore"):
        haz = pdf / sf
    haz[~np.isfinite(haz)] = np.nan
    return haz


# ---------------------------
# Plotting
# ---------------------------
BASE_COLOURS = {
    "Constant hazard (S→T)": (0.0000, 0.4470, 0.7410),
    "Increasing hazard (S0→S1→S2→S3→T)": (0.8500, 0.3250, 0.0980),
    "Decreasing hazard (S→R→T)": (0.4660, 0.6740, 0.1880),
}


def _curve_legend_handles():
    """Neutral proxy handles explaining the solid/dashed(/dotted) curve meaning."""
    handles = [
        Line2D([0], [0], color="0.35", lw=2.2, linestyle="-", label="Stochastic simulation"),
        Line2D([0], [0], color="0.10", lw=3.0, linestyle="--", label="Delayed Weibull fit"),
    ]
    if SHOW_EXACT_THEORY:
        handles.append(
            Line2D([0], [0], color="0.10", lw=1.4, linestyle=":", label="Exact Markov solution")
        )
    return handles


def make_combined_cdf_figure(results, output_dir):
    """Plot empirical CDF (solid) and fitted Weibull CDF (dashed) for all topologies."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    x_max = 0.0
    for label, res in results.items():
        emp_col = lighten_color(BASE_COLOURS[label], factor=0.90)
        fit_col = darken_color(BASE_COLOURS[label], factor=0.62)

        ax.fill_between(
            res["t_grid"],
            res["F_lo"],
            res["F_hi"],
            color=emp_col,
            alpha=0.16,
            linewidth=0,
        )
        # Solid: stochastic-simulation estimate (ground truth)
        ax.plot(res["t_grid"], res["F_emp"], linewidth=2.2, color=emp_col, label=label)
        # Dashed: FITTED delayed Weibull curve (the actual fit)
        ax.plot(res["t_grid"], res["F_weibull"], linewidth=3.0, linestyle="--", color=fit_col)
        # Optional dotted: exact analytic Markov solution
        if SHOW_EXACT_THEORY:
            ax.plot(res["t_grid"], res["F_theory"], linewidth=1.4, linestyle=":", color=fit_col)
        x_max = max(x_max, res["t_max"])

    ax.grid(True)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    topo_handles, topo_labels = ax.get_legend_handles_labels()
    leg1 = ax.legend(topo_handles, topo_labels, frameon=False, loc="lower right", fontsize=11)
    ax.add_artist(leg1)
    ax.legend(handles=_curve_legend_handles(), frameon=False, loc="center right", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "topologyModel_CDFs.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "topologyModel_CDFs.svg"), bbox_inches="tight")
    plt.close(fig)


def make_combined_hazard_figure(results, output_dir):
    """Plot empirical hazard (solid) and fitted Weibull hazard (dashed) for all topologies."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    y_max = 0.0
    for label, res in results.items():
        emp_col = lighten_color(BASE_COLOURS[label], factor=0.90)
        fit_col = darken_color(BASE_COLOURS[label], factor=0.62)

        ax.fill_between(res["haz_t"], res["haz_lo"], res["haz_hi"], color=emp_col, alpha=0.16, linewidth=0)
        # Solid: stochastic-simulation hazard estimate
        ax.plot(res["haz_t"], res["haz_emp"], linewidth=2.0, color=emp_col, label=label)
        # Dashed: FITTED delayed Weibull hazard
        ax.plot(res["haz_t"], res["haz_weibull"], linewidth=3.0, linestyle="--", color=fit_col)
        # Optional dotted: exact analytic Markov hazard
        if SHOW_EXACT_THEORY:
            ax.plot(res["haz_t"], res["haz_theory"], linewidth=1.4, linestyle=":", color=fit_col)

        # y-scale from the (bounded) empirical hazards, so the k<1 Weibull spike
        # near t=0 does not dominate the axis.
        finite_emp = res["haz_emp"][np.isfinite(res["haz_emp"])]
        if finite_emp.size:
            y_max = max(y_max, np.nanmax(finite_emp))

    x_max = min(res["haz_t"][-1] for res in results.values())

    ax.grid(True)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.15 * y_max if y_max > 0 else None)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    topo_handles, topo_labels = ax.get_legend_handles_labels()
    leg1 = ax.legend(topo_handles, topo_labels, frameon=False, loc="upper right", fontsize=11)
    ax.add_artist(leg1)
    ax.legend(handles=_curve_legend_handles(), frameon=False, loc="center right", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "topologyModel_hazards.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "topologyModel_hazards.svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------
def main():
    """Run stochastic simulations, fit Weibull curves, and save figures and summaries."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "topologyModelFigures")
    ensure_dir(output_dir)

    rng = np.random.default_rng(42)

    n_sim = 100_000_000
    t_max = 10.0

    # Model parameters
    r_const = 1.0
    rates_inc = (0.7, 0.9, 1.1, 1.3)
    r_tr = 2.2
    r_ad = 1.6
    r_ref = 0.025

    # Simulate first-passage times
    times_constant = simulate_constant_hazard(n=n_sim, r=r_const, rng=rng)
    times_increasing = simulate_increasing_hazard(n=n_sim, rates=rates_inc, rng=rng)
    times_decreasing = simulate_decreasing_hazard(
        n=n_sim,
        r_transition=r_tr,
        r_adapt=r_ad,
        r_refractory=r_ref,
        rng=rng,
    )

    panel_times = {
        "Constant hazard (S→T)": times_constant,
        "Increasing hazard (S0→S1→S2→S3→T)": times_increasing,
        "Decreasing hazard (S→R→T)": times_decreasing,
    }

    results = {}

    for label, times in panel_times.items():
        # Per-panel plotting range (decreasing panel may be extended to expose
        # the long-time Weibull-vs-exact divergence).
        panel_tmax = t_max
        if label == "Decreasing hazard (S→R→T)" and EXTEND_DECREASING_TMAX is not None:
            panel_tmax = float(EXTEND_DECREASING_TMAX)

        t_grid = np.linspace(0, panel_tmax, 800)
        F_emp, F_lo, F_hi = empirical_cdf_with_ci(times, t_grid)

        # Fit the delayed Weibull (t0 = 0, pi = 1 for these topologies) on a
        # grid restricted to the common range, matching the reported fits.
        lam_hat, k_hat = fit_weibull_to_cdf(times, t_grid=np.linspace(0, t_max, 800))

        # Fitted Weibull curve — THIS is what should be plotted as the fit.
        F_weibull = weibull_cdf(t_grid, lam_hat, k_hat)

        # Exact analytic Markov solution (optional overlay only).
        if label == "Constant hazard (S→T)":
            F_theory = theoretical_constant_cdf(t_grid, r_const)
            hz_theory_full = theoretical_constant_hazard(t_grid, r_const)
        elif label == "Increasing hazard (S0→S1→S2→S3→T)":
            F_theory = theoretical_increasing_cdf(t_grid, rates_inc)
            hz_theory_full = theoretical_increasing_hazard(t_grid, rates_inc)
        else:
            F_theory = theoretical_decreasing_cdf(t_grid, r_tr, r_ad, r_ref)
            hz_theory_full = theoretical_decreasing_hazard(t_grid, r_tr, r_ad, r_ref)

        haz_est = estimate_hazard_with_ci(
            times,
            t_max=panel_tmax,
            n_bins=501,
            smooth_sigma=6.0,
            min_risk=0,
        )

        hz_theory = np.interp(haz_est["t"], t_grid, hz_theory_full)
        # Fitted Weibull hazard on the hazard grid — the dashed hazard curve.
        haz_weibull = weibull_hazard(haz_est["t"], lam_hat, k_hat)

        results[label] = {
            "times": times,
            "t_grid": t_grid,
            "F_emp": F_emp,
            "F_lo": F_lo,
            "F_hi": F_hi,
            "F_weibull": F_weibull,
            "F_theory": F_theory,
            "lam_hat": lam_hat,
            "k_hat": k_hat,
            "haz_t": haz_est["t"],
            "haz_emp": haz_est["haz"],
            "haz_lo": haz_est["lo"],
            "haz_hi": haz_est["hi"],
            "haz_weibull": haz_weibull,
            "haz_theory": hz_theory,
            "t_max": panel_tmax,
        }

    make_combined_cdf_figure(results, output_dir)
    make_combined_hazard_figure(results, output_dir)

    summary_path = os.path.join(output_dir, "topologyModel_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Topology model simulation summary\n")
        f.write("================================\n\n")
        f.write(f"Number of simulated trajectories per model: {n_sim}\n")
        f.write(f"Common plotting range: t in [0, {t_max}]\n")
        f.write(f"SHOW_EXACT_THEORY = {SHOW_EXACT_THEORY}; EXTEND_DECREASING_TMAX = {EXTEND_DECREASING_TMAX}\n\n")

        f.write("Model definitions used\n")
        f.write("----------------------\n")
        f.write(f"Constant hazard: S -> T with rate r = {r_const}\n")
        f.write(f"Increasing hazard: S0 -> S1 -> S2 -> S3 -> T with rates = {rates_inc}\n")
        f.write(f"Decreasing hazard: S -> T with rate {r_tr}, S -> R with rate {r_ad}, R -> T with rate {r_ref}\n\n")

        f.write("Curves plotted\n")
        f.write("--------------\n")
        f.write("  Solid  : stochastic-simulation estimate (empirical CDF / hazard)\n")
        f.write("  Dashed : FITTED delayed Weibull curve, evaluated at (lambda_hat, k_hat)\n")
        if SHOW_EXACT_THEORY:
            f.write("  Dotted : exact analytic Markov solution (for comparison)\n")
        f.write("\n")

        f.write("Empirical Weibull fits\n")
        f.write("----------------------\n")
        for label, res in results.items():
            f.write(f"{label}\n")
            f.write(f"  fitted lambda = {res['lam_hat']:.6f}\n")
            f.write(f"  fitted k      = {res['k_hat']:.6f}\n\n")

        f.write("Hazard estimation settings\n")
        f.write("--------------------------\n")
        f.write("  piecewise hazard estimator\n")
        f.write("  95% pointwise approximate confidence bands\n")
        f.write("  501 bins over the plotting range\n")
        f.write("  Gaussian smoothing sigma = 6.0 bins\n")
        f.write("  no risk-set truncation (min_risk = 0)\n\n")

        f.write("Exact analytic solutions (used for optional dotted overlay)\n")
        f.write("----------------------------------------------------------\n")
        f.write("  Constant hazard: exact exponential CDF and hazard\n")
        f.write("  Increasing hazard: exact hypoexponential (convolution of the four steps)\n")
        f.write("  Decreasing hazard: exact CDF and hazard for S->T, S->R, R->T model\n")

    print(f"Saved outputs to: {output_dir}")
    print(f"Summary: {summary_path}")
    for label, res in results.items():
        print(f"{label}: fitted lambda = {res['lam_hat']:.4f}, fitted k = {res['k_hat']:.4f}")


if __name__ == "__main__":
    main()
