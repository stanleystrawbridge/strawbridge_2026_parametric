# simulate_topology_models.py
#
# Updated to:
# - increase simulations substantially
# - use wider bins for hazard estimation
# - smooth more strongly
# - make theoretical curves more visually prominent
# - tune decreasing-hazard model to be more cleanly monotone
# - improve constant-hazard empirical estimate
# - add empirical CDF error bounds
#
# Models:
#   Constant hazard:   S -> T
#   Increasing hazard: S0 -> S1 -> S2 -> S3 -> T
#   Decreasing hazard: S -> T, S -> R, R -> T
#
# Output folder:
#   topologyModelFigures


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gamma


# ---------------------------
# Style
# ---------------------------
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1.0


# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def lighten_color(rgb, factor=0.88):
    rgb = np.array(rgb, dtype=float)
    return tuple(1.0 - factor * (1.0 - rgb))


def darken_color(rgb, factor=0.62):
    rgb = np.array(rgb, dtype=float)
    return tuple(factor * rgb)


def weibull_cdf(t, lam, k):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = t >= 0
    out[mask] = 1.0 - np.exp(-((t[mask] / lam) ** k))
    return out


def empirical_cdf(times, t_grid):
    times = np.asarray(times)
    return np.searchsorted(np.sort(times), t_grid, side="right") / len(times)


def empirical_cdf_with_ci(times, t_grid, alpha=0.05):
    """
    Empirical CDF with simple pointwise binomial 95% confidence intervals.
    """
    F_emp = empirical_cdf(times, t_grid)
    n = len(times)

    se = np.sqrt(F_emp * (1.0 - F_emp) / n)
    z = 1.96  # for 95% CI

    F_lo = np.clip(F_emp - z * se, 0.0, 1.0)
    F_hi = np.clip(F_emp + z * se, 0.0, 1.0)

    return F_emp, F_lo, F_hi


def gaussian_kernel(size, sigma):
    x = np.arange(size) - (size - 1) / 2
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= np.sum(g)
    return g


def smooth_1d(y, sigma=3):
    size = int(np.ceil(6 * sigma)) | 1
    kernel = gaussian_kernel(size, sigma)
    return np.convolve(y, kernel, mode="same")


def fit_weibull_to_cdf(times, t_grid=None):
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
    Wider bins + stronger smoothing + tail truncation.
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
# Models
# ---------------------------
def simulate_constant_hazard(n, r=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.exponential(scale=1.0 / r, size=n)


def simulate_increasing_hazard(n, rates=(0.7, 0.9, 1.1, 1.3), rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x1 = rng.exponential(scale=1.0 / rates[0], size=n)
    x2 = rng.exponential(scale=1.0 / rates[1], size=n)
    x3 = rng.exponential(scale=1.0 / rates[2], size=n)
    x4 = rng.exponential(scale=1.0 / rates[3], size=n)
    return x1 + x2 + x3 + x4


def simulate_decreasing_hazard(n, r_transition=2.2, r_adapt=1.6, r_refractory=0.025, rng=None):
    """
    Tuned so the theoretical hazard is more cleanly front-loaded then decreasing.
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
# Theory
# ---------------------------
def theoretical_constant_cdf(t, r):
    return 1.0 - np.exp(-r * t)


def theoretical_constant_hazard(t, r):
    return np.full_like(t, r, dtype=float)


def theoretical_increasing_cdf(t, rates):
    rates = np.array(rates, dtype=float)
    if np.allclose(rates, rates[0]):
        return gamma.cdf(t, a=len(rates), scale=1.0 / rates[0])

    mean = np.sum(1.0 / rates)
    var = np.sum(1.0 / rates**2)
    shape = mean**2 / var
    scale = var / mean
    return gamma.cdf(t, a=shape, scale=scale)


def theoretical_increasing_hazard(t, rates):
    rates = np.array(rates, dtype=float)
    if np.allclose(rates, rates[0]):
        pdf = gamma.pdf(t, a=len(rates), scale=1.0 / rates[0])
        sf = gamma.sf(t, a=len(rates), scale=1.0 / rates[0])
    else:
        mean = np.sum(1.0 / rates)
        var = np.sum(1.0 / rates**2)
        shape = mean**2 / var
        scale = var / mean
        pdf = gamma.pdf(t, a=shape, scale=scale)
        sf = gamma.sf(t, a=shape, scale=scale)

    with np.errstate(divide="ignore", invalid="ignore"):
        haz = pdf / sf
    haz[~np.isfinite(haz)] = np.nan
    return haz


def theoretical_decreasing_cdf(t, r_transition, r_adapt, r_refractory):
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
def make_combined_cdf_figure(results, output_dir):
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    base_colours = {
        "Constant hazard (S→T)": (0.0000, 0.4470, 0.7410),
        "Increasing hazard (S0→S1→S2→S3→T)": (0.8500, 0.3250, 0.0980),
        "Decreasing hazard (S→R→T)": (0.4660, 0.6740, 0.1880),
    }

    for label, res in results.items():
        emp_col = lighten_color(base_colours[label], factor=0.90)
        theo_col = darken_color(base_colours[label], factor=0.62)

        ax.fill_between(
            res["t_grid"],
            res["F_lo"],
            res["F_hi"],
            color=emp_col,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(res["t_grid"], res["F_emp"], linewidth=2.2, color=emp_col, label=label)
        ax.plot(res["t_grid"], res["F_theory"], linewidth=3.0, linestyle="--", color=theo_col)

    ax.grid(True)
    ax.set_xlim(0, results["Constant hazard (S→T)"]["t_max"])
    ax.set_ylim(0, 1.05)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, "topologyModel_CDFs.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "topologyModel_CDFs.svg"), bbox_inches="tight")
    plt.close(fig)


def make_combined_hazard_figure(results, output_dir):
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    base_colours = {
        "Constant hazard (S→T)": (0.0000, 0.4470, 0.7410),
        "Increasing hazard (S0→S1→S2→S3→T)": (0.8500, 0.3250, 0.0980),
        "Decreasing hazard (S→R→T)": (0.4660, 0.6740, 0.1880),
    }

    for label, res in results.items():
        emp_col = lighten_color(base_colours[label], factor=0.90)
        theo_col = darken_color(base_colours[label], factor=0.62)

        ax.fill_between(res["haz_t"], res["haz_lo"], res["haz_hi"], color=emp_col, alpha=0.16, linewidth=0)
        ax.plot(res["haz_t"], res["haz_emp"], linewidth=2.0, color=emp_col, label=label)
        ax.plot(res["haz_t"], res["haz_theory"], linewidth=3.0, linestyle="--", color=theo_col)

    x_max = min(results["Constant hazard (S→T)"]["haz_t"][-1],
                results["Increasing hazard (S0→S1→S2→S3→T)"]["haz_t"][-1],
                results["Decreasing hazard (S→R→T)"]["haz_t"][-1])

    ax.grid(True)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, "topologyModel_hazards.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "topologyModel_hazards.svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "topologyModelFigures")
    ensure_dir(output_dir)

    rng = np.random.default_rng(42)

    # Increased simulations
    n_sim = 100_000_000
    t_max = 10.0

    # Parameters
    r_const = 1.0
    rates_inc = (0.7, 0.9, 1.1, 1.3)

    # Tuned decreasing-hazard parameters
    r_tr = 2.2
    r_ad = 1.6
    r_ref = 0.025

    # Simulate
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
        t_grid = np.linspace(0, t_max, 800)
        F_emp, F_lo, F_hi = empirical_cdf_with_ci(times, t_grid)
        lam_hat, k_hat = fit_weibull_to_cdf(times, t_grid=t_grid)

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
            t_max=t_max,
            n_bins=501,          # wider bins
            smooth_sigma=6.0,   # stronger smoothing
            min_risk=0,      # truncate unstable tail
        )

        hz_theory = np.interp(haz_est["t"], t_grid, hz_theory_full)

        results[label] = {
            "times": times,
            "t_grid": t_grid,
            "F_emp": F_emp,
            "F_lo": F_lo,
            "F_hi": F_hi,
            "F_theory": F_theory,
            "lam_hat": lam_hat,
            "k_hat": k_hat,
            "haz_t": haz_est["t"],
            "haz_emp": haz_est["haz"],
            "haz_lo": haz_est["lo"],
            "haz_hi": haz_est["hi"],
            "haz_theory": hz_theory,
            "t_max": t_max,
        }

    make_combined_cdf_figure(results, output_dir)
    make_combined_hazard_figure(results, output_dir)

    summary_path = os.path.join(output_dir, "topologyModel_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Topology model simulation summary\n")
        f.write("================================\n\n")
        f.write(f"Number of simulated trajectories per model: {n_sim}\n")
        f.write(f"Common plotting range: t in [0, {t_max}]\n\n")

        f.write("Model definitions used\n")
        f.write("----------------------\n")
        f.write(f"Constant hazard: S -> T with rate r = {r_const}\n")
        f.write(f"Increasing hazard: S0 -> S1 -> S2 -> S3 -> T with rates = {rates_inc}\n")
        f.write(f"Decreasing hazard: S -> T with rate {r_tr}, S -> R with rate {r_ad}, R -> T with rate {r_ref}\n\n")

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
        f.write("  90 bins over [0,10]\n")
        f.write("  Gaussian smoothing sigma = 6.0 bins\n")
        f.write("  tail truncated where risk set < 5000\n\n")

        f.write("Theory curves\n")
        f.write("-------------\n")
        f.write("  Constant hazard: exact exponential CDF and hazard\n")
        f.write("  Increasing hazard: gamma moment-matched theoretical approximation\n")
        f.write("  Decreasing hazard: exact CDF and hazard for S->T, S->R, R->T model\n")

    print(f"Saved outputs to: {output_dir}")
    print(f"Summary: {summary_path}")
    for label, res in results.items():
        print(f"{label}: fitted lambda = {res['lam_hat']:.4f}, fitted k = {res['k_hat']:.4f}")


if __name__ == "__main__":
    main()