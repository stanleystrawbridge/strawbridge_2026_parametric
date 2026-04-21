# analysis_hanna.py
# Hanna 2009 Nature
# DOI: 10.1038/nature08592
# Data from Figure 4D
#
# For each condition, fit a Weibull fraction model,
# compare characteristic timescale (lam), timing (t50),
# and synchronicity (k) across all pairs (post-hoc BH FDR),
# and plot empirical data with overlaid model curves.
#
# NOTE:
#   - t0 is fixed to 0.0
#   - pi is fixed to 1.0

import os
from pathlib import Path
from itertools import cycle, combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from cstk_fit import fit_weibull, fraction_population


# -----------------------
# Config
# -----------------------
DATA_PATH = os.path.join("data", "hanna2009nature.csv")
OUT_DIR = Path("hanna2009nature")
OUT_DIR.mkdir(exist_ok=True)

# Column-name heuristics (case-insensitive substring match)
CAND_TIME = ["time", "t", "hours", "hour", "h"]
CAND_COND = ["condition", "genotype", "cell_line", "cell line", "line", "group"]
CAND_VAL = ["fraction", "value", "y", "mean", "percentage", "percent", "prop", "proportion"]

# Used only for colouring and legend ordering
BASELINE_CONTAINS = ["NGFP1"]

ALPHA = 0.05


# -----------------------
# Helpers
# -----------------------
def find_column(df: pd.DataFrame, candidates) -> str:
    """Return the first column whose lowercase name matches any candidate token."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for lc, original in cols.items():
            if cand == lc or cand in lc:
                return original
    raise ValueError(
        f"Could not auto-detect a column among: {candidates}\n"
        f"Columns present: {list(df.columns)}"
    )


def normalize_to_unit(y: np.ndarray) -> np.ndarray:
    """
    Return values in [0,1].

    If values already lie in [0,1], return as-is.
    If values look like percentages, divide by 100.
    Otherwise raise an error.
    """
    y = y.astype(float)
    mmin, mmax = np.nanmin(y), np.nanmax(y)

    if (mmin >= -1e-9) and (mmax <= 1.0 + 1e-9):
        return y
    if mmax <= 120.0 and mmin >= -1.0:
        return y / 100.0

    raise ValueError("Values are not in [0,1] and do not look like percentages (0..100).")


def pick_baseline_label(levels) -> str:
    """Pick the baseline label using BASELINE_CONTAINS; otherwise return the first level."""
    for lev in levels:
        if all(k.lower() in lev.lower() for k in BASELINE_CONTAINS):
            return lev
    return list(levels)[0]


def var_t50_from_cov_fixed_t0(p: dict, cov: np.ndarray, fitted_keys) -> float:
    """
    Delta-method variance of t50 when t0 is fixed to 0:

        t50 = lam * (ln 2)^(1/k)

    Only lam and k enter the gradient.
    """
    if cov is None:
        return np.nan

    try:
        ln2 = np.log(2.0)
        g_full = {
            "lam": ln2 ** (1.0 / p["k"]),
            "k": p["lam"] * (ln2 ** (1.0 / p["k"])) * (-(np.log(ln2)) / (p["k"] ** 2)),
        }

        grad_keys = [k for k in fitted_keys if k in ("lam", "k")]
        if not grad_keys:
            return np.nan

        g = np.array([g_full[k] for k in grad_keys], float)
        idx = [fitted_keys.index(k) for k in grad_keys]
        C = np.asarray(cov)[np.ix_(idx, idx)]

        return float(g @ C @ g)

    except Exception:
        return np.nan


def extract_se(param_name: str, cov: np.ndarray, fitted_keys) -> float:
    """Return the standard error for one fitted parameter."""
    if (cov is None) or (param_name not in fitted_keys):
        return np.nan

    try:
        i = fitted_keys.index(param_name)
        v = float(np.asarray(cov)[i, i])
        return np.sqrt(v) if v >= 0 else np.nan
    except Exception:
        return np.nan


def wald_p_from_diff(diff: float, se: float):
    """
    Two-sided Wald p-value from a normal approximation.
    Returns (p, z). If se is invalid, returns (nan, nan).
    """
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan

    z = diff / se
    p = 2.0 * norm.sf(abs(z))
    return p, z


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR adjustment.
    Returns q-values with NaNs preserved.
    """
    p = np.array([pv if np.isfinite(pv) else np.nan for pv in pvals], dtype=float)
    m = np.sum(np.isfinite(p))
    q = np.full_like(p, np.nan)

    if m == 0:
        return q

    idx = np.argsort(np.where(np.isfinite(p), p, np.inf))
    idx = idx[:m]
    ranked = p[idx]

    bh = ranked * m / (np.arange(1, m + 1))
    bh_monotone = np.minimum.accumulate(bh[::-1])[::-1]

    q[idx] = bh_monotone
    return q


def pairwise_wald(df_sum, value_col, se_col, label_col="condition"):
    """
    Build a pairwise comparison table using Wald z-tests.

    diff = a - b
    se_diff = sqrt(se_a^2 + se_b^2), assuming independent fits
    """
    rows = []
    labels = list(df_sum[label_col])

    for a, b in combinations(labels, 2):
        ra = df_sum.loc[df_sum[label_col] == a].iloc[0]
        rb = df_sum.loc[df_sum[label_col] == b].iloc[0]

        diff = float(ra[value_col]) - float(rb[value_col])

        se_a = float(ra.get(se_col, np.nan))
        se_b = float(rb.get(se_col, np.nan))
        se_diff = np.sqrt(se_a ** 2 + se_b ** 2) if np.isfinite(se_a) and np.isfinite(se_b) else np.nan

        p, z = wald_p_from_diff(diff, se_diff)
        if np.isfinite(p):
            p = max(p, 1e-300)

        rows.append({
            f"{label_col}_a": a,
            f"{label_col}_b": b,
            "diff": diff,
            "se_diff": se_diff,
            "z": z,
            "p_two_sided": p,
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out["q_BH"] = bh_fdr(out["p_two_sided"].values)
        out["significant_FDR"] = (out["q_BH"] < ALPHA)

    return out


# -----------------------
# Load and tidy data
# -----------------------
df = pd.read_csv(DATA_PATH)

time_col = find_column(df, CAND_TIME)
cond_col = find_column(df, CAND_COND)
val_col = find_column(df, CAND_VAL)

df = df[[time_col, cond_col, val_col]].rename(
    columns={time_col: "time", cond_col: "condition", val_col: "value"}
)

# Convert values to fractions if needed
df["value"] = normalize_to_unit(df["value"].values)
df = df.dropna(subset=["time", "condition", "value"]).copy()
df["time"] = df["time"].astype(float)
df["condition"] = df["condition"].astype(str)

conditions = df["condition"].unique()
baseline = pick_baseline_label(conditions)  # used only for colouring/legend
print(f"Baseline (for legend) = '{baseline}'")


# -----------------------
# Fit per condition (t0 fixed to 0, pi fixed to 1)
# -----------------------
rows = []

for cond, sub in df.groupby("condition"):
    t = sub["time"].to_numpy(float)
    y = sub["value"].to_numpy(float)

    fit = fit_weibull(
        t,
        y,
        yerr=None,
        robust=True,
        return_cov=True,
        fix_t0=0.0,
        fix_pi=1.0,
    )

    p = fit.params
    fitted_keys = ["lam", "k"]  # free parameters when t0 and pi are fixed

    # Standard errors
    lam_se = extract_se("lam", fit.cov, fitted_keys)
    var_t50 = var_t50_from_cov_fixed_t0(p, fit.cov, fitted_keys)
    t50_se = np.sqrt(var_t50) if (np.isfinite(var_t50) and var_t50 >= 0) else np.nan
    k_se = extract_se("k", fit.cov, fitted_keys)

    rows.append({
        "condition": cond,
        "t0": p["t0"],
        "lam": p["lam"],
        "k": p["k"],
        "pi": p["pi"],
        "t10": fit.t_metrics["t10"],
        "t50": fit.t_metrics["t50"],
        "t90": fit.t_metrics["t90"],
        "lam_se": lam_se,
        "t50_se": t50_se,
        "k_se": k_se,
        "r2": fit.r2,
        "AIC": fit.aic,
        "BIC": fit.bic,
        "success": fit.success,
        "message": fit.message,
    })

summary = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)
summary.to_csv(OUT_DIR / "hanna2009nature_fit_summary.csv", index=False)
print("Saved:", OUT_DIR / "hanna2009nature_fit_summary.csv")


# -----------------------
# Pairwise Wald tests (post-hoc BH) for lam, t50, and k
# -----------------------
lam_pairs = pairwise_wald(summary, value_col="lam", se_col="lam_se", label_col="condition")
t50_pairs = pairwise_wald(summary, value_col="t50", se_col="t50_se", label_col="condition")
k_pairs = pairwise_wald(summary, value_col="k", se_col="k_se", label_col="condition")

lam_pairs.to_csv(OUT_DIR / "hanna2009nature_pairwise_lam.csv", index=False)
t50_pairs.to_csv(OUT_DIR / "hanna2009nature_pairwise_t50.csv", index=False)
k_pairs.to_csv(OUT_DIR / "hanna2009nature_pairwise_k.csv", index=False)

print("Saved:", OUT_DIR / "hanna2009nature_pairwise_lam.csv")
print("Saved:", OUT_DIR / "hanna2009nature_pairwise_t50.csv")
print("Saved:", OUT_DIR / "hanna2009nature_pairwise_k.csv")


# -----------------------
# Plot settings and colours
# -----------------------
xmin, xmax = 0.0, 60.0
ymin, ymax = 0.0, 1.0
tgrid = np.linspace(xmin, xmax, 600)

# Colour rules:
#   NGFP1             -> black
#   contains lin28    -> blue
#   all other NGFP1-* -> orange shades
nice_blue = "#1976d2"
from matplotlib.cm import get_cmap
oranges = get_cmap("Oranges")
orange_palette = [oranges(v) for v in np.linspace(0.35, 0.95, 6)]
orange_iter = cycle(orange_palette)


def color_for(cond: str, baseline_label: str = "NGFP1") -> str:
    """Assign a plotting colour to each condition."""
    cl = cond.strip().lower()
    if cl == baseline_label.strip().lower():
        return "#000000"
    if "lin28" in cl:
        return nice_blue
    return next(orange_iter)


cond_colors = {cond: color_for(cond, baseline_label=baseline) for cond in summary["condition"]}


# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=400)

for cond, sub in df.groupby("condition"):
    c = cond_colors[cond]

    # Empirical markers
    ax.plot(
        sub["time"], sub["value"],
        "o",
        ms=6.5,
        mfc="none",
        mec=c,
        mew=1.8,
        alpha=0.95,
        zorder=3,
    )

    # Model curve
    p = summary.loc[summary["condition"] == cond].iloc[0]
    yfit = fraction_population(tgrid, p["t0"], p["lam"], p["k"], p["pi"])
    ax.plot(
        tgrid, yfit,
        color=c,
        lw=2.6,
        alpha=0.95,
        zorder=2,
        label=cond,
    )

ax.set_title("Reprogramming Kinetics", fontsize=14, weight="bold")

# Legend order: baseline -> lin28 -> others alphabetically
legend_order = []
legend_order += [c for c in summary["condition"] if c.strip().lower() == baseline.strip().lower()]
legend_order += [c for c in summary["condition"] if "lin28" in c.lower()]
legend_order += sorted([c for c in summary["condition"] if c not in legend_order])

handles, labels = ax.get_legend_handles_labels()
handle_map = {lab: h for h, lab in zip(handles, labels)}
ax.legend(
    [handle_map[c] for c in legend_order if c in handle_map],
    [c for c in legend_order if c in handle_map],
    frameon=False,
    fontsize=9,
    loc="lower right",
    title="Condition",
    title_fontsize=10,
)

ax.set_xlim(xmin, xmax)
ax.set_xticks(np.arange(xmin, xmax + 1e-9, 12.0))
ax.set_ylim(ymin, ymax)
ax.set_yticks(np.arange(ymin, ymax + 1e-9, 0.1))
ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.5)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Fraction of Nanog-GFP+ Wells")

fig.tight_layout()
fig.savefig(OUT_DIR / "hanna2009nature_all_conditions.png")
fig.savefig(OUT_DIR / "hanna2009nature_all_conditions.svg")
plt.close(fig)


# -----------------------
# Console summaries
# -----------------------
print("\n=== Per-condition fits ===")
print(summary[["condition", "lam", "lam_se", "t50", "t50_se", "k", "k_se", "r2"]].to_string(index=False))

print("\n=== Pairwise Wald tests: lam (characteristic timescale) ===")
if not lam_pairs.empty:
    print(
        lam_pairs[["condition_a", "condition_b", "diff", "se_diff", "z", "p_two_sided", "q_BH", "significant_FDR"]]
        .to_string(index=False)
    )
else:
    print("No pairwise results (insufficient SEs).")

print("\n=== Pairwise Wald tests: t50 (timing) ===")
if not t50_pairs.empty:
    print(
        t50_pairs[["condition_a", "condition_b", "diff", "se_diff", "z", "p_two_sided", "q_BH", "significant_FDR"]]
        .to_string(index=False)
    )
else:
    print("No pairwise results (insufficient SEs).")

print("\n=== Pairwise Wald tests: k (synchronicity) ===")
if not k_pairs.empty:
    print(
        k_pairs[["condition_a", "condition_b", "diff", "se_diff", "z", "p_two_sided", "q_BH", "significant_FDR"]]
        .to_string(index=False)
    )
else:
    print("No pairwise results (insufficient SEs).")