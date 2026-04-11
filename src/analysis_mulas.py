# analysis_mulas.py
# Mulas 2017 Stem Cell Reports
# DOI: 10.1016/j.stemcr.2017.05.033
# Data for Primitive Streak like Differentiations is from Fig 2 A
# Data for Lateral Medoderm Differentiation is from Fig 2 C
# Data for Definitive Endoderm Differentiation is from Fig 2 E
# Data for Neural Differentiation is from Fig 2 K
#
# For each LINEAGE, fit Weibull fraction model to Mean by Cell Status,
# compare characteristic timescale (lam), timing (t50), synchronicity (k),
# and (if not fixed) pi across statuses, and plot data (mean±SD markers)
# with overlaid model curves.
#
# NOTE:
#    t0 is fixed to 0.0 for all fits.
#    For lineages 'Neural' and 'Primitive Streak', pi is fixed to 1.0.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from scipy.stats import norm

from cstk_fit import fit_weibull, fraction_population


# -----------------------
# Config
# -----------------------
DATA_PATH = os.path.join("data", "mulas2017stemCellReports.csv")
OUT_DIR = Path("mulas2017stemCellReports")
OUT_DIR.mkdir(exist_ok=True)

LINEAGE_COL = "Lineage"
GENE_COL    = "Gene"
STATUS_COL  = "Cell Status"
TIME_COL    = "Time"
MEAN_COL    = "Mean"
SD_COL      = "SD"

ALPHA = 0.05  # FDR threshold


# -----------------------
# Small utilities
# -----------------------
def canonical_status(label: str) -> str:
    """Map status labels to a consistent canonical form."""
    s = str(label).strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = " ".join(s.split())

    if "2i" in s:
        return "2i"

    if ("rex" in s) or ("rd2" in s):
        if ("low" in s) or (" lo" in s):
            return "rex1-lo"
        if ("high" in s) or (" hi" in s):
            return "rex1-hi"

    return s


def wald_p(diff, se):
    """Two-sided Wald p-value using a normal approximation."""
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan

    z = diff / se
    p = 2.0 * norm.sf(abs(z))
    return p, z


def bh_fdr(pvals):
    """Benjamini–Hochberg adjusted q-values (NaNs preserved)."""
    p = np.array([pv if np.isfinite(pv) else np.nan for pv in pvals], dtype=float)
    n = np.sum(np.isfinite(p))

    if n == 0:
        return p

    idx = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranked = p[idx]

    q = np.full_like(p, np.nan, dtype=float)
    prev = 1.0

    for i, pv in enumerate(ranked, start=1):
        prev = min(prev, pv * n / i)
        q[idx[i - 1]] = prev

    return q


def var_t50_from_cov_fixed_t0(p, cov, fitted_keys):
    """
    Delta-method variance of t50 with t0 fixed to 0.

    t50 = lam * (ln 2)^(1/k)

    Gradient is taken with respect to the free parameters in fitted_keys.
    """
    if cov is None:
        return np.nan

    try:
        ln2 = np.log(2.0)
        g_full = {
            "lam": ln2 ** (1.0 / p["k"]),
            "k": p["lam"] * (ln2 ** (1.0 / p["k"])) * (-(np.log(ln2)) / (p["k"] ** 2)),
            "pi": 0.0,
        }

        grad_keys = [k for k in fitted_keys if k in ("lam", "k")]
        if not grad_keys:
            return 0.0

        g = np.array([g_full[k] for k in grad_keys], dtype=float)
        idx = [fitted_keys.index(k) for k in grad_keys]
        C = np.asarray(cov)[np.ix_(idx, idx)]

        return float(g @ C @ g)

    except Exception:
        return np.nan


def safe_name(s: str) -> str:
    """Make a string safe for use in filenames."""
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def extract_se(param_name, cov, fitted_keys):
    """Extract the standard error of a scalar fitted parameter."""
    if (cov is None) or (param_name not in fitted_keys):
        return np.nan

    try:
        i = fitted_keys.index(param_name)
        v = float(np.asarray(cov)[i, i])
        return np.sqrt(v) if v >= 0 else np.nan
    except Exception:
        return np.nan


def pairwise_table(summary, value_col, se_col, label_col="status"):
    """
    Build a pairwise Wald table for a given fitted parameter or derived metric.

    Assumes independence between estimates when forming the SE of the difference.
    """
    rows = []

    for a, b in combinations(summary[label_col], 2):
        ra = summary.loc[summary[label_col] == a].iloc[0]
        rb = summary.loc[summary[label_col] == b].iloc[0]

        diff = float(ra[value_col]) - float(rb[value_col])

        se_a = float(ra[se_col])
        se_b = float(rb[se_col])

        if np.isfinite(se_a) and np.isfinite(se_b):
            se_diff = np.sqrt(se_a ** 2 + se_b ** 2)
        else:
            se_diff = np.nan

        p, z = wald_p(diff, se_diff)

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
        out["different_at_FDR"] = out["q_BH"] < ALPHA

    return out


# -----------------------
# Load & tidy
# -----------------------
df = pd.read_csv(DATA_PATH)

df = df[[LINEAGE_COL, GENE_COL, STATUS_COL, TIME_COL, MEAN_COL, SD_COL]].rename(
    columns={
        LINEAGE_COL: "lineage",
        GENE_COL: "gene",
        STATUS_COL: "status",
        TIME_COL: "time",
        MEAN_COL: "mean",
        SD_COL: "sd",
    }
)

df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
df["sd"]   = pd.to_numeric(df["sd"], errors="coerce")

df = df.dropna(subset=["lineage", "gene", "status", "time", "mean"]).copy()

df["y"] = df["mean"].astype(float)
df["yerr"] = df["sd"].astype(float)

mask = np.isfinite(df["time"].to_numpy(float)) & np.isfinite(df["y"].to_numpy(float))
df = df.loc[mask].copy()

df["status_canon"] = df["status"].map(canonical_status)

lineages = df["lineage"].astype(str).unique()


# -----------------------
# Fit per (lineage, status) with t0 fixed to 0
# -----------------------
all_summ = []

for lineage in lineages:
    sub_lin = df[df["lineage"] == lineage].copy()

    # Decide whether pi is fixed for this lineage
    lineage_pi_fixed = str(lineage).strip().lower() in {"neural", "primitive streak"}

    rows = []

    for status_canon, sub in sub_lin.groupby("status_canon"):
        t = sub["time"].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)

        m = np.isfinite(t) & np.isfinite(y)
        t, y = t[m], y[m]

        if t.size < 2:
            print(f"[{lineage}] skip {status_canon}: <2 finite points.")
            continue

        if (np.nanmax(y) - np.nanmin(y)) < 1e-8:
            print(f"[{lineage}] skip {status_canon}: nearly constant y; cannot fit.")
            continue

        # Fit with t0 fixed to 0; pi fixed only for specified lineages
        fit = fit_weibull(
            t,
            y,
            yerr=None,
            robust=True,
            return_cov=True,
            fix_t0=0.0,
            fix_pi=(1.0 if lineage_pi_fixed else None),
        )

        # Free-parameter order used in fit
        fitted_keys = ["lam", "k"] + ([] if lineage_pi_fixed else ["pi"])
        p = fit.params

        # Standard errors
        lam_se = extract_se("lam", fit.cov, fitted_keys)
        var_t50 = var_t50_from_cov_fixed_t0(p, fit.cov, fitted_keys)
        t50_se = np.sqrt(var_t50) if (np.isfinite(var_t50) and var_t50 >= 0) else np.nan
        k_se = extract_se("k", fit.cov, fitted_keys)
        pi_se = extract_se("pi", fit.cov, fitted_keys) if not lineage_pi_fixed else np.nan

        rows.append({
            "lineage": lineage,
            "gene": ", ".join(sorted(sub["gene"].astype(str).unique())),
            "status": sub["status"].iloc[0],
            "status_canon": status_canon,
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
            "pi_se": pi_se,
            "r2": fit.r2,
            "AIC": fit.aic,
            "BIC": fit.bic,
            "success": fit.success,
            "message": fit.message,
        })

    if not rows:
        print(f"[{lineage}] No statuses could be fitted.")
        continue

    summary = pd.DataFrame(rows).sort_values("t50").reset_index(drop=True)
    safe_lin = safe_name(lineage)

    summary_path = OUT_DIR / f"{safe_lin}_fit_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    # -----------------------
    # Pairwise statistics
    # -----------------------

    # (i) Characteristic timescale (lam)
    lam_pairs = pairwise_table(summary, value_col="lam", se_col="lam_se", label_col="status")
    if not lam_pairs.empty:
        pvals = 2.0 * norm.sf(np.abs(lam_pairs["z"].astype(float)))
        lam_pairs["p_two_sided"] = np.maximum(pvals, 1e-300)
        lam_pairs["q_BH"] = bh_fdr(lam_pairs["p_two_sided"].values)
        lam_pairs["different_at_FDR"] = lam_pairs["q_BH"] < ALPHA

        lam_path = OUT_DIR / f"{safe_lin}_pairwise_lam.csv"
        lam_pairs.to_csv(lam_path, index=False)
        print("Saved:", lam_path)

    # (ii) Time of differentiation (t50)
    t50_pairs = pairwise_table(summary, value_col="t50", se_col="t50_se", label_col="status")
    if not t50_pairs.empty:
        pvals = 2.0 * norm.sf(np.abs(t50_pairs["z"].astype(float)))
        t50_pairs["p_two_sided"] = np.maximum(pvals, 1e-300)
        t50_pairs["q_BH"] = bh_fdr(t50_pairs["p_two_sided"].values)
        t50_pairs["different_at_FDR"] = t50_pairs["q_BH"] < ALPHA

        t50_path = OUT_DIR / f"{safe_lin}_pairwise_t50.csv"
        t50_pairs.to_csv(t50_path, index=False)
        print("Saved:", t50_path)

    # (iii) Synchronicity (k)
    k_pairs = pairwise_table(summary, value_col="k", se_col="k_se", label_col="status")
    if not k_pairs.empty:
        pvals = 2.0 * norm.sf(np.abs(k_pairs["z"].astype(float)))
        k_pairs["p_two_sided"] = np.maximum(pvals, 1e-300)
        k_pairs["q_BH"] = bh_fdr(k_pairs["p_two_sided"].values)
        k_pairs["different_at_FDR"] = k_pairs["q_BH"] < ALPHA

        k_path = OUT_DIR / f"{safe_lin}_pairwise_k.csv"
        k_pairs.to_csv(k_path, index=False)
        print("Saved:", k_path)

    # (iv) Competent fraction (pi), only when pi is free
    if not lineage_pi_fixed:
        pi_pairs = pairwise_table(summary, value_col="pi", se_col="pi_se", label_col="status")
        if not pi_pairs.empty:
            pvals = 2.0 * norm.sf(np.abs(pi_pairs["z"].astype(float)))
            pi_pairs["p_two_sided"] = np.maximum(pvals, 1e-300)
            pi_pairs["q_BH"] = bh_fdr(pi_pairs["p_two_sided"].values)
            pi_pairs["different_at_FDR"] = pi_pairs["q_BH"] < ALPHA

            pi_path = OUT_DIR / f"{safe_lin}_pairwise_pi.csv"
            pi_pairs.to_csv(pi_path, index=False)
            print("Saved:", pi_path)

    all_summ.append(summary.assign(lineage=lineage))

    # -----------------------
    # Plot for this lineage
    # -----------------------
    t_end = float(sub_lin["time"].max())

    width = (t_end + 24.0) / 24.0
    fig, ax = plt.subplots(figsize=(width, 4.0), dpi=400)

    ax.set_title(f"{lineage}", fontsize=16, weight="bold")

    gene_label = ", ".join(sorted(sub_lin["gene"].astype(str).unique()))
    ax.set_ylabel(f"{gene_label} Cell Fraction")
    ax.set_xlabel("Time (h)")

    x_max = t_end + 24.0
    ax.set_xlim(0.0, x_max)

    xticks = list(np.arange(0.0, x_max, 12.0))
    if not np.isclose(xticks[-1] if xticks else -np.inf, x_max):
        xticks.append(x_max)
    ax.set_xticks(xticks)

    y_min = 0.0
    y_max = 1.0
    yticks = list(np.arange(0.0, 1.1, 0.1))
    ax.set_yticks(yticks)

    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.5)

    fixed_colors = {
        "rex1-lo": "#000000",  # black
        "rex1-hi": "#2e7d32",  # mid green
        "2i": "#00e676",       # bright green
    }
    base_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot data
    for status_canon, sub in sub_lin.groupby("status_canon"):
        c = fixed_colors.get(status_canon, base_palette[hash(status_canon) % len(base_palette)])

        y_mean = sub["mean"].astype(float)
        y_sd = sub["sd"].astype(float)

        ax.errorbar(
            sub["time"],
            y_mean,
            yerr=y_sd,
            fmt="o",
            ms=8.5,
            lw=2.0,
            elinewidth=2.2,
            capsize=3.5,
            color=c,
            ecolor=c,
            mec=c,
            mfc="none",
            mew=2.2,
            alpha=0.95,
            zorder=3,
        )

    # Plot fitted curves
    tgrid = np.linspace(0.0, x_max, 500)
    for _, row in summary.iterrows():
        s_canon = row["status_canon"]
        c = fixed_colors.get(s_canon, base_palette[hash(s_canon) % len(base_palette)])

        yfit = fraction_population(tgrid, row["t0"], row["lam"], row["k"], row["pi"])
        ax.plot(tgrid, yfit, color=c, lw=3.0, alpha=0.95, zorder=2)

    y_all = pd.concat(
        [sub_lin["mean"], sub_lin["mean"] + sub_lin["sd"], sub_lin["mean"] - sub_lin["sd"]],
        axis=0
    )
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()

    out_base = OUT_DIR / f"{safe_lin}"
    fig.savefig(f"{out_base}.png", dpi=300)
    fig.savefig(f"{out_base}.svg", dpi=300)
    plt.close(fig)

    print(f"Saved: {out_base}.png and {out_base}.svg")


# -----------------------
# Save combined summary
# -----------------------
if all_summ:
    all_summary_path = OUT_DIR / "ALL_fit_summary_by_lineage.csv"
    pd.concat(all_summ, ignore_index=True).to_csv(all_summary_path, index=False)
    print("Saved:", all_summary_path)