# analysis_leeb.py
# Leeb 2014 Cell Stem Cell
# DOI: 10.1016/j.stem.2013.12.008
# Data from Figure 3F and S3C
#
# For each gene, fit a Weibull fraction model,
# compare characteristic timescale (lam), timing (t50),
# synchronicity (k), and asymptote (pi),
# and plot empirical data with overlaid model curves.
#
# NOTE:
#   - t0 is FIXED to 0.0
#   - pi is free (fit) to allow incomplete down-regulation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

from cstk_fit import fit_weibull, fraction_population

# -----------------------
# Config
# -----------------------
DATA_PATH = os.path.join("data", "leeb2014cellStemCell.csv")
OUT_DIR = Path("leeb2014cellStemCell")
OUT_DIR.mkdir(exist_ok=True)

TIME_COL = "Time (h)"
COND_COL = "Gene"
VAL_COL  = "Expression"
ALPHA = 0.05  # significance threshold for adjusted p-values

# -----------------------
# Utilities
# -----------------------
def _norm_vec(v: np.ndarray) -> np.ndarray:
    """Normalize to [0,1]. If looks like percent (0..100), divide by 100 first."""
    v = np.asarray(v, float)
    vmax, vmin = np.nanmax(v), np.nanmin(v)
    if vmax > 1.2 and vmax <= 120.0 and vmin >= -1.0:
        v = v / 100.0
        vmax, vmin = np.nanmax(v), np.nanmin(v)
    if vmin >= -1e-9 and vmax <= 1.0 + 1e-9:
        return v.astype(float)
    rng = vmax - vmin
    if not np.isfinite(vmin) or not np.isfinite(vmax) or rng <= 0:
        raise ValueError("Cannot normalize: non-finite or zero range detected.")
    return ((v - vmin) / rng).astype(float)

def wald_p(diff, se):
    """Two-sided Wald p-value (normal), numerically stable."""
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan
    z = diff / se
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2.0))  # two-sided p
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
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

def holm_bonferroni(pvals):
    """Holm–Bonferroni adjusted p-values (step-down). NaNs preserved."""
    p = np.array([pv if np.isfinite(pv) else np.nan for pv in pvals], dtype=float)
    N = np.sum(np.isfinite(p))
    if N == 0:
        return p
    idx = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranked = p[idx]
    adj = np.full_like(p, np.nan, dtype=float)
    running_max = 0.0
    for k, pv in enumerate(ranked, start=1):
        factor = N - k + 1
        val = pv * factor
        if val > running_max:
            running_max = val
        adj[idx[k - 1]] = min(1.0, running_max)
    return adj

def var_t50_from_cov(p, cov, fitted_keys):
    """
    Delta-method variance of t50 when t0 is fixed (or even free; t50 ignores pi, and
    with fixed t0 the gradient is only in lam and k):
        t50 = t0 + lam * (ln 2)^(1/k)
    """
    if cov is None:
        return np.nan
    try:
        ln2 = np.log(2.0)
        c = (ln2) ** (1.0 / p["k"])
        g_full = {
            "lam": c,
            "k": p["lam"] * c * (-(np.log(ln2)) / (p["k"] ** 2)),
        }
        grad_keys = [k for k in fitted_keys if k in ("lam", "k")]
        if not grad_keys:
            return np.nan
        g = np.array([g_full[k] for k in grad_keys], dtype=float)
        idx = [fitted_keys.index(k) for k in grad_keys]
        C = np.asarray(cov)[np.ix_(idx, idx)]
        return float(g @ C @ g)
    except Exception:
        return np.nan

def extract_se(param_name: str, cov, fitted_keys):
    """SE for a given parameter from covariance matrix."""
    if (cov is None) or (param_name not in fitted_keys):
        return np.nan
    try:
        i = fitted_keys.index(param_name)
        v = float(np.asarray(cov)[i, i])
        return np.sqrt(v) if v >= 0 else np.nan
    except Exception:
        return np.nan

# -----------------------
# Load, tidy, normalize
# -----------------------
df = pd.read_csv(DATA_PATH)[[COND_COL, TIME_COL, VAL_COL]].rename(
    columns={COND_COL: "gene", TIME_COL: "time", VAL_COL: "expr"}
)

df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["expr"] = pd.to_numeric(df["expr"], errors="coerce")
df = df.dropna(subset=["gene", "time", "expr"]).copy()

# Normalize expression per gene to [0,1]
df["expr_norm"] = (
    df.groupby("gene", group_keys=False)["expr"]
      .transform(lambda s: _norm_vec(s.to_numpy(dtype=float)))
      .astype(float)
)

# Down-regulation signal to fit: fraction transitioned = 1 - normalized expression
df["frac"] = (1.0 - df["expr_norm"]).astype(float)

mask = np.isfinite(df["time"].to_numpy(dtype=float)) & np.isfinite(df["frac"].to_numpy(dtype=float))
df = df.loc[mask].copy()

# -----------------------
# Fit per gene (t0 = 0.0; pi free)
# -----------------------
summ_rows = []

tmin_plot = 0.0
tmax_plot = 56.0
time_grid = np.linspace(tmin_plot, tmax_plot, 400)

for g, sub in df.groupby("gene"):
    t = sub["time"].to_numpy(dtype=float)
    y = sub["frac"].to_numpy(dtype=float)

    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 3:
        print(f"[skip] {g}: <3 finite points.")
        continue
    if (np.nanmax(y) - np.nanmin(y)) < 1e-8:
        print(f"[skip] {g}: nearly constant signal; cannot fit.")
        continue

    fit = fit_weibull(t, y, robust=True, return_cov=True, fix_pi=None, fix_t0=0.0)

    p = fit.params
    fitted_keys = ["lam", "k", "pi"]  # t0 is fixed

    # SEs
    lam_se   = extract_se("lam", fit.cov, fitted_keys)
    var_t50  = var_t50_from_cov(p, fit.cov, fitted_keys)
    t50_se   = np.sqrt(var_t50) if (np.isfinite(var_t50) and var_t50 >= 0) else np.nan
    k_se     = extract_se("k", fit.cov, fitted_keys)
    pi_se    = extract_se("pi", fit.cov, fitted_keys)

    summ_rows.append({
        "gene": g,
        "t0": p["t0"],
        "lam": p["lam"],
        "k": p["k"],
        "pi": p["pi"],
        # classic t-metrics from the fitter (among responders)
        "t10": fit.t_metrics["t10"],
        "t50": fit.t_metrics["t50"],
        "t90": fit.t_metrics["t90"],
        # SEs
        "lam_se": lam_se,
        "t50_se": t50_se,
        "k_se": k_se,
        "pi_se": pi_se,
        "r2": fit.r2,
        "AIC": fit.aic,
        "BIC": fit.bic,
        "success": fit.success,
        "message": fit.message
    })

# Order by t50 (earliest → latest) for reporting/plotting
summary = pd.DataFrame(summ_rows).sort_values("t50").reset_index(drop=True)
summary.to_csv(OUT_DIR / "leeb2014_fit_summary.csv", index=False)
print("Saved:", OUT_DIR / "leeb2014_fit_summary.csv")

if summary.empty:
    raise SystemExit("No genes could be fitted (insufficient/invalid data).")

# -----------------------
# Pairwise comparisons (export each metric separately)
# -----------------------
def pairwise_tests(df_sum, value_col, se_col, out_csv, label):
    rows = []
    for a, b in combinations(df_sum["gene"], 2):
        ra = df_sum.loc[df_sum["gene"] == a].iloc[0]
        rb = df_sum.loc[df_sum["gene"] == b].iloc[0]
        diff = float(ra[value_col]) - float(rb[value_col])
        se_a, se_b = float(ra[se_col]), float(rb[se_col])
        se_diff = np.sqrt(se_a**2 + se_b**2) if np.isfinite(se_a) and np.isfinite(se_b) else np.nan
        p, z = wald_p(diff, se_diff)
        rows.append({
            "gene_a": a,
            "gene_b": b,
            "metric": label,
            "diff": diff,
            "se_diff": se_diff,
            "z": z,
            "p_two_sided": p
        })
    comp = pd.DataFrame(rows)
    if not comp.empty:
        comp["p_adj_holm"] = holm_bonferroni(comp["p_two_sided"].values)
        comp["q_BH"] = bh_fdr(comp["p_two_sided"].values)
        comp["significant"] = comp["p_adj_holm"] < ALPHA
        comp.to_csv(OUT_DIR / out_csv, index=False)
        print("Saved:", OUT_DIR / out_csv)
    return comp

# lam
comparisons_lam = pairwise_tests(summary, "lam", "lam_se", "leeb2014_lam_pairwise.csv", "lam")
# t50 (use delta-method SE)
comparisons_t50 = pairwise_tests(summary, "t50", "t50_se", "leeb2014_t50_pairwise.csv", "t50")
# k
comparisons_k   = pairwise_tests(summary, "k", "k_se", "leeb2014_k_pairwise.csv", "k")
# pi
comparisons_pi  = pairwise_tests(summary, "pi", "pi_se", "leeb2014_pi_pairwise.csv", "pi")

# Console summaries
print("\n=== Per-gene fits (ordered by t50) ===")
print(summary[["gene", "lam", "lam_se", "t50", "t50_se", "k", "k_se", "pi", "pi_se", "r2"]].to_string(index=False))

def _quick_report(df_comp, title):
    if df_comp.empty:
        print(f"\nNo pairwise results for {title}.")
        return
    shown = df_comp.sort_values("p_adj_holm")[["gene_a", "gene_b", "diff", "se_diff", "p_adj_holm", "q_BH", "significant"]]
    print(f"\n=== Pairwise comparisons: {title} (Holm-adjusted p) ===")
    print(shown.to_string(index=False))

_quick_report(comparisons_lam, "lam")
_quick_report(comparisons_t50, "t50")
_quick_report(comparisons_k,   "k")
_quick_report(comparisons_pi,  "pi")

# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=400)

# Colors by t50 rank (earliest → latest)
t50_palette = [
    "#1976d2",  # deep blue
    "#00838f",  # deep cyan/teal
    "#2e7d32",  # deep green
    "#f9a825",  # deep amber/yellow
    "#ef6c00",  # deep orange
    "#d32f2f",  # deep red
]
colors = {g: t50_palette[min(i, len(t50_palette) - 1)] for i, g in enumerate(summary["gene"])}

# Empirical markers
for g, sub in df.groupby("gene"):
    if g not in colors:
        continue
    ax.scatter(
        sub["time"], sub["expr_norm"],
        s=64, color=colors[g], alpha=0.9,
        label=f"{g} data", zorder=3, edgecolor="none"
    )

# Model curves: predicted expression = 1 - F(t)
for _, row in summary.iterrows():
    yfit_frac = fraction_population(time_grid, row["t0"], row["lam"], row["k"], row["pi"])
    ax.plot(
        time_grid, 1.0 - yfit_frac,
        color=colors[row["gene"]],
        lw=2.2, alpha=0.95,
        label=f"{row['gene']} model",
        zorder=2
    )

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Normalized Expression")
ax.set_xlim(0, 56)
ax.set_ylim(0, 1.0)
ax.set_xticks(np.arange(0, 56 + 1e-9, 8.0))
ax.set_yticks(np.arange(0, 1.0 + 1e-9, 0.1))
ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.5)

ax.legend(ncol=2, fontsize=8, frameon=False)
fig.tight_layout()

out_png = OUT_DIR / "leeb2014_all_genes.png"
out_svg = OUT_DIR / "leeb2014_all_genes.svg"
fig.savefig(out_png)
fig.savefig(out_svg)
print("Saved:", out_png)
print("Saved:", out_svg)

# ============================================================
# NEW SECTION: siPum1 vs siNeg per gene (S3C dataset)
# ============================================================

S3C_PATH = os.path.join("data", "leeb2014cellStemCell_S3C.csv")

# --- Load S3C ---
df_s3c_raw = pd.read_csv(S3C_PATH)

# Try to be robust to column naming
def _find_col(cands, cols):
    for c in cands:
        if c in cols:
            return c
    raise KeyError(f"Expected one of {cands} in columns: {list(cols)}")

GENE_COL = "gene"
COND_COL = "condition"
TIME_COL = "time"
VAL_COL = "relative_expression"

col_gene = _find_col([GENE_COL], df_s3c_raw.columns)
col_cond = _find_col([COND_COL], df_s3c_raw.columns)
col_time = _find_col([TIME_COL], df_s3c_raw.columns)
col_expr = _find_col([VAL_COL], df_s3c_raw.columns)

df_s3c = df_s3c_raw[[col_gene, col_time, col_expr, col_cond]].rename(
    columns={col_gene: "gene", col_time: "time", col_expr: "expr", col_cond: "cond"}
)

# Keep only the two conditions we care about; normalize names
df_s3c["cond"] = df_s3c["cond"].astype(str)
map_norm = {
    "siPum1": "siPum1", "sipum1": "siPum1", "siPUM1": "siPum1",
    "siNeg": "siNeg", "sineg": "siNeg", "siNEG": "siNeg"
}
df_s3c["cond"] = df_s3c["cond"].map(lambda x: map_norm.get(x, x))
df_s3c = df_s3c[df_s3c["cond"].isin(["siPum1", "siNeg"])].copy()

# Numeric & drop NAs
df_s3c["time"] = pd.to_numeric(df_s3c["time"], errors="coerce")
df_s3c["expr"] = pd.to_numeric(df_s3c["expr"], errors="coerce")
df_s3c = df_s3c.dropna(subset=["gene", "cond", "time", "expr"])

# Normalize expression per gene within each condition to [0,1], then use fraction = 1 - expr_norm
df_s3c["expr_norm"] = (
    df_s3c.groupby(["gene", "cond"], group_keys=False)["expr"]
          .transform(lambda s: _norm_vec(s.to_numpy(dtype=float)))
          .astype(float)
)
df_s3c["frac"] = (1.0 - df_s3c["expr_norm"]).astype(float)

# Time grid same as earlier
time_grid_s3c = np.linspace(tmin_plot, tmax_plot, 400)

# --- Fit per gene per condition ---
rows_fit = []
for (g, c), sub in df_s3c.groupby(["gene", "cond"]):
    t = sub["time"].to_numpy(float)
    y = sub["frac"].to_numpy(float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < 3 or (np.nanmax(y) - np.nanmin(y)) < 1e-8:
        print(f"[skip] {g}/{c}: insufficient data or nearly constant")
        continue

    fit_gc = fit_weibull(t, y, robust=True, return_cov=True, fix_pi=None, fix_t0=0.0)
    p = fit_gc.params
    fitted_keys_gc = ["lam", "k", "pi"]  # t0 fixed

    # SEs
    lam_se_gc  = extract_se("lam", fit_gc.cov, fitted_keys_gc)
    var_t50_gc = var_t50_from_cov(p, fit_gc.cov, fitted_keys_gc)
    t50_se_gc  = np.sqrt(var_t50_gc) if (np.isfinite(var_t50_gc) and var_t50_gc >= 0) else np.nan
    k_se_gc    = extract_se("k", fit_gc.cov, fitted_keys_gc)
    pi_se_gc   = extract_se("pi", fit_gc.cov, fitted_keys_gc)

    rows_fit.append({
        "gene": g,
        "cond": c,
        "t0": p["t0"],
        "lam": p["lam"],
        "k": p["k"],
        "pi": p["pi"],
        "t10": fit_gc.t_metrics["t10"],
        "t50": fit_gc.t_metrics["t50"],
        "t90": fit_gc.t_metrics["t90"],
        "lam_se": lam_se_gc,
        "t50_se": t50_se_gc,
        "k_se": k_se_gc,
        "pi_se": pi_se_gc,
        "r2": fit_gc.r2,
        "AIC": fit_gc.aic,
        "BIC": fit_gc.bic,
        "success": fit_gc.success,
        "message": fit_gc.message
    })

fit_s3c = pd.DataFrame(rows_fit)

# Save per-gene/condition fits
fit_s3c.to_csv(OUT_DIR / "leeb2014_S3C_fit_summary.csv", index=False)
print("Saved:", OUT_DIR / "leeb2014_S3C_fit_summary.csv")

# --- Pairwise (siPum1 vs siNeg) comparisons per gene ---
def compare_two_conditions(fit_table, metric, se_col, out_name):
    rows = []
    for g, sub in fit_table.groupby("gene"):
        if set(sub["cond"]) >= {"siPum1", "siNeg"}:
            ra = sub.loc[sub["cond"] == "siPum1"].iloc[0]
            rb = sub.loc[sub["cond"] == "siNeg"].iloc[0]
            diff = float(ra[metric]) - float(rb[metric])
            se_a, se_b = float(ra[se_col]), float(rb[se_col])
            se_diff = np.sqrt(se_a**2 + se_b**2) if np.isfinite(se_a) and np.isfinite(se_b) else np.nan
            p, z = wald_p(diff, se_diff)
            rows.append({
                "gene": g,
                "metric": metric,
                "siPum1_minus_siNeg": diff,
                "se_diff": se_diff,
                "z": z,
                "p_two_sided": p
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Adjust across genes for this metric
        out["p_adj_holm"] = holm_bonferroni(out["p_two_sided"].values)
        out["q_BH"] = bh_fdr(out["p_two_sided"].values)
        out["significant"] = out["p_adj_holm"] < ALPHA
        out.to_csv(OUT_DIR / out_name, index=False)
        print("Saved:", OUT_DIR / out_name)
    return out

cmp_lam = compare_two_conditions(fit_s3c, "lam", "lam_se", "leeb2014_S3C_comparisons_lam.csv")
cmp_t50 = compare_two_conditions(fit_s3c, "t50", "t50_se", "leeb2014_S3C_comparisons_t50.csv")
cmp_k   = compare_two_conditions(fit_s3c, "k",   "k_se",   "leeb2014_S3C_comparisons_k.csv")
cmp_pi  = compare_two_conditions(fit_s3c, "pi",  "pi_se",  "leeb2014_S3C_comparisons_pi.csv")

# --- Multiplot: 2 rows × 3 cols, ordered by original (first analysis) t50 ---
ordered_genes = list(summary["gene"])
ordered_genes = [g for g in ordered_genes if g in set(fit_s3c["gene"])]

n_panels = min(6, len(ordered_genes))
if n_panels < 1:
    print("No genes available for S3C multiplot.")
else:
    fig2, axes = plt.subplots(2, 3, figsize=(8.0, 6.0), dpi=400, sharex=True, sharey=True)
    axes = axes.flatten()

    fallback_color = "#1976d2"

    for idx, g in enumerate(ordered_genes[:6]):
        axp = axes[idx]
        col = colors.get(g, fallback_color)

        # Subset data
        df_g = df_s3c[df_s3c["gene"] == g]

        # siPum1: filled circles
        sub_pum1 = df_g[df_g["cond"] == "siPum1"]
        if not sub_pum1.empty:
            axp.scatter(
                sub_pum1["time"], sub_pum1["expr_norm"],
                s=40, facecolors=col, edgecolors="none",
                alpha=0.9, label="siPum1 (data)", zorder=3
            )

        # siNeg: open circles
        sub_neg = df_g[df_g["cond"] == "siNeg"]
        if not sub_neg.empty:
            axp.scatter(
                sub_neg["time"], sub_neg["expr_norm"],
                s=40, facecolors="none", edgecolors=col,
                linewidths=1.2, alpha=0.9,
                label="siNeg (data)", zorder=3
            )

        # Model curves: predicted expression = 1 - F(t)
        # siPum1 solid; siNeg dashed
        for cond_name, style in [("siPum1", "-"), ("siNeg", "--")]:
            row_fit = fit_s3c[(fit_s3c["gene"] == g) & (fit_s3c["cond"] == cond_name)]
            if not row_fit.empty:
                r = row_fit.iloc[0]
                yfrac = fraction_population(time_grid_s3c, r["t0"], r["lam"], r["k"], r["pi"])
                axp.plot(
                    time_grid_s3c, 1.0 - yfrac,
                    linestyle=style, color=col,
                    lw=2.0, alpha=0.95, label=f"{cond_name} (fit)"
                )

        axp.set_title(g, fontsize=10)
        axp.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.45)

        axp.set_xlim(0, 32)
        axp.set_ylim(0, 1.0)
        axp.set_xticks(np.arange(0, 32 + 1e-9, 8.0))
        axp.set_yticks(np.arange(0, 1.0 + 1e-9, 0.1))

        if idx == 0:
            axp.legend(fontsize=8, frameon=False, ncol=2)

    fig2.supxlabel("Time (hours)")
    fig2.supylabel("Normalized Expression")
    fig2.tight_layout()

    out_png2 = OUT_DIR / "leeb2014_S3C_siPum1_vs_siNeg.png"
    out_svg2 = OUT_DIR / "leeb2014_S3C_siPum1_vs_siNeg.svg"
    fig2.savefig(out_png2)
    fig2.savefig(out_svg2)
    print("Saved:", out_png2)
    print("Saved:", out_svg2)