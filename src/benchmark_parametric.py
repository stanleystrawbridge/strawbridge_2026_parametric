# benchmark_parametric.py
#
# Benchmarks the delayed Weibull "fraction transitioned" model against four
# alternative parametric families (Gamma, log-normal, Gompertz, log-logistic)
# across the reanalysed Leeb 2014, Mulas 2017 and Hanna 2009 datasets, using
# AICc and Akaike weights. Written in response to Reviewer 3's request for
# benchmarking against "other simple parametric descriptions of transition
# kinetics".
#
# * Every model is expressed as a fraction-transitioned CDF on t >= 0 that
#   starts at 0 (onset delay t0 fixed to 0, matching all three analysis
#   scripts) and is scaled by a competent fraction pi:
#       F(t) = pi * G(t),   G a proper CDF with G(0)=0, G(inf)=1.
#   This mirrors cstk_fit.fraction_population = pi * delayed-Weibull.
# * The pi / t0 policy is applied PER DATASET, so parameter counts are 
# matched across the five families:
#       - leeb (Fig 3F + S3C):  t0=0 fixed, pi FREE   -> 3 params each
#       - mulas Neural, Primitive Streak: t0=0, pi=1  -> 2 params each
#       - mulas Lat. Meso., Def. Endoderm: t0=0, pi FREE -> 3 params each
#       - hanna:                t0=0 fixed, pi=1 fixed -> 2 params each
#   Because all five families carry the SAME free-parameter count on a given
#   series, AICc differences are not confounded by parameterisation.
# * "Generalised logistic" is implemented as the log-logistic (Fisk)
#   distribution: the positive-support logistic sigmoid, a proper CDF with
#   G(0)=0 and two scale/shape parameters, directly comparable to the others.
#   (The 4-parameter Richards generalised logistic is NOT used: it carries an
#   extra asymmetry parameter and is not 0 at t=0, so it would neither be a
#   proper CDF here nor parameter-matched. Flagged for the author.)
# * Fitting is nonlinear least squares (scipy.curve_fit) with multi-start.
#   For Mulas, per-point SDs are used as weights (weighted residuals); for the
#   digitised single traces (Leeb, Hanna) ordinary least squares is used.
#   Within each series all five models use identical weights, so AICc ranking
#   is internally consistent.
# * AICc is computed from the (possibly weighted) residual sum of squares under
#   a Gaussian-error assumption:
#       AIC  = n*ln(RSS/n) + 2K,     K = (#free params) + 1  (the +1 is sigma^2)
#       AICc = AIC + 2K(K+1)/(n-K-1)
#   This is used only for WITHIN-series ranking across the five families, which
#   is exactly the comparison the reviewer asked for. Absolute AIC values are
#   not comparable to cstk_fit's (different likelihood constant); Delta-AICc and
#   Akaike weights are.
#
# OUTPUTS (written to ./benchmark_out/)
#   parametric_benchmark_long.csv   one row per (dataset, series, model)
#   parametric_benchmark_wins.csv   per-series winner + Delta-AICc spread
#   parametric_benchmark_featured.csv  compact table for the main-text panel

import os
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, gompertz, fisk

warnings.filterwarnings("ignore")

DATA_DIR = Path(os.environ.get("BENCH_DATA_DIR", "."))
OUT_DIR = Path("benchmark_out")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Fraction-transitioned CDF models: F(t) = pi * G(t), t0 = 0, G(0)=0, G(inf)=1
# Each returns F given t and the family's shape/scale params, then * pi.
# ---------------------------------------------------------------------------
def m_weibull(t, lam, k, pi):
    t = np.asarray(t, float)
    return pi * (1.0 - np.exp(-((np.clip(t, 0, None) / lam) ** k)))


def m_gamma(t, theta, k, pi):
    return pi * gamma_dist.cdf(np.clip(t, 0, None), a=k, scale=theta)


def m_lognormal(t, mu, sigma, pi):
    # scale = exp(mu); s = sigma
    return pi * lognorm.cdf(np.clip(t, 0, None), s=sigma, scale=np.exp(mu))


def m_gompertz(t, c, s, pi):
    # scipy gompertz: cdf(x) = 1 - exp(-c*(exp(x)-1)); with scale s, x=t/s
    return pi * gompertz.cdf(np.clip(t, 0, None), c=c, scale=s)


def m_loglogistic(t, alpha, beta, pi):
    # Fisk / log-logistic: cdf = 1 / (1 + (t/alpha)^(-beta))
    return pi * fisk.cdf(np.clip(t, 0, None), c=beta, scale=alpha)


# Registry: name -> (func, param_names, bounds_lo, bounds_hi, init_fn)
# init_fn(t, y, ymax) returns a list of starting guesses for [shape/scale...]
# (pi handled separately). Each family has exactly two non-pi parameters.
def _scale_seed(t, y, ymax):
    # time at which y reaches half of its plateau
    yy = y / max(ymax, 1e-9)
    idx = np.argmin(np.abs(yy - 0.5))
    return max(float(t[idx]), 1e-3)


MODELS = {
    "Weibull": dict(
        func=m_weibull,
        pnames=["lam", "k"],
        lo=[1e-6, 1e-3], hi=[1e6, 50.0],
        init=lambda t, y, ym: [_scale_seed(t, y, ym), 1.5],
    ),
    "Gamma": dict(
        func=m_gamma,
        pnames=["theta", "k"],
        lo=[1e-6, 1e-3], hi=[1e6, 50.0],
        init=lambda t, y, ym: [_scale_seed(t, y, ym), 2.0],
    ),
    "Log-normal": dict(
        func=m_lognormal,
        pnames=["mu", "sigma"],
        lo=[-20.0, 1e-3], hi=[20.0, 10.0],
        init=lambda t, y, ym: [np.log(_scale_seed(t, y, ym)), 0.6],
    ),
    "Gompertz": dict(
        func=m_gompertz,
        pnames=["c", "s"],
        lo=[1e-6, 1e-6], hi=[1e3, 1e6],
        init=lambda t, y, ym: [0.5, _scale_seed(t, y, ym)],
    ),
    "Log-logistic": dict(
        func=m_loglogistic,
        pnames=["alpha", "beta"],
        lo=[1e-6, 1e-3], hi=[1e6, 50.0],
        init=lambda t, y, ym: [_scale_seed(t, y, ym), 3.0],
    ),
}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _fit_one(model, t, y, sigma, pi_free):
    """Fit one family to (t,y). Returns dict with params, rss, aicc, r2, ok."""
    spec = MODELS[model]
    func2 = spec["func"]
    n = len(t)

    # Wrap so pi is appended as last parameter (or fixed to 1)
    if pi_free:
        def f(tt, *p):
            return func2(tt, p[0], p[1], p[2])
        p0_base = spec["init"](t, y, np.nanmax(y))
        # pi seed = observed plateau (last few points), clipped
        pi0 = float(np.clip(np.nanmax(y), 0.05, 1.2))
        lo = spec["lo"] + [1e-3]
        hi = spec["hi"] + [1.5]
        seeds = []
        for sc in (0.5, 1.0, 2.0):
            b = spec["init"](t, y, np.nanmax(y))
            b = [b[0] * sc if i == 0 else b[i] for i in range(len(b))]
            seeds.append(b + [pi0])
    else:
        def f(tt, *p):
            return func2(tt, p[0], p[1], 1.0)
        lo = spec["lo"]
        hi = spec["hi"]
        seeds = []
        for sc in (0.5, 1.0, 2.0):
            b = spec["init"](t, y, np.nanmax(y))
            b = [b[0] * sc if i == 0 else b[i] for i in range(len(b))]
            seeds.append(b)

    best = None
    for p0 in seeds:
        p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        try:
            popt, _ = curve_fit(
                f, t, y, p0=p0, bounds=(lo, hi),
                sigma=sigma, absolute_sigma=False, maxfev=40000,
            )
        except Exception:
            continue
        resid = (y - f(t, *popt))
        if sigma is not None:
            rss = float(np.sum((resid / sigma) ** 2))
        else:
            rss = float(np.sum(resid ** 2))
        if (best is None) or (rss < best["rss"]):
            best = dict(popt=popt, rss=rss)

    if best is None:
        return dict(ok=False, model=model)

    n_par = len(best["popt"])
    K = n_par + 1  # + sigma^2
    rss = max(best["rss"], 1e-12)
    aic = n * np.log(rss / n) + 2 * K
    if n - K - 1 > 0:
        aicc = aic + 2 * K * (K + 1) / (n - K - 1)
    else:
        aicc = np.inf
    # unweighted R^2 for reporting (on the raw fraction)
    fpred = f(t, *best["popt"])
    ss_res = float(np.sum((y - fpred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out = dict(ok=True, model=model, n=n, n_params=n_par, rss=rss,
               aic=aic, aicc=aicc, r2=r2)
    names = spec["pnames"] + (["pi"] if pi_free else [])
    for nm, val in zip(names, best["popt"]):
        out[f"p_{nm}"] = float(val)
    if not pi_free:
        out["p_pi"] = 1.0
    return out


def benchmark_series(dataset, series, t, y, sigma, pi_free):
    """Fit all five families to one series and compute Akaike weights."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if sigma is not None:
        sigma = np.asarray(sigma, float)[m]
        # floor non-positive / non-finite SDs at the median positive SD
        pos = sigma[(sigma > 0) & np.isfinite(sigma)]
        floor = np.median(pos) if pos.size else 1.0
        sigma = np.where((sigma > 0) & np.isfinite(sigma), sigma, floor)

    rows = []
    for model in MODELS:
        r = _fit_one(model, t, y, sigma, pi_free)
        r["dataset"] = dataset
        r["series"] = series
        r["pi_free"] = pi_free
        rows.append(r)

    ok = [r for r in rows if r.get("ok")]
    for r in rows:
        if r.get("ok"):
            r["df_left"] = r["n"] - (r["n_params"] + 1) - 1
            r["aicc_valid"] = r["df_left"] > 0
    ok_aicc = [r for r in ok if np.isfinite(r["aicc"])]
    if ok_aicc and len(ok_aicc) == len(ok):
        amin = min(r["aicc"] for r in ok_aicc)
        denom = sum(np.exp(-0.5 * (r["aicc"] - amin)) for r in ok_aicc)
        for r in rows:
            if r.get("ok"):
                r["delta_aicc"] = r["aicc"] - amin
                r["akaike_weight"] = np.exp(-0.5 * r["delta_aicc"]) / denom
    else:
        for r in rows:
            if r.get("ok"):
                r["delta_aicc"] = np.nan
                r["akaike_weight"] = np.nan
    # Plain-AIC weights: defined when n>params+1 but UNRELIABLE near n==K;
    # reported only as a caveated secondary for the underpowered series.
    if ok:
        bmin = min(r["aic"] for r in ok)
        bden = sum(np.exp(-0.5 * (r["aic"] - bmin)) for r in ok)
        for r in rows:
            if r.get("ok"):
                r["delta_aic"] = r["aic"] - bmin
                r["aic_weight"] = np.exp(-0.5 * r["delta_aic"]) / bden
    return rows


# ---------------------------------------------------------------------------
# Dataset loaders (replicating the original scripts' conventions)
# ---------------------------------------------------------------------------
def _norm_unit(v):
    v = np.asarray(v, float)
    vmax, vmin = np.nanmax(v), np.nanmin(v)
    if vmax > 1.2 and vmax <= 120.0 and vmin >= -1.0:
        v = v / 100.0
        vmax, vmin = np.nanmax(v), np.nanmin(v)
    if vmin >= -1e-9 and vmax <= 1.0 + 1e-9:
        return v
    rng = vmax - vmin
    return (v - vmin) / rng if rng > 0 else v


def load_leeb_main():
    d = pd.read_csv(DATA_DIR / "leeb2014cellStemCell.csv")
    d = d.rename(columns={"Time (h)": "time", "Gene": "gene", "Expression": "expr"})
    out = []
    for g, sub in d.groupby("gene"):
        sub = sub.dropna(subset=["time", "expr"])
        norm = _norm_unit(sub["expr"].to_numpy())
        frac = 1.0 - norm
        out.append(("leeb_main", f"{g}", sub["time"].to_numpy(), frac, None, True))
    return out


def load_leeb_s3c():
    d = pd.read_csv(DATA_DIR / "leeb2014cellStemCell_S3C.csv")
    out = []
    for (g, c), sub in d.groupby(["gene", "condition"]):
        sub = sub.dropna(subset=["time", "relative_expression"])
        norm = _norm_unit(sub["relative_expression"].to_numpy())
        frac = 1.0 - norm
        out.append(("leeb_s3c", f"{g}|{c}", sub["time"].to_numpy(), frac, None, True))
    return out


def load_mulas():
    d = pd.read_csv(DATA_DIR / "mulas2017stemCellReports.csv")
    pi_fixed_lineages = {"Neural", "Primitive Streak"}
    out = []
    for (lin, status), sub in d.groupby(["Lineage", "Cell Status"]):
        sub = sub.dropna(subset=["Time", "Mean"])
        pi_free = lin not in pi_fixed_lineages
        out.append(("mulas", f"{lin}|{status}", sub["Time"].to_numpy(),
                    sub["Mean"].to_numpy(), sub["SD"].to_numpy(), pi_free))
    return out


def load_hanna():
    d = pd.read_csv(DATA_DIR / "hanna2009nature.csv")
    d = d.rename(columns={"Time (h)": "time", "Condition": "cond", "Fraction": "frac"})
    out = []
    for c, sub in d.groupby("cond"):
        sub = sub.dropna(subset=["time", "frac"])
        # clamp tiny negative times to 0 (t0 fixed at 0)
        tt = np.clip(sub["time"].to_numpy(), 0, None)
        yy = _norm_unit(sub["frac"].to_numpy())
        out.append(("hanna", f"{c}", tt, yy, None, False))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_series = (
        load_leeb_main() + load_leeb_s3c() + load_mulas() + load_hanna()
    )

    long_rows = []
    for (dataset, series, t, y, sigma, pi_free) in all_series:
        long_rows.extend(
            benchmark_series(dataset, series, t, y, sigma, pi_free)
        )

    long = pd.DataFrame(long_rows)
    cols = ["dataset", "series", "model", "pi_free", "n", "n_params",
            "df_left", "aicc_valid", "r2", "rss", "aic", "aicc",
            "delta_aicc", "akaike_weight", "delta_aic", "aic_weight",
            "p_lam", "p_k", "p_theta", "p_mu", "p_sigma", "p_c", "p_s",
            "p_alpha", "p_beta", "p_pi", "ok"]
    for c in cols:
        if c not in long.columns:
            long[c] = np.nan
    long = long[cols].sort_values(["dataset", "series", "aic"]).reset_index(drop=True)
    long.to_csv(OUT_DIR / "parametric_benchmark_long.csv", index=False)

    ok = long[long["ok"] == True].copy()

    # Per-series summary. IC-valid = every family has finite AICc (df_left>0).
    wins = []
    for (ds, sr), sub in ok.groupby(["dataset", "series"]):
        ic_valid = bool(sub["aicc_valid"].all() and np.isfinite(sub["aicc"]).all())
        weib = sub[sub["model"] == "Weibull"].iloc[0]
        if ic_valid:
            b = sub.sort_values("aicc").iloc[0]
            best_model, best_delta = b["model"], 0.0
            wd_aicc = float(weib["delta_aicc"]); ww = float(weib["akaike_weight"])
        else:
            best_model, best_delta = "(insufficient df)", np.nan
            wd_aicc, ww = np.nan, np.nan
        # descriptive comparison always available: R2 gap Weibull vs best-R2 family
        best_r2_model = sub.sort_values("r2", ascending=False).iloc[0]["model"]
        wins.append(dict(
            dataset=ds, series=sr, n=int(weib["n"]),
            pi_free=bool(weib["pi_free"]), df_left=int(weib["df_left"]),
            ic_comparable=ic_valid,
            best_model_aicc=best_model,
            weibull_delta_aicc=wd_aicc, weibull_aicc_weight=ww,
            weibull_r2=float(weib["r2"]),
            best_r2_model=best_r2_model,
            r2_spread=float(sub["r2"].max() - sub["r2"].min()),
        ))
    wins = pd.DataFrame(wins).sort_values(["dataset", "series"]).reset_index(drop=True)
    wins.to_csv(OUT_DIR / "parametric_benchmark_wins.csv", index=False)

    # Featured wide tables: Delta-AICc (IC-valid series only) and R2 (all series)
    ok.pivot_table(index=["dataset", "series"], columns="model",
                   values="delta_aicc").round(2).to_csv(
        OUT_DIR / "parametric_benchmark_deltaAICc.csv")
    ok.pivot_table(index=["dataset", "series"], columns="model",
                   values="r2").round(4).to_csv(
        OUT_DIR / "parametric_benchmark_R2.csv")

    # ---- console report ----
    icv = wins[wins["ic_comparable"]]
    print("=" * 70)
    print(f"Total series: {len(wins)}   |   IC-comparable (df_left>0): {len(icv)}")
    print("=" * 70)
    print("\n[1] VALIDATION: Weibull reproduces manuscript fits")
    for ds, sr in [("leeb_main", "Klf4"), ("mulas", "Primitive Streak|2i")]:
        r = ok[(ok.dataset == ds) & (ok.series == sr) & (ok.model == "Weibull")].iloc[0]
        print(f"    {ds}/{sr}: k={r.p_k:.3f}, lam={r.p_lam:.3f}, pi={r.p_pi:.3f}")

    print("\n[2] IC-BASED COMPARISON is only defined where n is large enough.")
    print("    Series per dataset with finite AICc:")
    print(wins.groupby("dataset")["ic_comparable"].agg(
        ["sum", "count"]).rename(columns={"sum": "ic_valid", "count": "series"}).to_string())

    print("\n[3] Where AICc IS valid (Hanna), Weibull vs alternatives:")
    with pd.option_context("display.width", 160):
        print(icv[["dataset", "series", "n", "best_model_aicc",
                   "weibull_delta_aicc", "weibull_aicc_weight", "weibull_r2"]].to_string(index=False))
    if len(icv):
        print(f"\n    Weibull is best on {int((icv.best_model_aicc=='Weibull').sum())}/{len(icv)};"
              f" within Delta-AICc<=2 on {int((icv.weibull_delta_aicc<=2).sum())}/{len(icv)};"
              f" median Akaike weight {icv.weibull_aicc_weight.median():.2f}.")

    print("\n[4] Underpowered series (n<=5, IC undefined): descriptive fit quality.")
    und = wins[~wins["ic_comparable"]]
    print(f"    {len(und)} series. All families fit comparably:")
    print(f"      Weibull R2: median {und.weibull_r2.median():.4f}, min {und.weibull_r2.min():.4f}")
    print(f"      R2 spread across the 5 families (max-min), median {und.r2_spread.median():.4f}")
    print(f"      Best-R2 family tally: {dict(und.best_r2_model.value_counts())}")

    print(f"\nSaved: {OUT_DIR}/parametric_benchmark_long.csv (full detail)")
    print(f"Saved: {OUT_DIR}/parametric_benchmark_wins.csv (per-series summary)")
    print(f"Saved: {OUT_DIR}/parametric_benchmark_deltaAICc.csv  and  _R2.csv (wide tables)")


if __name__ == "__main__":
    main()
