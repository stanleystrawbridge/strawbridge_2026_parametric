# build_model_comparison_figure.py
#
# Main-text visual model comparison: representative fits of all
# five parametric families overlaid on the data, spanning the full range of
# information-criterion outcomes. Reuses the exact fits from
# benchmark_parametric.py, so curves match Supplemental Table S4.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("BENCH_DATA_DIR", os.path.join(_SCRIPT_DIR, "data"))
import benchmark_parametric as bp

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "mathtext.default": "regular",  # render $\Delta$AICc, $t_0$, $\pi$ in the sans font
    "font.size": 11,
    "axes.linewidth": 1.0,
    "svg.fonttype": "none",
})

ALL = {}
for rec in (bp.load_leeb_main() + bp.load_leeb_s3c() + bp.load_mulas() + bp.load_hanna()):
    d, s, t, y, sg, pf = rec
    ALL[(d, s)] = (np.asarray(t, float), np.asarray(y, float),
                   (np.asarray(sg, float) if sg is not None else None), pf)

STYLE = {
    "Weibull":      dict(color="black",   ls="-",  lw=2.6, z=5),
    "Gamma":        dict(color="#1f77b4", ls="--", lw=1.6, z=3),
    "Log-normal":   dict(color="#ff7f0e", ls="-.", lw=1.6, z=3),
    "Gompertz":     dict(color="#2ca02c", ls=":",  lw=2.2, z=4),
    "Log-logistic": dict(color="#9467bd", ls=(0, (5, 1)), lw=1.6, z=3),
}
ORDER = ["Weibull", "Gamma", "Log-normal", "Gompertz", "Log-logistic"]


def curve(name, row, tg):
    spec = bp.MODELS[name]
    args = [row[f"p_{pn}"] for pn in spec["pnames"]] + [row.get("p_pi", 1.0)]
    return spec["func"](tg, *args)


def panel(ax, dataset, series, title):
    t, y, sigma, pi_free = ALL[(dataset, series)]
    m = np.isfinite(t) & np.isfinite(y); t, y = t[m], y[m]
    sig = sigma[m] if sigma is not None else None
    rows = {r["model"]: r for r in bp.benchmark_series(dataset, series, t, y, sig, pi_free)
            if r.get("ok")}
    tg = np.linspace(0, t.max() * 1.05 + 1e-9, 400)
    for name in ORDER:
        if name in rows:
            st = STYLE[name]
            ax.plot(tg, curve(name, rows[name], tg), color=st["color"], ls=st["ls"],
                    lw=st["lw"], zorder=st["z"])
    if sig is not None:
        ax.errorbar(t, y, yerr=sig, fmt="o", ms=4.5, color="0.25", ecolor="0.6",
                    elinewidth=1, capsize=2, zorder=6)
    else:
        ax.plot(t, y, "o", ms=4.5, color="0.25", zorder=6)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlim(0, tg[-1]); ax.set_ylim(-0.03, 1.06); ax.grid(True, alpha=0.3)

    ic = all(r.get("aicc_valid", False) for r in rows.values())
    if ic:
        lines = [r"$\Delta$AICc"]
        for name in ORDER:
            dv = rows[name].get("delta_aicc", np.nan)
            if np.isfinite(dv):
                lines.append(f"{name[:4]}: {dv:.1f}")
        ax.text(0.035, 0.965, "\n".join(lines), transform=ax.transAxes, ha="left",
                va="top", fontsize=7.4,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    else:
        ax.text(0.035, 0.965, f"n = {len(t)}\nAICc undefined\n(indistinguishable)",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.4, color="0.35",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))


fig, ax = plt.subplots(2, 3, figsize=(13.2, 7.4))

# Top row: Hanna (AICc well posed) spanning win / tie / loss for the Weibull
panel(ax[0, 0], "hanna", "NGFP1",         "Reprogramming: NGFP1 (Hanna)\nWeibull best-supported")
panel(ax[0, 1], "hanna", "NGFP1-p21KD",   "Reprogramming: p21-KD (Hanna)\nWeibull \u2248 Gompertz (tied)")
panel(ax[0, 2], "hanna", "NGFP1-Lin28OE", "Reprogramming: Lin28-OE (Hanna)\nGompertz best-supported")
# Bottom row: sparse series across modalities (AICc undefined; families overlap)
panel(ax[1, 0], "leeb_main", "Klf4",                     "Bulk qPCR: Klf4 (Leeb)")
panel(ax[1, 1], "mulas", "Primitive Streak|2i",          "Directed diff.: Prim. streak, 2i (Mulas)")
panel(ax[1, 2], "mulas", "Lateral Mesoderm|Rex1-Hi",     "Directed diff.: Lat. mesoderm, Rex1-Hi (Mulas)")

for a in ax[:, 0]:
    a.set_ylabel("Transitioned fraction")
for a in ax[1, :]:
    a.set_xlabel("Time")

handles = ([Line2D([0], [0], marker="o", ls="none", ms=5, color="0.25", label="Data")]
           + [Line2D([0], [0], color=STYLE[n]["color"], ls=STYLE[n]["ls"],
                     lw=STYLE[n]["lw"], label=n) for n in ORDER])
fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, 1.005))
fig.tight_layout(rect=[0, 0, 1, 0.955])
_OUTDIR = os.path.join(_SCRIPT_DIR, "modelComparisonFigures")
os.makedirs(_OUTDIR, exist_ok=True)
fig.savefig(os.path.join(_OUTDIR, "figure_model_comparison.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(_OUTDIR, "figure_model_comparison.pdf"), bbox_inches="tight")
print(f"saved 2x3 figure to {_OUTDIR}")
