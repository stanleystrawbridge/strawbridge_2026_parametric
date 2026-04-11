# make_parameter_figures.py
#
# Save this script in:
# H:\Shared drives\strawbridge_lab\projects\smith_group\networkFailureAnalysis\src
#
# It will create:
# H:\Shared drives\strawbridge_lab\projects\smith_group\networkFailureAnalysis\src\parameterFigures
#
# and save PNG, SVG, and TXT summary files for panels B-E.
#
# Updated version:
# - no legend
# - common x-range across all panels
# - common y-range
# - slightly refined parameter values
# - more informative k sampling
# - same blue->orange gradient style
# - Arial font, grid on, no title, no axis labels

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex


# ---------------------------
# Style
# ---------------------------
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1.0


# ---------------------------
# Delayed Weibull CDF
# ---------------------------
def delayed_weibull_cdf(t, t0, lam, k, pi):
    """
    Delayed Weibull CDF with competence fraction pi.

    F(t) = 0,                                   t < t0
         = pi * [1 - exp(-((t - t0)/lam)^k)],  t >= t0
    """
    t = np.asarray(t)
    F = np.zeros_like(t, dtype=float)

    mask = t >= t0
    F[mask] = pi * (1.0 - np.exp(-((t[mask] - t0) / lam) ** k))

    return F


# ---------------------------
# Color utilities
# ---------------------------
def interpolate_colors(color1, color2, n):
    """
    Linear interpolation between two RGB colors.
    """
    color1 = np.array(color1, dtype=float)
    color2 = np.array(color2, dtype=float)

    if n == 1:
        return [tuple(color1)]

    colors = []
    for i in range(n):
        a = i / (n - 1)
        c = (1 - a) * color1 + a * color2
        colors.append(tuple(c))
    return colors


# MATLAB default blue and orange
MATLAB_BLUE = (0.0000, 0.4470, 0.7410)
MATLAB_ORANGE = (0.8500, 0.3250, 0.0980)


# ---------------------------
# Plot/save helper
# ---------------------------
def save_parameter_figure(
    output_dir,
    base_name,
    varied_name,
    varied_values,
    fixed_params,
    t_max,
    n_t=1400,
):
    """
    Make one figure, save PNG/SVG, and write TXT summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    t = np.linspace(0, t_max, n_t)
    colors = interpolate_colors(MATLAB_BLUE, MATLAB_ORANGE, len(varied_values))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    # Plot curves
    for value, color in zip(varied_values, colors):
        params = fixed_params.copy()
        params[varied_name] = value

        F = delayed_weibull_cdf(
            t=t,
            t0=params["t0"],
            lam=params["lambda"],
            k=params["k"],
            pi=params["pi"],
        )

        ax.plot(t, F, color=color, linewidth=2.8)

    # Axes style
    ax.grid(True)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.05)

    # No title, no axis labels
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Keep ticks
    ax.tick_params(direction="out", length=4, width=1)

    # Make x ticks uniform and readable
    ax.set_xticks(np.arange(0, t_max + 0.001, 1.0))
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    fig.tight_layout()

    # Save figure
    png_path = os.path.join(output_dir, f"{base_name}.png")
    svg_path = os.path.join(output_dir, f"{base_name}.svg")
    txt_path = os.path.join(output_dir, f"{base_name}_summary.txt")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    # Write summary text file
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Figure: {base_name}\n")
        f.write(f"Varied parameter: {varied_name}\n\n")

        f.write("Fixed parameters:\n")
        for key, value in fixed_params.items():
            if key != varied_name:
                f.write(f"  {key} = {value}\n")

        f.write("\nVaried values:\n")
        for i, (value, color) in enumerate(zip(varied_values, colors), start=1):
            hex_color = to_hex(color)
            f.write(f"  Curve {i}: {varied_name} = {value}, color = {hex_color}\n")

        f.write("\nTime range:\n")
        f.write(f"  t from 0 to {t_max}\n")
        f.write(f"  number of time points = {n_t}\n")

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {txt_path}")


# ---------------------------
# Main
# ---------------------------
def main():
    # Output folder beside this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "parameterFigures")

    # Use the same x-range for all panels for easier comparison
    common_t_max = 6.0

    # -----------------------
    # Panel B: vary t0
    # fixed: lambda = 1, k = 1, pi = 1
    # -----------------------
    t0_values = [0.0, 0.3, 0.6, 1.0, 1.4, 2.0]
    save_parameter_figure(
        output_dir=output_dir,
        base_name="B_vary_t0",
        varied_name="t0",
        varied_values=t0_values,
        fixed_params={"t0": 0.0, "lambda": 1.0, "k": 1.0, "pi": 1.0},
        t_max=common_t_max,
    )

    # -----------------------
    # Panel C: vary lambda
    # fixed: t0 = 0, k = 1, pi = 1
    # -----------------------
    lambda_values = [0.5, 0.75, 1.0, 1.4, 2.0, 2.8]
    save_parameter_figure(
        output_dir=output_dir,
        base_name="C_vary_lambda",
        varied_name="lambda",
        varied_values=lambda_values,
        fixed_params={"t0": 0.0, "lambda": 1.0, "k": 1.0, "pi": 1.0},
        t_max=common_t_max,
    )

    # -----------------------
    # Panel D: vary k
    # fixed: t0 = 0, lambda = 1, pi = 1
    # -----------------------
    k_values = [0.4, 0.6, 0.8, 1.0, 1.3, 1.8, 3.0]
    save_parameter_figure(
        output_dir=output_dir,
        base_name="D_vary_k",
        varied_name="k",
        varied_values=k_values,
        fixed_params={"t0": 0.0, "lambda": 1.0, "k": 1.0, "pi": 1.0},
        t_max=common_t_max,
    )

    # -----------------------
    # Panel E: vary pi
    # fixed: t0 = 0, lambda = 1, k = 1
    # -----------------------
    pi_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    save_parameter_figure(
        output_dir=output_dir,
        base_name="E_vary_pi",
        varied_name="pi",
        varied_values=pi_values,
        fixed_params={"t0": 0.0, "lambda": 1.0, "k": 1.0, "pi": 1.0},
        t_max=common_t_max,
    )


if __name__ == "__main__":
    main()