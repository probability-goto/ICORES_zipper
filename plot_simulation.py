"""
Produce the same 6 theory plots (3 for lambda1 sweeps, 3 for lambda2 sweeps)
and overlay simulation points (default 1.1M events, first 100k discarded as
warm-up, 1M used) on the theoretical curves.
The layout is 3 rows x 2 columns: left column = lambda1 sweeps
(E[L_main], E[L_merge]), right column = lambda2 sweeps.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

from qbd_model import DynamicThresholdQBD
from stationary_analysis import compute_stationary, calculate_performance_measures
from simulation import simulate

# import helper from plot_results to compute lambda1_max exactly the same way
from plot_results import _lambda1_max_for_params as compute_lambda1_max_for_params, make_model as plot_results_make_model


# constants used in plot_results.py (keep same values for consistency)
MU1_RAND = 40.0
MU2_RAND = 30.0
MU1_ZIP = 30.0
MU2_ZIP = 20.0
P = 0.8


def run_sim_for_points(points: Sequence[float], lam_other: float, sweep_type: str, K: int, m: int | None, p: float, sim_max_events: int, sim_warmup: int, seed_base: int):
    """Run simulation for a list of points.

    sweep_type: 'lambda1' or 'lambda2'. lam_other is the fixed other lambda.
    Returns dict point -> (E_main, E_merge, P_block)
    """
    res = {}
    for i, point in enumerate(points):
        if sweep_type == 'lambda1':
            lam1 = float(point)
            lam2 = float(lam_other)
        else:
            lam1 = float(lam_other)
            lam2 = float(point)

        try:
            sim_out = simulate(lam1, lam2, MU1_RAND, MU2_RAND, MU1_ZIP, MU2_ZIP, K, m if m is not None else max(1, int(round(K * 0.6))), p, max_events=sim_max_events, warmup_events=sim_warmup, seed=seed_base + i)
            res[point] = sim_out[:3]
        except Exception:
            res[point] = (np.nan, np.nan, np.nan)
    return res


def plot_6_theory_vs_sim(
    Ks=(5, 10, 15),
    lambda2_fixed_for_1_3: float = 10.0,
    lambda1_fixed_for_4_6: float = 10.0,
    sim_max_events: int = 1_100_000,
    sim_warmup: int = 100_000,
    seed_base: int = 123,
    num_sim_points: int = 20,
):
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 17,
        'legend.fontsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
    })
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # --- Left column: lambda1 sweeps for each K ---
    for row, K in enumerate(Ks):
        ax = axes[row, 0]
        # compute lambda1 grid same as plot_results.py, ensure at least 30 theory points
        lambda1_max = compute_lambda1_max_for_params(lambda2_fixed_for_1_3, K)
        num_theory_points = max(30, num_sim_points)
        if lambda1_max <= 0:
            lambda1_vals = np.linspace(0.1, 10, num_theory_points)
        else:
            lambda1_vals = np.linspace(0.1, max(0.5, 0.98 * lambda1_max), num_theory_points)

        E_main_vals = []
        E_merge_vals = []
        for lam1 in lambda1_vals:
            try:
                model = plot_results_make_model(lam1, lambda2_fixed_for_1_3, K)
                R = model.solve_R()
                pi0, pi1, _ = compute_stationary(model, R=R)
                E_main, E_merge, _ = calculate_performance_measures(model, R, pi0, pi1)
            except Exception:
                E_main, E_merge = np.nan, np.nan
            E_main_vals.append(E_main)
            E_merge_vals.append(E_merge)

        ax.plot(lambda1_vals, E_main_vals, label=r"Theory: $E[L_{main}]$", color="C0")
        ax.plot(lambda1_vals, E_merge_vals, label=r"Theory: $E[L_{merge}]$", color="C1")
        if lambda1_max > 0:
            ax.axvline(lambda1_max, color="k", linestyle="--", label=r"$\lambda_{1,max}$")
        ax.set_title(f"K={K}, $\\lambda_2$={lambda2_fixed_for_1_3}")
        ax.set_xlabel(r"$\lambda_1$")
        ax.set_ylabel(r"$E[L]$")
        ax.grid(True)

        # choose up to `num_sim_points` evenly spaced points along the lambda1 grid
        n_pts = min(num_sim_points, len(lambda1_vals))
        if n_pts <= 0:
            sim_points = []
        else:
            idxs = np.linspace(0, len(lambda1_vals) - 1, n_pts, dtype=int)
            # unique preserves order after sorting indices
            unique_idxs = np.unique(idxs)
            sim_points = [float(lambda1_vals[i]) for i in unique_idxs]
        # report how many theory/sim points will be used for this subplot
        print(f"[LEFT] K={K}, lambda2={lambda2_fixed_for_1_3}: theory_points={len(lambda1_vals)}, sim_points={len(sim_points)}")
        sim_res = run_sim_for_points(sim_points, lambda2_fixed_for_1_3, 'lambda1', K, None, P, sim_max_events, sim_warmup, seed_base + row * 10)

        # overlay simulation markers
        for j, lam1 in enumerate(sim_points):
            E_main_s, E_merge_s, _ = sim_res[lam1]
            ax.scatter([lam1], [E_main_s], marker='o', color='C0', edgecolor='k', zorder=5, label='Sim: E[L_{main}]' if j == 0 and row == 0 else "")
            ax.scatter([lam1], [E_merge_s], marker='s', color='C1', edgecolor='k', zorder=5, label='Sim: E[L_{merge}]' if j == 0 and row == 0 else "")
        if row == 0:
            ax.legend()

    # --- Right column: lambda2 sweeps for each K ---
    # ensure at least 30 theory points for lambda2 sweeps
    num_theory_points2 = max(30, num_sim_points)
    lambda2_vals = np.linspace(1, 30, num_theory_points2)
    for row, K in enumerate(Ks):
        ax = axes[row, 1]
        E_main_vals = []
        E_merge_vals = []
        for lam2 in lambda2_vals:
            try:
                model = plot_results_make_model(lambda1_fixed_for_4_6, lam2, K)
                R = model.solve_R()
                pi0, pi1, _ = compute_stationary(model, R=R)
                E_main, E_merge, _ = calculate_performance_measures(model, R, pi0, pi1)
            except Exception:
                E_main, E_merge = np.nan, np.nan
            E_main_vals.append(E_main)
            E_merge_vals.append(E_merge)

        ax.plot(lambda2_vals, E_main_vals, label=r"Theory: $E[L_{main}]$", color="C0")
        ax.plot(lambda2_vals, E_merge_vals, label=r"Theory: $E[L_{merge}]$", color="C1")
        ax.set_title(f"K={K}, $\\lambda_1$={lambda1_fixed_for_4_6}")
        ax.set_xlabel(r"$\lambda_2$")
        ax.set_ylabel(r"$E[L]$")
        ax.grid(True)

        # pick up to `num_sim_points` lambda2 points evenly spaced for simulation
        n_pts = min(num_sim_points, len(lambda2_vals))
        if n_pts <= 0:
            sim_points = []
        else:
            idxs = np.linspace(0, len(lambda2_vals) - 1, n_pts, dtype=int)
            unique_idxs = np.unique(idxs)
            sim_points = [float(lambda2_vals[i]) for i in unique_idxs]
        # report how many theory/sim points will be used for this subplot
        print(f"[RIGHT] K={K}, lambda1={lambda1_fixed_for_4_6}: theory_points={len(lambda2_vals)}, sim_points={len(sim_points)}")
        sim_res = run_sim_for_points(sim_points, lambda1_fixed_for_4_6, 'lambda2', K, None, P, sim_max_events, sim_warmup, seed_base + row * 10)

        for j, lam2 in enumerate(sim_points):
            E_main_s, E_merge_s, _ = sim_res[lam2]
            ax.scatter([lam2], [E_main_s], marker='o', color='C0', edgecolor='k', zorder=5, label='Sim: E[L_{main}]' if j == 0 and row == 0 else "")
            ax.scatter([lam2], [E_merge_s], marker='s', color='C1', edgecolor='k', zorder=5, label='Sim: E[L_{merge}]' if j == 0 and row == 0 else "")
        if row == 0:
            ax.legend()

    plt.tight_layout()
    outname = 'fig_results_sim_vs_theory.png'
    fig.savefig(outname, dpi=200)
    print('Saved', outname)


if __name__ == '__main__':
    plot_6_theory_vs_sim()
