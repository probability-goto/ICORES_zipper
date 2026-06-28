"""
論文の数値実験に相当する図を出力するスクリプト

注意: ループ内で行列反復や固有値計算を多数回行うため計算量が大きくなります。
      実行には時間がかかる可能性があるため、進捗表示に `tqdm` を使用します。

出力: `fig_results.png` に 8 個のサブプロットを保存します。
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from qbd_model import DynamicThresholdQBD
from stability_analysis import compute_exact_lambda1_max
from stationary_analysis import compute_stationary, calculate_performance_measures


# ベースラインパラメータ
MU1_RAND = 40
MU2_RAND = 30
MU1_ZIP = 30
MU2_ZIP = 20
P = 0.8


def make_model(lam1, lam2, K, m=None, p=P):
    """モデルを生成するヘルパー。m が None の場合は m = int(K*0.6) を使う。"""
    if m is None:
        m = max(1, int(round(K * 0.6)))
    return DynamicThresholdQBD(
        lam1=lam1,
        lam2=lam2,
        mu1_rand=MU1_RAND,
        mu2_rand=MU2_RAND,
        mu1_zip=MU1_ZIP,
        mu2_zip=MU2_ZIP,
        K=K,
        m=m,
        p=p,
    )


def _lambda1_max_for_params(lambda2, K, m=None, p=P):
    """与えられたパラメータに対して lambda1_max を返す。"""
    return compute_exact_lambda1_max(make_model(1.0, lambda2, K, m=m, p=p))


def plot_results():
    # グラフ設定
    Ks = [5, 10, 15]
    lambda2_fixed_for_1_3 = 10
    lambda1_fixed_for_4_6 = 10

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    # --- グラフ 1-3: lambda1 に対する E[L] (lambda2=10) ---
    for i, K in enumerate(Ks):
        ax = axes[i]
        lam2 = lambda2_fixed_for_1_3
        print(f"Computing lambda1_max for K={K}, lambda2={lam2}...")
        lambda1_max = _lambda1_max_for_params(lam2, K)

        if lambda1_max <= 0:
            lambda1_vals = np.linspace(0.1, 10, 20)
        else:
            lambda1_vals = np.linspace(0.1, max(0.5, 0.98 * lambda1_max), 20)

        E_main_vals = []
        E_merge_vals = []

        for lam1 in tqdm(lambda1_vals, desc=f"K={K} lambda1 sweep", leave=False):
            try:
                model = make_model(lam1, lam2, K)
                R = model.solve_R()
                pi0, pi1, _ = compute_stationary(model, R)
                E_main, E_merge, P_block = calculate_performance_measures(model, R, pi0, pi1)
            except Exception:
                E_main, E_merge = np.nan, np.nan
            E_main_vals.append(E_main)
            E_merge_vals.append(E_merge)

        ax.plot(lambda1_vals, E_main_vals, label=r"$E[L_{main}]$")
        ax.plot(lambda1_vals, E_merge_vals, label=r"$E[L_{merge}]$")
        if lambda1_max > 0:
            ax.axvline(lambda1_max, color="k", linestyle="--", label=r"$\lambda_{1,max}$")
        ax.set_title(f"K={K}, $\\lambda_2$={lam2}")
        ax.set_xlabel(r"$\lambda_1$")
        ax.set_ylabel(r"$E[L]$")
        ax.grid(True)
        ax.legend()

    # --- グラフ 4-6: lambda2 に対する E[L] (lambda1=10) ---
    lambda2_vals = np.linspace(1, 30, 30)
    for j, K in enumerate(Ks):
        ax = axes[3 + j]
        E_main_vals = []
        E_merge_vals = []
        for lam2 in tqdm(lambda2_vals, desc=f"K={K} lambda2 sweep", leave=False):
            try:
                model = make_model(10.0, lam2, K)
                R = model.solve_R()
                pi0, pi1, _ = compute_stationary(model, R)
                E_main, E_merge, P_block = calculate_performance_measures(model, R, pi0, pi1)
            except Exception:
                E_main, E_merge = np.nan, np.nan
            E_main_vals.append(E_main)
            E_merge_vals.append(E_merge)

        ax.plot(lambda2_vals, E_main_vals, label=r"$E[L_{main}]$")
        ax.plot(lambda2_vals, E_merge_vals, label=r"$E[L_{merge}]$")
        ax.set_title(f"K={K}, $\\lambda_1$=10")
        ax.set_xlabel(r"$\lambda_2$")
        ax.set_ylabel(r"$E[L]$")
        ax.grid(True)
        ax.legend()

    # --- グラフ 7: lambda2 に対する lambda1_max ---
    ax7 = axes[6]
    lambda2_range = np.arange(1, 31)
    for K in Ks:
        lambda1_max_list = []
        for lam2 in tqdm(lambda2_range, desc=f"compute lambda1_max K={K}", leave=False):
            try:
                val = compute_lambda1_max_for_params(lam2, K)
            except Exception:
                val = np.nan
            lambda1_max_list.append(val)
        ax7.plot(lambda2_range, lambda1_max_list, label=f"K={K}")
    ax7.set_title(r"$\lambda_{1,max}$ vs $\lambda_2$")
    ax7.set_xlabel(r"$\lambda_2$")
    ax7.set_ylabel(r"$\lambda_{1,max}$")
    ax7.grid(True)
    ax7.legend()

    # --- グラフ 8: K に対する lambda1_max と P_block の等高線的境界 ---
    ax8 = axes[7]
    Ks_full = np.arange(5, 31)
    lambda1_max_K = []
    # 閾値リスト
    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20]
    threshold_curves = {t: [] for t in thresholds}

    for K in tqdm(Ks_full, desc="compute for K range", leave=False):
        try:
            lam1_max = _lambda1_max_for_params(10.0, K)
        except Exception:
            lam1_max = np.nan
        lambda1_max_K.append(lam1_max)

        # P_block の境界を探索（安定領域内での近似探索）
        if np.isnan(lam1_max) or lam1_max <= 0:
            for t in thresholds:
                threshold_curves[t].append(np.nan)
            continue

        lam1_grid = np.linspace(0.1, max(0.5, 0.98 * lam1_max), 20)
        P_blocks = []
        for lam1 in lam1_grid:
            try:
                model = make_model(lam1, 10.0, K)
                R = model.solve_R()
                pi0, pi1, _ = compute_stationary(model, R)
                _, _, P_block = calculate_performance_measures(model, R, pi0, pi1)
            except Exception:
                P_block = np.nan
            P_blocks.append(P_block)

        P_blocks = np.array(P_blocks)
        # 各閾値について、P_block が閾値を超える最小の lam1 を記録
        for t in thresholds:
            # 無効値は除外
            valid = ~np.isnan(P_blocks)
            if not np.any(valid):
                threshold_curves[t].append(np.nan)
                continue
            lam_vals = lam1_grid[valid]
            pvals = P_blocks[valid]
            # もし最大値が閾値未満なら、境界は lam1_max を超える（ここでは nan）
            if np.max(pvals) < t:
                threshold_curves[t].append(np.nan)
                continue
            # 単純線形補間で閾値交差点を求める
            idx = np.argmax(pvals >= t)
            if idx == 0:
                lam_cross = lam_vals[0]
            else:
                x0, x1 = lam_vals[idx - 1], lam_vals[idx]
                y0, y1 = pvals[idx - 1], pvals[idx]
                if y1 == y0:
                    lam_cross = x1
                else:
                    lam_cross = x0 + (t - y0) * (x1 - x0) / (y1 - y0)
            threshold_curves[t].append(lam_cross)

    ax8.plot(Ks_full, lambda1_max_K, 'k-', label=r"$\lambda_{1,max}$")
    for t, curve in threshold_curves.items():
        ax8.plot(Ks_full, curve, '--', label=f"P_block={t:.2f}")
    ax8.set_title("K に対する最適化: $\\lambda_{1,max}$ と P_block 境界")
    ax8.set_xlabel("K")
    ax8.set_ylabel(r"$\lambda_1$")
    ax8.grid(True)
    ax8.legend()

    plt.tight_layout()
    fig.savefig("fig_results.png", dpi=200)
    print("Saved fig_results.png")


if __name__ == "__main__":
    plot_results()
