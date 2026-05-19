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
from stationary_analysis import compute_stationary, calculate_performance_measures

try:
    from scipy.linalg import null_space
except Exception:
    null_space = None


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


def _left_stationary_of_A(A):
    """行列 A の左定常分布 eta を求める（eta A = 0）．"""
    # null_space を使って A^T の零空間を取る
    ns = None
    if null_space is not None:
        ns = null_space(A.T)

    if ns is None or ns.size == 0:
        # SVD によるフォールバック
        U, s, Vt = np.linalg.svd(A.T, full_matrices=False)
        ns = Vt.T[:, -1:]

    if ns.size == 0:
        raise RuntimeError("A^T の零空間が得られませんでした。")

    eta = ns[:, 0].astype(float)
    eta = np.real_if_close(eta, tol=1000)
    # 正にする（必要なら符号反転）
    if np.sum(eta) < 0:
        eta = -eta
    s = np.sum(eta)
    if s == 0:
        raise RuntimeError("定常分布の正規化に失敗しました。")
    eta = eta / s
    return eta


def compute_lambda1_max_for_params(lambda2, K, m=None, p=P, tol=1e-6):
    """与えられた lambda2, K 等のパラメータに対して lambda1_max を数値的に求める。

    解法: 固定点方程式 lam1 = f(lam1) を満たす lam1 を求める（f(lam1) = eta(lam1) Q_-1 e）。
    二分探索（bisection）で解を探します。
    """

    def f(lam1):
        # lam1 に対する f を評価
        model = make_model(lam1, lambda2, K, m=m, p=p)
        A = model.Q_minus1 + model.Q0 + model.Q_plus1
        try:
            eta = _left_stationary_of_A(A)
        except Exception:
            # 零空間取得失敗時は 0 を返して探索を続行
            return 0.0
        ones = np.ones(A.shape[0])
        return float(eta @ (model.Q_minus1 @ ones))

    # f(0) >= 0 であるはず
    low = 0.0
    high = max(1.0, MU1_RAND * 2)
    # 括弧を探す（g(x)=f(x)-x で符号変化を探す）
    g_low = f(low) - low
    g_high = f(high) - high
    max_expand = 20
    expand = 0
    while g_high >= 0 and expand < max_expand:
        high *= 2.0
        g_high = f(high) - high
        expand += 1

    if g_high >= 0:
        # 符号変化が見つからなかった。f が大きく lambda1 を上回る場合、上限を high として返す
        return high

    # 二分探索
    for _ in range(50):
        mid = 0.5 * (low + high)
        g_mid = f(mid) - mid
        if abs(g_mid) < tol:
            return mid
        if g_mid > 0:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


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
        lambda1_max = compute_lambda1_max_for_params(lam2, K)

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
            lam1_max = compute_lambda1_max_for_params(10.0, K)
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
