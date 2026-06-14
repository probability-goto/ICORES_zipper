"""
実験群1：ハイブリッド制御の優位性の定量的証明

閾値 m を変化させたときの本線・合流車線の平均系内客数を比較する。
純粋なファスナー合流 (m=0) や純粋な乱雑合流 (m=K) と比較して、
中間のハイブリッド設定 (0 < m < K) が「閑散時の効率」と
「渋滞時の耐性」を両立していることを可視化する。

出力: experiment_1_results.png
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from scipy.linalg import null_space
except ImportError:
    null_space = None

from qbd_model import DynamicThresholdQBD
from stationary_analysis import compute_stationary, calculate_performance_measures

# --- パラメータ設定 ---
MU1_RAND = 45
MU2_RAND = 30
MU1_ZIP = 30
MU2_ZIP = 20
K = 10
LAM2 = 10
P = 0.5
M_VALUES = [0, 2, 5, 7, 10]
NUM_POINTS = 35  # lambda1 の分割点数

M_LABELS = {
    0:  r"$m=0$  (Zipper)",
    2:  r"$m=2$",
    5:  r"$m=5$",
    7:  r"$m=7$",
    10: r"$m=10$ (Random)",
}


# ---------------------------------------------------------------------------
# 安定限界の計算
# ---------------------------------------------------------------------------

def _left_stationary(A):
    """行列 A の左定常分布 η を求める（η A = 0, η e = 1）。"""
    ns = None
    if null_space is not None:
        ns = null_space(A.T)

    if ns is None or ns.size == 0:
        _, _, Vt = np.linalg.svd(A.T, full_matrices=False)
        ns = Vt.T[:, -1:]

    if ns.size == 0:
        raise RuntimeError("A^T の零空間が得られませんでした。")

    eta = np.real_if_close(ns[:, 0].astype(float), tol=1000)
    if np.sum(eta) < 0:
        eta = -eta
    s = np.sum(eta)
    if s == 0:
        raise RuntimeError("定常分布の正規化に失敗しました。")
    return eta / s


def compute_lambda1_max(m, lam2=LAM2, K=K, p=P, tol=1e-6):
    """フェーズ遷移行列 A = Q_0 + Q_{+1} + Q_{-1} の左定常分布 η から
    安定限界 λ_{1,max} = η Q_{-1} e を二分探索で求める。

    安定条件: λ_1 < η(λ_1) Q_{-1} e
    λ_{1,max} は g(λ_1) = η(λ_1) Q_{-1} e − λ_1 = 0 の根。
    """

    def _throughput(lam1):
        model = DynamicThresholdQBD(
            lam1=lam1, lam2=lam2,
            mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
            mu1_zip=MU1_ZIP,  mu2_zip=MU2_ZIP,
            K=K, m=m, p=p,
        )
        A = model.Q0 + model.Q_plus1 + model.Q_minus1
        try:
            eta = _left_stationary(A)
        except Exception:
            return 0.0
        return float(eta @ (model.Q_minus1 @ np.ones(A.shape[0])))

    low, high = 0.0, max(1.0, MU1_RAND * 2)

    # high を g(high) < 0 になるまで拡張
    for _ in range(20):
        if _throughput(high) - high < 0:
            break
        high *= 2.0

    if _throughput(high) - high >= 0:
        return high  # 発散しない（上限を返す）

    # 二分探索
    for _ in range(60):
        mid = 0.5 * (low + high)
        g_mid = _throughput(mid) - mid
        if abs(g_mid) < tol:
            return mid
        if g_mid > 0:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# メイン計算とプロット
# ---------------------------------------------------------------------------

def main():
    colors = plt.cm.tab10.colors

    fig, (ax_main, ax_merge) = plt.subplots(1, 2, figsize=(14, 6))

    total_tasks = len(M_VALUES) * NUM_POINTS
    pbar = tqdm(total=total_tasks, desc="Computing E[L]")

    for idx, m in enumerate(M_VALUES):
        color = colors[idx % len(colors)]
        label = M_LABELS.get(m, f"m={m}")

        lam1_max = compute_lambda1_max(m)
        lam1_vals = np.linspace(1.0, lam1_max * 0.99, NUM_POINTS)

        E_main_vals = []
        E_merge_vals = []

        for lam1 in lam1_vals:
            try:
                model = DynamicThresholdQBD(
                    lam1=lam1, lam2=LAM2,
                    mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
                    mu1_zip=MU1_ZIP,  mu2_zip=MU2_ZIP,
                    K=K, m=m, p=P,
                )
                R = model.solve_R()
                pi0, pi1, _ = compute_stationary(model, R)
                E_main, E_merge, _ = calculate_performance_measures(model, R, pi0, pi1)
            except Exception:
                E_main, E_merge = np.nan, np.nan

            E_main_vals.append(E_main)
            E_merge_vals.append(E_merge)
            pbar.update(1)

        lam1_arr   = lam1_vals
        E_main_arr = np.array(E_main_vals)
        E_merge_arr = np.array(E_merge_vals)

        ax_main.plot(lam1_arr, E_main_arr,  color=color, label=label, linewidth=2)
        ax_merge.plot(lam1_arr, E_merge_arr, color=color, label=label, linewidth=2)

    pbar.close()

    # --- グラフA: 本線の平均系内客数 ---
    ax_main.set_xlabel(r"$\lambda_1$", fontsize=13)
    ax_main.set_ylabel(r"$E[L_{\mathrm{main}}]$", fontsize=13)
    ax_main.set_title(r"(A) Mean Queue Length — Main Lane", fontsize=13)
    ax_main.set_ylim(0, 50)
    ax_main.legend(fontsize=11)
    ax_main.grid(True, linestyle="--", alpha=0.6)

    # --- グラフB: 合流車線の平均系内客数 ---
    ax_merge.set_xlabel(r"$\lambda_1$", fontsize=13)
    ax_merge.set_ylabel(r"$E[L_{\mathrm{merge}}]$", fontsize=13)
    ax_merge.set_title(r"(B) Mean Queue Length — Merge Lane", fontsize=13)
    ax_merge.legend(fontsize=11)
    ax_merge.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(
        (
            rf"($K={K}$, $\lambda_2={LAM2}$, $p={P}$, "
            rf"$\mu_1^{{\mathrm{{rand}}}}={MU1_RAND}$, $\mu_2^{{\mathrm{{rand}}}}={MU2_RAND}$, "
            rf"$\mu_1^{{\mathrm{{zip}}}}={MU1_ZIP}$, $\mu_2^{{\mathrm{{zip}}}}={MU2_ZIP}$)"
        ),
        fontsize=12,
    )

    plt.tight_layout()
    output_path = "experiment_1_results.png"
    fig.savefig(output_path, dpi=200)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
