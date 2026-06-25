"""
実験1拡張: P_block と合計スループットの追加

E[L_main], E[L_merge] に加えて、ブロッキング確率 P_block と合計スループットを可視化し、
ハイブリッド制御の優位性を独立な指標から裏付ける。

出力: fig_experiment_1_extended.png
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
# 実験1と同一。他実験との統一が必要な場合は MU1_RAND = 40 に変更する。
MU1_RAND = 45
MU2_RAND = 30
MU1_ZIP = 30
MU2_ZIP = 20
K = 10
LAM2 = 10
P = 0.5
M_VALUES = [0, 2, 5, 7, 10]
NUM_POINTS = 50

# m ごとの色: 実験1 (tab10) と一致させる
# tab10: 0=blue, 1=orange, 2=green, 3=red, 4=purple
COLORS = plt.cm.tab10.colors

M_LABELS = {
    0:  r"$m=0$  (Zipper)",
    2:  r"$m=2$",
    5:  r"$m=5$",
    7:  r"$m=7$",
    10: r"$m=10$ (Random)",
}


# ---------------------------------------------------------------------------
# 安定限界の計算 (plot_experiment_1.py と同一ロジック)
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
    """フェーズ遷移行列 A = Q_0 + Q_{+1} + Q_{-1} の左定常分布から
    安定限界 λ_{1,max} を二分探索で求める。"""

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

    for _ in range(20):
        if _throughput(high) - high < 0:
            break
        high *= 2.0

    if _throughput(high) - high >= 0:
        return high

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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_main, ax_merge, ax_pblock, ax_tput = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    total_tasks = len(M_VALUES) * NUM_POINTS
    pbar = tqdm(total=total_tasks, desc="Computing metrics")

    for idx, m in enumerate(M_VALUES):
        color = COLORS[idx % len(COLORS)]
        label = M_LABELS.get(m, f"m={m}")

        lam1_max = compute_lambda1_max(m)
        lam1_vals = np.linspace(0.1, lam1_max * 0.98, NUM_POINTS)

        E_main_vals  = []
        E_merge_vals = []
        P_block_vals = []
        tput_vals    = []

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
                E_main, E_merge, P_block = calculate_performance_measures(model, R, pi0, pi1)
                total_tput = lam1 + LAM2 * (1.0 - P_block)
            except Exception:
                E_main = E_merge = P_block = total_tput = np.nan

            E_main_vals.append(E_main)
            E_merge_vals.append(E_merge)
            P_block_vals.append(P_block)
            tput_vals.append(total_tput)
            pbar.update(1)

        lam1_arr    = lam1_vals
        E_main_arr  = np.array(E_main_vals)
        E_merge_arr = np.array(E_merge_vals)
        P_block_arr = np.array(P_block_vals)
        tput_arr    = np.array(tput_vals)

        kw = dict(color=color, linewidth=2)
        ax_main.plot(lam1_arr,  E_main_arr,  label=label, **kw)
        ax_merge.plot(lam1_arr, E_merge_arr, **kw)
        ax_pblock.plot(lam1_arr, P_block_arr, **kw)
        ax_tput.plot(lam1_arr,  tput_arr,    **kw)

    pbar.close()

    # --- パネル (A): E[L_main] ---
    ax_main.set_xlabel(r"$\lambda_1$", fontsize=13)
    ax_main.set_ylabel(r"$E[L_{\mathrm{main}}]$", fontsize=13)
    ax_main.set_title(r"(A) $E[L_{\mathrm{main}}]$", fontsize=13)
    ax_main.set_ylim(0, 70)
    ax_main.legend(fontsize=10)
    ax_main.grid(True, linestyle="--", alpha=0.3)

    # --- パネル (B): E[L_merge] ---
    ax_merge.set_xlabel(r"$\lambda_1$", fontsize=13)
    ax_merge.set_ylabel(r"$E[L_{\mathrm{merge}}]$", fontsize=13)
    ax_merge.set_title(r"(B) $E[L_{\mathrm{merge}}]$", fontsize=13)
    ax_merge.grid(True, linestyle="--", alpha=0.3)

    # --- パネル (C): P_block (対数スケール) ---
    ax_pblock.set_xlabel(r"$\lambda_1$", fontsize=13)
    ax_pblock.set_ylabel(r"$P_{\mathrm{block}}$", fontsize=13)
    ax_pblock.set_title(r"(C) Blocking Probability $P_{\mathrm{block}}$", fontsize=13)
    ax_pblock.set_yscale("log")
    ax_pblock.grid(True, linestyle="--", alpha=0.3)

    # --- パネル (D): 合計スループット ---
    ax_tput.set_xlabel(r"$\lambda_1$", fontsize=13)
    ax_tput.set_ylabel(r"$\lambda_1 + \lambda_2(1 - P_{\mathrm{block}})$", fontsize=13)
    ax_tput.set_title(r"(D) Total Throughput", fontsize=13)
    ax_tput.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        (
            rf"$K={K}$, $\lambda_2={LAM2}$, $p={P}$, "
            rf"$\mu_1^{{\mathrm{{rand}}}}={MU1_RAND}$, $\mu_2^{{\mathrm{{rand}}}}={MU2_RAND}$, "
            rf"$\mu_1^{{\mathrm{{zip}}}}={MU1_ZIP}$, $\mu_2^{{\mathrm{{zip}}}}={MU2_ZIP}$"
        ),
        fontsize=12,
    )

    plt.tight_layout()
    output_path = "fig_experiment_1_extended.png"
    fig.savefig(output_path, dpi=200)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
