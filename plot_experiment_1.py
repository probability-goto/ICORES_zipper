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

from qbd_model import DynamicThresholdQBD
from stability_analysis import compute_exact_lambda1_max
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

        probe = DynamicThresholdQBD(
            lam1=1.0, lam2=LAM2,
            mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
            mu1_zip=MU1_ZIP,  mu2_zip=MU2_ZIP,
            K=K, m=m, p=P,
        )
        lam1_max = compute_exact_lambda1_max(probe)
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
