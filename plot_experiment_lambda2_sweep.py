"""
λ2 スイープ実験: 合流車線への到着率 λ2 を変化させたときの 4 指標応答

固定パラメータ: K=10, λ1=10, p=0.5, µ1_rand=40, µ2_rand=30, µ1_zip=30, µ2_zip=20
スイープ: m ∈ {0, 2, 5, 7, 10},  λ2 ∈ [0.5, 30] (50 点)

出力: fig_experiment_lambda2_sweep.png

期待される観察:
  - m=0 では λ2 が小さいうちから P_block が増加 (µ2_zip=20 が低いため)
  - m=10 では合流車線が高速処理されるため P_block の急増が遅れる
  - 合計スループットは λ2 増加に対して線形に増えるが、
    P_block 急増領域で頭打ちになる
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from qbd_model import DynamicThresholdQBD
from stability_analysis import compute_exact_lambda1_max
from stationary_analysis import compute_stationary, calculate_performance_measures

# ---------------------------------------------------------------------------
# パラメータ
# ---------------------------------------------------------------------------
K        = 10
LAM1     = 10.0
P        = 0.5
MU1_RAND = 40
MU2_RAND = 30
MU1_ZIP  = 30
MU2_ZIP  = 20

M_VALUES  = [0, 2, 5, 7, 10]
LAM2_GRID = np.linspace(0.5, 30.0, 50)

M_COLORS = {0: "tab:blue", 2: "tab:orange", 5: "tab:green", 7: "tab:red", 10: "tab:purple"}

# ---------------------------------------------------------------------------
# 計算
# ---------------------------------------------------------------------------

def compute_sweep():
    results = {m: {"lam2": [], "E_L_main": [], "E_L_merge": [], "P_block": [], "throughput": []}
               for m in M_VALUES}

    total = len(M_VALUES) * len(LAM2_GRID)
    with tqdm(total=total, desc="λ2 sweep", unit="pt") as pbar:
        for m in M_VALUES:
            for lam2 in LAM2_GRID:
                # 安定性チェック: λ1_max(m, λ2) を計算し λ1=10 が収まるか確認
                probe = DynamicThresholdQBD(
                    lam1=LAM1, lam2=lam2,
                    mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
                    mu1_zip=MU1_ZIP,   mu2_zip=MU2_ZIP,
                    K=K, m=m, p=P,
                )
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    lam1_max = compute_exact_lambda1_max(probe, tol=1e-8)

                if LAM1 >= lam1_max:
                    # 系が不安定 → NaN を記録してスキップ
                    results[m]["lam2"].append(lam2)
                    results[m]["E_L_main"].append(np.nan)
                    results[m]["E_L_merge"].append(np.nan)
                    results[m]["P_block"].append(np.nan)
                    results[m]["throughput"].append(np.nan)
                    pbar.update(1)
                    continue

                try:
                    R = probe.solve_R()
                    pi0, pi1, _ = compute_stationary(probe, R)
                    E_L_main, E_L_merge, P_block = calculate_performance_measures(probe, R, pi0, pi1)
                    throughput = LAM1 + lam2 * (1.0 - P_block)

                    results[m]["lam2"].append(lam2)
                    results[m]["E_L_main"].append(E_L_main)
                    results[m]["E_L_merge"].append(E_L_merge)
                    results[m]["P_block"].append(P_block)
                    results[m]["throughput"].append(throughput)
                except Exception as e:
                    tqdm.write(f"  [WARN] m={m}, λ2={lam2:.2f}: {e}")
                    results[m]["lam2"].append(lam2)
                    results[m]["E_L_main"].append(np.nan)
                    results[m]["E_L_merge"].append(np.nan)
                    results[m]["P_block"].append(np.nan)
                    results[m]["throughput"].append(np.nan)

                pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# プロット
# ---------------------------------------------------------------------------

def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panel_labels = ["(A) $E[L_{\mathrm{main}}]$", "(B) $E[L_{\mathrm{merge}}]$", "(C) $P_{\mathrm{block}}$", "(D) Total Throughput"]
    metrics      = ["E_L_main", "E_L_merge", "P_block", "throughput"]
    ylabels      = [r"$E[L_{\mathrm{main}}]$", r"$E[L_{\mathrm{merge}}]$",
                    r"$P_{\mathrm{block}}$", "$\lambda_1 + \lambda_2(1 - P_{\mathrm{block}})$"]

    for idx, (metric, ylabel, panel) in enumerate(zip(metrics, ylabels, panel_labels)):
        ax = axes[idx // 2][idx % 2]
        for m in M_VALUES:
            lam2_arr = np.array(results[m]["lam2"])
            y_arr    = np.array(results[m][metric])
            ax.plot(lam2_arr, y_arr,
                    color=M_COLORS[m],
                    linewidth=2,
                    label=f"$m={m}$")

        ax.set_xlabel(r"$\lambda_2$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(panel, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)

        if metric == "P_block":
            ax.set_yscale("log")

        if idx == 0:
            ax.legend(fontsize=11, loc="upper left")

    fig.suptitle(
        rf"($K={K},\ \lambda_1={int(LAM1)},\ p={P},"
        rf"\ \mu_1^{{\rm rand}}={MU1_RAND},"
        rf"\ \mu_2^{{\rm rand}}={MU2_RAND},"
        rf"\ \mu_1^{{\rm zip}}={MU1_ZIP},"
        rf"\ \mu_2^{{\rm zip}}={MU2_ZIP}$)",
        fontsize=11,
        y=1.02,
    )

    plt.tight_layout()
    output_path = "fig_experiment_lambda2_sweep.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {output_path}")


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main():
    results = compute_sweep()
    plot(results)


if __name__ == "__main__":
    main()
