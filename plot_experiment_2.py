"""
実験群2：K と λ_{1,max} のトレードオフ緩和の検証

「合流車線容量 K を増やすと本線の最大スループット λ_{1,max} が低下する」
というトレードオフに対し、動的閾値 m を導入することで
この低下を抑制できるかを解析的に検証する。

出力: experiment_2_results.png
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from qbd_model import DynamicThresholdQBD
from stability_analysis import compute_exact_lambda1_max

# ---------------------------------------------------------------------------
# パラメータ設定
# ---------------------------------------------------------------------------
MU1_RAND = 40
MU2_RAND = 30
MU1_ZIP  = 30
MU2_ZIP  = 20
P        = 0.5
LAM1_DUMMY = 5.0   # lambda1_max 計算時のダミー値（結果に影響しない）

K_VALUES   = list(range(5, 21))   # 5 ～ 20
LAM2_VALUES = [5, 10, 20]

# (m の値または 'K', 凡例ラベル, マーカースタイル, 色)
M_SPECS = [
    (0,   r"$m=0$  (Zipper)",  "o-", "tab:blue"),
    (2,   r"$m=2$",       "s-", "tab:orange"),
    (5,   r"$m=5$",       "^-", "tab:green"),
    ("K", r"$m=K$  (Random)",  "D-", "tab:red"),
]

# ---------------------------------------------------------------------------
# λ_{1,max} の一括計算
# ---------------------------------------------------------------------------

def compute_all(K_values, lam2_values, m_specs):
    """全 (lam2, m_spec, K) の組み合わせで λ_{1,max} を計算して返す。

    Returns
    -------
    data : dict
        data[(m_label, lam2)] = (K_list, lam1_max_list)
    """
    # 有効な組み合わせ数をカウント（tqdm の total 用）
    total = 0
    for _, _, _, _ in m_specs:
        for K in K_values:
            for _ in lam2_values:
                total += 1

    data = {}
    pbar = tqdm(total=total, desc="Computing λ₁_max", unit="case")

    for m_val, label, _, _ in m_specs:
        for lam2 in lam2_values:
            key = (label, lam2)
            K_list, lam1_max_list = [], []

            for K in K_values:
                # m>Kの組み合わせをスキップ
                actual_m = K if m_val == "K" else m_val
                if actual_m > K:
                    pbar.update(1)
                    continue

                try:
                    model = DynamicThresholdQBD(
                        lam1=LAM1_DUMMY, lam2=lam2,
                        mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
                        mu1_zip=MU1_ZIP,  mu2_zip=MU2_ZIP,
                        K=K, m=actual_m, p=P,
                    )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        lam1_max = compute_exact_lambda1_max(model, tol=1e-8)
                    # 精度警告をそのまま転送（計算は続行）
                    for w in caught:
                        tqdm.write(
                            f"[WARNING] K={K}, m={actual_m}, lam2={lam2}: {w.message}"
                        )
                    K_list.append(K)
                    lam1_max_list.append(lam1_max)
                except Exception as e:
                    tqdm.write(f"[ERROR] K={K}, m={actual_m}, lam2={lam2}: {e}")
                    K_list.append(K)
                    lam1_max_list.append(np.nan)

                pbar.update(1)

            data[key] = (K_list, lam1_max_list)

    pbar.close()
    return data


# ---------------------------------------------------------------------------
# プロット
# ---------------------------------------------------------------------------

def plot(data, K_values, lam2_values, m_specs, output_path="experiment_2_results.png"):
    fig, axes = plt.subplots(
        1, len(lam2_values),
        figsize=(5.5 * len(lam2_values), 5.5),
        sharey=True,          # Y軸スケールを統一
    )

    subplot_labels = ["(A)", "(B)", "(C)"]

    for col, lam2 in enumerate(lam2_values):
        ax = axes[col]

        for m_val, label, marker_style, color in m_specs:
            key = (label, lam2)
            if key not in data:
                continue
            K_list, lam1_max_list = data[key]
            if not K_list:
                continue
            ax.plot(
                K_list, lam1_max_list,
                marker_style,
                label=label,
                color=color,
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Capacity $K$", fontsize=13)
        ax.set_title(
            rf"{subplot_labels[col]} $\lambda_2 = {lam2}$",
            fontsize=13,
        )
        ax.set_xticks(K_values[::2])   # 偶数 K のみ目盛り表示（密集防止）
        ax.grid(True, linestyle="--", alpha=0.55)
        ax.legend(fontsize=10, loc="best")

    # Y 軸ラベルは左端のサブプロットのみ
    axes[0].set_ylabel(r"Max Throughput $\lambda_{1,\mathrm{max}}$", fontsize=13)

    fig.suptitle(
        (
            "Experiment 2: Trade-off Mitigation between $K$ and $\\lambda_{1,\\mathrm{max}}$\n"
            rf"($p={P}$, "
            rf"$\mu_1^{{\mathrm{{rand}}}}={MU1_RAND}$, "
            rf"$\mu_2^{{\mathrm{{rand}}}}={MU2_RAND}$, "
            rf"$\mu_1^{{\mathrm{{zip}}}}={MU1_ZIP}$, "
            rf"$\mu_2^{{\mathrm{{zip}}}}={MU2_ZIP}$)"
        ),
        fontsize=12,
        y=1.02,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {output_path}")


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main():
    data = compute_all(K_VALUES, LAM2_VALUES, M_SPECS)
    plot(data, K_VALUES, LAM2_VALUES, M_SPECS)


if __name__ == "__main__":
    main()
