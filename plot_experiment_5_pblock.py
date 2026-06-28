"""
実験5 拡張: P_block の確率 p 感度分析

m=K, p=1 の「病的構成」が P_block の急騰として独立指標から確認できるかを検証する。

期待される観察:
  - 低負荷では全 m で P_block ≈ 0 のため、グラフが平坦になる可能性
  - 中負荷では m=10, p→1 で P_block が顕著に上昇するはず
  - 高負荷では m=10, p=1 で P_block がほぼ 1 に達する (「病的構成」の直接的証拠)
  - m=0 の曲線は p に対して完全平坦 (実験5 観察(I) と整合)

出力: fig_experiment_5_pblock.png
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from qbd_model import DynamicThresholdQBD
from stability_analysis import compute_exact_lambda1_max
from stationary_analysis import compute_stationary, calculate_performance_measures

# ---------------------------------------------------------------------------
# パラメータ設定
# ---------------------------------------------------------------------------
MU1_RAND   = 40
MU2_RAND   = 30
MU1_ZIP    = 30
MU2_ZIP    = 20
K          = 10
LAM2       = 10
LAM1_DUMMY = 5.0   # λ_{1,max} 計算時のダミー値 (A 行列は lam1 に依存しない)
LAM1_LOW   = 5.0   # 低負荷の固定値

M_VALUES = [0, 3, 5, 7, 10]
P_VALUES = np.arange(0.0, 1.01, 0.1)   # 0.0 〜 1.0 の 11 点

LOAD_NAMES = ["low", "medium", "high"]

# 各 m の色は実験5 と一致 (tab10 colormap)
COLORS = [plt.cm.tab10(i) for i in range(len(M_VALUES))]

ROW_LABELS = [
    r"Low Load  ($\lambda_1 = 5$)",
    r"Medium Load  ($\lambda_1 = 0.5\,\lambda_{1,\max}$)",
    r"High Load  ($\lambda_1 = 0.95\,\lambda_{1,\max}$)",
]

PBLOCK_CLIP_MIN = 1e-10   # P_block=0 を対数スケールでプロットするための下限値


# ---------------------------------------------------------------------------
# λ_{1,max} の計算
# ---------------------------------------------------------------------------

def _compute_lam1_max(m: int, p: float) -> float:
    """各 (m, p) ペアに対する λ1_max(m, p) を返す。"""
    model = DynamicThresholdQBD(
        lam1=LAM1_DUMMY, lam2=LAM2,
        mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
        mu1_zip=MU1_ZIP,  mu2_zip=MU2_ZIP,
        K=K, m=m, p=p,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val = compute_exact_lambda1_max(model, tol=1e-8)
    for w in caught:
        tqdm.write(f"  [WARN] λ1_max m={m}, p={p:.2f}: {w.message}")
    return float(val)


def _get_lam1(mi: int, pi_idx: int, load_level: str, lam1_max_table: np.ndarray) -> float:
    """指定の (m_idx, p_idx, 負荷水準) に対する λ1 を返す。"""
    if load_level == "low":
        return LAM1_LOW
    lm = lam1_max_table[mi, pi_idx]
    if np.isnan(lm) or lm <= 0:
        return np.nan
    if load_level == "medium":
        return lm * 0.5
    if load_level == "high":
        return lm * 0.95
    raise ValueError(f"Unknown load level: {load_level}")


# ---------------------------------------------------------------------------
# P_block の計算
# ---------------------------------------------------------------------------

def _compute_pblock(m: int, p: float, lam1: float) -> float:
    """P_block を返す。失敗時は例外を投げる。"""
    model = DynamicThresholdQBD(
        lam1=lam1, lam2=LAM2,
        mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
        mu1_zip=MU1_ZIP,  mu2_zip=MU2_ZIP,
        K=K, m=m, p=p,
    )
    R = model.solve_R()
    pi0, pi1, _ = compute_stationary(model, R)
    _, _, P_block = calculate_performance_measures(model, R, pi0, pi1)
    return P_block


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    n_m = len(M_VALUES)
    n_p = len(P_VALUES)

    # ---- Step 1: 全 (m, p) について λ_{1,max} を計算 ----
    print("Step 1: Building λ_{1,max}(m, p) table ...")
    lam1_max_table = np.full((n_m, n_p), np.nan)

    with tqdm(total=n_m * n_p, desc="λ1_max", unit="pt") as pbar:
        for mi, m in enumerate(M_VALUES):
            for pi_idx, p in enumerate(P_VALUES):
                try:
                    lam1_max_table[mi, pi_idx] = _compute_lam1_max(m, p)
                except Exception as e:
                    tqdm.write(f"  [ERROR] λ1_max m={m}, p={p:.2f}: {e}")
                pbar.update(1)

    # ---- Step 2: 全 (m, 負荷水準, p) の P_block を計算 ----
    # pblock[m_idx, load_idx, p_idx]
    pblock = np.full((n_m, 3, n_p), np.nan)

    total_pts = n_m * 3 * n_p   # = 165
    print(f"\nStep 2: Computing P_block ({total_pts} total points) ...")

    with tqdm(total=total_pts, desc="P_block", unit="pt") as pbar:
        for mi, m in enumerate(M_VALUES):
            for pi_idx, p in enumerate(P_VALUES):
                for li, load_level in enumerate(LOAD_NAMES):
                    lam1 = _get_lam1(mi, pi_idx, load_level, lam1_max_table)
                    if np.isnan(lam1) or lam1 <= 0:
                        tqdm.write(
                            f"  [SKIP] m={m}, p={p:.2f}, {load_level}: invalid λ1"
                        )
                        pbar.update(1)
                        continue

                    lam1_max = lam1_max_table[mi, pi_idx]
                    if not np.isnan(lam1_max) and lam1 >= lam1_max * 0.999:
                        tqdm.write(
                            f"  [SKIP] m={m}, p={p:.2f}, {load_level}: "
                            f"λ1={lam1:.4f} >= 0.999×λ1_max={lam1_max:.4f} (unstable)"
                        )
                        pbar.update(1)
                        continue

                    try:
                        pb = _compute_pblock(m, p, lam1)
                        if pb < 0 or pb > 1:
                            tqdm.write(
                                f"  [WARN] m={m}, p={p:.2f}, {load_level}: "
                                f"P_block={pb:.6f} out of [0, 1] → set NaN"
                            )
                            pb = np.nan
                        pblock[mi, li, pi_idx] = pb
                    except Exception as e:
                        tqdm.write(
                            f"  [WARN] m={m}, p={p:.2f}, {load_level}: {e}"
                        )
                    pbar.update(1)

    # ---- Step 3: グラフ描画 ----
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    m_labels = [f"m={m}" for m in M_VALUES]

    for li, (ax, row_label) in enumerate(zip(axes, ROW_LABELS)):
        for mi, m in enumerate(M_VALUES):
            y = pblock[mi, li, :].copy()

            # P_block の小さい値 (≤ 下限) を対数スケール用にクリップ (NaN は保持)
            y_plot = y.copy()
            valid = ~np.isnan(y_plot)
            y_plot[valid & (y_plot < PBLOCK_CLIP_MIN)] = PBLOCK_CLIP_MIN

            ls = "--" if m == 0 else "-"
            lw = 2.5  if m == 0 else 1.8

            ax.plot(
                P_VALUES, y_plot,
                color=COLORS[mi],
                linestyle=ls,
                linewidth=lw,
                marker="o",
                markersize=4,
                label=m_labels[mi],
            )

        ax.set_yscale("log")
        ax.set_xlabel("$p$", fontsize=11)
        ax.set_ylabel("$P_{\\mathrm{block}}$", fontsize=11)
        ax.set_title(row_label, fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.3)

    # 凡例は最上部パネル (低負荷) のみ
    handles, labs = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labs,
        loc="upper right",
        fontsize=9,
        title=r"Threshold $m$",
        title_fontsize=9,
    )

    fig.suptitle(
        "Experiment 5 Extended: $P_{\\mathrm{block}}$ Sensitivity to $p$\n"
        f"(K={K}, $\\lambda_2$={LAM2}, $\\mu_{{1,\\mathrm{{rand}}}}$={MU1_RAND}, "
        f"$\\mu_{{2,\\mathrm{{rand}}}}$={MU2_RAND}, "
        f"$\\mu_{{1,\\mathrm{{zip}}}}$={MU1_ZIP}, $\\mu_{{2,\\mathrm{{zip}}}}$={MU2_ZIP})",
        fontsize=11,
    )

    plt.tight_layout()
    output_path = "fig_experiment_5_pblock.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
