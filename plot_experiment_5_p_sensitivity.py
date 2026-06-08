"""
実験群5：確率パラメータ p の感度分析

本拡張モデル固有のパラメータ p（モードAでの本線選択確率）が
システム性能に与える影響を定量化する。特に以下3点を可視化する：
  1. E[L_total] を最小化する内点 p* の存在
  2. 最適 p* の λ1 依存性
  3. p の効果が m に依存すること（m=0 では消失）

出力: experiment_5_p_sensitivity.png
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
LAM1_DUMMY = 5.0   # λ_{1,max} 計算時のダミー値（A 行列は lam1 に依存しない）
LAM1_LOW   = 5.0   # 低負荷の固定値

M_VALUES = [0, 3, 5, 7, 10]
P_VALUES = np.arange(0.0, 1.01, 0.1)   # 0.0 〜 1.0 の 11 点

LOAD_NAMES  = ["low", "medium", "high"]
LOAD_FRACS  = [None, 0.5, 0.95]        # None は LAM1_LOW を使用

ROW_LABELS = [
    r"Low Load  ($\lambda_1 = 5$)",
    r"Medium Load  ($\lambda_1 = 0.5\,\lambda_{1,\max}$)",
    r"High Load  ($\lambda_1 = 0.95\,\lambda_{1,\max}$)",
]
COL_LABELS = [
    r"$E[L_{\mathrm{main}}]$",
    r"$E[L_{\mathrm{merge}}]$",
    r"$E[L_{\mathrm{total}}]$",
]


# ---------------------------------------------------------------------------
# λ_{1,max} の計算
# ---------------------------------------------------------------------------

def _compute_lam1_max(m: int) -> float:
    model = DynamicThresholdQBD(
        lam1=LAM1_DUMMY, lam2=LAM2,
        mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
        mu1_zip=MU1_ZIP,  mu2_zip=MU2_ZIP,
        K=K, m=m, p=0.5,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val = compute_exact_lambda1_max(model, tol=1e-8)
    for w in caught:
        tqdm.write(f"  [WARN] λ1_max m={m}: {w.message}")
    return float(val)


# ---------------------------------------------------------------------------
# 性能指標の計算
# ---------------------------------------------------------------------------

def _compute_metrics(m: int, p: float, lam1: float):
    """(E[L_main], E[L_merge], E[L_total]) を返す。失敗時は例外を投げる。"""
    model = DynamicThresholdQBD(
        lam1=lam1, lam2=LAM2,
        mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
        mu1_zip=MU1_ZIP,  mu2_zip=MU2_ZIP,
        K=K, m=m, p=p,
    )
    R = model.solve_R()
    pi0, pi1, _ = compute_stationary(model, R)
    E_L_main, E_L_merge, _ = calculate_performance_measures(model, R, pi0, pi1)
    return E_L_main, E_L_merge, E_L_main + E_L_merge


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    # ---- Step 1: λ_{1,max} を各 m について計算 ----
    print("Step 1: Computing λ_{1,max} for each m ...")
    lam1_max_dict: dict = {}
    for m in M_VALUES:
        try:
            val = _compute_lam1_max(m)
            lam1_max_dict[m] = val
            print(f"  m={m:2d}:  λ1_max = {val:.4f}")
        except Exception as e:
            tqdm.write(f"  [ERROR] λ1_max m={m}: {e}")
            lam1_max_dict[m] = None

    # ---- Step 2: λ1 水準を各 m について確定 ----
    # lam1_table[m] = [lam1_low, lam1_med, lam1_high]  （None はスキップ）
    lam1_table: dict = {}
    for m in M_VALUES:
        lam1_max = lam1_max_dict[m]
        if lam1_max is None:
            lam1_table[m] = [None, None, None]
        else:
            lam1_table[m] = [
                LAM1_LOW,
                lam1_max * 0.5,
                lam1_max * 0.95,
            ]

    # ---- Step 3: 全 (m, 負荷水準, p) の指標を計算 ----
    # results[m_idx, load_idx, p_idx, metric_idx]  metric: 0=main, 1=merge, 2=total
    n_m = len(M_VALUES)
    n_p = len(P_VALUES)
    results = np.full((n_m, 3, n_p, 3), np.nan)

    total = n_m * 3 * n_p   # = 165
    print(f"\nStep 2: Computing metrics ({total} total points) ...")

    with tqdm(total=total, desc="Computing", unit="pt") as pbar:
        for mi, m in enumerate(M_VALUES):
            for li, load_name in enumerate(LOAD_NAMES):
                lam1 = lam1_table[m][li]
                if lam1 is None:
                    tqdm.write(
                        f"  [SKIP] m={m}, load={load_name}: λ1_max unavailable"
                    )
                    pbar.update(n_p)
                    continue
                for pi_idx, p in enumerate(P_VALUES):
                    try:
                        metrics = _compute_metrics(m, p, lam1)
                        results[mi, li, pi_idx, :] = metrics
                    except Exception as e:
                        tqdm.write(
                            f"  [WARN] m={m}, {load_name}, p={p:.2f}: {e}"
                        )
                    pbar.update(1)

    # ---- Step 4: グラフ描画 ----
    colors = [plt.cm.tab10(i) for i in range(n_m)]
    m_labels = [f"m={m}" for m in M_VALUES]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for li in range(3):         # 行：λ1 水準
        for ci in range(3):     # 列：指標
            ax = axes[li, ci]

            for mi, m in enumerate(M_VALUES):
                y = results[mi, li, :, ci]

                # m=0 は破線・太線で視覚的に強調（p の効果が消失することを示す）
                ls = "--" if m == 0 else "-"
                lw = 2.5  if m == 0 else 1.8

                ax.plot(
                    P_VALUES, y,
                    color=colors[mi],
                    linestyle=ls,
                    linewidth=lw,
                    marker="o",
                    markersize=3,
                    label=m_labels[mi],
                )

                # 内点最小値の位置に縦の点線マーカーを追加
                valid = ~np.isnan(y)
                if valid.sum() >= 3:
                    min_idx = int(np.nanargmin(y))
                    if 0 < min_idx < n_p - 1:   # 両端を除く内点のみ
                        ax.axvline(
                            x=P_VALUES[min_idx],
                            color=colors[mi],
                            linestyle=":",
                            linewidth=1.0,
                            alpha=0.65,
                        )

            ax.set_xlabel("$p$", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlim(-0.02, 1.02)

    # 列ラベル（上側：指標名）
    for ci, label in enumerate(COL_LABELS):
        axes[0, ci].set_title(label, fontsize=13, pad=8)

    # 行ラベル（左側：λ1 水準名）
    for li, label in enumerate(ROW_LABELS):
        axes[li, 0].set_ylabel(label, fontsize=11)

    # 凡例は右上のサブプロット (0, 2) に1つだけ配置
    handles, labs = axes[0, 0].get_legend_handles_labels()
    axes[0, 2].legend(
        handles, labs,
        loc="upper right",
        fontsize=9,
        title=r"Threshold $m$",
        title_fontsize=9,
    )

    fig.suptitle(
        "Experiment 5: Sensitivity Analysis of Probability p\n"
        f"(K={K}, λ2={LAM2}, μ1_rand={MU1_RAND}, μ2_rand={MU2_RAND}, "
        f"μ1_zip={MU1_ZIP}, μ2_zip={MU2_ZIP})",
        fontsize=12, y=1.00,
    )

    plt.tight_layout()
    output_path = "experiment_5_p_sensitivity.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {output_path}")

    # ---- Step 5: 出力サマリ ----
    print()
    print("=" * 65)
    print(
        f"{'m':<4} {'λ1_level':<10} {'λ1':>8} {'best_p':>8} {'min_E_L_total':>14}"
    )
    print("-" * 65)

    for mi, m in enumerate(M_VALUES):
        for li, load_name in enumerate(LOAD_NAMES):
            lam1 = lam1_table[m][li]
            lam1_str = f"{lam1:.2f}" if lam1 is not None else "N/A"

            y_total = results[mi, li, :, 2]

            if np.all(np.isnan(y_total)):
                print(
                    f"{m:<4} {load_name:<10} {lam1_str:>8} {'----':>8} {'N/A':>14}"
                )
                continue

            min_idx = int(np.nanargmin(y_total))
            min_val = float(y_total[min_idx])
            best_p  = P_VALUES[min_idx]

            # m=0 は p に依存しないため "----" 表示
            if m == 0:
                best_p_str = "----"
            elif min_idx == 0 or min_idx == n_p - 1:
                best_p_str = f"{best_p:.2f}*"   # 端点最小は * で注記
            else:
                best_p_str = f"{best_p:.2f}"

            print(
                f"{m:<4} {load_name:<10} {lam1_str:>8} {best_p_str:>8} {min_val:>14.4f}"
            )

    print("=" * 65)
    print("(* = boundary optimum, not interior)")


if __name__ == "__main__":
    main()
