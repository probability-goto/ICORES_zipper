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
P_VALUES = np.arange(0.0, 1.01, 0.05)   # 0.0 〜 1.0 の 21 点
P_VALUES_HEATMAP = np.arange(0.0, 1.01, 0.1)   # ヒートマップ用：元の 0.1 刻み 11 点

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

def _compute_lam1_max(m: int, p: float) -> float:
    """各 (m, p) ペアに対する λ1_max(m, p) を返す。

    Q_{-1} と Q0 のモードAブロックに p が含まれるため、
    λ1_max は (m, p) の両方の関数である。
    """
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
    """指定の (m_idx, p_idx, 負荷水準) に対する λ1 を返す。

    load_level: 'low' | 'medium' | 'high'
    """
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
    # ---- Step 1: 全 (m, p) について λ_{1,max} を計算 ----
    print("Step 1: Building λ_{1,max}(m, p) table ...")
    n_m = len(M_VALUES)
    n_p = len(P_VALUES)
    lam1_max_table = np.full((n_m, n_p), np.nan)

    total_lam1 = n_m * n_p
    with tqdm(total=total_lam1, desc="λ1_max", unit="pt") as pbar:
        for mi, m in enumerate(M_VALUES):
            for pi_idx, p in enumerate(P_VALUES):
                try:
                    lam1_max_table[mi, pi_idx] = _compute_lam1_max(m, p)
                except Exception as e:
                    tqdm.write(f"  [ERROR] m={m}, p={p:.2f}: {e}")
                pbar.update(1)

    # ---- Step 2: 全 (m, 負荷水準, p) の指標を計算 ----
    # results[m_idx, load_idx, p_idx, metric_idx]  metric: 0=main, 1=merge, 2=total
    results = np.full((n_m, 3, n_p, 3), np.nan)

    total_metrics = n_m * 3 * n_p   # = 315
    print(f"\nStep 2: Computing metrics ({total_metrics} total points) ...")

    with tqdm(total=total_metrics, desc="E[L]", unit="pt") as pbar:
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

                    # 安全性チェック：λ1 が λ1_max を超えないか確認
                    lam1_max = lam1_max_table[mi, pi_idx]
                    if not np.isnan(lam1_max) and lam1 >= lam1_max * 0.999:
                        tqdm.write(
                            f"  [SKIP] m={m}, p={p:.2f}, {load_level}: "
                            f"λ1={lam1:.4f} >= 0.999×λ1_max={lam1_max:.4f} (unstable)"
                        )
                        pbar.update(1)
                        continue

                    try:
                        metrics = _compute_metrics(m, p, lam1)
                        results[mi, li, pi_idx, :] = metrics
                    except Exception as e:
                        tqdm.write(
                            f"  [WARN] m={m}, p={p:.2f}, {load_level}: {e}"
                        )
                    pbar.update(1)

    # ---- Step 3: グラフ描画 ----
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
        fontsize=14,
        title=r"Threshold $m$",
        title_fontsize=16,
    )

    fig.suptitle(
        f"(K={K}, λ2={LAM2}, μ1_rand={MU1_RAND}, μ2_rand={MU2_RAND}, "
        f"μ1_zip={MU1_ZIP}, μ2_zip={MU2_ZIP})",
        fontsize=14, y=1.00,
    )

    plt.tight_layout()
    output_path = "experiment_5_p_sensitivity.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {output_path}")

    # ---- Step 4: λ1_max(m, p) ヒートマップを出力 ----
    print("\nStep 3: Generating λ1_max(m, p) heatmap...")
    # ヒートマップ用に P_VALUES_HEATMAP（0.1 刻み）の点だけ抽出
    heatmap_indices = [int(np.argmin(np.abs(P_VALUES - ph))) for ph in P_VALUES_HEATMAP]
    lam1_max_table_heatmap = lam1_max_table[:, heatmap_indices]
    _plot_lam1_max_heatmap(lam1_max_table_heatmap, M_VALUES, P_VALUES_HEATMAP)

    # ---- Step 5: 出力サマリ ----
    print("\n" + "=" * 70)
    print(
        f"{'m':<4} {'λ1_level':<10} {'p':>6} {'λ1':>8} "
        f"{'best_p':>8} {'min_E_L_total':>14}"
    )
    print("-" * 70)

    for mi, m in enumerate(M_VALUES):
        for pi_idx, p in enumerate(P_VALUES):
            for li, load_name in enumerate(LOAD_NAMES):
                lam1 = _get_lam1(mi, pi_idx, load_name, lam1_max_table)
                lam1_str = f"{lam1:.2f}" if not np.isnan(lam1) else "N/A"

                y_total = results[mi, li, pi_idx, 2]

                if np.isnan(y_total):
                    print(
                        f"{m:<4} {load_name:<10} {p:>6.2f} {lam1_str:>8} "
                        f"{'----':>8} {'N/A':>14}"
                    )
                    continue

                # 内点最小値を別途探索
                y_series = results[mi, li, :, 2]
                if np.all(np.isnan(y_series)):
                    print(
                        f"{m:<4} {load_name:<10} {p:>6.2f} {lam1_str:>8} "
                        f"{'----':>8} {'N/A':>14}"
                    )
                    continue

                min_idx = int(np.nanargmin(y_series))
                min_val = float(y_series[min_idx])
                best_p  = P_VALUES[min_idx]

                # m=0 は p に依存しないため "----" 表示
                if m == 0:
                    best_p_str = "----"
                elif min_idx == 0 or min_idx == n_p - 1:
                    best_p_str = f"{best_p:.2f}*"   # 端点最小は * で注記
                else:
                    best_p_str = f"{best_p:.2f}"

                print(
                    f"{m:<4} {load_name:<10} {p:>6.2f} {lam1_str:>8} "
                    f"{best_p_str:>8} {min_val:>14.4f}"
                )

    print("=" * 70)
    print("(* = boundary optimum, not interior)")


# ---------------------------------------------------------------------------
# λ1_max(m, p) ヒートマップの出力
# ---------------------------------------------------------------------------

def _plot_lam1_max_heatmap(
    lam1_max_table: np.ndarray,
    M_VALUES: list,
    P_VALUES_HEATMAP: np.ndarray,
    output_path: str = "experiment_5_lam1_max_heatmap.png",
):
    """λ1_max(m, p) の2次元ヒートマップを出力する。"""
    fig, ax = plt.subplots(figsize=(11, 6))

    # masked_invalid で NaN を除外
    data_masked = np.ma.masked_invalid(lam1_max_table)

    im = ax.imshow(
        data_masked,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[P_VALUES_HEATMAP[0] - 0.05, P_VALUES_HEATMAP[-1] + 0.05, -0.5, len(M_VALUES) - 0.5],
    )

    ax.set_yticks(range(len(M_VALUES)))
    ax.set_yticklabels([f"m={m}" for m in M_VALUES])
    ax.set_xticks(P_VALUES_HEATMAP)
    ax.set_xlabel("$p$", fontsize=16)
    ax.set_ylabel("Threshold $m$", fontsize=16)
    ax.set_title(
        f"(K={K}, λ2={LAM2})",
        fontsize=16,
    )

    cbar = plt.colorbar(im, ax=ax, label=r"$\lambda_{1,\mathrm{max}}$")

    # セルに数値を表示
    for mi in range(len(M_VALUES)):
        for pi_idx, p in enumerate(P_VALUES_HEATMAP):
            val = lam1_max_table[mi, pi_idx]
            if not np.isnan(val):
                # テキストの色を背景に応じて調整
                mean_val = np.nanmean(lam1_max_table)
                text_color = "white" if val < mean_val else "black"
                ax.text(
                    p, mi,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    color=text_color,
                    weight="normal",
                )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
