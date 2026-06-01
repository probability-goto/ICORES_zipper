"""
実験群4：先行研究との比較によるトレードオフ完全

横軸 K（5〜30）、縦軸 λ1 の空間において、
  - 最大スループット λ_{1,max} の安定限界（黒太線）
  - ブロッキング確率 P_block の等高線（1%, 5%, 10%, 15%, 20%）
を描画する。

  左サブプロット : m = 0（先行研究 ICORES 2026 の再現）
  右サブプロット : m = int(0.6 * K)（提案手法）

提案手法では等高線が上に押し上げられ、同一容量 K でより高い λ1 に耐えられること
（トレードオフ完全突破）を視覚的に証明する。

出力: experiment_4_contour.png
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

from qbd_model import DynamicThresholdQBD
from stability_analysis import compute_exact_lambda1_max
from stationary_analysis import compute_stationary, calculate_performance_measures

# ---------------------------------------------------------------------------
# パラメータ設定
# ---------------------------------------------------------------------------
MU1_RAND   = 30
MU2_RAND   = 20
MU1_ZIP    = 30
MU2_ZIP    = 20
LAM2       = 10
P          = 0.5
LAM1_DUMMY = 5.0      # λ_{1,max} 計算時のダミー値（A 行列は lam1 に依存しない）

K_VALUES = list(range(5, 31))   # 5 から 30

CONTOUR_LEVELS = [0.01, 0.05, 0.10, 0.15, 0.20]
# CONTOUR_LEVELS = [0.0005, 0.001, 0.005, 0.01, 0.05]
N_LAM1_PER_K  = 20   # 各 K の λ1 グリッド点数

STRATEGIES = [
    {
        "label"  : r"$m = 0$",
        "m_func" : lambda K: 0,
    },
    {
        "label"  : r"$m = \lfloor 0.6K \rfloor$",
        "m_func" : lambda K: int(0.6 * K),
    },
]


# ---------------------------------------------------------------------------
# Step 1 + 2: λ_{1,max} と P_block グリッドの計算
# ---------------------------------------------------------------------------

def compute_strategy(m_func, desc):
    """
    戦略ごとに全 (K, λ1) 組み合わせの P_block を計算する。

    Returns
    -------
    lambda1_max_arr : ndarray, shape (n_K,)
        各 K の安定限界 λ_{1,max}
    lam1_grid_2d : ndarray, shape (N_LAM1_PER_K, n_K)
        各列が K ごとの λ1 グリッド（nan は未計算列）
    P_block_2d : ndarray, shape (N_LAM1_PER_K, n_K)
        対応する P_block（λ1 >= λ_{1,max} または計算失敗は nan）
    """
    n_K = len(K_VALUES)
    lambda1_max_arr = np.full(n_K, np.nan)
    lam1_grid_2d    = np.full((N_LAM1_PER_K, n_K), np.nan)
    P_block_2d      = np.full((N_LAM1_PER_K, n_K), np.nan)

    # ---- Phase 1: λ_{1,max} を全 K について計算 ----
    print(f"\n[{desc}]")
    print(f"  Phase 1/2: Computing λ_{{1,max}} for K = {K_VALUES[0]}..{K_VALUES[-1]}")
    for ki, K in enumerate(tqdm(K_VALUES, desc="  λ₁_max", leave=False)):
        m = m_func(K)
        try:
            model = DynamicThresholdQBD(
                lam1=LAM1_DUMMY, lam2=LAM2,
                mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
                mu1_zip=MU1_ZIP,  mu2_zip=MU2_ZIP,
                K=K, m=m, p=P,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                val = compute_exact_lambda1_max(model, tol=1e-8)
            for w in caught:
                tqdm.write(f"    [WARN] K={K}, m={m}: {w.message}")
            lambda1_max_arr[ki] = val
        except Exception as e:
            tqdm.write(f"    [ERROR] K={K}, m={m}: {e}")

    # ---- Phase 2: 各 (K, λ1) の P_block を計算 ----
    total = n_K * N_LAM1_PER_K
    print(f"  Phase 2/2: Computing P_block grid "
          f"({N_LAM1_PER_K} pts × {n_K} K-values = {total} pairs)")

    with tqdm(total=total, desc="  P_block", leave=True) as pbar:
        for ki, K in enumerate(K_VALUES):
            m          = m_func(K)
            lam1_max   = lambda1_max_arr[ki]

            if np.isnan(lam1_max) or lam1_max <= 1.0:
                pbar.update(N_LAM1_PER_K)
                continue

            lam1_upper = lam1_max * 0.98
            lam1_lower = 1.0
            lam1_vals  = np.linspace(lam1_lower, lam1_upper, N_LAM1_PER_K)
            lam1_grid_2d[:, ki] = lam1_vals

            for li, lam1 in enumerate(lam1_vals):
                try:
                    model = DynamicThresholdQBD(
                        lam1=lam1, lam2=LAM2,
                        mu1_rand=MU1_RAND, mu2_rand=MU2_RAND,
                        mu1_zip=MU1_ZIP,  mu2_zip=MU2_ZIP,
                        K=K, m=m, p=P,
                    )
                    R = model.solve_R()
                    pi0, pi1, _ = compute_stationary(model, R)
                    _, _, P_block = calculate_performance_measures(model, R, pi0, pi1)
                    P_block_2d[li, ki] = P_block
                except Exception as e:
                    tqdm.write(f"    [WARN] K={K}, m={m}, λ1={lam1:.3f}: {e}")
                    # 計算失敗は nan のまま
                pbar.update(1)

    return lambda1_max_arr, lam1_grid_2d, P_block_2d


# ---------------------------------------------------------------------------
# Step 3: 共通グリッドへの補間
# ---------------------------------------------------------------------------

def interpolate_to_common_grid(
    lambda1_max_arr, lam1_grid_2d, P_block_2d, lam1_common
):
    """
    各 K 列の P_block を共通の lam1_common グリッドへ線形補間する。
    λ_{1,max} 以上の領域は nan とする（plt.contour の未定義域）。

    Returns
    -------
    P_block_common : ndarray, shape (len(lam1_common), len(K_VALUES))
    """
    n_common = len(lam1_common)
    n_K      = len(K_VALUES)
    P_block_common = np.full((n_common, n_K), np.nan)

    for ki in range(n_K):
        col_lam1 = lam1_grid_2d[:, ki]
        col_pb   = P_block_2d[:, ki]
        valid    = ~np.isnan(col_lam1) & ~np.isnan(col_pb)

        if np.sum(valid) < 2:
            continue

        # 共通グリッドへ線形補間（範囲外は nan）
        interpolated = np.interp(
            lam1_common,
            col_lam1[valid],
            col_pb[valid],
            left=np.nan,
            right=np.nan,
        )

        # λ_{1,max} 以上は明示的に nan
        lam1_max_k = lambda1_max_arr[ki]
        if not np.isnan(lam1_max_k):
            interpolated[lam1_common >= lam1_max_k] = np.nan

        P_block_common[:, ki] = interpolated

    return P_block_common


# ---------------------------------------------------------------------------
# Step 4: 1 サブプロットの描画
# ---------------------------------------------------------------------------

def draw_subplot(ax, K_values, lambda1_max_arr, P_block_common, lam1_common, title):
    """
    単一サブプロットに安定限界と P_block 等高線を描画する。
    """
    K_arr = np.array(K_values, dtype=float)
    K_mesh, lam1_mesh = np.meshgrid(K_arr, lam1_common)

    # ---- P_block 等高線（着色なし輪郭線のみ）----
    P_masked = np.ma.masked_invalid(P_block_common)
    colors   = ["royalblue", "forestgreen", "darkorange", "crimson", "purple"]

    try:
        cs = ax.contour(
            K_mesh, lam1_mesh, P_masked,
            levels=CONTOUR_LEVELS,
            colors=colors,
            linewidths=1.8,
            zorder=3,
        )
        fmt = {lv: f"{int(round(lv * 100))}%" for lv in CONTOUR_LEVELS}
        ax.clabel(cs, fmt=fmt, fontsize=9, inline=True, inline_spacing=5)
    except Exception as e:
        print(f"  [WARN] contour failed for '{title}': {e}")

    # ---- 安定限界（黒太線）----
    ax.plot(
        K_arr, lambda1_max_arr,
        color="black", linewidth=2.8, linestyle="-",
        label=r"$\lambda_{1,\max}$ (stability boundary)",
        zorder=5,
    )

    # ---- 軸・グリッド設定 ----
    ax.set_xlim(K_arr[0] - 0.5, K_arr[-1] + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.set_xlabel("Capacity $K$", fontsize=13)
    ax.set_ylabel(r"Arrival Rate $\lambda_1$", fontsize=13)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, which="major", alpha=0.35, linestyle="--")
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")
    ax.legend(loc="upper left", fontsize=10)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    results = []
    for strategy in STRATEGIES:
        lam1_max, lam1_grid, P_block_grid = compute_strategy(
            strategy["m_func"], strategy["label"]
        )
        results.append((lam1_max, lam1_grid, P_block_grid))

    # ---- 共通 Y 軸グリッドを決定（全戦略の最大 λ_{1,max} を包含）----
    all_lam1_max = np.concatenate([r[0] for r in results])
    lam1_global_max = np.nanmax(all_lam1_max) * 1.02
    lam1_global_min = 0.8
    N_COMMON = 120
    lam1_common = np.linspace(lam1_global_min, lam1_global_max, N_COMMON)

    # ---- 各戦略を共通グリッドへ補間 ----
    P_common_list = [
        interpolate_to_common_grid(lam1_max, lam1_grid, P_block_grid, lam1_common)
        for lam1_max, lam1_grid, P_block_grid in results
    ]

    # ---- Figure 作成（1行2列, sharey=True）----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    titles = [
        r"$m = 0$   ",
        r"$m = \lfloor 0.6K \rfloor$",
    ]

    for ax, (lam1_max, _, _), P_common, title in zip(
        axes, results, P_common_list, titles
    ):
        draw_subplot(ax, K_VALUES, lam1_max, P_common, lam1_common, title)

    # sharey=True のため右サブプロットの Y ラベルは不要
    axes[1].set_ylabel("")

    # Y 軸範囲を共通設定
    axes[0].set_ylim(lam1_global_min, lam1_global_max)

    # ---- 全体タイトル ----
    param_str = (
        rf"$\lambda_2={LAM2}$, $p={P}$, "
        rf"$\mu_1^\mathrm{{rand}}={MU1_RAND}$, "
        rf"$\mu_2^\mathrm{{rand}}={MU2_RAND}$, "
        rf"$\mu_1^\mathrm{{zip}}={MU1_ZIP}$, "
        rf"$\mu_2^\mathrm{{zip}}={MU2_ZIP}$"
    )
    fig.suptitle(
        "Experiment 4: Proof of Trade-off Breakthrough\n"
        r"Contours of $P_\mathrm{block}$ and Stability Boundary in ($K$, $\lambda_1$) Space"
        f"\n({param_str})",
        fontsize=12,
        y=1.02,
    )

    plt.tight_layout()
    output_path = "experiment_4_contour.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
