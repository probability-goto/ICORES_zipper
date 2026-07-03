"""
実験群3：K と m の相互作用の 2 次元解析

合流車線容量 K（5〜30）と動的閾値 m（0〜K）のすべての組み合わせについて
λ_{1,max} を計算し、「最適なパラメータ帯（スウィートスポット）」が
比率 m/K として存在するかをヒートマップで可視化する。

出力: experiment_3_heatmap.png
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

from qbd_model import DynamicThresholdQBD
from stability_analysis import compute_exact_lambda1_max

# ---------------------------------------------------------------------------
# パラメータ設定
# ---------------------------------------------------------------------------
MU1_RAND   = 40
MU2_RAND   = 30
MU1_ZIP    = 30
MU2_ZIP    = 20
LAM2       = 10
P          = 0.5
LAM1_DUMMY = 5.0     # A = Q0+Q_{+1}+Q_{-1} は lam1 に依存しないためダミーで可

K_MIN, K_MAX = 5, 30
K_VALUES = np.arange(K_MIN, K_MAX + 1, dtype=int)   # 5 ～ 30 (26 種)

# m/K 軸の解像度（細かいほど滑らか。pcolormesh 用セル中心）
N_RATIO = 201   # 0.000 〜 1.000 を 201 点

# ---------------------------------------------------------------------------
# Step 1: 全有効 (K, m) の λ_{1,max} を事前計算
# ---------------------------------------------------------------------------

def compute_all() -> dict:
    """全 (K, m) ペアの λ_{1,max} を計算してキャッシュとして返す。"""
    pairs = [(int(K), m) for K in K_VALUES for m in range(0, int(K) + 1)]

    cache: dict[tuple[int, int], float] = {}
    for K, m in tqdm(pairs, desc="Computing λ₁_max", unit="pair"):
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
                tqdm.write(f"[WARNING] K={K}, m={m}: {w.message}")
            cache[(K, m)] = val
        except Exception as e:
            tqdm.write(f"[ERROR] K={K}, m={m}: {e}")
            cache[(K, m)] = np.nan

    return cache


# ---------------------------------------------------------------------------
# Step 2: 規則格子（K, m/K）への投影
# ---------------------------------------------------------------------------

def build_grid(cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    K_mesh, ratio_mesh : 2D meshgrid arrays  shape (N_RATIO, len(K_VALUES))
    Z                  : λ_{1,max} values     same shape
    """
    ratio_centers = np.linspace(0.0, 1.0, N_RATIO)

    Z = np.full((N_RATIO, len(K_VALUES)), np.nan)
    for ki, K in enumerate(K_VALUES):
        for ri, ratio in enumerate(ratio_centers):
            # ratio に最も近い整数 m にスナップ（0 ≤ m ≤ K を保証）
            m = int(np.clip(round(ratio * K), 0, K))
            Z[ri, ki] = cache.get((int(K), m), np.nan)

    K_mesh, ratio_mesh = np.meshgrid(K_VALUES, ratio_centers)
    return K_mesh, ratio_mesh, Z


# ---------------------------------------------------------------------------
# Step 3: ヒートマップとして可視化
# ---------------------------------------------------------------------------

def plot(K_mesh, ratio_mesh, Z, output_path="experiment_3_heatmap.png"):
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- ヒートマップ本体 ---
    pcm = ax.pcolormesh(
        K_mesh, ratio_mesh, Z,
        cmap="plasma",
        shading="nearest",
        vmin=np.nanmin(Z),
        vmax=np.nanmax(Z),
    )

    # --- 等高線オーバーレイ ---
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    # 視認性のため 8 本の等高線を均等配置
    contour_levels = np.linspace(z_min, z_max, 10)[1:-1]
    cs = ax.contour(
        K_mesh, ratio_mesh, Z,
        levels=contour_levels,
        colors="white",
        linewidths=0.9,
        alpha=0.7,
    )
    ax.clabel(cs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=4)

    # --- カラーバー ---
    cbar = fig.colorbar(pcm, ax=ax, pad=0.015, fraction=0.04)
    cbar.set_label(
        r"Max Throughput $\lambda_{1,\mathrm{max}}$",
        fontsize=13,
    )
    cbar.ax.tick_params(labelsize=10)

    # --- 軸設定 ---
    ax.set_xlabel("$K$", fontsize=16)
    ax.set_ylabel("Threshold Ratio $m/K$", fontsize=16)
    ax.set_xlim(K_VALUES[0] - 0.5, K_VALUES[-1] + 0.5)
    ax.set_ylim(-0.01, 1.01)

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis="both", which="major", labelsize=16)

    # --- 参照水平線（特定比率を強調） ---
    for ratio_ref, ls, alpha in [(0.25, "--", 0.55), (0.5, "-.", 0.55), (0.75, ":", 0.55)]:
        ax.axhline(ratio_ref, color="white", linestyle=ls, linewidth=1.1, alpha=alpha)

    # --- タイトル ---
    ax.set_title(
        rf"$\lambda_2={LAM2}$, $p={P}$, "
        rf"$\mu_1^{{\mathrm{{rand}}}}={MU1_RAND}$, "
        rf"$\mu_2^{{\mathrm{{rand}}}}={MU2_RAND}$, "
        rf"$\mu_1^{{\mathrm{{zip}}}}={MU1_ZIP}$, "
        rf"$\mu_2^{{\mathrm{{zip}}}}={MU2_ZIP}$",
        fontsize=18,
        pad=10,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {output_path}")


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main():
    cache = compute_all()
    K_mesh, ratio_mesh, Z = build_grid(cache)

    # --- 簡易サマリ（コンソール出力） ---
    print(f"\nλ₁_max range : {np.nanmin(Z):.4f} – {np.nanmax(Z):.4f}")
    # 各 K について λ_{1,max} を最大にする m/K を報告
    print(f"{'K':>4}  {'best m':>6}  {'best m/K':>8}  {'λ₁_max':>8}")
    for ki, K in enumerate(K_VALUES):
        col = Z[:, ki]
        if np.all(np.isnan(col)):
            continue
        best_ri = int(np.nanargmax(col))
        best_ratio = np.linspace(0, 1, N_RATIO)[best_ri]
        best_m = int(round(best_ratio * K))
        print(f"{K:>4}  {best_m:>6}  {best_ratio:>8.3f}  {col[best_ri]:>8.4f}")

    plot(K_mesh, ratio_mesh, Z)


if __name__ == "__main__":
    main()
