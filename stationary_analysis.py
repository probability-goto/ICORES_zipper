import numpy as np

try:
    from scipy.linalg import null_space
except Exception:
    null_space = None

from qbd_model import DynamicThresholdQBD


def compute_stationary(model: DynamicThresholdQBD, R: np.ndarray = None, tol=1e-12):
    """R を用いて定常分布 `pi_0` と `pi_1` を計算する。

    返り値は `(pi0, pi1, denom)` で、`pi0` は長さ `K+1`、`pi1` は長さ `2K+1`。
    手順:
    1. 境界方程式のブロック行列 M = [[B0, B_{+1}], [B_{-1}, Q0 + R Q_{-1}]] を作成
    2. M^T の零空間（`null_space`）を計算し、左零ベクトル x (x M = 0) を得る
       - `scipy.linalg.null_space` が空を返す場合は SVD によるフォールバックを行う
    3. x を `tilde_pi0`, `tilde_pi1` に分割し、正規化条件により正規化定数 denom を求める
    4. `pi0 = tilde_pi0 / denom`, `pi1 = tilde_pi1 / denom` を返す
    """

    if R is None:
        R = model.solve_R(tol=tol)

    B0 = model.B0
    B_plus1 = model.B_plus1
    B_minus1 = model.B_minus1
    Q0 = model.Q0
    Q_minus1 = model.Q_minus1

    # ブロック行列 M を構築
    bottom_right = Q0 + R @ Q_minus1
    M = np.block([[B0, B_plus1], [B_minus1, bottom_right]])

    # 左零ベクトル x (x M = 0) を得るため、M^T の零空間を求める
    ns = None
    if null_space is not None:
        ns = null_space(M.T)

    # null_space が存在しない、または空だった場合は SVD によるフォールバック
    if ns is None or ns.size == 0:
        # SVD を使って最小特異値に対応する右特異ベクトルを取る
        U, s, Vt = np.linalg.svd(M.T, full_matrices=False)
        ns = Vt.T[:, -1:]

    if ns.size == 0:
        raise RuntimeError("M^T の零空間が得られませんでした。数値的にフルランクです。")

    x = ns[:, 0]
    # 微小な虚部が混じっている場合は実部に丸める
    x = np.real_if_close(x, tol=1000)

    # 合計値が負であれば反転して正の成分を持たせる
    if np.sum(x) < 0:
        x = -x

    n0 = B0.shape[0]
    n = Q0.shape[0]

    tilde_pi0 = x[:n0].astype(float)
    tilde_pi1 = x[n0:n0 + n].astype(float)

    # (I - R)^{-1} を安全に計算
    I = np.eye(n)
    try:
        inv_I_minus_R = np.linalg.inv(I - R)
    except np.linalg.LinAlgError:
        inv_I_minus_R = np.linalg.pinv(I - R)

    ones0 = np.ones(n0)
    ones = np.ones(n)

    denom = float(tilde_pi0 @ ones0 + tilde_pi1 @ (inv_I_minus_R @ ones))
    if denom == 0 or np.isnan(denom):
        raise RuntimeError("正規化定数が0またはNaNです。")

    pi0 = tilde_pi0 / denom
    pi1 = tilde_pi1 / denom

    return pi0, pi1, denom


def calculate_performance_measures(model: DynamicThresholdQBD, R: np.ndarray, pi0: np.ndarray, pi1: np.ndarray):
        """定常分布から性能指標を計算する。

        返り値:
            - E_L_main: 本線の平均系内客数 E[L_main]
            - E_L_merge: 合流車線の平均系内客数 E[L_merge]
            - P_block: 合流車線が満杯である確率 P_block

        実装メモ:
            - (I - R)^{-1} や (I - R)^{-2} の計算は `np.linalg.solve` を用いて
                数値的安定性を高める（逆行列を直接計算しない）。
            - フェーズの並び順は `qbd_model.py` に従う。
        """

        n0 = model.num_phases_0
        n = model.num_phases

        # 基本行列
        I = np.eye(n)
        A = I - R

        # (I - R)^{-1} を RHS を指定して求める（解法を2回呼んで (I-R)^{-2} を得る）
        ones = np.ones(n)
        # solve(A, ones) = (I-R)^{-1} * ones
        y = np.linalg.solve(A, ones)
        # solve(A, y) = (I-R)^{-2} * ones
        z = np.linalg.solve(A, y)

        # E[L_main] = pi1 * (I - R)^{-2} * e
        E_L_main = float(pi1 @ z)

        # 合流車線の期待値 E[L_merge]
        # レベル0の n2 値配列: (0,0), (2,1), (2,2), ..., (2,K)
        n2_level0 = np.arange(0, n0)

        # レベル1以上のフェーズごとの n2 値配列: (1,0), (1,1), (2,1), (1,2), (2,2), ...
        n2_level1 = np.empty(n, dtype=int)
        n2_level1[0] = 0
        idx = 1
        for val in range(1, model.K + 1):
                n2_level1[idx] = val
                idx += 1
                n2_level1[idx] = val
                idx += 1

        # レベル1以上の各位相における総確率ベクトル p_levels = pi1 * (I - R)^{-1}
        # 数値的に安定に求めるため、(I-R)^T x = pi1^T を解き x^T = pi1 * (I-R)^{-1}
        p_levels = np.linalg.solve(A.T, pi1).T

        # E[L_merge] = sum_{level0} pi0[i] * n2_level0[i] + sum_{level>=1 phases} p_levels[j] * n2_level1[j]
        E_L_merge = float(np.dot(pi0, n2_level0) + np.dot(p_levels, n2_level1))

        # ブロッキング確率: レベル0 の末尾要素 + レベル1以上の n2=K に対応する末尾2要素の和
        P_block = float(pi0[-1] + np.sum(p_levels[-2:]))

        return E_L_main, E_L_merge, P_block


if __name__ == "__main__":
    # Example run and quick checks
    model = DynamicThresholdQBD(
        lam1=10, lam2=5, mu1_rand=40, mu2_rand=30, mu1_zip=30, mu2_zip=20, K=5, m=3, p=0.8
    )

    print("Computing R (this may take a few moments)...")
    R = model.solve_R(verbose=True)
    print("R computed. shape:", R.shape)

    pi0, pi1, denom = compute_stationary(model, R=R)

    n = R.shape[0]
    I = np.eye(n)
    inv_I_minus_R = np.linalg.inv(I - R)

    print("sum(pi0):", np.sum(pi0))
    print("pi1*(I-R)^{-1} ones sum:", float(pi1 @ (inv_I_minus_R @ np.ones(n))))
    print("normalization denom:", denom)
    print("total probability (should be 1):", np.sum(pi0) + float(pi1 @ (inv_I_minus_R @ np.ones(n))))

    print("pi0 (length {}):".format(pi0.size), pi0)
    print("pi1 (length {}):".format(pi1.size), pi1)

    # 性能指標を計算して表示
    E_L_main, E_L_merge, P_block = calculate_performance_measures(model, R, pi0, pi1)
    print()
    print("Performance measures:")
    print("E[L_main]:", E_L_main)
    print("E[L_merge]:", E_L_merge)
    print("P_block:", P_block)
