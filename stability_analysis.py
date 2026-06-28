import warnings
import numpy as np
from qbd_model import DynamicThresholdQBD

try:
    from scipy.linalg import null_space as scipy_null_space
except ImportError:
    scipy_null_space = None


def _residual(eta: np.ndarray, A: np.ndarray) -> float:
    return float(np.max(np.abs(eta @ A)))


def _validate_eta(eta: np.ndarray, A: np.ndarray, tol: float = 1e-6) -> None:
    """η A = 0 かつ η e = 1 を検証し、逸脱時に警告を発する。"""
    res = _residual(eta, A)
    if res > tol:
        warnings.warn(
            f"validate_eta: η A ≠ 0  (|ηA|_inf = {res:.3e} > tol={tol:.3e})",
            stacklevel=3,
        )
    norm_err = abs(float(np.sum(eta)) - 1.0)
    if norm_err > tol:
        warnings.warn(
            f"validate_eta: η e ≠ 1  (|Ση - 1| = {norm_err:.3e} > tol={tol:.3e})",
            stacklevel=3,
        )


def _normalize(v: np.ndarray) -> np.ndarray:
    """正規化: 符号を正方向に揃え sum(eta)=1 にする。"""
    if np.sum(v) < 0:
        v = -v
    return v / np.sum(v)


def _solve_via_null_space(A: np.ndarray) -> np.ndarray:
    """scipy null_space (SVD ベース) で eta を求める。"""
    ns = scipy_null_space(A.T)
    if ns.shape[1] == 0:
        raise RuntimeError("null_space returned empty basis")
    return _normalize(ns[:, 0].real)


def _solve_via_lstsq(A_T: np.ndarray, b: np.ndarray, A: np.ndarray) -> np.ndarray:
    """lstsq で最小二乗解を求め、正規化する。"""
    eta, _, _, _ = np.linalg.lstsq(A_T, b, rcond=None)
    # lstsq の解は sum=1 条件を満たすが、残差が大きい場合は再正規化する
    return _normalize(eta)


def compute_exact_lambda1_max(model: DynamicThresholdQBD, tol: float = 1e-8) -> float:
    """
    行列計算によって定常分布 eta を求め、安定限界 lambda_{1,max} を計算する。

    フェーズ遷移行列 A = Q0 + Q_{+1} + Q_{-1} の定常分布 eta を
    連立一次方程式として直接解き、
        lambda_{1,max} = eta @ Q_{-1} @ e
    を返す。

    Parameters
    ----------
    model : DynamicThresholdQBD
    tol : float
        残差 |eta @ A|_inf の許容閾値（デフォルト 1e-8）。
        超えた場合はフォールバック手法で再計算し、警告を発する。
    """
    # 1. フェーズ遷移行列 A の構築
    A = model.Q0 + model.Q_plus1 + model.Q_minus1
    n = A.shape[0]

    # 2. eta A = 0, eta e = 1 を線形系に変換
    A_T = A.T.copy()
    A_T[-1, :] = 1.0          # 最後の行を正規化条件に置換
    b = np.zeros(n)
    b[-1] = 1.0

    # 3. 主手法: np.linalg.solve（高速・直接法）
    eta = np.linalg.solve(A_T, b)
    res = _residual(eta, A)

    # 4. 残差チェック → 閾値超えならフォールバック
    if res > tol:
        warnings.warn(
            f"compute_exact_lambda1_max: numerical precision warning "
            f"(|eta @ A|_inf = {res:.3e} > tol={tol:.3e}, n={n}). "
            f"Trying fallback solvers.",
            stacklevel=2,
        )

        candidates: list[tuple[float, np.ndarray, str]] = [(res, eta, "solve")]

        # フォールバック 1: scipy null_space（SVD ベース、最も安定）
        if scipy_null_space is not None:
            try:
                eta_ns = _solve_via_null_space(A)
                candidates.append((_residual(eta_ns, A), eta_ns, "null_space"))
            except Exception:
                pass

        # フォールバック 2: np.linalg.lstsq（最小二乗、scipy なし環境向け）
        try:
            eta_ls = _solve_via_lstsq(A_T, b, A)
            candidates.append((_residual(eta_ls, A), eta_ls, "lstsq"))
        except Exception:
            pass

        best_res, eta, best_method = min(candidates, key=lambda x: x[0])

        if best_res > tol:
            warnings.warn(
                f"compute_exact_lambda1_max: all solvers exceeded tol "
                f"(best |eta @ A|_inf = {best_res:.3e} via {best_method}). "
                f"Result may be inaccurate.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"compute_exact_lambda1_max: fallback '{best_method}' succeeded "
                f"(|eta @ A|_inf = {best_res:.3e}).",
                stacklevel=2,
            )

    # 5. η A = 0, η e = 1 の最終検証
    _validate_eta(eta, A, tol=max(tol, 1e-6))

    # 6. lambda_{1,max} = eta @ Q_{-1} @ e
    ones = np.ones(n)
    return float(eta @ model.Q_minus1 @ ones)


def _spectral_radius_R(model: DynamicThresholdQBD) -> float:
    """公比行列 R のスペクトル半径を返す（安定性の確認用）。"""
    R = model.solve_R()
    return float(np.max(np.abs(np.linalg.eigvals(R))))


if __name__ == "__main__":
    # --- テスト 1: 基本的な動作確認 ---
    print("=" * 60)
    print("Test 1: K=10, m=5, p=0.5")
    print("=" * 60)

    base_params = dict(
        lam2=5.0,
        mu1_rand=20.0,
        mu2_rand=15.0,
        mu1_zip=15.0,
        mu2_zip=10.0,
        K=10,
        m=5,
        p=0.5,
    )

    # A = Q0 + Q_plus1 + Q_minus1 は lam1 に依存しない（Q_plus1 の寄与が Q0 対角で相殺）
    # ため、probe 用の lam1 は任意で良い
    model_probe = DynamicThresholdQBD(lam1=5.0, **base_params)
    lam1_max = compute_exact_lambda1_max(model_probe)
    print(f"lambda_1_max = {lam1_max:.6f}")

    # lam1 が変わっても A が不変なので lambda1_max は同じ値になることを確認
    for lam1_check in [1.0, 5.0, lam1_max * 0.5, lam1_max * 2.0]:
        m_check = DynamicThresholdQBD(lam1=lam1_check, **base_params)
        v = compute_exact_lambda1_max(m_check)
        print(f"  lam1={lam1_check:.4f} -> lambda1_max={v:.6f}  (diff={v - lam1_max:.2e})")

    # --- テスト 2: 安定・不安定の境界検証 ---
    print()
    print("=" * 60)
    print("Test 2: stability boundary verification")
    print("=" * 60)
    print("  [Note] For unstable QBDs, the minimal non-negative R has rho(R)->1^-")
    print("         numerically. Direct comparison lam1 vs lambda1_max is reliable.")

    for lam1_test in [lam1_max * 0.7, lam1_max * 0.9, lam1_max * 1.1, lam1_max * 1.3]:
        stable_by_formula = lam1_test < lam1_max
        model_test = DynamicThresholdQBD(lam1=lam1_test, **base_params)
        try:
            rho_R = _spectral_radius_R(model_test)
        except RuntimeError:
            rho_R = float("nan")
        print(
            f"  lam1={lam1_test:7.4f}  lambda1_max={lam1_max:.4f}"
            f"  stable={stable_by_formula}  rho(R)={rho_R:.6f}"
        )

    # --- テスト 3: 異なるパラメータ設定 ---
    print()
    print("=" * 60)
    print("Test 3: K=5, m=2, p=0.8")
    print("=" * 60)

    model2 = DynamicThresholdQBD(
        lam1=10.0, lam2=3.0,
        mu1_rand=30.0, mu2_rand=20.0,
        mu1_zip=25.0, mu2_zip=15.0,
        K=5, m=2, p=0.8,
    )
    lam1_max2 = compute_exact_lambda1_max(model2)
    print(f"lambda_1_max = {lam1_max2:.6f}")

    # eta の検証: eta @ A がほぼ零ベクトルであることを確認
    A2 = model2.Q0 + model2.Q_plus1 + model2.Q_minus1
    n2 = A2.shape[0]
    A2_T = A2.T.copy()
    A2_T[-1, :] = 1.0
    b2 = np.zeros(n2); b2[-1] = 1.0
    eta2 = np.linalg.solve(A2_T, b2)
    residual = np.max(np.abs(eta2 @ A2))
    print(f"  |eta @ A|_inf (should be ~0): {residual:.2e}")
    print(f"  sum(eta) (should be 1):       {np.sum(eta2):.10f}")
