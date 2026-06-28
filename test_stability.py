"""
stability_analysis.compute_exact_lambda1_max のテスト

検証対象:
  - η A = 0  (|ηA|_inf が tol 以下)
  - η e = 1  (sum(η) = 1)
  - _validate_eta が違反時に警告を発する
  - 各種パラメータで λ1_max が正値かつ λ1 依存しない
"""

import warnings
import numpy as np
import pytest

from qbd_model import DynamicThresholdQBD
from stability_analysis import compute_exact_lambda1_max, _validate_eta

# ---------------------------------------------------------------------------
# 共通パラメータ
# ---------------------------------------------------------------------------
BASE = dict(
    lam2=10.0,
    mu1_rand=40.0, mu2_rand=30.0,
    mu1_zip=30.0,  mu2_zip=20.0,
    K=10,
)


def _make(m, p, lam1=1.0, **kw):
    params = {**BASE, **kw}
    return DynamicThresholdQBD(lam1=lam1, m=m, p=p, **params)


def _eta_and_A(model):
    """η を直接計算して (η, A) を返す。"""
    A = model.Q0 + model.Q_plus1 + model.Q_minus1
    n = A.shape[0]
    A_T = A.T.copy()
    A_T[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    eta = np.linalg.solve(A_T, b)
    return eta, A


# ---------------------------------------------------------------------------
# η A = 0 の検証
# ---------------------------------------------------------------------------

class TestEtaAEqualsZero:
    """compute_exact_lambda1_max が返す η は η A ≈ 0 を満たす。"""

    @pytest.mark.parametrize("m,p", [
        (0, 0.0), (0, 0.5), (0, 1.0),
        (5, 0.0), (5, 0.5), (5, 1.0),
        (10, 0.0), (10, 0.5), (10, 1.0),
    ])
    def test_residual_below_tol(self, m, p):
        model = _make(m, p)
        eta, A = _eta_and_A(model)
        residual = float(np.max(np.abs(eta @ A)))
        assert residual < 1e-8, (
            f"m={m}, p={p}: |ηA|_inf = {residual:.3e} exceeds 1e-8"
        )

    @pytest.mark.parametrize("m,p", [(3, 0.3), (7, 0.7)])
    def test_residual_with_small_K(self, m, p):
        model = _make(m, p, K=5)
        eta, A = _eta_and_A(model)
        residual = float(np.max(np.abs(eta @ A)))
        assert residual < 1e-8, (
            f"K=5, m={m}, p={p}: |ηA|_inf = {residual:.3e} exceeds 1e-8"
        )


# ---------------------------------------------------------------------------
# η e = 1 の検証
# ---------------------------------------------------------------------------

class TestEtaSumEqualsOne:
    """compute_exact_lambda1_max が返す η は sum(η) = 1 を満たす。"""

    @pytest.mark.parametrize("m,p", [
        (0, 0.0), (0, 1.0),
        (5, 0.5),
        (10, 0.0), (10, 1.0),
    ])
    def test_sum_equals_one(self, m, p):
        model = _make(m, p)
        eta, _ = _eta_and_A(model)
        norm_err = abs(float(np.sum(eta)) - 1.0)
        assert norm_err < 1e-10, (
            f"m={m}, p={p}: |Ση - 1| = {norm_err:.3e} exceeds 1e-10"
        )

    @pytest.mark.parametrize("m,p", [(3, 0.3), (7, 0.7)])
    def test_sum_equals_one_small_K(self, m, p):
        model = _make(m, p, K=5)
        eta, _ = _eta_and_A(model)
        norm_err = abs(float(np.sum(eta)) - 1.0)
        assert norm_err < 1e-10, (
            f"K=5, m={m}, p={p}: |Ση - 1| = {norm_err:.3e} exceeds 1e-10"
        )


# ---------------------------------------------------------------------------
# _validate_eta の警告動作
# ---------------------------------------------------------------------------

class TestValidateEta:
    """_validate_eta は条件違反時のみ警告を発する。"""

    def _valid_eta_and_A(self):
        model = _make(m=5, p=0.5)
        return _eta_and_A(model)

    def test_no_warning_for_valid_eta(self):
        eta, A = self._valid_eta_and_A()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _validate_eta(eta, A, tol=1e-6)
        assert len(caught) == 0, (
            f"正常な η に対して警告が発生した: {[str(w.message) for w in caught]}"
        )

    def test_warns_when_eta_A_nonzero(self):
        eta, A = self._valid_eta_and_A()
        eta_bad = eta + 0.1  # η A ≠ 0 になるよう意図的に壊す
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _validate_eta(eta_bad, A, tol=1e-6)
        messages = [str(w.message) for w in caught]
        assert any("η A ≠ 0" in m for m in messages), (
            f"η A ≠ 0 の違反で警告が出なかった。実際のメッセージ: {messages}"
        )

    def test_warns_when_sum_not_one(self):
        eta, A = self._valid_eta_and_A()
        eta_bad = eta * 2.0  # sum(η) = 2 になるよう壊す
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _validate_eta(eta_bad, A, tol=1e-6)
        messages = [str(w.message) for w in caught]
        assert any("η e ≠ 1" in m for m in messages), (
            f"η e ≠ 1 の違反で警告が出なかった。実際のメッセージ: {messages}"
        )


# ---------------------------------------------------------------------------
# λ1_max の基本性質
# ---------------------------------------------------------------------------

class TestLambda1Max:
    """compute_exact_lambda1_max の出力が基本的な性質を満たす。"""

    @pytest.mark.parametrize("m,p", [
        (0, 0.0), (0, 0.5), (0, 1.0),
        (5, 0.5),
        (10, 0.0), (10, 1.0),
    ])
    def test_positive(self, m, p):
        model = _make(m, p)
        val = compute_exact_lambda1_max(model)
        assert val > 0, f"m={m}, p={p}: λ1_max = {val} ≤ 0"

    @pytest.mark.parametrize("m,p", [(5, 0.5), (3, 0.3)])
    def test_independent_of_lam1(self, m, p):
        """A は λ1 に依存しないので、異なる lam1 でも同じ λ1_max が得られる。"""
        results = [
            compute_exact_lambda1_max(_make(m, p, lam1=v))
            for v in [0.1, 1.0, 5.0, 20.0]
        ]
        spread = max(results) - min(results)
        assert spread < 1e-10, (
            f"m={m}, p={p}: lam1 を変えると λ1_max が変動する (spread={spread:.3e})"
        )

    def test_m0_independent_of_p(self):
        """m=0 のとき A は p に依存しないので λ1_max も p 不変のはず。"""
        results = [
            compute_exact_lambda1_max(_make(m=0, p=pv))
            for pv in np.linspace(0.0, 1.0, 6)
        ]
        spread = max(results) - min(results)
        assert spread < 1e-10, (
            f"m=0: p を変えると λ1_max が変動する (spread={spread:.3e})"
        )
