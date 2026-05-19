import numpy as np
from qbd_model import DynamicThresholdQBD


def make_model():
    return DynamicThresholdQBD(
        lam1=10, lam2=5, mu1_rand=40, mu2_rand=30, mu1_zip=30, mu2_zip=20, K=5, m=3, p=0.8
    )


def test_shapes():
    model = make_model()
    K = model.K
    n = 2 * K + 1
    n0 = K + 1

    assert model.Q0.shape == (n, n)
    assert model.Q_plus1.shape == (n, n)
    assert model.Q_minus1.shape == (n, n)

    assert model.B0.shape == (n0, n0)
    assert model.B_plus1.shape == (n0, n)
    assert model.B_minus1.shape == (n, n0)


def test_row_sums():
    model = make_model()

    # レベル1以上: Q_minus1 + Q0 + Q_plus1 の各行和が0
    A = model.Q_minus1 + model.Q0 + model.Q_plus1
    row_sums = np.sum(A, axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-10)

    # レベル0: B0 と B_plus1 の行和の合計が0
    B_rows = np.sum(model.B0, axis=1) + np.sum(model.B_plus1, axis=1)
    assert np.allclose(B_rows, 0.0, atol=1e-12)


def test_Q_minus1_block_structure():
    model = make_model()
    K = model.K

    # 合流車線処理中 (S=2) の行（偶数インデックス、n2>=1）には
    # レベル低下の遷移が存在しない（全て0）ことを確認
    for n2 in range(1, K + 1):
        idx = model._idx(2, n2)
        row = model.Q_minus1[idx, :]
        assert np.allclose(row, 0.0, atol=1e-12), f"Q_minus1 row at idx {idx} not zero: {row}"


def test_diagonals():
    model = make_model()

    # Q_plus1 の対角はすべて lam1
    lam1 = model.lam1
    assert np.allclose(np.diag(model.Q_plus1), lam1)

    # Q0 と B0 の対角はすべて負
    assert np.all(np.diag(model.Q0) < 0)
    assert np.all(np.diag(model.B0) < 0)
