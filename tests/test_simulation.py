import pytest

from qbd_model import DynamicThresholdQBD
from stationary_analysis import compute_stationary, calculate_performance_measures
from simulation import simulate


def test_simulation_matches_theory_quick():
    """
    Quick check: run a reduced-size simulation and compare to the QBD stationary results.

    Note: this is a stochastic check with finite samples. We use a modest relative
    tolerance to avoid flaky failures in CI. For more accurate comparison, run
    `simulate(..., max_events=1000000, warmup_events=100000)` locally.
    """
    # parameters (same as examples in other modules)
    lam1 = 10.0
    lam2 = 5.0
    mu1_rand = 40.0
    mu2_rand = 30.0
    mu1_zip = 30.0
    mu2_zip = 20.0
    K = 5
    m = 3
    p = 0.8

    # build model and compute theoretical stationary measures
    model = DynamicThresholdQBD(lam1, lam2, mu1_rand, mu2_rand, mu1_zip, mu2_zip, K, m, p)
    R = model.solve_R()
    pi0, pi1, _ = compute_stationary(model, R=R)
    E_L_main_th, E_L_merge_th, P_block_th = calculate_performance_measures(model, R, pi0, pi1)

    # run a smaller simulation for CI-speed
    E_L_main_sim, E_L_merge_sim, P_block_sim, sim_time = simulate(
        lam1,
        lam2,
        mu1_rand,
        mu2_rand,
        mu1_zip,
        mu2_zip,
        K,
        m,
        p,
        max_events=200_000,
        warmup_events=20_000,
        seed=12345,
    )

    # relative tolerances are generous due to stochastic noise on smaller runs
    assert E_L_main_sim == pytest.approx(E_L_main_th, rel=0.12)
    assert E_L_merge_sim == pytest.approx(E_L_merge_th, rel=0.12)
    assert P_block_sim == pytest.approx(P_block_th, abs=0.05)
