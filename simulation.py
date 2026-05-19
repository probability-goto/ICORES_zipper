"""
Discrete-event simulator for the dynamic-threshold QBD zipper model.

This implements a custom event loop using `heapq` (no external libraries).

Functions
- simulate(...): run event-driven simulation and return (E_L_main, E_L_merge, P_block, sim_time)

The simulation follows the specification provided by the user (mode A/B switching,
service-rate selection, merge-capacity K, warmup by events, time-average via area-under-curve).
"""
from __future__ import annotations

import random
from typing import Tuple, Optional

# Event types
ARRIVAL_MAIN = 1
ARRIVAL_MERGE = 2
SERVICE_COMPLETION = 3


def simulate(
    lam1: float,
    lam2: float,
    mu1_rand: float,
    mu2_rand: float,
    mu1_zip: float,
    mu2_zip: float,
    K: int,
    m: int,
    p: float,
    max_events: int = 1_000_000,
    warmup_events: int = 100_000,
    seed: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Run the discrete-event simulation.

    Returns: (E_L_main, E_L_merge, P_block, sim_time)

    - Time averages are computed after the warmup period using area-under-curve.
    - Blocking ratio is computed from merge arrivals after warmup.
    """

    rng = random.Random(seed)

    # basic state
    N1 = 0  # main
    N2 = 0  # merge
    S = 0  # server state: 0 idle, 1 main, 2 merge

    # bookkeeping
    last_time = 0.0
    processed = 0

    collecting = False
    sim_start_time: Optional[float] = None

    area_N1 = 0.0
    area_N2 = 0.0

    blocked_after_warmup = 0
    merge_arrivals_after_warmup = 0

    # For ModeB we need to know which lane was just served
    last_served = None

    # Gillespie loop (Markov jump process)
    while processed < max_events:
        # 1) compute current rates
        rate_arr_main = lam1
        rate_arr_merge = lam2
        rate_serv_main = (mu1_rand if N2 <= m else mu1_zip) if S == 1 else 0.0
        rate_serv_merge = (mu2_rand if N2 <= m else mu2_zip) if S == 2 else 0.0

        total_rate = rate_arr_main + rate_arr_merge + rate_serv_main + rate_serv_merge

        # if no events are possible, stop
        if total_rate <= 0.0:
            break

        # 2) time to next event
        dt = rng.expovariate(total_rate)

        # accumulate area under curve during dt if collecting
        if collecting:
            area_N1 += N1 * dt
            area_N2 += N2 * dt

        last_time += dt

        # 3) choose which event occurs
        r = rng.random() * total_rate
        threshold = rate_arr_main
        if r < threshold:
            # main arrival
            N1 += 1
            # arrival counts for merge arrivals unaffected
            if S == 0:
                S = 1
                last_served = 1
        else:
            threshold += rate_arr_merge
            if r < threshold:
                # merge arrival
                if N2 >= K:
                    if collecting:
                        blocked_after_warmup += 1
                        merge_arrivals_after_warmup += 1
                else:
                    N2 += 1
                    if collecting:
                        merge_arrivals_after_warmup += 1
                    if S == 0:
                        S = 2
                        last_served = 2
            else:
                # service completion
                threshold += rate_serv_main
                # determine which service completed (only non-zero service rates chosen)
                # prefer main service if r falls into service main range
                if r < threshold:
                    lane = 1
                else:
                    lane = 2

                prev_N2 = N2
                # decrement the served queue
                if lane == 1:
                    if N1 > 0:
                        N1 -= 1
                else:
                    if N2 > 0:
                        N2 -= 1

                # determine next server decision using priority rules
                # (Mode A/B classification uses prev_N2 per specification)
                if N1 == 0 and N2 == 0:
                    S = 0
                elif N1 == 0 and N2 > 0:
                    S = 2
                    last_served = 2
                elif N2 == 0 and N1 > 0:
                    S = 1
                    last_served = 1
                elif 0 < prev_N2 <= m and N1 > 0:
                    # Mode A: random choice using probability p (use prev_N2 for mode decision)
                    if rng.random() < p:
                        S = 1
                        last_served = 1
                    else:
                        S = 2
                        last_served = 2
                elif prev_N2 > m and N1 > 0:
                    # Mode B: must switch to the opposite lane from the one just served
                    if lane == 1:
                        S = 2
                        last_served = 2
                    else:
                        S = 1
                        last_served = 1

        processed += 1

        # start collecting exactly after warmup_events have been processed
        if (not collecting) and (processed >= warmup_events):
            collecting = True
            sim_start_time = last_time
            # reset accumulators and counters so that pre-warmup events are excluded
            area_N1 = 0.0
            area_N2 = 0.0
            blocked_after_warmup = 0
            merge_arrivals_after_warmup = 0

    # finalize
    if sim_start_time is None:
        # no collection period
        return float(N1), float(N2), 0.0, 0.0

    sim_time = last_time - sim_start_time
    E_L_main = area_N1 / sim_time if sim_time > 0 else 0.0
    E_L_merge = area_N2 / sim_time if sim_time > 0 else 0.0
    P_block = float(blocked_after_warmup) / merge_arrivals_after_warmup if merge_arrivals_after_warmup > 0 else 0.0

    return E_L_main, E_L_merge, P_block, sim_time


if __name__ == "__main__":
    # larger `max_events` (e.g. 1_000_000) and `warmup_events` (e.g. 100_000).
    params = dict(
        lam1=10.0,
        lam2=5.0,
        mu1_rand=40.0,
        mu2_rand=30.0,
        mu1_zip=30.0,
        mu2_zip=20.0,
        K=5,
        m=3,
        p=0.8,
        max_events=200_000,
        warmup_events=20_000,
        seed=12345,
    )
    print("Running quick simulation example (reduced events)...")
    res = simulate(**params)
    print("E[L_main], E[L_merge], P_block, sim_time:", res)
