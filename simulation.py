"""
Discrete-event simulator for the dynamic-threshold QBD zipper model.

This implements a custom event loop using `heapq` (no external libraries).

Functions
- simulate(...): run event-driven simulation and return (E_L_main, E_L_merge, P_block, sim_time)

The simulation follows the specification provided by the user (mode A/B switching,
service-rate selection, merge-capacity K, warmup by events, time-average via area-under-curve).
"""
from __future__ import annotations

import heapq
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

    # event calendar: tuples (time, seq, event_type, event_data)
    ev_heap = []
    seq = 0

    def push_event(time: float, ev_type: int, ev_data=None) -> None:
        nonlocal seq
        heapq.heappush(ev_heap, (time, seq, ev_type, ev_data))
        seq += 1

    # schedule initial arrivals
    push_event(rng.expovariate(lam1), ARRIVAL_MAIN)
    push_event(rng.expovariate(lam2), ARRIVAL_MERGE)

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

    # main loop
    while processed < max_events:
        time, _s, ev_type, ev_data = heapq.heappop(ev_heap)

        # advance time and accumulate area if collecting
        delta = time - last_time
        if collecting:
            area_N1 += N1 * delta
            area_N2 += N2 * delta
        last_time = time

        if ev_type == ARRIVAL_MAIN:
            # schedule next main arrival
            push_event(time + rng.expovariate(lam1), ARRIVAL_MAIN)

            # main arrival always accepted (no capacity)
            N1 += 1

            # if server idle, start service on main immediately
            if S == 0:
                S = 1
                last_served = 1
                # choose service rate depending on N2
                rate = mu1_rand if N2 <= m else mu1_zip
                push_event(time + rng.expovariate(rate), SERVICE_COMPLETION, 1)

        elif ev_type == ARRIVAL_MERGE:
            # schedule next merge arrival
            push_event(time + rng.expovariate(lam2), ARRIVAL_MERGE)

            # blocked if N2 == K
            if N2 >= K:
                if collecting:
                    blocked_after_warmup += 1
                    merge_arrivals_after_warmup += 1
            else:
                N2 += 1
                if collecting:
                    merge_arrivals_after_warmup += 1

                # if server idle, start service on merge immediately
                if S == 0:
                    S = 2
                    last_served = 2
                    rate = mu2_rand if N2 <= m else mu2_zip
                    push_event(time + rng.expovariate(rate), SERVICE_COMPLETION, 2)

        elif ev_type == SERVICE_COMPLETION:
            lane = ev_data

            # decrement the served queue
            if lane == 1:
                if N1 > 0:
                    N1 -= 1
            else:
                if N2 > 0:
                    N2 -= 1

            # determine next server decision using the specified priority rules
            if N1 == 0 and N2 == 0:
                S = 0
            elif N1 == 0 and N2 > 0:
                S = 2
                last_served = 2
                rate = mu2_rand if N2 <= m else mu2_zip
                push_event(time + rng.expovariate(rate), SERVICE_COMPLETION, 2)
            elif N2 == 0 and N1 > 0:
                S = 1
                last_served = 1
                rate = mu1_rand if N2 <= m else mu1_zip
                push_event(time + rng.expovariate(rate), SERVICE_COMPLETION, 1)
            elif 0 < N2 <= m and N1 > 0:
                # Mode A: random choice with prob p to serve main
                if rng.random() < p:
                    S = 1
                    last_served = 1
                    rate = mu1_rand if N2 <= m else mu1_zip
                    push_event(time + rng.expovariate(rate), SERVICE_COMPLETION, 1)
                else:
                    S = 2
                    last_served = 2
                    rate = mu2_rand if N2 <= m else mu2_zip
                    push_event(time + rng.expovariate(rate), SERVICE_COMPLETION, 2)
            elif N2 > m and N1 > 0:
                # Mode B: must switch to the opposite lane from the one just served
                if lane == 1:
                    S = 2
                    last_served = 2
                    rate = mu2_rand if N2 <= m else mu2_zip
                    push_event(time + rng.expovariate(rate), SERVICE_COMPLETION, 2)
                else:
                    S = 1
                    last_served = 1
                    rate = mu1_rand if N2 <= m else mu1_zip
                    push_event(time + rng.expovariate(rate), SERVICE_COMPLETION, 1)

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
    # Quick example run (smaller scale). For full accuracy run simulate(...) with
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
