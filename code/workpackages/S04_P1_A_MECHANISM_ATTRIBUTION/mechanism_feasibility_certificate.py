"""Exact-rational first-order Hawkes-minus-Poisson adaptivity-gap certificate.

Reads the protected WP3/WP4 certificate functions without editing them.
All arithmetic affecting bounds is Fraction or their rational exponential
intervals. Prints JSON; writes no output files.
"""
from __future__ import annotations

import json
import platform
import sys
from fractions import Fraction as Q
from math import factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "workpackages" / "WP3_count"))
sys.path.insert(0, str(ROOT / "workpackages" / "WP4_timestamps"))

from certificate_counts import dec_ceil, dec_floor, exp_minus, poisson_tail_ge
import certificate_timestamps as hawkes

MU = Q(1, 2)
T = Q(7, 5)
LAM = MU * T
N = 12
HAWKES_CELLS = 1024
POISSON_CELLS = 512
POISSON_ORACLE_LIPSCHITZ = Q(12, 5)
FINITE_ETA_REMAINDER = Q(84, 25)  # 2*1.19 + 2*0.49 = 3.36


def poisson_truncated_oracle_polynomial(t: Q) -> Q:
    """Finite x,y<=12 rowwise oracle, before multiplying by exp(-LAM)/4."""
    assert 0 <= t <= T
    m, n = MU * t, MU * (T - t)
    px = [m**x / factorial(x) for x in range(N + 1)]
    py = [n**y / factorial(y) for y in range(N + 1)]
    total = Q(0)
    for x in range(N + 1):
        row_a = row_b = Q(0)
        for y in range(N + 1):
            weight = px[x] * py[y]
            row_a += weight * abs(LAM - x - y)
            row_b += weight * abs(m - n - x + y)
        total += max(row_a, row_b)
    return total


def poisson_oracle_tail_upper() -> Q:
    """Omitted first-order oracle contribution outside [0,12]^2."""
    qn = poisson_tail_ge(N)
    qnext = poisson_tail_ge(N + 1)
    return LAM * (qn + 2 * qnext)


def certify() -> dict[str, str | int]:
    assert hawkes.MU == MU and hawkes.T == T and hawkes.N == N
    # WP4's fixed-B timestamp enclosure is valid at every rational node.
    sampled_h = max(
        hawkes.sampled_fixed_b_upper(T * Q(i, HAWKES_CELLS))
        for i in range(HAWKES_CELLS + 1)
    )
    fixed_h_upper = max(
        hawkes.fixed_a_upper(),
        sampled_h + hawkes.timestamp_tail_upper()
        + 5 * T / (2 * HAWKES_CELLS),
    )
    witness_h_lower = hawkes.witness_lower()
    gap_h_lower = witness_h_lower - fixed_h_upper

    sampled_p_polynomial = max(
        poisson_truncated_oracle_polynomial(T * Q(i, POISSON_CELLS))
        for i in range(POISSON_CELLS + 1)
    )
    exp_lam = exp_minus(LAM)
    sampled_p_upper = sampled_p_polynomial * exp_lam.hi / 4
    poisson_tail = poisson_oracle_tail_upper()
    poisson_timing = POISSON_ORACLE_LIPSCHITZ * T / (2 * POISSON_CELLS)
    oracle_p_upper = sampled_p_upper + poisson_tail + poisson_timing
    fixed_p_lower = LAM * exp_lam.lo / 2  # fixed A, valid for every t
    gap_p_upper = oracle_p_upper - fixed_p_lower

    d_lower = gap_h_lower - gap_p_upper
    assert d_lower > 0, "bounds did not decide the contrast"
    conservative_d = Q(10815375, 10**10)  # 0.0010815375
    assert conservative_d <= d_lower
    interval_end = conservative_d / FINITE_ETA_REMAINDER
    return {
        "status": "CERTIFIED_POSITIVE_FIRST_ORDER_AND_FINITE_ETA",
        "mu": "1/2", "beta": "1", "T": "7/5",
        "count_cutoff_inclusive": N,
        "hawkes_time_cells": HAWKES_CELLS,
        "poisson_time_cells": POISSON_CELLS,
        "hawkes_sampled_fixed_B_upper": dec_ceil(sampled_h, 15),
        "hawkes_fixed_timestamp_tail_upper": dec_ceil(hawkes.timestamp_tail_upper(), 15),
        "hawkes_global_timing_upper": dec_ceil(5 * T / (2 * HAWKES_CELLS), 15),
        "hawkes_fixed_global_upper": dec_ceil(fixed_h_upper, 15),
        "hawkes_feasible_adaptive_lower": dec_floor(witness_h_lower, 15),
        "hawkes_optimal_gap_lower": dec_floor(gap_h_lower, 15),
        "poisson_sampled_oracle_upper": dec_ceil(sampled_p_upper, 15),
        "poisson_oracle_count_tail_upper": dec_ceil(poisson_tail, 15),
        "poisson_oracle_timing_upper": dec_ceil(poisson_timing, 15),
        "poisson_oracle_global_upper": dec_ceil(oracle_p_upper, 15),
        "poisson_fixed_A_lower": dec_floor(fixed_p_lower, 15),
        "poisson_optimal_gap_upper": dec_ceil(gap_p_upper, 15),
        "dependence_first_order_d_lower": dec_floor(d_lower, 15),
        "conservative_d_for_finite_eta": dec_floor(conservative_d, 15),
        "finite_eta_remainder_coefficient": "3.36",
        "finite_eta_sufficient_endpoint_exact": str(interval_end),
        "finite_eta_sufficient_endpoint_decimal_lower": dec_floor(interval_end, 15),
        "python": platform.python_version(),
    }


if __name__ == "__main__":
    print(json.dumps(certify(), indent=2))
