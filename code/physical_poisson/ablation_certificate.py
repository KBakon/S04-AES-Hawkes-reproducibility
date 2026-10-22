"""Exact-rational first-order certificate for the diagnostic linear-score ablation.

This is a signed-score surrogate, not a finite-excitation Hawkes model.
Only the original cross-interaction term is set to zero. No original WP3/WP4
files are imported or changed. Run: python ablation_certificate.py
"""

from __future__ import annotations

import json
import platform
from fractions import Fraction as F
from math import factorial
from pathlib import Path

MU = F(1, 2)
T = F(7, 5)
LAMBDA = MU * T
T0 = F(21, 25)
COUNT_MAX = 12
TIME_CELLS = 256


def exp_minus_bounds(z: F) -> tuple[F, F]:
    """Alternating Taylor S_23 <= exp(-z) <= S_22, 0 <= z <= 7/10."""
    if not F(0) <= z <= LAMBDA:
        raise ValueError("exponential argument outside certified interval")
    term = F(1)
    total = term
    upper = None
    for k in range(1, 24):
        term *= -z / k
        total += term
        if k == 22:
            upper = total
    assert upper is not None and 0 < total <= upper
    return total, upper


def poisson_tail_upper(k: int) -> F:
    """Uniform P(Pois(a)>=k), a<=0.7, from a geometric ratio bound."""
    if k < 1:
        raise ValueError("k must be positive")
    return LAMBDA**k / factorial(k) / (1 - LAMBDA / (k + 1))


def scaled_probabilities(a: F, count_max: int = COUNT_MAX) -> list[F]:
    """e^a times Poisson probabilities; every entry is rational."""
    if not F(0) <= a <= LAMBDA:
        raise ValueError("Poisson mean outside certified interval")
    return [a**k / factorial(k) for k in range(count_max + 1)]


def scaled_signed_masses(t: F, x: int, y: int) -> tuple[F, F]:
    """e^lambda times (D_A, D_B_linear) at the count pair (x,y)."""
    if not F(0) <= t <= T or x < 0 or y < 0:
        raise ValueError("invalid time or count")
    m, n = MU * t, MU * (T - t)
    w = m**x / factorial(x) * n**y / factorial(y)
    return w * (m + n - x - y), w * (m - n - x + y)


def truncated_rows(t: F, count_max: int = COUNT_MAX) -> tuple[list[F], list[F]]:
    """Unsigned row masses before multiplication by exp(-lambda)/4."""
    if not F(0) <= t <= T:
        raise ValueError("time outside [0,T]")
    m, n = MU * t, MU * (T - t)
    px, py = scaled_probabilities(m, count_max), scaled_probabilities(n, count_max)
    rows_a, rows_b = [], []
    for x, wx in enumerate(px):
        a = b = F(0)
        for y, wy in enumerate(py):
            w = wx * wy
            a += w * abs(LAMBDA - x - y)
            b += w * abs(m - n - x + y)
        rows_a.append(a)
        rows_b.append(b)
    return rows_a, rows_b


def fixed_a_bounds() -> tuple[F, F, F]:
    """K_A=lambda*exp(-lambda)/2 exactly, as 0<lambda<1."""
    elo, ehi = exp_minus_bounds(LAMBDA)
    return LAMBDA * elo / 2, LAMBDA * ehi / 2, F(0)


def linear_count_tail_upper() -> F:
    """Uniform omitted (x,y) rectangle contribution to K_B_linear.

    |S_B_linear| <= lambda+X+Y. Union-bound X,Y>=N+1. The constant
    coefficients sum to 3*lambda, and the first moments sum to lambda*Q_N.
    """
    return LAMBDA * (poisson_tail_upper(COUNT_MAX)
                     + 3 * poisson_tail_upper(COUNT_MAX + 1)) / 4


def decimal_floor(q: F, digits: int = 15) -> str:
    scale = 10**digits
    v = q.numerator * scale // q.denominator
    return f"{v // scale}.{v % scale:0{digits}d}"


def decimal_ceil(q: F, digits: int = 15) -> str:
    scale = 10**digits
    v = -(-q.numerator * scale // q.denominator)
    return f"{v // scale}.{v % scale:0{digits}d}"


def certify() -> dict:
    elo, ehi = exp_minus_bounds(LAMBDA)
    a_lower, a_upper, a_tail = fixed_a_bounds()
    rows_a, rows_b = truncated_rows(T0)
    inherited = ["B" if 1 <= x <= 12 else "A" for x in range(COUNT_MAX + 1)]
    inherited_sum = sum((rows_b[x] if inherited[x] == "B" else rows_a[x]
                         for x in range(COUNT_MAX + 1)), F(0))
    inherited_lower = elo * inherited_sum / 4

    # A selected row beyond x=12 contributes nonnegative absolute mass.
    # Therefore this is a lower bound for the feasible unbounded-count rule.
    max_sampled_b_sum = F(0)
    maximizing_nodes = []
    for i in range(TIME_CELLS + 1):
        t = T * i / TIME_CELLS
        _, rb = truncated_rows(t)
        node_sum = sum(rb, F(0))
        if node_sum > max_sampled_b_sum:
            max_sampled_b_sum, maximizing_nodes = node_sum, [i]
        elif node_sum == max_sampled_b_sum:
            maximizing_nodes.append(i)
    sampled_b_upper = ehi * max_sampled_b_sum / 4
    b_tail = linear_count_tail_upper()
    radius = T / (2 * TIME_CELLS)
    timing_error = MU * radius
    global_b_upper = sampled_b_upper + b_tail + timing_error
    global_fixed_upper = max(a_upper, global_b_upper)
    gap_lower = inherited_lower - global_fixed_upper
    # Make the displayed gap follow from the separately outward-displayed
    # adaptive lower and fixed upper, even at the final decimal place.
    displayed_gap_lower = (F(decimal_floor(inherited_lower))
                           - F(decimal_ceil(global_fixed_upper)))
    # Exact comparisons for action crossing: these are strict because Y=1
    # has positive Poisson mass when n=7/25.
    m0, n0 = MU * T0, MU * (T - T0)
    assert m0 == F(21, 50) and n0 == F(7, 25)
    assert rows_a[0] > rows_b[0] and rows_b[1] > rows_a[1]
    assert gap_lower >= displayed_gap_lower > 0, "Certificate did not establish a positive gap"

    return {
        "status": "ABLATION_GAP_CERTIFIED_POSITIVE",
        "scope": "diagnostic first-order signed-score surrogate only; no finite-eta Hawkes claim",
        "arithmetic": "Python Fraction; all inequalities exact; decimal lower/upper displays directed",
        "parameters": {"mu": "1/2", "beta": "1", "T": "7/5", "t0": "21/25"},
        "count_cutoff_inclusive": COUNT_MAX,
        "time_cells": TIME_CELLS,
        "time_nodes": TIME_CELLS + 1,
        "sampled_B_maximizing_node_indices": maximizing_nodes,
        "exp_minus_lambda_lower": decimal_floor(elo, 20),
        "exp_minus_lambda_upper": decimal_ceil(ehi, 20),
        "exp_interval_width_upper": decimal_ceil(ehi - elo, 22),
        "tail_probability_Q12_upper": decimal_ceil(poisson_tail_upper(12), 20),
        "tail_probability_Q13_upper": decimal_ceil(poisson_tail_upper(13), 20),
        "fixed_A_tail_upper": decimal_ceil(a_tail, 20),
        "fixed_B_linear_tail_upper": decimal_ceil(b_tail, 20),
        "nearest_node_radius": str(radius),
        "fixed_B_linear_lipschitz": str(MU),
        "timing_error_upper": decimal_ceil(timing_error),
        "fixed_A_lower": decimal_floor(a_lower),
        "fixed_A_upper": decimal_ceil(a_upper),
        "sampled_fixed_B_linear_upper": decimal_ceil(sampled_b_upper),
        "global_fixed_B_linear_upper": decimal_ceil(global_b_upper),
        "global_fixed_upper": decimal_ceil(global_fixed_upper),
        "inherited_policy": "B iff 1<=X<=12; A otherwise",
        "inherited_adaptive_lower": decimal_floor(inherited_lower),
        "conservative_gap_lower": decimal_floor(displayed_gap_lower),
        "crossing_at_x0_and_x1": True,
        "python": platform.python_version(),
        "random_seeds": [],
    }


def main() -> None:
    result = certify()
    path = Path(__file__).with_name("raw_results.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
