"""Stationary triangular Hawkes first-order count likelihoods.

The source is a stationary Poisson process on the entire real line.  Conditional
on it, target immigrants have rate mu and its offspring have intensity
eta * sum_{source u<v} beta exp(-beta(v-u)).
"""
from __future__ import annotations

from math import exp, factorial


def poisson(m: float, k: int) -> float:
    return 0.0 if k < 0 else exp(-m) * m**k / factorial(k)


def first_order(t: float, x: int, y: int, mu: float = .5,
                beta: float = 1., T: float = 1.4) -> tuple[float, float]:
    """Return coefficients of P_AB-P_BA for second channel A and B."""
    if not (0 <= t <= T and x >= 0 and y >= 0):
        raise ValueError("invalid split or count")
    m, n = mu*t, mu*(T-t)
    c = mu*(1-exp(-beta*t))*(1-exp(-beta*(T-t)))/beta
    px, py = poisson(m, x), poisson(n, y)
    qx, qy = px-poisson(m, x-1), py-poisson(n, y-1)
    return m*qx*py+n*px*qy, m*qx*py-n*px*qy+c*qx*qy


def pgf_first_order(z: float, w: float, t: float, second: str,
                    mu: float = .5, beta: float = 1., T: float = 1.4) -> float:
    m, n = mu*t, mu*(T-t)
    c = mu*(1-exp(-beta*t))*(1-exp(-beta*(T-t)))/beta
    base = exp(m*(z-1)+n*(w-1))
    if second == "A":
        return -base*(m*(z-1)+n*(w-1))
    if second == "B":
        return base*(-m*(z-1)+n*(w-1)+c*(z-1)*(w-1))
    raise ValueError(second)


def coefficients(t: float, N: int = 16) -> tuple[float, float, float]:
    rows_a, rows_b = [], []
    for x in range(N+1):
        aa = bb = 0.
        for y in range(N+1):
            a, b = first_order(t, x, y)
            aa += abs(a)
            bb += abs(b)
        rows_a.append(aa)
        rows_b.append(bb)
    return sum(rows_a)/4, sum(rows_b)/4, sum(map(max, rows_a, rows_b))/4
