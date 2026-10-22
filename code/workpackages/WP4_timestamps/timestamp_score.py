"""First-order timestamp likelihood scores for stationary triangular Hawkes.

Under eta=0 the observed streams are independent homogeneous Poisson
processes of rate mu.  ``score`` is the Radon--Nikodym derivative of the
signed measure (P_AB - P_BA)'_0 relative to that reference law.
"""
from __future__ import annotations

from math import exp, expm1

MU = .5
BETA = 1.
T = 1.4


def score(t: float, first: tuple[float, ...], second: tuple[float, ...],
          channel: str, mu: float = MU, beta: float = BETA,
          horizon: float = T) -> float:
    if not (0 <= t <= horizon and channel in ("A", "B")):
        raise ValueError("invalid split or channel")
    if any(not 0 <= u < t for u in first) or any(not t <= v < horizon for v in second):
        raise ValueError("timestamp outside active window")
    x, y = len(first), len(second)
    m, n = mu*t, mu*(horizon-t)
    if channel == "A":
        return m+n-x-y
    # The exponential kernel makes the centered cross integral rank one.
    source = sum(exp(beta*u) for u in first) - mu*expm1(beta*t)/beta
    target = sum(exp(-beta*v) for v in second) - \
        mu*(exp(-beta*t)-exp(-beta*horizon))/beta
    return (m-x)+(y-n)+(beta/mu)*source*target


def count_conditional_score(t: float, x: int, y: int,
                            channel: str, mu: float = MU,
                            beta: float = BETA,
                            horizon: float = T) -> float:
    """E_0[score | X=x,Y=y], with endpoint limits built in."""
    m,n=mu*t,mu*(horizon-t)
    if channel == "A":
        return m+n-x-y
    if t == 0 or t == horizon:
        return (m-x)+(y-n)
    avg_source=expm1(beta*t)/(beta*t)
    avg_target=(exp(-beta*t)-exp(-beta*horizon))/(beta*(horizon-t))
    return (m-x)+(y-n)+(beta/mu)*(x-m)*(y-n)*avg_source*avg_target


def count_cross_mass(t: float, mu: float = MU,
                     beta: float = BETA, horizon: float = T) -> float:
    return mu*(-expm1(-beta*t))*(-expm1(-beta*(horizon-t)))/beta
