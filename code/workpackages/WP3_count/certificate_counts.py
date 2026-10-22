"""Exact-rational count-tail and global-time certificate for WP3.

No binary floating point or Decimal.exp participates in any bound.  Fraction
arithmetic and alternating Taylor enclosures supply every exponential.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from fractions import Fraction as Q
from math import factorial
from pathlib import Path

T = Q(7, 5)
MU = Q(1, 2)
BETA = Q(1)
N = 12
CELLS = 256


@dataclass(frozen=True)
class I:
    lo: Q
    hi: Q

    def __add__(self, other):
        other = to_i(other)
        return I(self.lo+other.lo, self.hi+other.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -to_i(other)

    def __rsub__(self, other):
        return to_i(other) + -self

    def __mul__(self, other):
        other = to_i(other)
        vals = (self.lo*other.lo, self.lo*other.hi,
                self.hi*other.lo, self.hi*other.hi)
        return I(min(vals), max(vals))

    __rmul__ = __mul__

    def abs(self):
        return I(0 if self.lo <= 0 <= self.hi else min(abs(self.lo), abs(self.hi)),
                 max(abs(self.lo), abs(self.hi)))


def to_i(x):
    return x if isinstance(x, I) else I(Q(x), Q(x))


def exp_minus(x: Q) -> I:
    assert 0 <= x <= T
    # S_23 <= exp(-x) <= S_22 for x>=0; terms decrease here.
    term = Q(1)
    s = term
    upper = None
    for k in range(1, 24):
        term *= -x/k
        s += term
        if k == 22:
            upper = s
    return I(s, upper)


def probabilities(m: Q):
    e = exp_minus(m)
    return [I(e.lo*m**k/factorial(k), e.hi*m**k/factorial(k))
            for k in range(N+1)]


def arrays(t: Q):
    m, n = MU*t, MU*(T-t)
    p, r = probabilities(m), probabilities(n)
    q = [p[k]-(p[k-1] if k else 0) for k in range(N+1)]
    s = [r[k]-(r[k-1] if k else 0) for k in range(N+1)]
    c = MU*(1-exp_minus(t))*(1-exp_minus(T-t))
    ra, rb = [], []
    for x in range(N+1):
        a = b = I(Q(0), Q(0))
        for y in range(N+1):
            common = m*q[x]*r[y]
            other = n*p[x]*s[y]
            a += (common+other).abs()
            b += (common-other+c*q[x]*s[y]).abs()
        ra.append(a)
        rb.append(b)
    return ra, rb


def metrics(t: Q):
    a, b = arrays(t)
    ka = sum(a, I(Q(0), Q(0)))
    kb = sum(b, I(Q(0), Q(0)))
    co = sum((I(max(u.lo,v.lo), max(u.hi,v.hi)) for u,v in zip(a,b)),
             I(Q(0), Q(0)))
    return tuple(I(v.lo/4, v.hi/4) for v in (ka,kb,co)), a, b


def poisson_tail_ge(k: int) -> Q:
    """Uniform Pr(Pois(m)>=k), 0<=m<=0.7, by geometric domination."""
    lam = MU*T
    return lam**k/factorial(k) / (1-lam/Q(k+1))


def tail_bounds():
    # Outside [0,N]^2: ||p_out|| <= Q_(N+1),
    # ||q_out|| <= Q_N + Q_(N+1). Product union bound.
    p = poisson_tail_ge(N+1)
    q = poisson_tail_ge(N)+p
    a = (MU*T)*(q+2*p)/4
    b = (MU*T)*(q+2*p)/4 + (MU/BETA)*q
    return a,b


def dec_floor(x: Q, digits=12) -> str:
    scale = 10**digits
    n = x.numerator*scale//x.denominator
    return f"{n//scale}.{n%scale:0{digits}d}"


def dec_ceil(x: Q, digits=12) -> str:
    scale = 10**digits
    n = -(-x.numerator*scale//x.denominator)
    return f"{n//scale}.{n%scale:0{digits}d}"


def main():
    ta,tb = tail_bounds()
    witness_t = Q(21,25)  # 0.84, prescribed before certificate execution
    witness, rows_a, rows_b = metrics(witness_t)
    # A feasible policy chooses the row with larger certified lower bound.
    policy = ["A" if a.lo >= b.lo else "B" for a,b in zip(rows_a,rows_b)]
    witness_lower = sum((a.lo if pick == "A" else b.lo
                         for a,b,pick in zip(rows_a,rows_b,policy)), Q(0))/4
    max_a = max_b = Q(0)
    for i in range(CELLS+1):
        t = T*Q(i,CELLS)
        (a,b,_),_,_ = metrics(t)
        max_a = max(max_a,a.hi)
        max_b = max(max_b,b.hi)
    # Global Lipschitz constants: ||D_A'||_1 <= 4 mu+4 mu T=4.8,
    # ||D_B'||_1 <= 4.8+4|c'|+8c <= 10.8.
    lips_a = Q(6,5)
    lips_b = Q(27,10)
    radius = T/Q(2*CELLS)
    upper_a = max_a + ta + lips_a*radius
    upper_b = max_b + tb + lips_b*radius
    fixed_upper = max(upper_a,upper_b)
    gap_lower = witness_lower-fixed_upper
    result = {
        "status":"CERTIFIED_POSITIVE" if gap_lower>0 else "NOT_SEPARATED",
        "parameters":{"mu":"1/2","beta":"1","T":"7/5","eta":"0+"},
        "count_truncation_inclusive":N,"time_cells":CELLS,
        "time_cell_width":str(T/CELLS),"nearest_grid_radius":str(radius),
        "tail_error_K_A_upper":dec_ceil(ta,15),
        "tail_error_K_B_upper":dec_ceil(tb,15),
        "time_error_K_A_upper":dec_ceil(lips_a*radius,12),
        "time_error_K_B_upper":dec_ceil(lips_b*radius,12),
        "sampled_K_A_upper":dec_ceil(max_a),
        "sampled_K_B_upper":dec_ceil(max_b),
        "global_fixed_A_upper":dec_ceil(upper_a),
        "global_fixed_B_upper":dec_ceil(upper_b),
        "global_fixed_upper":dec_ceil(fixed_upper),
        "witness_t":str(witness_t),
        "witness_policy_x_0_to_N":policy,
        "witness_lower":dec_floor(witness_lower),
        "witness_oracle_truncated_lower":dec_floor(witness[2].lo),
        "gap_lower":dec_floor(gap_lower),
        "python":platform.python_version(),"random_seeds":[]}
    Path(__file__).with_name("raw_results.json").write_text(
        json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
