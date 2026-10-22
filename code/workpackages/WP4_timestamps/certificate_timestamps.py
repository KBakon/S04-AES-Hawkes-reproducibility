"""Exact-rational global first-order timestamp certificate, protected triple.

Timestamp integration is bounded analytically, not numerically quadratured.
All exponential bounds come from WP3's exact rational Taylor enclosure.
"""
from __future__ import annotations

import json
import platform
import sys
from fractions import Fraction as Q
from pathlib import Path

HERE = Path(__file__).resolve().parent
WP3 = HERE.parent / "WP3_count"
sys.dont_write_bytecode = True  # keep protected WP3 inputs read-only
sys.path.insert(0, str(WP3))
from certificate_counts import I, exp_minus, poisson_tail_ge, dec_floor, dec_ceil

MU=Q(1,2)
T=Q(7,5)
N=12
CELLS=256
ZERO=I(Q(0),Q(0))


def exp_plus(x: Q) -> I:
    e=exp_minus(x)
    return I(1/e.hi,1/e.lo)


def probabilities(m: Q):
    e=exp_minus(m)
    p=[]
    coefficient=Q(1)
    for k in range(N+1):
        if k:
            coefficient *= m/k
        p.append(I(e.lo*coefficient,e.hi*coefficient))
    return p


def abs_score_upper(t: Q, x: int, y: int, et: I, emt: I,
                    eT: I, A: I, B: I) -> Q:
    """Upper E_0(|S_B| | X=x,Y=y) over continuous timestamps."""
    m,n=MU*t,MU*(T-t)
    r=I(Q(x)-A.hi,Q(x)*et.hi-A.lo)
    s=I(Q(y)*eT.lo-B.hi,Q(y)*emt.hi-B.lo)
    v=I(m-x+y-n,m-x+y-n)+2*r*s
    # Conditional timestamps are uniform. Their centered weighted sums have
    # these exact analytically integrated interval expectations.
    eu=I(Q(1),Q(1)) if t==0 else (et-1)*(1/t)
    ev=eT if t==T else (emt-eT)*(1/(T-t))
    er=x*eu-A
    es=y*ev-B
    mean=I(m-x+y-n,m-x+y-n)+2*er*es
    if v.lo>=0:
        return max(Q(0),mean.hi)
    if v.hi<=0:
        return max(Q(0),-mean.lo)
    if mean.lo>=0:
        return mean.hi+2*max(Q(0),-v.lo)
    if mean.hi<=0:
        return -mean.lo+2*max(Q(0),v.hi)
    return max(abs(v.lo),abs(v.hi))


def sampled_fixed_b_upper(t: Q) -> Q:
    m,n=MU*t,MU*(T-t)
    p,q=probabilities(m),probabilities(n)
    et=exp_plus(t)
    emt=exp_minus(t)
    eT=exp_minus(T)
    A=MU*(et-1)
    B=MU*(emt-eT)
    total=Q(0)
    for x in range(N+1):
        if p[x].hi==0:
            continue
        for y in range(N+1):
            if q[y].hi==0:
                continue
            total += p[x].hi*q[y].hi*abs_score_upper(t,x,y,et,emt,eT,A,B)
    return total/4


def fixed_a_upper() -> Q:
    p=probabilities(MU*T)
    finite=sum(p[k].hi*abs(MU*T-k) for k in range(N+1))/4
    # E[(mu*T+X) 1_{X>N}], X~Poisson(mu*T).
    tail=((MU*T)*poisson_tail_ge(N+1)+(MU*T)*poisson_tail_ge(N))/4
    return finite+tail


def timestamp_tail_upper() -> Q:
    # Union of X>N and Y>N. Under the reference law, both means <= .7.
    # |linear score| <= .7+X+Y, |R|<=5X+2, |S|<=Y+.5.
    # Applying E[X 1_{X>N}] <= .7 Q_N to each independent Poisson tail
    # gives <= [25*.7 Q_N+13.1 Q_(N+1)]/4 <= 8 Q_N.
    return 8*poisson_tail_ge(N)


def witness_lower() -> Q:
    # Feasible timestamp rule ignores timestamps, using WP3's exact-rational
    # count witness, and its classification may likewise ignore timestamps.
    from certificate_counts import metrics
    rows_a,rows_b=metrics(Q(21,25))[1:]
    return sum((max(a.lo,b.lo) for a,b in zip(rows_a,rows_b)),Q(0))/4


def main():
    assert exp_minus(T).lo > Q(1,5)  # proves e^T<5 for Lipschitz/tail bounds
    sampled=max(sampled_fixed_b_upper(T*Q(i,CELLS))
                for i in range(CELLS+1))
    tail=timestamp_tail_upper()
    radius=T/Q(2*CELLS)
    # K_B is 5-Lipschitz: couple the independent eta=0 A/B Poisson paths.
    fixed_b=sampled+tail+5*radius
    fixed_a=fixed_a_upper()
    fixed=max(fixed_a,fixed_b)
    witness=witness_lower()
    result={
        "status":"CERTIFIED_POSITIVE" if witness>fixed else "NOT_SEPARATED",
        "parameters":{"mu":"1/2","beta":"1","T":"7/5","eta":"0+"},
        "count_truncation_inclusive":N,"time_cells":CELLS,
        "sampled_fixed_B_upper":dec_ceil(sampled),
        "timestamp_point_count_tail_upper":dec_ceil(tail,12),
        "timestamp_integration_error_upper":"0 (analytic enclosure)",
        "global_timing_error_upper":dec_ceil(5*radius),
        "global_fixed_A_upper":dec_ceil(fixed_a),
        "global_fixed_B_upper":dec_ceil(fixed_b),
        "global_fixed_upper":dec_ceil(fixed),
        "feasible_timestamp_contingent_lower":dec_floor(witness),
        "gap_lower":dec_floor(witness-fixed),
        "witness_t":"21/25",
        "witness_policy":"A if X=0 or X>=13; B if 1<=X<=12",
        "python":platform.python_version(),"random_seeds":[]
    }
    (HERE/"raw_results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
