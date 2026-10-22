"""Independent cheap algebra/endpoint checks for the new certificate."""
import unittest
from fractions import Fraction as Q

import mechanism_feasibility_certificate as c


class MechanismFeasibilityTests(unittest.TestCase):
    def test_count_oracle_endpoints(self):
        # At either endpoint no first-stage decision can carry information.
        for t in (Q(0), c.T):
            finite = c.poisson_truncated_oracle_polynomial(t)
            self.assertEqual(finite, sum(
                (c.LAM**k / __import__("math").factorial(k))
                * abs(c.LAM-k) for k in range(c.N+1)
            ))

    def test_poisson_tail_and_lipschitz_budget(self):
        self.assertGreater(c.poisson_oracle_tail_upper(), 0)
        self.assertLess(c.poisson_oracle_tail_upper(), Q(1, 10**9))
        self.assertEqual(
            c.POISSON_ORACLE_LIPSCHITZ*c.T/(2*c.POISSON_CELLS),
            Q(21, 6400),
        )

    def test_physical_endpoint_cross_coefficient(self):
        # c(t)=mu(1-e^-t)(1-e^{-(T-t)}) for beta=1.
        # The interval formula vanishes exactly at either time endpoint.
        for t in (Q(0), c.T):
            term = c.MU*(1-c.exp_minus(t))*(1-c.exp_minus(c.T-t))
            self.assertEqual(term.lo, 0)
            self.assertEqual(term.hi, 0)


if __name__ == "__main__":
    unittest.main()
