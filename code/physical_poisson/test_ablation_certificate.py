"""Verification tests for the separate diagnostic score-ablation certificate.

Run: python -m unittest -v test_ablation_certificate.py
"""

import hashlib
import unittest
from fractions import Fraction as F
from math import factorial
from pathlib import Path

import ablation_certificate as a


class AblationCertificateTests(unittest.TestCase):
    def test_original_wp3_signed_count_identity_including_endpoints(self):
        # Independent coefficient extraction of the two first-order PGFs:
        # D_A/G0= -m(z-1)-n(w-1), D_B_lin/G0= -m(z-1)+n(w-1).
        # q_k(a)=p_k(a)-p_(k-1)(a), with p_k(a)=e^-a a^k/k!.
        for t in (F(0), a.T0, a.T / 2, a.T):
            m, n = a.MU * t, a.MU * (a.T - t)
            p, r = a.scaled_probabilities(m, 5), a.scaled_probabilities(n, 5)
            q = [p[k] - (p[k - 1] if k else 0) for k in range(6)]
            s = [r[k] - (r[k - 1] if k else 0) for k in range(6)]
            for x in range(6):
                for y in range(6):
                    wp3_a = m * q[x] * r[y] + n * p[x] * s[y]
                    wp3_b_without_cross = m * q[x] * r[y] - n * p[x] * s[y]
                    self.assertEqual((wp3_a, wp3_b_without_cross),
                                     a.scaled_signed_masses(t, x, y))

    def test_reference_normalization_and_zero_signed_mass(self):
        elo, ehi = a.exp_minus_bounds(a.LAMBDA)
        for t in (F(0), a.T0, a.T):
            m, n = a.MU * t, a.MU * (a.T - t)
            p, r = a.scaled_probabilities(m), a.scaled_probabilities(n)
            rectangle = sum(p, F(0)) * sum(r, F(0))
            self.assertLessEqual(elo * rectangle, 1)
            self.assertGreaterEqual(ehi * rectangle
                                    + 2 * a.poisson_tail_upper(a.COUNT_MAX + 1), 1)
            da = db = F(0)
            for x in range(a.COUNT_MAX + 1):
                for y in range(a.COUNT_MAX + 1):
                    va, vb = a.scaled_signed_masses(t, x, y)
                    da += va
                    db += vb
            # The infinite signed measures have zero mass; the finite sums
            # are bounded by the absolute mass omitted from the rectangle.
            self.assertLessEqual(abs(da) * ehi / 4,
                                 a.linear_count_tail_upper())
            self.assertLessEqual(abs(db) * ehi / 4,
                                 a.linear_count_tail_upper())

    def test_endpoints_and_absolute_value_branches(self):
        for t in (F(0), a.T):
            ra, rb = a.truncated_rows(t)
            self.assertEqual(sum(ra, F(0)), sum(rb, F(0)))
        for t in (F(0), a.T0, a.T):
            for x, y in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 4), (5, 2)):
                da, db = a.scaled_signed_masses(t, x, y)
                self.assertEqual(abs(da), max(da, -da))
                self.assertEqual(abs(db), max(db, -db))
        self.assertEqual(a.fixed_a_bounds()[2], 0)

    def test_elementary_action_crossing_is_strict(self):
        m, n = a.MU * a.T0, a.MU * (a.T - a.T0)
        self.assertEqual((m, n), (F(21, 50), F(7, 25)))
        self.assertEqual(a.LAMBDA, F(7, 10))
        # r_A(0)-r_B(0) >= 2(1-0.7)P(Y=1)>0;
        # r_B(1)-r_A(1) >= 2(1-0.86)P(Y=1)>0.
        elo_n, _ = a.exp_minus_bounds(n)
        self.assertGreater(2 * F(3, 10) * elo_n * n, 0)
        self.assertGreater(2 * F(7, 50) * elo_n * n, 0)
        ra, rb = a.truncated_rows(a.T0)
        self.assertGreater(ra[0], rb[0])
        self.assertGreater(rb[1], ra[1])

    def test_exponential_tail_and_global_certificate(self):
        for z in (F(0), F(7, 25), a.LAMBDA):
            lo, hi = a.exp_minus_bounds(z)
            self.assertLessEqual(lo, hi)
            self.assertGreater(lo, 0)
            self.assertEqual(hi - lo, z**23 / factorial(23))
        self.assertEqual(a.TIME_CELLS, 256)
        self.assertEqual(a.COUNT_MAX, 12)
        self.assertEqual(a.T / (2 * a.TIME_CELLS), F(7, 2560))
        result = a.certify()
        self.assertEqual(result["status"], "ABLATION_GAP_CERTIFIED_POSITIVE")
        self.assertEqual(result["sampled_B_maximizing_node_indices"], [17, 239])
        self.assertEqual(result["inherited_policy"], "B iff 1<=X<=12; A otherwise")
        self.assertGreater(F(result["conservative_gap_lower"]), 0)

    def test_protected_input_hashes_if_available(self):
        root = Path(__file__).resolve().parent.parent / "S04_RVPW_P1_ABLATION_INPUT"
        if not root.is_dir():
            self.skipTest("Original input ZIP not present beside the standalone return package")
        lines = (root / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines()
        checked = 0
        for line in lines:
            if not line.strip():
                continue
            expected, rel = line.split("  ", 1)
            path = root / rel.removeprefix("./")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
            checked += 1
        self.assertEqual(checked, 15)


if __name__ == "__main__":
    unittest.main()
