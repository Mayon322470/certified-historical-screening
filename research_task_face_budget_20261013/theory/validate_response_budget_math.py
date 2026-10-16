"""Exact/independent checks of the shared one-tau response interface.

One old certified nominal point is read-only reused; no parameter grid,
benchmark suite, simulator, calibration or holdout data is generated/read.
"""
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import response_budget_math as rb
import shared_response as sr


def case():
    return dict(past=((F(1, 5), F(1)),), future=((F(3, 5), F(1)),), H=F(3, 5), rho=F(0),
                packet=((F(0), F(1)),)*3,
                response=dict(kind=sr.KIND, epsilon_x=F(1, 100), L=F(4, 5)))


class ResponseBudgetTests(unittest.TestCase):
    def test_normalized_profile_exact_derivatives_and_global_origin(self):
        c = case()
        length, t = F(4, 5), F(1, 5)
        # Independent coefficient differentiation of x(t)=t^3/L^3.
        coefficients = (F(0), F(0), F(0), 1/length**3)
        derivative = tuple(i*coefficients[i] for i in range(1, len(coefficients)))
        second = tuple(i*derivative[i] for i in range(1, len(derivative)))
        evaluate = lambda polynomial: sum(value*t**i for i, value in enumerate(polynomial))
        self.assertEqual(sr.profile(c, t), (evaluate(coefficients), evaluate(derivative), evaluate(second)))
        self.assertEqual(sr.profile(c, 0), (F(0), F(0), F(0)))
        self.assertEqual(sr.profile(c, length)[0], F(1))
        self.assertNotEqual(sr.profile(c, F(1, 5)+F(1, 10)), sr.profile(c, F(1, 10)))

    def test_expansion_can_pass_every_row_without_shared_beta(self):
        c = case()
        psi = sr.profile(c, F(1, 5))
        b = F(1, 100)
        packet = ((psi[0]*b*F(4, 5), psi[0]*b*F(9, 10)),
                  (-psi[1]*b*F(9, 10), -psi[1]*b*F(4, 5)), (-psi[2]*b, psi[2]*b))
        c["packet"] = packet
        self.assertTrue(all(lo <= 0 <= hi for lo, hi in sr.expanded_packet(c)))
        self.assertIsNone(sr.beta_outer(((0, 0),)*3, packet, psi, b))
        self.assertIsNone(sr.beta_inner(((0, 0),)*3, packet, psi, b))

    def test_outer_vs_inner_and_zero_profile(self):
        self.assertEqual(sr.beta_outer(((0, 1),), ((0, F(1, 10)),), (1,), 1), (F(-1), F(1, 10)))
        self.assertIsNone(sr.beta_inner(((0, 1),), ((0, F(1, 10)),), (1,), 1))
        # Empty inner is not true inconsistency: p=.5,beta=-.5 is compatible.
        self.assertEqual(F(1, 2)+F(-1, 2), 0)
        self.assertEqual(sr.beta_outer(((0, 1),), ((F(1, 2), F(1, 2)),), (0,), 1), (F(-1), F(1)))
        self.assertIsNone(sr.beta_inner(((0, 1),), ((F(1, 2), F(1, 2)),), (0,), 1))
        self.assertIsNone(sr.beta_outer(((0, 1),), ((2, 2),), (0,), 1))

    def test_outer_beta_is_not_a_common_tau_witness(self):
        # Exact nominal functions p1=tau,p2=2*tau,tau∈[0,1]. The packets
        # force tau=1.5, but independent nominal cell ranges conceal this.
        beta = sr.beta_outer(((0, 1), (0, 2)), ((0, 0), (F(3, 2), F(3, 2))), (1, 1), 1)
        self.assertEqual(beta, (F(-1, 2), F(0)))
        self.assertGreater(F(3, 2), F(1))

    def test_known_nominal_exponential_point_and_independent_true_witness(self):
        old = Path(__file__).resolve().parents[2]/"research_zhao2026_comparison_20261012/implementation/certified_core.py"
        spec = importlib.util.spec_from_file_location("_response_old_nominal_readonly", old)
        cc = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cc
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(cc)
        finally:
            sys.dont_write_bytecode = previous
        c = case()
        point = cc.endpoint(F(1), (F(0),)*3, c["past"])
        nominal = tuple((value.lower(), value.upper()) for value in point)
        psi = sr.profile(c, F(1, 5))
        generating_beta = F(1, 200)
        width = F(1, 10**18)
        c["packet"] = tuple((value.midpoint()+generating_beta*p-width,
                              value.midpoint()+generating_beta*p+width) for value, p in zip(point, psi))
        inner = sr.beta_inner(nominal, c["packet"], psi, F(1, 100))
        self.assertIsNotNone(inner)
        independent_beta = (inner[0]+inner[1])/2
        for (nlo, nhi), (plo, phi), p in zip(nominal, c["packet"], psi):
            self.assertLessEqual(plo, nlo+independent_beta*p)
            self.assertLessEqual(nhi+independent_beta*p, phi)
        # beta=0 is NOT an original-history witness, despite packet expansion.
        self.assertTrue(any(nhi < plo or nlo > phi for (nlo, nhi), (plo, phi) in zip(nominal, c["packet"])))
        future = cc.endpoint(F(1), point, c["future"])[0]
        limit = future.upper()+generating_beta/2
        self.assertLess(future.upper(), limit)
        actual_lower = sr.future_lower(future.lower(), independent_beta, sr.profile(c, F(4, 5))[0])
        self.assertGreater(actual_lower, limit)

    def test_fm_rejection_pair_matches_shared_beta_contradiction(self):
        response = rb.CubicResponseClass(1, rb.Interval(-1, 1))
        lo = rb.HistoryRow("x", "x", 1, rb.Interval(0, 0), rb.Interval(F(4, 5), F(9, 10)))
        hi = rb.HistoryRow("v", "v", 1, rb.Interval(0, 0), rb.Interval(F(-27, 10), F(-12, 5)))
        fm = rb.fourier_motzkin_pair(response, lo, hi)
        self.assertEqual(fm, rb.Interval(F(24, 5), F(24, 5)))
        self.assertGreater(fm.lower, 0)
        self.assertIsNone(rb.shared_beta_interval(response, (lo, hi)))

    def test_future_outer_upper_exceedance_is_not_unsafe(self):
        outer = sr.beta_outer(((F(-1, 50), 0),), ((F(-1, 100), F(-1, 100)),), (1,), F(1, 100))
        inner = sr.beta_inner(((0, 0),), ((F(-1, 100), F(-1, 100)),), (1,), F(1, 100))
        self.assertEqual(inner, (F(-1, 100), F(-1, 100)))
        self.assertGreater(sr.future_upper(0, outer, 1), 0)
        self.assertLess(sr.future_lower(0, inner[0], 1), 0)

    def test_negative_response_requires_actual_curvature_allowance(self):
        c = dict(past=((F(1, 2), 0),), future=((F(1, 2), 0),), H=F(1, 2), rho=0,
                 packet=((0, 0), (1, 1), (-3, -3)),
                 response=dict(kind=sr.KIND, epsilon_x=1, L=1))
        # Nominal x=7t/4-3/4 is affine; beta=-1 adds -t^3.
        actual = lambda t: F(7, 4)*t-F(3, 4)-t**3
        self.assertEqual(actual(F(1, 2)), 0)
        self.assertEqual(actual(1), 0)
        self.assertEqual(actual(F(3, 4)), F(9, 64))
        allowance = sr.continuous_allowance(c, F(1, 2), expanded_nominal_a=(0, 0))
        self.assertEqual(allowance["nominal_acceleration_bound"], 0)
        self.assertEqual(allowance["response_curvature_bound"], 6)
        self.assertEqual(allowance["allowance"], F(3, 16))
        self.assertLess(actual(F(3, 4)), allowance["allowance"])

    def test_case_schema_and_non_rational_inputs_fail(self):
        c = case()
        c["response"]["L"] = 1
        with self.assertRaises(ValueError):
            sr.profile(c, 0)
        c = case()
        c["rho"] = 1
        with self.assertRaises(ValueError):
            sr.expanded_packet(c)
        with self.assertRaises(TypeError):
            sr.future_upper(.1, (0, 1), 1)
        with self.assertRaises(ValueError):
            sr.beta_inner(((0, 0),), ((0, 0),), (), 1)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ResponseBudgetTests))
    report = dict(sr.contract_metadata(), tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  passed=result.wasSuccessful(), validation_inputs="Selected synthetic analytic cases and one read-only certified nominal tau point",
                  new_tau_grid=False, benchmark_cases_generated=False, calibration_or_holdout_read=False,
                  checks=["profile derivatives/global clock", "expanded packets vs shared beta",
                          "outer/inner and zero-profile semantics", "outer interval lacks common tau witness",
                          "outward exponential nominal point and original-history/future witness",
                          "FM shared-beta strict pair", "upper exceedance is not unsafe",
                          "negative cubic actual-curvature bridge", "schema and float rejection"])
    Path(__file__).with_name("response_math_validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
