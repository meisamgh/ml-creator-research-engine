import unittest
from datetime import datetime, timedelta

from ml_creator.core import (DataState, Fact, FitGuard, PredictionEvent, Problem,
                             ResearchFeature, RunIntent, RunMode, calculate,
                             compile_feature, generate_variants, route)
from ml_creator.evaluation import evaluate_sealed_holdout, freeze, time_ordered_oof


T = datetime(2025, 1, 1)
FEATURE = ResearchFeature("txn_activity", "Transaction activity", "behavior",
                          "Recent activity may reflect financial stress",
                          ("customer identifier", "transaction timestamp", "amount"),
                          "transaction count in trailing window", ("example:manual",))


class CoreTests(unittest.TestCase):
    def event(self, entity="a", day=0, label=0):
        ts = T + timedelta(days=day)
        return PredictionEvent(entity, ts, label, ts + timedelta(days=1), 7)

    def test_bitemporal_boundaries(self):
        event = self.event()
        spec = compile_feature(FEATURE)
        facts = [Fact("a", T - timedelta(days=1), T - timedelta(days=1), 2),
                 Fact("a", T, T, 3),
                 Fact("a", T + timedelta(days=1), T, 100),
                 Fact("a", T - timedelta(days=1), T + timedelta(days=1), 100)]
        self.assertEqual(calculate(spec, event, facts), 2)

    def test_fit_guard_and_veto(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            FitGuard([0, 1]).check([0, 2])
        with self.assertRaisesRegex(ValueError, "join"):
            compile_feature(FEATURE, join_path=("transactions", "customers"))
        self.assertEqual(len(generate_variants(FEATURE)), 9)

    def test_six_routes_and_zero_candidates(self):
        p = Problem("default", "customer", 7)
        events = (self.event(),)
        facts = (Fact("a", T + timedelta(days=7), T + timedelta(days=7), 1),)
        for mode in RunMode:
            no_data = route(RunIntent(mode, DataState.NO_DATA), p, features=(FEATURE,))
            self.assertEqual(no_data["status"], "NO_DATA")
            self.assertNotIn("compiled_feature_specs", no_data)
            available = route(RunIntent(mode, DataState.DATA_AVAILABLE), p,
                              features=(FEATURE,), events=events, facts=facts)
            if mode != RunMode.STRUCTURES:
                self.assertEqual(len(available["compiled_feature_specs"]), 1)
        empty = route(RunIntent(RunMode.FEATURES, DataState.DATA_AVAILABLE), p,
                      events=events, facts=facts)
        self.assertEqual(empty["status"], "NO_CANDIDATES")
        self.assertNotIn("model_lift", empty)
        with self.assertRaisesRegex(ValueError, "horizon_days"):
            route(RunIntent(RunMode.FEATURES, DataState.DATA_AVAILABLE),
                  Problem("default", "customer"), events=events, facts=facts)

    def test_oof_freeze_replay_holdout(self):
        events = [self.event(str(i), i * 10, i % 2) for i in range(16)]
        values = [float(i % 2) for i in range(16)]
        first = time_ordered_oof(events, values)
        self.assertEqual(first, time_ordered_oof(events, values))
        portfolio = freeze([FEATURE.concept_id], "research-hash")
        result = evaluate_sealed_holdout(events, values, portfolio, first["holdout_indices"])
        self.assertEqual(result["status"], "HOLDOUT_EVALUATED")


if __name__ == "__main__":
    unittest.main()
