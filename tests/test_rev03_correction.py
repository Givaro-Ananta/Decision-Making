import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rev03", ROOT / "scripts/run_rev03_correction.py")
rev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rev)


def fixture(dates, statuses=None, counts=None, frp=None, grid="g1"):
    n = len(dates)
    date = pd.to_datetime(dates)
    detection = pd.DataFrame(dict(grid_id=[grid] * n, date=date, hotspot_count=counts or [0] * n,
                                  frp_sum=frp or [0.] * n, state=[0] * n))
    cov = pd.DataFrame(dict(grid_id=[grid] * n, date=date, predictors_available=[True] * n))
    for f in rev.FEATURES:
        cov[f] = np.arange(n, dtype=float) + 1
    ledger = pd.DataFrame(dict(date=date, usability=statuses or ["usable"] * n, reason=["fixture"] * n))
    return detection, cov, ledger


class ObservationContract(unittest.TestCase):
    def test_zero_and_unknown_are_distinct(self):
        data = fixture(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
                       ["usable", "unknown", "partial", "unavailable"], [0, 0, 1, 0], [0., 0., 20., 0.])
        panel = rev.corrected_panel(*data, 16.84)
        self.assertEqual(panel.state.iloc[0], 0)
        self.assertTrue(panel.loc[1:, ["state", "hotspot_count", "frp_sum"]].isna().all().all())
        self.assertEqual(panel.historical_hotspot_count.iloc[2], 1)
        self.assertEqual(str(panel.state.dtype), "Int8")

    def test_unknown_origin_and_target(self):
        panel = rev.corrected_panel(*fixture(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
                                             ["usable", "unknown", "usable", "usable"]), 16.84)
        t = rev.transitions(panel)
        self.assertEqual(t.valid_observation_transition.tolist(), [False, False, True, False])
        self.assertEqual(t.exclusion_reason.iloc[0], "target_observation_unknown")
        self.assertEqual(t.exclusion_reason.iloc[1], "origin_observation_unknown")

    def test_missing_origin_predictor_not_target_predictor(self):
        d, cov, ledger = fixture(["2020-01-01", "2020-01-02", "2020-01-03"])
        cov.loc[1, rev.FEATURES[0]] = np.nan
        t = rev.transitions(rev.corrected_panel(d, cov, ledger, 16.84))
        self.assertEqual(t.valid_observation_transition.tolist(), [True, False, False])
        self.assertEqual(t.exclusion_reason.iloc[1], "missing_origin_predictor")

    def test_leap_and_year_boundaries_target_split(self):
        panel = rev.corrected_panel(*fixture(["2020-02-28", "2020-02-29", "2020-03-01", "2022-12-31", "2023-01-01", "2024-12-31", "2025-01-01"]), 16.84)
        t = rev.transitions(panel)
        self.assertEqual(t.valid_observation_transition.tolist(), [True, True, False, True, False, True, False])
        self.assertEqual(t.split.iloc[3], "validation")
        self.assertEqual(t.split.iloc[5], "test")

    def test_no_cross_grid_or_missing_calendar_bridge(self):
        d1, c1, ledger = fixture(["2020-01-01", "2020-01-03"], grid="g1")
        d2, c2, _ = fixture(["2020-01-01", "2020-01-03"], grid="g2")
        panel = rev.corrected_panel(pd.concat([d1, d2]), pd.concat([c1, c2]), ledger, 16.84)
        t = rev.transitions(panel)
        self.assertFalse(t.valid_observation_transition.any())
        self.assertTrue(t.loc[t.next_date.notna(), "grid_id"].eq(t.loc[t.next_date.notna(), "next_grid_id"]).all())

    def test_exact_frp_boundary_and_zero_precedence(self):
        d, c, ledger = fixture(["2020-01-01", "2020-01-02", "2020-01-03"], counts=[1, 1, 0], frp=[16.839, 16.84, 0.])
        self.assertEqual(rev.corrected_panel(d, c, ledger, 16.84).state.tolist(), [1, 2, 0])

    def test_missing_ledger_and_duplicate_keys_fail(self):
        d, c, ledger = fixture(["2020-01-01", "2020-01-02"])
        with self.assertRaises(ValueError):
            rev.corrected_panel(d, c, ledger.iloc[:1], 16.84)
        with self.assertRaises(ValueError):
            rev.corrected_panel(pd.concat([d, d]), c, ledger, 16.84)
        ledger.loc[0, "usability"] = "invented_status"
        with self.assertRaises(ValueError):
            rev.corrected_panel(d, c, ledger, 16.84)

    def test_threshold_training_active_all_study_before_covariate_filter(self):
        d, cov, ledger = fixture(["2022-12-28", "2022-12-29", "2022-12-30", "2022-12-31", "2025-01-01"],
                                 ["usable", "usable", "unknown", "usable", "usable"], [1, 1, 1, 0, 1], [4., 12., 1000., 0., 9999.])
        cov["predictors_available"] = False
        boundary, active = rev.threshold_population(d, ledger)
        self.assertEqual(boundary, 10.)
        self.assertEqual(len(active), 2)
        d.loc[4, "frp_sum"] = 1e9
        self.assertEqual(rev.threshold_population(d, ledger)[0], boundary)
        ledger.loc[3, "usability"] = "unavailable"
        self.assertEqual(rev.threshold_population(d, ledger)[0], boundary)


class ModelAndProvenanceContract(unittest.TestCase):
    def test_proper_scores_and_argmax_class_order(self):
        p = np.array([[.6, .3, .1], [.2, .5, .3], [.1, .3, .6]])
        y = np.array([0, 2, 1])
        score, classes, matrix = rev.primary_scores(y, p, np.log(p))
        self.assertAlmostEqual(score["log_loss"], -(np.log(.6) + 2 * np.log(.3)) / 3)
        self.assertAlmostEqual(score["brier"], sum(np.square(p[i] - np.eye(3)[y[i]]).sum() for i in range(3)) / 3)
        self.assertEqual(matrix.tolist(), [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        self.assertEqual([c["support"] for c in classes], [1, 1, 1])
        with self.assertRaises(ValueError):
            rev.primary_scores(y, p * 2, np.log(p))

    def test_no_clipping_impossible_destination(self):
        p = np.array([[1., 0., 0.]])
        score, _, _ = rev.primary_scores([2], p, [[0., -np.inf, -np.inf]])
        self.assertTrue(score["log_loss_infinite"])
        self.assertIsNone(score["log_loss"])
        self.assertEqual(score["actual_probability_zero"], 1)

    def test_pinned_model_gradient_and_supported_class_mapping(self):
        screen, model, _ = rev.pinned_functions()
        self.assertEqual(screen["RULES"]["min_destination_count"], 10)
        X = np.c_[np.ones(6), [-2., -1., 0., .5, 1., 2.]]
        theta = np.array([.1, -.3, .2, .4])
        y = np.array([0, 1, 2, 0, 2, 1])
        _, grad = model["objective"](theta, X, y, 3)
        delta = 1e-6
        finite = [(model["objective"](theta + np.eye(4)[j] * delta, X, y, 3)[0] -
                   model["objective"](theta - np.eye(4)[j] * delta, X, y, 3)[0]) / (2 * delta) for j in range(4)]
        np.testing.assert_allclose(grad, finite, atol=1e-7)
        logs = model["predict_log_proba"]({"type": "deterministic", "next_state": 2}, pd.DataFrame(index=range(2)))[:, :3]
        np.testing.assert_array_equal(np.exp(logs), [[0, 0, 1], [0, 0, 1]])

    def test_hash_integrity_independent_of_mtime(self):
        import os
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(b"immutable fixture")
            relative = path.relative_to(ROOT).as_posix()
            mapping = {relative: rev.sha(path)}
            os.utime(path, (1, 1))
            rev.verified_hashes(mapping)
            path.write_bytes(b"changed fixture")
            with self.assertRaises(ValueError):
                rev.verified_hashes(mapping)
            with self.assertRaises(FileNotFoundError):
                rev.verified_hashes({relative + ".missing": "missing"})


if __name__ == "__main__":
    unittest.main()
