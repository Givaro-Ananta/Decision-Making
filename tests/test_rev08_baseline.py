import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline", ROOT / "scripts/compare_rev08_baseline.py")
api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)


def training():
    return pd.DataFrame(dict(grid_id=["A"] * 12, current_state=[s for s in range(3) for _ in range(4)],
                             next_state=[0, 0, 1, 2] * 3, next_date=pd.to_datetime(["2022-01-02"] * 12)))


class BaselineTests(unittest.TestCase):
    def test_hand_counts_probabilities_class_order(self):
        rates = api.rates_from_training(training())
        np.testing.assert_allclose(rates[[f"baseline_p_{j}" for j in range(3)]], [[.5, .25, .25]] * 3)
        self.assertEqual(rates.training_rows.tolist(), [4] * 3)

    def test_test_labels_cannot_enter_rates(self):
        t = training(); t.loc[0, "next_date"] = pd.Timestamp("2025-01-01")
        with self.assertRaises(ValueError): api.rates_from_training(t)

    def test_missing_current_state_and_unknown_destination_fail(self):
        t = training()
        with self.assertRaises(ValueError): api.rates_from_training(t[t.current_state.ne(2)])
        t.loc[0, "next_state"] = 3
        with self.assertRaises(ValueError): api.rates_from_training(t)

    def test_unobserved_destination_zero_and_infinite_loss_not_clipped(self):
        t = training(); t["next_state"] = 0
        rates = api.rates_from_training(t)
        np.testing.assert_array_equal(rates[[f"baseline_p_{j}" for j in range(3)]], [[1, 0, 0]] * 3)
        score = api.proper_scores([2], [[1., 0., 0.]])
        self.assertTrue(score["log_loss_infinite"]); self.assertIsNone(score["log_loss"])
        self.assertEqual(score["zero_probability_actual"], 1); self.assertEqual(score["brier"], 2.)

    def test_hand_calculated_proper_scores(self):
        score = api.proper_scores([0, 2], [[.5, .25, .25], [.25, .25, .5]])
        self.assertAlmostEqual(score["log_loss"], -np.log(.5))
        self.assertAlmostEqual(score["brier"], .375)
        self.assertEqual([score[f"support_{j}"] for j in range(3)], [1, 0, 1])

    def test_paired_rows_cannot_disappear_or_duplicate(self):
        pred = pd.DataFrame(dict(grid_id=["A", "B"], current_state=[0, 0], next_date=pd.to_datetime(["2025-01-01"] * 2)))
        with self.assertRaises(ValueError): api.attach_rates(pred, api.rates_from_training(training()))
        with self.assertRaises(ValueError): api.attach_rates(pd.concat([pred, pred]), api.rates_from_training(training()))

    def test_rates_do_not_use_evaluation_labels(self):
        pred = pd.DataFrame(dict(grid_id=["A"], current_state=[1], next_date=pd.to_datetime(["2025-01-01"]), actual_next_state=[0]))
        rates = api.rates_from_training(training())
        first = api.attach_rates(pred, rates)
        pred["actual_next_state"] = 2
        second = api.attach_rates(pred, rates)
        np.testing.assert_array_equal(first[[f"baseline_p_{j}" for j in range(3)]], second[[f"baseline_p_{j}" for j in range(3)]])

    def test_weighted_partition_reconciles_and_invalid_probability_fails(self):
        p = np.array([[.5, .25, .25], [.25, .5, .25], [.1, .2, .7]])
        y = [0, 2, 2]
        all_rows = api.proper_scores(y, p)
        a, b = api.proper_scores(y[:1], p[:1]), api.proper_scores(y[1:], p[1:])
        for metric in ("log_loss", "brier"):
            self.assertAlmostEqual(all_rows[metric], (a[metric] + 2 * b[metric]) / 3)
        with self.assertRaises(ValueError): api.proper_scores([0], [[.5, .5, .5]])


if __name__ == "__main__": unittest.main()
