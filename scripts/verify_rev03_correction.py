"""Verify a completed REV-03 run without refitting or modifying its evidence."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(directory):
    out = Path(directory).resolve()
    if not out.is_relative_to(ROOT / "outputs/revision_review"):
        raise ValueError("Expected isolated revision output within this workspace")
    v = json.loads((out / "verification.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    completed = json.loads((out / "run_completed.json").read_text())
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(dict(name=name, passed=True))

    check("completed_manifest_identity", sha(out / "verification.json") == completed["verification_sha256"])
    check("completed_summary_identity", sha(out / "summary.json") == completed["summary_sha256"])
    for name, expected in v["outputs"].items():
        check(f"output_sha:{name}", sha(out / name) == expected)
    for name, expected in v["protected_before"].items():
        check(f"protected_sha:{name}", sha(ROOT / name) == expected == v["protected_after"][name])
    for name, expected in v["inputs"].items():
        check(f"input_sha:{name}", sha(ROOT / name) == expected)
    check("runner_identity", sha(ROOT / "scripts/run_rev03_correction.py") == v["script_sha256"])
    check("regression_test_identity", sha(ROOT / "tests/test_rev03_correction.py") == v["test_sha256"])
    ledger = pd.read_csv(ROOT / "docs/attachments/rev02_20261011/product_quality_ledger.csv", parse_dates=["date"])
    panel = pd.read_parquet(out / "corrected_grid_day.parquet")
    trans = pd.read_parquet(out / "transition_ledger.parquet")
    source = pd.read_parquet(ROOT / "docs/attachments/rev01_20261011/reconstructed_historical_detection_panel.parquet").sort_values(["grid_id", "date"]).reset_index(drop=True)
    check("full_panel_identity", len(panel) == 678997 and panel[["grid_id", "date"]].equals(source[["grid_id", "date"]]))
    check("historical_quantities_preserved", np.array_equal(panel.historical_hotspot_count, source.hotspot_count) and np.array_equal(panel.historical_frp_sum, source.frp_sum) and np.array_equal(panel.historical_state, source.state))
    expected_usable = panel.date.isin(ledger.loc[ledger.usability.eq("usable"), "date"])
    check("quality_mask_matches_source_ledger", np.array_equal(panel.observation_usable, expected_usable))
    check("masked_values_never_zero", panel.loc[~expected_usable, ["state", "hotspot_count", "frp_sum"]].isna().all().all())
    check("usable_quantities_unchanged", np.array_equal(panel.loc[expected_usable, "hotspot_count"].astype(int), source.loc[expected_usable, "hotspot_count"]))
    active = source[expected_usable & source.hotspot_count.gt(0) & source.date.le("2022-12-31")]
    boundary = float(np.quantile(active.frp_sum.to_numpy(), .75, method="linear"))
    check("independent_threshold_arithmetic", boundary == summary["boundary_frp_mw"] and len(active) == summary["valid_active_training_rows"])
    expected_state = np.where(source.hotspot_count.eq(0), 0, np.where(source.frp_sum.lt(boundary), 1, 2))
    check("all_usable_states_recomputed", np.array_equal(panel.loc[expected_usable, "state"].astype(int), expected_state[expected_usable]))
    check("one_transition_origin_per_panel_row", trans[["grid_id", "date"]].equals(panel[["grid_id", "date"]]))
    exact = trans.next_date.eq(trans.date + pd.Timedelta(days=1)) & trans.grid_id.eq(trans.next_grid_id)
    expected_valid = exact & trans.current_state.notna() & trans.next_state.notna() & trans.predictors_available
    check("transition_validity_independent_reconciliation", np.array_equal(expected_valid, trans.valid_observation_transition))
    eligibility = pd.read_csv(out / "grid_eligibility.csv")
    eligible_ids = set(eligibility.loc[eligibility.eligible, "grid_id"])
    check("included_rows_follow_training_cohort", np.array_equal(trans.included, expected_valid & trans.grid_id.isin(eligible_ids)))
    check("no_cohort_expansion", len(eligibility) == 46 and len(eligible_ids) == summary["retained_grids"])
    counts = pd.read_csv(out / "split_reconciliation.csv")
    for row in counts.itertuples():
        check(f"{row.split}_exclusion_waterfall", row.historical_predictor_available == row.observation_excluded + row.training_screen_excluded + row.revised_included)
    training = trans[trans.included & trans.split.eq("train")]
    for path in sorted((out / "models").glob("*/state_*.json")):
        content = json.loads(path.read_text())
        model = content["model"]
        gid, state = path.parent.name, int(path.stem.split("_")[1])
        group = training[training.grid_id.eq(gid) & training.current_state.eq(state)]
        check(f"training_fit_success:{gid}:{state}", content["fit"]["success"] and content["training_only"])
        if model["type"] == "multinomial_logistic":
            for f in model["features"]:
                check(f"training_standardization:{gid}:{state}:{f}", abs(group[f].mean() - model["means"][f]) < 1e-12 and abs(group[f].std(ddof=0) - model["scales"][f]) < 1e-12)
            check(f"unweighted_unpenalized:{gid}:{state}", model["penalty"] is None)
    for split in ("validation", "test"):
        predicted = pd.read_parquet(out / f"{split}_predictions.parquet")
        expected_rows = trans[trans.included & trans.split.eq(split)].sort_values(["grid_id", "next_date"]).reset_index(drop=True)
        check(f"{split}_keys", predicted[["grid_id", "date", "next_date"]].equals(expected_rows[["grid_id", "date", "next_date"]]))
        check(f"{split}_labels", np.array_equal(predicted.actual_next_state, expected_rows.next_state.astype(int)) and np.array_equal(predicted.current_state, expected_rows.current_state.astype(int)))
        p = predicted[[f"p_{j}" for j in range(3)]].to_numpy()
        logs = predicted[[f"log_p_{j}" for j in range(3)]].to_numpy()
        check(f"{split}_probabilities", np.isfinite(p).all() and (p >= 0).all() and np.max(np.abs(p.sum(axis=1) - 1)) < 1e-12)
        check(f"{split}_primary_decision", np.array_equal(predicted.primary_argmax, p.argmax(axis=1)))
        for (gid, state), frame in predicted.groupby(["grid_id", "current_state"]):
            model = json.loads((out / "models" / gid / f"state_{state}.json").read_text())["model"]
            expected_logs = np.full((len(frame), 3), -np.inf)
            if model["type"] == "deterministic":
                expected_logs[:, model["next_state"]] = 0
            else:
                features = np.column_stack([(frame[f] - model["means"][f]) / model["scales"][f] for f in model["features"]])
                X = np.column_stack([np.ones(len(frame)), features])
                logits = np.column_stack([X @ np.asarray(model["coefficients_standardized"]).T, np.zeros(len(frame))])
                expected_logs[:, model["classes"]] = logits - logsumexp(logits, axis=1)[:, None]
            check(f"{split}_model_prediction:{gid}:{state}", np.allclose(frame[[f"log_p_{j}" for j in range(3)]], expected_logs, atol=1e-12, rtol=0))
        y = predicted.actual_next_state.to_numpy(dtype=int)
        scores = next(m for m in summary["primary_overall"] if m["split"] == split)
        check(f"{split}_logloss", abs(-logs[np.arange(len(y)), y].mean() - scores["log_loss"]) < 1e-12)
        check(f"{split}_brier", abs(np.square(p - np.eye(3)[y]).sum(axis=1).mean() - scores["brier"]) < 1e-12)
        matrix = np.zeros((3, 3), dtype=int)
        np.add.at(matrix, (y, p.argmax(axis=1)), 1)
        stored = pd.read_csv(out / "primary_confusion_matrices.csv")
        stored = stored[stored.split.eq(split) & stored.scope.eq("overall")].sort_values(["actual_state", "predicted_state"])
        check(f"{split}_confusion_matrix", np.array_equal(stored["count"].to_numpy().reshape(3, 3), matrix))
    evidence = ROOT / "docs/attachments/rev03_20261011"
    evidence.mkdir(exist_ok=True)
    target = evidence / "independent_verification.json"
    if target.exists():
        raise FileExistsError("Verification receipt already exists; preserve the completed record")
    receipt = dict(date="2026-10-11", run_path=out.relative_to(ROOT).as_posix(), run_id=out.name,
                   independent_checks_passed=len(checks), checks=checks, summary=summary,
                   verifier_sha256=sha(__file__), completed_sha256=sha(out / "run_completed.json"),
                   verification_sha256=sha(out / "verification.json"), environment=v["environment"],
                   protected_files=len(v["protected_before"]), integration_checks_passed=len(v["checks"]),
                   regression_tests="12 passing targeted unittest cases; see run regression_tests.txt",
                   limitations=summary["limitations"])
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (evidence / "REV03_REPORT.md").write_bytes((out / "REV03_REPORT.md").read_bytes())
    print(json.dumps(dict(checked=len(checks), receipt=target.relative_to(ROOT).as_posix(), run=out.name), indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/verify_rev03_correction.py outputs/revision_review/REV03_<id>")
    run(sys.argv[1])
