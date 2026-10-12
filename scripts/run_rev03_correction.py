"""Isolated, post-hoc REV-03 correction using verified reconstructed inputs.

Run from the repository root with .venv/Scripts/python.exe scripts/run_rev03_correction.py.
No historical evaluator, cache, guard, input, or model is overwritten.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import platform
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import pyarrow
from scipy.optimize import linprog
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "0745186a2435c693a990fa394031f0c036c24b60"
REV01 = ROOT / "docs/attachments/rev01_20261011"
REV02 = ROOT / "docs/attachments/rev02_20261011"
FEATURES = ["temperature_max_c_t", "relative_humidity_mean_t", "wind_speed_mean_t", "rainfall_mm_t"]
POLICY = {
    "id": "REV03-conservative-proxy-v1", "decision": "DEC-18",
    "usable": "retain provisionally for detected-export proxy",
    "unavailable": "mask observation/state; exclude touching transitions",
    "unknown": "mask conservatively; not a claim of proven error",
    "partial": "mask conservatively; local effect unresolved",
    "coverage": "No grid-specific observed-zero certificate; export completeness assumed outside flags",
    "threshold": "Q75 of valid active training grid-days across all 187 grids, before predictor/cohort filtering",
    "cohort": "Re-screen historical 46 only, unchanged training-only rules; no expansion",
    "model": "Independent unweighted unpenalized reference-category multinomial per grid/current state",
    "decision_rule": "Fixed argmax; ties use first class; no selection or tuning",
    "identity": "Post-hoc revised analysis, not untouched test or original-run reproduction",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified_hashes(mapping):
    for relative, expected in mapping.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Input/protected-file SHA mismatch: {relative}")


def threshold_population(detections, ledger):
    usable_dates = ledger.loc[ledger.usability.eq("usable"), "date"]
    active = detections.loc[
        detections.date.le(pd.Timestamp("2022-12-31"))
        & detections.date.isin(usable_dates) & detections.hotspot_count.gt(0),
        ["grid_id", "date", "frp_sum"],
    ].copy()
    if active.empty or not np.isfinite(active.frp_sum).all():
        raise ValueError("No valid finite active training population")
    boundary = float(active.frp_sum.quantile(.75, interpolation="linear"))
    if boundary <= 0:
        raise ValueError("Invalid positive FRP boundary")
    return boundary, active


def corrected_panel(detections, covariates, ledger, boundary):
    if detections.duplicated(["grid_id", "date"]).any() or ledger.date.duplicated().any():
        raise ValueError("Duplicate panel or ledger keys")
    statuses = set(ledger.usability)
    if not statuses.issubset({"usable", "unavailable", "unknown", "partial"}):
        raise ValueError(f"Unrecognized quality status: {statuses}")
    panel = detections.rename(columns={"state": "historical_state", "hotspot_count": "historical_hotspot_count", "frp_sum": "historical_frp_sum"})
    panel = panel.merge(covariates, on=["grid_id", "date"], validate="one_to_one", how="left")
    panel = panel.merge(ledger, on="date", validate="many_to_one", how="left")
    if panel.usability.isna().any() or panel.predictors_available.isna().any():
        raise ValueError("Incomplete ledger/covariate calendar")
    panel["observation_usable"] = panel.usability.eq("usable")
    ok = panel.observation_usable
    panel["hotspot_count"] = panel.historical_hotspot_count.astype("Int64").where(ok)
    panel["frp_sum"] = panel.historical_frp_sum.astype("Float64").where(ok)
    state = np.where(panel.historical_hotspot_count.eq(0), 0, np.where(panel.historical_frp_sum.lt(boundary), 1, 2))
    panel["state"] = pd.Series(state, index=panel.index, dtype="Int8").where(ok)
    finite = np.isfinite(panel[FEATURES].to_numpy(dtype=float)).all(axis=1)
    panel["predictors_available"] = panel.predictors_available.astype(bool) & finite
    return panel.sort_values(["grid_id", "date"]).reset_index(drop=True)


def transitions(panel):
    # Shift the complete calendar first: filtering unknown days first bridges gaps.
    result = panel.copy().sort_values(["grid_id", "date"]).reset_index(drop=True)
    grouped = result.groupby("grid_id", sort=False)
    for source, target in [("grid_id", "next_grid_id"), ("date", "next_date"), ("state", "next_state"),
                           ("observation_usable", "target_usable"), ("usability", "target_usability"),
                           ("reason", "target_reason"), ("historical_state", "historical_next_state")]:
        result[target] = grouped[source].shift(-1)
    result = result.rename(columns={"state": "current_state"})
    result["exact_next_day"] = result.next_date.eq(result.date + pd.Timedelta(days=1)) & result.grid_id.eq(result.next_grid_id)
    result["origin_unknown"] = ~result.observation_usable | result.current_state.isna()
    result["target_unknown"] = ~result.target_usable.fillna(False).astype(bool) | result.next_state.isna()
    result["missing_origin_predictor"] = ~result.predictors_available
    result["valid_observation_transition"] = result.exact_next_day & ~result.origin_unknown & ~result.target_unknown & ~result.missing_origin_predictor
    result["exclusion_reason"] = np.select(
        [~result.exact_next_day, result.origin_unknown, result.target_unknown, result.missing_origin_predictor],
        ["not_exact_next_day", "origin_observation_unknown", "target_observation_unknown", "missing_origin_predictor"],
        default="usable_before_cohort",
    )
    result["split"] = np.select(
        [result.next_date.isna(), result.next_date.le(pd.Timestamp("2022-12-31")), result.next_date.le(pd.Timestamp("2024-12-31"))],
        ["no_target", "train", "validation"], default="test",
    )
    return result


def pinned_functions():
    sources = {}
    for label, path in {"screening": "scripts/finalize_modeling_design_025.py", "model": "src/models/covariate_markov.py"}.items():
        sources[label] = subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)
    screen = ast.parse(sources["screening"])
    nodes = [n for n in screen.body if
             isinstance(n, ast.FunctionDef) and n.name in ("diagnose", "separation") or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("FEATURES", "RULES") for t in n.targets)]
    namespace = dict(np=np, pd=pd, json=json, linprog=linprog)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned_screening", "exec"), namespace)
    model = {}
    exec(compile(sources["model"], "pinned_model", "exec"), model)
    if namespace["FEATURES"] != FEATURES:
        raise ValueError("Pinned feature contract changed")
    return namespace, model, sources


def primary_scores(y, probabilities, logs):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    logs = np.asarray(logs, dtype=float)
    if len(y) == 0 or p.shape != (len(y), 3) or logs.shape != p.shape:
        raise ValueError("Empty/mismatched three-class evaluation")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("Invalid probabilities")
    if np.isnan(logs).any() or np.isposinf(logs).any() or not np.allclose(np.exp(logs), p, atol=1e-12, rtol=1e-10):
        raise ValueError("Invalid log/probability mapping")
    pred = p.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    matrix = confusion_matrix(y, pred, labels=[0, 1, 2])
    logloss = float(-logs[np.arange(len(y)), y].mean())
    result = {"n": len(y), "accuracy": float((pred == y).mean()),
              "balanced_accuracy": float(recall[support > 0].mean()), "macro_f1": float(f1.mean()),
              "weighted_f1": float(np.dot(f1, support) / len(y)),
              "log_loss": logloss if np.isfinite(logloss) else None,
              "log_loss_infinite": not np.isfinite(logloss),
              "brier": float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
              "actual_probability_zero": int(np.isneginf(logs[np.arange(len(y)), y]).sum())}
    classrows = []
    for s in range(3):
        binary = y == s
        classrows.append(dict(state=s, support=int(support[s]), precision=float(precision[s]), recall=float(recall[s]),
                              f1=float(f1[s]), auc=float(roc_auc_score(binary, p[:, s])) if 0 < binary.sum() < len(y) else None))
    return result, classrows, matrix


def json_file(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run():
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    rev01 = json.loads((REV01 / "handoff_verification.json").read_text())
    protected = json.loads((REV01 / "verification.json").read_text())["protected_after"]
    verified_hashes(protected)
    # Protect received evidence and unrelated work at its current bytes, including prior derived inputs.
    protected = {**protected, **{p.relative_to(ROOT).as_posix(): sha(p) for base in (REV01, REV02) for p in base.rglob("*") if p.is_file()}}
    for p in [ROOT / "README.md", ROOT / "docs/data-provenance.md", *sorted((ROOT / "notebooks").glob("*.ipynb")),
              *sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "tests").rglob("*.py"))]:
        protected[p.relative_to(ROOT).as_posix()] = sha(p)
    inputs = {f"docs/attachments/rev01_20261011/{name}": rev01["outputs"][name] for name in
              ("reconstructed_historical_detection_panel.parquet", "reconstructed_covariates.parquet")}
    inputs["docs/attachments/rev02_20261011/product_quality_ledger.csv"] = "77c8fe77a8e122fbead576056c02d176a2d21b46be8530788d835acaf8152462"
    inputs["final_test_050_3state.zip"] = "383dadbb682f02cf6734c5ba3d0257a09f20a63f7113676da1da6586a7d654cb"
    verified_hashes(inputs)
    fingerprint = hashlib.sha256(json.dumps(dict(policy=POLICY, inputs=inputs, script=sha(__file__)), sort_keys=True).encode()).hexdigest()[:12]
    out = ROOT / "outputs/revision_review" / f"REV03_20261011_{fingerprint}"
    out.mkdir(parents=True, exist_ok=False)
    json_file(out / "run_started.json", dict(utc=datetime.now(timezone.utc).isoformat(), inputs=inputs, protected_before=protected,
              policy=POLICY, pinned_commit=COMMIT, script_sha256=sha(__file__), run_id=out.name))
    try:
        test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_rev03_correction.py", "-v"], cwd=ROOT, capture_output=True, text=True)
        (out / "regression_tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
        check("targeted_regression_suite_passes", test.returncode == 0 and "Ran " in test.stderr)
        check("test_file_exists", (ROOT / "tests/test_rev03_correction.py").is_file())
        screen, model_api, sources = pinned_functions()
        for name, content in sources.items():
            (out / f"pinned_{name}_source.py").write_bytes(content)
        json_file(out / "analysis_plan.json", dict(policy=POLICY, rules=screen["RULES"], test_tuning=False,
                  fit_failure_policy="Fail run; no silent group/grid removal or relaxed criteria",
                  sources_sha256={k: hashlib.sha256(v).hexdigest() for k, v in sources.items()},
                  test_source_sha256=sha(ROOT / "tests/test_rev03_correction.py")))
        detections = pd.read_parquet(REV01 / "reconstructed_historical_detection_panel.parquet")
        covariates = pd.read_parquet(REV01 / "reconstructed_covariates.parquet")
        ledger = pd.read_csv(REV02 / "product_quality_ledger.csv", parse_dates=["date"])
        detections["date"] = pd.to_datetime(detections.date)
        covariates["date"] = pd.to_datetime(covariates.date)
        check("full_study_calendar", len(detections) == 678997 and detections.grid_id.nunique() == 187 and detections.date.nunique() == 3631)
        check("documented_status_counts", ledger.usability.value_counts().to_dict() == dict(usable=3472, unavailable=84, unknown=73, partial=2))
        boundary, active = threshold_population(detections, ledger)
        active.to_parquet(out / "threshold_training_population.parquet", index=False)
        historical_active = detections[detections.date.le("2022-12-31") & detections.hotspot_count.gt(0)]
        check("historical_population_identity", len(historical_active) == 41545 and abs(historical_active.frp_sum.quantile(.75) - 16.84) < 1e-12)
        check("threshold_training_only", active.date.max() <= pd.Timestamp("2022-12-31"))
        panel = corrected_panel(detections, covariates, ledger, boundary)
        check("unknown_observation_is_nullable", panel.loc[~panel.observation_usable, ["state", "hotspot_count", "frp_sum"]].isna().all().all())
        check("usable_zero_is_zero", panel.loc[panel.observation_usable & panel.historical_hotspot_count.eq(0), "state"].eq(0).all())
        check("state_class_contract", set(panel.state.dropna().unique()) == {0, 1, 2})
        panel.to_parquet(out / "corrected_grid_day.parquet", index=False)
        trans = transitions(panel)
        with zipfile.ZipFile(ROOT / "final_test_050_3state.zip") as archive:
            frozen = json.loads(archive.read("final_test_050_3state/frozen_design.json"))
            original_predictions = pd.read_parquet(io.BytesIO(archive.read("final_test_050_3state/test_predictions.parquet")))
        ids = sorted(frozen["eligible_grid_ids"])
        check("frozen_candidate_cohort", len(ids) == 46 and len(set(ids)) == 46)
        trans["historical_frozen_grid"] = trans.grid_id.isin(ids)
        training = trans[trans.historical_frozen_grid & trans.valid_observation_transition & trans.split.eq("train")].copy()
        for col in ("current_state", "next_state"):
            training[col] = training[col].astype(int)
        diagnostics, eligible = [], []
        for i, gid in enumerate(ids):
            ready = []
            for state in range(3):
                frame = training[training.grid_id.eq(gid) & training.current_state.eq(state)]
                d = screen["diagnose"](frame)
                ready.append(d["status"] in ("regression_ready", "deterministic_ready"))
                diagnostics.append(dict(d, grid_id=gid, current_state=state))
            eligible.append(dict(grid_id=gid, eligible=all(ready), state_statuses=";".join(diagnostics[-3 + j]["status"] for j in range(3))))
            if i % 10 == 0:
                print(f"Training-only re-screen: {i+1}/46 grids", flush=True)
        eligibility = pd.DataFrame(eligible)
        retained = sorted(eligibility.loc[eligibility.eligible, "grid_id"])
        check("cohort_no_expansion", bool(retained) and set(retained).issubset(ids))
        eligibility.to_csv(out / "grid_eligibility.csv", index=False)
        pd.DataFrame([dict(d, reasons=";".join(d["reasons"])) for d in diagnostics]).to_csv(out / "screening_diagnostics.csv", index=False)
        trans["revised_grid_eligible"] = trans.grid_id.isin(retained)
        trans["included"] = trans.valid_observation_transition & trans.revised_grid_eligible
        trans.loc[trans.valid_observation_transition & ~trans.historical_frozen_grid, "exclusion_reason"] = "outside_historical_candidate_cohort"
        trans.loc[trans.valid_observation_transition & trans.historical_frozen_grid & ~trans.revised_grid_eligible, "exclusion_reason"] = "training_screen_grid_drop"
        trans.loc[trans.included, "exclusion_reason"] = "included"
        trans.to_parquet(out / "transition_ledger.parquet", index=False)
        candidate = trans[trans.historical_frozen_grid & trans.exact_next_day]
        counts = []
        for split, original_n in [("train", 105340), ("validation", 33626), ("test", 27968)]:
            t = candidate[candidate.split.eq(split)]
            check(f"{split}_historical_predictor_count", int(t.predictors_available.sum()) == original_n)
            affected = t.predictors_available & (t.origin_unknown | t.target_unknown)
            expected = dict(train=1196, validation=2530, test=4278)[split]
            check(f"{split}_quality_union_reconciliation", int(affected.sum()) == expected)
            counts.append(dict(split=split, calendar_candidates=len(t), historical_predictor_available=original_n,
                               observation_excluded=int(affected.sum()), usable_before_rescreen=int(t.valid_observation_transition.sum()),
                               training_screen_excluded=int((t.valid_observation_transition & ~t.revised_grid_eligible).sum()),
                               revised_included=int(t.included.sum()),
                               threshold_changed_origin=int((t.valid_observation_transition & t.current_state.ne(t.historical_state)).sum()),
                               threshold_changed_target=int((t.valid_observation_transition & t.next_state.ne(t.historical_next_state)).sum())))
        split_counts = pd.DataFrame(counts)
        split_counts.to_csv(out / "split_reconciliation.csv", index=False)
        candidate.groupby(["grid_id", "split", "exclusion_reason"], observed=True).size().rename("rows").reset_index().to_csv(out / "per_grid_exclusions.csv", index=False)
        candidate.groupby(["split", "usability", "target_usability", "missing_origin_predictor"], observed=True).size().rename("rows").reset_index().to_csv(out / "endpoint_quality_counts.csv", index=False)
        check("training_changed_requires_refit", int(split_counts.loc[split_counts.split.eq("train"), "revised_included"].iloc[0]) < 105340)
        fitted, fitrows = {}, []
        for i, gid in enumerate(retained):
            for state in range(3):
                d = next(d for d in diagnostics if d["grid_id"] == gid and d["current_state"] == state)
                t = training[training.grid_id.eq(gid) & training.current_state.eq(state)]
                if d["status"] == "deterministic_ready":
                    model = {"type": "deterministic", "next_state": int(d["deterministic_next_state"])}
                    fit = dict(success=True, iterations=0)
                else:
                    model, fit = model_api["fit_transition"](t, json.loads(d["retained_features"]), json.loads(d["training_means"]), json.loads(d["training_scales"]))
                fitrows.append(dict(fit, grid_id=gid, current_state=state, training_rows=len(t)))
                pd.DataFrame(fitrows).to_csv(out / "fit_diagnostics.csv", index=False)
                check(f"fit_{gid}_state_{state}", fit["success"])
                fitted[(gid, state)] = model
                path = out / "models" / str(gid)
                path.mkdir(parents=True, exist_ok=True)
                json_file(path / f"state_{state}.json", dict(model=model, fit=fit, run_id=out.name, training_only=True))
            if i % 10 == 0:
                print(f"Same-family fits: {i+1}/{len(retained)} grids", flush=True)
        metrics, classes, matrices, coverage = [], [], [], []
        for split in ("validation", "test"):
            rows = trans[trans.included & trans.split.eq(split)].copy()
            for col in ("current_state", "next_state"):
                rows[col] = rows[col].astype(int)
            parts = []
            for (gid, state), frame in rows.groupby(["grid_id", "current_state"], sort=True):
                logs = model_api["predict_log_proba"](fitted[(gid, int(state))], frame)[:, :3]
                p = np.exp(logs)
                result = frame[["grid_id", "date", "next_date", "current_state", "next_state", *FEATURES]].rename(columns={"next_state": "actual_next_state"}).copy()
                for j in range(3):
                    result[f"p_{j}"] = p[:, j]
                    result[f"log_p_{j}"] = logs[:, j]
                result["primary_argmax"] = p.argmax(axis=1)
                result["run_id"] = out.name
                parts.append(result)
            predicted = pd.concat(parts, ignore_index=True).sort_values(["grid_id", "next_date"]).reset_index(drop=True)
            check(f"{split}_prediction_row_identity", len(predicted) == len(rows) and not predicted.duplicated(["grid_id", "next_date"]).any())
            check(f"{split}_same_day_contract", predicted.next_date.eq(predicted.date + pd.Timedelta(days=1)).all())
            predicted.to_parquet(out / f"{split}_predictions.parquet", index=False)
            groups = [("overall", "ALL", -1, predicted)] + [("per_grid", gid, -1, frame) for gid, frame in predicted.groupby("grid_id")]
            groups += [("per_current_state", "ALL", int(s), frame) for s, frame in predicted.groupby("current_state")]
            for scope, gid, state, frame in groups:
                meta = dict(split=split, scope=scope, grid_id=gid, current_state=state)
                score, cl, matrix = primary_scores(frame.actual_next_state, frame[[f"p_{j}" for j in range(3)]], frame[[f"log_p_{j}" for j in range(3)]])
                metrics.append(dict(meta, **score))
                classes.extend(dict(meta, **c) for c in cl)
                matrices.extend(dict(meta, actual_state=a, predicted_state=b, count=int(matrix[a, b])) for a in range(3) for b in range(3))
            coverage.extend(dict(split=split, grid_id=gid, current_state=int(s), rows=len(f)) for (gid, s), f in predicted.groupby(["grid_id", "current_state"]))
        pd.DataFrame(metrics).to_csv(out / "primary_metrics.csv", index=False)
        pd.DataFrame(classes).to_csv(out / "primary_class_metrics.csv", index=False)
        pd.DataFrame(matrices).to_csv(out / "primary_confusion_matrices.csv", index=False)
        pd.DataFrame(coverage).to_csv(out / "prediction_coverage.csv", index=False)
        check("test_not_used_in_threshold_or_fits", training.next_date.max() <= pd.Timestamp("2022-12-31"))
        historical_score = primary_scores(original_predictions.actual_next_state,
                original_predictions[["p_no_activity", "p_nonhigh", "p_high"]], original_predictions[["log_p_0", "log_p_1", "log_p_2"]])[0]
        verified_hashes(protected)
        verified_hashes(inputs)
        check("protected_inputs_and_unrelated_work_unchanged", True)
        summary = dict(run_id=out.name, boundary_frp_mw=boundary, original_boundary_frp_mw=16.84,
                       valid_active_training_rows=len(active), historical_active_training_rows=41545,
                       original_candidate_grids=len(ids), retained_grids=len(retained), dropped_grid_ids=sorted(set(ids) - set(retained)),
                       models=len(fitted), policy=POLICY, branch="training_changed_same_family_refit",
                       split_counts=counts, primary_overall=[m for m in metrics if m["scope"] == "overall"],
                       historical_test=historical_score,
                       comparison_warning="Different rows/cohort and potentially threshold/labels: metric differences do not demonstrate improvement",
                       limitations=["Conservative masks can exclude legitimate/reprocessed records; no confirmed-error count implied",
                                    "Export completeness and per-grid coverage outside flags remain unverified",
                                    "Fairbanks recovery/backfill beyond reviewed minimum interval unresolved",
                                    "Reconstructed inputs/new fits are not original intermediate/model bytes or original environment",
                                    "Already viewed test, post-hoc analysis; no independent superiority, calibration or operational validation"])
        json_file(out / "summary.json", summary)
        report = f"""# REV-03 — conservative post-hoc correction

Run `{out.name}`; decision DEC-18; pinned final model/screen source `{COMMIT}`.
This is a separately fitted retrospective detected-export-proxy analysis, not recovered originals or a new untouched test.

## Treatment and branch

Only ledger status usable is provisionally retained. Unavailable/unknown/partial observations become nullable count/FRP/state, never state zero or a fourth predicted class. Historical quantities remain separate for traceability. Usable no-detection days stay zero under the conditional completeness assumption. Both endpoints and origin predictors are required; shifting precedes filtering. Excluding uncertain dates is conservative and does not prove every excluded observation erroneous. Fairbanks recovery beyond the reviewed minimum remains unresolved.

Training transitions changed, so the same unweighted, unpenalized multinomial family was refitted with unchanged training-only screens. Historical 46 candidates only; retained {len(retained)}, dropped {len(ids)-len(retained)}. All {len(fitted)} fits pass the original success criteria. No new features, weights, regularization, decision-threshold tuning, eligibility expansion or baseline study.

## Training threshold and row accounting

Q75 is {boundary!r} MW from {len(active):,} valid active training grid-days across all 187 study grids before predictor/cohort filtering; historical 16.84 on 41,545 rows. No fixed historical threshold was forced. Target dates determine splits.

```json
{json.dumps(counts, indent=2)}
```

## Results and limits

`primary_metrics.csv`, `primary_class_metrics.csv`, `primary_confusion_matrices.csv` and the two prediction Parquets trace all revised results. `transition_ledger.parquet` retains every calendar origin, nullable endpoints and exclusion reasons; `per_grid_exclusions.csv`/`endpoint_quality_counts.csv` reconcile exclusions. Model files and diagnostics are separate and hashed. Primary argmax is unchanged.

```json
{json.dumps(summary['primary_overall'], indent=2)}
```

Original test metrics are retained in summary.json for historical context. Unequal rows/cohort and changed labels/threshold prohibit interpreting score differences as evidence of improved predictive performance. No original figures/manuscript are revised here. REV-05 must align final claims and review deliverables. Original model bytes/fitting environment, actual spatial observed-zero coverage, thermal type and independent covariate benefit remain unverified.

## Verification

Targeted unittest output is in regression_tests.txt; executed integration checks and environment are in verification.json. SHA checks protect original ZIPs, reporting evidence, prior reconstructions/quality ledger and unrelated current source/notebooks. Raw weather is not rewritten or reread by this stage. This does not replay the historical evaluator or reset any guard. Completed run directories cannot be reused.
"""
        (out / "REV03_REPORT.md").write_text(report, encoding="utf-8")
        import sklearn
        environment = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, pyarrow=pyarrow.__version__, sklearn=sklearn.__version__)
        json_file(out / "verification.json", dict(checks=checks, environment=environment,
                  protected_before=protected, protected_after={p: sha(ROOT / p) for p in protected}, inputs=inputs,
                  script_sha256=sha(__file__), test_sha256=sha(ROOT / "tests/test_rev03_correction.py"),
                  outputs={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}))
        json_file(out / "run_completed.json", dict(utc=datetime.now(timezone.utc).isoformat(), run_id=out.name,
                  checks_passed=len(checks), verification_sha256=sha(out / "verification.json"), summary_sha256=sha(out / "summary.json")))
        print(json.dumps(dict(output=out.relative_to(ROOT).as_posix(), checks=len(checks), boundary=boundary, retained_grids=len(retained), metrics=summary["primary_overall"]), indent=2), flush=True)
    except BaseException:
        json_file(out / "run_failed.json", dict(utc=datetime.now(timezone.utc).isoformat(), checks=checks, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    run()
