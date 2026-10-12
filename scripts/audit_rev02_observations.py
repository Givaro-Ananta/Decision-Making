"""Read-only REV-02 evidence reconciliation; never invoke the model pipeline.

Run from the repository root with .venv/Scripts/python.exe. Outputs are audit
annotations, not corrected data or evaluation results. Official-advisory rules
below were reviewed on 2026-10-11; rerunning does not refresh their web evidence.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/attachments/rev02_20261011"
COMMIT = "0745186a2435c693a990fa394031f0c036c24b60"
OUTAGES = "https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms/outages.html"
ISSUE477 = "https://landweb.modaps.eosdis.nasa.gov/displayissue?id=477"
ISSUE480 = "https://landweb.modaps.eosdis.nasa.gov/displayissue?id=480"
ISSUE1178 = "https://landweb.modaps.eosdis.nasa.gov/displayissue?id=1178"
RAW = [ROOT / "data/DL_FIRE_SV-C2_809701" / name for name in (
    "fire_archive_SV-C2_809701.csv", "fire_nrt_SV-C2_809701.csv")]
EXPECTED = [
    "eedc8f13f9c21d41040b8cbff5a5337f0e7e6ac818584c9db413a170c1f138e4",
    "4fdf0e2dfcda3e03b4453b988c5603781c1f64fb8dff09bd6ef702e665ea5ac1",
]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


def endpoint_mask(dates, flagged_dates):
    return pd.to_datetime(dates).isin(pd.to_datetime(flagged_dates))


def transition_flags(origin, target, flagged_dates):
    a, b = endpoint_mask(origin, flagged_dates), endpoint_mask(target, flagged_dates)
    return a, b, a | b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = sorted(set(RAW + [ROOT / "final_test_050_3state.zip", ROOT / "grid.parquet",
        ROOT / "final_artifacts-20261008T144712Z-1-001.zip"]
        + list((ROOT / "src").rglob("*.py")) + list((ROOT / "notebooks").glob("*.ipynb"))
        + list((ROOT / "docs/attachments/final_artifacts").glob("*"))))
    before = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in protected if p.is_file()}
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    inputs = {}

    def pinned(folder, name):
        path = f"outputs/reports/{folder}/{name}"
        content = git_bytes(path)
        (OUT / name).write_bytes(content)
        inputs[path] = hashlib.sha256(content).hexdigest()
        return pd.read_csv(io.BytesIO(content), parse_dates=["date"]) if name.endswith(".csv") and name not in (
            "source_advisory_windows.csv", "frozen_transition_impact.csv") else content

    daily = pinned("firms_observation_coverage", "daily_coverage_audit.csv")
    provider = pinned("final_validity_review", "firms_provider_crosscheck.csv")
    nasa = pinned("final_validity_review", "nasa_missing_dates.csv")
    pinned("final_validity_review", "nasa_missing_dates_response.json")
    old_windows = pd.read_csv(io.BytesIO(pinned("firms_observation_coverage", "source_advisory_windows.csv")))
    old_impact = pd.read_csv(io.BytesIO(pinned("firms_observation_coverage", "frozen_transition_impact.csv")))
    dates = pd.date_range("2016-09-22", "2026-08-31")
    check("complete_unique_daily_calendar", len(daily) == 3631 and daily.date.is_unique and daily.date.tolist() == list(dates))
    check("provider_dates_agree", provider.date.tolist() == list(dates))
    counts = pd.Series(0, index=dates, dtype="int64")
    for raw, expected in zip(RAW, EXPECTED):
        check(f"raw_hash_{raw.name}", sha(raw) == expected)
        for chunk in pd.read_csv(raw, usecols=["acq_date"], chunksize=300000):
            part = pd.to_datetime(chunk.acq_date).value_counts()
            counts = counts.add(part.reindex(dates, fill_value=0), fill_value=0).astype("int64")
    check("all_raw_daily_counts_reproduced", np.array_equal(counts.to_numpy(), daily.raw_records.to_numpy()))
    check("provider_daily_counts_agree", np.array_equal(provider.raw_records, daily.raw_records))
    nasa_dates = {}
    for product, column in [("standard", "nasa_standard_missing"), ("NRT", "nasa_nrt_missing")]:
        nasa_dates[product] = nasa.loc[nasa.satellite.eq("Suomi NPP (VIIRS)") & nasa["product"].eq(product), "date"]
        check(f"NASA_inventory_{product}_matches_provider", np.array_equal(endpoint_mask(dates, nasa_dates[product]), provider[column]))
    check("all_May_2026_standard_missing", provider.loc[provider.date.between("2026-05-01", "2026-05-31"), "nasa_standard_missing"].all())
    check("78_of_84_raw_empty_dates_listed", int(daily.raw_zero_records.sum()) == 84 and int((daily.raw_zero_records & (provider.nasa_standard_missing | provider.nasa_nrt_missing)).sum()) == 78)

    windows = old_windows[["window", "start", "end", "classification", "source_url"]].copy()
    windows["review_note"] = "Date envelope; partial boundaries and archive reprocessing need product QA."
    additions = [
        ("added_2023_november", "2023-11-01", "2023-11-05", "anomaly_and_quality_review", OUTAGES,
         "Not in old windows. Nominal notice Nov 6; activation Nov 2 is not quality clearance."),
        ("added_2025_fairbanks", "2025-07-03", "2025-07-11", "partial_reception_archive_backfill_unknown", ISSUE1178,
         "Minimum documented interval only. Jul 11 is issue update, NOT confirmed recovery; later boundary unresolved."),
        ("added_2025_september", "2025-09-24", "2025-09-24", "NRT_gap_standard_backfill_unknown", OUTAGES,
         "NRT-specific gap cannot establish missing standard archive granules."),
        ("added_2026_july_boundary", "2026-07-10", "2026-07-10", "partial_day_anomaly", OUTAGES,
         "Old focus started Jul 11; anomaly already began late Jul 10."),
    ]
    windows = pd.concat([windows, pd.DataFrame(additions, columns=windows.columns)], ignore_index=True)
    windows.to_csv(OUT / "reviewed_advisory_windows.csv", index=False)
    ledger = daily[["date", "raw_records", "retained_sumatra_records", "raw_zero_records", "sumatra_zero_retained_records"]].merge(
        provider[["date", "nasa_standard_missing", "nasa_nrt_missing"]], on="date", validate="one_to_one")
    ledger["platform"] = "Suomi NPP VIIRS"
    ledger["project_product"] = np.where(ledger.date.le(pd.Timestamp("2026-06-30")), "standard_archive", "NRT")
    ledger["matching_product_missing"] = np.where(ledger.project_product.eq("standard_archive"), ledger.nasa_standard_missing, ledger.nasa_nrt_missing)
    ledger["advisory_ids"] = ""
    ledger["advisory_sources"] = ""
    for row in windows.itertuples():
        hit = ledger.date.between(row.start, row.end)
        ledger.loc[hit, "advisory_ids"] += row.window + "|"
        ledger.loc[hit, "advisory_sources"] += row.source_url + "|"
    advisory = ledger.advisory_ids.ne("")
    ledger["usability"] = "usable"
    ledger["reason"] = "provisional_detection_proxy; export_completeness_assumed_not_verified"
    ledger.loc[advisory | ledger.matching_product_missing, ["usability", "reason"]] = ["unknown", "quality_or_product_backfill_unverified"]
    partial = ledger.advisory_ids.str.contains("2022_boundary|added_2026_july_boundary", regex=True)
    ledger.loc[partial, ["usability", "reason"]] = ["partial", "documented_partial_day_anomaly; local_spatial_effect_unverified"]
    ledger.loc[ledger.raw_zero_records, ["usability", "reason"]] = ["unavailable", "no_records_in_supplied_export; observed_zero_not_supported"]
    ledger["decision_confidence"] = np.where(ledger.raw_zero_records, "high_export_absence_not_fire_truth", "provisional")
    ledger["grid_observation_coverage_verified"] = False
    ledger["zero_interpretation"] = np.where(ledger.usability.eq("usable"), "provisional_no_retained_detection; not_confirmed_observed_zero", "unknown; do_not_certify_zero")
    ledger["review_candidate"] = ledger.usability.ne("usable")
    ledger["provider_evidence"] = "nasa_missing_dates_response.json@" + COMMIT
    ledger.to_csv(OUT / "product_quality_ledger.csv", index=False)
    unmatched = ledger[ledger.raw_zero_records & ~(ledger.nasa_standard_missing | ledger.nasa_nrt_missing)]
    check("all_six_inventory_unlisted_dates_have_advisory", len(unmatched) == 6 and unmatched.advisory_ids.ne("").all())
    unmatched.to_csv(OUT / "inventory_unlisted_dates.csv", index=False)

    with zipfile.ZipFile(ROOT / "final_test_050_3state.zip") as archive:
        pred_bytes = archive.read("final_test_050_3state/test_predictions.parquet")
        pred = pd.read_parquet(io.BytesIO(pred_bytes))
    check("original_prediction_identity", hashlib.sha256(pred_bytes).hexdigest() == "e92134c8f954316d85258415b1fffa33d78aefe2f6320b0616a61f44b1d779f3")
    geometry = pd.read_parquet(ROOT / "grid.parquet", columns=["grid_id"])
    check("actual_geometry_cohort", len(geometry) == 187 and geometry.grid_id.is_unique and set(pred.grid_id).issubset(set(geometry.grid_id)))
    check("original_test_shape_and_unique_keys", len(pred) == 27968 and pred.grid_id.nunique() == 46 and not pred.duplicated(["grid_id", "next_date"]).any())
    check("all_transitions_exact_one_day", (pred.next_date - pred.date).eq(pd.Timedelta(days=1)).all())
    trace = pred[["grid_id", "date", "next_date", "current_state", "actual_next_state"]].copy()
    for endpoint, column in [("origin", "date"), ("target", "next_date")]:
        lookup = ledger.set_index("date")
        for field in ["usability", "reason", "review_candidate", "raw_zero_records", "advisory_ids"]:
            trace[f"{endpoint}_{field}"] = trace[column].map(lookup[field])
    trace["either_endpoint_review_candidate"] = trace.origin_review_candidate | trace.target_review_candidate
    trace["either_endpoint_export_unavailable"] = trace.origin_raw_zero_records | trace.target_raw_zero_records
    trace.to_csv(OUT / "actual_test_transition_trace.csv", index=False)
    check("2392_actual_test_export_gap_transitions", int(trace.either_endpoint_export_unavailable.sum()) == 2392)
    check("2162_gap_test_targets_all_original_zero", int(trace.target_raw_zero_records.sum()) == 2162 and trace.loc[trace.target_raw_zero_records, "actual_next_state"].eq(0).all())
    check("2162_gap_test_origins_all_original_zero", int(trace.origin_raw_zero_records.sum()) == 2162 and trace.loc[trace.origin_raw_zero_records, "current_state"].eq(0).all())
    may = pd.date_range("2026-05-01", "2026-05-31")
    a, b, either = transition_flags(pred.date, pred.next_date, may)
    check("May_actual_endpoint_counts", int(b.sum()) == 1426 and int(either.sum()) == 1472)

    impact = []
    split_ranges = [("train", "2016-09-23", "2022-12-31", 15, 736),
                    ("validation", "2023-01-01", "2024-12-31", 22, 1288),
                    ("test", "2025-01-01", "2026-08-31", 47, 2392)]
    for split, start, end, expected_days, expected_rows in split_ranges:
        target_days = pd.date_range(start, end)
        zero_dates = ledger.loc[ledger.raw_zero_records, "date"]
        a, b, either = transition_flags(target_days - pd.Timedelta(days=1), target_days, zero_dates)
        check(f"{split}_raw_gap_dates", int(b.sum()) == expected_days)
        check(f"{split}_46grid_candidate_arithmetic", int(either.sum()) * 46 == expected_rows)
        old = old_impact[old_impact.flag.eq("raw_zero_records") & old_impact.split.eq(split)].iloc[0]
        check(f"{split}_reported_original_gap_counts_reconcile", old.either_endpoint_flagged == expected_rows and old.target_flagged == expected_days * 46)
        for flag, selected in [("export_unavailable", zero_dates), ("reviewed_quality_union", ledger.loc[ledger.review_candidate, "date"])]:
            a, b, either = transition_flags(target_days - pd.Timedelta(days=1), target_days, selected)
            impact.append(dict(split=split, flag=flag, flagged_target_dates=int(b.sum()),
                calendar_origin_rows=int(a.sum()) * 46, calendar_target_rows=int(b.sum()) * 46,
                calendar_either_rows=int(either.sum()) * 46,
                validation="actual_saved_rows_joined" if split == "test" else "calendar_candidates_only; original_rows_unavailable"))
    pd.DataFrame(impact).to_csv(OUT / "split_transition_impact.csv", index=False)
    check("test_quality_union_trace_matches_calendar", int(trace.either_endpoint_review_candidate.sum()) == impact[-1]["calendar_either_rows"])

    grid_days = pd.concat([
        pred[["grid_id", "date", "current_state"]].rename(columns={"current_state": "historical_state"}),
        pred[["grid_id", "next_date", "actual_next_state"]].rename(columns={"next_date": "date", "actual_next_state": "historical_state"})])
    check("shared_endpoint_states_agree", grid_days.groupby(["grid_id", "date"]).historical_state.nunique().eq(1).all())
    grid_days = grid_days.drop_duplicates(["grid_id", "date"]).merge(ledger[["date", "usability", "zero_interpretation", "grid_observation_coverage_verified"]], on="date", validate="many_to_one")
    grid_days["state_evidence"] = np.select([
        grid_days.usability.ne("usable"), grid_days.historical_state.gt(0)],
        ["quality_review_required; do_not_certify_historical_state", "historical_positive_detection; true_fire_not_certified"],
        default="provisional_no_retained_detection; grid_coverage_unverified")
    grid_days.to_csv(OUT / "actual_test_endpoint_ledger.csv", index=False)
    trace.groupby("grid_id", sort=True).agg(total_rows=("date", "size"),
        unavailable_either=("either_endpoint_export_unavailable", "sum"),
        reviewed_quality_either=("either_endpoint_review_candidate", "sum")).to_csv(OUT / "per_grid_test_impact.csv")
    trace.groupby(["either_endpoint_export_unavailable", "actual_next_state"]).size().rename("rows").to_csv(OUT / "historical_target_support_by_gap_flag.csv")
    examples = grid_days[grid_days.historical_state.eq(0)].groupby("usability", sort=True).head(1)
    examples.to_csv(OUT / "zero_interpretation_examples.csv", index=False)

    for name, origin, target, flagged, expected in [
        ("year_boundary_origin", ["2024-12-31"], ["2025-01-01"], ["2024-12-31"], [True]),
        ("leap_day_target", ["2024-02-28"], ["2024-02-29"], ["2024-02-29"], [True]),
        ("unflagged_pair", ["2025-01-01"], ["2025-01-02"], ["2025-01-03"], [False]),
        ("no_bridging_gap", ["2025-01-01", "2025-01-02"], ["2025-01-02", "2025-01-03"], ["2025-01-02"], [True, True])]:
        check(name, transition_flags(origin, target, flagged)[2].tolist() == expected)
    legitimate_proxy_zero = ledger.loc[ledger.sumatra_zero_retained_records & ~ledger.raw_zero_records & ~ledger.matching_product_missing & ledger.advisory_ids.eq("")]
    check("zero_inside_Sumatra_does_not_auto_exclude", len(legitimate_proxy_zero) > 0 and legitimate_proxy_zero.usability.eq("usable").all())
    check("standard_NRT_boundary_is_explicit", ledger.loc[ledger.date.eq("2026-06-30"), "project_product"].item() == "standard_archive" and ledger.loc[ledger.date.eq("2026-07-01"), "project_product"].item() == "NRT")
    check("quality_status_partition_complete", set(ledger.usability) == {"usable", "unavailable", "partial", "unknown"} and ledger.usability.notna().all())
    after = {name: sha(ROOT / name) for name in before}
    check("all_protected_input_bytes_unchanged", before == after)
    manifest = dict(review_date="2026-10-11", code_commit=COMMIT,
        audit_script_sha256=sha(Path(__file__)),
        official_advisories_checked="2026-10-11; static rules; no automatic web refresh",
        NASA_inventory="Pinned prior verified response, not freshly fetched by this script",
        inputs=inputs, protected_inputs_before=before, protected_inputs_after=after,
        environment=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, pyarrow=pyarrow.__version__),
        checks=checks, checks_passed=sum(c["passed"] for c in checks),
        ledger_status_counts={str(k): int(v) for k, v in ledger.usability.value_counts().items()},
        actual_test_review_candidate_rows=int(trace.either_endpoint_review_candidate.sum()),
        actual_test_unavailable_rows=int(trace.either_endpoint_export_unavailable.sum()),
        limitations=["Usable is conditional proxy usability, not a grid coverage certificate.",
            "Training/validation candidate counts are not original-row or label reproduction.",
            "Fairbanks recovery/backfill boundary is unresolved beyond reviewed minimum interval.",
            "No geometry point attribution/type filtering, correction, new metric evaluation, inference or refit."],
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "verification.json"})
    (OUT / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks_passed": manifest["checks_passed"], "quality_counts": manifest["ledger_status_counts"],
        "test_unavailable_either": manifest["actual_test_unavailable_rows"],
        "test_review_candidate_either": manifest["actual_test_review_candidate_rows"]}, indent=2))


if __name__ == "__main__":
    main()
