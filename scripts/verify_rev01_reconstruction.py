"""Verify saved REV-01 reconstruction and immutable inputs without refitting."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from reconstruct_rev01_lineage import ROOT, OUT, sha


def main():
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    detection = json.loads((OUT / "verification.json").read_text())
    cov = json.loads((OUT / "covariate_verification.json").read_text())
    check("detection_checks_pass", all(x["passed"] for x in detection["checks"]))
    check("covariate_checks_pass", all(x["passed"] for x in cov["checks"]))
    check("detection_script_identity", sha(ROOT / "scripts/reconstruct_rev01_lineage.py") == detection["audit_script_sha256"])
    check("covariate_script_identity", sha(ROOT / "scripts/reconstruct_rev01_covariates.py") == cov["script_sha256"])
    for name, expected in detection["protected_before"].items():
        check(f"protected_{name}", sha(ROOT / name) == expected)
    weather_assets = json.loads((OUT / "local_weather_asset_inventory.json").read_text())["assets"]
    for index, item in enumerate(weather_assets):
        check(f"raw_weather_bytes_{item['path']}", sha(ROOT / item["path"]) == item["sha256"])
        if index % 1000 == 0:
            print(f"Weather byte checks: {index+1}/{len(weather_assets)}", flush=True)
    provenance = json.loads((ROOT / "docs/attachments/final_artifacts/reporting_provenance.json").read_text())
    with zipfile.ZipFile(ROOT / "final_test_050_3state.zip") as archive:
        for path, expected in provenance["source_sha256"].items():
            name = path.rsplit("/", 1)[-1]
            actual = sha(ROOT / "grid.parquet") if name == "grid.parquet" else hashlib.sha256(archive.read("final_test_050_3state/"+name)).hexdigest()
            check(f"original_provenance_{name}", actual == expected)
        frozen = json.loads(archive.read("final_test_050_3state/frozen_design.json"))
    weather = pd.read_parquet(OUT / "reconstructed_covariates.parquet")
    reported = pd.read_csv(ROOT / "docs/attachments/rev02_20261011/daily_coverage_audit.csv", parse_dates=["date"]).set_index("date").predictor_available_grids
    actual = weather.groupby("date").predictors_available.sum().reindex(reported.index)
    check("all_3631_original_daily_availability_counts", np.array_equal(actual, reported))
    unreadable = json.loads((OUT / "unreadable_CHIRPS.json").read_text())
    for item in unreadable:
        check(f"unreadable_rain_stays_missing_{item['date']}", not weather.loc[weather.date.eq(item["date"]), "predictors_available"].any())
    trans = pd.read_parquet(OUT / "reconstructed_full_transitions.parquet")
    check("unique_transition_grid_target_keys", not trans.duplicated(["grid_id", "next_date"]).any())
    check("exact_daily_pairs", (trans.next_date-trans.date).eq(pd.Timedelta(days=1)).all())
    check("train_target_boundary", trans.loc[trans.split.eq("train"), "next_date"].max() == pd.Timestamp("2022-12-31"))
    check("validation_target_boundary", trans.loc[trans.split.eq("validation"), "next_date"].min() == pd.Timestamp("2023-01-01") and trans.loc[trans.split.eq("validation"), "next_date"].max() == pd.Timestamp("2024-12-31"))
    check("test_target_boundary", trans.loc[trans.split.eq("test"), "next_date"].min() == pd.Timestamp("2025-01-01"))
    eligible = pd.read_csv(OUT / "reconstructed_grid_eligibility.csv")
    check("original_eligible_grid_ids", sorted(eligible.loc[eligible.eligible, "grid_id"]) == sorted(frozen["eligible_grid_ids"]))
    ledger = pd.read_csv(ROOT / "docs/attachments/rev02_20261011/product_quality_ledger.csv", parse_dates=["date"])
    gaps = ledger.loc[ledger.raw_zero_records, "date"]
    impacts = []
    for split, expected in [("train", 736), ("validation", 1288), ("test", 2392)]:
        frame = trans[trans.historical_frozen_grid & trans.predictors_available & trans.split.eq(split)]
        origin, target = frame.date.isin(gaps), frame.next_date.isin(gaps)
        check(f"reconstructed_usable_{split}_gap_impact", int((origin|target).sum()) == expected)
        check(f"reconstructed_usable_{split}_gap_states", frame.loc[origin, "current_state"].eq(0).all() and frame.loc[target, "next_state"].eq(0).all())
        impacts.append(dict(split=split, total_rows=len(frame), gap_origins=int(origin.sum()), gap_targets=int(target.sum()),
            either_endpoint=int((origin|target).sum()), evidence="independent_reconstructed_full_predictor_mask; not_original_bytes"))
    pd.DataFrame(impacts).to_csv(OUT / "verified_reconstructed_split_gap_impact.csv", index=False)
    manifest = dict(date="2026-10-11", verification_script_sha256=sha(Path(__file__)),
        verification_checks_passed=len(checks), detection_checks_passed=len(detection["checks"]),
        covariate_checks_passed=len(cov["checks"]), checks=checks,
        reconstructed_not_original=True, original_model_environment_recovered=False,
        immutable_original_protected_files=len(detection["protected_before"]), local_weather_files_hashed=len(weather_assets),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "handoff_verification.json"})
    (OUT / "handoff_verification.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("checks", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
