"""Post-hoc, same-row probability comparison with training transition rates."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/revision_review/REV03_20261011_ec3d83da299e"
EVIDENCE = ROOT / "docs/attachments/rev08_20261011"
PLAN = dict(revision="REV-08", decision="DEC-21", source_run=SOURCE.name,
            baseline="Per-grid/current-state destination counts divided by training group size",
            training="Included corrected transitions with target date <= 2022-12-31; historical 46 candidates retained by REV-03",
            zero_count="Unobserved destination has probability zero; no smoothing or clipping; impossible actual yields infinite log loss",
            zero_denominator="Fail explicitly; no pooled fallback or test-derived rate",
            primary_scores=["log_loss", "multiclass_brier_sum_over_three_classes"],
            delta="Covariate minus baseline; negative is better for covariate",
            test_periods={"test_all": ["2025-01-01", "2026-08-31"], "test_2025": ["2025-01-01", "2025-12-31"], "test_2026_JanAug": ["2026-01-01", "2026-08-31"]},
            validation_period=["2023-01-01", "2024-12-31"],
            selection="No tuning, model replacement, new eligibility or threshold selection; test already viewed, descriptive post-hoc comparison",
            uncertainty="No significance interval or drift/causal/generalization claim from annual score differences")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rates_from_training(training):
    if training.empty or training.next_date.gt(pd.Timestamp("2022-12-31")).any():
        raise ValueError("Baseline requires nonempty training-only target dates")
    if not training.current_state.isin([0, 1, 2]).all() or not training.next_state.isin([0, 1, 2]).all():
        raise ValueError("Invalid training class")
    counts = training.groupby(["grid_id", "current_state", "next_state"], observed=True).size().unstack("next_state", fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    expected = pd.MultiIndex.from_product([sorted(training.grid_id.unique()), range(3)], names=["grid_id", "current_state"])
    counts = counts.reindex(expected, fill_value=0)
    totals = counts.sum(axis=1)
    if totals.le(0).any():
        raise ValueError("Zero-denominator grid/current-state group; no fallback")
    result = counts.rename(columns={j: f"n_{j}" for j in range(3)}).reset_index()
    result["training_rows"] = totals.to_numpy()
    for j in range(3):
        result[f"baseline_p_{j}"] = result[f"n_{j}"] / result.training_rows
    return result


def attach_rates(predictions, rates):
    if predictions.duplicated(["grid_id", "next_date"]).any():
        raise ValueError("Duplicate evaluation keys")
    if rates.duplicated(["grid_id", "current_state"]).any():
        raise ValueError("Duplicate baseline group")
    result = predictions.merge(rates, on=["grid_id", "current_state"], how="left", validate="many_to_one", sort=False)
    if len(result) != len(predictions) or result.training_rows.isna().any():
        raise ValueError("Missing training group for evaluation row; no fallback")
    return result


def proper_scores(y, p, logs=None):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not len(y) or p.shape != (len(y), 3) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Invalid score shape/classes")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("Invalid probabilities")
    if logs is None:
        logs = np.log(p, out=np.full_like(p, -np.inf), where=p > 0)
    logs = np.asarray(logs, dtype=float)
    if logs.shape != p.shape or np.isnan(logs).any() or np.isposinf(logs).any() or not np.allclose(np.exp(logs), p, atol=1e-12, rtol=1e-10):
        raise ValueError("Invalid log probabilities")
    actual_logs = logs[np.arange(len(y)), y]
    loss = float(-actual_logs.mean())
    return dict(n=len(y), log_loss=loss if np.isfinite(loss) else None,
                log_loss_infinite=not np.isfinite(loss), zero_probability_actual=int(np.isneginf(actual_logs).sum()),
                brier=float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
                **{f"support_{j}": int((y == j).sum()) for j in range(3)})


def compare(frame, period, grid="ALL"):
    y = frame.actual_next_state.to_numpy(dtype=int)
    baseline = proper_scores(y, frame[[f"baseline_p_{j}" for j in range(3)]])
    covariate = proper_scores(y, frame[[f"p_{j}" for j in range(3)]], frame[[f"log_p_{j}" for j in range(3)]])
    difference = None if baseline["log_loss_infinite"] or covariate["log_loss_infinite"] else covariate["log_loss"] - baseline["log_loss"]
    row = dict(period=period, grid_id=grid, n=len(frame), target_first=frame.next_date.min().date().isoformat(), target_last=frame.next_date.max().date().isoformat(),
               baseline_log_loss=baseline["log_loss"], covariate_log_loss=covariate["log_loss"], delta_log_loss=difference,
               baseline_log_loss_infinite=baseline["log_loss_infinite"], covariate_log_loss_infinite=covariate["log_loss_infinite"],
               baseline_zero_probability_actual=baseline["zero_probability_actual"], covariate_zero_probability_actual=covariate["zero_probability_actual"],
               baseline_brier=baseline["brier"], covariate_brier=covariate["brier"], delta_brier=covariate["brier"] - baseline["brier"],
               **{f"support_{j}": baseline[f"support_{j}"] for j in range(3)})
    return row


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    source_manifest = json.loads((ROOT / "docs/attachments/rev07_20261011/portable_inputs.json").read_text())
    protected = {name: meta["sha256"] for name, meta in source_manifest["profiles"]["revised-run"]["files"].items()}
    for name, expected in protected.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Protected input changed/missing: {name}")
    inputs = {name: sha(SOURCE / name) for name in ("transition_ledger.parquet", "validation_predictions.parquet", "test_predictions.parquet", "summary.json")}
    identity = hashlib.sha256(json.dumps(dict(plan=PLAN, inputs=inputs, script=sha(__file__)), sort_keys=True).encode()).hexdigest()[:12]
    out = ROOT / "outputs/revision_review" / f"REV08_20261011_{identity}"
    if EVIDENCE.exists():
        raise FileExistsError("Preserve prior REV-08 evidence; use a separately authorized revision")
    out.mkdir(parents=True, exist_ok=False)
    # Persist the policy before loading training outcomes or evaluating saved predictions.
    write_json(out / "analysis_plan.json", dict(plan=PLAN, inputs=inputs, script_sha256=sha(__file__)))
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_rev08_baseline.py", "-v"], cwd=ROOT, capture_output=True, text=True)
    (out / "regression_tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
    if test.returncode:
        raise RuntimeError(test.stdout + test.stderr)
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(dict(name=name, passed=True))

    cols = ["grid_id", "date", "next_date", "current_state", "next_state", "included", "split"]
    transitions = pd.read_parquet(SOURCE / "transition_ledger.parquet", columns=cols)
    training = transitions[transitions.included & transitions.split.eq("train")].copy()
    check("verified_training_count", len(training) == 104144 and training.grid_id.nunique() == 46)
    check("training_exact_calendar", training.next_date.eq(training.date + pd.Timedelta(days=1)).all())
    rates = rates_from_training(training)
    check("138_fixed_groups", len(rates) == 138)
    check("training_counts_reconcile", int(rates.training_rows.sum()) == len(training))
    rates.to_csv(out / "training_transition_rates.csv", index=False)
    overall, pergrid = [], []
    for split, n in (("validation", 31096), ("test", 23690)):
        saved = pd.read_parquet(SOURCE / f"{split}_predictions.parquet")
        frame = attach_rates(saved, rates)
        truth = transitions[transitions.included & transitions.split.eq(split)].sort_values(["grid_id", "next_date"]).reset_index(drop=True)
        check(f"{split}_same_row_identity", len(frame) == n and frame[["grid_id", "date", "next_date"]].equals(truth[["grid_id", "date", "next_date"]]))
        check(f"{split}_same_labels", np.array_equal(frame.actual_next_state, truth.next_state.astype(int)) and np.array_equal(frame.current_state, truth.current_state.astype(int)))
        check(f"{split}_covariate_probabilities_unchanged", frame[[f"p_{j}" for j in range(3)]].equals(saved[[f"p_{j}" for j in range(3)]]))
        frame.to_parquet(out / f"{split}_paired_predictions.parquet", index=False)
        periods = {"validation_all": PLAN["validation_period"]} if split == "validation" else PLAN["test_periods"]
        for period, (first, last) in periods.items():
            sub = frame[frame.next_date.between(first, last)]
            check(f"{period}_nonempty_cohort", not sub.empty and sub.grid_id.nunique() == 46)
            overall.append(compare(sub, period))
            pergrid.extend(compare(g, period, gid) for gid, g in sub.groupby("grid_id"))
    for metric in ("baseline_log_loss", "covariate_log_loss", "baseline_brier", "covariate_brier"):
        whole = next(r for r in overall if r["period"] == "test_all")
        years = [r for r in overall if r["period"] in ("test_2025", "test_2026_JanAug")]
        check(f"annual_weighted_{metric}", whole[metric] is None and any(r[metric] is None for r in years) or whole[metric] is not None and all(r[metric] is not None for r in years) and abs(sum(r[metric] * r["n"] for r in years) / whole["n"] - whole[metric]) < 1e-12)
    pd.DataFrame(overall).to_csv(out / "overall_comparison.csv", index=False)
    pd.DataFrame(pergrid).to_csv(out / "per_grid_comparison.csv", index=False)
    for name, expected in protected.items():
        check(f"protected:{name}", sha(ROOT / name) == expected)
    summary = dict(run_id=out.name, source_run=SOURCE.name, plan=PLAN, training_rows=len(training), training_groups=len(rates),
                   overall=overall, finite_score_convention="JSON null plus explicit infinite flag for impossible actual; no score clipping",
                   limitations=["Descriptive post-hoc same-row comparison; no untouched test or significance inference", "Unequal 2025/full versus 2026/Jan-Aug periods do not prove drift", "Conservative detected-proxy coverage, selected-cohort and original-run-attestation limits persist", "No covariate refit/inference, tuning, model replacement or baseline smoothing"])
    write_json(out / "summary.json", summary)
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    for name in ("analysis_plan.json", "training_transition_rates.csv", "overall_comparison.csv", "per_grid_comparison.csv", "summary.json", "regression_tests.txt"):
        (EVIDENCE / name).write_bytes((out / name).read_bytes())
    columns = ["period", "n", "baseline_log_loss", "covariate_log_loss", "delta_log_loss", "baseline_brier", "covariate_brier", "delta_brier"]
    lines = []
    for row in overall:
        cells = [str(row[c]) if c in ("period", "n") else f"{row[c]:.8f}" if row[c] is not None else "infinite/undefined (see flags)" for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    report = "\n".join(["# REV-08 — pembanding probabilitas transisi sederhana", "", f"11 Oktober 2026; run `{out.name}`, sumber `{SOURCE.name}`; DEC-21. Deskriptif post-hoc, bukan test konfirmatori baru.", "",
        "Baseline adalah frekuensi destination training per grid/current-state, tanpa kovariat/smoothing. Hanya 104.144 transisi corrected training (target sampai 2022-12-31); 138 group pada 46 grid. Destination tidak teramati mendapat p=0 dan log loss infinite bila menjadi actual; denominator nol/missing group menghentikan proses, tanpa fallback. Kebijakan ditulis sebelum membaca training/prediksi untuk skor. Tidak ada test-label influence pada rates, tuning, pooling, threshold/eligibility change atau refit model kovariat.", "",
        "Setiap perbandingan memakai grid/date/current-state/actual dan baris yang sama dengan REV-03, serta probabilitas kovariat tersimpan tanpa inference ulang. Validation 2023–2024; test_all 2025-01-01–2026-08-31; test_2025 setahun penuh; test_2026_JanAug hanya Januari–Agustus. Dukungan kelas, impossible-actual flags dan per-grid hasil ada di CSV. Brier adalah jumlah tiga squared errors yang dirata-ratakan antarbaris. Delta = kovariat minus baseline; negatif berarti skor kovariat lebih baik, positif berarti baseline lebih baik.", "",
        "| Periode | N | LL baseline | LL kovariat | Delta LL | Brier baseline | Brier kovariat | Delta Brier |", "|---|---:|---:|---:|---:|---:|---:|---:|", *lines, "",
        "Sumber tabel: overall_comparison.csv dalam direktori yang sama. Probabilitas pasangan disimpan pada output run; training_transition_rates.csv menelusuri setiap denominator dan count destination. Rerata tahunan dibobot jumlah baris dan direkonsiliasi ke test_all. Jangan membaca selisih tahunan sebagai bukti drift: panjang periode, musim, missingness dan class mix berbeda. Selisih point score bukan signifikansi statistik, kalibrasi atau manfaat kausal kovariat. Hasil hanya berlaku pada subset proxy/cohort terpilih; keterbatasan observasi/original-run tetap ada.", "",
        "[Hasil penelitian sebelumnya](../../FINAL_RESULTS.md) dipertahankan sebagai snapshot sebelum pembanding ini, termasuk pernyataan bahwa baseline belum dievaluasi pada saat REV-05. Gunakan addendum REV-08 ini dan keputusan terbaru untuk kesimpulan komparatif, bukan mengganti sumber/metrik historis. Tidak ada klaim operational/patroli/general superiority atau model replacement otomatis.", ""])
    (EVIDENCE / "REV08_REPORT.md").write_text(report, encoding="utf-8")
    import pyarrow
    record = dict(date="2026-10-11", run_path=out.relative_to(ROOT).as_posix(), checks=checks, checks_passed=len(checks),
                  inputs=inputs, script_sha256=sha(__file__), test_sha256=sha(ROOT / "tests/test_rev08_baseline.py"),
                  protected=protected, outputs={p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()},
                  evidence={p.name: sha(p) for p in EVIDENCE.iterdir() if p.is_file()},
                  environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, pyarrow=pyarrow.__version__))
    write_json(EVIDENCE / "verification.json", record)
    write_json(out / "run_completed.json", dict(verification_sha256=sha(EVIDENCE / "verification.json"), summary_sha256=sha(out / "summary.json")))
    print(json.dumps(dict(run=out.name, checks=len(checks), overall=overall), indent=2))


if __name__ == "__main__":
    main()
