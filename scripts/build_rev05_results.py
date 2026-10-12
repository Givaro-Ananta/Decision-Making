"""Verify frozen revised results and publish a traceable research-results handoff.

No fitting, inference, data correction, calibration or baseline selection.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/revision_review/REV03_20261011_ec3d83da299e"
OUT = ROOT / "docs/attachments/rev05_20261011"
COMMIT = "0745186a2435c693a990fa394031f0c036c24b60"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score(frame):
    y = frame.actual_next_state.to_numpy(dtype=int)
    p = frame[[f"p_{i}" for i in range(3)]].to_numpy()
    logs = frame[[f"log_p_{i}" for i in range(3)]].to_numpy()
    if not len(y) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Invalid target class or empty group")
    if not np.isfinite(p).all() or (p < 0).any() or np.max(np.abs(p.sum(axis=1) - 1)) > 1e-12:
        raise ValueError("Invalid class probabilities")
    if np.isnan(logs).any() or np.isposinf(logs).any() or not np.allclose(np.exp(logs), p, atol=1e-12, rtol=1e-10):
        raise ValueError("Invalid class log-probabilities")
    pred = p.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (y, pred), 1)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    tp = matrix.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(3), where=(support + predicted) > 0)
    loss = -logs[np.arange(len(y)), y].mean()
    result = dict(n=len(y), accuracy=float(tp.sum() / len(y)), balanced_accuracy=float(recall[support > 0].mean()),
                  macro_f1=float(f1.mean()), weighted_f1=float(np.dot(f1, support) / len(y)),
                  log_loss=float(loss) if np.isfinite(loss) else None, log_loss_infinite=not np.isfinite(loss),
                  brier=float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
                  actual_probability_zero=int(np.isneginf(logs[np.arange(len(y)), y]).sum()))
    classes = []
    for state in range(3):
        positive = y == state
        npos, nneg = int(positive.sum()), int((~positive).sum())
        # Rank-sum AUC independently checks the sklearn-generated source, including ties.
        auc = float((rankdata(p[:, state], method="average")[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None
        classes.append(dict(state=state, support=int(support[state]), precision=float(precision[state]),
                            recall=float(recall[state]), f1=float(f1[state]), auc=auc))
    return result, classes, matrix


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(dict(name=name, passed=True))

    v = json.loads((RUN / "verification.json").read_text())
    completed = json.loads((RUN / "run_completed.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    check("completed_run_manifest", sha(RUN / "verification.json") == completed["verification_sha256"])
    check("completed_summary", sha(RUN / "summary.json") == completed["summary_sha256"])
    for name, expected in v["outputs"].items():
        check(f"rev03_output:{name}", sha(RUN / name) == expected)
    for name, expected in v["protected_after"].items():
        check(f"pre_edit_protected:{name}", sha(ROOT / name) == expected)
    run_before = {p.relative_to(RUN).as_posix(): sha(p) for p in RUN.rglob("*") if p.is_file()}
    inputs = {name: sha(RUN / name) for name in ("summary.json", "primary_metrics.csv", "primary_class_metrics.csv", "primary_confusion_matrices.csv",
               "prediction_coverage.csv", "validation_predictions.parquet", "test_predictions.parquet", "split_reconciliation.csv", "verification.json", "run_completed.json")}
    out_sources = {}
    for path in ("README.md", "docs/final_validity_review.md"):
        raw = subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)
        out_sources[path] = dict(sha256=hashlib.sha256(raw).hexdigest(), commit=COMMIT)
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_rev03_correction.py", "-v"], cwd=ROOT, capture_output=True, text=True)
    check("unchanged_revision_regressions_pass", test.returncode == 0 and "Ran 12 tests" in test.stderr)
    table = pd.read_csv(RUN / "primary_metrics.csv")
    state_table = pd.read_csv(RUN / "primary_class_metrics.csv")
    matrix_table = pd.read_csv(RUN / "primary_confusion_matrices.csv")
    coverage = pd.read_csv(RUN / "prediction_coverage.csv")
    overall, classrows, matrices = [], [], []
    for split in ("validation", "test"):
        frame = pd.read_parquet(RUN / f"{split}_predictions.parquet")
        check(f"{split}_unique_keys", not frame.duplicated(["grid_id", "next_date"]).any())
        check(f"{split}_calendar", frame.next_date.eq(frame.date + pd.Timedelta(days=1)).all())
        check(f"{split}_run_identity", frame.run_id.eq(summary["run_id"]).all())
        check(f"{split}_46_grids", frame.grid_id.nunique() == 46)
        period = ("2023-01-01", "2024-12-31") if split == "validation" else ("2025-01-01", "2026-08-31")
        check(f"{split}_target_period", frame.next_date.between(*period).all())
        groups = [("overall", "ALL", -1, frame)]
        groups += [("per_grid", gid, -1, f) for gid, f in frame.groupby("grid_id")]
        groups += [("per_current_state", "ALL", int(s), f) for s, f in frame.groupby("current_state")]
        for scope, gid, state, group in groups:
            meta = dict(split=split, scope=scope, grid_id=gid, current_state=state)
            computed, classes, matrix = score(group)
            mask = table.split.eq(split) & table.scope.eq(scope) & table.grid_id.eq(gid) & table.current_state.eq(state)
            check(f"metric_unique:{split}:{scope}:{gid}:{state}", int(mask.sum()) == 1)
            stored = table[mask].iloc[0]
            for name, value in computed.items():
                check(f"metric:{split}:{scope}:{gid}:{state}:{name}", pd.isna(stored[name]) if value is None else abs(float(stored[name]) - float(value)) <= 1e-10)
            st = state_table[state_table.split.eq(split) & state_table.scope.eq(scope) & state_table.grid_id.eq(gid) & state_table.current_state.eq(state)]
            check(f"three_classes:{split}:{scope}:{gid}:{state}", len(st) == 3 and set(st.state) == {0, 1, 2})
            for c in classes:
                row = st[st.state.eq(c["state"])].iloc[0]
                for name, value in c.items():
                    check(f"class:{split}:{scope}:{gid}:{state}:{c['state']}:{name}", pd.isna(row[name]) if value is None else abs(float(row[name]) - value) <= 1e-10)
            mt = matrix_table[matrix_table.split.eq(split) & matrix_table.scope.eq(scope) & matrix_table.grid_id.eq(gid) & matrix_table.current_state.eq(state)]
            check(f"matrix:{split}:{scope}:{gid}:{state}", len(mt) == 9 and np.array_equal(mt.sort_values(["actual_state", "predicted_state"])["count"].to_numpy().reshape(3, 3), matrix))
            if scope == "overall":
                overall.append(dict(split=split, source_run=summary["run_id"], **computed))
                classrows.extend(dict(split=split, source_run=summary["run_id"], **c) for c in classes)
                matrices.extend(dict(split=split, actual_state=a, predicted_state=b, count=int(matrix[a, b])) for a in range(3) for b in range(3))
            if scope == "per_grid":
                check(f"coverage:{split}:{gid}", int(coverage[coverage.split.eq(split) & coverage.grid_id.eq(gid)].rows.sum()) == len(group))
        check(f"scope_rows:{split}", len(table[table.split.eq(split)]) == len(groups))
    with zipfile.ZipFile(ROOT / "final_test_050_3state.zip") as archive:
        historic = pd.read_parquet(io.BytesIO(archive.read("final_test_050_3state/test_predictions.parquet"))).rename(columns={"p_no_activity": "p_0", "p_nonhigh": "p_1", "p_high": "p_2"})
    historical, _, _ = score(historic)
    for key, value in summary["historical_test"].items():
        check(f"historical_context:{key}", value is None and historical[key] is None or value is not None and abs(float(historical[key]) - float(value)) <= 1e-10)
    split_counts = pd.read_csv(RUN / "split_reconciliation.csv")
    for row in split_counts.itertuples():
        check(f"row_waterfall:{row.split}", row.historical_predictor_available == row.observation_excluded + row.training_screen_excluded + row.revised_included)
    check("fixed_threshold_cohort", summary["boundary_frp_mw"] == 16.84 and summary["retained_grids"] == 46 and summary["models"] == 138)
    manuscript_files = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in (".docx", ".pptx", ".pdf") and p.relative_to(ROOT).parts[0] not in (".git", ".venv", "data")]
    before_readme = (ROOT / "README.md").read_bytes()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "README_before.txt").write_bytes(before_readme)
    (OUT / "regression_tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
    pd.DataFrame(overall).to_csv(OUT / "overall_results.csv", index=False)
    pd.DataFrame(classrows).to_csv(OUT / "state_results.csv", index=False)
    pd.DataFrame(matrices).to_csv(OUT / "confusion_matrices.csv", index=False)
    split_counts.to_csv(OUT / "split_accounting.csv", index=False)
    write_json(OUT / "reviewed_document_sources.json", dict(pinned_sources=out_sources, available_manuscript_slide_files=manuscript_files,
               findings=["Local README current notice predates REV-03; updated in REV-05 with old bytes archived",
                         "Pinned validity report describes pre-REV-03 no-correction status and older 1196/2254/3680 quality candidates; superseded by DEC-18 and revised source",
                         "Pinned recovery-before-experiment recommendation is not mandatory under DEC-16/DEC-18 conservative policy",
                         "Historical figures remain old 27968-row results, not revised performance graphics"],
               limits="No full manuscript/slides supplied in the searched workspace or pinned Git tree; no claim of review outside those locations"))
    render_report(summary, overall, classrows, matrices, split_counts, historical)
    old_text = before_readme.decode("utf-8")
    anchor = "Research project for **Decision Making"
    body = old_text[old_text.index(anchor):]
    notice = """# Decision Making — Retrospective Hotspot State Transitions (ITERA)

> **Current status — 11 October 2026:** Active scope is retrospective detected-state transition research (DEC-12), without patrol allocation. REV-01 input reconstruction, REV-02 quality audit and conservative REV-03 correction/refit are verified. The revised analysis retains 46 grids and 138 models on 23,690 test transitions; it is post-hoc and conditional on provisional detected-export usability, not validated observation coverage or operational forecasting. Read the [current results and methods](docs/FINAL_RESULTS.md), [progress](docs/PROGRESS.md), [audit](docs/FINAL_AUDIT.md), [plan](docs/REVISION_PLAN.md), [context](docs/PROJECT_CONTEXT.md), and [decisions](docs/DECISIONS.md). Full manuscript/slides remain unreviewed; historical figures below are not revised results. Heavy inputs/models are local and ignored by Git. Local HEAD remains `66bf5a8`, cached final commit `0745186`; no merge/push is implied. [AGENTS.md](AGENTS.md) defines maintenance instructions.

## Historical local proposal — superseded by DEC-12

The original proposal/status text below is retained as dated evidence, including obsolete patrol goals, missing-source statements and old results. It is not the current project specification. The exact pre-REV-05 README bytes are archived in `docs/attachments/rev05_20261011/README_before.txt`.

"""
    (ROOT / "README.md").write_bytes((notice + body).encode("utf-8"))
    check("historical_readme_body_preserved", (ROOT / "README.md").read_bytes().split(anchor.encode(), 1)[1] == before_readme.split(anchor.encode(), 1)[1])
    for name, expected in v["protected_after"].items():
        target = OUT / "README_before.txt" if name == "README.md" else ROOT / name
        check(f"post_edit_protected:{name}", sha(target) == expected)
    check("all_rev03_bytes_unchanged", run_before == {p.relative_to(RUN).as_posix(): sha(p) for p in RUN.rglob("*") if p.is_file()})
    write_json(OUT / "verification.json", dict(date="2026-10-11", stage="REV-05", status="available-result/document QA complete; unseen deliverable sign-off pending",
               source_run=summary["run_id"], inputs=inputs, checks=checks, checks_passed=len(checks), script_sha256=sha(__file__),
               reviewed_sources=out_sources, manuscript_slide_files=manuscript_files, regression_tests_passed=12,
               run_files_unchanged=len(run_before), protected_original_snapshot_files=len(v["protected_after"]),
               authorized_protected_file_change={"README.md": dict(before_sha256=sha(OUT / "README_before.txt"), after_sha256=sha(ROOT / "README.md"), archive="README_before.txt", reason="Update current notice/title; preserve historical body")},
               report_sha256=sha(ROOT / "docs/FINAL_RESULTS.md"), outputs={p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
               environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
               limitations=["No original run/model/environment attestation", "No physical coverage certificate or recovered backfill", "No manuscript/slides reviewed", "No new figures, baseline study, calibration or confirmatory test", "No refit/inference/dependency install/commit/push"]))
    print(json.dumps(dict(checks_passed=len(checks), report="docs/FINAL_RESULTS.md", evidence=OUT.relative_to(ROOT).as_posix(), manuscript_slide_files=manuscript_files), indent=2))


def render_report(summary, overall, classes, matrices, splits, historical):
    def fmt(value):
        return f"{value:.8f}"

    result_table = "\n".join(f"| {r['split']} | {r['n']:,} | {fmt(r['accuracy'])} | {fmt(r['balanced_accuracy'])} | {fmt(r['macro_f1'])} | {fmt(r['weighted_f1'])} | {fmt(r['log_loss'])} | {fmt(r['brier'])} |" for r in overall)
    class_table = "\n".join(f"| {r['state']} | {r['support']:,} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['auc'])} |" for r in classes if r["split"] == "test")
    matrix = np.array([r["count"] for r in matrices if r["split"] == "test"]).reshape(3, 3)
    matrix_rows = "\n".join(f"| {s} | {matrix[s,0]:,} | {matrix[s,1]:,} | {matrix[s,2]:,} | {matrix[s].sum():,} |" for s in range(3))
    rows = "\n".join(f"| {r.split} | {r.historical_predictor_available:,} | {r.observation_excluded:,} | {r.training_screen_excluded:,} | {r.revised_included:,} |" for r in splits.itertuples())
    report = fr"""# Hasil penelitian retrospektif — revisi konservatif REV-03

Disusun dan diverifikasi pada 11 Oktober 2026 melalui REV-05. Dokumen ini adalah laporan metode/hasil yang dapat ditelusuri, bukan klaim bahwa manuskrip jurnal atau slide akhir tim telah direview. Identitas hasil utama: `{summary['run_id']}`; kebijakan `REV03-conservative-proxy-v1` (DEC-18). Angka historis dipisahkan dan tidak menggantikan bukti asli.

## Pertanyaan dan ruang lingkup

Penelitian mengestimasi dan mengevaluasi secara retrospektif `P(S[g,t+1] | S[g,t], X[g,t])` untuk state intensitas **deteksi termal yang dipertahankan**, pada grid eligible di daratan utama Sumatra. Kontribusi yang diuji adalah penerapan dan evaluasi model transisi bergantung kovariat pada kohort terpilih. Belum ada bukti bahwa kovariat mengungguli model transisi homogen, model ini paling baik, atau temuan merupakan novelty pertama.

Kovariat historis dapat tersedia setelah tanggal representasinya; ini bukan kontrak prakiraan operasional dari informasi yang tersedia saat t. Tidak ada klaim kausal, manfaat patroli/Top-K, cakupan seluruh Sumatra, atau tingkat kerusakan kebakaran.

## Metode, definisi target, dan keputusan observasi

- Dukungan studi: 187 grid reguler 0,5°, origin barat 94,95° dan utara 6,05°, irisan daratan utama Sumatra dan extent sumber. Periode panel 22 September 2016–31 Agustus 2026. Kohort akhir 46 grid yang sudah menjadi kandidat historis; seluruhnya lolos ulang kriteria training, tanpa ekspansi ke grid baru. 46/187 adalah proporsi jumlah grid, bukan luas area atau representativitas geografis.
- State 0: tidak ada deteksi yang dipertahankan **pada hari yang secara provisional dinilai layak untuk proxy ekspor**; bukan bukti tidak ada api atau cakupan pengamatan lengkap. State 1: aktif dan FRP sum <16,84 MW; state 2: aktif dan FRP sum ≥16,84 MW. High adalah intensitas FRP deteksi, bukan severity/kerusakan atau energi harian terintegrasi. Jenis sumber termal pada geometri akhir belum seluruhnya tervalidasi; jangan menyamakannya otomatis dengan semua kejadian karhutla.
- Q75 dihitung ulang pada 41.254 hari-grid aktif training yang layak, di seluruh 187 grid sebelum filter prediktor/kohort; hasil tetap 16,84 MW. Populasi historis 41.545. Ini hasil perhitungan ulang, bukan pembekuan threshold demi mempertahankan skor.
- Ledger REV-02 mencakup 3.631 tanggal: 3.472 provisionally usable, 84 unavailable, 73 unknown, dua partial. Kebijakan konservatif memask count/FRP/state pada semua tanggal non-usable menjadi NULL. Kolom historis dipertahankan terpisah. Exclusion bukan bukti setiap baris tersebut salah; sebagian produk dapat valid atau sudah di-backfill.
- Kalender lengkap di-shift sebelum penyaringan. Transisi hanya dipakai bila grid sama, jarak tepat satu hari, kedua endpoint layak dan prediktor **origin t** tersedia. Tidak ada bridging tanggal, kelas keempat atau imputasi deteksi. Ketidaktersediaan prediktor target t+1 tidak menjadi syarat karena tidak dipakai model.
- Empat prediktor: maksimum suhu dari empat sampel ERA5-Land enam-jam (bukan maksimum hourly penuh), mean kelembapan relatif dari pasangan suhu/dewpoint, mean kecepatan angin, dan curah hujan harian CHIRPS. Tanggal ERA5 berasal dari valid time UTC; label harian CHIRPS berasal dari filename. Informasi produk harian/pentadal dan waktu rilis membatasi interpretasi prospektif. WorldCover merupakan konteks preprocessing, bukan prediktor final.
- Split berdasarkan **tanggal target**: training sampai 31 Desember 2022; validation 1 Januari 2023–31 Desember 2024; test 1 Januari 2025–31 Agustus 2026. Hari origin 31 Desember dapat masuk split berikutnya melalui target 1 Januari. Input direkonstruksi secara independen pada REV-01, bukan file intermediate asli yang berhasil dipulihkan.
- Model: 138 multinomial logistic reference-category independen, satu per grid × current state; unweighted, unpenalized. Screening, standardisasi dan seleksi fitur konstan/near-constant hanya memakai training, dengan fungsi pinned `0745186`. Minimum 50 transisi per current state, minimum 10 per destination yang teramati untuk regresi, ≥5 observasi per parameter bebas, full rank, VIF ≤10, condition number ≤1e8, dan tanpa complete/quasi separation; group single-destination dapat deterministik. Semua fit revisi memenuhi kriteria optimizer, gradient dan information matrix semula. Lolos screen bukan bukti stabilitas statistik.
- Training berubah sehingga diperlukan same-family refit (DEC-14/DEC-18); bukan koreksi arsitektur. Primary tetap argmax dengan tie pada kelas pertama. Tidak ada tuning model/threshold keputusan, weighting, regularisasi, recalibration atau pemilihan berdasarkan test. Periode test sudah pernah diperiksa: evaluasi ini **post-hoc**, bukan untouched confirmatory test.

Sumber metode: [REV-03 receipt](attachments/rev03_20261011/independent_verification.json), [rekonstruksi REV-01](attachments/rev01_20261011/REV01_REPORT.md), [ledger/QA REV-02](attachments/rev02_20261011/REV02_REPORT.md), [keputusan](DECISIONS.md). Bukti NASA berasal dari audit terdahulu; tidak diunduh ulang untuk REV-05.

## Rekonsiliasi baris

Sumber: run `{summary['run_id']}`, `split_reconciliation.csv`; salinan terverifikasi [split_accounting.csv](attachments/rev05_20261011/split_accounting.csv). Split mengikuti periode target di atas. Kolom awal sudah mengeluarkan origin predictor yang tidak tersedia.

| Split | Historis, prediktor tersedia | Exclusion observasi konservatif | Drop screening grid | Revisi dipakai |
|---|---:|---:|---:|---:|
{rows}

Kalender kandidat historis training memiliki 105.386 transisi; 46 origin tanpa rainfall sudah tidak dipakai pada hasil historis. Raster CHIRPS 23 September 2016 yang terpotong tidak diperbaiki atau diimputasi. Semua 46 grid tetap lolos; tidak ada perubahan state pada hari usable karena threshold tetap sama. Sebanyak 4.278 exclusion test adalah gabungan konservatif QA di kedua endpoint, bukan jumlah target terbukti salah. Kasus gap ekspor kosong murni lebih sempit: 2.392 transisi historis; perbedaan definisi ini harus dipertahankan.

NASA mencantumkan semua tanggal Mei 2026 sebagai missing standard SNPP pada inventory yang diaudit. Itu tidak membuktikan sensor mati sebulan atau membenarkan label nol. Data NASA/project berada dalam rantai sumber yang sama. Pemulihan seluruh gap bukan prasyarat kebijakan konservatif ini; reinstatement memerlukan bukti produk/QA baru dan versi analisis terpisah.

## Hasil primary revisi

Sumber: probabilitas tersimpan `{summary['run_id']}/{{validation,test}}_predictions.parquet`, dihitung ulang pada REV-05. Validation 2023–2024; test 2025-01-01–2026-08-31, hanya transisi yang dipakai pada 46 grid. [Tabel sumber terverifikasi](attachments/rev05_20261011/overall_results.csv). Brier multiclass adalah mean jumlah squared error ketiga probabilitas, tanpa pembagian tiga; log loss memakai log probabilitas asli tanpa clipping.

| Split | N | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
{result_table}

Per-class test pada sumber/periode/kohort yang sama; AUC one-vs-rest pooled, bukan ranking harian antargrid. [state_results.csv](attachments/rev05_20261011/state_results.csv).

| State | Support | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
{class_table}

Confusion matrix test, baris actual / kolom argmax; [confusion_matrices.csv](attachments/rev05_20261011/confusion_matrices.csv), sumber/periode/kohort sama.

| Actual state | Pred 0 | Pred 1 | Pred 2 | Support |
|---|---:|---:|---:|---:|
{matrix_rows}

Accuracy tidak cukup sebagai klaim keberhasilan: argmax hanya menangkap 106/2.330 target non-high (4,55%) dan 316/1.098 target High (28,78%) pada test revisi. Kinerja klasifikasi aktif masih terbatas. Pooled AUC tidak membuktikan kalibrasi, manfaat kovariat, kemampuan Top-K atau kesiapan penggunaan. Log loss/Brier adalah ringkasan probabilitas, belum bukti keunggulan tanpa pembanding yang sesuai.

## Hubungan dengan hasil historis

Sumber historis: received `final_test_050_3state.zip`, `test_predictions.parquet`; test 2025-01-01–2026-08-31, 46 grid, 27.968 transisi. Hasil ini tetap disimpan utuh dan aritmetikanya diverifikasi ulang pada REV-05: accuracy {fmt(historical['accuracy'])}, balanced accuracy {fmt(historical['balanced_accuracy'])}, macro-F1 {fmt(historical['macro_f1'])}, log loss {fmt(historical['log_loss'])}, Brier {fmt(historical['brier'])}. Ini **bukan** hasil koreksi dan bukan metrik dari notebook lokal binary/0,1°.

Jangan mencampur support/matrix historis dengan skor revisi, atau menafsirkan perbedaan skor sebagai efek kausal perbaikan/performa yang meningkat. Baris training dan evaluasi berbeda; missingness/exclusion dapat mengubah distribusi tanggal/kelas. Gambar asli di `docs/attachments/final_artifacts/` tetap historis 27.968-row; tidak ada gambar revisi baru pada tahap ini. Angka ECE/sensitivitas asli tidak dipindahkan sebagai hasil kalibrasi/sensitivity revisi. Laporan pinned `0745186:docs/final_validity_review.md` adalah review sebelum REV-03, termasuk rekomendasi pemulihan dan hitungan advisory lama; keputusan konservatif terbaru ada pada DEC-18, bukan klaim laporan lama telah diperbarui di GitHub.

## Keterbatasan dan kesimpulan yang dapat dipertahankan

1. Usable adalah penilaian provisional terhadap proxy ekspor, bukan sertifikat setiap grid teramati. Cloud/overpass, kelengkapan retained export, backfill/reprocessing dan recovery Fairbanks setelah interval minimum yang ditinjau belum dituntaskan. Missingness tidak diasumsikan acak; tidak ada estimasi performa untuk hari yang dikeluarkan.
2. Exclusion konservatif dapat membuang pengamatan valid; hasil berlaku pada subset yang dipakai dan kohort yang lolos screen, bukan seluruh lokasi/tanggal. Model independen/kelangkaan kelas membatasi stabilitas serta generalisasi; tidak ada uncertainty interval yang sudah diverifikasi.
3. Exact intermediate/model bytes dan dependency versions run asli belum tersedia. Model revisi, environment baru dan rekonstruksi yang cocok tidak membuktikan identitas run asli.
4. Tidak ada evaluasi baseline probabilitas transisi homogen (REV-08), validasi kalibrasi kondisional, validasi kausal, operational issuance/vintage atau manfaat patroli. Tidak ada klaim keunggulan yang memerlukan bukti-bukti tersebut.
5. Manuskrip/slide final tim belum tersedia pada lokasi repository/workspace yang diperiksa. Dokumen ini menyediakan metode/hasil/keterbatasan yang dapat dipakai untuk penyelarasan, tanpa mengklaim semua deliverable telah diperiksa atau persetujuan akademik telah diberikan.

Kesimpulan: pipeline revisi menangani observasi berstatus tidak layak secara eksplisit dan menghasilkan estimasi/evaluasi transisi yang dapat ditelusuri untuk proxy retrospektif bersyarat. Recovery kelas aktif oleh argmax tetap rendah. Klaim forecast operasional, ground-truth fire validity, cakupan pengamatan lengkap atau covariate superiority tidak didukung hasil ini. Siap sebagai bahan presentasi eksploratif **dengan batasan tersebut**; final submission tetap memerlukan pengecekan deliverable aktual, aturan mata kuliah dan verifikasi manusia atas bantuan AI.

## Reproduksi dan batas pengujian

Environment REV-03: Python 3.11.16, NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, PyArrow 25.0.1, scikit-learn 1.9.1. Tidak ada dependency baru dipasang. Kode final model/screen diambil dari Git object pinned `0745186`; checkout lokal tetap lama, sehingga menjalankan notebook lokal tidak mereproduksi hasil final.

Untuk inspeksi ringan gunakan CSV/receipt di docs/attachments/rev05_20261011 dan dokumen ini. Untuk QA penuh REV-05, butuh direktori run REV-03 lengkap (162 berkas), ZIP original test, evidence REV-01/REV-02, pinned Git objects dan environment di atas. Heavy inputs/models diabaikan Git dan tidak dianggap tersedia pada clone baru. Asset/hash daftar ada pada manifest REV-03 dan receipt REV-05. Jangan mengunduh atau meminta semua data lagi tanpa memeriksa assets yang sudah tersedia.

Perintah yang dijalankan pada tahap ini:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_rev03_correction.py -v
.\.venv\Scripts\python.exe scripts/build_rev05_results.py
```

Perintah pertama dibundel oleh perintah kedua; 12 tes regression terarah dijalankan, bukan seluruh suite historis. QA REV-05 menghitung ulang semua overall/per-grid/per-current-state primary metrics, class support/precision/recall/F1/AUC (AUC rank-sum independen), confusion matrices, coverage dan waterfall; membandingkannya dengan sumber pada toleransi ≤1e-10. Tidak ada fit/inference baru, koreksi data, recalibration, baseline experiment atau replay guarded test. Builder menolak menimpa folder evidence yang sudah ada; bukan perintah idempotent untuk dijalankan sembarangan. Receipt verifikasi dapat diinspeksi tanpa mengeksekusi builder.

README current notice diperbarui secara sengaja; exact bytes sebelumnya diarsipkan. Karena README tercakup manifest REV-03, verifier lama terhadap working-tree README akan menandai perubahan ini. Itu drift dokumentasi yang tercatat, bukan perubahan model/data; jangan mengubah manifest lama atau melemahkan hash guard. Verifikasi REV-05 mencocokkan hash README lama pada arsip, dan hash berkas protected lainnya pada path asli.

Sumber verifikasi: [receipt REV-05](attachments/rev05_20261011/verification.json), [reviewed source register](attachments/rev05_20261011/reviewed_document_sources.json), [builder](../scripts/build_rev05_results.py), [progress](PROGRESS.md). Dokumen ini tidak mengubah sembilan artefak asli, model/dataset revisi, source aplikasi, notebook, credential, commit atau remote.
"""
    (ROOT / "docs/FINAL_RESULTS.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
