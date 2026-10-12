"""Capture the authorized REV-07 integrity handoff; does not execute a model."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/attachments/rev07_20261011"
RUN = ROOT / "outputs/revision_review/REV03_20261011_ec3d83da299e"
spec = importlib.util.spec_from_file_location("portable_evidence", ROOT / "scripts/verify_portable_evidence.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def main():
    prior = json.loads((RUN / "verification.json").read_text())
    rev05 = json.loads((ROOT / "docs/attachments/rev05_20261011/verification.json").read_text())
    protected = dict(prior["protected_after"])
    protected.pop("README.md")
    protected["docs/attachments/rev05_20261011/README_before.txt"] = rev05["authorized_protected_file_change"]["README.md"]["before_sha256"]
    protected["README.md"] = rev05["authorized_protected_file_change"]["README.md"]["after_sha256"]
    protected.update({p.relative_to(ROOT).as_posix(): api.digest(p) for p in RUN.rglob("*") if p.is_file()})
    for name, expected in protected.items():
        if api.digest(ROOT / name) != expected:
            raise ValueError(f"Protected source drift: {name}")
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_portable_evidence.py", "-v"], cwd=ROOT, capture_output=True, text=True)
    if test.returncode or "Ran 9 tests" not in test.stderr:
        raise RuntimeError(test.stdout + test.stderr)
    legacy = subprocess.check_output(["git", "show", "0745186:outputs/reports/firms_observation_coverage/audit_manifest.json"], cwd=ROOT)
    legacy_json = json.loads(legacy)
    api.entries_for(legacy_json)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "original_coverage_audit_manifest.json").write_bytes(legacy)
    (OUT / "regression_tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
    light = [ROOT / "docs/FINAL_RESULTS.md", ROOT / "scripts/verify_portable_evidence.py", ROOT / "tests/test_portable_evidence.py"]
    for directory in ("final_artifacts", "rev05_20261011"):
        light += [p for p in (ROOT / "docs/attachments" / directory).iterdir() if p.is_file()]
    light += [ROOT / "docs/attachments/rev03_20261011/REV03_REPORT.md", ROOT / "docs/attachments/rev03_20261011/independent_verification.json", OUT / "original_coverage_audit_manifest.json"]

    def metadata(paths):
        return {p.relative_to(ROOT).as_posix(): dict(sha256=api.digest(p), mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(paths))}

    lightweight = metadata(light)
    full = metadata(light + [ROOT / p for p in protected])
    manifest = dict(schema="project-evidence/v1", revision="REV-07", decision="DEC-20", date="2026-10-11",
                    timestamp_meaning="New profiles record this handoff's filesystem timestamps, not original fitting time. Portable mode ignores them. Archived original manifest retains original values unchanged.",
                    profiles={"lightweight": dict(description="Available research report, source CSVs, prior receipts and original reporting artifacts; no raw inputs/model files or metric replay", files=lightweight),
                              "revised-run": dict(description="Lightweight plus all 162 completed REV-03 files and required protected input snapshots; not full original 314-file replay", files=full)})
    (OUT / "portable_inputs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results = {}
    for profile in manifest["profiles"]:
        result = api.verify(ROOT, api.entries_for(manifest, profile))
        if not result["success"]:
            raise AssertionError(profile)
        (OUT / f"{profile}_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results[profile] = result["required_files"]
    for name, expected in protected.items():
        if api.digest(ROOT / name) != expected:
            raise AssertionError(f"Changed protected file: {name}")
    record = dict(date="2026-10-11", regression_tests_passed=9, profiles_verified=results,
                  protected_before=protected, protected_after=protected,
                  original_manifest_sha256=api.digest(OUT / "original_coverage_audit_manifest.json"),
                  original_manifest_files=len(legacy_json["protected_files"]), original_314_asset_replay=False,
                  portable_manifest_sha256=api.digest(OUT / "portable_inputs.json"),
                  scripts={p: api.digest(ROOT / p) for p in ("scripts/verify_portable_evidence.py", "scripts/build_rev07_portable_manifest.py", "tests/test_portable_evidence.py")},
                  outputs={p.name: api.digest(p) for p in OUT.iterdir() if p.is_file()},
                  fitting=False, inference=False, model_or_data_changes=False, dependencies_installed=False,
                  limitations=["Byte integrity does not reproduce metrics or certify observations/original model identity", "Original 314 assets not all available; original builder left unchanged", "New profiles' timestamps are this handoff's snapshot, not historical fitting metadata", "No live remote or external data refresh", "REV-08 not executed"])
    (OUT / "verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(tests=9, profiles=results, protected_files=len(protected), legacy_manifest_files=len(legacy_json["protected_files"])), indent=2))


if __name__ == "__main__":
    main()
