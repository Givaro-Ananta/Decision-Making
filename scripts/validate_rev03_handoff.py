"""Check knowledge links/IDs, immutable REV-03 outputs and the no-overwrite guard."""
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/revision_review/REV03_20261011_ec3d83da299e"
DOCS = [ROOT / "AGENTS.md"] + [ROOT / "docs" / (name + ".md") for name in
        ("PROJECT_CONTEXT", "FINAL_AUDIT", "REVISION_PLAN", "DECISIONS", "PROGRESS")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    errors, links, definitions, references = [], [], [], set()
    for path in DOCS:
        content = path.read_text(encoding="utf-8")
        owner = {"FINAL_AUDIT.md": "AUD", "REVISION_PLAN.md": "REV", "DECISIONS.md": "DEC"}.get(path.name)
        if owner:
            definitions += re.findall(rf"^#{{2,3}} ({owner}-\d+)\b", content, flags=re.M)
        references.update(re.findall(r"\b(?:AUD|REV|DEC)-\d+\b", content))
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            if re.match(r"(?:https?://|mailto:|#)", target):
                continue
            resolved = (path.parent / target.strip("<>").split("#")[0]).resolve()
            links.append(dict(document=path.relative_to(ROOT).as_posix(), target=target, exists=resolved.exists()))
            if not resolved.exists():
                errors.append(f"Broken link {path.name}: {target}")
    if len(definitions) != len(set(definitions)):
        errors.append("Duplicate AUD/REV/DEC definitions")
    if references - set(definitions):
        errors.append(f"Undefined identifiers: {sorted(references - set(definitions))}")
    verification = json.loads((OUT / "verification.json").read_text())
    before = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob("*") if p.is_file()}
    guard = subprocess.run([sys.executable, "scripts/run_rev03_correction.py"], cwd=ROOT, capture_output=True, text=True)
    guard_pass = guard.returncode != 0 and "FileExistsError" in guard.stderr
    if not guard_pass:
        errors.append("Existing-run overwrite guard failed")
    after = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob("*") if p.is_file()}
    if before != after:
        errors.append("Completed output bytes changed")
    for relative, expected in verification["protected_before"].items():
        if sha(ROOT / relative) != expected:
            errors.append(f"Protected-file drift: {relative}")
    for path in [ROOT / "scripts" / f"{name}.py" for name in
                 ("run_rev03_correction", "verify_rev03_correction", "validate_rev03_handoff")] + [ROOT / "tests/test_rev03_correction.py"]:
        ast.parse(path.read_text(encoding="utf-8"))
    whitespace = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    if whitespace.returncode:
        errors.append(whitespace.stdout + whitespace.stderr)
    record = dict(date="2026-10-11", validator_sha256=sha(__file__), local_links_checked=len(links), links=links,
                  unique_definitions=len(set(definitions)), defined_ids=sorted(set(definitions)), errors=errors,
                  existing_run_guard_expected_rejection=guard_pass, completed_output_files_unchanged=len(before),
                  protected_files_unchanged=len(verification["protected_before"]), python_files_parse=4,
                  git_diff_whitespace_pass=whitespace.returncode == 0,
                  limitations=["External URLs not re-fetched; live GitHub state not rechecked", "No full original application suite or full raw-weather SHA rerun"])
    target = ROOT / "docs/attachments/rev03_20261011/knowledge_handoff_verification.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in ("links", "defined_ids")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
