"""Read-only checks for published REV-05 tables, evidence and knowledge references."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/attachments/rev05_20261011"
RUN = ROOT / "outputs/revision_review/REV03_20261011_ec3d83da299e"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(content, heading):
    lines = content.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(heading))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([v.strip() for v in line.strip("|").split("|")])
    return rows


def main():
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(dict(name=name, passed=True))

    receipt = json.loads((OUT / "verification.json").read_text())
    check("source_qa_passes", len(receipt["checks"]) == receipt["checks_passed"] == 3430 and all(c["passed"] for c in receipt["checks"]))
    check("executed_builder_identity", sha(ROOT / "scripts/build_rev05_results.py") == receipt["script_sha256"])
    check("report_identity", sha(ROOT / "docs/FINAL_RESULTS.md") == receipt["report_sha256"])
    for name, expected in receipt["inputs"].items():
        check(f"run_input:{name}", sha(RUN / name) == expected)
    for name, expected in receipt["outputs"].items():
        check(f"published_source:{name}", sha(OUT / name) == expected)
    revision = json.loads((RUN / "verification.json").read_text())
    for name, expected in revision["outputs"].items():
        check(f"immutable_revision:{name}", sha(RUN / name) == expected)
    for name, expected in revision["protected_after"].items():
        path = OUT / "README_before.txt" if name == "README.md" else ROOT / name
        check(f"protected_snapshot:{name}", sha(path) == expected)
    drift = receipt["authorized_protected_file_change"]["README.md"]
    check("authorized_readme_after", sha(ROOT / "README.md") == drift["after_sha256"])
    check("archived_readme_before", sha(OUT / "README_before.txt") == drift["before_sha256"])
    anchor = b"Research project for **Decision Making"
    check("original_readme_body_bytes", (ROOT / "README.md").read_bytes().split(anchor, 1)[1] == (OUT / "README_before.txt").read_bytes().split(anchor, 1)[1])
    report = (ROOT / "docs/FINAL_RESULTS.md").read_text(encoding="utf-8")
    check("no_unintended_control_characters", not any(ord(c) < 32 and c not in "\n\r\t" for c in report))
    overall = pd.read_csv(OUT / "overall_results.csv")
    fields = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss", "brier"]
    check("rendered_overall_cells", table(report, "| Split | N |") == [[r["split"], f"{r['n']:,}", *[f"{r[f]:.8f}" for f in fields]] for r in overall.to_dict("records")])
    classes = pd.read_csv(OUT / "state_results.csv")
    rows = [[str(r["state"]), f"{r['support']:,}", *[f"{r[f]:.8f}" for f in ("precision", "recall", "f1", "auc")]] for r in classes[classes.split.eq("test")].to_dict("records")]
    check("rendered_class_cells", table(report, "| State | Support |") == rows)
    matrices = pd.read_csv(OUT / "confusion_matrices.csv")
    matrix = matrices[matrices.split.eq("test")].sort_values(["actual_state", "predicted_state"])["count"].to_numpy().reshape(3, 3)
    check("rendered_confusion_cells", table(report, "| Actual state |") == [[str(s), *[f"{n:,}" for n in matrix[s]], f"{matrix[s].sum():,}"] for s in range(3)])
    splits = pd.read_csv(OUT / "split_accounting.csv")
    check("rendered_waterfall_cells", table(report, "| Split | Historis,") == [[r["split"], *[f"{r[f]:,}" for f in ("historical_predictor_available", "observation_excluded", "training_screen_excluded", "revised_included")]] for r in splits.to_dict("records")])
    paths = [ROOT / "AGENTS.md", ROOT / "README.md"] + [ROOT / "docs" / (name + ".md") for name in
             ("PROGRESS", "PROJECT_CONTEXT", "DECISIONS", "FINAL_AUDIT", "REVISION_PLAN", "FINAL_RESULTS")]
    definitions, refs, links = [], set(), []
    for path in paths:
        content = path.read_text(encoding="utf-8")
        owner = {"FINAL_AUDIT.md": "AUD", "REVISION_PLAN.md": "REV", "DECISIONS.md": "DEC"}.get(path.name)
        if owner:
            definitions += re.findall(rf"^#{{2,3}} ({owner}-\d+)\b", content, flags=re.M)
        refs.update(re.findall(r"\b(?:AUD|REV|DEC)-\d+\b", content))
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            if re.match(r"(?:https?://|mailto:|#)", target):
                continue
            exists = (path.parent / target.strip("<>").split("#")[0]).exists()
            links.append(dict(document=path.relative_to(ROOT).as_posix(), target=target, exists=exists))
            check(f"link:{path.name}:{target}", exists)
    check("unique_canonical_ids", len(definitions) == len(set(definitions)) == 38)
    check("references_resolve", not (refs - set(definitions)))
    for name in ("build_rev05_results", "verify_rev05_handoff"):
        ast.parse((ROOT / "scripts" / (name + ".py")).read_text(encoding="utf-8"))
        check(f"python_parse:{name}", True)
    whitespace = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    check("git_diff_whitespace", whitespace.returncode == 0)
    record = dict(date="2026-10-11", checks_passed=len(checks), checks=checks, local_links_checked=len(links),
                  links=links, unique_canonical_ids=len(set(definitions)), verifier_sha256=sha(__file__),
                  revision_verification_sha256=sha(RUN / "verification.json"), rev05_receipt_sha256=sha(OUT / "verification.json"),
                  document_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                  limitations=["No live remote/external URL recheck", "No raw-weather SHA rerun or full original application suite", "Unseen final manuscript/slides remain unreviewed"])
    (OUT / "knowledge_handoff_verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(checks_passed=len(checks), local_links_checked=len(links), unique_canonical_ids=len(set(definitions)), errors=[]), indent=2))


if __name__ == "__main__":
    main()
