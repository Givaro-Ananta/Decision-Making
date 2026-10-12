"""Read-only hash checks; no model calls or writes. Matches do not certify methods."""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "docs" / "attachments" / "final_artifacts"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes root: {relative}")
    return path


def check_file(path, expected):
    if not path.is_file():
        return "missing", None
    actual = sha256(path)
    return ("matched" if actual == expected else "mismatch"), actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT,
                        help="Original project root containing provenance-relative paths")
    parser.add_argument("--require-sources", action="store_true",
                        help="Fail unless all eight provenance-listed source files match")
    parser.add_argument("--renderer-path", type=Path,
                        help="Original renderer file to check against renderer_sha256")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        parser.error(f"Source root is not a directory: {source_root}")

    try:
        manifest = json.loads((ARTIFACT_DIR / "import_manifest.json").read_text(encoding="utf-8"))
        failed = False
        copies = []
        for item in manifest["files"]:
            path = contained_path(PROJECT_ROOT, item["copied_path"])
            if not path.is_relative_to(ARTIFACT_DIR.resolve()):
                raise ValueError(f"Copy outside artifact directory: {path}")
            status, actual = check_file(path, item["sha256"])
            if status == "matched" and path.stat().st_size != item["bytes"]:
                status = "size_mismatch"
            failed |= status != "matched"
            copies.append({"path": item["copied_path"], "status": status, "actual_sha256": actual})

        # The ignored original ZIP may be unavailable in a new clone.
        archive = contained_path(PROJECT_ROOT, manifest["archive_name"])
        archive_status, archive_actual = check_file(archive, manifest["archive_sha256"])
        failed |= archive_status == "mismatch"
        provenance = json.loads((ARTIFACT_DIR / "reporting_provenance.json").read_text(encoding="utf-8"))
        sources = []
        for relative, expected in provenance["source_sha256"].items():
            status, actual = check_file(contained_path(source_root, relative), expected)
            failed |= status == "mismatch" or (args.require_sources and status != "matched")
            sources.append({"path": relative, "status": status,
                            "expected_sha256": expected, "actual_sha256": actual})

        renderer = {"status": "not_supplied", "expected_sha256": provenance["renderer_sha256"]}
        if args.renderer_path is not None:
            path = args.renderer_path.resolve()
            status, actual = check_file(path, provenance["renderer_sha256"])
            failed |= status != "matched"
            renderer.update(path=str(path), status=status, actual_sha256=actual)

        matched = sum(item["status"] == "matched" for item in sources)
        print(json.dumps({
            "check_scope": "file_integrity_and_source_hashes_only",
            "methodology_verified": False,
            "source_root": str(source_root), "copies": copies,
            "archive": {"status": archive_status, "actual_sha256": archive_actual},
            "sources": sources, "matched_source_count": matched,
            "listed_source_count": len(sources),
            "source_hash_status": "complete" if matched == len(sources) else "incomplete",
            "renderer": renderer, "passed_requested_checks": not failed,
        }, indent=2, ensure_ascii=True))
        return 1 if failed else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Audit input error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
