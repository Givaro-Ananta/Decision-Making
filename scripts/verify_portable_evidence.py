"""Verify relative-path evidence by SHA-256 without replaying any experiment.

Portable mode ignores timestamp differences. Original-attestation mode additionally
requires the recorded mtime_ns. Neither mode skips missing required files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def entries_for(manifest, profile=None):
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    if "profiles" in manifest:
        if not isinstance(manifest["profiles"], dict) or profile not in manifest["profiles"]:
            raise ValueError("Choose an existing --profile; no automatic partial scope")
        entries = manifest["profiles"][profile].get("files")
    elif "protected_files" in manifest:
        if profile is not None:
            raise ValueError("Legacy protected_files manifest does not define profiles")
        entries = manifest["protected_files"]
    else:
        raise ValueError("Expected profiles or legacy protected_files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Required-file set must be a non-empty object")
    for name, meta in entries.items():
        if not isinstance(name, str) or not isinstance(meta, dict) or not re.fullmatch(r"[a-fA-F0-9]{64}", str(meta.get("sha256", ""))):
            raise ValueError("Each relative file identifier requires a SHA-256 value")
        if "mtime_ns" in meta and (not isinstance(meta["mtime_ns"], int) or isinstance(meta["mtime_ns"], bool)):
            raise ValueError("mtime_ns must be an integer when supplied")
    return entries


def local_path(root, name):
    relative = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or relative.is_absolute() or PureWindowsPath(name).drive or ".." in relative.parts:
        raise ValueError("Expected relative POSIX path within asset root")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Resolved asset escapes root or is not a file identifier")
    return path


def verify(root, entries, mode="portable"):
    if mode not in ("portable", "original-attestation"):
        raise ValueError("Unknown verification mode")
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Asset root must be an existing directory")
    results = []
    for name, meta in entries.items():
        row = dict(file=name, expected_sha256=meta["sha256"].lower())
        try:
            path = local_path(root, name)
            if not path.exists():
                row["status"] = "missing_required_file"
            elif not path.is_file():
                row["status"] = "not_regular_file"
            else:
                before = path.stat()
                observed = digest(path)
                after = path.stat()
                row.update(observed_sha256=observed, recorded_mtime_ns=meta.get("mtime_ns"), observed_mtime_ns=after.st_mtime_ns)
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    row["status"] = "file_changed_during_check"
                elif observed != meta["sha256"].lower():
                    row["status"] = "sha256_mismatch"
                elif mode == "original-attestation" and "mtime_ns" not in meta:
                    row["status"] = "missing_attestation_mtime"
                elif mode == "original-attestation" and after.st_mtime_ns != meta["mtime_ns"]:
                    row["status"] = "mtime_mismatch"
                else:
                    row["status"] = "pass"
        except ValueError:
            row["status"] = "unsafe_relative_path"
        except OSError as error:
            row.update(status="unreadable_file", error_type=type(error).__name__)
        results.append(row)
    failures = sum(row["status"] != "pass" for row in results)
    return dict(mode=mode, required_files=len(results), passed_files=len(results) - failures, failed_files=failures,
                success=failures == 0, results=results,
                scope="Only declared required files; no model/metric replay or original-run/coverage certification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Asset root; independent of working directory")
    parser.add_argument("--profile", help="Required for manifests with named profiles")
    parser.add_argument("--mode", choices=("portable", "original-attestation"), default="portable")
    parser.add_argument("--json-report", type=Path, help="Optional NEW report path; never overwrite")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        entries = entries_for(manifest, args.profile)
        result = verify(args.root, entries, args.mode)
        result.update(profile=args.profile, manifest_sha256=digest(args.manifest))
        if args.json_report is not None:
            # Refuse an output that could become an absent required input or overwrite evidence.
            destination = args.json_report.resolve()
            if destination == args.manifest.resolve() or any(destination == local_path(args.root.resolve(), name) for name in entries):
                raise ValueError("Report destination must differ from manifest and every required input")
            args.json_report.parent.mkdir(parents=True, exist_ok=True)
            with args.json_report.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print(json.dumps(dict(success=False, error_type=type(error).__name__, message=str(error))))
        return 2
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    for row in result["results"]:
        if row["status"] != "pass":
            print(f"FAIL {row['file']}: {row['status']}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
