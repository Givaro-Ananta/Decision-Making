import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_portable_evidence.py"
spec = importlib.util.spec_from_file_location("portable_evidence", SCRIPT)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class PortableEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "asset.bin"
        self.source.write_bytes(b"known immutable evidence")
        self.entries = {"asset.bin": dict(sha256=api.digest(self.source), mtime_ns=self.source.stat().st_mtime_ns)}

    def test_copy_with_new_timestamp_passes_portable_fails_attestation(self):
        copied = self.root / "different-machine"
        copied.mkdir()
        shutil.copyfile(self.source, copied / "asset.bin")
        os.utime(copied / "asset.bin", ns=(1000000000, 1000000000))
        self.assertTrue(api.verify(copied, self.entries)["success"])
        self.assertEqual(api.verify(copied, self.entries, "original-attestation")["results"][0]["status"], "mtime_mismatch")
        self.assertTrue(api.verify(self.root, self.entries, "original-attestation")["success"])

    def test_changed_byte_fails_even_with_original_mtime(self):
        self.source.write_bytes(b"changed immutable evidence")
        os.utime(self.source, ns=(self.entries["asset.bin"]["mtime_ns"],) * 2)
        for mode in ("portable", "original-attestation"):
            self.assertEqual(api.verify(self.root, self.entries, mode)["results"][0]["status"], "sha256_mismatch")

    def test_missing_file_never_silently_skipped(self):
        entries = dict(self.entries, missing=dict(sha256="0" * 64))
        result = api.verify(self.root, entries)
        self.assertFalse(result["success"])
        self.assertEqual((result["required_files"], result["passed_files"], result["failed_files"]), (2, 1, 1))
        self.assertEqual(result["results"][1]["status"], "missing_required_file")

    def test_legacy_manifest_and_strict_requires_metadata(self):
        self.assertEqual(api.entries_for({"protected_files": self.entries}), self.entries)
        entries = {"asset.bin": dict(sha256=api.digest(self.source))}
        self.assertTrue(api.verify(self.root, entries)["success"])
        self.assertEqual(api.verify(self.root, entries, "original-attestation")["results"][0]["status"], "missing_attestation_mtime")

    def test_invalid_or_empty_manifest_and_profile_fail(self):
        for manifest, profile in [({}, None), ({"protected_files": {}}, None), ({"protected_files": {"x": {"sha256": "bad"}}}, None),
                                  ({"profiles": {"light": {"files": self.entries}}}, None), ({"profiles": {"light": {"files": self.entries}}}, "other")]:
            with self.assertRaises(ValueError):
                api.entries_for(manifest, profile)

    def test_paths_cannot_escape_root(self):
        for name in ("../outside", "/etc/passwd", "C:/outside", "..\\outside", "", "."):
            result = api.verify(self.root, {name: dict(sha256="0" * 64)})
            self.assertEqual(result["results"][0]["status"], "unsafe_relative_path")

    def test_directory_is_not_an_asset(self):
        (self.root / "folder").mkdir()
        result = api.verify(self.root, {"folder": dict(sha256="0" * 64)})
        self.assertEqual(result["results"][0]["status"], "not_regular_file")

    def test_cli_runs_from_another_working_directory_and_refuses_overwrite(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"profiles": {"light": {"files": self.entries}}}))
        output = self.root / "receipt.json"
        command = [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--root", str(self.root), "--profile", "light", "--json-report", str(output)]
        first = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        previous = output.read_bytes()
        second = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(second.returncode, 2)
        self.assertEqual(output.read_bytes(), previous)
        missing = subprocess.run(command[:2] + ["--manifest", str(manifest), "--root", str(self.root), "--profile", "bad"], capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)

    def test_report_cannot_fill_a_missing_required_input(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"protected_files": {"missing": {"sha256": "0" * 64}}}))
        result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest), "--root", str(self.root), "--json-report", str(self.root / "missing")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "missing").exists())


if __name__ == "__main__":
    unittest.main()
