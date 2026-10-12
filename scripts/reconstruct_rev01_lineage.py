"""Reconstruct historical FIRMS lineage separately; never run fitting/evaluation.

Uses the existing rasterio-bundled GEOS C API, avoiding dependency installation.
API reference: https://libgeos.org/doxygen/geos__c_8h.html
"""
import ctypes as C
import hashlib
import io
import json
import platform
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/attachments/rev01_20261011"
COMMIT = "0745186a2435c693a990fa394031f0c036c24b60"
RAW = ROOT / "data/DL_FIRE_SV-C2_809701"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


class Geometry:
    def __init__(self):
        folder = Path(rasterio.__file__).parent.parent / "rasterio.libs"
        self.dll = next(folder.glob("geos_c-*.dll"))
        self.lib = C.CDLL(str(self.dll))
        specs = {
            "GEOS_init_r": (C.c_void_p, []),
            "GEOS_finish_r": (None, [C.c_void_p]),
            "GEOSWKBReader_create_r": (C.c_void_p, [C.c_void_p]),
            "GEOSWKBReader_read_r": (C.c_void_p, [C.c_void_p, C.c_void_p, C.c_char_p, C.c_size_t]),
            "GEOSWKBReader_destroy_r": (None, [C.c_void_p, C.c_void_p]),
            "GEOSGeom_destroy_r": (None, [C.c_void_p, C.c_void_p]),
            "GEOSPrepare_r": (C.c_void_p, [C.c_void_p, C.c_void_p]),
            "GEOSGeom_clone_r": (C.c_void_p, [C.c_void_p, C.c_void_p]),
            "GEOSUnion_r": (C.c_void_p, [C.c_void_p, C.c_void_p, C.c_void_p]),
            "GEOSPreparedGeom_destroy_r": (None, [C.c_void_p, C.c_void_p]),
            "GEOSPreparedIntersectsXY_r": (C.c_byte, [C.c_void_p, C.c_void_p, C.c_double, C.c_double]),
            "GEOSversion": (C.c_char_p, []),
        }
        for name, (result, args) in specs.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = result, args
        self.context = self.lib.GEOS_init_r()
        self.reader = self.lib.GEOSWKBReader_create_r(self.context)
        self.objects = []

    def prepared(self, wkb):
        geom = self.lib.GEOSWKBReader_read_r(self.context, self.reader, wkb, len(wkb))
        if not geom:
            raise ValueError("GEOS could not decode geometry")
        prepared = self.lib.GEOSPrepare_r(self.context, geom)
        if not prepared:
            raise ValueError("GEOS could not prepare geometry")
        self.objects.append((geom, prepared))
        return prepared

    def covers(self, prepared, x, y):
        # For points, intersects(polygon, point) equals covers(polygon, point),
        # including the exterior/hole boundary, unlike strict contains.
        result = self.lib.GEOSPreparedIntersectsXY_r(self.context, prepared, float(x), float(y))
        if result == 2:
            raise ValueError("GEOS predicate exception")
        return result == 1

    def union(self, geometries):
        result = self.lib.GEOSGeom_clone_r(self.context, geometries[0])
        for geom in geometries[1:]:
            merged = self.lib.GEOSUnion_r(self.context, result, geom)
            self.lib.GEOSGeom_destroy_r(self.context, result)
            if not merged:
                raise ValueError("GEOS union failed")
            result = merged
        prepared = self.lib.GEOSPrepare_r(self.context, result)
        self.objects.append((result, prepared))
        return prepared

    def close(self):
        for geom, prepared in reversed(self.objects):
            self.lib.GEOSPreparedGeom_destroy_r(self.context, prepared)
            self.lib.GEOSGeom_destroy_r(self.context, geom)
        self.lib.GEOSWKBReader_destroy_r(self.context, self.reader)
        self.lib.GEOS_finish_r(self.context)


def rectangle_wkb(rings):
    import struct
    return b"\x01" + struct.pack("<II", 3, len(rings)) + b"".join(
        struct.pack("<I", len(ring)) + b"".join(struct.pack("<dd", x, y) for x, y in ring) for ring in rings)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = [ROOT / "final_test_050_3state.zip", ROOT / "grid.parquet",
        ROOT / "final_artifacts-20261008T144712Z-1-001.zip", ROOT / "README.md", ROOT / "docs/data-provenance.md"]
    protected_paths += list((ROOT / "src").rglob("*.py")) + list((ROOT / "notebooks").glob("*.ipynb"))
    protected_paths += list(RAW.glob("*.csv")) + list((ROOT / "docs/attachments/final_artifacts").glob("*"))
    protected = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected_paths if p.is_file()}
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    design_bytes = subprocess.check_output(["git", "show", f"{COMMIT}:configs/modeling_design_050.json"], cwd=ROOT)
    design = json.loads(design_bytes)
    (OUT / "original_modeling_design_050.json").write_bytes(design_bytes)
    with zipfile.ZipFile(ROOT / "final_test_050_3state.zip") as archive:
        frozen = json.loads(archive.read("final_test_050_3state/frozen_design.json"))
        predictions = pd.read_parquet(io.BytesIO(archive.read("final_test_050_3state/test_predictions.parquet")))
    grid = pd.read_parquet(ROOT / "grid.parquet")
    geo = Geometry()
    try:
        square = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
        hole = [(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)]
        fixture = geo.prepared(rectangle_wkb([square, hole]))
        for label, point, expected in [
            ("polygon_interior", (.5, .5), True), ("polygon_boundary", (0, 2), True),
            ("hole_interior", (2, 2), False), ("hole_boundary", (1, 2), True),
            ("outside", (5, 2), False)]:
            check(label, geo.covers(fixture, *point) == expected)
        prepared = {row.grid_id: geo.prepared(row.study_geometry) for row in grid.itertuples()}
        domain = geo.union([geom for geom, prep in geo.objects[1:]])
        pieces, retention = [], []
        for path in sorted(RAW.glob("*.csv")):
            retained_n = 0
            for chunk in pd.read_csv(path, usecols=["latitude", "longitude", "acq_date", "confidence", "frp"], chunksize=200000):
                data = chunk[chunk.acq_date.between("2016-09-22", "2026-08-31") & chunk.confidence.isin(["n", "h"])].copy()
                row = np.floor(np.round((6.05 - data.latitude.to_numpy()) / .5, 10)).astype(int)
                col = np.floor(np.round((data.longitude.to_numpy() - 94.95) / .5, 10)).astype(int)
                data["grid_id"] = [f"SUM050_R{r:03d}_C{c:03d}" for r, c in zip(row, col)]
                data = data[data.grid_id.isin(prepared)].copy()
                inside = np.fromiter((geo.covers(domain, x, y) for g, x, y in
                    zip(data.grid_id, data.longitude, data.latitude)), dtype=bool, count=len(data))
                data = data[inside]
                check(f"finite_nonnegative_FRP_{path.name}_chunk_{retained_n}", np.isfinite(data.frp).all() and data.frp.ge(0).all())
                retained_n += len(data)
                pieces.append(data[["grid_id", "acq_date", "frp"]])
            retention.append(dict(file=path.name, retained=retained_n))
            print(f"Raw polygon retention: {path.name}: {retained_n}", flush=True)
        points = pd.concat(pieces, ignore_index=True).rename(columns={"acq_date": "date"})
        points["date"] = pd.to_datetime(points.date)
        active = points.groupby(["grid_id", "date"]).agg(hotspot_count=("frp", "size"), frp_sum=("frp", "sum")).reset_index()
        check("polygon_retained_total_305106", len(points) == 305106)
        check("archive_retained_290362", retention[0]["retained"] == 290362)
        check("NRT_retained_14744", retention[1]["retained"] == 14744)
        population = active[active.date.le("2022-12-31")]
        q50, q75 = population.frp_sum.quantile([.5, .75]).tolist()
        check("training_active_population_41545", len(population) == design["threshold_training_active_rows"] == 41545)
        check("training_Q50_5_96", abs(q50 - design["state_boundaries"]["low_moderate"]) <= 1e-12)
        check("training_Q75_16_84", abs(q75 - frozen["high_boundary_mw"]) <= 1e-12)
        active.to_parquet(OUT / "reconstructed_firms_active_grid_days.parquet", index=False)
        days = pd.date_range("2016-09-22", "2026-08-31")
        panel = pd.MultiIndex.from_product([sorted(grid.grid_id), days], names=["grid_id", "date"]).to_frame(index=False).merge(active, how="left", on=["grid_id", "date"], validate="one_to_one")
        panel["hotspot_count"] = panel.hotspot_count.fillna(0).astype("int64")
        panel["frp_sum"] = panel.frp_sum.fillna(0.)
        panel["state"] = np.where(panel.hotspot_count.eq(0), 0, np.where(panel.frp_sum.lt(q75), 1, 2)).astype("int8")
        check("historical_threshold_boundaries", np.where(np.array([0, 1, 1]) == 0, 0, np.where(np.array([0, q75-.01, q75]) < q75, 1, 2)).tolist() == [0, 1, 2])
        check("complete_187_by_3631_panel", len(panel) == 187 * 3631 and not panel.duplicated(["grid_id", "date"]).any())
        lookup = panel.set_index(["grid_id", "date"]).state
        for state, day in [("current_state", "date"), ("actual_next_state", "next_date")]:
            reconstructed = lookup.reindex(pd.MultiIndex.from_frame(predictions[["grid_id", day]].rename(columns={day: "date"}))).to_numpy()
            check(f"all_original_test_{state}_reconstructed", np.array_equal(reconstructed, predictions[state]))
        panel.to_parquet(OUT / "reconstructed_historical_detection_panel.parquet", index=False)
        transitions = panel.rename(columns={"state": "current_state"}).copy()
        transitions["next_date"] = transitions.groupby("grid_id").date.shift(-1)
        transitions["next_state"] = transitions.groupby("grid_id").current_state.shift(-1)
        transitions = transitions[transitions.next_date.eq(transitions.date + pd.Timedelta(days=1))].copy()
        transitions["next_state"] = transitions.next_state.astype("int8")
        transitions["split"] = np.select([transitions.next_date.le("2022-12-31"), transitions.next_date.le("2024-12-31")], ["train", "validation"], default="test")
        transitions["historical_frozen_grid"] = transitions.grid_id.isin(frozen["eligible_grid_ids"])
        transitions["predictor_availability"] = "not_reconstructed_in_this_detection_only_stage"
        transitions.to_parquet(OUT / "reconstructed_detection_transitions.parquet", index=False)
        transitions.groupby(["grid_id", "split", "current_state", "next_state"]).size().rename("rows").to_csv(OUT / "reconstructed_destination_counts.csv")
        diagnostics = []
        for (gid, state), frame in transitions[transitions.split.eq("train")].groupby(["grid_id", "current_state"]):
            counts = frame.next_state.value_counts().sort_index()
            diagnostics.append(dict(grid_id=gid, current_state=int(state), calendar_n=len(frame),
                destination_counts=json.dumps({int(k): int(v) for k, v in counts.items()}),
                sample_ge50=len(frame) >= 50, observed_destination_min_ge10=int(counts.min()) >= 10,
                qualification="necessary_count_screens_only; no_predictor_availability_or_numerical_screen"))
        pd.DataFrame(diagnostics).to_csv(OUT / "reconstructed_count_screen.csv", index=False)
        ledger = pd.read_csv(ROOT / "docs/attachments/rev02_20261011/product_quality_ledger.csv", parse_dates=["date"])
        gap_dates = set(ledger.loc[ledger.raw_zero_records, "date"])
        impacts = []
        for split, frame in transitions[transitions.historical_frozen_grid].groupby("split"):
            origin, target = frame.date.isin(gap_dates), frame.next_date.isin(gap_dates)
            impacts.append(dict(split=split, origin_rows=int(origin.sum()), target_rows=int(target.sum()),
                either_rows=int((origin | target).sum()), original_zero_targets=int(frame.loc[target, "next_state"].eq(0).sum()),
                evidence="independent_detection_reconstruction; not_original_transition_bytes_or_predictor_mask"))
        pd.DataFrame(impacts).to_csv(OUT / "reconstructed_gap_impact.csv", index=False)
        check("all_reconstructed_gap_targets_zero", all(r["target_rows"] == r["original_zero_targets"] for r in impacts))
        after = {name: sha(ROOT / name) for name in protected}
        check("all_protected_originals_unchanged", protected == after)
        manifest = dict(date="2026-10-11", pinned_code=COMMIT, reconstruction=True,
            observation_correction=False, fitting=False, inference=False,
            threshold=dict(training_active_grid_days=len(population), Q50=q50, Q75=q75,
                population="187 study grids, active days through 2022-12-31, before predictor/cohort filtering"),
            retained=retention, all_test_endpoint_states_match=True, checks=checks,
            environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                rasterio=rasterio.__version__, GEOS=geo.lib.GEOSversion().decode()),
            audit_script_sha256=sha(Path(__file__)), original_design_sha256=hashlib.sha256(design_bytes).hexdigest(),
            protected_before=protected, protected_after=after,
            limitations=["Historical zero filling is reproduced, not endorsed as observation-valid.",
                "No original training/validation bytes, four-predictor panel, full screening or original environment recovered."],
            outputs={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "verification.json"})
        (OUT / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(checks_passed=len(checks), threshold=manifest["threshold"], retained=retention), indent=2))
    finally:
        geo.close()


if __name__ == "__main__":
    main()
