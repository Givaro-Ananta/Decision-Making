"""Historical covariate/screen reconstruction in the REV-01 evidence directory.

No model fitting, test tuning, corrected observations or application writes.
"""
import ast
import ctypes as C
import io
import json
import subprocess
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_geom
from rasterio.windows import from_bounds
from scipy.optimize import linprog

from reconstruct_rev01_lineage import COMMIT, OUT, ROOT, Geometry, rectangle_wkb, sha

FEATURES = ["temperature_max_c_t", "relative_humidity_mean_t", "wind_speed_mean_t", "rainfall_mm_t"]


def area(geometry):
    if geometry["type"] == "GeometryCollection":
        return sum(area(g) for g in geometry["geometries"])
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    if geometry["type"] not in ("Polygon", "MultiPolygon"):
        return 0.
    value = 0.
    for polygon in polygons:
        for index, ring in enumerate(polygon):
            xy = np.asarray(ring)
            # Translate before shoelace to reduce cancellation at projected coordinates.
            xy = xy - xy[0]
            value += (1 if index == 0 else -1) * abs(np.sum(xy[:-1, 0]*xy[1:, 1] - xy[1:, 0]*xy[:-1, 1])) / 2
    return value


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    grid = pd.read_parquet(ROOT / "grid.parquet").sort_values("grid_id").reset_index(drop=True)
    geo = Geometry()
    specs = {
        "GEOSIntersection_r": (C.c_void_p, [C.c_void_p, C.c_void_p, C.c_void_p]),
        "GEOSArea_r": (C.c_int, [C.c_void_p, C.c_void_p, C.POINTER(C.c_double)]),
        "GEOSGeoJSONWriter_create_r": (C.c_void_p, [C.c_void_p]),
        "GEOSGeoJSONWriter_writeGeometry_r": (C.c_void_p, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]),
        "GEOSGeoJSONWriter_destroy_r": (None, [C.c_void_p, C.c_void_p]),
        "GEOSFree_r": (None, [C.c_void_p, C.c_void_p]),
        "GEOSPreparedContainsXY_r": (C.c_byte, [C.c_void_p, C.c_void_p, C.c_double, C.c_double]),
    }
    for name, (result, args) in specs.items():
        getattr(geo.lib, name).restype, getattr(geo.lib, name).argtypes = result, args
    writer = geo.lib.GEOSGeoJSONWriter_create_r(geo.context)
    source_files = sorted((ROOT / "data/era5").glob("*.nc")) + sorted((ROOT / "data/chirps").glob("*.tif"))
    before_stat = {p.relative_to(ROOT).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns] for p in source_files}
    window_hashes = {}
    try:
        for g in grid.itertuples():
            geo.prepared(g.study_geometry)
        domain = geo.union([geom for geom, prep in geo.objects])
        support = {g.grid_id: geo.objects[i][0] for i, g in enumerate(grid.itertuples())}
        weights = []
        for target, g in enumerate(grid.itertuples()):
            # Adjacent boundary cells are included, then screened by positive area.
            for r in range(max(0, g.row_index*5-1), min(121, (g.row_index+1)*5+1)):
                for c in range(max(0, g.col_index*5-1), min(111, (g.col_index+1)*5+1)):
                    x, y = 95+c*.1, 6-r*.1
                    wkb = rectangle_wkb([[(x-.05, y-.05), (x+.05, y-.05), (x+.05, y+.05), (x-.05, y+.05), (x-.05, y-.05)]])
                    base = geo.lib.GEOSWKBReader_read_r(geo.context, geo.reader, wkb, len(wkb))
                    overlap = geo.lib.GEOSIntersection_r(geo.context, base, support[g.grid_id])
                    geo.lib.GEOSGeom_destroy_r(geo.context, base)
                    if not overlap:
                        raise ValueError("Failed overlap")
                    a = C.c_double()
                    geo.lib.GEOSArea_r(geo.context, overlap, C.byref(a))
                    if a.value > 1e-12:
                        ptr = geo.lib.GEOSGeoJSONWriter_writeGeometry_r(geo.context, writer, overlap, 0)
                        if not ptr:
                            raise ValueError("Failed GeoJSON serialization")
                        shape = json.loads(C.string_at(ptr))
                        geo.lib.GEOSFree_r(geo.context, ptr)
                        projected = transform_geom("EPSG:4326", "EPSG:6933", shape)
                        weights.append(dict(grid_id=g.grid_id, target=target, row=r, col=c, base=r*111+c, area=area(projected)))
                    geo.lib.GEOSGeom_destroy_r(geo.context, overlap)
        weights = pd.DataFrame(weights)
        weights.to_parquet(OUT / "reconstructed_base_overlap.parquet", index=False)
        target = weights.target.to_numpy()
        source = weights.base.to_numpy()
        mass = weights.area.to_numpy()
        totals = np.bincount(target, weights=mass, minlength=187)
        check("overlap_land_area_conservation", np.max(np.abs(totals-grid.geometric_land_area_m2)/grid.geometric_land_area_m2) < 1e-3)

        def aggregate(values):
            v = values[source]
            valid = np.isfinite(v)
            denominator = np.bincount(target, weights=mass*valid, minlength=187)
            numerator = np.bincount(target, weights=mass*np.where(valid, v, 0), minlength=187)
            result = np.divide(numerator, denominator, out=np.full(187, np.nan), where=denominator>0)
            return result, denominator/totals

        records = []
        for index, path in enumerate(sorted((ROOT / "data/era5").glob("*.nc"))):
            with ExitStack() as stack:
                datasets = {v: stack.enter_context(rasterio.open(f"NETCDF:{path}:{v}")) for v in ("t2m", "d2m", "u10", "v10")}
                ref = datasets["t2m"]
                check(f"ERA_geometry_{path.stem}", (ref.width, ref.height) == (111, 121))
                bands = {}
                for b in range(1, ref.count+1):
                    day = pd.Timestamp(float(ref.tags(b)["NETCDF_DIM_valid_time"]), unit="s").normalize()
                    bands.setdefault(day, []).append(b)
                arrays = {v: s.read(masked=True).filled(np.nan) for v, s in datasets.items()}
                h = __import__("hashlib").sha256()
                for v in arrays:
                    h.update(v.encode()); h.update(arrays[v].tobytes())
                window_hashes[path.relative_to(ROOT).as_posix()] = h.hexdigest()
                for day, bs in bands.items():
                    if not pd.Timestamp("2016-09-22") <= day <= pd.Timestamp("2026-08-31"):
                        continue
                    if len(bs) != 4:
                        values = [np.full(13431, np.nan)]*3
                    else:
                        t, d, u, v = (arrays[k][np.array(bs)-1] for k in ("t2m", "d2m", "u10", "v10"))
                        tc, dc = t-273.15, d-273.15
                        rh = np.clip(100*np.exp(17.625*dc/(243.04+dc)-17.625*tc/(243.04+tc)), 0, 100)
                        # Reductions are strict over all four samples. Temperature conversion
                        # after reduction matches the original base-cell implementation.
                        values = [(np.max(t.astype("float64"), axis=0)-273.15).ravel(), np.mean(rh, axis=0, dtype="float64").ravel(), np.mean(np.hypot(u, v), axis=0, dtype="float64").ravel()]
                    frame = pd.DataFrame(dict(grid_id=grid.grid_id, date=day))
                    for f, value in zip(FEATURES[:3], values):
                        frame[f], frame[f+"_coverage"] = aggregate(value)
                    records.append(frame)
            if index % 12 == 0:
                print(f"ERA reconstruction: {index+1}/121 months", flush=True)
        weather = pd.concat(records, ignore_index=True)
        weather.to_parquet(OUT / "reconstructed_ERA_covariates.parquet", index=False)
        (OUT / "reconstructed_ERA_sources.json").write_text(json.dumps(window_hashes, indent=2)+"\n", encoding="utf-8")
        rainfall, layouts = [], {}
        unreadable = []
        for index, path in enumerate(sorted((ROOT / "data/chirps").glob("*.tif"))):
            day = pd.Timestamp(path.stem[-10:].replace(".", "-"))
            try:
                opened = rasterio.open(path)
            except rasterio.errors.RasterioIOError as error:
                # The original CHIRPS yearly builder also skips unreadable rasters;
                # retain their date with missing predictors, never impute rainfall.
                unreadable.append(dict(file=path.name, date=str(day.date()), error=str(error)))
                rainfall.append(pd.DataFrame(dict(grid_id=grid.grid_id, date=day, rainfall_mm_t=np.nan, rainfall_mm_t_coverage=0.)))
                continue
            with opened as src:
                check(f"CHIRPS_CRS_{day.date()}", src.crs.to_epsg() == 4326)
                window = from_bounds(94.95, -6.05, 106.05, 6.05, transform=src.transform).round_offsets().round_lengths()
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                key = (src.width, src.height, tuple(src.transform), tuple(window.flatten()))
                if key not in layouts:
                    rr, cc = np.indices((int(window.height), int(window.width)))
                    tr = src.window_transform(window)
                    xs, ys = tr.c+(cc.ravel()+.5)*tr.a, tr.f+(rr.ravel()+.5)*tr.e
                    br, bc = np.floor((6.05-ys)/.1).astype(int), np.floor((xs-94.95)/.1).astype(int)
                    ok = (br>=0)&(br<121)&(bc>=0)&(bc<111)
                    inside = np.zeros(len(xs), dtype=bool)
                    for j in np.flatnonzero(ok):
                        result = geo.lib.GEOSPreparedContainsXY_r(geo.context, domain, float(xs[j]), float(ys[j]))
                        if result == 2:
                            raise ValueError("CHIRPS polygon predicate failed")
                        inside[j] = result == 1
                    layouts[key] = (br*111+bc, inside)
                base_ids, inside = layouts[key]
                rain = src.read(1, window=window, masked=True).astype("float64").filled(np.nan).ravel()
                window_hashes[path.relative_to(ROOT).as_posix()] = __import__("hashlib").sha256(rain.tobytes()).hexdigest()
                valid = inside & np.isfinite(rain) & (rain != -9999)
                n = np.bincount(base_ids[valid], minlength=13431)
                sums = np.bincount(base_ids[valid], weights=rain[valid], minlength=13431)
                means = np.divide(sums, n, out=np.full(13431, np.nan), where=n>0)
                value, coverage = aggregate(means)
                rainfall.append(pd.DataFrame(dict(grid_id=grid.grid_id, date=day, rainfall_mm_t=value, rainfall_mm_t_coverage=coverage)))
            if index % 365 == 0:
                print(f"Rainfall reconstruction: {index+1}/3631 days", flush=True)
        weather = weather.merge(pd.concat(rainfall, ignore_index=True), on=["grid_id", "date"], validate="one_to_one")
        (OUT / "unreadable_CHIRPS.json").write_text(json.dumps(unreadable, indent=2)+"\n", encoding="utf-8")
        weather["predictors_available"] = np.isfinite(weather[FEATURES]).all(axis=1) & weather[[f+"_coverage" for f in FEATURES]].ge(1-1e-8).all(axis=1)
        weather.to_parquet(OUT / "reconstructed_covariates.parquet", index=False)
        panel = pd.read_parquet(OUT / "reconstructed_historical_detection_panel.parquet").merge(weather, on=["grid_id", "date"], validate="one_to_one")
        with zipfile.ZipFile(ROOT / "final_test_050_3state.zip") as archive:
            original = pd.read_parquet(io.BytesIO(archive.read("final_test_050_3state/test_predictions.parquet")))
            frozen = json.loads(archive.read("final_test_050_3state/frozen_design.json"))
        compared = original.merge(weather, on=["grid_id", "date"], suffixes=("_original", "_reconstructed"), validate="one_to_one")
        errors = {}
        for f in FEATURES:
            error = np.abs(compared[f+"_original"] - compared[f+"_reconstructed"])
            errors[f] = float(error.max())
            check(f"all_test_{f}_match", len(error) == 27968 and error.notna().all() and error.max() < 1e-7)
        check("all_test_predictors_available", compared.predictors_available.all())
        trans = pd.read_parquet(OUT / "reconstructed_detection_transitions.parquet").drop(columns="predictor_availability").merge(weather, on=["grid_id", "date"], validate="one_to_one")
        trans.to_parquet(OUT / "reconstructed_full_transitions.parquet", index=False)
        usable = trans[trans.predictors_available]
        usable[usable.historical_frozen_grid].groupby("split").size().rename("rows").to_csv(OUT / "reconstructed_frozen_split_counts.csv")
        for split, expected in [("train", 105340), ("validation", 33626), ("test", 27968)]:
            check(f"original_{split}_cohort_row_count", len(usable[usable.historical_frozen_grid & usable.split.eq(split)]) == expected)
        # Extract the pinned, side-effect-free screening functions, never its main.
        screening = subprocess.check_output(["git", "show", f"{COMMIT}:scripts/finalize_modeling_design_025.py"], cwd=ROOT)
        (OUT / "pinned_screening_source.py").write_bytes(screening)
        tree = ast.parse(screening)
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("separation", "diagnose") or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("FEATURES", "RULES") for t in n.targets)]
        namespace = dict(np=np, pd=pd, json=json, linprog=linprog)
        exec(compile(ast.Module(body=selected, type_ignores=[]), "pinned_screening", "exec"), namespace)
        diagnostics, eligible = [], []
        training = usable[usable.split.eq("train")]
        for index, gid in enumerate(grid.grid_id):
            ready = []
            for state in range(3):
                frame = training[training.grid_id.eq(gid) & training.current_state.eq(state)]
                counts = frame.next_state.value_counts()
                # A failed necessary count criterion cannot become eligible after LP.
                # For all possible passing groups execute the unchanged full diagnostic.
                if len(frame) < 50 or len(counts) == 0 or (len(counts)>1 and counts.min()<10):
                    d = dict(n_transitions=len(frame), status="excluded", reasons=["necessary_count_screen_failed; numerical_checks_not_run"], destination_counts=json.dumps({int(k):int(v) for k,v in counts.items()}))
                else:
                    d = namespace["diagnose"](frame)
                ready.append(d["status"] in ("regression_ready", "deterministic_ready"))
                diagnostics.append(dict(d, grid_id=gid, current_state=state, reasons=";".join(d["reasons"])))
            eligible.append(dict(grid_id=gid, eligible=all(ready)))
            if index % 20 == 0:
                print(f"Unchanged training screens: {index+1}/187 grids", flush=True)
        eligibility = pd.DataFrame(eligible)
        eligibility.to_csv(OUT / "reconstructed_grid_eligibility.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(OUT / "reconstructed_full_screening.csv", index=False)
        check("exact_original_46grid_eligibility", sorted(eligibility.loc[eligibility.eligible, "grid_id"]) == sorted(frozen["eligible_grid_ids"]))
        after_stat = {p.relative_to(ROOT).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns] for p in source_files}
        check("all_raw_weather_source_stats_unchanged", before_stat == after_stat)
        result = dict(date="2026-10-11", reconstruction=True, fitting=False, inference=False, correction=False,
            checks=checks, test_predictor_max_absolute_errors=errors, sources_before=before_stat, sources_after=after_stat,
            decoded_source_window_sha256=window_hashes, script_sha256=sha(Path(__file__)),
            screening_source_sha256=__import__("hashlib").sha256(screening).hexdigest(),
            limitations=["Reconstructed bytes are not original panel/transition/screening bytes.",
                "Current environment is not proven identical to original run.", "Original zero-filled labels are reproduced, not corrected.",
                "Decoded-window hashes cover read arrays; weather protection checks size/mtime, not full-file byte attestation."],
            outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ("covariate_verification.json", "verification.json")})
        (OUT / "covariate_verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(dict(checks_passed=len(checks), errors=errors, eligible=int(eligibility.eligible.sum())), indent=2))
    finally:
        geo.lib.GEOSGeoJSONWriter_destroy_r(geo.context, writer)
        geo.close()


if __name__ == "__main__":
    run()
