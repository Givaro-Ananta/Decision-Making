#!/usr/bin/env python3
"""Standalone, CPU-only Sumatra spatial audit. Requires DuckDB >= 1.4."""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

RESOLUTIONS = (0.1, 0.2, 0.25, 0.5)
ENV = ('temperature_t', 'precipitation_t', 'humidity_t', 'wind_speed_t', 'rain_t')
LEVELS = (10, 20, 30, 50, 100, 200)


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record(con, sql):
    cursor = con.execute(sql)
    return dict(zip([c[0] for c in cursor.description], cursor.fetchone()))


def export(con, sql, path):
    con.execute(f'COPY ({sql}) TO {literal(path)} (HEADER, DELIMITER \',\')')


def validate(con):
    columns = {r[0] for r in con.execute('DESCRIBE source').fetchall()}
    required = {'grid_id', 'date', 'centroid_lat', 'centroid_lon', 'row_index',
                'col_index', 'hotspot_count', 'frp_sum', 'frp_max'}
    if required - columns:
        raise ValueError(f'Missing required columns: {sorted(required - columns)}')
    con.execute('CREATE TABLE units AS SELECT DISTINCT grid_id, centroid_lat, centroid_lon, row_index, col_index FROM source')
    n = con.execute('SELECT count(*) FROM units').fetchone()[0]
    if not n or con.execute('SELECT count(DISTINCT grid_id) FROM units').fetchone()[0] != n:
        raise ValueError('Grid coordinates/indices are not persistent or panel is empty')
    bad = con.execute('''SELECT count(*) FROM units WHERE grid_id IS NULL OR
        centroid_lat IS NULL OR centroid_lon IS NULL OR row_index IS NULL OR col_index IS NULL OR
        abs(centroid_lat - (6.0 - row_index * 0.1)) > 1e-7 OR
        abs(centroid_lon - (95.0 + col_index * 0.1)) > 1e-7 OR
        row_index NOT BETWEEN 0 AND 120 OR col_index NOT BETWEEN 0 AND 110''').fetchone()[0]
    if bad:
        raise ValueError('Input does not match the verified ERA5-centered 0.1-degree lattice')
    con.execute('CREATE TABLE dates AS SELECT DISTINCT CAST(date AS DATE) AS date FROM source')
    d = record(con, 'SELECT count(*) AS n, min(date) AS first, max(date) AS last FROM dates')
    if d['first'] is None or (d['last'] - d['first']).days + 1 != d['n']:
        raise ValueError('Panel dates must be non-null and consecutive')
    months = con.execute("SELECT DISTINCT date_trunc('month', date)::DATE FROM dates ORDER BY 1").fetchall()
    for (month,) in months:
        where = f"date >= DATE {literal(month)} AND date < DATE {literal(month)} + INTERVAL '1 month'"
        stats = record(con, f'''SELECT count(*) AS n, count(DISTINCT CAST(date AS DATE)) AS days,
            count(*) FILTER (WHERE grid_id IS NULL OR date IS NULL OR hotspot_count IS NULL OR
                hotspot_count < 0 OR hotspot_count != floor(hotspot_count) OR frp_sum IS NULL OR
                NOT isfinite(frp_sum) OR frp_sum < 0 OR
                (hotspot_count = 0 AND (frp_sum != 0 OR frp_max IS NOT NULL)) OR
                (hotspot_count > 0 AND (frp_max IS NULL OR NOT isfinite(frp_max) OR frp_max < 0))) AS bad
            FROM source WHERE {where}''')
        dup = con.execute(f'SELECT count(*) FROM (SELECT grid_id, date FROM source WHERE {where} GROUP BY 1,2 HAVING count(*) != 1)').fetchone()[0]
        if stats['bad'] or dup or stats['n'] != n * stats['days']:
            raise ValueError(f'Invalid, duplicated, or incomplete source panel in {month}')
    if con.execute('SELECT count(*) FROM source WHERE date IS NULL').fetchone()[0]:
        raise ValueError('Null date')
    return columns, n, d, months


def audit_resolution(args, resolution, work):
    import duckdb
    con = duckdb.connect(str(work / 'audit.duckdb'))
    try:
        con.execute(f'SET memory_limit = {literal(args.memory_limit)}')
        con.execute(f'SET temp_directory = {literal(work / "spill")}')
        con.execute(f'SET threads = {args.threads}')
        con.execute('SET preserve_insertion_order = false')
        con.execute(f'CREATE VIEW source AS SELECT * FROM read_parquet({literal(args.input)})')
        columns, n_source, dates, months = validate(con)
        env = [c for c in ENV if c in columns]
        # Integer arithmetic avoids floating point ties at target boundaries.
        milli = round(resolution * 1000)
        con.execute(f'''CREATE TABLE mapping AS SELECT *,
            CAST(floor((50 + col_index * 100)::DOUBLE / {milli}) AS INTEGER) AS coarse_col,
            CAST(floor((50 + row_index * 100)::DOUBLE / {milli}) AS INTEGER) AS coarse_row
            FROM units''')
        con.execute("ALTER TABLE mapping ADD COLUMN coarse_grid_id VARCHAR")
        con.execute("UPDATE mapping SET coarse_grid_id = 'SUM_R' || coarse_row || '_C' || coarse_col")
        if con.execute('SELECT count(DISTINCT coarse_grid_id) FROM mapping').fetchone()[0] != n_source and resolution == 0.1:
            raise ValueError('Reference resolution failed to preserve base cells')
        label = f'{resolution:g}'
        export(con, 'SELECT * FROM mapping ORDER BY grid_id', args.output / f'grid_mapping_{label}.csv')
        classes = []
        if 'landcover_code' in columns:
            # Static category must be invariant per source grid; numeric codes are never averaged.
            bad = con.execute('''SELECT count(*) FROM (SELECT grid_id FROM source GROUP BY grid_id
                HAVING count(DISTINCT landcover_code) > 1 OR
                (count(landcover_code) > 0 AND count(landcover_code) != count(*)))''').fetchone()[0]
            if bad:
                raise ValueError('Land-cover categories change or have inconsistent missingness across dates')
            classes = [r[0] for r in con.execute('SELECT DISTINCT landcover_code FROM source WHERE landcover_code IS NOT NULL ORDER BY 1').fetchall()]
            if any(not math.isfinite(float(c)) or float(c) != int(c) or int(c) < 0 for c in classes):
                raise ValueError('Land-cover category must be a nonnegative integer code')
        aggregate = [
            'm.coarse_grid_id AS grid_id', 'CAST(s.date AS DATE) AS date',
            'count(*) AS n_source_cells', 'sum(s.hotspot_count)::BIGINT AS hotspot_count',
            'sum(s.frp_sum) AS frp_sum', 'max(s.frp_max) AS frp_max',
        ]
        for c in env:
            aggregate += [f'avg(s.{c}) AS {c}', f'count(s.{c}) AS {c}_n_observed']
        if 'landcover_code' in columns:
            aggregate += ['count(s.landcover_code) AS landcover_n_observed']
            aggregate += [f'avg(CASE WHEN s.landcover_code IS NULL THEN NULL ELSE (s.landcover_code = {int(c)})::INTEGER END) AS landcover_fraction_{int(c)}' for c in classes]
        select = ', '.join(aggregate)
        for index, (month,) in enumerate(months):
            sql = f'''SELECT {select} FROM source s JOIN mapping m USING (grid_id)
                WHERE s.date >= DATE {literal(month)} AND s.date < DATE {literal(month)} + INTERVAL '1 month'
                GROUP BY m.coarse_grid_id, CAST(s.date AS DATE)'''
            con.execute(('CREATE TABLE panel AS ' if index == 0 else 'INSERT INTO panel ') + sql)
        con.execute('''CREATE TABLE sparsity AS SELECT grid_id, count(*) AS total_days,
            count(*) FILTER (WHERE hotspot_count > 0) AS active_days,
            count(*) FILTER (WHERE hotspot_count = 0) AS inactive_days,
            avg((hotspot_count > 0)::INTEGER) AS active_fraction,
            sum(hotspot_count) AS total_hotspots, sum(frp_sum) AS total_frp
            FROM panel GROUP BY grid_id''')
        stats = record(con, '''SELECT count(*) AS n_grid_days,
            count(DISTINCT date) AS n_dates,
            count(*) FILTER (WHERE hotspot_count > 0) AS active_grid_days,
            sum(hotspot_count) AS total_hotspots, sum(frp_sum) AS total_frp FROM panel''')
        row = {'resolution': resolution, **stats}
        row.update(record(con, '''SELECT count(*) AS n_units,
            count(*) FILTER (WHERE active_days = 0) AS zero_event_units FROM sparsity'''))
        row['active_grid_day_pct'] = 100 * row['active_grid_days'] / row['n_grid_days']
        if row['n_grid_days'] != row['n_units'] * dates['n']:
            raise ValueError('Coarse panel completeness failed')
        original = record(con, 'SELECT sum(hotspot_count) AS total_hotspots, sum(frp_sum) AS total_frp FROM source')
        if row['total_hotspots'] != original['total_hotspots'] or not math.isclose(row['total_frp'], original['total_frp'], rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError('Coarsening failed event/FRP conservation')
        for column, quantiles in [('active_days', (0, .1, .25, .5, .75, .9, 1)), ('total_hotspots', (0, .25, .5, .75, .9, 1))]:
            for q in quantiles:
                prefix = 'min' if q == 0 else 'max' if q == 1 else 'median' if q == .5 else f'p{round(q*100)}'
                row[f'{prefix}_{column}'] = con.execute(f'SELECT quantile_cont({column}, {q}) FROM sparsity').fetchone()[0]
        for level in LEVELS:
            count = con.execute(f'SELECT count(*) FROM sparsity WHERE active_days >= {level}').fetchone()[0]
            row[f'units_ge_{level}_active_days'] = count
            row[f'pct_units_ge_{level}_active_days'] = 100 * count / row['n_units']
        intensity = []
        for column in ('hotspot_count', 'frp_sum', 'frp_max'):
            item = {'resolution': resolution, 'variable': column}
            item.update(record(con, f'SELECT count({column}) AS n, ' + ', '.join(f'quantile_cont({column}, {q}) AS p{round(100*q)}' for q in (.25, .5, .75, .9, .95, .99)) + f', max({column}) AS max FROM panel WHERE hotspot_count > 0'))
            intensity.append(item)
        # Record coverage so averaging available cells cannot conceal missing inputs.
        coverage = {'resolution': resolution, 'environmental_columns': env, 'absent_environmental_columns': [c for c in ENV if c not in env], 'landcover_category_codes': classes, 'landcover_column_available': 'landcover_code' in columns}
        for c in env:
            coverage[c] = record(con, f'SELECT sum(n_source_cells - {c}_n_observed) AS missing_source_grid_days, count(*) FILTER (WHERE {c} IS NULL) AS all_missing_coarse_grid_days FROM panel')
        if 'landcover_code' in columns:
            coverage['landcover'] = record(con, 'SELECT sum(n_source_cells - landcover_n_observed) AS missing_source_grid_days, count(*) FILTER (WHERE landcover_n_observed = 0) AS all_missing_coarse_grid_days FROM panel')
        export(con, 'SELECT * FROM sparsity ORDER BY grid_id', args.output / f'per_unit_sparsity_{label}.csv')
        if args.keep_panels:
            con.execute(f'COPY panel TO {literal(args.output / ("panel_" + label + ".parquet"))} (FORMAT PARQUET, COMPRESSION ZSTD)')
        return row, intensity, coverage, {'start_date': str(dates['first']), 'end_date': str(dates['last']), 'n_source_units': n_source, 'n_source_dates': dates['n']}
    finally:
        con.close()


def report(rows, intensities, metadata, args):
    passing = [r for r in rows if r['median_active_days'] >= args.min_median_active_days and r['pct_units_ge_20_active_days'] >= args.min_pct_units_ge_20]
    chosen = min(passing, key=lambda r: r['resolution']) if passing else None
    lines = ['# Spatial-resolution feasibility audit', '', '## Methodology', '',
        f"Input: `{args.input.name}`; {metadata['n_source_units']:,} source units, {metadata['n_source_dates']:,} consecutive dates ({metadata['start_date']} to {metadata['end_date']}). Input uniqueness, completeness, lattice coordinates, static land-cover invariance and event/FRP conservation are checked.", '',
        'Target bins share the verified northwest edge (longitude 94.95°, latitude 6.05°). Whole 0.1° cells are assigned by centroid using integer row/column arithmetic. The 0.25° bins do not nest into 0.1° cells: allocation alternates numbers of source cells and is an approximation, not exact geometric regridding. Only existing Sumatra source cells contribute. Each coarse unit is the union of its assigned source cells; full target rectangles are not added to the domain. No temporal aggregation occurs.', '',
        '## Aggregation rules', '',
        'Hotspot count and FRP sum are summed; FRP max is the maximum (null for inactive days). Available continuous environmental values are averaged equally over source cells; all-missing groups stay null. Observed-member counts retain missingness. Source cells have equal angular size, so these are equal-cell rather than exact land-area-weighted means. WorldCover codes become category shares among known source cells, not numeric-code averages or true within-cell land-cover fractions. No missing category is imputed. Existing fraction columns are not present in the verified input.', '',
        'Each resolution is processed separately in a disk-backed DuckDB database, aggregating one calendar month at a time. Temporary panels are removed after diagnostics unless `--keep-panels` is requested. Quantiles use exact continuous interpolation over active coarse grid-days; no state thresholds are selected.', '',
        '## Observed comparison', '',
        '| Resolution | Units | Active grid-days (%) | Zero-event units | Median active days | Units ≥20 (%) | Units ≥50 (%) | Units ≥100 (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['resolution']:g}° | {r['n_units']:,} | {r['active_grid_day_pct']:.3f} | {r['zero_event_units']:,} | {r['median_active_days']:.1f} | {r['pct_units_ge_20_active_days']:.1f} | {r['pct_units_ge_50_active_days']:.1f} | {r['pct_units_ge_100_active_days']:.1f} |")
    lines += ['', 'All requested counts and percentiles, including ≥10/30/200 active days, are in the comparison CSV; per-unit diagnostics and source-to-coarse mappings are exported separately.', '', '## Active-day intensity distributions', '', '| Resolution | Variable | n | p25 | p50 | p75 | p90 | p95 | p99 | max |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in intensities:
        values = ['NA' if item[k] is None else f'{item[k]:.6g}' for k in ('n', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max')]
        lines.append(f"| {item['resolution']:g}° | {item['variable']} | " + ' | '.join(values) + ' |')
    lines += ['', '## Observed sparsity', '']
    for r in rows:
        lines.append(f"- {r['resolution']:g}°: {r['zero_event_units']:,}/{r['n_units']:,} units have no events; the central 50% have {r['p25_active_days']:.1f}–{r['p75_active_days']:.1f} active days.")
    lines += ['', '## Recommendation', '', f'Exploratory screening rule: median active days ≥{args.min_median_active_days:g}, and ≥{args.min_pct_units_ge_20:g}% of units have at least 20 active days. These are configurable audit heuristics, not scientific eligibility thresholds or evidence that four-state models are estimable.']
    lines.append(f"The finest evaluated resolution passing this screen is **{chosen['resolution']:g}°**, provisionally recommended for state-definition assessment." if chosen else 'No evaluated resolution passes the screen. No resolution is recommended automatically; retain the research design for review with the observed sparsity evidence.')
    lines += ['', '## Remaining uncertainty', '', 'Active-day density does not establish sufficient observations per current state or diversity of next states. Low/Moderate/High definitions, training-period counts, transition frequencies, covariate availability, and usable complete-case samples still need assessment. Coarsening can dilute local activity and hide missing covariates. The 0.25° centroid allocation and equal-cell weighting warrant sensitivity checks. No predictive model has been fitted.', '', 'Covariate missingness and aggregation coverage are recorded in `spatial_resolution_run_metadata.json`.', '']
    (args.output / 'spatial_resolution_feasibility.md').write_text('\n'.join(lines), encoding='utf-8')
    return None if chosen is None else chosen['resolution']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resolutions', nargs='+', type=float, default=list(RESOLUTIONS))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--keep-panels', action='store_true')
    parser.add_argument('--memory-limit', default='4GB')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--min-median-active-days', type=float, default=50)
    parser.add_argument('--min-pct-units-ge-20', type=float, default=50)
    args = parser.parse_args(argv)
    args.input = args.input.resolve(); args.output = args.output.resolve()
    if not args.input.is_file():
        parser.error('Input Parquet file does not exist')
    if any(r not in RESOLUTIONS for r in args.resolutions) or len(set(args.resolutions)) != len(args.resolutions):
        parser.error('Use unique resolutions from 0.1 0.2 0.25 0.5')
    if args.threads < 1 or args.min_median_active_days < 0 or not 0 <= args.min_pct_units_ge_20 <= 100:
        parser.error('Invalid thread count or screening criteria')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and not args.overwrite:
        parser.error('Output directory is not empty; choose a new directory or use --overwrite')
    if args.input.parent == args.output or args.output in args.input.parents:
        parser.error('Output directory must not contain the source panel')
    try:
        import duckdb
    except ImportError:
        parser.error('Install DuckDB: python -m pip install "duckdb>=1.4,<2"')
    rows, intensities, coverage = [], [], []
    metadata = {}
    for resolution in sorted(args.resolutions):
        print(f'Auditing {resolution:g} degrees...', flush=True)
        with tempfile.TemporaryDirectory(prefix='spatial_audit_', dir=args.output) as scratch:
            row, detail, cov, metadata = audit_resolution(args, resolution, Path(scratch))
        rows.append(row); intensities.extend(detail); coverage.append(cov)
    write_csv(args.output / 'spatial_resolution_feasibility.csv', rows)
    write_csv(args.output / 'active_day_intensity.csv', intensities)
    recommendation = report(rows, intensities, metadata, args)
    metadata.update({'input': str(args.input), 'input_bytes': args.input.stat().st_size,
                     'duckdb_version': duckdb.__version__, 'resolutions': sorted(args.resolutions),
                     'memory_limit': args.memory_limit, 'threads': args.threads,
                     'coverage': coverage, 'recommended_resolution': recommendation,
                     'recommendation_screen': {'min_median_active_days': args.min_median_active_days, 'min_pct_units_ge_20': args.min_pct_units_ge_20}})
    (args.output / 'spatial_resolution_run_metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    print(f'Reports written to {args.output}', flush=True)


if __name__ == '__main__':
    main()
