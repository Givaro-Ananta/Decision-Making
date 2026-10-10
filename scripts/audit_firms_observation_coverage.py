#!/usr/bin/env python3
"""Read-only FIRMS record/coverage audit against saved frozen transitions.

Requires duckdb, pandas, shapely and pyarrow. Record presence is not a coverage
certificate; no target correction, metric calculation or model inference occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from shapely import prepare
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import START_DATE, END_DATE
from src.preprocessing.canonical_grid import GRID_EDGE_BOUNDS, load_mainland_sumatra_geometry
from src.preprocessing.preprocess_firms import filter_sumatra, FIRMS_FILES

NASA = 'https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms/outages.html'
NASA2022 = 'https://landweb.modaps.eosdis.nasa.gov/displayissue?id=477'
# Calendar envelopes, not assertions of full-day outage or corrected coverage.
EVENTS = [
    ('focus_2022_gap', '2022-07-27', '2022-08-10', 'documented_instrument_outage', NASA2022),
    ('focus_2026_may', '2026-05-01', '2026-05-31', 'local_gap_cause_unresolved_recovery_May1_0500UTC', NASA),
    ('focus_2026_july', '2026-07-11', '2026-07-16', 'documented_outage_partial_boundary_days', NASA),
    ('2022_boundary', '2022-07-26', '2022-07-26', 'partial_day_outage', NASA2022),
    ('2022_reprocessing', '2022-08-11', '2022-08-19', 'reprocessing_recommended_not_proof_archive_invalid', NASA2022),
    ('2023_july', '2023-07-26', '2023-07-27', 'documented_outage_partial_boundary_days', 'https://landweb.modaps.eosdis.nasa.gov/displayissue?id=480'),
    ('2024_may_june', '2024-05-25', '2024-06-04', 'geolocation_quality_and_production_hold', NASA),
    ('2024_july_gps', '2024-07-09', '2024-07-17', 'geolocation_quality_and_production_hold', NASA),
    ('2024_july_processing', '2024-07-24', '2024-07-30', 'production_hold', NASA),
    ('2024_september', '2024-09-24', '2024-09-30', 'geolocation_quality_and_production_hold', NASA),
    ('2024_november', '2024-11-02', '2024-11-08', 'geolocation_quality_and_production_hold', NASA),
    ('2025_november', '2025-11-11', '2025-11-12', 'ground_station_outage_and_recovery_partial_boundaries', NASA),
    ('2026_march_quality', '2026-03-09', '2026-03-16', 'anomaly_recovery_advisory_check_granule_QA', NASA),
    ('2026_april', '2026-04-28', '2026-04-30', 'documented_outage_and_partial_recovery', NASA),
    ('2026_june', '2026-06-01', '2026-06-10', 'outage_and_quality_caution', NASA),
    ('2026_july_quality', '2026-07-17', '2026-07-26', 'post_restart_quality_caution', NASA),
    ('2026_august', '2026-08-02', '2026-08-06', 'outage_and_quality_caution_partial_boundaries', NASA),
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def table(frame):
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(str(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None))


def impact(c, scope, flag, view='frozen_rows'):
    # Every row counted once even when both endpoints are flagged.
    return c.execute(f'''SELECT split, count(*) total_rows,
        count(*) FILTER(WHERE f0.{flag}) origin_flagged,
        count(*) FILTER(WHERE f1.{flag}) target_flagged,
        count(*) FILTER(WHERE f0.{flag} OR f1.{flag}) either_endpoint_flagged,
        count(*) FILTER(WHERE f1.{flag} AND actual_next_state=0) flagged_targets_labeled_no_activity,
        count(*) FILTER(WHERE f0.{flag} OR f1.{flag})*100.0/count(*) affected_pct
        FROM {view} r JOIN flags f0 ON CAST(r.date AS DATE)=f0.date
        JOIN flags f1 ON CAST(r.next_date AS DATE)=f1.date
        WHERE {scope} GROUP BY split ORDER BY split''').fetchdf()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/reports/firms_observation_coverage')
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Audit output already exists; use a new --output directory')
    frozen = ROOT / 'configs/final_primary_050_3state.json'
    design = json.loads(frozen.read_text())
    rawdir = ROOT / 'data/DL_FIRE_SV-C2_809701'
    trainval = ROOT / 'data/processed/state_sensitivity_050_3/transitions_train_validation.parquet'
    valpred = ROOT / 'outputs/reports/state_definition_sensitivity_050/validation_predictions_3state.parquet'
    testpred = ROOT / 'outputs/reports/final_test_050_3state/test_predictions.parquet'
    panel = ROOT / 'data/processed/modeling_050/grid_day.parquet'
    protected = sorted(set([rawdir / f for f in FIRMS_FILES] + [trainval, valpred, testpred, panel,
        ROOT / 'data/processed/sumatra_grid_day.parquet'] + list((ROOT / 'configs').glob('*.json')) +
        list((ROOT / 'outputs/models/state_sensitivity_050_3').glob('*/state_*.json')) +
        list((ROOT / 'outputs/final_artifacts').glob('*')) +
        [p for p in (ROOT / 'outputs/reports/final_test_050_3state').glob('*') if p.is_file()]))
    before = {str(p.relative_to(ROOT)): {'sha256': sha(p), 'size': p.stat().st_size,
        'mtime_ns': p.stat().st_mtime_ns} for p in protected if p.is_file()}
    out.mkdir(parents=True)
    domain = load_mainland_sumatra_geometry(ROOT).intersection(box(*GRID_EDGE_BOUNDS))
    prepare(domain)
    c = duckdb.connect()
    c.execute("SET memory_limit='1GB'; SET threads=1")
    counts = []
    inventory = []
    for name in FIRMS_FILES:
        path = rawdir / name
        raw_count = retained_count = 0
        bounds = []
        for data in pd.read_csv(path, chunksize=200000, usecols=['latitude', 'longitude', 'acq_date', 'confidence', 'frp', 'daynight', 'satellite', 'version']):
            # Count raw rows before temporal, confidence and geographic filtering.
            data['acq_date'] = pd.to_datetime(data.acq_date, errors='raise').dt.strftime('%Y-%m-%d')
            raw_count += len(data)
            bounds.extend([data.acq_date.min(), data.acq_date.max()])
            raw = data.groupby('acq_date').size().rename('raw_records')
            retained = filter_sumatra(data, domain)
            retained_count += len(retained)
            filtered = retained.groupby('acq_date').size().rename('retained_sumatra_records')
            stats = pd.concat([raw, filtered], axis=1).fillna(0).astype('int64').reset_index().rename(columns={'acq_date': 'date'})
            stats['source_file'] = name
            counts.append(stats)
        # All columns remain projected; source metadata summary is independent of polygon filtering.
        metadata = c.execute("SELECT satellite, version, count(*) n FROM read_csv_auto(?,all_varchar=true) GROUP BY ALL ORDER BY ALL", [str(path)]).fetchdf()
        inventory.append({'file': name, 'raw_records': raw_count, 'retained_sumatra_records': retained_count,
            'first_record_date': min(bounds), 'last_record_date': max(bounds), 'satellite_versions': metadata.to_dict('records')})
        print(f'Audited {name}: {raw_count:,} raw, {retained_count:,} retained', flush=True)
    file_counts = pd.concat(counts, ignore_index=True).groupby(['source_file', 'date'], as_index=False).sum()
    file_counts.to_csv(out / 'daily_records_by_file.csv', index=False)
    daily = file_counts.groupby('date')[['raw_records', 'retained_sumatra_records']].sum()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.reindex(pd.date_range(START_DATE, END_DATE)).fillna(0).astype('int64').rename_axis('date').reset_index()
    daily['raw_zero_records'] = daily.raw_records.eq(0)
    daily['sumatra_zero_retained_records'] = daily.retained_sumatra_records.eq(0)
    daily['observation_coverage_status'] = daily.raw_zero_records.map({True: 'unresolved_no_raw_records', False: 'records_present_grid_coverage_unverified'})
    daily['focus_window'] = False
    daily['documented_advisory_window'] = False
    daily['any_advisory_window'] = False
    for name, start, end, kind, source in EVENTS:
        mask = daily.date.between(start, end)
        daily.loc[mask, 'any_advisory_window'] = True
        if name.startswith('focus_'):
            daily.loc[mask, 'focus_window'] = True
        if name != 'focus_2026_may':
            daily.loc[mask, 'documented_advisory_window'] = True
    daily['audit_suspect_union'] = daily.raw_zero_records | daily.any_advisory_window
    c.register('flags', daily)
    ids = pd.DataFrame({'grid_id': design['eligible_grid_ids']})
    c.register('eligible_ids', ids)
    def pathsql(p):
        return "'" + str(p).replace("'", "''") + "'"
    c.execute(f'CREATE VIEW training_rows AS SELECT grid_id,date,next_date,current_state,next_state actual_next_state,split,eligible,predictors_available FROM read_parquet({pathsql(trainval)})')
    c.execute(f'''CREATE VIEW frozen_rows AS
        SELECT grid_id,date,next_date,current_state,actual_next_state,'train' split FROM training_rows
          WHERE split='train' AND predictors_available AND grid_id IN (SELECT grid_id FROM eligible_ids)
        UNION ALL SELECT grid_id,date,next_date,current_state,actual_next_state,'validation' FROM read_parquet({pathsql(valpred)})
        UNION ALL SELECT grid_id,date,next_date,current_state,actual_next_state,'test' FROM read_parquet({pathsql(testpred)})''')
    assert c.execute('SELECT count(*) FROM frozen_rows WHERE next_date<>date+INTERVAL 1 DAY OR grid_id NOT IN (SELECT grid_id FROM eligible_ids)').fetchone()[0] == 0
    assert c.execute('SELECT count(*) FROM (SELECT grid_id,date,split FROM frozen_rows GROUP BY ALL HAVING count(*)<>1)').fetchone()[0] == 0
    impacts = []
    for flag in ['focus_window', 'raw_zero_records', 'sumatra_zero_retained_records', 'documented_advisory_window', 'audit_suspect_union']:
        frame = impact(c, 'true', flag)
        frame.insert(0, 'flag', flag)
        impacts.append(frame)
    impact_summary = pd.concat(impacts, ignore_index=True)
    impact_summary.to_csv(out / 'frozen_transition_impact.csv', index=False)
    pergrid = c.execute('''SELECT r.split,r.grid_id,count(*) total_rows,
        count(*) FILTER(WHERE f0.audit_suspect_union OR f1.audit_suspect_union) suspect_endpoint_rows,
        count(*) FILTER(WHERE f0.raw_zero_records OR f1.raw_zero_records) raw_gap_endpoint_rows
        FROM frozen_rows r JOIN flags f0 ON CAST(r.date AS DATE)=f0.date JOIN flags f1 ON CAST(r.next_date AS DATE)=f1.date
        GROUP BY ALL ORDER BY ALL''').fetchdf()
    pergrid.to_csv(out / 'per_grid_transition_impact.csv', index=False)
    allimpact = impact(c, 'true', 'audit_suspect_union', 'training_rows')
    allimpact.to_csv(out / 'all_candidate_transition_impact.csv', index=False)
    windows = []
    event_impacts = []
    for name, start, end, kind, source in EVENTS:
        frame = daily[daily.date.between(start, end)]
        windows.append({'window': name, 'start': start, 'end': end, 'classification': kind,
            'days': len(frame), 'raw_records': int(frame.raw_records.sum()),
            'retained_sumatra_records': int(frame.retained_sumatra_records.sum()),
            'zero_raw_days': int(frame.raw_zero_records.sum()), 'source_url': source})
        c.register('window_dates', frame[['date']])
        affected = c.execute('''SELECT split,count(*) FILTER(WHERE CAST(date AS DATE) IN (SELECT date FROM window_dates)) origin_rows,
            count(*) FILTER(WHERE CAST(next_date AS DATE) IN (SELECT date FROM window_dates)) target_rows,
            count(*) FILTER(WHERE CAST(date AS DATE) IN (SELECT date FROM window_dates) OR CAST(next_date AS DATE) IN (SELECT date FROM window_dates)) either_rows,
            count(*) FILTER(WHERE CAST(next_date AS DATE) IN (SELECT date FROM window_dates) AND actual_next_state=0) target_no_activity_rows
            FROM frozen_rows GROUP BY split ORDER BY split''').fetchdf()
        affected.insert(0, 'window', name)
        event_impacts.append(affected)
    pd.DataFrame(windows).to_csv(out / 'source_advisory_windows.csv', index=False)
    event_impact = pd.concat(event_impacts, ignore_index=True)
    event_impact.to_csv(out / 'window_transition_impact.csv', index=False)
    runs = []
    group = daily.raw_zero_records.ne(daily.raw_zero_records.shift()).cumsum()
    for _, frame in daily[daily.raw_zero_records].groupby(group):
        runs.append({'start': str(frame.date.min().date()), 'end': str(frame.date.max().date()), 'days': len(frame),
            'any_documented_advisory': bool(frame.documented_advisory_window.any()), 'cause': 'requires_source_coverage_evidence'})
    pd.DataFrame(runs).to_csv(out / 'zero_raw_record_runs.csv', index=False)
    panelstats = c.execute(f'''SELECT date,count(*) grids,sum(hotspot_count) panel_hotspots,
        count(*) FILTER(WHERE hotspot_count=0) zero_hotspot_grids,
        count(*) FILTER(WHERE predictors_available) predictor_available_grids
        FROM read_parquet({pathsql(panel)}) GROUP BY date ORDER BY date''').fetchdf()
    daily = daily.merge(panelstats, on='date', validate='one_to_one')
    assert (daily.panel_hotspots == daily.retained_sumatra_records).all()
    daily.to_csv(out / 'daily_coverage_audit.csv', index=False)
    # Verify saved validation labels against stored transitions; never run inference.
    matched = c.execute(f'''SELECT count(*),count(*) FILTER(WHERE t.next_state<>v.actual_next_state OR t.current_state<>v.current_state)
        FROM read_parquet({pathsql(valpred)}) v JOIN read_parquet({pathsql(trainval)}) t
        ON v.grid_id=t.grid_id AND v.date=t.date AND v.next_date=t.next_date WHERE t.split='validation' ''').fetchone()
    assert matched == (33626, 0)
    model_support = c.execute('SELECT grid_id,current_state,count(*) n FROM frozen_rows WHERE split=\'train\' GROUP BY ALL').fetchdf()
    for row in model_support.itertuples(index=False):
        diag = json.loads((ROOT / 'outputs/models/state_sensitivity_050_3' / row.grid_id / f'state_{row.current_state}_diagnostics.json').read_text())
        assert diag['success'] and diag['n_train'] == row.n
    changed = [key for key, meta in before.items() if sha(ROOT / key) != meta['sha256'] or (ROOT / key).stat().st_mtime_ns != meta['mtime_ns']]
    assert not changed, changed
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(Path(__file__)),
        'period': [START_DATE, END_DATE], 'scope': 'frozen 0.5-degree mainland Sumatra', 'frozen_grids': len(ids),
        'inventory': inventory, 'zero_raw_days': int(daily.raw_zero_records.sum()),
        'zero_retained_sumatra_days': int(daily.sumatra_zero_retained_records.sum()),
        'protected_files': before, 'all_protected_files_unchanged': True, 'saved_validation_labels_matched': matched[0],
        'no_model_inference_fitting_or_metric_recalculation': True, 'coverage_certificate_available': False,
        'advisories_not_exhaustive': True, 'source_checked_date': '2026-10-11', 'source_windows': windows}
    (out / 'audit_manifest.json').write_text(json.dumps(manifest, indent=2))
    focus = pd.DataFrame(windows).iloc[:3][['window', 'days', 'raw_records', 'retained_sumatra_records', 'zero_raw_days']]
    report = f'''# FIRMS observation coverage audit — frozen outputs preserved

## Findings

Study dates: {START_DATE}–{END_DATE}. Frozen cohort: {len(ids)} grids. Raw archive/NRT rows are counted before spatial/confidence/date filtering; retained counts use the exact existing mainland polygon, extent and normal/high confidence filter. Raw CSVs contain detection records only, not observation footprints or per-grid coverage certificates.

{table(focus)}

Across the study period, **{manifest['zero_raw_days']} days have no raw records anywhere in the supplied files**, and {manifest['zero_retained_sumatra_days']} days have no retained Sumatra detections. These are different conditions. Neither an empty detection table nor record presence establishes complete observable grid coverage. May 2026 has no automatic sensor-outage explanation: NASA reports S-NPP safe from May 1 at 05:00 UTC. Its local gap requires download/distribution/archive provenance investigation; the cause is unresolved.

## Impact on the frozen cohort

{table(event_impact[event_impact.window.str.startswith('focus_')])}

Counts refer to saved predictor-available training transitions for the fixed 46 grids, saved validation predictions, and saved test predictions. Split follows target date. `either_rows` is a union of both endpoints, not their sum. `target_no_activity_rows` counts constructed labels potentially requiring verification; it does not count proven wrong labels. Window counts overlap and must not be summed.

{table(impact_summary[impact_summary.flag.isin(['raw_zero_records', 'audit_suspect_union'])])}

The broader suspect union combines all raw-zero dates and the listed outage/quality envelopes, including other validation/test advisories. It is a screening upper envelope, not a measured number of invalid transitions. Partial-day boundaries are included conservatively. This audit does not modify eligibility or calculate corrected performance. Candidate transitions across all 187 grids are tabulated separately; these were not all fitted.

## Source interpretation

[NASA 2022](https://landweb.modaps.eosdis.nasa.gov/displayissue?id=477) documents unrecoverable outage loss and recommends reprocessing August 11–19; that recommendation does not prove the downloaded archive lacks reprocessing. [NASA FIRMS]({NASA}) distinguishes July 2026 restart on July 16 from safe use on July 27. The CSV lists calendar envelopes for other advisories; these are not exact granule masks. One March notice has an inconsistent posting/recovery chronology, so its range is explicitly exploratory. This list is not a complete historical outage inventory.

[NASA 2023](https://landweb.modaps.eosdis.nasa.gov/displayissue?id=480) independently documents the July 26–27 gap found in validation. The [FIRMS missing-data table](https://firms.modaps.eosdis.nasa.gov/api/missing_data) was checked, but its rows failed to load in the available web response; no database-wide missing-date inventory is claimed.

## What is established and what remains unresolved

- The existing 0.5° panel totals match retained raw FIRMS detections on every date; missing detections were not introduced by the spatial aggregation. The panel fills calendar cells without separate target-observation coverage checks (`scripts/prepare_modeling_panel_025.py`).
- Both source CSVs identify SNPP, so another satellite is not supplying replacements within these files. Inspect `audit_manifest.json` for product versions and dates.
- A documented full instrument outage supports unavailable observations during the outage interval. Exact valid grid-days require acquisition footprints, cloud/valid-pixel QA and archive completeness. Partial-day loss can still produce records and underestimate intensity.
- No local download receipt, granule inventory or per-grid observation mask is established by the supplied Readme, which only links to generic documentation. `data/README.md` also leaves the FIRMS download provenance fields blank. May's cause cannot be determined from these CSVs alone. Absence of confidence-retained detections is not proof of absent observation.
- A valid next-day target transition needs verified observation support on both t and t+1. Weather publication availability is a separate unresolved operational gate; chronological splitting alone cannot certify as-of forecasting.
- Do not replace every zero with missing. Preserve legitimate observed no-detection days. No target mask or exclusion is applied by this audit.

## Decision boundary

The frozen results remain reproducible descriptions of constructed labels, but target observability is not yet certified. Documentation alone cannot resolve demonstrated source gaps; source recovery and coverage evidence come before deciding whether any versioned data correction is needed. No conclusion about required refitting or corrected scores is drawn here. Existing test results cannot be treated as an unseen test for a revised experiment.

## Verification and outputs

Raw daily counts reconcile exactly with all 0.5° panel hotspot totals. Saved validation origin/destination labels match {matched[0]:,} stored transitions. Training row counts match all 138 saved state-model fit diagnostics. Frozen rows are unique, remain in the 46 fixed grids and advance one day. SHA-256 and modification times confirm {len(before)} protected files unchanged, including raw CSVs, both panels, frozen models/configs, final reports and predictions. No inference, fitting, threshold changes, new evaluation metrics or target edits occurred.

CSV files provide daily records before/after filtering, contiguous raw-zero runs, advisory evidence, transition impacts per window/split/grid, and candidate-population counts. The manifest records input hashes and audit provenance. Reproduce with `python scripts/audit_firms_observation_coverage.py --output <new-directory>` using DuckDB, pandas, Shapely and PyArrow.
'''
    (out / 'firms_observation_coverage_audit.md').write_text(report)
    print(table(focus))
    print(table(impact_summary[impact_summary.flag.isin(['raw_zero_records', 'audit_suspect_union'])]))
    c.close()


if __name__ == '__main__':
    main()
