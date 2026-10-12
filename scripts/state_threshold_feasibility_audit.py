#!/usr/bin/env python3
"""Compare provisional four-state schemes at 0.2 and 0.25 degrees; never fit models."""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from spatial_resolution_audit import ENV, audit_resolution, export, literal, record, write_csv

RESOLUTIONS = (0.2, 0.25)
MINIMUMS = (10, 20, 30, 50)
SCHEMES = {
    'count_1_2_3plus': {
        'rule': '0: count=0; 1: count=1; 2: count=2; 3: count>=3',
        'rationale': 'Count median=1 and p75=2 at both resolutions; High begins above p75.',
        'sql': 'CASE WHEN hotspot_count=0 THEN 0 WHEN hotspot_count=1 THEN 1 WHEN hotspot_count=2 THEN 2 ELSE 3 END',
    },
    'count_1_2to4_5plus': {
        'rule': '0: count=0; 1: count=1; 2: count=2..4; 3: count>=5',
        'rationale': 'Count p90=5 at both resolutions; tests a narrower High tail.',
        'sql': 'CASE WHEN hotspot_count=0 THEN 0 WHEN hotspot_count=1 THEN 1 WHEN hotspot_count<5 THEN 2 ELSE 3 END',
    },
    'frp_sum_5_15': {
        'rule': '0: count=0; active 1: FRP sum<5; 2: 5<=FRP sum<15; 3: FRP sum>=15',
        'rationale': 'Rounded common boundaries near FRP-sum median (~5) and p75 (~13); reflects total intensity.',
        'sql': 'CASE WHEN hotspot_count=0 THEN 0 WHEN frp_sum<5 THEN 1 WHEN frp_sum<15 THEN 2 ELSE 3 END',
    },
    'frp_max_4_8': {
        'rule': '0: count=0; active 1: FRP max<4; 2: 4<=FRP max<8; 3: FRP max>=8',
        'rationale': 'Rounded common boundaries near FRP-max median (~4) and p75 (~8); reflects peak point intensity.',
        'sql': 'CASE WHEN hotspot_count=0 THEN 0 WHEN frp_max<4 THEN 1 WHEN frp_max<8 THEN 2 ELSE 3 END',
    },
}


def prepare_panels(args):
    """Reuse existing aggregation logic; cache generated coarse panels with input identity."""
    cache = args.output / 'coarse_panels'
    cache.mkdir(exist_ok=True)
    identity = {'path': str(args.input), 'bytes': args.input.stat().st_size,
                'mtime_ns': args.input.stat().st_mtime_ns}
    manifest = cache / 'source_manifest.json'
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    if previous and previous['source'] != identity:
        raise ValueError('Cached coarse panels belong to a different source; use a new output directory')
    metadata, intensities = {}, []
    for resolution in RESOLUTIONS:
        target = cache / f'panel_{resolution:g}.parquet'
        if previous and target.exists():
            print(f'Reusing {resolution:g}-degree panel', flush=True)
            metadata[str(resolution)] = previous['metadata'][str(resolution)]
            intensities.extend(previous['intensities'][str(resolution)])
            continue
        print(f'Aggregating {resolution:g}-degree panel', flush=True)
        settings = SimpleNamespace(input=args.input, output=cache, memory_limit=args.memory_limit,
                                   threads=args.threads, keep_panels=True)
        with tempfile.TemporaryDirectory(prefix='coarse_work_', dir=cache) as work:
            row, intensity, coverage, meta = audit_resolution(settings, resolution, Path(work))
        metadata[str(resolution)] = {'panel': meta, 'summary': row, 'covariates': coverage}
        intensities.extend(intensity)
    manifest.write_text(json.dumps({'source': identity, 'metadata': metadata,
        'intensities': {str(r): [i for i in intensities if i['resolution']==r] for r in RESOLUTIONS}}, indent=2))
    write_csv(args.output / 'active_day_intensity.csv', intensities)
    return cache, metadata, intensities


def make_sequence(con, scheme, env):
    """Window over a grid's complete timeline before applying date/availability filters."""
    available = ' AND '.join(f'{c} IS NOT NULL AND isfinite({c})' for c in env) or 'false'
    fully_observed = ' AND '.join(f'{c}_n_observed=n_source_cells' for c in env) or 'false'
    columns = {r[0] for r in con.execute('DESCRIBE panel').fetchall()}
    if 'landcover_n_observed' in columns:
        available += ' AND landcover_n_observed>0'
        fully_observed += ' AND landcover_n_observed=n_source_cells'
    else:
        available = fully_observed = 'false'
    # Missing requested features do not silently make a fully available transition.
    if set(env) != set(ENV):
        available = fully_observed = 'false'
    con.execute('DROP TABLE IF EXISTS sequence')
    con.execute(f'''CREATE TABLE sequence AS SELECT *,
        lead(date) OVER timeline AS next_date, lead(state) OVER timeline AS next_state
        FROM (SELECT grid_id, date, {scheme['sql']}::INTEGER AS state,
            ({available}) AS predictors_available,
            ({fully_observed}) AND ({available}) AS predictors_fully_observed FROM panel)
        WINDOW timeline AS (PARTITION BY grid_id ORDER BY date)''')
    # Never bridge a gap or filter out unavailable days before computing the target.
    con.execute('DROP VIEW IF EXISTS seq')
    con.execute('''CREATE VIEW seq AS SELECT *,
        next_date=date+INTERVAL '1 day' AS valid_transition FROM sequence''')


def diagnose_scope(con, resolution, name, scope, train_end):
    valid = 'valid_transition'
    date_filter = ''
    if scope == 'train':
        date_filter = f'WHERE date<=DATE {literal(train_end)}'
        valid += f' AND next_date<=DATE {literal(train_end)}'
    fields = ['count(*) AS n_days']
    for state in range(4):
        fields += [
            f'count(*) FILTER (WHERE state={state}) AS count_state_{state}',
            f'count(*) FILTER (WHERE state={state} AND ({valid})) AS transitions_from_state_{state}',
            f'count(DISTINCT next_state) FILTER (WHERE state={state} AND ({valid})) AS unique_next_states_from_state_{state}',
            f'count(*) FILTER (WHERE state={state} AND ({valid}) AND predictors_available) AS available_transitions_from_state_{state}',
            f'count(DISTINCT next_state) FILTER (WHERE state={state} AND ({valid}) AND predictors_available) AS available_unique_next_states_from_state_{state}',
            f'count(*) FILTER (WHERE state={state} AND ({valid}) AND predictors_fully_observed) AS fully_observed_transitions_from_state_{state}',
        ]
    con.execute('DROP TABLE IF EXISTS diagnostics')
    con.execute(f'CREATE TABLE diagnostics AS SELECT grid_id, {", ".join(fields)} FROM seq {date_filter} GROUP BY grid_id')
    if not con.execute('SELECT count(*) FROM diagnostics').fetchone()[0]:
        raise ValueError(f'No data for scope {scope}')
    header = f'{resolution} AS resolution, {literal(name)} AS scheme, {literal(scope)} AS scope'
    detail = con.execute(f'SELECT {header}, * FROM diagnostics ORDER BY grid_id')
    names = [r[0] for r in detail.description]
    per_grid = [dict(zip(names, row)) for row in detail.fetchall()]
    con.execute('DROP TABLE IF EXISTS pairs')
    con.execute(f'''CREATE TABLE pairs AS SELECT grid_id, state AS current_state,
        next_state, count(*) AS n_transitions,
        count(*) FILTER (WHERE predictors_available) AS n_available_transitions,
        count(*) FILTER (WHERE predictors_fully_observed) AS n_fully_observed_transitions
        FROM seq {date_filter + ' AND' if date_filter else 'WHERE'} ({valid}) GROUP BY 1,2,3''')
    matrix = con.execute(f'''SELECT {header}, u.grid_id, a.range AS current_state, b.range AS next_state,
        coalesce(p.n_transitions,0) AS n_transitions,
        coalesce(p.n_available_transitions,0) AS n_available_transitions,
        coalesce(p.n_fully_observed_transitions,0) AS n_fully_observed_transitions
        FROM diagnostics u CROSS JOIN range(4) a CROSS JOIN range(4) b LEFT JOIN pairs p
        ON u.grid_id=p.grid_id AND a.range=p.current_state AND b.range=p.next_state
        ORDER BY u.grid_id, current_state, next_state''')
    matrix_names = [r[0] for r in matrix.description]
    transitions = [dict(zip(matrix_names, row)) for row in matrix.fetchall()]
    # Independent accounting identities: four states partition days, and pairs agree with row counts.
    for row in per_grid:
        assert sum(row[f'count_state_{s}'] for s in range(4)) == row['n_days']
    assert sum(r['n_transitions'] for r in transitions) == sum(r[f'transitions_from_state_{s}'] for r in per_grid for s in range(4))
    summary = []
    total = len(per_grid)
    for minimum in MINIMUMS:
        r = {'resolution': resolution, 'scheme': name, 'scope': scope, 'minimum': minimum,
             'n_grids': total, 'zero_activity_grids': sum(sum(g[f'count_state_{s}'] for s in (1,2,3))==0 for g in per_grid)}
        for s in range(4):
            r[f'total_state_{s}_days'] = sum(g[f'count_state_{s}'] for g in per_grid)
            for label, column in [('observations', 'count_state'), ('transitions','transitions_from_state'), ('available_transitions','available_transitions_from_state')]:
                count = sum(g[f'{column}_{s}']>=minimum for g in per_grid)
                r[f'grids_state_{s}_{label}_ge_min'] = count
                r[f'pct_state_{s}_{label}_ge_min'] = 100*count/total
        for label, column in [('all_active_observations','count_state'), ('all_active_transitions','transitions_from_state'), ('all_active_available','available_transitions_from_state'), ('all_active_fully_observed','fully_observed_transitions_from_state')]:
            count = sum(all(g[f'{column}_{s}']>=minimum for s in (1,2,3)) for g in per_grid)
            r[f'grids_{label}_ge_min'] = count
            r[f'pct_{label}_ge_min'] = 100*count/total
        diverse = sum(all(g[f'available_transitions_from_state_{s}']>=minimum and g[f'available_unique_next_states_from_state_{s}']>=2 for s in (1,2,3)) for g in per_grid)
        r['grids_all_active_available_and_diverse_ge_min'] = diverse
        r['pct_all_active_available_and_diverse_ge_min'] = 100*diverse/total
        summary.append(r)
    return per_grid, transitions, summary


def markdown(args, summary, intensities, metadata):
    lines = ['# Provisional state and resolution feasibility audit', '',
        'Only 0.2° and 0.25° are compared. No model is fitted and no state definition is finalized.', '',
        '## Candidate definitions', '', '| Scheme | Definition | Empirical motivation |', '|---|---|---|']
    for name, scheme in SCHEMES.items():
        lines.append(f"| {name} | {scheme['rule']} | {scheme['rationale']} |")
    lines += ['', 'All candidate boundaries are common across the two resolutions. State 0 always overrides FRP when count=0. FRP values retain source units. Definitions based on count, total FRP, and maximum point FRP describe different aspects of activity; feasibility alone cannot decide their scientific meaning.', '',
        '## Active-day distributions', '', '| Resolution | Variable | n | p25 | p50 | p75 | p90 | p95 | p99 | max |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in intensities:
        lines.append(f"| {r['resolution']} | {r['variable']} | "+' | '.join(f'{r[k]:.6g}' for k in ('n','p25','p50','p75','p90','p95','p99','max'))+' |')
    lines += ['', '## Counting and transition methodology', '',
        f'Full-period counts use the existing complete daily panel. Training diagnostics use the repository cutoff **{args.train_end}**, and training transitions require both current date and next date ≤ that cutoff. These cutoff-specific results are diagnostics, not a new fixed split.', '',
        'States are assigned before LEAD is computed within each grid. Only next_date−date=1 day is accepted. The final date contributes a state observation but no outbound transition. Predictor availability refers only to the current day, never the target day. No unavailable-covariate day is removed before shifting.', '',
        'Available predictors require non-null, finite values for all five current environmental variables and at least one known WorldCover category. Fully observed additionally requires every assigned source cell to have each covariate. Existing averages over partially observed cells are retained, and their coverage is explicitly distinguished. Static land-cover fractions are constant within a grid: their independent effects cannot be estimated separately from that grid’s intercept.', '',
        'Each per-grid CSV row reports all four state-day counts, outbound transition counts, distinct observed next states, predictor-available transitions and fully-observed transitions. A separate transition matrix includes all 16 current/next pairs, even zero-count pairs.', '',
        '## Observation support for all three active states', '', '| Scope | Scheme | Resolution | Grids | ≥10 each | ≥20 each | ≥30 each | ≥50 each |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for scope in ('full','train'):
        for scheme in SCHEMES:
            for res in RESOLUTIONS:
                group=[r for r in summary if r['scope']==scope and r['scheme']==scheme and r['resolution']==res]
                counts=[]
                for minimum in MINIMUMS:
                    r=next(r for r in group if r['minimum']==minimum)
                    counts.append(f"{r['grids_all_active_observations_ge_min']} ({r['pct_all_active_observations_ge_min']:.1f}%)")
                lines.append(f"| {scope} | {scheme} | {res} | {group[0]['n_grids']} | "+' | '.join(counts)+' |')
    lines += ['', 'Individual-state coverage at each minimum is in `per_active_state_support.csv` and `state_support_summary.csv`; aggregate state balance is in `candidate_state_balance.csv`. The simultaneous requirement is stricter: a grid must pass for Low, Moderate AND High, rather than having enough active days in aggregate.', '',
        '## Predictor-available training transitions', '', '| Scheme | Resolution | ≥10 from each active state | ≥20 from each | ≥30 from each | ≥50 from each | ≥20 each AND ≥2 next states each |', '|---|---:|---:|---:|---:|---:|---:|']
    for scheme in SCHEMES:
        for res in RESOLUTIONS:
            group=[r for r in summary if r['scope']=='train' and r['scheme']==scheme and r['resolution']==res]
            counts=[]
            for minimum in MINIMUMS:
                r=next(r for r in group if r['minimum']==minimum)
                counts.append(f"{r['grids_all_active_available_ge_min']} ({r['pct_all_active_available_ge_min']:.1f}%)")
            r=next(r for r in group if r['minimum']==20)
            counts.append(f"{r['grids_all_active_available_and_diverse_ge_min']} ({r['pct_all_active_available_and_diverse_ge_min']:.1f}%)")
            lines.append(f'| {scheme} | {res} | '+' | '.join(counts)+' |')
    lines += ['', 'A one-next-state group can be represented deterministically, but cannot support estimation of covariate effects in that transition row. Observing two next states is necessary for a nontrivial regression, not sufficient for stable multinomial estimation; rare destination counts, feature rank, separation, autocorrelation and number of parameters still matter.', '',
        '## Scope and unresolved decisions', '',
        'The 0.2° cells nest into 0.1°. The 0.25° audit uses whole-cell centroid assignment, giving irregular 0.2°/0.3° supports. Only verified Sumatra source cells contribute; no cross-grid feature or newly filled target rectangle is added. Source hotspot totals and daily completeness are checked by the reused spatial aggregation routine.', '',
        'Next-state diversity across grids is summarized in `next_state_diversity_summary.csv`. 10/20/30/50 are sensitivity counts, not scientific model-eligibility guarantees. Candidate cutoffs are informed by full-period exploratory distributions; after choosing a research definition, any data-estimated final thresholds must be obtained from training data only. Resolution and threshold decisions remain open. An analyst recommendation based on these results is saved in `resolution_recommendation.md`.', '']
    (args.output/'state_threshold_feasibility.md').write_text('\n'.join(lines))



def write_support_details(output, summary, all_details):
    tidy = []
    for r in summary:
        for state in (1,2,3):
            row = {k:r[k] for k in ('resolution','scheme','scope','minimum','n_grids')}
            row['current_state'] = state
            for kind in ('observations','transitions','available_transitions'):
                row[f'grids_ge_min_{kind}'] = r[f'grids_state_{state}_{kind}_ge_min']
                row[f'pct_grids_ge_min_{kind}'] = r[f'pct_state_{state}_{kind}_ge_min']
            tidy.append(row)
    write_csv(output/'per_active_state_support.csv',tidy)
    balance=[]
    for r in summary:
        if int(r['minimum'])!=20:
            continue
        counts=[int(r[f'total_state_{state}_days']) for state in range(4)]
        row={key:r[key] for key in ('resolution','scheme','scope','n_grids')}
        for state in range(4):
            row[f'state_{state}_days']=counts[state]
            row[f'pct_all_days_state_{state}']=100*counts[state]/sum(counts)
            if state:
                row[f'pct_active_days_state_{state}']=100*counts[state]/sum(counts[1:]) if sum(counts[1:]) else None
        balance.append(row)
    write_csv(output/'candidate_state_balance.csv',balance)
    diversity = []
    for res in RESOLUTIONS:
        for scheme in SCHEMES:
            for scope in ('full','train'):
                group = [r for r in all_details if float(r['resolution'])==res and r['scheme']==scheme and r['scope']==scope]
                for state in range(4):
                    row = {'resolution':res,'scheme':scheme,'scope':scope,'current_state':state,'n_grids':len(group)}
                    for n in range(5):
                        row[f'grids_with_{n}_next_states'] = sum(int(r[f'unique_next_states_from_state_{state}'])==n for r in group)
                        row[f'available_grids_with_{n}_next_states'] = sum(int(r[f'available_unique_next_states_from_state_{state}'])==n for r in group)
                    diversity.append(row)
    write_csv(output/'next_state_diversity_summary.csv',diversity)


def recommendation(output, summary, train_end):
    def pick(scheme,res,minimum,scope='train'):
        return next(r for r in summary if r['scheme']==scheme and float(r['resolution'])==res and int(r['minimum'])==minimum and r['scope']==scope)
    dominates = all(
        float(pick(s,.25,20)['pct_all_active_available_and_diverse_ge_min']) > float(pick(s,.2,20)['pct_all_active_available_and_diverse_ge_min'])
        and int(pick(s,.25,50)['grids_all_active_available_ge_min']) > int(pick(s,.2,50)['grids_all_active_available_ge_min'])
        for s in SCHEMES)
    lines=['# Resolution recommendation from provisional state support','',
        '**Provisional recommendation: carry 0.25° forward for an eligibility-filtered per-grid four-state study.**' if dominates else '**The candidate schemes do not consistently establish a preferred resolution. Keep resolution selection open.**','',
        'This is an analyst judgment from relative temporal support, not a new numerical eligibility rule. The earlier 0.2° recommendation used total active-day density; splitting activity into three states reveals much weaker support. Neither resolution justifies fitting a four-state covariate-dependent model to every Sumatra grid. Thresholds and resolution are not finalized.','',
        f'## Evidence with the existing training cutoff ({train_end})','',
        '| Candidate | 0.2°: ≥20 usable transitions/state and ≥2 next states/state | 0.25°: same | 0.2°: ≥50 usable transitions/state | 0.25°: same |',
        '|---|---:|---:|---:|---:|']
    for s in SCHEMES:
        cells=[]
        for res in RESOLUTIONS:
            r=pick(s,res,20)
            cells.append(f"{int(r['grids_all_active_available_and_diverse_ge_min'])}/{int(r['n_grids'])} ({float(r['pct_all_active_available_and_diverse_ge_min']):.1f}%)")
        for res in RESOLUTIONS:
            r=pick(s,res,50)
            cells.append(f"{int(r['grids_all_active_available_ge_min'])}/{int(r['n_grids'])} ({float(r['pct_all_active_available_ge_min']):.1f}%)")
        lines.append(f'| {s} | '+' | '.join(cells)+' |')
    lines += ['', '## Full-period state-day support (exploratory)', '',
        '| Candidate | 0.2°: ≥20 days/state | 0.25°: same | 0.2°: ≥50 days/state | 0.25°: same |',
        '|---|---:|---:|---:|---:|']
    for candidate in SCHEMES:
        cells=[]
        for minimum in (20,50):
            for res in RESOLUTIONS:
                r=pick(candidate,res,minimum,'full')
                cells.append(f"{int(r['grids_all_active_observations_ge_min'])}/{int(r['n_grids'])} ({float(r['pct_all_active_observations_ge_min']):.1f}%)")
        lines.append(f'| {candidate} | '+' | '.join(cells)+' |')
    lines += ['',
        ('0.25° improves the training-support percentage at 20 transitions and the absolute number supported at 50 across every tested scheme in this run.' if dominates else 'Support gains are not consistent across the tested schemes; this audit does not establish a unique preferred resolution.'), 'With the planned covariates, the higher-count diagnostics deserve more weight than total active-day density. Neither resolution establishes broad support for covariate-dependent four-state estimation. A selected subset of 0.2° grids can still be studied independently, and might be preferable if spatial detail is the overriding priority. The 0.25° recommendation is a conservative tradeoff for stronger per-state samples, not a formal minimum-resolution guarantee.', 'Percentages have different denominators (997 versus 653 units in this source); they are not estimates of the land area that would be eligible. At the full-period 20-day screen, 0.2° can retain as many or more units for some schemes. A final choice also needs the eligible units’ geographic coverage.','',
        '## What this recommendation does not establish','',
        '- 20 or 50 observations does not prove stable estimation. With five dynamic covariates and four destination classes, one current-state multinomial row can require up to 18 free coefficients; destination imbalance, feature rank, separation and temporal dependence also matter.',
        '- A deterministic row is appropriate for a single observed next state, but provides no evidence for a covariate response. The audit reports such cases instead of fitting them.',
        f'- Counts are conditional on the current {train_end} training cutoff and configured covariate availability; a different chronological boundary or a scientifically justified covariate subset changes support. Missing data were not filled.',
        '- FRP-max schemes yield comparatively balanced state support, but measure peak intensity rather than total fire activity. Scientific interpretation must decide between count, FRP sum and FRP max; sample balance alone must not select thresholds.',
        '- The 0.25° grouping is an approximation using base-cell centroids. It is not yet an exact geometric 0.25° grid; resolve that limitation before calling the final spatial units exact 0.25° cells.',
        '- Static land cover does not vary within each independently fitted grid and cannot have a separately identifiable slope with that grid’s intercept.',
        '', '## Next decision', '',
        'Keep all four schemes provisional. Decide the scientific meaning of hotspot severity, review the spatial distribution of grids passing each support level, and settle a chronological training boundary and model complexity before setting eligibility criteria. Do not freeze a threshold scheme merely because it retains more grids.', '']
    (output/'resolution_recommendation.md').write_text('\n'.join(lines))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--train-end',default='2022-12-31',type=date.fromisoformat)
    parser.add_argument('--memory-limit',default='2GB')
    parser.add_argument('--threads',default=1,type=int)
    args=parser.parse_args(argv)
    args.input=args.input.resolve(); args.output=args.output.resolve()
    if not args.input.is_file() or args.threads<1:
        parser.error('Input must exist and threads must be positive')
    if args.output in args.input.parents:
        parser.error('Output must not contain the verified input')
    args.output.mkdir(parents=True,exist_ok=True)
    cache,metadata,intensities=prepare_panels(args)
    all_summary=[]
    all_details=[]
    import duckdb
    for resolution in RESOLUTIONS:
        print(f'Auditing candidate states at {resolution:g} degrees',flush=True)
        with tempfile.TemporaryDirectory(prefix='state_work_',dir=args.output) as work:
            con=duckdb.connect(str(Path(work)/'states.duckdb'))
            try:
                con.execute(f'SET memory_limit={literal(args.memory_limit)}')
                con.execute(f'SET temp_directory={literal(Path(work)/"spill")}')
                con.execute(f'SET threads={args.threads}')
                con.execute('SET preserve_insertion_order=false')
                con.execute(f'CREATE VIEW panel AS SELECT * FROM read_parquet({literal(cache/f"panel_{resolution:g}.parquet")})')
                env=[c for c in ENV if c in {r[0] for r in con.execute('DESCRIBE panel').fetchall()}]
                dates=record(con,'SELECT min(date) AS first,max(date) AS last,count(DISTINCT date) AS n FROM panel')
                if not dates['first']<=args.train_end<dates['last']:
                    raise ValueError('Training cutoff must lie inside the panel period with later dates remaining')
                detail_rows,matrix_rows=[],[]
                for name,scheme in SCHEMES.items():
                    make_sequence(con,scheme,env)
                    for scope in ('full','train'):
                        detail,matrix,summary=diagnose_scope(con,resolution,name,scope,args.train_end)
                        detail_rows.extend(detail); matrix_rows.extend(matrix); all_summary.extend(summary)
                all_details.extend(detail_rows)
                write_csv(args.output/f'per_grid_states_{resolution:g}.csv',detail_rows)
                write_csv(args.output/f'per_grid_transition_matrix_{resolution:g}.csv',matrix_rows)
            finally:
                con.close()
    write_csv(args.output/'state_support_summary.csv',all_summary)
    markdown(args,all_summary,intensities,metadata)
    write_support_details(args.output,all_summary,all_details)
    recommendation(args.output,all_summary,args.train_end)
    (args.output/'candidate_state_schemes.json').write_text(json.dumps({'status':'provisional; not finalized', 'schemes':SCHEMES,'train_end':str(args.train_end),'source':str(args.input),'memory_limit':args.memory_limit,'threads':args.threads,'aggregation':metadata},indent=2))
    print(f'Audit complete: {args.output}',flush=True)


if __name__=='__main__':
    main()
