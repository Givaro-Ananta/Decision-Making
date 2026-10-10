#!/usr/bin/env python3
"""Summarize existing pre-fit audit outputs; never fit a model."""
import argparse,json
from pathlib import Path
import pandas as pd

def summarize(directory,resolution):
    audit=json.loads((directory/'audit.json').read_text())
    diag=pd.read_csv(directory/'per_grid_state_diagnostics.csv')
    counts=diag.pivot(index='grid_id',columns='current_state',values='n_transitions')
    reasons=diag.reasons.dropna().str.split(';').explode().value_counts()
    result={'resolution':resolution,'total_grids':audit['n_grids'],'grids_ge50_every_state':int((counts>=50).all(axis=1).sum()),'grids_all_diagnostics':audit['n_eligible_grids'],'q50_frp_sum':audit['thresholds'][0],'q75_frp_sum':audit['thresholds'][1]}
    for state in range(4):
        for status in ['regression_ready','deterministic_ready']:
            result[f'{status}_state_{state}']=int(((diag.current_state==state)&(diag.status==status)).sum())
    for reason in ['fewer_than_50_transitions','destination_count_below_10','observations_per_parameter_below_5','separation_complete','separation_quasi','separation_solver_failure','vif_above_10','rank_deficiency','condition_number_above_1e8','no_observed_destination','no_varying_covariates']:
        result[reason]=int(reasons.get(reason,0))
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('outputs/reports/modeling_design_050'))
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows=[summarize(Path('outputs/reports/modeling_design_025'),.25),summarize(Path('outputs/reports/modeling_design_050'),.5)]
    pd.DataFrame(rows).to_csv(args.output/'resolution_comparison.csv',index=False)
    table='| Metric | 0.25° | 0.5° |\n|---|---:|---:|\n'+'\n'.join(f'| {key} | {rows[0][key]} | {rows[1][key]} |' for key in rows[0] if key!='resolution')
    conclusion=(f"0.5° admits {rows[1]['grids_all_diagnostics']} independently estimable candidate grids under the unchanged screens. This is support for that subset, not for all mainland Sumatra units; passing diagnostics is not proof of stable estimation or predictive accuracy." if rows[1]['grids_all_diagnostics'] else '0.5° does not supply a fully eligible grid under the unchanged screens. Coarsening has not made the intended four-current-state per-grid estimation ready for fitting.')
    text=f'''# Exact 0.25° versus 0.5° pre-fit comparison\n\n{table}\n\n## Interpretation\n\n{conclusion}\n\nBoth audits preserve mainland Sumatra intersected with the original verified extent, the same lattice origin (94.95° west, 6.05° north), daily dates, four FRP states, four dynamic predictors and strict complete land-support predictor availability. Q50/Q75 are recomputed independently from all active training grid-days through 2022-12-31 before eligibility filtering. Splits depend on target date; all transitions stay within one grid and advance exactly one day.\n\nEligibility and numerical diagnostics use the identical imported implementation: ≥50 usable training transitions per current state, ≥10 per observed destination for regression, ≥5 per free multinomial coefficient, training-only constant/near-constant removal, full rank, VIF≤10, condition number≤1e8 and no LP-detected separation. Single-destination groups with enough samples are deterministic-ready. No screens were relaxed.\n\nFailure counts above count grid × current-state groups and overlap. Separation counts include groups that already fail sample or destination support. Zero-destination groups cannot undergo separation tests; deterministic groups do not need regression diagnostics. Grids passing all diagnostics require all four current-state groups to pass.\n\n0.5° FIRMS counts/FRP are aggregated from point coordinates into exact cells. Environmental values reuse original base-cell preprocessing with equal-area land overlap, including the same corrected humidity cache. Missing observations are retained; the corrupt 2016-09-23 CHIRPS day is not imputed. Wider cells may lose complete predictor coverage when any part of their land support lacks source values.\n\nNo pooled estimation, neighbor features, model fitting or predictions were performed. Detailed evidence is in each resolution's per_grid_state_diagnostics.csv and grid_eligibility.csv. The original source panel and previous 0.25° outputs are preserved.\n\nReproduction:\n\n```bash\npython scripts/prepare_modeling_panel_025.py --resolution 0.5 --output data/processed/modeling_050 --humidity-cache data/processed/modeling_025/humidity_monthly\npython scripts/finalize_modeling_design_025.py --resolution 0.5 --input data/processed/modeling_050/grid_day.parquet --output data/processed/modeling_050 --report outputs/reports/modeling_design_050\npython scripts/compare_modeling_design_audits.py\n```\n\nThe current local environment requires `PYTHONPATH=/tmp/sumatra-audit-deps` for DuckDB commands.\n'''
    (args.output/'resolution_comparison.md').write_text(text)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
