#!/usr/bin/env python3
"""Training-only state design, daily transitions and pre-fit diagnostics. No model fitting."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
FEATURES=['temperature_max_c_t','relative_humidity_mean_t','wind_speed_mean_t','rainfall_mm_t']
RULES={'min_current_state_transitions':50,'min_destination_count':10,'observations_per_free_parameter':5,'max_vif':10,'max_condition_number':1e8,'separation_tolerance':1e-7,'near_constant_frequency':0.95,'near_constant_unique_fraction':0.1}
def literal(x): return "'"+str(x).replace("'","''")+"'"

def separation(X,y):
    """Detect multinomial complete/quasi separation with bounded-direction LPs."""
    classes=np.unique(y); k=len(classes); p=X.shape[1]; q=(k-1)*p
    labels=np.searchsorted(classes,y)
    margins=[]
    for row,label in zip(X,labels):
        for other in range(k):
            if other==label: continue
            m=np.zeros(q)
            if label<k-1: m[label*p:(label+1)*p]+=row
            if other<k-1: m[other*p:(other+1)*p]-=row
            margins.append(m)
    M=np.asarray(margins)
    options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    result=linprog(np.r_[np.zeros(q),-1],A_ub=np.c_[-M,np.ones(len(M))],b_ub=np.zeros(len(M)),bounds=[(-1,1)]*q+[(0,1)],method='highs',options=options)
    if not result.success: return 'solver_failure'
    if result.x[-1]>RULES['separation_tolerance']: return 'complete'
    result=linprog(-M.sum(axis=0),A_ub=-M,b_ub=np.zeros(len(M)),bounds=[(-1,1)]*q,method='highs',options=options)
    if not result.success: return 'solver_failure'
    return 'quasi' if (M@result.x).max()>RULES['separation_tolerance'] else 'none'

def diagnose(frame):
    n=len(frame); counts=frame.next_state.value_counts().sort_index(); k=len(counts)
    result={'n_transitions':n,'destination_counts':json.dumps({int(a):int(b) for a,b in counts.items()}),'n_destinations':k,'status':'excluded','reasons':[]}
    if n<50: result['reasons'].append('fewer_than_50_transitions')
    if k==1:
        result.update(status='deterministic_ready' if n>=50 else 'excluded',deterministic_next_state=int(counts.index[0]),retained_features='[]')
        return result
    if k<2: result['reasons'].append('no_observed_destination'); return result
    retained=[]; removed={}; means={}; scales={}
    for f in FEATURES:
        v=frame[f]; sd=float(v.std(ddof=0)); frequency=float(v.value_counts(normalize=True).iloc[0])
        if sd<=1e-10*max(1,float(v.abs().mean())): removed[f]='constant'
        elif frequency>=.95 and v.nunique()/n<=.1: removed[f]='near_zero_variance'
        else: retained.append(f); means[f]=float(v.mean()); scales[f]=sd
    result.update(retained_features=json.dumps(retained),removed_features=json.dumps(removed),training_means=json.dumps(means),training_scales=json.dumps(scales))
    if not retained: result['reasons'].append('no_varying_covariates'); return result
    Z=np.column_stack([(frame[f]-means[f])/scales[f] for f in retained]); X=np.c_[np.ones(n),Z]
    q=(k-1)*X.shape[1]; rank=int(np.linalg.matrix_rank(X)); cond=float(np.linalg.cond(X))
    result.update(free_parameters=q,design_rank=rank,design_columns=X.shape[1],condition_number=cond)
    if counts.min()<10: result['reasons'].append('destination_count_below_10')
    if n<max(50,5*q): result['reasons'].append('observations_per_parameter_below_5')
    if rank<X.shape[1]: result['reasons'].append('rank_deficiency'); result['max_vif']=None
    else:
        vif=float(np.max(np.diag(np.linalg.inv(Z.T@Z/n))))
        result['max_vif']=vif
        if vif>10: result['reasons'].append('vif_above_10')
    if cond>1e8: result['reasons'].append('condition_number_above_1e8')
    # LP remains useful even where destination/sample screens fail; rank failure is explicit.
    sep=separation(X,frame.next_state.to_numpy()); result['separation']=sep
    if sep!='none': result['reasons'].append('separation_'+sep)
    if not result['reasons']: result['status']='regression_ready'
    return result

def main():
    import duckdb
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('data/processed/modeling_025/grid_day.parquet'))
    ap.add_argument('--output',type=Path,default=Path('data/processed/modeling_025'))
    ap.add_argument('--report',type=Path,default=Path('outputs/reports/modeling_design_025'))
    ap.add_argument('--resolution',type=float,choices=[.25,.5],default=.25)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True); args.report.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(args.output/'design.duckdb')); con.execute("SET memory_limit='2GB'; SET threads=1")
    con.execute(f'CREATE OR REPLACE VIEW panel AS SELECT * FROM read_parquet({literal(args.input)})')
    q50,q75,n=con.execute("SELECT quantile_cont(frp_sum,.5),quantile_cont(frp_sum,.75),count(*) FROM panel WHERE date<=DATE '2022-12-31' AND hotspot_count>0").fetchone()
    assert 0<q50<q75
    design={'resolution_degrees':args.resolution,'origin_west':94.95,'origin_north':6.05,'train_end':'2022-12-31','validation_end':'2024-12-31','state_meaning':'Sum of observed FIRMS detection FRP in MW; relative observed intensity, not integrated daily energy','state_boundaries':{'low_moderate':q50,'moderate_high':q75},'threshold_training_active_rows':n,'threshold_population':'All active training grid-days before predictor availability or eligibility filtering','state_rule':'0 when count=0; otherwise 1 if FRP<Q50, 2 if Q50<=FRP<Q75, 3 if FRP>=Q75','features':FEATURES,'eligibility':RULES,'eligibility_scope':'Training transitions only; all four current-state groups must pass; single-destination groups deterministic','caution':'Operational safeguards; not a guarantee of statistical sufficiency or stable parameter estimates','split_rule':'Target next_date determines train/validation/test','fit_performed':False}
    Path('configs').mkdir(exist_ok=True); Path(f'configs/modeling_design_{int(round(args.resolution*100)):03d}.json').write_text(json.dumps(design,indent=2))
    con.execute(f'''CREATE OR REPLACE TABLE states AS SELECT *,CASE WHEN hotspot_count=0 THEN 0 WHEN frp_sum<{q50!r} THEN 1 WHEN frp_sum<{q75!r} THEN 2 ELSE 3 END::TINYINT AS state FROM panel''')
    con.execute('''CREATE OR REPLACE TABLE transitions AS WITH shifted AS (SELECT *,lead(grid_id) OVER w next_grid_id,lead(date) OVER w next_date,lead(state) OVER w next_state FROM states WINDOW w AS(PARTITION BY grid_id ORDER BY date)) SELECT * EXCLUDE(state),state AS current_state,CASE WHEN next_date<=DATE '2022-12-31' THEN 'train' WHEN next_date<=DATE '2024-12-31' THEN 'validation' ELSE 'test' END AS split FROM shifted WHERE next_date=date+INTERVAL 1 DAY''')
    assert con.execute('SELECT count(*) FROM transitions WHERE grid_id<>next_grid_id OR next_date<>date+INTERVAL 1 DAY').fetchone()[0]==0
    diagnostics=[]; eligible=[]; summaries=[]
    ids=con.execute('SELECT DISTINCT grid_id FROM states ORDER BY grid_id').fetchall()
    for i,(gid,) in enumerate(ids):
        frame=con.execute(f"SELECT current_state,next_state,{','.join(FEATURES)} FROM transitions WHERE grid_id={literal(gid)} AND split='train' AND predictors_available").fetchdf()
        statuses=[]
        for state in range(4):
            result=diagnose(frame[frame.current_state==state]); result.update(grid_id=gid,current_state=state)
            statuses.append(result['status']); result['reasons']=';'.join(result['reasons']); diagnostics.append(result)
        ok=all(s in ('regression_ready','deterministic_ready') for s in statuses)
        if ok: eligible.append(gid)
        summaries.append({'grid_id':gid,'eligible':ok,'usable_training_transitions':len(frame),'current_state_statuses':json.dumps(statuses)})
        if i%50==0: print(f'Pre-fit diagnostics: {i+1}/{len(ids)}; eligible={len(eligible)}',flush=True)
    pd.DataFrame(diagnostics).to_csv(args.report/'per_grid_state_diagnostics.csv',index=False)
    eligibility=pd.DataFrame(summaries); eligibility.to_csv(args.report/'grid_eligibility.csv',index=False)
    con.register('eligibility_frame',eligibility)
    con.execute('CREATE OR REPLACE TABLE final_transitions AS SELECT t.*,e.eligible AS grid_eligible FROM transitions t JOIN eligibility_frame e USING(grid_id)')
    for name,query in [('transitions','SELECT * FROM final_transitions'),('eligible_transitions','SELECT * FROM final_transitions WHERE grid_eligible AND predictors_available'),('grid_day_states','SELECT * FROM states')]:
        con.execute(f'COPY ({query} ORDER BY grid_id,date) TO {literal(args.output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
    counts=con.execute('SELECT split,count(*),count(*) FILTER(WHERE predictors_available),count(*) FILTER(WHERE predictors_available AND grid_eligible) FROM final_transitions GROUP BY split ORDER BY split').fetchall()
    report={'n_grids':len(ids),'n_eligible_grids':len(eligible),'thresholds':[q50,q75],'training_active_grid_days':n,'split_counts':counts,'fit_performed':False,'ready_for_fitting':len(eligible)>0}
    (args.report/'audit.json').write_text(json.dumps(report,indent=2))
    failures=pd.Series([reason for d in diagnostics for reason in d['reasons'].split(';') if reason]).value_counts()
    text=f'''# Frozen {args.resolution}° modeling design and pre-fit audit\n\nResolution: exact {args.resolution}° rectangles, origin west 94.95°, north 6.05°. Study support is mainland Sumatra intersected with the original verified extent. {len(ids)} units. FIRMS points are reassigned directly; existing environmental estimates are regridded by equal-area land overlap. This differs from the exploratory centroid-grouping audit.\n\nFRP states describe summed observed detections, not daily integrated energy. No Activity means no retained detection. All active training rows through 2022-12-31, before eligibility filtering, supply Q50={q50!r} and Q75={q75!r} MW. Low <Q50; Moderate Q50–<Q75; High ≥Q75. Training population: {n:,} active grid-days.\n\nPredictors: {', '.join(FEATURES)}. Strict complete land-support coverage is required for all four. Missing days remain missing. ERA5 accumulated rainfall is excluded; CHIRPS supplies rainfall. Humidity is recomputed from matched six-hourly temperature/dewpoint then averaged. Static land cover is excluded because it cannot vary within an independently modeled grid. No spatial-neighbor predictors.\n\nEligibility uses training only: ≥50 usable transitions from each of states 0–3. Single-destination groups can be deterministic. Regression groups require ≥2 destinations, ≥10 observations per observed destination, ≥5 observations per free multinomial coefficient, at least one varying predictor, full design rank, VIF≤10, condition number≤1e8, and no complete/quasi separation detected by linear programming. Constant/near-constant predictors are removed using training data only. These are predeclared operational screens, not scientifically guaranteed sample sizes. No rules were relaxed after seeing results.\n\nEligible grids: **{len(eligible)} / {len(ids)}**. All per-state evidence is in per_grid_state_diagnostics.csv; numerical preprocessing values are recorded there. Destination classes absent in training remain an uncertainty. Passing screens does not establish stable estimates or predictive performance.\n\nSplit uses next_date: training ≤2022-12-31, validation 2023–2024, test 2025 onward. Every transition is same-grid and exactly one day. Predictor columns belong to current date only. All rows are retained in transitions.parquet; eligible_transitions.parquet additionally requires usable predictors and an eligible grid. Missing covariates are never removed before shifting.\n\nSplit counts (split, total, predictor available, eligible usable): {counts!r}.\n\nFailure counts across grid/current-state groups:\n\n{failures.to_string()}\n\nNo model was fitted. Remaining uncertainty: operational thresholds and spatial approximation, observation coverage, FRP state interpretation, rare destination support, and finite-sample stability. The original verified panel is unchanged; see preparation_audit.json.\n\nERA5 accumulation conventions: [ECMWF documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=402639006). FRP meaning: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html). Separation: [Albert and Anderson](https://doi.org/10.1093/biomet/71.1.1).\n'''
    (args.report/'modeling_design_audit.md').write_text(text)
    print(json.dumps(report,indent=2),flush=True); con.close()
if __name__=='__main__': main()
