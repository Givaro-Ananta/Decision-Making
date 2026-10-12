#!/usr/bin/env python3
"""One-shot final test evaluation of frozen three-state unweighted models.

An exclusive start marker prevents repeat evaluation. Report rebuilding may use
saved test predictions, but this command never overwrites a completed test run.
"""
import sys,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.covariate_markov import predict_log_proba
from scripts.three_state_decision_sensitivity_050 import decide
P=['p_no_activity','p_nonhigh','p_high']
RULES=[{'name':'primary_argmax','family':'argmax'}, {'name':'sensitivity_guarded_020','family':'nonhigh_guarded','threshold':.20}, {'name':'sensitivity_two_stage_020_050','family':'two_stage','threshold':.20,'high_threshold':.50}]

def metrics(y,p,logs,pred):
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=range(3),zero_division=0)
    result={'n':len(y),'accuracy':float(np.mean(y==pred)),'balanced_accuracy':float(recall[support>0].mean()),'macro_f1':float(f1.mean()),'weighted_f1':float(np.dot(f1,support)/len(y)),'log_loss':float(-logs[np.arange(len(y)),y].mean()),'multiclass_brier':float(np.square(p-np.eye(3)[y]).sum(axis=1).mean())}
    classes=[]
    for j in range(3):
        binary=y==j
        classes.append({'destination_state':j,'n_true':int(support[j]),'n_predicted':int((pred==j).sum()),'precision':float(precision[j]),'recall':float(recall[j]),'f1':float(f1[j]),'auc_ovr':float(roc_auc_score(binary,p[:,j])) if 0<binary.sum()<len(y) else None,'average_precision':float(average_precision_score(binary,p[:,j])) if 0<binary.sum()<len(y) else None})
    return result,classes,confusion_matrix(y,pred,labels=range(3))

def main():
    import duckdb
    out=ROOT/'outputs/reports/final_test_050_3state';out.mkdir(parents=True,exist_ok=True)
    if (out/'evaluation_started.json').exists():raise RuntimeError('Final test evaluation has already started. No repeat inference/evaluation is permitted; inspect saved artifacts.')
    old=ROOT/'outputs/reports/state_definition_sensitivity_050';s=json.loads((old/'sensitivity_summary.json').read_text());ids=sorted(s['eligible_grids_3state']);assert len(ids)==46 and len(s['fully_fitted_grids'])==46
    original=json.loads((ROOT/'configs/modeling_design_050.json').read_text())
    features=original['features'];models={};modelhash={}
    for gid in ids:
        for state in range(3):
            path=ROOT/'outputs/models/state_sensitivity_050_3'/gid/f'state_{state}.json';raw=path.read_bytes();model=json.loads(raw)
            assert model['grid_id']==gid and model['current_state']==state and model['train_end']=='2022-12-31' and model['high_boundary']==16.84
            assert model['type']=='multinomial_logistic' and model['penalty'] is None and 'weight_power' not in model and model['features']==features
            models[(gid,state)]=model;modelhash[str(path)]=hashlib.sha256(raw).hexdigest()
    config={'status':'final_primary_frozen_before_test','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'user_authorized_final_design':True,'resolution_degrees':.5,'grid_geometry':'Exact regular rectangles; origin west=94.95, north=6.05; mainland Sumatra intersected with verified source extent','eligible_grid_ids':ids,'n_eligible_grids':46,'state_labels':{'0':'No Activity','1':'Active / non-high','2':'High'},'state_rule':'0 if hotspot_count=0; otherwise 1 if frp_sum<16.84 MW, else 2','high_boundary_mw':16.84,'boundary_provenance':'Q75 of all active 0.5-degree training grid-days before eligibility filtering','predictors':features,'training_end':'2022-12-31','validation_target_period':['2023-01-01','2024-12-31'],'test_target_period':['2025-01-01','2026-08-31'],'eligibility':original['eligibility'],'eligibility_scope':'Every three-state current-state group must pass original screens; fixed 46 grids, no test re-screening','model':'Independent unweighted unpenalized multinomial model per grid/current state','primary_decision':'standard argmax, NumPy first index resolves exact ties','sensitivity_rules_predeclared':RULES[1:],'validation_weighting_and_threshold_experiments':'Sensitivity only; not part of primary model','test_protocol':'One inference pass; no test tuning, fitting, threshold selection, calibration or design changes','model_sha256':modelhash}
    frozen=ROOT/'configs/final_primary_050_3state.json'
    if frozen.exists():raise RuntimeError('Final freeze file already exists; do not overwrite final design')
    frozen.write_text(json.dumps(config,indent=2));frozenbytes=frozen.read_bytes();(out/'frozen_design.json').write_bytes(frozenbytes)
    # Claim the evaluation before the first test-row read.
    with (out/'evaluation_started.json').open('x') as f:json.dump({'started_at_utc':datetime.now(timezone.utc).isoformat(),'design_sha256':hashlib.sha256(frozenbytes).hexdigest(),'status':'started','sensitivity_rule_names':[r['name'] for r in RULES[1:]]},f,indent=2)
    protected=[ROOT/'data/processed/modeling_050/grid_day.parquet',old/'design_3state.json',old/'validation_predictions_3state.parquet']
    fingerprints={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected}
    c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='1GB'")
    source=ROOT/'data/processed/modeling_050/grid_day.parquet';predictions=[];coverage=[]
    for i,gid in enumerate(ids):
        # Include 2024-12-31 predictor date to evaluate the 2025-01-01 target.
        query=f'''WITH allowed AS (SELECT grid_id,date,hotspot_count,frp_sum,predictors_available,{','.join(features)},CASE WHEN hotspot_count=0 THEN 0 WHEN frp_sum<16.84 THEN 1 ELSE 2 END::TINYINT state FROM read_parquet('{source}') WHERE grid_id=? AND date BETWEEN DATE '2024-12-31' AND DATE '2026-08-31'), shifted AS (SELECT *,lead(grid_id) OVER w next_grid_id,lead(date) OVER w next_date,lead(state) OVER w next_state FROM allowed WINDOW w AS(PARTITION BY grid_id ORDER BY date)) SELECT * EXCLUDE(state),state current_state FROM shifted WHERE next_date BETWEEN DATE '2025-01-01' AND DATE '2026-08-31' AND next_date=date+INTERVAL 1 DAY'''
        data=c.execute(query,[gid]).fetchdf();assert (data.grid_id==data.next_grid_id).all()
        expected=len(pd.date_range('2025-01-01','2026-08-31'));assert len(data)==expected
        usable=data[data.predictors_available].copy();assert np.isfinite(usable[features].to_numpy()).all()
        coverage.append({'grid_id':gid,'expected_test_transitions':expected,'actual_test_transitions':len(data),'predictor_available':len(usable),'missing_predictor_rows':len(data)-len(usable)})
        for state in range(3):
            rows=usable[usable.current_state==state].copy()
            if not len(rows):continue
            model=models[(gid,state)]
            logs=predict_log_proba(model,rows)[:,:3];p=np.exp(logs)
            assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1,atol=1e-12)
            result=rows[['grid_id','date','next_date','current_state','next_state',*features]].rename(columns={'next_state':'actual_next_state'}).copy()
            result['predicted_next_state']=p.argmax(axis=1);result['log_probability_actual']=logs[np.arange(len(rows)),result.actual_next_state.to_numpy()]
            for j,name in enumerate(P):result[name]=p[:,j];result[f'log_p_{j}']=logs[:,j]
            z=np.column_stack([(rows[f]-model['means'][f])/model['scales'][f] for f in features]);result['max_abs_training_z']=np.max(np.abs(z),axis=1)
            predictions.append(result)
        print(f'Test inference once: {i+1}/{len(ids)} grids',flush=True)
    prediction=pd.concat(predictions,ignore_index=True).sort_values(['grid_id','date']).reset_index(drop=True);assert not prediction[['grid_id','date']].duplicated().any()
    y=prediction.actual_next_state.to_numpy();p=prediction[P].to_numpy();logs=prediction[[f'log_p_{j}' for j in range(3)]].to_numpy()
    pred=p.argmax(axis=1);metricsrows=[];classes=[];confusions=[];bins=[];falsepositives=[]
    groups=[('overall','ALL',-1,np.arange(len(y)))]+[('per_grid',gid,-1,np.flatnonzero(prediction.grid_id.to_numpy()==gid)) for gid in ids]+[('per_current_state','ALL',state,np.flatnonzero(prediction.current_state.to_numpy()==state)) for state in range(3)]+[('per_grid_current_state',gid,state,np.flatnonzero((prediction.grid_id.to_numpy()==gid)&(prediction.current_state.to_numpy()==state))) for gid in ids for state in range(3)]
    for rule in RULES:
        decisions=decide(p,rule['family'],rule.get('threshold'),rule.get('high_threshold'));prediction[rule['name']]=decisions
        for scope,gid,state,index in groups:
            if not len(index):continue
            values,cl,cm=metrics(y[index],p[index],logs[index],decisions[index]);meta={'variant':rule['name'],'scope':scope,'grid_id':gid,'current_state':state};metricsrows.append(dict(meta,**values));classes.extend(dict(meta,**r) for r in cl)
            for a in range(3):
                for b in range(3):confusions.append(dict(meta,actual_state=a,predicted_state=b,count=int(cm[a,b])))
        tp=int(((y==1)&(decisions==1)).sum());fp=int(((y!=1)&(decisions==1)).sum());ref_tp=int(((y==1)&(pred==1)).sum());ref_fp=int(((y!=1)&(pred==1)).sum());delta_tp=tp-ref_tp
        falsepositives.append({'variant':rule['name'],'nonhigh_tp':tp,'nonhigh_fp':fp,'delta_nonhigh_tp':delta_tp,'delta_nonhigh_fp':fp-ref_fp,'extra_nonhigh_fp_per_extra_tp':(fp-ref_fp)/delta_tp if delta_tp>0 else None,'delta_no_activity_false_active':int(((y==0)&(decisions!=0)).sum()-((y==0)&(pred!=0)).sum()),'delta_high_tp':int(((y==2)&(decisions==2)).sum()-((y==2)&(pred==2)).sum())})
    # Descriptive diagnostics only: no fitted calibrator, adjustment or clipping.
    calibration=[]
    for scope,gid,state,index in groups:
        if not len(index):continue
        for j in range(3):
            probabilities=p[index,j];binids=np.minimum((probabilities*10).astype(int),9)
            for binid in range(10):
                mask=binids==binid
                if mask.any():bins.append({'scope':scope,'grid_id':gid,'current_state':state,'destination_state':j,'bin':binid,'n':int(mask.sum()),'mean_probability':float(probabilities[mask].mean()),'observed_fraction':float((y[index][mask]==j).mean())})
    binframe=pd.DataFrame(bins);overallbins=binframe[binframe.scope=='overall']
    for j in range(3):
        b=overallbins[overallbins.destination_state==j]
        calibration.append({'destination_state':j,'mean_probability':float(p[:,j].mean()),'observed_fraction':float((y==j).mean()),'mean_minus_observed':float(p[:,j].mean()-(y==j).mean()),'ece_10_equal_width_bins':float(np.sum(b.n*np.abs(b.mean_probability-b.observed_fraction))/b.n.sum())})
    confidence=p.max(axis=1);idsbin=np.minimum((confidence*10).astype(int),9);topbins=[]
    for b in range(10):
        mask=idsbin==b
        if mask.any():topbins.append({'bin':b,'n':int(mask.sum()),'mean_confidence':float(confidence[mask].mean()),'accuracy':float((pred[mask]==y[mask]).mean())})
    top=pd.DataFrame(topbins);top_ece=float(np.sum(top.n*np.abs(top.mean_confidence-top.accuracy))/len(y))
    prediction.to_parquet(out/'test_predictions.parquet',index=False);pd.DataFrame(coverage).to_csv(out/'test_coverage.csv',index=False);m=pd.DataFrame(metricsrows);m.to_csv(out/'test_metrics.csv',index=False);pd.DataFrame(classes).to_csv(out/'test_state_metrics_auc.csv',index=False);pd.DataFrame(confusions).to_csv(out/'test_confusion_matrices.csv',index=False);pd.DataFrame(falsepositives).to_csv(out/'sensitivity_false_positive_costs.csv',index=False)
    binframe.to_csv(out/'test_reliability_bins.csv',index=False);pd.DataFrame(calibration).to_csv(out/'test_calibration_in_the_large.csv',index=False);top.to_csv(out/'test_top_label_reliability.csv',index=False)
    prediction[prediction.log_probability_actual<np.log(.01)].to_csv(out/'test_true_probability_below_1pct.csv',index=False)
    primary=m[(m.variant=='primary_argmax')&(m.scope=='overall')].iloc[0].to_dict();grid=m[(m.variant=='primary_argmax')&(m.scope=='per_grid')];grid[['accuracy','balanced_accuracy','macro_f1','weighted_f1','log_loss']].agg(['mean','median','std','min','max']).to_csv(out/'test_across_grid_summary.csv')
    result={'evaluation_status':'completed_once','completed_at_utc':datetime.now(timezone.utc).isoformat(),'n_test_observations':len(y),'n_test_grids':prediction.grid_id.nunique(),'target_start':str(prediction.next_date.min().date()),'target_end':str(prediction.next_date.max().date()),'expected_transitions':sum(r['expected_test_transitions'] for r in coverage),'excluded_missing_predictors':sum(r['missing_predictor_rows'] for r in coverage),'primary_metrics':primary,'probability_min':float(p.min()),'probability_max':float(p.max()),'max_probability_sum_error':float(np.abs(p.sum(axis=1)-1).max()),'nonfinite_true_log_probabilities':int((~np.isfinite(prediction.log_probability_actual)).sum()),'true_probability_below_1pct':int((prediction.log_probability_actual<np.log(.01)).sum()),'rows_max_abs_training_z_ge3':int((prediction.max_abs_training_z>=3).sum()),'classwise_calibration':calibration,'top_label_ece_10_bins':top_ece,'no_refit_or_design_change':True,'models_unchanged':True,'frozen_design_sha256':hashlib.sha256(frozenbytes).hexdigest()}
    for name,digest in modelhash.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    for name,stat in fingerprints.items():assert (Path(name).stat().st_size,Path(name).stat().st_mtime_ns)==stat
    assert frozen.read_bytes()==frozenbytes
    (out/'test_summary.json').write_text(json.dumps(result,indent=2));write_report(out)
    (out/'evaluation_completed.json').write_text(json.dumps({'completed_at_utc':result['completed_at_utc'],'design_sha256':result['frozen_design_sha256'],'status':'completed_once'},indent=2))
    print(json.dumps(result,indent=2),flush=True);c.close()

def write_report(out):
    s=json.loads((out/'test_summary.json').read_text());m=pd.read_csv(out/'test_metrics.csv');cl=pd.read_csv(out/'test_state_metrics_auc.csv');cm=pd.read_csv(out/'test_confusion_matrices.csv');cal=pd.read_csv(out/'test_calibration_in_the_large.csv');fp=pd.read_csv(out/'sensitivity_false_positive_costs.csv')
    def table(f):return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))
    matrix=cm[(cm.variant=='primary_argmax')&(cm.scope=='overall')].pivot(index='actual_state',columns='predicted_state',values='count')
    text=f'''# Final one-shot test evaluation: frozen three-state 0.5° primary model\n\nThe final primary design was frozen before reading test observations in configs/final_primary_050_3state.json. It uses exactly 46 eligibility-approved grids and 138 existing unweighted multinomial transition models fitted only on training data through 2022-12-31. No fitting, re-screening, feature/state changes, weighting, calibration or tuning occurred. Standard argmax remains the primary decision rule. FRP High boundary remains the training-only Q75=16.84 MW.\n\nAn exclusive evaluation-start marker blocks rerunning inference/evaluation. The first target day 2025-01-01 uses predictor day 2024-12-31; all shifts occur within a grid and require a one-day gap. All available target dates through 2026-08-31 are included.\n\n## Primary results\n\n{s['n_test_observations']:,} usable test observations in {s['n_test_grids']} grids, targets {s['target_start']} through {s['target_end']}. Expected transitions {s['expected_transitions']:,}; missing-predictor exclusions {s['excluded_missing_predictors']}. Missing dates/zeros are retained before shift and covariate filtering.\n\n{table(m[(m.scope=='overall')&(m.variant=='primary_argmax')][['accuracy','balanced_accuracy','macro_f1','weighted_f1','log_loss','multiclass_brier']].round(6))}\n\n{table(cl[(cl.scope=='overall')&(cl.variant=='primary_argmax')][['destination_state','n_true','n_predicted','precision','recall','f1','auc_ovr','average_precision']].round(6))}\n\nStates: 0 No Activity; 1 Active/non-high; 2 High. Confusion matrix rows=actual, columns=predicted:\n\n```\n{matrix.to_string()}\n```\n\nPer-grid metrics, per-grid/current-state metrics and classwise AUC are in test_metrics.csv and test_state_metrics_auc.csv. Across-grid mean/median/std/min/max are in test_across_grid_summary.csv. Macro F1 uses the fixed three-label set with zero divisions set to zero; balanced accuracy averages recall over destinations present in each subset. Undefined AUCs are blank when a subgroup has no positive/negative contrast. Log loss uses stable model log probabilities with no clipping; multiclass Brier sums class squared errors, range 0–2. Pooled evaluation summarizes independent predictions; no pooled fitting.\n\n## Probability and calibration diagnostics\n\n{table(cal.round(6))}\n\nMinimum probability {s['probability_min']}; maximum {s['probability_max']}; maximum probability-vector sum error {s['max_probability_sum_error']}; nonfinite true-class log probabilities {s['nonfinite_true_log_probabilities']}. True-class probability below 1% occurs {s['true_probability_below_1pct']} times; detailed rows are in test_true_probability_below_1pct.csv. {s['rows_max_abs_training_z_ge3']} rows contain a predictor at least 3 training SD from its state-specific training mean. These are diagnostics only and do not exclude observations or alter predictions.\n\nTop-label ECE with ten equal-width bins: {s['top_label_ece_10_bins']}. Per-class and top-label reliability bins with sample counts are saved. Mean probability versus observed frequency and ECE are descriptive finite-sample checks; they do not establish calibration. Log loss and Brier reflect discrimination and calibration together. No calibrator was fitted. Calibration methodology: [scikit-learn documentation](https://scikit-learn.org/stable/modules/calibration.html).\n\n## Previously defined sensitivities only\n\nOnly guarded p_nonhigh≥0.20 with p_nonhigh≥p_high, and two-stage p_active≥0.20 followed by conditional p_high≥0.50, were frozen before the test read. They reuse exactly the primary probabilities; no threshold sweep, weighting or test-driven selection. The standard argmax primary output is retained regardless of sensitivity performance.\n\n{table(m[m.scope=='overall'][['variant','accuracy','balanced_accuracy','macro_f1','weighted_f1','log_loss']].round(6))}\n\n{table(cl[cl.scope=='overall'][['variant','destination_state','precision','recall','f1','auc_ovr']].round(6))}\n\n{table(fp.round(5))}\n\nSensitivity log loss and AUC are identical to primary because only hard decisions change. All three confusion matrices and per-grid results are supplied. Net false positives combine true No Activity/High labeled non-high; separate No Activity false-active and High TP changes expose other costs. None of these test results changes the final primary model or decision rule.\n\n## Frozen artifacts and one-shot status\n\nAll 138 source model hashes and the frozen design were checked unchanged after evaluation. Source panel and validation artifact size/mtime were unchanged. Predictions are in test_predictions.parquet with primary and sensitivity decisions plus original probabilities and logs. Final status: completed once. Do not rerun the evaluation command to tune or improve test performance. Further descriptive reporting can use these saved predictions without refitting or performing another test inference.\n'''
    (out/'final_test_report.md').write_text(text)
if __name__=='__main__':main()
