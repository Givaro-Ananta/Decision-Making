#!/usr/bin/env python3
"""Frozen-grid three-state sensitivity: training/validation only, unchanged screens."""
import sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,roc_auc_score,average_precision_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('prefit',ROOT/'scripts/finalize_modeling_design_025.py');prefit=importlib.util.module_from_spec(spec);spec.loader.exec_module(prefit)
from src.models.covariate_markov import fit_transition,predict_log_proba
FEATURES=prefit.FEATURES

def state3(count,frp,boundary):
    return np.where(np.asarray(count)==0,0,np.where(np.asarray(frp)<boundary,1,2))

def score(y,logp,nstates):
    p=np.exp(logp);pred=p.argmax(axis=1)
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=range(nstates),zero_division=0)
    result={'n':len(y),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1.mean()),'log_loss':float(-logp[np.arange(len(y)),y].mean())}
    classes=[]
    for j in range(nstates):
        binary=np.asarray(y)==j
        classes.append({'destination_state':j,'n_true':int(support[j]),'recall':float(recall[j]),'f1':float(f1[j]),'precision':float(precision[j]),'roc_auc':float(roc_auc_score(binary,p[:,j])) if 0<binary.sum()<len(y) else np.nan,'average_precision':float(average_precision_score(binary,p[:,j])) if 0<binary.sum()<len(y) else np.nan})
    return result,classes,confusion_matrix(y,pred,labels=range(nstates))

def main():
    import duckdb
    out=ROOT/'outputs/reports/state_definition_sensitivity_050';out.mkdir(parents=True,exist_ok=True)
    dataout=ROOT/'data/processed/state_sensitivity_050_3';dataout.mkdir(parents=True,exist_ok=True)
    models=ROOT/'outputs/models/state_sensitivity_050_3';models.mkdir(parents=True,exist_ok=True)
    frozen=ROOT/'configs/modeling_design_050.json';design=json.loads(frozen.read_text());assert design['features']==FEATURES and design['eligibility']==prefit.RULES
    protected=[frozen,ROOT/'configs/model_fitting_050.json',ROOT/'data/processed/modeling_050/grid_day.parquet',ROOT/'outputs/reports/markov_050_validation/validation_predictions.parquet']+list((ROOT/'outputs/models/markov_050').glob('*/state_*.json'))
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected}
    c=duckdb.connect(str(dataout/'sensitivity.duckdb'));c.execute("SET threads=1;SET memory_limit='1GB'")
    source=ROOT/'data/processed/modeling_050/grid_day.parquet'
    # Materialize only permitted days before window operations, including no-activity/missing rows.
    c.execute(f"CREATE OR REPLACE TABLE allowed_panel AS SELECT grid_id,date,hotspot_count,frp_sum,predictors_available,{','.join(FEATURES)} FROM read_parquet('{source}') WHERE date<=DATE '2024-12-31'")
    boundary,nactive=c.execute("SELECT quantile_cont(frp_sum,.75),count(*) FROM allowed_panel WHERE date<=DATE '2022-12-31' AND hotspot_count>0").fetchone();assert boundary==16.84
    c.execute(f'''CREATE OR REPLACE TABLE transitions AS WITH states AS(SELECT *,CASE WHEN hotspot_count=0 THEN 0 WHEN frp_sum<{boundary!r} THEN 1 ELSE 2 END::TINYINT state FROM allowed_panel), shifted AS(SELECT *,lead(grid_id) OVER w next_grid_id,lead(date) OVER w next_date,lead(state) OVER w next_state FROM states WINDOW w AS(PARTITION BY grid_id ORDER BY date)) SELECT * EXCLUDE(state),state AS current_state,CASE WHEN next_date<=DATE '2022-12-31' THEN 'train' ELSE 'validation' END split FROM shifted WHERE next_date=date+INTERVAL 1 DAY''')
    assert c.execute("SELECT count(*) FROM transitions WHERE next_grid_id<>grid_id OR next_date<>date+INTERVAL 1 DAY OR next_date>DATE '2024-12-31'").fetchone()[0]==0
    ids=[r[0] for r in c.execute('SELECT DISTINCT grid_id FROM transitions ORDER BY grid_id').fetchall()];diagnostics=[];eligibility=[];predictions=[];fitrows=[]
    for i,gid in enumerate(ids):
        frame=c.execute('SELECT * FROM transitions WHERE grid_id=? AND predictors_available',[gid]).fetchdf();train=frame[frame.split=='train'];val=frame[frame.split=='validation'];group=[]
        for state in range(3):
            result=prefit.diagnose(train[train.current_state==state]);result.update(grid_id=gid,current_state=state);group.append(result);diagnostics.append(dict(result,reasons=';'.join(result['reasons'])))
        eligible=all(d['status'] in ('regression_ready','deterministic_ready') for d in group)
        eligibility.append({'grid_id':gid,'eligible':eligible,'n_train':len(train),'n_validation':len(val),'ge50_every_state':all(d['n_transitions']>=50 for d in group)})
        if eligible:
            for state,d in enumerate(group):
                subset=train[train.current_state==state]
                features=json.loads(d['retained_features']);means=json.loads(d.get('training_means','{}'));scales=json.loads(d.get('training_scales','{}'))
                if d['status']=='deterministic_ready': model={'type':'deterministic','next_state':d['deterministic_next_state']};fit={'success':True,'iterations':0}
                else:model,fit=fit_transition(subset,features,means,scales)
                fit.update(grid_id=gid,current_state=state,n_train=len(subset));fitrows.append(fit)
                folder=models/gid;folder.mkdir(exist_ok=True);(folder/f'state_{state}_diagnostics.json').write_text(json.dumps(fit,indent=2))
                if not fit['success']:continue
                model.update(grid_id=gid,current_state=state,n_states=3,train_end='2022-12-31',high_boundary=boundary)
                (folder/f'state_{state}.json').write_text(json.dumps(model,indent=2))
                v=val[val.current_state==state].copy()
                if len(v):
                    logs=predict_log_proba(model,v)[:,:3];prob=np.exp(logs);assert np.allclose(prob.sum(axis=1),1,atol=1e-12)
                    v=v[['grid_id','date','next_date','current_state','next_state']].rename(columns={'next_state':'actual_next_state'})
                    v['predicted_next_state']=prob.argmax(axis=1);v['log_probability_actual']=logs[np.arange(len(v)),v.actual_next_state.to_numpy()]
                    for j in range(3):v[f'p_{j}']=prob[:,j];v[f'log_p_{j}']=logs[:,j]
                    predictions.append(v)
        if i%40==0:print(f'Screened {i+1}/{len(ids)} grids; eligible={sum(d["eligible"] for d in eligibility)}',flush=True)
    pd.DataFrame(diagnostics).to_csv(out/'per_grid_state_diagnostics.csv',index=False);e=pd.DataFrame(eligibility);e.to_csv(out/'grid_eligibility.csv',index=False);fit=pd.DataFrame(fitrows);fit.to_csv(out/'fit_diagnostics.csv',index=False)
    c.register('eligibility',e);c.execute(f"COPY (SELECT t.*,e.eligible FROM transitions t JOIN eligibility e USING(grid_id) ORDER BY grid_id,date) TO '{dataout/'transitions_train_validation.parquet'}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    prediction=pd.concat(predictions,ignore_index=True);prediction.to_parquet(out/'validation_predictions_3state.parquet',index=False)
    original=pd.read_parquet(ROOT/'outputs/reports/markov_050_validation/validation_predictions.parquet')
    original_ids=set(original.grid_id.unique());new_ids=set(prediction.grid_id.unique());common=sorted(original_ids&new_ids)
    allnew=sorted(new_ids);successfulgrids=sorted(fit.groupby('grid_id').success.all().loc[lambda x:x].index)
    cohorts=[('four_state_native_all_eligible',original,4),('three_state_all_eligible',prediction,3),('four_state_native_common',original[original.grid_id.isin(common)],4),('three_state_common',prediction[prediction.grid_id.isin(common)],3)]
    # Matched three-label task: collapse saved four-state probabilities, without changing/refitting originals.
    collapsed=original[original.grid_id.isin(common)].copy();collapsed['actual_next_state']=collapsed.actual_next_state.map({0:0,1:1,2:1,3:2});collapsed['current_state']=collapsed.current_state.map({0:0,1:1,2:1,3:2})
    collapsed['p_0']=collapsed.p_no_activity;collapsed['p_1']=collapsed.p_low+collapsed.p_moderate;collapsed['p_2']=collapsed.p_high
    for j in range(3):collapsed[f'log_p_{j}']=np.log(collapsed[f'p_{j}'])
    cohorts.append(('four_state_collapsed_common',collapsed,3))
    rows=[];classrows=[];cms=[]
    for cohort,frame,k in cohorts:
        if not len(frame):continue
        if k==4:
            p=frame[['p_no_activity','p_low','p_moderate','p_high']].to_numpy();logs=np.log(p)
        else:logs=frame[[f'log_p_{j}' for j in range(3)]].to_numpy()
        frame=frame.copy()
        for j in range(k):frame[f'eval_log_p_{j}']=logs[:,j]
        groups=[('overall','ALL',-1,frame)]+[('per_grid',g,-1,v) for g,v in frame.groupby('grid_id')]+[('per_current_state','ALL',int(s),v) for s,v in frame.groupby('current_state')]+[('per_grid_current_state',g,int(s),v) for (g,s),v in frame.groupby(['grid_id','current_state'])]
        for scope,gid,state,g in groups:
            m,classes,cm=score(g.actual_next_state.to_numpy(),g[[f'eval_log_p_{j}' for j in range(k)]].to_numpy(),k);meta={'cohort':cohort,'scope':scope,'grid_id':gid,'current_state':state,'n_states':k,'n_grids':g.grid_id.nunique()};rows.append(dict(meta,**m))
            for cl in classes:classrows.append(dict(meta,**cl))
            for a in range(k):
                for b in range(k):cms.append(dict(meta,actual_state=a,predicted_state=b,count=int(cm[a,b])))
    pd.DataFrame(rows).to_csv(out/'validation_metrics_comparison.csv',index=False);pd.DataFrame(classrows).to_csv(out/'validation_state_metrics_auc.csv',index=False);pd.DataFrame(cms).to_csv(out/'validation_confusions.csv',index=False)
    expected_old=original[original.grid_id.isin(common)].sort_values(['grid_id','date']);expected_new=prediction[prediction.grid_id.isin(common)].sort_values(['grid_id','date']);assert expected_old[['grid_id','date','next_date']].reset_index(drop=True).equals(expected_new[['grid_id','date','next_date']].reset_index(drop=True))
    summary={'boundary_training_q75':boundary,'training_active_days_threshold_population':nactive,'total_grids':len(ids),'eligible_grids_3state':e.loc[e.eligible,'grid_id'].tolist(),'ge50_every_state_3state':int(e.ge50_every_state.sum()),'fit_successful_state_models':int(fit.success.sum()),'fit_failed_state_models':int((~fit.success).sum()),'fully_fitted_grids':successfulgrids,'eligible_grids_4state':sorted(original_ids),'common_grids':common,'validation_rows_3state':len(prediction),'validation_rows_common':len(expected_new),'test_used':False,'original_artifacts_unchanged':True,'rules':prefit.RULES,'features':FEATURES,'current_state_conditioning':'Merged current Low/Moderate too; same-grid independent state-specific multinomial models','protected_sha256':hashes}
    for f,value in hashes.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==value
    (out/'sensitivity_summary.json').write_text(json.dumps(summary,indent=2));(out/'design_3state.json').write_text(json.dumps({'original_design':design,'state_labels':{'0':'No Activity','1':'Active / non-high','2':'High'},'high_boundary':boundary,'boundary_source':'Training-only active grid-day Q75, before eligibility filtering','frozen_primary_design_changed':False},indent=2))
    write_report(out)
    print(json.dumps({k:v for k,v in summary.items() if k!='protected_sha256'},indent=2),flush=True);c.close()

def write_report(out):
    summary=json.loads((out/'sensitivity_summary.json').read_text());m=pd.read_csv(out/'validation_metrics_comparison.csv');s=pd.read_csv(out/'validation_state_metrics_auc.csv')
    overall=m[m.scope=='overall'];states=s[s.scope=='overall']
    def table(f):return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))
    text=f'''# State-definition sensitivity at frozen 0.5°\n\nOnly the state partition is varied. Three states: 0 no retained detection; 1 active with FRP sum <16.84 MW; 2 active with FRP sum ≥16.84 MW. The boundary is recomputed as the active training grid-day Q75 through 2022-12-31 before eligibility filtering ({summary['training_active_days_threshold_population']:,} rows), and equals the original High boundary exactly. Both origin and destination Low/Moderate states are merged. No threshold search was performed.\n\nThe exact grid, mainland support, four predictors, strict predictor availability, split and eligibility/diagnostic implementation are unchanged. Each of the three current-state groups must pass the same ≥50 sample, ≥10 destination, ≥5/free-parameter, variance/rank/VIF/condition-number and separation screens. The number of states/classes changes parameter count algebraically; numerical cutoffs are not relaxed. Training-only scaling and unpenalized state-specific multinomial MLE use the original optimizer and fitting checks.\n\n## Eligibility and fitting\n\nTotal grids: {summary['total_grids']}. Three-state grids with ≥50 usable transitions from each state: {summary['ge50_every_state_3state']}; fully eligible: {len(summary['eligible_grids_3state'])}, versus 11 for four states. State fits successful: {summary['fit_successful_state_models']}; failed: {summary['fit_failed_state_models']}; fully fitted grids: {len(summary['fully_fitted_grids'])}.\n\nEligible three-state grid IDs:\n\n{', '.join(summary['eligible_grids_3state'])}\n\nCommon cohort: {len(summary['common_grids'])} grids, {summary['validation_rows_common']:,} validation rows, matched exactly by grid/date/target date. All-eligible three-state validation rows: {summary['validation_rows_3state']:,}.\n\n## Validation comparison\n\n{table(overall[['cohort','n_states','n_grids','n','balanced_accuracy','macro_f1','log_loss']])}\n\n{table(states[['cohort','destination_state','n_true','recall','f1','roc_auc','average_precision']])}\n\nNative four-state versus three-state metrics measure different target partitions. Their macro F1 and log loss cannot alone demonstrate better predictors: removing a split reduces target entropy and changes macro averaging. All-eligible cohorts also differ in difficulty. The common-cohort comparison controls spatial/sample composition. `four_state_collapsed_common` sums existing Low+Moderate probabilities and relabels both observed origin/destination for evaluation only; original fitted models and decision outputs are not overwritten. Its three-way argmax is solely a matched-task sensitivity diagnostic. Comparing it with the refitted three-state model controls the destination partition, though the latter also merges conditioning groups. It is not a new baseline algorithm.\n\nPer-grid and per-current-state recall/F1/AUC, confusion matrices and sample sizes are supplied in the CSVs. Missing AUC means no positive/negative class contrast in that subgroup. Pooled metrics summarize independent predictions; no pooled model is trained. Small per-state groups make AUC noisy. Successful fitting is not predictive adequacy.\n\nNo class weighting, custom decision thresholds, regularization, calibration or final test use. Frozen four-state models, configuration and validation predictions are verified unchanged by SHA-256. Separate three-state models are under outputs/models/state_sensitivity_050_3. Only source panel dates through 2024-12-31 are materialized before transitions; test-period observations are excluded.\n\nReproduce: `PYTHONPATH=/tmp/sumatra-audit-deps python scripts/state_definition_sensitivity_050.py`. This sensitivity result does not automatically replace the frozen primary design.\n'''
    common=overall[overall.cohort=='three_state_common'].iloc[0]
    collapsed=overall[overall.cohort=='four_state_collapsed_common'].iloc[0]
    active=states[(states.cohort=='three_state_common')&(states.destination_state==1)].iloc[0]
    active_all=states[(states.cohort=='three_state_all_eligible')&(states.destination_state==1)].iloc[0]
    text+=f'''
## Interpretation of the observed sensitivity

The Low/Moderate partition is empirically finer than these predictors separate well: prior diagnosis found pairwise AUC only 0.5601, and collapsing states increases fully eligible grids from 11 to {len(summary['eligible_grids_3state'])} under exactly the same numerical cutoffs. However, this is not evidence that the broader non-high state is reliably predicted. Its common-cohort recall is {active['recall']:.2%}, F1 {active['f1']:.4f}, and AUC {active['roc_auc']:.4f}; all-eligible recall is {active_all['recall']:.2%}.

On the matched three-destination task, the refitted three-state model has balanced accuracy {common['balanced_accuracy']:.4f}, macro F1 {common['macro_f1']:.4f}, and log loss {common['log_loss']:.4f}. Merely collapsing the original probabilities yields {collapsed['balanced_accuracy']:.4f}, {collapsed['macro_f1']:.4f}, and {collapsed['log_loss']:.4f}, respectively. The refit therefore offers little additional discrimination/classification improvement over relabeling; it has slightly lower log loss but slightly lower balanced accuracy and macro F1. No superiority or significance claim is made.

The interpretation is **better estimability and a more defensible intensity partition, but persistent weak non-high classification under unchanged argmax/predictors**. Most apparent native macro-F1/log-loss gains arise while changing the label task; expanded-cohort gains also reflect different grids. This sensitivity alone does not establish reliable active-state forecasting or warrant automatically freezing a replacement design.
'''
    (out/'sensitivity_report.md').write_text(text)
if __name__=='__main__':main()
