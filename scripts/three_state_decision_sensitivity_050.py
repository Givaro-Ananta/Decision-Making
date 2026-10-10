#!/usr/bin/env python3
"""Three-state decision/weight sensitivity on fixed eligible grids; no test data."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,balanced_accuracy_score,accuracy_score,roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.covariate_markov import predict_log_proba
from src.models.weighted_markov_sensitivity import fit_weighted
DIRECT_THRESHOLDS=[.05,.075,.10,.125,.15,.175,.20,.25,.30,.35,.40,.50]
ACTIVE_THRESHOLDS=[.10,.15,.20,.25,.30,.40,.50,.60]
HIGH_CONDITIONAL_THRESHOLDS=[.33,.50,.67]
WEIGHT_POWERS=[.5,1.]

def decide(p,family='argmax',threshold=None,high_threshold=None):
    if family=='argmax':return np.argmax(p,axis=1)
    if family.startswith('nonhigh_'):
        out=np.where(p[:,2]>p[:,0],2,0)
        yes=p[:,1]>=threshold
        if family=='nonhigh_guarded':yes &= p[:,1]>=p[:,2]
        out[yes]=1;return out
    if family=='two_stage':
        active=p[:,1]+p[:,2];qhigh=np.divide(p[:,2],active,out=np.zeros(len(p)),where=active>0)
        return np.where(active<threshold,0,np.where(qhigh>=high_threshold,2,1))
    raise ValueError(family)

def evaluation(y,p,logs,pred,reference):
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=range(3),zero_division=0);cm=confusion_matrix(y,pred,labels=range(3))
    fp=((y!=1)&(pred==1));oldfp=((y!=1)&(reference==1));tp=((y==1)&(pred==1));oldtp=((y==1)&(reference==1))
    dtp=int(tp.sum()-oldtp.sum());dfp=int(fp.sum()-oldfp.sum());dr=float(dtp/(y==1).sum())
    result={'n':len(y),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1.mean()),'log_loss':float(-logs[np.arange(len(y)),y].mean()),'nonhigh_tp':int(tp.sum()),'nonhigh_fp':int(fp.sum()),'nonhigh_fp_from_no_activity':int(cm[0,1]),'nonhigh_fp_from_high':int(cm[2,1]),'no_activity_false_active':int(cm[0,1]+cm[0,2]),'high_tp':int(cm[2,2]),'high_fp':int(cm[0,2]+cm[1,2]),'delta_nonhigh_tp':dtp,'delta_nonhigh_fp':dfp,'delta_nonhigh_recall_pp':dr*100,'extra_nonhigh_fp_per_extra_tp':float(dfp/dtp) if dtp>0 else np.nan,'extra_nonhigh_fp_per_recall_pp':float(dfp/(dr*100)) if dr>0 else np.nan,'new_nonhigh_fp':int((fp&~oldfp).sum()),'removed_nonhigh_fp':int((~fp&oldfp).sum()),'new_nonhigh_tp':int((tp&~oldtp).sum()),'lost_nonhigh_tp':int((~tp&oldtp).sum()),'delta_no_activity_false_active':int(cm[0,1]+cm[0,2]-((y==0)&(reference!=0)).sum()),'delta_high_tp':int(cm[2,2]-((y==2)&(reference==2)).sum())}
    classrows=[]
    for j in range(3):
        result.update({f'precision_{j}':float(precision[j]),f'recall_{j}':float(recall[j]),f'f1_{j}':float(f1[j])})
        binary=y==j
        classrows.append({'destination_state':j,'support':int(support[j]),'precision':float(precision[j]),'recall':float(recall[j]),'f1':float(f1[j]),'auc':float(roc_auc_score(binary,p[:,j])) if 0<binary.sum()<len(y) else np.nan,'average_precision':float(average_precision_score(binary,p[:,j])) if 0<binary.sum()<len(y) else np.nan})
    return result,classrows,cm

def main():
    import duckdb
    out=ROOT/'outputs/reports/three_state_decision_sensitivity_050';out.mkdir(parents=True,exist_ok=True)
    basepath=ROOT/'outputs/reports/state_definition_sensitivity_050';summary=json.loads((basepath/'sensitivity_summary.json').read_text());ids=summary['eligible_grids_3state'];assert len(ids)==46
    original=pd.read_parquet(basepath/'validation_predictions_3state.parquet').sort_values(['grid_id','date']).reset_index(drop=True);assert len(original)==33626 and original.next_date.between('2023-01-01','2024-12-31').all()
    protected=[basepath/'validation_predictions_3state.parquet',basepath/'design_3state.json',ROOT/'configs/modeling_design_050.json']+list((ROOT/'outputs/models/state_sensitivity_050_3').glob('*/state_*.json'))
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected}
    # Save the finite candidate list before evaluating any variant.
    plan={'eligible_grid_ids':ids,'direct_thresholds':DIRECT_THRESHOLDS,'active_thresholds':ACTIVE_THRESHOLDS,'conditional_high_thresholds':HIGH_CONDITIONAL_THRESHOLDS,'class_weight_powers':WEIGHT_POWERS,'weights':'Within each training grid/current-state group: (n/(k*n_class))**power, normalized to mean sample weight one','threshold_selection':'No operating point frozen; predefined sweep evaluated descriptively on validation only','boundary_frp_mw':16.84,'high_label':2,'nonhigh_label':1,'no_activity_label':0,'test_used':False,'eligibility_changed':False,'training_fit_end':'2022-12-31','validation_target_dates':['2023-01-01','2024-12-31']}
    (out/'analysis_plan.json').write_text(json.dumps(plan,indent=2))
    c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='1GB'")
    source=ROOT/'data/processed/state_sensitivity_050_3/transitions_train_validation.parquet';weightedparts={a:[] for a in WEIGHT_POWERS};fitrows=[]
    for i,gid in enumerate(ids):
        data=c.execute(f"SELECT * FROM read_parquet('{source}') WHERE grid_id=? AND predictors_available AND split IN ('train','validation')",[gid]).fetchdf();train=data[data.split=='train'];val=data[data.split=='validation'];assert train.next_date.max()<=pd.Timestamp('2022-12-31') and val.next_date.max()<=pd.Timestamp('2024-12-31')
        for state in range(3):
            base=json.loads((ROOT/'outputs/models/state_sensitivity_050_3'/gid/f'state_{state}.json').read_text());t=train[train.current_state==state];v=val[val.current_state==state]
            for power in WEIGHT_POWERS:
                model,fit=fit_weighted(t,base,power);fit.update(grid_id=gid,current_state=state);fitrows.append(fit)
                folder=ROOT/'outputs/models/three_state_weight_sensitivity_050'/f'power_{power}'/gid;folder.mkdir(parents=True,exist_ok=True)
                (folder/f'state_{state}_fit_diagnostics.json').write_text(json.dumps(fit,indent=2))
                if not fit['success']:continue
                (folder/f'state_{state}.json').write_text(json.dumps(model,indent=2))
                if len(v):
                    logs=predict_log_proba(model,v)[:,:3];p=np.exp(logs);assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1,atol=1e-12)
                    rows=v[['grid_id','date','next_date','current_state','next_state']].rename(columns={'next_state':'actual_next_state'}).copy()
                    for j in range(3):rows[f'p_{j}']=p[:,j];rows[f'log_p_{j}']=logs[:,j]
                    weightedparts[power].append(rows)
        if i%10==0:print(f'Weighted fits: {i+1}/{len(ids)} grids',flush=True)
    fits=pd.DataFrame(fitrows);fits['class_weights']=fits.class_weights.apply(json.dumps);fits.to_csv(out/'weighted_fit_diagnostics.csv',index=False)
    weighted={}
    for power,parts in weightedparts.items():
        frame=pd.concat(parts,ignore_index=True).sort_values(['grid_id','date']).reset_index(drop=True)
        frame.to_parquet(out/f'weighted_validation_probabilities_power_{power}.parquet',index=False)
        if not frame[['grid_id','date','next_date','current_state','actual_next_state']].equals(original[['grid_id','date','next_date','current_state','actual_next_state']]):
            raise RuntimeError('Weighted fit failure or missing prediction: cohort differs. Inspect diagnostics; no screening relaxation or silent comparison.')
        weighted[power]=frame
    variants=[{'name':'reference_argmax','family':'argmax','probability_source':'unweighted'}]
    variants += [{'name':f'weighted_argmax_power_{a}','family':'argmax','probability_source':f'weighted_{a}'} for a in WEIGHT_POWERS]
    for family in ['nonhigh_override','nonhigh_guarded']:
        for t in DIRECT_THRESHOLDS:variants.append({'name':f'{family}_t{t:g}','family':family,'threshold':t,'probability_source':'unweighted'})
    for t in ACTIVE_THRESHOLDS:
        for h in HIGH_CONDITIONAL_THRESHOLDS:variants.append({'name':f'two_stage_active{t:g}_high{h:g}','family':'two_stage','threshold':t,'high_threshold':h,'probability_source':'unweighted'})
    reference=original[['p_0','p_1','p_2']].to_numpy().argmax(axis=1);y=original.actual_next_state.to_numpy();metrics=[];classrows=[];confusions=[];decisions=original[['grid_id','date','next_date','current_state','actual_next_state']].copy()
    for variant in variants:
        sourceframe=original if variant['probability_source']=='unweighted' else weighted[float(variant['probability_source'].split('_')[1])]
        p=sourceframe[['p_0','p_1','p_2']].to_numpy();logs=sourceframe[['log_p_0','log_p_1','log_p_2']].to_numpy()
        predicted=decide(p,variant['family'],variant.get('threshold'),variant.get('high_threshold'));decisions[variant['name']]=predicted
        groups=[('overall','ALL',-1,np.arange(len(y)))]+[('per_grid',g,-1,np.flatnonzero(original.grid_id.to_numpy()==g)) for g in ids]+[('per_current_state','ALL',s,np.flatnonzero(original.current_state.to_numpy()==s)) for s in range(3)]
        for scope,gid,state,indices in groups:
            result,classes,cm=evaluation(y[indices],p[indices],logs[indices],predicted[indices],reference[indices]);meta=dict(variant,scope=scope,grid_id=gid,current_state=state)
            metrics.append(dict(meta,**result));classrows.extend(dict(meta,**cl) for cl in classes)
            for a in range(3):
                for b in range(3):confusions.append(dict(meta,actual_state=a,predicted_state=b,count=int(cm[a,b])))
    pd.DataFrame(metrics).to_csv(out/'variant_metrics.csv',index=False);pd.DataFrame(classrows).to_csv(out/'variant_state_metrics_auc.csv',index=False);pd.DataFrame(confusions).to_csv(out/'variant_confusion_matrices.csv',index=False);decisions.to_parquet(out/'validation_variant_decisions.parquet',index=False)
    allmetrics=pd.DataFrame(metrics);overall=allmetrics[allmetrics.scope=='overall'];overall.to_csv(out/'overall_comparison.csv',index=False)
    # Diagnostic Pareto points for recall versus non-high false positives, not a chosen operating rule.
    frontier=[]
    for _,row in overall.iterrows():
        dominated=((overall.recall_1>=row.recall_1)&(overall.nonhigh_fp<=row.nonhigh_fp)&((overall.recall_1>row.recall_1)|(overall.nonhigh_fp<row.nonhigh_fp))).any()
        if not dominated:frontier.append(row.to_dict())
    pd.DataFrame(frontier).to_csv(out/'recall_false_positive_frontier.csv',index=False)
    for f,value in hashes.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==value
    result={'n_grids':len(ids),'n_validation':len(original),'n_variants':len(variants),'weighted_successful_models':int(fits.success.sum()),'weighted_failed_models':int((~fits.success).sum()),'original_artifacts_unchanged':True,'test_used':False,'threshold_rule_frozen':False,'protected_sha256':hashes}
    (out/'run_summary.json').write_text(json.dumps(result,indent=2));write_report(out);print(json.dumps({k:v for k,v in result.items() if k!='protected_sha256'},indent=2))

def write_report(out):
    m=pd.read_csv(out/'overall_comparison.csv');s=json.loads((out/'run_summary.json').read_text())
    chosen=['reference_argmax','weighted_argmax_power_0.5','weighted_argmax_power_1.0','nonhigh_guarded_t0.1','nonhigh_guarded_t0.15','nonhigh_guarded_t0.2','nonhigh_override_t0.1','two_stage_active0.2_high0.5','two_stage_active0.3_high0.5','two_stage_active0.5_high0.5']
    display=m[m.name.isin(chosen)][['name','balanced_accuracy','macro_f1','log_loss','precision_1','recall_1','f1_1','recall_2','delta_nonhigh_tp','delta_nonhigh_fp','delta_no_activity_false_active','extra_nonhigh_fp_per_extra_tp']].copy()
    display=display.round(5)
    def table(f):return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))
    text=f'''# Three-state decision sensitivity at fixed 0.5°\n\nAll comparisons use the same {s['n_grids']} eligible grids and {s['n_validation']:,} validation rows (targets 2023–2024). States remain 0 No Activity, 1 Active/non-high (<16.84 MW on active days), 2 High (≥16.84 MW). No grid, predictor, eligibility screen, FRP boundary, split or independence assumption is changed.\n\n## Predeclared variants\n\nUnweighted saved-model argmax is the reference. Class-weighted multinomial fitting uses training group counts only, independently per grid/current state. Full inverse-frequency balanced weights n/(k*n_class), and their square roots, are each normalized to mean observation weight one. These are weighted versions of the same unpenalized likelihood, with the same training scaling and solver checks; no alternate model is introduced. Weighted fits successful: {s['weighted_successful_models']}; failures: {s['weighted_failed_models']}. All original models and probabilities are preserved. Weight formula: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html).\n\nDirect rule: assign non-high if p1≥tau; otherwise choose argmax between p0 and p2. Guarded direct rule additionally requires p1≥p2 before assigning non-high. Both are run at tau=0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25,0.30,0.35,0.40,0.50.\n\nTwo-stage rule: assign No Activity when p1+p2<tau_active; otherwise assign High when p2/(p1+p2)≥tau_high, else non-high. tau_active=0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60; tau_high=0.33,0.50,0.67. Stage two uses original multinomial probabilities, not another fitted model.\n\nThe finite variant list was saved before evaluation. Thresholds are global across eligible grids/current states; no per-grid tuning. Validation is being used exploratorily to inspect operating tradeoffs, so maximum validation scores are not unbiased confirmation or a frozen choice. No operating point is selected for final test use. Further threshold selection would require an explicitly agreed criterion/cost tradeoff and appropriate validation discipline; see [threshold-tuning cautions](https://scikit-learn.org/1.5/modules/classification_threshold.html).\n\n## Representative predefined points\n\n{table(display)}\n\nAll {s['n_variants']} variants are in overall_comparison.csv and variant_metrics.csv, including per-grid and originating-state metrics. Per-state precision/recall/F1 and AUC are in variant_state_metrics_auc.csv. Complete fixed 3×3 confusion matrices for every variant/cohort are in variant_confusion_matrices.csv. Validation decisions and weighted probabilities are retained in Parquet.\n\n## False-positive accounting\n\nNon-high FP means a true No Activity or High observation labeled non-high. Delta FP/TP counts are signed changes from original argmax. Extra FP per extra TP is delta_FP/delta_TP when delta_TP>0; positive recall-point cost is also supplied. New and removed FP, newly recovered and lost TP are recorded separately. No Activity false-active means true state 0 labeled either state 1 or 2; its delta tracks new fire alarms independently of High/non-high confusion. High TP/FP and their changes are supplied so increased non-high recall cannot hide losses of High detection. recall_false_positive_frontier.csv is a descriptive tradeoff frontier, not a deployed selection.\n\n## Probability and score interpretation\n\nChanging only decisions leaves log loss and AUC of the underlying unweighted probabilities unchanged. Log loss shown for a threshold rule is its source model's probability loss, not a score of its hard labels. Weighted fits change probabilities and have separately evaluated unweighted validation log loss; their reweighted training objective is not comparable to unweighted likelihood as a probability quality claim. No probability recalibration or prior correction is applied. Macro F1 uses all three labels; balanced accuracy uses mean recall over present destinations. Comparisons aggregate independently fitted predictions, never pooled fitting.\n\nFrozen artifacts passed SHA-256 unchanged checks. Training ends 2022-12-31, validation ends 2024-12-31. The only transition input is the previously created train/validation-only three-state table; no 2025+ test observations are read or evaluated.\n\nReproduce: `PYTHONPATH=/tmp/sumatra-audit-deps python scripts/three_state_decision_sensitivity_050.py`.\n'''
    allstates=pd.read_csv(out/'variant_state_metrics_auc.csv')
    displayed=allstates[(allstates.scope=='overall')&allstates.name.isin(chosen)]
    text+='\n## Per-state metrics at representative predefined points\n\n'+table(displayed[['name','destination_state','support','precision','recall','f1','auc']].round(5))+'\n'
    cmdata=pd.read_csv(out/'variant_confusion_matrices.csv')
    text+='\n## Representative confusion matrices\n\nRows are actual [No Activity, non-high, High]; columns are predictions in the same order.\n\n'
    for name in ['reference_argmax','weighted_argmax_power_0.5','weighted_argmax_power_1.0','nonhigh_guarded_t0.2','two_stage_active0.2_high0.5']:
        cm=cmdata[(cmdata.scope=='overall')&(cmdata.name==name)].pivot(index='actual_state',columns='predicted_state',values='count')
        text+=name+'\n\n```\n'+cm.to_string()+'\n```\n\n'
    text+='''
## Interpretation

Default argmax is a major bottleneck for non-high recall, but it is not the only limitation. The same frozen probability signal can produce substantially better balanced accuracy and macro F1: the predefined guarded p1≥0.20 rule reaches 30.19% non-high recall at 26.10% precision, versus 5.40% recall/33.94% precision for argmax. It adds 853 net recovered non-high observations and 2,580 net non-high false positives (3.02 extra FP per additional TP), with 2,179 extra No Activity false-active alarms and unchanged High TP. Macro F1 improves in 45/46 grids.

The predefined two-stage p_active≥0.20 / conditional p_high≥0.50 rule reaches 42.04% non-high recall, precision 23.29%, F1 0.2997, balanced accuracy 0.5688, and macro F1 0.5338. It adds 1,261 net non-high TP, 4,404 net non-high FP (3.49 per additional TP), 4,217 extra No Activity false-active alarms, and 120 more correctly identified High observations. Macro F1 and non-high recall improve in all 46 grids. These are observed validation tradeoffs, not chosen deployment thresholds or evidence of a known acceptable false-alarm cost.

A more conservative two-stage p_active≥0.50 / conditional p_high≥0.50 point reaches 10.26% non-high recall at 32.53% precision: 167 extra non-high TP, 370 extra non-high FP (2.22 per TP), 403 extra false-active alarms and 65 extra High TP. Thus some gains are available without the most aggressive alarm expansion.

Unrestricted non-high override p1≥0.10 inflates recall to 72.63% but precision is only 18.97%, and High recall collapses to 9.07% from 38.58%. This exposes why recall alone is not a sufficient criterion. Guarded and two-stage rules make the sacrificed error types explicit.

Class weighting is less attractive here. Square-root weights provide a modest gain (non-high recall 12.58%, macro F1 0.4962) while worsening log loss from 0.4202 to 0.5396. Full balanced weighting reaches 29.05% non-high recall but reduces macro F1 to 0.3769 and raises log loss to 0.9829; it adds 11,137 No Activity false-active alarms. Non-high AUC falls from 0.7443 to 0.7055 (square-root) and 0.5753 (full). Weighted fits all converge, so this is a generalization/probability-quality issue, not solver failure. Raw reweighted probabilities target a reweighted outcome distribution; they should not be interpreted as calibrated natural-frequency probabilities. No recalibration or prior adjustment is applied.

The predictors retain useful ranking information for tradeoffs (unweighted non-high AUC 0.7443, average precision 0.2347), but moderate precision at substantially increased recall shows limited separation. This analysis supports **decision-rule suppression plus imperfect discrimination**, rather than either “argmax is the whole problem” or “the probability signal is absent.” Whether any point is practically useful requires an agreed cost for false alarms and missed High/non-high activity; none was supplied, so no rule is frozen.

All variants were evaluated on a repeatedly inspected validation period. Any subsequent choice based on these scores is exploratory and does not establish performance on unseen dates. The 2025+ test set remains untouched.
'''
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for family,label in [('nonhigh_guarded','Guarded non-high'),('nonhigh_override','Non-high override')]:
        points=m[m.family==family].sort_values('recall_1');axes[0].plot(points.recall_1*100,points.precision_1*100,'o-',label=label);axes[1].plot(points.recall_1*100,points.nonhigh_fp,'o-',label=label)
    points=m[(m.family=='two_stage')&(m.high_threshold==.5)].sort_values('recall_1')
    axes[0].plot(points.recall_1*100,points.precision_1*100,'o-',label='Two-stage, high conditional=0.50');axes[1].plot(points.recall_1*100,points.nonhigh_fp,'o-',label='Two-stage, high conditional=0.50')
    for name,label,marker in [('reference_argmax','Argmax','*'),('weighted_argmax_power_0.5','Square-root weights','s'),('weighted_argmax_power_1.0','Balanced weights','D')]:
        r=m[m.name==name].iloc[0];axes[0].scatter(r.recall_1*100,r.precision_1*100,s=100,marker=marker,label=label);axes[1].scatter(r.recall_1*100,r.nonhigh_fp,s=100,marker=marker,label=label)
    for ax in axes:ax.set_xlabel('Active/non-high recall (%)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Active/non-high precision (%)');axes[1].set_ylabel('Non-high false positives (No Activity + High)')
    axes[0].legend(fontsize=8);fig.suptitle('Exploratory validation tradeoffs: 46 fixed grids, 33,626 observations');fig.tight_layout();fig.savefig(out/'recall_tradeoffs.png',dpi=180);fig.savefig(out/'recall_tradeoffs.svg');plt.close(fig)
    text+='\n![Validation recall tradeoffs](recall_tradeoffs.png)\n'
    (out/'decision_sensitivity_report.md').write_text(text)
if __name__=='__main__':main()
