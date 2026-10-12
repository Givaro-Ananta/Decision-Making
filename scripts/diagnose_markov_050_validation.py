#!/usr/bin/env python3
"""Diagnose saved validation failures; no fitting, changes, or test evaluation."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.covariate_markov import predict_log_proba
P=['p_no_activity','p_low','p_moderate','p_high'];FEATURES=['temperature_max_c_t','relative_humidity_mean_t','wind_speed_mean_t','rainfall_mm_t']

def distribution(values):
    values=np.asarray(values)
    return dict(zip(['min','p10','p25','p50','p75','p90','p95','p99','max'],np.quantile(values,[0,.1,.25,.5,.75,.9,.95,.99,1])))|{'mean':float(values.mean())}

def rank_true(probabilities,y):
    true=probabilities[np.arange(len(y)),y]
    return 1+(probabilities>true[:,None]).sum(axis=1)

def write_report(out):
    counts=pd.read_csv(out/'destination_counts.csv');overall=counts[counts.scope=='overall']
    disc=pd.read_csv(out/'discrimination.csv');local=disc[(disc.scope=='per_grid_current_state')&(disc.period=='validation')]
    confuse=pd.read_csv(out/'confusion_patterns.csv');confuse=confuse[(confuse.scope=='overall')&confuse.true_state.isin([1,2])]
    fail=pd.read_csv(out/'failure_concentration.csv');bygrid=fail[fail.scope=='per_grid'];byorigin=fail[fail.scope=='per_current_state']
    shift=pd.read_csv(out/'transition_shift.csv');extreme=pd.read_csv(out/'true_probability_below_1pct.csv')
    def table(frame):
        return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))
    local_stats=local[local.destination_state.isin([1,2])].groupby('destination_state').roc_auc.agg(['min','median','max'])
    report=f'''# Diagnosis of Low/Moderate validation recall — frozen 0.5° model

## Conclusion

The principal explanation is **argmax dominance combined with weak Low/Moderate discrimination**, rather than a small set of broken numerical fits. Temporal shift is a secondary contributor, especially for active-origin groups, but the same poor recall already exists on training resubstitution. The diagnosis does not establish a causal decomposition or justify any automatic redesign.

## Actual versus predicted destinations

{table(overall[['destination_state','n_train','training_predicted_count','training_recall','actual_validation','predicted_validation']])}

Training recall was only 1.14% for Low and 3.55% for Moderate; validation is 0.28% and 3.76%. These training figures are resubstitution diagnostics, not independent performance estimates. No model was refitted to produce them.

Validation has 714 true Low days but only 24 Low predictions (2 correct), and 399 true Moderate days but only 83 Moderate predictions (15 correct). No Activity is 78.32% of actual destinations but 91.17% of argmax predictions.

## Confusion patterns

{table(confuse[['true_state','predicted_state','count','fraction_of_true']])}

84.31% of Low and 76.69% of Moderate are assigned No Activity; another 12.04% and 18.30%, respectively, are assigned High. Under current No Activity, models choose No Activity for 99.14% of validation rows and never choose Low. Per-grid/current-state counts for all four destinations, including zero predictions, are in destination_counts.csv and confusion_patterns.csv.

## Probabilities and ranks

For true Low, median probabilities for [No Activity, Low, Moderate, High] are [0.7558, 0.0994, 0.0636, 0.0520]. For true Moderate they are [0.6349, 0.1146, 0.0882, 0.1005]. These are separate marginal medians and need not sum to one. Median winner-minus-true probability gaps are 0.6501 and 0.5841. Thus this is not mainly a close-tie problem.

Low's true probability ranks second in 64.85% of cases, and first in only 0.28%. Moderate ranks third in 52.13%, fourth in 27.32%, and first in 3.76%. Some Low probability mass is retained, but ranking second with about 10% probability is not evidence of strong class separation or conditional calibration. Full p10/p25/p50/p75/p90/p95/p99/min/max distributions of every predicted class conditional on every true class are in conditional_probability_distributions.csv, with rank frequencies in true_probability_ranks.csv. Rank uses one plus the number of strictly larger probabilities; exact ties share rank and are not alternative decision rules.

## Discrimination

Overall validation one-vs-rest AUC: Low 0.6456, Moderate 0.7376. Average precision: Low 0.1359 (prevalence 0.0888), Moderate 0.1123 (prevalence 0.0496). Those overall metrics mix grids and originating states; they overstate how consistently discrimination works within independent models.

Within grid/current-state validation groups, AUC summaries:

```
{local_stats.to_string()}
```

14 of 44 Low group AUCs and 11 of 44 Moderate group AUCs are ≤0.5. Pairwise Low-versus-Moderate AUC is 0.5601 after normalizing to their two probabilities. Pairwise Low-versus-No-Activity is 0.7147; Moderate-versus-No-Activity is 0.8044. Thus some separation from inactivity exists, but separation between the two intermediate destinations is particularly weak. Pairwise figures are diagnostics on selected true-label pairs, not changes to the classifier. Small subgroup samples make AUC estimates noisy.

Training-resubstitution one-vs-rest AUC falls from 0.7074 to 0.6456 on validation for Low, and from 0.7848 to 0.7376 for Moderate. Resubstitution is optimistic and its deterioration alone does not prove temporal shift. discrimination.csv records training/validation metrics for each grid/current-state group, positive counts, prevalence, and pairwise validation discrimination.

## Temporal distribution shift

Overall transition-frequency total variation is 0.03134 (3.13 percentage points of probability mass). Low prevalence increases from 7.64% to 8.88%; Moderate stays approximately stable (5.03% to 4.96%). Therefore the collapse cannot be explained by these destinations becoming substantially rarer during validation.

Conditional changes are larger: from current Moderate, No Activity destinations fall from 54.93% to 44.36%, while High rises from 16.65% to 24.31%. From current High, No Activity falls from 37.89% to 27.50%, while High rises from 37.09% to 45.63%. The largest group shifts are SUM050_R019_C016/current Moderate (TV 0.2770), SUM050_R005_C006/current High (TV 0.2713, only 9 validation rows), and SUM050_R019_C019/current High (TV 0.2648). Shift quantities are descriptive and are not formal time-series significance tests.

Overall environmental mean shifts, measured in training standard deviations: temperature +0.264, humidity −0.271, wind +0.125, rainfall −0.073. feature_shift.csv provides per-group means, validation quantiles, and counts outside training ranges. Neither frequency nor feature comparisons identify the causal contribution of shift. Stable training poor recall remains the strongest evidence that temporal shift is not the primary explanation.

## Concentration by grid and current state

{table(bygrid[['grid_id','true_state','n_true','correct','missed','recall','true_p_lt001']])}

Low recall is zero in 9 of 11 grids; only SUM050_R017_C016 and SUM050_R021_C020 have one correct Low each. Moderate recall is zero in 6 of 11 grids. Failures are widespread rather than isolated to one grid. Count concentration must be interpreted against each group's true-destination exposure.

{table(byorigin[['current_state','true_state','n_true','correct','missed','recall','true_p_lt001']])}

Current No Activity accounts for 439/712 Low misses and 188/384 Moderate misses. For Moderate, 11 of its 15 correct predictions originate from current Moderate. All current-state strata nevertheless show missed intermediate destinations. Complete per-grid/current-state evidence is in failure_concentration.csv.

## True-class probability below 1%

35 of 8,041 validation observations (0.435%) have true-class probability below 1%: Low 5, Moderate 16, High 11, No Activity 3. They arise from current states 0/1/2/3 in counts 24/1/5/5. These extreme tails are important for log loss but cannot explain all 1,096 missed Low/Moderate observations.

Only 5/35 are outside any corresponding training feature range; 5/35 have at least one absolute training-standardized feature ≥3. SUM050_R020_C020 contributes 10/35 extreme cases. Most extremes occur inside marginal training ranges, which does not guarantee joint covariate support. Nearest same-grid/current-state standardized training distances are supplied.

The worst observation is SUM050_R019_C019, current High, predictor date 2023-11-04, target date 2023-11-05: true Moderate, predicted No Activity, true probability 4.591e-13. Rainfall is 26.857 mm, +8.095 training SD for that particular state model, outside its training range; the nearest training feature vector is 4.005 standardized Euclidean units away. The rainfall term contributes +27.201 to the No-Activity-versus-Moderate log odds; total contrast is +28.284. This identifies a specific extrapolation-sensitive tail prediction, not an optimizer failure or an explanation for the widespread recall collapse.

true_probability_below_1pct.csv lists all 35 observations with probabilities, ranks, origin/destination, date, observed predictor values, training z scores, range flags, nearest-training distance, training/validation destination frequency, and per-term winner-versus-true log-odds contributions. Contributions explain the frozen fitted equation, not causal fire effects.

## Which explanation is supported?

1. **Argmax/class imbalance: strongly supported**, particularly Low ranking second while No Activity dominates. However, probabilities are not uniformly strong; large gaps and weak discrimination prevent calling this merely a decision-rule artifact.
2. **Poor intermediate-state discrimination: strongly supported**, including weak within-model AUC and Low-versus-Moderate discrimination. It combines with dominant inactivity to produce the tiny recall.
3. **Temporal shift: supported as a secondary/local factor**, with active-origin frequency changes and some feature extremes. It does not explain similarly poor training recall.
4. **Specific numerical model failures: not supported as the main cause.** All 44 saved fits converged with valid probabilities and replayed predictions; poor intermediate recall is widespread. Local extrapolation and finite-sample limitations remain, especially the extreme rainfall case.

No thresholds, features, eligibility rules, class weights, regularization, calibration or decision rules were changed. No model was fit/refit. Only training and 2023–2024 validation observations were selected; no 2025+ test observations were materialized or evaluated. SHA-256 integrity checks verified that models, frozen configurations and original validation predictions were unchanged. integrity.json records these checks.

Reproduce from repository root: `PYTHONPATH=/tmp/sumatra-audit-deps python scripts/diagnose_markov_050_validation.py`.
'''
    (out/'diagnosis.md').write_text(report)


def main():
    import duckdb
    out=ROOT/'outputs/reports/markov_050_validation_diagnosis';out.mkdir(parents=True,exist_ok=True)
    previous=ROOT/'outputs/reports/markov_050_validation'
    prediction=pd.read_parquet(previous/'validation_predictions.parquet');assert prediction.next_date.between('2023-01-01','2024-12-31').all()
    protected=[ROOT/'configs/modeling_design_050.json',ROOT/'configs/model_fitting_050.json',previous/'validation_predictions.parquet']+list((ROOT/'outputs/models/markov_050').glob('*/state_*.json'))
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected}
    c=duckdb.connect();c.execute("SET threads=1; SET memory_limit='1GB'")
    training=[];features=[];extremes=[]
    for gid in sorted(prediction.grid_id.unique()):
        source=c.execute(f"SELECT grid_id,date,next_date,current_state,next_state,hotspot_count,frp_sum,{','.join(FEATURES)} FROM read_parquet('data/processed/modeling_050/transitions.parquet') WHERE grid_id=? AND split IN ('train','validation') AND predictors_available",[gid]).fetchdf()
        train=source[source.next_date<='2022-12-31'];val=source[(source.next_date>'2022-12-31')&(source.next_date<='2024-12-31')]
        features.append(val)
        for state in range(4):
            t=train[train.current_state==state].copy();v=val[val.current_state==state].copy()
            model=json.loads((ROOT/'outputs/models/markov_050'/gid/f'state_{state}.json').read_text())
            logs=predict_log_proba(model,t);prob=np.exp(logs)
            t=t.rename(columns={'next_state':'actual_next_state'});t['predicted_next_state']=prob.argmax(axis=1)
            for j,p in enumerate(P): t[p]=prob[:,j]
            t['log_probability_actual']=logs[np.arange(len(t)),t.actual_next_state.to_numpy()];training.append(t)
            observed=prediction[(prediction.grid_id==gid)&(prediction.current_state==state)].merge(v,on=['grid_id','date','next_date','current_state'],validate='one_to_one')
            saved=observed[P].to_numpy();np.testing.assert_allclose(np.exp(predict_log_proba(model,observed)),saved,atol=1e-14)
            bad=observed[observed.log_probability_actual<np.log(.01)]
            Zt=np.column_stack([(t[f]-model['means'][f])/model['scales'][f] for f in FEATURES])
            for row in bad.itertuples(index=False):
                actual=int(row.actual_next_state);winner=int(row.predicted_next_state)
                x=np.array([getattr(row,f) for f in FEATURES]);z=np.array([(x[j]-model['means'][f])/model['scales'][f] for j,f in enumerate(FEATURES)])
                beta=np.vstack([np.asarray(model['coefficients_standardized']),np.zeros(5)])
                contrast=beta[winner]-beta[actual];contributions=contrast*np.r_[1,z]
                record={'grid_id':gid,'current_state':state,'date':str(row.date.date()),'next_date':str(row.next_date.date()),'actual_next_state':actual,'predicted_next_state':winner,'true_probability':float(np.exp(row.log_probability_actual)),'true_probability_rank':int(rank_true(saved[[np.flatnonzero(observed.date.to_numpy()==row.date)[0]]],np.array([actual]))[0]),'negative_log_probability':float(-row.log_probability_actual),'training_destination_count':int((t.actual_next_state==actual).sum()),'training_destination_fraction':float((t.actual_next_state==actual).mean()),'validation_destination_fraction':float((observed.actual_next_state==actual).mean()),'nearest_training_distance_standardized':float(cdist(z[None,:],Zt).min()),'max_absolute_training_z':float(np.abs(z).max()),'log_odds_winner_vs_true':float(contributions.sum()),'intercept_log_odds_contribution':float(contributions[0]),'current_hotspot_count':int(row.hotspot_count),'current_frp_sum':float(row.frp_sum)}
                for j,f in enumerate(FEATURES): record.update({f:float(x[j]),f+'_training_z':float(z[j]),f+'_outside_training_range':bool(x[j]<t[f].min() or x[j]>t[f].max()),f+'_winner_true_log_odds_contribution':float(contributions[j+1])})
                extremes.append(record)
    train=pd.concat(training,ignore_index=True);vfeatures=pd.concat(features,ignore_index=True)
    assert len(train)==25190 and len(prediction)==8041
    merged=prediction.merge(vfeatures.drop(columns='next_state'),on=['grid_id','date','next_date','current_state'],validate='one_to_one')
    for frame in [train,merged]:
        probabilities=frame[P].to_numpy();actual=frame.actual_next_state.to_numpy();frame['true_probability']=probabilities[np.arange(len(frame)),actual];frame['true_rank']=rank_true(probabilities,actual);frame['probability_margin_winner_true']=probabilities.max(axis=1)-frame.true_probability
    counts=[];confusions=[];distributions=[];ranks=[];discrimination=[];shift=[];concentration=[];feature_shift=[]
    scopes=[('overall','ALL',-1),*[( 'per_grid',g,-1) for g in sorted(prediction.grid_id.unique())],*[('per_current_state','ALL',s) for s in range(4)],*[('per_grid_current_state',g,s) for g in sorted(prediction.grid_id.unique()) for s in range(4)]]
    def select(frame,gid,state):
        mask=np.ones(len(frame),dtype=bool)
        if gid!='ALL': mask &= frame.grid_id==gid
        if state!=-1: mask &= frame.current_state==state
        return frame[mask]
    for scope,gid,state in scopes:
        meta={'scope':scope,'grid_id':gid,'current_state':state};t=select(train,gid,state);v=select(merged,gid,state)
        tf=t.actual_next_state.value_counts().reindex(range(4),fill_value=0).to_numpy();vf=v.actual_next_state.value_counts().reindex(range(4),fill_value=0).to_numpy();pf=v.predicted_next_state.value_counts().reindex(range(4),fill_value=0).to_numpy()
        shift.append(dict(meta,n_train=len(t),n_validation=len(v),transition_total_variation=float(np.abs(tf/len(t)-vf/len(v)).sum()/2),transition_js_divergence_bits=float(jensenshannon(tf/len(t),vf/len(v),base=2)**2)))
        for actual in range(4):
            counts.append(dict(meta,destination_state=actual,n_train=int(tf[actual]),training_predicted_count=int((t.predicted_next_state==actual).sum()),training_recall=float((t.loc[t.actual_next_state==actual,'predicted_next_state']==actual).mean()) if tf[actual] else np.nan,train_fraction=float(tf[actual]/len(t)),actual_validation=int(vf[actual]),actual_validation_fraction=float(vf[actual]/len(v)),predicted_validation=int(pf[actual]),predicted_validation_fraction=float(pf[actual]/len(v))))
            true=v[v.actual_next_state==actual]
            for predicted in range(4): confusions.append(dict(meta,true_state=actual,predicted_state=predicted,count=int((true.predicted_next_state==predicted).sum()),fraction_of_true=float((true.predicted_next_state==predicted).mean()) if len(true) else np.nan))
            if len(true):
                for j,p in enumerate(P): distributions.append(dict(meta,true_state=actual,probability_class=j,n=len(true),**distribution(true[p])))
                distributions.append(dict(meta,true_state=actual,probability_class='winner_minus_true_margin',n=len(true),**distribution(true.probability_margin_winner_true)))
                for rank in range(1,5): ranks.append(dict(meta,true_state=actual,rank=rank,n=int((true.true_rank==rank).sum()),fraction=float((true.true_rank==rank).mean())))
            if actual in [1,2]: concentration.append(dict(meta,true_state=actual,n_true=len(true),correct=int((true.predicted_next_state==actual).sum()),missed=int((true.predicted_next_state!=actual).sum()),recall=float((true.predicted_next_state==actual).mean()) if len(true) else np.nan,true_p_lt001=int((true.true_probability<.01).sum())))
            for period,frame in [('training_resubstitution',t),('validation',v)]:
                y=(frame.actual_next_state==actual).to_numpy();scores=frame[P[actual]].to_numpy()
                if 0<y.sum()<len(y): discrimination.append(dict(meta,period=period,destination_state=actual,n=len(y),n_positive=int(y.sum()),prevalence=float(y.mean()),roc_auc=float(roc_auc_score(y,scores)),average_precision=float(average_precision_score(y,scores)),mean_p_if_true=float(scores[y].mean()),mean_p_if_other=float(scores[~y].mean())))
            # Pairwise discrimination using conditional probabilities removes No Activity dominance.
            if actual in [1,2]:
                for other in [j for j in range(4) if j!=actual]:
                    pair=v[v.actual_next_state.isin([actual,other])];y=(pair.actual_next_state==actual).to_numpy()
                    if 0<y.sum()<len(y):
                        a=pair[P[actual]].to_numpy();b=pair[P[other]].to_numpy();score=a/(a+b)
                        discrimination.append(dict(meta,period='validation_pairwise',destination_state=actual,other_state=other,n=len(pair),n_positive=int(y.sum()),prevalence=float(y.mean()),roc_auc=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score))))
        for feature in FEATURES:
            feature_shift.append(dict(meta,feature=feature,train_mean=float(t[feature].mean()),validation_mean=float(v[feature].mean()),standardized_mean_shift=float((v[feature].mean()-t[feature].mean())/t[feature].std(ddof=0)),validation_outside_training_range=int(((v[feature]<t[feature].min())|(v[feature]>t[feature].max())).sum()),**{'validation_distribution_'+k:val for k,val in distribution(v[feature]).items()}))
    for name,data in [('destination_counts',counts),('confusion_patterns',confusions),('conditional_probability_distributions',distributions),('true_probability_ranks',ranks),('discrimination',discrimination),('transition_shift',shift),('failure_concentration',concentration),('feature_shift',feature_shift),('true_probability_below_1pct',extremes)]: pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
    for f,value in hashes.items(): assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==value
    overview={'validation_rows':len(merged),'training_rows_used_for_diagnostics':len(train),'test_read_or_evaluated':False,'refit':False,'protected_artifacts_unchanged':True,'true_probability_below_1pct':len(extremes),'protected_sha256':hashes}
    (out/'integrity.json').write_text(json.dumps(overview,indent=2))
    write_report(out)
    print(json.dumps({k:v for k,v in overview.items() if k!='protected_sha256'},indent=2))
if __name__=='__main__': main()
