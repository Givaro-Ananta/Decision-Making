#!/usr/bin/env python3
"""Fit frozen eligible grids on training only; evaluate validation only."""
import sys,json,hashlib,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.covariate_markov import fit_transition,predict_log_proba

def metrics(frame):
    y=frame.actual_next_state.to_numpy(); predicted=frame.predicted_next_state.to_numpy()
    precision,recall,f1,support=precision_recall_fscore_support(y,predicted,labels=range(4),zero_division=0)
    weighted=float(np.sum(f1*support)/len(y)); logs=frame.log_probability_actual.to_numpy()
    probs=frame[['p_no_activity','p_low','p_moderate','p_high']].to_numpy(); onehot=np.eye(4)[y]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); balanced=float(balanced_accuracy_score(y,predicted))
    return {'n':len(y),'accuracy':float(accuracy_score(y,predicted)),'balanced_accuracy':balanced,'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),'weighted_f1':weighted,'log_loss':float(-logs.mean()),'multiclass_brier':float(np.square(probs-onehot).sum(axis=1).mean())},precision,recall,f1,support

def main():
    import duckdb
    output=ROOT/'outputs/reports/markov_050_validation';output.mkdir(parents=True,exist_ok=True)
    modeldir=ROOT/'outputs/models/markov_050';modeldir.mkdir(parents=True,exist_ok=True)
    design_path=ROOT/'configs/modeling_design_050.json'; design_bytes=design_path.read_bytes(); design=json.loads(design_bytes)
    assert design['resolution_degrees']==.5 and design['train_end']=='2022-12-31' and design['state_boundaries']=={'low_moderate':5.96,'moderate_high':16.84}
    eligibility=pd.read_csv(ROOT/'outputs/reports/modeling_design_050/grid_eligibility.csv'); ids=sorted(eligibility.loc[eligibility.eligible,'grid_id']); assert len(ids)==11
    diagnostics=pd.read_csv(ROOT/'outputs/reports/modeling_design_050/per_grid_state_diagnostics.csv')
    c=duckdb.connect();c.execute("SET threads=1; SET memory_limit='1GB'")
    # Explicit split predicate: test rows are never materialized or evaluated.
    source=ROOT/'data/processed/modeling_050/transitions.parquet'
    fitting=[];coefs=[];predictions=[];coverage=[]
    for gid in ids:
        frame=c.execute(f"SELECT grid_id,date,next_date,current_state,next_state,predictors_available,{','.join(design['features'])} FROM read_parquet('{source}') WHERE grid_id=? AND split IN ('train','validation')",[gid]).fetchdf()
        train=frame[(frame.next_date<='2022-12-31')&frame.predictors_available]
        validation=frame[(frame.next_date>'2022-12-31')&(frame.next_date<='2024-12-31')&frame.predictors_available]
        assert len(train)>0 and train.next_date.max()<=pd.Timestamp('2022-12-31')
        for state in range(4):
            audit=diagnostics[(diagnostics.grid_id==gid)&(diagnostics.current_state==state)].iloc[0]
            assert audit.status in ['regression_ready','deterministic_ready']
            subset=train[train.current_state==state];assert len(subset)==audit.n_transitions
            features=json.loads(audit.retained_features);means=json.loads(audit.training_means);scales=json.loads(audit.training_scales)
            for f in features:
                assert np.isclose(means[f],subset[f].mean()) and np.isclose(scales[f],subset[f].std(ddof=0))
            model,result=fit_transition(subset,features,means,scales)
            result.update(grid_id=gid,current_state=state,n_train=len(subset));fitting.append(result)
            folder=modeldir/gid;folder.mkdir(exist_ok=True)
            (folder/f'state_{state}_fit_diagnostics.json').write_text(json.dumps(result,indent=2))
            if not result['success']: continue
            model.update(grid_id=gid,current_state=state,train_end=design['train_end'],design_sha256=hashlib.sha256(design_bytes).hexdigest())
            (folder/f'state_{state}.json').write_text(json.dumps(model,indent=2))
            if model['type']=='multinomial_logistic':
                standardized=np.asarray(model['coefficients_standardized']);sc=np.array([scales[f] for f in features]);mu=np.array([means[f] for f in features])
                raw=standardized.copy();raw[:,1:]/=sc;raw[:,0]-=raw[:,1:]@mu
                for j,destination in enumerate(model['classes'][:-1]):
                    for index,name in enumerate(['intercept']+features):
                        coefs.append({'grid_id':gid,'current_state':state,'destination_state':destination,'reference_state':model['reference_state'],'term':name,'coefficient_standardized':standardized[j,index],'odds_ratio_per_training_sd':float(np.exp(standardized[j,index])),'coefficient_original_unit':raw[j,index],'odds_ratio_per_original_unit':float(np.exp(raw[j,index]))})
            val=validation[validation.current_state==state].copy()
            if len(val):
                logs=predict_log_proba(model,val);prob=np.exp(logs);assert np.isfinite(prob).all() and np.allclose(prob.sum(axis=1),1,atol=1e-12)
                val=val[['grid_id','date','next_date','current_state','next_state']].rename(columns={'next_state':'actual_next_state'})
                val['predicted_next_state']=prob.argmax(axis=1);val['log_probability_actual']=logs[np.arange(len(val)),val.actual_next_state.to_numpy()]
                for j,name in enumerate(['p_no_activity','p_low','p_moderate','p_high']): val[name]=prob[:,j]
                predictions.append(val)
        coverage.append({'grid_id':gid,'validation_total':int((frame.next_date>'2022-12-31').sum()),'validation_predictor_available':len(validation)})
        print(f'Fitted grid {gid}',flush=True)
    fit=pd.DataFrame(fitting);fit.to_csv(output/'fit_diagnostics.csv',index=False);pd.DataFrame(coefs).to_csv(output/'coefficients_odds_ratios.csv',index=False);pd.DataFrame(coverage).to_csv(output/'validation_coverage.csv',index=False)
    if not predictions: raise RuntimeError('No validation predictions from successful fits')
    prediction=pd.concat(predictions,ignore_index=True).sort_values(['grid_id','date']);prediction.to_parquet(output/'validation_predictions.parquet',index=False)
    rows=[];classrows=[];confusions=[];calibration=[]
    groups=[('pooled_evaluation','ALL',-1,prediction)]
    groups += [('per_grid',gid,-1,g) for gid,g in prediction.groupby('grid_id')]
    groups += [('per_grid_current_state',gid,int(state),g) for (gid,state),g in prediction.groupby(['grid_id','current_state'])]
    groups += [('per_current_state','ALL',int(state),g) for state,g in prediction.groupby('current_state')]
    for scope,gid,state,g in groups:
        values,precision,recall,f1,support=metrics(g);meta={'scope':scope,'grid_id':gid,'current_state':state}; rows.append(dict(meta,**values))
        cm=confusion_matrix(g.actual_next_state,g.predicted_next_state,labels=range(4))
        for a in range(4):
            classrows.append(dict(meta,destination_state=a,precision=precision[a],recall=recall[a],f1=f1[a],support=int(support[a])))
            for b in range(4): confusions.append(dict(meta,actual_state=a,predicted_state=b,count=int(cm[a,b])))
            probabilities=g[['p_no_activity','p_low','p_moderate','p_high']].to_numpy()[:,a];bins=np.minimum((probabilities*10).astype(int),9)
            for binid in range(10):
                mask=bins==binid
                if mask.any(): calibration.append(dict(meta,destination_state=a,bin=binid,n=int(mask.sum()),mean_probability=float(probabilities[mask].mean()),observed_fraction=float((g.actual_next_state.to_numpy()[mask]==a).mean())))
    met=pd.DataFrame(rows);met.to_csv(output/'validation_metrics.csv',index=False);pd.DataFrame(classrows).to_csv(output/'validation_destination_metrics.csv',index=False);pd.DataFrame(confusions).to_csv(output/'validation_confusion_matrices.csv',index=False);cal=pd.DataFrame(calibration);cal.to_csv(output/'validation_reliability_bins.csv',index=False)
    pergrid=met[met.scope=='per_grid'];summary=pergrid[['accuracy','balanced_accuracy','macro_f1','weighted_f1','log_loss']].agg(['mean','median','std','min','max']);summary.to_csv(output/'validation_across_grid_summary.csv')
    probabilities=prediction[['p_no_activity','p_low','p_moderate','p_high']].to_numpy();pooled=met[met.scope=='pooled_evaluation'].iloc[0].to_dict()
    observed=prediction.actual_next_state.value_counts().reindex(range(4),fill_value=0)/len(prediction)
    calibration_summary={str(j):{'mean_probability':float(probabilities[:,j].mean()),'observed_fraction':float(observed[j])} for j in range(4)}
    run={'eligible_grids':ids,'successful_state_models':int(fit.success.sum()),'failed_state_models':int((~fit.success).sum()),'fully_fitted_grids':int(fit.groupby('grid_id').success.all().sum()),'validation_rows':len(prediction),'test_evaluated':False,'penalty':None,'thresholds_changed':False,'eligibility_changed':False,'design_sha256':hashlib.sha256(design_bytes).hexdigest(),'validation_overall':pooled,'probability_min':float(probabilities.min()),'probability_max':float(probabilities.max()),'probability_sum_max_error':float(np.abs(probabilities.sum(axis=1)-1).max()),'nonfinite_log_actual':int((~np.isfinite(prediction.log_probability_actual)).sum()),'calibration_in_the_large':calibration_summary}
    (output/'run_summary.json').write_text(json.dumps(run,indent=2));(output/'frozen_design.json').write_bytes(design_bytes)
    text=f'''# Frozen 0.5° Markov fitting and validation\n\n11 pre-fit eligible grids; {run['successful_state_models']}/44 state-specific models fitted successfully, {run['failed_state_models']} failures, {run['fully_fitted_grids']} fully fitted grids. Unpenalized multinomial maximum likelihood, one independent model per grid/current state, reference destination state 3 (all these training groups contain all four destinations). Coefficients, means and scales are training-only. No class weighting, regularization, neighbor predictors, pooled fitting, validation tuning or recalibration.\n\nFrozen Q50=5.96/Q75=16.84 MW, exact 0.5° grid, four dynamic predictors, original eligibility rules and split retained. Training targets through 2022-12-31; validation targets 2023–2024. SQL explicitly selects only train/validation splits. Test evaluation was not run. The audit design configuration remains unchanged; a byte-identical copy is retained.\n\nValidation predictions: {len(prediction):,}. Overall evaluation (aggregation of independently fitted predictions): {json.dumps(pooled)}.\n\nAcross-grid summary:\n\n```\n{summary.to_string()}\n```\n\nMacro metrics use the fixed four-state label set with zero division set to zero; balanced accuracy averages recall over destinations present in the evaluated subset. Per-current-state metrics condition on the originating state; destination metrics are also supplied. Confusion matrices use actual rows and predicted columns, states 0–3. Log loss uses stable log-softmax probabilities directly without clipping; multiclass Brier is the sum of squared class errors (range 0–2).\n\nCoefficients are destination-versus-reference log odds conditional on current state. Odds ratios per training SD and per original unit are reported; intercept odds ratios are baseline odds at the corresponding zero covariate values, not an exposure effect. Effects are conditional associations, not causal effects; no confidence intervals are claimed.\n\nNumerical evidence: no invalid probability vectors; max normalization error {run['probability_sum_max_error']}; minimum probability {run['probability_min']}; maximum {run['probability_max']}; nonfinite true-class log probabilities {run['nonfinite_log_actual']}. Optimizer status, summed gradient norm and observed-information eigenvalues/condition numbers are in fit_diagnostics.csv. Fits failing optimizer/gradient/positive-information checks are reported and not used.\n\nCalibration-in-the-large (mean predicted probability versus observed frequency): {json.dumps(calibration_summary)}. Reliability bins are descriptive validation diagnostics, with counts supplied; small state groups make them noisy. Log loss/Brier combine calibration and discrimination and do not by themselves establish calibration. No calibrator was fitted. See [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html). Optimizer stopping criteria: [SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).\n\nArtifacts: JSON models per grid/state under outputs/models/markov_050; coefficients_odds_ratios.csv; fit_diagnostics.csv; validation_predictions.parquet; validation_metrics.csv; validation_destination_metrics.csv; validation_confusion_matrices.csv; validation_reliability_bins.csv; validation_coverage.csv. Every fitted grid and current state is identified in those tables. No design changes were made based on validation.\n'''
    (output/'validation_report.md').write_text(text);assert design_path.read_bytes()==design_bytes
    print(json.dumps(run,indent=2),flush=True)
if __name__=='__main__': main()
