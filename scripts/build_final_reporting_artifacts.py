#!/usr/bin/env python3
"""Render publication artifacts from frozen output tables and saved grid geometry only."""
import os,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/final_artifacts'
OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Patch,FancyBboxPatch
from matplotlib.collections import PatchCollection
from shapely import from_wkb
from shapely.ops import unary_union
LABELS=['No Activity','Active/non-high','High']
COLORS=['#39748f','#db9c32','#b64f51']
REPORT=ROOT/'outputs/reports/final_test_050_3state'

def save(fig,name):
    fig.savefig(OUT/name,dpi=240,bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    files=[REPORT/'test_summary.json',REPORT/'test_metrics.csv',REPORT/'test_state_metrics_auc.csv',REPORT/'test_confusion_matrices.csv',REPORT/'test_reliability_bins.csv',REPORT/'sensitivity_false_positive_costs.csv',REPORT/'frozen_design.json',ROOT/'data/processed/modeling_050/grid.parquet']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    summary=json.loads((REPORT/'test_summary.json').read_text());design=json.loads((REPORT/'frozen_design.json').read_text())
    assert summary['evaluation_status']=='completed_once' and summary['n_test_grids']==46
    metrics=pd.read_csv(REPORT/'test_metrics.csv');states=pd.read_csv(REPORT/'test_state_metrics_auc.csv')
    primary=metrics[(metrics.variant=='primary_argmax')&(metrics.scope=='overall')].iloc[0]
    cls=states[(states.variant=='primary_argmax')&(states.scope=='overall')].sort_values('destination_state')
    rows=[]
    for key,title in [('accuracy','Accuracy'),('balanced_accuracy','Balanced Accuracy'),('macro_f1','Macro F1'),('weighted_f1','Weighted F1'),('log_loss','Log Loss')]:
        rows.append({'scope':'overall','state':'All states','metric':title,'value':float(primary[key]),'n_observations':summary['n_test_observations'],'prediction_rule':'primary_argmax'})
    for r in cls.itertuples():
        for key,title in [('precision','Precision'),('recall','Recall'),('f1','F1'),('auc_ovr','AUC (one-vs-rest)')]:
            rows.append({'scope':'per_state','state':LABELS[r.destination_state],'metric':title,'value':float(getattr(r,key)),'n_observations':r.n_true,'prediction_rule':'primary_argmax'})
    pd.DataFrame(rows).to_csv(OUT/'final_results_table.csv',index=False)
    cmdata=pd.read_csv(REPORT/'test_confusion_matrices.csv');cm=cmdata[(cmdata.variant=='primary_argmax')&(cmdata.scope=='overall')].pivot(index='actual_state',columns='predicted_state',values='count').reindex(index=range(3),columns=range(3)).to_numpy()
    assert cm.sum()==summary['n_test_observations']
    fig,ax=plt.subplots(figsize=(7,5.8));fractions=cm/cm.sum(axis=1,keepdims=True);im=ax.imshow(fractions,cmap='Blues',vmin=0,vmax=1)
    for a in range(3):
        for b in range(3):ax.text(b,a,f'{cm[a,b]:,}\n{fractions[a,b]:.1%}',ha='center',va='center',color='white' if fractions[a,b]>.55 else '#172b3a',fontsize=12)
    ax.set_xticks(range(3),LABELS);ax.set_yticks(range(3),LABELS);ax.set_xlabel('Predicted next-day state');ax.set_ylabel('Actual next-day state');ax.set_title('Final primary argmax — test confusion matrix\nCounts and percentages within each actual state',pad=14)
    fig.colorbar(im,ax=ax,label='Fraction of actual state');save(fig,'confusion_matrix.png')
    fig,ax=plt.subplots(figsize=(8,5));x=np.arange(3)
    for j,(key,title,color) in enumerate([('precision','Precision','#39748f'),('recall','Recall','#db9c32'),('f1','F1','#b64f51')]):
        vals=cls[key].to_numpy();bars=ax.bar(x+(j-1)*.25,vals,.25,label=title,color=color)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=3,fontsize=10)
    ax.set_xticks(x,LABELS);ax.set_ylim(0,1.13);ax.set_ylabel('Metric value');ax.set_title('Per-state test performance — primary argmax');ax.legend(loc='upper right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);save(fig,'per_state_metrics.png')
    grid=pd.read_parquet(ROOT/'data/processed/modeling_050/grid.parquet');eligible=set(design['eligible_grid_ids']);assert len(grid)==187 and len(eligible)==46 and eligible.issubset(set(grid.grid_id))
    study=[from_wkb(g) for g in grid.study_geometry];boundary=unary_union(study)
    def polygons(geometry):return [geometry] if geometry.geom_type=='Polygon' else [p for p in geometry.geoms if p.geom_type=='Polygon']
    def patches(geometry):return [Polygon(np.asarray(g.exterior.coords),closed=True) for g in polygons(geometry)]
    fig,ax=plt.subplots(figsize=(7.3,8.5))
    ax.add_collection(PatchCollection(patches(boundary),facecolor='#e7ecea',edgecolor='#455b55',linewidth=.8,zorder=1))
    allrect=[Polygon(np.asarray(from_wkb(g).exterior.coords),closed=True) for g in grid.geometry]
    ax.add_collection(PatchCollection(allrect,facecolor='none',edgecolor='#9ba6ad',linewidth=.5,zorder=2))
    fills=[];edges=[]
    for r in grid.itertuples():
        if r.grid_id in eligible:fills+=patches(from_wkb(r.study_geometry));edges+=patches(from_wkb(r.geometry))
    ax.add_collection(PatchCollection(fills,facecolor='#eaa044',edgecolor='none',alpha=.85,zorder=3));ax.add_collection(PatchCollection(edges,facecolor='none',edgecolor='#aa5c08',linewidth=1,zorder=4))
    xmin,ymin,xmax,ymax=boundary.bounds;ax.set_xlim(xmin-.45,xmax+.45);ax.set_ylim(ymin-.4,ymax+.4);ax.set_aspect(1/np.cos(np.radians((ymin+ymax)/2)))
    ax.set_xlabel('Longitude (°E)');ax.set_ylabel('Latitude (°; negative = south)');ax.grid(alpha=.15);ax.set_title('Mainland Sumatra — exact 0.5° grid\n46 eligible model grids / 187 study grids')
    ax.legend(handles=[Patch(facecolor='#e7ecea',edgecolor='#455b55',label='Frozen mainland study footprint'),Patch(facecolor='none',edgecolor='#9ba6ad',label='All 187 regular grid cells'),Patch(facecolor='#eaa044',edgecolor='#aa5c08',label='46 eligible grids (land support shaded)')],loc='lower left',fontsize=9)
    save(fig,'eligible_grids_map.png')
    bins=pd.read_csv(REPORT/'test_reliability_bins.csv');bins=bins[bins.scope=='overall'];fig,axes=plt.subplots(1,3,figsize=(12.5,4.2),sharex=True,sharey=True)
    for j,ax in enumerate(axes):
        b=bins[bins.destination_state==j].sort_values('bin');assert b.n.sum()==summary['n_test_observations']
        ax.plot([0,1],[0,1],'--',color='#7f8c8d',label='Ideal reliability');ax.plot(b.mean_probability,b.observed_fraction,color=COLORS[j],linewidth=1.6)
        ax.scatter(b.mean_probability,b.observed_fraction,s=25+120*np.sqrt(b.n/b.n.max()),color=COLORS[j],edgecolor='white',zorder=3,label='Saved test bins')
        e=summary['classwise_calibration'][j]['ece_10_equal_width_bins'];ax.set_title(f'{LABELS[j]}\nECE (10 bins): {e:.4f}');ax.set_xlabel('Mean predicted probability');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes[0].set_ylabel('Observed destination fraction');axes[-1].legend(fontsize=8,loc='lower right');fig.suptitle('Saved-probability reliability — no recalibration\n10 equal-width bins; marker area varies with bin count',y=1.05);fig.tight_layout();save(fig,'probability_calibration.png')
    variants=['primary_argmax','sensitivity_guarded_020','sensitivity_two_stage_020_050'];names=['Primary\nargmax','Guarded\n0.20','Two-stage\n0.20 / 0.50']
    cs=states[(states.scope=='overall')&(states.destination_state==1)].set_index('variant').loc[variants];fp=pd.read_csv(REPORT/'sensitivity_false_positive_costs.csv').set_index('variant').loc[variants]
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8));x=np.arange(3)
    for offset,key,title,color in [(-.19,'recall','Recall','#db9c32'),(.19,'precision','Precision','#39748f')]:
        bars=axes[0].bar(x+offset,cs[key],.36,label=title,color=color);axes[0].bar_label(bars,labels=[f'{v:.1%}' for v in cs[key]],padding=3)
    axes[0].set_ylim(0,.57);axes[0].set_ylabel('Active/non-high metric');axes[0].legend();axes[0].set_title('Recall gain versus precision')
    bars=axes[1].bar(x,fp.nonhigh_fp,color=['#39748f','#db9c32','#b64f51']);axes[1].bar_label(bars,labels=[f'{int(v):,}' for v in fp.nonhigh_fp],padding=4);axes[1].set_ylim(0,fp.nonhigh_fp.max()*1.22);axes[1].set_ylabel('Non-high false positives');axes[1].set_title('False positives: true No Activity or High')
    for i,r in enumerate(fp.itertuples()):axes[1].text(i,fp.nonhigh_fp.max()*1.13,'Reference' if i==0 else f'+{int(r.delta_nonhigh_fp):,} FP / +{int(r.delta_nonhigh_tp):,} TP',ha='center',fontsize=9)
    for ax in axes:ax.set_xticks(x,names);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Previously defined decision sensitivities — same frozen test probabilities\nPrimary argmax remains the final rule',y=1.04);fig.tight_layout();save(fig,'decision_rule_sensitivity.png')
    fig,ax=plt.subplots(figsize=(10,11));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    boxes=[('FIRMS + ERA5-Land + CHIRPS + WorldCover','WorldCover: coverage/context audit; not a final predictor'),('Sumatra filtering','Mainland polygon ∩ verified original spatial extent'),('Exact 0.5° daily grid-day panel','Complete dates; explicit no-detection zeros; same-grid covariates'),('FRP-based three states','No Activity; active <16.84 MW; active ≥16.84 MW'),('Chronological train / validation / test split','Train ≤2022-12-31; validation 2023–2024; test 2025–2026-08-31'),('Training-only eligibility screening','46 of 187 grids; unchanged sample and numerical screens'),('Per-grid covariate-dependent Markov model','One unweighted multinomial model per grid × current state'),('Final next-day state prediction','Three probabilities → standard argmax; same grid, exactly one day')]
    for i,(title,subtitle) in enumerate(boxes):
        y=.94-i*.112;rect=FancyBboxPatch((.07,y-.07),.86,.077,boxstyle='round,pad=0.012',facecolor='#edf3f5' if i!=7 else '#e7eee5',edgecolor='#39748f',linewidth=1.4);ax.add_patch(rect)
        ax.text(.5,y-.012,title,ha='center',va='center',fontsize=12,fontweight='bold');ax.text(.5,y-.045,subtitle,ha='center',va='center',fontsize=9)
        if i<7:ax.annotate('',xy=(.5,y-.107),xytext=(.5,y-.083),arrowprops={'arrowstyle':'->','color':'#39748f','lw':1.8})
    ax.text(.5,.025,'Threshold derivation, preprocessing scales and fitting use training only.\nValidation sensitivities did not alter the frozen primary; test inference completed once.',ha='center',fontsize=10,color='#34495e')
    ax.set_title('Frozen methodology — independent same-grid next-day prediction',fontsize=14,pad=16);save(fig,'methodology_flowchart.png')
    interpretations={
        'final_results_table.csv':'Primary pooled evaluation of independent predictions, with full-precision overall metrics and state-specific precision/recall/F1/one-vs-rest AUC. Values are proportions (0–1), except log loss; per-state n_observations is true-state support.',
        'confusion_matrix.png':'Primary argmax only. Counts and row-normalized percentages show 2,326 true non-high and 787 true High observations predicted as No Activity. Strong inactive recall dominates overall accuracy.',
        'per_state_metrics.png':'The three-state bars expose unequal recovery: non-high recall 4.50% and High recall 26.93%, compared with inactive recall 98.23%.',
        'eligible_grids_map.png':'All 187 exact regular cells intersecting the frozen mainland footprint are outlined. Orange land support and cell edges identify the 46 final modeled grids; unshaded cells were not modeled. Saved study_geometry defines the mask, with no reconstructed/new geographic filter.',
        'probability_calibration.png':'Uses the existing saved test reliability bins derived from saved probabilities; no model calls, new calibration fit or bin optimization. Points are per-class observed fractions versus mean predictions, with bin counts encoded by marker size. Sparse bins and aggregate ECE do not establish conditional calibration.',
        'decision_rule_sensitivity.png':'Guarded and two-stage rules raise non-high recall to 31.06% and 44.16%, with 2,416 and 4,318 additional non-high false positives. They remain sensitivities; primary argmax is unchanged.',
        'methodology_flowchart.png':'Shows data sources, frozen mainland domain, daily grid panel, three states, chronology, training-only screening, per-grid/current-state fitting and next-day argmax. The split precedes screening/fitting to reflect actual leakage controls. WorldCover is context only; the four predictors are dynamic.'}
    text='''# Final reporting artifacts

These artifacts render frozen outputs only. No fitting, inference, re-screening, threshold/state changes, weighting, calibration, decision optimization or test rerun was performed.

Frozen primary design: exact 0.5° grid; 46 eligible grids of 187 mainland Sumatra study grids; one independent unweighted multinomial transition model for each grid/current state (138 models). No neighbors or pooled fitting. Predictors: temperature_max_c_t, relative_humidity_mean_t, wind_speed_mean_t, rainfall_mm_t. WorldCover was audited but is not a final model predictor. Original numerical/sample screens remain unchanged.

States: 0 No Activity when hotspot_count=0; 1 Active/non-high when detections exist and FRP sum <16.84 MW; 2 High when detections exist and FRP sum ≥16.84 MW. High threshold is training-only active-grid-day Q75. These are observed detection-intensity states, not calibrated fire severity or daily energy.

Training cutoff 2022-12-31; validation targets 2023–2024; test targets 2025-01-01 through 2026-08-31. Final test evaluation completed once on 27,968 observations across all 46 grids, without missing-predictor exclusions. Standard argmax is the primary rule. Guarded 0.20 and two-stage 0.20/0.50 are previously declared sensitivities only.

| Primary metric | Value |
|---|---:|
'''
    for key,title in [('accuracy','Accuracy'),('balanced_accuracy','Balanced Accuracy'),('macro_f1','Macro F1'),('weighted_f1','Weighted F1'),('log_loss','Log Loss')]:text+=f'| {title} | {primary[key]:.6f} |\n'
    text+='\n| State | Precision | Recall | F1 | AUC |\n|---|---:|---:|---:|---:|\n'
    for r in cls.itertuples():text+=f'| {LABELS[r.destination_state]} | {r.precision:.6f} | {r.recall:.6f} | {r.f1:.6f} | {r.auc_ovr:.6f} |\n'
    text+='''
Main limitations: class imbalance makes 86.09% accuracy an incomplete account of activity recovery; balanced accuracy is 43.22%, non-high F1 0.078 and High F1 0.323. Independent eligibility limits inference to 46 grids; results do not establish performance for excluded grids. FIRMS no detection is not proof of absence of fire. The state partition is relative observed FRP intensity. Environmental regridding and source coverage limit covariate information. Only aggregate/descriptive calibration is shown; non-high and High probabilities overestimate observed frequency by 1.18 and 1.06 percentage points. There are 103 true-class probabilities below 1%. Threshold sensitivities recover more activity at a substantial false-positive cost; no sensitivity is adopted as primary.

Artifacts and interpretations:

'''
    for name,explanation in interpretations.items():text+=f'- [{name}]({name}): {explanation}\n\n'
    text+='Sources: configs/final_primary_050_3state.json; outputs/reports/final_test_050_3state/frozen_design.json and saved result tables; data/processed/modeling_050/grid.parquet. Reliability visualization reuses the saved test probability bins rather than performing any new inference. Source SHA-256 digests and renderer version are in reporting_provenance.json.\n'
    (OUT/'final_artifacts_summary.md').write_text(text)
    for name,digest in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    (OUT/'reporting_provenance.json').write_text(json.dumps({'source_sha256':hashes,'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inference_or_fitting_performed':False,'frozen_primary_changed':False,'eligible_grids':46,'study_grids':187,'test_observations':27968,'calibration_source':'Saved test reliability bins derived from saved test probabilities'},indent=2))
    print('Created final table, six figures, summary and provenance. Frozen source hashes unchanged.')
if __name__=='__main__':main()
