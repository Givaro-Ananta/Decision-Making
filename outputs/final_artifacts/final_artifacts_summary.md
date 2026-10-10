# Final reporting artifacts

These artifacts render frozen outputs only. No fitting, inference, re-screening, threshold/state changes, weighting, calibration, decision optimization or test rerun was performed.

Frozen primary design: exact 0.5° grid; 46 eligible grids of 187 mainland Sumatra study grids; one independent unweighted multinomial transition model for each grid/current state (138 models). No neighbors or pooled fitting. Predictors: temperature_max_c_t, relative_humidity_mean_t, wind_speed_mean_t, rainfall_mm_t. WorldCover was audited but is not a final model predictor. Original numerical/sample screens remain unchanged.

States: 0 No Activity when hotspot_count=0; 1 Active/non-high when detections exist and FRP sum <16.84 MW; 2 High when detections exist and FRP sum ≥16.84 MW. High threshold is training-only active-grid-day Q75. These are observed detection-intensity states, not calibrated fire severity or daily energy.

Training cutoff 2022-12-31; validation targets 2023–2024; test targets 2025-01-01 through 2026-08-31. Final test evaluation completed once on 27,968 observations across all 46 grids, without missing-predictor exclusions. Standard argmax is the primary rule. Guarded 0.20 and two-stage 0.20/0.50 are previously declared sensitivities only.

| Primary metric | Value |
|---|---:|
| Accuracy | 0.860912 |
| Balanced Accuracy | 0.432223 |
| Macro F1 | 0.443943 |
| Weighted F1 | 0.821760 |
| Log Loss | 0.411651 |

| State | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|
| No Activity | 0.883578 | 0.982329 | 0.930341 | 0.805361 |
| Active/non-high | 0.294404 | 0.045015 | 0.078090 | 0.737647 |
| High | 0.404645 | 0.269325 | 0.323400 | 0.858698 |

Main limitations: class imbalance makes 86.09% accuracy an incomplete account of activity recovery; balanced accuracy is 43.22%, non-high F1 0.078 and High F1 0.323. Independent eligibility limits inference to 46 grids; results do not establish performance for excluded grids. FIRMS no detection is not proof of absence of fire. The state partition is relative observed FRP intensity. Environmental regridding and source coverage limit covariate information. Only aggregate/descriptive calibration is shown; non-high and High probabilities overestimate observed frequency by 1.18 and 1.06 percentage points. There are 103 true-class probabilities below 1%. Threshold sensitivities recover more activity at a substantial false-positive cost; no sensitivity is adopted as primary.

Artifacts and interpretations:

- [final_results_table.csv](final_results_table.csv): Primary pooled evaluation of independent predictions, with full-precision overall metrics and state-specific precision/recall/F1/one-vs-rest AUC. Values are proportions (0–1), except log loss; per-state n_observations is true-state support.

- [confusion_matrix.png](confusion_matrix.png): Primary argmax only. Counts and row-normalized percentages show 2,326 true non-high and 787 true High observations predicted as No Activity. Strong inactive recall dominates overall accuracy.

- [per_state_metrics.png](per_state_metrics.png): The three-state bars expose unequal recovery: non-high recall 4.50% and High recall 26.93%, compared with inactive recall 98.23%.

- [eligible_grids_map.png](eligible_grids_map.png): All 187 exact regular cells intersecting the frozen mainland footprint are outlined. Orange land support and cell edges identify the 46 final modeled grids; unshaded cells were not modeled. Saved study_geometry defines the mask, with no reconstructed/new geographic filter.

- [probability_calibration.png](probability_calibration.png): Uses the existing saved test reliability bins derived from saved probabilities; no model calls, new calibration fit or bin optimization. Points are per-class observed fractions versus mean predictions, with bin counts encoded by marker size. Sparse bins and aggregate ECE do not establish conditional calibration.

- [decision_rule_sensitivity.png](decision_rule_sensitivity.png): Guarded and two-stage rules raise non-high recall to 31.06% and 44.16%, with 2,416 and 4,318 additional non-high false positives. They remain sensitivities; primary argmax is unchanged.

- [methodology_flowchart.png](methodology_flowchart.png): Shows data sources, frozen mainland domain, daily grid panel, three states, chronology, training-only screening, per-grid/current-state fitting and next-day argmax. The split precedes screening/fitting to reflect actual leakage controls. WorldCover is context only; the four predictors are dynamic.

Sources: configs/final_primary_050_3state.json; outputs/reports/final_test_050_3state/frozen_design.json and saved result tables; data/processed/modeling_050/grid.parquet. Reliability visualization reuses the saved test probability bins rather than performing any new inference. Source SHA-256 digests and renderer version are in reporting_provenance.json.
