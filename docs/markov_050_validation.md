# Frozen 0.5° training and validation

The exact design and eligible grid IDs are preserved in `configs/model_fitting_050.json`. The earlier `configs/modeling_design_050.json` remains the historical pre-fit design snapshot; its `fit_performed: false` describes that earlier audit stage. Completed fitting status is recorded in the fitting manifest and validation run summary.

Run from the repository root:

```bash
PYTHONPATH=/tmp/sumatra-audit-deps python scripts/fit_markov_050_validation.py
python -m unittest discover -s tests -v
```

Dependencies: DuckDB, pandas, NumPy, SciPy, scikit-learn, PyArrow. Scikit-learn supplies evaluation metrics only; estimation uses the unpenalized multinomial likelihood in `src/models/covariate_markov.py`.

Each grid/current-state JSON model contains destination classes, reference class, coefficients, training feature means/scales and design hash. Load it using `json.loads`, then call `predict_log_proba` from the model module on same-grid current-day feature rows. Exponentiate for probabilities in the fixed destination order 0–3. No pickle or external fitted estimator is needed.

Models are under `outputs/models/markov_050/<grid_id>/state_<state>.json`. Detailed validation results are under `outputs/reports/markov_050_validation/`. The original preprocessing/audit data are unchanged. Only train and validation splits are selected by the fitting script. No final test evaluation is implemented in this command.

No validation-dependent threshold, feature, eligibility, regularization or calibration changes were made. Validation exposes poor Low/Moderate classification; report this before making any scientific design change.
