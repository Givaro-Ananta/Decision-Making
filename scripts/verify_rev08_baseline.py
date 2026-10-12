"""Independently check frozen REV-08 counts, paired rows and score tables."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/attachments/rev08_20261011'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((EVIDENCE / 'verification.json').read_text())
    out = ROOT / receipt['run_path']
    source = ROOT / 'outputs/revision_review/REV03_20261011_ec3d83da299e'
    checks = []

    def check(name, value):
        if not bool(value):
            raise AssertionError(name)
        checks.append(name)

    for path, digest in receipt['protected'].items():
        check('protected:' + path, sha(ROOT / path) == digest)
    for name, digest in receipt['outputs'].items():
        check('output:' + name, sha(out / name) == digest)
    for name, digest in receipt['evidence'].items():
        check('evidence:' + name, sha(EVIDENCE / name) == digest)
    check('executed_script_identity', sha(EVIDENCE / 'executed_compare_rev08_baseline.py') == receipt['script_sha256'])
    check('maintained_script_identity', sha(ROOT / 'scripts/compare_rev08_baseline.py') == receipt['documentation_fix']['maintained_script_sha256'])
    check('completion_receipt_identity', json.loads((out / 'run_completed.json').read_text())['verification_sha256'] == sha(EVIDENCE / 'verification.json'))
    ledger = pd.read_parquet(source / 'transition_ledger.parquet')
    train = ledger[ledger['included'] & ledger['split'].eq('train')]
    check('train_only', len(train) == 104144 and train.next_date.max() <= pd.Timestamp('2022-12-31'))
    counts = Counter((str(g), int(c), int(y)) for g, c, y in train[['grid_id', 'current_state', 'next_state']].itertuples(index=False, name=None))
    probabilities = {}
    rates = pd.read_csv(out / 'training_transition_rates.csv')
    for row in rates.itertuples(index=False):
        ns = [counts[(str(row.grid_id), int(row.current_state), y)] for y in range(3)]
        check(f'counts:{row.grid_id}:{row.current_state}', ns == [row.n_0, row.n_1, row.n_2] and sum(ns) == row.training_rows and sum(ns) > 0)
        probabilities[(str(row.grid_id), int(row.current_state))] = np.array(ns) / sum(ns)
        check(f'rates:{row.grid_id}:{row.current_state}', np.allclose(probabilities[(str(row.grid_id), int(row.current_state))], [row.baseline_p_0, row.baseline_p_1, row.baseline_p_2], atol=1e-14, rtol=0))
    frames = {}
    for split in ['validation', 'test']:
        saved = pd.read_parquet(source / f'{split}_predictions.parquet')
        paired = pd.read_parquet(out / f'{split}_paired_predictions.parquet')
        check(split + ':all_saved_columns_unchanged', saved.equals(paired[saved.columns]))
        independent = np.vstack([probabilities[(str(g), int(c))] for g, c in paired[['grid_id', 'current_state']].itertuples(index=False, name=None)])
        check(split + ':independent_baseline', np.allclose(independent, paired[[f'baseline_p_{y}' for y in range(3)]], atol=1e-14, rtol=0))
        truth = ledger[ledger['included'] & ledger['split'].eq(split)].sort_values(['grid_id', 'next_date']).reset_index(drop=True)
        check(split + ':ledger_labels_and_keys', paired[['grid_id', 'date', 'next_date']].equals(truth[['grid_id', 'date', 'next_date']]) and np.array_equal(paired.current_state, truth.current_state.astype(int)) and np.array_equal(paired.actual_next_state, truth.next_state.astype(int)))
        frames[split + '_all'] = paired
    frames['test_2025'] = frames['test_all'][frames['test_all'].next_date.dt.year.eq(2025)]
    frames['test_2026_JanAug'] = frames['test_all'][frames['test_all'].next_date.dt.year.eq(2026)]
    for table in ['overall_comparison.csv', 'per_grid_comparison.csv']:
        rows = pd.read_csv(out / table)
        check(table + ':group_count', len(rows) == (4 if table.startswith('overall') else 184))
        for row in rows.itertuples(index=False):
            sub = frames[row.period]
            if row.grid_id != 'ALL':
                sub = sub[sub.grid_id.eq(str(row.grid_id))]
            label = f'{row.period}:{row.grid_id}'
            y = sub.actual_next_state.to_numpy(dtype=int)
            check(label + ':support', len(y) == row.n and all(int((y == j).sum()) == getattr(row, f'support_{j}') for j in range(3)))
            metrics = {}
            for model, prefix in [('baseline', 'baseline_p'), ('covariate', 'p')]:
                p = sub[[f'{prefix}_{j}' for j in range(3)]].to_numpy()
                actual = p[np.arange(len(y)), y]
                loss = -math.fsum(math.log(float(x)) for x in actual) / len(y)
                brier = float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))
                metrics[model] = loss, brier
                check(label + ':' + model, abs(loss - getattr(row, model + '_log_loss')) < 1e-12 and abs(brier - getattr(row, model + '_brier')) < 1e-12 and getattr(row, model + '_zero_probability_actual') == 0)
            check(label + ':deltas', abs(metrics['covariate'][0] - metrics['baseline'][0] - row.delta_log_loss) < 1e-12 and abs(metrics['covariate'][1] - metrics['baseline'][1] - row.delta_brier) < 1e-12)
    check('eight_tests_passed', 'Ran 8 tests' in (out / 'regression_tests.txt').read_text() and (out / 'regression_tests.txt').read_text().rstrip().endswith('OK'))
    pergrid = pd.read_csv(out / 'per_grid_comparison.csv')
    heterogeneity = {period: {'grids': len(g), 'covariate_lower_log_loss': int(g.delta_log_loss.lt(0).sum()), 'covariate_lower_brier': int(g.delta_brier.lt(0).sum())} for period, g in pergrid.groupby('period')}
    result = dict(date='2026-10-11', checks_passed=len(checks), checks=checks, heterogeneity=heterogeneity, verification_sha256=sha(EVIDENCE / 'verification.json'), verifier_sha256=sha(Path(__file__)), limitations='Read-only numerical/input check of available revised outputs; no full historical replay, physical observation coverage or significance inference')
    target = EVIDENCE / 'independent_verification.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(checks_passed=len(checks), heterogeneity=heterogeneity), indent=2))


if __name__ == '__main__':
    main()
