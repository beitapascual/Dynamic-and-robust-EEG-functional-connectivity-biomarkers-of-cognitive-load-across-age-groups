"""Step 5 — Statistical analysis of participant-level graph metrics.

Use paired Wilcoxon signed-rank tests to compare relative cognitive-load
classes across five metrics, five frequency bands and two brain regions.
Apply Benjamini–Hochberg FDR correction to 300 tests per dataset.
"""

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from corsi_04_compute_graph_metrics import (
    BANDS as GRAPH_BANDS, METRICS, REGIONS, OUTPUT_SUBDIR, TRIAL_SELECTION,
    ZERO_POLICY, require_success_only, fingerprint,
)

DATASETS = tuple(REGIONS)
INPUT_ROOT = Path('processed') / 'success_only'
BANDS = tuple(band for band in GRAPH_BANDS if band != 'broadband')
COMPARISONS = ((2, 1), (3, 1), (4, 1), (3, 2), (4, 2), (4, 3))
N_TESTS = 300
KEYS = ['subject_id', 'relative_class', 'band', 'region']
WILCOXON_OPTIONS = dict(alternative='two-sided', zero_method='wilcox',
                        correction=False, method='auto', nan_policy='raise')


def read_grouped_graph_metrics(path, dataset):
    """Read once, preserve subject IDs and reject duplicate inferential units."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing Step 4 participant metrics: {path}. Finish Step 4 first.')
    data = pd.read_csv(path, dtype={'subject_id': str})
    counts = [f'n_{metric}' for metric in METRICS]
    required = ['dataset', *KEYS, 'n_epochs', *METRICS, *counts]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f'Missing graph-metric columns: {sorted(missing)}')
    if data[required[:5]].isna().any().any() or not data['dataset'].eq(dataset).all():
        raise ValueError('Missing identifiers or mixed datasets in the graph-metrics CSV')
    if not data['band'].isin(GRAPH_BANDS).all():
        raise ValueError('Unknown frequency band in the graph-metrics CSV')
    data = data.loc[data['band'].isin(BANDS)].copy()
    if not data['region'].isin(REGIONS[dataset]).all():
        raise ValueError('Unknown region in the graph-metrics CSV')
    if not data['relative_class'].isin((1, 2, 3, 4)).all():
        raise ValueError('Expected relative classes 1, 2, 3 and 4')
    if data.duplicated(KEYS).any():
        raise ValueError('Duplicate participant/class/band/region graph-metric rows')
    expected = {(band, region) for band in BANDS for region in REGIONS[dataset]}
    if set(zip(data['band'], data['region'])) != expected:
        raise ValueError('The graph metrics CSV must contain all five bands and both regions')
    data[list(METRICS)] = data[list(METRICS)].apply(pd.to_numeric, errors='raise')
    for column in ['n_epochs', *counts]:
        values = pd.to_numeric(data[column], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f'Invalid epoch counts: {column}')
        data[column] = values.astype(np.int64)
    if (data['n_epochs'] == 0).any():
        raise ValueError('A participant/class metric row must refer to at least one epoch')
    for metric, count in zip(METRICS, counts):
        if (data[count] > data['n_epochs']).any() or not np.array_equal(data[count].eq(0), data[metric].isna()):
            raise ValueError(f'Inconsistent values/counts for {metric}')
    return data


def paired_test(values_a, values_b):
    """Use complete subject pairs for this contrast; pass differences to SciPy."""
    finite_a, finite_b = np.isfinite(values_a), np.isfinite(values_b)
    paired = finite_a & finite_b
    a, b = values_a[paired], values_b[paired]
    differences = a - b
    n_pairs = len(differences)
    n_nonzero = int(np.count_nonzero(differences))
    statistic, p_raw, z_value, effect_r, status = np.nan, np.nan, np.nan, np.nan, 'no_pairs'
    if n_pairs and not n_nonzero:
        statistic, p_raw, z_value, effect_r, status = 0.0, 1.0, 0.0, 0.0, 'all_zero_differences'
    elif n_pairs:
        if not np.isfinite(differences).all():
            raise ValueError('Nonfinite differences between finite subject metrics_table')
        result = wilcoxon(differences, **WILCOXON_OPTIONS)
        statistic, p_raw, status = float(result.statistic), float(result.pvalue), 'ok'
        if not np.isfinite(p_raw) or not 0 <= p_raw <= 1:
            raise ValueError('SciPy returned an invalid Wilcoxon p-value')
        # The automatic p-value is preserved. A separate asymptotic call exposes
        # SciPy's Z with its tie/zero corrections for the legacy r=|Z|/sqrt(N).
        asymptotic = wilcoxon(differences, **{**WILCOXON_OPTIONS, 'method': 'asymptotic'})
        z_value = float(asymptotic.zstatistic)
        effect_r = abs(z_value) / np.sqrt(n_pairs)
        if not np.isfinite(effect_r):
            raise ValueError('SciPy returned an invalid Wilcoxon effect size')
    return {
        'n_subjects_class_a': int(finite_a.sum()), 'n_subjects_class_b': int(finite_b.sum()),
        'n_pairs': n_pairs, 'n_nonzero_pairs': n_nonzero,
        'n_zero_differences': n_pairs - n_nonzero,
        'median_class_a': float(np.median(a)) if n_pairs else np.nan,
        'median_class_b': float(np.median(b)) if n_pairs else np.nan,
        'median_difference': float(np.median(differences)) if n_pairs else np.nan,
        'wilcoxon_statistic': statistic, 'wilcoxon_z': z_value,
        'effect_size_r': effect_r, 'p_raw': p_raw, 'status': status,
    }


def analyze_dataset(metrics_table, dataset, alpha=0.05):
    """Align subjects once, run 300 contrasts and call BH exactly once."""
    if not 0 < alpha < 1:
        raise ValueError('Alpha must be between 0 and 1')
    wide = metrics_table.pivot(index=['band', 'region', 'subject_id'],
                       columns='relative_class', values=list(METRICS))
    rows = []
    for band in BANDS:
        for region in REGIONS[dataset]:
            group = wide.loc[(band, region)]
            for metric in METRICS:
                values = group[metric].reindex(columns=(1, 2, 3, 4)).to_numpy(dtype=float)
                for class_a, class_b in COMPARISONS:
                    rows.append({
                        'dataset': dataset, 'metric': metric, 'band': band, 'region': region,
                        'comparison': f'CL{class_a}_vs_CL{class_b}',
                        'class_a': class_a, 'class_b': class_b,
                        **paired_test(values[:, class_a - 1], values[:, class_b - 1]),
                    })
    results = pd.DataFrame(rows)
    if len(results) != N_TESTS:
        raise ValueError(f'Expected {N_TESTS} contrasts, got {len(results)}')
    # An untestable contrast retains its slot in the planned 300-test family.
    # Its raw p remains NaN; only the FDR input uses conservative p=1.
    results['p_for_fdr'] = results['p_raw'].fillna(1.0)
    reject, adjusted, _, _ = multipletests(results['p_for_fdr'].to_numpy(),
                                          alpha=alpha, method='fdr_bh')
    results['p_global_bh_300'] = adjusted
    results['reject_global_bh_300'] = reject
    return results


def save_results(results, dataset, source, output_dir, alpha=0.05, *,
                 aggregation='metrics_of_mean_aec'):
    """Save test results and analysis settings."""
    source, output_dir = Path(source), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f'{dataset}_wilcoxon_global_bh_300.csv'
    results.to_csv(output, index=False)
    metadata = {
        'dataset': dataset, 'source_graph_metrics': str(source.resolve()),
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'inferential_unit': 'subject; one graph-metric value per class/band/region',
        'graph_aggregation': aggregation,
        'metrics': list(METRICS), 'bands': list(BANDS), 'regions': list(REGIONS[dataset]),
        'class_comparisons': [list(pair) for pair in COMPARISONS],
        'difference_direction': 'class_a minus class_b',
        'wilcoxon_options': WILCOXON_OPTIONS, 'difference_rounding': None,
        'effect_size_r': 'abs(SciPy asymptotic Z) / sqrt(n_pairs); unsigned magnitude',
        'effect_size_z_options': {**WILCOXON_OPTIONS, 'method': 'asymptotic'},
        'effect_size_note': 'Z includes SciPy tie/zero corrections. The automatic raw p-value and global BH family are unchanged.',
        'multiple_testing': 'one fdr_bh call over all 300 contrasts within this dataset',
        'family_size': N_TESTS, 'alpha': alpha,
        'no_pairs_policy': 'raw p=NaN; FDR input=1; retain family size 300',
        'all_zero_policy': 'statistic=0; raw p=1',
        'status_counts': {key: int(value) for key, value in results['status'].value_counts().items()},
        'significant_tests': int(results['reject_global_bh_300'].sum()),
        'versions': {name: version(name) for name in ('scipy', 'statsmodels', 'pandas', 'numpy')},
    }
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'{dataset} | contrasts={N_TESTS} | significant={metadata["significant_tests"]} '
          f'| no_pairs={metadata["status_counts"].get("no_pairs", 0)} | saved={output}')
    return output


def validate_graph_metrics(path, metrics_table, dataset):
    """Check HDF5 metrics and CSV summaries are consistent."""
    with h5py.File(path, 'r') as source:
        if source.attrs.get('graph_aggregation') != 'metrics_of_mean_aec' or source.attrs.get('dataset') != dataset:
            raise ValueError('Graph metric metadata do not match the expected analysis')
        if (tuple(source['metric_names'].asstr()[:]) != METRICS or
                tuple(source['region_names'].asstr()[:]) != tuple(REGIONS[dataset])):
            raise ValueError('Unexpected metric or region order')
        keys = pd.MultiIndex.from_arrays(
            [source['metadata/subject_id'].asstr()[:], source['metadata/relative_class'][:]],
            names=['subject_id', 'relative_class'])
        totals = source['metadata/n_epochs'][:]
        if keys.has_duplicates or np.any(totals <= 0):
            raise ValueError('Invalid subject/class groups or epoch counts')
        for band in BANDS:
            if any(f'{group}/{band}' not in source for group in ('aec', 'bands', 'status')):
                raise ValueError(f'Graph metrics are incomplete: missing {band}')
            status, values = source[f'status/{band}'][:], source[f'bands/{band}'][:]
            if (status.shape != (len(keys), 2, len(METRICS)) or values.shape != status.shape or
                    not np.isin(status, (1, 2)).all()):
                raise ValueError(f'Graph metrics are incomplete or invalid for {band}')
            for index, region in enumerate(REGIONS[dataset]):
                rows = metrics_table.loc[metrics_table.band.eq(band) & metrics_table.region.eq(region)]
                rows = rows.set_index(['subject_id', 'relative_class'])
                if set(rows.index) != set(keys):
                    raise ValueError(f'Graph metrics CSV subject/class keys differ for {band}/{region}')
                rows = rows.reindex(keys)
                if (not np.array_equal(rows.n_epochs.to_numpy(), totals) or
                        not np.array_equal(rows[[f'n_{metric}' for metric in METRICS]].to_numpy(), status[:, index] == 1) or
                        not np.allclose(rows[list(METRICS)].to_numpy(), values[:, index],
                                        rtol=1e-12, atol=1e-14, equal_nan=True)):
                    raise ValueError(f'Graph metrics CSV is stale/inconsistent for {band}/{region}')

def process_dataset(dataset, root=INPUT_ROOT, alpha=0.05):
    """Validate network metrics before paired participant comparisons."""
    root = Path(root)
    aec_path = root / f'{dataset}_AEC.h5'
    with h5py.File(aec_path, 'r') as aec:
        require_success_only(aec)
        n_epochs = len(aec['metadata/success'])
        n_subjects = len(np.unique(aec['metadata/subject_id'].asstr()[:]))

    metrics_dir = root / OUTPUT_SUBDIR
    summary = metrics_dir / f'{dataset}_graph_metrics.csv'
    graph_path = metrics_dir / f'{dataset}_graph_metrics.h5'
    metrics_table = read_grouped_graph_metrics(summary, dataset)
    with h5py.File(graph_path, 'r') as graph:
        require_success_only(graph)
        if graph.attrs.get('source_aec_fingerprint') != fingerprint(aec_path):
            raise ValueError('Source AEC changed; rerun Step 4')
        if graph.attrs.get('modularity_zero_policy') != ZERO_POLICY:
            raise ValueError('Unexpected modularity zero policy')
        graph_aggregation = graph.attrs['graph_aggregation']
    validate_graph_metrics(graph_path, metrics_table, dataset)
    results = analyze_dataset(metrics_table, dataset, alpha)
    path = save_results(results, dataset, summary, metrics_dir / 'statistics', alpha,
                        aggregation=graph_aggregation)
    provenance = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    provenance.update(trial_selection=TRIAL_SELECTION, retained_epochs=int(n_epochs),
                      retained_subjects=int(n_subjects),
                      modularity_zero_policy=ZERO_POLICY,
                      corsi_span='Original participant span from all trials')
    path.with_suffix('.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--alpha', type=float, default=.05)
    args = parser.parse_args()
    for dataset_name in args.datasets:
        process_dataset(dataset_name, args.input_root, args.alpha)
