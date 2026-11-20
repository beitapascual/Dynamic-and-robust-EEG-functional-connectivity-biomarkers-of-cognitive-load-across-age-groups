"""Step 4 — Compute graph metrics from participant-averaged AEC networks.

Only one aggregation order is used: for each participant, relative-load class
and frequency band, average the successful-trial AEC matrices first. Apply the
original frontal/parietal seed-network masks to the averaged connectivity,
then calculate the five MEDUSA graph metrics. Graph metrics from individual
epochs are never calculated or averaged.

The output stores the averaged 16 x 16 AEC matrices and the resulting regional
metrics, with the number of contributing epochs for each participant/class.
"""

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from medusa.graph_theory.clustering_coefficient import clustering_coefficient
from medusa.graph_theory.complexity import complexity
from medusa.graph_theory.degree import degree
from medusa.graph_theory.modularity import modularity
from medusa.graph_theory.path_length import path_length

from corsi_01_preprocess_eeg import CHANNEL_NAMES
from corsi_03_compute_orthogonalized_aec import BANDS

DATASETS = tuple(CHANNEL_NAMES)
INPUT_ROOT = Path('processed') / 'success_only'
OUTPUT_SUBDIR = 'graph_metrics'
TRIAL_SELECTION = 'success == 1; relative_class in 1..4'
MODULARITY_ZERO_ATOL = 1e-12
ZERO_POLICY = 'abs(modularity) < 1e-12 -> 0 before statistical analysis'
METRICS = ('strength', 'path_length', 'clustering_coefficient',
           'shannon_graph_complexity', 'modularity')
REGIONS = {
    'Old Adults Dataset': {
        'frontal': ('F7', 'F3', 'Fz', 'F4', 'F8', 'Fp1', 'Fp2'),
        'parietal': ('P3', 'Pz', 'P4', 'POz', 'C3', 'Cz', 'C4'),
    },
    'Young Adult Dataset': {
        'frontal': ('F7', 'F3', 'Fz', 'F4', 'FCz'),
        'parietal': ('C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4'),
    },
}


def fingerprint(path):
    """Identify a source file so downstream steps detect a changed AEC file."""
    p = Path(path).resolve()
    s = p.stat()
    return f'{p}|{s.st_size}|{s.st_mtime_ns}'


def require_success_only(src):
    """Reject accidental use of all-trial connectivity matrices."""
    if src.attrs.get('trial_selection') != TRIAL_SELECTION:
        raise ValueError('Expected the successful-trial AEC exported by Step 3')
    if 'metadata/success' in src and np.any(src['metadata/success'][:] != 1):
        raise ValueError('Unexpected unsuccessful trials in selected data')
    if not np.isin(src['metadata/relative_class'][:], [1, 2, 3, 4]).all():
        raise ValueError('AEC contains unclassified epochs')


def normalize_modularity(out, block_size=256):
    """Canonicalize numerical zero, retaining MEDUSA's modularity definition."""
    changed = 0
    for band in out['bands']:
        values_ds = out[f'bands/{band}']
        for start in range(0, len(values_ds), block_size):
            stop = min(start + block_size, len(values_ds))
            values = values_ds[start:stop]
            q = values[..., 4]
            mask = np.isfinite(q) & (np.abs(q) < MODULARITY_ZERO_ATOL)
            changed += int(np.count_nonzero(mask & (q != 0)))
            if mask.any():
                q[mask] = 0.
                values_ds[start:stop] = values
    out.attrs['modularity_zero_policy'] = ZERO_POLICY
    out.attrs['modularity_numerical_zeros_normalized'] = changed
    return changed


class _ComplexityWeights(np.ndarray):
    """Correct only MEDUSA 1.4's empty-graph check, without changing weights.

    Its entropy/divergence helpers use ``not W.all() == 0`` to test for edges.
    One zero (including the diagonal) therefore makes every graph 'empty'.
    On this private view only, ``all`` means ``any``. All other array operations
    and MEDUSA's entropy, divergence and normalization remain untouched. This
    is also harmless when MEDUSA uses the corrected ``W.any()`` check.
    """

    def all(self, *args, **kwargs):
        return np.asarray(self).any(*args, **kwargs)


def _seed_masks(names, dataset, n_channels):
    """Build masks for one actual montage; require every prescribed channel."""
    if len(names) != n_channels or len(set(names)) != n_channels:
        raise ValueError('Channel names do not match the connectivity matrix')
    masks = []
    for region, channels in REGIONS[dataset].items():
        missing = [channel for channel in channels if channel not in names]
        if missing:
            raise ValueError(f'{dataset} {region}: required channels missing: {missing}')
        indices = [names.index(channel) for channel in channels]
        mask = np.zeros((n_channels, n_channels), dtype=bool)
        mask[indices, :] = True
        mask[:, indices] = True
        np.fill_diagonal(mask, False)
        masks.append(mask)
    return np.asarray(masks)


def region_masks(src, dataset, n_channels):
    """Validate the single confirmed channel order and build both masks once."""
    names = tuple(src['channels/names'].asstr()[:])
    if names != CHANNEL_NAMES[dataset] or n_channels != 16:
        raise ValueError('AEC channel order differs from the verified acquisition montage')
    return names, 'confirmed_acquisition_order', _seed_masks(names, dataset, n_channels)


def _modularity(weights, gamma, seed):
    """MEDUSA uses NumPy's global RNG; seed each graph and restore caller state."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        if not weights.any():
            raise ValueError('Modularity is undefined for an edgeless graph')
        return modularity(weights, gamma=gamma)[0]
    finally:
        np.random.set_state(state)


def compute_metrics(weights, pending, gamma, seed):
    """Call MEDUSA once per requested metric on the shared regional matrix."""
    functions = (
        degree,
        # MEDUSA already maps weights to lengths internally. Zeros mean no edge;
        # its defaults exclude the diagonal and INCLUDE infinite distances.
        lambda w: path_length(w)[0],
        clustering_coefficient,
        lambda w: complexity(w.view(_ComplexityWeights)),
        lambda w: _modularity(w, gamma, seed),
    )
    values = np.full(len(METRICS), np.nan)
    errors = {}
    for index in np.flatnonzero(pending):
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                values[index] = np.mean(functions[index](weights))
            if np.isnan(values[index]) or np.isneginf(values[index]):
                raise ValueError('MEDUSA returned an undefined result')
            if np.isposinf(values[index]) and METRICS[index] != 'path_length':
                raise ValueError('MEDUSA returned an infinite result')
        except Exception as exc:
            values[index] = np.nan
            errors[index] = str(exc)
    return values, errors


def average_connectivity(src, band, groups, counts, block_size):
    """Read each epoch once; accumulate in float64, sharing means across regions."""
    matrices = src[f'bands/{band}']
    ready = src[f'checkpoint/{band}_processed'][:]
    selected = groups >= 0
    if matrices.shape != (len(groups), 16, 16) or ready.shape != groups.shape:
        raise ValueError(f'Incompatible AEC/checkpoint shape for {band}')
    if np.any(ready[selected] != 1):
        raise ValueError(f'Step 3 is incomplete for {band}; finish it before averaging AEC')
    sums = np.zeros((len(counts), 16, 16), dtype=np.float64)
    for start in range(0, len(groups), block_size):
        stop = start + block_size
        eligible = selected[start:stop]
        if not eligible.any():
            continue
        block = np.asarray(matrices[start:stop][eligible], dtype=np.float64)
        if (not np.isfinite(block).all() or np.any(block < 0) or
                not np.allclose(block, block.transpose(0, 2, 1), rtol=0, atol=1e-7)):
            raise ValueError(f'{band}: expected finite, nonnegative, symmetric AEC weights')
        np.add.at(sums, groups[start:stop][eligible], block)
    return sums / counts[:, None, None]


def compute_mean_aec_metrics(dataset, input_dir=INPUT_ROOT, output_dir=None,
                    bands=BANDS, block_size=256, gamma=1.0, seed=0):
    """Group AEC matrices first and compute metrics on each group mean."""
    input_dir = Path(input_dir)
    output_dir = input_dir / OUTPUT_SUBDIR if output_dir is None else Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError('Use a separate graph-metrics output directory')
    if block_size < 1 or not np.isfinite(gamma) or gamma < 0 or seed < 0:
        raise ValueError('Invalid block size, gamma or seed')
    if not bands or len(set(bands)) != len(bands) or set(bands) - set(BANDS):
        raise ValueError('Select distinct bands from BANDS')
    source = input_dir / f'{dataset}_AEC.h5'
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f'{dataset}_graph_metrics.h5'
    temporary = destination.with_suffix('.tmp.h5')
    rows = []
    pending = np.ones(len(METRICS), dtype=bool)
    with h5py.File(source, 'r') as src:
        names, _, masks = region_masks(src, dataset, 16)
        subjects = src['metadata/subject_id'].asstr()[:]
        classes = src['metadata/relative_class'][:]
        selected = np.isin(classes, (1, 2, 3, 4))
        keys = sorted(set(zip(subjects[selected], classes[selected])))
        if not keys:
            raise ValueError('No classified epochs to average')
        lookup = {key: index for index, key in enumerate(keys)}
        groups = np.array([lookup.get(key, -1) for key in zip(subjects, classes)])
        counts = np.bincount(groups[selected], minlength=len(keys))
        with h5py.File(temporary, 'w') as out, (output_dir / 'graph_metrics_errors.csv').open(
                'w', newline='', encoding='utf-8') as log:
            out.attrs.update(
                schema_version=1, dataset=dataset, graph_aggregation='metrics_of_mean_aec',
                source_hdf5=str(source.resolve()), medusa_version=version('medusa-kernel'),
                gamma=gamma, seed=seed, class_selection='relative_class in 1..4',
                grouping='subject_id x relative_class x band; epochs equally weighted',
                region_selection='seed; diagonal=0; outside_edges=0; all_nodes_retained',
                nodal_reduction='mean over all nodes in the seed graph',
                complexity_adapter='MEDUSA empty-graph guard: all -> any',
                path_length_infinite_distances='included (MEDUSA default)',
                created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
            strings = h5py.string_dtype('utf-8')
            for name, values in {
                    'channels/names': names, 'metric_names': METRICS,
                    'region_names': tuple(REGIONS[dataset]),
                    'metadata/subject_id': [key[0] for key in keys],
                    **{f'regions/{region}': channels for region, channels in REGIONS[dataset].items()}
            }.items():
                out.create_dataset(name, data=values, dtype=strings)
            out.create_dataset('metadata/relative_class', data=[key[1] for key in keys])
            out.create_dataset('metadata/n_epochs', data=counts)
            errors_csv = csv.writer(log)
            errors_csv.writerow(['dataset', 'band', 'subject_id', 'relative_class', 'region', 'metric', 'error'])
            for band in bands:
                averages = average_connectivity(src, band, groups, counts, block_size)
                out.create_dataset(f'aec/{band}', data=averages, compression='gzip')
                values = np.full((len(keys), 2, len(METRICS)), np.nan)
                status = np.ones(values.shape, dtype='u1')
                for index, ((subject, label), matrix) in enumerate(zip(keys, averages)):
                    for region_index, (region, mask) in enumerate(zip(REGIONS[dataset], masks)):
                        graph_seed = np.random.SeedSequence(
                            [seed, BANDS.index(band), index, region_index]).generate_state(1)[0]
                        computed, errors = compute_metrics(
                            np.where(mask, matrix, 0), pending, gamma, graph_seed)
                        values[index, region_index] = computed
                        for metric_index, message in errors.items():
                            status[index, region_index, metric_index] = 2
                            errors_csv.writerow([dataset, band, subject, label, region,
                                                 METRICS[metric_index], message])
                        rows.append([dataset, subject, label, band, region, counts[index],
                                     *computed, *(status[index, region_index] == 1).astype(int)])
                out.create_dataset(f'bands/{band}', data=values, compression='gzip')
                out.create_dataset(f'status/{band}', data=status, compression='gzip')
                print(f'{dataset} | {band} | mean AEC matrices={len(keys)} | failed metrics={int((status == 2).sum())}',
                      flush=True)
    temporary.replace(destination)
    summary = output_dir / f'{dataset}_graph_metrics_from_mean_aec.csv'
    staged_csv = summary.with_suffix('.tmp.csv')
    with staged_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'subject_id', 'relative_class', 'band', 'region', 'n_epochs',
                         *METRICS, *[f'n_{metric}' for metric in METRICS]])
        writer.writerows(rows)
    staged_csv.replace(summary)
    return destination

def process_dataset(dataset, root=INPUT_ROOT, block_size=256):
    """Use successful-trial AEC only; save one grouped metrics dataset."""
    root = Path(root)
    aec_path = root / f'{dataset}_AEC.h5'
    with h5py.File(aec_path, 'r') as aec:
        require_success_only(aec)
    output_dir = root / OUTPUT_SUBDIR
    path = compute_mean_aec_metrics(dataset, root, output_dir, block_size=block_size)
    with h5py.File(path, 'r+') as out:
        changed = normalize_modularity(out, block_size)
        out.attrs['trial_selection'] = TRIAL_SELECTION
        out.attrs['source_aec_fingerprint'] = fingerprint(aec_path)

    # Reconcile tiny numerical modularity zeros in the subject-level CSV too.
    summary_path = output_dir / f'{dataset}_graph_metrics_from_mean_aec.csv'
    summary = pd.read_csv(summary_path, dtype={'subject_id': str})
    summary.loc[summary['modularity'].abs() < MODULARITY_ZERO_ATOL, 'modularity'] = 0.
    summary.to_csv(summary_path, index=False)
    print(f'{dataset}: averaged AEC -> graph metrics; normalized {changed} numerical zero(s)')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--block-size', type=int, default=256)
    args = parser.parse_args()
    for dataset_name in args.datasets:
        process_dataset(dataset_name, args.input_root, args.block_size)
