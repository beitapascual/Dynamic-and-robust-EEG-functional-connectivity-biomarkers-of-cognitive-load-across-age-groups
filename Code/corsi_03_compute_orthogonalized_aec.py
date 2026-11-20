"""Step 3 — Estimate orthogonalized AEC for subsequent network averaging.

For each artifact-clean, one-second EEG epoch and frequency band, calculate
orthogonalized amplitude-envelope correlation using MEDUSA aec(ort=True).
An interrupted computation can be resumed via per-epoch checkpoints.

The final successful-trial dataset selects already computed epoch matrices
using the indices saved by Step 2. Step 4 averages these AEC matrices within
participant and relative-load class BEFORE estimating graph metrics.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from medusa.connectivity.amplitude_connectivity import aec
from corsi_01_preprocess_eeg import CHANNEL_NAMES


ROOT = Path('processed')
BANDS = ('broadband', 'theta', 'alpha', 'low_beta', 'high_beta', 'gamma')
DATASETS = ('Old Adults Dataset', 'Young Adult Dataset')


def build_epoch_metadata(eeg):
    """Align event metadata and Step 2 labels to the clean EEG epoch rows."""
    clean = np.asarray(eeg['clean_index'][:], dtype=np.int64)
    e = eeg['events']
    metadata = {
        'epoch_index': np.arange(len(clean), dtype=np.int64),
        'source_clean_index': np.arange(len(clean), dtype=np.int64),
        'source_event_index': clean,
        'subject_id': e['subject'].asstr()[:][clean],
        'session': e['session'].asstr()[:][clean],
        'recording': e['recording'].asstr()[:][clean],
        'trial_id': np.asarray(e['trial'][clean], dtype=np.int64),
        'sequence_length': np.asarray(e['L'][clean], dtype=np.int16),
        'serial_position': np.asarray(e['k'][clean], dtype=np.int16),
        'success': np.asarray(e['success'][clean], dtype=np.uint8),
        'pre_post': np.asarray(e['pre_post'][clean], dtype=np.uint8),
        'onset': np.asarray(e['onset'][clean], dtype=np.float64),
    }
    if 'labeled_epochs' not in eeg or 'excluded_label_epochs' not in eeg:
        raise ValueError('Step 2 class labels are missing')
    S = np.zeros(len(clean), dtype=np.int16)
    classes = np.zeros(len(clean), dtype=np.int8)
    labels = np.asarray(['missing_label'] * len(clean), dtype=object)
    labeled = eeg['labeled_epochs']
    for row, pos in enumerate(labeled['source_clean_index'][:]):
        S[pos] = labeled['corsi_span'][row]
        classes[pos] = labeled['relative_class'][row]
        labels[pos] = str(int(classes[pos]))
    excluded = eeg['excluded_label_epochs']
    for row, pos in enumerate(excluded['source_clean_index'][:]):
        S[pos] = excluded['corsi_span'][row]
        reason = excluded['exclusion_reason'].asstr()[row]
        labels[pos] = 'supra-span epochs' if 'k=' in reason and 'S=' in reason else reason
    metadata['corsi_span'] = S
    metadata['relative_class'] = classes
    metadata['class_label'] = labels
    if np.any((classes < 0) | (classes > 4)):
        raise ValueError('Relative classes outside expected 0–4 range')
    return metadata


def process_dataset(dataset, input_dir=ROOT, output_dir=ROOT,
                    bands=BANDS, block_size=8, flush_every=16,
                    retry_failed=False, max_epochs=None):
    """Compute all-trial AEC matrices and persist successful epoch checkpoints."""
    source_path = input_dir / f'{dataset}_Corsi_multiband.h5'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{dataset}_AEC.h5'
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if block_size < 1 or flush_every < 1:
        raise ValueError('block_size and flush_every must be positive')

    # A truncated HDF5 file cannot be resumed safely; keep it for inspection.
    try:
        out = h5py.File(output_path, 'a')
    except OSError as exc:
        if 'truncated file' not in str(exc).lower():
            raise
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        broken = output_path.with_name(f'{output_path.stem}.corrupt_{stamp}.h5')
        output_path.replace(broken)
        print(f'Preserved truncated output as {broken.name}')
        out = h5py.File(output_path, 'w')

    error_path = output_dir / 'aec_errors.csv'
    new_log = not error_path.is_file() or not error_path.stat().st_size
    with h5py.File(source_path, 'r') as eeg, out, error_path.open(
            'a', newline='', encoding='utf-8') as log:
        writer = csv.writer(log)
        if new_log:
            writer.writerow(['timestamp_utc', 'dataset', 'subject_id', 'band',
                             'epoch_index', 'source_event_index', 'error'])
        src = eeg[f'bands/{bands[0]}']
        n_epochs, _, n_channels = src.shape
        channels = tuple(eeg['channel_names'].asstr()[:])
        if channels != CHANNEL_NAMES[dataset] or n_channels != 16:
            raise ValueError('EEG channel order differs from the verified acquisition montage')

        if 'metadata' not in out:
            metadata = build_epoch_metadata(eeg)
            if len(metadata['epoch_index']) != n_epochs:
                raise ValueError('EEG epoch and label rows are misaligned')
            group = out.create_group('metadata')
            for name, values in metadata.items():
                arr = np.asarray(values)
                if arr.dtype.kind in 'UO':
                    group.create_dataset(name, data=np.asarray(values,dtype=object),
                                         dtype=h5py.string_dtype('utf-8'),
                                         compression='gzip' if len(arr) else None)
                else:
                    group.create_dataset(name, data=arr,
                                         compression='gzip' if len(arr) else None)
            group = out.create_group('channels')
            group.create_dataset('names', data=channels, dtype=h5py.string_dtype('utf-8'))
            group.attrs['source'] = 'confirmed_acquisition_order'
            out.attrs['dataset'] = dataset
            out.attrs['source_hdf5'] = str(source_path.resolve())
            out.attrs['n_epochs'] = n_epochs
            out.attrs['n_channels'] = n_channels
            out.attrs['aec_method'] = 'medusa.connectivity.amplitude_connectivity.aec'
            out.attrs['aec_orthogonalized'] = True
            out.attrs['matrix_dtype'] = 'float32'
            out.attrs['created_or_updated_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        else:
            if (out.attrs['dataset'] != dataset or out.attrs['n_epochs'] != n_epochs or
                    out.attrs['n_channels'] != n_channels or
                    tuple(out['channels/names'].asstr()[:]) != channels):
                raise ValueError('Existing AEC output is incompatible with Step 1 data')
            current = build_epoch_metadata(eeg)
            for key, values in current.items():
                field = out[f'metadata/{key}']
                previous = field.asstr()[:] if h5py.check_string_dtype(field.dtype) else field[:]
                if not np.array_equal(previous, values):
                    raise ValueError(f'Stale AEC metadata ({key}); rebuild the full AEC')

        out.attrs['block_size'] = int(block_size)
        out.attrs['flush_every'] = int(flush_every)
        out.flush()
        subject_ids = out['metadata/subject_id'].asstr()[:]
        source_events = out['metadata/source_event_index'][:]
        scan_until = n_epochs if max_epochs is None else min(n_epochs, int(max_epochs))

        for band in bands:
            eeg_band = eeg[f'bands/{band}']
            if eeg_band.shape[0] != n_epochs or eeg_band.shape[2] != n_channels:
                raise ValueError(f'{band}: unexpected source epoch layout')
            matrix_path = f'bands/{band}'
            done_path = f'checkpoint/{band}_processed'
            failed_path = f'failures/{band}_failed'
            if matrix_path not in out:
                out.create_dataset(matrix_path, shape=(n_epochs, n_channels, n_channels),
                                   dtype=np.float32, chunks=(1, n_channels, n_channels),
                                   compression='gzip', shuffle=True)
                out.create_dataset(done_path, data=np.zeros(n_epochs, dtype=np.uint8),
                                   compression='gzip')
                out.create_dataset(failed_path, data=np.zeros(n_epochs, dtype=np.uint8),
                                   compression='gzip')
            matrices = out[matrix_path]
            processed = out[done_path]
            failures = out[failed_path]
            if (matrices.shape != (n_epochs, n_channels, n_channels) or
                    processed.shape != (n_epochs,) or failures.shape != (n_epochs,)):
                raise ValueError(f'{band}: incompatible checkpoint/output dimensions')
            pending_since_flush = 0
            for start in range(0, scan_until, block_size):
                stop = min(start + block_size, scan_until)
                mask = (processed[start:stop] == 0) & (
                    retry_failed | (failures[start:stop] == 0))
                pending = start + np.flatnonzero(mask)
                if not len(pending):
                    continue
                block = eeg_band[start:stop][mask]
                print(f'{dataset} | {band} | epochs {pending[0]}–{pending[-1]} '
                      f'| pending={len(pending)}')
                try:
                    result = np.asarray(aec(block, ort=True))
                    if result.shape != (len(pending), n_channels, n_channels):
                        raise ValueError(f'Unexpected AEC shape: {result.shape}')
                    if not np.isfinite(result).all():
                        raise ValueError('Non-finite AEC values')
                except Exception as exc:
                    print(f'  Block failed ({exc}); retrying individual epochs')
                    # A failed batch need not imply every epoch is invalid.
                    result = [None] * len(pending)
                    for j, row in enumerate(block):
                        try:
                            single = np.asarray(aec(row[None, :, :], ort=True))
                            if single.shape != (1, n_channels, n_channels):
                                raise ValueError(f'Unexpected AEC shape: {single.shape}')
                            if not np.isfinite(single).all():
                                raise ValueError('Non-finite AEC values')
                            result[j] = single[0]
                        except Exception as error:
                            idx = int(pending[j])
                            failures[idx] = 1
                            processed[idx] = 0
                            writer.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                             dataset, subject_ids[idx], band, idx,
                                             source_events[idx], str(error)])
                            log.flush()
                for idx, matrix in zip(pending, result):
                    if matrix is None:
                        continue
                    idx = int(idx)
                    try:
                        matrices[idx] = np.asarray(matrix, dtype=np.float32)
                        failures[idx] = 0
                        processed[idx] = 1
                        pending_since_flush += 1
                    except Exception as error:
                        failures[idx] = 1
                        processed[idx] = 0
                        writer.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                         dataset, subject_ids[idx], band, idx,
                                         source_events[idx], str(error)])
                        log.flush()
                    if pending_since_flush >= flush_every:
                        out.flush()
                        pending_since_flush = 0
                out.flush()
            print(f'{dataset} | {band} | processed={int(processed[:scan_until].sum())}'
                  f'/{scan_until} | failures={int(failures[:scan_until].sum())}')
        out.flush()
    return output_path


def export_successful_trials(dataset, source_eeg, full_aec, output_dir, block_size=256):
    """Copy completed AEC matrices for classifiable successful-trial epochs.

    The selection is defined in Step 2 from the original participant-level
    span. No signal filtering or connectivity estimation is repeated here.
    """
    import pandas as pd

    slug = dataset
    selected_dir = output_dir / 'success_only'
    selected_dir.mkdir(parents=True, exist_ok=True)
    destination = selected_dir / f'{slug}_AEC.h5'
    selection_path = selected_dir / f'{slug}_epoch_selection.h5'
    tmp = destination.with_suffix('.tmp.h5')

    with h5py.File(source_eeg, 'r') as eeg, h5py.File(full_aec, 'r') as src:
        if 'successful_epoch_indices' not in eeg:
            raise ValueError('Step 2 successful-epoch indices not found; rerun Step 2')
        indices = np.asarray(eeg['successful_epoch_indices'][:], dtype=np.int64)
        clean = np.asarray(eeg['clean_index'][:], dtype=np.int64)
        if not len(indices):
            raise ValueError(f'{dataset}: no classifiable successful-trial epochs')
        if (np.any(indices < 0) or np.any(indices >= len(clean)) or
                np.any(np.diff(indices) <= 0)):
            raise ValueError('Success selection indices must be unique and increasing')
        meta = src['metadata']
        if not np.array_equal(meta['source_event_index'][:], clean):
            raise ValueError('Step 1 and AEC source rows are not aligned')
        if (not np.all(meta['success'][indices] == 1) or
                not np.all(np.isin(meta['relative_class'][indices], [1, 2, 3, 4]))):
            raise ValueError('Step 2 selection does not match AEC trial metadata')
        selected_label_indices = np.asarray(eeg['labeled_epochs/source_clean_index'][:], dtype=np.int64)
        successful_rows = eeg['labeled_epochs/success'][:] == 1
        if not np.array_equal(indices, selected_label_indices[successful_rows]):
            raise ValueError('Step 2 labels have changed since selection')
        for band in BANDS:
            if (f'bands/{band}' not in src or
                    f'checkpoint/{band}_processed' not in src or
                    np.any(src[f'checkpoint/{band}_processed'][indices] != 1)):
                raise ValueError(f'Incomplete AEC for {dataset}/{band}; finish all bands')

        # Record the source file signature for subsequent consistency checks.
        source_stat = full_aec.stat()
        source_signature = (f'{full_aec.resolve()}|{source_stat.st_size}|'
                            f'{source_stat.st_mtime_ns}')
        trial_selection = 'success == 1; relative_class in 1..4'
        with h5py.File(tmp, 'w') as out:
            for key, val in src.attrs.items():
                out.attrs[key] = val
            out.attrs['dataset'] = dataset
            out.attrs['trial_selection'] = trial_selection
            out.attrs['n_epochs'] = len(indices)
            out.attrs['n_source_epochs'] = len(clean)
            out.attrs['source_aec_hdf5'] = str(full_aec.resolve())
            out.attrs['source_hdf5'] = str(source_eeg.resolve())
            out.attrs['source_aec_fingerprint'] = source_signature
            out.attrs['aec_reused_without_recalculation'] = True
            out.attrs['corsi_span_method'] = 'Maximum successful L across all original trials'
            out.attrs['relative_class_method'] = 'ceil(4*k/S); original span retained'

            src.copy('channels', out)
            eeg.copy('corsi_span', out)
            out_meta = out.create_group('metadata')
            for field, ds in meta.items():
                values = ds.asstr()[:] if h5py.check_string_dtype(ds.dtype) else ds[:]
                if field == 'epoch_index':
                    values = np.arange(len(indices), dtype=np.int64)
                else:
                    values = values[indices]
                dtype = h5py.string_dtype('utf-8') if h5py.check_string_dtype(ds.dtype) else ds.dtype
                out_meta.create_dataset(field, data=values, dtype=dtype,
                                        compression='gzip' if len(values) else None)
            out_meta.create_dataset('source_aec_epoch_index', data=indices, compression='gzip')
            for band in BANDS:
                src_matrix = src[f'bands/{band}']
                dst = out.create_dataset(f'bands/{band}',
                                         shape=(len(indices), *src_matrix.shape[1:]),
                                         dtype=src_matrix.dtype,
                                         chunks=(1, *src_matrix.shape[1:]),
                                         compression='gzip', shuffle=True)
                for start in range(0, len(indices), block_size):
                    rows = indices[start:start + block_size]
                    dst[start:start + len(rows)] = src_matrix[rows]
                for group, name in (('checkpoint', f'{band}_processed'),
                                    ('failures', f'{band}_failed')):
                    source_data = src[f'{group}/{name}'][indices]
                    out.create_dataset(f'{group}/{name}', data=source_data,
                                       compression='gzip')
        tmp.replace(destination)

    # Retain selected epoch indices and original labels for auditability.
    with h5py.File(destination, 'r') as selected, h5py.File(selection_path, 'w') as table_h5:
        for key, val in selected.attrs.items():
            table_h5.attrs[key] = val
        for name in ('metadata', 'channels', 'corsi_span'):
            selected.copy(name, table_h5)
    with h5py.File(destination, 'r') as selected:
        meta = selected['metadata']
        table = pd.DataFrame({
            key: (ds.asstr()[:] if h5py.check_string_dtype(ds.dtype) else ds[:])
            for key, ds in meta.items()})
    table.to_csv(selected_dir / f'{slug}_corsi_epoch_master.csv', index=False)
    table.groupby(['subject_id', 'relative_class']).size().rename('n_epochs').to_csv(
        selected_dir / f'{slug}_epoch_counts.csv')
    report = {
        'dataset': dataset,
        'trial_selection': trial_selection,
        'source_epochs': int(len(clean)),
        'retained_epochs': int(len(indices)),
        'excluded_epochs': int(len(clean) - len(indices)),
        'subjects': int(table['subject_id'].nunique()),
        'retained_trials': int(table[['subject_id', 'recording', 'trial_id']].drop_duplicates().shape[0]),
        'class_counts': {str(k): int(v) for k, v in table['relative_class'].value_counts().sort_index().items()},
        'corsi_span_unchanged': True,
        'aec_reused_without_recalculation': True,
    }
    (selected_dir / f'{slug}_selection_report.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(f'{dataset} | successful/classifiable AEC: {len(indices)}/{len(clean)} epochs')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--bands', nargs='+', choices=BANDS, default=list(BANDS))
    parser.add_argument('--block-size', type=int, default=8)
    parser.add_argument('--flush-every', type=int, default=16)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--no-success-only', action='store_true',
                        help='Do not export the exact AEC subset from successful trials.')
    parser.add_argument('--max-epochs-per-band', type=int, default=None,
                        help='Smoke test only; omit for all epochs.')
    args = parser.parse_args()
    for dataset in args.datasets:
        path = process_dataset(dataset, args.input_root, args.output_root,
                               args.bands, args.block_size, args.flush_every,
                               args.retry_failed, args.max_epochs_per_band)
        if not args.no_success_only:
            eeg = args.input_root / f'{dataset}_Corsi_multiband.h5'
            export_successful_trials(dataset, eeg, path, args.output_root)


if __name__ == '__main__':
    main()
