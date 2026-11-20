"""Step 2 — Participant Corsi Span and relative cognitive-load labels.

The participant's Corsi Span (S) is the longest correctly recalled sequence,
using unique trials from all recordings. A clean epoch at serial position k is
assigned to C1–C4 by ceil(4*k/S), provided 1 <= k <= S.

The span is calculated across the recorded trials. The resulting class labels
and selected clean-epoch indices are stored for the connectivity analysis.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


INPUT_ROOT = Path('processed')
DATASETS = ('Old Adults Dataset', 'Young Adult Dataset')
STRING_DTYPE = h5py.string_dtype('utf-8')


def process_dataset(path):
    """Label the clean epochs in one Step 1 HDF5 file."""
    with h5py.File(path, 'r+') as hf:
        dataset = hf.attrs['dataset']
        if isinstance(dataset, bytes):
            dataset = dataset.decode('utf-8')
        if dataset not in DATASETS:
            raise ValueError(f'Unexpected dataset identifier: {dataset}')
        e = hf['events']
        clean_index = np.asarray(hf['clean_index'][:], dtype=np.int64)
        if len(clean_index) != len(hf['bands/broadband']):
            raise ValueError(f'{path.name}: EEG and clean-event indices differ')
        if np.any(clean_index < 0) or np.any(clean_index >= len(e['subject'])):
            raise ValueError(f'{path.name}: invalid clean-event indices')

        text_fields = {'subject': 'subject_id', 'session': 'session',
                       'recording': 'recording'}
        events = {output: np.asarray(e[source].asstr()[:], dtype=object)
                  for source, output in text_fields.items()}
        for source, output, dtype in (
                ('trial', 'trial_id', np.int64),
                ('L', 'sequence_length', np.int64),
                ('k', 'serial_position', np.int64),
                ('success', 'success', np.uint8),
                ('pre_post', 'pre_post', np.uint8),
                ('onset', 'onset', np.float64),
                ('noisy', 'noisy', np.uint8)):
            events[output] = np.asarray(e[source][:], dtype=dtype)
        n_events = len(events['subject_id'])
        if any(len(values) != n_events for values in events.values()):
            raise ValueError('Inconsistent event metadata lengths')

        # Collapse item rows into trials. Recording is part of the key because
        # the trial counter is restarted in each recording.
        trials = defaultdict(list)
        seen = set()
        for i in range(n_events):
            subject = events['subject_id'][i]
            key = (subject, events['recording'][i], int(events['trial_id'][i]))
            if key not in seen:
                seen.add(key)
                trials[subject].append((int(events['sequence_length'][i]),
                                        int(events['success'][i])))

        span = {}
        span_rows = []
        missing_span = []
        for subject in sorted(trials, key=lambda s: (
                s.rstrip('0123456789'), int(s[len(s.rstrip('0123456789')):] or 0), s)):
            successful = [length for length, ok in trials[subject] if ok == 1]
            if not successful:
                missing_span.append(subject)
                continue
            span[subject] = max(successful)
            span_rows.append(dict(dataset=dataset, subject_id=subject,
                                  corsi_span=span[subject], n_trials=len(trials[subject])))

        label_schema = {
            'dataset': 'str', 'subject_id': 'str', 'session': 'str',
            'recording': 'str', 'trial_id': np.int64,
            'sequence_length': np.int16, 'serial_position': np.int16,
            'corsi_span': np.int16, 'relative_class': np.int8,
            'success': np.uint8, 'pre_post': np.uint8,
            'onset': np.float64, 'source_event_index': np.int64,
            'source_clean_index': np.int64,
        }
        excluded_schema = {k: v for k, v in label_schema.items() if k != 'relative_class'}
        excluded_schema['exclusion_reason'] = 'str'
        labeled = []
        excluded = []
        master = []
        selected = []
        class_counts = Counter()

        for clean_row, event_idx in enumerate(clean_index):
            event_idx = int(event_idx)
            subject = events['subject_id'][event_idx]
            S = span.get(subject)
            k = int(events['serial_position'][event_idx])
            ok = int(events['success'][event_idx])
            class_id = None
            reason = None
            if S is None:
                reason = 'missing_corsi_span'
            elif k < 1:
                reason = f'Serial position k must start at 1, got {k}'
            elif k > S:
                reason = f'Cannot assign a 1..4 class when k={k} > S={S}'
            else:
                class_id = int(np.ceil(4 * k / S))
                if not 1 <= class_id <= 4:
                    raise ValueError(f'Invalid relative class {class_id}')

            row = {
                'dataset': dataset, 'subject_id': subject,
                'session': events['session'][event_idx],
                'recording': events['recording'][event_idx],
                'trial_id': int(events['trial_id'][event_idx]),
                'sequence_length': int(events['sequence_length'][event_idx]),
                'serial_position': k, 'corsi_span': S if S is not None else 0,
                'success': ok, 'pre_post': int(events['pre_post'][event_idx]),
                'onset': float(events['onset'][event_idx]),
                'source_event_index': event_idx,
                'source_clean_index': clean_row,
            }
            if class_id is None:
                excluded.append({**row, 'exclusion_reason': reason})
            else:
                labeled.append({**row, 'relative_class': class_id})
                class_counts[class_id] += 1
                if ok == 1:
                    selected.append(clean_row)
            master.append({
                'epoch_index': clean_row, 'participant_id': subject,
                'recording': events['recording'][event_idx],
                'L': row['sequence_length'], 'k': k,
                'S': '' if S is None else S,
                'class': '' if class_id is None else class_id,
                'success/failure': 'success' if ok else 'failure',
            })

        # HDF5 metadata tables use one dataset per column. The first two retain
        # all clean epochs with explicit inclusion/exclusion status.
        for name, rows, schema, description in (
            ('corsi_span', span_rows,
             {'dataset': 'str', 'subject_id': 'str',
              'corsi_span': np.int16, 'n_trials': np.int32},
             'Maximum successful sequence length per participant.'),
            ('labeled_epochs', labeled, label_schema,
             'Clean epochs with k <= S; relative_class = ceil(4*k/S).'),
            ('excluded_label_epochs', excluded, excluded_schema,
             'Clean epochs not assigned a valid relative class.'),
        ):
            if name in hf:
                del hf[name]
            group = hf.create_group(name)
            group.attrs['n_rows'] = len(rows)
            group.attrs['description'] = description
            for col, dtype in schema.items():
                values = [row[col] for row in rows]
                if dtype == 'str':
                    group.create_dataset(col, data=np.asarray(values, dtype=object),
                                         dtype=STRING_DTYPE)
                else:
                    numbers = np.asarray(values, dtype=dtype)
                    group.create_dataset(col, data=numbers,
                                         compression='gzip' if len(numbers) else None)

        if 'successful_epoch_indices' in hf:
            del hf['successful_epoch_indices']
        hf.create_dataset('successful_epoch_indices',
                          data=np.asarray(selected, dtype=np.int64),
                          compression='gzip' if selected else None)
        hf['successful_epoch_indices'].attrs['definition'] = (
            'Clean, classified epochs from correctly recalled trials; '
            'Corsi Span determined from all original trials.')
        hf.attrs['labeled_epoch_count'] = len(labeled)
        hf.attrs['excluded_label_epoch_count'] = len(excluded)

        # Distribution of all event rows, including epochs rejected as noisy.
        # These counts are not restricted to correctly recalled trials.
        distribution = defaultdict(Counter)
        for i in range(n_events):
            subject = events['subject_id'][i]
            S = span.get(subject)
            k = int(events['serial_position'][i])
            if S is None:
                class_label = 'no_class_missing_S'
            elif 1 <= k <= S:
                class_label = str(int(np.ceil(4 * k / S)))
            elif k > S:
                class_label = 'supra-span epochs'
            else:
                class_label = 'no_class_invalid'

            row = distribution[class_label]
            success = int(events['success'][i])
            noisy = int(events['noisy'][i])
            row['total_epochs'] += 1
            row['total_success' if success else 'total_failure'] += 1
            row['rejected_epochs' if noisy else 'clean_epochs'] += 1
            prefix = 'rejected' if noisy else 'clean'
            row[f'{prefix}_success' if success else f'{prefix}_failure'] += 1

        class_columns = ['total_epochs', 'clean_epochs', 'rejected_epochs',
                         'total_success', 'total_failure', 'clean_success',
                         'clean_failure', 'rejected_success', 'rejected_failure']
        labels = ['1', '2', '3', '4']
        labels += [k for k in sorted(distribution) if k not in labels]
        class_rows = [{'dataset': dataset, 'class': label, **{
            field: distribution[label][field] for field in class_columns}}
            for label in labels]

    return {'dataset': dataset, 'span_rows': span_rows,
            'master_rows': master, 'class_rows': class_rows,
            'n_labeled': len(labeled), 'n_excluded': len(excluded),
            'n_selected': len(selected), 'missing_span': missing_span,
            'class_counts': class_counts}


def main(root):
    """Process the two datasets and write compact human-readable summaries."""
    h5_paths = sorted(root.glob('*_Corsi_multiband.h5'))
    if not h5_paths:
        raise FileNotFoundError(f'No Step 1 datasets found in {root}')

    all_spans, all_classes = [], []
    for path in h5_paths:
        result = process_dataset(path)
        dataset = result['dataset']
        all_spans.extend(result['span_rows'])
        all_classes.extend(result['class_rows'])
        with (root / f'{dataset}_corsi_epoch_master.csv').open(
                'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'epoch_index', 'participant_id', 'recording', 'L', 'k',
                'S', 'class', 'success/failure'])
            writer.writeheader()
            writer.writerows(result['master_rows'])
        print(f'{dataset}: {len(result["span_rows"])} participants with S; '
              f'{result["n_labeled"]} classifiable clean epochs, '
              f'{result["n_selected"]} from successful trials')
        if result['missing_span']:
            print('  Participants without successful trials:',
                  ', '.join(result['missing_span']))

    for filename, rows, columns in (
        ('corsi_span_summary.csv', all_spans,
         ['dataset', 'subject_id', 'corsi_span', 'n_trials']),
        ('corsi_epoch_class_distribution.csv', all_classes,
         ['dataset', 'class', 'total_epochs', 'clean_epochs', 'rejected_epochs',
          'total_success', 'total_failure', 'clean_success', 'clean_failure',
          'rejected_success', 'rejected_failure']),
    ):
        with (root / filename).open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    args = parser.parse_args()
    main(args.input_root)
