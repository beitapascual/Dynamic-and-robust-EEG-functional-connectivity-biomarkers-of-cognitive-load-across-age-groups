"""Step 1 — Extract and preprocess Corsi-task EEG epochs.

The two cohorts have distinct acquisition montages and trial metadata formats.
For every stimulus, a 1-s window is extracted from the filtered, CAR-referenced
signal. Broadband artifact rejection supplies a shared mask across bands;
accepted epochs are z-scored independently by channel.

The electrode names below follow the verified physical acquisition order, not
potentially unreliable raw-file labels. Outputs retain event-to-epoch indices
for subsequent Corsi Span estimation and connectivity analysis.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import h5py
import numpy as np
from medusa import FIRFilter, car
from medusa.artifact_removal import reject_noisy_epochs
from medusa.components import Recording
from medusa.epoching import get_epochs_of_events, normalize_epochs
import medusa.bci.nft_paradigms



# Verified acquisition order (16 channels per cohort).
output_root = Path('processed')
CHANNEL_NAMES = {
    'Old Adults Dataset': (
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7',
        'C3', 'Cz', 'C4', 'T8', 'P3', 'Pz', 'P4', 'POz',
    ),
    'Young Adult Dataset': (
        'F7', 'F3', 'Fz', 'F4', 'F8', 'FCz', 'C3', 'Cz',
        'C4', 'CPz', 'P3', 'Pz', 'P4', 'PO7', 'POz', 'PO8',
    ),
}

# Frequency bands to export. The first band is also used as the reference band
# for artifact rejection, so all bands keep exactly the same clean event rows.
BANDS = {
    'broadband': (4, 60),
    'theta': (4, 8),
    'alpha': (8, 13),
    'low_beta': (13, 20),
    'high_beta': (20, 30),
    'gamma': (30, 60),
}


def _plain_tokens(value):
    """Return lowercase ASCII tokens so labels can be matched accent-free."""

    decomposed = unicodedata.normalize('NFKD', str(value))
    ascii_text = ''.join(char for char in decomposed if not unicodedata.combining(char))
    return set(re.findall(r'[a-z0-9]+', ascii_text.casefold()))


def _recording_context(dataset, root, path):
    """Return relative path, subject, and session labels for console logs."""

    rel = path.relative_to(root)
    if dataset == 'Old Adults Dataset':
        subject_match = re.search(r'Subject_(\d+)', str(rel), re.I)
        session_match = re.search(r'Session_(\d+)', str(rel), re.I)
        subject = subject_match.group(1) if subject_match else rel.parts[0]
        session = session_match.group(1) if session_match else 'unknown'
    else:
        subject = rel.parts[0]
        session = rel.parts[1] if len(rel.parts) > 1 else 'unknown'
    return rel, subject, session


def build_dataset(dataset, root):
    """Build one dataset; keep the original event-to-clean-epoch mapping."""

    if not root.is_dir():
        raise FileNotFoundError(f'Missing source folder for {dataset}: {root}')

    output_root.mkdir(parents=True, exist_ok=True)
    if dataset == 'Old Adults Dataset':
        paths = []
        for subject_dir in sorted(root.glob('Subject_*')):
            if not subject_dir.is_dir():
                continue
            for session in range(3, 12):
                session_dir = subject_dir / f'Session_{session}'
                for recording in (1, 15):
                    file_number = 27 if session == 3 and recording == 15 else recording
                    path = session_dir / f'R{file_number}.mtest.bson'
                    if path.is_file():
                        paths.append(path)
    else:
        paths = sorted(root.rglob('*.bson'))
        paths = [p for p in paths if 'Corsi' in p.parts]

    destination = output_root / f'{dataset}_Corsi_multiband.h5'
    total_events = total_clean = errors = processed = 0

    with h5py.File(destination, 'w') as hf:
        hf.attrs['dataset'] = dataset

        for path in paths:
            rel, subject, session = _recording_context(dataset, root, path)
            log_context = f'dataset={dataset} | subject={subject} | session={session} | file={path.name}'
            try:
                rec = Recording.load(str(path))
                raw = np.asarray(rec.eeg.signal, dtype=float)
                times = np.asarray(rec.eeg.times, dtype=float)
                fs = float(rec.eeg.fs)
                # Raw channel labels can be wrong; the confirmed physical
                # acquisition order is fixed within each cohort.
                channel_names = tuple(CHANNEL_NAMES[dataset])

                if (raw.ndim != 2 or not np.isfinite(raw).all() or
                        times.ndim != 1 or len(times) != len(raw) or
                        not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or
                        len(channel_names) != raw.shape[1] or
                        len(set(channel_names)) != len(channel_names)):
                    raise ValueError('Invalid EEG signal or timestamps')
                if ('channel_names' in hf and
                        tuple(hf['channel_names'].asstr()[:]) != channel_names):
                    raise ValueError('Recording channel order differs from the HDF5 layout')

                # Old Adults Dataset stores trial data in dictionaries, while Young Adult Dataset stores
                # the same fields as flat attributes.
                if dataset == 'Old Adults Dataset':
                    pre_post = 0 if path.name.lower().startswith('r1.') else 1
                    exp = rec.customexperimentdata
                    levels = exp.trial_started['level']
                    starts = exp.trial_started['onset']
                    successes = exp.trial_finished['performance']
                else:
                    subject = rel.parts[0]
                    session = rel.parts[1] if len(rel.parts) > 1 else 'unknown'
                    pre_post = 0 if 'pre' in _plain_tokens(session) else 1
                    exp = rec.customexperimentdata
                    levels, starts, successes = exp.trial_level, exp.trial_onset, exp.trial_success

                if not (len(levels) == len(starts) == len(successes)):
                    raise ValueError('Inconsistent trial counts')

                meta = {'success': [], 'L': [], 'k': [], 'trial': [], 'onset': []}
                for i, (L, start, success) in enumerate(zip(levels, starts, successes)):
                    L = int(L)
                    if L < 1:
                        raise ValueError(f'Invalid sequence length in trial {i}')
                    for k in range(1, L + 1):
                        onset = float(start) + k - 1

                        # Keep only events whose full one-second epoch is inside
                        # the recorded EEG time span.
                        if onset < times[0] or onset + 1 > times[-1]:
                            continue

                        for key, value in zip(meta, (int(bool(success)), L, k, i, onset)):
                            meta[key].append(value)

                if not meta['onset']:
                    print(f'NO EVENTS: {log_context}')
                    continue

                # Artifact rejection is decided once on the broadband signal and
                # then applied to every band so all band datasets stay aligned.
                onsets = np.asarray(meta['onset'], dtype=float)
                bands_data = {}
                notched = FIRFilter(1001, (49, 51), 'bandstop').fit_transform(raw, fs)

                for name, freq in BANDS.items():
                    # Filter the notch-cleaned EEG into the current band, then
                    # apply common average reference across channels.
                    signal = FIRFilter(1001, freq, 'bandpass').fit_transform(notched, fs)
                    signal = car(signal)

                    # Extract one-second epochs starting at each event onset.
                    epochs = np.asarray(get_epochs_of_events(
                        times, signal, onsets, fs, (0, 1000),
                        w_baseline_t=None, norm=None))

                    if epochs.ndim == 2 and len(onsets) == 1:
                        epochs = epochs[None, :, :]

                    if epochs.ndim != 3 or epochs.shape[0] != len(onsets):
                        raise ValueError(f'Invalid epochs in {name}: {epochs.shape}')

                    if name == 'broadband':
                        # Broadband epochs define the shared clean/noisy mask.
                        _, _, rejected = reject_noisy_epochs(
                            epochs, signal.mean(axis=0), signal.std(axis=0),
                            k=4.0, n_samp=2, n_cha=1)
                        rejected = np.asarray(rejected, dtype=bool)
                        if rejected.shape != (len(onsets),):
                            raise ValueError('Invalid artifact rejection mask')
                        keep = ~rejected
                        epoch_shape = epochs.shape[1:]
                    elif epochs.shape[1:] != epoch_shape:
                        raise ValueError(f'Incompatible epoch dimensions in {name}')

                    clean = epochs[keep]
                    if len(clean):
                        clean = normalize_epochs(clean, norm='z')
                        if not np.isfinite(clean).all():
                            raise ValueError(f'Invalid normalization in {name}')
                    bands_data[name] = np.asarray(clean, dtype=np.float32)

                if 'events' not in hf:
                    hf.attrs['fs'] = fs
                    hf.attrs['epoch_ms'] = (0, 1000)
                    hf.attrs['notch_hz'] = (49, 51)
                    hf.attrs['fir_taps'] = 1001
                    hf.attrs['artifact_k'] = 4.0
                    hf.attrs['artifact_n_samp'] = 2
                    hf.attrs['artifact_n_cha'] = 1
                    hf.attrs['zscore_per_epoch_channel'] = True
                    hf.attrs['stimulus_isi_seconds'] = 1.0
                    hf.create_dataset('channel_names', data=channel_names,
                                      dtype=h5py.string_dtype('utf-8'))

                    events = hf.create_group('events')
                    columns = {'subject': h5py.string_dtype(), 'session': h5py.string_dtype(),
                               'recording': h5py.string_dtype(), 'success': 'u1',
                               'noisy': 'u1', 'L': 'i4', 'k': 'i4', 'trial': 'i4',
                               'onset': 'f8', 'pre_post': 'u1'}
                    for col, dtype in columns.items():
                        events.create_dataset(col, shape=(0,), maxshape=(None,),
                                              dtype=dtype, chunks=True, compression='gzip')

                    hf.create_dataset('clean_index', shape=(0,), maxshape=(None,),
                                      dtype='i8', chunks=True, compression='gzip')

                    group = hf.create_group('bands')
                    for name, freq in BANDS.items():
                        ds = group.create_dataset(
                            name, shape=(0, *epoch_shape), maxshape=(None, *epoch_shape),
                            dtype='f4', chunks=(1, *epoch_shape), compression='gzip')
                        ds.attrs['frequency_hz'] = freq
                elif (not np.isclose(hf.attrs['fs'], fs) or
                      hf['bands/broadband'].shape[1:] != epoch_shape):
                    raise ValueError('Recording is incompatible with the HDF5 layout')

                n = len(onsets)
                rows = {**meta, 'subject': [subject] * n, 'session': [session] * n,
                        'recording': [rel.as_posix()] * n, 'pre_post': [pre_post] * n,
                        'noisy': rejected.astype('u1')}
                start_row = len(hf['events/noisy'])
                rows['clean_index'] = start_row + np.flatnonzero(keep)

                for key, values in rows.items():
                    ds = hf['clean_index'] if key == 'clean_index' else hf[f'events/{key}']
                    previous = len(ds)
                    ds.resize(previous + len(values), axis=0)
                    ds[previous:] = values

                for name, data in bands_data.items():
                    ds = hf[f'bands/{name}']
                    previous = len(ds)
                    ds.resize(previous + len(data), axis=0)
                    ds[previous:] = data

                processed += 1
                total_events += n
                total_clean += int(keep.sum())
                print(f'OK: {log_context} | events={n} | clean={keep.sum()}')

            except Exception as exc:
                errors += 1
                print(f'ERROR: {log_context} | {type(exc).__name__}: {exc}')

        hf.attrs['processed_recordings'] = processed
        hf.attrs['failed_recordings'] = errors

    print(f'{dataset}: {total_events} events, {total_clean} clean, {errors} errors -> {destination.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-adults-root', type=Path, default=Path('raw/old_adults'))
    parser.add_argument('--young-adult-root', type=Path, default=Path('raw/young_adults'))
    parser.add_argument('--output-root', type=Path, default=Path('processed'))
    parser.add_argument('--datasets', nargs='+',
                        choices=tuple(CHANNEL_NAMES), default=list(CHANNEL_NAMES))
    args = parser.parse_args()
    output_root = args.output_root
    roots = {'Old Adults Dataset': args.old_adults_root,
             'Young Adult Dataset': args.young_adult_root}
    for name in args.datasets:
        build_dataset(name, roots[name])
