"""Step 6 — Plot statistical results and regional graph metrics.

Generate adjusted p-value and effect-size matrices, regional violin plots
and band-level summary figures from the participant-level analysis.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageChops
from scipy.stats import gaussian_kde

from corsi_05_statistical_analysis import (BANDS, COMPARISONS, METRICS, N_TESTS,
                                           REGIONS, read_grouped_graph_metrics)
from corsi_04_compute_graph_metrics import OUTPUT_SUBDIR, TRIAL_SELECTION, ZERO_POLICY

DATASETS = ('Old Adults Dataset', 'Young Adult Dataset')
INPUT_ROOT = Path('processed') / 'success_only'
OUTPUT_ROOT = Path('figures') / 'success_only'
CLASS_LABELS = ('0–25', '25–50', '50–75', '75–100')
REGION_SHADING = {'frontal': '#DCEEFB', 'parietal': '#D5C5D4'}
CLASS_COLORS = dict(zip(range(1, 5), ('#90C06C', '#D5E1C5', '#FF9B2F', '#DA7F8F')))
BAND_LABELS = {'theta': 'θ', 'alpha': 'α', 'low_beta': 'Lβ',
               'high_beta': 'Hβ', 'gamma': 'γ'}
STYLES = {
    'p_global_bh_300': {
        'prefix': 'Wilcoxon', 'label': 'p (global BH-FDR, 300 tests)',
        'colors': ['#1c4e4f', '#436e6f', '#6a8e8f', '#879693', '#a49e97', '#F9FEFF'],
        'bounds': [0, .01, .02, .03, .04, .05, 1],
    },
    'effect_size_r': {
        'prefix': 'Effect_size', 'label': r'$r = |Z| / \sqrt{N_{pairs}}$',
        'colors': ['#F9FEFF'] * 3 + ['#C6D870'] * 2 + ['#84994F'] * 5,
        'bounds': np.linspace(0, 1, 11),
    },
}

def read_results(path, dataset):
    """Require all 300 planned contrasts and the effect sizes saved by Step 5."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing Step 5 results: {path}. Run Step 5 first.')
    data = pd.read_csv(path, float_precision='round_trip')
    keys = ['metric', 'band', 'region', 'class_a', 'class_b']
    required = {'dataset', 'status', *keys, *STYLES}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Missing columns {sorted(missing)}; rerun the current Step 5')
    expected = {(metric, band, region, a, b) for metric in METRICS for band in BANDS
                for region in REGIONS[dataset] for a, b in COMPARISONS}
    actual = set(data[keys].itertuples(index=False, name=None))
    if len(data) != N_TESTS or actual != expected or not data['dataset'].eq(dataset).all():
        raise ValueError(f'{dataset}: expected one complete family of 300 unique contrasts')
    if not data['status'].isin(('ok', 'all_zero_differences', 'no_pairs')).all():
        raise ValueError('Unknown statistical test status')
    for column in STYLES:
        values = data[column].to_numpy(dtype=float)
        tested = ~data['status'].eq('no_pairs').to_numpy()
        if not np.isfinite(values[tested]).all() or np.any(values[tested] < 0) or np.any(values[tested] > 1 + 1e-12):
            raise ValueError(f'Invalid values for {column}')
    return data

def comparison_matrix(data, metric, band, column):
    """Map class A>B to lower frontal / transposed upper parietal cells."""
    matrix = np.full((4, 4), np.nan)
    rows = data.loc[data['metric'].eq(metric) & data['band'].eq(band)]
    for row in rows.itertuples(index=False):
        if row.status == 'no_pairs':
            continue
        a, b = int(row.class_a) - 1, int(row.class_b) - 1
        i, j = (a, b) if row.region == 'frontal' else (b, a)
        matrix[i, j] = getattr(row, column)
    return matrix

def save_figure(fig, output_path, transparent=True, *, pad_inches=.15):
    """Finish the PNG before replacing its previous version."""
    output_path = Path(output_path)
    temporary = output_path.with_suffix('.tmp.png')
    fig.savefig(temporary, dpi=300, transparent=transparent,
                bbox_inches='tight', pad_inches=pad_inches)
    temporary.replace(output_path)

def plot_matrix(matrix, column, output_path):
    """Match the model's shaded frame/diagonal; preserve p/r cell palettes."""
    style = STYLES[column]
    cmap = ListedColormap(style['colors'])
    cmap.set_bad('white', alpha=0)
    norm = BoundaryNorm(style['bounds'], len(style['colors']), clip=True)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    try:
        fig.subplots_adjust(left=.18, right=.84, bottom=.18, top=.94)
        low, high = -.045, 1.045
        triangles = {
            'frontal': [(low, high), (low, low), (high, low)],
            'parietal': [(low, high), (high, high), (high, low)],
        }
        for region, vertices in triangles.items():
            ax.add_patch(Polygon(vertices, transform=ax.transAxes,
                                 facecolor=REGION_SHADING[region], edgecolor='none',
                                 clip_on=False, zorder=-1))
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm)
        ax.set(xlim=(-.5, 3.5), ylim=(3.5, -.5))
        ax.set_xticks(range(4), CLASS_LABELS, fontsize=14)
        ax.set_yticks(range(4), CLASS_LABELS, fontsize=14)
        ax.set_xlabel('% Cognitive load', fontsize=18, fontweight='bold', labelpad=12)
        ax.set_ylabel('% Cognitive load', fontsize=18, fontweight='bold', labelpad=12)
        ax.tick_params(axis='both', pad=8)
        ax.plot([low, high], [high, low], transform=ax.transAxes, color='black',
                linewidth=2.5, linestyle='--', dash_capstyle='butt', clip_on=False)
        colorbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.055,
                                ticks=style['bounds'])
        colorbar.ax.tick_params(labelsize=11)
        colorbar.ax.set_yticklabels([f'{value:.2f}' for value in style['bounds']])
        colorbar.set_label(style['label'], fontsize=12, labelpad=10)
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                value = matrix[i, j]
                text, color = 'NA', '#666666'
                if np.isfinite(value):
                    text = '<0.01' if column == 'p_global_bh_300' and value < .01 else f'{value:.2f}'
                    light = value < .03 if column == 'p_global_bh_300' else value > .5
                    color = 'white' if light else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=16)
        save_figure(fig, output_path)
    finally:
        plt.close(fig)

def read_violin_data(results_path, dataset, metrics_dir=None):
    """Read the participant table associated with Step 5."""
    if metrics_dir is None:
        metadata = results_path.with_suffix('.json')
        if not metadata.is_file():
            raise FileNotFoundError(f'Missing Step 5 provenance: {metadata}; check the provenance JSON')
        source = Path(json.loads(metadata.read_text(encoding='utf-8'))['source_graph_metrics'])
    else:
        source = Path(metrics_dir) / f'{dataset}_graph_metrics.csv'
    return read_grouped_graph_metrics(source, dataset)

def violin_limits(rows, metric, *, cut=0):
    """Share regional scales, including the KDE tails when requested."""
    values = rows[metric].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    low, high = values.min(), values.max()
    margin = .15 * (high - low) if high > low else max(abs(low) * .15, 1e-6)
    if cut > 0:
        keys = [key for key in ('region', 'relative_class') if key in rows]
        groups = rows.groupby(keys)[metric] if keys else [(None, rows[metric])]
        for _, group in groups:
            observations = group.to_numpy(dtype=float)
            observations = observations[np.isfinite(observations)]
            if len(observations) < 2 or np.ptp(observations) == 0:
                continue
            try:
                # Same SciPy estimator and Scott bandwidth used by seaborn.
                kde = gaussian_kde(observations, bw_method='scott')
            except np.linalg.LinAlgError:
                continue
            tail = cut * np.sqrt(kde.covariance[0, 0])
            low = min(low, observations.min() - tail)
            high = max(high, observations.max() + tail)
    return low - margin, high + margin

def plot_violin(rows, metric, region, limits, output_path, *, cut=2):
    """Keep the legacy class palette, quartiles and overlaid white box plots."""
    finite = rows.loc[np.isfinite(rows[metric].to_numpy(dtype=float))]
    with plt.rc_context({'font.family': ['DejaVu Sans']}):
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            ax.set_facecolor(REGION_SHADING[region])
            options = dict(data=finite, x='relative_class', y=metric, hue='relative_class',
                           order=(1, 2, 3, 4), hue_order=(1, 2, 3, 4),
                           palette=CLASS_COLORS, dodge=False, legend=False, ax=ax)
            if len(finite):
                sns.violinplot(**options, inner='quartile', linewidth=1.3,
                               bw_method='scott', cut=cut)
                sns.boxplot(**options, fliersize=0, linewidth=1.1, width=.3,
                            boxprops=dict(alpha=.5, facecolor='white'))
            else:
                ax.text(.5, .5, 'No finite subject values', ha='center', va='center',
                        transform=ax.transAxes, fontsize=16)
            ax.set_xticks(range(4), [f'{label}%' for label in CLASS_LABELS])
            ax.set_xlim(-.5, 3.5)
            ax.set_xlabel('% Cognitive Load', fontsize=24, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=24, fontweight='bold')
            ax.tick_params(axis='both', labelsize=20)
            ax.grid(True, axis='y', linestyle='--', alpha=.4)
            ax.set_axisbelow(True)
            if limits is not None:
                ax.set_ylim(limits)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            fig.tight_layout()
            save_figure(fig, output_path, transparent=False)
        finally:
            plt.close(fig)

def composite_panel(path, height=900):
    """Trim empty borders and load only the pixels needed at the final panel size."""
    with Image.open(path) as source:
        rgba = source.convert('RGBA')
        white = Image.new('RGBA', rgba.size, 'white')
        panel = Image.alpha_composite(white, rgba).convert('RGB')
        bounds = ImageChops.difference(panel, Image.new('RGB', panel.size, 'white')).getbbox()
        if bounds is None:
            raise ValueError(f'Empty figure: {path}')
        panel = panel.crop(bounds)
        width = round(panel.width * height / panel.height)
        return panel.resize((width, height), Image.Resampling.LANCZOS)

def plot_composite(dataset, metric, output_dir, bands=BANDS):
    """Reuse saved panels in band rows: frontal, parietal, adjusted p, effect r."""
    output_dir = Path(output_dir)
    sources = [[
        output_dir / f'violin_plot_AEC_{metric}_{band}_frontal_{dataset}.png',
        output_dir / f'violin_plot_AEC_{metric}_{band}_parietal_{dataset}.png',
        output_dir / f'Wilcoxon_{metric}_{band}_{dataset}.png',
        output_dir / f'Effect_size_{metric}_{band}_{dataset}.png',
    ] for band in bands]
    missing = [str(path) for row in sources for path in row if not path.is_file()]
    if missing:
        raise FileNotFoundError('Generate the individual plots first: ' + ', '.join(missing))
    panels = [[composite_panel(path) for path in row] for row in sources]
    row_height, gap, left, top, bottom = 3., .035, .42, .66, .025
    widths = [max(row[column].width / row[column].height for row in panels) * row_height
              for column in range(4)]
    width = left + sum(widths) + 3 * gap + .025
    height = top + len(bands) * row_height + (len(bands) - 1) * gap + bottom
    starts = left + np.r_[0., np.cumsum(np.array(widths[:-1]) + gap)]
    fig = plt.figure(figsize=(width, height), facecolor='white')
    try:
        fig.legend(handles=[Patch(facecolor=color, label=region.title())
                            for region, color in REGION_SHADING.items()],
                   loc='upper center', bbox_to_anchor=(.5, 1), frameon=False,
                   ncol=2, fontsize=16, handlelength=2.2, handleheight=1.2,
                   columnspacing=3., borderaxespad=.05)
        for column, label in enumerate(('Frontal violin', 'Parietal violin', 'Adjusted p', 'Effect size r')):
            fig.text((starts[column] + widths[column] / 2) / width, 1 - .53 / height,
                     label, ha='center', va='center', fontsize=14)
        for row_index, (band, row) in enumerate(zip(bands, panels)):
            y = height - top - row_height - row_index * (row_height + gap)
            fig.text(left * .43 / width, (y + row_height / 2) / height,
                     BAND_LABELS[band], ha='center', va='center', fontsize=18, fontweight='bold')
            for column, panel in enumerate(row):
                panel_width = panel.width / panel.height * row_height
                x = starts[column] + (widths[column] - panel_width) / 2
                ax = fig.add_axes([x / width, y / height, panel_width / width, row_height / height])
                ax.imshow(np.asarray(panel), interpolation='none')
                ax.set_axis_off()
        path = output_dir / f'{dataset}_{metric}_AEC.png'
        save_figure(fig, path, transparent=False, pad_inches=.02)
        return path
    finally:
        plt.close(fig)

def plot_dataset(dataset, input_dir, output_dir,
                    metrics=METRICS, bands=BANDS, *, metrics_dir=None,
                    plots=('matrices', 'violins', 'composites'), violin_cut=2):
    if not plots or set(plots) - {'matrices', 'violins', 'composites'}:
        raise ValueError('Choose matrices, violins and/or composites')
    results_path = Path(input_dir) / f'{dataset}_wilcoxon_global_bh_300.csv'
    data = read_results(results_path, dataset) if set(plots) & {'matrices', 'violins'} else None
    metrics_table = read_violin_data(results_path, dataset, metrics_dir) if 'violins' in plots else None
    by_band = {} if metrics_table is None else {band: metrics_table.loc[metrics_table.band.eq(band)] for band in bands}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric in metrics:
        for band in bands:
            if 'matrices' in plots:
                for column, style in STYLES.items():
                    matrix = comparison_matrix(data, metric, band, column)
                    path = output_dir / f'{style["prefix"]}_{metric}_{band}_{dataset}.png'
                    plot_matrix(matrix, column, path)
                    paths.append(path)
            if metrics_table is not None:
                rows = by_band[band]
                limits = violin_limits(rows, metric, cut=violin_cut)
                for region in REGIONS[dataset]:
                    path = output_dir / f'violin_plot_AEC_{metric}_{band}_{region}_{dataset}.png'
                    plot_violin(rows.loc[rows.region.eq(region)], metric, region, limits, path, cut=violin_cut)
                    paths.append(path)
        if 'composites' in plots:
            paths.append(plot_composite(dataset, metric, output_dir, bands))
    print(f'{dataset} | figures={len(paths)} | saved={output_dir}')
    return paths



def process_dataset(dataset, root=INPUT_ROOT, figures_root=OUTPUT_ROOT):
    """Produce figures from the validated statistical results."""
    metrics_dir = Path(root) / OUTPUT_SUBDIR
    statistics = metrics_dir / 'statistics'
    provenance_path = statistics / f'{dataset}_wilcoxon_global_bh_300.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if (provenance.get('trial_selection') != TRIAL_SELECTION or
            provenance.get('modularity_zero_policy') != ZERO_POLICY or
            provenance.get('graph_aggregation') != 'metrics_of_mean_aec'):
        raise ValueError('Step 5 provenance is inconsistent with the graph metrics')
    return plot_dataset(dataset, statistics, Path(figures_root), metrics_dir=metrics_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    for dataset_name in args.datasets:
        process_dataset(dataset_name, args.input_root, args.output_root)
