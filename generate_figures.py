"""
FORCE Paper - Figure Generation Script (Revised)
Generates publication-quality figures for MDPI Mathematics journal.

Compatible input formats:
(A) results_raw_*.csv:
    columns include: dataset, algorithm, run_id, time_ms, rmse, (optional n_samples/n_features)
(B) benchmark_per_run.csv (generate_trimmed_pearson_artifacts.py):
    columns include: dataset, method, run, time_ms, rmse

Figures:
1. Execution time comparison (log scale grouped bar chart)
2. Measured runtime versus reference RMSE, separately for each dataset
3. RMSE comparison (grouped bar chart)
4. FORCE speedup heatmap (bonus)
5. Combined performance (bonus)

Author: Sooyoung Jang (revised to support TrimmedPearson baselines and new pipeline)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Publication-quality defaults
# -------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
})

# -------------------------
# Algorithms, colors, categories
# -------------------------
# Color palette - professional and colorblind-friendly (extend for trimmed baselines)
COLORS = {
    'Pearson': '#1f77b4',                   # Blue
    'Spearman': '#ff7f0e',                  # Orange
    'Winsorized': '#2ca02c',                # Green
    'FastMCD': '#d62728',                   # Red
    'TrimmedPearsonExact(no TER)': '#8c564b',# Brown
    'TrimmedPearsonExact(TER)': '#e377c2',   # Pink
    'TP-Exact': '#8c564b',
    'TP-TER': '#e377c2',
    'FORCE': '#9467bd',                     # Purple
}

ALGO_CATEGORIES = {
    'Pearson': 'Moment-based',
    'FORCE': 'Moment-based',
    'Spearman': 'Rank-based',
    'Winsorized': 'Rank-based',
    'FastMCD': 'High-breakdown',
    'TrimmedPearsonExact(no TER)': 'Trimmed-moment',
    'TrimmedPearsonExact(TER)': 'Trimmed-moment',
}

# Order consistent with generate_trimmed_pearson_artifacts.py preference
PREFERRED_ALGO_ORDER = [
    'Pearson',
    'Spearman',
    'Winsorized',
    'FastMCD',
    'TrimmedPearsonExact(no TER)',
    'TrimmedPearsonExact(TER)',
    'FORCE',
]

# Dataset display names (extend as needed)
DATASET_NAMES = {
    'Synthetic': 'Synthetic',
    'SP500': 'S&P 500',
    'ODDS-mammography': 'Mammography',
    'ODDS-satellite': 'Satellite',
    'Genomics': 'Genomics',
}

PREFERRED_DATASET_ORDER = [
    'Synthetic',
    'SP500',
    'Genomics',
    'ODDS-mammography',
    'ODDS-satellite',
]


# -------------------------
# Utilities
# -------------------------
IDENTITY_COLUMNS = [
    'protocol', 'mode', 'execution_schedule', 'estimator', 'estimator_parameters',
    'bound_policy', 'policy_parameters', 'comparison_target', 'clean_reference_target',
    'target', 'target_definition', 'quantile_backend', 'n_samples', 'n_features',
    'replicate_kind', 'rmse_interpretation', 'identity_status',
]
COMPARISON_COLUMNS = [
    'comparison_target', 'clean_reference_target', 'target', 'target_definition',
    'n_samples', 'n_features', 'replicate_kind',
]
DATA_IDENTITY_COLUMNS = [
    'data_sha256', 'reference_sha256', 'data_identity', 'data_id', 'dataset_id',
    'dataset_sha256', 'source_sha256', 'reference_id',
    'base_data_sha256', 'order_sha256', 'order_seed',
    'eligible_mask_sha256', 'comparison_mask_sha256',
    'eligible_pair_count', 'compared_pair_count',
]
INDEPENDENT_REPLICATES = {'independent_data', 'independent_order'}
REQUIRED_IDENTITY = ['protocol', 'comparison_target', 'n_samples', 'n_features',
                     'data_sha256', 'reference_sha256']


def _validate_identity(frame: pd.DataFrame) -> None:
    # A method name cannot silently denote different protocols or modes.
    for name in ('protocol', 'mode'):
        if name in frame and frame[name].nunique(dropna=False) != 1:
            raise ValueError(f"Mixed {name} values; generate separate figures for each identity.")
    for _, group in frame.groupby(['dataset', 'method'], dropna=False):
        for name in IDENTITY_COLUMNS:
            if name in group and group[name].nunique(dropna=False) != 1:
                raise ValueError(f"Mixed {name} values within a dataset/method; split the input.")
    for _, group in frame.groupby('dataset', dropna=False):
        for name in COMPARISON_COLUMNS:
            if name in group and group[name].nunique(dropna=False) != 1:
                raise ValueError(f"Mixed {name} values across methods for one dataset.")
        kind = group['replicate_kind'].iloc[0] if 'replicate_kind' in group else None
        if kind == 'independent_order':
            for name in ('base_data_sha256', 'reference_sha256'):
                if name in group and group[name].nunique(dropna=False) != 1:
                    raise ValueError(f"Mixed {name} values in fixed-data order replicates.")
        if kind in INDEPENDENT_REPLICATES:
            if 'replicate_id' not in group or group['replicate_id'].isna().any():
                raise ValueError('Independent replicates require explicit replicate_id values.')
            if group.duplicated(['method', 'replicate_id']).any():
                raise ValueError('Duplicate replicate_id rows would count timing reruns as independent observations.')
            replicate_sets = [set(method['replicate_id']) for _, method in group.groupby('method')]
            if any(ids != replicate_sets[0] for ids in replicate_sets[1:]):
                raise ValueError('Methods must use the same independent replicate identities.')
            comparable = [replicate for _, replicate in group.groupby('replicate_id')]
        else:
            comparable = [group]
        for sample in comparable:
            for name in DATA_IDENTITY_COLUMNS:
                if name in sample and sample[name].nunique(dropna=False) != 1:
                    raise ValueError(f"Mixed {name} values across compared measurements.")


def standardize_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names without discarding estimator/target identity.

    Archival CSVs lacking provenance can be plotted, but are explicitly marked
    as unidentified. Failed or nonfinite measurements require attention instead
    of disappearing from the figures.
    """
    df = df.copy()
    if 'method' not in df.columns and 'algorithm' in df.columns:
        df = df.rename(columns={'algorithm': 'method'})
    if 'run' not in df.columns and 'run_id' in df.columns:
        df = df.rename(columns={'run_id': 'run'})
    required = {'dataset', 'method', 'run', 'time_ms', 'rmse'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError("Input measurements must be nonempty and have no missing required values.")
    if 'status' in df and not df['status'].eq('completed').all():
        raise ValueError("Input includes failed or incomplete measurements; inspect them before plotting.")
    if 'estimate_status' in df and not df['estimate_status'].eq('complete').all():
        raise ValueError("Input includes incomplete pair coverage; inspect validity before plotting.")
    if 'timing_status' in df and not df['timing_status'].eq('warmed').all():
        raise ValueError("Input includes timings without a successful warm-up.")
    if 'requested_runs' in df:
        for _, group in df.groupby(['dataset', 'method'], dropna=False):
            counts = pd.to_numeric(group['requested_runs'], errors='raise')
            if counts.isna().any() or counts.nunique() != 1 or counts.iloc[0] != len(group):
                raise ValueError("Input omits requested runs; success-conditioned figures need an explicit design.")
    for name in ('run', 'time_ms', 'rmse'):
        df[name] = pd.to_numeric(df[name], errors='raise')
        if not np.isfinite(df[name]).all():
            raise ValueError(f"Input column {name} must be finite.")
    if (df['time_ms'] <= 0).any() or (df['rmse'] < 0).any():
        raise ValueError("Execution times must be positive and RMSE nonnegative.")
    for name in ('dataset', 'method'):
        df[name] = df[name].astype(str)
    for name in ('estimator_parameters', 'policy_parameters'):
        if name in df:
            def canonical(value):
                if isinstance(value, str):
                    value = json.loads(value)
                return json.dumps(value, sort_keys=True, allow_nan=False)
            df[name] = df[name].map(canonical)
    missing_identity = [name for name in REQUIRED_IDENTITY
                        if name not in df or df[name].isna().any()]
    df['identity_status'] = 'provided' if not missing_identity else 'archival_partially_identified'
    _validate_identity(df)
    if missing_identity:
        warnings.warn('CSV lacks complete comparison identity (' + ', '.join(missing_identity)
                      + '); archival comparisons are not verified.', UserWarning)
    return df


def compute_summary(df_std: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated runs while retaining their verified identity."""
    _validate_identity(df_std)
    aggregation = dict(
        time_mean=('time_ms', 'mean'), time_std=('time_ms', 'std'),
        n_runs=('time_ms', 'count'), rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
    )
    for name in IDENTITY_COLUMNS:
        if name in df_std:
            aggregation[name] = (name, 'first')
    g = df_std.groupby(['dataset', 'method'], as_index=False).agg(**aggregation)
    g['time_std'] = g['time_std'].fillna(0.0)
    g['rmse_std'] = g['rmse_std'].fillna(0.0)
    # Timing repetition does not create independent statistical error samples.
    g['time_ci'] = 1.96 * g['time_std'] / np.sqrt(g['n_runs'])
    g['rmse_ci'] = np.nan
    g['rmse_ci_basis'] = 'not_estimated'
    if 'replicate_kind' in g:
        independent = g['replicate_kind'].isin(INDEPENDENT_REPLICATES) & (g['n_runs'] > 1)
        g.loc[independent, 'rmse_ci'] = 1.96 * g.loc[independent, 'rmse_std'] / np.sqrt(g.loc[independent, 'n_runs'])
        g.loc[independent, 'rmse_ci_basis'] = 'normal95_' + g.loc[independent, 'replicate_kind']
    return g


def resolve_orders(df_std: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Determine dataset/method order:
      - Use preferred orders filtered by presence
      - Append any remaining items alphabetically
    """
    datasets_present = list(df_std['dataset'].unique())
    methods_present = list(df_std['method'].unique())

    ds_order = [d for d in PREFERRED_DATASET_ORDER if d in datasets_present]
    ds_order += sorted([d for d in datasets_present if d not in ds_order])

    algo_order = [m for m in PREFERRED_ALGO_ORDER if m in methods_present]
    algo_order += sorted([m for m in methods_present if m not in algo_order])

    return ds_order, algo_order


def _method_color(method: str) -> str:
    return COLORS.get(method, '#7f7f7f')  # fallback gray


def _dataset_label(dataset: str) -> str:
    return DATASET_NAMES.get(dataset, dataset)


def _save_all_formats(fig, out_dir: Path, stem: str, summary: pd.DataFrame | None = None) -> None:
    """Save an editable vector export and a PNG preview."""
    if summary is not None and (
        'identity_status' not in summary
        or not summary['identity_status'].eq('provided').all()
    ):
        fig.text(0.5, -0.015, 'Archival input: incomplete data/reference identity; comparability is unverified.',
                 ha='center', fontsize=8)
    fig.savefig(out_dir / f"{stem}.png", format='png', dpi=300)
    fig.savefig(out_dir / f"{stem}.pdf", format='pdf')


# -------------------------
# Figure 1: Time comparison
# -------------------------
def figure1_time_comparison(summary: pd.DataFrame, datasets: list[str], methods: list[str], output_path: Path):
    """
    Execution time comparison across algorithms and datasets (log scale).
    Grouped bar chart with dynamic offsets (supports 5→7→N methods).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    n_datasets = len(datasets)
    n_methods = len(methods)

    x = np.arange(n_datasets)

    # Dynamic bar width: keep reasonable spacing
    width = min(0.12, 0.75 / max(n_methods, 1))
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    # Plot each method
    for i, method in enumerate(methods):
        mdata = summary[summary['method'] == method].set_index('dataset')

        times = [float(mdata.loc[d, 'time_mean']) if d in mdata.index else np.nan for d in datasets]
        errs = [float(mdata.loc[d, 'time_ci']) if d in mdata.index else np.nan for d in datasets]

        ax.bar(
            x + offsets[i],
            times,
            width,
            label=method,
            color=_method_color(method),
            yerr=errs,
            capsize=2,
            error_kw={'linewidth': 0.8, 'capthick': 0.8},
            edgecolor='black',
            linewidth=0.5,
        )

    ax.set_yscale('log')
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_dataset_label(d) for d in datasets])

    # Dynamic y-limits (safe for 2-run and 20-run cases)
    positive_times = summary['time_mean'].to_numpy()
    positive_times = positive_times[positive_times > 0]
    if positive_times.size:
        y_min = max(0.05, np.min(positive_times) / 3.0)
        y_max = np.max(positive_times) * 3.0
    else:
        y_min, y_max = 0.1, 2000

    ax.set_ylim(y_min, max(y_max, y_min * 10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'{v:g}'))

    # Tier lines (optional but useful in paper)
    for y in [1, 10, 100, 1000]:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.35)

    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend below figure in two rows
    ax.legend(
        loc='upper center',
        framealpha=0.95,
        ncol=4,  # 4 columns = 2 rows for 7 methods
        bbox_to_anchor=(0.5, -0.15),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legend below
    _save_all_formats(fig, output_path, 'time_comparison', summary)
    plt.close()


# -------------------------
# Figure 2: Estimator landscape
# -------------------------
def figure2_estimator_landscape(summary: pd.DataFrame, methods: list[str], output_path: Path):
    """Plot measured runtime/reference error without assigning breakdown points.

    Datasets remain separate: their reference targets and sample dimensions
    may differ. Each point is the supplied within-dataset repeated-run mean.
    """
    datasets = list(summary['dataset'].unique())
    ncols = min(2, len(datasets))
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.1, 2.7 * nrows + 0.8),
                             squeeze=False, layout='constrained')
    for ax, dataset in zip(axes.flat, datasets):
        subset = summary[summary['dataset'] == dataset].set_index('method')
        for index, method in enumerate(methods):
            if method not in subset.index:
                continue
            row = subset.loc[method]
            ax.scatter(float(row['time_mean']), float(row['rmse_mean']),
                       label=method, color=_method_color(method),
                       marker=['o', 's', '^', 'v', 'D', 'P', 'X'][index % 7], s=45)
        ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Mean execution time (ms)')
        ax.set_ylabel('Mean RMSE vs supplied reference')
        ax.set_title(_dataset_label(dataset))
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(datasets):]:
        ax.set_visible(False)
    handles = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    fig.legend(handles.values(), handles.keys(), loc='outside lower center',
               ncol=min(3, len(handles)), fontsize=8)
    fig.suptitle('Measured runtime and reference error; no breakdown point inferred', fontsize=10)
    _save_all_formats(fig, output_path, 'runtime_accuracy', summary)
    plt.close(fig)


# -------------------------
# Figure 3: RMSE comparison
# -------------------------
def figure3_rmse_comparison(summary: pd.DataFrame, datasets: list[str], methods: list[str], output_path: Path):
    """
    Reference errors; intervals require declared independent replicates.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    n_datasets = len(datasets)
    n_methods = len(methods)

    x = np.arange(n_datasets)
    width = min(0.12, 0.75 / max(n_methods, 1))
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    for i, method in enumerate(methods):
        mdata = summary[summary['method'] == method].set_index('dataset')

        vals = [float(mdata.loc[d, 'rmse_mean']) if d in mdata.index else np.nan for d in datasets]
        errs = [float(mdata.loc[d, 'rmse_ci']) if d in mdata.index else np.nan for d in datasets]

        ax.bar(
            x + offsets[i],
            vals,
            width,
            label=method,
            color=_method_color(method),
            yerr=errs if np.isfinite(errs).any() else None,
            capsize=2,
            error_kw={'linewidth': 0.8, 'capthick': 0.8},
            edgecolor='black',
            linewidth=0.5,
        )

    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('RMSE vs supplied reference', fontweight='bold')
    if (summary['rmse_ci'].notna().any() and 'replicate_kind' in summary
            and summary['replicate_kind'].eq('independent_order').all()):
        note = '95% intervals for mean error over random orderings of each fixed dataset'
    elif summary['rmse_ci'].notna().any():
        note = 'Error bars: 95% normal intervals over declared independent data/order replicates'
    elif 'replicate_kind' in summary and summary['replicate_kind'].eq('fixed_dataset_timing').all():
        note = 'Conditional fixed-dataset error; timing reruns provide no RMSE confidence interval'
    else:
        note = 'Replicate design unspecified or insufficient; no RMSE confidence interval'
    ax.set_title(note, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([_dataset_label(d) for d in datasets])

    # Dynamic y-limit
    positive_rmse = summary['rmse_mean'].to_numpy()
    positive_rmse = positive_rmse[positive_rmse > 0]
    if positive_rmse.size:
        ax.set_ylim(0, float(np.max(positive_rmse) * 1.25))
    else:
        ax.set_ylim(0, 1.0)

    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend below figure in two rows
    ax.legend(
        loc='upper center',
        framealpha=0.95,
        ncol=4,  # 4 columns = 2 rows for 7 methods
        bbox_to_anchor=(0.5, -0.15),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legend below
    _save_all_formats(fig, output_path, 'rmse_comparison', summary)
    plt.close()


# -------------------------
# Figure 4: FORCE speedup heatmap (bonus)
# -------------------------
def figure4_speedup_heatmap(summary: pd.DataFrame, datasets: list[str], methods: list[str], output_path: Path):
    """
    Heatmap: Speedup of FORCE over all other methods (time_baseline / time_FORCE).
    Includes TrimmedPearson baselines automatically if present.
    """
    force_name = 'FORCE'
    if force_name not in methods:
        print("Skipping speedup heatmap: FORCE not found in methods.")
        return

    baselines = [m for m in methods if m != force_name]

    # Build speedup matrix
    speedup_matrix = []
    for ds in datasets:
        force_rows = summary[(summary['dataset'] == ds) & (summary['method'] == force_name)]
        if force_rows.empty:
            speedup_matrix.append([np.nan] * len(baselines))
            continue
        force_time = float(force_rows['time_mean'].values[0])

        row = []
        for b in baselines:
            b_rows = summary[(summary['dataset'] == ds) & (summary['method'] == b)]
            if b_rows.empty:
                row.append(np.nan)
            else:
                b_time = float(b_rows['time_mean'].values[0])
                row.append(b_time / force_time if force_time > 0 else np.nan)
        speedup_matrix.append(row)

    speedup = np.array(speedup_matrix, dtype=float)

    # Log scale visualization
    log_speedup = np.log10(speedup)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(log_speedup, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=3.5)

    # Annotate values
    for i in range(len(datasets)):
        for j in range(len(baselines)):
            val = speedup[i, j]
            if np.isnan(val):
                txt = "—"
                color = "black"
            else:
                if val < 1:
                    txt = f'{val:.2f}×'
                    color = 'white'
                elif val < 10:
                    txt = f'{val:.1f}×'
                    color = 'black'
                else:
                    txt = f'{val:.0f}×'
                    color = 'white' if val > 100 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, fontweight='bold', color=color)

    ax.set_xticks(np.arange(len(baselines)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(baselines, rotation=30, ha='right')
    ax.set_yticklabels([_dataset_label(d) for d in datasets])

    ax.set_xlabel('Baseline Method', fontweight='bold')
    ax.set_ylabel('Dataset', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, label='Speedup (log₁₀ scale)')
    cbar.set_ticks([-1, 0, 1, 2, 3])
    cbar.set_ticklabels(['0.1×', '1×', '10×', '100×', '1000×'])

    ax.set_title('FORCE Speedup Over Other Methods', fontweight='bold', pad=10)

    plt.tight_layout()
    _save_all_formats(fig, output_path, 'speedup_heatmap', summary)
    plt.close()


# -------------------------
# Figure 5: Combined performance (bonus)
# -------------------------
def figure5_combined_performance(summary: pd.DataFrame, datasets: list[str], methods: list[str], output_path: Path):
    """
    Two-panel figure: time (log) and RMSE grouped bars.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    n_datasets = len(datasets)
    n_methods = len(methods)

    x = np.arange(n_datasets)
    width = min(0.12, 0.75 / max(n_methods, 1))
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    # (a) Time
    for i, method in enumerate(methods):
        mdata = summary[summary['method'] == method].set_index('dataset')
        times = [float(mdata.loc[d, 'time_mean']) if d in mdata.index else np.nan for d in datasets]
        ax1.bar(x + offsets[i], times, width, label=method, color=_method_color(method),
                edgecolor='black', linewidth=0.5)

    ax1.set_yscale('log')
    ax1.set_xlabel('Dataset', fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([_dataset_label(d) for d in datasets], ha='center')
    ax1.set_title('(a) Computational Efficiency', fontweight='bold')
    ax1.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax1.set_axisbelow(True)

    # (b) RMSE
    for i, method in enumerate(methods):
        mdata = summary[summary['method'] == method].set_index('dataset')
        vals = [float(mdata.loc[d, 'rmse_mean']) if d in mdata.index else np.nan for d in datasets]
        ax2.bar(x + offsets[i], vals, width, label=method, color=_method_color(method),
                edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Dataset', fontweight='bold')
    ax2.set_ylabel('RMSE vs supplied reference', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([_dataset_label(d) for d in datasets], ha='center')
    ax2.set_title('(b) Reference Error', fontweight='bold')
    ax2.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax2.set_axisbelow(True)

    # Single legend below figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend below
    _save_all_formats(fig, output_path, 'combined_performance', summary)
    plt.close()


def print_summary_statistics(summary: pd.DataFrame, datasets: list[str], methods: list[str]):
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (mean ± std; n runs; CI half-width)")
    print("=" * 70)

    print("\n--- Execution Time (ms) ---")
    for ds in datasets:
        print(f"\n{ds}:")
        ds_data = summary[summary['dataset'] == ds]
        for m in methods:
            row = ds_data[ds_data['method'] == m]
            if not row.empty:
                mean = float(row['time_mean'].values[0])
                std = float(row['time_std'].values[0])
                n = int(row['n_runs'].values[0])
                ci = float(row['time_ci'].values[0])
                print(f"  {m:26s}: {mean:8.3f} ± {std:7.3f} (n={n:2d}, CI±{ci:.3f})")

    print("\n--- RMSE ---")
    for ds in datasets:
        print(f"\n{ds}:")
        ds_data = summary[summary['dataset'] == ds]
        for m in methods:
            row = ds_data[ds_data['method'] == m]
            if not row.empty:
                mean = float(row['rmse_mean'].values[0])
                std = float(row['rmse_std'].values[0])
                n = int(row['n_runs'].values[0])
                ci = float(row['rmse_ci'].values[0])
                interval = f'CI±{ci:.5f}' if np.isfinite(ci) else 'RMSE CI not estimated'
                print(f"  {m:26s}: {mean:8.5f} ± {std:8.5f} (n={n:2d}, {interval})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True,
                    help="Path to results_raw_*.csv or benchmark_per_run.csv")
    ap.add_argument("--output_dir", type=str, default="generated_figures/benchmarks",
                    help="Directory for measured figure exports (PDF and PNG)")
    args = ap.parse_args()

    input_file = Path(args.input_csv)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("FORCE Paper - Figure Generation (Revised)")
    print("=" * 50)
    print(f"Input CSV: {input_file}")
    print(f"Output dir: {output_path}")

    df_raw = pd.read_csv(input_file)
    df = standardize_input_schema(df_raw)
    summary = compute_summary(df)

    datasets, methods = resolve_orders(df)

    print(f"\nLoaded {len(df)} rows (standardized), {len(summary)} summary entries")
    print(f"Datasets: {datasets}")
    print(f"Methods:  {methods}")

    print_summary_statistics(summary, datasets, methods)

    print("\nGenerating figures...")
    figure1_time_comparison(summary, datasets, methods, output_path)
    print(f"Saved: {output_path / 'time_comparison.*'}")

    figure2_estimator_landscape(summary, methods, output_path)
    print(f"Saved: {output_path / 'runtime_accuracy.*'}")

    figure3_rmse_comparison(summary, datasets, methods, output_path)
    print(f"Saved: {output_path / 'rmse_comparison.*'}")

    figure4_speedup_heatmap(summary, datasets, methods, output_path)
    print(f"Saved: {output_path / 'speedup_heatmap.*'}")

    figure5_combined_performance(summary, datasets, methods, output_path)
    print(f"Saved: {output_path / 'combined_performance.*'}")

    identities = [name for name in IDENTITY_COLUMNS + DATA_IDENTITY_COLUMNS + ['replicate_id'] if name in df]
    provenance = {
        'source_csv': str(input_file.resolve()),
        'source_sha256': hashlib.sha256(input_file.read_bytes()).hexdigest(),
        'identities': df[['dataset', 'method'] + identities].drop_duplicates().to_dict(orient='records'),
        'identity_status': df['identity_status'].iloc[0],
        'interpretation': 'Measured runtime and supplied-reference error; no breakdown guarantee.',
        'aggregation': 'Arithmetic means within dataset/method. Timing intervals describe supplied timing repetitions. RMSE intervals require explicitly declared independent_data/independent_order replicates with unique replicate_id values; fixed-dataset or unidentified repetitions have no RMSE interval. Order-only uncertainty is conditional on its fixed dataset.',
    }
    (output_path / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print("\nDone.")


if __name__ == "__main__":
    main()
