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
2. Estimator landscape (speed-robustness plane)
3. RMSE comparison (grouped bar chart)
4. FORCE speedup heatmap (bonus)
5. Combined performance (bonus)

Author: Sooyoung Jang (revised to support TrimmedPearson baselines and new pipeline)
"""

from __future__ import annotations

import argparse
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
def standardize_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to columns: dataset, method, run, time_ms, rmse
    Accepts either:
      - (dataset, algorithm, run_id, time_ms, rmse, ...)
      - (dataset, method, run, time_ms, rmse)
    """
    df = df.copy()

    if 'method' not in df.columns and 'algorithm' in df.columns:
        df = df.rename(columns={'algorithm': 'method'})
    if 'run' not in df.columns and 'run_id' in df.columns:
        df = df.rename(columns={'run_id': 'run'})

    required = {'dataset', 'method', 'run', 'time_ms', 'rmse'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns after standardization: {sorted(missing)}")

    # Keep only relevant columns but tolerate extras
    keep_cols = ['dataset', 'method', 'run', 'time_ms', 'rmse']
    df = df[keep_cols].copy()

    # Enforce types
    df['dataset'] = df['dataset'].astype(str)
    df['method'] = df['method'].astype(str)
    df['run'] = pd.to_numeric(df['run'], errors='coerce').astype('Int64')
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
    df['rmse'] = pd.to_numeric(df['rmse'], errors='coerce')

    df = df.dropna(subset=['dataset', 'method', 'time_ms', 'rmse'])
    return df


def compute_summary(df_std: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary per dataset-method with mean/std/count and 95% CI half-width.
    """
    g = df_std.groupby(['dataset', 'method'], as_index=False).agg(
        time_mean=('time_ms', 'mean'),
        time_std=('time_ms', 'std'),
        n_runs=('time_ms', 'count'),
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
    )

    # Replace NaN std for n=1 with 0 to avoid CI NaNs
    g['time_std'] = g['time_std'].fillna(0.0)
    g['rmse_std'] = g['rmse_std'].fillna(0.0)

    # 95% normal CI half-width (consistent with your original code)
    g['time_ci'] = 1.96 * g['time_std'] / np.sqrt(np.maximum(g['n_runs'], 1))
    g['rmse_ci'] = 1.96 * g['rmse_std'] / np.sqrt(np.maximum(g['n_runs'], 1))
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


def _save_all_formats(fig, out_dir: Path, stem: str) -> None:
    """Save figure as PNG only."""
    fig.savefig(out_dir / f"{stem}.png", format='png', dpi=300)


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

        times = [float(mdata.loc[d, 'time_mean']) if d in mdata.index else 0.0 for d in datasets]
        errs = [float(mdata.loc[d, 'time_ci']) if d in mdata.index else 0.0 for d in datasets]

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
    _save_all_formats(fig, output_path, 'time_comparison')
    plt.close()


# -------------------------
# Figure 2: Estimator landscape
# -------------------------
def figure2_estimator_landscape(summary: pd.DataFrame, methods: list[str], output_path: Path):
    """
    Scatter plot showing methods positioned by speed (x-axis, log) and robustness (y-axis).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Average time across datasets per method
    avg_times = summary.groupby('method')['time_mean'].mean()

    # Breakdown points (%): keep consistent with paper narrative
    # TrimmedPearson baselines use the same acceptance/trimming concept; we display ~25%.
    breakdown_points = {
        'Pearson': 0,
        'Spearman': 0,
        'Winsorized': 10,
        'FastMCD': 50,
        'TrimmedPearsonExact(no TER)': 25,
        'TrimmedPearsonExact(TER)': 25,
        'FORCE': 25,
    }

    # Marker styles for each method - differentiate TrimmedPearson variants
    marker_styles = {
        'Pearson': 'o',
        'Spearman': 'o',
        'Winsorized': 'o',
        'FastMCD': 'o',
        'TrimmedPearsonExact(no TER)': '^',  # Triangle up
        'TrimmedPearsonExact(TER)': 'v',      # Triangle down
        'FORCE': 's',                          # Square
    }

    # Y-axis offset to prevent overlap for methods with same breakdown point
    y_offsets = {
        'Pearson': 0,
        'Spearman': 0,
        'Winsorized': 0,
        'FastMCD': 0,
        'TrimmedPearsonExact(no TER)': 1.5,   # Slightly above 25%
        'TrimmedPearsonExact(TER)': -1.5,     # Slightly below 25%
        'FORCE': 0,
    }

    # Store scatter handles for legend
    scatter_handles = []
    scatter_labels = []

    for method in methods:
        if method not in avg_times.index:
            continue

        x = float(avg_times[method])
        y = float(breakdown_points.get(method, np.nan)) + y_offsets.get(method, 0)

        marker_size = 320 if method == 'FORCE' else 220
        marker = marker_styles.get(method, 'o')
        edgecolor = 'black' if method == 'FORCE' else 'white'
        linewidth = 2 if method == 'FORCE' else 1

        scatter = ax.scatter(
            x, y,
            s=marker_size,
            c=_method_color(method),
            marker=marker,
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=5,
            label=method
        )
        scatter_handles.append(scatter)
        scatter_labels.append(method)

    ax.set_xscale('log')
    ax.set_xlabel('Average Execution Time (ms)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Breakdown Point (%)', fontweight='bold', fontsize=11)

    # Limits - extend x-axis for FastMCD which can be very slow
    x_max = max(50000, float(avg_times.max()) * 2 if len(avg_times) else 50000)
    ax.set_xlim(0.05, x_max)
    ax.set_ylim(-5, 60)

    # Quadrant shading (adjusted for wider x range)
    rect1 = plt.Rectangle((0.05, -5), 10 - 0.05, 15, facecolor='#ffcccc', alpha=0.3, zorder=1)
    rect2 = plt.Rectangle((10, 15), x_max - 10, 60 - 15, facecolor='#ccccff', alpha=0.3, zorder=1)
    rect3 = plt.Rectangle((0.05, 15), 10 - 0.05, 60 - 15, facecolor='#ccffcc', alpha=0.4, zorder=1)
    ax.add_patch(rect1); ax.add_patch(rect2); ax.add_patch(rect3)

    ax.text(0.15, 2, 'Fast-Fragile', fontsize=9, color='#cc0000', style='italic', alpha=0.8)
    ax.text(500, 55, 'Slow-Robust', fontsize=9, color='#0000cc', style='italic', alpha=0.8)
    ax.text(0.15, 55, 'Fast-Robust\n(target)', fontsize=9, color='#006600', fontweight='bold', alpha=0.9)

    ax.axvline(x=10, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=15, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'{v:g}'))

    # Legend below figure in two rows (no text annotations in plot)
    ax.legend(
        loc='upper center',
        framealpha=0.95,
        ncol=4,
        bbox_to_anchor=(0.5, -0.12),
        fontsize=9
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend below
    _save_all_formats(fig, output_path, 'estimator_landscape')
    plt.close()


# -------------------------
# Figure 3: RMSE comparison
# -------------------------
def figure3_rmse_comparison(summary: pd.DataFrame, datasets: list[str], methods: list[str], output_path: Path):
    """
    Grouped bar chart showing RMSE values with error bars (95% CI).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    n_datasets = len(datasets)
    n_methods = len(methods)

    x = np.arange(n_datasets)
    width = min(0.12, 0.75 / max(n_methods, 1))
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    for i, method in enumerate(methods):
        mdata = summary[summary['method'] == method].set_index('dataset')

        vals = [float(mdata.loc[d, 'rmse_mean']) if d in mdata.index else 0.0 for d in datasets]
        errs = [float(mdata.loc[d, 'rmse_ci']) if d in mdata.index else 0.0 for d in datasets]

        ax.bar(
            x + offsets[i],
            vals,
            width,
            label=method,
            color=_method_color(method),
            yerr=errs,
            capsize=2,
            error_kw={'linewidth': 0.8, 'capthick': 0.8},
            edgecolor='black',
            linewidth=0.5,
        )

    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('RMSE (Lower is Better)', fontweight='bold')
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
    _save_all_formats(fig, output_path, 'rmse_comparison')
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
    fig.savefig(output_path / 'speedup_heatmap.png', format='png', dpi=300)
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
        times = [float(mdata.loc[d, 'time_mean']) if d in mdata.index else 0.0 for d in datasets]
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
        vals = [float(mdata.loc[d, 'rmse_mean']) if d in mdata.index else 0.0 for d in datasets]
        ax2.bar(x + offsets[i], vals, width, label=method, color=_method_color(method),
                edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Dataset', fontweight='bold')
    ax2.set_ylabel('RMSE', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([_dataset_label(d) for d in datasets], ha='center')
    ax2.set_title('(b) Estimation Accuracy', fontweight='bold')
    ax2.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax2.set_axisbelow(True)

    # Single legend below figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend below
    fig.savefig(output_path / 'combined_performance.png', format='png', dpi=300)
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
                print(f"  {m:26s}: {mean:8.5f} ± {std:8.5f} (n={n:2d}, CI±{ci:.5f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True,
                    help="Path to results_raw_*.csv or benchmark_per_run.csv")
    ap.add_argument("--output_dir", type=str, default="figures",
                    help="Directory to save figures (pdf/png/eps)")
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
    print(f"Saved: {output_path / 'estimator_landscape.*'}")

    figure3_rmse_comparison(summary, datasets, methods, output_path)
    print(f"Saved: {output_path / 'rmse_comparison.*'}")

    figure4_speedup_heatmap(summary, datasets, methods, output_path)
    print(f"Saved: {output_path / 'speedup_heatmap.*'}")

    figure5_combined_performance(summary, datasets, methods, output_path)
    print(f"Saved: {output_path / 'combined_performance.*'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
