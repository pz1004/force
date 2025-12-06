"""
FORCE Paper - Figure Generation Script
Generates publication-quality figures for MDPI Mathematics journal

Figures:
1. Execution time comparison (log scale bar chart)
2. Estimator landscape (speed-robustness plane)
3. RMSE comparison (grouped bar chart)

Author: Sooyoung Jang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path

# Set publication-quality defaults
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

# Color palette - professional and colorblind-friendly
COLORS = {
    'Pearson': '#1f77b4',      # Blue
    'Spearman': '#ff7f0e',     # Orange
    'Winsorized': '#2ca02c',   # Green
    'FastMCD': '#d62728',      # Red
    'FORCE': '#9467bd',        # Purple
}

# Algorithm categories for grouping
ALGO_CATEGORIES = {
    'Pearson': 'Moment-based',
    'FORCE': 'Moment-based',
    'Spearman': 'Rank-based',
    'Winsorized': 'Rank-based',
    'FastMCD': 'High-breakdown',
}

# Algorithm order for consistent plotting
ALGO_ORDER = ['Pearson', 'Spearman', 'Winsorized', 'FastMCD', 'FORCE']

# Dataset order
DATASET_ORDER = ['Synthetic', 'SP500', 'ODDS-mammography', 'ODDS-satellite']

# Dataset display names
DATASET_NAMES = {
    'Synthetic': 'Synthetic',
    'SP500': 'S&P 500',
    'ODDS-mammography': 'Mammography',
    'ODDS-satellite': 'Satellite'
}


def load_and_process_data(filepath):
    """Load raw results and compute summary statistics."""
    df = pd.read_csv(filepath)
    
    # Compute summary statistics per dataset-algorithm combination
    summary = df.groupby(['dataset', 'algorithm']).agg({
        'time_ms': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['dataset', 'algorithm', 'time_mean', 'time_std', 'n_runs', 
                       'rmse_mean', 'rmse_std']
    
    # Compute 95% CI
    summary['time_ci'] = 1.96 * summary['time_std'] / np.sqrt(summary['n_runs'])
    summary['rmse_ci'] = 1.96 * summary['rmse_std'] / np.sqrt(summary['n_runs'])
    
    return df, summary


def figure1_time_comparison(summary, output_path):
    """
    Figure 1: Execution time comparison across algorithms and datasets (log scale).
    
    Grouped bar chart showing execution times with error bars.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Prepare data
    datasets = DATASET_ORDER
    n_datasets = len(datasets)
    n_algorithms = len(ALGO_ORDER)
    
    # Bar positions
    x = np.arange(n_datasets)
    width = 0.15
    offsets = np.array([-2, -1, 0, 1, 2]) * width
    
    # Plot bars for each algorithm
    bars = []
    for i, algo in enumerate(ALGO_ORDER):
        algo_data = summary[summary['algorithm'] == algo].set_index('dataset')
        times = [algo_data.loc[d, 'time_mean'] if d in algo_data.index else 0 
                 for d in datasets]
        errors = [algo_data.loc[d, 'time_ci'] if d in algo_data.index else 0 
                  for d in datasets]
        
        bar = ax.bar(x + offsets[i], times, width, 
                     label=algo, color=COLORS[algo],
                     yerr=errors, capsize=2, 
                     error_kw={'linewidth': 0.8, 'capthick': 0.8},
                     edgecolor='black', linewidth=0.5)
        bars.append(bar)
    
    # Formatting
    ax.set_yscale('log')
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES[d] for d in datasets])
    
    # Y-axis formatting
    ax.set_ylim(0.1, 2000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
    
    # Add horizontal lines for performance tiers
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=10, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=1000, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Add tier annotations
    # ax.text(4.7, 0.3, 'Sub-ms', fontsize=8, color='gray', ha='right', style='italic')
    # ax.text(4.7, 3, '1-10 ms', fontsize=8, color='gray', ha='right', style='italic')
    # ax.text(4.7, 30, '10-100 ms', fontsize=8, color='gray', ha='right', style='italic')
    # ax.text(4.7, 500, '100+ ms', fontsize=8, color='gray', ha='right', style='italic')
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.95, ncol=5, 
              bbox_to_anchor=(0, 1.02, 1, 0.1), mode='expand')
    
    # Grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save in multiple formats
    fig.savefig(output_path / 'time_comparison.pdf', format='pdf')
    fig.savefig(output_path / 'time_comparison.png', format='png', dpi=300)
    fig.savefig(output_path / 'time_comparison.eps', format='eps')
    
    plt.close()
    print(f"Saved: {output_path / 'time_comparison.pdf'}")


def figure2_estimator_landscape(summary, output_path):
    """
    Figure 2: Position of correlation estimators in the speed-robustness plane.
    
    Scatter plot showing algorithms positioned by speed (x-axis) and robustness (y-axis).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute average execution time across datasets for each algorithm
    avg_times = summary.groupby('algorithm')['time_mean'].mean()
    
    # Define breakdown points (theoretical values)
    breakdown_points = {
        'Pearson': 0,
        'Spearman': 0,  # Technically ~0 for correlation estimation
        'Winsorized': 10,  # Depends on winsorization level (5th/95th percentile)
        'FastMCD': 50,
        'FORCE': 25
    }
    
    # Plot each algorithm
    for algo in ALGO_ORDER:
        x = avg_times[algo]
        y = breakdown_points[algo]
        
        # Different marker sizes and styles
        marker_size = 300 if algo == 'FORCE' else 200
        marker = 's' if algo == 'FORCE' else 'o'
        edgecolor = 'black' if algo == 'FORCE' else 'white'
        linewidth = 2 if algo == 'FORCE' else 1
        
        ax.scatter(x, y, s=marker_size, c=COLORS[algo], marker=marker,
                   label=algo, edgecolors=edgecolor, linewidths=linewidth,
                   zorder=5)
        
        # Add algorithm labels
        offset_x = 1.3 if algo != 'FastMCD' else 0.8
        offset_y = 3 if algo not in ['Pearson', 'Spearman'] else -4
        ha = 'left' if algo != 'FastMCD' else 'right'
        
        ax.annotate(algo, (x, y), 
                    xytext=(x * offset_x, y + offset_y),
                    fontsize=10, fontweight='bold' if algo == 'FORCE' else 'normal',
                    ha=ha, va='center')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Average Execution Time (ms)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Breakdown Point (%)', fontweight='bold', fontsize=11)
    
    # Axis limits
    ax.set_xlim(0.05, 3000)
    ax.set_ylim(-5, 60)
    
    # Add quadrant shading
    # Fast-Fragile quadrant (bottom-left)
    rect1 = plt.Rectangle((0.05, -5), 10-0.05, 15, 
                           facecolor='#ffcccc', alpha=0.3, zorder=1)
    ax.add_patch(rect1)
    ax.text(0.3, 2, 'Fast-Fragile', fontsize=9, color='#cc0000', 
            style='italic', alpha=0.8)
    
    # Slow-Robust quadrant (top-right)
    rect2 = plt.Rectangle((10, 15), 3000-10, 60-15, 
                           facecolor='#ccccff', alpha=0.3, zorder=1)
    ax.add_patch(rect2)
    ax.text(100, 55, 'Slow-Robust', fontsize=9, color='#0000cc', 
            style='italic', alpha=0.8)
    
    # Fast-Robust quadrant (top-left) - highlighted
    rect3 = plt.Rectangle((0.05, 15), 10-0.05, 60-15, 
                           facecolor='#ccffcc', alpha=0.4, zorder=1)
    ax.add_patch(rect3)
    ax.text(0.3, 55, 'Fast-Robust\n(FORCE)', fontsize=9, color='#006600', 
            fontweight='bold', alpha=0.9)
    
    # Add threshold lines
    ax.axvline(x=10, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=15, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add annotation for FORCE's unique position
    # ax.annotate('', xy=(0.63, 25), xytext=(5, 40),
    #             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
    #             zorder=6)
    # ax.text(5.5, 42, 'Unique\nposition', fontsize=8, color='green', 
    #         ha='left', va='bottom')
    
    # Grid
    ax.grid(True, linestyle='-', alpha=0.3, zorder=0)
    
    # X-axis tick formatting
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}'))
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path / 'estimator_landscape.pdf', format='pdf')
    fig.savefig(output_path / 'estimator_landscape.png', format='png', dpi=300)
    fig.savefig(output_path / 'estimator_landscape.eps', format='eps')
    
    plt.close()
    print(f"Saved: {output_path / 'estimator_landscape.pdf'}")


def figure3_rmse_comparison(summary, output_path):
    """
    Figure 3: RMSE comparison across algorithms and datasets.
    
    Grouped bar chart showing RMSE values.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Prepare data
    datasets = DATASET_ORDER
    n_datasets = len(datasets)
    
    # Bar positions
    x = np.arange(n_datasets)
    width = 0.15
    offsets = np.array([-2, -1, 0, 1, 2]) * width
    
    # Plot bars for each algorithm
    for i, algo in enumerate(ALGO_ORDER):
        algo_data = summary[summary['algorithm'] == algo].set_index('dataset')
        rmse_vals = [algo_data.loc[d, 'rmse_mean'] if d in algo_data.index else 0 
                     for d in datasets]
        errors = [algo_data.loc[d, 'rmse_ci'] if d in algo_data.index else 0 
                  for d in datasets]
        
        ax.bar(x + offsets[i], rmse_vals, width, 
               label=algo, color=COLORS[algo],
               yerr=errors, capsize=2,
               error_kw={'linewidth': 0.8, 'capthick': 0.8},
               edgecolor='black', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('RMSE (Lower is Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES[d] for d in datasets])
    
    # Y-axis limit
    ax.set_ylim(0, 0.85)
    
    # Add annotation for best performance on S&P 500
    sp500_idx = datasets.index('SP500')
    # ax.annotate('FORCE\nBest', 
    #             xy=(sp500_idx + offsets[4], 0.1091),
    #             xytext=(sp500_idx + offsets[4] + 0.3, 0.25),
    #             fontsize=8, color='#9467bd', fontweight='bold',
    #             arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1),
    #             ha='left')
    
    # Add annotation for breakdown on satellite
    sat_idx = datasets.index('ODDS-satellite')
    # ax.annotate('Breakdown\n(>25% contam.)', 
    #             xy=(sat_idx + offsets[4], 0.73),
    #             xytext=(sat_idx + offsets[4] - 0.5, 0.55),
    #             fontsize=8, color='#d62728',
    #             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1),
    #             ha='center')
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.95, ncol=5,
              bbox_to_anchor=(0, 1.02, 1, 0.1), mode='expand')
    
    # Grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path / 'rmse_comparison.pdf', format='pdf')
    fig.savefig(output_path / 'rmse_comparison.png', format='png', dpi=300)
    fig.savefig(output_path / 'rmse_comparison.eps', format='eps')
    
    plt.close()
    print(f"Saved: {output_path / 'rmse_comparison.pdf'}")


def figure4_speedup_heatmap(summary, output_path):
    """
    Figure 4 (Bonus): Speedup heatmap showing FORCE speedup over other methods.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Compute speedup matrix
    datasets = DATASET_ORDER
    baselines = ['Pearson', 'Spearman', 'Winsorized', 'FastMCD']
    
    speedup_matrix = []
    for dataset in datasets:
        force_time = summary[(summary['dataset'] == dataset) & 
                            (summary['algorithm'] == 'FORCE')]['time_mean'].values[0]
        row = []
        for baseline in baselines:
            baseline_time = summary[(summary['dataset'] == dataset) & 
                                   (summary['algorithm'] == baseline)]['time_mean'].values[0]
            speedup = baseline_time / force_time
            row.append(speedup)
        speedup_matrix.append(row)
    
    speedup_df = pd.DataFrame(speedup_matrix, 
                               index=[DATASET_NAMES[d] for d in datasets],
                               columns=baselines)
    
    # Create heatmap
    # Use log scale for better visualization
    log_speedup = np.log10(speedup_df.values)
    
    im = ax.imshow(log_speedup, cmap='RdYlGn', aspect='auto', 
                   vmin=-1, vmax=3.5)
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(baselines)):
            val = speedup_df.iloc[i, j]
            if val < 1:
                text = f'{val:.2f}×'
                color = 'white'
            elif val < 10:
                text = f'{val:.1f}×'
                color = 'black'
            else:
                text = f'{val:.0f}×'
                color = 'white' if val > 100 else 'black'
            ax.text(j, i, text, ha='center', va='center', 
                    fontsize=10, fontweight='bold', color=color)
    
    # Formatting
    ax.set_xticks(np.arange(len(baselines)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(baselines)
    ax.set_yticklabels([DATASET_NAMES[d] for d in datasets])
    
    ax.set_xlabel('Baseline Algorithm', fontweight='bold')
    ax.set_ylabel('Dataset', fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Speedup (log₁₀ scale)')
    cbar.set_ticks([-1, 0, 1, 2, 3])
    cbar.set_ticklabels(['0.1×', '1×', '10×', '100×', '1000×'])
    
    # Title
    ax.set_title('FORCE Speedup Over Baseline Algorithms', fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path / 'speedup_heatmap.pdf', format='pdf')
    fig.savefig(output_path / 'speedup_heatmap.png', format='png', dpi=300)
    
    plt.close()
    print(f"Saved: {output_path / 'speedup_heatmap.pdf'}")


def figure5_combined_performance(summary, output_path):
    """
    Figure 5 (Bonus): Combined view showing speed vs accuracy trade-off.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: Time comparison
    ax1 = axes[0]
    datasets = DATASET_ORDER
    x = np.arange(len(datasets))
    width = 0.15
    offsets = np.array([-2, -1, 0, 1, 2]) * width
    
    for i, algo in enumerate(ALGO_ORDER):
        algo_data = summary[summary['algorithm'] == algo].set_index('dataset')
        times = [algo_data.loc[d, 'time_mean'] if d in algo_data.index else 0 
                 for d in datasets]
        ax1.bar(x + offsets[i], times, width, label=algo, color=COLORS[algo],
                edgecolor='black', linewidth=0.5)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Dataset', fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([DATASET_NAMES[d] for d in datasets], ha='center')
    ax1.set_ylim(0.1, 2000)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('(a) Computational Efficiency', fontweight='bold')
    ax1.yaxis.grid(True, linestyle='-', alpha=0.3)
    
    # Right panel: RMSE comparison
    ax2 = axes[1]
    
    for i, algo in enumerate(ALGO_ORDER):
        algo_data = summary[summary['algorithm'] == algo].set_index('dataset')
        rmse_vals = [algo_data.loc[d, 'rmse_mean'] if d in algo_data.index else 0 
                     for d in datasets]
        ax2.bar(x + offsets[i], rmse_vals, width, label=algo, color=COLORS[algo],
                edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Dataset', fontweight='bold')
    ax2.set_ylabel('RMSE', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([DATASET_NAMES[d] for d in datasets], ha='center')
    ax2.set_ylim(0, 0.85)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_title('(b) Estimation Accuracy', fontweight='bold')
    ax2.yaxis.grid(True, linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path / 'combined_performance.pdf', format='pdf')
    fig.savefig(output_path / 'combined_performance.png', format='png', dpi=300)
    
    plt.close()
    print(f"Saved: {output_path / 'combined_performance.pdf'}")


def print_summary_statistics(summary):
    """Print summary statistics for verification."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\n--- Execution Time (ms) ---")
    for dataset in DATASET_ORDER:
        print(f"\n{dataset}:")
        ds_data = summary[summary['dataset'] == dataset]
        for algo in ALGO_ORDER:
            algo_data = ds_data[ds_data['algorithm'] == algo]
            if not algo_data.empty:
                mean = algo_data['time_mean'].values[0]
                std = algo_data['time_std'].values[0]
                print(f"  {algo:12s}: {mean:8.2f} ± {std:.2f} ms")
    
    print("\n--- RMSE ---")
    for dataset in DATASET_ORDER:
        print(f"\n{dataset}:")
        ds_data = summary[summary['dataset'] == dataset]
        for algo in ALGO_ORDER:
            algo_data = ds_data[ds_data['algorithm'] == algo]
            if not algo_data.empty:
                mean = algo_data['rmse_mean'].values[0]
                print(f"  {algo:12s}: {mean:.4f}")
    
    print("\n--- FORCE Speedup vs FastMCD ---")
    for dataset in DATASET_ORDER:
        ds_data = summary[summary['dataset'] == dataset]
        force_time = ds_data[ds_data['algorithm'] == 'FORCE']['time_mean'].values[0]
        mcd_time = ds_data[ds_data['algorithm'] == 'FastMCD']['time_mean'].values[0]
        speedup = mcd_time / force_time
        print(f"  {dataset:20s}: {speedup:8.1f}×")


def main():
    """Main function to generate all figures."""
    # Paths
    input_file = Path('benchmark_results/results_raw_20251201_153759.csv')
    output_path = Path('figures')
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("FORCE Paper - Figure Generation")
    print("="*50)
    
    # Load data
    print("\nLoading data...")
    raw_df, summary = load_and_process_data(input_file)
    print(f"Loaded {len(raw_df)} rows, {len(summary)} summary entries")
    
    # Print statistics
    print_summary_statistics(summary)
    
    # Generate figures
    print("\n" + "="*50)
    print("Generating figures...")
    print("="*50)
    
    print("\n1. Time comparison (log scale bar chart)...")
    figure1_time_comparison(summary, output_path)
    
    print("\n2. Estimator landscape (speed-robustness plane)...")
    figure2_estimator_landscape(summary, output_path)
    
    print("\n3. RMSE comparison (grouped bar chart)...")
    figure3_rmse_comparison(summary, output_path)
    
    print("\n4. Speedup heatmap (bonus)...")
    figure4_speedup_heatmap(summary, output_path)
    
    print("\n5. Combined performance view (bonus)...")
    figure5_combined_performance(summary, output_path)
    
    print("\n" + "="*50)
    print("All figures generated successfully!")
    print(f"Output directory: {output_path}")
    print("="*50)
    
    return summary


if __name__ == '__main__':
    summary = main()
