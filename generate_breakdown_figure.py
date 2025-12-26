"""
FORCE Paper - Breakdown Validation Figure
Illustrates how FORCE's breakdown point is validated using the ODDS-satellite dataset.

This figure shows:
1. Left panel: Conceptual illustration of how IQR-based bounds work under normal vs. high contamination
2. Right panel: RMSE vs. contamination rate showing the breakdown threshold

Author: Sooyoung Jang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

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
})

# Colors
COLOR_CLEAN = '#2ecc71'  # Green for clean data
COLOR_CONTAMINATED = '#e74c3c'  # Red for contaminated data
COLOR_FORCE = '#9467bd'  # Purple for FORCE
COLOR_FASTMCD = '#d62728'  # Red for FastMCD
COLOR_BOUND = '#3498db'  # Blue for bounds
COLOR_CORRUPTED_BOUND = '#e67e22'  # Orange for corrupted bounds


def create_breakdown_validation_figure():
    """Create a comprehensive breakdown validation figure."""
    
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)
    
    # =========================================================================
    # Panel A: Normal Operation (Contamination < 25%)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])
    
    # Generate synthetic data for illustration
    np.random.seed(42)
    n_clean = 85
    n_contam = 15  # 15% contamination
    
    # Clean data (normal distribution)
    clean_data = np.random.normal(0, 1, n_clean)
    # Contaminated data (shifted)
    contam_data = np.random.normal(5, 0.5, n_contam)
    
    all_data = np.concatenate([clean_data, contam_data])
    
    # Plot histogram
    bins = np.linspace(-4, 8, 40)
    ax1.hist(clean_data, bins=bins, alpha=0.7, color=COLOR_CLEAN, 
             label='Clean data', edgecolor='white', linewidth=0.5)
    ax1.hist(contam_data, bins=bins, alpha=0.7, color=COLOR_CONTAMINATED,
             label='Outliers (15%)', edgecolor='white', linewidth=0.5)
    
    # Calculate and show IQR-based bounds (on clean data, as intended)
    q25_clean = np.percentile(clean_data, 25)
    q75_clean = np.percentile(clean_data, 75)
    iqr_clean = q75_clean - q25_clean
    median_clean = np.median(clean_data)
    
    lower_bound = median_clean - 3 * iqr_clean / 1.349
    upper_bound = median_clean + 3 * iqr_clean / 1.349
    
    # Show acceptance region
    ax1.axvspan(lower_bound, upper_bound, alpha=0.2, color=COLOR_BOUND,
                label='Acceptance region')
    ax1.axvline(lower_bound, color=COLOR_BOUND, linestyle='--', linewidth=2)
    ax1.axvline(upper_bound, color=COLOR_BOUND, linestyle='--', linewidth=2)
    
    # Mark quartiles
    ax1.axvline(q25_clean, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axvline(q75_clean, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axvline(median_clean, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Annotations
    ax1.annotate('Q₂₅', (q25_clean, ax1.get_ylim()[1]*0.95), ha='center', fontsize=8)
    ax1.annotate('Q₇₅', (q75_clean, ax1.get_ylim()[1]*0.95), ha='center', fontsize=8)
    ax1.annotate('Median', (median_clean, ax1.get_ylim()[1]*0.85), ha='center', fontsize=8)
    
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Normal Operation\n(15% contamination < 25% limit)', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(-4, 8)
    
    # Add text box
    textstr = 'IQR intact\nOutliers excluded\nFORCE works ✓'
    props = dict(boxstyle='round', facecolor=COLOR_GREEN_LIGHT, alpha=0.8, edgecolor=COLOR_CLEAN)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # =========================================================================
    # Panel B: Breakdown (Contamination > 25%)
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])
    
    # Generate data with high contamination (31.7% like satellite)
    np.random.seed(42)
    n_clean_b = 68
    n_contam_b = 32  # 32% contamination (> 25%)
    
    clean_data_b = np.random.normal(0, 1, n_clean_b)
    contam_data_b = np.random.normal(4, 1.5, n_contam_b)
    
    all_data_b = np.concatenate([clean_data_b, contam_data_b])
    
    # Plot histogram
    ax2.hist(clean_data_b, bins=bins, alpha=0.7, color=COLOR_CLEAN,
             label='Clean data', edgecolor='white', linewidth=0.5)
    ax2.hist(contam_data_b, bins=bins, alpha=0.7, color=COLOR_CONTAMINATED,
             label='Outliers (32%)', edgecolor='white', linewidth=0.5)
    
    # Calculate corrupted IQR-based bounds (on ALL data, showing corruption)
    q25_corrupted = np.percentile(all_data_b, 25)
    q75_corrupted = np.percentile(all_data_b, 75)
    iqr_corrupted = q75_corrupted - q25_corrupted
    median_corrupted = np.median(all_data_b)
    
    lower_bound_corrupted = median_corrupted - 3 * iqr_corrupted / 1.349
    upper_bound_corrupted = median_corrupted + 3 * iqr_corrupted / 1.349
    
    # Show corrupted acceptance region (much wider)
    ax2.axvspan(lower_bound_corrupted, upper_bound_corrupted, alpha=0.2, 
                color=COLOR_CORRUPTED_BOUND, label='Corrupted acceptance region')
    ax2.axvline(lower_bound_corrupted, color=COLOR_CORRUPTED_BOUND, 
                linestyle='--', linewidth=2)
    ax2.axvline(upper_bound_corrupted, color=COLOR_CORRUPTED_BOUND,
                linestyle='--', linewidth=2)
    
    # Mark corrupted quartiles
    ax2.axvline(q25_corrupted, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axvline(q75_corrupted, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axvline(median_corrupted, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Annotations
    ax2.annotate('Q₂₅\n(shifted)', (q25_corrupted, ax2.get_ylim()[1]*0.9), 
                 ha='center', fontsize=8, color=COLOR_CORRUPTED_BOUND)
    ax2.annotate('Q₇₅\n(shifted)', (q75_corrupted, ax2.get_ylim()[1]*0.9),
                 ha='center', fontsize=8, color=COLOR_CORRUPTED_BOUND)
    
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) Breakdown Regime\n(32% contamination > 25% limit)', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(-4, 8)
    
    # Add text box
    textstr = 'IQR corrupted\nOutliers admitted\nFORCE fails ✗'
    props = dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8, edgecolor=COLOR_CONTAMINATED)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # =========================================================================
    # Panel C: RMSE vs Contamination Rate
    # =========================================================================
    ax3 = fig.add_subplot(gs[2])
    
    # Data points from experiments
    datasets = {
        'Genomics': (0.01, 0.0014, 0.0000),  # (contamination, FORCE RMSE, FastMCD RMSE)
        'Mammography': (0.023, 0.0151, 0.0096),
        'S&P 500': (0.10, 0.1091, 0.1606),
        'Synthetic': (0.10, 0.3650, 0.0170),
        'Satellite': (0.317, 0.7287, 0.0139),
    }
    
    # Extract data
    contam_rates = [v[0] for v in datasets.values()]
    force_rmse = [v[1] for v in datasets.values()]
    fastmcd_rmse = [v[2] for v in datasets.values()]
    names = list(datasets.keys())
    
    # Plot FORCE results
    ax3.scatter(contam_rates[:-1], force_rmse[:-1], s=120, c=COLOR_FORCE, 
                marker='o', label='FORCE (within limit)', zorder=5, edgecolors='black')
    ax3.scatter([contam_rates[-1]], [force_rmse[-1]], s=200, c=COLOR_CONTAMINATED,
                marker='X', label='FORCE (exceeds limit)', zorder=5, edgecolors='black', linewidths=2)
    
    # Plot FastMCD results
    ax3.scatter(contam_rates, fastmcd_rmse, s=80, c=COLOR_FASTMCD, marker='s',
                label='FastMCD', zorder=4, alpha=0.7, edgecolors='black')
    
    # Add breakdown threshold line
    ax3.axvline(0.25, color='red', linestyle='--', linewidth=2.5, 
                label='FORCE breakdown point (25%)')
    
    # Shade regions
    ax3.axvspan(0, 0.25, alpha=0.1, color=COLOR_CLEAN, label='FORCE operating regime')
    ax3.axvspan(0.25, 0.4, alpha=0.1, color=COLOR_CONTAMINATED)
    
    # Add annotations for each dataset
    offsets = {
        'Genomics': (0.02, 0.03),
        'Mammography': (0.02, 0.03),
        'S&P 500': (-0.03, 0.05),
        'Synthetic': (0.02, 0.03),
        'Satellite': (-0.06, -0.08),
    }
    
    for i, name in enumerate(names):
        ox, oy = offsets[name]
        ax3.annotate(name, (contam_rates[i] + ox, force_rmse[i] + oy),
                     fontsize=8, ha='left' if ox > 0 else 'right')
    
    # Add arrow showing breakdown
    ax3.annotate('', xy=(0.317, 0.7287), xytext=(0.28, 0.5),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.text(0.23, 0.52, 'Breakdown!\nTheory\nvalidated', fontsize=9, 
             color='red', ha='center', fontweight='bold')
    
    ax3.set_xlabel('Contamination Rate', fontweight='bold')
    ax3.set_ylabel('RMSE', fontweight='bold')
    ax3.set_title('(c) RMSE vs. Contamination Rate\nValidation of 25% Breakdown Point', fontweight='bold')
    ax3.set_xlim(0, 0.38)
    ax3.set_ylim(0, 0.85)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Add text annotations for regions
    ax3.text(0.12, 0.78, 'Safe Zone\n(<25%)', ha='center', fontsize=10,
             color=COLOR_CLEAN, fontweight='bold', alpha=0.8)
    ax3.text(0.32, 0.78, 'Breakdown\nZone', ha='center', fontsize=10,
             color=COLOR_CONTAMINATED, fontweight='bold', alpha=0.8)
    
    plt.tight_layout()
    
    return fig


# Color for text boxes
COLOR_GREEN_LIGHT = '#ccffcc'

def create_simple_breakdown_figure():
    """Create a simpler, cleaner breakdown validation figure."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # =========================================================================
    # Panel A: Conceptual Illustration of IQR Corruption
    # =========================================================================
    ax1 = axes[0]
    
    # Create schematic showing IQR shift
    # Normal operation
    clean_center = 0
    clean_iqr = 1.5
    clean_q25 = clean_center - clean_iqr/2
    clean_q75 = clean_center + clean_iqr/2
    clean_bounds = (clean_center - 3*clean_iqr/1.349, clean_center + 3*clean_iqr/1.349)
    
    # Draw clean distribution region
    rect1 = Rectangle((clean_bounds[0], 0.55), clean_bounds[1]-clean_bounds[0], 0.3,
                       facecolor=COLOR_GREEN_LIGHT, edgecolor=COLOR_CLEAN, linewidth=2,
                       label='Acceptance region')
    ax1.add_patch(rect1)
    
    # Mark IQR
    ax1.plot([clean_q25, clean_q75], [0.7, 0.7], 'k-', linewidth=4, label='IQR')
    ax1.plot([clean_q25], [0.7], 'ko', markersize=10)
    ax1.plot([clean_q75], [0.7], 'ko', markersize=10)
    ax1.plot([clean_center], [0.7], 'k^', markersize=12)
    
    # Add outliers (outside bounds - will be rejected)
    outlier_x = [4, 4.5, 5]
    for ox in outlier_x:
        ax1.plot(ox, 0.7, 'x', color=COLOR_CONTAMINATED, markersize=15, 
                 markeredgewidth=3)
    
    # Add label inside the green box (top-left, single line)
    ax1.text(clean_bounds[0] + 0.15, 0.835, 'Normal Operation (Contam. < 25%)',
             ha='left', va='top', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor=COLOR_CLEAN, linewidth=2))

    # Breakdown regime
    # IQR is now shifted/expanded due to contamination
    contam_center = 1.5  # Shifted right
    contam_iqr = 3.0  # Expanded
    contam_q25 = contam_center - contam_iqr/2
    contam_q75 = contam_center + contam_iqr/2
    contam_bounds = (contam_center - 3*contam_iqr/1.349, contam_center + 3*contam_iqr/1.349)

    # Draw corrupted region (much wider)
    rect2 = Rectangle((contam_bounds[0], 0.15), contam_bounds[1]-contam_bounds[0], 0.3,
                       facecolor='#ffdddd', edgecolor=COLOR_CONTAMINATED, linewidth=2,
                       linestyle='--')
    ax1.add_patch(rect2)

    # Mark corrupted IQR
    ax1.plot([contam_q25, contam_q75], [0.3, 0.3], color=COLOR_CORRUPTED_BOUND,
             linewidth=4, linestyle='--')
    ax1.plot([contam_q25], [0.3], 'o', color=COLOR_CORRUPTED_BOUND, markersize=10)
    ax1.plot([contam_q75], [0.3], 'o', color=COLOR_CORRUPTED_BOUND, markersize=10)
    ax1.plot([contam_center], [0.3], '^', color=COLOR_CORRUPTED_BOUND, markersize=12)

    # Add outliers (now inside bounds - will be admitted)
    for ox in outlier_x:
        ax1.plot(ox, 0.3, 'x', color=COLOR_CONTAMINATED, markersize=15,
                 markeredgewidth=3)

    # Add label inside the red box (top-left, single line)
    ax1.text(contam_bounds[0] + 0.15, 0.435, 'Breakdown Regime (Contam. > 25%)',
             ha='left', va='top', fontsize=10, fontweight='bold', color=COLOR_CONTAMINATED,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor=COLOR_CONTAMINATED, linewidth=2))
    
    # Arrows showing expansion
    ax1.annotate('', xy=(contam_q75, 0.45), xytext=(clean_q75, 0.55),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax1.annotate('', xy=(contam_q25, 0.45), xytext=(clean_q25, 0.55),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax1.text(-2.5, 0.5, 'IQR\ncorrupted', ha='center', va='center', fontsize=9, color='gray',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax1.set_xlim(-6, 8)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xlabel('(a) Mechanism of IQR Corruption\nWhen Contamination Exceeds 25%', fontweight='bold', fontsize=11)
    ax1.xaxis.set_label_coords(0.5, -0.12)
    
    # Legend - 3 columns, 2 rows at lower right
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=4, label='IQR (intact)'),
        Line2D([0], [0], color=COLOR_CORRUPTED_BOUND, linewidth=4, linestyle='--',
               label='IQR (corrupted)'),
        Line2D([0], [0], marker='x', color=COLOR_CONTAMINATED, linestyle='None',
               markersize=10, markeredgewidth=3, label='Outliers'),
        mpatches.Patch(facecolor=COLOR_GREEN_LIGHT, edgecolor=COLOR_CLEAN,
                       label='Valid region'),
        mpatches.Patch(facecolor='#ffdddd', edgecolor=COLOR_CONTAMINATED,
                       linestyle='--', label='Corrupted region'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.95, ncol=3)
    
    # =========================================================================
    # Panel B: RMSE vs Contamination with Breakdown Threshold
    # =========================================================================
    ax2 = axes[1]
    
    # Data from experiments
    datasets_ordered = [
        ('Genomics', 0.008, 0.0014),
        ('Mammography', 0.023, 0.0151),
        ('S&P 500', 0.10, 0.1091),
        ('Synthetic', 0.10, 0.3650),
        ('Satellite', 0.317, 0.7287),
    ]
    
    contam = [d[1] for d in datasets_ordered]
    rmse = [d[2] for d in datasets_ordered]
    names = [d[0] for d in datasets_ordered]
    
    # Separate within-limit and exceeds-limit
    within_contam = contam[:-1]
    within_rmse = rmse[:-1]
    exceed_contam = [contam[-1]]
    exceed_rmse = [rmse[-1]]
    
    # Shade regions
    ax2.axvspan(0, 0.25, alpha=0.15, color=COLOR_CLEAN, zorder=1)
    ax2.axvspan(0.25, 0.40, alpha=0.15, color=COLOR_CONTAMINATED, zorder=1)
    
    # Breakdown threshold
    ax2.axvline(0.25, color='red', linestyle='--', linewidth=3, zorder=2,
                label='Breakdown point (25%)')
    
    # Plot points
    ax2.scatter(within_contam, within_rmse, s=150, c=COLOR_FORCE, marker='o',
                edgecolors='black', linewidths=1.5, zorder=5,
                label='FORCE (within limit)')
    ax2.scatter(exceed_contam, exceed_rmse, s=250, c=COLOR_CONTAMINATED, marker='X',
                edgecolors='black', linewidths=2, zorder=5,
                label='FORCE (exceeds limit)')
    
    # Connect with trend line
    ax2.plot(contam, rmse, 'k--', alpha=0.3, linewidth=1, zorder=3)
    
    # Annotate points - Genomics with arrow at fixed position, others with text
    for i, name in enumerate(names):
        fontweight = 'bold' if name == 'Satellite' else 'normal'
        color = COLOR_CONTAMINATED if name == 'Satellite' else 'black'

        if name == 'Genomics':
            # Use arrow for Genomics at fixed position (1%, 0.1)
            ax2.annotate(name, xy=(contam[i], rmse[i]), xytext=(0.01, 0.1),
                        fontsize=9, ha='left', color=color, fontweight=fontweight,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1))
        else:
            # Regular text annotation for others
            if name == 'Mammography':
                dx, dy = 0.015, 0.02
            elif name == 'S&P 500':
                dx, dy = 0.015, 0.02
            elif name == 'Synthetic':
                dx, dy = 0.015, 0.02
            else:  # Satellite
                dx, dy = 0.015, 0.02

            ax2.annotate(name, (contam[i] + dx, rmse[i] + dy), fontsize=9,
                        ha='left', fontweight=fontweight, color=color)
    
    # Add arrow and annotation for breakdown
    ax2.annotate('Theory\nValidated!', xy=(0.317, 0.7287), xytext=(0.35, 0.55),
                 fontsize=11, color='red', fontweight='bold', ha='center',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Region labels - aligned at same height
    ax2.text(0.12, 0.82, 'FORCE Operating Regime', ha='center', fontsize=10,
             color=COLOR_CLEAN, fontweight='bold', alpha=0.9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLOR_CLEAN, linewidth=1.5))
    ax2.text(0.32, 0.82, 'Breakdown Regime', ha='center', fontsize=10,
             color=COLOR_CONTAMINATED, fontweight='bold', alpha=0.9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLOR_CONTAMINATED, linewidth=1.5))
    
    ax2.set_ylabel('RMSE', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, 0.40)
    ax2.set_ylim(0, 0.90)
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.3, zorder=0)
    ax2.set_xlabel('(b) RMSE vs. Contamination Rate\nEmpirical Validation of Breakdown Point', fontweight='bold', fontsize=11)
    ax2.xaxis.set_label_coords(0.5, -0.12)
    
    # Add percentage labels on x-axis
    ax2.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    ax2.set_xticklabels(['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%'])
    
    plt.tight_layout()
    
    return fig


def main():
    """Generate the breakdown validation figure."""
    from pathlib import Path
    
    output_path = Path('figures')
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating breakdown validation figure...")
    
    # Generate the simpler, cleaner version
    fig = create_simple_breakdown_figure()
    
    # Save in multiple formats
    # fig.savefig(output_path / 'breakdown_validation.pdf', format='pdf')
    fig.savefig(output_path / 'breakdown_validation.png', format='png', dpi=300)
    # fig.savefig(output_path / 'breakdown_validation.eps', format='eps')
    
    print(f"Saved: {output_path / 'breakdown_validation.pdf'}")
    
    plt.close()
    
    print("\nFigure generation complete!")


if __name__ == '__main__':
    main()
