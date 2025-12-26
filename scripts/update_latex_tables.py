"""
Script to update LaTeX tables in template_revised.tex with benchmark results.

This script reads the benchmark results CSV and generates updated LaTeX table content,
then automatically updates the template file.

Tables updated:
1. Table 1 (tab:fastmcd_baseline) - FastMCD baseline performance
2. Table 3 (tab:datasets) - Dataset summary
3. Table 4 (tab:time_results) - Execution time comparison (with TrimmedPearson variants)
4. Table 5 (tab:speedup) - Speedup factors (with TrimmedPearson variants)
5. Table 6 (tab:rmse_results) - RMSE comparison (with TrimmedPearson variants)
6. Table 7 (tab:satellite_validation) - Satellite validation (with TrimmedPearson variants)
7. Table 10 (tab:significance) - Statistical significance (with TrimmedPearson variants)

Header Abbreviations:
- TrimmedPearsonExact(no TER) → TP-Exact
- TrimmedPearsonExact(TER) → TP-TER
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import re

# Load the benchmark results
results_path = Path(__file__).parent.parent / "benchmark_results" / "results_raw_20251226_181952.csv"
df = pd.read_csv(results_path)

print("=" * 80)
print("LATEX TABLE AUTO-UPDATE SCRIPT")
print("=" * 80)
print(f"Input: {results_path}")

# Get unique datasets and algorithms
datasets = df['dataset'].unique()
algorithms = df['algorithm'].unique()

print(f"\nDatasets: {list(datasets)}")
print(f"Algorithms: {list(algorithms)}")

# Compute summary statistics
summary = df.groupby(['dataset', 'algorithm']).agg({
    'time_ms': ['mean', 'std', 'count'],
    'rmse': ['mean', 'std'],
    'n_samples': 'first',
    'n_features': 'first'
}).reset_index()

summary.columns = ['dataset', 'algorithm', 'time_mean', 'time_std', 'n_runs',
                   'rmse_mean', 'rmse_std', 'n_samples', 'n_features']

# Calculate 95% CI
summary['time_ci'] = 1.96 * summary['time_std'] / np.sqrt(summary['n_runs'])
summary['rmse_ci'] = 1.96 * summary['rmse_std'] / np.sqrt(summary['n_runs'])

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset ordering and display names
dataset_order = ['Synthetic', 'SP500', 'ODDS-mammography', 'ODDS-satellite', 'Genomics']
dataset_display = {
    'Synthetic': 'Synthetic',
    'SP500': 'S\\&P 500',
    'ODDS-mammography': 'ODDS-mammography',
    'ODDS-satellite': 'ODDS-satellite',
    'Genomics': 'Genomics'
}

# Algorithm ordering (including TrimmedPearson variants)
algo_order_full = ['Pearson', 'Spearman', 'Winsorized', 'FastMCD',
                   'TrimmedPearsonExact(no TER)', 'TrimmedPearsonExact(TER)', 'FORCE']
algo_order_base = ['Pearson', 'Spearman', 'Winsorized', 'FastMCD', 'FORCE']

# Algorithm display names (abbreviations for wide tables)
algo_display = {
    'Pearson': 'Pearson',
    'Spearman': 'Spearman',
    'Winsorized': 'Winsorized',
    'FastMCD': 'FastMCD',
    'TrimmedPearsonExact(no TER)': 'TP-Exact',
    'TrimmedPearsonExact(TER)': 'TP-TER',
    'FORCE': 'FORCE'
}

# Contamination rates (known from dataset sources)
contamination_rates = {
    'Synthetic': ('10\\%', '\\checkmark Yes ($<$25\\%)'),
    'SP500': ('$\\sim$10\\%', '\\checkmark Yes ($<$25\\%)'),
    'ODDS-mammography': ('2.3\\%', '\\checkmark Yes ($<$25\\%)'),
    'ODDS-satellite': ('31.7\\%', '$\\times$ \\textbf{No ($>$25\\%)}'),
    'Genomics': ('$<$1\\%', '\\checkmark Yes ($<$25\\%)')
}

contam_display = {
    'Synthetic': '10\\% \\checkmark',
    'SP500': '$\\sim$10\\% \\checkmark',
    'ODDS-mammography': '2.3\\% \\checkmark',
    'ODDS-satellite': '31.7\\% $\\times$',
    'Genomics': '$<$1\\% \\checkmark'
}

domain_map = {
    'Synthetic': 'Simulation',
    'SP500': 'Finance',
    'ODDS-mammography': 'Medical',
    'ODDS-satellite': 'Remote sensing',
    'Genomics': 'Genomics'
}

# Get dataset info
dataset_info = df.groupby('dataset').agg({
    'n_samples': 'first',
    'n_features': 'first'
}).reset_index()

# ============================================================================
# TABLE GENERATION FUNCTIONS
# ============================================================================

def generate_table1():
    """Table 1: FastMCD Baseline Performance"""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\caption{Baseline performance of the FastMCD algorithm across benchmark datasets. The execution times, measured in milliseconds, illustrate the computational bottleneck that precludes real-time deployment. Results represent the mean $\\pm$ standard deviation over 20 independent runs.\\label{tab:fastmcd_baseline}}")
    lines.append("\\begin{tabularx}{\\textwidth}{lCC}")
    lines.append("\\toprule")
    lines.append("\\textbf{Dataset} & \\textbf{Time (ms)} & \\textbf{RMSE} \\\\")
    lines.append("\\midrule")

    fastmcd_data = summary[summary['algorithm'] == 'FastMCD']
    for ds in dataset_order:
        row = fastmcd_data[fastmcd_data['dataset'] == ds]
        if len(row) > 0:
            row = row.iloc[0]
            display_name = dataset_display.get(ds, ds)
            lines.append(f"{display_name} & ${row['time_mean']:.2f} \\pm {row['time_std']:.2f}$ & ${row['rmse_mean']:.4f}$ \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_table3():
    """Table 3: Dataset Summary"""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\caption{Summary of benchmark datasets used for evaluation. The ``Within FORCE Limit?'' column indicates whether contamination is below FORCE's theoretical 25\\% breakdown point. The ODDS-satellite dataset is deliberately included to validate the breakdown point prediction.\\label{tab:datasets}}")
    lines.append("\\begin{tabularx}{\\textwidth}{lccCCl}")
    lines.append("\\toprule")
    lines.append("\\textbf{Dataset} & \\textbf{$N$} & \\textbf{$p$} &")
    lines.append("\\textbf{Contamination} & \\textbf{Within FORCE Limit?} & \\textbf{Domain} \\\\")
    lines.append("\\midrule")

    for ds in dataset_order:
        row = dataset_info[dataset_info['dataset'] == ds]
        if len(row) > 0:
            row = row.iloc[0]
            n_samples = int(row['n_samples'])
            n_features = int(row['n_features'])
            display_name = dataset_display.get(ds, ds)
            contam, within_limit = contamination_rates.get(ds, ('--', '--'))
            domain = domain_map.get(ds, '--')
            n_samples_fmt = f"{n_samples:,}"
            lines.append(f"{display_name} & {n_samples_fmt} & {n_features} & {contam} & {within_limit} & {domain} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_table4():
    """Table 4: Execution Time Comparison (with TrimmedPearson variants)"""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\caption{Execution time comparison (milliseconds) across five benchmark datasets. Values represent mean $\\pm$ standard deviation over 20 independent runs. The 95\\% confidence intervals are shown in brackets below each estimate. TP-Exact = TrimmedPearsonExact without TER; TP-TER = TrimmedPearsonExact with TER optimization.\\label{tab:time_results}}")
    lines.append("\\begin{adjustwidth}{-\\extralength}{0cm}")
    lines.append("\\begin{tabularx}{\\fulllength}{lCCCCCCC}")
    lines.append("\\toprule")

    # Header with abbreviated names
    header_parts = ["\\textbf{Dataset}"]
    for algo in algo_order_full:
        header_parts.append(f"\\textbf{{{algo_display[algo]}}}")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    for ds in dataset_order:
        ds_data = summary[summary['dataset'] == ds]
        display_name = dataset_display.get(ds, ds)

        row_vals = [display_name]
        row_cis = [""]
        for algo in algo_order_full:
            algo_row = ds_data[ds_data['algorithm'] == algo]
            if len(algo_row) > 0:
                algo_row = algo_row.iloc[0]
                row_vals.append(f"${algo_row['time_mean']:.2f} \\pm {algo_row['time_std']:.2f}$")
                ci_low = algo_row['time_mean'] - algo_row['time_ci']
                ci_high = algo_row['time_mean'] + algo_row['time_ci']
                row_cis.append(f"$[{ci_low:.2f}, {ci_high:.2f}]$")
            else:
                row_vals.append("--")
                row_cis.append("--")

        lines.append(" & ".join(row_vals) + " \\\\")
        lines.append(" & ".join(row_cis) + " \\\\")
        lines.append("\\midrule")

    # Remove last \midrule and add \bottomrule
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabularx}")
    lines.append("\\end{adjustwidth}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_table5():
    """Table 5: Speedup Factors (with TrimmedPearson variants)"""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\caption{Speedup factors achieved by FORCE relative to baseline algorithms. Values greater than 1.0 indicate FORCE is faster; values less than 1.0 indicate FORCE is slower. TP-Exact = TrimmedPearsonExact without TER; TP-TER = TrimmedPearsonExact with TER.\\label{tab:speedup}}")
    lines.append("\\begin{adjustwidth}{-\\extralength}{0cm}")
    lines.append("\\begin{tabularx}{\\fulllength}{lCCCCCC}")
    lines.append("\\toprule")

    # Speedup is vs all except FORCE
    speedup_algos = ['Pearson', 'Spearman', 'Winsorized', 'FastMCD',
                     'TrimmedPearsonExact(no TER)', 'TrimmedPearsonExact(TER)']

    header_parts = ["\\textbf{Dataset}"]
    for algo in speedup_algos:
        header_parts.append(f"\\textbf{{vs. {algo_display[algo]}}}")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    speedup_data = []
    for ds in dataset_order:
        ds_data = summary[summary['dataset'] == ds]
        force_time = ds_data[ds_data['algorithm'] == 'FORCE']['time_mean'].values

        if len(force_time) > 0:
            force_time = force_time[0]
            speedups = {}
            for algo in speedup_algos:
                algo_time = ds_data[ds_data['algorithm'] == algo]['time_mean'].values
                if len(algo_time) > 0:
                    speedups[algo] = algo_time[0] / force_time
                else:
                    speedups[algo] = None

            display_name = dataset_display.get(ds, ds)
            speedup_strs = [display_name]
            for algo in speedup_algos:
                if speedups[algo] is not None:
                    speedup_strs.append(f"${speedups[algo]:.2f}\\times$")
                else:
                    speedup_strs.append("--")

            lines.append(" & ".join(speedup_strs) + " \\\\")
            speedup_data.append(speedups)

    # Calculate averages
    if speedup_data:
        lines.append("\\midrule")
        avg_strs = ["\\textbf{Average}"]
        for algo in speedup_algos:
            vals = [s[algo] for s in speedup_data if s[algo] is not None]
            if vals:
                avg_strs.append(f"${np.mean(vals):.2f}\\times$")
            else:
                avg_strs.append("--")
        lines.append(" & ".join(avg_strs) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{adjustwidth}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_table6():
    """Table 6: RMSE Comparison (with TrimmedPearson variants)"""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\caption{RMSE comparison (lower is better) across five benchmark datasets. Best values per dataset are shown in bold. The contamination column indicates the contamination level and whether it is within FORCE's breakdown point (\\checkmark) or exceeds it ($\\times$). TP-Exact = TrimmedPearsonExact without TER; TP-TER = TrimmedPearsonExact with TER.\\label{tab:rmse_results}}")
    lines.append("\\begin{adjustwidth}{-\\extralength}{0cm}")
    lines.append("\\begin{tabularx}{\\fulllength}{lCCCCCCCC}")
    lines.append("\\toprule")

    header_parts = ["\\textbf{Dataset}", "\\textbf{Contam.}"]
    for algo in algo_order_full:
        header_parts.append(f"\\textbf{{{algo_display[algo]}}}")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    for ds in dataset_order:
        ds_data = summary[summary['dataset'] == ds]
        display_name = dataset_display.get(ds, ds)
        contam = contam_display.get(ds, '')

        rmse_vals = {}
        for algo in algo_order_full:
            algo_row = ds_data[ds_data['algorithm'] == algo]
            if len(algo_row) > 0:
                rmse_vals[algo] = algo_row.iloc[0]['rmse_mean']
            else:
                rmse_vals[algo] = None

        # Find minimum RMSE for bolding
        valid_rmses = {k: v for k, v in rmse_vals.items() if v is not None}
        min_rmse = min(valid_rmses.values()) if valid_rmses else None

        rmse_strs = [display_name, contam]
        for algo in algo_order_full:
            if rmse_vals[algo] is not None:
                if abs(rmse_vals[algo] - min_rmse) < 1e-6:
                    rmse_strs.append(f"$\\mathbf{{{rmse_vals[algo]:.4f}}}$")
                else:
                    rmse_strs.append(f"${rmse_vals[algo]:.4f}$")
            else:
                rmse_strs.append("--")

        lines.append(" & ".join(rmse_strs) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{adjustwidth}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_table7():
    """Table 7: Satellite Validation (with TrimmedPearson variants)"""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\caption{Empirical validation using the ODDS-satellite dataset (31.7\\% contamination). This dataset deliberately exceeds FORCE's theoretical 25\\% breakdown point to validate predictions. Algorithms with breakdown points below 31.7\\% show degraded RMSE performance. TP-Exact = TrimmedPearsonExact without TER; TP-TER = TrimmedPearsonExact with TER.\\label{tab:satellite_validation}}")
    lines.append("\\begin{tabularx}{\\textwidth}{lCCCC}")
    lines.append("\\toprule")
    lines.append("\\textbf{Algorithm} & \\textbf{RMSE} & \\textbf{Time (ms)} & \\textbf{Breakdown Point} & \\textbf{Status at 31.7\\%} \\\\")
    lines.append("\\midrule")

    satellite_data = summary[summary['dataset'] == 'ODDS-satellite']

    breakdown_points = {
        'Pearson': '0\\%',
        'Spearman': '$\\approx 0\\%$',
        'Winsorized': '$\\sim$10\\%',
        'TrimmedPearsonExact(no TER)': '25\\%',
        'TrimmedPearsonExact(TER)': '25\\%',
        'FORCE': '25\\%',
        'FastMCD': '$\\approx 50\\%$'
    }

    status_31 = {
        'Pearson': 'Exceeded',
        'Spearman': 'Exceeded',
        'Winsorized': 'Exceeded',
        'TrimmedPearsonExact(no TER)': '\\textbf{Exceeded}',
        'TrimmedPearsonExact(TER)': '\\textbf{Exceeded}',
        'FORCE': '\\textbf{Exceeded}',
        'FastMCD': '\\checkmark Within limit'
    }

    # Order for this table
    table7_order = ['Pearson', 'Spearman', 'Winsorized',
                    'TrimmedPearsonExact(no TER)', 'TrimmedPearsonExact(TER)',
                    'FORCE', 'FastMCD']

    # Get all RMSE values to find minimum
    all_rmse = {}
    for algo in table7_order:
        algo_row = satellite_data[satellite_data['algorithm'] == algo]
        if len(algo_row) > 0:
            all_rmse[algo] = algo_row.iloc[0]['rmse_mean']
    min_rmse = min(all_rmse.values()) if all_rmse else None

    for algo in table7_order:
        algo_row = satellite_data[satellite_data['algorithm'] == algo]
        if len(algo_row) > 0:
            row = algo_row.iloc[0]
            bp = breakdown_points.get(algo, '--')
            st = status_31.get(algo, '--')

            # Bold minimum RMSE
            if abs(row['rmse_mean'] - min_rmse) < 1e-6:
                rmse_str = f"$\\mathbf{{{row['rmse_mean']:.4f}}}$"
            else:
                rmse_str = f"${row['rmse_mean']:.4f}$"

            # Use abbreviated name
            algo_name = algo_display.get(algo, algo)
            lines.append(f"{algo_name} & {rmse_str} & ${row['time_mean']:.2f}$ & {bp} & {st} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_table10():
    """Table 10: Statistical Significance (with TrimmedPearson variants)"""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\caption{Statistical significance of execution time differences (paired $t$-test $p$-values). Comparisons show whether FORCE execution time differs significantly from each baseline. TP-Exact = TrimmedPearsonExact without TER; TP-TER = TrimmedPearsonExact with TER.\\label{tab:significance}}")
    lines.append("\\begin{adjustwidth}{-\\extralength}{0cm}")
    lines.append("\\begin{tabularx}{\\fulllength}{lCCCCCC}")
    lines.append("\\toprule")

    sig_algos = ['Pearson', 'Spearman', 'Winsorized', 'FastMCD',
                 'TrimmedPearsonExact(no TER)', 'TrimmedPearsonExact(TER)']

    header_parts = ["\\textbf{Dataset}"]
    for algo in sig_algos:
        header_parts.append(f"\\textbf{{vs. {algo_display[algo]}}}")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    for ds in dataset_order:
        ds_df = df[df['dataset'] == ds]
        force_times = ds_df[ds_df['algorithm'] == 'FORCE']['time_ms'].values

        display_name = dataset_display.get(ds, ds)
        p_strs = [display_name]

        for algo in sig_algos:
            algo_times = ds_df[ds_df['algorithm'] == algo]['time_ms'].values

            if len(force_times) > 0 and len(algo_times) > 0 and len(force_times) == len(algo_times):
                _, p_value = stats.ttest_rel(force_times, algo_times)

                if p_value < 0.001:
                    p_str = "$< 0.001$"
                elif p_value < 0.01:
                    p_str = "$< 0.01$"
                elif p_value < 0.05:
                    p_str = f"${p_value:.3f}$"
                else:
                    p_str = f"${p_value:.3f}$"

                p_strs.append(p_str)
            else:
                p_strs.append("--")

        lines.append(" & ".join(p_strs) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{adjustwidth}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ============================================================================
# AUTO-UPDATE TEMPLATE FILE
# ============================================================================

def update_template():
    """Update template_revised.tex with generated tables."""
    template_path = Path(__file__).parent.parent / "template_revised.tex"

    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        return False

    with open(template_path, 'r') as f:
        content = f.read()

    original_content = content
    updates_made = []

    # Table patterns: match from \begin{table} to \end{table} containing specific label
    def replace_table(content, label, new_table):
        # Find the label position first
        label_pattern = r'\\label\{' + re.escape(label) + r'\}'
        label_match = re.search(label_pattern, content)

        if not label_match:
            return content, False

        # Find the \begin{table} before this label (search backwards)
        label_pos = label_match.start()
        # Look for the nearest \begin{table} before the label
        before_label = content[:label_pos]
        begin_matches = list(re.finditer(r'\\begin\{table\}\[[Hh!]*\]', before_label))
        if not begin_matches:
            return content, False
        table_start = begin_matches[-1].start()  # Last match before label

        # Find the \end{table} after the label
        after_label_start = label_match.end()
        end_match = re.search(r'\\end\{table\}', content[after_label_start:])
        if not end_match:
            return content, False
        table_end = after_label_start + end_match.end()

        # Replace the table
        content = content[:table_start] + new_table + content[table_end:]
        return content, True

    # Update each table
    tables = [
        ('tab:fastmcd_baseline', generate_table1()),
        ('tab:datasets', generate_table3()),
        ('tab:time_results', generate_table4()),
        ('tab:speedup', generate_table5()),
        ('tab:rmse_results', generate_table6()),
        ('tab:satellite_validation', generate_table7()),
        ('tab:significance', generate_table10()),
    ]

    for label, new_table in tables:
        content, updated = replace_table(content, label, new_table)
        if updated:
            updates_made.append(label)
            print(f"  ✓ Updated {label}")
        else:
            print(f"  ! Could not find {label}")

    # Write updated content
    if content != original_content:
        with open(template_path, 'w') as f:
            f.write(content)
        print(f"\n✓ Successfully updated {len(updates_made)} tables in {template_path}")
        return True
    else:
        print("\n! No changes made to template")
        return False


# ============================================================================
# PRINT GENERATED TABLES AND UPDATE INLINE TEXT VALUES
# ============================================================================

def print_inline_text_updates():
    """Print values needed for inline text updates."""
    print("\n" + "=" * 80)
    print("INLINE TEXT VALUES FOR MANUAL UPDATE")
    print("=" * 80)

    # Calculate key statistics
    force_summary = summary[summary['algorithm'] == 'FORCE']
    fastmcd_summary = summary[summary['algorithm'] == 'FastMCD']

    # FastMCD speedups
    print("\n--- FastMCD Speedup Factors ---")
    for ds in dataset_order:
        force_time = force_summary[force_summary['dataset'] == ds]['time_mean'].values
        fastmcd_time = fastmcd_summary[fastmcd_summary['dataset'] == ds]['time_mean'].values
        if len(force_time) > 0 and len(fastmcd_time) > 0:
            speedup = fastmcd_time[0] / force_time[0]
            print(f"  {ds}: {speedup:.0f}×")

    # Average speedups
    print("\n--- Average Speedups ---")
    speedup_algos = ['Pearson', 'Spearman', 'Winsorized', 'FastMCD',
                     'TrimmedPearsonExact(no TER)', 'TrimmedPearsonExact(TER)']
    for algo in speedup_algos:
        speedups = []
        for ds in dataset_order:
            force_time = force_summary[force_summary['dataset'] == ds]['time_mean'].values
            algo_data = summary[(summary['dataset'] == ds) & (summary['algorithm'] == algo)]
            if len(force_time) > 0 and len(algo_data) > 0:
                speedup = algo_data.iloc[0]['time_mean'] / force_time[0]
                speedups.append(speedup)
        if speedups:
            print(f"  vs. {algo_display.get(algo, algo)}: avg {np.mean(speedups):.2f}×, range [{min(speedups):.2f}×, {max(speedups):.2f}×]")

    # FORCE execution times
    print("\n--- FORCE Execution Times ---")
    for ds in dataset_order:
        row = force_summary[force_summary['dataset'] == ds]
        if len(row) > 0:
            row = row.iloc[0]
            print(f"  {ds}: {row['time_mean']:.2f} ± {row['time_std']:.2f} ms")

    # RMSE comparison summary
    print("\n--- RMSE Summary by Dataset ---")
    for ds in dataset_order:
        ds_data = summary[summary['dataset'] == ds]
        print(f"\n  {ds}:")
        for algo in algo_order_full:
            algo_row = ds_data[ds_data['algorithm'] == algo]
            if len(algo_row) > 0:
                rmse = algo_row.iloc[0]['rmse_mean']
                print(f"    {algo_display.get(algo, algo):12s}: {rmse:.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GENERATING TABLES")
    print("=" * 80)

    # Print each generated table
    print("\n--- Table 1: FastMCD Baseline ---")
    print(generate_table1())

    print("\n--- Table 3: Dataset Summary ---")
    print(generate_table3())

    print("\n--- Table 4: Execution Time Comparison ---")
    print(generate_table4())

    print("\n--- Table 5: Speedup Factors ---")
    print(generate_table5())

    print("\n--- Table 6: RMSE Comparison ---")
    print(generate_table6())

    print("\n--- Table 7: Satellite Validation ---")
    print(generate_table7())

    print("\n--- Table 10: Statistical Significance ---")
    print(generate_table10())

    # Print inline text values
    print_inline_text_updates()

    # Update template file
    print("\n" + "=" * 80)
    print("UPDATING TEMPLATE FILE")
    print("=" * 80)
    update_template()

    # Save summary statistics
    output_dir = Path(__file__).parent.parent / "benchmark_results"
    summary.to_csv(output_dir / "summary_statistics.csv", index=False)
    print(f"\nSummary statistics saved to: {output_dir / 'summary_statistics.csv'}")
