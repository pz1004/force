# Reference table utility

`update_latex_tables.py.txt` is retained as source text. It requires these inputs,
which are not included in this repository:

- `benchmark_results/results_raw_20251226_181952.csv`
- `template_revised.tex`

The source text's description of the CSV as "committed" refers to its source
context. The two CSVs under `archive/results/` are different summaries and cannot
replace that input. The `.txt` suffix keeps this incomplete workflow separate
from executable package scripts.

Generate benchmark CSV, JSON and Markdown reports with
[`scripts/run_benchmark.py`](../../scripts/run_benchmark.py), as described in the
[evaluation contract](../../VERIFICATION.md#evaluation-contract). Run the script
with `--help` for available options. The reference utility's
statistical calculations are not used by that command.
