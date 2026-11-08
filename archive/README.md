# Reference materials

This directory holds figures, tables, a table-generation utility and development
records with their provenance. The numerical artifacts are retained reference
material; they are not outputs of the validation checks in
[VERIFICATION.md](../VERIFICATION.md).

| Location | Contents | Interpretation |
| --- | --- | --- |
| [paper-figures/](paper-figures/) | Six PNG figures | Archived presentation and numerical claims, subject to the qualification below. |
| [results/](results/) | Two S&P volatility-sensitivity CSV tables dated December 2025 in their filenames | The tables do not establish the ticker manifest, downloaded data or execution environment. |
| [tools/](tools/) | `update_latex_tables.py.txt` | Source text whose required inputs are absent. |
| [development/](development/) | Implementation review records and work plans | Dated engineering evidence, separate from the supported API documentation. |

The `paper-figures/breakdown_validation.png` image says "Theory Validated!"
and marks a 25% breakdown point. These statements do not establish that guarantee
for the default uncapped tail expansion ratio. One-sided contamination can
expand the acceptance bounds before the median and IQR break down. The image
is a reference artifact, not evidence for a universal robustness guarantee.
The [statistical interpretation](../VERIFICATION.md#statistical-interpretation)
explains this qualification.

Measured figures can be generated with
[`generate_breakdown_figure.py`](../generate_breakdown_figure.py) from diagnostic
reports and [`generate_figures.py`](../generate_figures.py) from benchmark CSVs.
Run either command with `--help` for input and output options; default outputs
go under `generated_figures/diagnostics/` and `generated_figures/benchmarks/`,
respectively.

## Provenance

[MANIFEST.json](MANIFEST.json) records source and archive paths, byte counts and
SHA-256 digests for the nine figure, table and utility files. Its commit field
identifies a provenance source; it is not required for installation or execution.
The PNG and CSV bytes match that source. The utility has its own recorded digest.
[Development records](development/README.md) have a separate checksum inventory.

These inventories establish file identity. They do not recover missing data or
establish generation commands, software versions or timestamps. Dates embedded
in filenames are source labels. The numerical artifacts have not been regenerated.
