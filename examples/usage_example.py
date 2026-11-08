"""Run with ``python examples/usage_example.py`` after installing the package."""

import numpy as np

from force import ForceEstimator, StreamingForceEstimator, generate_synthetic_data


def run_simple_demonstration():
    X, clean_correlation = generate_synthetic_data(
        n_samples=1000, n_features=5, contamination=0.1, seed=42
    )
    batch = ForceEstimator()
    batch_correlation = batch.fit(X)

    def source():
        # An application can reopen a file here. The same rows and row order
        # must be produced on each traversal; source buffering is caller-owned.
        for start in range(0, len(X), 37):
            yield X[start : start + 37]

    replay = ForceEstimator()
    replay_correlation = replay.fit_replayable(source, chunk_size=23)
    np.testing.assert_allclose(batch_correlation, replay_correlation, atol=1e-12)
    np.testing.assert_array_equal(batch.accepted_counts_, replay.accepted_counts_)
    print("Final-bound batch/replay correlation:")
    print(np.round(batch_correlation, 3))
    print("Jointly accepted counts:")
    print(batch.accepted_counts_)

    online = StreamingForceEstimator(window_size=100)
    for chunk in source():
        online.update_batch(chunk)
    print("Previous-window online correlation (different acceptance history):")
    print(np.round(online.finalize(), 3))
    print(
        f"Status: {online.status_}; calibration: {online.calibration_count_}; "
        f"estimation rows: {online.estimation_count_}"
    )
    print("Pair statuses:")
    print(online.pair_status_)

    mask = np.triu(np.ones_like(clean_correlation, dtype=bool), k=1)
    error = np.sqrt(np.mean((batch_correlation[mask] - clean_correlation[mask]) ** 2))
    print(f"Batch error against untrimmed clean correlation: {error:.4f}")
    print("This includes trimming-target bias; it is not quantile approximation error.")


if __name__ == "__main__":
    run_simple_demonstration()
