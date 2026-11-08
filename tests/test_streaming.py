"""Behavioral checks for the explicitly delayed-window streaming estimator."""

import warnings

import numpy as np
import pytest

from force import ForceEstimator
from force.streaming import StreamingForceEstimator

from .test_core import _independent_p2


pytestmark = pytest.mark.filterwarnings("ignore:window_size < 50:UserWarning")


def _window_bounds(window, lam, use_ter, cap, epsilon):
    """Independent P² reference followed by direct paper bound equations."""
    quantiles = np.array(
        [
            [_independent_p2(window[:, j], probability) for j in range(window.shape[1])]
            for probability in (0.01, 0.25, 0.50, 0.75, 0.99)
        ]
    )
    q01, q25, q50, q75, q99 = quantiles
    scale = np.maximum(0.0, (q75 - q25) / 1.349)
    ter = np.ones(window.shape[1])
    if use_ter:
        distance = abs(q50 - q01)
        ter = np.maximum(1.0, abs(q99 - q50) / (distance + epsilon))
        ter[distance < epsilon] = 1.0
        if cap is not None:
            ter = np.minimum(ter, cap)
    return q50 - lam * ter * scale, q50 + lam * ter * scale


def _offline_reference(X, window_size, lam=3.0, use_ter=True, cap=None, epsilon=1e-10):
    """Retain rows and historical masks solely in this test-only oracle."""
    p = X.shape[1]
    retained_rows, masks = [], []
    for start in range(window_size, len(X), window_size):
        lower, upper = _window_bounds(
            X[start - window_size : start], lam, use_ter, cap, epsilon
        )
        block = X[start : start + window_size]
        retained_rows.extend(block)
        masks.extend((block >= lower) & (block <= upper))
    rows = np.asarray(retained_rows, dtype=float).reshape(-1, p)
    accepted = np.asarray(masks, dtype=bool).reshape(-1, p)
    counts = np.zeros((p, p), dtype=np.int64)
    corr = np.eye(p)
    statuses = np.full((p, p), "insufficient_samples", dtype="<U24")
    for left in range(p):
        for right in range(left, p):
            pair = rows[accepted[:, left] & accepted[:, right]][:, [left, right]]
            counts[left, right] = counts[right, left] = len(pair)
            if len(pair) < 2:
                continue
            centered = pair - pair.mean(axis=0)
            if np.any(np.sum(centered**2, axis=0) == 0.0):
                statuses[left, right] = statuses[right, left] = "zero_variance"
                continue
            statuses[left, right] = statuses[right, left] = "ok"
            if left != right:
                value = np.corrcoef(pair, rowvar=False)[0, 1]
                corr[left, right] = corr[right, left] = value
    return corr, counts, statuses


@pytest.mark.parametrize("use_ter,cap", [(False, None), (True, None), (True, 2.0)])
def test_matches_independent_historical_acceptance_oracle(use_ter, cap):
    rng = np.random.default_rng(91)
    X = rng.normal(size=(143, 3))
    X[:, 1] += 0.7 * X[:, 0]
    X[31:49, 0] += 4.0
    X[81:99, 2] -= 3.0
    X[::7] = np.round(X[::7])
    estimator = StreamingForceEstimator(window_size=17, use_ter=use_ter, ter_max=cap)
    estimator.update_batch(X)
    expected, counts, statuses = _offline_reference(X, 17, use_ter=use_ter, cap=cap)
    np.testing.assert_allclose(estimator.finalize(), expected, atol=2e-13)
    np.testing.assert_array_equal(estimator.accepted_counts_, counts)
    np.testing.assert_array_equal(estimator.pair_status_, statuses)
    assert estimator.calibration_count_ == 17
    assert estimator.estimation_count_ == 126
    assert estimator.bound_version_ == 8
    assert estimator.window_count_ == 7


def test_row_and_arbitrary_chunk_calls_have_identical_window_schedule():
    X = np.random.default_rng(8).normal(size=(109, 4))
    complete = StreamingForceEstimator(window_size=13).update_batch(X)
    rowwise = StreamingForceEstimator(window_size=13)
    for row in X:
        assert rowwise.update(row) is rowwise
    chunks = StreamingForceEstimator(window_size=13)
    cuts = [0, 1, 4, 5, 12, 13, 14, 25, 27, 55, 56, 101, len(X)]
    for left, right in zip(cuts[:-1], cuts[1:]):
        assert chunks.update_batch(X[left:right]) is chunks
    for other in (rowwise, chunks):
        np.testing.assert_array_equal(other.quantiles_, complete.quantiles_)
        np.testing.assert_array_equal(other.accepted_counts_, complete.accepted_counts_)
        np.testing.assert_array_equal(other.finalize(), complete.finalize())
        assert other.bound_version_ == complete.bound_version_
        assert other.window_count_ == complete.window_count_


def test_calibration_boundary_is_excluded_and_snapshots_do_not_reset():
    X = np.random.default_rng(4).normal(size=(27, 2))
    estimator = StreamingForceEstimator(window_size=11)
    assert estimator.status_ == "calibrating"
    assert estimator.finalize() is None
    estimator.update_batch(X[:10])
    assert estimator.finalize() is None
    assert estimator.bound_version_ == 0
    estimator.update(X[10])
    assert estimator.status_ == "calibrated_no_estimates"
    assert estimator.finalize() is None
    assert estimator.bound_version_ == 1
    assert estimator.estimation_count_ == 0
    np.testing.assert_array_equal(estimator.accepted_counts_, np.zeros((2, 2)))
    estimator.update(X[11])
    assert estimator.status_ == "estimating"
    assert estimator.finalize() is not None
    assert estimator.estimation_count_ == 1
    assert estimator.pair_status_[0, 1] == "insufficient_samples"
    estimator.update_batch(X[12:17])
    snapshot = estimator.finalize()
    counts = estimator.accepted_counts_
    np.testing.assert_array_equal(snapshot, estimator.finalize())
    np.testing.assert_array_equal(counts, estimator.accepted_counts_)
    estimator.update_batch(X[17:])
    expected, expected_counts, _ = _offline_reference(X, 11)
    np.testing.assert_allclose(estimator.finalize(), expected, atol=2e-13)
    np.testing.assert_array_equal(estimator.accepted_counts_, expected_counts)


def test_inclusive_ties_use_old_bounds_through_the_last_row_of_each_window():
    estimator = StreamingForceEstimator(window_size=6, use_ter=False)
    estimator.update_batch(np.zeros((6, 2)))
    np.testing.assert_array_equal(estimator.thresholds_, np.zeros((2, 2)))
    estimator.update_batch(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    )
    # The last three rows would be accepted by the next bounds, but not the old ones.
    np.testing.assert_array_equal(estimator.accepted_counts_, [[2, 1], [1, 2]])
    assert estimator.pair_status_[0, 1] == "insufficient_samples"
    assert estimator.pair_status_[0, 0] == "zero_variance"
    assert estimator.bound_version_ == 2
    np.testing.assert_array_equal(estimator.thresholds_, np.ones((2, 2)))
    estimator.update([1.0, 1.0])
    np.testing.assert_array_equal(estimator.accepted_counts_, [[3, 2], [2, 3]])
    np.testing.assert_allclose(estimator.finalize(), np.ones((2, 2)), atol=2e-15)
    assert estimator.pair_status_[0, 1] == "ok"


def test_future_outliers_do_not_revise_existing_acceptance():
    X = np.random.default_rng(12).normal(size=(35, 2))
    estimator = StreamingForceEstimator(window_size=20, use_ter=False).update_batch(X)
    before = estimator.finalize()
    counts = estimator.accepted_counts_
    frozen = estimator.thresholds_
    estimator.update_batch(np.full((4, 2), 1e5))
    np.testing.assert_array_equal(estimator.finalize(), before)
    np.testing.assert_array_equal(estimator.accepted_counts_, counts)
    np.testing.assert_array_equal(estimator.thresholds_, frozen)
    assert estimator.estimation_count_ == 19


def test_online_output_intentionally_differs_from_final_bound_batch():
    X = np.vstack((np.zeros((6, 2)), np.full((6, 2), 10.0)))
    online = StreamingForceEstimator(window_size=6, use_ter=False).update_batch(X)
    batch = ForceEstimator(use_ter=False).fit(X)
    assert online.accepted_counts_[0, 1] == 0
    assert online.finalize()[0, 1] == 0.0
    assert batch[0, 1] == pytest.approx(1.0)


@pytest.mark.parametrize("size", [0, 4, 5, True, 5.0, np.nan])
def test_invalid_window_size(size):
    with pytest.raises(ValueError, match="window_size"):
        StreamingForceEstimator(window_size=size)


@pytest.mark.parametrize("size", [6, 49])
def test_small_windows_warn_about_repeated_quantile_warmup(size):
    with pytest.warns(UserWarning, match="small-sample.*Remark 3"):
        StreamingForceEstimator(window_size=size)


@pytest.mark.parametrize("size", [50, 1000])
def test_windows_above_warning_threshold_do_not_warn(size):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        StreamingForceEstimator(window_size=size)
    assert not caught


@pytest.mark.parametrize(
    "field,value",
    [
        ("window_size", 20),
        ("lambda_scale", 2.0),
        ("use_ter", False),
        ("ter_max", 3.0),
        ("epsilon", 1e-8),
    ],
)
def test_stream_configuration_cannot_change_after_observations(field, value):
    estimator = StreamingForceEstimator(window_size=9).update_batch(
        np.random.default_rng(31).normal(size=(13, 2))
    )
    original = getattr(estimator, field)
    snapshot = estimator.finalize()
    counts = estimator.accepted_counts_
    with pytest.raises(AttributeError):
        setattr(estimator, field, value)
    assert getattr(estimator, field) == original
    assert estimator.status_ == "estimating"
    assert estimator.estimation_count_ == 4
    np.testing.assert_array_equal(estimator.finalize(), snapshot)
    np.testing.assert_array_equal(estimator.accepted_counts_, counts)


def test_no_batch_exact_cutover_option():
    with pytest.raises(TypeError, match="exact_cutover"):
        StreamingForceEstimator(exact_cutover=100)


@pytest.mark.parametrize(
    "bad",
    [
        [[1.0, 2.0], [np.nan, 3.0]],
        [[1.0, 2.0], [3.0, np.inf]],
        [[1.0, 2.0], [3.0, -np.inf]],
        [[1.0, 2.0, 3.0]],
        [[1.0]],
        [1.0, 2.0],
        [[1.0 + 2.0j, 2.0]],
        np.array([[np.complex128(1.0 + 2.0j), 2.0]], dtype=object),
        [["not a number", 2.0]],
    ],
)
def test_invalid_chunk_rejected_before_any_row_is_incorporated(bad):
    estimator = StreamingForceEstimator(window_size=9).update_batch(
        [[1.0, 2.0], [2.0, 1.0]]
    )
    with pytest.raises(ValueError):
        estimator.update_batch(bad)
    assert estimator.n_samples_seen_ == 2
    assert estimator.status_ == "calibrating"
    assert estimator.bound_version_ == 0
    assert estimator.n_features_in_ == 2


def test_update_requires_one_row_and_empty_chunks_do_not_initialize():
    estimator = StreamingForceEstimator(window_size=9)
    estimator.update_batch(np.empty((0, 3)))
    assert estimator.n_features_in_ is None
    assert estimator.state_nbytes_ == 0
    with pytest.raises(ValueError, match="one-dimensional"):
        estimator.update([[1.0, 2.0]])
    estimator.update([1.0, 2.0])
    estimator.update_batch(np.empty((0, 2)))
    assert estimator.n_samples_seen_ == 1
    with pytest.raises(ValueError, match="Feature count"):
        estimator.update_batch(np.empty((0, 3)))


def test_diagnostics_are_copies_and_show_undefined_pairs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        estimator = StreamingForceEstimator(window_size=7).update_batch(
            np.ones((10, 3))
        )
    diagnostics = estimator.get_diagnostics()
    assert diagnostics["status"] == "estimating"
    assert diagnostics["calibration_count"] == 7
    assert diagnostics["estimation_count"] == 3
    assert diagnostics["bound_version"] == 1
    assert diagnostics["bound_status"]["degenerate_lower_tail"].all()
    assert diagnostics["bound_status"]["zero_iqr"].all()
    assert not diagnostics["bound_status"]["crossed_quartiles"].any()
    assert "pair_status" not in diagnostics
    assert diagnostics["pair_status_codes"].dtype == np.int8
    assert np.all(diagnostics["pair_status_codes"] == 2)
    assert diagnostics["pair_status_code_labels"][2] == "zero_variance"
    labeled = estimator.get_diagnostics(include_status_labels=True)
    assert np.all(labeled["pair_status"] == "zero_variance")
    diagnostics["accepted_counts"][:] = 123
    diagnostics["pair_status_codes"][:] = 123
    labeled["pair_status"][:] = "changed"
    diagnostics["thresholds"][0][:] = -123
    diagnostics["quantiles"][:] = 123
    diagnostics["bound_status"]["zero_iqr"][:] = False
    snapshot = estimator.finalize()
    snapshot[:] = 123
    np.testing.assert_array_equal(estimator.accepted_counts_, np.full((3, 3), 3))
    np.testing.assert_array_equal(estimator.thresholds_[0], np.ones(3))
    assert estimator.bound_status_["zero_iqr"].all()
    assert np.all(estimator.pair_status_codes_ == 2)
    assert np.all(estimator.pair_status_ == "zero_variance")
    np.testing.assert_array_equal(estimator.finalize(), np.eye(3))


@pytest.mark.parametrize(
    "requested,blocked",
    [
        ("finalize", ("finalize", "accepted_counts", "status_codes", "status_labels")),
        (
            "accepted_counts_",
            ("finalize", "correlation", "status_codes", "status_labels"),
        ),
        (
            "pair_status_codes_",
            ("finalize", "correlation", "accepted_counts", "status_labels"),
        ),
    ],
)
def test_pair_snapshots_only_extract_the_requested_array(
    requested, blocked, monkeypatch
):
    estimator = StreamingForceEstimator(window_size=50, use_ter=False).update_batch(
        np.random.default_rng(82).normal(size=(70, 3))
    )

    def unnecessary(*_args, **_kwargs):
        raise AssertionError("Unrequested diagnostic matrix was constructed")

    for name in blocked:
        monkeypatch.setattr(estimator._pair_moments, name, unnecessary)
    result = (
        estimator.finalize()
        if requested == "finalize"
        else getattr(estimator, requested)
    )
    assert result.shape == (3, 3)


def test_default_diagnostics_do_not_compute_correlation_or_status_labels(monkeypatch):
    estimator = StreamingForceEstimator(window_size=50).update_batch(
        np.random.default_rng(83).normal(size=(70, 3))
    )

    def unnecessary(*_args, **_kwargs):
        raise AssertionError("Unrequested diagnostic matrix was constructed")

    for name in ("finalize", "correlation", "status_labels"):
        monkeypatch.setattr(estimator._pair_moments, name, unnecessary)
    monkeypatch.setattr("force.streaming.status_labels_from_codes", unnecessary)
    diagnostics = estimator.get_diagnostics()
    assert diagnostics["accepted_counts"].shape == (3, 3)
    assert diagnostics["pair_status_codes"].shape == (3, 3)
    assert "pair_status" not in diagnostics


def test_diagnostics_before_initialization_and_label_flag_validation():
    estimator = StreamingForceEstimator(window_size=50)
    diagnostics = estimator.get_diagnostics(include_status_labels=True)
    assert diagnostics["accepted_counts"] is None
    assert diagnostics["pair_status_codes"] is None
    assert diagnostics["pair_status"] is None
    with pytest.raises(ValueError, match="boolean"):
        estimator.get_diagnostics(include_status_labels="yes")


def test_retained_numerical_state_is_independent_of_stream_length():
    rng = np.random.default_rng(5)
    estimator = StreamingForceEstimator(window_size=23, use_ter=False)
    estimator.update_batch(rng.normal(size=(24, 4)))
    initial_bytes = estimator.state_nbytes_
    for _ in range(40):
        estimator.update_batch(rng.normal(size=(23, 4)))
        assert estimator.state_nbytes_ == initial_bytes
    assert estimator.n_samples_seen_ == 944
    assert estimator.window_count_ == 1


def test_processing_failure_invalidates_future_updates_and_snapshots(monkeypatch):
    estimator = StreamingForceEstimator(window_size=9).update([0.0, 0.0])

    def fail(_):
        raise FloatingPointError("unsupported arithmetic")

    monkeypatch.setattr(estimator._quantile_state, "update_batch", fail)
    with pytest.raises(FloatingPointError, match="unsupported arithmetic"):
        estimator.update([1.0, 1.0])
    assert estimator.status_ == "failed"
    with pytest.raises(RuntimeError, match="invalid after"):
        estimator.finalize()
    with pytest.raises(RuntimeError, match="invalid after"):
        estimator.update([1.0, 1.0])


def test_failed_snapshot_invalidates_all_further_extraction(monkeypatch):
    estimator = StreamingForceEstimator(window_size=50, use_ter=False).update_batch(
        np.random.default_rng(86).normal(size=(70, 2))
    )

    def fail():
        raise FloatingPointError("unsupported arithmetic")

    monkeypatch.setattr(estimator._pair_moments, "correlation", fail)
    with pytest.raises(FloatingPointError, match="unsupported arithmetic"):
        estimator.finalize()
    assert estimator.status_ == "failed"
    with pytest.raises(RuntimeError, match="invalid after"):
        estimator.get_diagnostics()
    with pytest.raises(RuntimeError, match="invalid after"):
        _ = estimator.accepted_counts_
