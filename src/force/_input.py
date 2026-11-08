"""Bounded conversion and validation for replayable sources."""

from numbers import Integral

import numpy as np

from ._quantiles import _as_real_array


DEFAULT_CHUNK_SIZE = 4096


def validate_chunk_size(chunk_size):
    if (
        not isinstance(chunk_size, Integral)
        or isinstance(chunk_size, (bool, np.bool_))
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer.")
    return int(chunk_size)


def validated_chunks(source, chunk_size=DEFAULT_CHUNK_SIZE):
    """Yield float64 blocks from rows or 2-D blocks, never converting >B rows.

    Source-owned arrays/containers are not copied in their entirety. Empty
    blocks still validate their feature count. A source may mix rows and blocks.
    """
    chunk_size = validate_chunk_size(chunk_size)
    n_features = None
    pending = []

    def convert(block):
        nonlocal n_features
        try:
            matrix = _as_real_array(block)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Source must yield numeric rows or 2-D blocks.") from exc
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise ValueError("Source must provide 2-D blocks with at least 2 features.")
        if n_features is None:
            n_features = matrix.shape[1]
        elif matrix.shape[1] != n_features:
            raise ValueError("Source feature count changed.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Source must contain only finite values.")
        return np.ascontiguousarray(matrix)

    for item in source:
        if isinstance(item, np.ndarray):
            ndim = item.ndim
        elif isinstance(item, (list, tuple)):
            ndim = 2 if len(item) and not np.isscalar(item[0]) else 1
        else:
            raise ValueError("Source must yield array-like rows or 2-D blocks.")
        if ndim == 1:
            # Copy each row immediately: generators may reuse their row buffer.
            row = convert([item])[0].copy()
            pending.append(row)
            if len(pending) == chunk_size:
                yield convert(pending)
                pending.clear()
        elif ndim == 2:
            if pending:
                yield convert(pending)
                pending.clear()
            if len(item) == 0:
                convert(item)
            for start in range(0, len(item), chunk_size):
                yield convert(item[start : start + chunk_size])
        else:
            raise ValueError("Source must yield rows or two-dimensional blocks.")
    if pending:
        yield convert(pending)
