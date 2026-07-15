"""dataset-index schedules for balanced multi-dataset batches."""

from __future__ import annotations

import random


def balanced_batch_indices(
    n_datasets: int,
    batch_size: int,
    batch_number: int,
    rng: random.Random,
) -> list[int]:
    """return one shuffled batch with dataset counts differing by at most one.

    when ``batch_size`` is not divisible by ``n_datasets``, the remainder
    slots rotate across batches so every dataset receives the same long-run
    exposure.
    """
    if n_datasets < 1:
        raise ValueError(f"n_datasets must be >= 1; got {n_datasets}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    if batch_number < 0:
        raise ValueError(f"batch_number must be >= 0; got {batch_number}")

    per_dataset, remainder = divmod(batch_size, n_datasets)
    indices = [
        dataset_idx
        for dataset_idx in range(n_datasets)
        for _ in range(per_dataset)
    ]
    remainder_start = (batch_number * remainder) % n_datasets
    indices.extend(
        (remainder_start + offset) % n_datasets
        for offset in range(remainder)
    )
    rng.shuffle(indices)
    return indices
