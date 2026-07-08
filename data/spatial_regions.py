from __future__ import annotations


def region_row_col(region_idx: int, grid_x: int) -> tuple[int, int]:
    return region_idx // grid_x, region_idx % grid_x


def region_bounds_for_top_left(
    frame_y: int,
    frame_x: int,
    crop_size: int,
    grid_y: int,
    grid_x: int,
    region_idx: int,
) -> tuple[int, int, int, int]:
    """Return [y0, y1), [x0, x1) bounds for legal crop top-left positions."""
    if grid_y < 1 or grid_x < 1:
        raise ValueError(f"spatial region grid must be positive; got {grid_y}x{grid_x}")
    n_regions = grid_y * grid_x
    if not 0 <= region_idx < n_regions:
        raise ValueError(
            f"region_idx must be in [0, {n_regions}); got {region_idx}"
        )
    legal_y = frame_y - crop_size + 1
    legal_x = frame_x - crop_size + 1
    if legal_y < 1 or legal_x < 1:
        raise ValueError(
            f"crop_size={crop_size} exceeds frame size {frame_y}x{frame_x}"
        )

    row = region_idx // grid_x
    col = region_idx % grid_x
    y0 = (row * legal_y) // grid_y
    y1 = ((row + 1) * legal_y) // grid_y
    x0 = (col * legal_x) // grid_x
    x1 = ((col + 1) * legal_x) // grid_x
    if y0 >= y1 or x0 >= x1:
        raise ValueError(
            f"region {region_idx} in {grid_y}x{grid_x} grid has no legal "
            f"top-left positions for frame {frame_y}x{frame_x}, crop_size={crop_size}"
        )
    return y0, y1, x0, x1


def region_index_for_top_left(
    y: int,
    x: int,
    frame_y: int,
    frame_x: int,
    crop_size: int,
    grid_y: int,
    grid_x: int,
) -> int:
    """Map a legal crop top-left position to its row-major spatial region."""
    legal_y = frame_y - crop_size + 1
    legal_x = frame_x - crop_size + 1
    if not 0 <= y < legal_y or not 0 <= x < legal_x:
        raise ValueError(
            f"top-left ({y}, {x}) is outside legal range "
            f"[0, {legal_y}) x [0, {legal_x})"
        )
    row = next(
        r
        for r in range(grid_y)
        if (r * legal_y) // grid_y <= y < ((r + 1) * legal_y) // grid_y
    )
    col = next(
        c
        for c in range(grid_x)
        if (c * legal_x) // grid_x <= x < ((c + 1) * legal_x) // grid_x
    )
    return row * grid_x + col


def regions_are_adjacent(
    a: int,
    b: int,
    grid_x: int,
    min_gap: int,
) -> bool:
    """Return whether two regions are closer than the requested tile gap."""
    ar, ac = region_row_col(a, grid_x)
    br, bc = region_row_col(b, grid_x)
    return max(abs(ar - br), abs(ac - bc)) <= min_gap


def select_nonadjacent_regions(
    scores: dict[int, float],
    count: int,
    grid_x: int,
    min_gap: int,
    min_score: float | None = None,
) -> list[int]:
    """Greedily select high-scoring regions that do not touch each other."""
    if count < 1:
        raise ValueError(f"count must be >= 1; got {count}")
    eligible = [
        item
        for item in scores.items()
        if min_score is None or item[1] >= min_score
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))

    selected: list[int] = []
    for idx, _score in eligible:
        if all(not regions_are_adjacent(idx, prev, grid_x, min_gap) for prev in selected):
            selected.append(idx)
            if len(selected) == count:
                return selected

    # If the score threshold was too strict, keep the non-adjacent constraint
    # and fill from the best remaining regions so the caller can still proceed.
    if min_score is not None and len(selected) < count:
        fallback = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        for idx, _score in fallback:
            if idx in selected:
                continue
            if all(not regions_are_adjacent(idx, prev, grid_x, min_gap) for prev in selected):
                selected.append(idx)
                if len(selected) == count:
                    return selected

    if len(selected) < count:
        raise ValueError(
            f"could only select {len(selected)} non-adjacent regions; requested {count}"
        )
    return selected
