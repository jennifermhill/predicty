"""metadata lookup for zarr datasets.

each experiment has a row (or several) in training_data_index.xlsx. the
critical field for temporal sampling is `interval_minutes`, the wall-clock
spacing between frames.

callers pass a zarr_path containing a mixin token. on disk the folder names
look like `cryolite_mixin_test41_2024-01-09`, while the spreadsheet `expt`
column uses the compact form `mixin41`. we accept both spellings (mixin41,
mixin_test41, mixin-test-41, ...) and normalize to the compact form.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


# tolerate any non-digit junk between "mixin" and the digit suffix, so paths
# like ".../cryolite_mixin_test41_..." resolve to "mixin41".
_MIXIN_RE = re.compile(r"mixin[^0-9]*?(\d+)", re.IGNORECASE)


def extract_mixin_id(zarr_path: str) -> str:
    """pull the mixin id out of a zarr path and normalize to `mixinNN`."""
    match = _MIXIN_RE.search(zarr_path)
    if match is None:
        raise ValueError(
            f"could not find a mixin id (e.g. 'mixin41') in zarr_path: {zarr_path!r}"
        )
    return f"mixin{match.group(1)}"


def get_interval_minutes(zarr_path: str, index_path: str | Path) -> float:
    """return the unique interval_minutes for the experiment in `zarr_path`."""
    mixin_id = extract_mixin_id(zarr_path)
    return _interval_for_mixin(mixin_id, str(index_path))


@lru_cache(maxsize=8)
def _interval_for_mixin(mixin_id: str, index_path: str) -> float:
    """cached lookup keyed by (mixin_id, index_path)."""
    import pandas as pd

    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"training_data_index spreadsheet not found at {index_path}"
        )
    df = pd.read_excel(index_path)
    if "expt" not in df.columns or "interval_minutes" not in df.columns:
        raise ValueError(
            f"{index_path} is missing required columns 'expt' and/or 'interval_minutes'"
        )

    expt_series = df["expt"].astype(str).str.strip().str.lower()
    rows = df[expt_series == mixin_id]
    if rows.empty:
        raise ValueError(
            f"no rows in {index_path} for experiment {mixin_id!r}"
        )

    intervals = rows["interval_minutes"].dropna().unique().tolist()
    if not intervals:
        raise ValueError(
            f"interval_minutes is missing for experiment {mixin_id!r} in {index_path}"
        )
    if len(intervals) > 1:
        raise ValueError(
            f"inconsistent interval_minutes for {mixin_id!r}: {intervals}"
        )
    return float(intervals[0])
