"""metadata lookup for zarr datasets.

each experiment has one or more rows in training_data_index.xlsx, one row per
strain. a caller gives a zarr_path that contains a mixin token, for example
".../cryolite_mixin_test65_...". the spreadsheet `expt` column uses the compact
form `mixin65`. this module normalizes between the two forms.

a mixin can have more than one row. for example, mixin65 has the strains NC28.1
and WS205. in that case this module looks for the `strain` value of the row as
a substring of the zarr_path. Jennifer's pipeline writes paths that include the
strain name in a parent folder, in the filename, or in both, for example
".../NC28.1_overnight_day1/...".

a mixin with exactly one row is not ambiguous on its own, so a strain match is
not necessary. that is the common case.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

# the regex tolerates any non-digit text between "mixin" and the digit suffix.
# thus a path such as ".../cryolite_mixin_test41_..." resolves to "mixin41".
_MIXIN_RE = re.compile(r"mixin[^0-9]*?(\d+)", re.IGNORECASE)


def extract_mixin_id(zarr_path: str) -> str:
    """extract the mixin id from a zarr path and normalize it to `mixinNN`."""
    match = _MIXIN_RE.search(zarr_path)
    if match is None:
        raise ValueError(
            f"could not find a mixin id (e.g. 'mixin41') in zarr_path: {zarr_path!r}"
        )
    return f"mixin{match.group(1)}"


def get_interval_minutes(zarr_path: str, index_path: str | Path) -> float:
    """return interval_minutes for the (mixin, strain) row that matches zarr_path."""
    row = _resolve_row(zarr_path, str(index_path))
    val = row["interval_minutes"]
    if _is_blank(val):
        raise ValueError(
            f"interval_minutes is missing for {extract_mixin_id(zarr_path)!r} "
            f"(strain={row.get('strain')!r}) in {index_path}"
        )
    return float(val)


def get_z_limit(zarr_path: str, index_path: str | Path) -> int | None:
    """return the per-experiment z cap for the (mixin, strain) row that matches
    zarr_path.

    the 1-indexed inclusive max equals the 0-indexed exclusive upper bound. a
    blank cell means that the row is not-for-training. then this function
    returns None.
    """
    row = _resolve_row(zarr_path, str(index_path))
    val = row["z_limit"]
    if _is_blank(val):
        return None
    return int(val)


@lru_cache(maxsize=8)
def _load_index(index_path: str) -> Any:
    """load the spreadsheet and cache it. one read per (index_path) per process."""
    import pandas as pd

    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"training_data_index spreadsheet not found at {index_path}"
        )
    return pd.read_excel(index_path)


def _is_blank(val: Any) -> bool:
    import pandas as pd

    return val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val)


def _resolve_row(zarr_path: str, index_path: str) -> Any:
    """pick the spreadsheet row that matches zarr_path.

    for a mixin with more than one strain, this function matches the
    spreadsheet `strain` value as a substring of zarr_path. this function
    returns one pandas Series.
    """
    mixin_id = extract_mixin_id(zarr_path)
    df = _load_index(index_path)

    required = {"expt", "strain", "interval_minutes", "z_limit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{index_path} is missing required columns: {sorted(missing)}"
        )

    expt_series = df["expt"].astype(str).str.strip().str.lower()
    rows = df[expt_series == mixin_id]
    if rows.empty:
        raise ValueError(f"no rows in {index_path} for experiment {mixin_id!r}")
    if len(rows) == 1:
        return rows.iloc[0]

    # multi-row case: try to match the strain against the zarr path.
    path_lower = zarr_path.lower()
    strains = rows["strain"].astype(str).str.strip()
    valid = strains[strains.str.lower() != "nan"]
    matches = rows.loc[
        valid.index[[s.lower() in path_lower for s in valid]]
    ]

    if matches.empty:
        # the strain match failed. fall back to the well column. for example,
        # "B2" appears in a path such as ".../Field_B2/...".
        if "well" in rows.columns:
            wells = rows["well"].astype(str).str.strip()
            valid_wells = wells[wells.str.lower() != "nan"]
            well_matches = rows.loc[
                valid_wells.index[[w.lower() in path_lower for w in valid_wells]]
            ]
            if len(well_matches) == 1:
                return well_matches.iloc[0]
        raise ValueError(
            f"{mixin_id!r} has {len(rows)} rows (strains {strains.tolist()}) "
            f"but none of those strain tokens appear in zarr_path: {zarr_path}"
        )
    if len(matches) == 1:
        return matches.iloc[0]

    # more than one strain matches as a substring. for example, 'NC28' and
    # 'NC28.1' both fit. prefer the most specific match: the longest strain
    # string.
    lengths = matches["strain"].astype(str).str.len()
    best = matches[lengths == lengths.max()]
    if len(best) > 1:
        raise ValueError(
            f"ambiguous strain match for {mixin_id!r} in {zarr_path}; "
            f"tied strains: {best['strain'].tolist()}"
        )
    return best.iloc[0]
