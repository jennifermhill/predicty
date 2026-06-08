"""metadata lookup for zarr datasets.

each experiment has one or more rows in training_data_index.xlsx, one per
strain. callers pass a zarr_path containing a mixin token (e.g.
".../cryolite_mixin_test65_..."). the spreadsheet `expt` column uses the
compact form `mixin65`; we normalize between the two.

when a mixin has multiple rows (e.g. mixin65 has strains NC28.1 and WS205),
we disambiguate by looking for the row's `strain` value as a substring of
the zarr_path. paths produced by Jennifer's pipeline include the strain name
in a parent folder and/or the filename (".../NC28.1_overnight_day1/...").

if a mixin has exactly one row, we don't need to match the strain — the row
is unambiguous on its own. this is the common case for mixins where only one
strain was recorded.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as et
from functools import lru_cache
from pathlib import Path
from typing import Any

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
    """return interval_minutes for the (mixin, strain) row matching zarr_path."""
    row = _resolve_row(zarr_path, str(index_path))
    val = row["interval_minutes"]
    if _is_blank(val):
        raise ValueError(
            f"interval_minutes is missing for {extract_mixin_id(zarr_path)!r} "
            f"(strain={row.get('strain')!r}) in {index_path}"
        )
    return float(val)


def get_z_limit(zarr_path: str, index_path: str | Path) -> int | None:
    """return the per-experiment z cap for the (mixin, strain) row matching
    zarr_path. 1-indexed inclusive max == 0-indexed exclusive upper bound.
    a blank cell means the row is marked not-for-training; returns None.
    """
    row = _resolve_row(zarr_path, str(index_path))
    val = row["z_limit"]
    if _is_blank(val):
        return None
    return int(val)


@lru_cache(maxsize=8)
def _load_index(index_path: str) -> Any:
    """cached spreadsheet load. one read per (index_path) per process."""
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
    """pick the spreadsheet row matching zarr_path. disambiguates multi-strain
    mixins by substring-matching the spreadsheet `strain` value against
    zarr_path. returns a single pandas Series.
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

    # multi-row case: try to match strain against the zarr path.
    path_lower = zarr_path.lower()
    strains = rows["strain"].astype(str).str.strip()
    valid = strains[strains.str.lower() != "nan"]
    matches = rows.loc[
        valid.index[[s.lower() in path_lower for s in valid]]
    ]

    if matches.empty:
        # strain disambiguation failed; fall back to well column (e.g. "B2"
        # appears in a path like ".../Field_B2/...").
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

    # multiple strains substring-match (e.g. 'NC28' and 'NC28.1' both fit).
    # prefer the most specific match: longest strain string.
    lengths = matches["strain"].astype(str).str.len()
    best = matches[lengths == lengths.max()]
    if len(best) > 1:
        raise ValueError(
            f"ambiguous strain match for {mixin_id!r} in {zarr_path}; "
            f"tied strains: {best['strain'].tolist()}"
        )
    return best.iloc[0]

def get_voxel_dims(zarr_path, res_lvl=0):
    xml_file = Path(zarr_path) / "OME" / "METADATA.ome.xml"
    XML_tree = et.parse(xml_file)
    XML_root = XML_tree.getroot()

    namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

    images = XML_root.findall('ome:Image', namespaces)
    if res_lvl >= len(images):
        res_lvl = 0

    pixels = images[res_lvl].find('ome:Pixels', namespaces)

    pixel_size_x = float(pixels.get('PhysicalSizeX'))
    pixel_size_y = float(pixels.get('PhysicalSizeY'))
    pixel_size_z = float(pixels.get('PhysicalSizeZ'))
    return pixel_size_x, pixel_size_y, pixel_size_z
