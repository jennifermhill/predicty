"""interactive .zarr chooser for single-zarr CLI runs only.

the current multi-zarr training path cannot reach this module. that path sets
`cfg.zarr_paths` from the config and never calls `pick_zarr_path`. the chooser
applies only when you run a script with a single `--zarr-path`. the chooser
also applies when you give no `--zarr-path` on a machine with a display. on a
headless node with no `--zarr-path`, the chooser prints a message and exits
with status 2.
"""

import argparse
import os
import sys


def pick_zarr_path() -> str:
    """return the .zarr root from the cli or from tkinter.

    the dataset adds the pyramid path internally.
    """
    cli_path = _parse_cli_zarr_path()
    if cli_path is not None:
        return cli_path

    if _is_headless():
        print(
            "[picker] No display available and --zarr-path not provided. "
            "Pass --zarr-path /path/to/data.zarr to run headless.",
            file=sys.stderr,
        )
        sys.exit(2)

    # import tk here, not at module level, so headless workers skip tk.
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        path = filedialog.askdirectory(
            title="Select a .zarr root directory",
            mustexist=True,
        )
    finally:
        root.destroy()

    if not path:
        print("[picker] No zarr directory selected; aborting.", file=sys.stderr)
        sys.exit(2)
    return path


def _parse_cli_zarr_path() -> str | None:
    """read --zarr-path but do not consume the other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--zarr-path", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args.zarr_path


def _is_headless() -> bool:
    # on macos, tk works without a display.
    if sys.platform == "darwin":
        return False
    return not os.environ.get("DISPLAY")
