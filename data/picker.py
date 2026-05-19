import argparse
import os
import sys


def pick_zarr_path() -> str:
    """return the .zarr root from cli or tkinter.

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

    # import lazily so headless workers skip tk.
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
    """peek at --zarr-path without consuming other args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--zarr-path", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args.zarr_path


def _is_headless() -> bool:
    # macos tk works without display.
    if sys.platform == "darwin":
        return False
    return not os.environ.get("DISPLAY")
