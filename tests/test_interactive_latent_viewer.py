from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from eval.interactive_latent_viewer import (
    indices_inside_polygon,
    load_raw_crop,
    nearest_indices,
    representative_indices,
    robust_limits,
    temporal_pair,
    unique_crop_indices,
    validate_dataset_mapping,
)
from configs.config import Config


class InteractiveLatentViewerTest(unittest.TestCase):
    def test_nearest_indices_orders_by_distance(self) -> None:
        coords = np.asarray([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [8.0, 8.0]])
        result = nearest_indices(coords, (0.9, 0.0), 2)
        self.assertEqual(result.tolist(), [2, 0])

    def test_indices_inside_polygon(self) -> None:
        coords = np.asarray([[0.0, 0.0], [0.5, 0.5], [2.0, 2.0], [-1.0, 0.0]])
        vertices = [(-0.5, -0.5), (1.0, -0.5), (1.0, 1.0), (-0.5, 1.0)]
        self.assertEqual(indices_inside_polygon(coords, vertices).tolist(), [0, 1])

    def test_representative_indices_uses_selection_centroid(self) -> None:
        coords = np.asarray([[0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [2.1, 0.0]])
        selected = np.asarray([0, 1, 2, 3])
        self.assertEqual(representative_indices(coords, selected, 2).tolist(), [2, 3])

    def test_temporal_pair_handles_both_metadata_views(self) -> None:
        cfg = Config(temporal_direction="forward")
        self.assertEqual(
            temporal_pair({"t": "10", "frame_gap": "3", "view": "t1"}, cfg),
            (10, 13),
        )
        self.assertEqual(
            temporal_pair({"t": "13", "frame_gap": "3", "view": "t2"}, cfg),
            (10, 13),
        )

    def test_temporal_pair_handles_backward_direction(self) -> None:
        cfg = Config(temporal_direction="backward")
        self.assertEqual(
            temporal_pair({"t": "10", "frame_gap": "3", "view": "t1"}, cfg),
            (10, 7),
        )

    def test_unique_crop_indices_deduplicates_t1_and_t2_points(self) -> None:
        rows = [
            {
                "dataset_idx": "0", "view": "t1", "t": "10", "frame_gap": "3",
                "z": "1", "z_end": "3", "global_y": "4", "global_x": "5",
            },
            {
                "dataset_idx": "0", "view": "t2", "t": "13", "frame_gap": "3",
                "z": "1", "z_end": "3", "global_y": "4", "global_x": "5",
            },
            {
                "dataset_idx": "0", "view": "t1", "t": "20", "frame_gap": "3",
                "z": "1", "z_end": "3", "global_y": "4", "global_x": "5",
            },
        ]
        result = unique_crop_indices(np.asarray([0, 1, 2]), rows, Config())
        self.assertEqual(result.tolist(), [0, 2])

    def test_dataset_mapping_rejects_reordered_paths(self) -> None:
        rows = [{"dataset_idx": "0", "dataset_name": "original.zarr"}]
        with self.assertRaisesRegex(ValueError, "original path ordering"):
            validate_dataset_mapping(rows, ["/data/different.zarr"])

    def test_load_raw_crop_reconstructs_exact_metadata_window(self) -> None:
        import dask.array as da

        values = np.arange(2 * 1 * 3 * 5 * 6, dtype=np.float32).reshape(2, 1, 3, 5, 6)
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "sample.zarr"
            store = dataset / "0" / "0"
            da.from_array(values, chunks=(1, 1, 3, 5, 6)).to_zarr(str(store))
            row = {
                "dataset_idx": "0",
                "z": "1",
                "z_end": "3",
                "global_y": "2",
                "global_x": "3",
            }
            result = load_raw_crop(
                {}, [str(dataset)], row, Config(global_crop_size=2, channel=0), t=1
            )
            np.testing.assert_array_equal(result, values[1, 0, 1:3, 2:4, 3:5])

    def test_robust_limits_handles_constant_image(self) -> None:
        lo, hi = robust_limits(np.zeros((3, 3), dtype=np.float32))
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 1.0)


if __name__ == "__main__":
    unittest.main()
