from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.interactive_latent_viewer import (
    LatentCropViewer,
    compute_visible_mask,
    indices_inside_polygon,
    load_raw_crop,
    nearest_annotated_index,
    nearest_indices,
    representative_indices,
    robust_limits,
    select_click_indices,
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

    def test_nearest_annotated_index_finds_first_labeled_row_in_order(self) -> None:
        rows = [{"label": ""}, {"label": ""}, {"label": "slug"}, {"label": "aggregate"}]
        self.assertEqual(nearest_annotated_index(np.asarray([0, 1, 2, 3]), rows), 2)
        self.assertIsNone(nearest_annotated_index(np.asarray([0, 1]), rows))

    def test_select_click_indices_reserves_a_slot_for_nearest_annotated_crop(self) -> None:
        # A dense cluster of unlabeled points sits right on the click target;
        # the one labeled point is far away. Pure nearest-4 would never
        # surface it, so it should be swapped into the last slot instead.
        rows = [
            {
                "dataset_idx": "0", "view": "t1", "t": "10", "frame_gap": "3",
                "z": "1", "z_end": "3", "global_y": "4", "global_x": str(i),
                "label": "",
            }
            for i in range(5)
        ]
        rows.append(
            {
                "dataset_idx": "0", "view": "t1", "t": "10", "frame_gap": "3",
                "z": "1", "z_end": "3", "global_y": "4", "global_x": "99",
                "label": "slug",
            }
        )
        coords = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [10.0, 10.0]]
        )
        cfg = Config(temporal_direction="forward")
        result = select_click_indices(coords, rows, cfg, (0.0, 0.0), nearest=4)
        self.assertEqual(len(result), 4)
        self.assertIn(5, result.tolist())

    def test_select_click_indices_is_unchanged_when_no_annotations_exist(self) -> None:
        rows = [
            {
                "dataset_idx": "0", "view": "t1", "t": "10", "frame_gap": "3",
                "z": "1", "z_end": "3", "global_y": "4", "global_x": str(i),
                "label": "",
            }
            for i in range(4)
        ]
        coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        cfg = Config(temporal_direction="forward")
        result = select_click_indices(coords, rows, cfg, (0.0, 0.0), nearest=3)
        self.assertEqual(result.tolist(), [0, 1, 2])

    def test_compute_visible_mask_filters_by_annotation_status(self) -> None:
        rows = [{"label": "a"}, {"label": ""}, {"label": "b"}, {"label": ""}]
        self.assertEqual(
            compute_visible_mask(rows, True, True).tolist(), [True, True, True, True]
        )
        self.assertEqual(
            compute_visible_mask(rows, True, False).tolist(), [True, False, True, False]
        )
        self.assertEqual(
            compute_visible_mask(rows, False, True).tolist(), [False, True, False, True]
        )
        self.assertEqual(
            compute_visible_mask(rows, False, False).tolist(), [False, False, False, False]
        )

    def test_select_click_indices_never_returns_a_hidden_point(self) -> None:
        # Same near-cluster / far-annotated-point setup as the "reserves a
        # slot" test above, but the annotated point is hidden by a checkbox;
        # it must not be selected even though it would otherwise be promoted.
        rows = [
            {
                "dataset_idx": "0", "view": "t1", "t": "10", "frame_gap": "3",
                "z": "1", "z_end": "3", "global_y": "4", "global_x": str(i),
                "label": "",
            }
            for i in range(5)
        ]
        rows.append(
            {
                "dataset_idx": "0", "view": "t1", "t": "10", "frame_gap": "3",
                "z": "1", "z_end": "3", "global_y": "4", "global_x": "99",
                "label": "slug",
            }
        )
        coords = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [10.0, 10.0]]
        )
        cfg = Config(temporal_direction="forward")
        visible_mask = np.array([True, True, True, True, True, False])
        result = select_click_indices(
            coords, rows, cfg, (0.0, 0.0), nearest=4, visible_mask=visible_mask
        )
        self.assertNotIn(5, result.tolist())

    def test_select_click_indices_returns_empty_when_nothing_visible(self) -> None:
        rows = [
            {
                "dataset_idx": "0", "view": "t1", "t": "10", "frame_gap": "3",
                "z": "1", "z_end": "3", "global_y": "4", "global_x": str(i),
                "label": "",
            }
            for i in range(3)
        ]
        coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        cfg = Config(temporal_direction="forward")
        visible_mask = np.zeros(3, dtype=bool)
        result = select_click_indices(
            coords, rows, cfg, (0.0, 0.0), nearest=3, visible_mask=visible_mask
        )
        self.assertEqual(len(result), 0)

    def test_visibility_checkboxes_hide_and_show_map_points(self) -> None:
        rows = [
            {
                "dataset_idx": "0", "dataset_name": "dataset_a.zarr", "view": "t1",
                "t": "10", "frame_gap": "3", "z": "1", "z_end": "3",
                "global_y": "4", "global_x": "5", "label": "",
            },
            {
                "dataset_idx": "0", "dataset_name": "dataset_a.zarr", "view": "t1",
                "t": "50", "frame_gap": "3", "z": "1", "z_end": "3",
                "global_y": "4", "global_x": "5", "label": "slug",
            },
        ]
        coords = np.asarray([[0.0, 0.0], [5.0, 5.0]])
        cfg = Config(temporal_direction="forward")
        viewer = LatentCropViewer(
            rows, coords, cfg, ["/data/dataset_a.zarr"],
            nearest=4, max_crops=12, point_size=5.0,
        )
        try:
            # both checkboxes on by default: both points visible
            self.assertEqual(viewer.visible_mask.tolist(), [True, True])

            viewer._on_visibility_toggle("Unannotated")
            self.assertEqual(viewer.visible_mask.tolist(), [False, True])
            dataset_artist, _ = viewer.dataset_artists[0]
            self.assertEqual(len(dataset_artist.get_offsets()), 1)

            viewer._on_visibility_toggle("Annotated")
            self.assertEqual(viewer.visible_mask.tolist(), [False, False])
            label_artist, _ = viewer.label_artists[0]
            self.assertEqual(len(label_artist.get_offsets()), 0)
        finally:
            plt.close(viewer.map_fig)
            if viewer.crop_fig is not None:
                plt.close(viewer.crop_fig)

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

    def test_map_legend_includes_a_ring_entry_per_annotation_label(self) -> None:
        rows = [
            {
                "dataset_idx": "0", "dataset_name": "dataset_a.zarr", "view": "t1",
                "t": "10", "frame_gap": "3", "z": "1", "z_end": "3",
                "global_y": "4", "global_x": "5", "label": "",
            },
            {
                "dataset_idx": "0", "dataset_name": "dataset_a.zarr", "view": "t1",
                "t": "50", "frame_gap": "3", "z": "1", "z_end": "3",
                "global_y": "4", "global_x": "5", "label": "slug",
            },
            {
                "dataset_idx": "0", "dataset_name": "dataset_a.zarr", "view": "t1",
                "t": "90", "frame_gap": "3", "z": "1", "z_end": "3",
                "global_y": "4", "global_x": "5", "label": "aggregate",
            },
        ]
        coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
        cfg = Config(temporal_direction="forward")
        viewer = LatentCropViewer(
            rows, coords, cfg, ["/data/dataset_a.zarr"],
            nearest=4, max_crops=12, point_size=5.0,
        )
        try:
            legend_labels = {t.get_text() for t in viewer.map_ax.get_legend().get_texts()}
            self.assertIn("[annotated] slug", legend_labels)
            self.assertIn("[annotated] aggregate", legend_labels)
        finally:
            plt.close(viewer.map_fig)
            if viewer.crop_fig is not None:
                plt.close(viewer.crop_fig)

    def test_render_crops_shows_label_in_title_for_annotated_points(self) -> None:
        import dask.array as da

        values = np.zeros((3, 1, 3, 5, 6), dtype=np.float32)
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "sample.zarr"
            store = dataset / "0" / "0"
            da.from_array(values, chunks=(1, 1, 3, 5, 6)).to_zarr(str(store))

            rows = [
                {
                    "dataset_idx": "0", "dataset_name": "sample.zarr", "view": "t1",
                    "t": "1", "frame_gap": "1", "z": "1", "z_end": "3",
                    "global_y": "2", "global_x": "3", "label": "slug",
                },
            ]
            coords = np.asarray([[0.0, 0.0]])
            cfg = Config(temporal_direction="forward", global_crop_size=2, channel=0)
            viewer = LatentCropViewer(
                rows, coords, cfg, [str(dataset)],
                nearest=4, max_crops=12, point_size=5.0,
            )
            try:
                viewer.render_crops(np.asarray([0]), "test selection")
                title = viewer.crop_fig.axes[0].get_title()
                self.assertIn("ANNOTATED: slug", title)
            finally:
                plt.close(viewer.map_fig)
                if viewer.crop_fig is not None:
                    plt.close(viewer.crop_fig)

    def test_render_crops_marks_unannotated_points_explicitly(self) -> None:
        import dask.array as da

        values = np.zeros((3, 1, 3, 5, 6), dtype=np.float32)
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "sample.zarr"
            store = dataset / "0" / "0"
            da.from_array(values, chunks=(1, 1, 3, 5, 6)).to_zarr(str(store))

            rows = [
                {
                    "dataset_idx": "0", "dataset_name": "sample.zarr", "view": "t1",
                    "t": "1", "frame_gap": "1", "z": "1", "z_end": "3",
                    "global_y": "2", "global_x": "3", "label": "",
                },
            ]
            coords = np.asarray([[0.0, 0.0]])
            cfg = Config(temporal_direction="forward", global_crop_size=2, channel=0)
            viewer = LatentCropViewer(
                rows, coords, cfg, [str(dataset)],
                nearest=4, max_crops=12, point_size=5.0,
            )
            try:
                viewer.render_crops(np.asarray([0]), "test selection")
                title = viewer.crop_fig.axes[0].get_title()
                self.assertIn("not annotated", title)
                self.assertNotIn("ANNOTATED:", title)
            finally:
                plt.close(viewer.map_fig)
                if viewer.crop_fig is not None:
                    plt.close(viewer.crop_fig)


if __name__ == "__main__":
    unittest.main()
