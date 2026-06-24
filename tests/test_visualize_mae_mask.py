from __future__ import annotations

import unittest

import numpy as np

from eval.visualize_mae_mask import metric_text, reconstruction_metrics


class MaeVisualizationMetricsTest(unittest.TestCase):
    def test_metrics_exclude_visible_voxels(self) -> None:
        original = np.asarray([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
        reconstruction = np.asarray(
            [[[0.5, 0.5], [1000.0, -1000.0]]], dtype=np.float32
        )
        mask = np.asarray([[[1, 1], [0, 0]]], dtype=np.float32)

        metrics = reconstruction_metrics(original, reconstruction, mask, 0.75)

        self.assertEqual(metrics["hidden_voxels"], 2)
        self.assertAlmostEqual(metrics["mask_ratio"], 0.5)
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], 0.5)
        self.assertAlmostEqual(metrics["r2"], 0.0)
        self.assertIsNone(metrics["pearson_r"])
        self.assertAlmostEqual(metrics["objective_mse"], 0.75)

    def test_perfect_reconstruction_has_zero_error(self) -> None:
        original = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        mask = np.ones_like(original)
        metrics = reconstruction_metrics(original, original.copy(), mask, 0.0)

        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)
        self.assertEqual(metrics["pearson_r"], 1.0)
        self.assertTrue(np.isinf(metrics["psnr_db"]))

    def test_metric_text_labels_hidden_region(self) -> None:
        metrics = reconstruction_metrics(
            np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            np.ones((2, 2, 2), dtype=np.float32),
            0.0,
        )
        text = metric_text(metrics)
        self.assertIn("THIS CROP: HIDDEN-VOXEL METRICS", text)
        self.assertIn("visible patches excluded", text)


if __name__ == "__main__":
    unittest.main()
