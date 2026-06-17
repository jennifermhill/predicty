from __future__ import annotations

import unittest

from data.spatial_regions import (
    region_bounds_for_top_left,
    region_index_for_top_left,
    select_nonadjacent_regions,
)


class SpatialRegionsTest(unittest.TestCase):
    def test_two_by_two_regions_cover_legal_top_left_positions(self) -> None:
        frame_y, frame_x, crop = 10, 14, 4
        legal_y = frame_y - crop + 1
        legal_x = frame_x - crop + 1

        seen: set[tuple[int, int]] = set()
        for region_idx in range(4):
            y0, y1, x0, x1 = region_bounds_for_top_left(
                frame_y, frame_x, crop, 2, 2, region_idx
            )
            for y in range(y0, y1):
                for x in range(x0, x1):
                    self.assertNotIn((y, x), seen)
                    seen.add((y, x))
                    self.assertEqual(
                        region_index_for_top_left(y, x, frame_y, frame_x, crop, 2, 2),
                        region_idx,
                    )

        self.assertEqual(len(seen), legal_y * legal_x)

    def test_region_three_is_bottom_right_in_two_by_two_grid(self) -> None:
        y0, y1, x0, x1 = region_bounds_for_top_left(10, 10, 2, 2, 2, 3)

        self.assertGreaterEqual(y0, 4)
        self.assertEqual(y1, 9)
        self.assertGreaterEqual(x0, 4)
        self.assertEqual(x1, 9)

    def test_invalid_region_raises(self) -> None:
        with self.assertRaises(ValueError):
            region_bounds_for_top_left(10, 10, 2, 2, 2, 4)

    def test_selects_nonadjacent_high_scoring_regions(self) -> None:
        scores = {
            0: 0.99,
            1: 0.20,
            2: 0.98,
            3: 0.30,
            4: 0.10,
            5: 0.40,
            6: 0.97,
            7: 0.50,
            8: 0.96,
        }

        selected = select_nonadjacent_regions(
            scores,
            count=3,
            grid_x=3,
            min_gap=1,
            min_score=0.95,
        )

        self.assertEqual(selected, [0, 2, 6])

    def test_selection_falls_back_when_min_score_too_strict(self) -> None:
        scores = {idx: 0.1 + idx * 0.01 for idx in range(9)}
        selected = select_nonadjacent_regions(
            scores,
            count=2,
            grid_x=3,
            min_gap=1,
            min_score=100.0,
    )

        self.assertEqual(selected, [8, 6])


if __name__ == "__main__":
    unittest.main()
