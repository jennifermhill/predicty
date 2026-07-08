from __future__ import annotations

import random
import unittest
from collections import Counter

from data.balancing import balanced_batch_indices


class BalancedBatchIndicesTest(unittest.TestCase):
    def test_divisible_batch_has_equal_counts(self) -> None:
        indices = balanced_batch_indices(4, 16, 0, random.Random(0))

        self.assertEqual(len(indices), 16)
        self.assertEqual(Counter(indices), {0: 4, 1: 4, 2: 4, 3: 4})

    def test_remainder_rotates_evenly(self) -> None:
        extras = Counter()
        for batch_number in range(3):
            counts = Counter(
                balanced_batch_indices(
                    3, 8, batch_number, random.Random(batch_number)
                )
            )
            self.assertEqual(sorted(counts.values()), [2, 3, 3])
            extras.update(idx for idx, count in counts.items() if count == 3)

        self.assertEqual(extras, {0: 2, 1: 2, 2: 2})

    def test_more_datasets_than_samples_rotates_coverage(self) -> None:
        seen = Counter()
        for batch_number in range(5):
            indices = balanced_batch_indices(
                5, 2, batch_number, random.Random(batch_number)
            )
            self.assertEqual(len(indices), 2)
            self.assertEqual(len(set(indices)), 2)
            seen.update(indices)

        self.assertEqual(seen, {0: 2, 1: 2, 2: 2, 3: 2, 4: 2})


if __name__ == "__main__":
    unittest.main()
