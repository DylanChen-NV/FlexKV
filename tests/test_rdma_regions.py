import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "rdma_regions", Path(__file__).parents[1] / "flexkv/rdma_regions.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
split = module.split_read_regions


class CpuReadRegionsTest(unittest.TestCase):
    def test_default_keeps_original_vectors(self):
        src, dst, sizes = [1], [2], [3]
        actual = split(src, dst, sizes, 0)
        self.assertIs(actual[0], src)
        self.assertIs(actual[1], dst)
        self.assertIs(actual[2], sizes)

    def test_nondivisible_regions_preserve_ranges_order_and_total(self):
        self.assertEqual(split([100, 500], [1000, 5000], [9, 2], 4),
                         ([100, 104, 108, 500], [1000, 1004, 1008, 5000], [4, 4, 1, 2]))

    def test_invalid_opt_in_vectors_fail(self):
        for args in (([1], [2], [3], -1), ([1], [], [3], 4), ([1], [2], [0], 4)):
            with self.assertRaises(ValueError):
                split(*args)


if __name__ == "__main__":
    unittest.main()
