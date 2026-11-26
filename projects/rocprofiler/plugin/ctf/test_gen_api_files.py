"""Tests for enum-value parsing in the CTF API generator."""

import unittest

from gen_api_files import _enumerator_effective_val


class EnumeratorEffectiveValueTest(unittest.TestCase):
    def test_integer_value(self):
        self.assertEqual(
            _enumerator_effective_val({"name": "VALUE", "value": 5}),
            5,
        )

    def test_shift_expression_in_value(self):
        self.assertEqual(
            _enumerator_effective_val(
                {"name": "VALUE", "value": "(1 << 3)"}
            ),
            8,
        )

    def test_bitwise_expression_in_value(self):
        self.assertEqual(
            _enumerator_effective_val(
                {"name": "VALUE", "value": "(1 << 0) | (1 << 1)"}
            ),
            3,
        )

    def test_shift_expression_in_raw_value(self):
        self.assertEqual(
            _enumerator_effective_val(
                {"name": "VALUE", "value": None, "raw_value": "(1 << 2)"}
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
