"""CPU-only checks for the Aorta benchmark's independent output gate."""

import unittest

from consan_aorta_benchmark_workload import _check_cpu_reference


class AortaOracleTest(unittest.TestCase):
    def test_finite_repeatable_corruption_is_rejected(self):
        import torch

        reference = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            reference.weight.copy_(torch.eye(2))
        inputs = torch.tensor([[1.0, 2.0]])
        expected = reference(inputs).detach()
        self.assertEqual(_check_cpu_reference(reference, [(inputs, expected)])["cpu_oracle_max_abs_error"], 0)
        corrupted = expected + 1.0
        # This output is finite and perfectly repeatable. The independent CPU
        # reference must still reject it, on either operation in the cell.
        with self.assertRaises(AssertionError):
            _check_cpu_reference(reference, [(inputs, expected), (inputs, corrupted)])

    def test_missing_output_is_not_an_oracle_pass(self):
        with self.assertRaisesRegex(RuntimeError, "no model outputs"):
            _check_cpu_reference(None, [])

    def test_greedy_token_change_fails_even_below_numeric_error_bound(self):
        import torch

        inputs = torch.tensor([[1.0, 1.001]])
        with self.assertRaises(AssertionError):
            _check_cpu_reference(torch.nn.Identity(), [(inputs, inputs.flip(-1))])


if __name__ == "__main__":
    unittest.main()
