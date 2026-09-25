"""Guards the ASAN GPU-target policy in cmake/AsanGpuTargets.cmake.

Each case pins one family's observed clang behaviour, so reclassifying a
family fails here rather than producing an ASAN build with a silently
uninstrumented device image. Driven through `cmake -P`, like
test_device_coverage_cmake.py.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


RCCL_ROOT = Path(__file__).resolve().parents[4]
MODULE = RCCL_ROOT / "cmake" / "AsanGpuTargets.cmake"


@unittest.skipUnless(shutil.which("cmake"), "cmake not available on PATH")
class AsanGpuTargetsTest(unittest.TestCase):
    def adjust(self, targets):
        """Run rccl_asan_adjust_gpu_targets over the given targets.

        Returns (result_list, status_messages). Raises on FATAL_ERROR.
        """
        quoted = " ".join(f'"{t}"' for t in targets)
        with tempfile.TemporaryDirectory() as temp_dir:
            out_file = Path(temp_dir) / "out.txt"
            script = Path(temp_dir) / "probe.cmake"
            script.write_text(
                f'include("{MODULE}")\n'
                f"rccl_asan_adjust_gpu_targets(RESULT {quoted})\n"
                f'file(WRITE "{out_file}" "${{RESULT}}")\n'
            )
            completed = subprocess.run(
                ["cmake", "-P", str(script)],
                capture_output=True, text=True)
            if completed.returncode != 0:
                raise AssertionError(completed.stderr.strip())
            result = out_file.read_text()
        # message(STATUS) goes to stdout under cmake -P, FATAL_ERROR to stderr.
        return ([t for t in result.split(";") if t], completed.stdout)

    def test_gfx9_gets_xnack_plus_requested(self):
        """A bare gfx9 name compiles with the sanitizer ignored."""
        result, _ = self.adjust(["gfx90a", "gfx942", "gfx950"])

        self.assertEqual(result,
                         ["gfx90a:xnack+", "gfx942:xnack+", "gfx950:xnack+"])

    def test_gfx9_already_requesting_xnack_plus_is_unchanged(self):
        result, _ = self.adjust(["gfx942:xnack+"])

        self.assertEqual(result, ["gfx942:xnack+"])

    def test_gfx9_requesting_xnack_minus_is_corrected(self):
        """The old substring test for ":xnack+" produced gfx942:xnack-:xnack+."""
        result, _ = self.adjust(["gfx942:xnack-"])

        self.assertEqual(result, ["gfx942:xnack+"])

    def test_gfx9_keeps_its_other_target_features(self):
        """Only the xnack term is ours to rewrite; sramecc+ is what was asked for."""
        result, _ = self.adjust(["gfx942:sramecc+:xnack-"])

        self.assertEqual(result, ["gfx942:sramecc+:xnack+"])

    def test_both_xnack_variants_of_one_arch_collapse_to_one_entry(self):
        """A duplicate becomes a second add_library() in DeviceLinker.cmake."""
        result, _ = self.adjust(["gfx942:xnack+", "gfx942:xnack-"])

        self.assertEqual(result, ["gfx942:xnack+"])

    def test_gfx1250_is_kept_untouched(self):
        result, _ = self.adjust(["gfx1250"])

        self.assertEqual(result, ["gfx1250"])

    def test_gfx1250_loses_an_explicitly_requested_xnack(self):
        """clang rejects 'gfx1250:xnack+' outright: xnack is not selectable there."""
        result, _ = self.adjust(["gfx1250:xnack+"])

        self.assertEqual(result, ["gfx1250"])

    def test_gfx1250_with_and_without_xnack_collapse_to_one_entry(self):
        result, _ = self.adjust(["gfx1250", "gfx1250:xnack+"])

        self.assertEqual(result, ["gfx1250"])

    def test_archs_without_xnack_are_dropped(self):
        result, messages = self.adjust(["gfx942", "gfx1100", "gfx1201"])

        self.assertEqual(result, ["gfx942:xnack+"])
        self.assertIn("gfx1100", messages)
        self.assertIn("gfx1201", messages)

    def test_full_default_target_list(self):
        """Matches the old effective set, except gfx1250 is no longer lost.

        The input is a copy of DEFAULT_GPUS in the root CMakeLists.txt and has
        to be updated by hand when a family is added there.
        """
        result, _ = self.adjust([
            "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950",
            "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1151",
            "gfx1200", "gfx1201", "gfx1250"])

        self.assertEqual(result, [
            "gfx906:xnack+", "gfx908:xnack+", "gfx90a:xnack+",
            "gfx942:xnack+", "gfx950:xnack+", "gfx1250"])

    def test_empty_result_is_fatal(self):
        with self.assertRaises(AssertionError) as cm:
            self.adjust(["gfx1100"])

        self.assertIn("ASAN requires a GPU target that supports xnack",
                      str(cm.exception))


if __name__ == "__main__":
    unittest.main()
