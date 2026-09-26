import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


RCCL_ROOT = Path(__file__).resolve().parents[4]
MODULE = RCCL_ROOT / "cmake" / "DeviceLinkerTargets.cmake"


@unittest.skipUnless(shutil.which("cmake"), "cmake not available on PATH")
class DeviceLinkerTargetsCMakeTest(unittest.TestCase):
    """dl_parse_gpu_targets feeds --offload-arch to every device and fat-object
    compile in DeviceLinker.cmake, and the bundler entry the loader matches
    against the agent. A dropped suffix here is a silently featureless build.
    """

    def _parse(self, targets):
        """Call dl_parse_gpu_targets in script mode; return (bare, flags, ids)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "out.txt"
            script = Path(temp_dir) / "probe.cmake"
            script.write_text(
                f'include("{MODULE}")\n'
                f'dl_parse_gpu_targets(TARGETS {targets}\n'
                '  BARE_VAR BARE FLAGS_VAR FLAGS ID_PREFIX ID_)\n'
                'set(_ids "")\n'
                'foreach(_b ${BARE})\n'
                '  list(APPEND _ids "${_b}=${ID_${_b}}")\n'
                'endforeach()\n'
                'file(WRITE "${OUT}" "${BARE}\\n${FLAGS}\\n${_ids}\\n")\n'
            )
            subprocess.run(["cmake", f"-DOUT={out}", "-P", str(script)],
                           check=True, capture_output=True, text=True)
            bare, flags, ids = out.read_text().splitlines()
        return bare.split(";"), flags.split(";"), ids.split(";")

    def test_suffix_reaches_offload_arch_but_not_the_bare_name(self):
        bare, flags, ids = self._parse("gfx942:xnack+")

        self.assertEqual(bare, ["gfx942"])
        self.assertEqual(flags, ["--offload-arch=gfx942:xnack+"])
        self.assertEqual(ids, ["gfx942=gfx942:xnack+"])

    def test_a_bare_target_stays_bare(self):
        bare, flags, ids = self._parse("gfx950")

        self.assertEqual(bare, ["gfx950"])
        self.assertEqual(flags, ["--offload-arch=gfx950"])
        self.assertEqual(ids, ["gfx950=gfx950"])

    def test_multiple_features_all_survive(self):
        _, flags, ids = self._parse("gfx950:sramecc+:xnack-")

        self.assertEqual(flags, ["--offload-arch=gfx950:sramecc+:xnack-"])
        self.assertEqual(ids, ["gfx950=gfx950:sramecc+:xnack-"])

    def test_a_mixed_multi_arch_list_keeps_each_target_its_own_id(self):
        """The ASAN case: input order is preserved and each processor keeps its
        own suffix, so the per-target loop cannot pick up a neighbour's."""
        bare, flags, ids = self._parse("gfx942:xnack+ gfx950:xnack+ gfx1201")

        self.assertEqual(bare, ["gfx942", "gfx950", "gfx1201"])
        self.assertEqual(flags, ["--offload-arch=gfx942:xnack+",
                                 "--offload-arch=gfx950:xnack+",
                                 "--offload-arch=gfx1201"])
        self.assertEqual(ids, ["gfx942=gfx942:xnack+",
                               "gfx950=gfx950:xnack+",
                               "gfx1201=gfx1201"])


if __name__ == "__main__":
    unittest.main()
