import os
import subprocess
import tempfile
import unittest
from pathlib import Path


RCCL_ROOT = Path(__file__).resolve().parents[4]
DEVICE_COVERAGE_MODULE = RCCL_ROOT / "cmake" / "DeviceCoverage.cmake"


class DeviceCoverageCMakeTest(unittest.TestCase):
    def run_probe(self, create_runtime, reject_coverage=False):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resource_dir = root / "resource"
            runtime = (
                resource_dir
                / "lib"
                / "amdgcn-amd-amdhsa"
                / "libclang_rt.profile.a"
            )
            runtime.parent.mkdir(parents=True)
            if create_runtime:
                runtime.touch()

            compiler = root / "amdclang++"
            compiler.write_text(
                "#!/bin/sh\n"
                + (
                    "case \"$*\" in *-fcoverage-mapping*) "
                    "echo unsupported >&2; exit 2;; esac\n"
                    if reject_coverage
                    else ""
                )
                + f"printf '%s\\n' '{resource_dir}'\n"
            )
            compiler.chmod(0o755)

            result_file = root / "result.txt"
            script = root / "probe.cmake"
            script.write_text(
                f'include("{DEVICE_COVERAGE_MODULE}")\n'
                "rccl_find_device_profile_runtime("
                '"${COMPILER}" runtime reason)\n'
                'file(WRITE "${RESULT_FILE}" "${runtime}\\n${reason}")\n'
            )
            subprocess.run(
                [
                    "cmake",
                    f"-DCOMPILER={compiler}",
                    f"-DRESULT_FILE={result_file}",
                    "-P",
                    str(script),
                ],
                check=True,
                env=os.environ.copy(),
            )
            return runtime, result_file.read_text()

    def test_probe_finds_selected_compiler_profile_runtime(self):
        runtime, result = self.run_probe(create_runtime=True)

        self.assertEqual(result, f"{runtime}\n")

    def test_probe_explains_missing_profile_runtime(self):
        _, result = self.run_probe(create_runtime=False)

        self.assertTrue(result.startswith("\nthe selected compiler"))
        self.assertIn("libclang_rt.profile.a", result)

    def test_probe_rejects_compiler_without_device_coverage_flags(self):
        _, result = self.run_probe(
            create_runtime=True,
            reject_coverage=True,
        )

        self.assertTrue(result.startswith("\nthe selected compiler"))
        self.assertIn("rejected device coverage flags", result)


if __name__ == "__main__":
    unittest.main()
