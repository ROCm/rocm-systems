import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


RCCL_ROOT = Path(__file__).resolve().parents[4]
DEVICE_COVERAGE_MODULE = RCCL_ROOT / "cmake" / "DeviceCoverage.cmake"

# The amdgcn profile-runtime candidates probed by DeviceCoverage.cmake, in the
# same priority order as RCCL_DEVICE_PROFILE_RUNTIME_RELPATHS.
PRIMARY_RUNTIME_RELPATH = "lib/amdgcn-amd-amdhsa/libclang_rt.profile.a"
FALLBACK_RUNTIME_RELPATH = "lib/linux/libclang_rt.profile-amdgcn.a"


@unittest.skipUnless(shutil.which("cmake"), "cmake not available on PATH")
class DeviceCoverageCMakeTest(unittest.TestCase):
    def run_probe(self, create_runtime, reject_coverage=False,
                  runtime_relpath=PRIMARY_RUNTIME_RELPATH):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resource_dir = root / "resource"
            runtime = resource_dir / runtime_relpath
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

    def run_host_probe(self, create_runtime):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = root / "libclang_rt.profile_rocm.a"
            if create_runtime:
                runtime.touch()

            compiler = root / "amdclang++"
            compiler.write_text(
                "#!/bin/sh\n"
                f"printf '%s\\n' '{runtime if create_runtime else runtime.name}'\n"
            )
            compiler.chmod(0o755)

            result_file = root / "result.txt"
            script = root / "probe.cmake"
            script.write_text(
                f'include("{DEVICE_COVERAGE_MODULE}")\n'
                "rccl_find_host_rocm_profile_runtime("
                '"${COMPILER}" runtime)\n'
                'file(WRITE "${RESULT_FILE}" "${runtime}")\n'
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

    def test_probe_finds_linux_fallback_profile_runtime(self):
        # Only the second candidate (lib/linux/...-amdgcn.a) exists, exercising
        # the second loop iteration in rccl_find_device_profile_runtime.
        runtime, result = self.run_probe(
            create_runtime=True,
            runtime_relpath=FALLBACK_RUNTIME_RELPATH,
        )

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

    def test_host_probe_finds_companion_rocm_profile_runtime(self):
        runtime, result = self.run_host_probe(create_runtime=True)

        self.assertEqual(result, str(runtime))

    def test_host_probe_ignores_unresolved_runtime_name(self):
        _, result = self.run_host_probe(create_runtime=False)

        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
