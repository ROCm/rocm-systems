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
    def _run_cmake_probe(self, compiler_body, probe_call, extra_defs=None,
                         check=True):
        """Shared cmake -P scaffold: fake compiler, probe script, result file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            compiler = root / "amdclang++"
            compiler.write_text("#!/bin/sh\n" + compiler_body)
            compiler.chmod(0o755)

            result_file = root / "result.txt"
            argv_file = root / "argv.txt"
            script = root / "probe.cmake"
            script.write_text(
                f'include("{DEVICE_COVERAGE_MODULE}")\n' + probe_call
            )
            env = os.environ.copy()
            env["ARGV_FILE"] = str(argv_file)
            cmd = [
                "cmake",
                f"-DCOMPILER={compiler}",
                f"-DRESULT_FILE={result_file}",
                "-P",
                str(script),
            ]
            for key, value in (extra_defs or {}).items():
                cmd.insert(-2, f"-D{key}={value}")
            completed = subprocess.run(
                cmd,
                check=check,
                env=env,
                capture_output=True,
                text=True,
            )
            result = result_file.read_text() if result_file.exists() else ""
            argv = argv_file.read_text() if argv_file.exists() else ""
            return root, result, argv, completed

    def run_host_probe(self, create_runtime):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = root / "libclang_rt.profile_rocm.a"
            printed = str(runtime) if create_runtime else runtime.name
            if create_runtime:
                runtime.touch()

            compiler_body = (
                'printf "%s\\n" "$@" >> "${ARGV_FILE:-/dev/null}"\n'
                f"printf '%s\\n' '{printed}'\n"
            )
            probe_call = (
                "rccl_find_host_rocm_profile_runtime("
                '"${COMPILER}" runtime)\n'
                'file(WRITE "${RESULT_FILE}" "${runtime}")\n'
            )
            _root, result, argv, _completed = self._run_cmake_probe(
                compiler_body, probe_call
            )
            return runtime, result, argv

    def test_probe_finds_selected_compiler_profile_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resource_dir = root / "resource"
            runtime = resource_dir / PRIMARY_RUNTIME_RELPATH
            runtime.parent.mkdir(parents=True)
            runtime.touch()
            compiler_body = (
                'printf "%s\\n" "$@" >> "${ARGV_FILE:-/dev/null}"\n'
                f"printf '%s\\n' '{resource_dir}'\n"
            )
            probe_call = (
                "rccl_find_device_profile_runtime("
                '"${COMPILER}" runtime reason)\n'
                'file(WRITE "${RESULT_FILE}" "${runtime}\\n${reason}")\n'
            )
            _probe_root, result, argv, _ = self._run_cmake_probe(
                compiler_body, probe_call
            )

        self.assertEqual(result, f"{runtime}\n")
        self.assertIn("--target=amdgcn-amd-amdhsa", argv)
        self.assertIn("-fprofile-instr-generate", argv)
        self.assertIn("-fcoverage-mapping", argv)
        self.assertIn("-print-resource-dir", argv)

    def test_probe_finds_linux_fallback_profile_runtime(self):
        # Only the second candidate (lib/linux/...-amdgcn.a) exists, exercising
        # the second loop iteration in rccl_find_device_profile_runtime.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resource_dir = root / "resource"
            runtime = resource_dir / FALLBACK_RUNTIME_RELPATH
            runtime.parent.mkdir(parents=True)
            runtime.touch()
            compiler_body = (
                'printf "%s\\n" "$@" >> "${ARGV_FILE:-/dev/null}"\n'
                f"printf '%s\\n' '{resource_dir}'\n"
            )
            probe_call = (
                "rccl_find_device_profile_runtime("
                '"${COMPILER}" runtime reason)\n'
                'file(WRITE "${RESULT_FILE}" "${runtime}\\n${reason}")\n'
            )
            _probe_root, result, argv, _ = self._run_cmake_probe(
                compiler_body, probe_call
            )

        self.assertEqual(result, f"{runtime}\n")
        self.assertIn("-print-resource-dir", argv)

    def test_probe_explains_missing_profile_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_dir = Path(temp_dir) / "resource"
            resource_dir.mkdir()
            compiler_body = (
                'printf "%s\\n" "$@" >> "${ARGV_FILE:-/dev/null}"\n'
                f"printf '%s\\n' '{resource_dir}'\n"
            )
            probe_call = (
                "rccl_find_device_profile_runtime("
                '"${COMPILER}" runtime reason)\n'
                'file(WRITE "${RESULT_FILE}" "${runtime}\\n${reason}")\n'
            )
            _root, result, _argv, _ = self._run_cmake_probe(
                compiler_body, probe_call
            )

        self.assertTrue(result.startswith("\nthe selected compiler"))
        self.assertIn("libclang_rt.profile.a", result)

    def test_probe_rejects_compiler_without_device_coverage_flags(self):
        compiler_body = (
            'printf "%s\\n" "$@" >> "${ARGV_FILE:-/dev/null}"\n'
            "case \"$*\" in *-fcoverage-mapping*) "
            "echo unsupported >&2; exit 2;; esac\n"
            "printf '%s\\n' /tmp/resource\n"
        )
        probe_call = (
            "rccl_find_device_profile_runtime("
            '"${COMPILER}" runtime reason)\n'
            'file(WRITE "${RESULT_FILE}" "${runtime}\\n${reason}")\n'
        )
        _root, result, argv, _ = self._run_cmake_probe(compiler_body, probe_call)

        self.assertTrue(result.startswith("\nthe selected compiler"))
        self.assertIn("rejected device coverage flags", result)
        self.assertIn("-fcoverage-mapping", argv)

    def test_host_probe_finds_companion_rocm_profile_runtime(self):
        runtime, result, argv = self.run_host_probe(create_runtime=True)

        self.assertEqual(result, str(runtime))
        self.assertIn(
            f"-print-file-name={Path(runtime).name}", argv.replace("\n", " ")
        )

    def test_host_probe_ignores_unresolved_runtime_name(self):
        _, result, _argv = self.run_host_probe(create_runtime=False)

        self.assertEqual(result, "")

    def _resolve(self, request, linker, rocm_ok, rocm_version, pretty,
                 cap_error, check=True):
        probe_call = (
            "rccl_resolve_full_coverage("
            '"${REQUEST}" "${LINKER}" "${ROCM_OK}" "${ROCM_VERSION}" '
            '"${ROCM_PRETTY}" "${CAP_ERROR}" resolved blocker)\n'
            'file(WRITE "${RESULT_FILE}" "${resolved}\\n${blocker}")\n'
        )
        return self._run_cmake_probe(
            "true\n",
            probe_call,
            extra_defs={
                "REQUEST": request,
                "LINKER": linker,
                "ROCM_OK": rocm_ok,
                "ROCM_VERSION": rocm_version,
                "ROCM_PRETTY": pretty,
                "CAP_ERROR": cap_error,
            },
            check=check,
        )

    def test_resolve_auto_enables_when_available(self):
        _root, result, _argv, completed = self._resolve(
            "AUTO", "ON", "7.15.0", "71500", "7.15.0", ""
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(result.splitlines()[0], "ON")
        self.assertEqual(result.splitlines()[1] if len(result.splitlines()) > 1 else "", "")

    def test_resolve_auto_falls_back_when_rocm_too_old(self):
        _root, result, _argv, completed = self._resolve(
            "AUTO", "ON", "7.1.0", "70100", "7.1.0", ""
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(result.splitlines()[0], "OFF")
        self.assertIn("ROCm 7.15", result)

    def test_resolve_auto_falls_back_when_device_linker_disabled(self):
        _root, result, _argv, completed = self._resolve(
            "AUTO", "OFF", "7.15.0", "71500", "7.15.0", ""
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(result.splitlines()[0], "OFF")
        self.assertIn("device linker is disabled", result)

    def test_resolve_auto_falls_back_when_rocm_version_unparsed(self):
        _root, result, _argv, completed = self._resolve(
            "AUTO", "ON", "", "0", "unknown", ""
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(result.splitlines()[0], "OFF")
        self.assertIn("could not be parsed", result)

    def test_resolve_auto_falls_back_when_compiler_lacks_coverage(self):
        cap_error = "rejected device coverage flags: unsupported"
        _root, result, _argv, completed = self._resolve(
            "AUTO", "ON", "7.15.0", "71500", "7.15.0", cap_error
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(result.splitlines()[0], "OFF")
        self.assertIn("does not support it", result)
        self.assertIn(cap_error, result)

    def test_resolve_on_is_fatal_when_blocked(self):
        _root, result, _argv, completed = self._resolve(
            "ON", "OFF", "7.15.0", "71500", "7.15.0", "",
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        stderr = " ".join(completed.stderr.split())
        self.assertIn(
            "ENABLE_FULL_COVERAGE=ON is unavailable because the device linker is disabled",
            stderr,
        )
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
