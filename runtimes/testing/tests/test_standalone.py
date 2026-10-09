# SPDX-License-Identifier: MIT

"""Build and execute real first-slice consumers; no device is needed for this harness."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

SOURCE = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")
CTEST = shutil.which("ctest")


class StandaloneTests(unittest.TestCase):
    @classmethod
    def command(cls, *arguments, success=True, environment=None):
        result = subprocess.run(
            [str(argument) for argument in arguments],
            capture_output=True,
            text=True,
            env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"} | (environment or {}),
        )
        if success and result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)
        if not success and result.returncode == 0:
            raise AssertionError(f"Unexpected success: {arguments}\n{result.stdout}")
        return result

    @classmethod
    def configure(cls, source, build, *settings, success=True):
        return cls.command(
            CMAKE, "-S", source, "-B", build, "-G", "Ninja", *settings, success=success
        )

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="runtime-test-contracts-")
        cls.root = Path(cls.temporary.name)
        cls.addClassCleanup(cls.temporary.cleanup)

        headers = cls.root / "header build"
        installed_headers = cls.root / "original headers"
        cls.configure(SOURCE / "api-headers", headers)
        cls.command(CMAKE, "--install", headers, "--prefix", installed_headers)
        cls.headers = cls.root / "moved headers"
        installed_headers.rename(cls.headers)
        cls.prefixes = {}
        cls.builds = {}
        for family in ("runtime-api", "ddi", "kfd"):
            build = cls.root / f"{family} build"
            settings = [f"-DROCM_RUNTIMES_TEST_FAMILIES={family}"]
            if family == "runtime-api":
                settings.append(
                    f"-DROCmRuntimeAPI_DIR={cls.headers}/lib/cmake/ROCmRuntimeAPI"
                )
            else:
                settings.append("-DROCmRuntimeAPI_DIR=/absent/header-package")
            if family != "ddi":
                settings += [
                    "-DROCM_RUNTIMES_CARGO=/absent/cargo",
                    "-DROCM_RUNTIMES_RUSTC=/absent/rustc",
                    "-DROCM_RUNTIMES_LLD=/absent/lld",
                ]
            cls.configure(SOURCE / "testing", build, *settings)
            cls.command(CMAKE, "--build", build)
            prefix = cls.root / f"moved {family} prefix"
            cls.command(CMAKE, "--install", build, "--prefix", prefix)
            cls.prefixes[family] = prefix
            cls.builds[family] = build
        # The API executable was already built before this producer existed.
        cls.runtime = cls.root / "separate runtime"
        cls.configure(SOURCE, cls.runtime, "-DBUILD_TESTING=OFF")
        cls.command(CMAKE, "--build", cls.runtime)

    def run_suite(
        self, family, suite, *, success=True, prefix=None, results=None, **inputs
    ):
        prefix = prefix or self.prefixes[family]
        results = results or self.root / f"results {self.id().split('.')[-1]} {suite}"
        result = self.command(
            CTEST,
            "-S",
            prefix / "cmake/RunTests.cmake",
            "-V",
            f"-DTEST_SUITE:STRING={suite}",
            f"-DRESULTS_DIRECTORY:PATH={results}",
            *(f"-D{name}:STRING={value}" for name, value in inputs.items()),
            success=success,
            # No compiler, Cargo, Python or CMake executable is available by PATH.
            environment={"PATH": str(self.root / "no tools")},
        )
        return result, results

    def test_real_host_contracts_from_relocated_prefixes(self):
        for family, suite in (("runtime-api", "amdf-host"), ("ddi", "ddi-host")):
            with self.subTest(family=family):
                inputs = (
                    {"RUNTIME_PREFIX": self.runtime} if family == "runtime-api" else {}
                )
                prefix = self.prefixes[family]
                paths = [prefix, *prefix.rglob("*")]
                modes = {path: path.stat().st_mode for path in paths}
                build = self.builds[family]
                hidden_build = build.with_name(build.name + " unavailable")
                try:
                    build.rename(hidden_build)
                    for path, mode in modes.items():
                        path.chmod(mode & ~0o222)
                    _, results = self.run_suite(family, suite, **inputs)
                    report = ET.parse(results / "junit.xml").getroot()
                    self.assertEqual(report.attrib["tests"], "1")
                    self.assertEqual(report.attrib["failures"], "0")
                    self.assertEqual(report.attrib["skipped"], "0")
                    self.assertIn("status=passed", (results / "run.txt").read_text())
                finally:
                    for path, mode in modes.items():
                        path.chmod(mode)
                    hidden_build.rename(build)

    def test_native_api_has_no_runtime_binary_dependency(self):
        result = self.command(
            "readelf",
            "-d",
            self.prefixes["runtime-api"] / "bin/runtime-api/amdf-host-contract",
        )
        self.assertNotIn("libamdf", result.stdout)
        self.assertNotIn("libhsa", result.stdout)

    def test_ddi_build_omits_frontend_artifacts_and_native_discovery(self):
        build = self.builds["ddi"]
        names = [path.name for path in (build / "ddi/rocddi/cargo").rglob("*")]
        self.assertFalse(any("libamdf" in name or "libhsa" in name for name in names))
        self.assertNotIn("CMAKE_C_COMPILER:", (build / "CMakeCache.txt").read_text())
        self.assertEqual(
            [path.name for path in (self.prefixes["ddi"] / "bin").iterdir()], ["ddi"]
        )

    def test_public_header_package_is_scoped(self):
        self.assertEqual(
            [path.name for path in (self.headers / "include").iterdir()], ["amdf"]
        )
        targets = (
            self.headers / "lib/cmake/ROCmRuntimeAPI/ROCmRuntimeAPITargets.cmake"
        ).read_text()
        self.assertNotIn(str(SOURCE), targets)
        self.assertNotIn(str(self.root), targets)

    def test_unknown_empty_and_unsupported_selections_fail(self):
        for index, settings in enumerate(
            [
                ["-DROCM_RUNTIMES_TEST_FAMILIES="],
                ["-DROCM_RUNTIMES_TEST_FAMILIES=unknown"],
                ["-DBUILD_TESTING=OFF"],
                ["-DROCM_RUNTIMES_TEST_FAMILIES=integration"],
                ["-DCMAKE_SYSTEM_NAME=Windows"],
            ]
        ):
            with self.subTest(settings=settings):
                self.configure(
                    SOURCE / "testing",
                    self.root / f"invalid selection {index}",
                    *settings,
                    success=False,
                )

    def test_invalid_explicit_api_package_cannot_fall_back(self):
        result = self.configure(
            SOURCE / "testing",
            self.root / "invalid explicit package",
            "-DROCmRuntimeAPI_DIR=/absent/package",
            f"-DCMAKE_PREFIX_PATH={self.headers}",
            success=False,
        )
        self.assertIn("Explicit ROCmRuntimeAPI_DIR", result.stderr)

    def test_family_reselection_cannot_leave_stale_artifacts(self):
        build = self.root / "selection ownership"
        self.configure(SOURCE / "testing", build, "-DROCM_RUNTIMES_TEST_FAMILIES=kfd")
        result = self.configure(
            SOURCE / "testing",
            build,
            "-DROCM_RUNTIMES_TEST_FAMILIES=ddi",
            success=False,
        )
        self.assertIn("fresh build directory", result.stderr)

    def test_invalid_explicit_ddi_inputs_fail(self):
        for index, setting in enumerate(
            [
                "-DROCM_RUNTIMES_CARGO_CONFIG=/absent/config.toml",
                "-DROCM_RUNTIMES_CARGO_HOME=/absent/cargo-home",
                "-DROCM_RUNTIMES_CARGO=/absent/cargo",
                "-DROCM_RUNTIMES_RUSTC=/absent/rustc",
                "-DCMAKE_BUILD_TYPE=Unknown",
            ]
        ):
            with self.subTest(setting=setting):
                self.configure(
                    SOURCE / "testing",
                    self.root / f"invalid DDI {index}",
                    "-DROCM_RUNTIMES_TEST_FAMILIES=ddi",
                    setting,
                    success=False,
                )

    def test_combined_selection_has_one_owner_per_suite(self):
        build = self.root / "combined build"
        self.configure(
            SOURCE / "testing",
            build,
            "-DROCM_RUNTIMES_TEST_FAMILIES=runtime-api;ddi;kfd;ddi",
            f"-DROCmRuntimeAPI_DIR={self.headers}/lib/cmake/ROCmRuntimeAPI",
        )
        self.command(CMAKE, "--build", build)
        prefix = build / "tests"
        self.assertEqual(
            sorted(path.stem for path in (prefix / "cmake/suites").iterdir()),
            ["amdf-host", "ddi-host", "kfd"],
        )
        self.run_suite("ddi", "ddi-host", prefix=prefix)

    def test_nonexecutable_required_test_is_not_a_pass(self):
        prefix = self.root / "nonexecutable prefix"
        shutil.copytree(self.prefixes["ddi"], prefix)
        executable = prefix / "bin/ddi/rocddi-host-contract"
        executable.chmod(executable.stat().st_mode & ~0o111)
        _, results = self.run_suite("ddi", "ddi-host", prefix=prefix, success=False)
        self.assertIn("status=failed", (results / "run.txt").read_text())

    def test_unknown_suite_and_missing_runtime_fail_with_records(self):
        for suite in ("missing", "amdf-host"):
            with self.subTest(suite=suite):
                _, results = self.run_suite("runtime-api", suite, success=False)
                self.assertIn("status=failed", (results / "run.txt").read_text())

    def test_ddi_and_kfd_reject_runtime_substitution(self):
        for family, suite in (("ddi", "ddi-host"), ("kfd", "kfd")):
            with self.subTest(family=family):
                result, _ = self.run_suite(
                    family, suite, success=False, RUNTIME_PREFIX=self.runtime
                )
                self.assertIn("does not consume RUNTIME_PREFIX", result.stderr)

    def test_results_cannot_mutate_test_or_runtime_prefix(self):
        alias = self.root / "test prefix alias"
        alias.symlink_to(self.prefixes["runtime-api"], target_is_directory=True)
        for prefix in (self.prefixes["runtime-api"], alias, self.runtime):
            with self.subTest(prefix=prefix):
                results = prefix / "nested" / "must not exist"
                result, _ = self.run_suite(
                    "runtime-api",
                    "amdf-host",
                    success=False,
                    results=results,
                    RUNTIME_PREFIX=self.runtime,
                )
                self.assertIn("outside the completed", result.stderr)
                self.assertFalse((prefix / "nested").exists())

    def test_existing_results_are_preserved(self):
        results = self.root / "retained results"
        results.mkdir()
        sentinel = results / "keep.txt"
        sentinel.write_text("prior evidence")
        self.run_suite("ddi", "ddi-host", success=False, results=results)
        self.assertEqual(sentinel.read_text(), "prior evidence")
        self.assertEqual(list(results.iterdir()), [sentinel])

    def test_missing_artifacts_cannot_pass(self):
        for index, relative in enumerate(
            [
                "bin/ddi/rocddi-host-contract",
                "cmake/suites/ddi-host.cmake",
                "share/rocm-runtime-tests/ddi.txt",
            ]
        ):
            with self.subTest(artifact=relative):
                prefix = self.root / f"incomplete prefix {index}"
                shutil.copytree(self.prefixes["ddi"], prefix)
                (prefix / relative).unlink()
                _, results = self.run_suite(
                    "ddi",
                    "ddi-host",
                    prefix=prefix,
                    success=False,
                    results=self.root / f"incomplete results {index}",
                )
                self.assertIn("status=failed", (results / "run.txt").read_text())

    def test_loader_rejects_missing_symbols_and_incompatible_negotiation(self):
        # These are invalid providers used only to check loader failure paths.
        # They never supply positive runtime qualification evidence.
        for index, source in enumerate(
            [
                "int not_amdf(void) { return 0; }\n",
                "#include <amdf/amdf.h>\namdf_status_t amdf_query_api(amdf_abi_version_t a, "
                "amdf_abi_version_t b, const amdf_api_t **out) { return "
                "amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH); }\n",
            ]
        ):
            with self.subTest(provider=index):
                prefix = self.root / f"invalid provider {index}"
                (prefix / "lib").mkdir(parents=True)
                source_file = prefix / "provider.c"
                source_file.write_text(source)
                self.command(
                    "cc",
                    "-shared",
                    "-fPIC",
                    "-I",
                    self.headers / "include",
                    source_file,
                    "-o",
                    prefix / "lib/libamdf.so",
                )
                _, results = self.run_suite(
                    "runtime-api",
                    "amdf-host",
                    success=False,
                    results=self.root / f"invalid provider results {index}",
                    RUNTIME_PREFIX=prefix,
                )
                report = ET.parse(results / "junit.xml").getroot()
                self.assertEqual(report.attrib["failures"], "1")

    def test_loader_rejects_entry_point_from_a_dependency(self):
        prefix = self.root / "facade provider"
        (prefix / "lib").mkdir(parents=True)
        source = prefix / "facade.c"
        source.write_text("int not_amdf(void) { return 0; }\n")
        self.command(
            "cc",
            "-shared",
            "-fPIC",
            source,
            "-Wl,--no-as-needed",
            "-L",
            self.runtime / "lib",
            f"-Wl,-rpath,{self.runtime / 'lib'}",
            "-lhsa-runtime64",
            "-o",
            prefix / "lib/libamdf.so",
        )
        _, results = self.run_suite(
            "runtime-api", "amdf-host", success=False, RUNTIME_PREFIX=prefix
        )
        self.assertIn(
            "resolved outside selected image", (results / "junit.xml").read_text()
        )

    @unittest.skipIf(
        Path("/dev/kfd").exists(), "Missing-device check requires /dev/kfd to be absent"
    )
    def test_missing_kfd_is_a_failure_not_a_skip(self):
        _, results = self.run_suite("kfd", "kfd", success=False)
        report = ET.parse(results / "junit.xml").getroot()
        self.assertEqual(report.attrib["tests"], "1")
        self.assertEqual(report.attrib["failures"], "1")
        self.assertEqual(report.attrib["skipped"], "0")
        self.assertIn(
            "Required KFD access unavailable", (results / "junit.xml").read_text()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
