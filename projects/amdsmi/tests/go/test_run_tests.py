# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_tests


class CommandTests(unittest.TestCase):
    def test_selectors_and_checks(self) -> None:
        args = run_tests.parse_args(
            ["--run", "^TestCore", "--package", "./amdsmi", "--race", "--checkptr"]
        )
        command = run_tests.go_command(args=args, output=Path("telemetry"))
        self.assertIn("-race", command)
        self.assertIn("-gcflags=all=-d=checkptr=2", command)
        self.assertEqual(command[-4:], ["-run", "^TestCore", "-v", "./amdsmi"])
        args = run_tests.parse_args(["--asan"])
        self.assertEqual(args.cc, "gcc")
        self.assertIn("-asan", run_tests.go_command(args=args, output=Path("telemetry")))

    def test_invalid_options(self) -> None:
        for arguments in (
            ["--native"],
            ["--native", "--include-dir", "/include"],
            ["--native", "--library-dir", "/lib"],
            ["--library-dir", "/lib"],
            ["--vet", "--build-example"],
            ["--race", "--asan"],
            ["--asan", "--cc", "clang"],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    run_tests.parse_args(arguments)
                self.assertEqual(error.exception.code, 2)


class EnvironmentTests(unittest.TestCase):
    def test_environment_is_controlled(self) -> None:
        removed = """
            LD_PRELOAD LD_AUDIT LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
            GOFLAGS GOOS GOARCH CGO_CPPFLAGS CGO_CXXFLAGS CGO_FFLAGS GOCACHEPROG
            CGO_CFLAGS_ALLOW CGO_CFLAGS_DISALLOW CGO_LDFLAGS_ALLOW CGO_LDFLAGS_DISALLOW
        """.split()
        expected = {
            "GOTOOLCHAIN": "local",
            "GOENV": "off",
            "GOWORK": "off",
            "GOPROXY": "off",
            "GOSUMDB": "off",
            "GOVCS": "*:off",
            "GONOPROXY": "none",
            "GOPRIVATE": "",
            "CGO_ENABLED": "1",
            "GODEBUG": "cgocheck=1",
            "CC": "custom-cc",
        }
        polluted = dict.fromkeys(removed + list(expected), "/wrong")
        polluted.update(CGO_CFLAGS="-I/wrong", CGO_LDFLAGS="-L/wrong", GOEXPERIMENT="wrong")
        with tempfile.TemporaryDirectory(prefix="amdsmi-agent-runner-test-") as directory:
            root = Path(directory)
            with patch.dict(os.environ, polluted):
                env = run_tests.make_env(
                    root=root,
                    include_dir=root / "include space",
                    library_dir=root / "lib space",
                    cc="custom-cc",
                    cgocheck2=True,
                )
            for key in removed:
                self.assertNotIn(key, env)
            for key, value in expected.items():
                self.assertEqual(env[key], value)
            self.assertEqual(env["GOEXPERIMENT"], "cgocheck2")
            self.assertEqual(shlex.split(env["CGO_CFLAGS"]), ["-I" + str(root / "include space")])
            self.assertEqual(
                shlex.split(env["CGO_LDFLAGS"]),
                ["-L" + str(root / "lib space"), "-Wl,-rpath," + str(root / "lib space")],
            )
            self.assertEqual(env["LD_LIBRARY_PATH"], str(root / "lib space"))
            self.assertTrue(Path(env["GOTMPDIR"]).is_dir())
            self.assertEqual(Path(env["GOMODCACHE"]), root / "go-mod-cache")

    def test_private_cache_is_reused_without_reusing_fixture_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="amdsmi-agent-runner-test-") as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            with (
                patch("run_tests.tempfile.gettempdir", return_value=directory),
                patch.dict(os.environ, {"GOEXPERIMENT": "wrong"}),
            ):
                envs = [
                    run_tests.make_env(
                        root=path, include_dir=root, library_dir=path, cc="cc", cgocheck2=False
                    )
                    for path in (first, second)
                ]
            for env in envs:
                self.assertNotIn("GOEXPERIMENT", env)
            self.assertEqual(envs[0]["GOCACHE"], envs[1]["GOCACHE"])
            cache = Path(envs[0]["GOCACHE"])
            self.assertEqual(cache.stat().st_uid, os.getuid())
            self.assertEqual(cache.stat().st_mode & 0o777, 0o700)
            self.assertNotEqual(envs[0]["CGO_LDFLAGS"], envs[1]["CGO_LDFLAGS"])
            self.assertNotEqual(envs[0]["GOTMPDIR"], envs[1]["GOTMPDIR"])

    def test_cache_rejects_symlinks_and_shared_permissions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="amdsmi-agent-runner-test-") as directory:
            root = Path(directory)
            with patch("run_tests.tempfile.gettempdir", return_value=directory):
                cache = run_tests.go_cache_dir()
                cache.chmod(0o777)
                with self.assertRaisesRegex(ValueError, "private"):
                    run_tests.go_cache_dir()
                cache.chmod(0o700)
                cache.rmdir()
                cache.symlink_to(root, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "private"):
                    run_tests.go_cache_dir()


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="amdsmi-agent-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "amd_smi").mkdir()
        (self.root / "amd_smi" / "amdsmi.h").touch()
        self.args = run_tests.parse_args([])
        self.env = {"PATH": "/tools", "GOTOOLCHAIN": "local", "GOPROXY": "off"}

    def validate(self) -> None:
        run_tests.validate_tools(
            args=self.args, env=self.env, include_dir=self.root, library_dir=self.root
        )

    def test_missing_tools(self) -> None:
        for missing in ("go", "cc"):
            with self.subTest(missing=missing), patch("run_tests.shutil.which") as which:
                which.side_effect = lambda program, **kwargs: (
                    None if program == missing else program
                )
                with patch("run_tests.subprocess.run") as run:
                    with self.assertRaisesRegex(FileNotFoundError, missing):
                        self.validate()
                    run.assert_not_called()

    def test_missing_native_inputs(self) -> None:
        self.args.native = True
        with (
            patch("run_tests.shutil.which", return_value="tool"),
            patch("run_tests.subprocess.run") as run,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "shared library"):
                self.validate()
            (self.root / "amd_smi" / "amdsmi.h").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "header"):
                self.validate()
            run.assert_not_called()

    def test_requires_linux(self) -> None:
        with patch("run_tests.sys.platform", "darwin"):
            with self.assertRaisesRegex(ValueError, "Linux"):
                self.validate()

    def test_local_version_gates(self) -> None:
        cases = [
            ("go1.19.13", False, False),
            ("go1.20.14", False, True),
            ("go1.20.14", True, False),
            ("go1.21.0", True, True),
            ("devel unknown", False, False),
        ]
        for version, experiment, accepted in cases:
            self.args.cgocheck2 = experiment
            result = subprocess.CompletedProcess(
                ["go", "version"], 0, stdout="go version " + version + " linux/amd64"
            )
            with (
                self.subTest(version=version, experiment=experiment),
                patch("run_tests.shutil.which", return_value="tool"),
                patch("run_tests.subprocess.run", return_value=result),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if accepted:
                    self.validate()
                else:
                    with self.assertRaisesRegex(ValueError, "local Go toolchain"):
                        self.validate()

    def test_native_checks_never_compile_fixture_or_run_example(self) -> None:
        for options, action in (
            (["--run", "^TestNativeVersion$", "--checkptr"], "test"),
            (["--vet", "--race"], "vet"),
            (["--build-example", "--asan"], "build"),
        ):
            args = run_tests.parse_args(
                ["--native", "--include-dir", str(self.root), "--library-dir", str(self.root)]
                + options
            )
            with (
                self.subTest(options=options),
                patch("run_tests.validate_tools"),
                patch("run_tests.subprocess.run") as run,
            ):
                run_tests.run_fixture(project=self.root, args=args)
                run.assert_called_once()
                command = run.call_args.args[0]
                self.assertEqual(command[:2], ["go", action])
                self.assertNotIn("-tags=amdsmi_mock", command)
                self.assertEqual(run.call_args.kwargs["env"]["LD_LIBRARY_PATH"], str(self.root))

    def test_no_sources_fails_before_compilation(self) -> None:
        with patch("run_tests.validate_tools"), patch("run_tests.subprocess.run") as run:
            with self.assertRaisesRegex(FileNotFoundError, "fixture sources"):
                run_tests.run_fixture(project=self.root, args=self.args)
            run.assert_not_called()

    def test_failures_return_nonzero(self) -> None:
        failures = [
            FileNotFoundError("missing"),
            ValueError("invalid"),
            subprocess.CalledProcessError(9, "go"),
        ]
        for failure in failures:
            with self.subTest(failure=failure), patch("run_tests.sys.argv", ["run_tests.py"]):
                with patch("run_tests.run_fixture", side_effect=failure):
                    with contextlib.redirect_stderr(io.StringIO()) as output:
                        self.assertNotEqual(run_tests.main(), 0)
                    self.assertIn(str(failure), output.getvalue())


class SmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        if sys.platform != "linux":
            self.skipTest("Go/CGO smoke tests require Linux")
        for tool in ("go", "gcc"):
            if shutil.which(tool) is None:
                self.skipTest("Go/CGO smoke tests require " + tool)

    def run_cli(self, *arguments: str, success: bool = True) -> str:
        result = subprocess.run(
            [sys.executable, "-B", run_tests.__file__, *arguments],
            cwd=tempfile.gettempdir(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode == 0, success, result.stdout)
        return result.stdout

    def assert_tests_passed(self, output: str, *names: str) -> None:
        self.assertNotIn("[no tests to run]", output)
        self.assertNotIn("--- SKIP:", output)
        for name in names:
            self.assertIn("--- PASS: " + name + " (", output, output)
        self.assertRegex(
            output, r"(?m)^ok\s+github.com/ROCm/rocm-systems/projects/amdsmi/go/amdsmi\s"
        )

    def test_default_fixture_tests(self) -> None:
        self.assert_tests_passed(
            self.run_cli(),
            "TestPartitionIDArray",
            "TestCoreLifecycleReferences",
            "TestCoreDiscovery",
            "TestIdentityASIC",
            "TestTelemetryTemperature",
            "TestECCCount",
            "TestNativeVersion",
        )

    def test_asan_partition_and_lifecycle(self) -> None:
        self.assert_tests_passed(
            self.run_cli(
                "--asan",
                "--cc",
                "gcc",
                "--package",
                "./amdsmi",
                "--run",
                "^(TestPartitionIDArray|TestCoreLifecycle)",
            ),
            "TestPartitionIDArray",
            "TestCoreLifecycleCompatibleVersion",
            "TestCoreLifecycleReferences",
            "TestCoreLifecycleSerialization",
        )

    def test_vet_and_example_build(self) -> None:
        for action in ("--vet", "--build-example"):
            with self.subTest(action=action):
                output = self.run_cli(action, "--cc", "gcc")
                self.assertNotIn("AMD SMI ", output)

    def test_compiler_failure_returns_nonzero(self) -> None:
        output = self.run_cli("--cc", "/bin/false", success=False)
        self.assertIn("returned non-zero exit status", output)
        self.assertNotIn("--- PASS:", output)
        self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()
