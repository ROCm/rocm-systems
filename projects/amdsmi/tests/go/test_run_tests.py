# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_tests


class CommandTests(unittest.TestCase):
    def test_default_uses_mock_tag(self) -> None:
        args = run_tests.parse_args([])
        self.assertEqual(
            run_tests.go_command(args=args, output=Path("telemetry")),
            [
                "go",
                "test",
                "-tags=amdsmi_mock",
                "-count=1",
                "-timeout=120s",
                "-run",
                ".",
                "-v",
                "./...",
            ],
        )

    def test_selectors_and_checks(self) -> None:
        args = run_tests.parse_args(
            ["--run", "^TestCore", "--package", "./amdsmi", "--race", "--checkptr"]
        )
        command = run_tests.go_command(args=args, output=Path("telemetry"))
        self.assertIn("-race", command)
        self.assertIn("-gcflags=all=-d=checkptr=2", command)
        self.assertEqual(command[-4:], ["-run", "^TestCore", "-v", "./amdsmi"])

    def test_asan_uses_gcc_and_go_instrumentation(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                args = run_tests.parse_args(["--asan"])
            except SystemExit:
                self.fail("runner does not accept --asan")
        self.assertEqual(args.cc, "gcc")
        self.assertIn("-asan", run_tests.go_command(args=args, output=Path("telemetry")))

    def test_native_has_no_mock_tag(self) -> None:
        args = run_tests.parse_args(
            ["--native", "--include-dir", "/include", "--library-dir", "/lib", "--vet"]
        )
        self.assertEqual(
            run_tests.go_command(args=args, output=Path("telemetry")), ["go", "vet", "./..."]
        )

    def test_example_is_only_built(self) -> None:
        args = run_tests.parse_args(["--build-example"])
        self.assertEqual(
            run_tests.go_command(args=args, output=Path("/tmp/telemetry")),
            ["go", "build", "-tags=amdsmi_mock", "-o", "/tmp/telemetry", "./examples/telemetry"],
        )

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
        removed = (
            "LD_PRELOAD",
            "LD_AUDIT",
            "LIBRARY_PATH",
            "CPATH",
            "C_INCLUDE_PATH",
            "CPLUS_INCLUDE_PATH",
            "GOFLAGS",
            "GOOS",
            "GOARCH",
            "CGO_CPPFLAGS",
            "CGO_CXXFLAGS",
            "CGO_FFLAGS",
            "CGO_CFLAGS_ALLOW",
            "CGO_CFLAGS_DISALLOW",
            "CGO_LDFLAGS_ALLOW",
            "CGO_LDFLAGS_DISALLOW",
            "GOCACHEPROG",
        )
        polluted = dict.fromkeys(removed, "/wrong")
        polluted.update(
            {
                "GOTOOLCHAIN": "auto",
                "CGO_CFLAGS": "-I/wrong",
                "CGO_LDFLAGS": "-L/wrong",
                "GOEXPERIMENT": "wrong",
                "GODEBUG": "cgocheck=0",
            }
        )
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
            for key, value in {
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
                "GOEXPERIMENT": "cgocheck2",
                "CC": "custom-cc",
            }.items():
                self.assertEqual(env[key], value)
            self.assertEqual(shlex.split(env["CGO_CFLAGS"]), ["-I" + str(root / "include space")])
            self.assertEqual(
                shlex.split(env["CGO_LDFLAGS"]),
                ["-L" + str(root / "lib space"), "-Wl,-rpath," + str(root / "lib space")],
            )
            self.assertEqual(env["LD_LIBRARY_PATH"], str(root / "lib space"))
            self.assertTrue(Path(env["GOTMPDIR"]).is_dir())
            self.assertEqual(Path(env["GOMODCACHE"]), root / "go-mod-cache")

    def test_default_has_no_experiment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="amdsmi-agent-runner-test-") as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"GOEXPERIMENT": "wrong"}):
                env = run_tests.make_env(
                    root=root, include_dir=root, library_dir=root, cc="cc", cgocheck2=False
                )
            self.assertNotIn("GOEXPERIMENT", env)

    def test_private_cache_is_reused_without_reusing_fixture_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="amdsmi-agent-runner-test-") as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            with patch("run_tests.tempfile.gettempdir", return_value=directory):
                envs = [
                    run_tests.make_env(
                        root=path, include_dir=root, library_dir=path, cc="cc", cgocheck2=False
                    )
                    for path in (first, second)
                ]
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

    def test_missing_header(self) -> None:
        (self.root / "amd_smi" / "amdsmi.h").unlink()
        with (
            patch("run_tests.shutil.which", return_value="tool"),
            patch("run_tests.subprocess.run") as run,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "header"):
                self.validate()
            run.assert_not_called()

    def test_missing_native_library(self) -> None:
        self.args.native = True
        with (
            patch("run_tests.shutil.which", return_value="tool"),
            patch("run_tests.subprocess.run") as run,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "shared library"):
                self.validate()
            run.assert_not_called()

    def test_requires_linux(self) -> None:
        with patch("run_tests.sys.platform", "darwin"), patch("run_tests.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "Linux"):
                self.validate()
            run.assert_not_called()

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
            with self.subTest(version=version, experiment=experiment):
                with patch("run_tests.shutil.which", return_value="tool"):
                    with patch("run_tests.subprocess.run", return_value=result) as run:
                        with contextlib.redirect_stdout(io.StringIO()):
                            if accepted:
                                self.validate()
                            else:
                                with self.assertRaisesRegex(ValueError, "local Go toolchain"):
                                    self.validate()
                        self.assertEqual(run.call_args[0][0], ["go", "version"])
                        self.assertIs(run.call_args[1]["env"], self.env)
                        self.assertTrue(run.call_args[1]["check"])

    def test_version_subprocess_failure_propagates(self) -> None:
        with patch("run_tests.shutil.which", return_value="tool"):
            with patch(
                "run_tests.subprocess.run", side_effect=subprocess.CalledProcessError(7, "go")
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    self.validate()


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="amdsmi-agent-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.include = self.project / "include"
        (self.include / "amd_smi").mkdir(parents=True)
        (self.include / "amd_smi" / "amdsmi.h").touch()
        self.sources = self.project / "go" / "amdsmi" / "testdata"
        self.sources.mkdir(parents=True)
        for name in ("mock_core.c", "mock_identity.c"):
            (self.sources / name).touch()
        self.version = subprocess.CompletedProcess(
            ["go", "version"], 0, stdout="go version go1.24.1 linux/amd64"
        )

    def test_asan_instruments_fixture_and_go_with_gcc(self) -> None:
        args = run_tests.parse_args(["--asan"])
        with patch("run_tests.shutil.which", return_value="tool"):
            with patch("run_tests.subprocess.run", return_value=self.version) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_tests.run_fixture(project=self.project, args=args)
        compiler, go = run.call_args_list[1:]
        self.assertEqual(compiler[0][0][0], "gcc")
        self.assertIn("-fsanitize=address", compiler[0][0])
        self.assertIn("-fno-omit-frame-pointer", compiler[0][0])
        self.assertIn("-asan", go[0][0])
        self.assertEqual(go[1]["env"]["CC"], "gcc")

    def test_fixture_compilation_precedes_go_test(self) -> None:
        args = run_tests.parse_args(["--cc", "custom-cc", "--run", "^TestNativeVersion$"])
        with patch("run_tests.shutil.which", return_value="tool"):
            with patch("run_tests.subprocess.run", return_value=self.version) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_tests.run_fixture(project=self.project, args=args)
        self.assertEqual(run.call_count, 3)
        version, compiler, go = run.call_args_list
        self.assertEqual(version[0][0], ["go", "version"])
        command = compiler[0][0]
        self.assertEqual(command[0], "custom-cc")
        self.assertIn("-Werror", command)
        self.assertIn("-I" + str(self.include), command)
        self.assertEqual(
            command[-2:], [str(path) for path in sorted(self.sources.glob("mock_*.c"))]
        )
        library = Path(command[command.index("-o") + 1])
        self.assertEqual(library.name, "libamd_smi.so")
        self.assertNotEqual(library.parent, self.project)
        self.assertIs(version[1]["env"], compiler[1]["env"])
        self.assertIs(compiler[1]["env"], go[1]["env"])
        self.assertEqual(go[1]["env"]["CC"], "custom-cc")
        self.assertEqual(go[1]["env"]["LD_LIBRARY_PATH"], str(library.parent))
        self.assertEqual(go[1]["cwd"], self.project / "go")
        self.assertEqual(go[0][0][0:3], ["go", "test", "-tags=amdsmi_mock"])
        self.assertTrue(all(call[1]["check"] for call in run.call_args_list))
        self.assertFalse(library.parent.exists())

    def test_native_build_does_not_compile_fixture_or_execute_example(self) -> None:
        library = self.project / "native"
        library.mkdir()
        (library / "libamd_smi.so").touch()
        args = run_tests.parse_args(
            [
                "--native",
                "--include-dir",
                str(self.include),
                "--library-dir",
                str(library),
                "--build-example",
            ]
        )
        with patch("run_tests.shutil.which", return_value="tool"):
            with patch("run_tests.subprocess.run", return_value=self.version) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_tests.run_fixture(project=self.project, args=args)
        self.assertEqual(run.call_count, 2)
        command = run.call_args_list[1][0][0]
        self.assertEqual(command[:3], ["go", "build", "-o"])
        self.assertEqual(command[-1], "./examples/telemetry")
        self.assertNotIn("-tags=amdsmi_mock", command)

    def test_no_sources_fails_before_compilation(self) -> None:
        for source in self.sources.glob("mock_*.c"):
            source.unlink()
        with patch("run_tests.shutil.which", return_value="tool"):
            with patch("run_tests.subprocess.run", return_value=self.version) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(FileNotFoundError, "fixture sources"):
                        run_tests.run_fixture(project=self.project, args=run_tests.parse_args([]))
        self.assertEqual(run.call_count, 1)

    def test_compilation_failure_stops_go(self) -> None:
        failure = subprocess.CalledProcessError(4, "cc")
        with patch("run_tests.shutil.which", return_value="tool"):
            with patch("run_tests.subprocess.run", side_effect=[self.version, failure]) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(subprocess.CalledProcessError) as error:
                        run_tests.run_fixture(project=self.project, args=run_tests.parse_args([]))
        self.assertIs(error.exception, failure)
        self.assertEqual(run.call_count, 2)


class MainTests(unittest.TestCase):
    def test_success(self) -> None:
        with patch("run_tests.sys.argv", ["run_tests.py"]), patch("run_tests.run_fixture") as run:
            self.assertEqual(run_tests.main(), 0)
        self.assertEqual(run.call_args[1]["project"], Path(run_tests.__file__).resolve().parents[2])

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


if __name__ == "__main__":
    unittest.main()
