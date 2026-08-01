#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest import mock

import yaml

import consan_tensile_support as tensile_support
import consan_tensile_validation as tensile_validation
from consan_validation_test_support import temporary_root


class TensileValidationTest(unittest.TestCase):
    @staticmethod
    def _make_fake_paths(root: Path) -> tensile_support.TensileValidationPaths:
        tensilelite = root / "tensilelite"
        (tensilelite / "Tensile").mkdir(parents=True)
        hardware_header = tensilelite / "include" / "Tensile" / "AMDGPU.hpp"
        hardware_header.parent.mkdir(parents=True)
        hardware_header.write_text(
            'std::getenv("TENSILE_STREAMK_FIXED_GRID");\n',
            encoding="utf-8",
        )
        rocm = root / "rocm"
        (rocm / "bin").mkdir(parents=True)
        executables = {
            "amdclang": rocm / "bin" / "amdclang++",
            "client": root / "tensilelite-client",
            "wrapper": root / "wrapper",
            "rocjitsu": root / "rocjitsu",
            "readelf": root / "llvm-readelf",
        }
        for path in executables.values():
            path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            path.chmod(0o755)
        rocjitsu_config = root / "gfx1250.json"
        rocjitsu_config.write_text("{}\n", encoding="utf-8")
        return tensile_support.TensileValidationPaths(
            tensilelite=tensilelite,
            rocm=rocm,
            client=executables["client"],
            wrapper=executables["wrapper"],
            rocjitsu=executables["rocjitsu"],
            rocjitsu_config=rocjitsu_config,
            llvm_readelf=executables["readelf"],
        )

    @staticmethod
    def _install_tensile_stub(
        paths: tensile_support.TensileValidationPaths,
        validation: str,
        *,
        sleep_seconds: int = 0,
    ) -> None:
        package = paths.tensilelite / "Tensile"
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "Tensile.py").write_text(
            "from pathlib import Path\n"
            "import time\n"
            "def Tensile(args):\n"
            "    output = Path(args[1])\n"
            "    (output / 'kernel.hsaco').write_bytes(b'elf')\n"
            "    print('run,problem,solution,validation,time-us', flush=True)\n"
            f"    print('0,problem,kernel_SK3_shape,{validation},10', flush=True)\n"
            f"    time.sleep({sleep_seconds})\n",
            encoding="utf-8",
        )

    @staticmethod
    def _main_argv(root: Path, *, timeout_seconds: int = 5) -> list[str]:
        return [
            "consan_tensile_validation.py",
            "--workspace",
            str(root),
            "--config",
            str(root / "case.yaml"),
            "--output-dir",
            str(root / "output"),
            "--label",
            "unit",
            "--timeout-seconds",
            str(timeout_seconds),
            "--expect-numeric-rows",
            "1",
            "--streamk-fixed-grid",
            "4",
            "--require-streamk-mode",
            "3",
        ]

    @staticmethod
    def _amdgpu_header() -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            (),
            0,
            stdout=(
                "ELF Header:\n"
                "  Machine:                           EM_AMDGPU\n"
                "  Flags:                             0x549, gfx1250\n"
            ),
            stderr="",
        )

    def test_oracle_result_is_retained_and_forwarded(self) -> None:
        with temporary_root() as root:
            forwarded = root / "row.json"
            retained = root / "oracle.json"
            with mock.patch.dict(
                os.environ,
                {"CONSAN_ROW_RESULT_PATH": str(forwarded)},
                clear=True,
            ):
                tensile_validation._write_oracle_result(
                    "pass",
                    {"numeric_rows": 1},
                    retained_path=retained,
                )
            forwarded_payload = json.loads(forwarded.read_text(encoding="utf-8"))
            retained_payload = json.loads(retained.read_text(encoding="utf-8"))
        self.assertEqual(forwarded_payload, retained_payload)
        self.assertEqual(retained_payload["oracle"], "pass")
        self.assertEqual(retained_payload["detail"]["numeric_rows"], 1)

    def test_config_must_be_a_workspace_file(self) -> None:
        with temporary_root() as root:
            config = root / "repo" / "case.yaml"
            config.parent.mkdir()
            config.write_text("case\n", encoding="utf-8")
            self.assertEqual(
                tensile_validation._resolve_config(root, Path("repo/case.yaml")),
                config,
            )
            self.assertEqual(
                tensile_validation._resolve_config(root, config),
                config,
            )
            with self.assertRaisesRegex(ValueError, "outside"):
                tensile_validation._resolve_config(root, root.parent / "case.yaml")
            with self.assertRaisesRegex(ValueError, "not a file"):
                tensile_validation._resolve_config(root, Path("missing.yaml"))

    def test_prerequisites_require_executables_and_fixed_grid_support(self) -> None:
        with temporary_root() as root:
            paths = self._make_fake_paths(root)
            config = root / "case.yaml"
            config.write_text("case\n", encoding="utf-8")
            self.assertEqual(
                tensile_validation._prerequisite_errors(
                    paths, config, streamk_fixed_grid=4
                ),
                [],
            )
            paths.wrapper.chmod(0o644)
            errors = tensile_validation._prerequisite_errors(
                paths, config, streamk_fixed_grid=4
            )
            self.assertTrue(
                any("missing Tensile launcher wrapper" in error for error in errors)
            )
            paths.wrapper.chmod(0o755)
            (paths.tensilelite / "include" / "Tensile" / "AMDGPU.hpp").write_text(
                "// variable removed\n", encoding="utf-8"
            )
            errors = tensile_validation._prerequisite_errors(
                paths, config, streamk_fixed_grid=4
            )
            self.assertTrue(
                any(
                    "does not expose TENSILE_STREAMK_FIXED_GRID" in error
                    for error in errors
                )
            )

    def test_positive_integer_parser_rejects_zero(self) -> None:
        self.assertEqual(tensile_validation._positive_int("3"), 3)
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "must be positive"):
            tensile_validation._positive_int("0")

    def test_gpu_target_parser_accepts_architecture_names(self) -> None:
        self.assertEqual(tensile_validation._gpu_target("gfx950"), "gfx950")
        self.assertEqual(tensile_validation._gpu_target("gfx1250"), "gfx1250")
        for invalid in ("950", "gfx", "gfx950:xnack-", "gfx9_50"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                argparse.ArgumentTypeError, "must name a gfx target"
            ):
                tensile_validation._gpu_target(invalid)

    def test_numeric_validation_requires_a_real_passed_row(self) -> None:
        passed = (
            "noise\n"
            "run,problem,solution,validation,time-us\n"
            '0,"(33,33,1,65)",kernel_SK3_shape,PASSED,10\n'
        )
        self.assertEqual(
            tensile_validation._numeric_validation_errors(
                passed, expected_result_count=1
            ),
            (1, []),
        )
        self.assertEqual(
            tensile_validation._numeric_validation_errors(
                passed, required_streamk_mode=3
            ),
            (1, []),
        )

        count, errors = tensile_validation._numeric_validation_errors(
            passed.replace("PASSED", "FAILED")
        )
        self.assertEqual(count, 1)
        self.assertIn("not PASSED", errors[0])

        count, errors = tensile_validation._numeric_validation_errors(
            passed, required_streamk_mode=2
        )
        self.assertEqual(count, 1)
        self.assertIn("not Stream-K mode 2", errors[0])

        count, errors = tensile_validation._numeric_validation_errors("noise\n")
        self.assertEqual(count, 0)
        self.assertIn("no result rows", errors[0])

    def test_numeric_validation_rejects_wrong_hardware_and_malformed_rows(self) -> None:
        output = (
            "WRONG_HARDWARE\n" "run,problem,validation,time-us\n" "0,problem,PASSED\n"
        )
        count, errors = tensile_validation._numeric_validation_errors(output)
        self.assertEqual(count, 0)
        self.assertTrue(any("incompatible hardware" in error for error in errors))
        self.assertTrue(any("malformed numeric row" in error for error in errors))

    def test_numeric_validation_is_line_scoped_and_checks_cardinality(self) -> None:
        output = (
            '"unbalanced unrelated log text\n'
            "run,problem,solution,validation,time-us\n"
            '0,"(33,33,1,65)",kernel_SK3_shape,PASSED,10\n'
            "1\n"
        )
        count, errors = tensile_validation._numeric_validation_errors(
            output,
            expected_result_count=2,
            required_streamk_mode=3,
        )
        self.assertEqual(count, 1)
        self.assertTrue(
            any("line 4: malformed numeric row" in error for error in errors)
        )
        self.assertTrue(
            any("expected 2 numeric result rows, found 1" in error for error in errors)
        )

    def test_main_writes_pass_and_failed_numeric_oracles(self) -> None:
        for numeric_verdict, expected_returncode, expected_oracle in (
            ("PASSED", 0, "pass"),
            ("FAILED", 1, "fail"),
        ):
            with self.subTest(
                numeric_verdict=numeric_verdict
            ), temporary_root() as root:
                paths = self._make_fake_paths(root)
                self._install_tensile_stub(paths, numeric_verdict)
                (root / "case.yaml").write_text("case\n", encoding="utf-8")
                forwarded = root / "row.json"
                with (
                    mock.patch.object(
                        tensile_validation,
                        "resolve_tensile_validation_paths",
                        return_value=paths,
                    ),
                    mock.patch.object(
                        tensile_validation.subprocess,
                        "run",
                        return_value=self._amdgpu_header(),
                    ),
                    mock.patch.object(
                        sys,
                        "argv",
                        self._main_argv(root),
                    ),
                    mock.patch.dict(
                        os.environ,
                        {"CONSAN_ROW_RESULT_PATH": str(forwarded)},
                        clear=True,
                    ),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    returncode = tensile_validation.main()
                payload = json.loads(forwarded.read_text(encoding="utf-8"))
                retained = tuple((root / "output").rglob("oracle.json"))
            self.assertEqual(returncode, expected_returncode)
            self.assertEqual(payload["oracle"], expected_oracle)
            self.assertEqual(payload["detail"]["numeric_rows"], 1)
            self.assertEqual(payload["detail"]["expected_numeric_rows"], 1)
            self.assertEqual(len(retained), 1)

    def test_main_reports_prerequisite_failure(self) -> None:
        with temporary_root() as root:
            paths = tensile_support.TensileValidationPaths(
                tensilelite=root / "missing-tensilelite",
                rocm=root / "missing-rocm",
                client=root / "missing-client",
                wrapper=root / "missing-wrapper",
                rocjitsu=root / "missing-rocjitsu",
                rocjitsu_config=root / "missing-config",
                llvm_readelf=root / "missing-readelf",
            )
            (root / "case.yaml").write_text("case\n", encoding="utf-8")
            forwarded = root / "row.json"
            with (
                mock.patch.object(
                    tensile_validation,
                    "resolve_tensile_validation_paths",
                    return_value=paths,
                ),
                mock.patch.object(sys, "argv", self._main_argv(root)),
                mock.patch.dict(
                    os.environ,
                    {"CONSAN_ROW_RESULT_PATH": str(forwarded)},
                    clear=True,
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                returncode = tensile_validation.main()
            payload = json.loads(forwarded.read_text(encoding="utf-8"))
        self.assertEqual(returncode, 2)
        self.assertEqual(payload["oracle"], "fail")
        self.assertGreaterEqual(len(payload["detail"]["reasons"]), 8)

    def test_main_timeout_is_single_cause_and_skips_object_claims(self) -> None:
        with temporary_root() as root:
            paths = self._make_fake_paths(root)
            self._install_tensile_stub(paths, "PASSED", sleep_seconds=30)
            (root / "case.yaml").write_text("case\n", encoding="utf-8")
            forwarded = root / "row.json"
            with (
                mock.patch.object(
                    tensile_validation,
                    "resolve_tensile_validation_paths",
                    return_value=paths,
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    self._main_argv(root, timeout_seconds=1),
                ),
                mock.patch.dict(
                    os.environ,
                    {"CONSAN_ROW_RESULT_PATH": str(forwarded)},
                    clear=True,
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                returncode = tensile_validation.main()
            payload = json.loads(forwarded.read_text(encoding="utf-8"))
        self.assertEqual(returncode, 1)
        self.assertEqual(payload["oracle"], "fail")
        self.assertEqual(payload["detail"]["verified_code_objects"], [])
        self.assertEqual(len(payload["detail"]["reasons"]), 1)
        self.assertIn("execution budget", payload["detail"]["reasons"][0])

    def test_outer_termination_is_forwarded_to_the_inner_process_group(self) -> None:
        with temporary_root() as root:
            pid_file = root / "pids"
            inner = (
                "import os, pathlib, subprocess, sys, time;"
                "child=subprocess.Popen(['sleep','60']);"
                "pathlib.Path(sys.argv[1]).write_text("
                "f'{os.getpid()} {child.pid}', encoding='utf-8');"
                "time.sleep(60)"
            )
            wrapper = (
                "import os, sys;"
                "import consan_tensile_validation as runner;"
                f"runner._run_command([sys.executable,'-c',{inner!r},sys.argv[1]],"
                "os.environ.copy(),60)"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", wrapper, str(pid_file)],
                cwd=Path(tensile_validation.__file__).parent,
                start_new_session=True,
            )
            pids: tuple[int, ...] = ()
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not pid_file.is_file():
                    time.sleep(0.02)
                self.assertTrue(pid_file.is_file())
                pids = tuple(
                    int(value) for value in pid_file.read_text(encoding="utf-8").split()
                )
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    live = [
                        pid
                        for pid in pids
                        if Path(f"/proc/{pid}/stat").is_file()
                        and Path(f"/proc/{pid}/stat")
                        .read_text(encoding="utf-8")
                        .split()[2]
                        != "Z"
                    ]
                    if not live:
                        break
                    time.sleep(0.02)
                self.assertEqual(live, [])
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_code_objects_must_all_declare_gfx1250(self) -> None:
        with temporary_root() as root:
            (root / "kernel.hsaco").write_bytes(b"elf")
            (root / "library.co").write_bytes(b"elf")
            header = subprocess.CompletedProcess(
                (),
                0,
                stdout=(
                    "ELF Header:\n"
                    "  Machine:                           EM_AMDGPU\n"
                    "  Flags:                             0x549, gfx1250\n"
                ),
                stderr="",
            )
            with mock.patch.object(
                tensile_validation.subprocess, "run", return_value=header
            ) as run:
                artifacts, errors = tensile_validation._code_object_errors(
                    root, Path("/llvm-readelf")
                )
        self.assertEqual(
            [path.name for path in artifacts],
            ["kernel.hsaco", "library.co"],
        )
        self.assertEqual(errors, [])
        self.assertEqual(run.call_count, 2)

    def test_code_object_target_check_fails_closed(self) -> None:
        with temporary_root() as root:
            (root / "kernel.hsaco").write_bytes(b"elf")
            header = subprocess.CompletedProcess(
                (),
                0,
                stdout=(
                    "ELF Header:\n"
                    "  Machine:                           EM_AMDGPU\n"
                    "  Flags:                             0x4a3, gfx1201\n"
                ),
                stderr="",
            )
            with mock.patch.object(
                tensile_validation.subprocess, "run", return_value=header
            ):
                _, errors = tensile_validation._code_object_errors(
                    root, Path("/llvm-readelf")
                )
        self.assertEqual(len(errors), 1)
        self.assertIn("does not declare gfx1250", errors[0])

    def test_code_object_checks_share_the_end_to_end_budget(self) -> None:
        with temporary_root() as root:
            (root / "kernel.hsaco").write_bytes(b"elf")
            with mock.patch.object(tensile_validation.subprocess, "run") as run:
                _, errors = tensile_validation._code_object_errors(
                    root,
                    Path("/llvm-readelf"),
                    deadline=time.monotonic() - 1,
                )
        run.assert_not_called()
        self.assertEqual(errors, ["code-object verification exceeded its budget"])

    def test_path_overrides_form_one_coherent_toolchain(self) -> None:
        with temporary_root() as root:
            overrides = {
                tensile_support.TENSILELITE_ROOT_ENV: str(root / "tensile"),
                tensile_support.ROCM_ROOT_ENV: str(root / "rocm"),
                tensile_support.TENSILE_CLIENT_ENV: str(root / "client"),
                tensile_support.TENSILE_WRAPPER_ENV: str(root / "wrapper"),
                tensile_support.ROCJITSU_EXE_ENV: str(root / "rocjitsu"),
                tensile_support.ROCJITSU_CONFIG_ENV: str(root / "gfx1250.json"),
                tensile_support.LLVM_READELF_ENV: str(root / "llvm-readelf"),
            }
            with mock.patch.dict(os.environ, overrides, clear=True):
                paths = tensile_support.resolve_tensile_validation_paths(root)
        self.assertEqual(paths.tensilelite, root / "tensile")
        self.assertEqual(paths.rocm, root / "rocm")
        self.assertEqual(paths.client, root / "client")
        self.assertEqual(paths.wrapper, root / "wrapper")
        self.assertEqual(paths.rocjitsu, root / "rocjitsu")
        self.assertEqual(paths.rocjitsu_config, root / "gfx1250.json")
        self.assertEqual(paths.llvm_readelf, root / "llvm-readelf")

    def test_target_selects_matching_default_rocjitsu_config(self) -> None:
        with temporary_root() as root:
            overrides = {
                tensile_support.TENSILELITE_ROOT_ENV: str(root / "tensile"),
                tensile_support.ROCM_ROOT_ENV: str(root / "rocm"),
                tensile_support.TENSILE_CLIENT_ENV: str(root / "client"),
                tensile_support.TENSILE_WRAPPER_ENV: str(root / "wrapper"),
                tensile_support.ROCJITSU_EXE_ENV: str(root / "rocjitsu"),
                tensile_support.LLVM_READELF_ENV: str(root / "llvm-readelf"),
            }
            with mock.patch.dict(os.environ, overrides, clear=True):
                paths = tensile_support.resolve_tensile_validation_paths(root, "gfx950")
        self.assertEqual(
            paths.rocjitsu_config,
            root
            / "rocm-systems"
            / "emulation"
            / "rocjitsu"
            / "configs"
            / "gfx950_cdna4.json",
        )

    def test_unknown_target_requires_an_explicit_rocjitsu_config(self) -> None:
        with temporary_root() as root:
            overrides = {
                tensile_support.TENSILELITE_ROOT_ENV: str(root / "tensile"),
                tensile_support.ROCM_ROOT_ENV: str(root / "rocm"),
                tensile_support.TENSILE_CLIENT_ENV: str(root / "client"),
                tensile_support.TENSILE_WRAPPER_ENV: str(root / "wrapper"),
                tensile_support.ROCJITSU_EXE_ENV: str(root / "rocjitsu"),
                tensile_support.LLVM_READELF_ENV: str(root / "llvm-readelf"),
            }
            with mock.patch.dict(os.environ, overrides, clear=True):
                with self.assertRaisesRegex(ValueError, "no default RocJITsu config"):
                    tensile_support.resolve_tensile_validation_paths(root, "gfx1100")

    def test_launcher_discovery_matches_the_hook_build_order(self) -> None:
        with temporary_root() as root:
            preferred = root / "rocjitsu-build" / "tools" / "rocjitsu" / "rocjitsu"
            fallback = root / "build" / "tools" / "rocjitsu" / "rocjitsu"
            preferred.parent.mkdir(parents=True)
            fallback.parent.mkdir(parents=True)
            preferred.write_bytes(b"preferred")
            fallback.write_bytes(b"fallback")
            overrides = {
                tensile_support.TENSILELITE_ROOT_ENV: str(root / "tensile"),
                tensile_support.ROCM_ROOT_ENV: str(root / "rocm"),
                tensile_support.TENSILE_CLIENT_ENV: str(root / "client"),
                tensile_support.ROCJITSU_CONFIG_ENV: str(root / "gfx1250.json"),
                tensile_support.LLVM_READELF_ENV: str(root / "llvm-readelf"),
            }
            with mock.patch.dict(os.environ, overrides, clear=True):
                paths = tensile_support.resolve_tensile_validation_paths(root)
        self.assertEqual(paths.rocjitsu, preferred)
        self.assertEqual(
            paths.wrapper,
            Path(tensile_support.__file__).with_name(
                "run_tensile_client_with_rocjitsu.sh"
            ),
        )

    def test_retained_fixture_has_bounded_streamk_shape(self) -> None:
        fixture_path = (
            Path(tensile_validation.__file__).with_name("fixtures")
            / "gfx1250_tensile_streamk_smoke.yaml"
        )
        fixture = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
        self.assertNotIn("TestParameters", fixture)
        problem_type, parameters = fixture["BenchmarkProblems"][0]
        self.assertEqual(problem_type["OperationType"], "GEMM")
        fork_parameters = {
            key: value
            for entry in parameters["ForkParameters"]
            for key, value in entry.items()
        }
        self.assertEqual(fork_parameters["StreamK"], [3])
        self.assertEqual(fork_parameters["WavefrontSize"], [32])
        self.assertEqual(fork_parameters["DepthU"], [64])
        self.assertEqual(
            fork_parameters["MatrixInstruction"],
            [[16, 16, 4, 1, 1, 1, 1, 1, 2]],
        )
        self.assertEqual(
            parameters["BenchmarkFinalParameters"],
            [{"ProblemSizes": [{"Exact": [33, 33, 1, 65]}]}],
        )
        for retired_key in (
            "InitialSolutionParameters",
            "BenchmarkForkParameters",
            "JoinParameters",
            "BenchmarkJoinParameters",
        ):
            self.assertNotIn(retired_key, parameters)


if __name__ == "__main__":
    unittest.main()
