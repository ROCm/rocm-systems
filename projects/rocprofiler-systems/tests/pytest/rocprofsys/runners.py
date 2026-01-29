# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Test runners for different rocprofiler-systems instrumentation modes.

Provides classes for running tests with:
- Baseline execution (no instrumentation)
- Sampling instrumentation
- Binary rewrite instrumentation
- Runtime instrumentation
- rocprof-sys-run wrapper
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
from typing import Optional
from .config import RocprofsysConfig


def _safe_remove_file(filepath: Path) -> None:
    """Safely remove a file, ignoring errors."""
    try:
        if filepath.is_file():
            filepath.unlink()
    except OSError:
        pass


def _safe_remove_directory(dirpath: Path) -> None:
    """Safely remove a directory recursively, ignoring errors."""
    try:
        if dirpath.is_dir():
            shutil.rmtree(dirpath)
    except OSError:
        pass


def _decode_bytes(data: bytes | None, encoding: str = "utf-8") -> str:
    """Decode bytes to string, returning empty string if None."""
    if data is None:
        return ""
    return data.decode(encoding, errors="replace")


@dataclass
class TestResult:
    """Result of a test execution

    Attributes:
        returncode: Process exit code
        test_output: Standard output and error content
        extra_output: Extra output set by the test itself
                      (as of now, only used for timeout errors)
        output_dir: Directory containing output files
        command: The command that was executed
        env: Environment variables used
        duration: Execution time in seconds (if measured)
        _instrumented_files: List of instrumented binary files created
    """

    returncode: int
    test_output: str
    output_dir: Path
    command: list[str]
    environment: dict[str, str]
    extra_output: Optional[str] = None
    duration: Optional[float] = None
    _instrumented_files: list[Path] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if test execution succeeded.

        Returns True only if:
        - Return code is 0
        """
        return self.returncode == 0

    @property
    def perfetto_file(self) -> Optional[Path]:
        candidates = [
            self.output_dir / "perfetto-trace.proto",
            self.output_dir / "perfetto-trace-0.proto",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        protos = list(self.output_dir.glob("perfetto-trace*.proto"))
        return protos[0] if protos else None

    @property
    def rocpd_file(self) -> Optional[Path]:
        candidate = self.output_dir / "rocpd.db"
        if candidate.exists():
            return candidate
        # Try globbing
        dbs = list(self.output_dir.glob("*.db"))
        return dbs[0] if dbs else None

    @property
    def timemory_files(self) -> list[Path]:
        """List of timemory output files."""
        return list(self.output_dir.glob("*.json")) + list(self.output_dir.glob("*.txt"))

    def get_output_file(self, pattern: str) -> Optional[Path]:
        """Get an output file matching the given pattern.

        Args:
            pattern: Glob pattern to match

        Returns:
            First matching file or None
        """
        matches = list(self.output_dir.glob(pattern))
        return matches[0] if matches else None

    def cleanup(self, keep_on_failure: bool = True) -> None:
        """Clean up test output files.

        Args:
            keep_on_failure: If True, keep files when test failed for debugging
        """
        if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "1") == "1":
            return

        if keep_on_failure and not self.success:
            return

        # Clean up instrumented binaries
        for inst_file in self._instrumented_files:
            _safe_remove_file(inst_file)

        # Clean up output directory
        if self.output_dir.exists():
            _safe_remove_directory(self.output_dir)

    def cleanup_instrumented_binaries(self) -> None:
        """Clean up only the instrumented binary files."""
        if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "1") == "1":
            return

        for inst_file in self._instrumented_files:
            _safe_remove_file(inst_file)

        # Also clean any .inst files in output directory
        if self.output_dir.exists():
            for inst_file in self.output_dir.glob("*.inst"):
                _safe_remove_file(inst_file)


class BaseRunner(ABC):
    """Abstract base class for test runners."""

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        run_args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        timeout: int = 300,
        mpi_ranks: int = 0,
        working_directory: Optional[Path] = None,
    ):

        self.config = config
        self.target = target
        self.target_exe = config.get_target_executable(target)
        self.output_dir = Path(output_dir)
        self.run_args = run_args or []
        self.timeout = timeout
        self.mpi_ranks = mpi_ranks
        self.working_directory = working_directory or config.rocprofsys_build_dir
        self.env = config.get_fundamental_environment()
        self.env.update(config.get_base_environment())
        self.env["ROCPROFSYS_OUTPUT_PATH"] = str(self.output_dir)
        if env:
            self.env.update(env)

    @abstractmethod
    def build_command(self) -> list[str]:
        """Build the command to execute.

        Returns:
            List of command components
        """
        pass

    def _wrap_with_mpi(self, command: list[str]) -> list[str]:
        """Wrap command with MPI launcher if needed.

        Args:
            command: Base command

        Returns:
            Command wrapped with mpiexec if MPI is enabled
        """
        if self.mpi_ranks > 0 and self.config.mpiexec:
            mpi_cmd = [
                str(self.config.mpiexec),
                "-n",
                str(self.mpi_ranks),
            ]

            try:
                result = subprocess.run(
                    [str(self.config.mpiexec), "--oversubscribe", "-n", "1", "true"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=5,
                )
                if result.returncode == 0:
                    mpi_cmd.insert(1, "--oversubscribe")
            except (subprocess.TimeoutExpired, OSError):
                pass

            return mpi_cmd + command

        return command

    def run(self) -> TestResult:
        """Execute the test.

        Returns:
            TestResult with execution results
        """
        import time

        self.output_dir.mkdir(parents=True, exist_ok=True)

        command = self.build_command()
        command = self._wrap_with_mpi(command)

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout,
                env=self.env,
                cwd=self.working_directory,
            )

            duration = time.time() - start_time
            test_result = TestResult(
                returncode=result.returncode,
                test_output=result.stdout,
                output_dir=self.output_dir,
                command=command,
                environment=self.env,
                duration=duration,
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            stdout = _decode_bytes(e.stdout)
            stderr = _decode_bytes(e.stderr)

            test_result = TestResult(
                returncode=-1,
                test_output=stdout,
                extra_output=f"Timeout after {self.timeout}s\n{stderr}",
                output_dir=self.output_dir,
                command=command,
                environment=self.env,
                duration=duration,
            )

        return test_result


class BaselineRunner(BaseRunner):
    """Run target without any instrumentation.

    Can also be used to run arbitrary commands by providing the `command` parameter.
     - command + run_args are executed as a single command
    If a rocprof-sys binary is provided, uses "base_binary_environment" instead of "base_environment".

    Args:
        config: rocprofiler-systems configuration
        target: Name of target executable (used if command is None)
        output_dir: Directory for output files
        command: Optional full command to run instead of target executable
        **kwargs: Additional arguments passed to BaseRunner
    """

    # rocprof-sys binaries that should use get_base_binary_environment()
    ROCPROFSYS_BINARIES = {
        "rocprof-sys-instrument",
        "rocprof-sys-sample",
        "rocprof-sys-run",
        "rocprof-sys-avail",
    }

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        command: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(config, target, output_dir, **kwargs)
        self.command = command

        # If target is a rocprof-sys binary, use binary environment instead
        if target in self.ROCPROFSYS_BINARIES:
            self.env = config.get_fundamental_environment()
            self.env.update(config.get_base_binary_environment())
            self.env["ROCPROFSYS_OUTPUT_PATH"] = str(self.output_dir)
            # Re-apply any custom env passed via kwargs
            if "env" in kwargs and kwargs["env"]:
                self.env.update(kwargs["env"])

    def build_command(self) -> list[str]:
        if self.command:
            return self.command + self.run_args
        return [str(self.target_exe)] + self.run_args


class SamplingRunner(BaseRunner):
    """Run target with sampling instrumentation."""

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        sample_args: Optional[list[str]] = None,
        **kwargs,
    ):
        """Initialize sampling runner.

        Args:
            config: rocprofiler-systems configuration
            target: Name of target executable
            output_dir: Directory for output files
            sample_args: Arguments for rocprof-sys-sample
            **kwargs: Additional arguments passed to BaseRunner
        """
        super().__init__(config, target, output_dir, **kwargs)
        self.sample_args = sample_args or []

    def build_command(self) -> list[str]:
        return (
            [str(self.config.rocprofsys_sample)]
            + self.sample_args
            + ["--", str(self.target_exe)]
            + self.run_args
        )


class BinaryRewriteRunner(BaseRunner):
    """Run binary rewrite instrumentation (two-phase: rewrite then run)."""

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        rewrite_args: Optional[list[str]] = None,
        cleanup_on_success: bool = False,
        **kwargs,
    ):
        """Initialize binary rewrite runner.

        Args:
            config: rocprofiler-systems configuration
            target: Name of target executable
            output_dir: Directory for output files
            rewrite_args: Arguments for rocprof-sys-instrument
            cleanup_on_success: Whether to clean up instrumented binary immediately
                after successful run. Default is False - let the test_output_dir
                fixture handle cleanup after validation completes.
            **kwargs: Additional arguments passed to BaseRunner
        """
        super().__init__(config, target, output_dir, **kwargs)
        self.rewrite_args = rewrite_args or []
        self.instrumented_exe = output_dir / f"{target}.inst"
        self.cleanup_on_success = cleanup_on_success
        self._instrumented_files: list[Path] = []

    def rewrite(self) -> TestResult:
        """Perform binary rewrite phase.

        Returns:
            TestResult from rewrite operation
        """
        import time

        self.output_dir.mkdir(parents=True, exist_ok=True)

        command = (
            [str(self.config.rocprofsys_instrument)]
            + ["-o", str(self.instrumented_exe)]
            + self.rewrite_args
            + ["--print-instrumented", "functions"]
            + ["--", str(self.target_exe)]
        )

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout,
                env=self.env,
                cwd=self.config.rocprofsys_build_dir,
            )

            duration = time.time() - start_time
            test_result = TestResult(
                returncode=result.returncode,
                test_output=result.stdout,
                output_dir=self.output_dir,
                command=command,
                environment=self.env,
                duration=duration,
                _instrumented_files=self._instrumented_files.copy(),
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            stdout = _decode_bytes(e.stdout)
            stderr = _decode_bytes(e.stderr)

            test_result = TestResult(
                returncode=-1,
                test_output=stdout,
                extra_output=f"Timeout after {self.timeout}s\n{stderr}",
                output_dir=self.output_dir,
                command=command,
                environment=self.env,
                duration=duration,
                _instrumented_files=self._instrumented_files.copy(),
            )

        # Track instrumented files for cleanup
        if self.instrumented_exe.exists():
            self._instrumented_files.append(self.instrumented_exe)

        return test_result

    def build_command(self) -> list[str]:
        """Build command to run the instrumented binary."""
        return [
            str(self.config.rocprofsys_run),
            "--",
            str(self.instrumented_exe),
        ] + self.run_args

    def run(self) -> TestResult:
        """Execute full rewrite + run sequence.

        Returns:
            TestResult from full rewrite + run sequence

        Note:
            By default, cleanup is handled by the test_output_dir fixture
            AFTER the test completes (including validation). Set cleanup_on_success=True
            only if you want immediate cleanup of .inst files (validation files are
            preserved regardless).
        """
        # First, perform rewrite
        rewrite_result = self.rewrite()
        if not rewrite_result.success:
            return rewrite_result

        # Then run the instrumented binary
        run_result = super().run()

        # Add instrumented files to result for cleanup (used by fixtures)
        run_result._instrumented_files = self._instrumented_files.copy()

        # Optional immediate cleanup of .inst files only (NOT validation files)
        # Default is False - let test_output_dir fixture handle all cleanup
        # after validation completes
        if self.cleanup_on_success and run_result.success:
            run_result.cleanup_instrumented_binaries()

        # Combine rewrite and run output
        run_result.test_output = (
            f"=== REWRITE PHASE ===\n{rewrite_result.test_output}\n"
            f"=== RUN PHASE ===\n{run_result.test_output}"
        )
        run_result.duration = rewrite_result.duration + run_result.duration
        extra_parts = []
        if rewrite_result.extra_output:
            extra_parts.append(f"=== REWRITE PHASE ===\n{rewrite_result.extra_output}")
        if run_result.extra_output:
            extra_parts.append(f"=== RUN PHASE ===\n{run_result.extra_output}")
        if extra_parts:
            run_result.extra_output = "\n".join(extra_parts)

        return run_result

    def cleanup(self) -> None:
        """Clean up instrumented binary files."""
        if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "1") == "1":
            return

        for inst_file in self._instrumented_files:
            _safe_remove_file(inst_file)

        # Also clean any .inst files in output directory
        if self.output_dir.exists():
            for inst_file in self.output_dir.glob("*.inst"):
                _safe_remove_file(inst_file)


class RuntimeInstrumentRunner(BaseRunner):
    """Run target with runtime instrumentation."""

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        instrument_args: Optional[list[str]] = None,
        **kwargs,
    ):
        """Initialize runtime instrument runner.

        Args:
            config: rocprofiler-systems configuration
            target: Name of target executable
            output_dir: Directory for output files
            instrument_args: Arguments for rocprof-sys-instrument
            **kwargs: Additional arguments passed to BaseRunner
        """
        super().__init__(config, target, output_dir, **kwargs)
        self.instrument_args = instrument_args or []

    def build_command(self) -> list[str]:
        return (
            [str(self.config.rocprofsys_instrument)]
            + self.instrument_args
            + ["--print-instrumented", "functions"]
            + ["--", str(self.target_exe)]
            + self.run_args
        )


class SysRunRunner(BaseRunner):
    """Run target with rocprof-sys-run wrapper."""

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        sysrun_args: Optional[list[str]] = None,
        **kwargs,
    ):
        """Initialize sys-run runner.

        Args:
            config: rocprofiler-systems configuration
            target: Name of target executable
            output_dir: Directory for output files
            sysrun_args: Arguments for rocprof-sys-run (before --)
            **kwargs: Additional arguments passed to BaseRunner
        """
        super().__init__(config, target, output_dir, **kwargs)
        self.sysrun_args = sysrun_args or []

    def build_command(self) -> list[str]:
        return (
            [str(self.config.rocprofsys_run)]
            + self.sysrun_args
            + ["--", str(self.target_exe)]
            + self.run_args
        )
