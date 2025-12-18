# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Test runners for different rocprofiler-systems instrumentation modes.

Provides classes for running tests with:
- Baseline execution (no instrumentation)
- Sampling instrumentation
- Binary rewrite instrumentation
- Runtime instrumentation
- rocprof-sys-run wrapper
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Optional
from .config import RocprofsysConfig

# Global registry for test results (cleared after each test)
_test_results: list["TestResult"] = []


def _register_result(result: "TestResult") -> None:
    """Register a test result for potential output display."""
    _test_results.append(result)


def _get_and_clear_results() -> list["TestResult"]:
    """Get all registered results and clear the registry."""
    global _test_results
    results = _test_results.copy()
    _test_results = []
    return results

ROCPROFSYS_ABORT_FAIL_REGEX = [
    r"### ERROR ###",
    r"unknown-hash=",
    r"address of faulting memory reference",
    r"exiting with non-zero exit code",
    r"terminate called after throwing an instance",
    r"calling abort\.\. in ",
    r"Exit code: [1-9]",
]

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

@dataclass
class TestResult:
    """Result of a test execution

    Attributes:
        returncode: Process exit code
        test_output: Standard output and error content
        extra_output: Extra output set by the test itself (as of now, only used for timeout errors)
        output_dir: Directory containing output files
        command: The command that was executed
        env: Environment variables used
        duration: Execution time in seconds (if measured)
        pass_regex: Optional regex pattern(s) that must be found for success
        fail_regex: Regex pattern(s) that must NOT be found (defaults to ROCPROFSYS_ABORT_FAIL_REGEX)
        _instrumented_files: List of instrumented binary files created
    """

    returncode: int
    test_output: str
    output_dir: Path
    command: list[str]
    environment: dict[str, str]
    extra_output: Optional[str] = None
    duration: Optional[float] = None
    pass_regex: Optional[list[str]] = None
    fail_regex: list[str] = field(default_factory=lambda: ROCPROFSYS_ABORT_FAIL_REGEX.copy())
    _instrumented_files: list[Path] = field(default_factory=list)

    def _check_patterns(self) -> tuple[bool, Optional[str]]:
        """Check all fail/pass patterns in a single scan.

        Returns:
            Tuple of (success, failure_reason)
        """
        if self.returncode != 0:
            return False, f"Non-zero return code: {self.returncode}"

        # Build combined regex for single-pass scanning
        all_patterns = []
        fail_indices = set()
        pass_indices = set()

        for i, pattern in enumerate(self.fail_regex):
            all_patterns.append(f"(?P<f{i}>{pattern})")
            fail_indices.add(f"f{i}")

        if self.pass_regex:
            for i, pattern in enumerate(self.pass_regex):
                all_patterns.append(f"(?P<p{i}>{pattern})")
                pass_indices.add(f"p{i}")

        if not all_patterns:
            return True, None

        # Single scan with combined regex
        combined_regex = re.compile("|".join(all_patterns))
        found_pass = set()

        for match in combined_regex.finditer(self.test_output):
            matched_group = match.lastgroup

            # Fail pattern found - immediate failure
            if matched_group in fail_indices:
                original_idx = int(matched_group[1:])
                return False, f"Fail pattern matched: {self.fail_regex[original_idx]}"

            # Track found pass patterns
            if matched_group in pass_indices:
                found_pass.add(matched_group)

        # Check if all pass patterns were found
        if self.pass_regex:
            missing = pass_indices - found_pass
            if missing:
                missing_idx = int(next(iter(missing))[1:])
                return False, f"Pass pattern not found: {self.pass_regex[missing_idx]}"

        return True, None

    @property
    def success(self) -> bool:
        """Check if test execution succeeded.

        Returns True only if:
        - Return code is 0
        - No fail_regex patterns found in test_output
        - All pass_regex patterns found (if specified)

        Uses single-pass scanning for efficiency on large outputs.
        """
        success, _ = self._check_patterns()
        return success

    @property
    def failure_reason(self) -> Optional[str]:
        """Get a description of why the test failed, or None if successful."""
        _, reason = self._check_patterns()
        return reason

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
        return list(self.output_dir.glob("*.json")) + list(
            self.output_dir.glob("*.txt")
        )

    def get_output_file(self, pattern: str) -> Optional[Path]:
        """Get an output file matching the given pattern.

        Args:
            pattern: Glob pattern to match

        Returns:
            First matching file or None
        """
        matches = list(self.output_dir.glob(pattern))
        return matches[0] if matches else None

    def assert_file_exists(self, filename: str) -> Path:
        """Assert that an output file exists and return its path.

        Args:
            filename: Name of the file to check

        Returns:
            Path to the file

        Raises:
            AssertionError: If file doesn't exist
        """
        path = self.output_dir / filename
        assert path.exists(), f"Expected output file not found: {path}"
        return path

    def cleanup(self, keep_on_failure: bool = True) -> None:
        """Clean up test output files.

        Args:
            keep_on_failure: If True, keep files when test failed for debugging
        """
        if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") == "1":
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
        if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") == "1":
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
        pass_regex: Optional[list[str]] = None,
        fail_regex: Optional[list[str]] = None,
    ):

        self.config = config
        self.target = target
        self.target_exe = config.get_target_executable(target)
        self.output_dir = Path(output_dir)
        self.run_args = run_args or []
        self.timeout = timeout
        self.mpi_ranks = mpi_ranks
        self.pass_regex = pass_regex
        self.fail_regex = fail_regex if fail_regex is not None else ROCPROFSYS_ABORT_FAIL_REGEX.copy()

        self.env = config.get_base_environment()
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
        full_env = os.environ.copy()
        full_env.update(self.env)

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout,
                env=full_env,
                cwd=self.config.rocprofsys_root_dir,
            )

            duration = time.time() - start_time

            test_result = TestResult(
                returncode=result.returncode,
                test_output=result.stdout,
                output_dir=self.output_dir,
                command=command,
                environment=self.env,
                duration=duration,
                pass_regex=self.pass_regex,
                fail_regex=self.fail_regex,
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            test_result = TestResult(
                returncode=-1,
                test_output=e.stdout or "",
                extra_output=f"Timeout after {self.timeout}s\n{e.stderr or ''}",
                output_dir=self.output_dir,
                command=command,
                environment=self.env,
                duration=duration,
                pass_regex=self.pass_regex,
                fail_regex=self.fail_regex,
            )

        # Register result for pytest output hooks
        _register_result(test_result)
        return test_result

class BaselineRunner(BaseRunner):
    """Run target without any instrumentation."""

    def build_command(self) -> list[str]:
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
        self.output_dir.mkdir(parents=True, exist_ok=True)

        command = (
            [str(self.config.rocprofsys_instrument)]
            + ["-o", str(self.instrumented_exe)]
            + self.rewrite_args
            + ["--print-instrumented", "functions"]
            + ["--", str(self.target_exe)]
        )

        full_env = os.environ.copy()
        full_env.update(self.env)

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=self.timeout,
            env=full_env,
            cwd=self.config.rocprofsys_root_dir,
        )

        # Track instrumented files for cleanup
        if self.instrumented_exe.exists():
            self._instrumented_files.append(self.instrumented_exe)

        return TestResult(
            returncode=result.returncode,
            test_output=result.stdout,
            output_dir=self.output_dir,
            command=command,
            environment=self.env,
            _instrumented_files=self._instrumented_files.copy(),
        )

    def build_command(self) -> list[str]:
        """Build command to run the instrumented binary."""
        return (
            [str(self.config.rocprofsys_run), "--", str(self.instrumented_exe)]
            + self.run_args
        )

    def run(self) -> TestResult:
        """Execute full rewrite + run sequence.

        Returns:
            TestResult from run phase (rewrite must succeed first)

        Note:
            By default, cleanup is handled by the test_output_dir fixture
            AFTER the test completes (including validation). Set cleanup_on_success=True
            only if you want immediate cleanup of .inst files (validation files are
            preserved regardless).
        """
        # First, perform rewrite
        rewrite_result = self.rewrite()
        _register_result(rewrite_result)
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
            if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") != "1":
                run_result.cleanup_instrumented_binaries()

        return run_result

    def cleanup(self) -> None:
        """Clean up instrumented binary files."""
        if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") == "1":
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

    def build_command(self) -> list[str]:
        return (
            [str(self.config.rocprofsys_run), "--", str(self.target_exe)]
            + self.run_args
        )

class SysBinaryRunner(BaseRunner):
    """Run a rocprof-sys binary

    A rocprof-sys binary is one of the following:
        - rocprof-sys-instrument
        - rocprof-sys-sample
        - rocprof-sys-run
        - rocprof-sys-avail

    Args:
        config: rocprofiler-systems configuration
        target: Name of target binary (e.g., 'rocprof-sys-instrument')
        output_dir: Directory for output files
        args: Arguments to pass to the target binary
        env: Custom environment variables. If None, uses default test environment
        timeout: Test timeout in seconds
        pass_regex: Patterns that must be found for success
        fail_regex: Patterns that must NOT be found. If pass_regex is set,
                    this must contain ABORT_FAIL_REGEX_MARKER placeholder
        command: Optional full command to run instead of target
        **kwargs: Additional arguments passed to BaseRunner
    """
    # Marker for abort fail regex replacement
    ABORT_FAIL_REGEX_MARKER = "|ROCPROFSYS_ABORT_FAIL_REGEX"

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        timeout: int = 300,
        pass_regex: Optional[list[str]] = None,
        fail_regex: Optional[list[str]] = None,
        command: Optional[list[str]] = None,
        working_directory: Optional[Path] = None,
    ):
        # Regex validation
        if pass_regex and fail_regex:
            fail_regex_str = "|".join(fail_regex) if isinstance(fail_regex, list) else str(fail_regex)
            if self.ABORT_FAIL_REGEX_MARKER not in fail_regex_str:
                raise ValueError(
                    f"Test has set pass and fail regexes but fail regex does not include "
                    f"'{self.ABORT_FAIL_REGEX_MARKER}'"
                )

        # Replace marker with actual regex
        processed_fail_regex = None
        if fail_regex:
            processed_fail_regex = []
            for pattern in fail_regex:
                if self.ABORT_FAIL_REGEX_MARKER in pattern:
                    # Replace marker with actual abort fail patterns
                    # Split on marker, add default patterns in between
                    parts = pattern.split(self.ABORT_FAIL_REGEX_MARKER)
                    for i, part in enumerate(parts):
                        if part:
                            processed_fail_regex.append(part)
                        if i < len(parts) - 1:
                            processed_fail_regex.extend(ROCPROFSYS_ABORT_FAIL_REGEX)
                else:
                    processed_fail_regex.append(pattern)
        else:
            processed_fail_regex = ROCPROFSYS_ABORT_FAIL_REGEX.copy()

        # Handle environment variables
        base_env = config.get_base_binary_environment()
        if env:
            base_env.update(env)

        # Initialize base runner
        super().__init__(
            config=config,
            target=target,
            output_dir=output_dir,
            run_args=args or [],
            env=base_env,
            timeout=timeout,
            pass_regex=pass_regex,
            fail_regex=processed_fail_regex,
        )

        self.command = command
        # This is only here for the basename-only library test
        self.working_directory = working_directory or config.rocprofsys_root_dir

    def build_command(self) -> list[str]:
        """Build the command to execute.

        If a custom command was provided, use it with args appended.
        Otherwise, use target binary with args.

        Returns:
            List of command components
        """

        if self.command:
            return self.command + self.run_args
        return [str(self.target_exe)] + self.run_args

    def run(self) -> TestResult:
        """Execute the test.

        Returns:
            TestResult with execution results
        """
        import time

        self.output_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command()
        full_env = os.environ.copy()
        full_env.update(self.env)

        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout,
                env=full_env,
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
                pass_regex=self.pass_regex,
                fail_regex=self.fail_regex,
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            test_result = TestResult(
                returncode=-1,
                test_output=e.stdout or "",
                extra_output=f"Timeout after {self.timeout}s\n{e.stderr or ''}",
                output_dir=self.output_dir,
                command=command,
                environment=self.env,
                duration=duration,
                pass_regex=self.pass_regex,
                fail_regex=self.fail_regex,
            )

        _register_result(test_result)
        return test_result
