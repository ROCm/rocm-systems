#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Test diagnostics and error handling utilities for rocprofiler-sdk tests.
Provides error categorization, environment validation, and helpful diagnostics.
"""

import os
import sys
import subprocess
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Any


class ErrorCategory(Enum):
    """Categorizes test failures for quick diagnosis"""
    SETUP = "SETUP"              # Missing dependencies, wrong environment
    PERMISSIONS = "PERMISSIONS"  # File/device permissions issues
    INFRASTRUCTURE = "INFRASTRUCTURE"  # GPU unavailable, driver issues
    DATA = "DATA"                # Data validation failures
    TIMEOUT = "TIMEOUT"          # Execution timeout
    CORRUPTION = "CORRUPTION"    # Corrupted output or files


class LogLevel(Enum):
    """Logging verbosity levels"""
    MINIMAL = 1   # CI: errors and critical diagnostics only
    NORMAL = 2    # Local: moderate verbosity
    VERBOSE = 3   # Debug: full details


def is_ci_environment() -> bool:
    """Detect if running in CI environment"""
    ci_indicators = ['CI', 'CONTINUOUS_INTEGRATION', 'JENKINS_HOME', 'GITLAB_CI']
    return any(os.getenv(var) for var in ci_indicators)


def get_log_level() -> LogLevel:
    """Determine logging level based on environment"""
    override = os.getenv('ROCPROFILER_TEST_LOG_LEVEL', '').upper()
    if override == 'VERBOSE':
        return LogLevel.VERBOSE
    elif override == 'MINIMAL':
        return LogLevel.MINIMAL

    return LogLevel.MINIMAL if is_ci_environment() else LogLevel.NORMAL


class DiagnosticHelper:
    """Provides diagnostic messages and self-diagnosis for test failures"""

    def __init__(self, log_level: Optional[LogLevel] = None):
        self.log_level = log_level or get_log_level()
        self._gpu_checked = False
        self._gpu_available = None

    def format_error(
        self,
        category: ErrorCategory,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Format a comprehensive error message with category and diagnostics.

        Args:
            category: Error category for quick identification
            message: Primary error message
            context: Additional context (file, line, values, etc.)
            suggestions: List of diagnostic steps or fixes

        Returns:
            Formatted error string
        """
        parts = [f"[{category.value}] {message}"]

        if context and self.log_level != LogLevel.MINIMAL:
            parts.append("Context:")
            for key, value in context.items():
                parts.append(f"  {key}: {value}")

        if suggestions:
            parts.append("Diagnostic steps:")
            for i, suggestion in enumerate(suggestions, 1):
                parts.append(f"  {i}. {suggestion}")

        return "\n".join(parts)

    def check_gpu_available(self) -> tuple[bool, str]:
        """
        Check if GPU/HSA runtime is available.

        Returns:
            (is_available, diagnostic_message)
        """
        if self._gpu_checked:
            return self._gpu_available, ""

        self._gpu_checked = True

        # Check for ROCm installation
        rocm_paths = ['/opt/rocm', '/opt/rocm-*']
        if not any(Path(p).exists() for p in ['/opt/rocm'] + list(Path('/opt').glob('rocm-*'))):
            self._gpu_available = False
            return False, "ROCm installation not found in /opt/rocm"

        # Check for GPU devices
        try:
            result = subprocess.run(
                ['rocminfo'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and 'Agent' in result.stdout:
                self._gpu_available = True
                return True, ""
            else:
                self._gpu_available = False
                return False, "rocminfo returned no agents"
        except FileNotFoundError:
            self._gpu_available = False
            return False, "rocminfo not found in PATH"
        except subprocess.TimeoutExpired:
            self._gpu_available = False
            return False, "rocminfo timed out (potential driver issue)"
        except Exception as e:
            self._gpu_available = False
            return False, f"GPU check failed: {str(e)}"

    def check_file_permissions(self, filepath: Path) -> tuple[bool, str]:
        """
        Check if file is readable and provide diagnostic.

        Returns:
            (can_read, diagnostic_message)
        """
        if not filepath.exists():
            return False, f"File does not exist: {filepath}"

        if not os.access(filepath, os.R_OK):
            stat_info = filepath.stat()
            return False, f"Permission denied: mode={oct(stat_info.st_mode)}, owner={stat_info.st_uid}"

        return True, ""

    def suggest_data_mismatch_diagnosis(
        self,
        expected: Any,
        actual: Any,
        field_name: str
    ) -> List[str]:
        """Generate diagnostic suggestions for data mismatches"""
        suggestions = [
            f"Check if test data was corrupted or truncated",
            f"Verify {field_name} generation logic in source",
        ]

        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if expected > 0 and actual == 0:
                suggestions.append(
                    f"{field_name} is zero - possible initialization or timing issue"
                )
            elif expected != 0 and actual != 0:
                ratio = abs(actual / expected) if expected != 0 else float('inf')
                if 0.8 <= ratio <= 1.2:
                    suggestions.append(
                        f"Values are close ({ratio:.2%}) - may be timing/race condition"
                    )

        return suggestions


# Global diagnostic helper instance
_diagnostic = DiagnosticHelper()


def assert_with_diagnostic(
    condition: bool,
    category: ErrorCategory,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None
):
    """
    Enhanced assert with categorized error messages and diagnostics.

    Example:
        assert_with_diagnostic(
            len(data) > 0,
            ErrorCategory.DATA,
            "Expected non-empty data array",
            context={"actual_length": len(data), "source": filename},
            suggestions=["Check if profiling was enabled", "Verify trace file is complete"]
        )
    """
    if not condition:
        error_msg = _diagnostic.format_error(category, message, context, suggestions)
        raise AssertionError(error_msg)


def check_test_prerequisites() -> Optional[str]:
    """
    Perform common setup checks before running tests.

    Returns:
        Error message if prerequisites fail, None if all pass
    """
    # Check GPU availability
    gpu_ok, gpu_msg = _diagnostic.check_gpu_available()
    if not gpu_ok:
        return _diagnostic.format_error(
            ErrorCategory.INFRASTRUCTURE,
            "GPU/HSA runtime not available",
            context={"detail": gpu_msg},
            suggestions=[
                "Verify ROCm is installed: ls /opt/rocm",
                "Check GPU devices: rocminfo",
                "Ensure user has GPU access: groups | grep render",
                "Check driver loaded: lsmod | grep amdgpu"
            ]
        )

    # Check LD_LIBRARY_PATH includes ROCm
    ld_path = os.getenv('LD_LIBRARY_PATH', '')
    if '/opt/rocm' not in ld_path and not any('rocm' in p for p in ld_path.split(':')):
        if _diagnostic.log_level != LogLevel.MINIMAL:
            print(
                "WARNING: LD_LIBRARY_PATH may not include ROCm libraries",
                file=sys.stderr
            )

    return None


def log_verbose(message: str):
    """Log message only if verbosity is enabled (non-CI or override)"""
    if _diagnostic.log_level == LogLevel.VERBOSE:
        print(f"[VERBOSE] {message}", file=sys.stderr)


def log_info(message: str):
    """Log informational message (suppressed in CI unless error)"""
    if _diagnostic.log_level >= LogLevel.NORMAL:
        print(f"[INFO] {message}", file=sys.stderr)
