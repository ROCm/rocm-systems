"""
GoogleTest runner -- executes the rdctst binary in embedded batch mode and the
GoogleTest binaries that run without a GPU.
"""

import subprocess
import os
import pytest


class TestGoogleTestSuite:
    """Run the rdctst GoogleTest binary and report results."""

    def test_rdctst_embedded_batch(self, rdctst_path, rdcd_path, has_gpu):
        if not has_gpu:
            pytest.skip("No AMD GPU -- cannot run rdctst")

        env = os.environ.copy()
        env["RDC_DISABLE_ROCP"] = "yes"

        result = subprocess.run(
            [rdctst_path, "-b", f"--start_rdcd={rdcd_path}"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        assert result.returncode == 0, (
            f"rdctst batch mode failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
        )

    def test_rdc_partition_count(self, rocm_dir, rdctst_path):
        # rdctst_path skips this test unless the RDC test package is installed, and
        # every build that installs rdctst also installs rdc_partition_count_tests.
        binary = os.path.join(rocm_dir, "share", "rdc", "rdctst_tests", "rdc_partition_count_tests")
        assert os.path.isfile(binary), f"{binary} is missing from the RDC test package"

        result = subprocess.run([binary], capture_output=True, text=True, timeout=120)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        assert result.returncode == 0, (
            f"rdc_partition_count_tests failed (rc={result.returncode}):\n{result.stdout[-2000:]}"
        )

    def test_rdc_client_shutdown(self, rocm_dir, rdctst_path):
        # rdctst_path skips this test unless the RDC test package is installed. rdctst
        # is built only with the standalone client, so rdc_client_tests must be too.
        binary = os.path.join(rocm_dir, "share", "rdc", "rdctst_tests", "rdc_client_tests")
        assert os.path.isfile(binary), f"{binary} is missing from the RDC test package"

        result = subprocess.run([binary], capture_output=True, text=True, timeout=120)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        assert result.returncode == 0, (
            f"rdc_client_tests failed (rc={result.returncode}):\n{result.stdout[-2000:]}"
        )

    def test_rdc_power_usage(self, rocm_dir, rdctst_path):
        # rdctst_path skips this test unless the RDC test package is installed, and
        # every build that installs rdctst also installs rdc_power_usage_tests.
        binary = os.path.join(rocm_dir, "share", "rdc", "rdctst_tests", "rdc_power_usage_tests")
        assert os.path.isfile(binary), f"{binary} is missing from the RDC test package"

        result = subprocess.run([binary], capture_output=True, text=True, timeout=120)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        assert result.returncode == 0, (
            f"rdc_power_usage_tests failed (rc={result.returncode}):\n{result.stdout[-2000:]}"
        )

    def test_rdc_link_status(self, rocm_dir, rdctst_path):
        # rdctst_path skips this test unless the RDC test package is installed, and
        # every build that installs rdctst also installs rdc_link_status_tests.
        binary = os.path.join(rocm_dir, "share", "rdc", "rdctst_tests", "rdc_link_status_tests")
        assert os.path.isfile(binary), f"{binary} is missing from the RDC test package"

        result = subprocess.run([binary], capture_output=True, text=True, timeout=120)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        assert result.returncode == 0, (
            f"rdc_link_status_tests failed (rc={result.returncode}):\n{result.stdout[-2000:]}"
        )

    def test_rdctst_help(self, rdctst_path):
        result = subprocess.run(
            [rdctst_path, "--rdctst_help"], capture_output=True, text=True, timeout=30
        )
        combined = (result.stdout + result.stderr).lower()
        assert "batch_mode" in combined or "rdctst_help" in combined
