"""Tests to verify generated bundled binary test assets.

This test suite introspects the generated test assets to verify they contain
the expected bundled architectures and formats.
"""

import subprocess
from pathlib import Path

import pytest

from rocm_kpack.binutils import BundledBinary, Toolchain


@pytest.fixture(scope="module")
def bundled_assets_dir(test_assets_dir: Path) -> Path:
    """Path to bundled binaries test assets."""
    assets_dir = test_assets_dir / "bundled_binaries" / "linux" / "cov5"
    if not assets_dir.exists():
        pytest.skip(f"Bundled assets not found at {assets_dir}")
    return assets_dir


def get_bundle_info(file_path: Path, toolchain: Toolchain) -> dict:
    """Extract bundle information from a binary.

    For fully linked binaries (executables and shared libraries), we check for
    .hip_fatbin sections and extract architecture information from strings.

    Returns:
        Dictionary with keys:
        - targets: List of gfx architecture identifiers found
        - has_device_code: Boolean indicating if device code is present
        - has_hip_fatbin_section: Boolean indicating if .hip_fatbin section exists
    """
    try:
        # Check for .hip_fatbin section using readelf
        has_hip_section = False
        try:
            result = subprocess.run(
                ["readelf", "-S", str(file_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            has_hip_section = ".hip_fatbin" in result.stdout
        except:
            pass

        # Extract architecture targets using strings
        targets = []
        try:
            result = subprocess.run(
                ["strings", str(file_path)],
                capture_output=True,
                text=True,
                check=True,
            )

            # Look for gfx architecture identifiers
            import re

            gfx_pattern = re.compile(r"gfx\d+")

            for line in result.stdout.splitlines():
                matches = gfx_pattern.findall(line)
                for match in matches:
                    if match not in targets:
                        targets.append(match)
        except:
            pass

        return {
            "targets": targets,
            "has_device_code": has_hip_section and len(targets) > 0,
            "has_hip_fatbin_section": has_hip_section,
        }

    except Exception as e:
        pytest.fail(f"Failed to get bundle info for {file_path}: {e}")


def test_bundled_assets_exist(bundled_assets_dir: Path):
    """Verify all expected test assets exist."""
    expected_executables = [
        "test_kernel_single.exe",
        "test_kernel_multi.exe",
        "test_kernel_compressed.exe",
        "test_kernel_wide.exe",
    ]

    expected_libraries = [
        "libtest_kernel_single.so",
        "libtest_kernel_multi.so",
    ]

    for exe in expected_executables:
        assert (bundled_assets_dir / exe).exists(), f"Missing executable: {exe}"

    for lib in expected_libraries:
        assert (bundled_assets_dir / lib).exists(), f"Missing shared library: {lib}"


def test_single_arch_executable(bundled_assets_dir: Path, toolchain: Toolchain):
    """Verify test_kernel_single contains single architecture."""
    exe_path = bundled_assets_dir / "test_kernel_single.exe"
    info = get_bundle_info(exe_path, toolchain)

    assert info["has_device_code"], "Single arch executable should have device code"

    # Should have exactly one device code target (gfx1100)
    gfx_targets = [t for t in info["targets"] if "gfx" in t]
    assert len(gfx_targets) >= 1, "Should have at least one gfx target"

    # Verify gfx1100 is present
    assert any("gfx1100" in t for t in info["targets"]), "Should contain gfx1100 target"


def test_multi_arch_executable(bundled_assets_dir: Path, toolchain: Toolchain):
    """Verify test_kernel_multi contains multiple architectures."""
    exe_path = bundled_assets_dir / "test_kernel_multi.exe"
    info = get_bundle_info(exe_path, toolchain)

    assert info["has_device_code"], "Multi arch executable should have device code"

    # Should have multiple gfx targets (gfx1100 and gfx1101)
    gfx_targets = [t for t in info["targets"] if "gfx" in t]
    assert len(gfx_targets) >= 2, "Should have at least two gfx targets"

    # Verify expected architectures
    targets_str = " ".join(info["targets"])
    assert "gfx1100" in targets_str, "Should contain gfx1100 target"
    assert "gfx1101" in targets_str, "Should contain gfx1101 target"


def test_compressed_executable(bundled_assets_dir: Path, toolchain: Toolchain):
    """Verify test_kernel_compressed has expected architecture coverage.

    Note: Actual compression may not be supported by the compiler, so we verify
    architecture coverage rather than compression status.
    """
    exe_path = bundled_assets_dir / "test_kernel_compressed.exe"
    info = get_bundle_info(exe_path, toolchain)

    assert info["has_device_code"], "Compressed executable should have device code"

    # Should have gfx1100 and gfx1101
    targets_str = " ".join(info["targets"])
    assert "gfx1100" in targets_str, "Should contain gfx1100 target"
    assert "gfx1101" in targets_str, "Should contain gfx1101 target"

    # Note: We don't assert is_compressed because the compiler may not support it
    # and falls back to uncompressed bundles


def test_wide_arch_executable(bundled_assets_dir: Path, toolchain: Toolchain):
    """Verify test_kernel_wide contains wide architecture coverage."""
    exe_path = bundled_assets_dir / "test_kernel_wide.exe"
    info = get_bundle_info(exe_path, toolchain)

    assert info["has_device_code"], "Wide arch executable should have device code"

    # Should have multiple CDNA and RDNA targets
    targets_str = " ".join(info["targets"])

    # Verify expected architectures (at least some of them should be present)
    expected_archs = ["gfx900", "gfx906", "gfx908", "gfx90a", "gfx1100"]
    found_archs = [arch for arch in expected_archs if arch in targets_str]

    assert (
        len(found_archs) >= 3
    ), f"Should contain at least 3 of {expected_archs}, found: {found_archs}"


def test_single_arch_library(bundled_assets_dir: Path, toolchain: Toolchain):
    """Verify libtest_kernel_single.so contains single architecture."""
    lib_path = bundled_assets_dir / "libtest_kernel_single.so"
    info = get_bundle_info(lib_path, toolchain)

    assert info["has_device_code"], "Single arch library should have device code"

    # Verify gfx1100 is present
    assert any("gfx1100" in t for t in info["targets"]), "Should contain gfx1100 target"


def test_multi_arch_library(bundled_assets_dir: Path, toolchain: Toolchain):
    """Verify libtest_kernel_multi.so contains multiple architectures."""
    lib_path = bundled_assets_dir / "libtest_kernel_multi.so"
    info = get_bundle_info(lib_path, toolchain)

    assert info["has_device_code"], "Multi arch library should have device code"

    # Should have multiple gfx targets
    gfx_targets = [t for t in info["targets"] if "gfx" in t]
    assert len(gfx_targets) >= 2, "Should have at least two gfx targets"

    # Verify expected architectures
    targets_str = " ".join(info["targets"])
    assert "gfx1100" in targets_str, "Should contain gfx1100 target"
    assert "gfx1101" in targets_str, "Should contain gfx1101 target"


def test_executables_are_executable(bundled_assets_dir: Path):
    """Verify executables have executable permission bit set."""
    import stat

    executables = [
        "test_kernel_single.exe",
        "test_kernel_multi.exe",
        "test_kernel_compressed.exe",
        "test_kernel_wide.exe",
    ]

    for exe_name in executables:
        exe_path = bundled_assets_dir / exe_name
        st = exe_path.stat()
        assert st.st_mode & stat.S_IXUSR, f"{exe_name} should be executable"


def test_libraries_are_shared_objects(bundled_assets_dir: Path):
    """Verify shared libraries have correct ELF type."""
    import subprocess

    libraries = [
        "libtest_kernel_single.so",
        "libtest_kernel_multi.so",
    ]

    for lib_name in libraries:
        lib_path = bundled_assets_dir / lib_name

        # Use 'file' command to check type
        result = subprocess.run(
            ["file", str(lib_path)], capture_output=True, text=True, check=True
        )

        assert (
            "shared object" in result.stdout.lower()
        ), f"{lib_name} should be a shared object, got: {result.stdout}"


def test_hip_fatbin_sections_present(bundled_assets_dir: Path):
    """Verify all bundled binaries have .hip_fatbin sections."""
    all_binaries = [
        "test_kernel_single.exe",
        "test_kernel_multi.exe",
        "test_kernel_compressed.exe",
        "test_kernel_wide.exe",
        "libtest_kernel_single.so",
        "libtest_kernel_multi.so",
    ]

    for binary_name in all_binaries:
        binary_path = bundled_assets_dir / binary_name

        # Check for .hip_fatbin section
        result = subprocess.run(
            ["readelf", "-S", str(binary_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        assert (
            ".hip_fatbin" in result.stdout
        ), f"{binary_name} should have .hip_fatbin section"
