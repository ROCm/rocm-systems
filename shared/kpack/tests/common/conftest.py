"""
Cross-platform test fixtures for common tests.

These fixtures parameterize tests by (platform, co_version) tuples,
allowing the same test logic to run against both ELF and COFF binaries.
"""

import pytest
from pathlib import Path


# Asset configurations: (platform, co_version)
# Parameterized to allow future expansion (e.g., cov6)
ASSET_CONFIGS = [
    ("linux", "cov5"),
    ("windows", "cov5"),
]


@pytest.fixture(params=ASSET_CONFIGS, ids=lambda x: f"{x[0]}-{x[1]}")
def asset_config(request):
    """Returns (platform, co_version) tuple for parameterized tests."""
    return request.param


@pytest.fixture
def bundled_assets_dir(test_assets_dir: Path, asset_config: tuple[str, str]) -> Path:
    """Path to bundled binaries for this platform/co_version."""
    platform, co_version = asset_config
    assets_dir = test_assets_dir / "bundled_binaries" / platform / co_version
    if not assets_dir.exists():
        raise FileNotFoundError(
            f"Required test assets directory not found: {assets_dir}"
        )
    return assets_dir


@pytest.fixture
def single_arch_exe(bundled_assets_dir: Path) -> Path:
    """Path to single-arch executable for this platform."""
    path = bundled_assets_dir / "test_kernel_single.exe"
    if not path.exists():
        raise FileNotFoundError(f"Required test asset not found: {path}")
    return path


@pytest.fixture
def multi_arch_exe(bundled_assets_dir: Path) -> Path:
    """Path to multi-arch executable for this platform."""
    path = bundled_assets_dir / "test_kernel_multi.exe"
    if not path.exists():
        raise FileNotFoundError(f"Required test asset not found: {path}")
    return path


@pytest.fixture
def single_arch_lib(bundled_assets_dir: Path, asset_config: tuple[str, str]) -> Path:
    """Path to single-arch shared library for this platform."""
    platform, _ = asset_config
    if platform == "windows":
        path = bundled_assets_dir / "test_kernel_single.dll"
    else:
        path = bundled_assets_dir / "libtest_kernel_single.so"
    if not path.exists():
        raise FileNotFoundError(f"Required test asset not found: {path}")
    return path


@pytest.fixture
def host_only_exe(bundled_assets_dir: Path) -> Path:
    """Path to host-only executable (no GPU code)."""
    path = bundled_assets_dir / "host_only.exe"
    if not path.exists():
        raise FileNotFoundError(f"Required test asset not found: {path}")
    return path


@pytest.fixture
def host_only_lib(bundled_assets_dir: Path, asset_config: tuple[str, str]) -> Path:
    """Path to host-only shared library (no GPU code)."""
    platform, _ = asset_config
    if platform == "windows":
        path = bundled_assets_dir / "host_only.dll"
    else:
        path = bundled_assets_dir / "libhost_only.so"
    if not path.exists():
        raise FileNotFoundError(f"Required test asset not found: {path}")
    return path
