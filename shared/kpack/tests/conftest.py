import os
import pathlib
import shutil

import pytest

# Configure Windows console for UTF-8 output before any test output
from rocm_kpack.platform_utils import configure_windows_console

configure_windows_console()

from rocm_kpack.binutils import Toolchain
from elf_test_utils import patch_hip_fatbin_size


@pytest.fixture(scope="session")
def test_assets_dir() -> pathlib.Path:
    """Provides a pathlib.Path to the shared test_assets directory."""
    test_assets_path = pathlib.Path(__file__).parent.parent / "test_assets"
    if not test_assets_path.is_dir():
        raise FileNotFoundError(
            f"test_assets directory not found at: {test_assets_path}"
        )
    return test_assets_path.resolve()


@pytest.fixture(scope="session")
def toolchain() -> Toolchain:
    """Provides a Toolchain, using ROCm installation if available.

    Tool discovery order:
    1. KPACK_LLVM_BIN environment variable (path to LLVM bin directory)
    2. System PATH (via shutil.which)

    Set KPACK_LLVM_BIN to your LLVM bin directory if tools aren't on PATH:
        export KPACK_LLVM_BIN=/opt/rocm/llvm/bin
    """
    llvm_bin = os.environ.get("KPACK_LLVM_BIN")

    # Check KPACK_LLVM_BIN first
    if llvm_bin:
        llvm_bin_path = pathlib.Path(llvm_bin)
        bundler = llvm_bin_path / "clang-offload-bundler"
        if not bundler.exists():
            # Try with .exe extension on Windows
            bundler = llvm_bin_path / "clang-offload-bundler.exe"
        if bundler.exists():
            return Toolchain(clang_offload_bundler=bundler)

    # Fall back to system PATH
    bundler_path = shutil.which("clang-offload-bundler")
    if bundler_path:
        return Toolchain(clang_offload_bundler=pathlib.Path(bundler_path))

    # Return default Toolchain - will fail on first tool access with helpful error
    return Toolchain()


# =============================================================================
# Patched binary fixtures for edge case testing
# =============================================================================
#
# These fixtures create modified versions of test binaries to trigger edge
# cases that are difficult to reproduce with real compiled code. See
# tests/elf_test_utils.py for detailed background on why these cases matter.


@pytest.fixture
def small_hip_fatbin_binary(
    test_assets_dir: pathlib.Path, tmp_path: pathlib.Path
) -> pathlib.Path:
    """
    Create a binary with a .hip_fatbin section too small to zero-page.

    The section is set to 3000 bytes, which is less than one 4KB page.
    This means conservative_zero_page() cannot find any full pages to
    remove, causing it to fail.

    This reproduces CI failures like:
        WARNING: Section too small or misaligned - no full pages to zero
          Section range: [0xf000, 0xfd6d)
        Error: Zero-page optimization failed

    Returns:
        Path to the patched binary
    """
    source = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
    output = tmp_path / "small_hip_fatbin.exe"
    return patch_hip_fatbin_size(source, output, new_size=3000)


@pytest.fixture
def marginal_hip_fatbin_binary(
    test_assets_dir: pathlib.Path, tmp_path: pathlib.Path
) -> pathlib.Path:
    """
    Create a binary where kpack overhead exceeds zero-page savings.

    The section is set to 5000 bytes (~1.2 pages). Zero-paging can remove
    one full page (4KB), but the structural changes required add more:
    - Padding for mmap alignment (up to 4KB)
    - PHDR table relocation with spare slots
    - Section data relocation for alignment

    Net result: binary grows by ~1-2KB instead of shrinking.

    This reproduces CI failures like:
        Error: Binary was not stripped or grew in size: .../test_tuple
        Original: 764864 bytes, New: 766019 bytes

    Returns:
        Path to the patched binary
    """
    source = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
    output = tmp_path / "marginal_hip_fatbin.exe"
    return patch_hip_fatbin_size(source, output, new_size=5000)
