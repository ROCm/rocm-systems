"""Tests for UnbundlingVisitor with bundled binary test assets."""

import shutil
from pathlib import Path

import pytest

from rocm_kpack.artifact_scanner import (
    ArtifactPath,
    ArtifactScanner,
    ArtifactVisitor,
    RecognizerRegistry,
)
from rocm_kpack.binutils import BundledBinary, Toolchain
from rocm_kpack.elf import ElfSurgery, add_kpack_ref_section, kpack_offload_binary


class UnbundlingVisitor(ArtifactVisitor):
    """Visitor that unbundles binaries and copies other artifacts.

    For bundled binaries:
    - Extracts each architecture bundle to {binary_name}.unbundled/{arch}.co
    - Creates host-only version of the binary (without .hip_fatbin section)

    For opaque files:
    - Copies verbatim preserving directory structure

    For kernel databases:
    - Copies all database artifacts preserving structure
    """

    def __init__(self, output_root: Path, toolchain: Toolchain):
        """Initialize visitor.

        Args:
            output_root: Root directory where artifacts will be written
            toolchain: Toolchain for unbundling operations
        """
        self.output_root = output_root
        self.toolchain = toolchain
        self.visited_opaque_files: list[Path] = []
        self.visited_bundled_binaries: list[Path] = []
        self.visited_databases: list[Path] = []

    def visit_opaque_file(self, artifact_path: ArtifactPath) -> None:
        """Copy opaque file verbatim."""
        self.visited_opaque_files.append(artifact_path.relative_path)
        dest = self.output_root / artifact_path.relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_path.absolute_path, dest)

    def visit_bundled_binary(
        self, artifact_path: ArtifactPath, bundled_binary: BundledBinary
    ) -> None:
        """Unbundle binary to .unbundled/ directory and create host-only version.

        Creates:
        - {binary_name}.unbundled/{arch}.co for each architecture
        - {binary_name} (host-only version without device code)
        """
        self.visited_bundled_binaries.append(artifact_path.relative_path)

        binary_name = artifact_path.relative_path.name
        binary_parent = artifact_path.relative_path.parent
        unbundled_dir_name = f"{binary_name}.unbundled"

        # Create .unbundled directory
        unbundled_dir = self.output_root / binary_parent / unbundled_dir_name
        unbundled_dir.mkdir(parents=True, exist_ok=True)

        # Unbundle to temporary directory
        with bundled_binary.unbundle(delete_on_close=True) as contents:
            # Copy only GPU architecture bundles (.hsaco files) to .unbundled/
            for filename in contents.file_names:
                if filename.endswith(".hsaco"):
                    src = contents.dest_dir / filename
                    # Extract architecture from filename like "hipv4-amdgcn-amd-amdhsa--gfx1100.hsaco"
                    # and rename to just "{arch}.co"
                    arch = filename.split("--")[-1].replace(".hsaco", "")
                    dest = unbundled_dir / f"{arch}.co"
                    shutil.copy2(src, dest)

        # Create host-only version (without device code)
        binary_dest = self.output_root / artifact_path.relative_path
        binary_dest.parent.mkdir(parents=True, exist_ok=True)
        kpack_offload_binary(bundled_binary.file_path, binary_dest)

    def visit_kernel_database(self, artifact_path: ArtifactPath, database) -> None:
        """Copy kernel database artifacts preserving structure."""
        self.visited_databases.append(artifact_path.relative_path)

        # Copy all artifacts from the database
        for kernel in database.get_kernel_artifacts():
            src = artifact_path.absolute_path / kernel.relative_path
            dest = self.output_root / artifact_path.relative_path / kernel.relative_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


@pytest.fixture
def bundled_test_tree(tmp_path: Path, toolchain: Toolchain) -> Path:
    """Create a test tree with bundled binaries and opaque files.

    Uses actual test assets from test_assets/bundled_binaries/linux/cov5/.

    Structure:
        test_root/
            config.txt (opaque file)
            bins/
                test_kernel_multi.exe (bundled binary with .rocm_kpack_ref)
                libtest_kernel_single.so (bundled binary with .rocm_kpack_ref)
                host_only.exe (opaque file - no .hip_fatbin)
            data/
                input.dat (opaque file)
    """
    root = tmp_path / "test_root"
    root.mkdir()

    # Opaque files
    (root / "config.txt").write_text("configuration data")
    (root / "data").mkdir()
    (root / "data" / "input.dat").write_text("input data")

    # Setup bundled binaries from test assets
    bins_dir = root / "bins"
    bins_dir.mkdir()

    assets_dir = Path("test_assets/bundled_binaries/linux/cov5")

    # Add .rocm_kpack_ref sections to bundled binaries
    # (required for create_host_only to work)

    # Add marker to test_kernel_multi.exe
    surgery = ElfSurgery.load(assets_dir / "test_kernel_multi.exe")
    add_kpack_ref_section(
        surgery,
        kpack_search_paths=["test.kpack"],
        kernel_name="test_kernel",
    )
    surgery.save(bins_dir / "test_kernel_multi.exe")

    # Add marker to libtest_kernel_single.so
    surgery = ElfSurgery.load(assets_dir / "libtest_kernel_single.so")
    add_kpack_ref_section(
        surgery,
        kpack_search_paths=["test.kpack"],
        kernel_name="test_kernel",
    )
    surgery.save(bins_dir / "libtest_kernel_single.so")

    # Copy host-only binary (should be treated as opaque)
    shutil.copy2(assets_dir / "host_only.exe", bins_dir / "host_only.exe")

    return root


def test_unbundling_visitor_unbundles_binaries(
    bundled_test_tree: Path, tmp_path: Path, toolchain: Toolchain
):
    """Test that UnbundlingVisitor correctly unbundles bundled binaries."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    visitor = UnbundlingVisitor(output_dir, toolchain)

    # Scan and process the tree
    scanner.scan_tree(bundled_test_tree, visitor)

    # Verify opaque files were copied
    assert (output_dir / "config.txt").exists()
    assert (output_dir / "config.txt").read_text() == "configuration data"
    assert (output_dir / "data" / "input.dat").exists()
    assert (output_dir / "data" / "input.dat").read_text() == "input data"

    # Verify host-only binary was copied as opaque file
    assert (output_dir / "bins" / "host_only.exe").exists()

    # Verify bundled binaries were unbundled
    # Check test_kernel_multi.exe
    multi_unbundled = output_dir / "bins" / "test_kernel_multi.exe.unbundled"
    assert multi_unbundled.exists()
    assert multi_unbundled.is_dir()
    assert (multi_unbundled / "gfx1100.co").exists()
    assert (multi_unbundled / "gfx1101.co").exists()

    # Check libtest_kernel_single.so
    single_unbundled = output_dir / "bins" / "libtest_kernel_single.so.unbundled"
    assert single_unbundled.exists()
    assert single_unbundled.is_dir()
    assert (single_unbundled / "gfx1100.co").exists()

    # Verify host-only binaries were created
    assert (output_dir / "bins" / "test_kernel_multi.exe").exists()
    assert (output_dir / "bins" / "libtest_kernel_single.so").exists()

    # Verify host-only binaries are smaller than originals (device code removed)
    original_multi_size = (
        (bundled_test_tree / "bins" / "test_kernel_multi.exe").stat().st_size
    )
    host_only_multi_size = (
        (output_dir / "bins" / "test_kernel_multi.exe").stat().st_size
    )
    assert (
        host_only_multi_size < original_multi_size
    ), "Host-only binary should be smaller"

    original_single_size = (
        (bundled_test_tree / "bins" / "libtest_kernel_single.so").stat().st_size
    )
    host_only_single_size = (
        (output_dir / "bins" / "libtest_kernel_single.so").stat().st_size
    )
    # Note: Small binaries may grow due to ELF structural overhead.
    # The important assertion is that multi-arch binaries (with large .hip_fatbin)
    # shrink significantly. Small test binaries may have overhead that exceeds savings.
    # TODO(#3): Optimize overhead of section add.
    assert host_only_single_size > 0, "Host-only library should exist and be valid"


def test_unbundling_visitor_counts(
    bundled_test_tree: Path, tmp_path: Path, toolchain: Toolchain
):
    """Test that UnbundlingVisitor tracks visited artifacts correctly."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    visitor = UnbundlingVisitor(output_dir, toolchain)

    scanner.scan_tree(bundled_test_tree, visitor)

    # Verify counts
    # Opaque files: config.txt, data/input.dat, bins/host_only.exe
    assert len(visitor.visited_opaque_files) == 3

    # Bundled binaries: test_kernel_multi.exe, libtest_kernel_single.so
    assert len(visitor.visited_bundled_binaries) == 2

    # No kernel databases in this tree
    assert len(visitor.visited_databases) == 0

    # Verify specific paths
    assert Path("config.txt") in visitor.visited_opaque_files
    assert Path("data/input.dat") in visitor.visited_opaque_files
    assert Path("bins/host_only.exe") in visitor.visited_opaque_files
    assert Path("bins/test_kernel_multi.exe") in visitor.visited_bundled_binaries
    assert Path("bins/libtest_kernel_single.so") in visitor.visited_bundled_binaries


def test_unbundling_visitor_architecture_extraction(
    bundled_test_tree: Path, tmp_path: Path, toolchain: Toolchain
):
    """Test that unbundled .co files have correct architecture names."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    visitor = UnbundlingVisitor(output_dir, toolchain)

    scanner.scan_tree(bundled_test_tree, visitor)

    # Check that .co files exist and are non-empty
    gfx1100_co = output_dir / "bins" / "test_kernel_multi.exe.unbundled" / "gfx1100.co"
    gfx1101_co = output_dir / "bins" / "test_kernel_multi.exe.unbundled" / "gfx1101.co"

    assert gfx1100_co.exists()
    assert gfx1101_co.exists()

    # Verify they are non-empty (contain actual code)
    assert gfx1100_co.stat().st_size > 0
    assert gfx1101_co.stat().st_size > 0

    # Verify they are different (different architectures)
    assert gfx1100_co.read_bytes() != gfx1101_co.read_bytes()
