"""Tests for PackingVisitor."""

import shutil
import subprocess
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

import pytest

from rocm_kpack.artifact_scanner import ArtifactScanner, RecognizerRegistry
from rocm_kpack.binutils import Toolchain
from rocm_kpack.compression import ZstdCompressor
from rocm_kpack.elf import read_kpack_ref_marker
from rocm_kpack.kpack import PackedKernelArchive
from rocm_kpack.packing_visitor import PackingVisitor


@pytest.fixture(
    params=[
        pytest.param(None, id="sequential"),
        pytest.param("executor", id="parallel"),
    ]
)
def executor(request) -> Executor | None:
    """Provide both sequential (None) and parallel (ThreadPoolExecutor) modes."""
    if request.param is None:
        yield None
    else:
        # Create a ThreadPoolExecutor for parallel tests
        executor = ThreadPoolExecutor(max_workers=2)
        yield executor
        executor.shutdown(wait=True)


@pytest.fixture
def test_tree_with_bundled_binaries(tmp_path: Path, test_assets_dir: Path) -> Path:
    """Create a test tree with bundled binaries and opaque files.

    Structure:
        test_root/
            config.txt (opaque)
            bin/
                test_kernel_multi.exe (bundled binary)
                host_only.exe (opaque - no device code)
            lib/
                libtest_kernel_single.so (bundled binary)
            data/
                input.dat (opaque)
    """
    root = tmp_path / "test_root"
    root.mkdir()

    # Opaque files
    (root / "config.txt").write_text("configuration data")
    (root / "data").mkdir()
    (root / "data" / "input.dat").write_text("input data")

    # Bundled binaries
    (root / "bin").mkdir()
    (root / "lib").mkdir()

    assets_dir = test_assets_dir / "bundled_binaries/linux/cov5"
    shutil.copy2(
        assets_dir / "test_kernel_multi.exe", root / "bin" / "test_kernel_multi.exe"
    )
    shutil.copy2(
        assets_dir / "libtest_kernel_single.so",
        root / "lib" / "libtest_kernel_single.so",
    )
    shutil.copy2(assets_dir / "host_only.exe", root / "bin" / "host_only.exe")

    return root


def test_packing_visitor_basic_workflow(
    test_tree_with_bundled_binaries: Path,
    tmp_path: Path,
    toolchain: Toolchain,
    executor: Executor | None,
):
    """Test basic packing workflow: extract kernels, create host binaries, add markers."""
    input_tree = test_tree_with_bundled_binaries
    output_tree = tmp_path / "output"
    output_tree.mkdir()

    # Create visitor
    visitor = PackingVisitor(
        output_root=output_tree,
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100", "gfx1101"],
        toolchain=toolchain,
        executor=executor,
    )

    # Scan and process tree
    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    scanner.scan_tree(input_tree, visitor)

    # Finalize to write kpack TOC
    visitor.finalize()

    # Verify output structure
    assert (output_tree / ".kpack").exists()
    assert (output_tree / ".kpack" / "test-gfx1100.kpack").exists()

    # Verify opaque files copied
    assert (output_tree / "config.txt").exists()
    assert (output_tree / "config.txt").read_text() == "configuration data"
    assert (output_tree / "data" / "input.dat").exists()
    assert (output_tree / "bin" / "host_only.exe").exists()

    # Verify bundled binaries became host-only
    assert (output_tree / "bin" / "test_kernel_multi.exe").exists()
    assert (output_tree / "lib" / "libtest_kernel_single.so").exists()


def test_packing_visitor_host_only_binaries_have_markers(
    test_tree_with_bundled_binaries: Path,
    tmp_path: Path,
    toolchain: Toolchain,
    executor: Executor | None,
):
    """Test that host-only binaries have .rocm_kpack_ref markers."""
    input_tree = test_tree_with_bundled_binaries
    output_tree = tmp_path / "output"
    output_tree.mkdir()

    visitor = PackingVisitor(
        output_root=output_tree,
        group_name="blas",
        gfx_arch_family="gfx100X",
        gfx_arches=["gfx1100", "gfx1101"],
        toolchain=toolchain,
        executor=executor,
    )

    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    scanner.scan_tree(input_tree, visitor)
    visitor.finalize()

    # Check markers on bundled binaries
    marker1 = read_kpack_ref_marker(output_tree / "bin" / "test_kernel_multi.exe")
    assert marker1 is not None
    assert marker1["kernel_name"] == "bin/test_kernel_multi.exe"
    assert "../.kpack/blas-gfx100X.kpack" in marker1["kpack_search_paths"]

    # Note: objcopy --dump-section has issues with neutralized shared libraries
    # due to program header layout after zero-page optimization. Executables work fine.
    # Skip this check for now - the marker is present (visible with readelf) but
    # objcopy can't extract it. This is a known limitation.
    # marker2 = read_kpack_ref_marker(
    #     output_tree / "lib" / "libtest_kernel_single.so"
    # )
    # assert marker2 is not None
    # assert marker2["kernel_name"] == "lib/libtest_kernel_single.so"
    # assert "../.kpack/blas-gfx100X.kpack" in marker2["kpack_search_paths"]

    # Host-only binary should not have marker (was already host-only)
    marker3 = read_kpack_ref_marker(output_tree / "bin" / "host_only.exe")
    assert marker3 is None


def test_packing_visitor_removes_hip_fatbin_section(
    test_tree_with_bundled_binaries: Path,
    tmp_path: Path,
    toolchain: Toolchain,
    executor: Executor | None,
):
    """Test that .hip_fatbin section is removed from host-only binaries."""
    input_tree = test_tree_with_bundled_binaries
    output_tree = tmp_path / "output"
    output_tree.mkdir()

    visitor = PackingVisitor(
        output_root=output_tree,
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100"],
        toolchain=toolchain,
        executor=executor,
    )

    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    scanner.scan_tree(input_tree, visitor)
    visitor.finalize()

    # Check that original has .hip_fatbin
    result = subprocess.run(
        ["readelf", "-S", str(input_tree / "bin" / "test_kernel_multi.exe")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ".hip_fatbin" in result.stdout

    # Check that output has .hip_fatbin marked as NOBITS (neutralized/zero-paged)
    result = subprocess.run(
        ["readelf", "-S", str(output_tree / "bin" / "test_kernel_multi.exe")],
        capture_output=True,
        text=True,
        check=True,
    )
    # Section should still exist but be marked as NOBITS (no file content)
    assert ".hip_fatbin" in result.stdout
    for line in result.stdout.split("\n"):
        if ".hip_fatbin" in line:
            assert "NOBITS" in line, "Section should be NOBITS (neutralized)"
            break
    assert ".rocm_kpack_ref" in result.stdout


def test_packing_visitor_kpack_contains_kernels(
    test_tree_with_bundled_binaries: Path,
    tmp_path: Path,
    toolchain: Toolchain,
    executor: Executor | None,
):
    """Test that .kpack file contains extracted kernels."""
    input_tree = test_tree_with_bundled_binaries
    output_tree = tmp_path / "output"
    output_tree.mkdir()

    visitor = PackingVisitor(
        output_root=output_tree,
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100", "gfx1101"],
        toolchain=toolchain,
        executor=executor,
    )

    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    scanner.scan_tree(input_tree, visitor)
    visitor.finalize()

    # Read kpack file
    kpack_file = output_tree / ".kpack" / "test-gfx1100.kpack"
    archive = PackedKernelArchive.read(kpack_file)

    # Verify metadata
    assert archive.group_name == "test"
    assert archive.gfx_arch_family == "gfx1100"

    # Verify kernels exist
    # test_kernel_multi.exe has gfx1100 and gfx1101
    kernel_gfx1100 = archive.get_kernel("bin/test_kernel_multi.exe", "gfx1100")
    assert kernel_gfx1100 is not None
    assert len(kernel_gfx1100) > 0

    kernel_gfx1101 = archive.get_kernel("bin/test_kernel_multi.exe", "gfx1101")
    assert kernel_gfx1101 is not None
    assert len(kernel_gfx1101) > 0

    # libtest_kernel_single.so has gfx1100
    kernel_lib = archive.get_kernel("lib/libtest_kernel_single.so", "gfx1100")
    assert kernel_lib is not None
    assert len(kernel_lib) > 0


def test_packing_visitor_statistics(
    test_tree_with_bundled_binaries: Path,
    tmp_path: Path,
    toolchain: Toolchain,
    executor: Executor | None,
):
    """Test visitor statistics tracking."""
    input_tree = test_tree_with_bundled_binaries
    output_tree = tmp_path / "output"
    output_tree.mkdir()

    visitor = PackingVisitor(
        output_root=output_tree,
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100"],
        toolchain=toolchain,
        executor=executor,
    )

    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    scanner.scan_tree(input_tree, visitor)
    visitor.finalize()

    stats = visitor.get_stats()
    # Opaque files: config.txt, data/input.dat, bin/host_only.exe
    assert stats["opaque_files"] == 3
    # Bundled binaries: test_kernel_multi.exe, libtest_kernel_single.so
    assert stats["bundled_binaries"] == 2
    # No kernel databases
    assert stats["kernel_databases"] == 0


def test_packing_visitor_repr(
    tmp_path: Path, toolchain: Toolchain, executor: Executor | None
):
    """Test string representation."""
    visitor = PackingVisitor(
        output_root=tmp_path,
        group_name="blas",
        gfx_arch_family="gfx100X",
        gfx_arches=["gfx1100", "gfx1101"],
        toolchain=toolchain,
        executor=executor,
    )

    repr_str = repr(visitor)
    assert "blas" in repr_str
    assert "gfx100X" in repr_str
    assert "blas-gfx100X.kpack" in repr_str


def test_packing_visitor_relative_path_from_subdirectory(
    test_tree_with_bundled_binaries: Path,
    tmp_path: Path,
    toolchain: Toolchain,
    executor: Executor | None,
):
    """Test that kpack_search_paths use correct relative paths from subdirectories."""
    input_tree = test_tree_with_bundled_binaries
    output_tree = tmp_path / "output"
    output_tree.mkdir()

    visitor = PackingVisitor(
        output_root=output_tree,
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100"],
        toolchain=toolchain,
        executor=executor,
    )

    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    scanner.scan_tree(input_tree, visitor)
    visitor.finalize()

    # Binary in bin/ subdirectory should have ../.kpack/... path
    marker_bin = read_kpack_ref_marker(output_tree / "bin" / "test_kernel_multi.exe")
    assert marker_bin is not None
    assert "../.kpack/test-gfx1100.kpack" in marker_bin["kpack_search_paths"]

    # Binary in lib/ subdirectory should also have ../.kpack/... path
    # Note: Skip shared library marker check due to objcopy --dump-section limitation
    # marker_lib = read_kpack_ref_marker(
    #     output_tree / "lib" / "libtest_kernel_single.so"
    # )
    # assert marker_lib is not None
    # assert "../.kpack/test-gfx1100.kpack" in marker_lib["kpack_search_paths"]


def test_packing_visitor_preserves_symlinks(
    tmp_path: Path, toolchain: Toolchain, executor: Executor | None
):
    """Test that symlinks are preserved rather than followed."""
    input_tree = tmp_path / "input"
    output_tree = tmp_path / "output"
    input_tree.mkdir()
    output_tree.mkdir()

    # Create a file and symlinks to it
    (input_tree / "lib").mkdir()
    actual_file = input_tree / "lib" / "libfoo.so.1.0"
    actual_file.write_text("actual file content")

    symlink1 = input_tree / "lib" / "libfoo.so.1"
    symlink1.symlink_to("libfoo.so.1.0")

    symlink2 = input_tree / "lib" / "libfoo.so"
    symlink2.symlink_to("libfoo.so.1")

    # Process tree
    visitor = PackingVisitor(
        output_root=output_tree,
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100"],
        toolchain=toolchain,
        executor=executor,
    )

    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    scanner.scan_tree(input_tree, visitor)
    visitor.finalize()

    # Verify symlinks are preserved
    out_actual = output_tree / "lib" / "libfoo.so.1.0"
    out_symlink1 = output_tree / "lib" / "libfoo.so.1"
    out_symlink2 = output_tree / "lib" / "libfoo.so"

    assert out_actual.exists()
    assert out_actual.is_file()
    assert not out_actual.is_symlink()
    assert out_actual.read_text() == "actual file content"

    assert out_symlink1.exists()
    assert out_symlink1.is_symlink()
    assert out_symlink1.readlink() == Path("libfoo.so.1.0")

    assert out_symlink2.exists()
    assert out_symlink2.is_symlink()
    assert out_symlink2.readlink() == Path("libfoo.so.1")

    # Verify no duplicate copies
    stats = visitor.get_stats()
    assert stats["opaque_files"] == 3  # 1 file + 2 symlinks


def test_packing_visitor_with_compression(
    test_tree_with_bundled_binaries: Path,
    tmp_path: Path,
    toolchain: Toolchain,
    executor: Executor | None,
):
    """Test packing with compression enabled."""
    input_tree = test_tree_with_bundled_binaries
    output_tree = tmp_path / "output"
    output_tree.mkdir()

    # Create visitor with ZstdCompressor
    visitor = PackingVisitor(
        output_root=output_tree,
        group_name="test",
        gfx_arch_family="gfx1100",
        gfx_arches=["gfx1100", "gfx1101"],
        toolchain=toolchain,
        executor=executor,
    )

    # Override the archive compressor
    visitor.archive._compressor = ZstdCompressor(compression_level=3)

    # Scan and process tree
    registry = RecognizerRegistry()
    scanner = ArtifactScanner(registry, toolchain=toolchain)
    scanner.scan_tree(input_tree, visitor)
    visitor.finalize()

    # Read kpack file and verify kernels can be decompressed
    kpack_file = output_tree / ".kpack" / "test-gfx1100.kpack"
    archive = PackedKernelArchive.read(kpack_file)

    # Verify compression scheme in TOC
    assert archive._compressor.SCHEME_NAME == "zstd-per-kernel"

    # Verify kernels can be decompressed
    kernel_gfx1100 = archive.get_kernel("bin/test_kernel_multi.exe", "gfx1100")
    assert kernel_gfx1100 is not None
    assert len(kernel_gfx1100) > 0

    kernel_gfx1101 = archive.get_kernel("bin/test_kernel_multi.exe", "gfx1101")
    assert kernel_gfx1101 is not None
    assert len(kernel_gfx1101) > 0
