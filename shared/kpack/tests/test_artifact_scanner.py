"""Unit tests for artifact scanner functionality."""

import shutil
from pathlib import Path
from typing import Iterator

import pytest

from rocm_kpack.artifact_scanner import (
    ArtifactPath,
    ArtifactScanner,
    ArtifactVisitor,
    DatabaseRecognizer,
    KernelArtifact,
    KernelDatabase,
    RecognizerRegistry,
)


# Dummy database implementation for testing
class DummyKernelDatabase(KernelDatabase):
    """A simple kernel database for testing.

    This represents a directory containing marker file 'kernel.db' and
    architecture-specific kernel files.
    """

    def get_architectures(self) -> list[str]:
        """Scan for gfx targets in filenames."""
        architectures = set()
        for item in self.absolute_path.iterdir():
            if item.is_file() and item.stem.startswith("kernel_"):
                # Extract gfx target from filename like kernel_gfx1100.hsaco
                parts = item.stem.split("_")
                if len(parts) >= 2 and parts[1].startswith("gfx"):
                    architectures.add(parts[1])
        return sorted(architectures)

    def get_kernel_artifacts(self) -> Iterator[KernelArtifact]:
        """Yield kernel artifacts from this database."""
        for item in self.absolute_path.iterdir():
            if item.is_file() and item.stem.startswith("kernel_"):
                parts = item.stem.split("_")
                if len(parts) >= 2 and parts[1].startswith("gfx"):
                    gfx_target = parts[1]
                    artifact_type = "hsaco" if item.suffix == ".hsaco" else "metadata"
                    yield KernelArtifact(
                        relative_path=Path(item.name),
                        gfx_target=gfx_target,
                        artifact_type=artifact_type,
                    )


class DummyDatabaseRecognizer(DatabaseRecognizer):
    """Recognizes directories containing a 'kernel.db' marker file."""

    def can_recognize(self, artifact_path: ArtifactPath) -> bool:
        """Check if directory contains kernel.db marker."""
        abs_path = artifact_path.absolute_path
        return abs_path.is_dir() and (abs_path / "kernel.db").exists()

    def recognize(self, artifact_path: ArtifactPath) -> KernelDatabase | None:
        """Create a DummyKernelDatabase if recognized."""
        if not self.can_recognize(artifact_path):
            return None
        return DummyKernelDatabase(artifact_path)


# CopyVisitor implementation
class CopyVisitor(ArtifactVisitor):
    """Visitor that copies artifacts to an output directory."""

    def __init__(self, output_root: Path):
        """Initialize with output directory.

        Args:
            output_root: Root directory where artifacts will be copied
        """
        self.output_root = output_root
        self.visited_opaque_files: list[Path] = []
        self.visited_databases: list[Path] = []
        self.visited_bundled_binaries: list[Path] = []

    def visit_opaque_file(self, artifact_path: ArtifactPath) -> None:
        """Copy opaque file preserving relative path structure."""
        self.visited_opaque_files.append(artifact_path.relative_path)
        dest = self.output_root / artifact_path.relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_path.absolute_path, dest)

    def visit_bundled_binary(self, artifact_path, bundled_binary) -> None:
        """Record bundled binary visits (not implemented for this test)."""
        self.visited_bundled_binaries.append(artifact_path.relative_path)

    def visit_kernel_database(self, artifact_path, database) -> None:
        """Copy kernel database artifacts."""
        self.visited_databases.append(artifact_path.relative_path)

        # Copy the database marker file
        marker_src = artifact_path.absolute_path / "kernel.db"
        marker_dest = self.output_root / artifact_path.relative_path / "kernel.db"
        marker_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(marker_src, marker_dest)

        # Copy kernel artifacts
        for kernel in database.get_kernel_artifacts():
            src = artifact_path.absolute_path / kernel.relative_path
            dest = self.output_root / artifact_path.relative_path / kernel.relative_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


# Tests
@pytest.fixture
def test_tree(tmp_path: Path) -> Path:
    """Create a test directory tree with opaque files and a dummy database.

    Structure:
        root/
            file1.txt
            subdir1/
                file2.txt
                file3.dat
            subdir2/
                kernels/
                    kernel.db (marker file)
                    kernel_gfx1100.hsaco
                    kernel_gfx1201.hsaco
                    kernel_gfx1100.dat
            file4.log
    """
    root = tmp_path / "test_root"
    root.mkdir()

    # Opaque files
    (root / "file1.txt").write_text("content1")
    (root / "subdir1").mkdir()
    (root / "subdir1" / "file2.txt").write_text("content2")
    (root / "subdir1" / "file3.dat").write_text("content3")
    (root / "file4.log").write_text("content4")

    # Kernel database
    kernels_dir = root / "subdir2" / "kernels"
    kernels_dir.mkdir(parents=True)
    (kernels_dir / "kernel.db").write_text("database marker")
    (kernels_dir / "kernel_gfx1100.hsaco").write_text("gfx1100 kernel code")
    (kernels_dir / "kernel_gfx1201.hsaco").write_text("gfx1201 kernel code")
    (kernels_dir / "kernel_gfx1100.dat").write_text("gfx1100 metadata")

    return root


def test_copy_visitor_copies_tree_correctly(test_tree: Path, tmp_path: Path):
    """Test that CopyVisitor correctly copies a tree with opaque files and a database."""
    # Setup
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    registry = RecognizerRegistry()
    registry.register(DummyDatabaseRecognizer())

    scanner = ArtifactScanner(registry, toolchain=None)
    visitor = CopyVisitor(output_dir)

    # Execute
    scanner.scan_tree(test_tree, visitor)

    # Verify opaque files were copied
    assert (output_dir / "file1.txt").read_text() == "content1"
    assert (output_dir / "subdir1" / "file2.txt").read_text() == "content2"
    assert (output_dir / "subdir1" / "file3.dat").read_text() == "content3"
    assert (output_dir / "file4.log").read_text() == "content4"

    # Verify database was copied
    assert (output_dir / "subdir2" / "kernels" / "kernel.db").read_text() == (
        "database marker"
    )
    assert (
        output_dir / "subdir2" / "kernels" / "kernel_gfx1100.hsaco"
    ).read_text() == ("gfx1100 kernel code")
    assert (
        output_dir / "subdir2" / "kernels" / "kernel_gfx1201.hsaco"
    ).read_text() == ("gfx1201 kernel code")
    assert (output_dir / "subdir2" / "kernels" / "kernel_gfx1100.dat").read_text() == (
        "gfx1100 metadata"
    )

    # Verify visitor tracking
    assert Path("file1.txt") in visitor.visited_opaque_files
    assert Path("subdir1/file2.txt") in visitor.visited_opaque_files
    assert Path("subdir1/file3.dat") in visitor.visited_opaque_files
    assert Path("file4.log") in visitor.visited_opaque_files
    assert Path("subdir2/kernels") in visitor.visited_databases


def test_database_architectures(test_tree: Path):
    """Test that the dummy database correctly identifies architectures."""
    registry = RecognizerRegistry()
    registry.register(DummyDatabaseRecognizer())

    scanner = ArtifactScanner(registry)
    databases_found = []

    class RecordingVisitor(ArtifactVisitor):
        def visit_kernel_database(self, artifact_path, database):
            databases_found.append(database)

    visitor = RecordingVisitor()
    scanner.scan_tree(test_tree, visitor)

    assert len(databases_found) == 1
    db = databases_found[0]
    assert set(db.get_architectures()) == {"gfx1100", "gfx1201"}


def test_database_kernel_artifacts(test_tree: Path):
    """Test that the dummy database correctly lists kernel artifacts."""
    registry = RecognizerRegistry()
    registry.register(DummyDatabaseRecognizer())

    scanner = ArtifactScanner(registry)
    databases_found = []

    class RecordingVisitor(ArtifactVisitor):
        def visit_kernel_database(self, artifact_path, database):
            databases_found.append(database)

    visitor = RecordingVisitor()
    scanner.scan_tree(test_tree, visitor)

    assert len(databases_found) == 1
    db = databases_found[0]

    artifacts = list(db.get_kernel_artifacts())
    assert len(artifacts) == 3

    # Check that we have the expected artifacts
    artifact_paths = {a.relative_path for a in artifacts}
    assert Path("kernel_gfx1100.hsaco") in artifact_paths
    assert Path("kernel_gfx1201.hsaco") in artifact_paths
    assert Path("kernel_gfx1100.dat") in artifact_paths

    # Check gfx targets
    gfx_targets = {a.gfx_target for a in artifacts}
    assert gfx_targets == {"gfx1100", "gfx1201"}


def test_scanner_does_not_double_visit_database_files(test_tree: Path):
    """Test that files inside a recognized database are not visited as opaque files."""
    registry = RecognizerRegistry()
    registry.register(DummyDatabaseRecognizer())

    scanner = ArtifactScanner(registry)
    visitor = CopyVisitor(tmp_path := Path("/tmp/unused"))

    scanner.scan_tree(test_tree, visitor)

    # Database files should NOT appear in opaque files list
    for opaque_path in visitor.visited_opaque_files:
        assert not opaque_path.is_relative_to(Path("subdir2/kernels"))

    # But database should be visited
    assert Path("subdir2/kernels") in visitor.visited_databases
