"""
Integration tests for the artifact splitter.

These tests simulate real artifact splitting scenarios with mock data.
"""

import shutil
from argparse import Namespace
from pathlib import Path

import pytest

from rocm_kpack.artifact_splitter import (
    ArtifactSplitter,
    FileClassificationVisitor,
    ExtractedKernel,
)
from rocm_kpack.artifact_utils import read_artifact_manifest, write_artifact_manifest
from rocm_kpack.database_handlers import RocBLASHandler
from rocm_kpack.tools.split_artifacts import batch_split, parse_artifact_name
from rocm_kpack.tools.verify_artifacts import ArtifactVerifier


class TestArtifactSplitterIntegration:
    """Integration tests for the complete artifact splitting workflow."""

    @pytest.fixture
    def create_test_artifact(self, tmp_path):
        """Create a test artifact directory structure."""

        def _create(
            prefixes,
            files_per_prefix=5,
            include_fat_binaries=False,
            include_db_files=False,
        ):
            artifact_dir = tmp_path / "test_artifact"
            artifact_dir.mkdir()

            # Write artifact manifest
            write_artifact_manifest(artifact_dir, prefixes)

            # Create prefix directories and files
            for prefix in prefixes:
                prefix_dir = artifact_dir / prefix
                prefix_dir.mkdir(parents=True)

                # Create regular files
                lib_dir = prefix_dir / "lib"
                lib_dir.mkdir(parents=True, exist_ok=True)

                for i in range(files_per_prefix):
                    file_path = lib_dir / f"libtest{i}.so"
                    file_path.write_text(f"Mock library content {i}")

                # Optionally create fat binaries (mock)
                if include_fat_binaries:
                    fat_bin = lib_dir / "libfat.so"
                    fat_bin.write_text("Mock fat binary with device code")
                    # Mark it for our tests
                    (lib_dir / ".test_fat_marker").write_text("libfat.so")

                # Optionally create database files
                if include_db_files:
                    db_dir = lib_dir / "rocblas" / "library"
                    db_dir.mkdir(parents=True, exist_ok=True)

                    # Create mock rocBLAS database files
                    (db_dir / "TensileLibrary_gfx1100.dat").write_text(
                        "Mock tensor data"
                    )
                    (db_dir / "TensileLibrary_gfx1100.co").write_text(
                        "Mock code object"
                    )
                    (db_dir / "kernels.db").write_text("Mock kernel database")

            return artifact_dir

        return _create

    def test_simple_artifact_split(self, create_test_artifact, toolchain, tmp_path):
        """Test splitting a simple artifact without fat binaries or databases."""
        # Create test artifact with plain text files (not ELF)
        input_dir = create_test_artifact(
            prefixes=["math-libs/BLAS/rocBLAS/stage"], files_per_prefix=3
        )

        output_dir = tmp_path / "output"

        # Create splitter with real toolchain
        splitter = ArtifactSplitter(
            artifact_prefix="test_lib",
            toolchain=toolchain,
            database_handlers=[],
            verbose=True,
        )

        # Run the split - text files won't be detected as fat binaries
        splitter.split(input_dir, output_dir)

        # Verify output structure
        assert output_dir.exists()

        # Should have created generic artifact
        generic_dir = output_dir / "test_lib_generic"
        assert generic_dir.exists()

        # Check manifest was copied
        generic_manifest = read_artifact_manifest(generic_dir)
        assert generic_manifest == ["math-libs/BLAS/rocBLAS/stage"]

        # Check files were copied
        generic_prefix = generic_dir / "math-libs/BLAS/rocBLAS/stage"
        assert generic_prefix.exists()
        assert (generic_prefix / "lib" / "libtest0.so").exists()
        assert (generic_prefix / "lib" / "libtest1.so").exists()
        assert (generic_prefix / "lib" / "libtest2.so").exists()

    def test_artifact_with_fat_binaries(self, test_assets_dir, toolchain, tmp_path):
        """Test splitting artifact with real fat binaries from test assets."""
        # Create test artifact structure
        input_dir = tmp_path / "test_artifact"
        input_dir.mkdir()

        # Create artifact manifest
        prefix = "test/lib/stage"
        write_artifact_manifest(input_dir, [prefix])

        # Create prefix directory
        prefix_dir = input_dir / prefix
        lib_dir = prefix_dir / "lib"
        lib_dir.mkdir(parents=True)

        # Copy real fat binary from test assets
        fat_binary_src = (
            test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_multi.so"
        )
        fat_binary_dest = lib_dir / "libtest.so"
        shutil.copy2(fat_binary_src, fat_binary_dest)

        # Also copy a host-only library
        host_only_src = test_assets_dir / "bundled_binaries/linux/cov5/libhost_only.so"
        host_only_dest = lib_dir / "libhost.so"
        shutil.copy2(host_only_src, host_only_dest)

        output_dir = tmp_path / "output"

        # Create splitter with real toolchain - run everything live
        splitter = ArtifactSplitter(
            artifact_prefix="test_lib",
            toolchain=toolchain,
            database_handlers=[],
            verbose=True,
        )

        # Run the full split operation
        splitter.split(input_dir, output_dir)

        # Verify generic artifact was created
        generic_dir = output_dir / "test_lib_generic"
        assert generic_dir.exists()

        # Check that both libraries were copied to generic
        generic_lib_dir = generic_dir / prefix / "lib"
        assert (generic_lib_dir / "libtest.so").exists()
        assert (generic_lib_dir / "libhost.so").exists()

        # Verify fat binary was detected and kernels extracted
        # Should have created architecture-specific artifacts
        # The test binary has gfx1100 and gfx1101 kernels
        arch_artifacts = list(output_dir.glob("test_lib_gfx*"))
        assert (
            len(arch_artifacts) >= 1
        ), "Should have created at least one architecture-specific artifact"

        # Check that kpack files were created (using original prefix, not synthetic)
        for arch_artifact in arch_artifacts:
            kpack_files = list(arch_artifact.glob(f"{prefix}/.kpack/*.kpack"))
            assert (
                len(kpack_files) == 1
            ), f"Should have one kpack file in {arch_artifact}/{prefix}/.kpack/"

        # Verify no .kpm manifest was created (patterns replace manifests)
        kpm_files = (
            list((generic_dir / prefix / ".kpack").glob("*.kpm"))
            if (generic_dir / prefix / ".kpack").exists()
            else []
        )
        assert (
            len(kpm_files) == 0
        ), f"No .kpm manifest files should exist, found: {kpm_files}"

        # Verify device code was stripped from fat binary in generic artifact
        original_size = fat_binary_src.stat().st_size
        stripped_size = (generic_lib_dir / "libtest.so").stat().st_size
        assert (
            stripped_size < original_size
        ), "Stripped binary should be smaller than original"

        # Run the artifact verifier to check all invariants
        verifier = ArtifactVerifier(output_dir, toolchain, verbose=False)
        all_checks_passed = verifier.run_all_checks()
        assert all_checks_passed, "Artifact verification should pass all checks"

    def test_artifact_with_database_files(
        self, create_test_artifact, toolchain, tmp_path
    ):
        """Test splitting artifact with kernel database files."""
        # Create test artifact with database files
        input_dir = create_test_artifact(
            prefixes=["math-libs/BLAS/rocBLAS/stage"],
            files_per_prefix=2,
            include_db_files=True,
        )

        output_dir = tmp_path / "output"

        # Create splitter with rocBLAS handler
        rocblas_handler = RocBLASHandler()
        splitter = ArtifactSplitter(
            artifact_prefix="rocblas_lib",
            toolchain=toolchain,
            database_handlers=[rocblas_handler],
            verbose=True,
        )

        # Run the split - text files won't be detected as fat binaries
        splitter.split(input_dir, output_dir)

        # Verify generic artifact exists
        generic_dir = output_dir / "rocblas_lib_generic"
        assert generic_dir.exists()

        # Verify database files were moved to architecture-specific artifact
        arch_dir = output_dir / "rocblas_lib_gfx1100"
        assert arch_dir.exists()

        # Check database files in arch-specific artifact
        db_path = arch_dir / "math-libs/BLAS/rocBLAS/stage/lib/rocblas/library"
        assert (db_path / "TensileLibrary_gfx1100.dat").exists()
        assert (db_path / "TensileLibrary_gfx1100.co").exists()

        # Verify database files are NOT in generic artifact
        generic_db_path = (
            generic_dir / "math-libs/BLAS/rocBLAS/stage/lib/rocblas/library"
        )
        if generic_db_path.exists():
            # Directory might exist but should be empty or not have database files
            assert not (generic_db_path / "TensileLibrary_gfx1100.dat").exists()
            assert not (generic_db_path / "TensileLibrary_gfx1100.co").exists()

    def test_multiple_prefixes(self, create_test_artifact, toolchain, tmp_path):
        """Test splitting artifact with multiple prefixes."""
        # Create artifact with multiple prefixes
        prefixes = [
            "math-libs/BLAS/rocBLAS/stage",
            "math-libs/BLAS/hipBLASLt/stage",
            "kpack/stage",
        ]

        input_dir = create_test_artifact(prefixes=prefixes, files_per_prefix=2)

        output_dir = tmp_path / "output"

        # Create splitter with real toolchain
        splitter = ArtifactSplitter(
            artifact_prefix="multi_lib",
            toolchain=toolchain,
            database_handlers=[],
            verbose=True,
        )

        # Run the split - text files won't be detected as fat binaries
        splitter.split(input_dir, output_dir)

        # Verify generic artifact has all prefixes
        generic_dir = output_dir / "multi_lib_generic"
        assert generic_dir.exists()

        for prefix in prefixes:
            prefix_dir = generic_dir / prefix
            assert prefix_dir.exists(), f"Missing prefix: {prefix}"

    def test_file_classification_visitor(self, test_assets_dir, toolchain, tmp_path):
        """Test the FileClassificationVisitor directly with real files."""
        # Create test directory
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        lib_dir = test_dir / "lib"
        lib_dir.mkdir()

        # Copy real binaries from test assets
        fat_binary_src = (
            test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_multi.so"
        )
        fat_binary = lib_dir / "fat.so"
        shutil.copy2(fat_binary_src, fat_binary)

        host_only_src = test_assets_dir / "bundled_binaries/linux/cov5/libhost_only.so"
        regular_binary = lib_dir / "regular.so"
        shutil.copy2(host_only_src, regular_binary)

        # Create rocBLAS database files
        db_dir = lib_dir / "rocblas" / "library"
        db_dir.mkdir(parents=True)
        (db_dir / "TensileLibrary_gfx1100.dat").write_text("data")

        # Create visitor with handlers
        visitor = FileClassificationVisitor(
            toolchain=toolchain, database_handlers=[RocBLASHandler()], verbose=True
        )

        # Visit files - now using real ELF analysis
        visitor.visit_file(regular_binary, test_dir)
        visitor.visit_file(fat_binary, test_dir)
        visitor.visit_file(db_dir / "TensileLibrary_gfx1100.dat", test_dir)

        # Check classification results
        assert len(visitor.fat_binaries) == 1
        assert visitor.fat_binaries[0].name == "fat.so"

        assert "gfx1100" in visitor.database_files_by_arch
        assert len(visitor.database_files_by_arch["gfx1100"]) == 1

        assert len(visitor.exclude_from_generic) == 1  # Only database file

    def test_extracted_kernel_dataclass(self):
        """Test the ExtractedKernel dataclass."""
        kernel = ExtractedKernel(
            target_name="hipv4-amdgcn-amd-amdhsa--gfx906",
            kernel_data=b"kernel binary data",
            source_binary_relpath="lib/libtest.so",
            source_prefix="math-libs/BLAS/rocBLAS/stage",
            architecture="gfx906",
        )

        assert kernel.target_name == "hipv4-amdgcn-amd-amdhsa--gfx906"
        assert kernel.kernel_data == b"kernel binary data"
        assert kernel.source_binary_relpath == "lib/libtest.so"
        assert kernel.source_prefix == "math-libs/BLAS/rocBLAS/stage"
        assert kernel.architecture == "gfx906"

    def test_error_handling_missing_input(self, toolchain, tmp_path):
        """Test error handling for missing input directory."""
        splitter = ArtifactSplitter(
            artifact_prefix="test",
            toolchain=toolchain,
            database_handlers=[],
            verbose=False,
        )

        non_existent = tmp_path / "non_existent"
        output_dir = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            splitter.split(non_existent, output_dir)

    def test_error_handling_missing_manifest(self, toolchain, tmp_path):
        """Test error handling for missing artifact manifest."""
        # Create directory without manifest
        input_dir = tmp_path / "no_manifest"
        input_dir.mkdir()

        output_dir = tmp_path / "output"

        splitter = ArtifactSplitter(
            artifact_prefix="test",
            toolchain=toolchain,
            database_handlers=[],
            verbose=False,
        )

        with pytest.raises(FileNotFoundError, match="artifact_manifest.txt not found"):
            splitter.split(input_dir, output_dir)

    def test_batch_split_cli(self, create_test_artifact, toolchain, tmp_path):
        """Test batch mode split through the CLI interface."""
        # Create a parent directory with multiple arch-specific artifacts
        parent_dir = tmp_path / "shard"
        parent_dir.mkdir()

        # Create several arch-specific artifacts
        artifacts_to_create = [
            ("blas_lib_gfx1100", "blas_lib"),
            ("blas_dev_gfx1100", "blas_dev"),
            ("fft_lib_gfx1151", "fft_lib"),
            ("support_dev_generic", None),  # Should be skipped
        ]

        for artifact_name, _ in artifacts_to_create:
            artifact_dir = parent_dir / artifact_name
            artifact_dir.mkdir()
            write_artifact_manifest(artifact_dir, ["test/stage"])

            # Create some mock files
            lib_dir = artifact_dir / "test/stage/lib"
            lib_dir.mkdir(parents=True)
            (lib_dir / "libtest.so").write_text("mock library")

        # Test parse_artifact_name function
        assert parse_artifact_name("blas_lib_gfx1100") == "blas_lib"
        assert parse_artifact_name("support_dev_generic") is None
        assert parse_artifact_name("fft_lib_gfx1151") == "fft_lib"

        # Create args namespace for batch mode
        output_dir = tmp_path / "output"
        args = Namespace(
            input_dir=parent_dir,
            output_dir=output_dir,
            split_databases=None,
            verbose=False,
            tmp_dir=tmp_path / "tmp",
        )

        # Run batch split
        batch_split(args, toolchain)

        # Verify output artifacts were created
        # Since we didn't create any fat binaries or database files, only generic artifacts are created
        assert (output_dir / "blas_lib_generic").exists()
        assert (output_dir / "blas_dev_generic").exists()
        assert (output_dir / "fft_lib_generic").exists()

        # Arch-specific artifacts are only created if there's device code (fat binaries or databases)
        # In this test, we only have text files, so no arch-specific artifacts
        assert not (output_dir / "blas_lib_gfx1100").exists()
        assert not (output_dir / "blas_dev_gfx1100").exists()
        assert not (output_dir / "fft_lib_gfx1151").exists()

        # support_dev_generic should have been skipped (no artifacts created)
        # Since it ends in _generic, it won't create any output
        support_artifacts = list(output_dir.glob("support_dev_*"))
        assert len(support_artifacts) == 0, "support_dev_generic should be skipped"

    def test_batch_split_with_database_handlers(
        self, create_test_artifact, toolchain, tmp_path
    ):
        """Test batch mode with database handlers."""
        # Create parent directory with BLAS artifacts
        parent_dir = tmp_path / "shard"
        parent_dir.mkdir()

        # Create blas_lib artifact with database files
        blas_artifact = parent_dir / "blas_lib_gfx1100"
        blas_artifact.mkdir()
        write_artifact_manifest(blas_artifact, ["math-libs/BLAS/rocBLAS/stage"])

        # Create mock library and database files
        lib_dir = blas_artifact / "math-libs/BLAS/rocBLAS/stage/lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / "librocblas.so").write_text("mock library")

        db_dir = lib_dir / "rocblas" / "library"
        db_dir.mkdir(parents=True)
        (db_dir / "TensileLibrary_gfx1100.dat").write_text("mock database")
        (db_dir / "TensileLibrary_gfx1100.co").write_text("mock code object")

        # Create args with database handlers
        output_dir = tmp_path / "output"
        args = Namespace(
            input_dir=parent_dir,
            output_dir=output_dir,
            split_databases=["rocblas"],
            verbose=False,
            tmp_dir=tmp_path / "tmp",
        )

        # Run batch split
        batch_split(args, toolchain)

        # Verify artifacts were created
        assert (output_dir / "blas_lib_generic").exists()
        assert (output_dir / "blas_lib_gfx1100").exists()

        # Verify database files were moved to arch-specific artifact
        arch_db_path = (
            output_dir
            / "blas_lib_gfx1100/math-libs/BLAS/rocBLAS/stage/lib/rocblas/library"
        )
        assert (arch_db_path / "TensileLibrary_gfx1100.dat").exists()
        assert (arch_db_path / "TensileLibrary_gfx1100.co").exists()

    def test_kpack_uses_original_prefix_not_synthetic(
        self, test_assets_dir, toolchain, tmp_path
    ):
        """
        Test that kpack files are placed in original prefix directory, not synthetic kpack/stage.

        This is critical for bootstrap overlay: when generic and arch-specific artifacts
        are extracted to the same location, the .kpack/ directory must merge correctly.

        Before fix: rand_lib_gfx1201/kpack/stage/.kpack/rand_lib_gfx1201.kpack
        After fix:  rand_lib_gfx1201/math-libs/rocRAND/stage/.kpack/rand_lib_gfx1201.kpack
        """
        # Create test artifact structure with fat binary
        input_dir = tmp_path / "test_artifact"
        input_dir.mkdir()

        # Use a realistic prefix path
        prefix = "math-libs/rocRAND/stage"
        write_artifact_manifest(input_dir, [prefix])

        # Create prefix directory with fat binary
        prefix_dir = input_dir / prefix
        lib_dir = prefix_dir / "lib"
        lib_dir.mkdir(parents=True)

        # Copy real fat binary from test assets
        fat_binary_src = (
            test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_multi.so"
        )
        shutil.copy2(fat_binary_src, lib_dir / "librocrand.so")

        output_dir = tmp_path / "output"

        # Run split
        splitter = ArtifactSplitter(
            artifact_prefix="rand_lib",
            toolchain=toolchain,
            database_handlers=[],
            verbose=True,
        )
        splitter.split(input_dir, output_dir)

        # Find arch-specific artifacts
        arch_artifacts = list(output_dir.glob("rand_lib_gfx*"))
        assert (
            len(arch_artifacts) >= 1
        ), "Should have at least one arch-specific artifact"

        for arch_artifact in arch_artifacts:
            # CRITICAL: kpack should be in original prefix, NOT kpack/stage
            wrong_path = arch_artifact / "kpack/stage/.kpack"
            correct_path = arch_artifact / prefix / ".kpack"

            assert (
                not wrong_path.exists()
            ), f"kpack file should NOT be in synthetic kpack/stage/ path: {wrong_path}"
            assert (
                correct_path.exists()
            ), f"kpack file should be in original prefix path: {correct_path}"

            # Verify kpack file exists in correct location
            kpack_files = list(correct_path.glob("*.kpack"))
            assert (
                len(kpack_files) == 1
            ), f"Should have exactly one kpack file in {correct_path}"

    def test_generic_manifest_includes_all_prefixes(
        self, test_assets_dir, toolchain, tmp_path
    ):
        """
        Test that generic artifact manifest includes ALL prefixes, not just the last one.

        Before fix: manifest only contained last processed prefix (overwrite bug)
        After fix:  manifest contains all prefixes
        """
        # Create artifact with multiple prefixes
        input_dir = tmp_path / "test_artifact"
        input_dir.mkdir()

        prefixes = [
            "math-libs/rocRAND/stage",
            "math-libs/hipRAND/stage",
        ]
        write_artifact_manifest(input_dir, prefixes)

        # Create directories and files for each prefix
        for prefix in prefixes:
            prefix_dir = input_dir / prefix
            lib_dir = prefix_dir / "lib"
            lib_dir.mkdir(parents=True)
            (lib_dir / "libtest.so").write_text("mock library")

        output_dir = tmp_path / "output"

        # Run split
        splitter = ArtifactSplitter(
            artifact_prefix="rand_lib",
            toolchain=toolchain,
            database_handlers=[],
            verbose=True,
        )
        splitter.split(input_dir, output_dir)

        # Check generic artifact manifest
        generic_dir = output_dir / "rand_lib_generic"
        generic_manifest = read_artifact_manifest(generic_dir)

        # CRITICAL: All prefixes must be in the manifest
        assert len(generic_manifest) == len(
            prefixes
        ), f"Generic manifest should have {len(prefixes)} prefixes, got {len(generic_manifest)}"
        for prefix in prefixes:
            assert (
                prefix in generic_manifest
            ), f"Prefix '{prefix}' missing from generic manifest: {generic_manifest}"

    def test_symlinks_preserved_in_generic_artifact(self, toolchain, tmp_path):
        """
        Test that symlinks are preserved when copying to generic artifact.

        Before fix: Only regular files were copied, symlinks were lost
        After fix:  Symlinks are preserved with their original targets
        """
        # Create artifact with symlinks (simulating .so versioning)
        input_dir = tmp_path / "test_artifact"
        input_dir.mkdir()

        prefix = "math-libs/rocRAND/stage"
        write_artifact_manifest(input_dir, [prefix])

        # Create prefix directory with library and version symlinks
        prefix_dir = input_dir / prefix
        lib_dir = prefix_dir / "lib"
        lib_dir.mkdir(parents=True)

        # Create the actual library file
        real_lib = lib_dir / "librocrand.so.1.1"
        real_lib.write_text("mock library content")

        # Create symlinks (typical Linux .so versioning)
        import os

        os.symlink("librocrand.so.1.1", lib_dir / "librocrand.so.1")
        os.symlink("librocrand.so.1", lib_dir / "librocrand.so")

        output_dir = tmp_path / "output"

        # Run split
        splitter = ArtifactSplitter(
            artifact_prefix="rand_lib",
            toolchain=toolchain,
            database_handlers=[],
            verbose=True,
        )
        splitter.split(input_dir, output_dir)

        # Check generic artifact
        generic_lib_dir = output_dir / "rand_lib_generic" / prefix / "lib"

        # CRITICAL: Both symlinks and real file must exist
        assert (
            generic_lib_dir / "librocrand.so.1.1"
        ).exists(), "Real library file missing"
        assert (
            generic_lib_dir / "librocrand.so.1"
        ).is_symlink(), "Version symlink missing"
        assert (
            generic_lib_dir / "librocrand.so"
        ).is_symlink(), "SONAME symlink missing"

        # Verify symlink targets are correct
        assert os.readlink(generic_lib_dir / "librocrand.so.1") == "librocrand.so.1.1"
        assert os.readlink(generic_lib_dir / "librocrand.so") == "librocrand.so.1"

        # Verify the symlink chain works (can resolve to real file)
        assert (generic_lib_dir / "librocrand.so").resolve().name == "librocrand.so.1.1"

    def test_overlay_produces_merged_kpack_directory(
        self, test_assets_dir, toolchain, tmp_path
    ):
        """
        Test that extracting generic + arch artifacts to same location merges correctly.

        This simulates the bootstrap scenario where both artifacts are extracted
        to reconstitute a complete stage/ directory.
        """
        # Create test artifact with fat binary
        input_dir = tmp_path / "test_artifact"
        input_dir.mkdir()

        prefix = "math-libs/rocRAND/stage"
        write_artifact_manifest(input_dir, [prefix])

        prefix_dir = input_dir / prefix
        lib_dir = prefix_dir / "lib"
        lib_dir.mkdir(parents=True)

        # Copy real fat binary
        fat_binary_src = (
            test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_multi.so"
        )
        shutil.copy2(fat_binary_src, lib_dir / "librocrand.so")

        output_dir = tmp_path / "output"

        # Run split
        splitter = ArtifactSplitter(
            artifact_prefix="rand_lib",
            toolchain=toolchain,
            database_handlers=[],
            verbose=True,
        )
        splitter.split(input_dir, output_dir)

        # Simulate bootstrap overlay
        overlay_dir = tmp_path / "overlay"
        overlay_dir.mkdir()

        # Extract generic first
        generic_dir = output_dir / "rand_lib_generic"
        shutil.copytree(generic_dir, overlay_dir, dirs_exist_ok=True)

        # Extract arch-specific on top (should merge .kpack directory)
        arch_artifacts = list(output_dir.glob("rand_lib_gfx*"))
        for arch_artifact in arch_artifacts:
            shutil.copytree(arch_artifact, overlay_dir, dirs_exist_ok=True)

        # Verify .kpack directory has .kpack files (no .kpm manifests)
        kpack_dir = overlay_dir / prefix / ".kpack"
        assert kpack_dir.exists(), ".kpack directory should exist after overlay"

        kpack_files = list(kpack_dir.glob("*.kpack"))
        kpm_files = list(kpack_dir.glob("*.kpm"))

        assert (
            len(kpack_files) >= 1
        ), f"Should have at least one .kpack kernel file, got {kpack_files}"
        assert (
            len(kpm_files) == 0
        ), f"Should have no .kpm manifest files (patterns replace manifests), got {kpm_files}"
