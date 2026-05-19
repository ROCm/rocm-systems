"""
Artifact combination for the recombination phase.

This module combines generic and architecture-specific artifacts from the
map phase into unified output artifacts organized by package group.
"""

import shutil
from pathlib import Path

from rocm_kpack.artifact_collector import ArtifactCollector, CollectedArtifact
from rocm_kpack.artifact_utils import write_artifact_manifest
from rocm_kpack.packaging_config import ArchitectureGroup


class ArtifactCombiner:
    """
    Combines split artifacts into package-group artifacts.

    This class takes generic and architecture-specific artifacts from the
    map phase and combines them according to package group configuration.
    """

    def __init__(
        self,
        collector: ArtifactCollector,
        verbose: bool = False,
    ):
        """
        Initialize artifact combiner.

        Args:
            collector: ArtifactCollector with discovered artifacts
            verbose: Enable verbose output
        """
        self.collector = collector
        self.verbose = verbose
        self._created_generic_artifacts: set[str] = set()

    def combine_component(
        self,
        component_name: str,
        group_name: str,
        arch_group: ArchitectureGroup,
        output_dir: Path,
    ) -> None:
        """
        Combine artifacts for a component and architecture group.

        This creates two separate artifacts:
        1. Generic artifact (once per component): Host code only, no .kpack files
        2. Arch-specific artifact (once per group): .kpack files and arch databases only

        Args:
            component_name: Component name (e.g., "rocblas_lib")
            group_name: Package group name (e.g., "gfx110X")
            arch_group: Architecture group configuration
            output_dir: Output directory for combined artifacts

        Raises:
            ValueError: If required artifacts are missing or invalid
        """
        if self.verbose:
            print(f"\nCombining component '{component_name}' for group '{group_name}'")

        # Get generic artifact
        generic_artifact = self.collector.get_generic_artifact(component_name)
        if generic_artifact is None:
            raise ValueError(
                f"Generic artifact not found for component '{component_name}'"
            )

        # Check which architectures are available
        availability = self.collector.validate_availability(
            component_name, arch_group.architectures, require_generic=True
        )

        # Determine if this is a generic-only component (no device code)
        has_device_code = bool(availability.available)

        if self.verbose:
            if availability.available:
                print(f"  Available architectures: {', '.join(availability.available)}")
            if availability.missing:
                print(f"  Missing architectures: {', '.join(availability.missing)}")
            if not has_device_code:
                print(f"  Component has no device code (generic-only)")

        # Create generic artifact (once per component)
        if component_name not in self._created_generic_artifacts:
            generic_output_dir = output_dir / f"{component_name}_generic"
            generic_output_dir.mkdir(parents=True, exist_ok=True)

            if self.verbose:
                print(f"  Creating generic artifact: {generic_output_dir}")

            self._create_generic_artifact(generic_artifact, generic_output_dir)
            self._created_generic_artifacts.add(component_name)

        # Skip arch-specific artifact if component has no device code
        if not has_device_code:
            if self.verbose:
                print(
                    f"  ✓ Generic-only artifact created (no arch-specific artifact needed)"
                )
            return

        # Create architecture-specific artifact for this group
        arch_output_dir = output_dir / f"{component_name}_{group_name}"
        arch_output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"  Creating arch-specific artifact: {arch_output_dir}")

        # Collect all prefixes from arch-specific artifacts
        arch_prefixes: set[str] = set()
        for arch in availability.available:
            arch_artifact = self.collector.get_arch_artifact(component_name, arch)
            if arch_artifact is None:
                # Should not happen since validate_availability returned it as available
                raise RuntimeError(f"Architecture artifact {arch} unexpectedly missing")

            arch_prefixes.update(arch_artifact.prefixes)
            self._copy_arch_content_only(arch_artifact, arch_output_dir)

        # Write artifact manifest for arch-specific artifact
        write_artifact_manifest(arch_output_dir, sorted(arch_prefixes))

        if self.verbose:
            print(f"  ✓ Artifacts created successfully")

    def _create_generic_artifact(
        self, generic_artifact: CollectedArtifact, output_dir: Path
    ) -> None:
        """
        Create generic artifact containing only host code (no .kpack files).

        This copies all files and directories from the generic artifact,
        excluding .kpack directories which belong in arch-specific artifacts.

        Args:
            generic_artifact: Generic artifact to copy
            output_dir: Destination directory
        """
        if self.verbose:
            print(f"    Copying generic host code from {generic_artifact.path}")

        def ignore_kpack_dirs(dir_path: str, names: list[str]) -> list[str]:
            """Ignore .kpack directories during copy."""
            return [name for name in names if name == ".kpack"]

        # Copy each prefix, excluding .kpack directories
        for prefix in generic_artifact.prefixes:
            src_prefix = generic_artifact.path / prefix
            dst_prefix = output_dir / prefix

            if not src_prefix.exists():
                if self.verbose:
                    print(f"      Skipping missing prefix: {prefix}")
                continue

            if self.verbose:
                print(f"      Copying prefix: {prefix}")

            # Copy the entire prefix directory tree, excluding .kpack
            if dst_prefix.exists():
                raise RuntimeError(
                    f"Destination prefix already exists: {dst_prefix}\n"
                    f"This indicates a duplicate copy or previous failed run"
                )

            shutil.copytree(
                src_prefix, dst_prefix, symlinks=True, ignore=ignore_kpack_dirs
            )

            # Validate copy succeeded
            if not dst_prefix.exists():
                raise RuntimeError(
                    f"Failed to copy generic artifact prefix: {src_prefix} -> {dst_prefix}"
                )

        # Write artifact manifest for generic artifact
        write_artifact_manifest(output_dir, generic_artifact.prefixes)

    def _copy_arch_content_only(
        self, arch_artifact: CollectedArtifact, output_dir: Path
    ) -> None:
        """
        Copy ONLY architecture-specific content (kpack files and arch databases).

        This does NOT copy any host code - that belongs in the generic artifact.

        Args:
            arch_artifact: Architecture-specific artifact
            output_dir: Destination directory
        """
        arch = arch_artifact.architecture
        if arch is None:
            raise ValueError("Architecture artifact has no architecture set")

        if self.verbose:
            print(f"      Copying {arch} kpack files and databases")

        # Copy each prefix's architecture-specific content
        for prefix in arch_artifact.prefixes:
            src_prefix = arch_artifact.path / prefix
            dst_prefix = output_dir / prefix

            if not src_prefix.exists():
                continue

            # Copy kpack directory (.kpack/*.kpack files)
            src_kpack_dir = src_prefix / ".kpack"
            if src_kpack_dir.exists():
                dst_kpack_dir = dst_prefix / ".kpack"
                dst_kpack_dir.mkdir(parents=True, exist_ok=True)

                # Copy .kpack files
                for kpack_file in src_kpack_dir.glob("*.kpack"):
                    dst_kpack_file = dst_kpack_dir / kpack_file.name
                    if self.verbose:
                        print(f"        {kpack_file.name}")
                    shutil.copy2(kpack_file, dst_kpack_file)

                    # Validate kpack file was copied successfully
                    if not dst_kpack_file.exists():
                        raise RuntimeError(f"Failed to copy kpack file: {kpack_file}")
                    if dst_kpack_file.stat().st_size == 0:
                        raise RuntimeError(
                            f"Kpack file is empty after copy: {dst_kpack_file}"
                        )
                    if dst_kpack_file.stat().st_size != kpack_file.stat().st_size:
                        raise RuntimeError(
                            f"Kpack file size mismatch after copy: "
                            f"{kpack_file.stat().st_size} -> {dst_kpack_file.stat().st_size}"
                        )

            # Copy kpack directory structure (for stage/.kpack/*.kpack layout)
            src_kpack_stage = src_prefix / "kpack" / "stage" / ".kpack"
            if src_kpack_stage.exists():
                dst_kpack_stage = dst_prefix / "kpack" / "stage" / ".kpack"
                dst_kpack_stage.mkdir(parents=True, exist_ok=True)

                for kpack_file in src_kpack_stage.glob("*.kpack"):
                    dst_kpack_file = dst_kpack_stage / kpack_file.name
                    if self.verbose:
                        print(f"        {kpack_file.name}")
                    shutil.copy2(kpack_file, dst_kpack_file)

                    # Validate kpack file was copied successfully
                    if not dst_kpack_file.exists():
                        raise RuntimeError(f"Failed to copy kpack file: {kpack_file}")
                    if dst_kpack_file.stat().st_size == 0:
                        raise RuntimeError(
                            f"Kpack file is empty after copy: {dst_kpack_file}"
                        )
                    if dst_kpack_file.stat().st_size != kpack_file.stat().st_size:
                        raise RuntimeError(
                            f"Kpack file size mismatch after copy: "
                            f"{kpack_file.stat().st_size} -> {dst_kpack_file.stat().st_size}"
                        )

            # Copy architecture-specific database files
            # These are already organized by architecture in the artifact
            # Just copy any files that aren't in the generic artifact
            self._copy_arch_specific_files(src_prefix, dst_prefix, arch)

    def _should_copy_arch_file(self, file_path: Path, arch: str) -> bool:
        """
        Check if file should be copied for this architecture.

        Args:
            file_path: Path to file to check
            arch: Architecture name

        Returns:
            True if file should be copied, False otherwise
        """
        # Skip .kpack directory (already handled separately)
        if ".kpack" in file_path.parts:
            return False

        # Skip kpack/stage/.kpack (already handled separately)
        if "kpack" in file_path.parts and "stage" in file_path.parts:
            return False

        # Copy files that contain the architecture in their name
        # (e.g., TensileLibrary_gfx1100.dat, Kernels_gfx1101.so)
        return arch in file_path.name

    def _copy_arch_specific_files(
        self, src_dir: Path, dst_dir: Path, arch: str
    ) -> None:
        """
        Copy architecture-specific files (e.g., database files).

        Args:
            src_dir: Source directory
            dst_dir: Destination directory
            arch: Architecture name
        """
        # Walk the source directory
        for src_file in src_dir.rglob("*"):
            if not src_file.is_file():
                continue

            if not self._should_copy_arch_file(src_file, arch):
                continue

            rel_path = src_file.relative_to(src_dir)
            dst_file = dst_dir / rel_path

            # Create parent directories
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            if self.verbose:
                print(f"      Copying {rel_path}")
            shutil.copy2(src_file, dst_file)

            # Validate file was copied successfully
            if not dst_file.exists():
                raise RuntimeError(f"Failed to copy arch-specific file: {src_file}")
            if dst_file.stat().st_size == 0:
                raise RuntimeError(
                    f"Arch-specific file is empty after copy: {dst_file}"
                )
            if dst_file.stat().st_size != src_file.stat().st_size:
                raise RuntimeError(
                    f"Arch-specific file size mismatch after copy: "
                    f"{src_file.stat().st_size} -> {dst_file.stat().st_size}"
                )
