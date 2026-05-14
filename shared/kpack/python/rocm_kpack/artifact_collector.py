"""
Artifact collection for the recombination phase.

This module discovers and organizes split artifacts from multiple map phase
shards, preparing them for recombination into package groups.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from rocm_kpack.artifact_utils import read_artifact_manifest


@dataclass
class CollectedArtifact:
    """Represents a collected artifact directory."""

    path: Path
    shard_name: str
    component_name: str
    architecture: str | None  # None for generic artifacts
    prefixes: list[str]

    @property
    def is_generic(self) -> bool:
        """Check if this is a generic artifact."""
        return self.architecture is None

    @property
    def is_architecture_specific(self) -> bool:
        """Check if this is an architecture-specific artifact."""
        return self.architecture is not None


class ArtifactNameInfo(NamedTuple):
    """Parsed artifact directory name."""

    component_name: str
    architecture: str | None  # None for generic artifacts


@dataclass
class AvailabilityResult:
    """Result of architecture availability check."""

    available: list[str]
    missing: list[str]


class ArtifactCollector:
    """
    Collects and organizes split artifacts from multiple map phase shards.

    This class discovers generic and architecture-specific artifacts across
    multiple build shards and prepares them for recombination.
    """

    def __init__(self, shards_dir: Path, primary_shard: str, verbose: bool = False):
        """
        Initialize artifact collector.

        Args:
            shards_dir: Directory containing shard subdirectories from map phase
            primary_shard: Name of shard to use for generic artifacts
            verbose: Enable verbose output
        """
        self.shards_dir = shards_dir
        self.primary_shard = primary_shard
        self.verbose = verbose

        # Shard -> component -> artifact
        self.shards: dict[str, dict[str, CollectedArtifact]] = {}

        # Component -> arch -> artifact (across all shards)
        self.arch_artifacts: dict[str, dict[str, CollectedArtifact]] = {}

        # Component -> generic artifact (from primary shard only)
        self.generic_artifacts: dict[str, CollectedArtifact] = {}

    def collect(self) -> None:
        """
        Discover and collect all artifacts from shards.

        Raises:
            FileNotFoundError: If shards directory doesn't exist
            ValueError: If artifact structure is invalid or primary shard missing
        """
        if not self.shards_dir.exists():
            raise FileNotFoundError(
                f"Shards directory does not exist: {self.shards_dir}"
            )

        if not self.shards_dir.is_dir():
            raise ValueError(f"Shards path is not a directory: {self.shards_dir}")

        if self.verbose:
            print(f"Collecting artifacts from shards in: {self.shards_dir}")
            print(f"Primary shard for generic artifacts: {self.primary_shard}")

        # Discover shards
        shard_dirs = [d for d in self.shards_dir.iterdir() if d.is_dir()]

        if not shard_dirs:
            raise ValueError(f"No shard directories found in {self.shards_dir}")

        # Check that primary shard exists
        primary_shard_dir = self.shards_dir / self.primary_shard
        if not primary_shard_dir.exists():
            raise ValueError(
                f"Primary shard '{self.primary_shard}' not found in {self.shards_dir}"
            )

        if self.verbose:
            print(f"\nFound {len(shard_dirs)} shards:")
            for shard_dir in sorted(shard_dirs):
                marker = " (primary)" if shard_dir.name == self.primary_shard else ""
                print(f"  - {shard_dir.name}{marker}")
            print()

        # Scan each shard for artifacts
        for shard_dir in sorted(shard_dirs):
            self._scan_shard(shard_dir)

        # Validate that primary shard has generic artifacts
        if not self.generic_artifacts:
            raise ValueError(
                f"No generic artifacts found in primary shard '{self.primary_shard}'"
            )

        if self.verbose:
            print(f"\nCollection summary:")
            print(
                f"  Generic artifacts (from {self.primary_shard}): {len(self.generic_artifacts)}"
            )
            total_arch_artifacts = sum(
                len(archs) for archs in self.arch_artifacts.values()
            )
            print(
                f"  Architecture-specific artifacts (all shards): {total_arch_artifacts}"
            )

    def _scan_shard(self, shard_dir: Path) -> None:
        """
        Scan a shard directory for artifacts.

        Args:
            shard_dir: Shard directory to scan
        """
        shard_name = shard_dir.name

        if self.verbose:
            print(f"Scanning shard: {shard_name}")

        if shard_name not in self.shards:
            self.shards[shard_name] = {}

        # Scan for artifact directories in this shard
        for artifact_dir in shard_dir.iterdir():
            if not artifact_dir.is_dir():
                continue

            # Check if it has artifact_manifest.txt
            manifest_file = artifact_dir / "artifact_manifest.txt"
            if not manifest_file.exists():
                if self.verbose:
                    print(f"  Skipping {artifact_dir.name}: no artifact_manifest.txt")
                continue

            # Parse artifact name to extract component and architecture
            artifact_info = self._parse_artifact_name(artifact_dir.name)
            if artifact_info is None:
                if self.verbose:
                    print(
                        f"  Skipping {artifact_dir.name}: invalid artifact name format"
                    )
                continue

            component_name = artifact_info.component_name
            architecture = artifact_info.architecture

            # Read artifact manifest
            try:
                prefixes = read_artifact_manifest(artifact_dir)
            except FileNotFoundError as e:
                raise ValueError(
                    f"Artifact manifest not found in {artifact_dir}"
                ) from e
            except OSError as e:
                raise ValueError(
                    f"Cannot read manifest from {artifact_dir}: {e}"
                ) from e

            # Create collected artifact
            artifact = CollectedArtifact(
                path=artifact_dir,
                shard_name=shard_name,
                component_name=component_name,
                architecture=architecture,
                prefixes=prefixes,
            )

            # Store in shard collection
            artifact_key = f"{component_name}_{architecture or 'generic'}"
            self.shards[shard_name][artifact_key] = artifact

            # Store appropriately based on type
            if artifact.is_generic:
                # Generic artifacts: only from primary shard
                if shard_name == self.primary_shard:
                    if component_name in self.generic_artifacts:
                        raise ValueError(
                            f"Duplicate generic artifact for component '{component_name}' "
                            f"in primary shard '{self.primary_shard}'"
                        )
                    self.generic_artifacts[component_name] = artifact
                    if self.verbose:
                        print(f"  Found generic artifact: {component_name}")
                else:
                    if self.verbose:
                        print(
                            f"  Skipping generic artifact {component_name} (not from primary shard)"
                        )
            else:
                # Architecture-specific artifacts: from any shard
                if component_name not in self.arch_artifacts:
                    self.arch_artifacts[component_name] = {}

                if architecture in self.arch_artifacts[component_name]:
                    # Multiple shards have same arch - use first one found, warn if different
                    existing = self.arch_artifacts[component_name][architecture]
                    if self.verbose:
                        print(
                            f"  Duplicate {architecture} artifact for '{component_name}': "
                            f"using {existing.shard_name}, ignoring {shard_name}"
                        )
                else:
                    self.arch_artifacts[component_name][architecture] = artifact
                    if self.verbose:
                        print(
                            f"  Found arch-specific artifact: {component_name} ({architecture})"
                        )

    def _parse_artifact_name(self, name: str) -> ArtifactNameInfo | None:
        """
        Parse artifact directory name to extract component and architecture.

        Expected formats:
        - Generic: {component}_generic
        - Architecture-specific: {component}_gfx{arch}

        Args:
            name: Artifact directory name

        Returns:
            ArtifactNameInfo with component_name and architecture (None for generic),
            or None if name doesn't match expected format
        """
        # Check for generic suffix
        if name.endswith("_generic"):
            component_name = name[: -len("_generic")]
            return ArtifactNameInfo(component_name=component_name, architecture=None)

        # Check for architecture suffix (gfxXXXX)
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            component_name, potential_arch = parts
            if potential_arch.startswith("gfx"):
                return ArtifactNameInfo(
                    component_name=component_name, architecture=potential_arch
                )

        return None

    def get_generic_artifact(self, component_name: str) -> CollectedArtifact | None:
        """
        Get generic artifact for a component (from primary shard).

        Args:
            component_name: Component name

        Returns:
            CollectedArtifact or None if not found
        """
        return self.generic_artifacts.get(component_name)

    def get_arch_artifact(
        self, component_name: str, architecture: str
    ) -> CollectedArtifact | None:
        """
        Get architecture-specific artifact for a component (from any shard).

        Args:
            component_name: Component name
            architecture: Architecture (e.g., "gfx1100")

        Returns:
            CollectedArtifact or None if not found
        """
        if component_name not in self.arch_artifacts:
            return None
        return self.arch_artifacts[component_name].get(architecture)

    def get_available_architectures(self, component_name: str) -> list[str]:
        """
        Get list of available architectures for a component (across all shards).

        Args:
            component_name: Component name

        Returns:
            List of architecture names (e.g., ["gfx1100", "gfx1101"])
        """
        if component_name not in self.arch_artifacts:
            return []
        return sorted(self.arch_artifacts[component_name].keys())

    def get_component_names(self) -> set[str]:
        """
        Get all component names found in artifacts.

        Returns:
            Set of component names
        """
        components = set(self.generic_artifacts.keys())
        components.update(self.arch_artifacts.keys())
        return components

    def validate_availability(
        self,
        component_name: str,
        required_architectures: list[str],
        require_generic: bool = True,
    ) -> AvailabilityResult:
        """
        Validate that required artifacts are available.

        Args:
            component_name: Component name
            required_architectures: List of required architectures
            require_generic: Whether generic artifact is required

        Returns:
            AvailabilityResult with available and missing architectures

        Raises:
            ValueError: If generic artifact is required but missing
        """
        # Check generic artifact
        if require_generic and component_name not in self.generic_artifacts:
            raise ValueError(
                f"Generic artifact not found for component '{component_name}' "
                f"in primary shard '{self.primary_shard}'"
            )

        # Check architecture-specific artifacts
        available = self.get_available_architectures(component_name)
        missing = [arch for arch in required_architectures if arch not in available]

        return AvailabilityResult(
            available=[arch for arch in required_architectures if arch in available],
            missing=missing,
        )
