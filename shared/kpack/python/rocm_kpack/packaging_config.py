"""
Configuration for the artifact recombination phase.

This module defines the configuration schema for combining split artifacts
from the map phase into final package groups.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArchitectureGroup:
    """Defines a package group and its member architectures."""

    display_name: str
    architectures: list[str]

    def __post_init__(self):
        if not self.architectures:
            raise ValueError(
                f"Architecture group '{self.display_name}' must have at least one architecture"
            )

        # Validate architecture format (gfxXXXX)
        for arch in self.architectures:
            if not arch.startswith("gfx"):
                raise ValueError(
                    f"Invalid architecture '{arch}': must start with 'gfx'"
                )


@dataclass
class ValidationRules:
    """Validation rules for the recombination process."""

    error_on_duplicate_device_code: bool = True
    verify_generic_artifacts_match: bool = False
    error_on_missing_architecture: bool = False


@dataclass
class PackagingConfig:
    """
    Configuration for artifact recombination.

    This configuration determines how split artifacts from the map phase
    are combined into final package groups.
    """

    primary_shard: str
    architecture_groups: dict[str, ArchitectureGroup]
    validation: ValidationRules = field(default_factory=ValidationRules)

    def __post_init__(self):
        if not self.primary_shard:
            raise ValueError("primary_shard must be specified")

        if not self.architecture_groups:
            raise ValueError("At least one architecture group must be defined")

    @classmethod
    def from_json(cls, json_path: Path) -> "PackagingConfig":
        """
        Load configuration from a JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            PackagingConfig instance

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If JSON is invalid or configuration is malformed
        """
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        if json_path.stat().st_size == 0:
            raise ValueError(f"Configuration file is empty: {json_path}")

        try:
            with open(json_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_path}: {e}") from e
        except OSError as e:
            raise RuntimeError(
                f"Cannot read configuration file {json_path}: {e}"
            ) from e

        # Parse architecture groups
        groups = {}
        if "architecture_groups" not in data:
            raise ValueError("Missing required field: architecture_groups")

        for group_name, group_data in data["architecture_groups"].items():
            if not isinstance(group_data, dict):
                raise ValueError(f"Architecture group '{group_name}' must be an object")

            if "architectures" not in group_data:
                raise ValueError(
                    f"Architecture group '{group_name}' missing 'architectures' field"
                )

            display_name = group_data.get("display_name", group_name)
            architectures = group_data["architectures"]

            if not isinstance(architectures, list):
                raise ValueError(
                    f"Architecture group '{group_name}' architectures must be a list"
                )

            groups[group_name] = ArchitectureGroup(
                display_name=display_name, architectures=architectures
            )

        # Parse validation rules
        validation = ValidationRules()
        if "validation" in data:
            val_data = data["validation"]
            if not isinstance(val_data, dict):
                raise ValueError("'validation' field must be an object")

            validation = ValidationRules(
                error_on_duplicate_device_code=val_data.get(
                    "error_on_duplicate_device_code", True
                ),
                verify_generic_artifacts_match=val_data.get(
                    "verify_generic_artifacts_match", False
                ),
                error_on_missing_architecture=val_data.get(
                    "error_on_missing_architecture", False
                ),
            )

        # Get primary shard
        if "primary_shard" not in data:
            raise ValueError("Missing required field: primary_shard")

        primary_shard = data["primary_shard"]
        if not isinstance(primary_shard, str):
            raise ValueError("'primary_shard' must be a string")

        return cls(
            primary_shard=primary_shard,
            architecture_groups=groups,
            validation=validation,
        )

    def to_json(self, json_path: Path) -> None:
        """
        Write configuration to a JSON file.

        Args:
            json_path: Path where JSON file will be written
        """
        # Build architecture groups dict
        groups_dict = {}
        for group_name, group in self.architecture_groups.items():
            groups_dict[group_name] = {
                "display_name": group.display_name,
                "architectures": group.architectures,
            }

        data = {
            "primary_shard": self.primary_shard,
            "architecture_groups": groups_dict,
            "validation": {
                "error_on_duplicate_device_code": self.validation.error_on_duplicate_device_code,
                "verify_generic_artifacts_match": self.validation.verify_generic_artifacts_match,
                "error_on_missing_architecture": self.validation.error_on_missing_architecture,
            },
        }

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Validate output
        if not json_path.exists():
            raise RuntimeError(f"Failed to write configuration file: {json_path}")
        if json_path.stat().st_size == 0:
            raise RuntimeError(f"Written configuration file is empty: {json_path}")
