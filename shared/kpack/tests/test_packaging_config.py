"""Tests for packaging configuration module."""

import json
from pathlib import Path

import pytest

from rocm_kpack.packaging_config import (
    ArchitectureGroup,
    PackagingConfig,
    ValidationRules,
)


class TestArchitectureGroup:
    """Tests for ArchitectureGroup dataclass."""

    def test_create_valid_group(self):
        """Test creating a valid architecture group."""
        group = ArchitectureGroup(
            display_name="ROCm gfx110X", architectures=["gfx1100", "gfx1101", "gfx1102"]
        )

        assert group.display_name == "ROCm gfx110X"
        assert group.architectures == ["gfx1100", "gfx1101", "gfx1102"]

    def test_empty_architectures_raises(self):
        """Test that empty architectures list raises error."""
        with pytest.raises(ValueError, match="must have at least one architecture"):
            ArchitectureGroup(display_name="Empty Group", architectures=[])

    def test_invalid_architecture_format_raises(self):
        """Test that invalid architecture format raises error."""
        with pytest.raises(ValueError, match="must start with 'gfx'"):
            ArchitectureGroup(display_name="Invalid", architectures=["invalid"])


class TestPackagingConfig:
    """Tests for PackagingConfig."""

    @pytest.fixture
    def sample_config_dict(self):
        """Sample configuration dictionary."""
        return {
            "primary_shard": "gfx110X_build",
            "architecture_groups": {
                "gfx110X": {
                    "display_name": "ROCm gfx110X",
                    "architectures": ["gfx1100", "gfx1101", "gfx1102"],
                },
                "gfx115X": {
                    "display_name": "ROCm gfx115X",
                    "architectures": ["gfx1150", "gfx1151"],
                },
            },
            "validation": {
                "error_on_duplicate_device_code": True,
                "verify_generic_artifacts_match": False,
                "error_on_missing_architecture": False,
            },
        }

    def test_from_json_valid(self, tmp_path, sample_config_dict):
        """Test loading valid JSON configuration."""
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(sample_config_dict, f)

        config = PackagingConfig.from_json(config_file)

        assert config.primary_shard == "gfx110X_build"
        assert len(config.architecture_groups) == 2
        assert "gfx110X" in config.architecture_groups
        assert "gfx115X" in config.architecture_groups

        gfx110x_group = config.architecture_groups["gfx110X"]
        assert gfx110x_group.display_name == "ROCm gfx110X"
        assert gfx110x_group.architectures == ["gfx1100", "gfx1101", "gfx1102"]

        assert config.validation.error_on_duplicate_device_code is True
        assert config.validation.verify_generic_artifacts_match is False

    def test_from_json_missing_file_raises(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        config_file = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            PackagingConfig.from_json(config_file)

    def test_from_json_invalid_json_raises(self, tmp_path):
        """Test that invalid JSON raises ValueError."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises(ValueError, match="Invalid JSON"):
            PackagingConfig.from_json(config_file)

    def test_from_json_missing_required_field_raises(self, tmp_path):
        """Test that missing required fields raise ValueError."""
        config_file = tmp_path / "config.json"

        # Missing architecture_groups
        config_file.write_text('{"primary_shard": "gfx110X_build"}')
        with pytest.raises(ValueError, match="architecture_groups"):
            PackagingConfig.from_json(config_file)

        # Missing primary_shard
        config_file.write_text('{"architecture_groups": {}}')
        with pytest.raises(ValueError, match="primary_shard"):
            PackagingConfig.from_json(config_file)

    def test_to_json_roundtrip(self, tmp_path, sample_config_dict):
        """Test writing and reading configuration."""
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(sample_config_dict, f)

        config1 = PackagingConfig.from_json(config_file)

        # Write to new file
        output_file = tmp_path / "output.json"
        config1.to_json(output_file)

        # Read back
        config2 = PackagingConfig.from_json(output_file)

        # Compare
        assert config2.primary_shard == config1.primary_shard
        assert config2.architecture_groups.keys() == config1.architecture_groups.keys()
        assert (
            config2.validation.error_on_duplicate_device_code
            == config1.validation.error_on_duplicate_device_code
        )
