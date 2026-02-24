#!/usr/bin/env python3
"""
Bump version in VERSION file for ROCprofiler-SDK.

Supports semantic versioning (MAJOR.MINOR.PATCH format).
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple


class VersionBumper:
    """Handle version file reading, parsing, and bumping."""

    VERSION_PATTERN = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')

    def __init__(self, version_file: Path):
        """
        Initialize version bumper.

        Args:
            version_file: Path to VERSION file
        """
        self.version_file = version_file

    def read_version(self) -> Tuple[int, int, int]:
        """
        Read and parse current version from file.

        Returns:
            Tuple of (major, minor, patch)

        Raises:
            ValueError: If version format is invalid
            FileNotFoundError: If version file doesn't exist
        """
        if not self.version_file.exists():
            raise FileNotFoundError(f"Version file not found: {self.version_file}")

        version_text = self.version_file.read_text().strip()
        match = self.VERSION_PATTERN.match(version_text)

        if not match:
            raise ValueError(
                f"Invalid version format: {version_text}. "
                f"Expected MAJOR.MINOR.PATCH"
            )

        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3))

        return (major, minor, patch)

    def write_version(self, major: int, minor: int, patch: int):
        """
        Write version to file.

        Args:
            major: Major version number
            minor: Minor version number
            patch: Patch version number
        """
        version_str = f"{major}.{minor}.{patch}\n"
        self.version_file.write_text(version_str)

    def bump_major(self) -> Tuple[int, int, int]:
        """
        Bump major version, reset minor and patch to 0.

        Returns:
            New version tuple
        """
        major, minor, patch = self.read_version()
        return (major + 1, 0, 0)

    def bump_minor(self) -> Tuple[int, int, int]:
        """
        Bump minor version, reset patch to 0.

        Returns:
            New version tuple
        """
        major, minor, patch = self.read_version()
        return (major, minor + 1, 0)

    def bump_patch(self) -> Tuple[int, int, int]:
        """
        Bump patch version.

        Returns:
            New version tuple
        """
        major, minor, patch = self.read_version()
        return (major, minor, patch + 1)

    def bump(self, bump_type: str) -> Tuple[int, int, int]:
        """
        Bump version based on type.

        Args:
            bump_type: One of 'major', 'minor', 'patch'

        Returns:
            New version tuple

        Raises:
            ValueError: If bump_type is invalid
        """
        bump_type = bump_type.lower()

        if bump_type == 'major':
            new_version = self.bump_major()
        elif bump_type == 'minor':
            new_version = self.bump_minor()
        elif bump_type == 'patch':
            new_version = self.bump_patch()
        else:
            raise ValueError(
                f"Invalid bump type: {bump_type}. "
                f"Expected 'major', 'minor', or 'patch'"
            )

        # Write new version
        self.write_version(*new_version)

        return new_version

    def get_version_string(self, version: Tuple[int, int, int] = None) -> str:
        """
        Get version as string.

        Args:
            version: Version tuple, or None to read current version

        Returns:
            Version string in MAJOR.MINOR.PATCH format
        """
        if version is None:
            version = self.read_version()

        return f"{version[0]}.{version[1]}.{version[2]}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Bump version in VERSION file'
    )
    parser.add_argument(
        '--version-file',
        type=Path,
        required=True,
        help='Path to VERSION file'
    )
    parser.add_argument(
        '--bump-type',
        choices=['major', 'minor', 'patch'],
        required=True,
        help='Type of version bump'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    try:
        bumper = VersionBumper(args.version_file)

        # Get current version
        current_version = bumper.read_version()
        current_str = bumper.get_version_string(current_version)

        # Calculate new version
        if args.bump_type == 'major':
            new_version = bumper.bump_major()
        elif args.bump_type == 'minor':
            new_version = bumper.bump_minor()
        elif args.bump_type == 'patch':
            new_version = bumper.bump_patch()

        new_str = bumper.get_version_string(new_version)

        # Output results
        print(f"Current version: {current_str}")
        print(f"New version: {new_str}")
        print(f"Bump type: {args.bump_type}")

        # Perform bump (unless dry run)
        if args.dry_run:
            print("Dry run - no changes made")
        else:
            bumper.write_version(*new_version)
            print(f"Version file updated: {args.version_file}")

        # Set output for GitHub Actions
        # https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#setting-an-output-parameter
        print(f"::set-output name=old_version::{current_str}")
        print(f"::set-output name=new_version::{new_str}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
