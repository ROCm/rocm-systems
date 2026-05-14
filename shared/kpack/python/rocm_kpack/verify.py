"""
Common verification utilities for binary formats.

This module provides the base VerificationResult class and BinaryVerifier
abstract base class that ELF and COFF verifiers extend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VerificationResult:
    """Result of binary verification.

    Used by both ELF and COFF verifiers to report errors and warnings
    in a consistent format.
    """

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Add an error (verification failed)."""
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        """Add a warning (verification passed but with concerns)."""
        self.warnings.append(msg)

    def merge(self, other: "VerificationResult") -> None:
        """Merge another result into this one."""
        if not other.passed:
            self.passed = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.passed:
            lines.append("Verification PASSED")
        else:
            lines.append("Verification FAILED")

        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  - {e}")

        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)


class BinaryVerifier(ABC):
    """Abstract base class for binary verifiers.

    ELF and COFF verifiers extend this to provide format-specific
    structural validation.
    """

    @classmethod
    @abstractmethod
    def verify(cls, path: Path) -> VerificationResult:
        """Verify a binary file on disk.

        Args:
            path: Path to binary

        Returns:
            VerificationResult with any errors/warnings
        """
        ...

    @classmethod
    @abstractmethod
    def verify_data(cls, data: bytes | bytearray) -> VerificationResult:
        """Verify binary data in memory.

        Args:
            data: Binary data

        Returns:
            VerificationResult with any errors/warnings
        """
        ...

    @abstractmethod
    def run_all_checks(self) -> VerificationResult:
        """Run all internal structural checks."""
        ...
