# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Core contracts for the rocprofsys-validator framework.

Defines the three load-bearing classes that all future phases import:
- CheckResult: immutable record of a single validation check outcome
- FormatReader: abstract base class for all format adapters
- AssertionBase: base for all Phase 3+ fluent assertion builders

These contracts are frozen after Plan 02 — changing field names or ABC signatures
breaks every downstream validator and reader.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, TypeVar

_T = TypeVar("_T", bound="FormatReader")

@dataclass
class CheckResult:
    """Outcome of a single validation check. Mutable to support soft-assertion accumulation.

    Fields are locked (D-05 in CONTEXT.md). Changing them breaks every validator.
    """

    passed: bool
    validator_name: str
    message: str
    expected: Any = None
    actual: Any = None
    details: dict[str, Any] = field(default_factory=dict)

    def assert_or_raise(self) -> None:
        """Raise AssertionError if this validation failed."""
        if not self.passed:
            raise AssertionError(self._format_message())

    def _format_message(self) -> str:
        """Build a human-readable failure message.

        First line: "[validator_name] FAILED: message"
        If expected is not None: "  expected: <repr>"
        If actual is not None:   "  actual:   <repr>"
        For each k, v in details: "  k: <repr>"
        Lines are joined with newline.
        """
        parts = [f"[{self.validator_name}] FAILED: {self.message}"]
        if self.expected is not None:
            parts.append(f"  expected: {self.expected!r}")
        if self.actual is not None:
            parts.append(f"  actual:   {self.actual!r}")
        for k, v in self.details.items():
            parts.append(f"  {k}: {v!r}")
        return "\n".join(parts)

class FormatReader(abc.ABC):
    """Abstract base class for all format adapters.

    Concrete subclasses implement validate() and are registered via @reader.
    Use as a context manager to ensure resource cleanup.

    Contract (D-07 in CONTEXT.md):
    - Single abstract method: validate() -> list[CheckResult]
    - No I/O in the ABC itself
    - Context manager protocol for resource cleanup
    """

    @abc.abstractmethod
    def validate(self) -> list[CheckResult]:
        """Run all validations and return results.

        Returns:
            list[CheckResult]: All validation results (pass or fail).
        """
        ...

    def close(self) -> None:
        """Release resources (file handles, subprocesses). Override in subclasses."""

    def __enter__(self: _T) -> _T:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

class AssertionBase:
    """Base for all Phase 3+ fluent assertion builders.

    Overrides __eq__ and __bool__ to prevent silent assertion discard.
    Phase 3 subclasses (ExpectBuilder hierarchy) override __bool__ to raise
    AssertionError (not TypeError) with formatted failure messages — this produces
    pytest FAILED output (not ERROR) when `assert expect(reader).has_track(...)` fails.

    __eq__ raises TypeError (accidental equality guard — do not override in subclasses).
    __bool__ raises TypeError here; subclasses raise AssertionError with failure details.
    Users: call .validate() on readers; use `assert expect(reader)...` for builders.
    """

    __hash__ = None  # Explicitly unhashable — assertion builders should not be used as dict keys

    def __eq__(self, other: object) -> bool:
        raise TypeError(
            "Use .validate() to evaluate assertions — "
            "don't compare or bool() the builder"
        )

    def __bool__(self) -> bool:
        raise TypeError(
            "Use .validate() to evaluate assertions — "
            "don't compare or bool() the builder"
        )

    def __ne__(self, other: object) -> bool:
        # Delegates to __eq__ which raises TypeError — intentional
        return not self.__eq__(other)  # type: ignore[return-value]
