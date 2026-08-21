# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configuration for C++ code generation."""

from __future__ import annotations

from dataclasses import dataclass, field

AMDGPU_INCLUDE_BASE = 'rocjitsu/isa/arch/amdgpu'
AMDGPU_GENERATED_INCLUDE_BASE = f'{AMDGPU_INCLUDE_BASE}/generated'


@dataclass
class CodegenConfig:
    """Configuration for C++ code generation paths and namespaces.

    Attributes:
        namespace: Top-level C++ namespace enclosing all generated code.
        include_base: Base path prefix for handwritten AMDGPU includes.
        generated_include_base: Base path prefix for generated AMDGPU includes.
        unshared_execute_keys: Shared execute body keys that must stay
            ISA-local because different ISAs generated different bodies.
    """

    namespace: str = 'rocjitsu'
    include_base: str = AMDGPU_INCLUDE_BASE
    generated_include_base: str = AMDGPU_GENERATED_INCLUDE_BASE
    unshared_execute_keys: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def handwritten_include(self, *parts: str) -> str:
        """Return an include path rooted in handwritten AMDGPU sources."""
        return '/'.join((self.include_base, *parts))

    def generated_include(self, *parts: str) -> str:
        """Return an include path rooted in generated AMDGPU sources."""
        return '/'.join((self.generated_include_base, *parts))
