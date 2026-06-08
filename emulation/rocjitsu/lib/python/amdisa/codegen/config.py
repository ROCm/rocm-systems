# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configuration for C++ code generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodegenConfig:
    """Configuration for C++ code generation paths and namespaces.

    Attributes:
        namespace: Top-level C++ namespace enclosing all generated code.
        include_base: Base path prefix for architecture-specific includes
            (e.g. ``'rocjitsu/isa/arch/amdgpu'``).
    """

    namespace: str = 'rocjitsu'
    include_base: str = 'rocjitsu/isa/arch/amdgpu'
