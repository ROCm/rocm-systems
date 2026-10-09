# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configuration for C++ code generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

AMDGPU_INCLUDE_BASE = 'rocjitsu/isa/arch/amdgpu'
AMDGPU_GENERATED_INCLUDE_BASE = f'{AMDGPU_INCLUDE_BASE}/generated'


@dataclass
class CodegenConfig:
    """Configuration for C++ code generation paths and namespaces.

    Attributes:
        namespace: Top-level C++ namespace enclosing all generated code.
        include_base: Base path prefix for handwritten AMDGPU includes.
        generated_include_base: Base path prefix for generated AMDGPU includes.
        shared_generated_include_base: Base path prefix for shared generated
            AMDGPU includes.
        use_shared_execute_helpers: Whether generated execute bodies may call
            helpers emitted by multi-ISA generation.
        unshared_execute_keys: Shared execute body keys that must stay
            ISA-local because different ISAs generated different bodies.
    """

    namespace: str = 'rocjitsu'
    include_base: str = AMDGPU_INCLUDE_BASE
    generated_include_base: str = AMDGPU_GENERATED_INCLUDE_BASE
    shared_generated_include_base: str = AMDGPU_GENERATED_INCLUDE_BASE
    use_shared_execute_helpers: bool = True
    unshared_execute_keys: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    @classmethod
    def for_output(
        cls,
        isa_output: str,
        *,
        include_root: str | None = None,
        handwritten_include_base: str = AMDGPU_INCLUDE_BASE,
        use_shared_execute_helpers: bool = True,
    ) -> CodegenConfig:
        """Create path settings for an output tree.

        Without an include root, absolute includes make relative output
        independent of the generator's working directory. Supplying an include
        root instead emits stable paths relative to that compiler include root.
        Handwritten includes retain their source-tree prefix unless an
        independent prefix is supplied.
        """
        output = Path(isa_output).resolve()
        if include_root is None:
            generated_base = output.as_posix()
        else:
            root = Path(include_root).resolve()
            try:
                generated_base = output.relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f'ISA output {output} is not under include root {root}'
                ) from error

        return cls(
            include_base=handwritten_include_base,
            generated_include_base=generated_base,
            shared_generated_include_base=generated_base,
            use_shared_execute_helpers=use_shared_execute_helpers,
        )

    def handwritten_include(self, *parts: str) -> str:
        """Return an include path rooted in handwritten AMDGPU sources."""
        return '/'.join((self.include_base, *parts))

    def generated_include(self, *parts: str) -> str:
        """Return an include path rooted in generated AMDGPU sources."""
        return '/'.join((self.generated_include_base, *parts))

    def shared_generated_include(self, *parts: str) -> str:
        """Return an include path rooted in shared generated AMDGPU sources."""
        return '/'.join((self.shared_generated_include_base, 'shared', *parts))
