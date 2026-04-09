# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD machine-readable ISA specification parser and C++ code generator."""

from amdisa.codegen import CodegenConfig, CodeGenerator, CppFile
from amdisa.gpuisa import (
    InstEncoding,
    Instruction,
    IsaSpec,
    Operand,
)
from amdisa.isa_profile import (
    Cdna1Profile,
    Cdna2Profile,
    CdnaProfile,
    EncodingModifier,
    IsaProfile,
    MemoryCoherencyModel,
    MnemonicRule,
    Rdna1Profile,
    Rdna2Profile,
    Rdna3Profile,
    Rdna3_5Profile,
    Rdna4Profile,
)
from amdisa.parser import Parser
from amdisa.xml_schema import SchemaValueError, SchemaVersion
from amdisa.semantics import (
    InstructionSemantics,
    SemanticsSpec,
    derive_all_semantics,
    derive_semantics,
)

__all__ = [
    'Cdna1Profile',
    'Cdna2Profile',
    'CdnaProfile',
    'CodegenConfig',
    'CodeGenerator',
    'CppFile',
    'EncodingModifier',
    'InstEncoding',
    'Instruction',
    'IsaProfile',
    'MemoryCoherencyModel',
    'MnemonicRule',
    'Rdna1Profile',
    'Rdna2Profile',
    'Rdna3Profile',
    'Rdna3_5Profile',
    'Rdna4Profile',
    'IsaSpec',
    'Operand',
    'Parser',
    'SchemaValueError',
    'SchemaVersion',
    'InstructionSemantics',
    'SemanticsSpec',
    'derive_all_semantics',
    'derive_semantics',
]
