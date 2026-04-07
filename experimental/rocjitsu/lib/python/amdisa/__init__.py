# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD machine-readable ISA specification parser and C++ code generator."""

from amdisa.codegen import CodeGenerator, CppFile
from amdisa.gpuisa import (
    DecodeTableEntry,
    InstBase,
    InstEncoding,
    Instruction,
    IsaSpec,
    MicrocodeField,
    Operand,
    OperandNamePattern,
    OperandSelector,
)
from amdisa.isa_profile import (
    CdnaProfile,
    EncodingModifier,
    IsaProfile,
    MnemonicRule,
    Rdna1Profile,
    Rdna3Profile,
    Rdna4Profile,
)
from amdisa.parser import Parser
from amdisa.xml_schema import SchemaValueError
from amdisa.semantics import (
    InstructionSemantics,
    SemanticsSpec,
    derive_all_semantics,
    derive_semantics,
)

__all__ = [
    'CdnaProfile',
    'CodeGenerator',
    'CppFile',
    'DecodeTableEntry',
    'EncodingModifier',
    'InstBase',
    'InstEncoding',
    'Instruction',
    'IsaProfile',
    'MnemonicRule',
    'Rdna1Profile',
    'Rdna3Profile',
    'Rdna4Profile',
    'IsaSpec',
    'MicrocodeField',
    'Operand',
    'OperandNamePattern',
    'OperandSelector',
    'Parser',
    'SchemaValueError',
    'InstructionSemantics',
    'SemanticsSpec',
    'derive_all_semantics',
    'derive_semantics',
]
