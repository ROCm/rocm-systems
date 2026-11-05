# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared rejection guard for execution paths requiring a VGPR operand."""


def append_required_vgpr_offset(
    lines: list[str], name: str, operand: str, *, uses_vgpr_msb_indexing: bool
) -> None:
    """Resolve a required VGPR, reporting failure before the caller uses it."""
    args = f'{operand}.opr_type_, {operand}.encoding_value_'
    if uses_vgpr_msb_indexing:
        args = f'wf, {args}, {operand}.vgpr_msb_role()'
    lines.append(f'  auto {name} = Isa::resolved_vgpr_offset({args});')
    lines.append(f'  if (!{name}) {{')
    lines.append('    wf.report_instruction_execution_error(')
    lines.append('        amdgpu::InstructionExecutionError::UnsupportedOperandValue);')
    lines.append('    return;')
    lines.append('  }')
