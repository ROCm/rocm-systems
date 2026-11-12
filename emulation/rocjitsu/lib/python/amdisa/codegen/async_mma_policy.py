# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate numeric membership for the simulator's asynchronous MMA policy.

These sets belong to isa/arch/amdgpu/async_mma_policy.h. They preserve its
qualified inline instruction families; they do not describe general ISA
semantics or prove that an instruction is safe to execute concurrently.
The C++ policy still checks decoded modifiers, register destinations, and
control effects before admitting an instruction.
"""

from __future__ import annotations

from .cpp_file import CppFile

_MEMORY_PREFIXES = (
    'GLOBAL_LOAD_',
    'GLOBAL_STORE_',
    'BUFFER_LOAD_',
    'BUFFER_STORE_',
)

_ARITHMETIC_PREFIXES = (
    'S_MOV_',
    'S_ADD_',
    'S_ADDC_',
    'S_SUB_',
    'S_SUBB_',
    'S_MUL_',
    'S_MAD_',
    'S_AND_',
    'S_OR_',
    'S_XOR_',
    'S_LSHL_',
    'S_LSHR_',
    'S_ASHR_',
    'S_CMP_',
    'S_CSELECT_',
    'V_MOV_',
    'V_ADD_',
    'V_ADDC_',
    'V_SUB_',
    'V_SUBREV_',
    'V_MUL_',
    'V_MAD_',
    'V_FMA_',
    'V_LSHL',
    'V_LSHR',
    'V_ASHR',
    'V_AND_',
    'V_OR_',
    'V_XOR_',
    'V_CVT_',
    'V_CMP_',
    'V_CNDMASK_',
    'V_MAX_',
    'V_MIN_',
    'V_BFE_',
    'V_BFI_',
    'V_ALIGNBIT_',
    'V_ALIGNBYTE_',
)


def emit_async_inline_operations(output_root: str, operation_names: list[str]) -> None:
    """Emit an O(1) policy lookup in the same order as the Operation enum.

    Operation zero is Unknown and always rejects admission. Requiring the
    canonical sorted input and asserting the final enum index keeps the compact
    byte table coupled to the operation emitter without runtime name lookup.
    """
    if operation_names != sorted(set(operation_names)):
        raise ValueError('async inline policy requires sorted, unique operations')

    values = [0]
    for name in operation_names:
        values.append(
            1
            if name.startswith(_MEMORY_PREFIXES)
            else 2 if name.startswith(_ARITHMETIC_PREFIXES) else 0
        )
    rows = [
        '  ' + ', '.join(str(value) for value in values[start : start + 32]) + ','
        for start in range(0, len(values), 32)
    ]
    body = '\n'.join(
        [
            'namespace async_mma_policy {',
            '',
            '/// Membership in the qualified inline families; not an execution guarantee.',
            'enum class InlineOperationKind : uint8_t { None, Memory, Arithmetic };',
            '',
            f'inline constexpr std::array<uint8_t, {len(values)}> kInlineOperationKinds = {{',
            *rows,
            '};',
            '',
            (
                f'static_assert(static_cast<size_t>(Operation::{operation_names[-1]}) == '
                f'{len(operation_names)});'
                if operation_names
                else 'static_assert(static_cast<size_t>(Operation::Unknown) == 0);'
            ),
            '',
            '[[nodiscard]] constexpr InlineOperationKind inline_operation_kind(Operation operation) {',
            '  const size_t index = static_cast<size_t>(operation);',
            '  return index < kInlineOperationKinds.size()',
            '             ? static_cast<InlineOperationKind>(kInlineOperationKinds[index])',
            '             : InlineOperationKind::None;',
            '}',
            '',
            '} // namespace async_mma_policy',
        ]
    )
    CppFile(
        'async_inline_operations',
        output_root,
        True,
        [
            ('rocjitsu/isa/arch/amdgpu/generated/shared/operations.h', False),
            ('array', True),
            ('cstddef', True),
            ('cstdint', True),
        ],
        [],
        [body],
        'amdgpu',
        generated_dir_name='shared',
    ).gen_code()
