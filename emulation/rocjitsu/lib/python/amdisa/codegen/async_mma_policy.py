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

from .config import CodegenConfig
from .cpp_file import CppFile

_MEMORY_PREFIXES = (
    'GLOBAL_LOAD_',
    'GLOBAL_STORE_',
    'GLOBAL_ATOMIC_',
    'BUFFER_LOAD_',
    'BUFFER_STORE_',
    'BUFFER_ATOMIC_',
    'SCRATCH_LOAD_',
    'SCRATCH_STORE_',
    'FLAT_LOAD_',
    'FLAT_STORE_',
    'FLAT_ATOMIC_',
    'S_LOAD_',
    'S_STORE_',
    'S_BUFFER_LOAD_',
    'S_BUFFER_STORE_',
    'S_BUFFER_ATOMIC_',
)

# Scalar and vector ALU instructions qualify as arithmetic unless they carry
# wave state the policy cannot see in their register operands: control flow,
# waits, hardware and mode registers, EXEC, VGPR indexing or addressing modes,
# messages, caches, matrix instructions (admitted through candidate()), and
# interpolation (implicit LDS reads). The C++ policy still rejects explicit
# special-register destinations, DPP and secondary operations.
_ALU_PREFIXES = ('S_', 'V_')
_ALU_DENIED_PREFIXES = (
    'S_ADD_PC',
    'S_ALLOC_VGPR',
    'S_ATC_PROBE',
    'S_ATOMIC_',
    'S_BARRIER',
    'S_BRANCH',
    'S_BUFFER_PREFETCH',
    'S_CALL',
    'S_CBRANCH',
    'S_CODE_END',
    'S_DCACHE_',
    'S_DENORM_MODE',
    'S_ENDPGM',
    'S_GET_BARRIER_STATE',
    'S_GL1_INV',
    'S_ICACHE_INV',
    'S_ILLEGAL',
    'S_INST_PREFETCH',
    'S_MEMREALTIME',
    'S_MEMTIME',
    'S_MONITOR_SLEEP',
    'S_MOVREL',
    'S_PREFETCH_',
    'S_RFE',
    'S_ROUND_MODE',
    'S_SCRATCH_',
    'S_SENDMSG',
    'S_SETHALT',
    'S_SETKILL',
    'S_SETPC',
    'S_SETREG',
    'S_SETVSKIP',
    'S_SET_GPR_IDX',
    'S_SET_INST_PREFETCH',
    'S_SET_PC',
    'S_SET_VALU_COEXEC_MODE',
    'S_SET_VGPR_MSB',
    'S_SLEEP',
    'S_SWAPPC',
    'S_SWAP_PC',
    'S_TRAP',
    'S_TTRACEDATA',
    'S_VERSION',
    'S_WAIT',
    'S_WAKEUP',
    'V_CMPX',
    'V_ILLEGAL',
    'V_INTERP',
    'V_MFMA',
    'V_MOVREL',
    'V_SET_INACTIVE',
    'V_SMFMAC',
    'V_SWAPREL',
    'V_SWMMAC',
    'V_WMMA',
)
# EXEC writers whose names do not start with a denied prefix.
_ALU_DENIED_INFIXES = ('SAVEEXEC', 'WREXEC')


def _inline_operation_kind(name: str) -> int:
    if name.startswith(_MEMORY_PREFIXES):
        return 1
    if (
        name.startswith(_ALU_PREFIXES)
        and not name.startswith(_ALU_DENIED_PREFIXES)
        and not any(infix in name for infix in _ALU_DENIED_INFIXES)
    ):
        return 2
    return 0


def emit_async_inline_operations(
    output_root: str,
    operation_names: list[str],
    *,
    config: CodegenConfig | None = None,
) -> None:
    """Emit an O(1) policy lookup in the same order as the Operation enum.

    Operation zero is Unknown and always rejects admission. Requiring the
    canonical sorted input and asserting the final enum index keeps the compact
    byte table coupled to the operation emitter without runtime name lookup.
    """
    config = config or CodegenConfig()
    if operation_names != sorted(set(operation_names)):
        raise ValueError('async inline policy requires sorted, unique operations')

    values = [0]
    for name in operation_names:
        values.append(_inline_operation_kind(name))
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
            (config.operations_include(), False),
            ('array', True),
            ('cstddef', True),
            ('cstdint', True),
        ],
        [],
        [body],
        'amdgpu',
        generated_dir_name='shared',
    ).gen_code()
