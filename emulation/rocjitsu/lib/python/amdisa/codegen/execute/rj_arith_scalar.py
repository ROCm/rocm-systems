# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rj-arith scalar arithmetic bindings, preserving explicit SGPR and SCC writes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarShape:
    result_bits: int
    input_bits: tuple[int, ...]
    writes_scc: bool
    reads_scc: bool = False


# Widths are API widths, including mixed-width counts for 64-bit shifts.
SCALAR = {}
for bits in (32, 64):
    for op in ('AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'AND_NOT1', 'OR_NOT1'):
        SCALAR[f'S_{op}_B{bits}'] = ScalarShape(bits, (bits, bits), True)
    SCALAR[f'S_NOT_B{bits}'] = ScalarShape(bits, (bits,), True)
    SCALAR[f'S_BREV_B{bits}'] = ScalarShape(bits, (bits,), False)
    for op in ('LSHL_B', 'LSHR_B', 'ASHR_I'):
        SCALAR[f'S_{op}{bits}'] = ScalarShape(bits, (bits, 32), True)
    for ones in (0, 1):
        SCALAR[f'S_BCNT{ones}_I32_B{bits}'] = ScalarShape(32, (bits,), True)
    SCALAR[f'S_CTZ_I32_B{bits}'] = ScalarShape(32, (bits,), False)
    SCALAR[f'S_CLZ_I32_U{bits}'] = ScalarShape(32, (bits,), False)
    for op in ('QUADMASK', 'WQM'):
        SCALAR[f'S_{op}_B{bits}'] = ScalarShape(bits, (bits,), True)
    SCALAR[f'S_CSELECT_B{bits}'] = ScalarShape(bits, (bits, bits), False, True)
for name in ('S_CLS_I32', 'S_SEXT_I32_I8', 'S_SEXT_I32_I16'):
    SCALAR[name] = ScalarShape(32, (32,), False)
SCALAR['S_CLS_I32_I64'] = ScalarShape(32, (64,), False)
SCALAR['S_ABS_I32'] = ScalarShape(32, (32,), True)
for name in (
    'S_ADD_U32',
    'S_SUB_U32',
    'S_ADD_I32',
    'S_SUB_I32',
    'S_ADD_CO_U32',
    'S_SUB_CO_U32',
    'S_ADD_CO_I32',
    'S_SUB_CO_I32',
    'S_ABSDIFF_I32',
    'S_MIN_I32',
    'S_MIN_U32',
    'S_MAX_I32',
    'S_MAX_U32',
    'S_LSHL1_ADD_U32',
    'S_LSHL2_ADD_U32',
    'S_LSHL3_ADD_U32',
    'S_LSHL4_ADD_U32',
):
    SCALAR[name] = ScalarShape(32, (32, 32), True)
for name in ('S_MUL_I32', 'S_MUL_HI_U32', 'S_MUL_HI_I32'):
    SCALAR[name] = ScalarShape(32, (32, 32), False)
for name in ('S_ADD_NC_U64', 'S_SUB_NC_U64', 'S_MUL_U64'):
    SCALAR[name] = ScalarShape(64, (64, 64), False)
for name in ('S_ADDC_U32', 'S_SUBB_U32', 'S_ADD_CO_CI_U32', 'S_SUB_CO_CI_U32'):
    SCALAR[name] = ScalarShape(32, (32, 32), True, True)
for bits in (32, 64):
    for sign in ('I', 'U'):
        SCALAR[f'S_BFE_{sign}{bits}'] = ScalarShape(bits, (bits, 32), True)
    SCALAR[f'S_BFM_B{bits}'] = ScalarShape(bits, (32, 32), False)
for halves in ('LL', 'LH', 'HL', 'HH'):
    SCALAR[f'S_PACK_{halves}_B32_B16'] = ScalarShape(32, (32, 32), False)
SCALAR['S_BITREPLICATE_B64_B32'] = ScalarShape(64, (32,), False)

# Comparisons have only an SCC output. CMPK's encoded sdst is an input.
COMPARISONS = {}
for dtype in ('I32', 'U32'):
    for predicate in ('EQ', 'LG', 'GT', 'GE', 'LT', 'LE'):
        COMPARISONS[f'S_CMP_{predicate}_{dtype}'] = (32, 32)
        COMPARISONS[f'S_CMPK_{predicate}_{dtype}'] = (32, 16)
for predicate in ('EQ', 'LG'):
    COMPARISONS[f'S_CMP_{predicate}_U64'] = (64, 64)
for bits in (32, 64):
    for bit in (0, 1):
        COMPARISONS[f'S_BITCMP{bit}_B{bits}'] = (bits, 32)

# Immediate arithmetic reads and replaces sdst. Only addition writes SCC.
IMMEDIATE = {'S_ADDK_I32': True, 'S_ADDK_CO_I32': True, 'S_MULK_I32': False}
BITSET = {f'S_BITSET{value}_B{bits}': bits for value in (0, 1) for bits in (32, 64)}
FLOAT_COMPARISONS = {
    f'S_CMP_{predicate}_F{bits}': bits
    for bits in (16, 32)
    for predicate in (
        'LT',
        'EQ',
        'LE',
        'GT',
        'LG',
        'GE',
        'O',
        'U',
        'NGE',
        'NLG',
        'NGT',
        'NLE',
        'NEQ',
        'NLT',
    )
}
FLOAT_ROUNDING = {
    f'S_{op}_F{bits}': bits
    for bits in (16, 32)
    for op in ('CEIL', 'FLOOR', 'TRUNC', 'RNDNE')
}


@dataclass(frozen=True)
class ScalarFloatOperation:
    inputs: int
    input_fp_bits: int
    output_fp_bits: int
    accumulator: bool = False


# Zero precision means integer bits, so that side has no denormal policy.
FLOAT_ARITHMETIC = {
    f'S_{op}_F{bits}': ScalarFloatOperation(2, bits, bits)
    for bits in (16, 32)
    for op in ('ADD', 'SUB', 'MUL', 'MIN_NUM', 'MAX_NUM', 'MINIMUM', 'MAXIMUM')
}
FLOAT_ARITHMETIC.update(
    {
        'S_CVT_F32_I32': ScalarFloatOperation(1, 0, 32),
        'S_CVT_F32_U32': ScalarFloatOperation(1, 0, 32),
        'S_CVT_I32_F32': ScalarFloatOperation(1, 32, 0),
        'S_CVT_U32_F32': ScalarFloatOperation(1, 32, 0),
        'S_CVT_F16_F32': ScalarFloatOperation(1, 32, 16),
        'S_CVT_F32_F16': ScalarFloatOperation(1, 16, 32),
        'S_CVT_HI_F32_F16': ScalarFloatOperation(1, 16, 32),
        'S_CVT_PK_RTZ_F16_F32': ScalarFloatOperation(2, 32, 16),
        'S_FMAC_F32': ScalarFloatOperation(2, 32, 32, accumulator=True),
        'S_FMAC_F16': ScalarFloatOperation(2, 16, 16, accumulator=True),
        'S_FMAAK_F32': ScalarFloatOperation(3, 32, 32),
        'S_FMAMK_F32': ScalarFloatOperation(3, 32, 32),
    }
)
MOVES = {f'S_{op}_B{bits}' for op in ('MOV', 'CMOV') for bits in (32, 64)} | {
    'S_MOVK_I32',
    'S_CMOVK_I32',
}
EXEC_UPDATES = {
    f'S_{op}_{action}_B{bits}'
    for bits in (32, 64)
    for action, operations in (
        (
            'SAVEEXEC',
            (
                'AND',
                'OR',
                'XOR',
                'NAND',
                'NOR',
                'XNOR',
                'AND_NOT0',
                'OR_NOT0',
                'AND_NOT1',
                'OR_NOT1',
            ),
        ),
        ('WREXEC', ('AND_NOT0', 'AND_NOT1')),
    )
    for op in operations
}
RELATIVE_MOVES = {
    'S_MOVRELS_B32',
    'S_MOVRELS_B64',
    'S_MOVRELD_B32',
    'S_MOVRELD_B64',
    'S_MOVRELSD_2_B32',
}


MODE_UPDATES = {'S_ROUND_MODE', 'S_DENORM_MODE'}


SCALAR_NAMES = (
    SCALAR.keys()
    | MOVES
    | RELATIVE_MOVES
    | MODE_UPDATES
    | EXEC_UPDATES
    | COMPARISONS.keys()
    | IMMEDIATE.keys()
    | BITSET.keys()
    | FLOAT_COMPARISONS.keys()
    | FLOAT_ROUNDING.keys()
    | FLOAT_ARITHMETIC.keys()
)


def _call(mnemonic, arguments, flags='wf.cu().arithmetic_flags()'):
    return [
        f'  const int status = rj_arith_{mnemonic.lower()}(',
        f'      {flags}, 0, {arguments});',
        '  if (status != RJ_ARITH_SUCCESS) {',
        '    wf.report_instruction_execution_error(amdgpu::InstructionExecutionError::UnsupportedOperandValue);',
        '    return;',
        '  }',
    ]


def _gen_comparison(mnemonic, dst, src):
    if mnemonic.startswith('S_CMPK_'):
        if len(dst) != 1 or len(src) != 1:
            return None
        src = [dst[0], src[0]]
    elif dst or len(src) != 2:
        return None
    lines = ['  amdgpu::RegisterAccess registers(wf);']
    for i, (operand, bits) in enumerate(zip(src, COMPARISONS[mnemonic])):
        suffix = '64' if bits == 64 else ''
        lines.append(
            f'  const uint{bits}_t input{i} = uint{bits}_t(registers.read_scalar{suffix}({operand}));'
        )
    lines.append('  uint32_t scc = 0;')
    lines += _call(mnemonic, '&scc, input0, input1')
    lines.append('  wf.write_scc(scc);')
    return '\n'.join(lines)


def _gen_immediate(mnemonic, dst, src):
    if len(dst) != 1 or len(src) != 1:
        return None
    lines = [
        '  amdgpu::RegisterAccess registers(wf);',
        f'  uint32_t result = registers.read_scalar({dst[0]});',
        f'  const uint16_t immediate = uint16_t(registers.read_scalar({src[0]}));',
    ]
    if IMMEDIATE[mnemonic]:
        lines.append('  uint32_t scc = 0;')
    arguments = '&result, immediate' + (', &scc' if IMMEDIATE[mnemonic] else '')
    lines += _call(mnemonic, arguments)
    lines.append(f'  registers.write_scalar({dst[0]}, result);')
    if IMMEDIATE[mnemonic]:
        lines.append('  wf.write_scc(scc);')
    return '\n'.join(lines)


def _gen_bitset(mnemonic, dst, src):
    if len(dst) != 1 or len(src) != 1:
        return None
    bits = BITSET[mnemonic]
    suffix = '64' if bits == 64 else ''
    lines = [
        '  amdgpu::RegisterAccess registers(wf);',
        f'  uint{bits}_t result = registers.read_scalar{suffix}({dst[0]});',
        f'  const uint32_t bit = registers.read_scalar({src[0]});',
    ]
    lines += _call(mnemonic, '&result, bit')
    lines.append(f'  registers.write_scalar{suffix}({dst[0]}, result);')
    return '\n'.join(lines)


def _gen_float(mnemonic, dst, src):
    comparison = mnemonic in FLOAT_COMPARISONS
    if mnemonic in FLOAT_ARITHMETIC:
        operation = FLOAT_ARITHMETIC[mnemonic]
    else:
        bits = (FLOAT_COMPARISONS if comparison else FLOAT_ROUNDING)[mnemonic]
        operation = ScalarFloatOperation(2 if comparison else 1, bits, 0)
    if len(dst) != (0 if comparison else 1) or len(src) != operation.inputs:
        return None
    lines = [
        '  amdgpu::RegisterAccess registers(wf);',
        f'  uint64_t flags = wf.cu().arithmetic_flags();',
    ]
    for bits, mask, flag in (
        (operation.input_fp_bits, 1, 'INPUT'),
        (operation.output_fp_bits, 2, 'OUTPUT'),
    ):
        if bits:
            denorm = 'fp_denorm_mode_f32' if bits == 32 else 'fp_denorm_mode_f16_f64'
            lines += [
                f'  if (!(wf.{denorm}() & {mask}u))',
                f'    flags |= RJ_ARITH_FP_FLUSH_{flag}_DENORMALS;',
            ]
    # Scalar half arithmetic ignores FP16_OVFL; only narrowing CVT consumes it.
    if mnemonic == 'S_CVT_F16_F32':
        lines += ['  if (wf.fp16_ovfl())', '    flags |= RJ_ARITH_FP16_OVFL;']
    if mnemonic in ('S_FMAC_F32', 'S_FMAAK_F32', 'S_FMAMK_F32'):
        lines.append('  flags |= uint64_t(wf.fp_round_mode_f32()) << 24;')
    lines.append(
        '  const bool report_fp_status = wf.cu().arithmetic_reports_fp_status();'
    )
    for i, operand in enumerate(src):
        # Numeric FP16 inlines differ from raw scalar register/literal carriers.
        if operation.input_fp_bits == 16:
            high = ', true' if mnemonic == 'S_CVT_HI_F32_F16' else ''
            read = f'amdgpu::read_scalar_f16_carrier({operand}, wf{high})'
        else:
            read = f'registers.read_scalar({operand})'
        lines.append(f'  const uint32_t input{i} = {read};')
    initial = f'registers.read_scalar({dst[0]})' if operation.accumulator else '0'
    lines += [f'  uint32_t result = {initial};', '  uint32_t causes = 0;']
    arguments = '&result, ' + ', '.join(f'input{i}' for i in range(len(src)))
    lines += _call(mnemonic, arguments + ', &causes', 'flags')
    lines.append(
        '  wf.write_scc(result);'
        if comparison
        else f'  registers.write_scalar({dst[0]}, result);'
    )
    lines += [
        '  if (report_fp_status) {',
        '    wf.set_trapsts(wf.trapsts() | causes);',
        '    wf.raise_alu_causes(causes);',
        '  }',
    ]
    return '\n'.join(lines)


def _gen_relative_move(mnemonic, dst, src):
    if len(dst) != 1 or len(src) != 1:
        return None
    # rocjitsu owns selector/bank resolution (including special registers).
    # Once resolved, the operation is the same raw-bit copy as ordinary MOV.
    wide = mnemonic.endswith('B64')
    width = 2 if wide else 1
    bits = width * 32
    split = mnemonic == 'S_MOVRELSD_2_B32'
    relative_src = mnemonic.startswith('S_MOVRELS')
    relative_dst = mnemonic.startswith('S_MOVRELD') or split
    lines = []
    source, destination = src[0], dst[0]
    if relative_dst:
        lines += [
            f'  auto dst_reg = amdgpu::relative_scalar_selector(wf, {destination}.encoding_value(), wf.m0(), {width}, true, {str(split).lower()});',
            '  if (!dst_reg) return;',
            f'  Operand indexed_dst({bits}, OperandType::OPR_SDST, static_cast<int>(*dst_reg));',
        ]
        destination = 'indexed_dst'
    if relative_src:
        lines += [
            f'  auto src_reg = amdgpu::relative_scalar_selector(wf, {source}.encoding_value(), wf.m0(), {width}, false, {str(split).lower()});',
            f'  Operand indexed_src({bits}, OperandType::OPR_SSRC, static_cast<int>(src_reg.value_or(0)));',
        ]
        source = 'indexed_src'
    lines.append(_gen_move(f'S_MOV_B{bits}', [destination], [source]))
    return '\n'.join(lines)


def _gen_move(mnemonic, dst, src):
    if len(dst) != 1 or len(src) != 1:
        return None
    conditional = mnemonic.startswith('S_CMOV')
    immediate = 'MOVK_' in mnemonic
    bits = 64 if mnemonic.endswith('B64') else 32
    suffix = '64' if bits == 64 else ''
    lines = []
    if conditional:
        # Untaken CMOV observes neither source nor destination and preserves SCC.
        lines += [
            '  const uint32_t input_scc = wf.read_scc();',
            '  if (!input_scc) return;',
        ]
    lines += ['  amdgpu::RegisterAccess registers(wf);']
    value = (
        f'uint16_t({src[0]}.encoding_value())'
        if immediate
        else f'registers.read_scalar{suffix}({src[0]})'
    )
    lines += [
        f'  const uint{16 if immediate else bits}_t input = {value};',
        f'  uint{bits}_t result = 0;',
    ]
    lines += _call(mnemonic, '&result, input' + (', input_scc' if conditional else ''))
    lines += [f'  registers.write_scalar{suffix}({dst[0]}, result);']
    return '\n'.join(lines)


def _gen_exec_update(mnemonic, dst, src):
    if len(dst) != 1 or len(src) != 1:
        return None
    wide = mnemonic.endswith('B64')
    bits = 64 if wide else 32
    suffix = '64' if wide else ''
    read_exec = 'wf.exec_raw()' if wide else 'uint32_t(wf.exec())'
    write_exec = 'write_exec' if wide else 'set_exec'
    lines = [
        '  amdgpu::RegisterAccess registers(wf);',
        f'  const uint{bits}_t input = registers.read_scalar{suffix}({src[0]});',
        f'  const uint{bits}_t exec_mask = {read_exec};',
        f'  uint{bits}_t result = 0, new_exec = 0;',
        '  uint32_t scc = 0;',
    ]
    # EXEC and SCC are functional outputs, not optional exception history.
    lines += _call(mnemonic, '&result, input, exec_mask, &new_exec, &scc')
    lines += [
        f'  registers.write_scalar{suffix}({dst[0]}, result);',
        f'  wf.{write_exec}(new_exec);',
        '  wf.write_scc(scc);',
    ]
    return '\n'.join(lines)


def gen_scalar(arch, mnemonic, dst, src):
    if arch not in ('rdna3', 'rdna4'):
        return None
    if (
        mnemonic in FLOAT_COMPARISONS
        or mnemonic in FLOAT_ROUNDING
        or mnemonic in FLOAT_ARITHMETIC
    ):
        return _gen_float(mnemonic, dst, src) if arch == 'rdna4' else None
    if mnemonic in EXEC_UPDATES:
        return _gen_exec_update(mnemonic, dst, src)
    if mnemonic in MODE_UPDATES:
        if dst or len(src) != 1:
            return None
        # Guest MODE is functional state, independent of optional exception history.
        lines = ['  uint32_t mode = wf.mode_raw();']
        lines += _call(mnemonic, f'uint16_t({src[0]}.encoding_value()), &mode')
        lines += ['  wf.set_mode_raw(mode);']
        return '\n'.join(lines)
    if mnemonic in RELATIVE_MOVES:
        return _gen_relative_move(mnemonic, dst, src)
    if mnemonic in MOVES:
        return _gen_move(mnemonic, dst, src)
    if mnemonic in COMPARISONS:
        return _gen_comparison(mnemonic, dst, src)
    if mnemonic in IMMEDIATE:
        return _gen_immediate(mnemonic, dst, src)
    if mnemonic in BITSET:
        return _gen_bitset(mnemonic, dst, src)
    shape = SCALAR.get(mnemonic)
    if shape is None:
        return None
    if len(dst) != 1 or len(src) != len(shape.input_bits):
        return None
    lines = ['  amdgpu::RegisterAccess registers(wf);']
    if shape.reads_scc:
        lines.append('  const uint32_t input_scc = wf.read_scc();')
    for i, (operand, bits) in enumerate(zip(src, shape.input_bits)):
        suffix = '64' if bits == 64 else ''
        read = f'registers.read_scalar{suffix}({operand})'
        if mnemonic.startswith('S_CSELECT_'):
            # Preserve rocjitsu's observation of only the selected source.
            condition = 'input_scc' if i == 0 else '!input_scc'
            read = f'{condition} ? {read} : 0'
        lines.append(f'  const uint{bits}_t input{i} = {read};')
    lines.append(f'  uint{shape.result_bits}_t result = 0;')
    if shape.writes_scc:
        lines.append('  uint32_t scc = 0;')
    arguments = ', '.join(f'input{i}' for i in range(len(src)))
    if mnemonic in ('S_ADDC_U32', 'S_SUBB_U32') or mnemonic.startswith('S_CSELECT_'):
        arguments += ', input_scc'
    if shape.writes_scc:
        arguments += ', &scc'
    if mnemonic in ('S_ADD_CO_CI_U32', 'S_SUB_CO_CI_U32'):
        arguments += ', input_scc'
    lines += _call(mnemonic, '&result, ' + arguments)
    suffix = '64' if shape.result_bits == 64 else ''
    lines.append(f'  registers.write_scalar{suffix}({dst[0]}, result);')
    if shape.writes_scc:
        lines.append('  wf.write_scc(scc);')
    return '\n'.join(lines)
