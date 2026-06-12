# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CTS test data generator.

Usage:
    python -m amdisa.codegen.cts.cts_generator \\
        --multi cdna1:/path/to/cdna1.xml cdna4:/path/to/cdna4.xml ... \\
        --output-dir tests/cts_data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from amdisa import Parser, derive_all_semantics
from amdisa.codegen.cts.category_registry import cts_category
from amdisa.codegen.cts.encoding_builder import (
    CTS_DS_REGS,
    CTS_MFMA_REGS,
    CTS_SCALAR_REGS_BINARY,
    CTS_SCALAR_REGS_UNARY,
    CTS_SOPC_REGS,
    CTS_VOP1_REGS,
    CTS_VOP2_REGS,
    CTS_VOPC_REGS,
    CTS_VOP3_BINARY_REGS,
    CTS_VOP3_MAD64_REGS,
    CTS_VOP3P_REGS,
    CTS_VOP3_TERNARY_REGS,
    CTS_VOP3_UNARY_REGS,
    CTS_WMMA_REGS,
    build_encoding,
)
from amdisa.codegen.cts.golden_compute import (
    compute_bitop3_golden,
    compute_dot_product_golden,
    compute_ds_atomic_golden,
    compute_mad_64_32_golden,
    compute_packed_fp16_golden,
    compute_packed_int_golden,
    compute_scalar_alu_golden,
    compute_scalar_cmp_golden,
    compute_vector_alu_golden,
    compute_vector_cmp_golden,
    compute_vector_modifier_golden,
    compute_vector_ternary_golden,
)
from amdisa.codegen.cts.input_vectors import (
    dot_product_inputs,
    ds_addr_offset_pairs,
    ds_atomic_pairs,
    ds_test_data,
    inputs_for_dtype,
    packed_f16_inputs,
    packed_f16_ternary_inputs,
    packed_i16_inputs,
)
from amdisa.codegen.cts.modifier_enumerator import (
    binary_modifier_configs,
    unary_modifier_configs,
)


def _detect_profile(xml_path: str) -> 'IsaProfile':
    """Auto-detect ISA profile from XML filename."""
    import xml.etree.ElementTree as ET

    from amdisa.isa_profile import (
        Cdna1Profile,
        Cdna2Profile,
        CdnaProfile,
        Gfx1250Profile,
        Rdna1Profile,
        Rdna2Profile,
        Rdna3Profile,
        Rdna3_5Profile,
        Rdna4Profile,
    )

    stem = Path(xml_path).stem.lower()
    if 'gfx1250' in stem or 'mi450' in stem:
        return Gfx1250Profile()
    if 'cdna1' in stem:
        return Cdna1Profile()
    if 'cdna2' in stem:
        return Cdna2Profile()
    if 'cdna3' in stem:
        return CdnaProfile()
    if 'cdna4' in stem:
        return CdnaProfile()
    if 'rdna1' in stem:
        return Rdna1Profile()
    if 'rdna2' in stem:
        return Rdna2Profile()
    if 'rdna3_5' in stem or 'rdna35' in stem:
        return Rdna3_5Profile()
    if 'rdna3' in stem:
        return Rdna3Profile()
    if 'rdna4' in stem:
        return Rdna4Profile()

    root = ET.parse(xml_path).getroot()
    arch = root.get('Architecture', '').lower()
    if 'cdna' in arch:
        return CdnaProfile()
    if 'rdna4' in arch:
        return Rdna4Profile()
    return Rdna3Profile()


def _emit_test_array(
    struct_name: str, array_name: str, count_name: str, test_cases: list[str]
) -> list[str]:
    """Emit a constexpr test array (or nullptr + 0 if empty)."""
    if test_cases:
        return [
            f'inline constexpr {struct_name} {array_name}[] = {{',
            *test_cases,
            '};',
            '',
            f'inline constexpr size_t {count_name} = {len(test_cases)};',
        ]
    return [
        f'inline constexpr {struct_name} *{array_name} = nullptr;',
        '',
        f'inline constexpr size_t {count_name} = 0;',
    ]


_SKIP_64BIT_SUFFIXES = ('_b64', '_i64', '_u64')

_SKIP_SCC_INPUT_OPS = frozenset(
    {
        'addc',
        'subb',
        'add_co_ci',
        'sub_co_ci',
    }
)

_SKIP_RMW_OPS = frozenset({'bitset0', 'bitset1'})

_SKIP_CLASSES = frozenset({'scalar_cselect', 'scalar_cmov'})

_SKIP_FP_DTYPES = frozenset({'f16', 'f32', 'f64', 'bf16'})


_SKIP_FP_OPS_PREFIX = ('cvt_f', 'cvt_hi_f')


def _should_skip_scalar(mnemonic: str, sem) -> bool:
    """Return True if this scalar instruction can't be tested with our harness."""
    if any(mnemonic.endswith(s) for s in _SKIP_64BIT_SUFFIXES):
        return True
    if sem.semantic_class in _SKIP_CLASSES:
        return True
    if sem.operation in _SKIP_SCC_INPUT_OPS:
        return True
    if sem.operation in _SKIP_RMW_OPS:
        return True
    if sem.data_type in _SKIP_FP_DTYPES:
        return True
    if '_co_' in mnemonic:
        return True
    if sem.operation and any(sem.operation.startswith(p) for p in _SKIP_FP_OPS_PREFIX):
        return True
    return False


def _scalar_operand_count(semantic_class: str) -> int:
    if semantic_class in ('scalar_unary', 'scalar_mov', 'scalar_cmov'):
        return 1
    return 2


def _generate_scalar_alu(isa_name: str, isa_spec, semantics, output_dir: str) -> int:
    """Generate scalar_alu.h test data for one ISA.

    Returns the number of test cases generated.
    """
    test_cases: list[str] = []
    skipped = 0
    profile = isa_spec.profile

    test_child_encs: dict[str, list] = {}
    for enc in isa_spec.inst_encodings:
        if enc.insts and profile.is_alt_encoding(enc.enc_name):
            parent_name = profile.derive_parent_enc_name(enc.enc_name)
            test_child_encs.setdefault(parent_name, []).append(enc)

    for enc in isa_spec.inst_encodings:
        if profile.is_alt_encoding(enc.enc_name):
            continue

        all_insts = list(enc.insts)
        for child in test_child_encs.get(enc.enc_name, []):
            all_insts.extend(child.insts)

        for inst in all_insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            cat = cts_category(sem.semantic_class)
            if cat != 'scalar_alu':
                continue

            if _should_skip_scalar(inst.mnemonic, sem):
                skipped += 1
                continue

            n_src = _scalar_operand_count(sem.semantic_class)
            regs = CTS_SCALAR_REGS_UNARY if n_src == 1 else CTS_SCALAR_REGS_BINARY
            encoding = build_encoding(isa_spec, enc, inst, regs)
            if encoding is None:
                continue

            w0, w1 = encoding
            inputs = inputs_for_dtype(sem.data_type)

            for i in range(0, len(inputs), n_src):
                if i + n_src > len(inputs):
                    break
                src_vals = inputs[i : i + n_src]

                golden = compute_scalar_alu_golden(
                    sem.semantic_class,
                    sem.operation,
                    sem.data_type,
                    src_vals,
                    scc_in=False,
                )
                if golden is None:
                    continue

                scc_str = (
                    'true'
                    if golden.scc
                    else 'false' if golden.scc is not None else 'std::nullopt'
                )
                src_list = ', '.join(f'0x{v:08X}U' for v in src_vals)

                test_cases.append(
                    f'  {{"{inst.mnemonic}", '
                    f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                    f'{{{src_list}}}, '
                    f'0x{golden.output_bits:08X}U, '
                    f'{scc_str}, '
                    f'{n_src}}},'
                )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_SCALAR_ALU_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} scalar ALU.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <optional>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct ScalarAluTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 2> inputs;',
        '  uint32_t expected_output;',
        '  std::optional<bool> expected_scc;',
        '  uint32_t num_inputs;',
        '};',
        '',
        'inline constexpr ScalarAluTestCase SCALAR_ALU_TESTS[] = {',
    ]
    lines.extend(test_cases)
    lines.extend(
        [
            '};',
            '',
            f'inline constexpr size_t NUM_SCALAR_ALU_TESTS = {len(test_cases)};',
            '',
            f'}} // namespace rocjitsu::cts::{isa_name}',
            '',
            f'#endif // {guard}',
            '',
        ]
    )

    out_path = os.path.join(isa_dir, 'scalar_alu.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  Vector ALU integer filtering
# ---------------------------------------------------------------------------

_VECTOR_INT_DTYPES = frozenset({'b32', 'u32', 'i32', 'u24', 'i24'})

_SKIP_VECTOR_OPS = frozenset(
    {
        'fmac',
        'ldexp',
        'mul_legacy',
        'cvt',
        'cvt_f32_fp8',
        'cvt_f32_bf8',
        'cvt_f32_bf16',
        'cvt_norm_i16_f16',
        'cvt_norm_u16_f16',
        'cvt_off_f32_i4',
        'rcp',
        'rcp_iflag',
        'rsq',
        'sqrt',
        'sin',
        'cos',
        'tanh',
        'log2',
        'exp2',
        'ceil',
        'floor',
        'trunc',
        'fract',
        'rndne',
        'frexp_exp_f32',
        'frexp_exp_f16',
        'frexp_mant_f32',
    }
)


def _should_skip_vector(mnemonic: str, sem) -> bool:
    """Return True if this vector instruction can't be tested in Phase 2."""
    if sem.data_type and sem.data_type not in _VECTOR_INT_DTYPES:
        return True
    if sem.operation and sem.operation in _SKIP_VECTOR_OPS:
        return True
    if sem.semantic_class in ('vector_cndmask', 'vector_fmamk', 'vector_fmaak'):
        return True
    if '_b64' in mnemonic:
        return True
    if 'accvgpr' in mnemonic:
        return True
    return False


def _vector_operand_count(semantic_class: str) -> int:
    if semantic_class in ('vector_unary', 'vector_mov', 'vector_cvt'):
        return 1
    return 2


def _generate_vector_alu(
    isa_name: str, isa_spec, semantics, output_dir: str, category: str
) -> tuple[int, int]:
    """Generate vector_alu_unary.h or vector_alu_binary.h for one ISA."""
    target_classes = {
        'vector_alu_unary': ('vector_unary', 'vector_mov', 'vector_cvt'),
        'vector_alu_binary': ('vector_binop',),
    }[category]

    test_cases: list[str] = []
    skipped = 0
    profile = isa_spec.profile

    test_child_encs: dict[str, list] = {}
    for enc in isa_spec.inst_encodings:
        if enc.insts and profile.is_alt_encoding(enc.enc_name):
            parent_name = profile.derive_parent_enc_name(enc.enc_name)
            test_child_encs.setdefault(parent_name, []).append(enc)

    target_encs = frozenset({'ENC_VOP1', 'ENC_VOP2'})

    for enc in isa_spec.inst_encodings:
        if profile.is_alt_encoding(enc.enc_name):
            continue
        if enc.enc_name.upper() not in target_encs:
            continue

        all_insts = list(enc.insts)
        for child in test_child_encs.get(enc.enc_name, []):
            all_insts.extend(child.insts)

        for inst in all_insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            if sem.semantic_class not in target_classes:
                continue

            if _should_skip_vector(inst.mnemonic, sem):
                skipped += 1
                continue

            n_src = _vector_operand_count(sem.semantic_class)
            regs = CTS_VOP1_REGS if n_src == 1 else CTS_VOP2_REGS
            encoding = build_encoding(isa_spec, enc, inst, regs)
            if encoding is None:
                continue

            w0, w1 = encoding
            inputs = inputs_for_dtype(sem.data_type)

            for i in range(0, len(inputs), n_src):
                if i + n_src > len(inputs):
                    break
                src_vals = inputs[i : i + n_src]

                golden = compute_vector_alu_golden(
                    sem.semantic_class,
                    sem.operation,
                    sem.data_type,
                    src_vals,
                )
                if golden is None:
                    continue

                src_list = ', '.join(f'0x{v:08X}U' for v in src_vals)

                test_cases.append(
                    f'  {{"{inst.mnemonic}", '
                    f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                    f'{{{src_list}}}, '
                    f'0x{golden.output_bits:08X}U, '
                    f'{n_src}}},'
                )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_{category.upper()}_H_'
    struct_suffix = ''.join(w.capitalize() for w in category.split('_'))
    struct_name = f'{struct_suffix}TestCase'
    array_name = category.upper() + '_TESTS'
    count_name = 'NUM_' + array_name

    lines = [
        f'// Auto-generated CTS test data for {isa_name} {category}.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        f'struct {struct_name} {{',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 2> inputs;',
        '  uint32_t expected_output;',
        '  uint32_t num_inputs;',
        '};',
        '',
        f'inline constexpr {struct_name} {array_name}[] = {{',
    ]
    lines.extend(test_cases)
    lines.extend(
        [
            '};',
            '',
            f'inline constexpr size_t {count_name} = {len(test_cases)};',
            '',
            f'}} // namespace rocjitsu::cts::{isa_name}',
            '',
            f'#endif // {guard}',
            '',
        ]
    )

    out_path = os.path.join(isa_dir, f'{category}.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  Scalar comparison
# ---------------------------------------------------------------------------

_SKIP_CMP_DTYPES = frozenset({'f16', 'f64', 'bf16', 'i64', 'u64'})
_SKIP_CMP_CLASSES = frozenset({'scalar_cmpk'})


def _should_skip_cmp(mnemonic: str, sem) -> bool:
    if sem.semantic_class in _SKIP_CMP_CLASSES:
        return True
    if sem.data_type in _SKIP_CMP_DTYPES:
        return True
    if any(mnemonic.endswith(s) for s in _SKIP_64BIT_SUFFIXES):
        return True
    return False


def _generate_scalar_cmp(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate scalar_cmp.h test data for one ISA."""
    test_cases: list[str] = []
    skipped = 0
    profile = isa_spec.profile

    test_child_encs: dict[str, list] = {}
    for enc in isa_spec.inst_encodings:
        if enc.insts and profile.is_alt_encoding(enc.enc_name):
            parent_name = profile.derive_parent_enc_name(enc.enc_name)
            test_child_encs.setdefault(parent_name, []).append(enc)

    for enc in isa_spec.inst_encodings:
        if profile.is_alt_encoding(enc.enc_name):
            continue

        all_insts = list(enc.insts)
        for child in test_child_encs.get(enc.enc_name, []):
            all_insts.extend(child.insts)

        for inst in all_insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            cat = cts_category(sem.semantic_class)
            if cat != 'scalar_cmp':
                continue
            if _should_skip_cmp(inst.mnemonic, sem):
                skipped += 1
                continue

            regs = CTS_SOPC_REGS
            encoding = build_encoding(isa_spec, enc, inst, regs)
            if encoding is None:
                continue

            w0, w1 = encoding
            inputs = inputs_for_dtype(sem.data_type)

            for i in range(0, len(inputs), 2):
                if i + 2 > len(inputs):
                    break
                src_vals = inputs[i : i + 2]

                golden = compute_scalar_cmp_golden(
                    sem.semantic_class,
                    sem.operation,
                    sem.data_type,
                    src_vals,
                )
                if golden is None or golden.scc is None:
                    continue

                src_list = ', '.join(f'0x{v:08X}U' for v in src_vals)
                scc_str = 'true' if golden.scc else 'false'

                test_cases.append(
                    f'  {{"{inst.mnemonic}", '
                    f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                    f'{{{src_list}}}, '
                    f'{scc_str}}},'
                )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_SCALAR_CMP_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} scalar comparisons.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct ScalarCmpTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 2> inputs;',
        '  bool expected_scc;',
        '};',
        '',
        'inline constexpr ScalarCmpTestCase SCALAR_CMP_TESTS[] = {',
    ]
    lines.extend(test_cases)
    lines.extend(
        [
            '};',
            '',
            f'inline constexpr size_t NUM_SCALAR_CMP_TESTS = {len(test_cases)};',
            '',
            f'}} // namespace rocjitsu::cts::{isa_name}',
            '',
            f'#endif // {guard}',
            '',
        ]
    )

    out_path = os.path.join(isa_dir, 'scalar_cmp.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  Vector comparison (VOPC)
# ---------------------------------------------------------------------------

_SKIP_CMP_VECTOR_DTYPES = frozenset({'f16', 'f64', 'bf16', 'i64', 'u64', 'i16', 'u16'})


def _should_skip_vector_cmp(mnemonic: str, sem) -> bool:
    if sem.data_type in _SKIP_CMP_VECTOR_DTYPES:
        return True
    if any(mnemonic.endswith(s) for s in _SKIP_64BIT_SUFFIXES):
        return True
    if sem.semantic_class == 'vector_cmp_class':
        return True
    if sem.semantic_class == 'vector_cmpx':
        return True
    return False


def _generate_vector_cmp(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate vector_cmp.h test data for one ISA."""
    test_cases: list[str] = []
    skipped = 0
    profile = isa_spec.profile

    test_child_encs: dict[str, list] = {}
    for enc in isa_spec.inst_encodings:
        if enc.insts and profile.is_alt_encoding(enc.enc_name):
            parent_name = profile.derive_parent_enc_name(enc.enc_name)
            test_child_encs.setdefault(parent_name, []).append(enc)

    for enc in isa_spec.inst_encodings:
        if profile.is_alt_encoding(enc.enc_name):
            continue
        if enc.enc_name.upper() != 'ENC_VOPC':
            continue

        all_insts = list(enc.insts)
        for child in test_child_encs.get(enc.enc_name, []):
            all_insts.extend(child.insts)

        for inst in all_insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            cat = cts_category(sem.semantic_class)
            if cat != 'vector_cmp':
                continue
            if _should_skip_vector_cmp(inst.mnemonic, sem):
                skipped += 1
                continue

            encoding = build_encoding(isa_spec, enc, inst, CTS_VOPC_REGS)
            if encoding is None:
                continue

            w0, w1 = encoding
            inputs = inputs_for_dtype(sem.data_type)

            for i in range(0, len(inputs), 2):
                if i + 2 > len(inputs):
                    break
                src_vals = inputs[i : i + 2]

                golden = compute_vector_cmp_golden(
                    sem.semantic_class,
                    sem.operation,
                    sem.data_type,
                    src_vals,
                )
                if golden is None or golden.scc is None:
                    continue

                src_list = ', '.join(f'0x{v:08X}U' for v in src_vals)
                vcc_str = 'true' if golden.scc else 'false'

                test_cases.append(
                    f'  {{"{inst.mnemonic}", '
                    f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                    f'{{{src_list}}}, '
                    f'{vcc_str}}},'
                )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_VECTOR_CMP_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} vector comparisons.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct VectorCmpTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 2> inputs;',
        '  bool expected_vcc_bit;',
        '};',
        '',
        'inline constexpr VectorCmpTestCase VECTOR_CMP_TESTS[] = {',
    ]
    lines.extend(test_cases)
    lines.extend(
        [
            '};',
            '',
            f'inline constexpr size_t NUM_VECTOR_CMP_TESTS = {len(test_cases)};',
            '',
            f'}} // namespace rocjitsu::cts::{isa_name}',
            '',
            f'#endif // {guard}',
            '',
        ]
    )

    out_path = os.path.join(isa_dir, 'vector_cmp.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  Vector ternary (VOP3)
# ---------------------------------------------------------------------------

_TERNARY_INT_DTYPES = frozenset({'b32', 'u32', 'i32'})

_SKIP_TERNARY_OPS = frozenset(
    {
        'mad',
        'fma',
        'cubeid',
        'cubesc',
        'cubetc',
        'cubema',
        'minimum3',
        'maximum3',
        'minimummaximum',
        'maximumminimum',
        'maxmin_num',
        'minmax_num',
        'ashr_pk_i8_i32',
        'ashr_pk_u8_i32',
    }
)


def _should_skip_ternary(mnemonic: str, sem) -> bool:
    if sem.data_type and sem.data_type not in _TERNARY_INT_DTYPES:
        return True
    if sem.operation and sem.operation in _SKIP_TERNARY_OPS:
        return True
    if '_b64' in mnemonic or '_u64' in mnemonic or '_i64' in mnemonic:
        return True
    return False


def _generate_vector_ternary(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate vector_alu_ternary.h test data for one ISA."""
    test_cases: list[str] = []
    skipped = 0
    profile = isa_spec.profile

    test_child_encs: dict[str, list] = {}
    for enc in isa_spec.inst_encodings:
        if enc.insts and profile.is_alt_encoding(enc.enc_name):
            parent_name = profile.derive_parent_enc_name(enc.enc_name)
            test_child_encs.setdefault(parent_name, []).append(enc)

    target_encs = frozenset({'ENC_VOP3', 'ENC_VOP3_SDST_ENC'})

    for enc in isa_spec.inst_encodings:
        if profile.is_alt_encoding(enc.enc_name):
            continue
        if enc.enc_name.upper() not in target_encs:
            continue

        all_insts = list(enc.insts)
        for child in test_child_encs.get(enc.enc_name, []):
            all_insts.extend(child.insts)

        for inst in all_insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            if sem.semantic_class != 'vector_ternary':
                continue
            if _should_skip_ternary(inst.mnemonic, sem):
                skipped += 1
                continue

            encoding = build_encoding(isa_spec, enc, inst, CTS_VOP3_TERNARY_REGS)
            if encoding is None:
                continue

            w0, w1 = encoding
            inputs = inputs_for_dtype(sem.data_type)

            for i in range(0, len(inputs), 3):
                if i + 3 > len(inputs):
                    break
                src_vals = inputs[i : i + 3]

                golden = compute_vector_ternary_golden(
                    sem.semantic_class,
                    sem.operation,
                    sem.data_type,
                    src_vals,
                )
                if golden is None:
                    continue

                src_list = ', '.join(f'0x{v:08X}U' for v in src_vals)

                test_cases.append(
                    f'  {{"{inst.mnemonic}", '
                    f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                    f'{{{src_list}}}, '
                    f'0x{golden.output_bits:08X}U}},'
                )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_VECTOR_ALU_TERNARY_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} vector ternary ALU.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct VectorAluTernaryTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 3> inputs;',
        '  uint32_t expected_output;',
        '};',
        '',
        'inline constexpr VectorAluTernaryTestCase VECTOR_ALU_TERNARY_TESTS[] = {',
    ]
    lines.extend(test_cases)
    lines.extend(
        [
            '};',
            '',
            f'inline constexpr size_t NUM_VECTOR_ALU_TERNARY_TESTS = {len(test_cases)};',
            '',
            f'}} // namespace rocjitsu::cts::{isa_name}',
            '',
            f'#endif // {guard}',
            '',
        ]
    )

    out_path = os.path.join(isa_dir, 'vector_alu_ternary.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  VOP3 modifier tests (FP operations with neg/abs/omod/clamp)
# ---------------------------------------------------------------------------

_MODIFIER_FP_UNARY_OPS = frozenset(
    {
        'floor',
        'ceil',
        'trunc',
        'rndne',
        'fract',
        'rcp',
        'rsq',
        'sqrt',
    }
)

_SKIP_MODIFIER_MNEMONICS = frozenset(
    {
        'v_s_rcp_f32',
        'v_s_rsq_f32',
        'v_s_sqrt_f32',
    }
)

_MODIFIER_FP_BINARY_OPS = frozenset(
    {
        'add',
        'sub',
        'rsub',
        'mul',
        'max',
        'min',
    }
)

_MODIFIER_FP_INPUTS_PER_CONFIG = 8


def _generate_vector_modifier(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate vector_modifier.h — VOP3 FP instructions with modifiers."""
    test_cases: list[str] = []
    skipped = 0
    profile = isa_spec.profile

    for enc in isa_spec.inst_encodings:
        if enc.enc_name.upper() != 'ENC_VOP3':
            continue

        for inst in enc.insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            if sem.data_type != 'f32':
                continue

            if inst.mnemonic in _SKIP_MODIFIER_MNEMONICS:
                skipped += 1
                continue

            is_unary = (
                sem.semantic_class == 'vector_unary'
                and sem.operation in _MODIFIER_FP_UNARY_OPS
            )
            is_binary = (
                sem.semantic_class == 'vector_binop'
                and sem.operation in _MODIFIER_FP_BINARY_OPS
            )
            if not is_unary and not is_binary:
                skipped += 1
                continue

            n_src = 1 if is_unary else 2
            regs = CTS_VOP3_UNARY_REGS if is_unary else CTS_VOP3_BINARY_REGS
            configs = (
                unary_modifier_configs() if is_unary else binary_modifier_configs()
            )
            inputs = inputs_for_dtype('f32')

            for cfg in configs:
                mod_fields = {}
                if cfg.neg:
                    mod_fields['neg'] = cfg.neg
                if cfg.abs:
                    mod_fields['abs'] = cfg.abs
                if cfg.omod:
                    mod_fields['omod'] = cfg.omod
                if cfg.clamp:
                    mod_fields['clamp'] = cfg.clamp

                encoding = build_encoding(
                    isa_spec, enc, inst, regs, mod_fields if mod_fields else None
                )
                if encoding is None:
                    continue

                w0, w1 = encoding

                for i in range(
                    0, min(len(inputs), _MODIFIER_FP_INPUTS_PER_CONFIG * n_src), n_src
                ):
                    if i + n_src > len(inputs):
                        break
                    src_vals = inputs[i : i + n_src]

                    golden = compute_vector_modifier_golden(
                        sem.semantic_class,
                        sem.operation,
                        sem.data_type,
                        src_vals,
                        neg=cfg.neg,
                        abs_=cfg.abs,
                        omod=cfg.omod,
                        clamp=cfg.clamp,
                    )
                    if golden is None:
                        continue

                    src_list = ', '.join(f'0x{v:08X}U' for v in src_vals)
                    # Pad to 2 inputs for uniform struct
                    if n_src == 1:
                        src_list += ', 0x00000000U'

                    test_cases.append(
                        f'  {{"{inst.mnemonic}", '
                        f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                        f'{{{src_list}}}, '
                        f'0x{golden.output_bits:08X}U, '
                        f'{n_src}}},'
                    )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_VECTOR_MODIFIER_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} VOP3 modifier tests.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct VectorModifierTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 2> inputs;',
        '  uint32_t expected_output;',
        '  uint32_t num_inputs;',
        '};',
        '',
        'inline constexpr VectorModifierTestCase VECTOR_MODIFIER_TESTS[] = {',
    ]
    lines.extend(test_cases)
    lines.extend(
        [
            '};',
            '',
            f'inline constexpr size_t NUM_VECTOR_MODIFIER_TESTS = {len(test_cases)};',
            '',
            f'}} // namespace rocjitsu::cts::{isa_name}',
            '',
            f'#endif // {guard}',
            '',
        ]
    )

    out_path = os.path.join(isa_dir, 'vector_modifier.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  Bitop3 (truth-table-based 3-input bitwise operation)
# ---------------------------------------------------------------------------

_BITOP3_TRUTH_TABLES = [
    0x00,
    0xFF,
    0x80,
    0xFE,
    0x78,
    0x01,
    0x7F,
    0xF0,
    0xCC,
    0xAA,
    0x60,
    0xCA,
    0xE8,
    0x96,
    0x17,
    0x5A,
]

_BITOP3_INPUTS = [
    (0x00000000, 0x00000000, 0x00000000),
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
    (0x55555555, 0xAAAAAAAA, 0x0F0F0F0F),
    (0xDEADBEEF, 0xCAFEBABE, 0x12345678),
    (0x00000001, 0x00000001, 0x00000001),
    (0x80000000, 0x7FFFFFFF, 0xFFFF0000),
    (0x0000FFFF, 0xFFFF0000, 0x00FF00FF),
    (0xA5A5A5A5, 0x5A5A5A5A, 0xC3C3C3C3),
    (0x11111111, 0x22222222, 0x44444444),
    (0xF0F0F0F0, 0x0F0F0F0F, 0xFF00FF00),
]


def _generate_vector_bitop3(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate vector_bitop3.h — bitop3_b32 truth table tests."""
    test_cases: list[str] = []
    skipped = 0
    profile = isa_spec.profile

    for enc in isa_spec.inst_encodings:
        if enc.enc_name.upper() != 'ENC_VOP3':
            continue

        for inst in enc.insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            if sem.semantic_class != 'vector_bitop3':
                continue
            if sem.data_type != 'b32':
                skipped += 1
                continue

            for tt in _BITOP3_TRUTH_TABLES:
                mod_fields = {
                    'neg': tt & 7,
                    'abs': (tt >> 3) & 7,
                    'omod': (tt >> 6) & 3,
                }
                encoding = build_encoding(
                    isa_spec, enc, inst, CTS_VOP3_TERNARY_REGS, mod_fields
                )
                if encoding is None:
                    continue

                w0, w1 = encoding

                for src0, src1, src2 in _BITOP3_INPUTS:
                    golden = compute_bitop3_golden(tt, [src0, src1, src2])
                    if golden is None:
                        continue

                    test_cases.append(
                        f'  {{"{inst.mnemonic}", '
                        f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                        f'{{0x{src0:08X}U, 0x{src1:08X}U, 0x{src2:08X}U}}, '
                        f'0x{golden.output_bits:08X}U, 0x{tt:02X}U}},'
                    )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_VECTOR_BITOP3_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} bitop3.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct VectorBitop3TestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 3> inputs;',
        '  uint32_t expected_output;',
        '  uint32_t truth_table;',
        '};',
        '',
        *_emit_test_array(
            'VectorBitop3TestCase',
            'VECTOR_BITOP3_TESTS',
            'NUM_VECTOR_BITOP3_TESTS',
            test_cases,
        ),
        '',
        f'}} // namespace rocjitsu::cts::{isa_name}',
        '',
        f'#endif // {guard}',
        '',
    ]

    out_path = os.path.join(isa_dir, 'vector_bitop3.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  Integer dot products (VOP3P)
# ---------------------------------------------------------------------------


def _vop3p_default_modifiers(enc) -> dict[str, int]:
    """Return modifier dict to set op_sel_hi=3 for VOP3P (field name varies by ISA)."""
    field_names = {f.name for f in enc.ucode_fields}
    mods = {}
    if 'op_sel_hi' in field_names:
        mods['op_sel_hi'] = 3
    elif 'opsel_hi' in field_names:
        mods['opsel_hi'] = 3
    return mods


_DOT_PRODUCT_CLASSES = frozenset(
    {
        'dot4_i32_i8',
        'dot4_u32_u8',
        'dot8_i32_i4',
        'dot8_u32_u4',
        'dot2_i32_i16',
        'dot2_u32_u16',
    }
)


def _generate_dot_product(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate dot_product.h — integer dot product tests."""
    test_cases: list[str] = []
    skipped = 0

    inputs = dot_product_inputs()

    for enc in isa_spec.inst_encodings:
        if enc.enc_name.upper() != 'ENC_VOP3P':
            continue

        for inst in enc.insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            if sem.semantic_class not in _DOT_PRODUCT_CLASSES:
                continue

            mods = _vop3p_default_modifiers(enc)
            encoding = build_encoding(
                isa_spec, enc, inst, CTS_VOP3P_REGS, mods if mods else None
            )
            if encoding is None:
                continue

            w0, w1 = encoding

            for i in range(0, len(inputs), 3):
                if i + 3 > len(inputs):
                    break
                src_vals = inputs[i : i + 3]

                golden = compute_dot_product_golden(
                    sem.semantic_class,
                    src_vals,
                )
                if golden is None:
                    continue

                src_list = ', '.join(f'0x{v:08X}U' for v in src_vals)
                test_cases.append(
                    f'  {{"{inst.mnemonic}", '
                    f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                    f'{{{src_list}}}, '
                    f'0x{golden.output_bits:08X}U}},'
                )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_DOT_PRODUCT_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} dot products.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct DotProductTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 3> inputs;',
        '  uint32_t expected_output;',
        '};',
        '',
        *_emit_test_array(
            'DotProductTestCase',
            'DOT_PRODUCT_TESTS',
            'NUM_DOT_PRODUCT_TESTS',
            test_cases,
        ),
        '',
        f'}} // namespace rocjitsu::cts::{isa_name}',
        '',
        f'#endif // {guard}',
        '',
    ]

    out_path = os.path.join(isa_dir, 'dot_product.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  Packed 16-bit integer ops (VOP3P)
# ---------------------------------------------------------------------------

_PACKED_INT_OPS = frozenset(
    {
        'add',
        'sub',
        'mul',
        'min',
        'max',
        'shl',
        'shr',
        'ashr',
    }
)

_PACKED_INT_DTYPES = frozenset({'i16', 'u16'})


def _generate_packed_int(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate packed_int.h — packed 16-bit integer tests."""
    test_cases: list[str] = []
    skipped = 0

    inputs = packed_i16_inputs()

    for enc in isa_spec.inst_encodings:
        if enc.enc_name.upper() != 'ENC_VOP3P':
            continue

        for inst in enc.insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            if sem.semantic_class != 'pk_binop':
                continue
            if sem.data_type not in _PACKED_INT_DTYPES:
                skipped += 1
                continue
            if sem.operation not in _PACKED_INT_OPS:
                skipped += 1
                continue

            mods = _vop3p_default_modifiers(enc)
            encoding = build_encoding(
                isa_spec, enc, inst, CTS_VOP3P_REGS, mods if mods else None
            )
            if encoding is None:
                continue

            w0, w1 = encoding

            for i in range(0, len(inputs), 2):
                if i + 2 > len(inputs):
                    break
                src_vals = inputs[i : i + 2]

                golden = compute_packed_int_golden(
                    sem.operation,
                    sem.data_type,
                    src_vals,
                )
                if golden is None:
                    continue

                src_list = ', '.join(f'0x{v:08X}U' for v in src_vals)
                test_cases.append(
                    f'  {{"{inst.mnemonic}", '
                    f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                    f'{{{src_list}}}, '
                    f'0x{golden.output_bits:08X}U}},'
                )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_PACKED_INT_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} packed integer ops.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct PackedIntTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 2> inputs;',
        '  uint32_t expected_output;',
        '};',
        '',
        *_emit_test_array(
            'PackedIntTestCase', 'PACKED_INT_TESTS', 'NUM_PACKED_INT_TESTS', test_cases
        ),
        '',
        f'}} // namespace rocjitsu::cts::{isa_name}',
        '',
        f'#endif // {guard}',
        '',
    ]

    out_path = os.path.join(isa_dir, 'packed_int.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  Packed FP16 ops (VOP3P)
# ---------------------------------------------------------------------------

_PACKED_FP16_BIN_OPS = frozenset({'add', 'mul', 'min', 'max'})

_PACKED_FP16_TERNARY_OPS = frozenset({'fma', 'min3', 'max3'})

_PACKED_FP16_SKIP_TERNARY_OPS = frozenset({'fmac'})


def _vop3p_ternary_modifiers(enc) -> dict[str, int]:
    """Return modifier dict for VOP3P ternary: op_sel_hi=3 + op_sel_hi_2=1."""
    field_names = {f.name for f in enc.ucode_fields}
    mods = {}
    if 'op_sel_hi' in field_names:
        mods['op_sel_hi'] = 3
    elif 'opsel_hi' in field_names:
        mods['opsel_hi'] = 3
    if 'op_sel_hi_2' in field_names:
        mods['op_sel_hi_2'] = 1
    elif 'opsel_hi_2' in field_names:
        mods['opsel_hi_2'] = 1
    elif 'pad_14' in field_names:
        mods['pad_14'] = 1
    return mods


def _generate_packed_fp16(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate packed_fp16.h — packed FP16 binary and ternary tests."""
    bin_cases: list[str] = []
    fma_cases: list[str] = []
    skipped = 0

    bin_inputs = packed_f16_inputs()
    fma_inputs = packed_f16_ternary_inputs()

    for enc in isa_spec.inst_encodings:
        if enc.enc_name.upper() != 'ENC_VOP3P':
            continue

        for inst in enc.insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            if sem.data_type != 'f16':
                continue

            if (
                sem.semantic_class == 'pk_binop'
                and sem.operation in _PACKED_FP16_BIN_OPS
            ):
                mods = _vop3p_default_modifiers(enc)
                encoding = build_encoding(
                    isa_spec, enc, inst, CTS_VOP3P_REGS, mods if mods else None
                )
                if encoding is None:
                    continue

                w0, w1 = encoding

                for i in range(0, len(bin_inputs), 2):
                    if i + 2 > len(bin_inputs):
                        break
                    src_vals = bin_inputs[i : i + 2]
                    golden = compute_packed_fp16_golden(
                        sem.semantic_class,
                        sem.operation,
                        src_vals,
                    )
                    if golden is None:
                        continue

                    src_list = ', '.join(f'0x{v:08X}U' for v in src_vals)
                    bin_cases.append(
                        f'  {{"{inst.mnemonic}", '
                        f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                        f'{{{src_list}}}, '
                        f'0x{golden.output_bits:08X}U}},'
                    )

            elif (
                sem.semantic_class == 'pk_ternary'
                and sem.operation in _PACKED_FP16_TERNARY_OPS
            ):
                mods = _vop3p_ternary_modifiers(enc)
                encoding = build_encoding(
                    isa_spec, enc, inst, CTS_VOP3P_REGS, mods if mods else None
                )
                if encoding is None:
                    continue

                w0, w1 = encoding

                for i in range(0, len(fma_inputs), 3):
                    if i + 3 > len(fma_inputs):
                        break
                    src_vals = fma_inputs[i : i + 3]
                    golden = compute_packed_fp16_golden(
                        sem.semantic_class,
                        sem.operation,
                        src_vals,
                    )
                    if golden is None:
                        continue

                    src_list = ', '.join(f'0x{v:08X}U' for v in src_vals)
                    fma_cases.append(
                        f'  {{"{inst.mnemonic}", '
                        f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                        f'{{{src_list}}}, '
                        f'0x{golden.output_bits:08X}U}},'
                    )

            elif (
                sem.semantic_class == 'pk_ternary'
                and sem.operation in _PACKED_FP16_SKIP_TERNARY_OPS
            ):
                skipped += 1

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_PACKED_FP16_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} packed FP16 ops.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct PackedFp16BinTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 2> inputs;',
        '  uint32_t expected_output;',
        '};',
        '',
        *_emit_test_array(
            'PackedFp16BinTestCase',
            'PACKED_FP16_BIN_TESTS',
            'NUM_PACKED_FP16_BIN_TESTS',
            bin_cases,
        ),
        '',
        'struct PackedFp16FmaTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 3> inputs;',
        '  uint32_t expected_output;',
        '};',
        '',
        *_emit_test_array(
            'PackedFp16FmaTestCase',
            'PACKED_FP16_FMA_TESTS',
            'NUM_PACKED_FP16_FMA_TESTS',
            fma_cases,
        ),
        '',
        f'}} // namespace rocjitsu::cts::{isa_name}',
        '',
        f'#endif // {guard}',
        '',
    ]

    out_path = os.path.join(isa_dir, 'packed_fp16.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    total = len(bin_cases) + len(fma_cases)
    return total, skipped


# ---------------------------------------------------------------------------
#  64-bit multiply-accumulate (mad_u64_u32, mad_i64_i32)
# ---------------------------------------------------------------------------

_MAD64_INPUTS = [
    (0, 0, 0, 0),
    (1, 1, 0, 0),
    (0xFFFFFFFF, 0xFFFFFFFF, 0, 0),
    (0x7FFFFFFF, 2, 0, 0),
    (0x80000000, 2, 0, 0),
    (100, 200, 0x0000FFFF, 0),
    (0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0),
    (0, 0xFFFFFFFF, 1, 0),
    (2, 3, 0xFFFFFFFF, 0xFFFFFFFF),
    (0x10000, 0x10000, 0, 0),
]


def _generate_mad_64_32(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate mad_64.h — mad_u64_u32 and mad_i64_i32 tests."""
    test_cases: list[str] = []
    skipped = 0

    target_encs = frozenset({'ENC_VOP3', 'VOP3_SDST_ENC'})

    for enc in isa_spec.inst_encodings:
        if enc.enc_name.upper() not in target_encs:
            continue

        for inst in enc.insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            if sem.semantic_class != 'vector_mad_64_32':
                continue

            dtype = 'i64' if sem.data_type == 'i64' else 'u64'

            encoding = build_encoding(isa_spec, enc, inst, CTS_VOP3_MAD64_REGS)
            if encoding is None:
                continue

            w0, w1 = encoding

            for src0, src1, src2_lo, src2_hi in _MAD64_INPUTS:
                golden = compute_mad_64_32_golden(
                    dtype,
                    [src0, src1, src2_lo, src2_hi],
                )
                if golden is None:
                    continue

                result_lo, result_hi = golden

                test_cases.append(
                    f'  {{"{inst.mnemonic}", '
                    f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                    f'{{0x{src0:08X}U, 0x{src1:08X}U, 0x{src2_lo:08X}U, 0x{src2_hi:08X}U}}, '
                    f'0x{result_lo:08X}U, 0x{result_hi:08X}U}},'
                )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_MAD_64_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} mad_u64_u32/mad_i64_i32.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct Mad64TestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  std::array<uint32_t, 4> inputs;',
        '  uint32_t expected_lo;',
        '  uint32_t expected_hi;',
        '};',
        '',
        *_emit_test_array(
            'Mad64TestCase', 'MAD_64_TESTS', 'NUM_MAD_64_TESTS', test_cases
        ),
        '',
        f'}} // namespace rocjitsu::cts::{isa_name}',
        '',
        f'#endif // {guard}',
        '',
    ]

    out_path = os.path.join(isa_dir, 'mad_64.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  DS (LDS) read / write / read2 / write2
# ---------------------------------------------------------------------------

_DS_ENC_NAMES = frozenset({'ENC_DS', 'ENC_VDS'})

_DS_SKIP_CLASSES = frozenset(
    {
        'ds_read_addtid',
        'ds_write_addtid',
        'ds_read_tr_b4',
        'ds_read_tr_b6',
        'ds_read_tr_b8',
        'ds_read_tr_b16',
        'ds_append_consume',
        'ds_mskor',
        'ds_barrier_arrive',
        'ds_permute',
        'ds_swizzle',
    }
)

_DS_ATOMIC_TARGET_OPS = frozenset(
    {
        'add',
        'sub',
        'rsub',
        'swap',
        'smin',
        'smax',
        'umin',
        'umax',
        'and',
        'or',
        'xor',
        'inc',
        'dec',
    }
)


def _ds_offset_modifiers(enc, offset: int) -> dict[str, int]:
    """Build modifier dict to encode a 16-bit offset into offset0/offset1."""
    return {'offset0': offset & 0xFF, 'offset1': (offset >> 8) & 0xFF}


def _ds_dual_offset_modifiers(enc, off0: int, off1: int) -> dict[str, int]:
    """Build modifier dict for ds_read2/ds_write2 with separate offsets."""
    return {'offset0': off0 & 0xFF, 'offset1': off1 & 0xFF}


def _should_skip_ds(name: str, sem) -> bool:
    """Skip DS instructions we can't test yet."""
    upper = name.upper()
    if sem.semantic_class in _DS_SKIP_CLASSES:
        return True
    if 'ADDTID' in upper:
        return True
    if sem.num_elems and sem.num_elems > 2:
        return True
    if sem.d16_hi or sem.d16_lo:
        return True
    if sem.sign_extend:
        return True
    if 'ST64' in upper or 'STRIDE64' in upper:
        return True
    return False


def _generate_ds_read_write(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate ds.h — DS read/write/read2/write2 tests."""
    read_cases: list[str] = []
    write_cases: list[str] = []
    read2_cases: list[str] = []
    write2_cases: list[str] = []
    skipped = 0

    data_vals = ds_test_data()
    addr_pairs = ds_addr_offset_pairs()

    for enc in isa_spec.inst_encodings:
        if enc.enc_name.upper() not in _DS_ENC_NAMES:
            continue

        for inst in enc.insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            if sem.semantic_class not in (
                'ds_read',
                'ds_write',
                'ds_read2',
                'ds_write2',
            ):
                continue
            if _should_skip_ds(inst.name, sem):
                skipped += 1
                continue
            if sem.elem_size not in (4, 8):
                skipped += 1
                continue

            is_64 = sem.elem_size == 8

            if sem.semantic_class == 'ds_read':
                for addr_val, offset in addr_pairs:
                    mods = _ds_offset_modifiers(enc, offset)
                    encoding = build_encoding(isa_spec, enc, inst, CTS_DS_REGS, mods)
                    if encoding is None:
                        continue
                    w0, w1 = encoding
                    eff_addr = addr_val + offset

                    for dv in data_vals[:8]:
                        if is_64:
                            dv_hi = dv ^ 0x12345678
                            read_cases.append(
                                f'  {{"{inst.mnemonic}", '
                                f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                                f'0x{addr_val:08X}U, 0x{eff_addr:08X}U, '
                                f'0x{dv:08X}U, 0x{dv_hi:08X}U, '
                                f'0x{dv:08X}U, 0x{dv_hi:08X}U, true}},'
                            )
                        else:
                            read_cases.append(
                                f'  {{"{inst.mnemonic}", '
                                f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                                f'0x{addr_val:08X}U, 0x{eff_addr:08X}U, '
                                f'0x{dv:08X}U, 0x00000000U, '
                                f'0x{dv:08X}U, 0x00000000U, false}},'
                            )

            elif sem.semantic_class == 'ds_write':
                for addr_val, offset in addr_pairs:
                    mods = _ds_offset_modifiers(enc, offset)
                    encoding = build_encoding(isa_spec, enc, inst, CTS_DS_REGS, mods)
                    if encoding is None:
                        continue
                    w0, w1 = encoding
                    eff_addr = addr_val + offset

                    for dv in data_vals[:8]:
                        if is_64:
                            dv_hi = dv ^ 0x12345678
                            write_cases.append(
                                f'  {{"{inst.mnemonic}", '
                                f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                                f'0x{addr_val:08X}U, 0x{dv:08X}U, 0x{dv_hi:08X}U, '
                                f'0x{eff_addr:08X}U, '
                                f'0x{dv:08X}U, 0x{dv_hi:08X}U, true}},'
                            )
                        else:
                            write_cases.append(
                                f'  {{"{inst.mnemonic}", '
                                f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                                f'0x{addr_val:08X}U, 0x{dv:08X}U, 0x00000000U, '
                                f'0x{eff_addr:08X}U, '
                                f'0x{dv:08X}U, 0x00000000U, false}},'
                            )

            elif sem.semantic_class == 'ds_read2' and not is_64:
                off0, off1 = 0, 4
                for addr_val, _ in addr_pairs[:4]:
                    mods = _ds_dual_offset_modifiers(enc, off0, off1)
                    encoding = build_encoding(isa_spec, enc, inst, CTS_DS_REGS, mods)
                    if encoding is None:
                        continue
                    w0, w1 = encoding
                    eff0 = addr_val + off0 * 4
                    eff1 = addr_val + off1 * 4

                    for i in range(0, min(len(data_vals), 8), 2):
                        d0, d1 = data_vals[i], data_vals[i + 1]
                        read2_cases.append(
                            f'  {{"{inst.mnemonic}", '
                            f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                            f'0x{addr_val:08X}U, '
                            f'{off0}U, {off1}U, '
                            f'0x{eff0:08X}U, 0x{eff1:08X}U, '
                            f'0x{d0:08X}U, 0x{d1:08X}U, '
                            f'0x{d0:08X}U, 0x{d1:08X}U}},'
                        )

            elif sem.semantic_class == 'ds_write2' and not is_64:
                off0, off1 = 0, 4
                for addr_val, _ in addr_pairs[:4]:
                    mods = _ds_dual_offset_modifiers(enc, off0, off1)
                    encoding = build_encoding(isa_spec, enc, inst, CTS_DS_REGS, mods)
                    if encoding is None:
                        continue
                    w0, w1 = encoding
                    eff0 = addr_val + off0 * 4
                    eff1 = addr_val + off1 * 4

                    for i in range(0, min(len(data_vals), 8), 2):
                        d0, d1 = data_vals[i], data_vals[i + 1]
                        write2_cases.append(
                            f'  {{"{inst.mnemonic}", '
                            f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                            f'0x{addr_val:08X}U, '
                            f'0x{d0:08X}U, 0x{d1:08X}U, '
                            f'0x{eff0:08X}U, 0x{eff1:08X}U, '
                            f'0x{d0:08X}U, 0x{d1:08X}U}},'
                        )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_DS_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} DS read/write.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct DsReadTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  uint32_t addr_value;',
        '  uint32_t lds_addr;',
        '  uint32_t lds_data_lo;',
        '  uint32_t lds_data_hi;',
        '  uint32_t expected_lo;',
        '  uint32_t expected_hi;',
        '  bool is_64bit;',
        '};',
        '',
        *_emit_test_array(
            'DsReadTestCase', 'DS_READ_TESTS', 'NUM_DS_READ_TESTS', read_cases
        ),
        '',
        'struct DsWriteTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  uint32_t addr_value;',
        '  uint32_t data_lo;',
        '  uint32_t data_hi;',
        '  uint32_t expected_lds_addr;',
        '  uint32_t expected_lo;',
        '  uint32_t expected_hi;',
        '  bool is_64bit;',
        '};',
        '',
        *_emit_test_array(
            'DsWriteTestCase', 'DS_WRITE_TESTS', 'NUM_DS_WRITE_TESTS', write_cases
        ),
        '',
        'struct DsRead2TestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  uint32_t addr_value;',
        '  uint32_t offset0;',
        '  uint32_t offset1;',
        '  uint32_t lds_addr0;',
        '  uint32_t lds_addr1;',
        '  uint32_t lds_data0;',
        '  uint32_t lds_data1;',
        '  uint32_t expected_vdst0;',
        '  uint32_t expected_vdst1;',
        '};',
        '',
        *_emit_test_array(
            'DsRead2TestCase', 'DS_READ2_TESTS', 'NUM_DS_READ2_TESTS', read2_cases
        ),
        '',
        'struct DsWrite2TestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  uint32_t addr_value;',
        '  uint32_t data0;',
        '  uint32_t data1;',
        '  uint32_t expected_lds_addr0;',
        '  uint32_t expected_lds_addr1;',
        '  uint32_t expected_lds0;',
        '  uint32_t expected_lds1;',
        '};',
        '',
        *_emit_test_array(
            'DsWrite2TestCase', 'DS_WRITE2_TESTS', 'NUM_DS_WRITE2_TESTS', write2_cases
        ),
        '',
        f'}} // namespace rocjitsu::cts::{isa_name}',
        '',
        f'#endif // {guard}',
        '',
    ]

    out_path = os.path.join(isa_dir, 'ds.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    total = len(read_cases) + len(write_cases) + len(read2_cases) + len(write2_cases)
    return total, skipped


# ---------------------------------------------------------------------------
#  DS atomic RTN
# ---------------------------------------------------------------------------


def _generate_ds_atomic_rtn(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate ds_atomic.h — DS atomic RTN tests."""
    test_cases: list[str] = []
    skipped = 0

    pairs = ds_atomic_pairs()
    addr_val = 256
    offset = 0

    for enc in isa_spec.inst_encodings:
        if enc.enc_name.upper() not in _DS_ENC_NAMES:
            continue

        for inst in enc.insts:
            if inst.name not in semantics:
                continue
            sem = semantics[inst.name]
            if sem.semantic_class != 'ds_atomic':
                continue
            if sem.elem_size != 4:
                skipped += 1
                continue
            if 'RTN' not in inst.name.upper():
                skipped += 1
                continue
            if sem.operation not in _DS_ATOMIC_TARGET_OPS:
                skipped += 1
                continue

            mods = _ds_offset_modifiers(enc, offset)
            encoding = build_encoding(isa_spec, enc, inst, CTS_DS_REGS, mods)
            if encoding is None:
                continue

            w0, w1 = encoding
            eff_addr = addr_val + offset

            for initial, operand in pairs:
                old_val, new_val = compute_ds_atomic_golden(
                    sem.operation,
                    initial,
                    operand,
                )

                test_cases.append(
                    f'  {{"{inst.mnemonic}", '
                    f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                    f'0x{addr_val:08X}U, 0x{eff_addr:08X}U, '
                    f'0x{operand:08X}U, 0x{initial:08X}U, '
                    f'0x{old_val:08X}U, 0x{new_val:08X}U}},'
                )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_DS_ATOMIC_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} DS atomic RTN.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <array>',
        '#include <cstdint>',
        '#include <string_view>',
        '',
        f'namespace rocjitsu::cts::{isa_name} {{',
        '',
        'struct DsAtomicRtnTestCase {',
        '  std::string_view mnemonic;',
        '  std::array<uint32_t, 2> encoding;',
        '  uint32_t addr_value;',
        '  uint32_t lds_addr;',
        '  uint32_t data_operand;',
        '  uint32_t initial_lds;',
        '  uint32_t expected_vdst;',
        '  uint32_t expected_lds;',
        '};',
        '',
        *_emit_test_array(
            'DsAtomicRtnTestCase',
            'DS_ATOMIC_RTN_TESTS',
            'NUM_DS_ATOMIC_RTN_TESTS',
            test_cases,
        ),
        '',
        f'}} // namespace rocjitsu::cts::{isa_name}',
        '',
        f'#endif // {guard}',
        '',
    ]

    out_path = os.path.join(isa_dir, 'ds_atomic.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


# ---------------------------------------------------------------------------
#  MFMA / WMMA matrix instruction tests
# ---------------------------------------------------------------------------

_MFMA_ENC_NAMES = frozenset({'VOP3P_MFMA'})
_WMMA_ENC_NAMES = frozenset({'ENC_VOP3P'})

_MFMA_SKIP_PREFIXES = (
    'V_SMFMAC_',
    'V_SWMMAC_',
    'V_MFMA_F64_',
    'V_WMMA_BF16F32_',
    'V_WMMA_SCALE_',
)

_MFMA_SKIP_SUFFIXES = (
    '_F8F6F4',
    '_F4',
)


def _isa_group_from_name(isa_name: str) -> str:
    """Map ISA name to mma_golden isa_group."""
    if isa_name.startswith('cdna'):
        return 'cdna'
    if isa_name == 'gfx1250':
        return 'gfx1250'
    if isa_name.startswith('rdna'):
        return 'rdna'
    return 'cdna'


def _should_skip_mfma(name: str) -> bool:
    """Return True for instructions we can't test yet."""
    upper = name.upper()
    if any(upper.startswith(p) for p in _MFMA_SKIP_PREFIXES):
        return True
    if any(upper.endswith(s) for s in _MFMA_SKIP_SUFFIXES):
        return True
    return False


def _emit_vgpr_array(
    name: str, vgprs: list[list[int]], indent: str = '  '
) -> list[str]:
    """Emit a constexpr uint32_t array for VGPR data."""
    lines = [f'{indent}static constexpr uint32_t {name}[] = {{']
    for vi, lane_data in enumerate(vgprs):
        vals = ', '.join(f'0x{v:08X}U' for v in lane_data)
        lines.append(f'{indent}  {vals},')
    lines.append(f'{indent}}};')
    return lines


def _generate_mfma(
    isa_name: str, isa_spec, semantics, output_dir: str
) -> tuple[int, int]:
    """Generate mfma.h — MFMA/WMMA matrix instruction tests."""
    from amdisa.codegen.cts.mma_golden import (
        generate_test_case,
        parse_mfma_params,
    )

    test_cases: list[str] = []
    data_arrays: list[str] = []
    skipped = 0
    profile = isa_spec.profile
    isa_group = _isa_group_from_name(isa_name)
    is_cdna = isa_group == 'cdna'
    isa_detail = isa_name  # pass full ISA name for B-value disambiguation

    case_idx = 0

    inst_enc_pairs: list[tuple] = []
    for enc in isa_spec.inst_encodings:
        for inst in enc.insts:
            if inst.name in semantics and semantics[inst.name].semantic_class == 'mfma':
                inst_enc_pairs.append((inst, enc))

    for inst, enc in inst_enc_pairs:
        if _should_skip_mfma(inst.name):
            skipped += 1
            continue

        mn = inst.mnemonic.lower()
        p = parse_mfma_params(mn, isa_group)
        if p is None:
            skipped += 1
            continue

        regs = CTS_MFMA_REGS if is_cdna else CTS_WMMA_REGS

        configs = [
            ('basic', False, 0, 0, 0),
            ('with_acc', True, 0, 0, 0),
        ]
        if p.has_cbsz and p.B > 1:
            configs.append(('cbsz1', False, 1, 0, 0))
        if p.has_cbsz:
            configs.append(('blgp3', False, 0, 0, 3))

        for config_label, has_acc, cbsz, abid, blgp in configs:
            enc_mods: dict[str, int] = {}
            if is_cdna:
                enc_mods['acc_cd'] = 1
            if cbsz != 0:
                enc_mods['cbsz'] = cbsz
                enc_mods['abid'] = abid
            if blgp != 0:
                enc_mods['blgp'] = blgp

            encoding = build_encoding(
                isa_spec,
                enc,
                inst,
                regs,
                enc_mods if enc_mods else None,
            )
            if encoding is None:
                continue

            w0, w1 = encoding

            td = generate_test_case(
                mn,
                isa_group,
                has_acc=has_acc,
                cbsz=cbsz,
                abid=abid,
                blgp=blgp,
                config_label=config_label,
                isa_name=isa_detail,
            )
            if td is None:
                continue

            prefix = f'tc_{case_idx}'
            case_idx += 1

            data_arrays.extend(_emit_vgpr_array(f'{prefix}_src0', td.src0_vgprs))
            data_arrays.append('')
            data_arrays.extend(_emit_vgpr_array(f'{prefix}_src1', td.src1_vgprs))
            data_arrays.append('')

            if td.acc_vgprs is not None:
                data_arrays.extend(_emit_vgpr_array(f'{prefix}_acc', td.acc_vgprs))
                data_arrays.append('')

            data_arrays.extend(
                _emit_vgpr_array(f'{prefix}_expected', td.expected_vgprs)
            )
            data_arrays.append('')

            ACC_VGPR_OFFSET = 256
            src0_base = 0
            src1_base = 16
            if isa_group == 'gfx1250':
                dst_base = 32
            else:
                dst_base = ACC_VGPR_OFFSET + 32
            acc_base = dst_base

            has_acc_str = 'true' if has_acc else 'false'
            acc_ptr = f'{prefix}_acc' if has_acc else 'nullptr'

            test_cases.append(
                f'  {{"{inst.mnemonic}", '
                f'{{0x{w0:08X}U, 0x{w1:08X}U}}, '
                f'{len(td.src0_vgprs)}, {len(td.src1_vgprs)}, '
                f'{len(td.expected_vgprs)}, '
                f'{src0_base}, {src1_base}, {dst_base}, {acc_base}, '
                f'{td.eff_wave_size}, '
                f'{has_acc_str}, '
                f'{prefix}_src0, {prefix}_src1, '
                f'{acc_ptr}, {prefix}_expected}},'
            )

    isa_dir = os.path.join(output_dir, isa_name)
    os.makedirs(isa_dir, exist_ok=True)

    max_chunk_bytes = 400_000
    chunk_files: list[str] = []
    total_data_bytes = sum(len(l) + 1 for l in data_arrays)
    if total_data_bytes > max_chunk_bytes:
        chunk: list[str] = []
        chunk_bytes = 0
        chunk_idx = 0
        for line in data_arrays:
            chunk.append(line)
            chunk_bytes += len(line) + 1
            if chunk_bytes >= max_chunk_bytes and line.strip() == '};':
                fname = f'mfma_data_{chunk_idx}.h'
                chunk_files.append(fname)
                dg = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_MFMA_DATA_{chunk_idx}_H_'
                dlines = [
                    f'// Auto-generated MFMA data chunk {chunk_idx} for {isa_name}.',
                    '// Do not edit — regenerate with cts_generator.py.',
                    '',
                    f'#ifndef {dg}',
                    f'#define {dg}',
                    '',
                    '#include <cstdint>',
                    '',
                    f'namespace rocjitsu::cts::{isa_name} {{',
                    '',
                    *chunk,
                    '',
                    f'}} // namespace rocjitsu::cts::{isa_name}',
                    '',
                    f'#endif // {dg}',
                    '',
                ]
                with open(os.path.join(isa_dir, fname), 'w') as f:
                    f.write('\n'.join(dlines))
                chunk = []
                chunk_bytes = 0
                chunk_idx += 1
        if chunk:
            fname = f'mfma_data_{chunk_idx}.h'
            chunk_files.append(fname)
            dg = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_MFMA_DATA_{chunk_idx}_H_'
            dlines = [
                f'// Auto-generated MFMA data chunk {chunk_idx} for {isa_name}.',
                '// Do not edit — regenerate with cts_generator.py.',
                '',
                f'#ifndef {dg}',
                f'#define {dg}',
                '',
                '#include <cstdint>',
                '',
                f'namespace rocjitsu::cts::{isa_name} {{',
                '',
                *chunk,
                '',
                f'}} // namespace rocjitsu::cts::{isa_name}',
                '',
                f'#endif // {dg}',
                '',
            ]
            with open(os.path.join(isa_dir, fname), 'w') as f:
                f.write('\n'.join(dlines))

    guard = f'ROCJITSU_CTS_DATA_{isa_name.upper()}_MFMA_H_'
    lines = [
        f'// Auto-generated CTS test data for {isa_name} MFMA/WMMA.',
        '// Do not edit — regenerate with cts_generator.py.',
        '',
        f'#ifndef {guard}',
        f'#define {guard}',
        '',
        '#include <cstdint>',
        '#include <string_view>',
        '',
    ]

    if chunk_files:
        for cf in chunk_files:
            lines.append(f'#include "{cf}"')
        lines.append('')
    lines.append(f'namespace rocjitsu::cts::{isa_name} {{')
    lines.append('')

    if not chunk_files:
        lines.extend(data_arrays)
        lines.append('')

    lines.append('struct MfmaTestCase {')
    lines.append('  std::string_view mnemonic;')
    lines.append('  uint32_t encoding[2];')
    lines.append('  uint16_t num_src0_vgprs;')
    lines.append('  uint16_t num_src1_vgprs;')
    lines.append('  uint16_t num_dst_vgprs;')
    lines.append('  uint16_t src0_base;')
    lines.append('  uint16_t src1_base;')
    lines.append('  uint16_t dst_base;')
    lines.append('  uint16_t acc_base;')
    lines.append('  uint16_t wf_size;')
    lines.append('  bool has_acc;')
    lines.append('  const uint32_t *src0_data;')
    lines.append('  const uint32_t *src1_data;')
    lines.append('  const uint32_t *acc_data;')
    lines.append('  const uint32_t *expected;')
    lines.append('};')
    lines.append('')

    lines.extend(
        _emit_test_array('MfmaTestCase', 'MFMA_TESTS', 'NUM_MFMA_TESTS', test_cases)
    )
    lines.append('')
    lines.append(f'}} // namespace rocjitsu::cts::{isa_name}')
    lines.append('')
    lines.append(f'#endif // {guard}')
    lines.append('')

    out_path = os.path.join(isa_dir, 'mfma.h')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(test_cases), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate CTS test data')
    parser.add_argument(
        '--multi',
        nargs='+',
        required=True,
        help='ISA entries as name:xml_path (e.g., cdna1:/path/to/cdna1.xml)',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for generated test data headers',
    )
    args = parser.parse_args()

    total = 0
    for entry in args.multi:
        if ':' not in entry:
            print(
                f'error: --multi entry must be name:xml_path, got: {entry}',
                file=sys.stderr,
            )
            sys.exit(1)
        name, xml_path = entry.split(':', 1)
        profile = _detect_profile(xml_path)
        spec = Parser(xml_path, profile).parse()
        semantics = derive_all_semantics(spec)

        count, n_skipped = _generate_scalar_alu(name, spec, semantics, args.output_dir)
        total += count
        print(f'  {name}: {count} scalar ALU test cases ({n_skipped} insts skipped)')

        count, n_skipped = _generate_scalar_cmp(name, spec, semantics, args.output_dir)
        total += count
        print(f'  {name}: {count} scalar CMP test cases ({n_skipped} insts skipped)')

        for vcat in ('vector_alu_unary', 'vector_alu_binary'):
            count, n_skipped = _generate_vector_alu(
                name, spec, semantics, args.output_dir, vcat
            )
            total += count
            print(f'  {name}: {count} {vcat} test cases ({n_skipped} insts skipped)')

        count, n_skipped = _generate_vector_cmp(name, spec, semantics, args.output_dir)
        total += count
        print(f'  {name}: {count} vector_cmp test cases ({n_skipped} insts skipped)')

        count, n_skipped = _generate_vector_ternary(
            name, spec, semantics, args.output_dir
        )
        total += count
        print(
            f'  {name}: {count} vector_alu_ternary test cases ({n_skipped} insts skipped)'
        )

        count, n_skipped = _generate_vector_modifier(
            name, spec, semantics, args.output_dir
        )
        total += count
        print(
            f'  {name}: {count} vector_modifier test cases ({n_skipped} insts skipped)'
        )

        count, n_skipped = _generate_vector_bitop3(
            name, spec, semantics, args.output_dir
        )
        total += count
        print(f'  {name}: {count} vector_bitop3 test cases ({n_skipped} insts skipped)')

        count, n_skipped = _generate_dot_product(name, spec, semantics, args.output_dir)
        total += count
        print(f'  {name}: {count} dot_product test cases ({n_skipped} insts skipped)')

        count, n_skipped = _generate_packed_int(name, spec, semantics, args.output_dir)
        total += count
        print(f'  {name}: {count} packed_int test cases ({n_skipped} insts skipped)')

        count, n_skipped = _generate_packed_fp16(name, spec, semantics, args.output_dir)
        total += count
        print(f'  {name}: {count} packed_fp16 test cases ({n_skipped} insts skipped)')

        count, n_skipped = _generate_mad_64_32(name, spec, semantics, args.output_dir)
        total += count
        print(f'  {name}: {count} mad_64 test cases ({n_skipped} insts skipped)')

        count, n_skipped = _generate_ds_read_write(
            name, spec, semantics, args.output_dir
        )
        total += count
        print(f'  {name}: {count} ds_read_write test cases ({n_skipped} insts skipped)')

        count, n_skipped = _generate_ds_atomic_rtn(
            name, spec, semantics, args.output_dir
        )
        total += count
        print(f'  {name}: {count} ds_atomic_rtn test cases ({n_skipped} insts skipped)')

        count, n_skipped = _generate_mfma(name, spec, semantics, args.output_dir)
        total += count
        print(f'  {name}: {count} mfma test cases ({n_skipped} insts skipped)')

    print(f'\nTotal: {total} test cases across {len(args.multi)} ISAs')


if __name__ == '__main__':
    main()
