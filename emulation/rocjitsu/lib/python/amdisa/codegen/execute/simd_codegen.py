# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations
from dataclasses import dataclass
from amdisa.codegen.execute import (
    conversion,
    float_compare,
    float_fract,
    float_ldexp,
    float_minmax,
    input_policy,
)
from amdisa.codegen.execute.floating_policy import (
    F16_FLUSHED_SOURCE_OPS,
    F16_TRANSCENDENTAL_OPS,
    FLUSH_NEAREST_F32_OPS,
    INPUT_FLUSHED_ROUNDING,
    TRANS_F32_OPS,
)
from amdisa.codegen.execute.fp8_formats import uses_shared_fp8_widening
from amdisa.semantics import F64_TO_INTEGER_DTYPES

"""SIMD fast-path probes for generated AMDGPU execute kernels.

Instruction tables select functors and helpers for each format and encoding.
Probes return after successful SIMD execution; scalar bodies remain fallbacks.
Toolchains without ``<experimental/simd>`` use helpers that return false.

IEEE-2019 min/max uses float_minmax.simd_probe and shared raw-bit selection,
including source/output modifiers on VOP3. VOP2 F16 *_NUM entries use the same
selection through SIMD_VOP2_BINARY for register-half handling. Older host-float
min/max functors remain separate, with their documented NaN/signed-zero limits.
"""


# Operation and operands of each F32 binary functor. SUB and SUBREV add the
# negated subtrahend, keeping the sources in operand order (see _vec_binop_expr).
_BINARY_F32_FORMS = {
    'add': ('ADD', 'a, b'),
    'sub': ('ADD', 'a, -b'),
    'subrev': ('ADD', '-a, b'),
    'mul': ('MUL', 'a, b'),
}


@dataclass(frozen=True)
class ArithmeticCore:
    """How a shared arithmetic core must be driven by whatever probe emits it.

    ``round_mode`` is the rounding the probe installs for the whole instruction;
    ``guest_gate`` is the separate MODE predicate the route needs, or None for any
    MODE. Keeping them apart matters for F16, which needs a nearest-even
    environment *and* rejects directed guest rounding.
    """

    round_mode: str
    guest_gate: str | None


# The single source of truth for environment ownership: a functor naming one of
# these cores leaves its guest FP environment to the probe. The VOP2 and local
# word probes install the recipe's round_mode. The VOP3 F32 and VOP3P packed-F32
# helpers install F32 MODE rounding in C++ instead, so _vop3_f32_core admits only
# cores whose recipe asks for that rounding; the packed-F32 ternary functor uses
# its unscoped core without consulting this table. Host-float functors (MAD, min/max, DIV_FIXUP) are absent on purpose: they
# retain their existing code generation, host NaN selection and flag behavior. A
# scope would contain the flags they raise, which is why VOPD takes one around its
# own host multiply in simd_coverage.h; the generated routes stay unscoped.
_ARITHMETIC_CORES: dict[str, ArithmeticCore] = {
    'amdgpu::binary_f32_simd_unscoped<': ArithmeticCore('wf.fp_round_mode_f32()', None),
    'amdgpu::fma_f32_simd_unscoped(': ArithmeticCore('wf.fp_round_mode_f32()', None),
}


def arithmetic_core(cpp_op: str) -> ArithmeticCore | None:
    """The environment recipe a functor requires, or None if it needs no scope."""
    for core, recipe in _ARITHMETIC_CORES.items():
        if core in cpp_op:
            return recipe
    return None


def _vop3_f32_core(cpp_op: str) -> bool:
    """Whether a VOP3 F32 functor needs its probe's instruction environment.

    Those helpers install F32 MODE rounding themselves, so they can adopt only
    cores whose recipe asks for it.
    """
    recipe = arithmetic_core(cpp_op)
    if recipe is None:
        return False
    assert recipe.round_mode == 'wf.fp_round_mode_f32()', cpp_op
    return True


def _binary_f32_op(name: str, *, modifiers: bool = False) -> str:
    operation, operands = _BINARY_F32_FORMS[name]
    capture = '&inst, &wf' if modifiers else '&wf'
    omod = ', amdgpu::effective_vop3_omod_f32(wf, inst.inst_.omod)' if modifiers else ''
    return (
        f'[{capture}](auto a, auto b) {{ return '
        f'amdgpu::binary_f32_simd_unscoped<amdgpu::fp_mode::Arithmetic::{operation}>('
        f'{operands}, wf{omod}); }}'
    )


# template_name -> (cpp_element_type, cpp_binary_op_functor)
#
# template_name matches the symbol emitted by _generator.gen_shared_execute:
#   f"{inst.mnemonic}_{enc_key}"  (e.g. "v_add_f32_vop2").
#
# The functor is invoked as `bin_op(simd<T>, simd<T>) -> simd<T>` inside
# try_execute_binary_vop2_simd. Use std::*<> for stateless ops.
SIMD_VOP2_BINARY: dict[str, tuple[str, str]] = {
    # --- float32 (IEEE-754 single-rounded, bit-identical to scalar body) ---
    "v_add_f32_vop2": ("float32_t", _binary_f32_op('add')),
    'v_sub_f32_vop2': ('float32_t', _binary_f32_op('sub')),
    'v_subrev_f32_vop2': ('float32_t', _binary_f32_op('subrev')),
    'v_mul_f32_vop2': ('float32_t', _binary_f32_op('mul')),
}


# template_name -> (cpp_in_type, cpp_out_type, cpp_unary_op_functor)
#
# Same keying as SIMD_VOP2_BINARY. The functor is invoked as
#   un_op(simd<Tin>) -> simd<Tout>
# inside try_execute_unary_vop1_simd. Tin and Tout are both 32-bit lane
# types and may differ (e.g. int32->float32 for v_cvt_f32_i32). Eligible
# kernels preserve the generated scalar result: elementwise bit operations,
# exact casts, and floating-point helpers with matching guest policies.
# LOG/EXP and RCP/RSQ/SQRT use the shared hardware mappings. SIN/COS stay scalar.
# VOP3 F32 twins that require floating source and output modifiers.
# MOV_B32/ACCVGPR_MOV_B32 use the generic raw-copy path: their scalar
# bodies ignore floating modifiers.
_VOP3_UNARY_FP_F32 = {
    'v_rcp_f32',
    'v_rcp_iflag_f32',
    'v_rsq_f32',
    'v_sqrt_f32',
    'v_exp_f32',
    'v_log_f32',
}


_VOP3_UNARY_SKIP = {'v_cvt_f32_bf8', 'v_cvt_f32_fp8', 'v_mov_b16'}


def _flush_input_then(fmt: str, functor: str) -> str:
    """Flush MODE input denormals in the raw source before ``functor``.

    CEIL and FLOOR see the flushed zero, as gfx1201 does: ceil(+tiny) is +0.
    """
    policy = input_policy.policy_expr(fmt.lower())
    return f'amdgpu::flush_input_then<amdgpu::fp_format::{fmt}>({policy}, {functor})'


SIMD_VOP1_UNARY: dict[str, tuple[str, str, str]] = {
    # RDNA3+ 16-bit move / not / int16<->int32 conversions (low-16, zero-extend
    # to the 32-bit VGPR). The _vop3 twins auto-route through the same VOP1 path,
    # except v_mov_b16 whose VOP3 form applies float omod/clamp (see
    # _VOP3_UNARY_SKIP above).
    'v_mov_b16_vop1': ('uint32_t', 'uint32_t', '[](auto a) { return a & 0xFFFFu; }'),
    'v_not_b16_vop1': ('uint32_t', 'uint32_t', '[](auto a) { return (~a) & 0xFFFFu; }'),
    # --- f8 -> f32 (E4M3 / E5M2 8-bit float expand) ----------------------------
    # Scalar reads the low byte of src0 and expands via util::fp8_e4m3_to_f32 /
    # bf8_e5m2_to_f32 (no op_sel / byte-select in the body), so the unary functor
    # mirrors them bit-for-bit through the vector ports (denormal, ±0, max-normal /
    # Inf, NaN all handled). The VOP3 twins are modifier-free -> auto-route here.
    'v_cvt_f32_fp8_vop1': (
        'uint32_t',
        'float32_t',
        '[](auto a) { return util::fp8_e4m3_to_f32_simd(a); }',
    ),
    'v_cvt_f32_bf8_vop1': (
        'uint32_t',
        'float32_t',
        '[](auto a) { return util::bf8_e5m2_to_f32_simd(a); }',
    ),
}


# template_name -> (cpp_type, k_literal_expr, cpp_fma_op_functor)
#
# Same keying as SIMD_VOP2_BINARY. The VOP2 FMA/MAC/MAD family has three operand
# shapes, all built on the single-rounded fused multiply-add (the scalar bodies
# use std::fma). The functor is invoked as
#   fma_op(simd<T> src0, simd<T> vsrc1, simd<T> vdst, simd<T> k) -> simd<T>
# inside try_execute_ternary_vop2_simd for literal forms, or
# try_execute_ternary_vop2_acc_simd for dst-accumulate forms; `k` is the
# broadcast inline literal (`k_literal_expr`, an inst.-qualified expression, or
# "0u" when there is none).
# Shapes:
#   dst-accumulate (fmac/mac):     fma(s0, s1, dvst)        -- ignores k
#   literal addend (fmaak/madak):  fma(s0, s1, k)           -- ignores dvst
#   literal mult  (fmamk/madmk):   fma(s0, k, s1)           -- ignores dvst
# The inline-literal entries read K through the RegisterAccess facade
# (`read_scalar(inst.simm32)`), matching the scalar FMA-K executors so read-
# observation plugins see the literal. read_scalar on the fieldless OPR_SIMM32
# returns the literal's encoding bits and broadcast<T> bit-casts them, so this is
# value-identical to a raw `inst.simm32.encoding_value_` read. The operand name
# is hard-coded to `simm32` (correct for every ISA shared here; gfx1250 names its
# field-bearing literal `literal` and is kept out of this path by the sharing
# preflight). `simd_ternary_literal_operand_name()` below exposes that assumption
# so the generator can assert it per ISA.
# The f16 entries here are the legacy MAD literal forms. They widen each operand
# to f32 and narrow once after the arithmetic, matching their scalar bodies.
# Architectural F16 FMA forms use rj-arith, which owns guest rounding and CPU
# dispatch. v_fmac_f64 retains the split-VGPR, MODE-aware SIMD_VOP2_FMA_F64 path.
_FMA_ACC_F32 = '[&wf](auto a, auto b, auto d, auto) { return amdgpu::fma_f32_simd_unscoped(a, b, d, wf); }'


_FMA_ADDK_F32 = '[&wf](auto a, auto b, auto, auto k) { return amdgpu::fma_f32_simd_unscoped(a, b, k, wf); }'


_FMA_MULK_F32 = '[&wf](auto a, auto b, auto, auto k) { return amdgpu::fma_f32_simd_unscoped(a, k, b, wf); }'


# Inline-literal K read, routed through the RegisterAccess facade (see comment
# above). Hard-coded operand name `simm32`; simd_ternary_literal_operand_name()
# parses it back out for the generator's per-ISA assertion.
_FMA_K_READ = 'amdgpu::RegisterAccess(wf).read_scalar(inst.simm32)'


SIMD_VOP2_TERNARY: dict[str, tuple[str, str, str]] = {
    # --- f32 dst-accumulate ---
    'v_fmac_f32_vop2': ('float32_t', '0u', _FMA_ACC_F32),
    'v_mac_f32_vop2': ('float32_t', '0u', _FMA_ACC_F32),
    # --- f32 inline literal ---
    'v_fmaak_f32_vop2': ('float32_t', _FMA_K_READ, _FMA_ADDK_F32),
    'v_madak_f32_vop2': ('float32_t', _FMA_K_READ, _FMA_ADDK_F32),
    'v_fmamk_f32_vop2': ('float32_t', _FMA_K_READ, _FMA_MULK_F32),
    'v_madmk_f32_vop2': ('float32_t', _FMA_K_READ, _FMA_MULK_F32),
}


def simd_ternary_literal_operand_name(template_name: str) -> str | None:
    """Return the C++ operand name a shared SIMD ternary template reads its
    inline literal from, or None if it has none.

    Inline-literal FMA entries render ``k`` from
    ``amdgpu::RegisterAccess(wf).read_scalar(inst.<name>)``, hard-coding
    ``<name>`` (today always ``simm32``). The generator asserts the instruction
    carries that operand, so an ISA whose literal is named differently fails at
    generation instead of emitting non-compiling C++. Dst-accumulate forms
    (``k`` == ``"0u"``) return None.
    """
    spec = SIMD_VOP2_TERNARY.get(template_name)
    if spec is None:
        return None
    k_expr = spec[1]
    prefix = 'amdgpu::RegisterAccess(wf).read_scalar(inst.'
    suffix = ')'
    if k_expr.startswith(prefix) and k_expr.endswith(suffix):
        return k_expr[len(prefix) : -len(suffix)]
    return None


SIMD_VOP2_TERNARY_ACCUMULATE = {
    'v_fmac_f32_vop2',
    'v_mac_f32_vop2',
}


# template_name -> cpp_bin_op (over native<double>, no modifiers). VOP2 f64
# binary forms: scalar bodies read src0/vsrc1 as read_lane64, no abs/neg/omod/
# clamp. Min/max read raw encodings instead; see float_minmax.simd_probe.
SIMD_VOP2_BINARY_FP64: dict[str, str] = {
    'v_add_f64_vop2': '[](auto a, auto b) { return a + b; }',
    'v_mul_f64_vop2': '[](auto a, auto b) { return a * b; }',
}


# --- 64-bit-lane VOP1 unary (f64 math + v_mov_b64) -------------------------
#
# Maps template_name -> (lane_cpp_type, unary functor). Read/written as
# native<T> through the split lo/hi VGPR-pair path (read_simd64/write_simd64).
# The math ops use T = double and mirror the scalar body verbatim: the scalar
# rcp/rsq write `1.0f / x` (the float 1.0f converts exactly to 1.0 and the
# division is done in double), so the SIMD form uses native<double>(1.0). All
# map to correctly-rounded IEEE ops (vroundpd / vsqrtpd / vdivpd), bit-identical
# to std::* for finite/Inf inputs; NaN-input payload divergence is accepted (see
# the glue note + UtilSimd.*F64*_BitExact guards). v_mov_b64 is a pure 64-bit
# copy (T = uint64_t).
SIMD_VOP1_UNARY_F64: dict[str, tuple[str, str]] = {
    'v_mov_b64_vop1': ('uint64_t', '[](auto a) { return a; }'),
}


# FREXP exponent extraction still needs a mixed-width FP64-to-32-bit bridge.
SIMD_CVT_F64_TO_B32: dict[str, tuple[str, str]] = {}


# v_dot2_f32_{f16,bf16} — two half-precision products + an f32 accumulator into
# one f32 lane. Eligibility: the HostF32 accumulation policy and the default
# op_sel packing only; floating DOT ignores CLAMP. The set spans BOTH 16-bit
# float formats, which share the entire dot2 structure but differ in how each
# half is widened to f32 (f16 has a 5-bit exponent with denormal renormalization;
# bf16 has an 8-bit exponent and is a pure left-shift). Both route through the
# same ROCJITSU_TRY_SIMD_VOP3P_DOT_F16 glue, which takes the format as an
# argument so it selects util::{f16,bf16}_to_f32_simd accordingly; see
# try_execute_vop3p_dot_f16_simd in simd_glue.h.
SIMD_VOP3P_DOT_F16_OR_BF16: set[str] = {
    'v_dot2_f32_f16_vop3p',
    'v_dot2_f32_bf16_vop3p',
}


# suffix -> (lane_cpp_type, operand-conversion template using {x})
_VOPC_SUFFIX: dict[str, tuple[str, str]] = {
    'f32': ('float32_t', '{x}'),
    'f16': ('uint32_t', 'util::f16_to_f32_simd({x})'),
    # 64-bit lanes: read directly as the native 64-bit lane type (read_simd64),
    # so the conversion is the identity. Routed through the VOPC64 glue.
    'f64': ('double', '{x}'),
}


# relation -> mask expression over converted operands {a}, {b}. These mirror the
# generated scalar comparison expressions exactly (incl. float ordered/unordered
# NaN semantics): the n* forms are the logical negation of the base relation, so
# they are true on NaN; o/u test orderedness directly.
_VOPC_REL: dict[str, str] = {
    'eq': '{a} == {b}',
    'lt': '{a} < {b}',
    'le': '{a} <= {b}',
    'gt': '{a} > {b}',
    'ge': '{a} >= {b}',
    'ne': '{a} != {b}',  # integer not-equal
    # float less-or-greater: ordered, FALSE on NaN. Scalar body is (a<b)||(a>b),
    # NOT a!=b (which is TRUE on NaN). nlg is its logical negation.
    'lg': '({a} < {b}) || ({a} > {b})',
    'neq': '{a} != {b}',  # float not-equal
    'nge': '!({a} >= {b})',
    'ngt': '!({a} > {b})',
    'nle': '!({a} <= {b})',
    'nlg': '!(({a} < {b}) || ({a} > {b}))',
    'nlt': '!({a} < {b})',
    'o': '!util::stdx::isnan({a}) && !util::stdx::isnan({b})',
    'u': 'util::stdx::isnan({a}) || util::stdx::isnan({b})',
}


# Constant relations carry no comparison: f is always-false, t/tru always-true.
# The mask type is taken from the converted-operand compare so it matches lane
# width regardless of suffix.
_VOPC_CONST: dict[str, str] = {'f': 'false', 't': 'true', 'tru': 'true'}


_VOPC_FLOAT_RELS = [
    'eq',
    'ge',
    'gt',
    'le',
    'lg',
    'lt',
    'neq',
    'nge',
    'ngt',
    'nle',
    'nlg',
    'nlt',
    'o',
    'u',
    'f',
    'tru',
]


_VOPC_INT_RELS = ['eq', 'ge', 'gt', 'le', 'lt', 'ne', 'f', 't']


def _vopc_functor(conv: str, rel: str) -> str:
    ca = conv.format(x='a')
    cb = conv.format(x='b')
    if rel in _VOPC_CONST:
        body = f'decltype({ca} == {cb})({_VOPC_CONST[rel]})'
    else:
        body = _VOPC_REL[rel].format(a=ca, b=cb)
    return f'[](auto a, auto b) {{ return {body}; }}'


def _float_vopc_entry(suf: str, rel: str) -> tuple[str, str]:
    """Float relations evaluate raw encodings through shared/comparison.h."""
    if float_compare.is_float_relation(suf, rel):
        return float_compare.lane_type(suf), float_compare.simd_functor(suf, rel)
    lane_t, conv = _VOPC_SUFFIX[suf]
    return lane_t, _vopc_functor(conv, rel)


def _build_simd_vopc() -> dict[str, tuple[str, str]]:
    table: dict[str, tuple[str, str]] = {}
    # 16-/32-bit lane suffixes only; f64/i64/u64 (64-bit lane) wired separately.
    for suf in ('f16', 'f32'):
        for rel in _VOPC_FLOAT_RELS:
            table[f'v_cmp_{rel}_{suf}_vopc'] = _float_vopc_entry(suf, rel)
    return table


def _build_simd_vopc64() -> dict[str, tuple[str, str]]:
    """FP64 VOPC compares, routed through the VOPC64 glue."""
    table: dict[str, tuple[str, str]] = {}
    for rel in _VOPC_FLOAT_RELS:
        table[f'v_cmp_{rel}_f64_vopc'] = _float_vopc_entry('f64', rel)
    return table


SIMD_VOPC: dict[str, tuple[str, str]] = _build_simd_vopc()


SIMD_VOPC64: dict[str, tuple[str, str]] = _build_simd_vopc64()


# --- VOP3 forms of the relational VOPC compares ----------------------------
#
# The VOP3 form of v_cmp_<rel>_<suffix> differs from the VOPC form in three
# ways: (1) it reads src0/src1 (not src0/vsrc1), (2) abs/neg per-source
# modifiers apply to floating-point operands, and (3) the per-lane compare
# result is packed into a lane mask and passed to the generated result writer.
# The instruction wrapper owns the final SGPR/EXEC merge, including DPP
# invalid-source preservation and destination-mask zeroing. The lane packing is
# otherwise identical to the VOPC path; this is what the VOP3 VOPC glue
# templates implement.
#
# Integer/bitwise VOPC bodies apply no modifiers, so their functors are the
# same as the VOPC ones (built by _vopc_functor); they go through
# try_execute_vopc_vop3_int_simd (32-bit lane) or
# try_execute_vopc64_vop3_int_simd (64-bit lane).
#
# Floating-point VOPC bodies in VOP3 form read the raw encodings through the
# same raw-lane glue; shared/comparison.h applies the captured abs/neg fields,
# MODE input flushing and NaN ordering inside the functor. F16 sources occupy
# 32-bit lanes and may select their half with true16 OPSEL.
#
# VOP3 fp adds two extra rels vs VOPC: 't' alongside 'tru' (both constant
# always-true), so the table keys span the union of the two.
_VOP3_FLOAT_RELS = _VOPC_FLOAT_RELS + ['t']


def _build_simd_vopc_vop3_int_32() -> dict[str, tuple[str, str]]:
    return {}


SIMD_VOPC_VOP3_INT_32: dict[str, tuple[str, str]] = _build_simd_vopc_vop3_int_32()


def _build_simd_vopc_vop3_float() -> dict[str, tuple[str, str]]:
    """VOP3 form of the f16/f32/f64 VOPC relations (16 from VOPC + 't').

    Keyed by ``_vop3``; values are (raw lane type, functor). Constant relations
    ignore their operands.
    """
    modifiers = ('inst.inst_.abs', 'inst.inst_.neg')
    table: dict[str, tuple[str, str]] = {}
    for suf in ('f16', 'f32', 'f64'):
        lane_t = float_compare.lane_type(suf)
        for rel in _VOP3_FLOAT_RELS:
            if float_compare.is_float_relation(suf, rel):
                functor = float_compare.simd_functor(suf, rel, modifiers)
            else:
                functor = _vopc_functor('{x}', rel)
            table[f'v_cmp_{rel}_{suf}_vop3'] = (lane_t, functor)
    return table


SIMD_VOPC_VOP3_FLOAT: dict[str, tuple[str, str]] = _build_simd_vopc_vop3_float()


# --- VOP3 integer/bitwise ternary (3-source) -------------------------------
#
# A handful of integer 3-source VOP3 ops are plain element-wise functions of
# (src0, src1, src2) with no modifiers and no widening: routed through
# try_execute_ternary_vop3_simd<T>. Functor sig: `(native<T> a, native<T> b,
# native<T> c) -> native<T>`. Excluded from this table: the float ternary
# family (fma/mad — needs modifier glue), med3/min3/max3 (NaN /
# signed-integer-vs-fmin semantics — see project_pr6470_review_findings), the
# bfe ops (branchy mask), the byte-permute ops (perm/alignbyte/lerp/sad/msad —
# byte-wise / table-driven), and 64-bit-lane ternary (would need a 64-bit
# ternary glue).
# --- Extra plain integer/bitwise binary VOP3 ops ---------------------------
#
# VOP3-only integer/bitwise binary ops that have no VOP2 twin (so the existing
# _vop3 -> _vop2 fallback doesn't pick them up). All read src0/src1 and apply
# no modifiers; routed through try_execute_binary_vop3_simd<T>.
#
# 16-bit forms compute a 32-bit add/sub and mask the low 16 bits, matching the
# scalar body's `uint32_t(uint16_t(int16_t(low16(a)+low16(b))))` (or the u16
# variant) — both reduce to `(a + b) & 0xFFFFu` / `(a - b) & 0xFFFFu` when
# CLAMP is clear because
# unsigned 32-bit wrap-around at the low 16 bits is identical to signed/unsigned
# 16-bit wrap. 32-bit forms use the wrap-around add/sub on uint32 lanes;
# signed-vs-unsigned wraps the same way. The shared glue falls back to scalar
# execution when CLAMP requests saturating arithmetic.
SIMD_VOP3_BINARY_INT_EXTRA: dict[str, tuple[str, str]] = {
    'v_add_i16_vop3': ('uint32_t', '[](auto a, auto b) { return (a + b) & 0xFFFFu; }'),
    'v_sub_i16_vop3': ('uint32_t', '[](auto a, auto b) { return (a - b) & 0xFFFFu; }'),
    # 16-bit-lane bitwise binary VOP3 ops (RDNA3+; CDNA4 does not decode).
    # The scalar body computes `uint16_t(uint16_t(a) OP uint16_t(b))` and
    # writes the result as a zero-extended uint32; SIMD reproduces with a
    # 32-bit `& 0xFFFFu` mask on the result. No modifiers (integer ops).
    'v_and_b16_vop3': ('uint32_t', '[](auto a, auto b) { return (a & b) & 0xFFFFu; }'),
    'v_or_b16_vop3': ('uint32_t', '[](auto a, auto b) { return (a | b) & 0xFFFFu; }'),
    'v_xor_b16_vop3': ('uint32_t', '[](auto a, auto b) { return (a ^ b) & 0xFFFFu; }'),
}


SIMD_VOP3_BINARY_TRUE16_SRC: dict[str, tuple[str, str]] = {
    # True16 source selectors choose the f16 half of src0/src1, but these
    # instructions write a full packed b32 result.
}


# True16 VOP3 scalar semantics select 16-bit source halves and merge a selected
# destination half. The generic 32-bit VOP3 SIMD glue overwrites the whole dword.
SIMD_VOP3_TRUE16_UNSAFE = frozenset(
    {
        'v_add_i16_vop3',
        'v_add_nc_i16_vop3',
        'v_add_nc_u16_vop3',
        'v_add_u16_vop3',
        'v_and_b16_vop3',
        'v_mul_lo_u16_vop3',
        'v_not_b16_vop3',
        'v_or_b16_vop3',
        'v_sub_i16_vop3',
        'v_sub_nc_i16_vop3',
        'v_sub_nc_u16_vop3',
        'v_sub_u16_vop3',
        'v_xor_b16_vop3',
    }
)


# VOP3 unary integer ops without a VOP1 twin. Reuse the VOP1 unary glue
# (operand shape is identical: src0 in, vdst out, 32-bit lanes) — only the
# probe routing key differs. RDNA3+; CDNA4 does not decode.
SIMD_VOP3_UNARY_INT_EXTRA: dict[str, tuple[str, str, str]] = {
    # v_not_b16: `uint16_t(~src0)`, zero-extended. Same shape as v_not_b32 but
    # masked to 16 bits.
    'v_not_b16_vop3': (
        'uint32_t',
        'uint32_t',
        '[](auto a) { return (~a) & 0xFFFFu; }',
    ),
}


# Plain f64 unary: scalar bodies are std::ceil / std::floor / std::trunc /
# Fixed-direction rounding of the modifier-applied double, then omod/clamp on
# the result. util::{ceil,floor,trunc,rndne}_simd wrap the stdx native<double>
# rounding primitives and repair the two edge cases libstdc++ gets wrong at
# narrow widths (sign-of-zero on a zero-magnitude result, NaN-payload
# preservation), so they are bit-identical to the scalar libm calls for every
# finite / Inf / NaN / signed-zero input.
SIMD_VOP3_UNARY_FP64: dict[str, str] = {}


# --- VOP3 floating-point ternary (FMA / MAD family) ------------------------
#
# F32 FMA and accumulator forms share scalar NaN policy; finite lanes retain
# native SIMD arithmetic. MAD retains its non-fused multiply/add semantics.
SIMD_VOP3_TERNARY_FP32: dict[str, str] = {
    'v_fma_f32_vop3': '[&inst, &wf](auto a, auto b, auto c) { return amdgpu::fma_f32_simd_unscoped(a, b, c, wf, amdgpu::effective_vop3_omod_f32(wf, inst.inst_.omod)); }',
    'v_mad_f32_vop3': '[](auto a, auto b, auto c) { return a * b + c; }',
    'v_mad_legacy_f32_vop3': '[](auto a, auto b, auto c) { return a * b + c; }',
}


# Non-fused F16 ternaries: widen -> ABS/NEG -> evaluate -> round F16 -> OMOD/CLAMP.
# MAD retains its existing MODE policy; selection uses the shared raw-bit stage.
# Fused FMA uses the
# MODE-aware native<double> route selected by SIMD_VOP3_FMA_MODE_FP16 below.
SIMD_VOP3_TERNARY_FP16: dict[str, str] = {}


# f64 ternary FMA.
SIMD_VOP3_TERNARY_FP64: dict[str, str] = {}


# --- VOP3 dst-accumulate FMA / MAC (vdst is the accumulator) ----------------
#
# v_fmac / v_mac per-isa classes only initialize src0+src1+vdst; the third FMA
# operand IS vdst (no src2 Operand). The accumulate-form glue reads inst.vdst
# as the third operand and applies abs/neg only to src0/src1 (per scalar body).
# F32 NaN selection matches the non-accumulate ternary slice.
SIMD_VOP3_FMAC_FP32: dict[str, str] = {
    'v_fmac_f32_vop3': '[&inst, &wf](auto a, auto b, auto c) { return amdgpu::fma_f32_simd_unscoped(a, b, c, wf, amdgpu::effective_vop3_omod_f32(wf, inst.inst_.omod)); }',
    'v_mac_f32_vop3': '[&inst, &wf](auto a, auto b, auto c) { return amdgpu::fma_f32_simd_unscoped(a, b, c, wf, amdgpu::effective_vop3_omod_f32(wf, inst.inst_.omod)); }',
}


SIMD_VOP3_TERNARY_INT: dict[str, tuple[str, str]] = {}


def _indent_probe(probe: str) -> str:
    return '\n'.join(('  ' + line) if line else line for line in probe.splitlines())


_PACKED_NORMALIZED_VOP3 = ()


def _modified_conversion_op(cpp_op: str, *, bits: int, arity: int = 2) -> str:
    """Apply floating source signs before a raw-word SIMD conversion."""
    operands = ('a', 'b')[:arity]
    parameters = ', '.join(f'auto {operand}' for operand in operands)
    sign = f'0x{1 << (bits - 1):x}u'
    modifiers = ''.join(
        f' if (inst.inst_.abs & {1 << i}u) {operand} &= ~{sign};'
        f' if (inst.inst_.neg & {1 << i}u) {operand} ^= {sign};'
        for i, operand in enumerate(operands)
    )
    return (
        f'[&inst]({parameters}) {{'
        + modifiers
        + f' return ({cpp_op})({", ".join(operands)}); }}'
    )


def _prepared_f16_conversion_op(cpp_op: str) -> str:
    """Apply shared ABS/NEG, then MODE input flushing, to two raw F16 sources."""
    prepare = ''.join(
        f' {operand} = amdgpu::input_denormal::prepare<amdgpu::fp_format::F16>('
        f'amdgpu::source_modifier::apply<amdgpu::fp_format::F16>('
        f'{operand}, {i}, inst.inst_.abs, inst.inst_.neg), {input_policy.NAME});'
        for i, operand in enumerate(('a', 'b'))
    )
    return (
        f'[&inst, {input_policy.NAME} = {input_policy.policy_expr("f16")}](auto a, auto b) {{'
        + prepare
        + f' return ({cpp_op})(a, b); }}'
    )


def _mode_aware_f16_result_simd_probe(default_probe: str, ovfl_probe: str) -> str:
    default_body = _indent_probe(default_probe)
    ovfl_body = _indent_probe(ovfl_probe)
    return f'  if (wf.fp16_ovfl()) {{\n{ovfl_body}\n  }} else {{\n{default_body}\n  }}'


def _probe_uses_fp16_ovfl_f16_result_narrow(cpp_op: str) -> bool:
    return 'f32_to_f16_simd' in cpp_op


def _fp16_ovfl_cpp_op(cpp_op: str) -> str:
    return cpp_op.replace('f32_to_f16_simd', 'f32_to_f16_ovfl_simd')


def simd_probe_line(
    template_name: str,
    *,
    true16_vop3: bool = False,
    result_writer: str | None = None,
) -> str | None:
    probe = _simd_probe_line(
        template_name, true16_vop3=true16_vop3, result_writer=result_writer
    )
    return _guard_mode_arithmetic_probe(template_name, probe)


def _guard_mode_arithmetic_probe(template_name: str, probe: str | None) -> str | None:
    """Retain native arithmetic only when it implements the guest FP policy."""
    arithmetic_ops = (
        'add',
        'sub',
        'subrev',
        'mul',
        'fma',
        'fmac',
        'fmaak',
        'fmamk',
        'mac',
        'madak',
        'madmk',
    )
    fields = template_name.split('_')
    if (
        probe is None
        or len(fields) < 3
        or fields[0] != 'v'
        or fields[1] not in arithmetic_ops
    ):
        return probe
    dtype = next((field for field in fields if field in ('f16', 'f32', 'f64')), None)
    if dtype is None:
        return probe
    recipe = arithmetic_core(probe)
    if recipe is not None:
        # The probe owns the guest environment and output scaling is raw-bit, so
        # callbacks keep the caller's FP state. Only the core's own guest gate,
        # if any, narrows the route.
        if recipe.guest_gate is None:
            return probe
        return f'  if ({recipe.guest_gate}) {{\n{probe}\n  }}'
    if dtype in ('f16', 'f64') and fields[1] in ('fma', 'fmac', 'fmaak', 'fmamk'):
        return probe
    mode = 'f32' if dtype == 'f32' else 'f16_f64'
    return (
        f'  if (amdgpu::fp_mode::native_arithmetic_matches(wf.fp_round_mode_{mode}(), '
        f'wf.fp_denorm_mode_{mode}())) {{\n{probe}\n  }}'
    )


def _simd_probe_line(
    template_name: str,
    *,
    true16_vop3: bool = False,
    result_writer: str | None = None,
) -> str | None:
    """Return the SIMD fast-path probe block for a kernel, or None."""
    minmax_probe = float_minmax.simd_probe(template_name)
    if minmax_probe is not None:
        return minmax_probe
    ldexp_probe = float_ldexp.simd_probe(template_name)
    if ldexp_probe is not None:
        return ldexp_probe
    # Conversions to floating formats, which round in the destination's MODE.
    conversion_probe = conversion.simd_probe(template_name, true16_vop3=true16_vop3)
    if conversion_probe is not None:
        return conversion_probe
    if template_name in SIMD_VOP3_TRUE16_UNSAFE:
        return None
    spec2 = SIMD_VOP2_BINARY.get(template_name)
    if spec2 is not None:
        cpp_t, cpp_op = spec2
        recipe = arithmetic_core(cpp_op)
        if recipe is not None:
            probe = (
                f'  ROCJITSU_TRY_SIMD_VOP2_BINARY_SCOPED('
                f'{cpp_t}, {cpp_op}, {recipe.round_mode});'
            )
        else:
            probe = f'  ROCJITSU_TRY_SIMD_VOP2_BINARY({cpp_t}, {cpp_op});'
        if _probe_uses_fp16_ovfl_f16_result_narrow(cpp_op):
            ovfl_probe = f'  ROCJITSU_TRY_SIMD_VOP2_BINARY({cpp_t}, {_fp16_ovfl_cpp_op(cpp_op)});'
            return _mode_aware_f16_result_simd_probe(probe, ovfl_probe)
        return probe
    spec1 = SIMD_VOP1_UNARY.get(template_name)
    if spec1 is not None:
        cpp_tin, cpp_tout, cpp_op = spec1
        probe = f'  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {cpp_op});'
        if _probe_uses_fp16_ovfl_f16_result_narrow(cpp_op):
            ovfl_probe = f'  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {_fp16_ovfl_cpp_op(cpp_op)});'
            return _mode_aware_f16_result_simd_probe(probe, ovfl_probe)
        return probe
    spect = SIMD_VOP2_TERNARY.get(template_name)
    if spect is not None:
        cpp_t, k_expr, cpp_op = spect
        accumulate = template_name in SIMD_VOP2_TERNARY_ACCUMULATE
        recipe = arithmetic_core(cpp_op)
        if recipe is not None:
            scoped = (
                'ROCJITSU_TRY_SIMD_VOP2_TERNARY_ACC_SCOPED'
                if accumulate
                else 'ROCJITSU_TRY_SIMD_VOP2_TERNARY_SCOPED'
            )
            return f'  {scoped}({cpp_t}, {k_expr}, {cpp_op}, {recipe.round_mode});'
        macro = (
            'ROCJITSU_TRY_SIMD_VOP2_TERNARY_ACC'
            if accumulate
            else 'ROCJITSU_TRY_SIMD_VOP2_TERNARY'
        )
        probe = f'  {macro}({cpp_t}, {k_expr}, {cpp_op});'
        if _probe_uses_fp16_ovfl_f16_result_narrow(cpp_op):
            ovfl_probe = f'  {macro}({cpp_t}, {k_expr}, {_fp16_ovfl_cpp_op(cpp_op)});'
            return _mode_aware_f16_result_simd_probe(probe, ovfl_probe)
        return probe
    spec2binf64 = SIMD_VOP2_BINARY_FP64.get(template_name)
    if spec2binf64 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP2_BINARY_FP64({spec2binf64});'
    spec1f64 = SIMD_VOP1_UNARY_F64.get(template_name)
    if spec1f64 is not None:
        lane_t, cpp_op = spec1f64
        return f'  ROCJITSU_TRY_SIMD_VOP1_UNARY_F64({lane_t}, {cpp_op});'
    speccvtout = SIMD_CVT_F64_TO_B32.get(template_name)
    if speccvtout is not None:
        out_t, cpp_op = speccvtout
        return f'  ROCJITSU_TRY_SIMD_CVT_F64_TO_B32({out_t}, {cpp_op});'
    specvopc = SIMD_VOPC.get(template_name)
    if specvopc is not None:
        lane_t, cpp_op = specvopc
        return f'  ROCJITSU_TRY_SIMD_VOPC({lane_t}, {cpp_op});'
    specvopc64 = SIMD_VOPC64.get(template_name)
    if specvopc64 is not None:
        lane_t, cpp_op = specvopc64
        return f'  ROCJITSU_TRY_SIMD_VOPC64({lane_t}, {cpp_op});'
    # VOP3 form of the integer/bitwise VOPC compares (i16/u16/i32/u32 and
    # i64/u64). Same functor as the VOPC table (no modifiers); result-writer
    # variants return the packed mask to the generated wrapper for commit.
    specvopcv3i32 = SIMD_VOPC_VOP3_INT_32.get(template_name)
    if specvopcv3i32 is not None:
        lane_t, cpp_op = specvopcv3i32
        writer_arg = f'{result_writer}, ' if result_writer is not None else ''
        suffix = '_RESULT' if result_writer is not None else ''
        return f'  ROCJITSU_TRY_SIMD_VOPC_VOP3_INT{suffix}({writer_arg}{lane_t}, {cpp_op});'
    # VOP3 form of the f16/f32/f64 VOPC compares (17 ops each). Raw lanes go
    # through the integer glue; the functor applies the captured abs/neg.
    specvopcv3float = SIMD_VOPC_VOP3_FLOAT.get(template_name)
    if specvopcv3float is not None:
        lane_t, cpp_op = specvopcv3float
        if lane_t == 'uint64_t':
            macro = 'ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT'
        elif true16_vop3 and '_f16_' in template_name:
            macro = 'ROCJITSU_TRY_SIMD_VOPC_VOP3_TRUE16_INT'
        else:
            macro = 'ROCJITSU_TRY_SIMD_VOPC_VOP3_INT'
        writer_arg = f'{result_writer}, ' if result_writer is not None else ''
        suffix = '_RESULT' if result_writer is not None else ''
        return f'  {macro}{suffix}({writer_arg}{lane_t}, {cpp_op});'
    # VOP3 integer/bitwise ternary ops (add3/or3/xor3/lshl_add/add_lshl/bfi).
    # Plain element-wise functor of (src0, src1, src2); no modifiers.
    spec3tern = SIMD_VOP3_TERNARY_INT.get(template_name)
    if spec3tern is not None:
        cpp_t, cpp_op = spec3tern
        return f'  ROCJITSU_TRY_SIMD_VOP3_TERNARY_INT({cpp_t}, {cpp_op});'
    # VOP3 fp ternary (FMA / MAD family). Per-source abs/neg + omod/clamp in
    # the f32 / f16 / f64 domain. NaN-input lanes skipped by test (gcc-13
    # packed FMA quiets a different NaN operand vs scalar std::fma).
    spec3tf32 = SIMD_VOP3_TERNARY_FP32.get(template_name)
    if spec3tf32 is not None:
        ternary_macro = (
            'ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP32_SCOPED'
            if _vop3_f32_core(spec3tf32)
            else 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP32'
        )
        return f'  {ternary_macro}({spec3tf32});'
    spec3tf16 = SIMD_VOP3_TERNARY_FP16.get(template_name)
    if spec3tf16 is not None:
        macro = (
            'ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_FP16'
            if true16_vop3
            else 'ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16'
        )
        return f'  {macro}({spec3tf16});'
    spec3tf64 = SIMD_VOP3_TERNARY_FP64.get(template_name)
    if spec3tf64 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP64({spec3tf64});'
    if template_name in SIMD_VOP3P_DOT_F16_OR_BF16:
        fmt = 'BF16' if template_name == 'v_dot2_f32_bf16_vop3p' else 'F16'
        return f'  ROCJITSU_TRY_SIMD_VOP3P_DOT_F16({fmt});'
    # VOP3 dst-accumulate FMA/MAC (vdst is the third operand). Per-isa class
    # has no src2; the accumulate glue reads vdst instead.
    specfmacf32 = SIMD_VOP3_FMAC_FP32.get(template_name)
    if specfmacf32 is not None:
        fmac_macro = (
            'ROCJITSU_TRY_SIMD_FMAC_VOP3_FP32_SCOPED'
            if _vop3_f32_core(specfmacf32)
            else 'ROCJITSU_TRY_SIMD_FMAC_VOP3_FP32'
        )
        return f'  {fmac_macro}({specfmacf32});'
    # VOP3 unary integer ops with no VOP1 twin (e.g. v_not_b16). Reuse the
    # VOP1 unary glue — operand shape (src0, vdst, 32-bit lanes) matches.
    spec3unai = SIMD_VOP3_UNARY_INT_EXTRA.get(template_name)
    if spec3unai is not None:
        if true16_vop3:
            return None
        cpp_tin, cpp_tout, cpp_op = spec3unai
        return f'  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {cpp_op});'
    # Extra plain integer binary VOP3 ops without a VOP2 twin (add_i32/i16,
    # sub_*, nc_* variants). Routed through the int VOP3 binary glue.
    spec3bin16 = SIMD_VOP3_BINARY_TRUE16_SRC.get(template_name)
    if spec3bin16 is not None:
        cpp_t, cpp_op = spec3bin16
        if template_name in _PACKED_NORMALIZED_VOP3:
            cpp_op = _prepared_f16_conversion_op(cpp_op)
        return f'  ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_SRC({cpp_t}, {cpp_op});'
    spec3binx = SIMD_VOP3_BINARY_INT_EXTRA.get(template_name)
    if spec3binx is not None:
        if true16_vop3:
            return None
        cpp_t, cpp_op = spec3binx
        if template_name in _PACKED_NORMALIZED_VOP3:
            cpp_op = _modified_conversion_op(cpp_op, bits=32)
        return f'  ROCJITSU_TRY_SIMD_VOP3_BINARY_INT({cpp_t}, {cpp_op});'
    # VOP3 f64 unary (ceil/floor/trunc/rndne/sqrt). Same modifier policy.
    spec3unaf64 = SIMD_VOP3_UNARY_FP64.get(template_name)
    if spec3unaf64 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP64({spec3unaf64});'
    # VOP3-encoded twins of the SIMD VOP2 binary ops. Same operator/lane type;
    # the VOP3 form reads src0/src1 and carries abs/neg/omod/clamp modifiers.
    # f32 ops apply the modifiers in-vector (bit-exact); integer/bitwise ops
    # apply none (their scalar bodies ignore them), so the plain op suffices.
    if template_name.endswith('_vop3'):
        base = template_name[: -len('_vop3')]
        spec2v3 = SIMD_VOP2_BINARY.get(base + '_vop2')
        if spec2v3 is not None:
            cpp_t, cpp_op = spec2v3
            if cpp_t == 'float32_t':
                if base in ('v_add_f32', 'v_mul_f32', 'v_sub_f32', 'v_subrev_f32'):
                    cpp_op = _binary_f32_op(base.split('_')[1], modifiers=True)
                binary_macro = (
                    'ROCJITSU_TRY_SIMD_VOP3_BINARY_FP_SCOPED'
                    if _vop3_f32_core(cpp_op)
                    else 'ROCJITSU_TRY_SIMD_VOP3_BINARY_FP'
                )
                return f'  {binary_macro}({cpp_t}, {cpp_op});'
            # Remaining F16 binaries reuse unmodified VOP2 functors and reject
            # modifiers. ADD/SUB/SUBREV/MUL use the MODE-aware route above;
            # min/max has its own route. Keep a probe for the shared cross-ISA plan.
            if base.endswith('_f16'):
                macro = (
                    'ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_F16'
                    if true16_vop3
                    else 'ROCJITSU_TRY_SIMD_VOP3_BINARY_F16'
                )
                probe = f'  {macro}({cpp_t}, {cpp_op});'
                if _probe_uses_fp16_ovfl_f16_result_narrow(cpp_op):
                    ovfl_probe = f'  {macro}({cpp_t}, {_fp16_ovfl_cpp_op(cpp_op)});'
                    return _mode_aware_f16_result_simd_probe(probe, ovfl_probe)
                return probe
            if true16_vop3:
                return None
            probe = f'  ROCJITSU_TRY_SIMD_VOP3_BINARY_INT({cpp_t}, {cpp_op});'
            if _probe_uses_fp16_ovfl_f16_result_narrow(cpp_op):
                ovfl_probe = f'  ROCJITSU_TRY_SIMD_VOP3_BINARY_INT({cpp_t}, {_fp16_ovfl_cpp_op(cpp_op)});'
                return _mode_aware_f16_result_simd_probe(probe, ovfl_probe)
            return probe
        # VOP3 twins: F32 modifier glue or eligible VOP1 functors.
        # Specialized F16 routes precede this block; _VOP3_UNARY_SKIP only
        # blocks generic reuse. Integer conversions have modifier gates below.
        spec1v3 = SIMD_VOP1_UNARY.get(base + '_vop1')
        if spec1v3 is not None:
            cpp_tin, cpp_tout, cpp_op = spec1v3
            if true16_vop3:
                return None
            if base in _VOP3_UNARY_SKIP:
                return None
            if base in _VOP3_UNARY_FP_F32:
                if base.split('_')[1] in ('ceil', 'floor', 'trunc', 'rndne'):
                    return f'  ROCJITSU_TRY_SIMD_VOP3_UNARY_ROUNDED_FP32({cpp_op});'
                flush = 'true' if base.upper() in FLUSH_NEAREST_F32_OPS else 'false'
                # TRANS results ignore MODE.FP_ROUND, so their OMOD overflow
                # rounds to nearest even; the scalar lowering selects the same way.
                round_mode = (
                    '0' if base.upper() in TRANS_F32_OPS else 'wf.fp_round_mode_f32()'
                )
                modifier_args = (
                    f', {flush} /* force_output_flush */, {round_mode} /* round_mode */'
                )
                return f'  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP(float32_t, float32_t, {cpp_op}{modifier_args});'
            if base in ('v_cvt_i16_f16', 'v_cvt_u16_f16'):
                cpp_op = _modified_conversion_op(cpp_op, bits=16, arity=1)
            probe = f'  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {cpp_op});'
            if base in (
                'v_cvt_f32_ubyte0',
                'v_cvt_f32_ubyte1',
                'v_cvt_f32_ubyte2',
                'v_cvt_f32_ubyte3',
            ):
                # The VOP1 shortcut has no output scaling or saturation.
                return f'  if (!inst.inst_.omod && !inst.inst_.clamp) {{\n{probe}\n  }}'
            if base in (
                'v_cvt_i32_f32',
                'v_cvt_u32_f32',
            ):
                # The VOP1 shortcut does not apply floating source modifiers.
                # Modified F32 inputs use the scalar conversion body.
                return f'  if (!inst.inst_.abs && !inst.inst_.neg) {{\n{probe}\n  }}'
            if _probe_uses_fp16_ovfl_f16_result_narrow(cpp_op):
                ovfl_probe = f'  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {_fp16_ovfl_cpp_op(cpp_op)});'
                return _mode_aware_f16_result_simd_probe(probe, ovfl_probe)
            return probe
        # FREXP exponent ignores ABS/NEG and reuses its VOP1 lane operation.
        speccvtoutv3 = SIMD_CVT_F64_TO_B32.get(base + '_vop1')
        if speccvtoutv3 is not None:
            if base in _VOP3_UNARY_SKIP:
                return None
            out_t, cpp_op = speccvtoutv3
            probe = f'  ROCJITSU_TRY_SIMD_CVT_F64_TO_B32({out_t}, {cpp_op});'
            return probe
    return None


def vop3p_local_simd_probe_line(
    template_name: str,
    vop3p_opsel_fields: tuple[str, str],
    op_sel_hi_2_expr: str = 'inst_.op_sel_hi_2',
) -> str | None:
    """Return a profile-aware SIMD probe for an ISA-local VOP3P body.

    Shared VOP3P templates use the canonical ``op_sel`` field spelling.  Some
    profiles keep equivalent semantics under renamed fields, so their inline
    scalar bodies need a probe that passes the normalized selector values to
    SIMD glue instead of delegating to the canonical shared template.
    """
    if vop3p_opsel_fields == ('op_sel', 'op_sel_hi'):
        return None
    op_sel, op_sel_hi = vop3p_opsel_fields
    return simd_probe_line(template_name)


def simd_probe_arch_portable(
    template_name: str,
    vop3p_opsel_fields: tuple[str, str] = ('op_sel', 'op_sel_hi'),
) -> bool:
    """Whether a SIMD-probe kernel can be force-routed through the shared
    execute template on an ISA that is not in its cross-ISA shared group.

    Most SIMD fast-path kernels have an arch-independent body (arithmetic /
    compare on `inst.src0` / `inst.vsrc1` / `inst.vdst`), so an ISA whose
    operand/field signature kept it out of the shared plan can still safely
    delegate to the one shared template — the body is identical. The exception
    is the inline-literal FMA family (v_fmaak/fmamk/madak/madmk): those carry
    instruction-local literal state and are left to the genuine shared plan
    instead of being force-routed into a shared SIMD probe. They are identified
    by a non-``"0u"`` literal expression in SIMD_VOP2_TERNARY; the
    dst-accumulate forms (literal ``"0u"``: v_fmac/v_mac, and v_fmac_f64) are
    portable.

    A second non-portable family is VOP3P: the shared VOP3P execute template
    reads the op_sel field by its canonical member name (``op_sel`` /
    ``op_sel_hi``), but RDNA4/gfx1250 rename it to ``opsel`` / ``opsel_hi`` in
    their ``Vop3pMachineInst`` struct.  An ISA that renames the field cannot
    compile against the canonical-named shared body, so it must NOT be
    force-routed through it — it falls back to an inline body generated with
    its own (renamed) field accessors. Supported families may add a
    profile-aware local SIMD probe to that body. ``vop3p_opsel_fields`` carries
    the calling ISA's profile field names; when they differ from the canonical
    pair, VOP3P probes are not portable for that ISA.
    """
    if simd_probe_line(template_name) is None:
        return False
    spect = SIMD_VOP2_TERNARY.get(template_name)
    if spect is not None and spect[1] != '0u':
        return False
    if template_name.endswith('_vop3p') and vop3p_opsel_fields != (
        'op_sel',
        'op_sel_hi',
    ):
        return False
    return True


def simd_extra_includes() -> list[str]:
    """Extra `#include` lines required by the SIMD probe call sites.

    The helper templates live in ``simd_glue.h``, which pulls in
    ``util/simd.h`` transitively (for ``util::has_stdx_simd``), so this is
    the only SIMD-specific include the generated shared header needs.
    """
    return ['#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"']


def local_coverage_probe(
    template_name: str,
    *,
    true16_vop3: bool = False,
    e32: bool = False,
    cmpx_writes_vcc: bool = False,
    result_writer: str | None = None,
    e32_half_dst: bool = True,
    e32_half_inputs: int = 1,
    arch_name: str | None = None,
) -> str | None:
    probe = _local_coverage_probe(
        template_name,
        true16_vop3=true16_vop3,
        e32=e32,
        cmpx_writes_vcc=cmpx_writes_vcc,
        result_writer=result_writer,
        e32_half_dst=e32_half_dst,
        e32_half_inputs=e32_half_inputs,
        arch_name=arch_name,
    )
    return _guard_mode_arithmetic_probe(template_name, probe)


def _local_coverage_probe(
    template_name: str,
    *,
    true16_vop3: bool = False,
    e32: bool = False,
    cmpx_writes_vcc: bool = False,
    result_writer: str | None = None,
    e32_half_dst: bool = True,
    e32_half_inputs: int = 1,
    arch_name: str | None = None,
) -> str | None:
    """Emit local probes without conflating profile-specific result semantics.

    ``arch_name`` selects the target's FP8/BF8 decoder through
    fp8_formats.fp8_policy.
    """

    def call(arity: int, half_dst: bool, half_inputs: int, functor: str) -> str:
        functor = functor.replace('float32_t', 'float')
        args = f'{arity}, {str(e32).lower()}, {str(half_dst).lower()}, {half_inputs}'
        # Callbacks on a shared core need the probe to install its environment
        # once for the instruction, as the shared probes do.
        recipe = arithmetic_core(functor)
        if recipe is not None:
            return (
                f'  if (amdgpu::try_execute_words_simd<{args}, true>('
                f'inst, wf, {functor}, {recipe.round_mode})) return;'
            )
        return f'  if (amdgpu::try_execute_words_simd<{args}>(inst, wf, {functor})) return;'

    if template_name.startswith('v_cmpx_'):
        ordinary = template_name.replace('v_cmpx_', 'v_cmp_', 1)
        writer = 'commit_result' if result_writer else 'wf.set_exec'
        prefix = ''
        if cmpx_writes_vcc:
            prefix = (
                'amdgpu::write_wave_mask_scalar(inst.vdst, wf, value); '
                if template_name.endswith('_vop3')
                else 'wf.set_vcc_mask(value); '
            )
        commit = f'[&](uint64_t value) {{ {prefix}{writer}(value); }}'
        if template_name.endswith('_vopc'):
            for table, helper in (
                (SIMD_VOPC, 'try_execute_vopc_simd'),
                (SIMD_VOPC64, 'try_execute_vopc64_simd'),
            ):
                if ordinary in table:
                    t, op = table[ordinary]
                    t = t.replace('float32_t', 'float')
                    return f'  if (amdgpu::{helper}<{t}>(inst, wf, {op}, {commit})) return;'
        return local_coverage_probe(
            ordinary, true16_vop3=true16_vop3, result_writer=commit
        ) or simd_probe_line(ordinary, true16_vop3=true16_vop3, result_writer=commit)

    if true16_vop3 and template_name in SIMD_VOPC_VOP3_INT_32:
        t, op = SIMD_VOPC_VOP3_INT_32[template_name]
        commit = (
            result_writer
            or '[&](uint64_t v) { amdgpu::write_wave_mask_scalar(inst.vdst, wf, v); }'
        )
        return f'  if (amdgpu::try_execute_vopc_vop3_int_simd<{t}, true>(inst, wf, {op}, {commit})) return;'

    if template_name in (
        'v_cvt_pk_f16_f32_vop3',
        'v_cvt_pk_bf16_f32_vop3',
    ):
        if 'bf16' in template_name:
            convert = 'util::pack_bf16_simd({x}, wf.fp16_ovfl())'
        else:
            convert = 'util::f32_to_f16_mode_simd(std::bit_cast<util::native<float>>({x}), wf.fp16_ovfl())'
        src0, src1 = 'a', 'b'
        if arch_name == 'cdna4':
            apply = 'amdgpu::source_modifier::apply<amdgpu::fp_format::F32>'
            src0 = f'{apply}(a, 0, inst.inst_.abs, inst.inst_.neg)'
            src1 = f'{apply}(b, 1, inst.inst_.abs, inst.inst_.neg)'
        low = convert.format(x=src0)
        high = convert.format(x=src1)
        packed = (
            f'amdgpu::fp_mode::pack_f16_results({low}, {high}, wf.fp_denorm_mode_f16_f64())'
            if arch_name == 'cdna5' and template_name == 'v_cvt_pk_f16_f32_vop3'
            else f'{low} | ({high} << 16)'
        )
        functor = f'[&](auto a, auto b) {{ return {packed}; }}'
        return call(2, False, 0, functor)
    if template_name == 'v_cvt_f32_bf16_vop1':
        return call(
            1,
            False,
            e32_half_inputs if e32 else 0,
            '[](auto a) { return std::bit_cast<util::native<uint32_t>>(util::bf16_to_f32_simd(a)); }',
        )
    if (e32 or true16_vop3) and conversion.needs_half_glue(template_name):
        return conversion.simd_probe(
            template_name,
            true16_vop3=true16_vop3,
            e32=e32,
            e32_half_inputs=e32_half_inputs,
        )
    if template_name in ('v_cvt_f32_fp8_vop3', 'v_cvt_f32_bf8_vop3'):
        if not uses_shared_fp8_widening(arch_name):
            # util's decoders, which the target's scalar body uses.
            convert = (
                'util::bf8_e5m2_to_f32_simd(byte)'
                if 'bf8' in template_name
                else '(amdgpu::vop3_fp8_decode_e5m3(inst) ? util::fp8_e5m3_to_f32_simd(byte) : util::fp8_e4m3_to_f32_simd(byte))'
            )
            decoded = f'std::bit_cast<util::native<uint32_t>>({convert})'
        else:
            decoded = conversion.decode_fp8_expr(template_name[10:13], 'byte')
            if 'fp8' in template_name:
                decoded = (
                    '(amdgpu::vop3_fp8_decode_e5m3(inst) ? '
                    'std::bit_cast<util::native<uint32_t>>(util::fp8_e5m3_to_f32_simd(byte)) : '
                    f'{decoded})'
                )
        return call(
            1,
            False,
            0,
            '[&](auto a) { const uint32_t sel = amdgpu::vop3_opsel(inst.inst_); '
            'auto byte = (a >> ((((sel & 1u) << 1) | ((sel & 2u) >> 1)) * 8u)) & 0xffu; '
            f'return {decoded}; }}',
        )
    if template_name in ('v_cvt_f32_bf16_vop3',):
        # VOP3 conversion applies source half selection and floating modifiers.
        return None
    if template_name in ('v_bitop3_b32_vop3', 'v_bitop3_b16_vop3'):
        half = 'b16' in template_name
        mask = ' & 0xffffu' if half else ''
        return call(
            3,
            half and true16_vop3,
            7 if half and true16_vop3 else 0,
            '[&](auto a, auto b, auto c) { return amdgpu::bitop3_words(a, b, c, '
            'static_cast<uint8_t>((inst.inst_.omod << 6) | (inst.inst_.abs << 3) | inst.inst_.neg))'
            + mask
            + '; }',
        )
    if true16_vop3:
        binary = {
            'v_and_b16_vop3': 'a & b',
            'v_or_b16_vop3': 'a | b',
            'v_xor_b16_vop3': 'a ^ b',
            'v_add_i16_vop3': 'a + b',
            'v_add_u16_vop3': 'a + b',
            'v_sub_i16_vop3': 'a - b',
            'v_sub_u16_vop3': 'a - b',
        }
        if template_name in binary:
            # Integer CLAMP retains its exact scalar implementation.
            return (
                '  if (!inst.inst_.clamp) {\n'
                + call(
                    2,
                    True,
                    3,
                    f'[](auto a, auto b) {{ return {binary[template_name]}; }}',
                )
                + '\n  }'
            )
        if template_name in ('v_not_b16_vop3', 'v_mov_b16_vop3'):
            expression = '~a' if 'not' in template_name else 'a'
            return call(1, True, 1, f'[](auto a) {{ return {expression}; }}')
    if e32:
        if template_name in SIMD_VOP1_UNARY:
            tin, tout, op = SIMD_VOP1_UNARY[template_name]
            if _probe_uses_fp16_ovfl_f16_result_narrow(op):
                op = f'[&](auto a) {{ if (wf.fp16_ovfl()) return ({_fp16_ovfl_cpp_op(op)})(a); return ({op})(a); }}'
            return call(
                1,
                e32_half_dst,
                e32_half_inputs,
                f'[&](auto a) {{ return std::bit_cast<util::native<uint32_t>>(({op})(std::bit_cast<util::native<{tin}>>(a))); }}',
            )
        if template_name in SIMD_VOP2_BINARY:
            tin, op = SIMD_VOP2_BINARY[template_name]
            if _probe_uses_fp16_ovfl_f16_result_narrow(op):
                op = f'[&](auto a, auto b) {{ if (wf.fp16_ovfl()) return ({_fp16_ovfl_cpp_op(op)})(a, b); return ({op})(a, b); }}'
            return call(
                2,
                e32_half_dst,
                e32_half_inputs,
                f'[&](auto a, auto b) {{ return std::bit_cast<util::native<uint32_t>>(({op})(std::bit_cast<util::native<{tin}>>(a), std::bit_cast<util::native<{tin}>>(b))); }}',
            )
    return None
