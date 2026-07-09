# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SIMD specialization codegen for AMDGPU VOP2 execute kernels.

Emits an `<experimental/simd>`-based fast path on top of the generated
scalar per-lane bodies. The scalar body is preserved verbatim as a
fallback; the SIMD probe is a single line at the start of the kernel:

    ROCJITSU_TRY_SIMD_VOP2_BINARY(T, op_functor);

That macro (defined in ``simd_glue.h``) expands to a call to
``try_execute_binary_vop2_simd<T>`` plus an early ``return`` on success.

`try_execute_binary_vop2_simd` in ``simd_glue.h`` is a constrained
template (``requires(util::has_stdx_simd)``) plus an unconstrained
fallback that returns ``false``. On toolchains without
``<experimental/simd>``, overload resolution picks the fallback and the
compiler inlines the probe to a dead branch.

Eligible kernels are listed in :data:`SIMD_VOP2_BINARY` — only those
whose host SIMD result is bit-identical to the scalar generated body
(IEEE-754 single-rounded fp arithmetic, wrap-around integer arithmetic,
elementwise bitwise ops). NaN-sensitive ops (min/max), VCC-writing ops
(add_co), and modifier-bearing forms (VOP3 with abs/neg/clamp/omod) are
excluded — those need their own helpers.
"""

from __future__ import annotations

# template_name -> (cpp_element_type, cpp_binary_op_functor)
#
# template_name matches the symbol emitted by _generator.gen_shared_execute:
#   f"{inst.mnemonic}_{enc_key}"  (e.g. "v_add_f32_vop2").
#
# The functor is invoked as `bin_op(simd<T>, simd<T>) -> simd<T>` inside
# try_execute_binary_vop2_simd. Use std::*<> for stateless ops.
SIMD_VOP2_BINARY: dict[str, tuple[str, str]] = {
    # --- float32 (IEEE-754 single-rounded, bit-identical to scalar body) ---
    "v_add_f32_vop2": ("float32_t", "std::plus<>{}"),
    'v_sub_f32_vop2': ('float32_t', 'std::minus<>{}'),
    'v_subrev_f32_vop2': ('float32_t', '[](auto a, auto b) { return b - a; }'),
    'v_mul_f32_vop2': ('float32_t', 'std::multiplies<>{}'),
    # Legacy / DX9 zero-multiply: (a==0 || b==0) ? 0 : a*b. The ==0 matches both
    # ±0 (as the scalar `a == 0.0f` does). Routed via the VOP3 binary fp glue for
    # the _vop3 twin (which applies abs/neg/omod/clamp around this functor).
    'v_mul_legacy_f32_vop2': (
        'float32_t',
        '[](auto a, auto b) {'
        ' auto r = a * b;'
        ' util::stdx::where(a == 0.0f || b == 0.0f, r) = util::native<float32_t>(0.0f);'
        ' return r; }',
    ),
    'v_mul_dx9_zero_f32_vop2': (
        'float32_t',
        '[](auto a, auto b) {'
        ' auto r = a * b;'
        ' util::stdx::where(a == 0.0f || b == 0.0f, r) = util::native<float32_t>(0.0f);'
        ' return r; }',
    ),
    # --- uint32 (wrap-around / bitwise, bit-identical to scalar body) ---
    "v_add_u32_vop2": ("uint32_t", "std::plus<>{}"),
    'v_sub_u32_vop2': ('uint32_t', 'std::minus<>{}'),
    'v_subrev_u32_vop2': ('uint32_t', '[](auto a, auto b) { return b - a; }'),
    # RDNA "no-carry" int add/sub — bit-identical to add_u32/sub_u32/subrev_u32
    # (no VCC interaction), just renamed.
    'v_add_nc_u32_vop2': ('uint32_t', 'std::plus<>{}'),
    'v_sub_nc_u32_vop2': ('uint32_t', 'std::minus<>{}'),
    'v_subrev_nc_u32_vop2': ('uint32_t', '[](auto a, auto b) { return b - a; }'),
    'v_and_b32_vop2': ('uint32_t', 'std::bit_and<>{}'),
    'v_or_b32_vop2': ('uint32_t', 'std::bit_or<>{}'),
    'v_xor_b32_vop2': ('uint32_t', 'std::bit_xor<>{}'),
    'v_xnor_b32_vop2': ('uint32_t', '[](auto a, auto b) { return ~(a ^ b); }'),
    # rev: shift value is vsrc1 (b), shift count is src0 (a), masked to 5 bits.
    'v_lshlrev_b32_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return ::rocjitsu::amdgpu::simd_lshl_u32(b, a); }',
    ),
    'v_lshrrev_b32_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return ::rocjitsu::amdgpu::simd_lshr_u32(b, a); }',
    ),
    'v_mul_u32_u24_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return (a & 0x00FFFFFFu) * (b & 0x00FFFFFFu); }',
    ),
    # Signed 24-bit multiply, low 32 bits. Sign-extend the low 24 bits to int32
    # (matching the scalar (int32_t)(x<<8)>>8); the int32 product's low 32 bits
    # are exact, so no widening is needed.
    'v_mul_i32_i24_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return ::rocjitsu::amdgpu::simd_mul_i24_u32(a, b); }',
    ),
    # High 32 bits of the 24-bit multiply (48-bit product). The 32x32->high32
    # step uses util::mul_hi_{u,i}32_simd, a 16x16 partial-product decomposition
    # in pure native<uint32_t> arithmetic. It deliberately avoids
    # fixed_size_simd<{u,i}64, N>: clang + libstdc++ miscompile the 64-bit-lane
    # multiply/shift of an over-native-width fixed_size_simd, so the int64
    # widening diverges from the scalar uint64/int64 intermediates at every SIMD
    # width. The decomposition is bit-identical to the scalar reference.
    'v_mul_hi_u32_u24_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' return util::mul_hi_u32_simd(a & 0x00FFFFFFu, b & 0x00FFFFFFu); }',
    ),
    'v_mul_hi_i32_i24_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto sa = (util::stdx::static_simd_cast<util::native<int32_t>>(a) << 8) >> 8;'
        ' auto sb = (util::stdx::static_simd_cast<util::native<int32_t>>(b) << 8) >> 8;'
        ' return util::mul_hi_i32_simd('
        ' util::stdx::static_simd_cast<util::native<uint32_t>>(sa),'
        ' util::stdx::static_simd_cast<util::native<uint32_t>>(sb)); }',
    ),
    'v_max_u32_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return util::stdx::max(a, b); }',
    ),
    'v_min_u32_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return util::stdx::min(a, b); }',
    ),
    # --- int32 (signed: arithmetic shift / signed min-max) ---
    'v_ashrrev_i32_vop2': ('int32_t', '[](auto a, auto b) { return b >> (a & 31); }'),
    'v_max_i32_vop2': (
        'int32_t',
        '[](auto a, auto b) { return util::stdx::max(a, b); }',
    ),
    'v_min_i32_vop2': (
        'int32_t',
        '[](auto a, auto b) { return util::stdx::min(a, b); }',
    ),
    # --- 16-bit integer (low 16 bits, result zero-extended to 32). Lane type
    # stays uint32_t; the functor masks/sign-extends the low 16 bits and writes
    # back the zero-extended 16-bit result, matching write_lane semantics. ---
    'v_add_u16_vop2': ('uint32_t', '[](auto a, auto b) { return (a + b) & 0xFFFFu; }'),
    'v_sub_u16_vop2': ('uint32_t', '[](auto a, auto b) { return (a - b) & 0xFFFFu; }'),
    'v_subrev_u16_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return (b - a) & 0xFFFFu; }',
    ),
    'v_mul_lo_u16_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return ((a & 0xFFFFu) * (b & 0xFFFFu)) & 0xFFFFu; }',
    ),
    # rev: shift value is vsrc1 (b), count is src0 (a) masked to 4 bits.
    'v_lshlrev_b16_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return (b << (a & 15u)) & 0xFFFFu; }',
    ),
    'v_lshrrev_b16_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return (b & 0xFFFFu) >> (a & 15u); }',
    ),
    # ashr: sign-extend the low 16 bits to int32, arithmetic-shift by count & 15
    # (the scalar masks the i16 count to 4 bits), then take the low 16 bits.
    'v_ashrrev_i16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto sb = (util::stdx::static_simd_cast<util::native<int32_t>>(b) << 16) >> 16;'
        ' auto sa = (util::stdx::static_simd_cast<util::native<int32_t>>(a) << 16) >> 16;'
        ' return util::stdx::static_simd_cast<util::native<uint32_t>>(sb >> (sa & 15)) & 0xFFFFu; }',
    ),
    'v_max_i16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto sa = (util::stdx::static_simd_cast<util::native<int32_t>>(a) << 16) >> 16;'
        ' auto sb = (util::stdx::static_simd_cast<util::native<int32_t>>(b) << 16) >> 16;'
        ' return util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::max(sa, sb))'
        ' & 0xFFFFu; }',
    ),
    'v_min_i16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto sa = (util::stdx::static_simd_cast<util::native<int32_t>>(a) << 16) >> 16;'
        ' auto sb = (util::stdx::static_simd_cast<util::native<int32_t>>(b) << 16) >> 16;'
        ' return util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::min(sa, sb))'
        ' & 0xFFFFu; }',
    ),
    'v_max_u16_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return util::stdx::max(a & 0xFFFFu, b & 0xFFFFu); }',
    ),
    'v_min_u16_vop2': (
        'uint32_t',
        '[](auto a, auto b) { return util::stdx::min(a & 0xFFFFu, b & 0xFFFFu); }',
    ),
    # --- float min/max. util::stdx::fmax/fmin match the scalar std::fmax/fmin
    # used by the generated bodies for all finite/Inf inputs. Two accepted
    # divergences (per project direction): (1) NaN inputs may differ in NaN
    # payload; (2) a signed-zero tie returns the opposite-signed zero — scalar
    # std::fmax/fmin returns the first operand, the packed vmaxps/vminps the
    # second, so e.g. fmax(-0,+0) is -0 (scalar) vs +0 (SIMD). Both are
    # numerically equal results; the guard tests skip NaN-input and zero-tie
    # lanes (UtilSimd.Fmax/Fmin_VectorMatchesScalar_BitExact). All other inputs
    # are bit-exact. (The earlier "not reproducible" note conflated these two
    # accepted corners with a hard blocker.) ---
    'v_max_f32_vop2': (
        'float32_t',
        '[](auto a, auto b) { return util::stdx::fmax(a, b); }',
    ),
    'v_min_f32_vop2': (
        'float32_t',
        '[](auto a, auto b) { return util::stdx::fmin(a, b); }',
    ),
    'v_max_f16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' return util::f32_to_f16_simd('
        'util::stdx::fmax(util::f16_to_f32_simd(a), util::f16_to_f32_simd(b))); }',
    ),
    'v_min_f16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' return util::f32_to_f16_simd('
        'util::stdx::fmin(util::f16_to_f32_simd(a), util::f16_to_f32_simd(b))); }',
    ),
    # IEEE-2019 maximumNumber/minimumNumber (gfx1250/rdna4). Their generated
    # scalar bodies are std::fmax / std::fmin — byte-for-byte the legacy
    # v_max_f*/v_min_f* bodies above — so the functors are identical and inherit
    # the same accepted NaN-payload / signed-zero-tie carve-out. The _vop3 twins
    # auto-route from these _vop2 entries (f32 -> VOP3_BINARY_FP, f16 ->
    # VOP3_BINARY_INT with the widening functor, mirroring v_max_f16_vop3).
    'v_max_num_f32_vop2': (
        'float32_t',
        '[](auto a, auto b) { return util::stdx::fmax(a, b); }',
    ),
    'v_min_num_f32_vop2': (
        'float32_t',
        '[](auto a, auto b) { return util::stdx::fmin(a, b); }',
    ),
    'v_max_num_f16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' return util::f32_to_f16_simd('
        'util::stdx::fmax(util::f16_to_f32_simd(a), util::f16_to_f32_simd(b))); }',
    ),
    'v_min_num_f16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' return util::f32_to_f16_simd('
        'util::stdx::fmin(util::f16_to_f32_simd(a), util::f16_to_f32_simd(b))); }',
    ),
    # v_cvt_pkrtz_f16_f32 (both spellings): pack two f32 -> two f16 (round toward
    # zero is what f32_to_f16/f32_to_f16_simd implement; proven bit-identical).
    # Inputs arrive as raw u32 lanes, bit_cast to f32. The VOP3 twins carry no
    # modifiers (verified) so they auto-route to VOP3_BINARY_INT with this functor.
    'v_cvt_pkrtz_f16_f32_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto lo = util::f32_to_f16_simd(std::bit_cast<util::native<float>>(a));'
        ' auto hi = util::f32_to_f16_simd(std::bit_cast<util::native<float>>(b));'
        ' return lo | (hi << 16); }',
    ),
    'v_cvt_pk_rtz_f16_f32_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto lo = util::f32_to_f16_simd(std::bit_cast<util::native<float>>(a));'
        ' auto hi = util::f32_to_f16_simd(std::bit_cast<util::native<float>>(b));'
        ' return lo | (hi << 16); }',
    ),
    # --- f16 binary (low 16 bits f16, result zero-extended). Same f32
    # intermediate as the scalar bodies (single final round) ⇒ bit-identical. ---
    'v_add_f16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' return util::f32_to_f16_simd(util::f16_to_f32_simd(a) + util::f16_to_f32_simd(b)); }',
    ),
    'v_sub_f16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' return util::f32_to_f16_simd(util::f16_to_f32_simd(a) - util::f16_to_f32_simd(b)); }',
    ),
    'v_subrev_f16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' return util::f32_to_f16_simd(util::f16_to_f32_simd(b) - util::f16_to_f32_simd(a)); }',
    ),
    'v_mul_f16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' return util::f32_to_f16_simd(util::f16_to_f32_simd(a) * util::f16_to_f32_simd(b)); }',
    ),
    # v_ldexp_f16: dst = f16(ldexp(f16->f32(src0), (int16)vsrc1)). src0 is an f16
    # multiplicand, vsrc1 a signed-16-bit exponent (widened to a fixed_size int
    # lane). std::ldexp is a power-of-2 scale with a single correctly-rounded
    # result; util::stdx::ldexp matches it bit-for-bit (verified full-range incl
    # NaN/Inf/denormal), and the f16<->f32 conversions are bit-exact, so the
    # composition matches the scalar body. No NaN-operand ambiguity (unlike fma).
    'v_ldexp_f16_vop2': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto x = util::f16_to_f32_simd(a);'
        ' auto n = util::stdx::static_simd_cast<'
        'util::stdx::fixed_size_simd<int, util::native<float>::size()>>('
        '(util::stdx::static_simd_cast<util::native<int32_t>>(b) << 16) >> 16);'
        ' return util::f32_to_f16_simd(util::stdx::ldexp(x, n)); }',
    ),
}

# template_name -> (cpp_in_type, cpp_out_type, cpp_unary_op_functor)
#
# Same keying as SIMD_VOP2_BINARY. The functor is invoked as
#   un_op(simd<Tin>) -> simd<Tout>
# inside try_execute_unary_vop1_simd. Tin and Tout are both 32-bit lane
# types and may differ (e.g. int32->float32 for v_cvt_f32_i32). Eligible
# kernels are those whose host SIMD result is bit-identical to the scalar
# generated body: elementwise bit ops, exact int<->float casts, and the
# correctly-rounded IEEE operations (div, sqrt, and — verified per toolchain
# via the parity test — exp2/log2). NaN/clamp-bearing conversions and the
# inexact transcendentals (sin/cos) are excluded.
# VOP1 base mnemonics whose VOP3 form applies float abs/neg/omod/clamp modifiers
# over an f32 source and result (so the VOP3 twin routes through the f32 unary
# modifier glue rather than reusing the plain VOP1 path).
# NOTE: v_mov_b32 / v_accvgpr_mov_b32 are deliberately NOT here. They are integer
# bit-moves: OMOD/CLAMP are float-only modifiers, so their generated VOP3 scalar
# body is a raw copy that ignores them. Routing them through the f32 modifier glue
# made the SIMD path apply clamp/omod the scalar never does (clamp=1: scalar keeps
# the raw bits vs simd clamps to 1.0). They fall through to the plain VOP1 unary
# raw-copy path below, matching the scalar body for every modifier combination.
_VOP3_UNARY_FP_F32 = {
    'v_floor_f32',
    'v_ceil_f32',
    'v_trunc_f32',
    'v_rndne_f32',
    'v_fract_f32',
    'v_rcp_f32',
    'v_rcp_iflag_f32',
    'v_rsq_f32',
    'v_sqrt_f32',
    'v_exp_f32',
    'v_log_f32',
    'v_frexp_mant_f32',
}

# VOP1 base mnemonics whose VOP3 twin stays scalar: the f16 rounding/transcendental
# forms carry modifiers applied around an f16<->f32 round trip (not yet handled).
# v_mov_b16 is here for the same reason — its VOP3 body reads src0 as a u16 value,
# widens it to f32, applies omod/clamp, then narrows back to u16 (a 16-bit->f32->16-bit
# modifier round trip). The plain `a & 0xFFFFu` VOP1 functor it would otherwise reuse
# ignores those modifiers, so with clamp/omod set the SIMD path diverged from scalar
# (clamp=1: scalar 0x1 vs simd 0xffff). The f32 FP8/BF8 VOP3 decoders use op_sel[1:0]
# as a bit-swapped byte selector, while the VOP1 SIMD functor always consumes byte 0. Leaving these
# VOP3 twins scalar keeps both modifier and byte-select behavior correct; the
# modifier-free / byte0 VOP1 forms still take the fast path.
_VOP3_UNARY_SKIP = {
    'v_floor_f16',
    'v_ceil_f16',
    'v_trunc_f16',
    'v_rndne_f16',
    'v_fract_f16',
    'v_rcp_f16',
    'v_rsq_f16',
    'v_sqrt_f16',
    'v_exp_f16',
    'v_log_f16',
    'v_mov_b16',
    'v_frexp_exp_i32_f32',
    'v_frexp_exp_i32_f64',
    'v_frexp_mant_f16',
    'v_frexp_exp_i16_f16',
    'v_cvt_f32_fp8',
    'v_cvt_f32_bf8',
}

SIMD_VOP1_UNARY: dict[str, tuple[str, str, str]] = {
    # --- bitwise / move (uint32, bit-identical) ---
    'v_mov_b32_vop1': ('uint32_t', 'uint32_t', '[](auto a) { return a; }'),
    'v_accvgpr_mov_b32_vop1': ('uint32_t', 'uint32_t', '[](auto a) { return a; }'),
    'v_not_b32_vop1': ('uint32_t', 'uint32_t', '[](auto a) { return ~a; }'),
    # RDNA3+ 16-bit move / not / int16<->int32 conversions (low-16, zero-extend
    # to the 32-bit VGPR). The _vop3 twins auto-route through the same VOP1 path,
    # except v_mov_b16 whose VOP3 form applies float omod/clamp (see
    # _VOP3_UNARY_SKIP above).
    'v_mov_b16_vop1': ('uint32_t', 'uint32_t', '[](auto a) { return a & 0xFFFFu; }'),
    'v_not_b16_vop1': ('uint32_t', 'uint32_t', '[](auto a) { return (~a) & 0xFFFFu; }'),
    'v_cvt_i32_i16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto x = util::stdx::static_simd_cast<util::native<int32_t>>(a & 0xFFFFu);'
        ' return util::stdx::static_simd_cast<util::native<uint32_t>>((x << 16) >> 16); }',
    ),
    'v_cvt_u32_u16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) { return a & 0xFFFFu; }',
    ),
    # RDNA f32->i32 round-toward-floor / round-to-nearest-even conversions
    # (cdna4 spells these v_cvt_flr_i32_f32 / v_cvt_rpi_i32_f32). floor via
    # stdx::floor; nearest via ceil(s - 0.5) (round-half-to-even on the .5 path).
    # Out-of-range saturates to INT32_MIN/MAX and NaN -> 0, matching the scalar.
    'v_cvt_floor_i32_f32_vop1': (
        'float32_t',
        'int32_t',
        '[](auto s) {'
        ' auto r = util::stdx::floor(s);'
        ' return ::rocjitsu::amdgpu::simd_cvt_i32_f32(r); }',
    ),
    'v_cvt_nearest_i32_f32_vop1': (
        'float32_t',
        'int32_t',
        '[](auto s) {'
        ' auto r = util::stdx::ceil(s - util::native<float32_t>(0.5f));'
        ' return ::rocjitsu::amdgpu::simd_cvt_i32_f32(r); }',
    ),
    # --- bit-scan (SWAR, no stdx primitive) -----------------------------------
    # All return uint32_t. Most special-case the zero input to 0xFFFFFFFF,
    # matching the scalar bodies (std::countl_zero / countr_zero / popcount);
    # cls_i32 instead maps the all-zero/all-one case to 31. The VOP3
    # twins share these (modifier-free integer bodies, auto-routed below). f16<->
    # f32 round-trip not involved -> not in _VOP3_UNARY_SKIP.
    #
    # ffbh_u32 / clz_i32_u32: count leading zeros; 0 -> 0xFFFFFFFF.
    'v_ffbh_u32_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto c = util::clz_u32_simd(a);'
        ' util::stdx::where(a == 0u, c) = 0xFFFFFFFFu;'
        ' return c; }',
    ),
    'v_clz_i32_u32_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto c = util::clz_u32_simd(a);'
        ' util::stdx::where(a == 0u, c) = 0xFFFFFFFFu;'
        ' return c; }',
    ),
    # ffbl_b32 / ctz_i32_b32: count trailing zeros; 0 -> 0xFFFFFFFF.
    'v_ffbl_b32_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto c = util::ctz_u32_simd(a);'
        ' util::stdx::where(a == 0u, c) = 0xFFFFFFFFu;'
        ' return c; }',
    ),
    'v_ctz_i32_b32_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto c = util::ctz_u32_simd(a);'
        ' util::stdx::where(a == 0u, c) = 0xFFFFFFFFu;'
        ' return c; }',
    ),
    # ffbh_i32: count leading sign bits = clz of (s < 0 ? ~s : s);
    # 0 (== all-zero or all-one source) -> 0xFFFFFFFF.
    'v_ffbh_i32_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' util::native<uint32_t> u = a;'
        ' util::stdx::where((a & 0x80000000u) != 0u, u) = ~a;'
        ' auto c = util::clz_u32_simd(u);'
        ' util::stdx::where(u == 0u, c) = 0xFFFFFFFFu;'
        ' return c; }',
    ),
    'v_cls_i32_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' util::native<uint32_t> u = a;'
        ' util::stdx::where((a & 0x80000000u) != 0u, u) = ~a;'
        ' auto c = util::clz_u32_simd(u) - 1u;'
        ' util::stdx::where(u == 0u, c) = 31u;'
        ' return c; }',
    ),
    # v_bfrev_b32: reverse the 32 bits of src0. The scalar body loops bit-by-bit;
    # this is the branchless swap-by-strides equivalent (1/2/4/8/16-bit groups),
    # bit-identical for every input. Pure uint32 bitwise ops.
    'v_bfrev_b32_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto x = a;'
        ' x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);'
        ' x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);'
        ' x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);'
        ' x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);'
        ' return (x << 16) | (x >> 16); }',
    ),
    # --- frexp (f32 split into mantissa / exponent) ----------------------------
    # frexp_mant returns the significand m with |m| in [0.5,1) (±0/Inf/NaN pass
    # through); read as float so the VOP3 twin reuses the f32 unary FP glue for
    # abs/neg/omod/clamp (added to _VOP3_UNARY_FP_F32 below). frexp_exp returns
    # the raw int32 exponent bits; its VOP3 twin applies float omod/clamp to
    # float(exp) and bit-casts (a different output encoding than the VOP1 int),
    # so the VOP3 form stays scalar (_VOP3_UNARY_SKIP).
    'v_frexp_mant_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::frexp_mant_f32_simd(std::bit_cast<util::native<uint32_t>>(a)); }',
    ),
    'v_frexp_exp_i32_f32_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) { return util::frexp_exp_f32_simd(a); }',
    ),
    # frexp f16: the scalar widens src to f32, runs std::frexp, then narrows the
    # mantissa (or float(exp)) back to f16 via f32_to_f16. Compose the bit-exact
    # f16<->f32 ports with the f32 frexp helpers; the f16<->f32 round trips and
    # frexp are each bit-identical, so the composition matches. The VOP3 twins
    # carry omod/clamp around that round trip (not reproduced) -> _VOP3_UNARY_SKIP.
    'v_frexp_mant_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' return util::f32_to_f16_simd(util::frexp_mant_f32_simd('
        'std::bit_cast<util::native<uint32_t>>(util::f16_to_f32_simd(a)))); }',
    ),
    'v_frexp_exp_i16_f16_vop1': (
        'uint32_t',
        'uint32_t',
        # The scalar narrows `static_cast<uint32_t>(exp)` to f16 -- the int->float
        # conversion is UNSIGNED, so a negative exponent (0xFFFF_FFxx) becomes a
        # ~4.29e9 float and f32_to_f16 saturates it to +Inf (0x7C00). Mirror that
        # with an unsigned static_simd_cast (no signed intermediate).
        '[](auto a) {'
        ' auto e = util::frexp_exp_f32_simd('
        'std::bit_cast<util::native<uint32_t>>(util::f16_to_f32_simd(a)));'
        ' return util::f32_to_f16_simd(util::stdx::static_simd_cast<util::native<float>>(e)); }',
    ),
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
    # --- ubyte -> f32 (extract byte N, exact int->float) -----------------------
    # dst = float(byte_N(src0)); byte value is 0..255 so the int->float cast is
    # exact and bit-identical to the scalar static_cast<float>.
    'v_cvt_f32_ubyte0_vop1': (
        'uint32_t',
        'float32_t',
        '[](auto a) { return util::stdx::static_simd_cast<util::native<float32_t>>(a & 0xFFu); }',
    ),
    'v_cvt_f32_ubyte1_vop1': (
        'uint32_t',
        'float32_t',
        '[](auto a) {'
        ' return util::stdx::static_simd_cast<util::native<float32_t>>((a >> 8) & 0xFFu); }',
    ),
    'v_cvt_f32_ubyte2_vop1': (
        'uint32_t',
        'float32_t',
        '[](auto a) {'
        ' return util::stdx::static_simd_cast<util::native<float32_t>>((a >> 16) & 0xFFu); }',
    ),
    'v_cvt_f32_ubyte3_vop1': (
        'uint32_t',
        'float32_t',
        '[](auto a) {'
        ' return util::stdx::static_simd_cast<util::native<float32_t>>((a >> 24) & 0xFFu); }',
    ),
    # --- int<->float casts (single-rounded, bit-identical) ---
    'v_cvt_f32_i32_vop1': (
        'int32_t',
        'float32_t',
        '[](auto a) { return util::stdx::static_simd_cast<util::native<float32_t>>(a); }',
    ),
    'v_cvt_f32_u32_vop1': (
        'uint32_t',
        'float32_t',
        '[](auto a) { return util::stdx::static_simd_cast<util::native<float32_t>>(a); }',
    ),
    # --- float rounding (bit-identical to std::* on host) ---
    'v_floor_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::floor_simd(a); }',
    ),
    'v_ceil_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::ceil_simd(a); }',
    ),
    'v_trunc_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::trunc_simd(a); }',
    ),
    'v_rndne_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::rndne_simd(a); }',
    ),
    'v_fract_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return a - util::floor_simd(a); }',
    ),
    # --- transcendental div / sqrt. These mirror amdgpu::transcendental::*_f32
    # exactly via util::*_f32_simd (FTZ input/output flush + canonical-qNaN and
    # NaN-input-preservation blends), so the SIMD result is bit-identical to the
    # forced-scalar body on every input incl NaN/Inf/denormal. ---
    'v_rcp_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::rcp_f32_simd(a); }',
    ),
    'v_rcp_iflag_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::rcp_f32_simd(a); }',
    ),
    'v_rsq_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::rsq_f32_simd(a); }',
    ),
    'v_sqrt_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::sqrt_f32_simd(a); }',
    ),
    # --- transcendental exp2/log2. util::{exp,log}_f32_simd wrap stdx::exp2/log2
    # with the same FTZ flush / special-case guards as the scalar transcendental
    # reference; the underlying vector libm is bit-exact to scalar std::* on the
    # supported toolchains (libstdc++ 13 / AVX-512), guarded by
    # UtilSimd.Exp2/Log2_*_BitExact. v_sin/v_cos excluded: vector libm ~1 ULP off. ---
    'v_exp_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::exp_f32_simd(a); }',
    ),
    'v_log_f32_vop1': (
        'float32_t',
        'float32_t',
        '[](auto a) { return util::log_f32_simd(a); }',
    ),
    # --- float -> int conversions with NaN->0 and saturating clamp. The float
    # comparison masks are re-typed to the int lane via simd_mask_as<> before
    # the where-blend. Masks are mutually exclusive so application order is
    # irrelevant. Matches the scalar bodies' truncate-toward-zero cast. ---
    'v_cvt_i32_f32_vop1': (
        'float32_t',
        'int32_t',
        '[](auto s) { return ::rocjitsu::amdgpu::simd_cvt_i32_f32(s); }',
    ),
    'v_cvt_u32_f32_vop1': (
        'float32_t',
        'uint32_t',
        '[](auto s) { return ::rocjitsu::amdgpu::simd_cvt_u32_f32(s); }',
    ),
    'v_cvt_flr_i32_f32_vop1': (
        'float32_t',
        'int32_t',
        '[](auto s) {'
        ' auto r = util::stdx::floor(s);'
        ' return ::rocjitsu::amdgpu::simd_cvt_i32_f32(r); }',
    ),
    'v_cvt_rpi_i32_f32_vop1': (
        'float32_t',
        'int32_t',
        '[](auto s) {'
        ' auto r = util::stdx::ceil(s - util::native<float32_t>(0.5f));'
        ' return ::rocjitsu::amdgpu::simd_cvt_i32_f32(r); }',
    ),
    # --- f16 (half) ops. Scalar bodies route through an f32 intermediate with a
    # single final round, so the SIMD path (f16_to_f32_simd -> f32 op ->
    # f32_to_f16_simd) is bit-identical. The conversions are bit-exact (see
    # UtilSimd.F16ToF32/F32ToF16 guards). f16 result occupies the low 16 bits of
    # the dst, high zeroed (Tout=uint32_t), matching write_lane zero-extension. ---
    'v_floor_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) { return util::f32_to_f16_simd(util::floor_simd(util::f16_to_f32_simd(a))); }',
    ),
    'v_ceil_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) { return util::f32_to_f16_simd(util::ceil_simd(util::f16_to_f32_simd(a))); }',
    ),
    'v_trunc_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) { return util::f32_to_f16_simd(util::trunc_simd(util::f16_to_f32_simd(a))); }',
    ),
    'v_rndne_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' return util::f32_to_f16_simd(util::rndne_simd(util::f16_to_f32_simd(a))); }',
    ),
    'v_fract_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto f = util::f16_to_f32_simd(a);'
        ' return util::f32_to_f16_simd(f - util::floor_simd(f)); }',
    ),
    # f16 transcendentals mirror the scalar f32_to_f16(<op>_f32(f16_to_f32(x)))
    # by applying the f32-domain util::*_f32_simd helper (FTZ flush + canonical
    # qNaN / NaN-input guards) on the f16->f32 intermediate.
    'v_rcp_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' return util::f32_to_f16_simd(util::rcp_f32_simd(util::f16_to_f32_simd(a))); }',
    ),
    'v_rsq_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' return util::f32_to_f16_simd(util::rsq_f32_simd(util::f16_to_f32_simd(a))); }',
    ),
    'v_sqrt_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' return util::f32_to_f16_simd(util::sqrt_f32_simd(util::f16_to_f32_simd(a))); }',
    ),
    # exp/log_f16 inherit the exp2/log2 toolchain guard (UtilSimd.Exp2/Log2_*).
    'v_exp_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' return util::f32_to_f16_simd(util::exp_f32_simd(util::f16_to_f32_simd(a))); }',
    ),
    'v_log_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' return util::f32_to_f16_simd(util::log_f32_simd(util::f16_to_f32_simd(a))); }',
    ),
    # --- f16 <-> f32 / int16 conversions ---
    'v_cvt_f32_f16_vop1': (
        'uint32_t',
        'float32_t',
        '[](auto a) { return util::f16_to_f32_simd(a); }',
    ),
    'v_cvt_f16_f32_vop1': (
        'float32_t',
        'uint32_t',
        '[](auto a) { return util::f32_to_f16_simd(a); }',
    ),
    'v_cvt_f16_i16_vop1': (
        'int32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto i = (a << 16) >> 16;'  # sign-extend low 16 bits
        ' return util::f32_to_f16_simd(util::stdx::static_simd_cast<util::native<float32_t>>(i)); }',
    ),
    'v_cvt_f16_u16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto u = a & 0xFFFFu;'
        ' return util::f32_to_f16_simd(util::stdx::static_simd_cast<util::native<float32_t>>(u)); }',
    ),
    'v_cvt_i16_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto s = util::f16_to_f32_simd(a);'
        ' return ::rocjitsu::amdgpu::simd_cvt_i16_f32_to_u32(s); }',
    ),
    'v_cvt_u16_f16_vop1': (
        'uint32_t',
        'uint32_t',
        '[](auto a) {'
        ' auto s = util::f16_to_f32_simd(a);'
        ' return ::rocjitsu::amdgpu::simd_cvt_u16_f32_to_u32(s); }',
    ),
}


# template_name -> cpp_carry_op_functor
#
# Same keying as SIMD_VOP2_BINARY. The functor is invoked as
#   carry_op(simd<uint32_t> src0, simd<uint32_t> vsrc1, simd<uint32_t> cin)
#     -> SimdCarry<simd<uint32_t>, mask>
# inside try_execute_binary_vop2_carry_simd (lane type fixed to uint32_t).
# `cin` is the incoming VCC bit (0/1 per lane); the add_co/sub_co/subrev_co
# forms have no carry-in and ignore it (unnamed third parameter). Each functor
# returns the 32-bit result and a per-lane carry/borrow mask via make_simd_carry;
# the glue masked-stores the result and merges the carry into VCC for active
# lanes only. Bit-identical to the scalar bodies:
#   add_co:     w = (u64)a + (u64)b;          carry  = w > 0xFFFFFFFF
#   sub_co:     dst = a - b;                  borrow = a < b
#   subrev_co:  dst = b - a;                  borrow = b < a   (operands swapped)
#   addc:       w = (u64)a + (u64)b + cin;    carry  = w > 0xFFFFFFFF
#   subb:       dst = a - b - cin;            borrow = a < b + cin
#   subbrev:    dst = b - a - cin;            borrow = b < a + cin (operands swapped)
# The unsigned-wraparound carry/borrow identities (e.g. add carry = (a+b) < a;
# subtract-with-borrow chain) reproduce the scalar u64-domain results exactly.
SIMD_VOP2_CARRY: dict[str, str] = {
    'v_add_co_u32_vop2': (
        '[](auto a, auto b, auto) {'
        ' auto s = a + b;'
        ' return make_simd_carry(s, s < a); }'
    ),
    'v_sub_co_u32_vop2': (
        '[](auto a, auto b, auto) { return make_simd_carry(a - b, a < b); }'
    ),
    'v_subrev_co_u32_vop2': (
        '[](auto a, auto b, auto) { return make_simd_carry(b - a, b < a); }'
    ),
    'v_addc_co_u32_vop2': (
        '[](auto a, auto b, auto cin) {'
        ' auto t1 = a + b; auto c1 = t1 < a;'
        ' auto t2 = t1 + cin; auto c2 = t2 < t1;'
        ' return make_simd_carry(t2, c1 | c2); }'
    ),
    'v_subb_co_u32_vop2': (
        '[](auto a, auto b, auto cin) {'
        ' auto t1 = a - b; auto bw1 = a < b;'
        ' auto t2 = t1 - cin; auto bw2 = t1 < cin;'
        ' return make_simd_carry(t2, bw1 | bw2); }'
    ),
    'v_subbrev_co_u32_vop2': (
        '[](auto a, auto b, auto cin) {'
        ' auto t1 = b - a; auto bw1 = b < a;'
        ' auto t2 = t1 - cin; auto bw2 = t1 < cin;'
        ' return make_simd_carry(t2, bw1 | bw2); }'
    ),
    # NOTE: the RDNA VOP2 carry-in aliases (v_add_co_ci / sub_co_ci /
    # subrev_co_ci _vop2) are intentionally NOT wired here. On RDNA their decoded
    # form shares routing with the VOP3 add_co_ci path, and a VOP2-carry probe
    # (carry-in from VCC) diverges from the VOP3 src2 carry-in the kernels feed;
    # the _vop3 forms in SIMD_VOP3_CARRY_CIN cover these ops correctly.
}


# template_name -> (cpp_type, k_literal_expr, cpp_fma_op_functor)
#
# Same keying as SIMD_VOP2_BINARY. The VOP2 FMA/MAC/MAD family has three operand
# shapes, all built on the single-rounded fused multiply-add (the scalar bodies
# use std::fma). The functor is invoked as
#   fma_op(simd<T> src0, simd<T> vsrc1, simd<T> vdst, simd<T> k) -> simd<T>
# inside try_execute_ternary_vop2_simd; `k` is the broadcast inline literal
# (`k_literal_expr`, an inst.-qualified expression, or "0u" when there is none).
# Shapes:
#   dst-accumulate (fmac/mac):     fma(s0, s1, dvst)        -- ignores k
#   literal addend (fmaak/madak):  fma(s0, s1, k)           -- ignores dvst
#   literal mult  (fmamk/madmk):   fma(s0, k, s1)           -- ignores dvst
# f16 forms (lane type uint32_t) convert each operand via f16_to_f32_simd and
# round the result with f32_to_f16_simd (single final round, matching scalar).
# util::stdx::fma is bit-identical to std::fma for all finite/Inf inputs
# (UtilSimd.Fma_VectorMatchesScalar_BitExact, NaN inputs excluded); the f16<->f32
# conversions are already bit-exact, so the f16 forms match by composition. When
# an input is NaN the packed and scalar FMA may pick a different NaN operand to
# propagate (toolchain-dependent payload); that NaN-payload divergence is
# accepted (the result is a NaN either way). Note the two
# distinct literal members: fmaak/fmamk use inst.simm32_, while madak/madmk use
# inst.simm32.encoding_value_ (matching the scalar bodies). v_fmac_f64 is
# excluded (64-bit / 2-VGPR lanes — a separate width).
_FMA_ACC_F32 = '[](auto a, auto b, auto d, auto) { return util::stdx::fma(a, b, d); }'
_FMA_ADDK_F32 = '[](auto a, auto b, auto, auto k) { return util::stdx::fma(a, b, k); }'
_FMA_MULK_F32 = '[](auto a, auto b, auto, auto k) { return util::stdx::fma(a, k, b); }'
_FMA_ACC_F16 = (
    '[](auto a, auto b, auto d, auto) {'
    ' return util::f32_to_f16_simd(util::stdx::fma('
    'util::f16_to_f32_simd(a), util::f16_to_f32_simd(b), util::f16_to_f32_simd(d))); }'
)
_FMA_ADDK_F16 = (
    '[](auto a, auto b, auto, auto k) {'
    ' return util::f32_to_f16_simd(util::stdx::fma('
    'util::f16_to_f32_simd(a), util::f16_to_f32_simd(b), util::f16_to_f32_simd(k))); }'
)
_FMA_MULK_F16 = (
    '[](auto a, auto b, auto, auto k) {'
    ' return util::f32_to_f16_simd(util::stdx::fma('
    'util::f16_to_f32_simd(a), util::f16_to_f32_simd(k), util::f16_to_f32_simd(b))); }'
)
SIMD_VOP2_TERNARY: dict[str, tuple[str, str, str]] = {
    # --- f32 dst-accumulate ---
    'v_fmac_f32_vop2': ('float32_t', '0u', _FMA_ACC_F32),
    'v_fmac_dx9_zero_f32_vop2': ('float32_t', '0u', _FMA_ACC_F32),
    'v_mac_f32_vop2': ('float32_t', '0u', _FMA_ACC_F32),
    # --- f32 inline literal ---
    'v_fmaak_f32_vop2': ('float32_t', 'inst.simm32_', _FMA_ADDK_F32),
    'v_madak_f32_vop2': ('float32_t', 'inst.simm32.encoding_value_', _FMA_ADDK_F32),
    'v_fmamk_f32_vop2': ('float32_t', 'inst.simm32_', _FMA_MULK_F32),
    'v_madmk_f32_vop2': ('float32_t', 'inst.simm32.encoding_value_', _FMA_MULK_F32),
    # --- f16 dst-accumulate ---
    'v_fmac_f16_vop2': ('uint32_t', '0u', _FMA_ACC_F16),
    'v_mac_f16_vop2': ('uint32_t', '0u', _FMA_ACC_F16),
    # --- f16 inline literal ---
    'v_madak_f16_vop2': ('uint32_t', 'inst.simm32.encoding_value_', _FMA_ADDK_F16),
    'v_fmamk_f16_vop2': ('uint32_t', 'inst.simm32_', _FMA_MULK_F16),
    'v_madmk_f16_vop2': ('uint32_t', 'inst.simm32.encoding_value_', _FMA_MULK_F16),
    # v_fmaak_f16 (RDNA only): dst = fma(s0, s1, K), K = f16(simm32_). Same f16
    # FMA functor as v_madak_f16, differing only in the literal field (simm32_ vs
    # simm32.encoding_value_); the SIMD path is identical to the tested madak_f16.
    'v_fmaak_f16_vop2': ('uint32_t', 'inst.simm32_', _FMA_ADDK_F16),
}


# template_name -> cpp_fma_op_functor (dst-accumulate, over native<double>).
#
# 64-bit-lane VOP2 FMA. The only f64 VOP2 op reachable on CDNA4 is v_fmac_f64
# (dst = fma(src0, vsrc1, dst), all f64). The functor is invoked as
#   fma_op(simd<double> src0, simd<double> vsrc1, simd<double> vdst) -> simd<double>
# inside try_execute_ternary_vop2_f64_simd (lane type fixed to double, read/written
# through the split lo/hi 32-bit VGPR-pair path). util::stdx::fma over native<double>
# is bit-identical to the scalar std::fma for all finite/Inf inputs; NaN-input lanes
# may differ in propagated NaN payload (accepted). Guarded by
# UtilSimd.FmaF64_VectorMatchesScalar_BitExact.
SIMD_VOP2_FMA_F64: dict[str, str] = {
    'v_fmac_f64_vop2': '[](auto a, auto b, auto d) { return util::stdx::fma(a, b, d); }',
}


# template_name -> cpp_bin_op (over native<double>, no modifiers). VOP2 f64
# binary forms: scalar bodies read src0/vsrc1 as read_lane64, no abs/neg/omod/
# clamp. add/mul are bit-exact; max_num/min_num use util::stdx::fmax/fmin (scalar
# is std::fmax/std::fmin) with the same accepted NaN-payload / signed-zero-tie
# carve-out as the f64 vop3 forms in SIMD_VOP3_BINARY_FP64.
SIMD_VOP2_BINARY_FP64: dict[str, str] = {
    'v_add_f64_vop2': '[](auto a, auto b) { return a + b; }',
    'v_mul_f64_vop2': '[](auto a, auto b) { return a * b; }',
    'v_max_num_f64_vop2': '[](auto a, auto b) { return util::stdx::fmax(a, b); }',
    'v_min_num_f64_vop2': '[](auto a, auto b) { return util::stdx::fmin(a, b); }',
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
    'v_ceil_f64_vop1': ('double', '[](auto a) { return util::ceil_simd(a); }'),
    'v_floor_f64_vop1': ('double', '[](auto a) { return util::floor_simd(a); }'),
    'v_trunc_f64_vop1': ('double', '[](auto a) { return util::trunc_simd(a); }'),
    'v_rndne_f64_vop1': ('double', '[](auto a) { return util::rndne_simd(a); }'),
    'v_fract_f64_vop1': ('double', '[](auto a) { return a - util::floor_simd(a); }'),
    'v_rcp_f64_vop1': (
        'double',
        '[](auto a) { return util::native<double>(1.0) / a; }',
    ),
    'v_rsq_f64_vop1': (
        'double',
        '[](auto a) { return util::native<double>(1.0) / util::stdx::sqrt(a); }',
    ),
    'v_sqrt_f64_vop1': ('double', '[](auto a) { return util::sqrt_f64_simd(a); }'),
    'v_mov_b64_vop1': ('uint64_t', '[](auto a) { return a; }'),
    # frexp mantissa: significand in [0.5,1) via bitfield rebias + denormal
    # renorm (see util::frexp_mant_f64_simd). VOP3 twin in SIMD_VOP3_UNARY_FP64.
    'v_frexp_mant_f64_vop1': (
        'double',
        '[](auto a) { return util::frexp_mant_f64_simd(a); }',
    ),
}


# --- mixed-width f64 <-> 32-bit conversions -------------------------------
#
# These VOP1 cvt ops bridge an 8-wide (native_width64) f64 chunk and the same
# number of 32-bit lanes, so they use dedicated glue rather than the equal-width
# unary path. The 32-bit side is a util::narrow32<T> (fixed_size_simd<T,8>); a
# direct static_simd_cast bridges it to/from native<double> with no bit_cast.
#
# f64 source -> 32-bit dst. template_name -> (out_lane_cpp_type, functor); the
# functor is invoked as cvt_op(native<double>) -> narrow32<Tout> inside
# try_execute_cvt_f64_to_b32_simd. cvt_f32_f64 is a single correctly-rounded
# narrowing cast (vcvtpd2ps); the int forms do NaN->0 and the saturating clamp in
# the double domain (all where-masks native<double>, so no cross-width mask cast)
# then one truncating cast to the 8-wide int — bit-identical to the scalar body
# for finite/Inf inputs. A NaN *result* of cvt_f32_f64 may differ in payload
# (accepted; the A/B test skips it). INT32_MAX/MIN and UINT32_MAX are all exactly
# representable in double, so the clamp constants cast back to the exact integers.
SIMD_CVT_F64_TO_B32: dict[str, tuple[str, str]] = {
    'v_cvt_f32_f64_vop1': (
        'float32_t',
        '[](auto s) { return util::stdx::static_simd_cast<util::narrow32<float32_t>>(s); }',
    ),
    'v_cvt_i32_f64_vop1': (
        'int32_t',
        '[](auto s) {'
        ' auto r = util::cvt_i32_f64_saturate_input_simd(s);'
        ' return util::stdx::static_simd_cast<util::narrow32<int32_t>>(r); }',
    ),
    'v_cvt_u32_f64_vop1': (
        'uint32_t',
        '[](auto s) {'
        ' auto r = util::cvt_u32_f64_saturate_input_simd(s);'
        ' return util::stdx::static_simd_cast<util::narrow32<uint32_t>>(r); }',
    ),
    # frexp exponent (f64 -> int32). util::frexp_exp_f64_simd returns the int32 in
    # the low 32 bits of each 64-bit lane; the narrowing cast keeps them. The VOP3
    # twin applies float omod/clamp to float(exp) + bit-casts (a different output
    # encoding), so it stays scalar (_VOP3_UNARY_SKIP gates the cvt auto-route).
    'v_frexp_exp_i32_f64_vop1': (
        'int32_t',
        '[](auto s) {'
        ' return util::stdx::static_simd_cast<util::narrow32<int32_t>>('
        'util::frexp_exp_f64_simd(s)); }',
    ),
}

# 32-bit source -> f64 dst. template_name -> (in_lane_cpp_type, functor); the
# functor is invoked as cvt_op(narrow32<Tin>) -> native<double> inside
# try_execute_cvt_b32_to_f64_simd. Each is an exact widening static_simd_cast
# (vcvtps2pd for f32; int->double for i32/u32), bit-identical to the scalar body
# (static_cast<double>) for every input.
SIMD_CVT_B32_TO_F64: dict[str, tuple[str, str]] = {
    'v_cvt_f64_f32_vop1': (
        'float32_t',
        '[](auto in) { return util::stdx::static_simd_cast<util::native<double>>(in); }',
    ),
    'v_cvt_f64_i32_vop1': (
        'int32_t',
        '[](auto in) { return util::stdx::static_simd_cast<util::native<double>>(in); }',
    ),
    'v_cvt_f64_u32_vop1': (
        'uint32_t',
        '[](auto in) { return util::stdx::static_simd_cast<util::native<double>>(in); }',
    ),
}


# template_name set for v_cndmask_b32 (VCC-driven per-lane select). No functor:
# the op is fixed (dst = (VCC bit) ? vsrc1 : src0), a pure 32-bit bit select, so
# the SIMD result is bit-identical to the scalar body for every input.
SIMD_VOP2_CNDMASK: set[str] = {
    'v_cndmask_b32_vop2',
}

# VOP3 form of v_cndmask_b32: same per-lane select, but the 64-bit selector is
# read from the SGPR-pair `src2` instead of VCC. Also fixed-op / functorless.
SIMD_VOP3_CNDMASK: set[str] = {
    'v_cndmask_b32_vop3',
}

# 16-bit variant — RDNA3+. Low-16 of each source selected per lane, high-16
# zero (matches scalar `uint32_t(uint16_t(...))` write pattern).
SIMD_VOP3_CNDMASK_B16: set[str] = {
    'v_cndmask_b16_vop3',
}

# VOP3 div_fmas: fma(src0, src1, src2) followed by a VCC-bit-gated
# ldexp(result, 32) (f32) or ldexp(result, 64) (f64). Fixed-op / functorless.
SIMD_VOP3_DIV_FMAS_FP32: set[str] = {
    'v_div_fmas_f32_vop3',
}
SIMD_VOP3_DIV_FMAS_FP64: set[str] = {
    'v_div_fmas_f64_vop3',
}

# VOP3P fma_mix / mad_mix family. The six ops share one body (`a*b + c` plus
# optional clamp to [0,1]); only the destination shape differs:
#  - F32     -> v_fma_mix_f32_vop3p (RDNA3+), v_mad_mix_f32_vop3p (CDNA1-4)
#  - F16_LO  -> v_fma_mixlo_f16_vop3p, v_mad_mixlo_f16_vop3p
#  - F16_HI  -> v_fma_mixhi_f16_vop3p, v_mad_mixhi_f16_vop3p
# Per-source op_sel_hi gates the f16<->f32 widening shape; op_sel picks the f16
# half. neg flips the sign bit. No abs, no omod. Functorless / fixed-op.
SIMD_VOP3P_FMA_MIX_F32: set[str] = {
    'v_fma_mix_f32_vop3p',
    'v_mad_mix_f32_vop3p',
}
SIMD_VOP3P_FMA_MIX_F16_LO: set[str] = {
    'v_fma_mixlo_f16_vop3p',
    'v_mad_mixlo_f16_vop3p',
}
SIMD_VOP3P_FMA_MIX_F16_HI: set[str] = {
    'v_fma_mixhi_f16_vop3p',
    'v_mad_mixhi_f16_vop3p',
}

# VOP3P packed-16 integer binary family. Each 32-bit lane holds {low16,
# high16}. The glue gates op_sel == 0 && op_sel_hi == 3 (default packing)
# and bails to scalar otherwise; the functor receives the two u32 simd
# vectors as packed pairs and returns the same shape with the per-half op
# applied. Scalar bodies for these ops do NOT apply neg/neg_hi/clamp on
# integer operands, so the SIMD path also passes through. mul_lo / shift
# functors mask each half to 16 bits before packing to drop any product
# overflow / shift-into-bit-16 leakage between halves.
SIMD_VOP3P_PK_BINARY_INT: dict[str, str] = {
    'v_pk_add_u16_vop3p': (
        '[](auto a, auto b) {'
        ' auto lo = (a + b) & 0xFFFFu;'
        ' auto hi = ((a >> 16) + (b >> 16)) & 0xFFFFu;'
        ' return lo | (hi << 16); }'
    ),
    # add_i16 is bit-identical to add_u16 (mod-2^16 wrap matches).
    'v_pk_add_i16_vop3p': (
        '[](auto a, auto b) {'
        ' auto lo = (a + b) & 0xFFFFu;'
        ' auto hi = ((a >> 16) + (b >> 16)) & 0xFFFFu;'
        ' return lo | (hi << 16); }'
    ),
    'v_pk_sub_u16_vop3p': (
        '[](auto a, auto b) {'
        ' auto lo = (a - b) & 0xFFFFu;'
        ' auto hi = ((a >> 16) - (b >> 16)) & 0xFFFFu;'
        ' return lo | (hi << 16); }'
    ),
    'v_pk_sub_i16_vop3p': (
        '[](auto a, auto b) {'
        ' auto lo = (a - b) & 0xFFFFu;'
        ' auto hi = ((a >> 16) - (b >> 16)) & 0xFFFFu;'
        ' return lo | (hi << 16); }'
    ),
    'v_pk_mul_lo_u16_vop3p': (
        '[](auto a, auto b) {'
        ' auto lo = ((a & 0xFFFFu) * (b & 0xFFFFu)) & 0xFFFFu;'
        ' auto hi = ((a >> 16) * (b >> 16)) & 0xFFFFu;'
        ' return lo | (hi << 16); }'
    ),
    # Reverse-shift forms: src1 holds the value, src0 holds the count.
    # Shift count masked to low 4 bits per scalar (`& 15u`).
    'v_pk_lshlrev_b16_vop3p': (
        '[](auto a, auto b) {'
        ' auto lo = ((b & 0xFFFFu) << (a & 15u)) & 0xFFFFu;'
        ' auto hi = (((b >> 16) & 0xFFFFu) << ((a >> 16) & 15u)) & 0xFFFFu;'
        ' return lo | (hi << 16); }'
    ),
    'v_pk_lshrrev_b16_vop3p': (
        '[](auto a, auto b) {'
        ' auto lo = ((b & 0xFFFFu) >> (a & 15u)) & 0xFFFFu;'
        ' auto hi = ((b >> 16) >> ((a >> 16) & 15u)) & 0xFFFFu;'
        ' return lo | (hi << 16); }'
    ),
    # Arithmetic right shift on i16: sign-extend each half to int32 via
    # (x << 16) >> 16, shift, mask back to 16.
    'v_pk_ashrrev_i16_vop3p': (
        '[](auto a, auto b) {'
        ' using I = util::native<int32_t>;'
        ' auto bv_lo = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;'
        ' auto bv_hi = (util::stdx::static_simd_cast<I>(b >> 16) << 16) >> 16;'
        ' auto sh_lo = util::stdx::static_simd_cast<I>(a & 15u);'
        ' auto sh_hi = util::stdx::static_simd_cast<I>((a >> 16) & 15u);'
        ' auto rlo = util::stdx::static_simd_cast<util::native<uint32_t>>(bv_lo >> sh_lo) & 0xFFFFu;'
        ' auto rhi = util::stdx::static_simd_cast<util::native<uint32_t>>(bv_hi >> sh_hi) & 0xFFFFu;'
        ' return rlo | (rhi << 16); }'
    ),
    'v_pk_min_u16_vop3p': (
        '[](auto a, auto b) {'
        ' auto lo = util::stdx::min(a & 0xFFFFu, b & 0xFFFFu);'
        ' auto hi = util::stdx::min(a >> 16, b >> 16);'
        ' return (lo & 0xFFFFu) | ((hi & 0xFFFFu) << 16); }'
    ),
    'v_pk_max_u16_vop3p': (
        '[](auto a, auto b) {'
        ' auto lo = util::stdx::max(a & 0xFFFFu, b & 0xFFFFu);'
        ' auto hi = util::stdx::max(a >> 16, b >> 16);'
        ' return (lo & 0xFFFFu) | ((hi & 0xFFFFu) << 16); }'
    ),
    'v_pk_min_i16_vop3p': (
        '[](auto a, auto b) {'
        ' using I = util::native<int32_t>;'
        ' auto a_lo = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;'
        ' auto a_hi = (util::stdx::static_simd_cast<I>(a >> 16) << 16) >> 16;'
        ' auto b_lo = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;'
        ' auto b_hi = (util::stdx::static_simd_cast<I>(b >> 16) << 16) >> 16;'
        ' auto rlo = util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::min(a_lo, b_lo)) & 0xFFFFu;'
        ' auto rhi = util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::min(a_hi, b_hi)) & 0xFFFFu;'
        ' return rlo | (rhi << 16); }'
    ),
    'v_pk_max_i16_vop3p': (
        '[](auto a, auto b) {'
        ' using I = util::native<int32_t>;'
        ' auto a_lo = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;'
        ' auto a_hi = (util::stdx::static_simd_cast<I>(a >> 16) << 16) >> 16;'
        ' auto b_lo = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;'
        ' auto b_hi = (util::stdx::static_simd_cast<I>(b >> 16) << 16) >> 16;'
        ' auto rlo = util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::max(a_lo, b_lo)) & 0xFFFFu;'
        ' auto rhi = util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::max(a_hi, b_hi)) & 0xFFFFu;'
        ' return rlo | (rhi << 16); }'
    ),
}

# VOP3P packed-16 integer ternary (pk_mad_i16 / pk_mad_u16). Same default
# packing gate as the binary table (op_sel/op_sel_hi/op_sel_hi_2). Scalar
# truncates to 16 bits via uint16_t cast, so the SIMD path masks each half
# to 16 bits before pack.
SIMD_VOP3P_PK_TERNARY_INT: dict[str, str] = {
    'v_pk_mad_i16_vop3p': (
        '[](auto a, auto b, auto c) {'
        ' using I = util::native<int32_t>;'
        ' auto a_lo = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;'
        ' auto a_hi = (util::stdx::static_simd_cast<I>(a >> 16) << 16) >> 16;'
        ' auto b_lo = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;'
        ' auto b_hi = (util::stdx::static_simd_cast<I>(b >> 16) << 16) >> 16;'
        ' auto c_lo = (util::stdx::static_simd_cast<I>(c & 0xFFFFu) << 16) >> 16;'
        ' auto c_hi = (util::stdx::static_simd_cast<I>(c >> 16) << 16) >> 16;'
        ' auto rlo = util::stdx::static_simd_cast<util::native<uint32_t>>(a_lo * b_lo + c_lo) & 0xFFFFu;'
        ' auto rhi = util::stdx::static_simd_cast<util::native<uint32_t>>(a_hi * b_hi + c_hi) & 0xFFFFu;'
        ' return rlo | (rhi << 16); }'
    ),
    'v_pk_mad_u16_vop3p': (
        '[](auto a, auto b, auto c) {'
        ' auto a_lo = a & 0xFFFFu;'
        ' auto a_hi = a >> 16;'
        ' auto b_lo = b & 0xFFFFu;'
        ' auto b_hi = b >> 16;'
        ' auto c_lo = c & 0xFFFFu;'
        ' auto c_hi = c >> 16;'
        ' auto rlo = (a_lo * b_lo + c_lo) & 0xFFFFu;'
        ' auto rhi = (a_hi * b_hi + c_hi) & 0xFFFFu;'
        ' return rlo | (rhi << 16); }'
    ),
}

# VOP3P packed-16 f16 binary family. Each 32-bit lane holds 2 f16 values.
# Glue widens halves to f32, applies neg/neg_hi (sign-bit toggle), runs the
# per-half functor in f32, narrows back to f16, packs. No clamp on any
# pk_*_f16 scalar body (verified line 15109, 15519). NaN-input lanes can
# diverge in payload (same as the existing f16 ternary slice).
SIMD_VOP3P_PK_BINARY_FP16: dict[str, str] = {
    'v_pk_add_f16_vop3p': '[](auto a, auto b) { return a + b; }',
    'v_pk_mul_f16_vop3p': '[](auto a, auto b) { return a * b; }',
    'v_pk_max_f16_vop3p': '[](auto a, auto b) { return util::stdx::fmax(a, b); }',
    'v_pk_min_f16_vop3p': '[](auto a, auto b) { return util::stdx::fmin(a, b); }',
}

# pk_fma_f16 — 3-source FMA per half. NaN-input payload divergence accepted.
SIMD_VOP3P_PK_TERNARY_FP16: dict[str, str] = {
    'v_pk_fma_f16_vop3p': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
}

# VOP3P packed-f32 binary. Each operand is a 64-bit consecutive-VGPR pair of two
# f32 (lo/hi). Glue extracts each f32 half (narrow32 width), applies neg/neg_hi
# (sign-bit toggle), runs the per-half functor, repacks. No clamp on any pk_f32
# scalar body. Default packing only (op_sel = 0, op_sel_hi = 3).
SIMD_VOP3P_PK_BINARY_F32: dict[str, str] = {
    'v_pk_add_f32_vop3p': '[](auto a, auto b) { return a + b; }',
    'v_pk_mul_f32_vop3p': '[](auto a, auto b) { return a * b; }',
}

# pk_fma_f32 — 3-source FMA per half. op_sel_hi_2 == 1 gate (src2-hi select).
# NaN-input payload divergence accepted.
SIMD_VOP3P_PK_TERNARY_F32: dict[str, str] = {
    'v_pk_fma_f32_vop3p': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
}

# v_pk_mov_b32 — default-packing-only fast path. Each src is a 64-bit pair
# (consecutive VGPRs), result is (src0_lo, src1_hi). Functorless / fixed-op.
SIMD_VOP3P_MOV_B32: set[str] = {
    'v_pk_mov_b32_vop3p',
}

# VOP3P integer dot products. dst is a single 32-bit lane (not packed) and
# src2 is the accumulator; the dot reduction happens within each lane so the
# fast path vectorizes across lanes. Parameterised by (ElemBits, Signed):
# the 8/4-bit forms ignore op_sel in the scalar body, the 16-bit forms gate
# on default packing inside the glue. Signed forms lower-clamp to 0 when
# inst.clamp is set; the unsigned scalar bodies have no clamp branch.
SIMD_VOP3P_DOT_INT: dict[str, str] = {
    'v_dot4_i32_i8_vop3p': '8, true',
    'v_dot4_u32_u8_vop3p': '8, false',
    'v_dot8_i32_i4_vop3p': '4, true',
    'v_dot8_u32_u4_vop3p': '4, false',
    'v_dot2_i32_i16_vop3p': '16, true',
    'v_dot2_u32_u16_vop3p': '16, false',
}

# v_dot2_f32_{f16,bf16} — two half-precision products + an f32 accumulator into
# one f32 lane. op_sel half-select (gated default), neg/neg_hi sign flips,
# optional clamp to [0,1]. Functorless / fixed-op. The set spans BOTH 16-bit
# float formats: v_dot2_f32_bf16's generated scalar body is byte-identical to the
# f16 form (it widens each half via util::f16_to_f32 with the same
# op_sel/neg/clamp handling, verified by diff), so both route through the same
# glue. Named _F16_OR_BF16 to make that span explicit (the emitted macro keeps
# the shorter ROCJITSU_TRY_SIMD_VOP3P_DOT_F16 name — the widening path is shared).
SIMD_VOP3P_DOT_F16_OR_BF16: set[str] = {
    'v_dot2_f32_f16_vop3p',
    'v_dot2_f32_bf16_vop3p',
}

# VOP3P mixed-sign integer dots: per-operand signedness from inst.neg bits 0/1
# at runtime (src0/src1); int32 accumulate from src2; lower-clamp-to-0; no
# op_sel/neg_hi in the scalar bodies. key -> ElemBits.
SIMD_VOP3P_DOT_INT_MIXED: dict[str, str] = {
    'v_dot4_i32_iu8_vop3p': '8',
    'v_dot8_i32_iu4_vop3p': '4',
}

# VOP2/VOP3 dst-accumulate integer dots (the "c" forms). Accumulator is vdst
# (read + write), NOT src2; all signed; no op_sel/neg/clamp. Keyed by full
# template_name (incl _vop2/_vop3) — the only structural difference is the 2nd
# source member (vsrc1 vs src1), passed as the Vop3 bool. Value = "ElemBits, Vop3".
SIMD_DOTC_INT: dict[str, str] = {
    'v_dot2c_i32_i16_vop2': '16, false',
    'v_dot2c_i32_i16_vop3': '16, true',
    'v_dot4c_i32_i8_vop2': '8, false',
    'v_dot4c_i32_i8_vop3': '8, true',
    'v_dot8c_i32_i4_vop2': '4, false',
    'v_dot8c_i32_i4_vop3': '4, true',
}

# VOP2/VOP3 dst-accumulate f16 dot (v_dot2c_f32_f16). Accumulator is vdst (f32
# bits); bracketing acc + (a0*b0 + a1*b1) matches the scalar facc += ... form.
# Value = "Vop3".
SIMD_DOTC_F16: dict[str, str] = {
    'v_dot2c_f32_f16_vop2': 'false',
    'v_dot2c_f32_f16_vop3': 'true',
    'v_dot2acc_f32_f16_vop2': 'false',
}


# --- VOPC compare -> VCC ---------------------------------------------------
#
# 198 VOPC opcodes on CDNA4 are the single biggest breadth. Each writes one bit
# into VCC per active EXEC lane (inactive bits preserved); CDNA4 has no v_cmpx
# (EXEC-writing) form, so one glue shape (try_execute_vopc_simd) covers them all.
# The table maps template_name -> (lane_cpp_type, cmp_functor); the functor is
# invoked as cmp_op(native<T> src0, native<T> vsrc1) -> simd_mask and must mirror
# the scalar body's comparison expression *verbatim* (esp. the ordered vs
# unordered NaN behaviour, e.g. v_cmp_nlt = !(a < b), true on NaN). It is built
# programmatically below from per-suffix operand conversions and per-relation
# expressions.
#
# Lane width buckets: the 32-bit (f32/i32/u32) and 16-bit (f16/i16/u16) suffixes
# all read as 32-bit lanes — the 16-bit ones narrow/convert inside the functor —
# so they share the existing read_simd<T> path. The 64-bit suffixes (f64/i64/u64)
# need the 64-bit-lane infra and are wired separately. v_cmp_class_* (a bitfield
# class test, not a relational compare) is left scalar.

# suffix -> (lane_cpp_type, operand-conversion template using {x})
_VOPC_SUFFIX: dict[str, tuple[str, str]] = {
    'f32': ('float32_t', '{x}'),
    'f16': ('uint32_t', 'util::f16_to_f32_simd({x})'),
    'i32': ('int32_t', '{x}'),
    'u32': ('uint32_t', '{x}'),
    # 16-bit integers read as uint32 lanes; sign-extend / mask the low 16 bits to
    # match the scalar static_cast<int16_t>/<uint16_t>.
    'i16': (
        'uint32_t',
        '((util::stdx::static_simd_cast<util::native<int32_t>>({x}) << 16) >> 16)',
    ),
    'u16': ('uint32_t', '({x} & 0xFFFFu)'),
    # 64-bit lanes: read directly as the native 64-bit lane type (read_simd64),
    # so the conversion is the identity. Routed through the VOPC64 glue.
    'f64': ('double', '{x}'),
    'i64': ('int64_t', '{x}'),
    'u64': ('uint64_t', '{x}'),
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


def _build_simd_vopc() -> dict[str, tuple[str, str]]:
    table: dict[str, tuple[str, str]] = {}
    # 16-/32-bit lane suffixes only; f64/i64/u64 (64-bit lane) wired separately.
    for suf in ('f16', 'f32'):
        lane_t, conv = _VOPC_SUFFIX[suf]
        for rel in _VOPC_FLOAT_RELS:
            table[f'v_cmp_{rel}_{suf}_vopc'] = (lane_t, _vopc_functor(conv, rel))
    for suf in ('i16', 'u16', 'i32', 'u32'):
        lane_t, conv = _VOPC_SUFFIX[suf]
        for rel in _VOPC_INT_RELS:
            table[f'v_cmp_{rel}_{suf}_vopc'] = (lane_t, _vopc_functor(conv, rel))
    return table


def _build_simd_vopc64() -> dict[str, tuple[str, str]]:
    """64-bit-lane VOPC compares (f64/i64/u64), routed through the VOPC64 glue."""
    table: dict[str, tuple[str, str]] = {}
    lane_t, conv = _VOPC_SUFFIX['f64']
    for rel in _VOPC_FLOAT_RELS:
        table[f'v_cmp_{rel}_f64_vopc'] = (lane_t, _vopc_functor(conv, rel))
    for suf in ('i64', 'u64'):
        lane_t, conv = _VOPC_SUFFIX[suf]
        for rel in _VOPC_INT_RELS:
            table[f'v_cmp_{rel}_{suf}_vopc'] = (lane_t, _vopc_functor(conv, rel))
    return table


SIMD_VOPC: dict[str, tuple[str, str]] = _build_simd_vopc()
SIMD_VOPC64: dict[str, tuple[str, str]] = _build_simd_vopc64()


# --- VOPC v_cmp_class -> VCC ----------------------------------------------
#
# v_cmp_class tests src0's IEEE-754 float class against a 10-bit class mask in
# vsrc1 and writes one VCC bit per active lane. It is NOT a relational compare
# (no src0-vs-vsrc1 ordering), so it needs a class-decode functor rather than the
# relational builder above — but the VCC merge / lane packing is identical, so the
# f16/f32 forms reuse try_execute_vopc_simd (lane type uint32_t, raw bits). src0 is
# read as raw bits and vsrc1 as the mask; the functor partitions the value into
# exactly one of the 10 mutually exclusive classes (one bit set in `cls`) and
# returns `(cls & mask) != 0`, which equals the scalar body's OR of mask-gated
# class predicates. The classification is done purely from the exponent / mantissa
# / sign bits (matching the scalar std::isnan/isinf/isnormal/signbit outcomes,
# which for finite/Inf reduce to the same bit tests), so it is bit-exact for every
# input including NaN payloads. f64 (a 64-bit value vs a 32-bit mask) needs a
# mixed-width glue and is wired separately.
#
# Class-bit layout (low 10 bits of the mask): 0x001 sNaN, 0x002 qNaN, 0x004 -Inf,
# 0x008 -normal, 0x010 -denormal, 0x020 -0, 0x040 +0, 0x080 +denormal,
# 0x100 +normal, 0x200 +Inf. The qNaN bit is the mantissa MSB
# (f32 0x00400000, f16 0x0200).
_CMP_CLASS_F32 = (
    '[](auto a, auto b) {'
    ' using U = util::native<uint32_t>;'
    ' U exp = (a >> 23) & 0xFFu;'
    ' U mant = a & 0x7FFFFFu;'
    ' auto sgn = ((a >> 31) & 1u) != 0u;'
    ' auto qnan = ((a >> 22) & 1u) != 0u;'
    ' auto is_nan = (exp == 0xFFu) && (mant != 0u);'
    ' auto is_inf = (exp == 0xFFu) && (mant == 0u);'
    ' auto is_zero = (exp == 0u) && (mant == 0u);'
    ' auto is_den = (exp == 0u) && (mant != 0u);'
    ' auto is_norm = (exp >= 1u) && (exp <= 0xFEu);'
    ' U cls(0u);'
    ' util::stdx::where(is_nan && !qnan, cls) = 0x001u;'
    ' util::stdx::where(is_nan && qnan, cls) = 0x002u;'
    ' util::stdx::where(is_inf && sgn, cls) = 0x004u;'
    ' util::stdx::where(is_norm && sgn, cls) = 0x008u;'
    ' util::stdx::where(is_den && sgn, cls) = 0x010u;'
    ' util::stdx::where(is_zero && sgn, cls) = 0x020u;'
    ' util::stdx::where(is_zero && !sgn, cls) = 0x040u;'
    ' util::stdx::where(is_den && !sgn, cls) = 0x080u;'
    ' util::stdx::where(is_norm && !sgn, cls) = 0x100u;'
    ' util::stdx::where(is_inf && !sgn, cls) = 0x200u;'
    ' return (cls & b) != 0u; }'
)
_CMP_CLASS_F16 = (
    '[](auto a, auto b) {'
    ' using U = util::native<uint32_t>;'
    ' U h = a & 0xFFFFu;'
    ' U exp = (h >> 10) & 0x1Fu;'
    ' U mant = h & 0x3FFu;'
    ' auto sgn = ((h >> 15) & 1u) != 0u;'
    ' auto qnan = ((h >> 9) & 1u) != 0u;'
    ' auto is_nan = (exp == 0x1Fu) && (mant != 0u);'
    ' auto is_inf = (exp == 0x1Fu) && (mant == 0u);'
    ' auto is_zero = (exp == 0u) && (mant == 0u);'
    ' auto is_den = (exp == 0u) && (mant != 0u);'
    ' auto is_norm = (exp >= 1u) && (exp <= 30u);'
    ' U cls(0u);'
    ' util::stdx::where(is_nan && !qnan, cls) = 0x001u;'
    ' util::stdx::where(is_nan && qnan, cls) = 0x002u;'
    ' util::stdx::where(is_inf && sgn, cls) = 0x004u;'
    ' util::stdx::where(is_norm && sgn, cls) = 0x008u;'
    ' util::stdx::where(is_den && sgn, cls) = 0x010u;'
    ' util::stdx::where(is_zero && sgn, cls) = 0x020u;'
    ' util::stdx::where(is_zero && !sgn, cls) = 0x040u;'
    ' util::stdx::where(is_den && !sgn, cls) = 0x080u;'
    ' util::stdx::where(is_norm && !sgn, cls) = 0x100u;'
    ' util::stdx::where(is_inf && !sgn, cls) = 0x200u;'
    ' return (cls & b) != 0u; }'
)
SIMD_VOPC_CLASS: dict[str, tuple[str, str]] = {
    'v_cmp_class_f16_vopc': ('uint32_t', _CMP_CLASS_F16),
    'v_cmp_class_f32_vopc': ('uint32_t', _CMP_CLASS_F32),
}


# v_cmp_class_f64: a 64-bit f64 value (src0) tested against a 32-bit class mask
# (vsrc1), so it needs the mixed-width class glue (try_execute_vopc_class_f64_simd)
# rather than the equal-width VOPC path. The functor receives src0 as
# a uint64 SIMD raw-bit vector and vsrc1 as the matching uint32 SIMD class-mask
# mask; it classifies the f64 from its raw bits (qNaN bit = mantissa MSB
# 0x0008000000000000), partitions into one of the 10 mutually exclusive classes,
# casts the small class code down to the 32-bit mask width, and returns
# (cls & mask) != 0. Pure bit decode, bit-exact with the scalar body. The
# class-bit layout matches the f16/f32 forms above.
_CMP_CLASS_F64 = (
    '[](auto s, auto m) {'
    ' using U = std::decay_t<decltype(s)>;'
    ' using M = std::decay_t<decltype(m)>;'
    ' U exp = (s >> 52) & 0x7FFu;'
    ' U mant = s & 0xFFFFFFFFFFFFFull;'
    ' auto sgn = ((s >> 63) & 1u) != 0u;'
    ' auto qnan = ((s >> 51) & 1u) != 0u;'
    ' auto is_nan = (exp == 0x7FFu) && (mant != 0u);'
    ' auto is_inf = (exp == 0x7FFu) && (mant == 0u);'
    ' auto is_zero = (exp == 0u) && (mant == 0u);'
    ' auto is_den = (exp == 0u) && (mant != 0u);'
    ' auto is_norm = (exp >= 1u) && (exp <= 0x7FEu);'
    ' U cls(0u);'
    ' util::stdx::where(is_nan && !qnan, cls) = 0x001u;'
    ' util::stdx::where(is_nan && qnan, cls) = 0x002u;'
    ' util::stdx::where(is_inf && sgn, cls) = 0x004u;'
    ' util::stdx::where(is_norm && sgn, cls) = 0x008u;'
    ' util::stdx::where(is_den && sgn, cls) = 0x010u;'
    ' util::stdx::where(is_zero && sgn, cls) = 0x020u;'
    ' util::stdx::where(is_zero && !sgn, cls) = 0x040u;'
    ' util::stdx::where(is_den && !sgn, cls) = 0x080u;'
    ' util::stdx::where(is_norm && !sgn, cls) = 0x100u;'
    ' util::stdx::where(is_inf && !sgn, cls) = 0x200u;'
    ' auto cls32 = util::stdx::static_simd_cast<M>(cls);'
    ' return (cls32 & m) != 0u; }'
)
SIMD_VOPC_CLASS_F64: dict[str, str] = {
    'v_cmp_class_f64_vopc': _CMP_CLASS_F64,
}


# VOP3 forms of v_cmp_class. Same classification as the VOPC forms (the functors
# are reused verbatim), but the VOP3 glue additionally applies the abs/neg source
# modifiers to src0's raw bits before classifying, reads the class mask from src1,
# and merges the result into the SGPR-pair dst (inst.vdst.read/write_scalar64)
# rather than VCC. The per-op sign-bit mask (abs clears it, neg flips it) is passed
# to the glue: 0x8000 (f16) / 0x80000000 (f32) share a uint32 lane, 0x8000…0 (f64)
# is 64-bit. f16/f32 go through the 32-bit-value glue; f64 through the 64-bit one.
SIMD_VOP3_CLASS: dict[str, tuple[str, str]] = {
    'v_cmp_class_f16_vop3': ('0x8000u', _CMP_CLASS_F16),
    'v_cmp_class_f32_vop3': ('0x80000000u', _CMP_CLASS_F32),
}
SIMD_VOP3_CLASS_F64: dict[str, str] = {
    'v_cmp_class_f64_vop3': _CMP_CLASS_F64,
}


# --- VOP3 forms of the relational VOPC compares ----------------------------
#
# The VOP3 form of v_cmp_<rel>_<suffix> differs from the VOPC form in three
# ways: (1) it reads src0/src1 (not src0/vsrc1), (2) abs/neg per-source
# modifiers apply to floating-point operands, and (3) the per-lane compare
# result merges into an arbitrary SGPR-pair dst via
# inst.vdst.read/write_scalar64 instead of the fixed VCC. The lane-pack /
# inactive-bit-preservation merge is identical to the VOPC path; this is what
# the VOP3 VOPC glue templates implement.
#
# Integer/bitwise VOPC bodies apply no modifiers, so their functors are the
# same as the VOPC ones (built by _vopc_functor); they go through
# try_execute_vopc_vop3_int_simd (32-bit lane) or
# try_execute_vopc64_vop3_int_simd (64-bit lane).
#
# Floating-point VOPC bodies in VOP3 form apply abs (std::fabs) then neg per
# source on the already-widened/converted operand; the float-bucket tables
# below build new functors that take the post-modifier value (no in-functor
# widen), and the corresponding glue applies the modifier outside before
# calling.
#
# VOP3 fp adds two extra rels vs VOPC: 't' alongside 'tru' (both constant
# always-true), so the table keys span the union of the two.
_VOP3_FLOAT_RELS = _VOPC_FLOAT_RELS + ['t']


def _build_simd_vopc_vop3_int_32() -> dict[str, tuple[str, str]]:
    """VOP3 form of the 32-bit-lane integer VOPC relations (i16/u16/i32/u32).

    Keyed by ``_vop3``; the functor matches the VOPC one (no modifiers).
    """
    table: dict[str, tuple[str, str]] = {}
    for suf in ('i16', 'u16', 'i32', 'u32'):
        lane_t, conv = _VOPC_SUFFIX[suf]
        for rel in _VOPC_INT_RELS:
            table[f'v_cmp_{rel}_{suf}_vop3'] = (lane_t, _vopc_functor(conv, rel))
    return table


def _build_simd_vopc_vop3_int_64() -> dict[str, tuple[str, str]]:
    """VOP3 form of the 64-bit-lane integer VOPC relations (i64/u64). Same
    keying / functor as the VOPC64 path; no modifiers."""
    table: dict[str, tuple[str, str]] = {}
    for suf in ('i64', 'u64'):
        lane_t, conv = _VOPC_SUFFIX[suf]
        for rel in _VOPC_INT_RELS:
            table[f'v_cmp_{rel}_{suf}_vop3'] = (lane_t, _vopc_functor(conv, rel))
    return table


SIMD_VOPC_VOP3_INT_32: dict[str, tuple[str, str]] = _build_simd_vopc_vop3_int_32()
SIMD_VOPC_VOP3_INT_64: dict[str, tuple[str, str]] = _build_simd_vopc_vop3_int_64()


def _build_simd_vopc_vop3_f32() -> dict[str, str]:
    """VOP3 form of the f32 VOPC relations (16 from VOPC + 't' constant).

    Keyed by ``_vop3``; the functor is the same shape as the VOPC f32 one
    (identity operand conversion), and operates on already-modifier-applied
    `native<float>` arguments — the VOP3 fp32 glue applies abs/neg outside the
    functor.
    """
    table: dict[str, str] = {}
    _, conv = _VOPC_SUFFIX['f32']
    for rel in _VOP3_FLOAT_RELS:
        table[f'v_cmp_{rel}_f32_vop3'] = _vopc_functor(conv, rel)
    return table


SIMD_VOPC_VOP3_F32: dict[str, str] = _build_simd_vopc_vop3_f32()


def _build_simd_vopc_vop3_f16() -> dict[str, str]:
    """VOP3 form of the f16 VOPC relations (17 — same set as f32).

    The f16 VOP3 glue widens raw lanes via util::f16_to_f32_simd and applies
    abs/neg on the f32 outside the functor; the functor itself takes
    already-widened `native<float>` arguments, so it is the same functor as
    the f32 path (identity operand conversion).
    """
    table: dict[str, str] = {}
    _, conv = _VOPC_SUFFIX['f32']
    for rel in _VOP3_FLOAT_RELS:
        table[f'v_cmp_{rel}_f16_vop3'] = _vopc_functor(conv, rel)
    return table


SIMD_VOPC_VOP3_F16: dict[str, str] = _build_simd_vopc_vop3_f16()


def _build_simd_vopc_vop3_f64() -> dict[str, str]:
    """VOP3 form of the f64 VOPC relations (17 — same set as f32/f16).

    Lane type `double`; the glue applies abs/neg outside the functor on the
    f64 value, so the functor itself takes already-modified `native<double>`
    arguments and reuses the VOPC64 f64 builder (identity operand conversion).
    """
    table: dict[str, str] = {}
    _, conv = _VOPC_SUFFIX['f64']
    for rel in _VOP3_FLOAT_RELS:
        table[f'v_cmp_{rel}_f64_vop3'] = _vopc_functor(conv, rel)
    return table


SIMD_VOPC_VOP3_F64: dict[str, str] = _build_simd_vopc_vop3_f64()


# --- VOP3 integer/bitwise ternary (3-source) -------------------------------
#
# A handful of integer 3-source VOP3 ops are plain element-wise functions of
# (src0, src1, src2) with no modifiers and no widening: routed through
# try_execute_ternary_vop3_simd<T>. Functor sig: `(native<T> a, native<T> b,
# native<T> c) -> native<T>`. Excluded from this table: the float ternary
# family (fma/mad/mad_mix — needs modifier glue), med3/min3/max3 (NaN /
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
# variant) — both reduce to `(a + b) & 0xFFFFu` / `(a - b) & 0xFFFFu` because
# unsigned 32-bit wrap-around at the low 16 bits is identical to signed/unsigned
# 16-bit wrap. 32-bit forms use the wrap-around add/sub on uint32 lanes;
# signed-vs-unsigned wraps the same way.
SIMD_VOP3_BINARY_INT_EXTRA: dict[str, tuple[str, str]] = {
    # Pack two clamped 32-bit ints into the hi/lo 16-bit halves of the dst.
    # v_cvt_pk_i16_i32: signed-clamp each source to [-32768, 32767]; u16_u32:
    # unsigned-saturate each to 0xFFFF. Pure element-wise, no modifiers.
    'v_cvt_pk_i16_i32_vop3': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' using I = util::native<int32_t>;'
        ' using U = util::native<uint32_t>;'
        ' I lo = util::stdx::static_simd_cast<I>(a);'
        ' util::stdx::where(lo < I(-32768), lo) = I(-32768);'
        ' util::stdx::where(lo > I(32767), lo) = I(32767);'
        ' I hi = util::stdx::static_simd_cast<I>(b);'
        ' util::stdx::where(hi < I(-32768), hi) = I(-32768);'
        ' util::stdx::where(hi > I(32767), hi) = I(32767);'
        ' return ((util::stdx::static_simd_cast<U>(hi) & 0xFFFFu) << 16) |'
        ' (util::stdx::static_simd_cast<U>(lo) & 0xFFFFu); }',
    ),
    'v_cvt_pk_u16_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto lo = a; util::stdx::where(a > 0xFFFFu, lo) = 0xFFFFu;'
        ' auto hi = b; util::stdx::where(b > 0xFFFFu, hi) = 0xFFFFu;'
        ' return (hi << 16) | lo; }',
    ),
    # v_cvt_pk_i16_f32 / v_cvt_pk_u16_f32 are deliberately NOT SIMD-specialized.
    # Their scalar bodies are FLOAT-input converts: bit_cast each src to f32, then
    # clamp the float value (i16: [-32768, 32767]; u16: [0, 65535]) and truncate
    # before packing the low/high 16-bit halves. The integer VOP3 binary glue
    # (try_execute_binary_vop3_simd<uint32_t>) feeds the functor the raw 32-bit
    # lane *bits*, so any int-domain functor here would clamp the float's bit
    # pattern as an integer — a different result for essentially every finite
    # input (e.g. 1.0f -> bits 0x3F800000 -> clamps to 32767 instead of 1). Leave
    # them on the (correct) scalar path. Only the int-domain twins
    # v_cvt_pk_i16_i32 / v_cvt_pk_u16_u32 above are safe to route through this glue.
    # Normalized f32 pack-converts. src read as raw f32 (bit_cast), scale by K,
    # clamp, isnan->0, truncate, pack 16|16. Helpers in util/simd.h.
    # QUIRK: only the no-underscore v_cvt_pknorm_i16_f32 uses the true i16 lambda
    # (K=32767, signed). EVERY OTHER spelling/width — including the
    # underscore-spelled v_cvt_pk_norm_i16_f32 and the _f16 forms (which also read
    # src as raw f32, not f16) — uses the u16 lambda (K=65535, [0,65535]). This
    # mirrors the generated scalar bodies verbatim (a codegen quirk; do not fix).
    'v_cvt_pknorm_i16_f32_vop3': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' using U = util::native<uint32_t>;'
        ' auto lo = util::cvt_pknorm_i16_f32_simd(std::bit_cast<util::native<float>>(a));'
        ' auto hi = util::cvt_pknorm_i16_f32_simd(std::bit_cast<util::native<float>>(b));'
        ' return ((util::stdx::static_simd_cast<U>(hi) & 0xFFFFu) << 16) |'
        ' (util::stdx::static_simd_cast<U>(lo) & 0xFFFFu); }',
    ),
    'v_cvt_pk_norm_i16_f32_vop3': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto lo = util::cvt_pknorm_u16_f32_simd(std::bit_cast<util::native<float>>(a));'
        ' auto hi = util::cvt_pknorm_u16_f32_simd(std::bit_cast<util::native<float>>(b));'
        ' return ((hi & 0xFFFFu) << 16) | (lo & 0xFFFFu); }',
    ),
    'v_cvt_pk_norm_i16_f16_vop3': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto lo = util::cvt_pknorm_u16_f32_simd(std::bit_cast<util::native<float>>(a));'
        ' auto hi = util::cvt_pknorm_u16_f32_simd(std::bit_cast<util::native<float>>(b));'
        ' return ((hi & 0xFFFFu) << 16) | (lo & 0xFFFFu); }',
    ),
    'v_cvt_pknorm_u16_f32_vop3': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto lo = util::cvt_pknorm_u16_f32_simd(std::bit_cast<util::native<float>>(a));'
        ' auto hi = util::cvt_pknorm_u16_f32_simd(std::bit_cast<util::native<float>>(b));'
        ' return ((hi & 0xFFFFu) << 16) | (lo & 0xFFFFu); }',
    ),
    'v_cvt_pk_norm_u16_f32_vop3': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto lo = util::cvt_pknorm_u16_f32_simd(std::bit_cast<util::native<float>>(a));'
        ' auto hi = util::cvt_pknorm_u16_f32_simd(std::bit_cast<util::native<float>>(b));'
        ' return ((hi & 0xFFFFu) << 16) | (lo & 0xFFFFu); }',
    ),
    'v_cvt_pk_norm_u16_f16_vop3': (
        'uint32_t',
        '[](auto a, auto b) {'
        ' auto lo = util::cvt_pknorm_u16_f32_simd(std::bit_cast<util::native<float>>(a));'
        ' auto hi = util::cvt_pknorm_u16_f32_simd(std::bit_cast<util::native<float>>(b));'
        ' return ((hi & 0xFFFFu) << 16) | (lo & 0xFFFFu); }',
    ),
    'v_add_i32_vop3': ('uint32_t', '[](auto a, auto b) { return a + b; }'),
    'v_sub_i32_vop3': ('uint32_t', '[](auto a, auto b) { return a - b; }'),
    'v_add_nc_i32_vop3': ('uint32_t', '[](auto a, auto b) { return a + b; }'),
    'v_add_nc_u32_vop3': ('uint32_t', '[](auto a, auto b) { return a + b; }'),
    'v_sub_nc_i32_vop3': ('uint32_t', '[](auto a, auto b) { return a - b; }'),
    'v_sub_nc_u32_vop3': ('uint32_t', '[](auto a, auto b) { return a - b; }'),
    'v_subrev_nc_u32_vop3': ('uint32_t', '[](auto a, auto b) { return b - a; }'),
    'v_add_i16_vop3': ('uint32_t', '[](auto a, auto b) { return (a + b) & 0xFFFFu; }'),
    'v_sub_i16_vop3': ('uint32_t', '[](auto a, auto b) { return (a - b) & 0xFFFFu; }'),
    'v_add_nc_i16_vop3': (
        'uint32_t',
        '[](auto a, auto b) { return (a + b) & 0xFFFFu; }',
    ),
    'v_add_nc_u16_vop3': (
        'uint32_t',
        '[](auto a, auto b) { return (a + b) & 0xFFFFu; }',
    ),
    'v_sub_nc_i16_vop3': (
        'uint32_t',
        '[](auto a, auto b) { return (a - b) & 0xFFFFu; }',
    ),
    'v_sub_nc_u16_vop3': (
        'uint32_t',
        '[](auto a, auto b) { return (a - b) & 0xFFFFu; }',
    ),
    # 32-bit integer multiply: low 32 bits of the product is just `a * b`
    # (uint32 wrap is identical signed/unsigned for the low half).
    'v_mul_lo_u32_vop3': ('uint32_t', '[](auto a, auto b) { return a * b; }'),
    # High 32 bits of the 32x32 -> 64 multiply via util::mul_hi_{u,i}32_simd
    # (16x16 partial-product decomposition in pure native<uint32_t>). Avoids
    # fixed_size_simd<{u,i}64, N>, which clang + libstdc++ miscompile at every
    # SIMD width (over-native-width 64-bit-lane multiply/shift). Bit-identical to
    # the scalar uint64/int64 `(a * b) >> 32`.
    'v_mul_hi_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b) { return util::mul_hi_u32_simd(a, b); }',
    ),
    'v_mul_hi_i32_vop3': (
        'uint32_t',
        '[](auto a, auto b) { return util::mul_hi_i32_simd(a, b); }',
    ),
    # v_bfm_b32: ((1 << (a & 31)) - 1) << (b & 31). Two shift counts both
    # masked to low 5 bits — same vpsllvd-vs-shl rationale as v_lshl_add_u32.
    'v_bfm_b32_vop3': (
        'uint32_t',
        '[](auto a, auto b) { return ::rocjitsu::amdgpu::simd_bfm_b32(a, b); }',
    ),
    # v_pack_b32_f16: pack two f16 halves into a b32. low16(src0) into the low
    # half, low16(src1) into the high half. The scalar body applies no abs/neg
    # despite the f16 typing (it pre-masks the operands to 16 bits before the
    # shift), and clamp/omod are likewise unused. Pure integer bit-pack.
    'v_pack_b32_f16_vop3': (
        'uint32_t',
        '[](auto a, auto b) { return (a & 0xFFFFu) | ((b & 0xFFFFu) << 16); }',
    ),
    # 16-bit-lane bitwise binary VOP3 ops (RDNA3+; CDNA4 does not decode).
    # The scalar body computes `uint16_t(uint16_t(a) OP uint16_t(b))` and
    # writes the result as a zero-extended uint32; SIMD reproduces with a
    # 32-bit `& 0xFFFFu` mask on the result. No modifiers (integer ops).
    'v_and_b16_vop3': ('uint32_t', '[](auto a, auto b) { return (a & b) & 0xFFFFu; }'),
    'v_or_b16_vop3': ('uint32_t', '[](auto a, auto b) { return (a | b) & 0xFFFFu; }'),
    'v_xor_b16_vop3': ('uint32_t', '[](auto a, auto b) { return (a ^ b) & 0xFFFFu; }'),
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
        'v_cmp_eq_i16_vop3',
        'v_cmp_eq_u16_vop3',
        'v_cmp_ge_i16_vop3',
        'v_cmp_ge_u16_vop3',
        'v_cmp_gt_i16_vop3',
        'v_cmp_gt_u16_vop3',
        'v_cmp_le_i16_vop3',
        'v_cmp_le_u16_vop3',
        'v_cmp_lt_i16_vop3',
        'v_cmp_lt_u16_vop3',
        'v_cmp_ne_i16_vop3',
        'v_cmp_ne_u16_vop3',
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
    # v_bcnt_u32_b32: VOP3-only (no VOP1 twin). The scalar body is a plain
    # std::popcount(src0) -> vdst (src1 is read but unused by this codebase's
    # body), so the unary src0->vdst glue is bit-exact.
    'v_bcnt_u32_b32_vop3': (
        'uint32_t',
        'uint32_t',
        '[](auto a) { return util::popcount_u32_simd(a); }',
    ),
}


# --- VOP3 f64 binary + unary -----------------------------------------------
#
# VOP3 f64 ops carry per-source abs/neg and result omod/clamp modifiers, all
# applied in the f64 domain (apply_vop3_src_mod_f64 / apply_vop3_dst_mod_f64).
# The VOPC table key convention is preserved: dict[name] -> functor string.
#
# For v_max_f64 / v_min_f64, stdx::fmax / stdx::fmin match scalar std::fmax /
# std::fmin for all finite/Inf inputs except (a) NaN payload and (b) signed-zero
# tie (matching the f32 finding) — accepted divergences, with the A/B test
# skipping NaN-input and zero-tie lanes (same convention as v_max_f32 / v_min_f32
# in SIMD_VOP2_BINARY).
# VOP3-only f32 binary ops (no VOP2 twin, so not reachable via the _vop3
# auto-route). Per-source abs/neg + result omod/clamp applied by the f32 binary
# glue; the functor sees already-modified native<float> args. Currently the
# IEEE-2019 maximum/minimum (NaN-propagating, signed-zero-ordered) forms.
SIMD_VOP3_BINARY_FP32: dict[str, str] = {
    'v_maximum_f32_vop3': '[](auto a, auto b) { return util::ieee_maximum_simd(a, b); }',
    'v_minimum_f32_vop3': '[](auto a, auto b) { return util::ieee_minimum_simd(a, b); }',
}

SIMD_VOP3_BINARY_FP64: dict[str, str] = {
    'v_add_f64_vop3': '[](auto a, auto b) { return a + b; }',
    'v_mul_f64_vop3': '[](auto a, auto b) { return a * b; }',
    'v_max_f64_vop3': '[](auto a, auto b) { return util::stdx::fmax(a, b); }',
    'v_min_f64_vop3': '[](auto a, auto b) { return util::stdx::fmin(a, b); }',
    # IEEE-2019 num twins: scalar bodies are the same std::fmax / std::fmin.
    'v_max_num_f64_vop3': '[](auto a, auto b) { return util::stdx::fmax(a, b); }',
    'v_min_num_f64_vop3': '[](auto a, auto b) { return util::stdx::fmin(a, b); }',
    # IEEE-2019 maximum/minimum (NaN-propagating, signed-zero-ordered).
    'v_maximum_f64_vop3': '[](auto a, auto b) { return util::ieee_maximum_simd(a, b); }',
    'v_minimum_f64_vop3': '[](auto a, auto b) { return util::ieee_minimum_simd(a, b); }',
}

# Plain f64 unary: scalar bodies are std::ceil / std::floor / std::trunc /
# std::nearbyint applied to the (modifier-applied) double, then omod/clamp on
# the result. util::{ceil,floor,trunc,rndne}_simd wrap the stdx native<double>
# rounding primitives and repair the two edge cases libstdc++ gets wrong at
# narrow widths (sign-of-zero on a zero-magnitude result, NaN-payload
# preservation), so they are bit-identical to the scalar libm calls for every
# finite / Inf / NaN / signed-zero input.
SIMD_VOP3_UNARY_FP64: dict[str, str] = {
    'v_ceil_f64_vop3': '[](auto a) { return util::ceil_simd(a); }',
    'v_floor_f64_vop3': '[](auto a) { return util::floor_simd(a); }',
    'v_trunc_f64_vop3': '[](auto a) { return util::trunc_simd(a); }',
    'v_rndne_f64_vop3': '[](auto a) { return util::rndne_simd(a); }',
    # sqrt_f64 is correctly-rounded IEEE (scalar uses transcendental::sqrt_f64
    # which is `std::sqrt` after NaN/negative guards); stdx::sqrt matches.
    'v_sqrt_f64_vop3': ('[](auto a) { return util::sqrt_f64_simd(a); }'),
    # v_fract_f64: scalar = v - std::floor(v); util::floor_simd matches
    # std::floor bit-exact incl. sign-of-zero (NaN-floor(NaN) = NaN; NaN result
    # skipped by the test like any other NaN-result lane).
    'v_fract_f64_vop3': '[](auto a) { return a - util::floor_simd(a); }',
    # frexp mantissa: same functor as the VOP1 form; the f64 unary FP glue applies
    # abs/neg on the source and omod/clamp on the mantissa, matching the scalar.
    'v_frexp_mant_f64_vop3': '[](auto a) { return util::frexp_mant_f64_simd(a); }',
    # v_rcp_f64 / v_rsq_f64: scalar uses transcendental::*_f64 with explicit
    # NaN passthrough, ±0 -> copysign(Inf, x), ±Inf -> copysign(0, x); negative
    # rsq inputs -> qNaN. Plain 1.0 / x and 1.0 / sqrt match the IEEE result
    # for all non-NaN inputs; NaN-result lanes are skipped by the A/B test.
    'v_rcp_f64_vop3': '[](auto a) { return util::native<double>(1.0) / a; }',
    'v_rsq_f64_vop3': '[](auto a) { return util::native<double>(1.0) / util::stdx::sqrt(a); }',
    # NOTE: v_mov_b64 is deliberately NOT here. It is an integer 64-bit bit-move;
    # OMOD/CLAMP are float-only, so its generated VOP3 scalar body is a raw copy
    # that ignores them. Routing it through the f64 modifier glue made the SIMD
    # path apply clamp/omod the scalar never does (clamp=1: scalar keeps the raw
    # bits vs simd clamps to 1.0). With no entry here it stays scalar (a 64-bit
    # register copy gains nothing from vectorization anyway).
}


# VOP3 f16 unary: widen f16->f32, apply abs/neg, op, omod/clamp, narrow back.
# Rounding ops (ceil/floor/trunc/rndne) have no FTZ. sqrt is also no-FTZ
# (transcendental::sqrt_f32 keeps denormals). The four transcendentals
# (rcp/rsq/exp/log) reuse util::*_f32_simd which already wraps the scalar
# transcendental::flush_denorm_f32 carve-outs (FTZ input + matching ±0/Inf
# blends + NaN-passthrough), so the f16 scalar
# f32_to_f16(transcendental::op_f32(f16_to_f32(...))) maps directly.
SIMD_VOP3_UNARY_FP16: dict[str, str] = {
    'v_ceil_f16_vop3': '[](auto a) { return util::ceil_simd(a); }',
    'v_floor_f16_vop3': '[](auto a) { return util::floor_simd(a); }',
    'v_trunc_f16_vop3': '[](auto a) { return util::trunc_simd(a); }',
    'v_rndne_f16_vop3': '[](auto a) { return util::rndne_simd(a); }',
    'v_sqrt_f16_vop3': (
        '[](auto a) {'
        ' auto r = util::stdx::sqrt(a);'
        ' util::stdx::where(util::stdx::isnan(a), r) = a;'
        ' util::stdx::where(a < 0.0f, r) = std::numeric_limits<float>::quiet_NaN();'
        ' return r; }'
    ),
    'v_rcp_f16_vop3': '[](auto a) { return util::rcp_f32_simd(a); }',
    'v_rsq_f16_vop3': '[](auto a) { return util::rsq_f32_simd(a); }',
    'v_exp_f16_vop3': '[](auto a) { return util::exp_f32_simd(a); }',
    'v_log_f16_vop3': '[](auto a) { return util::log_f32_simd(a); }',
    # v_fract_f16: x - floor(x) in the widened f32 domain (the glue widens/narrows
    # and applies abs/neg/omod/clamp), mirroring the covered v_fract_f16_vop1.
    'v_fract_f16_vop3': '[](auto a) { return a - util::floor_simd(a); }',
}


# VOP3 frexp twins routed through the EXISTING unary FP glue. Their generated
# scalar bodies apply src0 abs/neg, compute the frexp exp/mantissa, then apply
# result omod/clamp on the (float-domain) value and bit_cast / f32_to_f16 it to
# the dst — exactly what the f32 / f16 unary FP glue produces. So only the
# functor differs from the proven VOP1 forms.
#   route 'fp16' -> ROCJITSU_TRY_SIMD_VOP3_UNARY_FP16   (f16<->f32 widen glue)
#   route 'fp32' -> ROCJITSU_TRY_SIMD_VOP3_UNARY_FP(float32_t, float32_t, ...)
# frexp_exp returns the uint32 two's-complement exponent (±0/Inf/NaN -> 0); the
# static_simd_cast<float> reproduces the scalar `(float)(uint32_t)exp` exactly
# (negative exp -> large float -> f16 +Inf, like scalar). frexp_mant_f32_simd is
# the already-proven VOP1 helper. (v_frexp_exp_i32_f64 needs a new f64->f32
# mixed-width glue and stays in _VOP3_UNARY_SKIP / scalar for now.)
SIMD_VOP3_FREXP_FP: dict[str, tuple[str, str]] = {
    'v_frexp_mant_f16_vop3': (
        'fp16',
        '[](auto a) { return util::frexp_mant_f32_simd(std::bit_cast<util::native<uint32_t>>(a)); }',
    ),
    'v_frexp_exp_i16_f16_vop3': (
        'fp16',
        '[](auto a) { return util::stdx::static_simd_cast<util::native<float>>('
        'util::frexp_exp_f32_simd(std::bit_cast<util::native<uint32_t>>(a))); }',
    ),
    'v_frexp_exp_i32_f32_vop3': (
        'fp32',
        '[](auto a) { return util::stdx::static_simd_cast<util::native<float>>('
        'util::frexp_exp_f32_simd(std::bit_cast<util::native<uint32_t>>(a))); }',
    ),
    # f64 source -> float(exp) dst. Mixed-width (read64 / narrow32 store): the new
    # cvt-vop3 f64->b32 glue applies the f64 src abs/neg + float omod/clamp.
    # frexp_exp_f64_simd returns the int32 exp in the low 32 bits of each 64-bit
    # lane; narrow to uint32 (scalar `(uint32_t)exp`) then to float (scalar
    # unsigned `(float)(uint32_t)exp`). The intermediate uint32 narrow is
    # load-bearing — a direct uint64->float would convert the full 64-bit value.
    'v_frexp_exp_i32_f64_vop3': (
        'fp64cvt',
        '[](auto s) { return util::stdx::static_simd_cast<util::narrow32<float32_t>>('
        'util::stdx::static_simd_cast<util::narrow32<uint32_t>>('
        'util::frexp_exp_f64_simd(s))); }',
    ),
}


# --- VOP3 floating-point ternary (FMA / MAD family) ------------------------
#
# v_fma_*: util::stdx::fma (fused multiply-add, single-rounded). v_fmac/v_mac:
# same body (the scalar generator emits std::fma for both because of dst-
# accumulate semantics — src2 == vdst). v_mad: non-fused `a * b + c`. NaN-input
# divergence between stdx::fma and std::fma (gcc-13 packed FMA picks a
# different NaN operand to quiet) is accepted, same as the existing VOP2
# ternary FMA slice — the A/B test skips NaN-input lanes.
SIMD_VOP3_TERNARY_FP32: dict[str, str] = {
    'v_fma_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
    'v_mad_f32_vop3': '[](auto a, auto b, auto c) { return a * b + c; }',
    'v_mad_legacy_f32_vop3': '[](auto a, auto b, auto c) { return a * b + c; }',
    # min3/max3/med3 (f32): the scalar body composes std::fmax/std::fmin
    # (max3 = fmax(fmax(a,b),c); min3 = fmin(fmin(a,b),c);
    # med3 = fmax(fmin(fmax(a,b),c), fmin(a,b))). stdx::fmax/fmin match scalar
    # for all finite/Inf inputs except NaN-payload and signed-zero tie (same
    # accepted carve-out as v_max_f32 / v_min_f32; the A/B test skips NaN-input
    # lanes and uses no ±0 inputs). omod/clamp applied by the glue.
    # v_fma_dx9_zero_f32: the scalar body is a plain fused multiply-add (the
    # DX9 zero-multiply special-case is not applied to the FMA form).
    'v_fma_dx9_zero_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
    'v_max3_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmax(util::stdx::fmax(a, b), c); }',
    'v_min3_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmin(util::stdx::fmin(a, b), c); }',
    'v_med3_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmax(util::stdx::fmin(util::stdx::fmax(a, b), c), util::stdx::fmin(a, b)); }',
    # minmax = max(min(a,b),c); maxmin = min(max(a,b),c). (RDNA3+.)
    'v_minmax_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmax(util::stdx::fmin(a, b), c); }',
    'v_maxmin_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmin(util::stdx::fmax(a, b), c); }',
    # IEEE-2019 num twins (gfx1250/rdna4): identical fmax/fmin compositions as
    # the legacy max3/min3/minmax/maxmin bodies above.
    'v_max3_num_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmax(util::stdx::fmax(a, b), c); }',
    'v_min3_num_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmin(util::stdx::fmin(a, b), c); }',
    'v_minmax_num_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmax(util::stdx::fmin(a, b), c); }',
    'v_maxmin_num_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmin(util::stdx::fmax(a, b), c); }',
    # IEEE-2019 maximum/minimum 3-input + combined forms. The scalar bodies are
    # the exact nested composition of the binary maximum/minimum (NaN-propagating,
    # signed-zero-ordered) — see util::ieee_{maximum,minimum}_simd.
    'v_maximum3_f32_vop3': '[](auto a, auto b, auto c) { return util::ieee_maximum_simd(util::ieee_maximum_simd(a, b), c); }',
    'v_minimum3_f32_vop3': '[](auto a, auto b, auto c) { return util::ieee_minimum_simd(util::ieee_minimum_simd(a, b), c); }',
    'v_maximumminimum_f32_vop3': '[](auto a, auto b, auto c) { return util::ieee_minimum_simd(util::ieee_maximum_simd(a, b), c); }',
    'v_minimummaximum_f32_vop3': '[](auto a, auto b, auto c) { return util::ieee_maximum_simd(util::ieee_minimum_simd(a, b), c); }',
    # Cubemap face ops: ternary f32, per-source abs/neg + result omod/clamp via
    # the glue; the face-selection is a bit-exact where-cascade (helpers in
    # simd_glue.h). cubema is inline (exact ×2 of fmax of abs).
    'v_cubeid_f32_vop3': '[](auto x, auto y, auto z) { return util::cube_id_f32_simd(x, y, z); }',
    'v_cubema_f32_vop3': '[](auto x, auto y, auto z) { return 2.0f * util::stdx::fmax(util::stdx::abs(x), util::stdx::fmax(util::stdx::abs(y), util::stdx::abs(z))); }',
    'v_cubesc_f32_vop3': '[](auto x, auto y, auto z) { return util::cube_sc_f32_simd(x, y, z); }',
    'v_cubetc_f32_vop3': '[](auto x, auto y, auto z) { return util::cube_tc_f32_simd(x, y, z); }',
    # v_div_fixup_f32: per-AMD-spec `else if` cascade selecting the result
    # among NaN/Inf/zero copysign cases. Lives as a helper in simd_glue.h
    # (div_fixup_f32_simd) — bit-exact match to the scalar body's predicate
    # tree applied lowest-priority-first so higher-priority `where` blends
    # overwrite. Omod/clamp DO apply (scalar applies them at end).
    'v_div_fixup_f32_vop3': (
        '[](auto p, auto b, auto c) { return ::rocjitsu::amdgpu::div_fixup_f32_simd(p, b, c); }'
    ),
    # v_div_fixup_f16 / _legacy_f16: despite the _f16 name, the generated CDNA4
    # scalar body reads/writes the operands as raw f32 (std::bit_cast<float>, not
    # f16_to_f32) — bit-identical to the f32 div_fixup body — so it routes through
    # the same f32 ternary glue + div_fixup_f32_simd cascade.
    'v_div_fixup_f16_vop3': (
        '[](auto p, auto b, auto c) { return ::rocjitsu::amdgpu::div_fixup_f32_simd(p, b, c); }'
    ),
    'v_div_fixup_legacy_f16_vop3': (
        '[](auto p, auto b, auto c) { return ::rocjitsu::amdgpu::div_fixup_f32_simd(p, b, c); }'
    ),
}

# f16 ternary — widen each src to f32, op in f32, narrow back. Same NaN
# carve-out as the f32 path.
SIMD_VOP3_TERNARY_FP16: dict[str, str] = {
    'v_fma_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
    'v_mad_f16_vop3': '[](auto a, auto b, auto c) { return a * b + c; }',
    'v_mad_legacy_f16_vop3': '[](auto a, auto b, auto c) { return a * b + c; }',
    'v_fma_legacy_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
    # min3/max3/med3 (f16): widened to f32 by the glue; same fmax/fmin
    # composition as the f32 forms (see SIMD_VOP3_TERNARY_FP32).
    'v_max3_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmax(util::stdx::fmax(a, b), c); }',
    'v_min3_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmin(util::stdx::fmin(a, b), c); }',
    'v_med3_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmax(util::stdx::fmin(util::stdx::fmax(a, b), c), util::stdx::fmin(a, b)); }',
    'v_minmax_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmax(util::stdx::fmin(a, b), c); }',
    'v_maxmin_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmin(util::stdx::fmax(a, b), c); }',
    # IEEE-2019 num twins (f16): same fmax/fmin compositions, widened by the glue.
    'v_max3_num_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmax(util::stdx::fmax(a, b), c); }',
    'v_min3_num_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmin(util::stdx::fmin(a, b), c); }',
    'v_minmax_num_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmax(util::stdx::fmin(a, b), c); }',
    'v_maxmin_num_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fmin(util::stdx::fmax(a, b), c); }',
    # IEEE-2019 maximum/minimum 3-input + combined (f16; widened to f32 by glue).
    'v_maximum3_f16_vop3': '[](auto a, auto b, auto c) { return util::ieee_maximum_simd(util::ieee_maximum_simd(a, b), c); }',
    'v_minimum3_f16_vop3': '[](auto a, auto b, auto c) { return util::ieee_minimum_simd(util::ieee_minimum_simd(a, b), c); }',
    'v_maximumminimum_f16_vop3': '[](auto a, auto b, auto c) { return util::ieee_minimum_simd(util::ieee_maximum_simd(a, b), c); }',
    'v_minimummaximum_f16_vop3': '[](auto a, auto b, auto c) { return util::ieee_maximum_simd(util::ieee_minimum_simd(a, b), c); }',
}

# f64 ternary FMA.
SIMD_VOP3_TERNARY_FP64: dict[str, str] = {
    'v_fma_f64_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
    # v_div_fixup_f64: 64-bit-lane div_fixup cascade (same shape as f32, see
    # SIMD_VOP3_TERNARY_FP32 above).
    'v_div_fixup_f64_vop3': (
        '[](auto p, auto b, auto c) { return ::rocjitsu::amdgpu::div_fixup_f64_simd(p, b, c); }'
    ),
}

# --- VOP3 dst-accumulate FMA / MAC (vdst is the accumulator) ----------------
#
# v_fmac / v_mac per-isa classes only initialize src0+src1+vdst; the third FMA
# operand IS vdst (no src2 Operand). The accumulate-form glue reads inst.vdst
# as the third operand and applies abs/neg only to src0/src1 (per scalar body).
# NaN payload divergence accepted, same as the non-accumulate ternary slice.
SIMD_VOP3_FMAC_FP32: dict[str, str] = {
    'v_fmac_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
    'v_mac_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
    'v_fmac_dx9_zero_f32_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
}
SIMD_VOP3_FMAC_FP16: dict[str, str] = {
    'v_fmac_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
    'v_mac_f16_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
}
SIMD_VOP3_FMAC_FP64: dict[str, str] = {
    'v_fmac_f64_vop3': '[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }',
}

# --- VOP3 ldexp (mixed-width: fp src0 + int32 src1 exp) --------------------
#
# stdx::ldexp on native<float|double> with same-size int simd is bit-exact to
# std::ldexp for every input including NaN (proven in the v_ldexp_f16 VOP2
# slice).
SIMD_VOP3_LDEXP_FP32: dict[str, str] = {
    # stdx::ldexp wants the integer arg as fixed_size_simd<int, size> matching
    # the float abi, not the native<int32_t> the operand reader returns.
    'v_ldexp_f32_vop3': (
        '[](auto a, auto e) {'
        ' return util::stdx::ldexp(a, util::stdx::static_simd_cast<'
        'util::stdx::fixed_size_simd<int, util::native<float>::size()>>(e)); }'
    ),
}
SIMD_VOP3_LDEXP_FP64: dict[str, str] = {
    # narrow32<int32_t> is already an 8-wide fixed_size_simd; just re-cast to
    # the int-typed equivalent so stdx::ldexp accepts the matching abi.
    'v_ldexp_f64_vop3': (
        '[](auto a, auto e) {'
        ' return util::stdx::ldexp(a, util::stdx::static_simd_cast<'
        'util::stdx::fixed_size_simd<int, util::native_width64>>(e)); }'
    ),
}


SIMD_VOP3_TERNARY_INT: dict[str, tuple[str, str]] = {
    # Sum-of-absolute-differences (byte / 16-bit / 32-bit) + accumulate src2.
    # Pure integer byte SWAR (see util::sad_bytes_u32_simd etc); no modifiers.
    'v_sad_u8_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return c + util::sad_bytes_u32_simd(a, b); }',
    ),
    'v_sad_hi_u8_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return (util::sad_bytes_u32_simd(a, b) << 16) + c; }',
    ),
    'v_sad_u16_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' using U = util::native<uint32_t>;'
        ' U al = a & 0xFFFFu, ah = (a >> 16) & 0xFFFFu;'
        ' U bl = b & 0xFFFFu, bh = (b >> 16) & 0xFFFFu;'
        ' U dl = al - bl; util::stdx::where(bl > al, dl) = bl - al;'
        ' U dh = ah - bh; util::stdx::where(bh > ah, dh) = bh - ah;'
        ' return c + dl + dh; }',
    ),
    'v_sad_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' auto d = a - b; util::stdx::where(b > a, d) = b - a; return c + d; }',
    ),
    'v_msad_u8_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return c + util::msad_bytes_u32_simd(a, b); }',
    ),
    'v_lerp_u8_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return util::lerp_u8_simd(a, b, c); }',
    ),
    'v_add3_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return a + b + c; }',
    ),
    'v_or3_b32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return a | b | c; }',
    ),
    'v_xor3_b32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return a ^ b ^ c; }',
    ),
    'v_xad_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return (a ^ b) + c; }',
    ),
    'v_and_or_b32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return (a & b) | c; }',
    ),
    'v_lshl_or_b32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return ::rocjitsu::amdgpu::simd_lshl_u32(a, b) | c; }',
    ),
    # v_mad_i32_i24: low-24 sign-extended a, b -> int32 multiply (low 32 of the
    # 48-bit product, identical signed/unsigned for the low half) + int32(c).
    'v_mad_i32_i24_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return ::rocjitsu::amdgpu::simd_mad_i24_u32(a, b, c); }',
    ),
    # v_mad_u32_u24: low-24 mask a, b, multiply, add c.
    'v_mad_u32_u24_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return (a & 0x00FFFFFFu) * (b & 0x00FFFFFFu) + c; }',
    ),
    # v_alignbit_b32: low 32 of ((u64(a) << 32) | b) >> (c & 31). Per-lane
    # variable shift on a 64-bit-lane fixed_size_simd<u64> — proven on the
    # widening mul_hi pattern; shift count is masked to [0, 31] so vpsrlvq
    # and scalar shr agree on the result.
    'v_alignbit_b32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' using U64 = util::stdx::fixed_size_simd<uint64_t, util::native<uint32_t>::size()>;'
        ' auto va = util::stdx::static_simd_cast<U64>(a);'
        ' auto vb = util::stdx::static_simd_cast<U64>(b);'
        ' auto val = (va << 32) | vb;'
        ' auto sh = util::stdx::static_simd_cast<U64>(c & 31u);'
        ' return util::stdx::static_simd_cast<util::native<uint32_t>>(val >> sh); }',
    ),
    # v_alignbyte_b32: same widen but shift count is (c & 3) * 8 = {0,8,16,24}.
    'v_alignbyte_b32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' using U64 = util::stdx::fixed_size_simd<uint64_t, util::native<uint32_t>::size()>;'
        ' auto va = util::stdx::static_simd_cast<U64>(a);'
        ' auto vb = util::stdx::static_simd_cast<U64>(b);'
        ' auto val = (va << 32) | vb;'
        ' auto sh = util::stdx::static_simd_cast<U64>((c & 3u) * 8u);'
        ' return util::stdx::static_simd_cast<util::native<uint32_t>>(val >> sh); }',
    ),
    # The shift count is masked to the low 5 bits to match the scalar body's
    # x86 `shl` semantics (which mask cl to 5 bits) — stdx's `<<` on native<u32>
    # lowers to vpsllvd, which zeros out lanes where the count is >= 32 rather
    # than masking, so SIMD without the explicit `& 31u` diverges from scalar
    # whenever src1 (or src2 below) is >= 32 (verified empirically on the test
    # value 0x80000000). The scalar body's `<<` is C++ UB at those counts; the
    # masked SIMD form reproduces the x86 scalar result for every value of the
    # shift operand on this host.
    'v_lshl_add_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return ::rocjitsu::amdgpu::simd_lshl_u32(a, b) + c; }',
    ),
    'v_add_lshl_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return ::rocjitsu::amdgpu::simd_lshl_u32(a + b, c); }',
    ),
    'v_bfi_b32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return (a & b) | (~a & c); }',
    ),
    # Byte permute: each output byte selected from the 8 bytes of {src0:src1} by
    # a selector byte of src2 (see util::perm_b32_simd). Pure byte gather.
    'v_perm_b32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return util::perm_b32_simd(a, b, c); }',
    ),
    # min3/max3/med3 (integer). The scalar body uses std::max/std::min for the
    # 16/32-bit forms (and the minmax/maxmin scalar bodies use std::fmax/fmin on
    # *integers* — order-preserving and exactly representable, so the integer
    # min/max here matches for every input; this is the correct SIMD form, NOT
    # the std::fmin-on-int pattern flagged in project_pr6470_review_findings).
    # 32-bit: signed lane type for i32 (signed compare), unsigned for u32.
    # max3 = max(max(a,b),c); min3 = min(min(a,b),c);
    # med3 = max(min(max(a,b),c), min(a,b)).
    'v_max3_i32_vop3': (
        'int32_t',
        '[](auto a, auto b, auto c) { return util::stdx::max(util::stdx::max(a, b), c); }',
    ),
    'v_min3_i32_vop3': (
        'int32_t',
        '[](auto a, auto b, auto c) { return util::stdx::min(util::stdx::min(a, b), c); }',
    ),
    'v_med3_i32_vop3': (
        'int32_t',
        '[](auto a, auto b, auto c) {'
        ' return util::stdx::max(util::stdx::min(util::stdx::max(a, b), c),'
        ' util::stdx::min(a, b)); }',
    ),
    'v_max3_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return util::stdx::max(util::stdx::max(a, b), c); }',
    ),
    'v_min3_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return util::stdx::min(util::stdx::min(a, b), c); }',
    ),
    'v_med3_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' return util::stdx::max(util::stdx::min(util::stdx::max(a, b), c),'
        ' util::stdx::min(a, b)); }',
    ),
    # NOTE: integer v_minmax/v_maxmin i32/u32 are intentionally NOT wired. Unlike
    # min3/max3 (whose scalar uses std::max/std::min), the minmax/maxmin scalar
    # bodies use std::fmax/std::fmin on *ints* and write the double result back
    # through a double->uint32 conversion that saturates negatives to 0xFFFFFFFF
    # (x86 cvttsd2si overflow). That scalar is UB and the integer SIMD min/max
    # cannot match it — left scalar-authoritative (see project_pr6470 findings).
    # 16-bit: the scalar body sign-/zero-extends the low 16 bits, takes the
    # min/max, then truncates to uint16 and zero-extends to uint32. SIMD lane is
    # the raw uint32; i16 sign-extends via (cast<int32> << 16) >> 16 (same
    # pattern as v_mad_i32_i24's 24-bit extend), u16 masks & 0xFFFF; the result
    # is masked back to the low 16 bits.
    'v_max3_i16_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' using I = util::native<int32_t>;'
        ' auto sa = (util::stdx::static_simd_cast<I>(a) << 16) >> 16;'
        ' auto sb = (util::stdx::static_simd_cast<I>(b) << 16) >> 16;'
        ' auto sc = (util::stdx::static_simd_cast<I>(c) << 16) >> 16;'
        ' return util::stdx::static_simd_cast<util::native<uint32_t>>('
        'util::stdx::max(util::stdx::max(sa, sb), sc)) & 0xFFFFu; }',
    ),
    'v_min3_i16_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' using I = util::native<int32_t>;'
        ' auto sa = (util::stdx::static_simd_cast<I>(a) << 16) >> 16;'
        ' auto sb = (util::stdx::static_simd_cast<I>(b) << 16) >> 16;'
        ' auto sc = (util::stdx::static_simd_cast<I>(c) << 16) >> 16;'
        ' return util::stdx::static_simd_cast<util::native<uint32_t>>('
        'util::stdx::min(util::stdx::min(sa, sb), sc)) & 0xFFFFu; }',
    ),
    'v_med3_i16_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' using I = util::native<int32_t>;'
        ' auto sa = (util::stdx::static_simd_cast<I>(a) << 16) >> 16;'
        ' auto sb = (util::stdx::static_simd_cast<I>(b) << 16) >> 16;'
        ' auto sc = (util::stdx::static_simd_cast<I>(c) << 16) >> 16;'
        ' return util::stdx::static_simd_cast<util::native<uint32_t>>('
        'util::stdx::max(util::stdx::min(util::stdx::max(sa, sb), sc),'
        ' util::stdx::min(sa, sb))) & 0xFFFFu; }',
    ),
    'v_max3_u16_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' auto sa = a & 0xFFFFu; auto sb = b & 0xFFFFu; auto sc = c & 0xFFFFu;'
        ' return util::stdx::max(util::stdx::max(sa, sb), sc) & 0xFFFFu; }',
    ),
    'v_min3_u16_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' auto sa = a & 0xFFFFu; auto sb = b & 0xFFFFu; auto sc = c & 0xFFFFu;'
        ' return util::stdx::min(util::stdx::min(sa, sb), sc) & 0xFFFFu; }',
    ),
    'v_med3_u16_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' auto sa = a & 0xFFFFu; auto sb = b & 0xFFFFu; auto sc = c & 0xFFFFu;'
        ' return util::stdx::max(util::stdx::min(util::stdx::max(sa, sb), sc),'
        ' util::stdx::min(sa, sb)) & 0xFFFFu; }',
    ),
    # v_mad_i32_i16: sign-extend the low 16 bits of src0/src1 to int32, multiply
    # (the int16 product fits in int32), add the full int32 src2, store as uint32
    # — matches the scalar `(int16)a * (int16)b + (int32)c`.
    'v_mad_i32_i16_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) {'
        ' auto sa = ::rocjitsu::amdgpu::simd_sign_extend_u32(a, 16);'
        ' auto sb = ::rocjitsu::amdgpu::simd_sign_extend_u32(b, 16);'
        ' return sa * sb + c; }',
    ),
    # v_mad_u32_u16: zero-extend low 16 of src0/src1, multiply, add full uint32 c.
    'v_mad_u32_u16_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return (a & 0xFFFFu) * (b & 0xFFFFu) + c; }',
    ),
    # v_mad_i16 / v_mad_u16 need true16 VOP3 op_sel handling for each source
    # and for the destination half, so their executors stay scalar.
    # v_bfe_u32: bitfield extract. off = src1 & 31, w = src2 & 31, result =
    # (src >> off) & ((1<<w)-1). The scalar `if (w==0) return 0` is redundant —
    # ((1u<<w)-1) is already 0 for w==0 — and `w>=32` is dead since w is masked
    # to 31, so the SIMD form is branchless. Variable logical shift (vpsrlvd /
    # vpsllvd) matches scalar for every count in [0,31].
    'v_bfe_u32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return ::rocjitsu::amdgpu::simd_bfe_u32(a, b, c); }',
    ),
    # v_bfe_i32: signed bitfield extract. Same off/width masking; the field is
    # extracted with an arithmetic shift (vpsravd, so off+w>32 sign-fill matches
    # the scalar int32 `src >> off`) then sign-extended from bit w-1 via the
    # branchless `(x ^ s) - s` trick (s = 1<<(w-1) = (mask+1)>>1, which is 0 for
    # w==0 so the result is 0, matching the scalar early return).
    'v_bfe_i32_vop3': (
        'uint32_t',
        '[](auto a, auto b, auto c) { return ::rocjitsu::amdgpu::simd_bfe_i32(a, b, c); }',
    ),
}


# --- VOP3 64-bit reverse shifts --------------------------------------------
#
# v_lshlrev_b64 / v_lshrrev_b64 / v_ashrrev_i64: shift the 64-bit src1 by the
# 32-bit src0 (masked to [0,63]). Routed through try_execute_shift64_vop3_simd,
# whose functor receives (native<uint64_t> value, native<uint64_t> shift) with
# the count already widened+masked. Logical shifts are plain `<< / >>` on the
# uint64 lane; the signed (arithmetic) form casts through int64.
SIMD_SHIFT64_VOP3: dict[str, str] = {
    'v_lshlrev_b64_vop3': '[](auto v, auto sh) { return ::rocjitsu::amdgpu::simd_lshl_u64(v, sh); }',
    'v_lshrrev_b64_vop3': '[](auto v, auto sh) { return ::rocjitsu::amdgpu::simd_lshr_u64(v, sh); }',
    'v_ashrrev_i64_vop3': (
        '[](auto v, auto sh) {' ' return ::rocjitsu::amdgpu::simd_ashr_i64(v, sh); }'
    ),
}

# v_lshl_add_u64: (src0 << (src1 & 63)) + src2 — 64-bit value/addend, 32-bit
# shift. Fixed op, dedicated glue (operand widths differ from the rev-shifts).
SIMD_LSHL_ADD_U64: set[str] = {'v_lshl_add_u64_vop3'}

# --- VOP3 wide 32x32->64 multiply-add (v_mad_u64_u32 / v_mad_i64_i32) -------
#
# 32-bit src0/src1 multiplicands widened to 64 (zero-/sign-extend), 64-bit
# low-half product, plus the 64-bit src2 addend, with the carry/overflow mask
# written to sdst. Routed through
# try_execute_mad_wide64_vop3_simd; the functor receives the two narrow operands
# and the 64-bit addend and returns SimdCarry<native<uint64_t>, mask>.
SIMD_MAD_WIDE64: dict[str, str] = {
    'v_mad_u64_u32_vop3': (
        '[](auto a, auto b, auto c) {'
        ' auto product = util::stdx::static_simd_cast<util::native<uint64_t>>(a)'
        ' * util::stdx::static_simd_cast<util::native<uint64_t>>(b);'
        ' auto result = product + c;'
        ' return make_simd_carry(result, result < product); }'
    ),
    'v_mad_i64_i32_vop3': (
        '[](auto a, auto b, auto c) {'
        ' auto wa = util::stdx::static_simd_cast<util::native<uint64_t>>('
        'util::stdx::static_simd_cast<util::native<int64_t>>('
        'util::stdx::static_simd_cast<util::narrow32<int32_t>>(a)));'
        ' auto wb = util::stdx::static_simd_cast<util::native<uint64_t>>('
        'util::stdx::static_simd_cast<util::native<int64_t>>('
        'util::stdx::static_simd_cast<util::narrow32<int32_t>>(b)));'
        ' auto product = wa * wb;'
        ' auto result = product + c;'
        ' return make_simd_carry(result,'
        ' ((~(product ^ c) & (product ^ result)) & (1ULL << 63)) != 0); }'
    ),
}

# --- VOP3 carry binary (carry-out -> SGPR sdst, carry-in <- SGPR src2) ------
#
# Same SimdCarry functors as the VOP2 carry family, keyed by _vop3, split by
# whether the form has a carry-in (and thus a src2 member): the carry-OUT forms
# (add_co/sub_co/subrev_co) have no src2 and route through the _co glue; the
# carry-IN forms (addc/subb/subbrev) read src2 and route through the _cin glue.
_VOP3_CARRY_CIN_NAMES = {'v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'}
SIMD_VOP3_CARRY_CO: dict[str, str] = {
    name.replace('_vop2', '_vop3'): functor
    for name, functor in SIMD_VOP2_CARRY.items()
    if name.replace('_vop2', '') not in _VOP3_CARRY_CIN_NAMES
}
SIMD_VOP3_CARRY_CIN: dict[str, str] = {
    name.replace('_vop2', '_vop3'): functor
    for name, functor in SIMD_VOP2_CARRY.items()
    if name.replace('_vop2', '') in _VOP3_CARRY_CIN_NAMES
}
# RDNA-only carry-in aliases (add_co_ci / sub_co_ci / subrev_co_ci): same
# per-lane add/sub-with-carry functor as the CDNA addc / subb / subbrev forms;
# the decoder binds src2/sdst to VCC, but the body is uniform.
SIMD_VOP3_CARRY_CIN.update(
    {
        'v_add_co_ci_u32_vop3': SIMD_VOP2_CARRY['v_addc_co_u32_vop2'],
        'v_sub_co_ci_u32_vop3': SIMD_VOP2_CARRY['v_subb_co_u32_vop2'],
        'v_subrev_co_ci_u32_vop3': SIMD_VOP2_CARRY['v_subbrev_co_u32_vop2'],
    }
)


def simd_probe_line(template_name: str) -> str | None:
    """Return the SIMD fast-path probe block for a kernel, or None."""
    if template_name in SIMD_VOP3_TRUE16_UNSAFE:
        return None
    if template_name in SIMD_VOP2_CNDMASK:
        return '  ROCJITSU_TRY_SIMD_VOP2_CNDMASK();'
    if template_name in SIMD_VOP3_CNDMASK:
        return '  ROCJITSU_TRY_SIMD_VOP3_CNDMASK();'
    if template_name in SIMD_VOP3_CNDMASK_B16:
        return '  ROCJITSU_TRY_SIMD_VOP3_CNDMASK_B16();'
    spec2 = SIMD_VOP2_BINARY.get(template_name)
    if spec2 is not None:
        cpp_t, cpp_op = spec2
        return f'  ROCJITSU_TRY_SIMD_VOP2_BINARY({cpp_t}, {cpp_op});'
    spec1 = SIMD_VOP1_UNARY.get(template_name)
    if spec1 is not None:
        cpp_tin, cpp_tout, cpp_op = spec1
        return f'  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {cpp_op});'
    specc = SIMD_VOP2_CARRY.get(template_name)
    if specc is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP2_CARRY({specc});'
    spect = SIMD_VOP2_TERNARY.get(template_name)
    if spect is not None:
        cpp_t, k_expr, cpp_op = spect
        return f'  ROCJITSU_TRY_SIMD_VOP2_TERNARY({cpp_t}, {k_expr}, {cpp_op});'
    specf64 = SIMD_VOP2_FMA_F64.get(template_name)
    if specf64 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP2_FMA_F64({specf64});'
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
    speccvtin = SIMD_CVT_B32_TO_F64.get(template_name)
    if speccvtin is not None:
        in_t, cpp_op = speccvtin
        return f'  ROCJITSU_TRY_SIMD_CVT_B32_TO_F64({in_t}, {cpp_op});'
    specvopc = SIMD_VOPC.get(template_name)
    if specvopc is not None:
        lane_t, cpp_op = specvopc
        return f'  ROCJITSU_TRY_SIMD_VOPC({lane_t}, {cpp_op});'
    specclass = SIMD_VOPC_CLASS.get(template_name)
    if specclass is not None:
        lane_t, cpp_op = specclass
        return f'  ROCJITSU_TRY_SIMD_VOPC({lane_t}, {cpp_op});'
    specclass64 = SIMD_VOPC_CLASS_F64.get(template_name)
    if specclass64 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOPC_CLASS_F64({specclass64});'
    spec3class = SIMD_VOP3_CLASS.get(template_name)
    if spec3class is not None:
        sm, cpp_op = spec3class
        return f'  ROCJITSU_TRY_SIMD_VOP3_CLASS_B32({sm}, {cpp_op});'
    spec3class64 = SIMD_VOP3_CLASS_F64.get(template_name)
    if spec3class64 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_CLASS_F64(0x8000000000000000ull, {spec3class64});'
    specvopc64 = SIMD_VOPC64.get(template_name)
    if specvopc64 is not None:
        lane_t, cpp_op = specvopc64
        return f'  ROCJITSU_TRY_SIMD_VOPC64({lane_t}, {cpp_op});'
    # VOP3 form of the integer/bitwise VOPC compares (i16/u16/i32/u32 and
    # i64/u64). Same functor as the VOPC table (no modifiers), but the merge
    # writes the SGPR-pair dst instead of VCC.
    specvopcv3i32 = SIMD_VOPC_VOP3_INT_32.get(template_name)
    if specvopcv3i32 is not None:
        lane_t, cpp_op = specvopcv3i32
        return f'  ROCJITSU_TRY_SIMD_VOPC_VOP3_INT({lane_t}, {cpp_op});'
    specvopcv3i64 = SIMD_VOPC_VOP3_INT_64.get(template_name)
    if specvopcv3i64 is not None:
        lane_t, cpp_op = specvopcv3i64
        return f'  ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT({lane_t}, {cpp_op});'
    # VOP3 form of the f32 VOPC relational compares (17 ops: 16 relations +
    # 't' constant). Per-source abs/neg modifiers applied outside the functor
    # via the fp32 VOPC glue; SGPR-pair dst merge identical to the integer path.
    specvopcv3f32 = SIMD_VOPC_VOP3_F32.get(template_name)
    if specvopcv3f32 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOPC_VOP3_FP32({specvopcv3f32});'
    # VOP3 form of the f16 VOPC relational compares (17 ops). The glue widens
    # raw lanes to f32 then applies abs/neg in f32 domain — matching the scalar
    # body's f16_to_f32 -> std::fabs/-x order. Same functor as the f32 path.
    specvopcv3f16 = SIMD_VOPC_VOP3_F16.get(template_name)
    if specvopcv3f16 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOPC_VOP3_FP16({specvopcv3f16});'
    # VOP3 form of the f64 VOPC relational compares (17 ops). 64-bit-lane,
    # per-source abs/neg modifiers applied in the f64 domain outside the functor.
    specvopcv3f64 = SIMD_VOPC_VOP3_F64.get(template_name)
    if specvopcv3f64 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOPC64_VOP3_FP64({specvopcv3f64});'
    # VOP3 integer/bitwise ternary ops (add3/or3/xor3/lshl_add/add_lshl/bfi).
    # Plain element-wise functor of (src0, src1, src2); no modifiers.
    spec3tern = SIMD_VOP3_TERNARY_INT.get(template_name)
    if spec3tern is not None:
        cpp_t, cpp_op = spec3tern
        return f'  ROCJITSU_TRY_SIMD_VOP3_TERNARY_INT({cpp_t}, {cpp_op});'
    # 64-bit reverse shifts (src0 = 32-bit count, src1 = 64-bit value).
    specshift64 = SIMD_SHIFT64_VOP3.get(template_name)
    if specshift64 is not None:
        return f'  ROCJITSU_TRY_SIMD_SHIFT64_VOP3({specshift64});'
    if template_name in SIMD_LSHL_ADD_U64:
        return '  ROCJITSU_TRY_SIMD_LSHL_ADD_U64();'
    specmad64 = SIMD_MAD_WIDE64.get(template_name)
    if specmad64 is not None:
        return f'  ROCJITSU_TRY_SIMD_MAD_WIDE64_VOP3({specmad64});'
    specv3co = SIMD_VOP3_CARRY_CO.get(template_name)
    if specv3co is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_CO({specv3co});'
    specv3cin = SIMD_VOP3_CARRY_CIN.get(template_name)
    if specv3cin is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_CIN({specv3cin});'
    # VOP3 fp ternary (FMA / MAD family). Per-source abs/neg + omod/clamp in
    # the f32 / f16 / f64 domain. NaN-input lanes skipped by test (gcc-13
    # packed FMA quiets a different NaN operand vs scalar std::fma).
    spec3tf32 = SIMD_VOP3_TERNARY_FP32.get(template_name)
    if spec3tf32 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP32({spec3tf32});'
    spec3tf16 = SIMD_VOP3_TERNARY_FP16.get(template_name)
    if spec3tf16 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16({spec3tf16});'
    spec3tf64 = SIMD_VOP3_TERNARY_FP64.get(template_name)
    if spec3tf64 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP64({spec3tf64});'
    if template_name in SIMD_VOP3_DIV_FMAS_FP32:
        return '  ROCJITSU_TRY_SIMD_DIV_FMAS_VOP3_FP32();'
    if template_name in SIMD_VOP3_DIV_FMAS_FP64:
        return '  ROCJITSU_TRY_SIMD_DIV_FMAS_VOP3_FP64();'
    # VOP3P packed-16 integer binary family (pk_add/sub/mul_lo/min/max for
    # i16/u16 + pk_lshlrev/lshrrev/ashrrev for b16/i16). Gated on default
    # op_sel = 0 / op_sel_hi = 3 inside the glue.
    specpk = SIMD_VOP3P_PK_BINARY_INT.get(template_name)
    if specpk is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_INT({specpk});'
    specpkt = SIMD_VOP3P_PK_TERNARY_INT.get(template_name)
    if specpkt is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_INT({specpkt});'
    specpkf16 = SIMD_VOP3P_PK_BINARY_FP16.get(template_name)
    if specpkf16 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_FP16({specpkf16});'
    specpkf16t = SIMD_VOP3P_PK_TERNARY_FP16.get(template_name)
    if specpkf16t is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_FP16({specpkf16t});'
    specpkf32 = SIMD_VOP3P_PK_BINARY_F32.get(template_name)
    if specpkf32 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_F32({specpkf32});'
    specpkf32t = SIMD_VOP3P_PK_TERNARY_F32.get(template_name)
    if specpkf32t is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_F32({specpkf32t});'
    if template_name in SIMD_VOP3P_MOV_B32:
        return '  ROCJITSU_TRY_SIMD_VOP3P_MOV_B32();'
    # VOP3P integer dot products (dot4 i8/u8, dot8 i4/u4, dot2 i16/u16).
    specdot = SIMD_VOP3P_DOT_INT.get(template_name)
    if specdot is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3P_DOT_INT({specdot});'
    if template_name in SIMD_VOP3P_DOT_F16_OR_BF16:
        return '  ROCJITSU_TRY_SIMD_VOP3P_DOT_F16();'
    # VOP3P mixed-sign integer dots (dot4 iu8, dot8 iu4).
    specdotm = SIMD_VOP3P_DOT_INT_MIXED.get(template_name)
    if specdotm is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3P_DOT_INT_MIXED({specdotm});'
    # VOP2/VOP3 dst-accumulate integer dots (v_dot{2,4,8}c_*; vdst is the accum).
    specdotc = SIMD_DOTC_INT.get(template_name)
    if specdotc is not None:
        return f'  ROCJITSU_TRY_SIMD_DOTC_INT({specdotc});'
    specdotcf16 = SIMD_DOTC_F16.get(template_name)
    if specdotcf16 is not None:
        return f'  ROCJITSU_TRY_SIMD_DOTC_F16({specdotcf16});'
    # VOP3P fma_mix / mad_mix (six ops, three destination shapes). Same body
    # for all; the routing picks the matching glue specialization.
    if template_name in SIMD_VOP3P_FMA_MIX_F32:
        return '  ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F32();'
    if template_name in SIMD_VOP3P_FMA_MIX_F16_LO:
        return '  ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F16_LO();'
    if template_name in SIMD_VOP3P_FMA_MIX_F16_HI:
        return '  ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F16_HI();'
    # VOP3 dst-accumulate FMA/MAC (vdst is the third operand). Per-isa class
    # has no src2; the accumulate glue reads vdst instead.
    specfmacf32 = SIMD_VOP3_FMAC_FP32.get(template_name)
    if specfmacf32 is not None:
        return f'  ROCJITSU_TRY_SIMD_FMAC_VOP3_FP32({specfmacf32});'
    specfmacf16 = SIMD_VOP3_FMAC_FP16.get(template_name)
    if specfmacf16 is not None:
        return f'  ROCJITSU_TRY_SIMD_FMAC_VOP3_FP16({specfmacf16});'
    specfmacf64 = SIMD_VOP3_FMAC_FP64.get(template_name)
    if specfmacf64 is not None:
        return f'  ROCJITSU_TRY_SIMD_FMAC_VOP3_FP64({specfmacf64});'
    # VOP3 ldexp: mixed-width fp src0 + int32 src1 exp.
    specldexpf32 = SIMD_VOP3_LDEXP_FP32.get(template_name)
    if specldexpf32 is not None:
        return f'  ROCJITSU_TRY_SIMD_LDEXP_VOP3_FP32({specldexpf32});'
    specldexpf64 = SIMD_VOP3_LDEXP_FP64.get(template_name)
    if specldexpf64 is not None:
        return f'  ROCJITSU_TRY_SIMD_LDEXP_VOP3_FP64({specldexpf64});'
    # VOP3 unary integer ops with no VOP1 twin (e.g. v_not_b16). Reuse the
    # VOP1 unary glue — operand shape (src0, vdst, 32-bit lanes) matches.
    spec3unai = SIMD_VOP3_UNARY_INT_EXTRA.get(template_name)
    if spec3unai is not None:
        cpp_tin, cpp_tout, cpp_op = spec3unai
        return f'  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {cpp_op});'
    # Extra plain integer binary VOP3 ops without a VOP2 twin (add_i32/i16,
    # sub_*, nc_* variants). Routed through the int VOP3 binary glue.
    spec3binx = SIMD_VOP3_BINARY_INT_EXTRA.get(template_name)
    if spec3binx is not None:
        cpp_t, cpp_op = spec3binx
        return f'  ROCJITSU_TRY_SIMD_VOP3_BINARY_INT({cpp_t}, {cpp_op});'
    # VOP3-only f32 binary (no VOP2 twin): IEEE maximum/minimum. Per-source
    # abs/neg + result omod/clamp applied by the f32 binary glue.
    spec3binf32 = SIMD_VOP3_BINARY_FP32.get(template_name)
    if spec3binf32 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_BINARY_FP(float32_t, {spec3binf32});'
    # VOP3 f64 binary (add/mul/max/min). Per-source abs/neg + result omod/clamp
    # in the f64 domain.
    spec3binf64 = SIMD_VOP3_BINARY_FP64.get(template_name)
    if spec3binf64 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_BINARY_FP64({spec3binf64});'
    # VOP3 f64 unary (ceil/floor/trunc/rndne/sqrt). Same modifier policy.
    spec3unaf64 = SIMD_VOP3_UNARY_FP64.get(template_name)
    if spec3unaf64 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP64({spec3unaf64});'
    # VOP3 f16 unary (widen-then-modify-then-narrow). FTZ-free ops only:
    # ceil/floor/trunc/rndne/sqrt. Transcendentals deferred.
    specfrexp = SIMD_VOP3_FREXP_FP.get(template_name)
    if specfrexp is not None:
        route, fn = specfrexp
        if route == 'fp16':
            return f'  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP16({fn});'
        if route == 'fp64cvt':
            return f'  ROCJITSU_TRY_SIMD_CVT_VOP3_F64_TO_B32_FP({fn});'
        return f'  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP(float32_t, float32_t, {fn});'
    spec3unaf16 = SIMD_VOP3_UNARY_FP16.get(template_name)
    if spec3unaf16 is not None:
        return f'  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP16({spec3unaf16});'
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
                return f'  ROCJITSU_TRY_SIMD_VOP3_BINARY_FP({cpp_t}, {cpp_op});'
            # f16 float binaries (v_add/sub/subrev/mul/max/min/ldexp_f16) are
            # uint32-typed (the functor widens f16->f32 by hand), but their VOP3
            # twin applies abs/neg/omod/clamp around the f16<->f32 round trip (see
            # the generated scalar body). The plain integer VOP3 glue
            # (ROCJITSU_TRY_SIMD_VOP3_BINARY_INT) does NOT apply those, so it would
            # silently drop the modifiers and return the VOP2-style result. No fp16
            # VOP3 binary modifier glue exists yet, so route them through the f16
            # variant that bails to the (modifier-applying) scalar body whenever a
            # modifier field is set, and takes the fast path only for the common
            # unmodified case. (Keeping a probe present — rather than returning None
            # — also avoids perturbing the cross-ISA shared plan via the
            # simd_probe_arch_portable gate.)
            if base.endswith('_f16'):
                return f'  ROCJITSU_TRY_SIMD_VOP3_BINARY_F16({cpp_t}, {cpp_op});'
            return f'  ROCJITSU_TRY_SIMD_VOP3_BINARY_INT({cpp_t}, {cpp_op});'
        # VOP3-encoded twins of the SIMD VOP1 unary ops. The plain int/cvt forms
        # apply no modifiers and read the same src0/vdst operands as VOP1, so they
        # reuse the VOP1 unary path verbatim. The f32 forms (and the float-domain
        # v_mov_b32) carry abs/neg/omod/clamp and route through the f32 unary glue.
        # The f16 forms also carry modifiers, but applying them around the f16<->f32
        # round trip is not yet handled, so they are left scalar (_VOP3_UNARY_SKIP).
        spec1v3 = SIMD_VOP1_UNARY.get(base + '_vop1')
        if spec1v3 is not None:
            cpp_tin, cpp_tout, cpp_op = spec1v3
            if base in _VOP3_UNARY_SKIP:
                return None
            if base in _VOP3_UNARY_FP_F32:
                return f'  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP(float32_t, float32_t, {cpp_op});'
            return f'  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {cpp_op});'
        # VOP3 twins of the mixed-width f64<->b32 cvt ops. Their generated VOP3
        # bodies drop the abs/neg/omod/clamp modifier reads (verified per-op),
        # so routing through the existing cvt glue is bit-exact. (A symmetric
        # SIMD_VOP1_UNARY_F64 fallback was considered but explicitly NOT added:
        # the f64-unary VOP3 forms apply modifiers via apply_vop3_*_mod_f64 —
        # routed through SIMD_VOP3_UNARY_FP64 above instead — and the rcp/rsq
        # forms use transcendental::*_f64 with NaN/±0/±Inf carve-outs not
        # present in the plain VOP1 functors.)
        speccvtoutv3 = SIMD_CVT_F64_TO_B32.get(base + '_vop1')
        if speccvtoutv3 is not None:
            # Unlike the plain cvt ops, v_frexp_exp_i32_f64's VOP3 body keeps the
            # modifiers (float omod/clamp on float(exp) + bit_cast) — a different
            # output encoding than the VOP1 int. Keep that twin scalar.
            if base in _VOP3_UNARY_SKIP:
                return None
            out_t, cpp_op = speccvtoutv3
            return f'  ROCJITSU_TRY_SIMD_CVT_F64_TO_B32({out_t}, {cpp_op});'
        speccvtinv3 = SIMD_CVT_B32_TO_F64.get(base + '_vop1')
        if speccvtinv3 is not None:
            in_t, cpp_op = speccvtinv3
            return f'  ROCJITSU_TRY_SIMD_CVT_B32_TO_F64({in_t}, {cpp_op});'
    return None


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
    is the inline-literal FMA family (v_fmaak/fmamk/madak/madmk): those read the
    32-bit literal through an ISA-divergent member (`simm32_` on some ISAs vs a
    `simm32` Operand with `.encoding_value_` on others), so a single shared body
    cannot satisfy every ISA. Those are identified by a non-``"0u"`` literal
    expression in SIMD_VOP2_TERNARY and are left to the genuine shared plan;
    the dst-accumulate forms (literal ``"0u"``: v_fmac/v_mac, and v_fmac_f64)
    are portable.

    A second non-portable family is VOP3P: the shared VOP3P execute template
    reads the op_sel field by its canonical member name (``op_sel`` /
    ``op_sel_hi``), but RDNA4/gfx1250 rename it to ``opsel`` / ``opsel_hi`` in
    their ``Vop3pMachineInst`` struct.  An ISA that renames the field cannot
    compile against the canonical-named shared body, so it must NOT be
    force-routed through it — it falls back to an inline body generated with
    its own (renamed) field accessors.  ``vop3p_opsel_fields`` carries the
    calling ISA's profile field names; when they differ from the canonical
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
