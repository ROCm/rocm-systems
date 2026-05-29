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
    "v_sub_f32_vop2": ("float32_t", "std::minus<>{}"),
    "v_subrev_f32_vop2": ("float32_t", "[](auto a, auto b) { return b - a; }"),
    "v_mul_f32_vop2": ("float32_t", "std::multiplies<>{}"),
    # Legacy / DX9 zero multiply: `(a == 0 || b == 0) ? 0 : a * b`. The scalar
    # bodies for `v_mul_legacy_f32` and `v_mul_dx9_zero_f32` are bit-identical.
    # The where-blend respects ±0 (== 0 matches both -0 and +0), matching
    # the scalar `a == 0.0f`. Routed via the VOP3 binary fp glue for the
    # vop3 form (which applies abs/neg/omod/clamp around this functor,
    # matching the modifier-bearing scalar body).
    "v_mul_legacy_f32_vop2": (
        "float32_t",
        "[](auto a, auto b) {"
        " auto r = a * b;"
        " util::stdx::where(a == 0.0f || b == 0.0f, r) = util::native<float32_t>(0.0f);"
        " return r; }",
    ),
    "v_mul_dx9_zero_f32_vop2": (
        "float32_t",
        "[](auto a, auto b) {"
        " auto r = a * b;"
        " util::stdx::where(a == 0.0f || b == 0.0f, r) = util::native<float32_t>(0.0f);"
        " return r; }",
    ),
    # --- uint32 (wrap-around / bitwise, bit-identical to scalar body) ---
    "v_add_u32_vop2": ("uint32_t", "std::plus<>{}"),
    "v_sub_u32_vop2": ("uint32_t", "std::minus<>{}"),
    "v_subrev_u32_vop2": ("uint32_t", "[](auto a, auto b) { return b - a; }"),
    # RDNA-style "no-carry" int add/sub — same body as plain add_u32/sub_u32/
    # subrev_u32, just renamed (the scalar bodies are bit-identical add/sub
    # without any VCC interaction).
    "v_add_nc_u32_vop2": ("uint32_t", "std::plus<>{}"),
    "v_sub_nc_u32_vop2": ("uint32_t", "std::minus<>{}"),
    "v_subrev_nc_u32_vop2": ("uint32_t", "[](auto a, auto b) { return b - a; }"),
    "v_and_b32_vop2": ("uint32_t", "std::bit_and<>{}"),
    "v_or_b32_vop2": ("uint32_t", "std::bit_or<>{}"),
    "v_xor_b32_vop2": ("uint32_t", "std::bit_xor<>{}"),
    "v_xnor_b32_vop2": ("uint32_t", "[](auto a, auto b) { return ~(a ^ b); }"),
    # rev: shift value is vsrc1 (b), shift count is src0 (a), masked to 5 bits.
    "v_lshlrev_b32_vop2": ("uint32_t", "[](auto a, auto b) { return b << (a & 31u); }"),
    "v_lshrrev_b32_vop2": ("uint32_t", "[](auto a, auto b) { return b >> (a & 31u); }"),
    "v_mul_u32_u24_vop2": (
        "uint32_t",
        "[](auto a, auto b) { return (a & 0x00FFFFFFu) * (b & 0x00FFFFFFu); }",
    ),
    # Signed 24-bit multiply, low 32 bits. Sign-extend the low 24 bits to int32
    # (matching the scalar (int32_t)(x<<8)>>8); the int32 product's low 32 bits
    # are exact, so no widening is needed.
    "v_mul_i32_i24_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " auto sa = (util::stdx::static_simd_cast<util::native<int32_t>>(a) << 8) >> 8;"
        " auto sb = (util::stdx::static_simd_cast<util::native<int32_t>>(b) << 8) >> 8;"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(sa * sb); }",
    ),
    # High 32 bits of the 24-bit multiply (48-bit product). The 32x32->64 step is
    # done by widening the lanes to a 64-bit fixed_size_simd, multiplying, and
    # shifting right by 32 (arithmetic for the signed form) before truncating
    # back to uint32 — bit-identical to the scalar uint64/int64 intermediates.
    "v_mul_hi_u32_u24_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " using U64 = util::stdx::fixed_size_simd<uint64_t, util::native<uint32_t>::size()>;"
        " auto pa = util::stdx::static_simd_cast<U64>(a & 0x00FFFFFFu);"
        " auto pb = util::stdx::static_simd_cast<U64>(b & 0x00FFFFFFu);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>((pa * pb) >> decltype(pa)(32)); }",
    ),
    "v_mul_hi_i32_i24_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " auto sa = (util::stdx::static_simd_cast<util::native<int32_t>>(a) << 8) >> 8;"
        " auto sb = (util::stdx::static_simd_cast<util::native<int32_t>>(b) << 8) >> 8;"
        " using I64 = util::stdx::fixed_size_simd<int64_t, util::native<int32_t>::size()>;"
        " auto pa = util::stdx::static_simd_cast<I64>(sa);"
        " auto pb = util::stdx::static_simd_cast<I64>(sb);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>((pa * pb) >> decltype(pa)(32)); }",
    ),
    "v_max_u32_vop2": (
        "uint32_t",
        "[](auto a, auto b) { return util::stdx::max(a, b); }",
    ),
    "v_min_u32_vop2": (
        "uint32_t",
        "[](auto a, auto b) { return util::stdx::min(a, b); }",
    ),
    # --- int32 (signed: arithmetic shift / signed min-max) ---
    "v_ashrrev_i32_vop2": ("int32_t", "[](auto a, auto b) { return b >> (a & 31); }"),
    "v_max_i32_vop2": (
        "int32_t",
        "[](auto a, auto b) { return util::stdx::max(a, b); }",
    ),
    "v_min_i32_vop2": (
        "int32_t",
        "[](auto a, auto b) { return util::stdx::min(a, b); }",
    ),
    # --- 16-bit integer (low 16 bits, result zero-extended to 32). Lane type
    # stays uint32_t; the functor masks/sign-extends the low 16 bits and writes
    # back the zero-extended 16-bit result, matching write_lane semantics. ---
    "v_add_u16_vop2": ("uint32_t", "[](auto a, auto b) { return (a + b) & 0xFFFFu; }"),
    "v_sub_u16_vop2": ("uint32_t", "[](auto a, auto b) { return (a - b) & 0xFFFFu; }"),
    "v_subrev_u16_vop2": (
        "uint32_t",
        "[](auto a, auto b) { return (b - a) & 0xFFFFu; }",
    ),
    "v_mul_lo_u16_vop2": (
        "uint32_t",
        "[](auto a, auto b) { return ((a & 0xFFFFu) * (b & 0xFFFFu)) & 0xFFFFu; }",
    ),
    # rev: shift value is vsrc1 (b), count is src0 (a) masked to 4 bits.
    "v_lshlrev_b16_vop2": (
        "uint32_t",
        "[](auto a, auto b) { return (b << (a & 15u)) & 0xFFFFu; }",
    ),
    "v_lshrrev_b16_vop2": (
        "uint32_t",
        "[](auto a, auto b) { return (b & 0xFFFFu) >> (a & 15u); }",
    ),
    # ashr: sign-extend the low 16 bits to int32, arithmetic-shift by count & 15
    # (the scalar masks the i16 count to 4 bits), then take the low 16 bits.
    "v_ashrrev_i16_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " auto sb = (util::stdx::static_simd_cast<util::native<int32_t>>(b) << 16) >> 16;"
        " auto sa = (util::stdx::static_simd_cast<util::native<int32_t>>(a) << 16) >> 16;"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(sb >> (sa & 15)) & 0xFFFFu; }",
    ),
    "v_max_i16_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " auto sa = (util::stdx::static_simd_cast<util::native<int32_t>>(a) << 16) >> 16;"
        " auto sb = (util::stdx::static_simd_cast<util::native<int32_t>>(b) << 16) >> 16;"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::max(sa, sb))"
        " & 0xFFFFu; }",
    ),
    "v_min_i16_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " auto sa = (util::stdx::static_simd_cast<util::native<int32_t>>(a) << 16) >> 16;"
        " auto sb = (util::stdx::static_simd_cast<util::native<int32_t>>(b) << 16) >> 16;"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::min(sa, sb))"
        " & 0xFFFFu; }",
    ),
    "v_max_u16_vop2": (
        "uint32_t",
        "[](auto a, auto b) { return util::stdx::max(a & 0xFFFFu, b & 0xFFFFu); }",
    ),
    "v_min_u16_vop2": (
        "uint32_t",
        "[](auto a, auto b) { return util::stdx::min(a & 0xFFFFu, b & 0xFFFFu); }",
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
    "v_max_f32_vop2": (
        "float32_t",
        "[](auto a, auto b) { return util::stdx::fmax(a, b); }",
    ),
    "v_min_f32_vop2": (
        "float32_t",
        "[](auto a, auto b) { return util::stdx::fmin(a, b); }",
    ),
    "v_max_f16_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " return util::f32_to_f16_simd("
        "util::stdx::fmax(util::f16_to_f32_simd(a), util::f16_to_f32_simd(b))); }",
    ),
    "v_min_f16_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " return util::f32_to_f16_simd("
        "util::stdx::fmin(util::f16_to_f32_simd(a), util::f16_to_f32_simd(b))); }",
    ),
    # --- f16 binary (low 16 bits f16, result zero-extended). Same f32
    # intermediate as the scalar bodies (single final round) ⇒ bit-identical. ---
    "v_add_f16_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " return util::f32_to_f16_simd(util::f16_to_f32_simd(a) + util::f16_to_f32_simd(b)); }",
    ),
    "v_sub_f16_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " return util::f32_to_f16_simd(util::f16_to_f32_simd(a) - util::f16_to_f32_simd(b)); }",
    ),
    "v_subrev_f16_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " return util::f32_to_f16_simd(util::f16_to_f32_simd(b) - util::f16_to_f32_simd(a)); }",
    ),
    "v_mul_f16_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " return util::f32_to_f16_simd(util::f16_to_f32_simd(a) * util::f16_to_f32_simd(b)); }",
    ),
    # v_ldexp_f16: dst = f16(ldexp(f16->f32(src0), (int16)vsrc1)). src0 is an f16
    # multiplicand, vsrc1 a signed-16-bit exponent (widened to a fixed_size int
    # lane). std::ldexp is a power-of-2 scale with a single correctly-rounded
    # result; util::stdx::ldexp matches it bit-for-bit (verified full-range incl
    # NaN/Inf/denormal), and the f16<->f32 conversions are bit-exact, so the
    # composition matches the scalar body. No NaN-operand ambiguity (unlike fma).
    "v_ldexp_f16_vop2": (
        "uint32_t",
        "[](auto a, auto b) {"
        " auto x = util::f16_to_f32_simd(a);"
        " auto n = util::stdx::static_simd_cast<"
        "util::stdx::fixed_size_simd<int, util::native<float>::size()>>("
        "(util::stdx::static_simd_cast<util::native<int32_t>>(b) << 16) >> 16);"
        " return util::f32_to_f16_simd(util::stdx::ldexp(x, n)); }",
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
# modifier glue rather than reusing the plain VOP1 path). v_mov_b32's VOP3 body
# treats src0 as f32 and applies the same modifiers, so it belongs here too.
_VOP3_UNARY_FP_F32 = {
    "v_mov_b32",
    "v_floor_f32",
    "v_ceil_f32",
    "v_trunc_f32",
    "v_rndne_f32",
    "v_fract_f32",
    "v_rcp_f32",
    "v_rcp_iflag_f32",
    "v_rsq_f32",
    "v_sqrt_f32",
    "v_exp_f32",
    "v_log_f32",
}

# VOP1 base mnemonics whose VOP3 twin stays scalar: the f16 rounding/transcendental
# forms carry modifiers applied around an f16<->f32 round trip (not yet handled).
_VOP3_UNARY_SKIP = {
    "v_floor_f16",
    "v_ceil_f16",
    "v_trunc_f16",
    "v_rndne_f16",
    "v_fract_f16",
    "v_rcp_f16",
    "v_rsq_f16",
    "v_sqrt_f16",
    "v_exp_f16",
    "v_log_f16",
}

SIMD_VOP1_UNARY: dict[str, tuple[str, str, str]] = {
    # --- bitwise / move (uint32, bit-identical) ---
    "v_mov_b32_vop1": ("uint32_t", "uint32_t", "[](auto a) { return a; }"),
    "v_not_b32_vop1": ("uint32_t", "uint32_t", "[](auto a) { return ~a; }"),
    # v_bfrev_b32: reverse the 32 bits of src0. The scalar body loops bit-by-bit;
    # this is the branchless swap-by-strides equivalent (1/2/4/8/16-bit groups),
    # bit-identical for every input. Pure uint32 bitwise ops.
    "v_bfrev_b32_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " auto x = a;"
        " x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);"
        " x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);"
        " x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);"
        " x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);"
        " return (x << 16) | (x >> 16); }",
    ),
    # --- ubyte -> f32 (extract byte N, exact int->float) -----------------------
    # dst = float(byte_N(src0)); byte value is 0..255 so the int->float cast is
    # exact and bit-identical to the scalar static_cast<float>.
    "v_cvt_f32_ubyte0_vop1": (
        "uint32_t",
        "float32_t",
        "[](auto a) { return util::stdx::static_simd_cast<util::native<float32_t>>(a & 0xFFu); }",
    ),
    "v_cvt_f32_ubyte1_vop1": (
        "uint32_t",
        "float32_t",
        "[](auto a) {"
        " return util::stdx::static_simd_cast<util::native<float32_t>>((a >> 8) & 0xFFu); }",
    ),
    "v_cvt_f32_ubyte2_vop1": (
        "uint32_t",
        "float32_t",
        "[](auto a) {"
        " return util::stdx::static_simd_cast<util::native<float32_t>>((a >> 16) & 0xFFu); }",
    ),
    "v_cvt_f32_ubyte3_vop1": (
        "uint32_t",
        "float32_t",
        "[](auto a) {"
        " return util::stdx::static_simd_cast<util::native<float32_t>>((a >> 24) & 0xFFu); }",
    ),
    # --- int<->float casts (single-rounded, bit-identical) ---
    "v_cvt_f32_i32_vop1": (
        "int32_t",
        "float32_t",
        "[](auto a) { return util::stdx::static_simd_cast<util::native<float32_t>>(a); }",
    ),
    "v_cvt_f32_u32_vop1": (
        "uint32_t",
        "float32_t",
        "[](auto a) { return util::stdx::static_simd_cast<util::native<float32_t>>(a); }",
    ),
    # --- float rounding (bit-identical to std::* on host) ---
    "v_floor_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return util::floor_simd(a); }",
    ),
    "v_ceil_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return util::ceil_simd(a); }",
    ),
    "v_trunc_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return util::trunc_simd(a); }",
    ),
    "v_rndne_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return util::rndne_simd(a); }",
    ),
    "v_fract_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return a - util::floor_simd(a); }",
    ),
    # --- transcendental div / sqrt. These mirror amdgpu::transcendental::*_f32
    # exactly via util::*_f32_simd (FTZ input/output flush + canonical-qNaN and
    # NaN-input-preservation blends), so the SIMD result is bit-identical to the
    # forced-scalar body on every input incl NaN/Inf/denormal. ---
    "v_rcp_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return util::rcp_f32_simd(a); }",
    ),
    "v_rcp_iflag_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return util::rcp_f32_simd(a); }",
    ),
    "v_rsq_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return util::rsq_f32_simd(a); }",
    ),
    "v_sqrt_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return util::sqrt_f32_simd(a); }",
    ),
    # --- transcendental exp2/log2. util::{exp,log}_f32_simd wrap stdx::exp2/log2
    # with the same FTZ flush / special-case guards as the scalar transcendental
    # reference; the underlying vector libm is bit-exact to scalar std::* on the
    # supported toolchains (libstdc++ 13 / AVX-512), guarded by
    # UtilSimd.Exp2/Log2_*_BitExact. v_sin/v_cos excluded: vector libm ~1 ULP off. ---
    "v_exp_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return util::exp_f32_simd(a); }",
    ),
    "v_log_f32_vop1": (
        "float32_t",
        "float32_t",
        "[](auto a) { return util::log_f32_simd(a); }",
    ),
    # --- float -> int conversions with NaN->0 and saturating clamp. The float
    # comparison masks are re-typed to the int lane via simd_mask_as<> before
    # the where-blend. Masks are mutually exclusive so application order is
    # irrelevant. Matches the scalar bodies' truncate-toward-zero cast. ---
    "v_cvt_i32_f32_vop1": (
        "float32_t",
        "int32_t",
        "[](auto s) {"
        " util::native<int32_t> out = util::stdx::static_simd_cast<util::native<int32_t>>(s);"
        " util::stdx::where(simd_mask_as<int32_t>(s >= 2147483648.0f), out) = 2147483647;"
        " util::stdx::where(simd_mask_as<int32_t>(s < -2147483648.0f), out) = (-2147483647 - 1);"
        " util::stdx::where(simd_mask_as<int32_t>(util::stdx::isnan(s)), out) = 0;"
        " return out; }",
    ),
    "v_cvt_u32_f32_vop1": (
        "float32_t",
        "uint32_t",
        "[](auto s) {"
        " util::native<uint32_t> out = util::stdx::static_simd_cast<util::native<uint32_t>>(s);"
        " util::stdx::where(simd_mask_as<uint32_t>(s >= 4294967296.0f), out) = 4294967295u;"
        " util::stdx::where(simd_mask_as<uint32_t>(util::stdx::isnan(s) || s < 0.0f), out) = 0u;"
        " return out; }",
    ),
    "v_cvt_flr_i32_f32_vop1": (
        "float32_t",
        "int32_t",
        "[](auto s) {"
        " auto r = util::floor_simd(s);"
        " util::native<int32_t> out = util::stdx::static_simd_cast<util::native<int32_t>>(r);"
        " util::stdx::where(simd_mask_as<int32_t>(r >= 2147483648.0f), out) = 2147483647;"
        " util::stdx::where(simd_mask_as<int32_t>(r < -2147483648.0f), out) = (-2147483647 - 1);"
        " util::stdx::where(simd_mask_as<int32_t>(util::stdx::isnan(r)), out) = 0;"
        " return out; }",
    ),
    "v_cvt_rpi_i32_f32_vop1": (
        "float32_t",
        "int32_t",
        "[](auto s) {"
        " auto r = util::ceil_simd(s - util::native<float32_t>(0.5f));"
        " util::native<int32_t> out = util::stdx::static_simd_cast<util::native<int32_t>>(r);"
        " util::stdx::where(simd_mask_as<int32_t>(r >= 2147483648.0f), out) = 2147483647;"
        " util::stdx::where(simd_mask_as<int32_t>(r < -2147483648.0f), out) = (-2147483647 - 1);"
        " util::stdx::where(simd_mask_as<int32_t>(util::stdx::isnan(r)), out) = 0;"
        " return out; }",
    ),
    # --- f16 (half) ops. Scalar bodies route through an f32 intermediate with a
    # single final round, so the SIMD path (f16_to_f32_simd -> f32 op ->
    # f32_to_f16_simd) is bit-identical. The conversions are bit-exact (see
    # UtilSimd.F16ToF32/F32ToF16 guards). f16 result occupies the low 16 bits of
    # the dst, high zeroed (Tout=uint32_t), matching write_lane zero-extension. ---
    "v_floor_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) { return util::f32_to_f16_simd(util::floor_simd(util::f16_to_f32_simd(a))); }",
    ),
    "v_ceil_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) { return util::f32_to_f16_simd(util::ceil_simd(util::f16_to_f32_simd(a))); }",
    ),
    "v_trunc_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) { return util::f32_to_f16_simd(util::trunc_simd(util::f16_to_f32_simd(a))); }",
    ),
    "v_rndne_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " return util::f32_to_f16_simd(util::rndne_simd(util::f16_to_f32_simd(a))); }",
    ),
    "v_fract_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " auto f = util::f16_to_f32_simd(a);"
        " return util::f32_to_f16_simd(f - util::floor_simd(f)); }",
    ),
    # f16 transcendentals mirror the scalar f32_to_f16(<op>_f32(f16_to_f32(x)))
    # by applying the f32-domain util::*_f32_simd helper (FTZ flush + canonical
    # qNaN / NaN-input guards) on the f16->f32 intermediate.
    "v_rcp_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " return util::f32_to_f16_simd(util::rcp_f32_simd(util::f16_to_f32_simd(a))); }",
    ),
    "v_rsq_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " return util::f32_to_f16_simd(util::rsq_f32_simd(util::f16_to_f32_simd(a))); }",
    ),
    "v_sqrt_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " return util::f32_to_f16_simd(util::sqrt_f32_simd(util::f16_to_f32_simd(a))); }",
    ),
    # exp/log_f16 inherit the exp2/log2 toolchain guard (UtilSimd.Exp2/Log2_*).
    "v_exp_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " return util::f32_to_f16_simd(util::exp_f32_simd(util::f16_to_f32_simd(a))); }",
    ),
    "v_log_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " return util::f32_to_f16_simd(util::log_f32_simd(util::f16_to_f32_simd(a))); }",
    ),
    # --- f16 <-> f32 / int16 conversions ---
    "v_cvt_f32_f16_vop1": (
        "uint32_t",
        "float32_t",
        "[](auto a) { return util::f16_to_f32_simd(a); }",
    ),
    "v_cvt_f16_f32_vop1": (
        "float32_t",
        "uint32_t",
        "[](auto a) { return util::f32_to_f16_simd(a); }",
    ),
    "v_cvt_f16_i16_vop1": (
        "int32_t",
        "uint32_t",
        "[](auto a) {"
        " auto i = (a << 16) >> 16;"  # sign-extend low 16 bits
        " return util::f32_to_f16_simd(util::stdx::static_simd_cast<util::native<float32_t>>(i)); }",
    ),
    "v_cvt_f16_u16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " auto u = a & 0xFFFFu;"
        " return util::f32_to_f16_simd(util::stdx::static_simd_cast<util::native<float32_t>>(u)); }",
    ),
    "v_cvt_i16_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " auto s = util::f16_to_f32_simd(a);"
        " util::native<int32_t> o = util::stdx::static_simd_cast<util::native<int32_t>>(s);"
        " util::stdx::where(simd_mask_as<int32_t>(s >= 32768.0f), o) = 32767;"
        " util::stdx::where(simd_mask_as<int32_t>(s < -32768.0f), o) = -32768;"
        " util::stdx::where(simd_mask_as<int32_t>(util::stdx::isnan(s)), o) = 0;"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(o) & 0xFFFFu; }",
    ),
    "v_cvt_u16_f16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " auto s = util::f16_to_f32_simd(a);"
        " util::native<uint32_t> o = util::stdx::static_simd_cast<util::native<uint32_t>>(s);"
        " util::stdx::where(simd_mask_as<uint32_t>(s >= 65536.0f), o) = 65535u;"
        " util::stdx::where(simd_mask_as<uint32_t>(util::stdx::isnan(s) || s < 0.0f), o) = 0u;"
        " return o & 0xFFFFu; }",
    ),
    # --- VOP1 b16 unary (RDNA3+ only): low-16 mov / not. The scalar bodies
    # ignore the high 16 bits of src0 and zero-extend the 16-bit result. No
    # modifiers in VOP1 form (cf. the VOP3 mov_b16 form which carries omod/
    # clamp — handled by the dedicated try_execute_mov_b16_vop3_simd glue).
    "v_mov_b16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) { return a & 0xFFFFu; }",
    ),
    "v_not_b16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) { return (~a) & 0xFFFFu; }",
    ),
    # --- int16 / uint16 → 32-bit zero/sign extend. Scalar:
    #   cvt_i32_i16: (int32_t)(int16_t)(src0 & 0xFFFF)  (sign-extend)
    #   cvt_u32_u16: src0 & 0xFFFF                       (zero-extend)
    "v_cvt_i32_i16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) {"
        " auto x = util::stdx::static_simd_cast<util::native<int32_t>>(a & 0xFFFFu);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>((x << 16) >> 16); }",
    ),
    "v_cvt_u32_u16_vop1": (
        "uint32_t",
        "uint32_t",
        "[](auto a) { return a & 0xFFFFu; }",
    ),
    # --- aliases for v_cvt_flr_i32_f32 / v_cvt_rpi_i32_f32 (same scalar body,
    # RDNA naming). Reuse the floor / ceil(s - 0.5) functors verbatim.
    "v_cvt_floor_i32_f32_vop1": (
        "float32_t",
        "int32_t",
        "[](auto s) {"
        " auto r = util::floor_simd(s);"
        " util::native<int32_t> out = util::stdx::static_simd_cast<util::native<int32_t>>(r);"
        " util::stdx::where(simd_mask_as<int32_t>(r >= 2147483648.0f), out) = 2147483647;"
        " util::stdx::where(simd_mask_as<int32_t>(r < -2147483648.0f), out) = (-2147483647 - 1);"
        " util::stdx::where(simd_mask_as<int32_t>(util::stdx::isnan(r)), out) = 0;"
        " return out; }",
    ),
    "v_cvt_nearest_i32_f32_vop1": (
        "float32_t",
        "int32_t",
        "[](auto s) {"
        " auto r = util::ceil_simd(s - util::native<float32_t>(0.5f));"
        " util::native<int32_t> out = util::stdx::static_simd_cast<util::native<int32_t>>(r);"
        " util::stdx::where(simd_mask_as<int32_t>(r >= 2147483648.0f), out) = 2147483647;"
        " util::stdx::where(simd_mask_as<int32_t>(r < -2147483648.0f), out) = (-2147483647 - 1);"
        " util::stdx::where(simd_mask_as<int32_t>(util::stdx::isnan(r)), out) = 0;"
        " return out; }",
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
    "v_add_co_u32_vop2": (
        "[](auto a, auto b, auto) {"
        " auto s = a + b;"
        " return make_simd_carry(s, s < a); }"
    ),
    "v_sub_co_u32_vop2": (
        "[](auto a, auto b, auto) { return make_simd_carry(a - b, a < b); }"
    ),
    "v_subrev_co_u32_vop2": (
        "[](auto a, auto b, auto) { return make_simd_carry(b - a, b < a); }"
    ),
    "v_addc_co_u32_vop2": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = a + b; auto c1 = t1 < a;"
        " auto t2 = t1 + cin; auto c2 = t2 < t1;"
        " return make_simd_carry(t2, c1 | c2); }"
    ),
    "v_subb_co_u32_vop2": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = a - b; auto bw1 = a < b;"
        " auto t2 = t1 - cin; auto bw2 = t1 < cin;"
        " return make_simd_carry(t2, bw1 | bw2); }"
    ),
    "v_subbrev_co_u32_vop2": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = b - a; auto bw1 = b < a;"
        " auto t2 = t1 - cin; auto bw2 = t1 < cin;"
        " return make_simd_carry(t2, bw1 | bw2); }"
    ),
    # RDNA-style co/ci VOP2 carry-cin forms — bit-identical to the addc/subb/
    # subbrev_co bodies above (cin read from VCC, co written to VCC).
    "v_add_co_ci_u32_vop2": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = a + b; auto c1 = t1 < a;"
        " auto t2 = t1 + cin; auto c2 = t2 < t1;"
        " return make_simd_carry(t2, c1 | c2); }"
    ),
    "v_sub_co_ci_u32_vop2": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = a - b; auto bw1 = a < b;"
        " auto t2 = t1 - cin; auto bw2 = t1 < cin;"
        " return make_simd_carry(t2, bw1 | bw2); }"
    ),
    "v_subrev_co_ci_u32_vop2": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = b - a; auto bw1 = b < a;"
        " auto t2 = t1 - cin; auto bw2 = t1 < cin;"
        " return make_simd_carry(t2, bw1 | bw2); }"
    ),
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
_FMA_ACC_F32 = "[](auto a, auto b, auto d, auto) { return util::stdx::fma(a, b, d); }"
_FMA_ADDK_F32 = "[](auto a, auto b, auto, auto k) { return util::stdx::fma(a, b, k); }"
_FMA_MULK_F32 = "[](auto a, auto b, auto, auto k) { return util::stdx::fma(a, k, b); }"
_FMA_ACC_F16 = (
    "[](auto a, auto b, auto d, auto) {"
    " return util::f32_to_f16_simd(util::stdx::fma("
    "util::f16_to_f32_simd(a), util::f16_to_f32_simd(b), util::f16_to_f32_simd(d))); }"
)
_FMA_ADDK_F16 = (
    "[](auto a, auto b, auto, auto k) {"
    " return util::f32_to_f16_simd(util::stdx::fma("
    "util::f16_to_f32_simd(a), util::f16_to_f32_simd(b), util::f16_to_f32_simd(k))); }"
)
_FMA_MULK_F16 = (
    "[](auto a, auto b, auto, auto k) {"
    " return util::f32_to_f16_simd(util::stdx::fma("
    "util::f16_to_f32_simd(a), util::f16_to_f32_simd(k), util::f16_to_f32_simd(b))); }"
)
SIMD_VOP2_TERNARY: dict[str, tuple[str, str, str]] = {
    # --- f32 dst-accumulate ---
    "v_fmac_f32_vop2": ("float32_t", "0u", _FMA_ACC_F32),
    "v_fmac_dx9_zero_f32_vop2": ("float32_t", "0u", _FMA_ACC_F32),
    "v_mac_f32_vop2": ("float32_t", "0u", _FMA_ACC_F32),
    # --- f32 inline literal ---
    "v_fmaak_f32_vop2": ("float32_t", "inst.simm32_", _FMA_ADDK_F32),
    "v_madak_f32_vop2": ("float32_t", "inst.simm32.encoding_value_", _FMA_ADDK_F32),
    "v_fmamk_f32_vop2": ("float32_t", "inst.simm32_", _FMA_MULK_F32),
    "v_madmk_f32_vop2": ("float32_t", "inst.simm32.encoding_value_", _FMA_MULK_F32),
    # --- f16 dst-accumulate ---
    "v_fmac_f16_vop2": ("uint32_t", "0u", _FMA_ACC_F16),
    "v_mac_f16_vop2": ("uint32_t", "0u", _FMA_ACC_F16),
    # --- f16 inline literal ---
    "v_madak_f16_vop2": ("uint32_t", "inst.simm32.encoding_value_", _FMA_ADDK_F16),
    "v_fmamk_f16_vop2": ("uint32_t", "inst.simm32_", _FMA_MULK_F16),
    "v_madmk_f16_vop2": ("uint32_t", "inst.simm32.encoding_value_", _FMA_MULK_F16),
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
    "v_fmac_f64_vop2": "[](auto a, auto b, auto d) { return util::stdx::fma(a, b, d); }",
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
    "v_ceil_f64_vop1": ("double", "[](auto a) { return util::ceil_simd(a); }"),
    "v_floor_f64_vop1": ("double", "[](auto a) { return util::floor_simd(a); }"),
    "v_trunc_f64_vop1": ("double", "[](auto a) { return util::trunc_simd(a); }"),
    "v_rndne_f64_vop1": ("double", "[](auto a) { return util::rndne_simd(a); }"),
    "v_fract_f64_vop1": ("double", "[](auto a) { return a - util::floor_simd(a); }"),
    "v_rcp_f64_vop1": (
        "double",
        "[](auto a) { return util::native<double>(1.0) / a; }",
    ),
    "v_rsq_f64_vop1": (
        "double",
        "[](auto a) { return util::native<double>(1.0) / util::stdx::sqrt(a); }",
    ),
    "v_sqrt_f64_vop1": ("double", "[](auto a) { return util::stdx::sqrt(a); }"),
    "v_mov_b64_vop1": ("uint64_t", "[](auto a) { return a; }"),
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
    "v_cvt_f32_f64_vop1": (
        "float32_t",
        "[](auto s) { return util::stdx::static_simd_cast<util::narrow32<float32_t>>(s); }",
    ),
    "v_cvt_i32_f64_vop1": (
        "int32_t",
        "[](auto s) {"
        " auto r = s;"
        " util::stdx::where(util::stdx::isnan(s), r) = 0.0;"
        " util::stdx::where(s >= 2147483648.0, r) = 2147483647.0;"
        " util::stdx::where(s < -2147483648.0, r) = -2147483648.0;"
        " return util::stdx::static_simd_cast<util::narrow32<int32_t>>(r); }",
    ),
    "v_cvt_u32_f64_vop1": (
        "uint32_t",
        "[](auto s) {"
        " auto r = s;"
        " util::stdx::where(util::stdx::isnan(s) || s < 0.0, r) = 0.0;"
        " util::stdx::where(s >= 4294967296.0, r) = 4294967295.0;"
        " return util::stdx::static_simd_cast<util::narrow32<uint32_t>>(r); }",
    ),
}

# 32-bit source -> f64 dst. template_name -> (in_lane_cpp_type, functor); the
# functor is invoked as cvt_op(narrow32<Tin>) -> native<double> inside
# try_execute_cvt_b32_to_f64_simd. Each is an exact widening static_simd_cast
# (vcvtps2pd for f32; int->double for i32/u32), bit-identical to the scalar body
# (static_cast<double>) for every input.
SIMD_CVT_B32_TO_F64: dict[str, tuple[str, str]] = {
    "v_cvt_f64_f32_vop1": (
        "float32_t",
        "[](auto in) { return util::stdx::static_simd_cast<util::native<double>>(in); }",
    ),
    "v_cvt_f64_i32_vop1": (
        "int32_t",
        "[](auto in) { return util::stdx::static_simd_cast<util::native<double>>(in); }",
    ),
    "v_cvt_f64_u32_vop1": (
        "uint32_t",
        "[](auto in) { return util::stdx::static_simd_cast<util::native<double>>(in); }",
    ),
}


# template_name set for v_cndmask_b32 (VCC-driven per-lane select). No functor:
# the op is fixed (dst = (VCC bit) ? vsrc1 : src0), a pure 32-bit bit select, so
# the SIMD result is bit-identical to the scalar body for every input.
SIMD_VOP2_CNDMASK: set[str] = {
    "v_cndmask_b32_vop2",
}

# VOP3 form of v_cndmask_b32: same per-lane select, but the 64-bit selector is
# read from the SGPR-pair `src2` instead of VCC. Also fixed-op / functorless.
SIMD_VOP3_CNDMASK: set[str] = {
    "v_cndmask_b32_vop3",
}

# 16-bit variant — RDNA3+. Low-16 of each source selected per lane, high-16
# zero (matches scalar `uint32_t(uint16_t(...))` write pattern).
SIMD_VOP3_CNDMASK_B16: set[str] = {
    "v_cndmask_b16_vop3",
}

# VOP3 div_fmas: fma(src0, src1, src2) followed by a VCC-bit-gated
# ldexp(result, 32) (f32) or ldexp(result, 64) (f64). Fixed-op / functorless.
SIMD_VOP3_DIV_FMAS_FP32: set[str] = {
    "v_div_fmas_f32_vop3",
}
SIMD_VOP3_DIV_FMAS_FP64: set[str] = {
    "v_div_fmas_f64_vop3",
}

# VOP3 sdst-enc carry forms (no-carry-in subset). Same per-lane carry/borrow
# functor as the VOP2 carry family — the glue layer
# (try_execute_binary_vop3_carry_simd) handles the VOP3 src1/sdst differences
# (the functor is identical).
SIMD_VOP3_CARRY: dict[str, str] = {
    "v_add_co_u32_vop3": (
        "[](auto a, auto b, auto) {"
        " auto s = a + b;"
        " return make_simd_carry(s, s < a); }"
    ),
    "v_sub_co_u32_vop3": (
        "[](auto a, auto b, auto) { return make_simd_carry(a - b, a < b); }"
    ),
    "v_subrev_co_u32_vop3": (
        "[](auto a, auto b, auto) { return make_simd_carry(b - a, b < a); }"
    ),
}

# VOP3 sdst-enc carry forms with src2 carry-in. Six ops sharing one shape:
# CDNA4 sdst-enc `addc_co_u32 / subb_co_u32 / subbrev_co_u32` and the RDNA-only
# `add_co_ci_u32 / sub_co_ci_u32 / subrev_co_ci_u32`. All read per-lane cin from
# `inst.src2.read_scalar64(wf)` and write co into `inst.sdst.write_scalar64`
# (the RDNA decoder binds src2/sdst to VCC, but the body is uniform across the
# six). Glue `try_execute_binary_vop3_carry_src2_simd` handles the cin source.
SIMD_VOP3_CARRY_SRC2: dict[str, str] = {
    "v_addc_co_u32_vop3": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = a + b;"
        " auto c1 = t1 < a;"
        " auto t2 = t1 + cin;"
        " auto c2 = t2 < t1;"
        " return make_simd_carry(t2, c1 | c2); }"
    ),
    "v_subb_co_u32_vop3": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = a - b;"
        " auto bw1 = a < b;"
        " auto t2 = t1 - cin;"
        " auto bw2 = t1 < cin;"
        " return make_simd_carry(t2, bw1 | bw2); }"
    ),
    "v_subbrev_co_u32_vop3": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = b - a;"
        " auto bw1 = b < a;"
        " auto t2 = t1 - cin;"
        " auto bw2 = t1 < cin;"
        " return make_simd_carry(t2, bw1 | bw2); }"
    ),
    "v_add_co_ci_u32_vop3": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = a + b;"
        " auto c1 = t1 < a;"
        " auto t2 = t1 + cin;"
        " auto c2 = t2 < t1;"
        " return make_simd_carry(t2, c1 | c2); }"
    ),
    "v_sub_co_ci_u32_vop3": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = a - b;"
        " auto bw1 = a < b;"
        " auto t2 = t1 - cin;"
        " auto bw2 = t1 < cin;"
        " return make_simd_carry(t2, bw1 | bw2); }"
    ),
    "v_subrev_co_ci_u32_vop3": (
        "[](auto a, auto b, auto cin) {"
        " auto t1 = b - a;"
        " auto bw1 = b < a;"
        " auto t2 = t1 - cin;"
        " auto bw2 = t1 < cin;"
        " return make_simd_carry(t2, bw1 | bw2); }"
    ),
}

# VOP3 v_mov_b16 — RDNA3+ only. Reads low 16 of src0 as an integer, treats it
# as a float for omod (*2 / *4 / *0.5) + clamp ([0, 1]), then truncate-casts
# back through int32 and masks 16. No abs/neg/op_sel; functorless / fixed-op.
SIMD_VOP3_MOV_B16: set[str] = {
    "v_mov_b16_vop3",
}

# VOP3P fma_mix / mad_mix family. The six ops share one body (`a*b + c` plus
# optional clamp to [0,1]); only the destination shape differs:
#  - F32     -> v_fma_mix_f32_vop3p (RDNA3+), v_mad_mix_f32_vop3p (CDNA1-4)
#  - F16_LO  -> v_fma_mixlo_f16_vop3p, v_mad_mixlo_f16_vop3p
#  - F16_HI  -> v_fma_mixhi_f16_vop3p, v_mad_mixhi_f16_vop3p
# Per-source op_sel_hi gates the f16<->f32 widening shape; op_sel picks the f16
# half. neg flips the sign bit. No abs, no omod. Functorless / fixed-op.
SIMD_VOP3P_FMA_MIX_F32: set[str] = {
    "v_fma_mix_f32_vop3p",
    "v_mad_mix_f32_vop3p",
}
SIMD_VOP3P_FMA_MIX_F16_LO: set[str] = {
    "v_fma_mixlo_f16_vop3p",
    "v_mad_mixlo_f16_vop3p",
}
SIMD_VOP3P_FMA_MIX_F16_HI: set[str] = {
    "v_fma_mixhi_f16_vop3p",
    "v_mad_mixhi_f16_vop3p",
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
    "v_pk_add_u16_vop3p": (
        "[](auto a, auto b) {"
        " auto lo = (a + b) & 0xFFFFu;"
        " auto hi = ((a >> 16) + (b >> 16)) & 0xFFFFu;"
        " return lo | (hi << 16); }"
    ),
    # add_i16 is bit-identical to add_u16 (mod-2^16 wrap matches).
    "v_pk_add_i16_vop3p": (
        "[](auto a, auto b) {"
        " auto lo = (a + b) & 0xFFFFu;"
        " auto hi = ((a >> 16) + (b >> 16)) & 0xFFFFu;"
        " return lo | (hi << 16); }"
    ),
    "v_pk_sub_u16_vop3p": (
        "[](auto a, auto b) {"
        " auto lo = (a - b) & 0xFFFFu;"
        " auto hi = ((a >> 16) - (b >> 16)) & 0xFFFFu;"
        " return lo | (hi << 16); }"
    ),
    "v_pk_sub_i16_vop3p": (
        "[](auto a, auto b) {"
        " auto lo = (a - b) & 0xFFFFu;"
        " auto hi = ((a >> 16) - (b >> 16)) & 0xFFFFu;"
        " return lo | (hi << 16); }"
    ),
    "v_pk_mul_lo_u16_vop3p": (
        "[](auto a, auto b) {"
        " auto lo = ((a & 0xFFFFu) * (b & 0xFFFFu)) & 0xFFFFu;"
        " auto hi = ((a >> 16) * (b >> 16)) & 0xFFFFu;"
        " return lo | (hi << 16); }"
    ),
    # Reverse-shift forms: src1 holds the value, src0 holds the count.
    # Shift count masked to low 4 bits per scalar (`& 15u`).
    "v_pk_lshlrev_b16_vop3p": (
        "[](auto a, auto b) {"
        " auto lo = ((b & 0xFFFFu) << (a & 15u)) & 0xFFFFu;"
        " auto hi = (((b >> 16) & 0xFFFFu) << ((a >> 16) & 15u)) & 0xFFFFu;"
        " return lo | (hi << 16); }"
    ),
    "v_pk_lshrrev_b16_vop3p": (
        "[](auto a, auto b) {"
        " auto lo = ((b & 0xFFFFu) >> (a & 15u)) & 0xFFFFu;"
        " auto hi = ((b >> 16) >> ((a >> 16) & 15u)) & 0xFFFFu;"
        " return lo | (hi << 16); }"
    ),
    # Arithmetic right shift on i16: sign-extend each half to int32 via
    # (x << 16) >> 16, shift, mask back to 16.
    "v_pk_ashrrev_i16_vop3p": (
        "[](auto a, auto b) {"
        " using I = util::native<int32_t>;"
        " auto bv_lo = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;"
        " auto bv_hi = (util::stdx::static_simd_cast<I>(b >> 16) << 16) >> 16;"
        " auto sh_lo = util::stdx::static_simd_cast<I>(a & 15u);"
        " auto sh_hi = util::stdx::static_simd_cast<I>((a >> 16) & 15u);"
        " auto rlo = util::stdx::static_simd_cast<util::native<uint32_t>>(bv_lo >> sh_lo) & 0xFFFFu;"
        " auto rhi = util::stdx::static_simd_cast<util::native<uint32_t>>(bv_hi >> sh_hi) & 0xFFFFu;"
        " return rlo | (rhi << 16); }"
    ),
    "v_pk_min_u16_vop3p": (
        "[](auto a, auto b) {"
        " auto lo = util::stdx::min(a & 0xFFFFu, b & 0xFFFFu);"
        " auto hi = util::stdx::min(a >> 16, b >> 16);"
        " return (lo & 0xFFFFu) | ((hi & 0xFFFFu) << 16); }"
    ),
    "v_pk_max_u16_vop3p": (
        "[](auto a, auto b) {"
        " auto lo = util::stdx::max(a & 0xFFFFu, b & 0xFFFFu);"
        " auto hi = util::stdx::max(a >> 16, b >> 16);"
        " return (lo & 0xFFFFu) | ((hi & 0xFFFFu) << 16); }"
    ),
    "v_pk_min_i16_vop3p": (
        "[](auto a, auto b) {"
        " using I = util::native<int32_t>;"
        " auto a_lo = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;"
        " auto a_hi = (util::stdx::static_simd_cast<I>(a >> 16) << 16) >> 16;"
        " auto b_lo = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;"
        " auto b_hi = (util::stdx::static_simd_cast<I>(b >> 16) << 16) >> 16;"
        " auto rlo = util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::min(a_lo, b_lo)) & 0xFFFFu;"
        " auto rhi = util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::min(a_hi, b_hi)) & 0xFFFFu;"
        " return rlo | (rhi << 16); }"
    ),
    "v_pk_max_i16_vop3p": (
        "[](auto a, auto b) {"
        " using I = util::native<int32_t>;"
        " auto a_lo = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;"
        " auto a_hi = (util::stdx::static_simd_cast<I>(a >> 16) << 16) >> 16;"
        " auto b_lo = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;"
        " auto b_hi = (util::stdx::static_simd_cast<I>(b >> 16) << 16) >> 16;"
        " auto rlo = util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::max(a_lo, b_lo)) & 0xFFFFu;"
        " auto rhi = util::stdx::static_simd_cast<util::native<uint32_t>>(util::stdx::max(a_hi, b_hi)) & 0xFFFFu;"
        " return rlo | (rhi << 16); }"
    ),
}

# VOP3P packed-16 integer ternary (pk_mad_i16 / pk_mad_u16). Same default
# packing gate as the binary table (op_sel/op_sel_hi/op_sel_hi_2). Scalar
# truncates to 16 bits via uint16_t cast, so the SIMD path masks each half
# to 16 bits before pack.
SIMD_VOP3P_PK_TERNARY_INT: dict[str, str] = {
    "v_pk_mad_i16_vop3p": (
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " auto a_lo = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;"
        " auto a_hi = (util::stdx::static_simd_cast<I>(a >> 16) << 16) >> 16;"
        " auto b_lo = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;"
        " auto b_hi = (util::stdx::static_simd_cast<I>(b >> 16) << 16) >> 16;"
        " auto c_lo = (util::stdx::static_simd_cast<I>(c & 0xFFFFu) << 16) >> 16;"
        " auto c_hi = (util::stdx::static_simd_cast<I>(c >> 16) << 16) >> 16;"
        " auto rlo = util::stdx::static_simd_cast<util::native<uint32_t>>(a_lo * b_lo + c_lo) & 0xFFFFu;"
        " auto rhi = util::stdx::static_simd_cast<util::native<uint32_t>>(a_hi * b_hi + c_hi) & 0xFFFFu;"
        " return rlo | (rhi << 16); }"
    ),
    "v_pk_mad_u16_vop3p": (
        "[](auto a, auto b, auto c) {"
        " auto a_lo = a & 0xFFFFu;"
        " auto a_hi = a >> 16;"
        " auto b_lo = b & 0xFFFFu;"
        " auto b_hi = b >> 16;"
        " auto c_lo = c & 0xFFFFu;"
        " auto c_hi = c >> 16;"
        " auto rlo = (a_lo * b_lo + c_lo) & 0xFFFFu;"
        " auto rhi = (a_hi * b_hi + c_hi) & 0xFFFFu;"
        " return rlo | (rhi << 16); }"
    ),
}

# VOP3P packed-16 f16 binary family. Each 32-bit lane holds 2 f16 values.
# Glue widens halves to f32, applies neg/neg_hi (sign-bit toggle), runs the
# per-half functor in f32, narrows back to f16, packs. No clamp on any
# pk_*_f16 scalar body (verified line 15109, 15519). NaN-input lanes can
# diverge in payload (same as the existing f16 ternary slice).
SIMD_VOP3P_PK_BINARY_FP16: dict[str, str] = {
    "v_pk_add_f16_vop3p": "[](auto a, auto b) { return a + b; }",
    "v_pk_mul_f16_vop3p": "[](auto a, auto b) { return a * b; }",
    "v_pk_max_f16_vop3p": "[](auto a, auto b) { return util::stdx::fmax(a, b); }",
    "v_pk_min_f16_vop3p": "[](auto a, auto b) { return util::stdx::fmin(a, b); }",
}

# pk_fma_f16 — 3-source FMA per half. NaN-input payload divergence accepted.
SIMD_VOP3P_PK_TERNARY_FP16: dict[str, str] = {
    "v_pk_fma_f16_vop3p": "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }",
}

# v_pk_mov_b32 — default-packing-only fast path. Each src is a 64-bit pair
# (consecutive VGPRs), result is (src0_lo, src1_hi). Functorless / fixed-op.
SIMD_VOP3P_MOV_B32: set[str] = {
    "v_pk_mov_b32_vop3p",
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
    "f32": ("float32_t", "{x}"),
    "f16": ("uint32_t", "util::f16_to_f32_simd({x})"),
    "i32": ("int32_t", "{x}"),
    "u32": ("uint32_t", "{x}"),
    # 16-bit integers read as uint32 lanes; sign-extend / mask the low 16 bits to
    # match the scalar static_cast<int16_t>/<uint16_t>.
    "i16": (
        "uint32_t",
        "((util::stdx::static_simd_cast<util::native<int32_t>>({x}) << 16) >> 16)",
    ),
    "u16": ("uint32_t", "({x} & 0xFFFFu)"),
    # 64-bit lanes: read directly as the native 64-bit lane type (read_simd64),
    # so the conversion is the identity. Routed through the VOPC64 glue.
    "f64": ("double", "{x}"),
    "i64": ("int64_t", "{x}"),
    "u64": ("uint64_t", "{x}"),
}

# relation -> mask expression over converted operands {a}, {b}. These mirror the
# generated scalar comparison expressions exactly (incl. float ordered/unordered
# NaN semantics): the n* forms are the logical negation of the base relation, so
# they are true on NaN; o/u test orderedness directly.
_VOPC_REL: dict[str, str] = {
    "eq": "{a} == {b}",
    "lt": "{a} < {b}",
    "le": "{a} <= {b}",
    "gt": "{a} > {b}",
    "ge": "{a} >= {b}",
    "ne": "{a} != {b}",  # integer not-equal
    # float less-or-greater: ordered, FALSE on NaN. Scalar body is (a<b)||(a>b),
    # NOT a!=b (which is TRUE on NaN). nlg is its logical negation.
    "lg": "({a} < {b}) || ({a} > {b})",
    "neq": "{a} != {b}",  # float not-equal
    "nge": "!({a} >= {b})",
    "ngt": "!({a} > {b})",
    "nle": "!({a} <= {b})",
    "nlg": "!(({a} < {b}) || ({a} > {b}))",
    "nlt": "!({a} < {b})",
    "o": "!util::stdx::isnan({a}) && !util::stdx::isnan({b})",
    "u": "util::stdx::isnan({a}) || util::stdx::isnan({b})",
}

# Constant relations carry no comparison: f is always-false, t/tru always-true.
# The mask type is taken from the converted-operand compare so it matches lane
# width regardless of suffix.
_VOPC_CONST: dict[str, str] = {"f": "false", "t": "true", "tru": "true"}

_VOPC_FLOAT_RELS = [
    "eq",
    "ge",
    "gt",
    "le",
    "lg",
    "lt",
    "neq",
    "nge",
    "ngt",
    "nle",
    "nlg",
    "nlt",
    "o",
    "u",
    "f",
    "tru",
]
_VOPC_INT_RELS = ["eq", "ge", "gt", "le", "lt", "ne", "f", "t"]


def _vopc_functor(conv: str, rel: str) -> str:
    ca = conv.format(x="a")
    cb = conv.format(x="b")
    if rel in _VOPC_CONST:
        body = f"decltype({ca} == {cb})({_VOPC_CONST[rel]})"
    else:
        body = _VOPC_REL[rel].format(a=ca, b=cb)
    return f"[](auto a, auto b) {{ return {body}; }}"


def _build_simd_vopc() -> dict[str, tuple[str, str]]:
    table: dict[str, tuple[str, str]] = {}
    # 16-/32-bit lane suffixes only; f64/i64/u64 (64-bit lane) wired separately.
    for suf in ("f16", "f32"):
        lane_t, conv = _VOPC_SUFFIX[suf]
        for rel in _VOPC_FLOAT_RELS:
            table[f"v_cmp_{rel}_{suf}_vopc"] = (lane_t, _vopc_functor(conv, rel))
    for suf in ("i16", "u16", "i32", "u32"):
        lane_t, conv = _VOPC_SUFFIX[suf]
        for rel in _VOPC_INT_RELS:
            table[f"v_cmp_{rel}_{suf}_vopc"] = (lane_t, _vopc_functor(conv, rel))
    return table


def _build_simd_vopc64() -> dict[str, tuple[str, str]]:
    """64-bit-lane VOPC compares (f64/i64/u64), routed through the VOPC64 glue."""
    table: dict[str, tuple[str, str]] = {}
    lane_t, conv = _VOPC_SUFFIX["f64"]
    for rel in _VOPC_FLOAT_RELS:
        table[f"v_cmp_{rel}_f64_vopc"] = (lane_t, _vopc_functor(conv, rel))
    for suf in ("i64", "u64"):
        lane_t, conv = _VOPC_SUFFIX[suf]
        for rel in _VOPC_INT_RELS:
            table[f"v_cmp_{rel}_{suf}_vopc"] = (lane_t, _vopc_functor(conv, rel))
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
    "[](auto a, auto b) {"
    " using U = util::native<uint32_t>;"
    " U exp = (a >> 23) & 0xFFu;"
    " U mant = a & 0x7FFFFFu;"
    " auto sgn = ((a >> 31) & 1u) != 0u;"
    " auto qnan = ((a >> 22) & 1u) != 0u;"
    " auto is_nan = (exp == 0xFFu) && (mant != 0u);"
    " auto is_inf = (exp == 0xFFu) && (mant == 0u);"
    " auto is_zero = (exp == 0u) && (mant == 0u);"
    " auto is_den = (exp == 0u) && (mant != 0u);"
    " auto is_norm = (exp >= 1u) && (exp <= 0xFEu);"
    " U cls(0u);"
    " util::stdx::where(is_nan && !qnan, cls) = 0x001u;"
    " util::stdx::where(is_nan && qnan, cls) = 0x002u;"
    " util::stdx::where(is_inf && sgn, cls) = 0x004u;"
    " util::stdx::where(is_norm && sgn, cls) = 0x008u;"
    " util::stdx::where(is_den && sgn, cls) = 0x010u;"
    " util::stdx::where(is_zero && sgn, cls) = 0x020u;"
    " util::stdx::where(is_zero && !sgn, cls) = 0x040u;"
    " util::stdx::where(is_den && !sgn, cls) = 0x080u;"
    " util::stdx::where(is_norm && !sgn, cls) = 0x100u;"
    " util::stdx::where(is_inf && !sgn, cls) = 0x200u;"
    " return (cls & b) != 0u; }"
)
_CMP_CLASS_F16 = (
    "[](auto a, auto b) {"
    " using U = util::native<uint32_t>;"
    " U h = a & 0xFFFFu;"
    " U exp = (h >> 10) & 0x1Fu;"
    " U mant = h & 0x3FFu;"
    " auto sgn = ((h >> 15) & 1u) != 0u;"
    " auto qnan = ((h >> 9) & 1u) != 0u;"
    " auto is_nan = (exp == 0x1Fu) && (mant != 0u);"
    " auto is_inf = (exp == 0x1Fu) && (mant == 0u);"
    " auto is_zero = (exp == 0u) && (mant == 0u);"
    " auto is_den = (exp == 0u) && (mant != 0u);"
    " auto is_norm = (exp >= 1u) && (exp <= 30u);"
    " U cls(0u);"
    " util::stdx::where(is_nan && !qnan, cls) = 0x001u;"
    " util::stdx::where(is_nan && qnan, cls) = 0x002u;"
    " util::stdx::where(is_inf && sgn, cls) = 0x004u;"
    " util::stdx::where(is_norm && sgn, cls) = 0x008u;"
    " util::stdx::where(is_den && sgn, cls) = 0x010u;"
    " util::stdx::where(is_zero && sgn, cls) = 0x020u;"
    " util::stdx::where(is_zero && !sgn, cls) = 0x040u;"
    " util::stdx::where(is_den && !sgn, cls) = 0x080u;"
    " util::stdx::where(is_norm && !sgn, cls) = 0x100u;"
    " util::stdx::where(is_inf && !sgn, cls) = 0x200u;"
    " return (cls & b) != 0u; }"
)
SIMD_VOPC_CLASS: dict[str, tuple[str, str]] = {
    "v_cmp_class_f16_vopc": ("uint32_t", _CMP_CLASS_F16),
    "v_cmp_class_f32_vopc": ("uint32_t", _CMP_CLASS_F32),
}


# v_cmp_class_f64: a 64-bit f64 value (src0) tested against a 32-bit class mask
# (vsrc1), so it needs the mixed-width class glue (try_execute_vopc_class_f64_simd)
# rather than the equal-width VOPC path. The functor receives src0 as
# native<uint64_t> raw bits and vsrc1 as a native_width64-wide narrow32<uint32_t>
# mask; it classifies the f64 from its raw bits (qNaN bit = mantissa MSB
# 0x0008000000000000), partitions into one of the 10 mutually exclusive classes,
# casts the small class code down to the 32-bit mask width, and returns
# (cls & mask) != 0. Pure bit decode, bit-exact with the scalar body. The
# class-bit layout matches the f16/f32 forms above.
_CMP_CLASS_F64 = (
    "[](auto s, auto m) {"
    " using U = util::native<uint64_t>;"
    " U exp = (s >> 52) & 0x7FFu;"
    " U mant = s & 0xFFFFFFFFFFFFFull;"
    " auto sgn = ((s >> 63) & 1u) != 0u;"
    " auto qnan = ((s >> 51) & 1u) != 0u;"
    " auto is_nan = (exp == 0x7FFu) && (mant != 0u);"
    " auto is_inf = (exp == 0x7FFu) && (mant == 0u);"
    " auto is_zero = (exp == 0u) && (mant == 0u);"
    " auto is_den = (exp == 0u) && (mant != 0u);"
    " auto is_norm = (exp >= 1u) && (exp <= 0x7FEu);"
    " U cls(0u);"
    " util::stdx::where(is_nan && !qnan, cls) = 0x001u;"
    " util::stdx::where(is_nan && qnan, cls) = 0x002u;"
    " util::stdx::where(is_inf && sgn, cls) = 0x004u;"
    " util::stdx::where(is_norm && sgn, cls) = 0x008u;"
    " util::stdx::where(is_den && sgn, cls) = 0x010u;"
    " util::stdx::where(is_zero && sgn, cls) = 0x020u;"
    " util::stdx::where(is_zero && !sgn, cls) = 0x040u;"
    " util::stdx::where(is_den && !sgn, cls) = 0x080u;"
    " util::stdx::where(is_norm && !sgn, cls) = 0x100u;"
    " util::stdx::where(is_inf && !sgn, cls) = 0x200u;"
    " auto cls32 = util::stdx::static_simd_cast<util::narrow32<uint32_t>>(cls);"
    " return (cls32 & m) != 0u; }"
)
SIMD_VOPC_CLASS_F64: dict[str, str] = {
    "v_cmp_class_f64_vopc": _CMP_CLASS_F64,
}


# VOP3 forms of v_cmp_class. Same classification as the VOPC forms (the functors
# are reused verbatim), but the VOP3 glue additionally applies the abs/neg source
# modifiers to src0's raw bits before classifying, reads the class mask from src1,
# and merges the result into the SGPR-pair dst (inst.vdst.read/write_scalar64)
# rather than VCC. The per-op sign-bit mask (abs clears it, neg flips it) is passed
# to the glue: 0x8000 (f16) / 0x80000000 (f32) share a uint32 lane, 0x8000…0 (f64)
# is 64-bit. f16/f32 go through the 32-bit-value glue; f64 through the 64-bit one.
SIMD_VOP3_CLASS: dict[str, tuple[str, str]] = {
    "v_cmp_class_f16_vop3": ("0x8000u", _CMP_CLASS_F16),
    "v_cmp_class_f32_vop3": ("0x80000000u", _CMP_CLASS_F32),
}
SIMD_VOP3_CLASS_F64: dict[str, str] = {
    "v_cmp_class_f64_vop3": _CMP_CLASS_F64,
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
_VOP3_FLOAT_RELS = _VOPC_FLOAT_RELS + ["t"]


def _build_simd_vopc_vop3_int_32() -> dict[str, tuple[str, str]]:
    """VOP3 form of the 32-bit-lane integer VOPC relations (i16/u16/i32/u32).

    Keyed by ``_vop3``; the functor matches the VOPC one (no modifiers).
    """
    table: dict[str, tuple[str, str]] = {}
    for suf in ("i16", "u16", "i32", "u32"):
        lane_t, conv = _VOPC_SUFFIX[suf]
        for rel in _VOPC_INT_RELS:
            table[f"v_cmp_{rel}_{suf}_vop3"] = (lane_t, _vopc_functor(conv, rel))
    return table


def _build_simd_vopc_vop3_int_64() -> dict[str, tuple[str, str]]:
    """VOP3 form of the 64-bit-lane integer VOPC relations (i64/u64). Same
    keying / functor as the VOPC64 path; no modifiers."""
    table: dict[str, tuple[str, str]] = {}
    for suf in ("i64", "u64"):
        lane_t, conv = _VOPC_SUFFIX[suf]
        for rel in _VOPC_INT_RELS:
            table[f"v_cmp_{rel}_{suf}_vop3"] = (lane_t, _vopc_functor(conv, rel))
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
    _, conv = _VOPC_SUFFIX["f32"]
    for rel in _VOP3_FLOAT_RELS:
        table[f"v_cmp_{rel}_f32_vop3"] = _vopc_functor(conv, rel)
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
    _, conv = _VOPC_SUFFIX["f32"]
    for rel in _VOP3_FLOAT_RELS:
        table[f"v_cmp_{rel}_f16_vop3"] = _vopc_functor(conv, rel)
    return table


SIMD_VOPC_VOP3_F16: dict[str, str] = _build_simd_vopc_vop3_f16()


def _build_simd_vopc_vop3_f64() -> dict[str, str]:
    """VOP3 form of the f64 VOPC relations (17 — same set as f32/f16).

    Lane type `double`; the glue applies abs/neg outside the functor on the
    f64 value, so the functor itself takes already-modified `native<double>`
    arguments and reuses the VOPC64 f64 builder (identity operand conversion).
    """
    table: dict[str, str] = {}
    _, conv = _VOPC_SUFFIX["f64"]
    for rel in _VOP3_FLOAT_RELS:
        table[f"v_cmp_{rel}_f64_vop3"] = _vopc_functor(conv, rel)
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
    "v_add_i32_vop3": ("uint32_t", "[](auto a, auto b) { return a + b; }"),
    "v_sub_i32_vop3": ("uint32_t", "[](auto a, auto b) { return a - b; }"),
    "v_add_nc_i32_vop3": ("uint32_t", "[](auto a, auto b) { return a + b; }"),
    "v_add_nc_u32_vop3": ("uint32_t", "[](auto a, auto b) { return a + b; }"),
    "v_sub_nc_i32_vop3": ("uint32_t", "[](auto a, auto b) { return a - b; }"),
    "v_sub_nc_u32_vop3": ("uint32_t", "[](auto a, auto b) { return a - b; }"),
    "v_subrev_nc_u32_vop3": ("uint32_t", "[](auto a, auto b) { return b - a; }"),
    "v_add_i16_vop3": ("uint32_t", "[](auto a, auto b) { return (a + b) & 0xFFFFu; }"),
    "v_sub_i16_vop3": ("uint32_t", "[](auto a, auto b) { return (a - b) & 0xFFFFu; }"),
    "v_add_nc_i16_vop3": (
        "uint32_t",
        "[](auto a, auto b) { return (a + b) & 0xFFFFu; }",
    ),
    "v_add_nc_u16_vop3": (
        "uint32_t",
        "[](auto a, auto b) { return (a + b) & 0xFFFFu; }",
    ),
    "v_sub_nc_i16_vop3": (
        "uint32_t",
        "[](auto a, auto b) { return (a - b) & 0xFFFFu; }",
    ),
    "v_sub_nc_u16_vop3": (
        "uint32_t",
        "[](auto a, auto b) { return (a - b) & 0xFFFFu; }",
    ),
    # 32-bit integer multiply: low 32 bits of the product is just `a * b`
    # (uint32 wrap is identical signed/unsigned for the low half).
    "v_mul_lo_u32_vop3": ("uint32_t", "[](auto a, auto b) { return a * b; }"),
    # High 32 bits of the 32x32 -> 64 multiply, via the same widening pattern
    # as v_mul_hi_u32_u24: cast lanes to a 64-bit fixed_size_simd, multiply,
    # shift right by 32, narrow back. Unsigned uses uint64_t / logical shift;
    # signed uses int64_t / arithmetic shift (preserves the sign).
    "v_mul_hi_u32_vop3": (
        "uint32_t",
        "[](auto a, auto b) {"
        " using U64 = util::stdx::fixed_size_simd<uint64_t, util::native<uint32_t>::size()>;"
        " auto pa = util::stdx::static_simd_cast<U64>(a);"
        " auto pb = util::stdx::static_simd_cast<U64>(b);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>((pa * pb) >> decltype(pa)(32)); }",
    ),
    "v_mul_hi_i32_vop3": (
        "uint32_t",
        "[](auto a, auto b) {"
        " using I64 = util::stdx::fixed_size_simd<int64_t, util::native<int32_t>::size()>;"
        " auto sa = util::stdx::static_simd_cast<util::native<int32_t>>(a);"
        " auto sb = util::stdx::static_simd_cast<util::native<int32_t>>(b);"
        " auto pa = util::stdx::static_simd_cast<I64>(sa);"
        " auto pb = util::stdx::static_simd_cast<I64>(sb);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>((pa * pb) >> decltype(pa)(32)); }",
    ),
    # v_bfm_b32: ((1 << (a & 31)) - 1) << (b & 31). Two shift counts both
    # masked to low 5 bits — same vpsllvd-vs-shl rationale as v_lshl_add_u32.
    "v_bfm_b32_vop3": (
        "uint32_t",
        "[](auto a, auto b) { return ((util::native<uint32_t>(1u) << (a & 31u)) - 1u) << (b & 31u); }",
    ),
    # v_pack_b32_f16: pack two f16 halves into a b32. low16(src0) into the low
    # half, low16(src1) into the high half. The scalar body applies no abs/neg
    # despite the f16 typing (it pre-masks the operands to 16 bits before the
    # shift), and clamp/omod are likewise unused. Pure integer bit-pack.
    "v_pack_b32_f16_vop3": (
        "uint32_t",
        "[](auto a, auto b) { return (a & 0xFFFFu) | ((b & 0xFFFFu) << 16); }",
    ),
    # 16-bit-lane bitwise binary VOP3 ops (RDNA3+; CDNA4 does not decode).
    # The scalar body computes `uint16_t(uint16_t(a) OP uint16_t(b))` and
    # writes the result as a zero-extended uint32; SIMD reproduces with a
    # 32-bit `& 0xFFFFu` mask on the result. No modifiers (integer ops).
    "v_and_b16_vop3": ("uint32_t", "[](auto a, auto b) { return (a & b) & 0xFFFFu; }"),
    "v_or_b16_vop3": ("uint32_t", "[](auto a, auto b) { return (a | b) & 0xFFFFu; }"),
    "v_xor_b16_vop3": ("uint32_t", "[](auto a, auto b) { return (a ^ b) & 0xFFFFu; }"),
}

# VOP3 unary integer ops without a VOP1 twin. Reuse the VOP1 unary glue
# (operand shape is identical: src0 in, vdst out, 32-bit lanes) — only the
# probe routing key differs. RDNA3+; CDNA4 does not decode.
SIMD_VOP3_UNARY_INT_EXTRA: dict[str, tuple[str, str, str]] = {
    # v_not_b16: `uint16_t(~src0)`, zero-extended. Same shape as v_not_b32 but
    # masked to 16 bits.
    "v_not_b16_vop3": (
        "uint32_t",
        "uint32_t",
        "[](auto a) { return (~a) & 0xFFFFu; }",
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
SIMD_VOP3_BINARY_FP64: dict[str, str] = {
    "v_add_f64_vop3": "[](auto a, auto b) { return a + b; }",
    "v_mul_f64_vop3": "[](auto a, auto b) { return a * b; }",
    "v_max_f64_vop3": "[](auto a, auto b) { return util::stdx::fmax(a, b); }",
    "v_min_f64_vop3": "[](auto a, auto b) { return util::stdx::fmin(a, b); }",
}

# Plain f64 unary: scalar bodies are std::ceil / std::floor / std::trunc /
# std::nearbyint applied to the (modifier-applied) double, then omod/clamp on
# the result. stdx provides ceil/floor/trunc/nearbyint as native<double>
# primitives — bit-identical to the scalar libm calls for every finite/Inf
# input (and the NaN result has the same payload because the scalar libm
# rounding ops are sign/payload preserving on NaN inputs).
SIMD_VOP3_UNARY_FP64: dict[str, str] = {
    "v_ceil_f64_vop3": "[](auto a) { return util::ceil_simd(a); }",
    "v_floor_f64_vop3": "[](auto a) { return util::floor_simd(a); }",
    "v_trunc_f64_vop3": "[](auto a) { return util::trunc_simd(a); }",
    "v_rndne_f64_vop3": "[](auto a) { return util::rndne_simd(a); }",
    # sqrt_f64 is correctly-rounded IEEE (scalar uses transcendental::sqrt_f64
    # which is `std::sqrt` after NaN/negative guards); stdx::sqrt matches.
    "v_sqrt_f64_vop3": (
        "[](auto a) {"
        " auto r = util::stdx::sqrt(a);"
        " util::stdx::where(util::stdx::isnan(a), r) = a;"
        " util::stdx::where(a < 0.0, r) = std::numeric_limits<double>::quiet_NaN();"
        " return r; }"
    ),
    # v_fract_f64: scalar = v - std::floor(v); stdx::floor on native<double>
    # matches std::floor bit-exact (NaN-floor(NaN) = NaN; NaN result skipped
    # by the test like any other NaN-result lane).
    "v_fract_f64_vop3": "[](auto a) { return a - util::floor_simd(a); }",
    # v_rcp_f64 / v_rsq_f64: scalar uses transcendental::*_f64 with explicit
    # NaN passthrough, ±0 -> copysign(Inf, x), ±Inf -> copysign(0, x); negative
    # rsq inputs -> qNaN. Plain 1.0 / x and 1.0 / sqrt match the IEEE result
    # for all non-NaN inputs; NaN-result lanes are skipped by the A/B test.
    "v_rcp_f64_vop3": "[](auto a) { return util::native<double>(1.0) / a; }",
    "v_rsq_f64_vop3": "[](auto a) { return util::native<double>(1.0) / util::stdx::sqrt(a); }",
    # v_mov_b64 with VOP3 modifiers: scalar bit_casts to double, applies
    # abs/neg/omod/clamp, bit_casts back. The f64 unary glue operates entirely
    # in native<double> domain, so an identity functor + the same modifier
    # helpers reproduce the scalar bit pattern exactly.
    "v_mov_b64_vop3": "[](auto a) { return a; }",
}


# VOP3 f16 unary: widen f16->f32, apply abs/neg, op, omod/clamp, narrow back.
# Rounding ops (ceil/floor/trunc/rndne) have no FTZ. sqrt is also no-FTZ
# (transcendental::sqrt_f32 keeps denormals). The four transcendentals
# (rcp/rsq/exp/log) reuse util::*_f32_simd which already wraps the scalar
# transcendental::flush_denorm_f32 carve-outs (FTZ input + matching ±0/Inf
# blends + NaN-passthrough), so the f16 scalar
# f32_to_f16(transcendental::op_f32(f16_to_f32(...))) maps directly.
SIMD_VOP3_UNARY_FP16: dict[str, str] = {
    "v_ceil_f16_vop3": "[](auto a) { return util::ceil_simd(a); }",
    "v_floor_f16_vop3": "[](auto a) { return util::floor_simd(a); }",
    "v_trunc_f16_vop3": "[](auto a) { return util::trunc_simd(a); }",
    "v_rndne_f16_vop3": "[](auto a) { return util::rndne_simd(a); }",
    "v_fract_f16_vop3": "[](auto a) { return a - util::floor_simd(a); }",
    "v_sqrt_f16_vop3": (
        "[](auto a) {"
        " auto r = util::stdx::sqrt(a);"
        " util::stdx::where(util::stdx::isnan(a), r) = a;"
        " util::stdx::where(a < 0.0f, r) = std::numeric_limits<float>::quiet_NaN();"
        " return r; }"
    ),
    "v_rcp_f16_vop3": "[](auto a) { return util::rcp_f32_simd(a); }",
    "v_rsq_f16_vop3": "[](auto a) { return util::rsq_f32_simd(a); }",
    "v_exp_f16_vop3": "[](auto a) { return util::exp_f32_simd(a); }",
    "v_log_f16_vop3": "[](auto a) { return util::log_f32_simd(a); }",
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
    "v_fma_f32_vop3": "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }",
    "v_mad_f32_vop3": "[](auto a, auto b, auto c) { return a * b + c; }",
    "v_mad_legacy_f32_vop3": "[](auto a, auto b, auto c) { return a * b + c; }",
    # v_fma_dx9_zero_f32: the scalar body uses plain std::fma without any
    # zero-killing semantics — DX9 zero rules are NOT actually applied in
    # the generated body (verified in execute_shared.h). Alias of fma.
    "v_fma_dx9_zero_f32_vop3": (
        "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }"
    ),
    # v_div_fixup_f32: per-AMD-spec `else if` cascade selecting the result
    # among NaN/Inf/zero copysign cases. Lives as a helper in simd_glue.h
    # (div_fixup_f32_simd) — bit-exact match to the scalar body's predicate
    # tree applied lowest-priority-first so higher-priority `where` blends
    # overwrite. Omod/clamp DO apply (scalar applies them at end).
    "v_div_fixup_f32_vop3": (
        "[](auto p, auto b, auto c) { return ::rocjitsu::amdgpu::div_fixup_f32_simd(p, b, c); }"
    ),
    # v_div_fixup_f16 / v_div_fixup_legacy_f16: the SCALAR bodies read src0/1/2
    # as raw f32 (no f16<->f32 widening — they bit_cast<float>(read_lane(uint32))
    # and write the f32 result back through bit_cast<uint32>). Verified inline
    # at line 9389 (f16) and 9570 (legacy_f16). Effectively the same operation
    # as v_div_fixup_f32_vop3, just under a different mnemonic. Reuses the
    # helper verbatim.
    "v_div_fixup_f16_vop3": (
        "[](auto p, auto b, auto c) { return ::rocjitsu::amdgpu::div_fixup_f32_simd(p, b, c); }"
    ),
    "v_div_fixup_legacy_f16_vop3": (
        "[](auto p, auto b, auto c) { return ::rocjitsu::amdgpu::div_fixup_f32_simd(p, b, c); }"
    ),
}

# f16 ternary — widen each src to f32, op in f32, narrow back. Same NaN
# carve-out as the f32 path.
SIMD_VOP3_TERNARY_FP16: dict[str, str] = {
    "v_fma_f16_vop3": "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }",
    "v_mad_f16_vop3": "[](auto a, auto b, auto c) { return a * b + c; }",
    "v_mad_legacy_f16_vop3": "[](auto a, auto b, auto c) { return a * b + c; }",
    "v_fma_legacy_f16_vop3": "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }",
}

# f64 ternary FMA.
SIMD_VOP3_TERNARY_FP64: dict[str, str] = {
    "v_fma_f64_vop3": "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }",
    # v_div_fixup_f64: 64-bit-lane div_fixup cascade (same shape as f32, see
    # SIMD_VOP3_TERNARY_FP32 above).
    "v_div_fixup_f64_vop3": (
        "[](auto p, auto b, auto c) { return ::rocjitsu::amdgpu::div_fixup_f64_simd(p, b, c); }"
    ),
}

# --- VOP3 dst-accumulate FMA / MAC (vdst is the accumulator) ----------------
#
# v_fmac / v_mac per-isa classes only initialize src0+src1+vdst; the third FMA
# operand IS vdst (no src2 Operand). The accumulate-form glue reads inst.vdst
# as the third operand and applies abs/neg only to src0/src1 (per scalar body).
# NaN payload divergence accepted, same as the non-accumulate ternary slice.
SIMD_VOP3_FMAC_FP32: dict[str, str] = {
    "v_fmac_f32_vop3": "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }",
    "v_mac_f32_vop3": "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }",
    # v_fmac_dx9_zero_f32: scalar body uses plain std::fma — DX9 zero
    # semantics NOT applied (verified). Alias of fmac.
    "v_fmac_dx9_zero_f32_vop3": (
        "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }"
    ),
}
SIMD_VOP3_FMAC_FP16: dict[str, str] = {
    "v_fmac_f16_vop3": "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }",
    "v_mac_f16_vop3": "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }",
}
SIMD_VOP3_FMAC_FP64: dict[str, str] = {
    "v_fmac_f64_vop3": "[](auto a, auto b, auto c) { return util::stdx::fma(a, b, c); }",
}

# --- VOP3 ldexp (mixed-width: fp src0 + int32 src1 exp) --------------------
#
# stdx::ldexp on native<float|double> with same-size int simd is bit-exact to
# std::ldexp for every input including NaN (proven in the v_ldexp_f16 VOP2
# slice).
SIMD_VOP3_LDEXP_FP32: dict[str, str] = {
    # stdx::ldexp wants the integer arg as fixed_size_simd<int, size> matching
    # the float abi, not the native<int32_t> the operand reader returns.
    "v_ldexp_f32_vop3": (
        "[](auto a, auto e) {"
        " return util::stdx::ldexp(a, util::stdx::static_simd_cast<"
        "util::stdx::fixed_size_simd<int, util::native<float>::size()>>(e)); }"
    ),
}
SIMD_VOP3_LDEXP_FP64: dict[str, str] = {
    # narrow32<int32_t> is already an 8-wide fixed_size_simd; just re-cast to
    # the int-typed equivalent so stdx::ldexp accepts the matching abi.
    "v_ldexp_f64_vop3": (
        "[](auto a, auto e) {"
        " return util::stdx::ldexp(a, util::stdx::static_simd_cast<"
        "util::stdx::fixed_size_simd<int, util::native_width64>>(e)); }"
    ),
}


SIMD_VOP3_TERNARY_INT: dict[str, tuple[str, str]] = {
    "v_add3_u32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) { return a + b + c; }",
    ),
    "v_or3_b32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) { return a | b | c; }",
    ),
    "v_xor3_b32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) { return a ^ b ^ c; }",
    ),
    "v_xad_u32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) { return (a ^ b) + c; }",
    ),
    "v_and_or_b32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) { return (a & b) | c; }",
    ),
    "v_lshl_or_b32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) { return (a << (b & 31u)) | c; }",
    ),
    # v_mad_i32_i24: low-24 sign-extended a, b -> int32 multiply (low 32 of the
    # 48-bit product, identical signed/unsigned for the low half) + int32(c).
    "v_mad_i32_i24_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " auto sa = (util::stdx::static_simd_cast<util::native<int32_t>>(a) << 8) >> 8;"
        " auto sb = (util::stdx::static_simd_cast<util::native<int32_t>>(b) << 8) >> 8;"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>("
        "sa * sb + util::stdx::static_simd_cast<util::native<int32_t>>(c)); }",
    ),
    # v_mad_u32_u24: low-24 mask a, b, multiply, add c.
    "v_mad_u32_u24_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) { return (a & 0x00FFFFFFu) * (b & 0x00FFFFFFu) + c; }",
    ),
    # v_alignbit_b32: low 32 of ((u64(a) << 32) | b) >> (c & 31). Per-lane
    # variable shift on a 64-bit-lane fixed_size_simd<u64> — proven on the
    # widening mul_hi pattern; shift count is masked to [0, 31] so vpsrlvq
    # and scalar shr agree on the result.
    "v_alignbit_b32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using U64 = util::stdx::fixed_size_simd<uint64_t, util::native<uint32_t>::size()>;"
        " auto va = util::stdx::static_simd_cast<U64>(a);"
        " auto vb = util::stdx::static_simd_cast<U64>(b);"
        " auto val = (va << 32) | vb;"
        " auto sh = util::stdx::static_simd_cast<U64>(c & 31u);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(val >> sh); }",
    ),
    # v_alignbyte_b32: same widen but shift count is (c & 3) * 8 = {0,8,16,24}.
    "v_alignbyte_b32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using U64 = util::stdx::fixed_size_simd<uint64_t, util::native<uint32_t>::size()>;"
        " auto va = util::stdx::static_simd_cast<U64>(a);"
        " auto vb = util::stdx::static_simd_cast<U64>(b);"
        " auto val = (va << 32) | vb;"
        " auto sh = util::stdx::static_simd_cast<U64>((c & 3u) * 8u);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(val >> sh); }",
    ),
    # v_bfe_u32: unsigned bitfield extract. off = src1 & 31, w = src2 & 31.
    # Scalar returns 0 when w == 0; the mask formula `(1 << w) - 1` evaluates
    # to 0 at w == 0 so `(src >> off) & 0 == 0` already matches without a
    # special case. Per-lane shift counts are pre-masked to 5 bits (scalar
    # body's `& 31u`) so vpsrlvd / vpsllvd produce the same result as scalar
    # shr / shl. Functorless / direct uint32 expression.
    "v_bfe_u32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " auto off = b & 31u;"
        " auto w = c & 31u;"
        " auto mask = (util::native<uint32_t>(1u) << w) - 1u;"
        " return (a >> off) & mask; }",
    ),
    # v_bfe_i32: signed bitfield extract. After the unsigned extract step
    # (mask-and-shift in the int32 domain to keep the shift arithmetic), if
    # the extracted field's top bit (bit w-1) is set then the upper bits are
    # OR'd with ~mask, matching the scalar `val |= -(1 << w)` sign-extend.
    # `(w - 1) & 31u` keeps the top-bit shift well-defined when w == 0; the
    # whole result is then forced to 0 on that lane.
    "v_bfe_i32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " using U = util::native<uint32_t>;"
        " auto off = b & 31u;"
        " auto w = c & 31u;"
        " auto sa = util::stdx::static_simd_cast<I>(a);"
        " auto soff = util::stdx::static_simd_cast<I>(off);"
        " auto mask = (U(1u) << w) - 1u;"
        " auto smask = util::stdx::static_simd_cast<I>(mask);"
        " auto val = (sa >> soff) & smask;"
        " auto top = util::stdx::static_simd_cast<I>(U(1u) << ((w - 1u) & 31u));"
        " util::stdx::where((val & top) != I(0), val) = val | ~smask;"
        " util::stdx::where(simd_mask_as<int32_t>(w == 0u), val) = I(0);"
        " return util::stdx::static_simd_cast<U>(val); }",
    ),
    # The shift count is masked to the low 5 bits to match the scalar body's
    # x86 `shl` semantics (which mask cl to 5 bits) — stdx's `<<` on native<u32>
    # lowers to vpsllvd, which zeros out lanes where the count is >= 32 rather
    # than masking, so SIMD without the explicit `& 31u` diverges from scalar
    # whenever src1 (or src2 below) is >= 32 (verified empirically on the test
    # value 0x80000000). The scalar body's `<<` is C++ UB at those counts; the
    # masked SIMD form reproduces the x86 scalar result for every value of the
    # shift operand on this host.
    "v_lshl_add_u32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) { return (a << (b & 31u)) + c; }",
    ),
    "v_add_lshl_u32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) { return (a + b) << (c & 31u); }",
    ),
    "v_bfi_b32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) { return (a & b) | (~a & c); }",
    ),
    # --- int min3 / max3 / med3 (u32 / i32 / u16 / i16). Scalar bodies route
    # through std::min/max on the typed (16/32-bit) values. SIMD analogs use
    # stdx::min/max on the matching simd type, then truncate/sign-extend back
    # to uint32_t to match the scalar's zero/sign-extension semantics.
    "v_min3_u32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " return util::stdx::min(util::stdx::min(a, b), c); }",
    ),
    "v_max3_u32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " return util::stdx::max(util::stdx::max(a, b), c); }",
    ),
    "v_med3_u32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " return util::stdx::max(util::stdx::min(util::stdx::max(a, b), c),"
        " util::stdx::min(a, b)); }",
    ),
    "v_min3_i32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " auto sa = util::stdx::static_simd_cast<I>(a);"
        " auto sb = util::stdx::static_simd_cast<I>(b);"
        " auto sc = util::stdx::static_simd_cast<I>(c);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>("
        "util::stdx::min(util::stdx::min(sa, sb), sc)); }",
    ),
    "v_max3_i32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " auto sa = util::stdx::static_simd_cast<I>(a);"
        " auto sb = util::stdx::static_simd_cast<I>(b);"
        " auto sc = util::stdx::static_simd_cast<I>(c);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>("
        "util::stdx::max(util::stdx::max(sa, sb), sc)); }",
    ),
    "v_med3_i32_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " auto sa = util::stdx::static_simd_cast<I>(a);"
        " auto sb = util::stdx::static_simd_cast<I>(b);"
        " auto sc = util::stdx::static_simd_cast<I>(c);"
        " auto r = util::stdx::max(util::stdx::min(util::stdx::max(sa, sb), sc),"
        " util::stdx::min(sa, sb));"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(r); }",
    ),
    # 16-bit forms: scalar masks low 16 bits + casts to i16 / u16, runs the
    # 3-source min/max/med, then zero-extends back to uint32_t. SIMD: do the
    # mask + signed shift to sign-extend (i16 path), run the op, then mask
    # 0xFFFF on the result.
    "v_min3_u16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " auto ua = a & 0xFFFFu;"
        " auto ub = b & 0xFFFFu;"
        " auto uc = c & 0xFFFFu;"
        " return util::stdx::min(util::stdx::min(ua, ub), uc); }",
    ),
    "v_max3_u16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " auto ua = a & 0xFFFFu;"
        " auto ub = b & 0xFFFFu;"
        " auto uc = c & 0xFFFFu;"
        " return util::stdx::max(util::stdx::max(ua, ub), uc); }",
    ),
    "v_med3_u16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " auto ua = a & 0xFFFFu;"
        " auto ub = b & 0xFFFFu;"
        " auto uc = c & 0xFFFFu;"
        " return util::stdx::max(util::stdx::min(util::stdx::max(ua, ub), uc),"
        " util::stdx::min(ua, ub)); }",
    ),
    "v_min3_i16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " auto sa = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;"
        " auto sb = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;"
        " auto sc = (util::stdx::static_simd_cast<I>(c & 0xFFFFu) << 16) >> 16;"
        " auto r = util::stdx::min(util::stdx::min(sa, sb), sc);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(r) & 0xFFFFu; }",
    ),
    "v_max3_i16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " auto sa = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;"
        " auto sb = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;"
        " auto sc = (util::stdx::static_simd_cast<I>(c & 0xFFFFu) << 16) >> 16;"
        " auto r = util::stdx::max(util::stdx::max(sa, sb), sc);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(r) & 0xFFFFu; }",
    ),
    "v_med3_i16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " auto sa = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;"
        " auto sb = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;"
        " auto sc = (util::stdx::static_simd_cast<I>(c & 0xFFFFu) << 16) >> 16;"
        " auto r = util::stdx::max(util::stdx::min(util::stdx::max(sa, sb), sc),"
        " util::stdx::min(sa, sb));"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(r) & 0xFFFFu; }",
    ),
    # v_minmax / v_maxmin int forms DEFERRED. The scalar body uses
    # std::fmin/std::fmax on integer operands → double round-trip; for
    # negative int32 the resulting double, when cast back through
    # vdst.write_lane (uint32_t), goes through an implementation-defined
    # double→uint conversion that GCC saturates to UINT_MAX. The vector
    # path computes the correct integer-domain result, so SIMD vs scalar
    # diverges on negative-int32 inputs (test demonstrated lane=26 rot=13
    # case). Pending PR 6470 scalar fix.
    # --- int16 / uint16 mad (low-16 mul-add, result masked to 16 bits) and
    # mad_legacy_i16 / mad_legacy_u16 (identical scalar bodies; legacy
    # naming). SIMD operates in int32 width — 16x16 mul + add stays within
    # 32 bits, sign/zero-extend on inputs and mask 0xFFFF on output.
    "v_mad_i16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " auto sa = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;"
        " auto sb = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;"
        " auto sc = (util::stdx::static_simd_cast<I>(c & 0xFFFFu) << 16) >> 16;"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(sa * sb + sc) & 0xFFFFu; }",
    ),
    "v_mad_legacy_i16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " auto sa = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;"
        " auto sb = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;"
        " auto sc = (util::stdx::static_simd_cast<I>(c & 0xFFFFu) << 16) >> 16;"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(sa * sb + sc) & 0xFFFFu; }",
    ),
    "v_mad_u16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " return ((a & 0xFFFFu) * (b & 0xFFFFu) + (c & 0xFFFFu)) & 0xFFFFu; }",
    ),
    "v_mad_legacy_u16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " return ((a & 0xFFFFu) * (b & 0xFFFFu) + (c & 0xFFFFu)) & 0xFFFFu; }",
    ),
    # Widening 16x16+32 → 32-bit forms. src0/src1 are 16-bit (sign or zero
    # extended to int32), src2 is full 32-bit, result is 32-bit.
    "v_mad_i32_i16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {"
        " using I = util::native<int32_t>;"
        " auto sa = (util::stdx::static_simd_cast<I>(a & 0xFFFFu) << 16) >> 16;"
        " auto sb = (util::stdx::static_simd_cast<I>(b & 0xFFFFu) << 16) >> 16;"
        " auto sc = util::stdx::static_simd_cast<I>(c);"
        " return util::stdx::static_simd_cast<util::native<uint32_t>>(sa * sb + sc); }",
    ),
    "v_mad_u32_u16_vop3": (
        "uint32_t",
        "[](auto a, auto b, auto c) {" " return (a & 0xFFFFu) * (b & 0xFFFFu) + c; }",
    ),
}


def simd_probe_line(template_name: str) -> str | None:
    """Return the SIMD fast-path probe block for a kernel, or None."""
    if template_name in SIMD_VOP2_CNDMASK:
        return "  ROCJITSU_TRY_SIMD_VOP2_CNDMASK();"
    if template_name in SIMD_VOP3_CNDMASK:
        return "  ROCJITSU_TRY_SIMD_VOP3_CNDMASK();"
    if template_name in SIMD_VOP3_CNDMASK_B16:
        return "  ROCJITSU_TRY_SIMD_VOP3_CNDMASK_B16();"
    spec2 = SIMD_VOP2_BINARY.get(template_name)
    if spec2 is not None:
        cpp_t, cpp_op = spec2
        return f"  ROCJITSU_TRY_SIMD_VOP2_BINARY({cpp_t}, {cpp_op});"
    spec1 = SIMD_VOP1_UNARY.get(template_name)
    if spec1 is not None:
        cpp_tin, cpp_tout, cpp_op = spec1
        return f"  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {cpp_op});"
    specc = SIMD_VOP2_CARRY.get(template_name)
    if specc is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP2_CARRY({specc});"
    spect = SIMD_VOP2_TERNARY.get(template_name)
    if spect is not None:
        cpp_t, k_expr, cpp_op = spect
        return f"  ROCJITSU_TRY_SIMD_VOP2_TERNARY({cpp_t}, {k_expr}, {cpp_op});"
    specf64 = SIMD_VOP2_FMA_F64.get(template_name)
    if specf64 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP2_FMA_F64({specf64});"
    spec1f64 = SIMD_VOP1_UNARY_F64.get(template_name)
    if spec1f64 is not None:
        lane_t, cpp_op = spec1f64
        return f"  ROCJITSU_TRY_SIMD_VOP1_UNARY_F64({lane_t}, {cpp_op});"
    speccvtout = SIMD_CVT_F64_TO_B32.get(template_name)
    if speccvtout is not None:
        out_t, cpp_op = speccvtout
        return f"  ROCJITSU_TRY_SIMD_CVT_F64_TO_B32({out_t}, {cpp_op});"
    speccvtin = SIMD_CVT_B32_TO_F64.get(template_name)
    if speccvtin is not None:
        in_t, cpp_op = speccvtin
        return f"  ROCJITSU_TRY_SIMD_CVT_B32_TO_F64({in_t}, {cpp_op});"
    specvopc = SIMD_VOPC.get(template_name)
    if specvopc is not None:
        lane_t, cpp_op = specvopc
        return f"  ROCJITSU_TRY_SIMD_VOPC({lane_t}, {cpp_op});"
    specclass = SIMD_VOPC_CLASS.get(template_name)
    if specclass is not None:
        lane_t, cpp_op = specclass
        return f"  ROCJITSU_TRY_SIMD_VOPC({lane_t}, {cpp_op});"
    specclass64 = SIMD_VOPC_CLASS_F64.get(template_name)
    if specclass64 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOPC_CLASS_F64({specclass64});"
    spec3class = SIMD_VOP3_CLASS.get(template_name)
    if spec3class is not None:
        sm, cpp_op = spec3class
        return f"  ROCJITSU_TRY_SIMD_VOP3_CLASS_B32({sm}, {cpp_op});"
    spec3class64 = SIMD_VOP3_CLASS_F64.get(template_name)
    if spec3class64 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3_CLASS_F64(0x8000000000000000ull, {spec3class64});"
    specvopc64 = SIMD_VOPC64.get(template_name)
    if specvopc64 is not None:
        lane_t, cpp_op = specvopc64
        return f"  ROCJITSU_TRY_SIMD_VOPC64({lane_t}, {cpp_op});"
    # VOP3 form of the integer/bitwise VOPC compares (i16/u16/i32/u32 and
    # i64/u64). Same functor as the VOPC table (no modifiers), but the merge
    # writes the SGPR-pair dst instead of VCC.
    specvopcv3i32 = SIMD_VOPC_VOP3_INT_32.get(template_name)
    if specvopcv3i32 is not None:
        lane_t, cpp_op = specvopcv3i32
        return f"  ROCJITSU_TRY_SIMD_VOPC_VOP3_INT({lane_t}, {cpp_op});"
    specvopcv3i64 = SIMD_VOPC_VOP3_INT_64.get(template_name)
    if specvopcv3i64 is not None:
        lane_t, cpp_op = specvopcv3i64
        return f"  ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT({lane_t}, {cpp_op});"
    # VOP3 form of the f32 VOPC relational compares (17 ops: 16 relations +
    # 't' constant). Per-source abs/neg modifiers applied outside the functor
    # via the fp32 VOPC glue; SGPR-pair dst merge identical to the integer path.
    specvopcv3f32 = SIMD_VOPC_VOP3_F32.get(template_name)
    if specvopcv3f32 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOPC_VOP3_FP32({specvopcv3f32});"
    # VOP3 form of the f16 VOPC relational compares (17 ops). The glue widens
    # raw lanes to f32 then applies abs/neg in f32 domain — matching the scalar
    # body's f16_to_f32 -> std::fabs/-x order. Same functor as the f32 path.
    specvopcv3f16 = SIMD_VOPC_VOP3_F16.get(template_name)
    if specvopcv3f16 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOPC_VOP3_FP16({specvopcv3f16});"
    # VOP3 form of the f64 VOPC relational compares (17 ops). 64-bit-lane,
    # per-source abs/neg modifiers applied in the f64 domain outside the functor.
    specvopcv3f64 = SIMD_VOPC_VOP3_F64.get(template_name)
    if specvopcv3f64 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOPC64_VOP3_FP64({specvopcv3f64});"
    # VOP3 integer/bitwise ternary ops (add3/or3/xor3/lshl_add/add_lshl/bfi).
    # Plain element-wise functor of (src0, src1, src2); no modifiers.
    spec3tern = SIMD_VOP3_TERNARY_INT.get(template_name)
    if spec3tern is not None:
        cpp_t, cpp_op = spec3tern
        return f"  ROCJITSU_TRY_SIMD_VOP3_TERNARY_INT({cpp_t}, {cpp_op});"
    # VOP3 fp ternary (FMA / MAD family). Per-source abs/neg + omod/clamp in
    # the f32 / f16 / f64 domain. NaN-input lanes skipped by test (gcc-13
    # packed FMA quiets a different NaN operand vs scalar std::fma).
    spec3tf32 = SIMD_VOP3_TERNARY_FP32.get(template_name)
    if spec3tf32 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP32({spec3tf32});"
    spec3tf16 = SIMD_VOP3_TERNARY_FP16.get(template_name)
    if spec3tf16 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16({spec3tf16});"
    spec3tf64 = SIMD_VOP3_TERNARY_FP64.get(template_name)
    if spec3tf64 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP64({spec3tf64});"
    if template_name in SIMD_VOP3_DIV_FMAS_FP32:
        return "  ROCJITSU_TRY_SIMD_DIV_FMAS_VOP3_FP32();"
    if template_name in SIMD_VOP3_DIV_FMAS_FP64:
        return "  ROCJITSU_TRY_SIMD_DIV_FMAS_VOP3_FP64();"
    if template_name in SIMD_VOP3_MOV_B16:
        return "  ROCJITSU_TRY_SIMD_VOP3_MOV_B16();"
    # VOP3 sdst-enc carry forms.
    specv3carry = SIMD_VOP3_CARRY.get(template_name)
    if specv3carry is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3_CARRY({specv3carry});"
    # VOP3 sdst-enc carry forms with src2 carry-in (CDNA4 addc/subb/subbrev +
    # RDNA add/sub/subrev_co_ci). All six share the same shape; cin is loaded
    # from inst.src2.read_scalar64 inside the glue.
    specv3carry_src2 = SIMD_VOP3_CARRY_SRC2.get(template_name)
    if specv3carry_src2 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3_CARRY_SRC2({specv3carry_src2});"
    # VOP3P packed-16 integer binary family (pk_add/sub/mul_lo/min/max for
    # i16/u16 + pk_lshlrev/lshrrev/ashrrev for b16/i16). Gated on default
    # op_sel = 0 / op_sel_hi = 3 inside the glue.
    specpk = SIMD_VOP3P_PK_BINARY_INT.get(template_name)
    if specpk is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_INT({specpk});"
    specpkt = SIMD_VOP3P_PK_TERNARY_INT.get(template_name)
    if specpkt is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_INT({specpkt});"
    specpkf16 = SIMD_VOP3P_PK_BINARY_FP16.get(template_name)
    if specpkf16 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_FP16({specpkf16});"
    specpkf16t = SIMD_VOP3P_PK_TERNARY_FP16.get(template_name)
    if specpkf16t is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_FP16({specpkf16t});"
    if template_name in SIMD_VOP3P_MOV_B32:
        return "  ROCJITSU_TRY_SIMD_VOP3P_MOV_B32();"
    # VOP3P fma_mix / mad_mix (six ops, three destination shapes). Same body
    # for all; the routing picks the matching glue specialization.
    if template_name in SIMD_VOP3P_FMA_MIX_F32:
        return "  ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F32();"
    if template_name in SIMD_VOP3P_FMA_MIX_F16_LO:
        return "  ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F16_LO();"
    if template_name in SIMD_VOP3P_FMA_MIX_F16_HI:
        return "  ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F16_HI();"
    # VOP3 dst-accumulate FMA/MAC (vdst is the third operand). Per-isa class
    # has no src2; the accumulate glue reads vdst instead.
    specfmacf32 = SIMD_VOP3_FMAC_FP32.get(template_name)
    if specfmacf32 is not None:
        return f"  ROCJITSU_TRY_SIMD_FMAC_VOP3_FP32({specfmacf32});"
    specfmacf16 = SIMD_VOP3_FMAC_FP16.get(template_name)
    if specfmacf16 is not None:
        return f"  ROCJITSU_TRY_SIMD_FMAC_VOP3_FP16({specfmacf16});"
    specfmacf64 = SIMD_VOP3_FMAC_FP64.get(template_name)
    if specfmacf64 is not None:
        return f"  ROCJITSU_TRY_SIMD_FMAC_VOP3_FP64({specfmacf64});"
    # VOP3 ldexp: mixed-width fp src0 + int32 src1 exp.
    specldexpf32 = SIMD_VOP3_LDEXP_FP32.get(template_name)
    if specldexpf32 is not None:
        return f"  ROCJITSU_TRY_SIMD_LDEXP_VOP3_FP32({specldexpf32});"
    specldexpf64 = SIMD_VOP3_LDEXP_FP64.get(template_name)
    if specldexpf64 is not None:
        return f"  ROCJITSU_TRY_SIMD_LDEXP_VOP3_FP64({specldexpf64});"
    # VOP3 unary integer ops with no VOP1 twin (e.g. v_not_b16). Reuse the
    # VOP1 unary glue — operand shape (src0, vdst, 32-bit lanes) matches.
    spec3unai = SIMD_VOP3_UNARY_INT_EXTRA.get(template_name)
    if spec3unai is not None:
        cpp_tin, cpp_tout, cpp_op = spec3unai
        return f"  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {cpp_op});"
    # Extra plain integer binary VOP3 ops without a VOP2 twin (add_i32/i16,
    # sub_*, nc_* variants). Routed through the int VOP3 binary glue.
    spec3binx = SIMD_VOP3_BINARY_INT_EXTRA.get(template_name)
    if spec3binx is not None:
        cpp_t, cpp_op = spec3binx
        return f"  ROCJITSU_TRY_SIMD_VOP3_BINARY_INT({cpp_t}, {cpp_op});"
    # VOP3 f64 binary (add/mul/max/min). Per-source abs/neg + result omod/clamp
    # in the f64 domain.
    spec3binf64 = SIMD_VOP3_BINARY_FP64.get(template_name)
    if spec3binf64 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3_BINARY_FP64({spec3binf64});"
    # VOP3 f64 unary (ceil/floor/trunc/rndne/sqrt). Same modifier policy.
    spec3unaf64 = SIMD_VOP3_UNARY_FP64.get(template_name)
    if spec3unaf64 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP64({spec3unaf64});"
    # VOP3 f16 unary (widen-then-modify-then-narrow). FTZ-free ops only:
    # ceil/floor/trunc/rndne/sqrt. Transcendentals deferred.
    spec3unaf16 = SIMD_VOP3_UNARY_FP16.get(template_name)
    if spec3unaf16 is not None:
        return f"  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP16({spec3unaf16});"
    # VOP3-encoded twins of the SIMD VOP2 binary ops. Same operator/lane type;
    # the VOP3 form reads src0/src1 and carries abs/neg/omod/clamp modifiers.
    # f32 ops apply the modifiers in-vector (bit-exact); integer/bitwise ops
    # apply none (their scalar bodies ignore them), so the plain op suffices.
    if template_name.endswith("_vop3"):
        base = template_name[: -len("_vop3")]
        spec2v3 = SIMD_VOP2_BINARY.get(base + "_vop2")
        if spec2v3 is not None:
            cpp_t, cpp_op = spec2v3
            if cpp_t == "float32_t":
                return f"  ROCJITSU_TRY_SIMD_VOP3_BINARY_FP({cpp_t}, {cpp_op});"
            return f"  ROCJITSU_TRY_SIMD_VOP3_BINARY_INT({cpp_t}, {cpp_op});"
        # VOP3-encoded twins of the SIMD VOP1 unary ops. The plain int/cvt forms
        # apply no modifiers and read the same src0/vdst operands as VOP1, so they
        # reuse the VOP1 unary path verbatim. The f32 forms (and the float-domain
        # v_mov_b32) carry abs/neg/omod/clamp and route through the f32 unary glue.
        # The f16 forms also carry modifiers, but applying them around the f16<->f32
        # round trip is not yet handled, so they are left scalar (_VOP3_UNARY_SKIP).
        spec1v3 = SIMD_VOP1_UNARY.get(base + "_vop1")
        if spec1v3 is not None:
            cpp_tin, cpp_tout, cpp_op = spec1v3
            if base in _VOP3_UNARY_SKIP:
                return None
            if base in _VOP3_UNARY_FP_F32:
                return f"  ROCJITSU_TRY_SIMD_VOP3_UNARY_FP(float32_t, float32_t, {cpp_op});"
            return f"  ROCJITSU_TRY_SIMD_VOP1_UNARY({cpp_tin}, {cpp_tout}, {cpp_op});"
        # VOP3 twins of the mixed-width f64<->b32 cvt ops. Their generated VOP3
        # bodies drop the abs/neg/omod/clamp modifier reads (verified per-op),
        # so routing through the existing cvt glue is bit-exact. (A symmetric
        # SIMD_VOP1_UNARY_F64 fallback was considered but explicitly NOT added:
        # the f64-unary VOP3 forms apply modifiers via apply_vop3_*_mod_f64 —
        # routed through SIMD_VOP3_UNARY_FP64 above instead — and the rcp/rsq
        # forms use transcendental::*_f64 with NaN/±0/±Inf carve-outs not
        # present in the plain VOP1 functors.)
        speccvtoutv3 = SIMD_CVT_F64_TO_B32.get(base + "_vop1")
        if speccvtoutv3 is not None:
            out_t, cpp_op = speccvtoutv3
            return f"  ROCJITSU_TRY_SIMD_CVT_F64_TO_B32({out_t}, {cpp_op});"
        speccvtinv3 = SIMD_CVT_B32_TO_F64.get(base + "_vop1")
        if speccvtinv3 is not None:
            in_t, cpp_op = speccvtinv3
            return f"  ROCJITSU_TRY_SIMD_CVT_B32_TO_F64({in_t}, {cpp_op});"
    return None


def simd_probe_arch_portable(template_name: str) -> bool:
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
    """
    if simd_probe_line(template_name) is None:
        return False
    spect = SIMD_VOP2_TERNARY.get(template_name)
    if spect is not None and spect[1] != "0u":
        return False
    return True


def simd_extra_includes() -> list[str]:
    """Extra `#include` lines required by the SIMD probe call sites.

    The helper templates live in ``simd_glue.h``, which pulls in
    ``util/simd.h`` transitively (for ``util::has_stdx_simd``), so this is
    the only SIMD-specific include the generated shared header needs.
    """
    return ['#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"']
