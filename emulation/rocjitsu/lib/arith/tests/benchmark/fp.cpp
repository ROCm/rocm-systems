// SPDX-License-Identifier: MIT

// Elementwise floating-point workload registration across CPU and semantics modes.

#include "benchmark/workloads.h"

namespace rocjitsu::arith::bench {

// Append FP workloads and their supported CPU/semantics variants.
void add_fp(std::vector<Benchmark> &cases) {
  add(cases, "fp", "v_div_fmas_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fmas_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, 0, &r.status);
  });
  add(cases, "fp", "v_div_fmas_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fmas_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, 0, &r.status);
  });
  add(cases, "fp", "v_div_fmas_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fmas_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, 0, &r.status);
  });
  add(cases, "fp", "v_div_fmas_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fmas_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, 0, &r.status);
  });
  add(cases, "fp", "v_div_fmas_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fmas_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, 0,
                                             &r.status);
  });
  add(cases, "fp", "v_div_fmas_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fmas_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, 0,
                                             &r.status);
  });
  add(cases, "fp", "v_div_fmas_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fmas_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, 0,
                                             &r.status);
  });
  add(cases, "fp", "v_div_fmas_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fmas_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, 0,
                                             &r.status);
  });
  add(cases, "fp", "v_div_scale_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_scale_f32_w32(flags, UINT32_MAX, 0, r.d, &r.condition32, r.a, r.b, r.c);
  });
  add(cases, "fp", "v_div_scale_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_scale_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, &r.condition32, r.a, r.b,
                                              r.c);
  });
  add(cases, "fp", "v_div_scale_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_scale_f64_w32(flags, UINT32_MAX, 0, r.d, &r.condition32, r.a, r.b, r.c);
  });
  add(cases, "fp", "v_div_scale_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_scale_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, &r.condition32, r.a, r.b,
                                              r.c);
  });
  add(cases, "fp", "v_div_scale_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_scale_f32_w64(flags, UINT64_MAX, 0, r.d, &r.condition64, r.a, r.b, r.c);
  });
  add(cases, "fp", "v_div_scale_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_scale_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, &r.condition64, r.a, r.b,
                                              r.c);
  });
  add(cases, "fp", "v_div_scale_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_scale_f64_w64(flags, UINT64_MAX, 0, r.d, &r.condition64, r.a, r.b, r.c);
  });
  add(cases, "fp", "v_div_scale_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_scale_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, &r.condition64, r.a, r.b,
                                              r.c);
  });
  add(cases, "fp", "v_div_fixup_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_div_fixup_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_div_fixup_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_dx9_zero_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_dx9_zero_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "fp", "v_fma_dx9_zero_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_dx9_zero_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "fp", "v_fmac_dx9_zero_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmac_dx9_zero_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b);
  });
  add(cases, "fp", "v_fmac_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmac_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fmac_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmac_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fmamk_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmamk_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.literal, r.b, &r.status);
  });
  add(cases, "fp", "v_fmamk_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmamk_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.literal, r.b,
                                          &r.status);
  });
  add(cases, "fp", "v_fmaak_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmaak_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.literal, &r.status);
  });
  add(cases, "fp", "v_fmaak_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmaak_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.literal,
                                          &r.status);
  });
  add(cases, "fp", "v_fmac_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmac_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fmac_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmac_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fmac_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmac_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fmac_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmac_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fmamk_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmamk_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, uint16_t(r.literal), r.b,
                                    &r.status);
  });
  add(cases, "fp", "v_fmamk_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmamk_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, uint16_t(r.literal), r.b,
                                          &r.status);
  });
  add(cases, "fp", "v_fmaak_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmaak_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, uint16_t(r.literal),
                                    &r.status);
  });
  add(cases, "fp", "v_fmaak_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmaak_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, uint16_t(r.literal),
                                          &r.status);
  });
  add(cases, "fp", "v_fma_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fmac_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmac_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fmac_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmac_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fmamk_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmamk_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, uint16_t(r.literal), r.b,
                                    &r.status);
  });
  add(cases, "fp", "v_fmamk_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmamk_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, uint16_t(r.literal), r.b,
                                          &r.status);
  });
  add(cases, "fp", "v_fmaak_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmaak_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, uint16_t(r.literal),
                                    &r.status);
  });
  add(cases, "fp", "v_fmaak_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmaak_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, uint16_t(r.literal),
                                          &r.status);
  });
  add(cases, "fp", "v_fmamk_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmamk_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.literal, r.b, &r.status);
  });
  add(cases, "fp", "v_fmamk_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmamk_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.literal, r.b,
                                          &r.status);
  });
  add(cases, "fp", "v_fmaak_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmaak_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.literal, &r.status);
  });
  add(cases, "fp", "v_fmaak_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmaak_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.literal,
                                          &r.status);
  });
  add(cases, "fp", "v_fma_dx9_zero_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_dx9_zero_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "fp", "v_fma_dx9_zero_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_dx9_zero_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "fp", "v_fmac_dx9_zero_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fmac_dx9_zero_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b);
  });
  add(cases, "fp", "v_pk_fma_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_fma_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_pk_fmac_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_fmac_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_fma_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_fma_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_pk_fmac_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_fmac_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_fma_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_fma_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_pk_fmac_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_fmac_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_fma_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_fma_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_pk_fmac_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_fmac_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_add_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_add_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_add_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_add_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_mul_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_mul_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_mul_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_mul_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_min_num_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_min_num_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_max_num_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_max_num_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_minimum_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_minimum_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_maximum_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_maximum_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_add_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_add_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_sub_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sub_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_sub_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sub_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_subrev_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_subrev_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_subrev_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_subrev_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min_num_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_num_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_num_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_num_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_minimum_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimum_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_maximum_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximum_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_add_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_add_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_sub_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sub_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_sub_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sub_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_subrev_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_subrev_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_subrev_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_subrev_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min_num_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_num_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_num_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_num_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_minimum_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimum_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_maximum_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximum_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_dx9_zero_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_dx9_zero_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_dx9_zero_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_dx9_zero_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_add_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_add_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fma_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_trunc_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_trunc_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_ceil_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_ceil_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rndne_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rndne_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_floor_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_floor_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_fract_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_fract_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_sqrt_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_sqrt_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rcp_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rcp_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rsq_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rsq_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_min_num_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_num_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_num_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_num_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_minimum_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimum_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_maximum_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximum_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_frexp_mant_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_mant_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_min_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min3_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min3_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_max3_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max3_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minmax_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minmax_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maxmin_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maxmin_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_med3_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_med3_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_min3_num_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min3_num_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_max3_num_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max3_num_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minmax_num_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minmax_num_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maxmin_num_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maxmin_num_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minimum3_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimum3_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maximum3_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximum3_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minimummaximum_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimummaximum_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maximumminimum_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximumminimum_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_med3_num_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_med3_num_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_min3_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min3_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_max3_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max3_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minmax_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minmax_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maxmin_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maxmin_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_med3_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_med3_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_min3_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min3_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_max3_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max3_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minmax_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minmax_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maxmin_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maxmin_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_med3_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_med3_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_min3_num_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min3_num_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_max3_num_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max3_num_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minmax_num_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minmax_num_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maxmin_num_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maxmin_num_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minimum3_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimum3_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maximum3_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximum3_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minimummaximum_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimummaximum_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maximumminimum_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximumminimum_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_med3_num_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_med3_num_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_min3_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min3_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_max3_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max3_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minmax_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minmax_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maxmin_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maxmin_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_med3_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_med3_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_pk_min_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_min_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_max_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_max_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_add_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_add_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_sub_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sub_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_sub_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sub_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_subrev_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_subrev_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_subrev_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_subrev_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min_num_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_num_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_num_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_num_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_minimum_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimum_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_maximum_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximum_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_add_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_add_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_add_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_add_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_mul_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_mul_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_mul_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_mul_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_min_num_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_min_num_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_max_num_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_max_num_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_minimum_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_minimum_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_maximum_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_maximum_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_min_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_min_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_pk_max_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_pk_max_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min3_num_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min3_num_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_max3_num_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max3_num_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minmax_num_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minmax_num_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maxmin_num_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maxmin_num_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minimum3_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimum3_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maximum3_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximum3_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minimummaximum_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimummaximum_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maximumminimum_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximumminimum_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_mul_dx9_zero_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_dx9_zero_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_dx9_zero_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_dx9_zero_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_med3_num_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_med3_num_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_add_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_add_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fma_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_min_num_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_num_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_num_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_num_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_minimum_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimum_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_maximum_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximum_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_trunc_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_trunc_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_ceil_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_ceil_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rndne_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rndne_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_floor_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_floor_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_fract_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_fract_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_sqrt_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_sqrt_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rcp_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rcp_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rsq_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_rsq_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_mant_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_mant_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i32_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i32_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i32_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i32_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i32_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i32_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i32_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i32_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i32_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i32_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i32_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i32_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i32_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i32_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i32_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i32_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_ldexp_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_ldexp_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_ldexp_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_ldexp_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_ldexp_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_ldexp_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_ldexp_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_ldexp_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_fma_mix_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mix_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mix_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mix_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mixlo_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mixlo_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mixlo_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mixlo_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mixhi_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mixhi_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mixhi_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mixhi_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mix_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mix_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mix_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mix_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mixlo_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mixlo_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mixlo_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mixlo_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mixhi_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mixhi_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_fma_mixhi_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fma_mixhi_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_add_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_add_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_add_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_sub_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sub_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_sub_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sub_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_subrev_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_subrev_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_subrev_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_subrev_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_mul_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_mul_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min_num_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min_num_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_max_num_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max_num_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_minimum_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimum_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_maximum_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximum_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_min3_num_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_min3_num_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_max3_num_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_max3_num_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minmax_num_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minmax_num_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maxmin_num_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maxmin_num_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minimum3_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimum3_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maximum3_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximum3_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_minimummaximum_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_minimummaximum_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_maximumminimum_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_maximumminimum_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_med3_num_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_med3_num_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "fp", "v_ldexp_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_ldexp_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i16_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i16_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i16_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i16_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_ldexp_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_ldexp_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ldexp_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i16_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i16_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "fp", "v_frexp_exp_i16_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_exp_i16_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
}

} // namespace rocjitsu::arith::bench
