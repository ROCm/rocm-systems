// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unary math workload registration across CPU and semantics modes.

#include "benchmark/workloads.h"

namespace rocjitsu::arith::bench {

// Append MATH workloads and their supported CPU/semantics variants.
void add_math(std::vector<Benchmark> &cases) {
  add(cases, "math", "v_rcp_iflag_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_iflag_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_iflag_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_iflag_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_iflag_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_iflag_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_iflag_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_iflag_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_trig_preop_f64_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trig_preop_f64_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "math", "v_trig_preop_f64_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trig_preop_f64_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "math", "v_trig_preop_f64_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trig_preop_f64_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "math", "v_trig_preop_f64_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trig_preop_f64_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "math", "v_sin_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sin_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sin_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sin_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_cos_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_cos_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_cos_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_cos_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sin_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sin_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sin_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sin_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_cos_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_cos_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_cos_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_cos_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sin_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sin_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sin_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sin_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_cos_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_cos_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_cos_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_cos_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sin_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sin_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sin_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sin_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_cos_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_cos_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_cos_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_cos_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_fract_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_fract_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_frexp_mant_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_frexp_mant_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_trunc_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_trunc_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_ceil_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_ceil_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rndne_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rndne_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_floor_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_floor_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sqrt_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sqrt_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rsq_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rsq_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_exp_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_exp_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_exp_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_exp_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_log_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_log_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_log_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_log_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_fract_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_fract_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_frexp_mant_f32_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f32_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_frexp_mant_f32_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f32_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_trunc_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_trunc_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_ceil_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_ceil_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rndne_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rndne_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_floor_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_floor_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sqrt_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sqrt_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rsq_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rsq_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_exp_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_exp_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_exp_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_exp_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_log_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_log_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_log_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_log_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_fract_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_fract_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_frexp_mant_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_frexp_mant_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_trunc_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_trunc_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_ceil_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_ceil_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rndne_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rndne_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_floor_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_floor_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sqrt_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sqrt_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rsq_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rsq_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_exp_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_exp_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_exp_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_exp_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_log_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_log_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_log_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_log_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_fract_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_fract_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_fract_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_frexp_mant_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_frexp_mant_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_frexp_mant_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_trunc_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_trunc_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_trunc_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_ceil_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_ceil_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_ceil_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rndne_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rndne_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rndne_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_floor_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_floor_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_floor_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sqrt_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_sqrt_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_sqrt_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rcp_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rcp_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rsq_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_rsq_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_rsq_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_exp_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_exp_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_exp_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_exp_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_log_f32_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_log_f32_w32(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
  add(cases, "math", "v_log_f32_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_log_f32_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, &r.status);
  });
}

} // namespace rocjitsu::arith::bench
