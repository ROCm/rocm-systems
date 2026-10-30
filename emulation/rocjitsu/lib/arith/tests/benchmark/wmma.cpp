// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense and sparse matrix workload registration across CPU and semantics modes.

#include "benchmark/workloads.h"

namespace rocjitsu::arith::bench {

// Append WMMA workloads and their supported CPU/semantics variants.
void add_wmma(std::vector<Benchmark> &cases) {
  add(cases, "wmma", "v_swmmac_i32_16x16x32_iu8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_i32_16x16x32_iu8_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_i32_16x16x32_iu4_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_i32_16x16x32_iu4_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_i32_16x16x64_iu4_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_i32_16x16x64_iu4_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_i32_16x16x32_iu8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_i32_16x16x32_iu8_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_i32_16x16x32_iu4_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_i32_16x16x32_iu4_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_i32_16x16x64_iu4_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_i32_16x16x64_iu4_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_fp8_fp8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_fp8_fp8_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_fp8_bf8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_fp8_bf8_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_bf8_fp8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_bf8_fp8_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_bf8_bf8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_bf8_bf8_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_fp8_fp8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_fp8_fp8_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_fp8_bf8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_fp8_bf8_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_bf8_fp8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_bf8_fp8_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_bf8_bf8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_bf8_bf8_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_f16_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_bf16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_bf16_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f16_16x16x32_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f16_16x16x32_f16_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_bf16_16x16x32_bf16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_bf16_16x16x32_bf16_w32(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_f16_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f32_16x16x32_bf16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f32_16x16x32_bf16_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_f16_16x16x32_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_f16_16x16x32_f16_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_swmmac_bf16_16x16x32_bf16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_swmmac_bf16_16x16x32_bf16_w64(flags, 0, r.d, r.a, r.b, r.index);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_f16_w32(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_f16_w64(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_bf16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_bf16_w32(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_bf16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_bf16_w64(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f16_16x16x16_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f16_16x16x16_f16_w32(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f16_16x16x16_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f16_16x16x16_f16_w64(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_bf16_16x16x16_bf16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_bf16_16x16x16_bf16_w32(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_bf16_16x16x16_bf16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_bf16_16x16x16_bf16_w64(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_i32_16x16x16_iu8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_i32_16x16x16_iu8_w32(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_i32_16x16x16_iu8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_i32_16x16x16_iu8_w64(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_i32_16x16x16_iu4_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_i32_16x16x16_iu4_w32(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_i32_16x16x16_iu4_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_i32_16x16x16_iu4_w64(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_f16_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_bf16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_bf16_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_f16_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_bf16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_bf16_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f16_16x16x16_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f16_16x16x16_f16_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_bf16_16x16x16_bf16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_bf16_16x16x16_bf16_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f16_16x16x16_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f16_16x16x16_f16_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_bf16_16x16x16_bf16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_bf16_16x16x16_bf16_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_fp8_bf8_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_bf8_fp8_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_fp8_fp8_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_fp8_bf8_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_bf8_fp8_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_f32_16x16x16_bf8_bf8_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_i32_16x16x16_iu8_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_i32_16x16x16_iu8_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_i32_16x16x16_iu4_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_i32_16x16x16_iu4_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_i32_16x16x32_iu4_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_i32_16x16x32_iu4_w32_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_i32_16x16x16_iu8_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_i32_16x16x16_iu8_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_i32_16x16x16_iu4_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_i32_16x16x16_iu4_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "wmma", "v_wmma_i32_16x16x32_iu4_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_wmma_i32_16x16x32_iu4_w64_gfx12(flags, 0, r.d, r.a, r.b, r.c);
  });
}

} // namespace rocjitsu::arith::bench
