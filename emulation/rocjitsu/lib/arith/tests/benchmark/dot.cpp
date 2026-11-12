// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// DOT workload registration across supported CPU and semantics modes.

#include "benchmark/workloads.h"
#include "rj_arith/detail/rj_arith_dot.h"

namespace rocjitsu::arith::test {

// Append DOT workloads and their supported CPU/semantics variants.
void add_dot(std::vector<Benchmark> &cases) {
  add(cases, "dot", "v_dot2_f16_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f16_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_f16_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f16_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_bf16_bf16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_bf16_bf16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_bf16_bf16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_bf16_bf16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_f16_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f16_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_f16_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f16_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_bf16_bf16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_bf16_bf16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_bf16_bf16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_bf16_bf16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_f32_f16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f32_f16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_f32_bf16_w32_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f32_bf16_w32_gfx12(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_f32_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f32_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_f32_bf16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f32_bf16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2acc_f32_f16_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2acc_f32_f16_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "dot", "v_dot2_f32_f16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f32_f16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_f32_bf16_w64_gfx12", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f32_bf16_w64_gfx12(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_f32_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f32_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2_f32_bf16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2_f32_bf16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c, &r.status);
  });
  add(cases, "dot", "v_dot2acc_f32_f16_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot2acc_f32_f16_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, &r.status);
  });
  add(cases, "dot", "v_dot4_f32_fp8_fp8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot4_f32_fp8_fp8_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "dot", "v_dot4_f32_fp8_bf8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot4_f32_fp8_bf8_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "dot", "v_dot4_f32_bf8_fp8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot4_f32_bf8_fp8_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "dot", "v_dot4_f32_bf8_bf8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot4_f32_bf8_bf8_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "dot", "v_dot4_f32_fp8_fp8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot4_f32_fp8_fp8_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "dot", "v_dot4_f32_fp8_bf8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot4_f32_fp8_bf8_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "dot", "v_dot4_f32_bf8_fp8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot4_f32_bf8_fp8_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add(cases, "dot", "v_dot4_f32_bf8_bf8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot4_f32_bf8_bf8_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add_flag_variant(
      cases, "dot", "v_dot4_i32_iu8_w32",
      [](Operands &r, uint64_t flags, uint64_t modifiers) {
        return rj_arith_v_dot4_i32_iu8_w32(flags, UINT32_MAX, modifiers, r.d, r.a, r.b, r.c);
      },
      RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B, "signed A/B",
      OperandLayout::SignedIntegerInputs);
  add(cases, "dot", "v_dot4_u32_u8_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot4_u32_u8_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add_flag_variant(
      cases, "dot", "v_dot8_i32_iu4_w32",
      [](Operands &r, uint64_t flags, uint64_t modifiers) {
        return rj_arith_v_dot8_i32_iu4_w32(flags, UINT32_MAX, modifiers, r.d, r.a, r.b, r.c);
      },
      RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B, "signed A/B",
      OperandLayout::SignedIntegerInputs);
  add(cases, "dot", "v_dot8_u32_u4_w32", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot8_u32_u4_w32(flags, UINT32_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add_flag_variant(
      cases, "dot", "v_dot4_i32_iu8_w64",
      [](Operands &r, uint64_t flags, uint64_t modifiers) {
        return rj_arith_v_dot4_i32_iu8_w64(flags, UINT64_MAX, modifiers, r.d, r.a, r.b, r.c);
      },
      RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B, "signed A/B",
      OperandLayout::SignedIntegerInputs);
  add(cases, "dot", "v_dot4_u32_u8_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot4_u32_u8_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c);
  });
  add_flag_variant(
      cases, "dot", "v_dot8_i32_iu4_w64",
      [](Operands &r, uint64_t flags, uint64_t modifiers) {
        return rj_arith_v_dot8_i32_iu4_w64(flags, UINT64_MAX, modifiers, r.d, r.a, r.b, r.c);
      },
      RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B, "signed A/B",
      OperandLayout::SignedIntegerInputs);
  add(cases, "dot", "v_dot8_u32_u4_w64", [](Operands &r, uint64_t flags) {
    return rj_arith_v_dot8_u32_u4_w64(flags, UINT64_MAX, 0, r.d, r.a, r.b, r.c);
  });
}

} // namespace rocjitsu::arith::test
