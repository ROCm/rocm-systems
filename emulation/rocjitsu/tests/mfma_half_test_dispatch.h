// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Retained half-input MFMA observer and overlap fixtures call the public arithmetic API.
#pragma once
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_dense_matrix_exec.h"
#include "rocjitsu/isa/isa_family.h"
#include "util/simd_test_hooks.h"
#include <gtest/gtest.h>

namespace rocjitsu::test {
// Execute one real half-input MFMA shape without retaining a second arithmetic implementation.
template <bool Bf16, uint32_t M, uint32_t N, uint32_t K, uint32_t Batch = 1>
void execute_mfma_half_shape(amdgpu::ComputeUnitCore &cu, uint32_t d, uint32_t a, uint32_t b,
                             uint32_t c, uint32_t constant, uint32_t cbsz, uint32_t abid,
                             uint32_t blgp) {
  static_assert(M == N);
  auto *wave = cu.wf(0);
  ASSERT_NE(wave, nullptr);
  using Entry = decltype(&rj_arith_v_mfma_f32_16x16x16_f16_w64);
  Entry entry = nullptr;
  if constexpr (M == 32 && K == 4 && Batch == 2) {
    if constexpr (!Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna1) {
        entry = rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx908;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna4) {
        entry = rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx950;
      } else {
        entry = rj_arith_v_mfma_f32_32x32x4_2b_f16_w64;
      }
    } else if constexpr (Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna4) {
        entry = rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64_gfx950;
      } else {
        entry = rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64;
      }
    }
  } else if constexpr (M == 16 && K == 4 && Batch == 4) {
    if constexpr (!Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna1) {
        entry = rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx908;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna4) {
        entry = rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx950;
      } else {
        entry = rj_arith_v_mfma_f32_16x16x4_4b_f16_w64;
      }
    } else if constexpr (Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna4) {
        entry = rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx950;
      } else {
        entry = rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64;
      }
    }
  } else if constexpr (M == 4 && K == 4 && Batch == 16) {
    if constexpr (!Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna1) {
        entry = rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx908;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna4) {
        entry = rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx950;
      } else {
        entry = rj_arith_v_mfma_f32_4x4x4_16b_f16_w64;
      }
    } else if constexpr (Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna4) {
        entry = rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx950;
      } else {
        entry = rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64;
      }
    }
  } else if constexpr (M == 32 && K == 8 && Batch == 1) {
    if constexpr (!Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna1) {
        entry = rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx908;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna4) {
        entry = rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx950;
      } else {
        entry = rj_arith_v_mfma_f32_32x32x8_f16_w64;
      }
    } else if constexpr (Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_32x32x8_bf16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna4) {
        entry = rj_arith_v_mfma_f32_32x32x8_bf16_w64_gfx950;
      } else {
        entry = rj_arith_v_mfma_f32_32x32x8_bf16_w64;
      }
    }
  } else if constexpr (M == 16 && K == 16 && Batch == 1) {
    if constexpr (!Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna1) {
        entry = rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx908;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna4) {
        entry = rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx950;
      } else {
        entry = rj_arith_v_mfma_f32_16x16x16_f16_w64;
      }
    } else if constexpr (Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna4) {
        entry = rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx950;
      } else {
        entry = rj_arith_v_mfma_f32_16x16x16_bf16_w64;
      }
    }
  } else if constexpr (M == 16 && K == 32 && Batch == 1) {
    if constexpr (!Bf16) {
      {
        entry = rj_arith_v_mfma_f32_16x16x32_f16_w64;
      }
    } else if constexpr (Bf16) {
      { entry = rj_arith_v_mfma_f32_16x16x32_bf16_w64; }
    }
  } else if constexpr (M == 32 && K == 16 && Batch == 1) {
    if constexpr (!Bf16) {
      {
        entry = rj_arith_v_mfma_f32_32x32x16_f16_w64;
      }
    } else if constexpr (Bf16) {
      { entry = rj_arith_v_mfma_f32_32x32x16_bf16_w64; }
    }
  } else if constexpr (M == 32 && K == 2 && Batch == 2 && Bf16) {
    if constexpr (Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_32x32x2_2b_bf16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna1) {
        entry = rj_arith_v_mfma_f32_32x32x2_2b_bf16_w64_gfx908;
      }
    }
  } else if constexpr (M == 16 && K == 2 && Batch == 4 && Bf16) {
    if constexpr (Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_16x16x2_4b_bf16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna1) {
        entry = rj_arith_v_mfma_f32_16x16x2_4b_bf16_w64_gfx908;
      }
    }
  } else if constexpr (M == 4 && K == 2 && Batch == 16 && Bf16) {
    if constexpr (Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_4x4x2_16b_bf16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna1) {
        entry = rj_arith_v_mfma_f32_4x4x2_16b_bf16_w64_gfx908;
      }
    }
  } else if constexpr (M == 32 && K == 4 && Batch == 1 && Bf16) {
    if constexpr (Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_32x32x4_bf16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna1) {
        entry = rj_arith_v_mfma_f32_32x32x4_bf16_w64_gfx908;
      }
    }
  } else if constexpr (M == 16 && K == 8 && Batch == 1 && Bf16) {
    if constexpr (Bf16) {
      if (cu.arch() == rocjitsu::IsaFamily::Cdna2) {
        entry = rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx90a;
      } else if (cu.arch() == rocjitsu::IsaFamily::Cdna1) {
        entry = rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx908;
      }
    }
  }
  ASSERT_NE(entry, nullptr);
  uint64_t flags = cu.arithmetic_flags();
  if (util::force_scalar()) {
    flags &= ~RJ_ARITH_CPU_MASK;
  }
  const bool cdna3 = cu.arch() == rocjitsu::IsaFamily::Cdna3;
  const unsigned block = cu.arch() == rocjitsu::IsaFamily::Cdna2 ? M : 0;
  const uint64_t a_lanes =
      cdna3 && Batch == 1 ? ~uint64_t{0} : amdgpu::mfma_permuted_lane_mask([&](uint32_t lane) {
        return cdna3 ? rj_arith_gfx942_batched_mfma_a_lane(lane, cbsz, abid, M)
                     : amdgpu::permute_a_lane(lane, cbsz, abid, block);
      });
  const uint64_t b_lanes =
      cdna3 && Batch == 1 ? ~uint64_t{0} : amdgpu::mfma_permuted_lane_mask([&](uint32_t lane) {
        return amdgpu::permute_b_lane(lane, blgp);
      });
  amdgpu::arith::exec_dense_fp_matrix<M * K * Batch / 128, M * M * Batch / 64, 64>(
      *wave, entry, flags, cbsz | (uint64_t(abid) << 3) | (uint64_t(blgp) << 7), d, a, b, c,
      constant == amdgpu::ACC_FROM_VGPR ? std::nullopt : std::optional<uint64_t>(constant), a_lanes,
      b_lanes);
}
} // namespace rocjitsu::test
