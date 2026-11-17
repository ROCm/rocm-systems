// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Intrinsic-backed SDWA scaling of rounded FP16/FP32 results.
#pragma once

#include "common/isa.h"
#include "common/sdwa_output.h"
#include "conversion/x86_64/lanes.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Scale rounded results in full vectors and store only EXEC-selected lanes.
template <SdwaOutput Output>
inline void scale_sdwa_simd(uint64_t flags, uint64_t exec_mask, int lanes, uint32_t mode,
                            uint32_t *data) {
  using V = conversion_simd::Lanes<32>;
  const auto policy = sdwa_output_policy(flags, mode);
  for (int lane = 0; lane < lanes; lane += V::size()) {
    sdwa_scale_word<Output>(V(data + lane), mode, policy)
        .store(data + lane, uint32_t(exec_mask >> lane));
  }
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
