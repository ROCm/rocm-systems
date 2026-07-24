// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file WMMA execution helpers for gfx1250.

#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"

#include <cstdint>
#include <utility>

namespace rocjitsu::gfx1250 {

template <typename F>
inline uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc,
                            F &&get_const) {
  return amdgpu::resolve_acc<amdgpu::AccMode::Unified>(vb, dst, src2_ev, const_acc,
                                                       std::forward<F>(get_const));
}

} // namespace rocjitsu::gfx1250
