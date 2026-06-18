// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/zifencei.h"

namespace rocjitsu {
namespace risc_v {
namespace detail {

FenceIInst::FenceIInst(uint32_t raw) : IType("fence.i", raw, make_exec_fn<FenceIInst>()) {}
void FenceIInst::execute_impl(HartState &ctx) {
  (void)ctx; // No-op for single-hart simulation with no I-cache.
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu
