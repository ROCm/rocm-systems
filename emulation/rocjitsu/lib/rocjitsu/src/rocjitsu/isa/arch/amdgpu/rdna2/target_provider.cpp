// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna2/target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/execution_backend.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::rdna2 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder() {
  return make_isa_decoder<Isa>(&execution_backend());
}

} // namespace rocjitsu::rdna2
