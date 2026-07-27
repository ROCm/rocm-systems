// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/gfx1250/target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/gfx1250/execution_backend.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::gfx1250 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder() {
  return make_isa_decoder<Isa>(&execution_backend());
}

} // namespace rocjitsu::gfx1250
