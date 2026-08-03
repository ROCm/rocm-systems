// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna1/target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::rdna1 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder() { return make_isa_decoder<Isa>(); }

} // namespace rocjitsu::rdna1
