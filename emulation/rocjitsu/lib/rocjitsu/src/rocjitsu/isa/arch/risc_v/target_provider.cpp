// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/target_provider.h"

#include "rocjitsu/isa/arch/risc_v/isa.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::risc_v {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder() { return make_isa_decoder<Isa>(); }

} // namespace rocjitsu::risc_v
