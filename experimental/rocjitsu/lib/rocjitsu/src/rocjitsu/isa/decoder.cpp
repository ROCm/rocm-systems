// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/decoder.h"

#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/instruction.h"
#include <memory>

namespace rocjitsu {

std::unique_ptr<Decoder> Decoder::create(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return std::make_unique<IsaDecoder<cdna3::Isa>>();
  case ROCJITSU_CODE_ARCH_CDNA4:
    return std::make_unique<IsaDecoder<cdna4::Isa>>();
  default:
    return nullptr;
  }
}

} // namespace rocjitsu
