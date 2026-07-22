// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decoder_factory.cpp
/// @brief Full decoder registry for public and simulator entry points.

#include "rocjitsu/isa/decoder.h"

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/instruction.h"

#include <memory>

namespace rocjitsu {

std::unique_ptr<Decoder> Decoder::create(rj_code_arch_t arch) {
  /*
   * \NPI new ISA family: #include its arch/amdgpu/<isa>/isa.h above and add a \
   * case returning std::make_unique<IsaDecoder<<isa>::Isa>>() here.
   */
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return std::make_unique<IsaDecoder<cdna1::Isa>>();
  case ROCJITSU_CODE_ARCH_CDNA2:
    return std::make_unique<IsaDecoder<cdna2::Isa>>();
  case ROCJITSU_CODE_ARCH_CDNA3:
    return std::make_unique<IsaDecoder<cdna3::Isa>>();
  case ROCJITSU_CODE_ARCH_CDNA4:
    return std::make_unique<IsaDecoder<cdna4::Isa>>();
  case ROCJITSU_CODE_ARCH_RDNA1:
    return std::make_unique<IsaDecoder<rdna1::Isa>>();
  case ROCJITSU_CODE_ARCH_RDNA2:
    return std::make_unique<IsaDecoder<rdna2::Isa>>();
  case ROCJITSU_CODE_ARCH_RDNA3:
    return std::make_unique<IsaDecoder<rdna3::Isa>>();
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return std::make_unique<IsaDecoder<rdna3_5::Isa>>();
  case ROCJITSU_CODE_ARCH_RDNA4:
    return std::make_unique<IsaDecoder<rdna4::Isa>>();
  case ROCJITSU_CODE_ARCH_GFX1250:
    return std::make_unique<IsaDecoder<gfx1250::Isa>>();
  default:
    return nullptr;
  }
}

} // namespace rocjitsu
