// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcheck_decoder.cpp
/// @brief Decode-only factory used by the unload-safe waitcheck shared library.

#include "rocjitsu/isa/decoder.h"

#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/instruction.h"

#include <memory>

namespace rocjitsu {

Decoder::~Decoder() {
  if (Instruction::alloc_pool_ == &pool_)
    deactivate_pool();
}

Instruction *Decoder::decode(const rj_code_binary_inst_t *inst, uint64_t src_loc) {
  Instruction *decoded = decode(inst);
  if (decoded != nullptr)
    decoded->src_loc_ = src_loc;
  return decoded;
}

std::unique_ptr<Decoder> Decoder::create(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return std::make_unique<IsaDecoder<cdna3::Isa>>();
  case ROCJITSU_CODE_ARCH_CDNA4:
    return std::make_unique<IsaDecoder<cdna4::Isa>>();
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

void Decoder::activate_pool(AllocFn alloc, DeallocFn dealloc, void *pool) {
  Instruction::alloc_fn_ = alloc;
  Instruction::dealloc_fn_ = dealloc;
  Instruction::alloc_pool_ = pool;
}

void Decoder::deactivate_pool() {
  Instruction::alloc_fn_ = nullptr;
  Instruction::dealloc_fn_ = nullptr;
  Instruction::alloc_pool_ = nullptr;
}

} // namespace rocjitsu
