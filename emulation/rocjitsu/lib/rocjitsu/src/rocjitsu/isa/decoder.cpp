// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/decoder.h"

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/instruction.h"
#include <memory>

namespace rocjitsu {

Decoder::~Decoder() {
  // If this decoder's pool is still the active one, deactivate it so
  // surviving instructions (held by callers in unique_ptr/vectors) fall
  // back to ::operator delete instead of following a dangling pool pointer.
  if (Instruction::alloc_pool_ == &pool_)
    deactivate_pool();
}

std::unique_ptr<Decoder> Decoder::create(rj_code_arch_t arch) {
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
