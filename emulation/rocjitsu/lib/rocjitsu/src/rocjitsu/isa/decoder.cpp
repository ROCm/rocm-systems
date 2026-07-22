// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/decoder.h"

#include "rocjitsu/isa/instruction.h"

namespace rocjitsu {

Decoder::~Decoder() {
  // If this decoder's pool is still the active one, deactivate it so
  // surviving instructions (held by callers in unique_ptr/vectors) fall
  // back to ::operator delete instead of following a dangling pool pointer.
  if (Instruction::alloc_pool_ == &pool_)
    deactivate_pool();
}

Instruction *Decoder::decode(const rj_code_binary_inst_t *inst, uint64_t src_loc) {
  Instruction *decoded = decode(inst);
  if (decoded != nullptr)
    decoded->src_loc_ = src_loc;
  return decoded;
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
