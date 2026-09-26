// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/decoder.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu {

std::unique_ptr<Decoder> Decoder::create(const IsaTargetRegistry &registry,
                                         std::string_view target_id) {
  const IsaTargetDescriptor *target = registry.find(target_id);
  return target == nullptr ? nullptr : target->decoder_factory();
}

std::unique_ptr<Decoder> Decoder::create(const IsaTargetRegistry &registry, rj_code_arch_t arch) {
  const IsaTargetDescriptor *target = registry.find(arch);
  return target == nullptr ? nullptr : target->decoder_factory();
}

Decoder::~Decoder() {
  // Clear direct and temporarily suppressed references to this pool so no
  // later allocation scope can restore a pointer into a destroyed decoder.
  Instruction::invalidate_allocator_pool(&pool_);
}

void Decoder::disable_pool() { Instruction::invalidate_allocator_pool(&pool_); }

DecodeResult Decoder::decode(const rj_code_binary_inst_t *inst, uint64_t src_loc,
                             const DecodeErrorEmitter &emit_error) {
  DecodeResult decoded = decode(inst, emit_error);
  if (decoded.succeeded())
    decoded.value()->src_loc_ = src_loc;
  return decoded;
}

void Decoder::activate_pool(AllocFn alloc, DeallocFn dealloc, void *pool) {
  Instruction::alloc_fn_ = alloc;
  Instruction::dealloc_fn_ = dealloc;
  Instruction::alloc_pool_ = pool;
}

Result Decoder::validate_instruction_operands(const Instruction &inst,
                                              const DecodeErrorEmitter &emit_error) {
  for (int index = 0; index < inst.num_src_operands(); ++index) {
    if (const Operand *operand = inst.src_operand(index))
      if (operand->validate_encoding(emit_error).failed()) [[unlikely]]
        return Result::failure();
  }
  for (int index = 0; index < inst.num_dst_operands(); ++index) {
    if (const Operand *operand = inst.dst_operand(index))
      if (operand->validate_encoding(emit_error).failed()) [[unlikely]]
        return Result::failure();
  }
  return Result::success();
}

} // namespace rocjitsu
