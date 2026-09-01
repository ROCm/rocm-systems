// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/decoder.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu {

std::unique_ptr<Decoder> Decoder::create(const IsaTargetRegistry &registry,
                                         std::string_view target_id) {
  const IsaTargetDescriptor *target = registry.find(target_id);
  if (target == nullptr)
    return nullptr;
  const IsaGpuTargetDescription *binding = registry.find_gpu_target_by_code_object_id(target_id);
  if (binding != nullptr && target->variant_decoder_factory != nullptr)
    return target->variant_decoder_factory(*binding);
  if (target->variant_decoder_factory != nullptr) {
    binding = registry.find_default_gpu_target(*target);
    return binding == nullptr ? nullptr : target->variant_decoder_factory(*binding);
  }
  return target->decoder_factory();
}

std::unique_ptr<Decoder> Decoder::create(const IsaTargetRegistry &registry,
                                         rj_code_target_id_t target_id) {
  const IsaTargetDescriptor *target = registry.find(target_id);
  const IsaGpuTargetDescription *binding = registry.find_gpu_target(target_id);
  if (target == nullptr || binding == nullptr)
    return nullptr;
  return target->variant_decoder_factory == nullptr ? target->decoder_factory()
                                                    : target->variant_decoder_factory(*binding);
}

std::unique_ptr<Decoder> Decoder::create(const IsaTargetRegistry &registry, rj_code_arch_t arch) {
  const IsaTargetDescriptor *target = registry.find(arch);
  if (target == nullptr)
    return nullptr;
  if (target->variant_decoder_factory == nullptr)
    return target->decoder_factory();
  const IsaGpuTargetDescription *binding = registry.find_default_gpu_target(*target);
  return binding == nullptr ? nullptr : target->variant_decoder_factory(*binding);
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
