// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/decoder.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu {

namespace {

thread_local std::unique_ptr<Decoder::Pool> worker_decoder_pool;

} // namespace

std::unique_ptr<Decoder> Decoder::create(const IsaTargetRegistry &registry,
                                         std::string_view target_id) {
  const IsaTargetDescriptor *target = registry.find(target_id);
  return target == nullptr ? nullptr : target->decoder_factory();
}

std::unique_ptr<Decoder> Decoder::create(const IsaTargetRegistry &registry, rj_code_arch_t arch) {
  const IsaTargetDescriptor *target = registry.find(arch);
  return target == nullptr ? nullptr : target->decoder_factory();
}

void Decoder::enable_thread_pool() {
  if (!worker_decoder_pool)
    worker_decoder_pool = std::make_unique<Pool>();
  if (Instruction::alloc_pool_ == worker_decoder_pool.get())
    return;
  activate_pool([](void *p, size_t s) -> void * { return static_cast<Pool *>(p)->allocate(s); },
                [](void *p, void *ptr) { static_cast<Pool *>(p)->deallocate(ptr); },
                worker_decoder_pool.get());
}

void Decoder::disable_thread_pool() {
  if (!worker_decoder_pool)
    return;
  Instruction::invalidate_allocator_pool(worker_decoder_pool.get());
  worker_decoder_pool.reset();
}

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
