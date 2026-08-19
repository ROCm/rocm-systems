// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/decoder.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/except.h"

#include <algorithm>
#include <vector>

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

DecodeResult Decoder::decode_window(std::span<const rj_code_binary_inst_t> words, uint64_t src_loc,
                                    const DecodeErrorEmitter &emit_error) {
  if (words.empty())
    throw util::InvalidInst("empty decode window");

  const std::size_t maximum_words = max_instruction_words();
  if (maximum_words == 0)
    throw util::InvalidInst("decoder reported a zero-width decode window");
  std::vector<rj_code_binary_inst_t> window(maximum_words, 0);
  const bool needs_padding = words.size() < window.size();
  const rj_code_binary_inst_t *decode_words = words.data();
  if (needs_padding) {
    std::copy(words.begin(), words.end(), window.begin());
    decode_words = window.data();
  }

  DecodeResult decode_result = decode(decode_words, src_loc, emit_error);
  if (decode_result.failed())
    return Result::failure();
  std::unique_ptr<Instruction> decoded_owner = std::move(decode_result).value();
  Instruction *decoded = decoded_owner.get();

  const int decoded_size = decoded->size();
  if (decoded_size <= 0 || decoded_size % static_cast<int>(sizeof(window.front())) != 0 ||
      static_cast<std::size_t>(decoded_size) > window.size() * sizeof(window.front())) {
    throw util::InvalidInst("decoder exceeded the maximum instruction size");
  }
  if (static_cast<std::size_t>(decoded_size) > words.size_bytes()) {
    throw util::InvalidInst("truncated instruction encoding");
  }

  // Most generated instructions own a copy of their encoding. A few combined
  // encodings, as well as external Decoder implementations, retain the input
  // pointer instead. Keep that established lifetime contract when a padded
  // tail window was required.
  if (needs_padding && decoded->raw_encoding_ == window.data())
    decoded->raw_encoding_ = words.data();
  return decoded_owner;
}

Instruction *Decoder::decode_window(std::span<const rj_code_binary_inst_t> words,
                                    uint64_t src_loc) {
  DecodeResult result = decode_window(words, src_loc, DecodeErrorEmitter{});
  return result.failed() ? nullptr : std::move(result).value().release();
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
