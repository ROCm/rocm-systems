// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/decoder.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/except.h"

#include <algorithm>
#include <array>

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

Instruction *Decoder::decode_window(std::span<const rj_code_binary_inst_t> words,
                                    uint64_t src_loc) {
  if (words.empty())
    throw util::InvalidInst("empty decode window");

  std::array<rj_code_binary_inst_t, kMaximumInstructionWords> window{};
  const bool needs_padding = words.size() < window.size();
  const rj_code_binary_inst_t *decode_words = words.data();
  if (needs_padding) {
    std::copy(words.begin(), words.end(), window.begin());
    decode_words = window.data();
  }

  Instruction *decoded = decode(decode_words, src_loc);
  if (decoded == nullptr)
    return nullptr;

  const int decoded_size = decoded->size();
  if (decoded_size <= 0 || decoded_size % static_cast<int>(sizeof(window.front())) != 0 ||
      static_cast<std::size_t>(decoded_size) > window.size() * sizeof(window.front())) {
    delete decoded;
    throw util::InvalidInst("decoder exceeded the maximum instruction size");
  }
  if (static_cast<std::size_t>(decoded_size) > words.size_bytes()) {
    delete decoded;
    throw util::InvalidInst("truncated instruction encoding");
  }

  // Most generated instructions own a copy of their encoding. A few combined
  // encodings, as well as external Decoder implementations, retain the input
  // pointer instead. Keep that established lifetime contract when a padded
  // tail window was required.
  if (needs_padding && decoded->raw_encoding_ == window.data())
    decoded->raw_encoding_ = words.data();
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
