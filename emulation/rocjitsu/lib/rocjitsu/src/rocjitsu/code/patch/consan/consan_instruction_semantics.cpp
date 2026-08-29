// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <cstring>
#include <string_view>
#include <utility>

namespace rocjitsu {

std::unique_ptr<Instruction> decode_bounded_instruction(Decoder &decoder,
                                                        std::span<const uint32_t> words,
                                                        uint64_t source_offset) {
  DecodeResult decoded = decoder.decode_window(words, source_offset, DecodeErrorEmitter{});
  if (decoded.failed())
    return {};
  return std::move(decoded).value();
}

int32_t sign_extend_24(uint32_t value) { return static_cast<int32_t>(value << 8) >> 8; }

bool is_barrier_instruction(const Instruction &instruction) {
  const std::string_view mnemonic = instruction.mnemonic();
  return instruction.is_barrier() || mnemonic.starts_with("s_barrier") ||
         mnemonic.find("barrier") != std::string_view::npos;
}

ConSanBarrierSite::Scope barrier_scope_for_id(int32_t barrier_id) {
  switch (barrier_id) {
  case -4:
  case -3:
    return ConSanBarrierSite::Scope::Cluster;
  case -2:
  case -1:
    return ConSanBarrierSite::Scope::Workgroup;
  default:
    return barrier_id >= 1 && barrier_id <= 16 ? ConSanBarrierSite::Scope::Workgroup
                                               : ConSanBarrierSite::Scope::Unknown;
  }
}

void decode_barrier_operand(const Instruction &instruction,
                            std::span<const uint8_t> instruction_bytes, ConSanBarrierSite &site) {
  if (instruction.mnemonic() == "s_barrier") {
    site.barrier_id = -1;
    site.operand_source = ConSanBarrierSite::OperandSource::Immediate;
    site.scope = ConSanBarrierSite::Scope::Workgroup;
    return;
  }
  if (instruction.mnemonic() == "s_barrier_wait") {
    if (instruction_bytes.size() < sizeof(uint32_t))
      return;
    uint32_t word = 0;
    std::memcpy(&word, instruction_bytes.data(), sizeof(word));
    site.raw_simm16 = static_cast<uint16_t>(word);
    site.barrier_id = static_cast<int16_t>(word & 0xffffu);
    site.operand_source = ConSanBarrierSite::OperandSource::Immediate;
    site.scope = barrier_scope_for_id(*site.barrier_id);
    return;
  }
  if (instruction.mnemonic() == "s_barrier_leave") {
    if (instruction_bytes.size() < sizeof(uint32_t))
      return;
    uint32_t word = 0;
    std::memcpy(&word, instruction_bytes.data(), sizeof(word));
    site.raw_simm16 = static_cast<uint16_t>(word);
    return;
  }
  const std::string_view mnemonic = instruction.mnemonic();
  if (mnemonic != "s_barrier_signal" && mnemonic != "s_barrier_signal_isfirst" &&
      mnemonic != "s_barrier_init" && mnemonic != "s_barrier_join" &&
      mnemonic != "s_wakeup_barrier" && mnemonic != "s_get_barrier_state")
    return;
  if (instruction_bytes.size() < sizeof(uint32_t))
    return;
  uint32_t word = 0;
  std::memcpy(&word, instruction_bytes.data(), sizeof(word));
  const uint32_t source = word & 0xffu;
  site.raw_operand_selector = source;
  if (source == 125u) {
    site.operand_source = ConSanBarrierSite::OperandSource::DynamicM0;
    return;
  }
  if (source >= 128u && source <= 192u) {
    site.barrier_id = static_cast<int32_t>(source - 128u);
  } else if (source >= 193u && source <= 208u) {
    site.barrier_id = -static_cast<int32_t>(source - 192u);
  } else if (source == 255u && instruction_bytes.size() >= 2u * sizeof(uint32_t)) {
    int32_t literal = 0;
    std::memcpy(&literal, instruction_bytes.data() + sizeof(uint32_t), sizeof(literal));
    site.barrier_id = literal;
    site.operand_source = ConSanBarrierSite::OperandSource::Literal32;
    site.literal_width_bits = 32u;
    site.literal_value = static_cast<uint32_t>(literal);
    site.scope = barrier_scope_for_id(*site.barrier_id);
    return;
  } else if (source == 254u && instruction_bytes.size() >= 3u * sizeof(uint32_t)) {
    uint64_t literal = 0;
    std::memcpy(&literal, instruction_bytes.data() + sizeof(uint32_t), sizeof(literal));
    site.operand_source = ConSanBarrierSite::OperandSource::Literal64;
    site.literal_width_bits = 64u;
    site.literal_value = literal;
    return;
  } else {
    return;
  }
  site.operand_source = ConSanBarrierSite::OperandSource::Immediate;
  site.scope = barrier_scope_for_id(*site.barrier_id);
}

bool is_s_clause(const Instruction &instruction) { return instruction.mnemonic() == "s_clause"; }

uint32_t s_clause_following_instruction_count(const Instruction &instruction) {
  if (!is_s_clause(instruction) || instruction.raw_encoding() == nullptr ||
      instruction.size() != sizeof(uint32_t))
    return 0;
  uint32_t word = 0;
  std::memcpy(&word, instruction.raw_encoding(), sizeof(word));
  return (word & 0xffffu) + 1u;
}

} // namespace rocjitsu
