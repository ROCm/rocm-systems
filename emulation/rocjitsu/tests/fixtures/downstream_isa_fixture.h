// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file downstream_isa_fixture.h
/// @brief Minimal out-of-tree-style ISA used to test static provider selection.

#pragma once

#include "rocjitsu/isa/instruction.h"

#include <cstddef>
#include <memory>

namespace rocjitsu::test {

class DownstreamInstruction final : public Instruction {
public:
  explicit DownstreamInstruction(const rj_code_binary_inst_t *raw)
      : Instruction("downstream_nop", nullptr) {
    size_ = sizeof(*raw);
    raw_encoding_ = raw;
  }
};

class DownstreamIsa {
public:
  class Decoder {
  public:
    // Source-integrated decoders publish their maximum encoded width and raw lookahead.
    static constexpr std::size_t kMaxInstructionWords = 1;
    static std::unique_ptr<Instruction> decode(const rj_code_binary_inst_t *raw) {
      return std::make_unique<DownstreamInstruction>(raw);
    }
  };
};

} // namespace rocjitsu::test
