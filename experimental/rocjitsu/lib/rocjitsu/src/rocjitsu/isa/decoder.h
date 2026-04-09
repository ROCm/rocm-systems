// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decoder.h
/// @brief Instruction decoder interface and ISA-parameterized implementation.

#ifndef ROCJITSU_ISA_DECODER_H_
#define ROCJITSU_ISA_DECODER_H_

#include "rocjitsu/base/api.h"
#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <memory>

namespace rocjitsu {

class Instruction;

/// @brief Abstract interface for decoding binary instructions into Instruction objects.
class Decoder {
public:
  virtual ~Decoder() = default;

  /// @brief Decode a single binary instruction.
  /// @param[in] inst Pointer to the binary instruction encoding.
  /// @returns Decoded Instruction, or nullptr if the encoding is invalid.
  virtual std::unique_ptr<Instruction> decode(const rj_code_binary_inst_t *inst) const = 0;

  /// @brief Create a decoder for the given architecture.
  /// @param[in] arch Architecture to create a decoder for.
  /// @returns Architecture-specific decoder instance.
  static std::unique_ptr<Decoder> create(rj_code_arch_t arch);
};

/// @brief ISA-parameterized decoder that delegates to the ISA's static decode method.
/// @tparam Isa ISA traits type that provides a nested Decoder with a static decode() method.
template <typename Isa> class IsaDecoder final : public Decoder {
public:
  /// @brief Decode a binary instruction using the ISA-specific decoder.
  /// @param[in] inst Pointer to the binary instruction encoding.
  /// @returns Decoded Instruction, or nullptr if the encoding is invalid.
  std::unique_ptr<Instruction> decode(const rj_code_binary_inst_t *inst) const override {
    return Isa::Decoder::decode(inst);
  }
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_DECODER_H_
