// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_FUZZ_DECODE_DECODE_FUZZ_CORE_H_
#define ROCJITSU_FUZZ_DECODE_DECODE_FUZZ_CORE_H_

#include "util/diagnostic.h"
#include "util/result.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

class Decoder;

namespace decode_fuzz {

inline constexpr size_t kWindowSize = 16;

struct DecodeRecord {
  bool valid = false;
  int size = 0;
  uint16_t encoding_id = 0;
  uint16_t opcode = 0;
  std::string mnemonic;
  std::string disassembly;
  std::string rejection;
  std::vector<std::string> destinations;
  std::vector<std::string> sources;
};

/// @brief Return failure for an unsupported target, emitting its reason to diagnostic.
util::FailureOr<std::unique_ptr<Decoder>> create_decoder(std::string_view target,
                                                         util::DiagnosticEmitter diagnostic = {});
/// @brief Return failure for an invalid window size; rejected guest encodings are records.
/// @details Internal decoder invariant violations terminate the process, including under NDEBUG.
util::FailureOr<DecodeRecord> decode_window(Decoder &decoder, std::span<const uint8_t> bytes,
                                            util::DiagnosticEmitter diagnostic = {});
std::string record_json(const DecodeRecord &record, std::span<const uint8_t> bytes,
                        std::string_view input_name = {});

} // namespace decode_fuzz
} // namespace rocjitsu

#endif // ROCJITSU_FUZZ_DECODE_DECODE_FUZZ_CORE_H_
