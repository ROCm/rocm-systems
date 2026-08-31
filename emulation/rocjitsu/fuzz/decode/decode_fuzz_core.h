// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_FUZZ_DECODE_DECODE_FUZZ_CORE_H_
#define ROCJITSU_FUZZ_DECODE_DECODE_FUZZ_CORE_H_

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

std::unique_ptr<Decoder> create_decoder(std::string_view target);
DecodeRecord decode_window(Decoder &decoder, std::span<const uint8_t> bytes);
std::string record_json(const DecodeRecord &record, std::span<const uint8_t> bytes,
                        std::string_view input_name = {});

} // namespace decode_fuzz
} // namespace rocjitsu

#endif // ROCJITSU_FUZZ_DECODE_DECODE_FUZZ_CORE_H_
