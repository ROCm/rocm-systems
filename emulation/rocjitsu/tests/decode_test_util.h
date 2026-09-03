// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TESTS_DECODE_TEST_UTIL_H_
#define ROCJITSU_TESTS_DECODE_TEST_UTIL_H_

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/decoder.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace rocjitsu {

/// Decode an instruction encoding expected to be valid by the surrounding test.
///
/// Returning nullptr preserves the existing assertion style at valid-decode
/// call sites while rejection-specific tests inspect DecodeResult directly.
inline Instruction *decode_valid(Decoder &decoder, const rj_code_binary_inst_t *inst) {
  DecodeResult result = decoder.decode(inst);
  if (result.failed())
    return nullptr;
  return result.value().release();
}

inline Instruction *decode_valid(Decoder &decoder, const rj_code_binary_inst_t *inst,
                                 uint64_t src_loc) {
  DecodeResult result = decoder.decode(inst, src_loc);
  if (result.failed())
    return nullptr;
  return result.value().release();
}

inline bool decode_fails(Decoder &decoder, const rj_code_binary_inst_t *inst) {
  return decoder.decode(inst).failed();
}

/// Unwrap a block build expected to succeed by the surrounding test.
inline std::vector<std::unique_ptr<BasicBlock>>
build_valid_blocks(const CodeObject &co, Decoder &decoder, rj_code_arch_t arch,
                   std::span<const uint64_t> extra_leaders = {},
                   ExternalEntryPolicy entry_policy = ExternalEntryPolicy::InferPredecessorless,
                   std::span<const uint64_t> extra_split_points = {}) {
  util::StringDiagnostic decode_error;
  auto result = BasicBlock::build(co, decoder, arch, decode_error.emitter(), extra_leaders,
                                  entry_policy, extra_split_points);
  if (result.failed()) {
    ADD_FAILURE() << "failed to build expected-valid blocks: " << decode_error.message();
    return {};
  }
  return std::move(result).value();
}

} // namespace rocjitsu

#endif // ROCJITSU_TESTS_DECODE_TEST_UTIL_H_
