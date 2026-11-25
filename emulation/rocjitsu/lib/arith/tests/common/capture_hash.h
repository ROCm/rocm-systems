// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Stable hashes of captured result words and byte streams.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith::test {

static const uint64_t capture_hash_seed = 14695981039346656037ULL;

// Mix one whole word, as used by word-wise hardware capture digests.
inline uint64_t capture_hash_word(uint64_t hash, uint64_t word) {
  return (hash ^ word) * 1099511628211ULL;
}

// Mix the low byte_count bytes in low-to-high order, independent of host endian.
// byte_count must not exceed eight. This differs from mixing one whole word.
inline uint64_t capture_hash_bytes(uint64_t hash, uint64_t word, int byte_count) {
  for (int byte = 0; byte < byte_count; ++byte) {
    hash = capture_hash_word(hash, (word >> (8 * byte)) & 255);
  }
  return hash;
}

} // namespace rocjitsu::arith::test
