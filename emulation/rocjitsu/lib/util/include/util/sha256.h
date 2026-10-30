// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace util {

/// @brief Incremental SHA-256, shared by code-object identities and capture codecs.
/// @details finish() consumes the state; construct a new instance for another digest.
class Sha256 {
public:
  void update(const void *data, size_t size) {
    const auto *data_bytes = static_cast<const uint8_t *>(data);
    bit_count_ += static_cast<uint64_t>(size) * 8u;
    while (size > 0) {
      const size_t take = std::min(size, sizeof(buffer_) - buffered_);
      std::memcpy(buffer_ + buffered_, data_bytes, take);
      buffered_ += take;
      data_bytes += take;
      size -= take;
      if (buffered_ == sizeof(buffer_)) {
        compress(buffer_);
        buffered_ = 0;
      }
    }
  }

  void update(std::string_view text) { update(text.data(), text.size()); }

  std::array<uint8_t, 32> finish() {
    const uint64_t bits = bit_count_;
    const uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t zero = 0;
    while (buffered_ != 56)
      update(&zero, 1);
    uint8_t tail[8];
    for (int index = 0; index < 8; ++index)
      tail[index] = static_cast<uint8_t>(bits >> (56 - index * 8));
    // Length is appended directly: routing it through update() would grow
    // bit_count_ again and corrupt the padding invariant.
    std::memcpy(buffer_ + buffered_, tail, sizeof(tail));
    compress(buffer_);
    std::array<uint8_t, 32> out{};
    for (int index = 0; index < 8; ++index) {
      out[index * 4] = static_cast<uint8_t>(state_[index] >> 24);
      out[index * 4 + 1] = static_cast<uint8_t>(state_[index] >> 16);
      out[index * 4 + 2] = static_cast<uint8_t>(state_[index] >> 8);
      out[index * 4 + 3] = static_cast<uint8_t>(state_[index]);
    }
    return out;
  }

private:
  uint32_t state_[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  uint64_t bit_count_ = 0;
  uint8_t buffer_[64] = {};
  size_t buffered_ = 0;

  static uint32_t rotr(uint32_t value, uint32_t shift) {
    return (value >> shift) | (value << (32u - shift));
  }

  void compress(const uint8_t *block) {
    static constexpr uint32_t kRoundConstants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u};
    uint32_t schedule[64];
    for (int index = 0; index < 16; ++index)
      schedule[index] = (static_cast<uint32_t>(block[index * 4]) << 24) |
                        (static_cast<uint32_t>(block[index * 4 + 1]) << 16) |
                        (static_cast<uint32_t>(block[index * 4 + 2]) << 8) |
                        static_cast<uint32_t>(block[index * 4 + 3]);
    for (int index = 16; index < 64; ++index) {
      const uint32_t sigma0 = rotr(schedule[index - 15], 7) ^ rotr(schedule[index - 15], 18) ^
                              (schedule[index - 15] >> 3);
      const uint32_t sigma1 = rotr(schedule[index - 2], 17) ^ rotr(schedule[index - 2], 19) ^
                              (schedule[index - 2] >> 10);
      schedule[index] = schedule[index - 16] + sigma0 + schedule[index - 7] + sigma1;
    }
    uint32_t working_a = state_[0], working_b = state_[1], working_c = state_[2],
             working_d = state_[3];
    uint32_t working_e = state_[4], working_f = state_[5], working_g = state_[6],
             working_h = state_[7];
    for (int index = 0; index < 64; ++index) {
      const uint32_t sigma1 = rotr(working_e, 6) ^ rotr(working_e, 11) ^ rotr(working_e, 25);
      const uint32_t choose = (working_e & working_f) ^ (~working_e & working_g);
      const uint32_t temp1 = working_h + sigma1 + choose + kRoundConstants[index] + schedule[index];
      const uint32_t sigma0 = rotr(working_a, 2) ^ rotr(working_a, 13) ^ rotr(working_a, 22);
      const uint32_t majority =
          (working_a & working_b) ^ (working_a & working_c) ^ (working_b & working_c);
      const uint32_t temp2 = sigma0 + majority;
      working_h = working_g;
      working_g = working_f;
      working_f = working_e;
      working_e = working_d + temp1;
      working_d = working_c;
      working_c = working_b;
      working_b = working_a;
      working_a = temp1 + temp2;
    }
    state_[0] += working_a;
    state_[1] += working_b;
    state_[2] += working_c;
    state_[3] += working_d;
    state_[4] += working_e;
    state_[5] += working_f;
    state_[6] += working_g;
    state_[7] += working_h;
  }
};

} // namespace util
