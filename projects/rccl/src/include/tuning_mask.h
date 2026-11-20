/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TUNING_MASK_H_
#define RCCL_TUNING_MASK_H_

#include <stdint.h>
#include <type_traits>

// A set of cost-model tuning ids. No constructor, so it stays trivial: value-initialize it (`rcclTuningMask m{}`).
struct rcclTuningMask {
  static constexpr int kBits = 128;

  // {id}, for 0 <= id < kBits.
  static constexpr rcclTuningMask bit(int id) {
    rcclTuningMask m{};
    m.w[id / 64] = uint64_t(1) << (id % 64);
    return m;
  }
  // {lo, ..., hi - 1}, for 0 <= lo <= hi <= kBits.
  static constexpr rcclTuningMask range(int lo, int hi) {
    rcclTuningMask m{};
    for (int i = 0; i < 2; i++) {
      int l = clampToWord(lo - 64 * i), h = clampToWord(hi - 64 * i);
      if (l < h) m.w[i] = lowBits(h) & ~lowBits(l);
    }
    return m;
  }
  // The ids shift + i for each bit i set in bits (those below kBits), for 0 <= shift < kBits.
  static constexpr rcclTuningMask fromWord(uint64_t bits, int shift) {
    rcclTuningMask m{};
    if (shift >= 64) {
      m.w[1] = bits << (shift - 64);
    } else {
      m.w[0] = bits << shift;
      m.w[1] = shift == 0 ? 0 : bits >> (64 - shift);
    }
    return m;
  }

  constexpr bool test(int id) const {
    return (w[id / 64] >> (id % 64)) & 1;
  }
  constexpr bool any() const {
    return (w[0] | w[1]) != 0;
  }
  constexpr bool none() const {
    return !any();
  }
  // Ids shift .. shift + 63 as the bits of a word (ids past kBits read as 0), for 0 <= shift < kBits.
  constexpr uint64_t word(int shift) const {
    if (shift >= 64) return w[1] >> (shift - 64);
    return shift == 0 ? w[0] : (w[0] >> shift) | (w[1] << (64 - shift));
  }

  constexpr rcclTuningMask& operator&=(rcclTuningMask o) {
    w[0] &= o.w[0];
    w[1] &= o.w[1];
    return *this;
  }
  constexpr rcclTuningMask& operator|=(rcclTuningMask o) {
    w[0] |= o.w[0];
    w[1] |= o.w[1];
    return *this;
  }
  constexpr rcclTuningMask& operator^=(rcclTuningMask o) {
    w[0] ^= o.w[0];
    w[1] ^= o.w[1];
    return *this;
  }
  // Complement over all kBits ids, ids past NCCL_TUNING_COUNT included: AND the result with a mask of real ids.
  constexpr rcclTuningMask operator~() const {
    rcclTuningMask m{};
    m.w[0] = ~w[0];
    m.w[1] = ~w[1];
    return m;
  }
  friend constexpr rcclTuningMask operator&(rcclTuningMask a, rcclTuningMask b) {
    return a &= b;
  }
  friend constexpr rcclTuningMask operator|(rcclTuningMask a, rcclTuningMask b) {
    return a |= b;
  }
  friend constexpr rcclTuningMask operator^(rcclTuningMask a, rcclTuningMask b) {
    return a ^= b;
  }
  friend constexpr bool operator==(rcclTuningMask a, rcclTuningMask b) {
    return a.w[0] == b.w[0] && a.w[1] == b.w[1];
  }
  friend constexpr bool operator!=(rcclTuningMask a, rcclTuningMask b) {
    return !(a == b);
  }

private:
  uint64_t w[2]; // tuning id i is bit i % 64 of w[i / 64]

  static constexpr int clampToWord(int n) {
    return n < 0 ? 0 : n > 64 ? 64 : n;
  }
  // The low n bits of a word, for 0 <= n <= 64.
  static constexpr uint64_t lowBits(int n) {
    return n >= 64 ? ~uint64_t(0) : (uint64_t(1) << n) - 1;
  }
};
static_assert(std::is_trivial<rcclTuningMask>::value, "rcclTuningMask must stay trivial");

#endif
