/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TUNING_MASK_H_
#define RCCL_TUNING_MASK_H_

#include <stdint.h>

// A set of cost-model tuning ids. NCCL keeps tuning masks in a uint64_t, which holds at most 64 tuning ids:
// NCCL_TUNING_COUNT is already 63, and each symmetric kernel RCCL adds needs one more. This widens the mask to 128 bits,
// as two 64-bit words.
struct ncclTuningMask {
  static constexpr int kBits = 128;
  uint64_t w[2]; // tuning id i is bit i % 64 of w[i / 64]

  constexpr ncclTuningMask() : w{0, 0} {}

  // {id}, for 0 <= id < kBits.
  static constexpr ncclTuningMask bit(int id) {
    ncclTuningMask m;
    m.w[id / 64] = uint64_t(1) << (id % 64);
    return m;
  }
  // {lo, ..., hi - 1}, for 0 <= lo <= hi <= kBits.
  static constexpr ncclTuningMask range(int lo, int hi) {
    ncclTuningMask m;
    for (int i = 0; i < 2; i++) {
      int l = clampToWord(lo - 64 * i), h = clampToWord(hi - 64 * i);
      if (l < h) m.w[i] = lowBits(h) & ~lowBits(l);
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

  constexpr ncclTuningMask& operator&=(ncclTuningMask o) {
    w[0] &= o.w[0];
    w[1] &= o.w[1];
    return *this;
  }
  constexpr ncclTuningMask& operator|=(ncclTuningMask o) {
    w[0] |= o.w[0];
    w[1] |= o.w[1];
    return *this;
  }
  constexpr ncclTuningMask& operator^=(ncclTuningMask o) {
    w[0] ^= o.w[0];
    w[1] ^= o.w[1];
    return *this;
  }
  // Complement over all kBits ids, ids past NCCL_TUNING_COUNT included: AND the result with a mask of real ids.
  constexpr ncclTuningMask operator~() const {
    ncclTuningMask m;
    m.w[0] = ~w[0];
    m.w[1] = ~w[1];
    return m;
  }
  friend constexpr ncclTuningMask operator&(ncclTuningMask a, ncclTuningMask b) {
    return a &= b;
  }
  friend constexpr ncclTuningMask operator|(ncclTuningMask a, ncclTuningMask b) {
    return a |= b;
  }
  friend constexpr ncclTuningMask operator^(ncclTuningMask a, ncclTuningMask b) {
    return a ^= b;
  }
  friend constexpr bool operator==(ncclTuningMask a, ncclTuningMask b) {
    return a.w[0] == b.w[0] && a.w[1] == b.w[1];
  }
  friend constexpr bool operator!=(ncclTuningMask a, ncclTuningMask b) {
    return !(a == b);
  }

private:
  static constexpr int clampToWord(int n) {
    return n < 0 ? 0 : n > 64 ? 64 : n;
  }
  // The low n bits of a word, for 0 <= n <= 64.
  static constexpr uint64_t lowBits(int n) {
    return n >= 64 ? ~uint64_t(0) : (uint64_t(1) << n) - 1;
  }
};

#endif
