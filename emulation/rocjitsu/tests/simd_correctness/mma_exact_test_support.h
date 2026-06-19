// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mma_exact_test_support.h
/// @brief Shared fixture for the expensive MFMA/WMMA SIMD-vs-scalar bit-exact
/// suites (enabled by RJ_ENABLE_EXPENSIVE_CHECKS).
///
/// The fused-FMA SIMD path and the non-fused scalar path can only round apart
/// when an intermediate K-sum actually rounds, so the random generators seed
/// integer-valued elements that stay exact through every product and partial
/// sum (max |prod| 64, max K-sum 8192 < 2^24); boundary modes drive the
/// rounding-free corners directly (NaN, +/-Inf, +/-0, denormals, max-finite
/// overflow, +1/-1 cancellation). Under those inputs SIMD and scalar must be
/// bit-identical, and the suites assert word-for-word EXPECT_EQ on the dst.

#pragma once

#include "../mma_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace mma_exact {

using namespace rocjitsu;

// Input element formats the executors consume.
enum class Fmt { F16, BF16, F32, F64, FP8, BF8, RAW4, RAW6, I8 };

// Per-trial seeding pattern.
enum class Mode {
  RandomInt,  // small integer-valued elements (exact through every sum)
  Zeros,      // +0 everywhere
  SignedZero, // alternating +0 / -0
  NaN,        // NaN in A, ones in B (skipped for formats without NaN)
  Inf,        // +/-Inf in A, ones in B (skipped for formats without Inf)
  Denorm,     // min subnormal everywhere
  MaxFinite,  // max finite everywhere (sums overflow to Inf identically)
  Cancel,     // alternating +1 / -1
};

inline bool fmt_has_nan(Fmt f) { return f != Fmt::RAW4 && f != Fmt::RAW6 && f != Fmt::I8; }
inline bool fmt_has_inf(Fmt f) {
  return f == Fmt::F16 || f == Fmt::BF16 || f == Fmt::F32 || f == Fmt::F64 || f == Fmt::BF8;
}
inline uint32_t fmt_bits(Fmt f) {
  switch (f) {
  case Fmt::F32:
    return 32;
  case Fmt::F64:
    return 64;
  case Fmt::F16:
  case Fmt::BF16:
    return 16;
  case Fmt::RAW4:
    return 4;
  case Fmt::RAW6:
    return 6;
  default:
    return 8;
  }
}

// Encode one element for `mode`; `idx` decorrelates alternating patterns and
// `rng` drives the random-int mode. Returned value is the raw element bits.
inline uint64_t encode_element(Fmt f, Mode mode, uint32_t idx, std::mt19937 &rng) {
  auto pick_int = [&](int lo, int hi) {
    return static_cast<float>(std::uniform_int_distribution<int>(lo, hi)(rng));
  };
  float v = 0.0f;
  switch (mode) {
  case Mode::RandomInt:
    v = pick_int(f == Fmt::BF8 ? -4 : -8, f == Fmt::BF8 ? 4 : 7);
    break;
  case Mode::Zeros:
    v = 0.0f;
    break;
  case Mode::SignedZero:
    v = (idx & 1) ? -0.0f : 0.0f;
    break;
  case Mode::Cancel:
    v = (idx & 1) ? -1.0f : 1.0f;
    break;
  default:
    break; // NaN / Inf / Denorm / MaxFinite handled per-format below
  }
  switch (f) {
  case Fmt::F16:
    if (mode == Mode::NaN)
      return 0x7E00;
    if (mode == Mode::Inf)
      return (idx & 1) ? 0xFC00 : 0x7C00;
    if (mode == Mode::Denorm)
      return 0x0001;
    if (mode == Mode::MaxFinite)
      return 0x7BFF;
    return util::f32_to_f16(v);
  case Fmt::BF16:
    if (mode == Mode::NaN)
      return 0x7FC0;
    if (mode == Mode::Inf)
      return (idx & 1) ? 0xFF80 : 0x7F80;
    if (mode == Mode::Denorm)
      return 0x0001;
    if (mode == Mode::MaxFinite)
      return 0x7F7F;
    return util::f32_to_bf16(v);
  case Fmt::F32:
    if (mode == Mode::NaN)
      return 0x7FC00000u;
    if (mode == Mode::Inf)
      return (idx & 1) ? 0xFF800000u : 0x7F800000u;
    if (mode == Mode::Denorm)
      return 0x00000001u;
    if (mode == Mode::MaxFinite)
      return 0x7F7FFFFFu;
    return std::bit_cast<uint32_t>(v);
  case Fmt::F64:
    if (mode == Mode::NaN)
      return 0x7FF8000000000000ull;
    if (mode == Mode::Inf)
      return (idx & 1) ? 0xFFF0000000000000ull : 0x7FF0000000000000ull;
    if (mode == Mode::Denorm)
      return 0x0000000000000001ull;
    if (mode == Mode::MaxFinite)
      return 0x7FEFFFFFFFFFFFFFull;
    return std::bit_cast<uint64_t>(static_cast<double>(v));
  case Fmt::FP8: // e4m3: NaN 0x7F, no Inf, denorm 0x01, max 0x7E (448)
    if (mode == Mode::NaN)
      return 0x7F;
    if (mode == Mode::Denorm)
      return 0x01;
    if (mode == Mode::MaxFinite)
      return 0x7E;
    return util::f32_to_fp8_e4m3_rne(v);
  case Fmt::BF8: // e5m2: NaN 0x7E, Inf 0x7C, denorm 0x01, max 0x7B
    if (mode == Mode::NaN)
      return 0x7E;
    if (mode == Mode::Inf)
      return (idx & 1) ? 0xFC : 0x7C;
    if (mode == Mode::Denorm)
      return 0x01;
    if (mode == Mode::MaxFinite)
      return 0x7B;
    return util::f32_to_bf8_e5m2_rne(v);
  case Fmt::RAW4: // every e2m1 pattern is small and exact; random bits suffice
    return (mode == Mode::RandomInt) ? (rng() & 0xF) : (mode == Mode::Zeros ? 0 : (idx * 5) & 0xF);
  case Fmt::RAW6:
    return (mode == Mode::RandomInt) ? (rng() & 0x3F)
                                     : (mode == Mode::Zeros ? 0 : (idx * 11) & 0x3F);
  case Fmt::I8:
    if (mode == Mode::MaxFinite)
      return 0x7F;
    if (mode == Mode::Denorm)
      return 0x80; // INT8_MIN: the int analogue of the magnitude corner
    if (mode == Mode::Cancel)
      return (idx & 1) ? 0xFF : 0x01;
    if (mode == Mode::RandomInt)
      return rng() & 0xFF;
    return 0;
  }
  return 0;
}

// One CU + one wave; the suite seeds raw VGPR words, so the architecture only
// matters for wave width (CDNA4 MFMA = wave64, gfx1250 WMMA = wave32).
struct ExactFixture {
  static constexpr uint32_t SGPRS = 106;
  static constexpr uint32_t VGPRS = 256;

  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t vbase = 0;
  uint32_t wf_size;

  explicit ExactFixture(rj_code_arch_t arch, uint32_t wave)
      : gpu_mem("mma_exact_mem"), l2("mma_exact_l2"), wf_size(wave) {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS;
    cfg.vgprs_per_wf = VGPRS;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_mma_exact", cfg, &gpu_mem, &l2);
    wf = cu->dispatch_wf(0, 0, SGPRS, VGPRS);
    if (wf)
      vbase = wf->vgpr_alloc().base;
  }

  // Fill `regs` registers with elements of `fmt` in `mode`. 64-bit elements
  // occupy (lo,hi) register pairs; `regs` must be even for those.
  void seed(uint32_t off, uint32_t regs, Fmt fmt, Mode mode, uint32_t seed) {
    std::mt19937 rng(seed);
    const uint32_t bits = fmt_bits(fmt);
    uint32_t idx = 0;
    if (bits == 64) {
      for (uint32_t reg = 0; reg + 1 < regs; reg += 2)
        for (uint32_t lane = 0; lane < wf_size; ++lane) {
          uint64_t v = encode_element(fmt, mode, idx++, rng);
          cu->write_vgpr(vbase + off + reg, lane, static_cast<uint32_t>(v));
          cu->write_vgpr(vbase + off + reg + 1, lane, static_cast<uint32_t>(v >> 32));
        }
      return;
    }
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < wf_size; ++lane) {
        uint64_t word = 0;
        for (uint32_t pos = 0; pos < 32; pos += bits)
          word |= encode_element(fmt, mode, idx++, rng) << pos;
        cu->write_vgpr(vbase + off + reg, lane, static_cast<uint32_t>(word));
      }
  }

  void seed_words(uint32_t off, uint32_t regs, uint32_t seed) {
    std::mt19937 rng(seed);
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < wf_size; ++lane)
        cu->write_vgpr(vbase + off + reg, lane, rng());
  }

  std::vector<uint32_t> snapshot(uint32_t off, uint32_t regs) const {
    std::vector<uint32_t> out(static_cast<size_t>(regs) * wf_size);
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < wf_size; ++lane)
        out[static_cast<size_t>(reg) * wf_size + lane] = cu->read_vgpr(vbase + off + reg, lane);
    return out;
  }
};

// Run `kernel` forced-scalar then SIMD over identical pre-seeded state and
// assert the dst window matches bit-for-bit. `reseed_acc` restores the
// accumulator window between runs (the kernels write dst == acc in-place for
// the WMMA suites).
inline void expect_bit_exact(const char *label, Mode mode, ExactFixture &fx,
                             const std::function<void()> &reseed_acc,
                             const std::function<void()> &kernel, uint32_t dst_off,
                             uint32_t dst_regs) {
  reseed_acc();
  util::set_force_scalar_for_testing(true);
  kernel();
  util::set_force_scalar_for_testing(false);
  auto scalar = fx.snapshot(dst_off, dst_regs);

  reseed_acc();
  kernel();
  auto simd = fx.snapshot(dst_off, dst_regs);

  for (size_t i = 0; i < scalar.size(); ++i)
    ASSERT_EQ(scalar[i], simd[i]) << label << " mode=" << static_cast<int>(mode)
                                  << ": SIMD diverges from scalar at word " << i << " (scalar=0x"
                                  << std::hex << scalar[i] << " simd=0x" << simd[i] << ")";
}

// All modes applicable to `fmt`, plus 4 random seeds.
inline std::vector<std::pair<Mode, uint32_t>> trials_for(Fmt fmt) {
  std::vector<std::pair<Mode, uint32_t>> t = {
      {Mode::RandomInt, 11}, {Mode::RandomInt, 22}, {Mode::RandomInt, 33},
      {Mode::RandomInt, 44}, {Mode::Zeros, 0},      {Mode::SignedZero, 0},
      {Mode::Denorm, 0},     {Mode::MaxFinite, 0},  {Mode::Cancel, 0},
  };
  if (fmt_has_nan(fmt))
    t.push_back({Mode::NaN, 0});
  if (fmt_has_inf(fmt))
    t.push_back({Mode::Inf, 0});
  return t;
}

} // namespace mma_exact
