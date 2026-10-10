// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared setup for a few decoded RDNA matrix regressions.
#pragma once

#include "fixtures/float_dot/rdna_matrix_sample.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <ios>
#include <span>
#include <utility>

namespace rocjitsu::matrix_test {
// Restore the test-only scalar dispatch override on scope exit.
class ForceScalarGuard {
public:
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(old_); }

private:
  bool old_ = util::force_scalar();
};

using Registers = std::array<std::array<uint32_t, 64>, 33>;
// Reproduce the integer capture inputs, including legal sparse index pairs.
inline Registers integer_input(uint32_t width, bool gfx12) {
  Registers input{};
  uint32_t rng = 0x12056101u;
  constexpr uint32_t c[] = {0,           1,           0xffffffffu, 0x7fffffffu,
                            0x80000000u, 0x7ffff000u, 0x80001000u, 0x12345678u};
  constexpr uint32_t pairs[] = {4, 8, 12, 9, 13, 14};
  for (uint32_t reg = 0; reg < 33; ++reg) {
    for (uint32_t lane = 0; lane < width; ++lane) {
      rng ^= rng << 13;
      rng ^= rng >> 17;
      rng ^= rng << 5;
      uint32_t value = reg >= 16 && reg < 32 ? c[(rng >> 24) & 7] : rng;
      if (reg == 32) {
        value = 0;
        for (uint32_t i = 0; i < 8; ++i)
          value |= pairs[((rng >> (4 * i)) & 15) % 6] << (4 * i);
      }
      input[reg][lane] = !gfx12 && reg < 16 && lane >= 16 ? input[reg][lane % 16] : value;
    }
  }
  return input;
}

// Execute one instruction, check selected destination words, and verify that
// registers outside the destination range are unchanged.
inline void check(rj_code_arch_t arch, uint32_t width, std::array<uint32_t, 2> words, uint32_t mode,
                  const Registers &input, std::span<const Sample> expected,
                  uint32_t output_words = 256) {
  SCOPED_TRACE(testing::Message() << "arch=" << arch << " width=" << width << " word0=" << std::hex
                                  << words[0] << " word1=" << words[1] << " mode=" << mode);
  ASSERT_NE(expected.size(), 0u);
  amdgpu::GpuMemory memory{"matrix_memory"};
  amdgpu::L2Cache cache{"matrix_cache"};
  cache.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("matrix_test", cfg, &memory, &cache);
  auto *wave = cu->dispatch_wf(0, 0, 106, 256, width);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(width == 64 ? ~uint64_t{0} : 0xffffffffu);
  wave->set_mode_raw(mode);
  const uint32_t base = wave->vgpr_alloc().base, dst = words[0] & 255;
  for (uint32_t reg = 0; reg < 33; ++reg)
    for (uint32_t lane = 0; lane < width; ++lane)
      cu->write_vgpr(base + reg, lane, input[reg][lane]);
  const std::array<uint32_t, 4> code{words[0], words[1], 0, 0};
  auto decoded = Decoder::create(arch)->decode(code.data());
  ASSERT_TRUE(decoded.succeeded());
  auto inst = std::move(decoded).value();
  ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
  for (const auto &[word, bits] : expected) {
    ASSERT_LT(word, output_words);
    EXPECT_EQ(cu->read_vgpr(base + dst + word / width, word % width), bits) << "word=" << word;
  }
  for (uint32_t reg = 0; reg < 33; ++reg) {
    if (reg < dst || reg >= dst + output_words / width) {
      for (uint32_t lane = 0; lane < width; ++lane) {
        EXPECT_EQ(cu->read_vgpr(base + reg, lane), input[reg][lane])
            << "reg=" << reg << " lane=" << lane;
      }
    }
  }
  wave->halt();
}
// Allow short expectations to stay inline at their call sites.
inline void check(rj_code_arch_t arch, uint32_t width, std::array<uint32_t, 2> words, uint32_t mode,
                  const Registers &input, std::initializer_list<Sample> expected,
                  uint32_t output_words = 256) {
  check(arch, width, words, mode, input, std::span<const Sample>(expected.begin(), expected.size()),
        output_words);
}
} // namespace rocjitsu::matrix_test
