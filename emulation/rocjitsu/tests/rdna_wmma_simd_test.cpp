// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx11_dot2.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_dot.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_dot_simd.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"
#include <array>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
namespace {
using namespace rocjitsu;
struct HostState {
  fenv_t environment;
  bool scalar = util::force_scalar();
  HostState() { std::fegetenv(&environment); }
  ~HostState() {
    std::fesetenv(&environment);
    util::set_force_scalar_for_testing(scalar);
  }
};
class DotMachine {
public:
  explicit DotMachine(rj_code_arch_t arch, uint32_t width = 32)
      : memory("dot_memory"), cache("dot_cache") {
    cache.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("dot_test", cfg, &memory, &cache);
    decoder = Decoder::create(arch);
    wave = cu->dispatch_wf(0, 0, 106, 256, width);
    wave->set_exec(width == 64 ? ~uint64_t{0} : 0xffffffff);
    base = wave->vgpr_alloc().base;
  }
  ~DotMachine() { wave->halt(); }
  void set(uint32_t reg, uint32_t lane, uint32_t bits) { cu->write_vgpr(base + reg, lane, bits); }
  uint32_t get(uint32_t reg, uint32_t lane) { return cu->read_vgpr(base + reg, lane); }
  void execute(std::array<uint32_t, 2> encoding) {
    std::array<uint32_t, 4> words{encoding[0], encoding[1], 0, 0};
    auto result = decoder->decode(words.data());
    ASSERT_TRUE(result.succeeded());
    auto instruction = std::move(result).value();
    ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
  }
  amdgpu::GpuMemory memory;
  amdgpu::L2Cache cache;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave;
  uint32_t base;
};

#if __has_include(<experimental/simd>)
template <bool Gfx12, bool Bf16> void compare_dot() {
  namespace simd = amdgpu::rdna_dot_simd;
  constexpr unsigned n = Gfx12 ? 4 : 2;
  constexpr unsigned width = simd::width;
  std::mt19937 random(0x71a2);
  const std::array<uint16_t, 12> special = {0,      0x8000, 1,      0x8001, 0x3c00, 0xbc00,
                                            0x3f80, 0xbf80, 0x7c00, 0x7f80, 0x7fff, 0xffff};
  for (unsigned trial = 0; trial != 32768; ++trial) {
    std::array<simd::U, n> a, b;
    for (unsigned j = 0; j < n; ++j) {
      a[j] = simd::U([&](auto lane) -> uint32_t {
        return trial < 8192 ? uint16_t(trial * width + lane) : uint16_t(random());
      });
      b[j] = simd::U([&](auto lane) -> uint32_t {
        return trial < 8192 ? special[(trial + lane + j) % special.size()] : uint16_t(random());
      });
    }
    const simd::U c([&](auto) -> uint32_t { return random(); });
    simd::U result;
    if constexpr (Gfx12)
      result = simd::dot4<Bf16>(a, b, c);
    else
      result = simd::dot2<Bf16>(a, b, c);
    for (unsigned lane = 0; lane < width; ++lane) {
      uint32_t expected;
      if constexpr (Gfx12) {
        std::array<uint16_t, 4> aa, bb;
        for (unsigned j = 0; j < 4; ++j) {
          aa[j] = a[j][lane];
          bb[j] = b[j][lane];
        }
        expected = amdgpu::gfx12_dot_f32<Bf16>(aa, bb, c[lane]);
      } else
        expected =
            amdgpu::gfx11_dot2_f32<Bf16>(a[0][lane], b[0][lane], a[1][lane], b[1][lane], c[lane]);
      ASSERT_EQ(uint32_t(result[lane]), expected) << "trial " << trial << " lane " << lane;
    }
  }
}
TEST(RdnaWmmaSimd, DotArithmeticMatchesScalarBits) {
  compare_dot<false, false>();
  compare_dot<false, true>();
  compare_dot<true, false>();
  compare_dot<true, true>();
}
#endif

TEST(RdnaWmmaSimd, DecodedRawBitsModifiersAliasesAndWaveSizes) {
  HostState restore;
  std::mt19937 random(0x27aa);
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4})
    for (unsigned width : {32u, 64u}) {
      DotMachine machine(arch, width);
      for (bool bf16 : {false, true})
        for (unsigned data = 0; data < 3; ++data) {
          std::vector<uint32_t> initial(128 * width);
          for (auto &word : initial) {
            word = random();
            if (data == 0)
              word &= bf16 ? 0xbfffbfff : 0xbbffbbff;
            if (data == 1) {
              constexpr uint16_t special[] = {0,      0x8000, 1,      0x8001, 0x7c00,
                                              0xfc00, 0x7f80, 0xff80, 0x7fff, 0xffff};
              word = special[random() % 10] | (uint32_t(special[random() % 10]) << 16);
            }
          }
          for (uint8_t mods = 0; mods < 64; ++mods)
            for (uint8_t dst : {0, 32, 64, 96})
              for (bool constant : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << arch << ' ' << width << ' ' << bf16 << ' ' << data << ' '
                             << unsigned(mods) << ' ' << unsigned(dst) << ' ' << constant);
                const uint16_t src2 = constant ? 193 : 320; // inline -1 versus C.
                const auto words = arch == ROCJITSU_CODE_ARCH_RDNA4
                                       ? rdna4::build_vop3p(bf16 ? rdna4::kVWmmaF3216x16x16Bf16Vop3p
                                                                 : rdna4::kVWmmaF3216x16x16F16Vop3p,
                                                            {.vdst = dst,
                                                             .neg_hi = uint8_t(mods >> 3),
                                                             .src0 = 256,
                                                             .src1 = 288,
                                                             .src2 = src2,
                                                             .opsel_hi = 3,
                                                             .neg = uint8_t(mods & 7)})
                                       : rdna3::build_vop3p(bf16 ? rdna3::kVWmmaF3216x16x16Bf16Vop3p
                                                                 : rdna3::kVWmmaF3216x16x16F16Vop3p,
                                                            {.vdst = dst,
                                                             .neg_hi = uint8_t(mods >> 3),
                                                             .src0 = 256,
                                                             .src1 = 288,
                                                             .src2 = src2,
                                                             .op_sel_hi = 3,
                                                             .neg = uint8_t(mods & 7)});
                machine.wave->set_exec(mods & 1 ? 0x5555555555555555ull &
                                                      (width == 64 ? ~uint64_t{0} : 0xffffffffu)
                                       : width == 64 ? ~uint64_t{0}
                                                     : 0xffffffffu);
                auto execute = [&](bool scalar) {
                  util::set_force_scalar_for_testing(scalar);
                  for (unsigned reg = 0; reg < 128; ++reg)
                    for (unsigned lane = 0; lane < width; ++lane)
                      machine.set(reg, lane, initial[reg * width + lane]);
                  const int rounding =
                      std::array{FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}[mods & 3];
                  const int exceptions = mods & 4 ? FE_INEXACT : 0;
                  std::fesetround(rounding);
                  std::feclearexcept(FE_ALL_EXCEPT);
                  std::feraiseexcept(exceptions);
                  machine.execute(words);
                  EXPECT_EQ(std::fegetround(), rounding);
                  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
                  std::vector<uint32_t> result;
                  for (unsigned reg = 0; reg < 128; ++reg)
                    for (unsigned lane = 0; lane < width; ++lane)
                      result.push_back(machine.get(reg, lane));
                  return result;
                };
                const auto scalar = execute(true);
                EXPECT_EQ(execute(false), scalar);
              }
        }
    }
}
} // namespace
