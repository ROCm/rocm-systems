// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fixtures/float_dot/gfx1201_dot4_edge_cases.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_dot.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <array>
#include <cfenv>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>

namespace {
using namespace rocjitsu;

struct Dot4HostState {
  fenv_t environment;
  bool scalar = util::force_scalar();
  Dot4HostState() { std::fegetenv(&environment); }
  ~Dot4HostState() {
    std::fesetenv(&environment);
    util::set_force_scalar_for_testing(scalar);
  }
};

struct Dot4Family {
  uint16_t opcode;
  bool a_fp8, b_fp8;
  std::span<const Gfx12Dot4EdgeCase> cases;
};

const std::array<Dot4Family, 4> kFamilies{{
    {rdna4::kVDot4F32Fp8Fp8Vop3p, true, true, kGfx12Dot4Fp8Fp8Cases},
    {rdna4::kVDot4F32Fp8Bf8Vop3p, true, false, kGfx12Dot4Fp8Bf8Cases},
    {rdna4::kVDot4F32Bf8Fp8Vop3p, false, true, kGfx12Dot4Bf8Fp8Cases},
    {rdna4::kVDot4F32Bf8Bf8Vop3p, false, false, kGfx12Dot4Bf8Bf8Cases},
}};

struct Dot4Encoding {
  uint16_t source;
  uint32_t extension;
  bool dpp16, fi, bound_ctrl;
};

const std::array<Dot4Encoding, 7> kEncodings{{
    {256, 0, false, true, false},
    {0xfa, 0x3d051100, true, true, false},
    {0xea, 0x05397700, false, true, false},
    {0xfa, 0x3d011100, true, false, false},
    {0xfa, 0x3d0d1100, true, true, true},
    {0xfa, 0x3d091100, true, false, true},
    {0xe9, 0x05397700, false, false, false},
}};

uint32_t dot4(const Dot4Family &family, const Gfx12Dot4EdgeCase &sample) {
  std::array<uint8_t, 4> a, b;
  for (uint32_t i = 0; i < 4; ++i) {
    a[i] = static_cast<uint8_t>(sample.a >> (8 * i));
    b[i] = static_cast<uint8_t>(sample.b >> (8 * i));
  }
  if (family.a_fp8)
    return family.b_fp8 ? amdgpu::gfx12_dot4_f32<true, true>(a, b, sample.c)
                        : amdgpu::gfx12_dot4_f32<true, false>(a, b, sample.c);
  return family.b_fp8 ? amdgpu::gfx12_dot4_f32<false, true>(a, b, sample.c)
                      : amdgpu::gfx12_dot4_f32<false, false>(a, b, sample.c);
}

class Dot4Machine {
public:
  explicit Dot4Machine(uint32_t width) : memory("dot4_memory"), cache("dot4_cache") {
    cache.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("dot4_test", cfg, &memory, &cache);
    decoder = Decoder::create(cfg.arch);
    wave = cu->dispatch_wf(0, 0, 106, 256, width);
    base = wave->vgpr_alloc().base;
  }
  ~Dot4Machine() { wave->halt(); }
  void set(uint32_t reg, uint32_t lane, uint32_t value) { cu->write_vgpr(base + reg, lane, value); }
  uint32_t get(uint32_t reg, uint32_t lane) { return cu->read_vgpr(base + reg, lane); }
  void execute(const std::array<uint32_t, 4> &words) {
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
  }
  amdgpu::Wavefront *wave;

private:
  amdgpu::GpuMemory memory;
  amdgpu::L2Cache cache;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  uint32_t base;
};

TEST(Gfx12Dot4, HardwareArithmeticIgnoresHostFpState) {
  Dot4HostState restore;
  for (int round : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    std::fesetround(round);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INEXACT);
    for (const auto &family : kFamilies)
      for (const auto &sample : family.cases)
        EXPECT_EQ(dot4(family, sample), sample.expected)
            << "opcode=" << family.opcode << " rounding=" << round << " A=" << std::hex << sample.a
            << " B=" << sample.b << " C=" << sample.c;
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INEXACT);
    EXPECT_EQ(std::fegetround(), round);
  }
}

TEST(Gfx12Dot4, DecodedHardwareArithmeticIncludesDppAndOverlappingRegisters) {
  Dot4HostState restore;
  for (uint32_t width : {32u, 64u}) {
    Dot4Machine machine(width);
    for (bool scalar : {false, true}) {
      util::set_force_scalar_for_testing(scalar);
      for (const auto &family : kFamilies) {
        for (const auto &encoding : kEncodings) {
          for (uint8_t dst : {uint8_t(0), uint8_t(1), uint8_t(2), uint8_t(16)}) {
            for (uint64_t mask : {~uint64_t{0}, uint64_t{0xa55a5aa55aa5a55a}}) {
              const uint64_t exec = width == 64 ? mask : mask & 0xffffffffu;
              machine.wave->set_exec(exec);
              for (uint32_t offset = 0; offset < family.cases.size(); ++offset) {
                std::array<uint32_t, 64> old_dst{};
                for (uint32_t lane = 0; lane < width; ++lane) {
                  // Place the captured A in the lane selected by the modifier.
                  const uint32_t source = encoding.source == 256 ? lane
                                          : encoding.dpp16 ? (lane + 15) % 16 + (lane / 16) * 16
                                                           : (lane ^ 7);
                  machine.set(0, source, family.cases[(offset + lane) % family.cases.size()].a);
                  const auto &sample = family.cases[(offset + lane) % family.cases.size()];
                  machine.set(1, lane, sample.b);
                  machine.set(2, lane, sample.c);
                  machine.set(16, lane, 0xfeedface);
                }
                for (uint32_t lane = 0; lane < width; ++lane)
                  old_dst[lane] = machine.get(dst, lane);
                const auto base = rdna4::build_vop3p(family.opcode, {.vdst = dst,
                                                                     .src0 = encoding.source,
                                                                     .src1 = 257,
                                                                     .src2 = 258,
                                                                     .opsel_hi = 3});
                machine.execute({base[0], base[1], encoding.extension, 0});
                for (uint32_t lane = 0; lane < width; ++lane) {
                  const bool edge = encoding.dpp16 && lane % 16 == 0;
                  const uint32_t source = encoding.source == 256 ? lane
                                          : encoding.dpp16       ? lane - 1
                                                                 : (lane ^ 7);
                  const bool zero = edge || (!encoding.fi && !(exec & (uint64_t{1} << source)));
                  const bool writes = (exec & (uint64_t{1} << lane)) &&
                                      (!encoding.dpp16 || ((!zero || encoding.bound_ctrl) &&
                                                           ((3u >> (lane / 16)) & 1) &&
                                                           ((13u >> ((lane % 16) / 4)) & 1)));
                  const auto &sample = family.cases[(offset + lane) % family.cases.size()];
                  const uint32_t expected = zero ? sample.zero_a_expected : sample.expected;
                  EXPECT_EQ(machine.get(dst, lane), writes ? expected : old_dst[lane])
                      << "opcode=" << family.opcode << " width=" << width
                      << " src0=" << encoding.source << " fi=" << encoding.fi
                      << " bound_ctrl=" << encoding.bound_ctrl << " dst=" << unsigned(dst)
                      << " offset=" << offset << " lane=" << lane;
                }
              }
            }
          }
        }
      }
    }
  }
}
} // namespace
