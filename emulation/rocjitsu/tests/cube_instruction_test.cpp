// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>

namespace {
using namespace rocjitsu;

struct ForceScalarGuard {
  explicit ForceScalarGuard(bool force_scalar) : old_force_scalar(util::force_scalar()) {
    util::set_force_scalar_for_testing(force_scalar);
  }
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(old_force_scalar); }
  bool old_force_scalar;
};

struct CubeCase {
  std::array<uint32_t, 3> input;
  // Physical gfx1100 outputs, IEEE=0, RNE: ID, SC, TC, MA.
  std::array<uint32_t, 4> output;
  // Physical CUBEMA outputs for round-up, round-down and round-to-zero.
  std::array<uint32_t, 3> directed_major;
};

// Captured from actual cube opcodes on gfx1100 and cross-checked on gfx1201.
// Face ties, all signs, signed zeros, subnormals, overflow, infinities and NaNs.
constexpr CubeCase kCases[] = {
    {{0x40000000u, 0x3f800000u, 0x3f000000u},
     {0x00000000u, 0xbf000000u, 0xbf800000u, 0x40800000u},
     {0x40800000u, 0x40800000u, 0x40800000u}},
    {{0xc0000000u, 0x3f800000u, 0x3f000000u},
     {0x3f800000u, 0x3f000000u, 0xbf800000u, 0xc0800000u},
     {0xc0800000u, 0xc0800000u, 0xc0800000u}},
    {{0x3f800000u, 0x40000000u, 0x3f000000u},
     {0x40000000u, 0x3f800000u, 0x3f000000u, 0x40800000u},
     {0x40800000u, 0x40800000u, 0x40800000u}},
    {{0x3f800000u, 0xc0000000u, 0x3f000000u},
     {0x40400000u, 0x3f800000u, 0xbf000000u, 0xc0800000u},
     {0xc0800000u, 0xc0800000u, 0xc0800000u}},
    {{0x3f800000u, 0x3f000000u, 0x40000000u},
     {0x40800000u, 0x3f800000u, 0xbf000000u, 0x40800000u},
     {0x40800000u, 0x40800000u, 0x40800000u}},
    {{0x3f800000u, 0x3f000000u, 0xc0000000u},
     {0x40a00000u, 0xbf800000u, 0xbf000000u, 0xc0800000u},
     {0xc0800000u, 0xc0800000u, 0xc0800000u}},
    {{0x3f800000u, 0x3f800000u, 0x00000000u},
     {0x40000000u, 0x3f800000u, 0x00000000u, 0x40000000u},
     {0x40000000u, 0x40000000u, 0x40000000u}},
    {{0x3f800000u, 0x3f800000u, 0x3f800000u},
     {0x40800000u, 0x3f800000u, 0xbf800000u, 0x40000000u},
     {0x40000000u, 0x40000000u, 0x40000000u}},
    {{0xbf800000u, 0xbf800000u, 0xbf800000u},
     {0x40a00000u, 0x3f800000u, 0x3f800000u, 0xc0000000u},
     {0xc0000000u, 0xc0000000u, 0xc0000000u}},
    {{0x40000000u, 0x40000000u, 0x3f800000u},
     {0x40000000u, 0x40000000u, 0x3f800000u, 0x40800000u},
     {0x40800000u, 0x40800000u, 0x40800000u}},
    {{0x3f800000u, 0x40000000u, 0x40000000u},
     {0x40800000u, 0x3f800000u, 0xc0000000u, 0x40800000u},
     {0x40800000u, 0x40800000u, 0x40800000u}},
    {{0x40000000u, 0x3f800000u, 0x40000000u},
     {0x40800000u, 0x40000000u, 0xbf800000u, 0x40800000u},
     {0x40800000u, 0x40800000u, 0x40800000u}},
    {{0x00000000u, 0x00000000u, 0x00000000u},
     {0x40800000u, 0x00000000u, 0x80000000u, 0x00000000u},
     {0x00000000u, 0x00000000u, 0x00000000u}},
    {{0x80000000u, 0x80000000u, 0x80000000u},
     {0x40800000u, 0x80000000u, 0x00000000u, 0x00000000u},
     {0x00000000u, 0x80000000u, 0x00000000u}},
    {{0x80000000u, 0x00000000u, 0x80000000u},
     {0x40800000u, 0x80000000u, 0x80000000u, 0x00000000u},
     {0x00000000u, 0x80000000u, 0x00000000u}},
    {{0x00000001u, 0x80000001u, 0x007fffffu},
     {0x40800000u, 0x00000001u, 0x00000001u, 0x00000000u},
     {0x00000000u, 0x00000000u, 0x00000000u}},
    {{0x807fffffu, 0x007fffffu, 0x80000001u},
     {0x40800000u, 0x807fffffu, 0x807fffffu, 0x00000000u},
     {0x00000000u, 0x80000000u, 0x00000000u}},
    {{0x00800000u, 0x80800000u, 0x00000000u},
     {0x40400000u, 0x00800000u, 0x80000000u, 0x81000000u},
     {0x81000000u, 0x81000000u, 0x81000000u}},
    {{0x00000001u, 0x00800000u, 0x80000001u},
     {0x40000000u, 0x00000001u, 0x80000001u, 0x01000000u},
     {0x01000000u, 0x01000000u, 0x01000000u}},
    {{0x80000001u, 0x00000001u, 0x80800000u},
     {0x40a00000u, 0x00000001u, 0x80000001u, 0x81000000u},
     {0x81000000u, 0x81000000u, 0x81000000u}},
    {{0x7f7fffffu, 0x3f800000u, 0x3f000000u},
     {0x00000000u, 0xbf000000u, 0xbf800000u, 0x7f800000u},
     {0x7f800000u, 0x7f7fffffu, 0x7f7fffffu}},
    {{0xff7fffffu, 0x3f800000u, 0x3f000000u},
     {0x3f800000u, 0x3f000000u, 0xbf800000u, 0xff800000u},
     {0xff7fffffu, 0xff800000u, 0xff7fffffu}},
    {{0x3f800000u, 0x7f000000u, 0x3f000000u},
     {0x40000000u, 0x3f800000u, 0x3f000000u, 0x7f800000u},
     {0x7f800000u, 0x7f7fffffu, 0x7f7fffffu}},
    {{0x3f800000u, 0xff000000u, 0x3f000000u},
     {0x40400000u, 0x3f800000u, 0xbf000000u, 0xff800000u},
     {0xff7fffffu, 0xff800000u, 0xff7fffffu}},
    {{0x3f800000u, 0x3f000000u, 0x7f800000u},
     {0x40800000u, 0x3f800000u, 0xbf000000u, 0x7f800000u},
     {0x7f800000u, 0x7f800000u, 0x7f800000u}},
    {{0x3f800000u, 0x3f000000u, 0xff800000u},
     {0x40a00000u, 0xbf800000u, 0xbf000000u, 0xff800000u},
     {0xff800000u, 0xff800000u, 0xff800000u}},
    {{0x7fc12345u, 0x3f800000u, 0x40000000u},
     {0x00000000u, 0xc0000000u, 0xbf800000u, 0x7fc12345u},
     {0x7fc12345u, 0x7fc12345u, 0x7fc12345u}},
    {{0x3f800000u, 0x7fc12345u, 0x40000000u},
     {0x00000000u, 0xc0000000u, 0xffc12345u, 0x40000000u},
     {0x40000000u, 0x40000000u, 0x40000000u}},
    {{0x3f800000u, 0x40000000u, 0x7fc12345u},
     {0x40000000u, 0x3f800000u, 0x7fc12345u, 0x40800000u},
     {0x40800000u, 0x40800000u, 0x40800000u}},
    {{0x7f800001u, 0x3f800000u, 0x40000000u},
     {0x00000000u, 0xc0000000u, 0xbf800000u, 0x7f800001u},
     {0x7f800001u, 0x7f800001u, 0x7f800001u}},
    {{0x3f800000u, 0x7f800001u, 0x40000000u},
     {0x00000000u, 0xc0000000u, 0xff800001u, 0x40000000u},
     {0x40000000u, 0x40000000u, 0x40000000u}},
    {{0x3f800000u, 0x40000000u, 0x7f800001u},
     {0x40000000u, 0x3f800000u, 0x7f800001u, 0x40800000u},
     {0x40800000u, 0x40800000u, 0x40800000u}},
    {{0xff800001u, 0x7f800001u, 0xff800001u},
     {0x00000000u, 0x7f800001u, 0xff800001u, 0xff800001u},
     {0xff800001u, 0xff800001u, 0xff800001u}},
    {{0x7f800001u, 0xff800001u, 0x7f800001u},
     {0x00000000u, 0xff800001u, 0x7f800001u, 0x7f800001u},
     {0x7f800001u, 0x7f800001u, 0x7f800001u}},
    {{0x7f800000u, 0xff800000u, 0x7f800000u},
     {0x40800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u},
     {0x7f800000u, 0x7f800000u, 0x7f800000u}},
    {{0xff800000u, 0x7f800000u, 0xff800000u},
     {0x40a00000u, 0x7f800000u, 0xff800000u, 0xff800000u},
     {0xff800000u, 0xff800000u, 0xff800000u}},
    {{0x00000001u, 0x3f800000u, 0x00000000u},
     {0x40000000u, 0x00000001u, 0x00000000u, 0x40000000u},
     {0x40000000u, 0x40000000u, 0x40000000u}},
    {{0x3f800000u, 0x00000001u, 0x00000000u},
     {0x00000000u, 0x80000000u, 0x80000001u, 0x40000000u},
     {0x40000000u, 0x40000000u, 0x40000000u}},
    {{0x00000000u, 0x80000001u, 0x3f800000u},
     {0x40800000u, 0x00000000u, 0x00000001u, 0x40000000u},
     {0x40000000u, 0x40000000u, 0x40000000u}},
    {{0x807fffffu, 0x3f800000u, 0x00000000u},
     {0x40000000u, 0x807fffffu, 0x00000000u, 0x40000000u},
     {0x40000000u, 0x40000000u, 0x40000000u}},
    {{0x00000000u, 0x807fffffu, 0xbf800000u},
     {0x40a00000u, 0x80000000u, 0x007fffffu, 0xc0000000u},
     {0xc0000000u, 0xc0000000u, 0xc0000000u}},
    {{0x00000000u, 0x00000000u, 0xff800001u},
     {0x40000000u, 0x00000000u, 0xff800001u, 0x00000000u},
     {0x00000000u, 0x00000000u, 0x00000000u}},
};

class CubeInstructionTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  bool gfx9() const {
    const auto arch = GetParam();
    return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
           arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
  }
  uint32_t first_opcode() const {
    const auto arch = GetParam();
    const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
    return gfx9() ? 452 : gfx10 ? 324 : 524;
  }
  amdgpu::ComputeUnitCore::Config config() const {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = GetParam();
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = gfx9() ? 102 : 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    return cfg;
  }
};

INSTANTIATE_TEST_SUITE_P(AllArchitectures, CubeInstructionTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA1,
                                           ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                                           ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5));

TEST_P(CubeInstructionTest, FaceCoordinatesAndModeMatchPhysicalWitnesses) {
  const auto arch = GetParam();
  for (bool scalar : {false, true}) {
    ForceScalarGuard guard(scalar);
    amdgpu::GpuMemory memory("cube_memory");
    amdgpu::L2Cache l2("cube_l2");
    const auto cfg = config();
    auto cu = amdgpu::ComputeUnitCore::create("cube_cu", cfg, &memory, &l2);
    auto decoder = Decoder::create(arch);
    auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wf, nullptr);
    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t op = 0; op < 4; ++op) {
      const uint32_t words[] = {(gfx9() ? 0xd0000003u : 0xd4000003u) |
                                    ((first_opcode() + op) << 16),
                                256u | (257u << 9) | (258u << 18)};
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
      ASSERT_NE(inst, nullptr);
      for (uint32_t mode = 0; mode < 32; ++mode) {
        const uint32_t round = mode & 3u;
        const bool ieee = mode & 16u;
        wf->set_mode_raw(round | (((mode >> 2) & 3u) << 4) | (ieee ? 512u : 0u));
        for (uint64_t exec : {wf->wf_size() == 64 ? ~0ull : 0xffffffffull, 0x55555555ull}) {
          wf->set_exec(exec);
          for (uint32_t start = 0; start < std::size(kCases); start += wf->wf_size()) {
            for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
              const auto &c = kCases[(start + lane) % std::size(kCases)];
              for (uint32_t source = 0; source < 3; ++source)
                cu->write_vgpr(vb + source, lane, c.input[source]);
              cu->write_vgpr(vb + 3, lane, 0xdeadbeefu);
            }
            ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
            for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
              const auto &c = kCases[(start + lane) % std::size(kCases)];
              uint32_t expected = op == 3 && round ? c.directed_major[round - 1] : c.output[op];
              // RDNA4/CDNA5 always quiet sNaNs; earlier targets honor IEEE.
              if ((ieee || arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5) &&
                  (expected & 0x7fffffffu) > 0x7f800000u)
                expected |= 0x00400000u;
              if (!(exec & (1ull << lane)))
                expected = 0xdeadbeefu;
              EXPECT_EQ(cu->read_vgpr(vb + 3, lane), expected)
                  << "case=" << (start + lane) % std::size(kCases) << " op=" << op
                  << " mode=" << mode << " scalar=" << scalar << " exec=" << exec;
            }
          }
        }
      }
    }
    wf->halt();
  }
}

struct CubeModifierCase {
  std::array<uint32_t, 3> input;
  uint32_t mode;
  uint32_t absolute;
  uint32_t negate;
  uint32_t omod;
  uint32_t clamp;
  std::array<uint32_t, 4> legacy_output;
  std::array<uint32_t, 4> modern_output;
};

// Raw gfx1100/gfx1201 captures: source modifiers, scaling, directed overflow,
// underflow, IEEE OMOD suppression and DX10_CLAMP signaling-NaN behavior.
constexpr CubeModifierCase kModifierCases[] = {
    {{0xc0000000u, 0x3f800001u, 0x40000000u},
     0u,
     0u,
     0u,
     0u,
     0u,
     {0x40800000u, 0xc0000000u, 0xbf800001u, 0x40800000u},
     {0x40800000u, 0xc0000000u, 0xbf800001u, 0x40800000u}},
    {{0x3f7fffffu, 0xff800001u, 0xff800001u},
     0u,
     1u,
     0u,
     0u,
     0u,
     {0x00000000u, 0x7f800001u, 0x7f800001u, 0x3fffffffu},
     {0x00000000u, 0x7fc00001u, 0x7fc00001u, 0x3fffffffu}},
    {{0x80000001u, 0x00800000u, 0x7f800000u},
     0u,
     2u,
     0u,
     0u,
     0u,
     {0x40800000u, 0x80000001u, 0x80800000u, 0x7f800000u},
     {0x40800000u, 0x80000001u, 0x80800000u, 0x7f800000u}},
    {{0x7f800000u, 0x80000000u, 0x807fffffu},
     0u,
     4u,
     0u,
     0u,
     0u,
     {0x00000000u, 0x807fffffu, 0x00000000u, 0x7f800000u},
     {0x00000000u, 0x807fffffu, 0x00000000u, 0x7f800000u}},
    {{0x80000000u, 0x00000001u, 0x00800000u},
     0u,
     0u,
     1u,
     0u,
     0u,
     {0x40800000u, 0x00000000u, 0x80000001u, 0x01000000u},
     {0x40800000u, 0x00000000u, 0x80000001u, 0x01000000u}},
    {{0x3e800000u, 0xbf800000u, 0xc0400000u},
     0u,
     0u,
     2u,
     0u,
     0u,
     {0x40a00000u, 0xbe800000u, 0xbf800000u, 0xc0c00000u},
     {0x40a00000u, 0xbe800000u, 0xbf800000u, 0xc0c00000u}},
    {{0xbf800001u, 0x007fffffu, 0x3f000000u},
     0u,
     0u,
     4u,
     0u,
     0u,
     {0x3f800000u, 0xbf000000u, 0x807fffffu, 0xc0000001u},
     {0x3f800000u, 0xbf000000u, 0x807fffffu, 0xc0000001u}},
    {{0x7f000000u, 0xc0000000u, 0x80000000u},
     0u,
     7u,
     7u,
     0u,
     0u,
     {0x3f800000u, 0x80000000u, 0x40000000u, 0xff800000u},
     {0x3f800000u, 0x80000000u, 0x40000000u, 0xff800000u}},
    {{0x3f000000u, 0x807fffffu, 0x80800000u},
     0u,
     0u,
     0u,
     1u,
     0u,
     {0x00000000u, 0x01000000u, 0x00000000u, 0x40000000u},
     {0x00000000u, 0x01000000u, 0x00000000u, 0x40000000u}},
    {{0xbf000000u, 0x7fc12345u, 0x7f800001u},
     0u,
     0u,
     0u,
     1u,
     0u,
     {0x40000000u, 0x7f800001u, 0xffc12345u, 0xc0000000u},
     {0x40000000u, 0x7fc00001u, 0xffc12345u, 0xc0000000u}},
    {{0x3e800000u, 0xbf800000u, 0xc0400000u},
     32u,
     0u,
     0u,
     1u,
     0u,
     {0x41200000u, 0xbf000000u, 0x40000000u, 0xc1400000u},
     {0x41200000u, 0xbf000000u, 0x40000000u, 0xc1400000u}},
    {{0x3e800000u, 0xbf800000u, 0xc0400000u},
     512u,
     0u,
     0u,
     1u,
     0u,
     {0x40a00000u, 0xbe800000u, 0x3f800000u, 0xc0c00000u},
     {0x41200000u, 0xbf000000u, 0x40000000u, 0xc1400000u}},
    {{0x7f7fffffu, 0x80000001u, 0xff800000u},
     1u,
     0u,
     0u,
     1u,
     0u,
     {0x41200000u, 0xff7fffffu, 0x00000000u, 0xff800000u},
     {0x41200000u, 0xff7fffffu, 0x00000000u, 0xff800000u}},
    {{0x7f7fffffu, 0x80000001u, 0xff800000u},
     2u,
     0u,
     0u,
     1u,
     0u,
     {0x41200000u, 0xff800000u, 0x00000000u, 0xff800000u},
     {0x41200000u, 0xff800000u, 0x00000000u, 0xff800000u}},
    {{0x7f7fffffu, 0x80000001u, 0xff800000u},
     3u,
     0u,
     0u,
     1u,
     0u,
     {0x41200000u, 0xff7fffffu, 0x00000000u, 0xff800000u},
     {0x41200000u, 0xff7fffffu, 0x00000000u, 0xff800000u}},
    {{0x40000000u, 0xbf000000u, 0xff000000u},
     3u,
     0u,
     0u,
     2u,
     0u,
     {0x41a00000u, 0xc1000000u, 0x40000000u, 0xff7fffffu},
     {0x41a00000u, 0xc1000000u, 0x40000000u, 0xff7fffffu}},
    {{0x00800000u, 0xff800000u, 0x80000001u},
     0u,
     0u,
     0u,
     3u,
     0u,
     {0x3fc00000u, 0x00000000u, 0x00000000u, 0xff800000u},
     {0x3fc00000u, 0x00000000u, 0x00000000u, 0xff800000u}},
    {{0x80800000u, 0x3f800000u, 0x00000000u},
     0u,
     0u,
     0u,
     3u,
     0u,
     {0x3f800000u, 0x80000000u, 0x00000000u, 0x3f800000u},
     {0x3f800000u, 0x80000000u, 0x00000000u, 0x3f800000u}},
    {{0x80000000u, 0x00000001u, 0x00800000u},
     2u,
     0u,
     0u,
     1u,
     0u,
     {0x41000000u, 0x00000000u, 0x00000000u, 0x01800000u},
     {0x41000000u, 0x00000000u, 0x00000000u, 0x01800000u}},
    {{0x80000000u, 0x00000001u, 0x00800000u},
     0u,
     0u,
     0u,
     1u,
     0u,
     {0x41000000u, 0x00000000u, 0x00000000u, 0x01800000u},
     {0x41000000u, 0x00000000u, 0x00000000u, 0x01800000u}},
    {{0xbf000000u, 0x7fc12345u, 0x7f800001u},
     0u,
     0u,
     0u,
     1u,
     1u,
     {0x3f800000u, 0x7f800001u, 0xffc12345u, 0x00000000u},
     {0x3f800000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {{0xbf000000u, 0x7fc12345u, 0x7f800001u},
     256u,
     0u,
     0u,
     1u,
     1u,
     {0x3f800000u, 0x00000000u, 0x00000000u, 0x00000000u},
     {0x3f800000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {{0xbf000000u, 0x7fc12345u, 0x7f800001u},
     512u,
     0u,
     0u,
     1u,
     1u,
     {0x3f800000u, 0x7fc00001u, 0xffc12345u, 0x00000000u},
     {0x3f800000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {{0xbf000000u, 0x7fc12345u, 0x7f800001u},
     768u,
     0u,
     0u,
     1u,
     1u,
     {0x3f800000u, 0x00000000u, 0x00000000u, 0x00000000u},
     {0x3f800000u, 0x00000000u, 0x00000000u, 0x00000000u}},
    {{0x80800000u, 0x3f800000u, 0x00000000u},
     0u,
     0u,
     0u,
     3u,
     1u,
     {0x3f800000u, 0x00000000u, 0x00000000u, 0x3f800000u},
     {0x3f800000u, 0x00000000u, 0x00000000u, 0x3f800000u}},
    {{0xff800001u, 0x40000000u, 0x7f000000u},
     2u,
     0u,
     0u,
     2u,
     0u,
     {0x00000000u, 0xff800000u, 0xc1000000u, 0xff800001u},
     {0x00000000u, 0xff800000u, 0xc1000000u, 0xffc00001u}},
    {{0xff800001u, 0x40000000u, 0x7f000000u},
     0u,
     7u,
     7u,
     1u,
     0u,
     {0x00000000u, 0x7f800000u, 0x40800000u, 0xff800001u},
     {0x00000000u, 0x7f800000u, 0x40800000u, 0xffc00001u}},
    {{0x7f7fffffu, 0x80000001u, 0xff800000u},
     1u,
     7u,
     7u,
     2u,
     0u,
     {0x41a00000u, 0x7f800000u, 0x00000000u, 0xff800000u},
     {0x41a00000u, 0x7f800000u, 0x00000000u, 0xff800000u}},
    {{0xff800001u, 0x40000000u, 0x7f000000u},
     768u,
     7u,
     7u,
     3u,
     1u,
     {0x00000000u, 0x3f800000u, 0x3f800000u, 0x00000000u},
     {0x00000000u, 0x3f800000u, 0x3f800000u, 0x00000000u}},
};

TEST_P(CubeInstructionTest, SourceAndOutputModifiersMatchPhysicalWitnesses) {
  const auto arch = GetParam();
  const bool modern = arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
  for (bool scalar : {false, true}) {
    ForceScalarGuard guard(scalar);
    amdgpu::GpuMemory memory("cube_modifier_memory");
    amdgpu::L2Cache l2("cube_modifier_l2");
    const auto cfg = config();
    auto cu = amdgpu::ComputeUnitCore::create("cube_modifier_cu", cfg, &memory, &l2);
    auto decoder = Decoder::create(arch);
    auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wf, nullptr);
    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t index = 0; index < std::size(kModifierCases); ++index) {
      const auto &c = kModifierCases[index];
      wf->set_mode_raw(c.mode);
      for (uint32_t op = 0; op < 4; ++op) {
        const uint32_t words[] = {
            (gfx9() ? 0xd0000003u : 0xd4000003u) | ((first_opcode() + op) << 16) |
                (c.absolute << 8) | (c.clamp << 15),
            256u | (257u << 9) | (258u << 18) | (c.omod << 27) | (c.negate << 29)};
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
        ASSERT_NE(inst, nullptr);
        for (uint64_t exec : {wf->wf_size() == 64 ? ~0ull : 0xffffffffull, 0x55555555ull}) {
          wf->set_exec(exec);
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
            for (uint32_t source = 0; source < 3; ++source)
              cu->write_vgpr(vb + source, lane, c.input[source]);
            cu->write_vgpr(vb + 3, lane, 0xdeadbeefu);
          }
          ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
            const uint32_t expected = exec & (1ull << lane)
                                          ? (modern ? c.modern_output : c.legacy_output)[op]
                                          : 0xdeadbeefu;
            EXPECT_EQ(cu->read_vgpr(vb + 3, lane), expected)
                << "case=" << index << " op=" << op << " scalar=" << scalar << " exec=" << exec;
          }
        }
      }
    }
    wf->halt();
  }
}
} // namespace
