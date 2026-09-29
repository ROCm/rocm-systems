// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hwfloat_mul_f32_execution_test.cpp
/// @brief Target-qualified V_MUL_F32 execution on gfx1100 and gfx1201.
///
/// @details The physically qualified plain policy reports INVALID for zero
/// times infinity, including observer and scalar execution. Unqualified
/// targets and modified forms retain their existing numerical policy.

#include "decode_test_util.h"
#include "legacy_gpu_memory_fixture.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/dx9_mul_f32.h"
#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/mul_f32.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu::hwfloat;
namespace cause = mul_f32_cause;

constexpr uint32_t kSgprs = 106;
constexpr uint32_t kVgprs = 64;
constexpr uint32_t kUnwritten = 0xdeadbeefu;
constexpr uint32_t kKernelAddr = 0x1000;
constexpr uint32_t kSEndpgm = 0xBFB00000u;
constexpr uint32_t kIeeeMode = 1u << 9;
constexpr uint32_t kDx10Clamp = 1u << 8;

struct QualifiedTarget {
  rj_code_arch_t arch;
  rj_code_target_id_t target;
  uint32_t vop2_opcode;
  uint32_t vop3_opcode;
  const char *name;
};

constexpr QualifiedTarget kQualifiedTargets[] = {
    {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_TARGET_GFX1100, rdna3::kVMulF32Vop2,
     rdna3::kVMulF32Vop3, "gfx1100"},
    {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201, rdna4::kVMulF32Vop2,
     rdna4::kVMulF32Vop3, "gfx1201"},
};

constexpr uint32_t vgpr(uint32_t index) { return 256 + index; }

// GFX11/GFX12 VOP2: op[30:25], vdst[24:17], vsrc1[16:9], src0[8:0].
constexpr uint32_t vop2(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return (opcode << 25) | (vdst << 17) | (vsrc1 << 9) | src0;
}

struct Vop3Modifiers {
  uint32_t abs = 0;
  uint32_t neg = 0;
  uint32_t op_sel = 0;
  uint32_t clamp = 0;
  uint32_t omod = 0;
};

// GFX11/GFX12 VOP3: vdst[7:0], abs[10:8], op_sel[14:11], clamp[15], op[25:16],
// encoding 0x35 in [31:26]; src0[8:0], src1[17:9], src2[26:18], omod[28:27],
// neg[31:29].
constexpr std::array<uint32_t, 2> vop3(uint32_t opcode, uint32_t vdst, uint32_t src0, uint32_t src1,
                                       Vop3Modifiers m = {}) {
  return {vdst | (m.abs << 8) | (m.op_sel << 11) | (m.clamp << 15) | (opcode << 16) | (0x35u << 26),
          src0 | (src1 << 9) | (m.omod << 27) | (m.neg << 29)};
}

constexpr uint32_t mode(uint32_t round, uint32_t denorm, uint32_t flags = 0) {
  return round | (denorm << 4) | flags;
}

MulF32Policy policy_for(const QualifiedTarget &target, uint32_t mode_bits) {
  return {.round = mode_bits & 3,
          .denorm = (mode_bits >> 4) & 3,
          .quiet_nan = target.arch == ROCJITSU_CODE_ARCH_RDNA4 || (mode_bits & kIeeeMode) != 0};
}

uint32_t apply_modifiers(uint32_t value, uint32_t abs, uint32_t neg, uint32_t index) {
  if (abs & (1u << index))
    value &= 0x7fffffffu;
  if (neg & (1u << index))
    value ^= 0x80000000u;
  return value;
}

struct ForceScalarOverride {
  explicit ForceScalarOverride(bool value) : old(util::force_scalar()) {
    util::set_force_scalar_for_testing(value);
  }
  ~ForceScalarOverride() { util::set_force_scalar_for_testing(old); }

  bool old;
};

struct SavedHostEnvironment {
  int error_number = errno;
  std::fenv_t environment{};
#if defined(__x86_64__)
  uint32_t mxcsr = _mm_getcsr();
#endif

  SavedHostEnvironment() { std::fegetenv(&environment); }
  ~SavedHostEnvironment() { restore(); }

  void restore() const {
    std::fesetenv(&environment);
#if defined(__x86_64__)
    _mm_setcsr(mxcsr);
#endif
    errno = error_number;
  }
};

struct MulFixture {
  test::LegacyGpuMemoryFixture gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;

  MulFixture(rj_code_arch_t arch, rj_code_target_id_t target,
             amdgpu::MemoryWaitDiagnostics diagnostics = amdgpu::MemoryWaitDiagnostics::Off)
      : gpu_mem("hwfloat_mul_mem"), l2("hwfloat_mul_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.target = target;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = kSgprs;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    cfg.memory_wait_diagnostics = diagnostics;
    cu = amdgpu::ComputeUnitCore::create("hwfloat_mul_cu", cfg, &gpu_mem, &l2);
    l2.set_gpu_vm(&gpu_mem.gpu_vm());
    cu->set_gpu_vm(&gpu_mem.gpu_vm());
    decoder = Decoder::create(arch);
  }

  amdgpu::Wavefront *dispatch(uint32_t wave_size, uint32_t vgprs = kVgprs, uint64_t pc = 0) {
    return cu->dispatch_wf(0, pc, kSgprs, vgprs, wave_size);
  }

  void fill_vgpr(const amdgpu::Wavefront &wf, uint32_t reg, uint32_t value) {
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
      cu->write_vgpr(wf.vgpr_alloc().base + reg, lane, value);
  }

  uint32_t vgpr_value(const amdgpu::Wavefront &wf, uint32_t reg, uint32_t lane) const {
    return cu->read_vgpr_storage(wf.vgpr_alloc().base + reg, lane);
  }

  std::unique_ptr<Instruction> decode(std::span<const uint32_t> words) {
    std::array<uint32_t, 4> buffer{};
    std::copy(words.begin(), words.end(), buffer.begin());
    return std::unique_ptr<Instruction>(decode_valid(*decoder, buffer.data()));
  }
};

struct Outcome {
  uint32_t trapsts = 0;
  uint32_t pending = 0;
};

Outcome run_mul(MulFixture &f, amdgpu::Wavefront &wf, std::span<const uint32_t> words) {
  std::unique_ptr<Instruction> inst = f.decode(words);
  EXPECT_NE(inst, nullptr);
  if (!inst)
    return {};
  wf.clear_pending_alu_causes();
  EXPECT_TRUE(f.cu->execute_instruction(inst.get(), wf).succeeded());
  return {wf.trapsts(), wf.pending_alu_causes()};
}

Outcome run_mul(MulFixture &f, amdgpu::Wavefront &wf, uint32_t word, uint32_t literal = 0) {
  const std::array<uint32_t, 2> words{word, literal};
  return run_mul(f, wf, words);
}

// Lane inputs covering every fact. Lane i uses pair (5 * i) % 16, so lanes 0,
// 16, 32, and 48 hold the exact product and lane 33 multiplies zero by
// infinity.
constexpr std::array<std::pair<uint32_t, uint32_t>, 16> kLanePairs = {{
    {0x3fc00000u, 0x3fc00000u}, // exact
    {0x3f8ccccdu, 0x3f8ccccdu}, // inexact
    {0x1f800001u, 0x207ffffeu}, // tiny before, not after, significand rounding
    {0x1fffffffu, 0x20000000u}, // tiny after significand rounding
    {0x00000001u, 0x3f800000u}, // subnormal source
    {0x00000000u, 0x7f800000u}, // zero times infinity
    {0x7fa00000u, 0x3f800000u}, // signaling NaN
    {0x7fc00001u, 0xffa00000u}, // quiet NaN before a signaling NaN
    {0x7f7fffffu, 0x40000000u}, // overflow
    {0x00000001u, 0x7f800000u}, // subnormal times infinity
    {0x00800000u, 0x3f000000u}, // exponent sum 127
    {0x7f000000u, 0x3f000000u}, // exponent sum 380
    {0x9fffffffu, 0x20000000u}, // negative, tiny after significand rounding
    {0x80000000u, 0x3f800000u}, // negative zero
    {0x00400000u, 0x00400000u}, // subnormal squared
    {0x3f7fffffu, 0x00800000u}, // just below the smallest normal
}};

constexpr std::pair<uint32_t, uint32_t> lane_pair(uint32_t lane) {
  return kLanePairs[(5 * lane) % kLanePairs.size()];
}

constexpr uint64_t kExactLanes = 0x0001'0001'0001'0001ull;

TEST(HwfloatMulF32ExecutionTest, ResultsAndCausesFollowTheIntegerModel) {
  ForceScalarOverride simd(false);
  constexpr uint32_t kSeed = (1u << 6) | cause::kUnderflow;
  for (const QualifiedTarget &target : kQualifiedTargets) {
    for (uint32_t wave_size : {32u, 64u}) {
      MulFixture f(target.arch, target.target);
      amdgpu::Wavefront *wf = f.dispatch(wave_size);
      ASSERT_NE(wf, nullptr);
      for (uint32_t lane = 0; lane < wave_size; ++lane) {
        f.cu->write_vgpr(wf->vgpr_alloc().base + 1, lane, lane_pair(lane).first);
        f.cu->write_vgpr(wf->vgpr_alloc().base + 2, lane, lane_pair(lane).second);
      }
      const uint64_t lanes = wave_size == 64 ? ~uint64_t{0} : 0xffff'ffffull;
      const uint64_t masks[] = {lanes, 0x0000'0002'8000'0421ull & lanes,
                                wave_size == 64 ? uint64_t{1} << 33 : uint64_t{1} << 5,
                                kExactLanes & lanes};
      for (uint32_t round = 0; round < 4; ++round) {
        for (uint32_t denorm : {0u, 1u, 2u, 3u}) {
          for (uint32_t flags : {0u, kIeeeMode}) {
            for (uint64_t exec : masks) {
              const uint32_t mode_bits = mode(round, denorm, flags);
              SCOPED_TRACE(testing::Message() << target.name << " wave" << wave_size << " mode "
                                              << mode_bits << " exec " << std::hex << exec);
              f.fill_vgpr(*wf, 3, kUnwritten);
              wf->set_mode_raw(mode_bits);
              wf->set_exec(exec);
              wf->set_trapsts(kSeed);
              const Outcome outcome = run_mul(f, *wf, vop2(target.vop2_opcode, 3, 2, vgpr(1)));

              const MulF32Policy p = policy_for(target, mode_bits);
              uint32_t expected_causes = 0;
              for (uint32_t lane = 0; lane < wave_size; ++lane) {
                const bool active = (exec >> lane) & 1;
                const auto [a, b] = lane_pair(lane);
                const MulF32Result r = multiply_f32(a, b, p);
                if (active)
                  expected_causes |= plain_mul_f32_causes(r, p);
                EXPECT_EQ(f.vgpr_value(*wf, 3, lane), active ? r.bits : kUnwritten)
                    << "lane " << lane;
                EXPECT_EQ(f.vgpr_value(*wf, 1, lane), a) << "lane " << lane;
                EXPECT_EQ(f.vgpr_value(*wf, 2, lane), b) << "lane " << lane;
              }
              if (exec == (kExactLanes & lanes)) {
                EXPECT_EQ(expected_causes, 0u);
              }
              EXPECT_EQ(outcome.pending, expected_causes);
              EXPECT_EQ(outcome.trapsts, kSeed | expected_causes);
              if (HasFailure())
                return;
            }
          }
        }
      }
    }
  }
}

TEST(HwfloatMulF32ExecutionTest, ScalarSourcesModifiersAndDestinationAliasing) {
  ForceScalarOverride simd(false);
  constexpr uint32_t kScalar = 0x3f8ccccdu;
  constexpr uint32_t kInlineHalf = 240;
  constexpr uint32_t kLiteral = 255;
  constexpr uint32_t kSgpr = 4;
  constexpr uint32_t kHighSgpr = 100;
  for (const QualifiedTarget &target : kQualifiedTargets) {
    for (uint32_t wave_size : {32u, 64u}) {
      MulFixture f(target.arch, target.target);
      amdgpu::Wavefront *wf = f.dispatch(wave_size);
      ASSERT_NE(wf, nullptr);
      const uint64_t exec = wave_size == 64 ? 0xf0f0'0002'ffff'0001ull : 0x8fff'0001ull;
      const uint32_t mode_bits = mode(0, 3);
      const MulF32Policy p = policy_for(target, mode_bits);
      f.cu->write_sgpr(wf->sgpr_alloc().base + kSgpr, kScalar);
      f.cu->write_sgpr(wf->sgpr_alloc().base + kHighSgpr, kScalar);

      // Source selectors: 1 and 2 name v1 and v2; the others name scalar values.
      constexpr uint32_t kScalarSource = 100;
      constexpr uint32_t kHalfSource = 101;
      struct Case {
        std::string name;
        std::array<uint32_t, 3> words;
        uint32_t dst;
        uint32_t lhs;
        uint32_t rhs;
        uint32_t abs = 0;
        uint32_t neg = 0;
      };
      std::vector<Case> cases = {
          {"sgpr src0", {vop2(target.vop2_opcode, 3, 2, kSgpr), 0, 0}, 3, kScalarSource, 2},
          {"sgpr src0 above vgpr allocation",
           {vop2(target.vop2_opcode, 3, 2, kHighSgpr), 0, 0},
           3,
           kScalarSource,
           2},
          {"literal src0",
           {vop2(target.vop2_opcode, 3, 2, kLiteral), kScalar, 0},
           3,
           kScalarSource,
           2},
          {"inline src0", {vop2(target.vop2_opcode, 3, 2, kInlineHalf), 0, 0}, 3, kHalfSource, 2},
          {"vdst is src0", {vop2(target.vop2_opcode, 1, 2, vgpr(1)), 0, 0}, 1, 1, 2},
          {"vdst is vsrc1", {vop2(target.vop2_opcode, 2, 2, vgpr(1)), 0, 0}, 2, 1, 2},
          {"vdst is both sources", {vop2(target.vop2_opcode, 1, 1, vgpr(1)), 0, 0}, 1, 1, 1},
      };
      {
        const auto w = vop3(target.vop3_opcode, 3, vgpr(1), kSgpr);
        cases.push_back({"vop3 sgpr src1", {w[0], w[1], 0}, 3, 1, kScalarSource});
      }
      {
        const auto w = vop3(target.vop3_opcode, 3, vgpr(1), kHighSgpr);
        cases.push_back(
            {"vop3 sgpr src1 above vgpr allocation", {w[0], w[1], 0}, 3, 1, kScalarSource});
      }
      {
        const auto w = vop3(target.vop3_opcode, 3, vgpr(1), kLiteral);
        cases.push_back({"vop3 literal src1", {w[0], w[1], kScalar}, 3, 1, kScalarSource});
      }
      for (uint32_t abs = 0; abs < 4; ++abs) {
        for (uint32_t neg = 0; neg < 4; ++neg) {
          const auto w = vop3(target.vop3_opcode, 2, vgpr(1), vgpr(2), {.abs = abs, .neg = neg});
          cases.push_back({"vop3 abs " + std::to_string(abs) + " neg " + std::to_string(neg),
                           {w[0], w[1], 0},
                           2,
                           1,
                           2,
                           abs,
                           neg});
        }
      }

      for (const Case &c : cases) {
        SCOPED_TRACE(testing::Message() << target.name << " wave" << wave_size << " " << c.name);
        std::array<uint32_t, 64> lhs_in{};
        std::array<uint32_t, 64> rhs_in{};
        for (uint32_t lane = 0; lane < wave_size; ++lane) {
          // Inactive lanes multiply zero by infinity.
          const auto [a, b] = (exec >> lane) & 1 ? lane_pair(lane) : kLanePairs[5];
          f.cu->write_vgpr(wf->vgpr_alloc().base + 1, lane, a);
          f.cu->write_vgpr(wf->vgpr_alloc().base + 2, lane, b);
          f.cu->write_vgpr(wf->vgpr_alloc().base + 3, lane, kUnwritten);
          lhs_in[lane] = a;
          rhs_in[lane] = b;
        }
        auto source = [&](uint32_t selector, const std::array<uint32_t, 64> &v1,
                          const std::array<uint32_t, 64> &v2, uint32_t lane) {
          if (selector == kScalarSource)
            return kScalar;
          if (selector == kHalfSource)
            return 0x3f000000u;
          return selector == 1 ? v1[lane] : v2[lane];
        };
        wf->set_mode_raw(mode_bits);
        wf->set_exec(exec);
        wf->set_trapsts(0);
        const Outcome outcome = run_mul(f, *wf, c.words);
        uint32_t expected_causes = 0;
        for (uint32_t lane = 0; lane < wave_size; ++lane) {
          const bool active = (exec >> lane) & 1;
          const uint32_t a = apply_modifiers(source(c.lhs, lhs_in, rhs_in, lane), c.abs, c.neg, 0);
          const uint32_t b = apply_modifiers(source(c.rhs, lhs_in, rhs_in, lane), c.abs, c.neg, 1);
          const MulF32Result r = multiply_f32(a, b, p);
          if (active)
            expected_causes |= plain_mul_f32_causes(r, p);
          const uint32_t old_dst = c.dst == 1   ? lhs_in[lane]
                                   : c.dst == 2 ? rhs_in[lane]
                                                : kUnwritten;
          EXPECT_EQ(f.vgpr_value(*wf, c.dst, lane), active ? r.bits : old_dst) << "lane " << lane;
        }
        EXPECT_EQ(outcome.pending, expected_causes);
        EXPECT_EQ(outcome.trapsts, expected_causes);
        if (HasFailure())
          return;
      }
    }
  }
}

struct DeclineCase {
  std::string name;
  rj_code_arch_t arch;
  rj_code_target_id_t target;
  std::array<uint32_t, 3> words;
  uint32_t mode_bits = 0;
  amdgpu::MemoryWaitDiagnostics diagnostics = amdgpu::MemoryWaitDiagnostics::Off;
  bool force_scalar = false;
  bool debug_active = false;
  uint32_t vgprs = kVgprs;
};

// Identity DPP16 quad permutation and DPP8 lane selection for v1.
constexpr uint32_t kDpp16Identity = 1u | (0xe4u << 8) | (0xfu << 24) | (0xfu << 28);
constexpr uint32_t kDpp8Identity = 1u | (0xfac688u << 8);

TEST(HwfloatMulF32ExecutionTest, UnqualifiedTargetsFormsAndStatesKeepTheGenericPath) {
  ForceScalarOverride simd(false);
  std::vector<DeclineCase> cases;
  for (const QualifiedTarget &target : kQualifiedTargets) {
    const uint32_t plain = vop2(target.vop2_opcode, 3, 2, vgpr(1));
    const std::string name = target.name;
    cases.push_back(
        {name + " architecture only", target.arch, ROCJITSU_CODE_TARGET_INVALID, {plain, 0, 0}});
    cases.push_back(
        {name + " DX10_CLAMP", target.arch, target.target, {plain, 0, 0}, mode(0, 0, kDx10Clamp)});
    cases.push_back({name + " DPP16",
                     target.arch,
                     target.target,
                     {vop2(target.vop2_opcode, 3, 2, amdgpu::SRC_DPP), kDpp16Identity, 0}});
    cases.push_back({name + " DPP8",
                     target.arch,
                     target.target,
                     {vop2(target.vop2_opcode, 3, 2, amdgpu::SRC_DPP8_FI_0), kDpp8Identity, 0}});
    cases.push_back({name + " destination outside allocation",
                     target.arch,
                     target.target,
                     {vop2(target.vop2_opcode, 200, 2, vgpr(1)), 0, 0},
                     0,
                     amdgpu::MemoryWaitDiagnostics::Off,
                     false,
                     false,
                     8});
    const Vop3Modifiers declined_modifiers[] = {
        {.omod = 1}, {.omod = 3}, {.clamp = 1}, {.op_sel = 1}, {.abs = 4}, {.neg = 4},
    };
    for (const Vop3Modifiers &m : declined_modifiers) {
      const auto w = vop3(target.vop3_opcode, 3, vgpr(1), vgpr(2), m);
      cases.push_back({name + " vop3 abs " + std::to_string(m.abs) + " neg " +
                           std::to_string(m.neg) + " op_sel " + std::to_string(m.op_sel) +
                           " clamp " + std::to_string(m.clamp) + " omod " + std::to_string(m.omod),
                       target.arch,
                       target.target,
                       {w[0], w[1], 0}});
    }
  }
  cases.push_back({"gfx1200",
                   ROCJITSU_CODE_ARCH_RDNA4,
                   ROCJITSU_CODE_TARGET_GFX1200,
                   {vop2(rdna4::kVMulF32Vop2, 3, 2, vgpr(1)), 0, 0}});

  for (const DeclineCase &c : cases) {
    SCOPED_TRACE(c.name);
    ForceScalarOverride scalar(c.force_scalar);
    MulFixture f(c.arch, c.target, c.diagnostics);
    f.cu->set_debug_active(c.debug_active);
    amdgpu::Wavefront *wf = f.dispatch(32, c.vgprs);
    ASSERT_NE(wf, nullptr);
    f.fill_vgpr(*wf, 1, 0x00000000u);
    f.fill_vgpr(*wf, 2, 0x7f800000u);
    wf->set_mode_raw(c.mode_bits);
    wf->set_exec(0x0000'ff01u);
    wf->set_trapsts(0);
    const Outcome outcome = run_mul(f, *wf, c.words);
    EXPECT_EQ(outcome.pending & cause::kInvalid, 0u);
    EXPECT_EQ(outcome.trapsts & cause::kInvalid, 0u);
  }
}

class VgprAccessRecorder : public ExecutionPlugin {
public:
  VgprAccessRecorder() : ExecutionPlugin("hwfloat_vgpr_access") {}

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t) override {
    reads.emplace_back(physical_reg, lane_mask);
    events.push_back({false, wf->pending_alu_causes(), wf->trapsts(), std::fegetround(),
                      std::fetestexcept(FE_ALL_EXCEPT), errno});
  }

  void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                              uint64_t lane_mask, uint8_t) override {
    writes.emplace_back(physical_reg, lane_mask);
    events.push_back({true, wf->pending_alu_causes(), wf->trapsts(), std::fegetround(),
                      std::fetestexcept(FE_ALL_EXCEPT), errno});
  }

  struct Event {
    bool write;
    uint32_t pending;
    uint32_t sticky;
    int host_round;
    int host_flags;
    int error_number;
  };
  std::vector<Event> events;

  std::vector<std::pair<uint32_t, uint64_t>> reads;
  std::vector<std::pair<uint32_t, uint64_t>> writes;
};

TEST(HwfloatMulF32ExecutionTest, RegisterObserversKeepReadAndDeliveryOrder) {
  ForceScalarOverride simd(false);
  for (const QualifiedTarget &target : kQualifiedTargets) {
    SCOPED_TRACE(target.name);
    MulFixture f(target.arch, target.target);
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<VgprAccessRecorder>();
    VgprAccessRecorder *recorder = plugin.get();
    ASSERT_TRUE(group->add(std::move(plugin)));
    f.cu->set_plugin_group(group);
    group->onInit();

    amdgpu::Wavefront *wf = f.dispatch(32);
    ASSERT_NE(wf, nullptr);
    f.fill_vgpr(*wf, 1, 0x00000000u);
    f.fill_vgpr(*wf, 2, 0x7f800000u);
    wf->set_mode_raw(mode(0, 0));
    constexpr uint64_t kExec = 0x0000'ff01u;
    wf->set_exec(kExec);
    wf->set_trapsts(cause::kInexact);
    recorder->reads.clear();
    recorder->writes.clear();
    recorder->events.clear();
    const Outcome outcome = run_mul(f, *wf, vop2(target.vop2_opcode, 3, 2, vgpr(1)));
    EXPECT_EQ(outcome.pending, cause::kInvalid);
    EXPECT_EQ(outcome.trapsts, cause::kInvalid | cause::kInexact);
    // The classifier retains its two original observed source-view reads.
    // Only then are current causes raised and sticky status updated; result
    // reads and every destination write observe that completed update.
    ASSERT_GT(recorder->events.size(), 2u);
    for (size_t i = 0; i < recorder->events.size(); ++i) {
      const auto &event = recorder->events[i];
      if (i < 2) {
        EXPECT_FALSE(event.write);
        EXPECT_EQ(event.pending, 0u);
        EXPECT_EQ(event.sticky, cause::kInexact);
      } else {
        EXPECT_EQ(event.pending, cause::kInvalid);
        EXPECT_EQ(event.sticky, cause::kInvalid | cause::kInexact);
      }
    }

    const uint32_t base = wf->vgpr_alloc().base;
    uint64_t v1_lanes = 0;
    uint64_t v2_lanes = 0;
    uint64_t v3_lanes = 0;
    for (const auto &[reg, lanes] : recorder->reads) {
      if (reg == base + 1)
        v1_lanes |= lanes;
      else if (reg == base + 2)
        v2_lanes |= lanes;
    }
    for (const auto &[reg, lanes] : recorder->writes)
      if (reg == base + 3)
        v3_lanes |= lanes;
    EXPECT_EQ(v1_lanes, kExec);
    EXPECT_EQ(v2_lanes, kExec);
    EXPECT_EQ(v3_lanes, kExec);
    group->onShutdown();
  }
}

TEST(HwfloatMulF32ExecutionTest, ScalarAndDebugPlainCausesMatchPhysicalWitnesses) {
  struct Witness {
    uint32_t lhs, rhs, mode_bits, result, causes;
  };
  // Direct gfx1100/gfx1201 captures; the signaling NaN result is quieted on
  // gfx1201, while the status cause is identical on both concrete targets.
  constexpr Witness witnesses[] = {
      {0x7f800000, 0x00000000, 0xf0, 0xffc00000, 0x01},
      {0x00800000, 0x3f000000, 0xf0, 0x00400000, 0x00},
      {0x00800000, 0x3f7fffff, 0xf0, 0x00800000, 0x30},
      {0x7f7fffff, 0x40000000, 0xf3, 0x7f7fffff, 0x28},
      {0x7fc12345, 0x00000001, 0xf0, 0x7fc12345, 0x00},
      {0x7f812345, 0x00000001, 0xf0, 0x7f812345, 0x01},
      {0x00000001, 0x3f800000, 0xc0, 0x00000000, 0x00},
  };
  for (const QualifiedTarget &target : kQualifiedTargets) {
    constexpr uint32_t wave_size = 64;
    for (uint32_t control : {0u, 1u, 2u}) {
      SCOPED_TRACE(target.name);
      SCOPED_TRACE(wave_size);
      SCOPED_TRACE(control);
      ForceScalarOverride scalar(control == 0);
      MulFixture f(target.arch, target.target,
                   control == 2 ? amdgpu::MemoryWaitDiagnostics::Warn
                                : amdgpu::MemoryWaitDiagnostics::Off);
      f.cu->set_debug_active(control == 1);
      auto *wf = f.dispatch(wave_size);
      ASSERT_NE(wf, nullptr);
      const auto w3 = vop3(target.vop3_opcode, 3, vgpr(1), vgpr(2));
      const std::array<uint32_t, 3> forms[] = {{vop2(target.vop2_opcode, 3, 2, vgpr(1)), 0, 0},
                                               {w3[0], w3[1], 0}};
      for (const auto &words : forms) {
        auto inst = f.decode(words);
        ASSERT_NE(inst, nullptr);
        for (const auto &witness : witnesses) {
          f.fill_vgpr(*wf, 1, witness.lhs);
          f.fill_vgpr(*wf, 2, witness.rhs);
          f.fill_vgpr(*wf, 3, kUnwritten);
          wf->set_mode_raw(witness.mode_bits);
          const uint64_t exec = 1u | (uint64_t{1} << (wave_size - 1));
          wf->set_exec(exec);
          // A previously latched sticky cause must not suppress this occurrence.
          wf->set_trapsts(0x40u | witness.causes);
          wf->clear_pending_alu_causes();
          const SavedHostEnvironment saved;
          ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
          std::feclearexcept(FE_ALL_EXCEPT);
          std::feraiseexcept(FE_OVERFLOW | FE_INEXACT);
#if defined(__x86_64__)
          _mm_setcsr(_mm_getcsr() | (1u << 6) | (1u << 15));
          const uint32_t expected_mxcsr = _mm_getcsr();
#endif
          errno = EILSEQ;
          const bool succeeded = f.cu->execute_instruction(inst.get(), *wf).succeeded();
          const int actual_errno = errno;
          const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
          const int actual_rounding = std::fegetround();
#if defined(__x86_64__)
          const uint32_t actual_mxcsr = _mm_getcsr();
#endif
          saved.restore();
          ASSERT_TRUE(succeeded);
          EXPECT_EQ(actual_errno, EILSEQ);
          EXPECT_EQ(actual_flags, FE_OVERFLOW | FE_INEXACT);
          EXPECT_EQ(actual_rounding, FE_UPWARD);
#if defined(__x86_64__)
          EXPECT_EQ(actual_mxcsr, expected_mxcsr);
#endif
          EXPECT_EQ(wf->pending_alu_causes(), witness.causes);
          EXPECT_EQ(wf->trapsts(), 0x40u | witness.causes);
          uint32_t expected = witness.result;
          if (target.arch == ROCJITSU_CODE_ARCH_RDNA4 && witness.lhs == 0x7f812345)
            expected |= 0x00400000;
          for (uint32_t lane = 0; lane < wave_size; ++lane) {
            EXPECT_EQ(f.vgpr_value(*wf, 3, lane),
                      (exec & (uint64_t{1} << lane)) ? expected : kUnwritten);
          }
          if (HasFailure())
            return;
        }
      }
    }
  }
}

TEST(HwfloatMulF32ExecutionTest, EmptyExecHasNoEffect) {
  ForceScalarOverride simd(false);
  for (const QualifiedTarget &target : kQualifiedTargets) {
    SCOPED_TRACE(target.name);
    MulFixture f(target.arch, target.target);
    amdgpu::Wavefront *wf = f.dispatch(32);
    ASSERT_NE(wf, nullptr);
    f.fill_vgpr(*wf, 1, 0x00000000u);
    f.fill_vgpr(*wf, 2, 0x7f800000u);
    f.fill_vgpr(*wf, 3, kUnwritten);
    wf->set_mode_raw(mode(0, 0));
    wf->set_exec(0);
    wf->set_trapsts(cause::kInexact);
    const Outcome outcome = run_mul(f, *wf, vop2(target.vop2_opcode, 3, 2, vgpr(1)));
    EXPECT_EQ(outcome.pending, 0u);
    EXPECT_EQ(outcome.trapsts, cause::kInexact);
    for (uint32_t lane = 0; lane < 32; ++lane) {
      EXPECT_EQ(f.vgpr_value(*wf, 3, lane), kUnwritten);
    }
  }
}

void set_trap_enables(amdgpu::Wavefront &wf, rj_code_arch_t arch, uint32_t causes) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    wf.set_gfx12_trap_ctrl_raw(causes);
  else
    wf.set_mode_raw(causes << 12);
}

// Delivery depends on the causes raised by this instruction, not on whether
// the sticky TRAPSTS bit changes.
TEST(HwfloatMulF32ExecutionTest, TrapDeliveryUsesCurrentCausesWithStickyStatus) {
  ForceScalarOverride simd(false);
  constexpr uint32_t kHuge = 0x7e967699u; // 1e38f
  constexpr uint32_t kOverflowInexact = cause::kOverflow | cause::kInexact;
  for (const QualifiedTarget &target : kQualifiedTargets) {
    SCOPED_TRACE(target.name);
    MulFixture f(target.arch, target.target);
    const uint32_t square_v0 = vop2(target.vop2_opcode, 0, 0, vgpr(0));
    const uint32_t program[] = {square_v0, square_v0, square_v0,
                                vop2(target.vop2_opcode, 1, 2, vgpr(2)), kSEndpgm};
    for (uint32_t i = 0; i < std::size(program); ++i)
      f.gpu_mem.write32(kKernelAddr + 4 * i, program[i]);
    amdgpu::Wavefront *wf = f.dispatch(32, kVgprs, kKernelAddr);
    ASSERT_NE(wf, nullptr);
    wf->set_mode_raw(mode(0, 0));
    wf->set_exec(1);
    const uint32_t base = wf->vgpr_alloc().base;
    f.cu->write_vgpr(base + 2, 0, 0x3fc00000u);

    uint32_t handler_calls = 0;
    f.cu->set_alu_exception_handler([&](amdgpu::Wavefront &) {
      ++handler_calls;
      return false;
    });

    // Enabled overflow with the sticky bit already latched.
    wf->set_trapsts(cause::kOverflow);
    set_trap_enables(*wf, target.arch, cause::kOverflow);
    f.cu->write_vgpr(base, 0, kHuge);
    f.cu->step();
    EXPECT_EQ(handler_calls, 1u);
    EXPECT_EQ(wf->pending_alu_causes(), kOverflowInexact);
    EXPECT_EQ(wf->trapsts() & 0x7fu, kOverflowInexact);
    EXPECT_EQ(f.cu->read_vgpr_storage(base, 0), 0x7f800000u);

    // Disabled occurrence: sticky only.
    set_trap_enables(*wf, target.arch, 0);
    f.cu->write_vgpr(base, 0, kHuge);
    f.cu->step();
    EXPECT_EQ(handler_calls, 1u);
    EXPECT_EQ(wf->pending_alu_causes(), kOverflowInexact);
    EXPECT_EQ(wf->trapsts() & 0x7fu, kOverflowInexact);

    // Enabling after a disabled occurrence delivers the next occurrence.
    set_trap_enables(*wf, target.arch, cause::kOverflow);
    f.cu->write_vgpr(base, 0, kHuge);
    f.cu->step();
    EXPECT_EQ(handler_calls, 2u);

    // A latched, enabled cause that this instruction does not raise is not an event.
    set_trap_enables(*wf, target.arch, cause::kInexact);
    f.cu->step();
    EXPECT_EQ(handler_calls, 2u);
    EXPECT_EQ(wf->pending_alu_causes(), 0u);
    EXPECT_EQ(wf->trapsts() & 0x7fu, kOverflowInexact);
    EXPECT_EQ(f.cu->read_vgpr_storage(base + 1, 0), 0x40100000u);
  }
}

TEST(HwfloatMulF32ExecutionTest, HostFloatingPointStateDoesNotAffectResults) {
  ForceScalarOverride simd(false);
  for (const QualifiedTarget &target : kQualifiedTargets) {
    SCOPED_TRACE(target.name);
    MulFixture f(target.arch, target.target);
    amdgpu::Wavefront *wf = f.dispatch(32);
    ASSERT_NE(wf, nullptr);
    for (uint32_t lane = 0; lane < 32; ++lane) {
      f.cu->write_vgpr(wf->vgpr_alloc().base + 1, lane, lane_pair(lane).first);
      f.cu->write_vgpr(wf->vgpr_alloc().base + 2, lane, lane_pair(lane).second);
    }
    const uint32_t mode_bits = mode(0, 3);
    wf->set_mode_raw(mode_bits);
    wf->set_exec(0xffff'ffffu);
    wf->set_trapsts(0);
    std::unique_ptr<Instruction> inst =
        f.decode(std::array<uint32_t, 1>{vop2(target.vop2_opcode, 3, 2, vgpr(1))});
    ASSERT_NE(inst, nullptr);

    const SavedHostEnvironment saved;
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
#if defined(__x86_64__)
    _mm_setcsr(_mm_getcsr() | (1u << 6) | (1u << 15)); // DAZ and FTZ
#endif
    std::feclearexcept(FE_ALL_EXCEPT);
#if defined(__x86_64__)
    const uint32_t hostile_mxcsr = _mm_getcsr();
#endif
    errno = EDOM;
    wf->clear_pending_alu_causes();
    const bool succeeded = f.cu->execute_instruction(inst.get(), *wf).succeeded();
    const int error_number = errno;
    const int flags = std::fetestexcept(FE_ALL_EXCEPT);
    const int rounding = std::fegetround();
#if defined(__x86_64__)
    const uint32_t mxcsr = _mm_getcsr();
#endif
    saved.restore();

    EXPECT_TRUE(succeeded);
    EXPECT_EQ(error_number, EDOM);
    EXPECT_EQ(flags, 0);
    EXPECT_EQ(rounding, FE_UPWARD);
#if defined(__x86_64__)
    EXPECT_EQ(mxcsr, hostile_mxcsr);
#endif
    const MulF32Policy p = policy_for(target, mode_bits);
    uint32_t expected_causes = 0;
    for (uint32_t lane = 0; lane < 32; ++lane) {
      const MulF32Result r = multiply_f32(lane_pair(lane).first, lane_pair(lane).second, p);
      expected_causes |= plain_mul_f32_causes(r, p);
      EXPECT_EQ(f.vgpr_value(*wf, 3, lane), r.bits) << "lane " << lane;
    }
    EXPECT_EQ(wf->pending_alu_causes(), expected_causes);
    EXPECT_EQ(wf->trapsts(), expected_causes);
  }
}

// Directed physical gfx1100/gfx1201 DX9 captures. Expected bits and flags are
// literal observations, independent of the new arithmetic implementation.
struct Dx9Witness {
  uint32_t a, b, mode, omod, clamp, rdna3_bits, rdna4_bits, rdna3_causes, rdna4_causes;
};
constexpr Dx9Witness kDx9Witnesses[] = {
    {0x3f800001, 0x3f800003, 0xc0, 0, 0, 0x3f800004, 0x3f800004, 0x20, 0x20},
    {0x3f800001, 0x3f800003, 0xc1, 0, 0, 0x3f800005, 0x3f800005, 0x20, 0x20},
    {0x3f800001, 0x3f800003, 0xc2, 0, 0, 0x3f800004, 0x3f800004, 0x20, 0x20},
    {0x3f800001, 0x3f800003, 0xc3, 0, 0, 0x3f800004, 0x3f800004, 0x20, 0x20},
    {0x00000001, 0x4b000000, 0xc0, 0, 0, 0x00000000, 0x00000000, 0x00, 0x00},
    {0x00000001, 0x4b000000, 0xd0, 0, 0, 0x00800000, 0x00800000, 0x02, 0x02},
    {0x00000001, 0x4b000000, 0xe0, 0, 0, 0x00000000, 0x00000000, 0x00, 0x00},
    {0x00000001, 0x4b000000, 0xf0, 0, 0, 0x00800000, 0x00800000, 0x02, 0x02},
    {0x00800000, 0x3f7fffff, 0xc0, 0, 0, 0x00000000, 0x00000000, 0x30, 0x30},
    {0x80800000, 0x3f7fffff, 0xc0, 0, 0, 0x80000000, 0x80000000, 0x30, 0x30},
    {0x00800000, 0x3f7fffff, 0xf0, 0, 0, 0x00800000, 0x00800000, 0x30, 0x30},
    {0x00000000, 0x7f800000, 0xc0, 0, 0, 0x00000000, 0x00000000, 0x00, 0x00},
    {0x00000000, 0x7f812345, 0xc0, 0, 0, 0x00000000, 0x00000000, 0x01, 0x01},
    {0x7f812345, 0x3f800000, 0xc0, 0, 0, 0x7f812345, 0x7fc12345, 0x01, 0x01},
    {0x7f812345, 0x3f800000, 0x2c0, 0, 0, 0x7fc12345, 0x7fc12345, 0x01, 0x01},
    {0x7f812345, 0x3f800000, 0x1c0, 0, 0, 0x7f812345, 0x7fc12345, 0x00, 0x01},
    {0x00000000, 0x00000001, 0xd0, 0, 0, 0x00000000, 0x00000000, 0x02, 0x02},
    {0x7f812345, 0x00000001, 0xd0, 0, 0, 0x7f812345, 0x7fc12345, 0x01, 0x01},
    {0x00800000, 0x3f7fffff, 0xf0, 1, 0, 0x00800000, 0x00000000, 0x00, 0x00},
    {0x7f7fffff, 0x40000000, 0xc0, 1, 0, 0x7f800000, 0x7f800000, 0x00, 0x08},
    {0x3f800001, 0x3f800003, 0xc0, 1, 0, 0x40000004, 0x40000004, 0x00, 0x00},
    {0x7f812345, 0x3f800000, 0xc0, 0, 1, 0x7f812345, 0x00000000, 0x00, 0x00},
    {0x7f812345, 0x3f800000, 0x1c0, 0, 1, 0x00000000, 0x00000000, 0x00, 0x00},
};

TEST(HwfloatDx9MulF32ExecutionTest, PhysicalResultAndStickyWitnesses) {
  SavedHostEnvironment host;
  for (const auto &target : kQualifiedTargets) {
    for (uint32_t lanes : {32u, 64u}) {
      MulFixture f(target.arch, target.target);
      auto *wf = f.dispatch(lanes);
      ASSERT_NE(wf, nullptr);
      for (bool force_scalar : {false, true}) {
        ForceScalarOverride scalar(force_scalar);
        for (bool encoded_vop3 : {false, true}) {
          for (const auto &w : kDx9Witnesses) {
            if (!encoded_vop3 && (w.omod || w.clamp))
              continue;
            SCOPED_TRACE(target.name);
            SCOPED_TRACE(w.mode);
            const bool is_rdna4 = target.arch == ROCJITSU_CODE_ARCH_RDNA4;
            const uint32_t expected = is_rdna4 ? w.rdna4_bits : w.rdna3_bits;
            const uint32_t causes = is_rdna4 ? w.rdna4_causes : w.rdna3_causes;
            for (uint64_t exec : {uint64_t{0}, uint64_t{5}, uint64_t{1} << (lanes - 1)}) {
              // Alias destination with lhs; inactive lanes must remain untouched.
              f.fill_vgpr(*wf, 1, w.a);
              f.fill_vgpr(*wf, 2, w.b);
              wf->set_mode_raw(w.mode);
              wf->set_exec(exec);
              wf->set_trapsts(0x40);
              std::fesetround(FE_DOWNWARD);
              std::feclearexcept(FE_ALL_EXCEPT);
              std::feraiseexcept(FE_DIVBYZERO);
              errno = EILSEQ;
              Outcome outcome;
              if (encoded_vop3) {
                const auto opcode =
                    is_rdna4 ? rdna4::kVMulDx9ZeroF32Vop3 : rdna3::kVMulDx9ZeroF32Vop3;
                outcome = run_mul(
                    f, *wf, vop3(opcode, 1, vgpr(1), vgpr(2), {.clamp = w.clamp, .omod = w.omod}));
              } else {
                const auto opcode =
                    is_rdna4 ? rdna4::kVMulDx9ZeroF32Vop2 : rdna3::kVMulDx9ZeroF32Vop2;
                outcome = run_mul(f, *wf, vop2(opcode, 1, 2, vgpr(1)));
              }
              const int error = errno;
              const int rounding = std::fegetround();
              const int flags = std::fetestexcept(FE_ALL_EXCEPT);
              EXPECT_EQ(error, EILSEQ);
              EXPECT_EQ(rounding, FE_DOWNWARD);
              EXPECT_EQ(flags, FE_DIVBYZERO);
              EXPECT_EQ(outcome.pending, exec ? causes : 0u);
              EXPECT_EQ(outcome.trapsts, 0x40u | (exec ? causes : 0u));
              for (uint32_t lane = 0; lane < lanes; ++lane) {
                EXPECT_EQ(f.vgpr_value(*wf, 1, lane),
                          exec & (uint64_t{1} << lane) ? expected : w.a);
              }
            }
          }
        }
      }
    }
  }
}

TEST(HwfloatDx9MulF32ExecutionTest, ObserversRetainPerLaneReadsWritesAndRepeatedCauses) {
  SavedHostEnvironment host;
  for (const auto &target : kQualifiedTargets) {
    MulFixture f(target.arch, target.target);
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<VgprAccessRecorder>();
    auto *recorder = plugin.get();
    ASSERT_TRUE(group->add(std::move(plugin)));
    f.cu->set_plugin_group(group);
    group->onInit();
    auto *wf = f.dispatch(64);
    ASSERT_NE(wf, nullptr);
    f.fill_vgpr(*wf, 1, 0);
    f.fill_vgpr(*wf, 2, 0x7f812345);
    wf->set_mode_raw(0xc0);
    constexpr uint64_t exec = 5ull | (1ull << 63);
    wf->set_exec(exec);
    wf->set_trapsts(0x41);
    const bool is_rdna4 = target.arch == ROCJITSU_CODE_ARCH_RDNA4;
    for (bool encoded_vop3 : {false, true}) {
      for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        recorder->events.clear();
        recorder->reads.clear();
        recorder->writes.clear();
        std::fesetround(FE_UPWARD);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_OVERFLOW);
        errno = EILSEQ;
        Outcome outcome;
        if (encoded_vop3) {
          const auto opcode = is_rdna4 ? rdna4::kVMulDx9ZeroF32Vop3 : rdna3::kVMulDx9ZeroF32Vop3;
          outcome = run_mul(f, *wf, vop3(opcode, 3, vgpr(1), vgpr(2)));
        } else {
          const auto opcode = is_rdna4 ? rdna4::kVMulDx9ZeroF32Vop2 : rdna3::kVMulDx9ZeroF32Vop2;
          outcome = run_mul(f, *wf, vop2(opcode, 3, 2, vgpr(1)));
        }
        EXPECT_EQ(outcome.pending, 1u);
        EXPECT_EQ(outcome.trapsts, 0x41u);
        ASSERT_EQ(recorder->events.size(), 9u);
        for (size_t i = 0; i < recorder->events.size(); ++i) {
          const bool write = i % 3 == 2;
          EXPECT_EQ(recorder->events[i].write, write);
          EXPECT_EQ(recorder->events[i].pending, i < 2 ? 0u : 1u);
          EXPECT_EQ(recorder->events[i].host_round,
                    encoded_vop3 && !write ? FE_TONEAREST : FE_UPWARD);
          EXPECT_EQ(recorder->events[i].host_flags, encoded_vop3 && !write ? 0 : FE_OVERFLOW);
          EXPECT_EQ(recorder->events[i].error_number, EILSEQ);
        }
        ASSERT_EQ(recorder->reads.size(), 6u);
        ASSERT_EQ(recorder->writes.size(), 3u);
        for (size_t i = 0; i < 3; ++i) {
          const uint64_t lane_mask = uint64_t{1} << (i == 0 ? 0 : i == 1 ? 2 : 63);
          EXPECT_EQ(recorder->reads[2 * i].second, lane_mask);
          EXPECT_EQ(recorder->reads[2 * i + 1].second, lane_mask);
          EXPECT_EQ(recorder->writes[i].second, lane_mask);
        }
      }
    }
    group->onShutdown();
  }
}

TEST(HwfloatDx9MulF32ExecutionTest, ArchitectureOnlyTargetRetainsUnqualifiedPolicy) {
  for (const auto &target : kQualifiedTargets) {
    MulFixture f(target.arch, ROCJITSU_CODE_TARGET_INVALID);
    auto *wf = f.dispatch(32);
    ASSERT_NE(wf, nullptr);
    f.fill_vgpr(*wf, 1, 0x00800000);
    f.fill_vgpr(*wf, 2, 0x3f7fffff);
    wf->set_mode_raw(0xc0);
    wf->set_exec(1);
    const auto opcode = target.arch == ROCJITSU_CODE_ARCH_RDNA4 ? rdna4::kVMulDx9ZeroF32Vop2
                                                                : rdna3::kVMulDx9ZeroF32Vop2;
    const auto outcome = run_mul(f, *wf, vop2(opcode, 3, 2, vgpr(1)));
    EXPECT_EQ(outcome.pending, 0u);
    EXPECT_EQ(f.vgpr_value(*wf, 3, 0), 0x00800000u);
  }
}

} // namespace
