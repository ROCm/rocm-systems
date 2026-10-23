// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

namespace {
using namespace rocjitsu;

enum class SourceKind { Sgpr, Ttmp, Vcc, Literal, Inline, M0, Scc, Null };
using Parameters = std::tuple<unsigned, SourceKind, bool>;

constexpr std::array<uint16_t, 3> kOpcodes = {cdna5::kVPkAddF32Vop3p, cdna5::kVPkMulF32Vop3p,
                                              cdna5::kVPkFmaF32Vop3p};
constexpr std::array<const char *, 3> kOperationNames = {"Add", "Mul", "Fma"};
constexpr std::array<const char *, 8> kSourceNames = {"Sgpr",   "Ttmp", "Vcc", "Literal",
                                                      "Inline", "M0",   "Scc", "Null"};
constexpr std::array<uint16_t, 8> kSelectors = {8,
                                                110,
                                                cdna5::OPR_SRC_VCC_LO,
                                                cdna5::OPR_SRC_SRC_LITERAL,
                                                cdna5::OPR_SRC_FLOAT_NEG_TWO,
                                                cdna5::OPR_SRC_M0,
                                                cdna5::OPR_SRC_SRC_SCC,
                                                cdna5::OPR_SRC_NULL};
constexpr std::array<uint32_t, 8> kScalarBits = {0x3fc00000, 0x40a00000, 0x40800000, 0x3f000000,
                                                 0xc0000000, 0x40500000, 1,          0};
constexpr uint32_t kSentinel = 0xdeadbeef;

uint32_t initial_vgpr(uint32_t reg, uint32_t lane) {
  if (reg >= 6)
    return kSentinel;
  return std::bit_cast<uint32_t>(static_cast<float>(2 + reg + lane % 3));
}

class Cdna5PackedF32ScalarSources : public testing::TestWithParam<Parameters> {
protected:
  void SetUp() override {
    ASSERT_EQ(std::fegetenv(&saved_environment_), 0);
    saved_force_scalar_ = util::force_scalar();
    // Reference computations include SCC=1, the smallest positive FP32 subnormal.
    ASSERT_EQ(std::fesetenv(FE_DFL_ENV), 0);
    if (!std::get<2>(GetParam()) && !util::has_stdx_simd)
      GTEST_SKIP() << "SIMD execution is unavailable";

    amdgpu::ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_CDNA5;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cache_.set_backing_memory(&memory_);
    cu_ = amdgpu::ComputeUnitCore::create("packed_f32_scalar_sources", config, &memory_, &cache_);
    decoder_ = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    wave_ = cu_->dispatch_wf(0, 0, 106, 256);
    ASSERT_NE(wave_, nullptr);
  }

  void TearDown() override {
    if (wave_)
      wave_->halt();
    util::set_force_scalar_for_testing(saved_force_scalar_);
    std::fesetenv(&saved_environment_);
  }

  unsigned source_count() const { return std::get<0>(GetParam()) == 2 ? 3 : 2; }

  void check(unsigned scalar_position, uint8_t lo, uint8_t hi, uint8_t neg, uint8_t neg_hi,
             uint8_t dst = 6, uint64_t exec = ~uint64_t{0}) {
    const auto [operation, source_kind, force_scalar] = GetParam();
    const auto kind = static_cast<unsigned>(source_kind);
    SCOPED_TRACE(testing::Message()
                 << "source_position=" << scalar_position << " lo=" << unsigned(lo)
                 << " hi=" << unsigned(hi) << " neg=" << unsigned(neg)
                 << " neg_hi=" << unsigned(neg_hi) << " dst=" << unsigned(dst) << " exec=" << exec);
    std::array<uint16_t, 3> sources = {256, 258, 260};
    sources[scalar_position] = kSelectors[kind];
    const auto words = cdna5::build_vop3p(kOpcodes[operation], {.vdst = dst,
                                                                .neg_hi = neg_hi,
                                                                .opsel = lo,
                                                                .opsel_hi_2 = uint8_t(hi >> 2),
                                                                .src0 = sources[0],
                                                                .src1 = sources[1],
                                                                .src2 = sources[2],
                                                                .opsel_hi = uint8_t(hi & 3),
                                                                .neg = neg});
    const std::array<uint32_t, 4> encoding = {words[0], words[1], kScalarBits[kind], 0};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder_, encoding.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->size(), source_kind == SourceKind::Literal ? 12 : 8);
    ASSERT_EQ(inst->num_src_operands(), static_cast<int>(source_count()));
    ASSERT_EQ(inst->dst_operand(0)->size_bits(), 64);
    for (unsigned source = 0; source < source_count(); ++source) {
      ASSERT_EQ(inst->src_operand(source)->size_bits(), 64);
      if (!force_scalar) {
        ASSERT_TRUE(inst->src_operand(source)->simd_capable());
      }
    }

    const uint32_t base = wave_->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
      for (uint32_t reg = 0; reg < 8; ++reg)
        cu_->write_vgpr(base + reg, lane, initial_vgpr(reg, lane));
    wave_->debug_write_sgpr(8, kScalarBits[static_cast<unsigned>(SourceKind::Sgpr)]);
    wave_->debug_write_sgpr(9, 0xc0e00000); // -7.0 must never be selected from s9.
    wave_->set_ttmp(2, kScalarBits[static_cast<unsigned>(SourceKind::Ttmp)]);
    wave_->set_ttmp(3, 0x41300000); // 11.0 must never be selected from ttmp3.
    wave_->set_vcc(kScalarBits[static_cast<unsigned>(SourceKind::Vcc)]);
    wave_->set_m0(kScalarBits[static_cast<unsigned>(SourceKind::M0)]);
    wave_->write_scc(true);
    wave_->set_mode_raw(0xf0); // Round to nearest; preserve input/output denormals.
    wave_->set_exec(exec);
    util::set_force_scalar_for_testing(force_scalar);
    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wave_).succeeded());

    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      for (unsigned half = 0; half < 2; ++half) {
        uint32_t expected = initial_vgpr(dst + half, lane);
        if (exec & (uint64_t{1} << lane)) {
          std::array<double, 3> values{};
          for (unsigned source = 0; source < source_count(); ++source) {
            // CDNA5 ISA 7.7.1: a scalar source is replicated; OPSEL applies to VGPRs.
            const unsigned selected_half = ((half ? hi : lo) >> source) & 1;
            const uint32_t bits = source == scalar_position
                                      ? kScalarBits[kind]
                                      : initial_vgpr(2 * source + selected_half, lane);
            values[source] = std::bit_cast<float>(bits);
            if ((half ? neg_hi : neg) & (1u << source))
              values[source] = -values[source];
          }
          // Small binary fractions give exact arithmetic, apart from SCC's tiny
          // contribution to a normal sum. None is a rounding-boundary case.
          // This reference does not invoke the simulator's scalar execution.
          const double result = operation == 0   ? values[0] + values[1]
                                : operation == 1 ? values[0] * values[1]
                                                 : values[0] * values[1] + values[2];
          expected = std::bit_cast<uint32_t>(static_cast<float>(result));
        }
        ASSERT_EQ(cu_->read_vgpr(base + dst + half, lane), expected)
            << "lane=" << lane << " half=" << half;
      }
    }
  }

private:
  std::fenv_t saved_environment_{};
  bool saved_force_scalar_ = false;
  amdgpu::GpuMemory memory_{"packed_f32_scalar_sources_memory"};
  amdgpu::L2Cache cache_{"packed_f32_scalar_sources_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
};

TEST_P(Cdna5PackedF32ScalarSources, AllSourcePositionsAndSelectors) {
  const uint8_t selections = 1u << source_count();
  for (unsigned source = 0; source < source_count(); ++source)
    for (uint8_t lo = 0; lo < selections; ++lo)
      for (uint8_t hi = 0; hi < selections; ++hi)
        // Zero scalar selectors are assembler-supported controls. Raw encodings
        // also exercise the manual's ignored scalar OPSEL bits; assemblers can
        // reject those otherwise redundant high-half selections.
        check(source, lo, hi, 0, 0);
}

TEST_P(Cdna5PackedF32ScalarSources, PerHalfNegationAliasingAndExec) {
  const uint8_t mask = (1u << source_count()) - 1;
  for (unsigned source = 0; source < source_count(); ++source)
    for (uint8_t neg = 0; neg <= mask; ++neg)
      for (uint8_t dst : {0, 2, 4, 6})
        for (uint64_t exec : {uint64_t{0}, uint64_t{0xa5c30f69}, ~uint64_t{0}})
          check(source, 5 & mask, 2 & mask, neg, neg ^ mask, dst, exec);
}

std::string parameter_name(const testing::TestParamInfo<Parameters> &info) {
  const auto [operation, source, force_scalar] = info.param;
  return std::string(kOperationNames[operation]) + kSourceNames[static_cast<unsigned>(source)] +
         (force_scalar ? "Scalar" : "Simd");
}

INSTANTIATE_TEST_SUITE_P(Cdna5, Cdna5PackedF32ScalarSources,
                         testing::Combine(testing::Values(0u, 1u, 2u),
                                          testing::Values(SourceKind::Sgpr, SourceKind::Ttmp,
                                                          SourceKind::Vcc, SourceKind::Literal,
                                                          SourceKind::Inline, SourceKind::M0,
                                                          SourceKind::Scc, SourceKind::Null),
                                          testing::Bool()),
                         parameter_name);
} // namespace
