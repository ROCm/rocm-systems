// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simd_coverage_test.cpp
/// @brief Decoded packed, true16 and compare arithmetic checks with masks and overlaps.

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/isa_family.h"
#include "util/host_target.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using O = cdna5::Operand;
using K = cdna5::OperandType;

class SimdCoverage : public ::testing::Test {
protected:
  const IsaExecutionBackend backend{.operand_backend = O::full_execution_backend()};
  ScopedIsaExecutionBackend backend_scope{&backend};
  GpuMemory memory{"simd_coverage_memory"};
  L2Cache l2{"simd_coverage_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  Wavefront *wf = nullptr;
  bool old_force_scalar = util::force_scalar();
  void SetUp() override {
    util::set_force_scalar_for_testing(false);
    ComputeUnitCore::Config config{};
    config.arch = rocjitsu::IsaFamily::Cdna5;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(
        rocjitsu::test::initialize_compute_unit(cu, "simd_coverage_cu", config, &memory, &l2));
    wf = cu->dispatch_wf(0, 0, 106, 256, 32);
    wf->set_mode_raw(3u << 6);
    wf->set_exec(0xffffffffu);
  }
  void TearDown() override { util::set_force_scalar_for_testing(old_force_scalar); }
  void put(uint32_t reg, uint32_t lane, uint32_t value) {
    cu->write_vgpr(wf->vgpr_alloc().base + reg, lane, value);
  }
  uint32_t get(uint32_t reg, uint32_t lane) {
    return cu->read_vgpr(wf->vgpr_alloc().base + reg, lane);
  }
};

TEST_F(SimdCoverage, DecodedTrue16MovePreservesScalarAndVectorSelectors) {
  auto decoder = Decoder::create(rocjitsu::IsaFamily::Cdna5);
  constexpr uint64_t exec = 0x55555555u;
  wf->set_exec(exec);
  cu->write_sgpr(wf->sgpr_alloc().base + 9, 0xc0003c00u);
  // Scalar register, integer/floating inline constants, literal and both VGPR halves.
  for (uint32_t source : {9u, 129u, 242u, 255u, 256u, 384u}) {
    for (bool high : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "source=" << source << " high=" << high);
      const uint32_t words[] = {0x7e003800u | ((7u | (high ? 128u : 0u)) << 17) | source,
                                0x76544321u};
      std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words));
      ASSERT_NE(instruction, nullptr);
      std::array<uint32_t, 32> expected;
      for (bool scalar : {true, false}) {
        util::set_force_scalar_for_testing(scalar);
        for (uint32_t lane = 0; lane < 32; ++lane) {
          put(0, lane, 0xc0003c00u + lane);
          put(7, lane, 0xabcd1234u);
          put(9, lane, 0xdeadbeefu); // Detect accidental scalar-to-vector reinterpretation.
        }
        ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wf).succeeded());
        for (uint32_t lane = 0; lane < 32; ++lane) {
          if (scalar) {
            expected[lane] = get(7, lane);
          } else {
            EXPECT_EQ(get(7, lane), expected[lane]) << "lane=" << lane;
          }
          if (!(exec & (uint64_t{1} << lane))) {
            EXPECT_EQ(get(7, lane), 0xabcd1234u);
          } else if (source == 9) {
            EXPECT_EQ(get(7, lane), high ? 0x3c001234u : 0xabcd3c00u);
          }
        }
      }
    }
  }
}

class VopdReadRecorder final : public ExecutionPlugin {
public:
  VopdReadRecorder() : ExecutionPlugin("vopd_read_recorder") {}

  void onAmdgpuReadVgprLanes(const Wavefront *wave, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask) override {
    const uint32_t reg = physical_reg - wave->vgpr_alloc().base;
    ASSERT_LT(reg, lanes.size());
    lanes[reg] |= lane_mask;
    bytes[reg] |= byte_mask;
  }

  std::array<uint64_t, 8> lanes{};
  std::array<uint8_t, 8> bytes{};
};

TEST_F(SimdCoverage, DecodedVopdMovAndFloatPairReadsOnlyUsedRegisters) {
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<VopdReadRecorder>();
  auto *recorder = plugin.get();
  group->add(std::move(plugin));
  cu->set_plugin_group(group);
  auto decoder = Decoder::create(rocjitsu::IsaFamily::Cdna5);
  for (uint32_t mov_source : {0u, 4u}) {
    // MOV v6,v0 has unused VSRCX1=5; MOV v6,v4 has unused VSRCX1=0.
    // Both pair with MUL v7,v2,v3, selecting the floating-point VOPD helper.
    const uint32_t words[] = {mov_source == 0 ? 0xca060b00u : 0xca060104u, 0x06060702u};
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words));
    ASSERT_NE(instruction, nullptr);
    for (uint64_t exec : {uint64_t{0}, uint64_t{1}, uint64_t{0x55555555}}) {
      for (bool force_scalar : {true, false}) {
        SCOPED_TRACE(::testing::Message()
                     << "source=" << mov_source << " exec=" << exec << " scalar=" << force_scalar);
        for (uint32_t reg = 0; reg < 8; ++reg) {
          for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
            put(reg, lane, 0x3f800000u);
          }
        }
        wf->set_exec(exec);
        recorder->lanes = {};
        recorder->bytes = {};
        util::set_force_scalar_for_testing(force_scalar);
        ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wf).succeeded());
        for (uint32_t reg = 0; reg < 8; ++reg) {
          const bool used = reg == mov_source || reg == 2 || reg == 3;
          EXPECT_EQ(recorder->lanes[reg], used ? exec : 0) << "v" << reg;
          EXPECT_EQ(recorder->bytes[reg], used && exec != 0 ? 0xf : 0) << "v" << reg;
        }
      }
    }
  }
}

TEST_F(SimdCoverage, DecodedPackedFmacPreservesDirectedZeroSigns) {
  struct ZeroCase {
    uint32_t multiplicand, multiplier, addend;
    bool matching_negative_zero, cancellation;
  };
  constexpr ZeroCase cases[] = {
      {0x3c003c00u, 0x3c003c00u, 0xbc00bc00u, false, true},
      {0xbc00bc00u, 0x3c003c00u, 0x3c003c00u, false, true},
      {0x80008000u, 0x3c003c00u, 0x80008000u, true, false},
      {0x00000000u, 0x3c003c00u, 0x00000000u, false, false},
      {0x00000000u, 0x3c003c00u, 0x80008000u, false, true},
      {0x80008000u, 0x3c003c00u, 0x00000000u, false, true},
  };
  for (auto arch : {rocjitsu::IsaFamily::Cdna5, rocjitsu::IsaFamily::Rdna4}) {
    auto decoder = Decoder::create(arch);
    const uint32_t words[] = {0x780c0300u}; // v_pk_fmac_f16 v6, v0, v1
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words));
    ASSERT_NE(instruction, nullptr);
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    std::unique_ptr<ComputeUnitCore> target;
    ASSERT_NO_FATAL_FAILURE(
        rocjitsu::test::initialize_compute_unit(target, "packed_fmac_zero", config, &memory, &l2));
    auto *wave = target->dispatch_wf(0, 0, 106, 256, 32);
    ASSERT_NE(wave, nullptr);
    for (uint32_t rounding = 0; rounding < 4; ++rounding) {
      wave->set_mode_raw(rounding << 2);
      for (const auto &test : cases) {
        for (bool force_scalar : {true, false}) {
          SCOPED_TRACE(::testing::Message()
                       << "arch=" << arch << " rounding=" << rounding << " scalar=" << force_scalar
                       << " addend=" << test.addend);
          for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
            target->write_vgpr(wave->vgpr_alloc().base, lane, test.multiplicand);
            target->write_vgpr(wave->vgpr_alloc().base + 1, lane, test.multiplier);
            target->write_vgpr(wave->vgpr_alloc().base + 6, lane, test.addend);
          }
          wave->set_exec(0x55555555u);
          util::set_force_scalar_for_testing(force_scalar);
          ASSERT_TRUE(target->execute_instruction(instruction.get(), *wave).succeeded());
          const bool negative = test.matching_negative_zero || (test.cancellation && rounding == 2);
          for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
            EXPECT_EQ(target->read_vgpr(wave->vgpr_alloc().base + 6, lane), lane % 2   ? test.addend
                                                                            : negative ? 0x80008000u
                                                                                       : 0u);
          }
        }
      }
    }
  }
}

struct DecodedCase {
  rocjitsu::IsaFamily arch;
  const char *assembly;
  uint32_t words[4];
};
// Encodings assembled with the local gfx1250-capable LLVM assembler.
const DecodedCase kDecodedCases[] = {
    {rocjitsu::IsaFamily::Cdna4, "v_cmpx_lt_f32_e64 s[8:9], v0, v1", {0xd0510008u, 0x00020300u}},
    {rocjitsu::IsaFamily::Rdna4, "v_div_scale_f32 v6, s8, v0, v1, v2", {0xd6fc0806u, 0x040a0300u}},
    {rocjitsu::IsaFamily::Cdna5,
     "v_pk_add_f16 v6, v0, 1.0 op_sel_hi:[1,0]",
     {0xcc0f4006u, 0x0a01e500u}},
    {rocjitsu::IsaFamily::Cdna5,
     "v_pk_mul_f16 v6, 0x3000, v1 op_sel_hi:[0,1]",
     {0xcc104006u, 0x120202ffu, 0x00003000u}},
    {rocjitsu::IsaFamily::Cdna5, "v_pk_fma_f16 v6, v0, v1, v2", {0xcc0e4006u, 0x1c0a0300u}},
    {rocjitsu::IsaFamily::Cdna5, "v_pk_min_f16 v6, v0, v1", {0xcc1b4006u, 0x1a020300u}},
    {rocjitsu::IsaFamily::Cdna5, "v_cvt_pk_f16_f32 v6, v0, v1", {0xd76f0006u, 0x02020300u}},
    {rocjitsu::IsaFamily::Cdna5, "v_cmpx_lt_f32 v0, v1", {0x7d220300u}},
    {rocjitsu::IsaFamily::Cdna5, "v_cmpx_lt_f32_e64 v0, v1", {0xd491007eu, 0x02020300u}},
    {rocjitsu::IsaFamily::Cdna5, "v_mov_b16 v6.h, v0.h", {0x7f0c3980u}},
    {rocjitsu::IsaFamily::Cdna5,
     "v_mov_b16 v6.h, v0.h quad_perm:[1,0,3,2]",
     {0x7f0c38fau, 0xff00b180u}},
    {rocjitsu::IsaFamily::Cdna5, "v_add_f16 v6.h, v0.h, v1.l", {0x650c0380u}},
    {rocjitsu::IsaFamily::Cdna5, "v_and_b16 v6.h, v0.h, v1.l", {0xd7624806u, 0x02020300u}},
    {rocjitsu::IsaFamily::Cdna5,
     "v_dual_add_f32 v6, v0, v1 :: v_dual_mul_f32 v7, v2, v3",
     {0xc9060300u, 0x06060702u}},
    {rocjitsu::IsaFamily::Cdna5, "v_pk_fmac_f16 v6, v0, v1", {0x780c0300u}},
    {rocjitsu::IsaFamily::Cdna5, "v_cvt_f16_f32 v6.h, v0", {0x7f0c1500u}},
    {rocjitsu::IsaFamily::Cdna5, "v_cmpx_lt_f16_e64 v0.h, v1.l", {0xd481087eu, 0x02020300u}},
    {rocjitsu::IsaFamily::Cdna5, "v_cvt_pk_bf16_f32 v6, v0, v1", {0xd76d0006u, 0x02020300u}},
    {rocjitsu::IsaFamily::Cdna5,
     "v_bitop3_b32 v6, v0, v1, v2 bitop3:0x96",
     {0xd6340206u, 0xd40a0300u}},
    {rocjitsu::IsaFamily::Cdna5, "v_pk_add_bf16 v6, v0, v1", {0xcc234006u, 0x1a020300u}},
    {rocjitsu::IsaFamily::Cdna5, "v_pk_fma_bf16 v6, v0, v1, v2", {0xcc114006u, 0x1c0a0300u}},
    {rocjitsu::IsaFamily::Cdna5, "v_cvt_f32_bf16 v6, v0.h", {0x7e0ce580u}},
    {rocjitsu::IsaFamily::Cdna5, "v_div_scale_f32 v6, s8, v0, v1, v2", {0xd6fc0806u, 0x040a0300u}},
    {rocjitsu::IsaFamily::Cdna4,
     "v_pk_add_f16 v6, v0, 1.0 op_sel_hi:[1,0]",
     {0xd38f4006u, 0x0801e500u}},
    {rocjitsu::IsaFamily::Cdna4, "v_pk_fma_f16 v6, v0, v1, v2", {0xd38e4006u, 0x1c0a0300u}},
    {rocjitsu::IsaFamily::Cdna4, "v_pk_min_f16 v6, v0, v1", {0xd3914006u, 0x18020300u}},
    {rocjitsu::IsaFamily::Cdna4, "v_cvt_pk_f16_f32 v6, v0, v1", {0xd2670006u, 0x00020300u}},
    {rocjitsu::IsaFamily::Cdna4, "v_cmpx_lt_f32 v0, v1", {0x7ca20300u}},
    {rocjitsu::IsaFamily::Cdna4, "v_cvt_pk_bf16_f32 v6, v0, v1", {0xd2680006u, 0x00020300u}},
    {rocjitsu::IsaFamily::Cdna4,
     "v_bitop3_b32 v6, v0, v1, v2 bitop3:0x96",
     {0xd2340206u, 0xd40a0300u}},
    {rocjitsu::IsaFamily::Cdna4,
     "v_div_scale_f32 v6, s[8:9], v0, v1, v2",
     {0xd1e00806u, 0x040a0300u}},
    {rocjitsu::IsaFamily::Rdna4,
     "v_pk_add_f16 v6, v0, 1.0 op_sel_hi:[1,0]",
     {0xcc0f4006u, 0x0a01e500u}},
    {rocjitsu::IsaFamily::Rdna4,
     "v_pk_mul_f16 v6, 0x3000, v1 op_sel_hi:[0,1]",
     {0xcc104006u, 0x120202ffu, 0x00003000u}},
    {rocjitsu::IsaFamily::Rdna4, "v_pk_fma_f16 v6, v0, v1, v2", {0xcc0e4006u, 0x1c0a0300u}},
    {rocjitsu::IsaFamily::Rdna4, "v_pk_min_f16 v6, v0, v1", {0xcc1b4006u, 0x1a020300u}},
    {rocjitsu::IsaFamily::Rdna4, "v_cmpx_lt_f32 v0, v1", {0x7d220300u}},
    {rocjitsu::IsaFamily::Rdna4, "v_cmpx_lt_f32_e64 v0, v1", {0xd491007eu, 0x02020300u}},
    {rocjitsu::IsaFamily::Rdna4, "v_mov_b16 v6.h, v0.h", {0x7f0c3980u}},
    {rocjitsu::IsaFamily::Rdna4,
     "v_mov_b16 v6.h, v0.h quad_perm:[1,0,3,2]",
     {0x7f0c38fau, 0xff00b180u}},
    {rocjitsu::IsaFamily::Rdna4, "v_add_f16 v6.h, v0.h, v1.l", {0x650c0380u}},
    {rocjitsu::IsaFamily::Rdna4, "v_and_b16 v6.h, v0.h, v1.l", {0xd7624806u, 0x02020300u}},
    {rocjitsu::IsaFamily::Rdna4,
     "v_dual_add_f32 v6, v0, v1 :: v_dual_mul_f32 v7, v2, v3",
     {0xc9060300u, 0x06060702u}},
    {rocjitsu::IsaFamily::Rdna4,
     "v_dual_cndmask_b32 v7, v7, v9 :: v_dual_cndmask_b32 v6, v6, v8",
     {0xca521307u, 0x07061106u}},
    {rocjitsu::IsaFamily::Rdna4, "v_pk_fmac_f16 v6, v0, v1", {0x780c0300u}},
    {rocjitsu::IsaFamily::Rdna4, "v_cvt_f16_f32 v6.h, v0", {0x7f0c1500u}},
    {rocjitsu::IsaFamily::Rdna4, "v_cmpx_lt_f16_e64 v0.h, v1.l", {0xd481087eu, 0x02020300u}},
};

TEST_F(SimdCoverage, DecodedTargetPathsMatchScalar) {
  constexpr uint32_t input[] = {0x3c004000u, 0xbc004200u, 0x80000000u, 0x00008000u, 0x3f800000u,
                                0x40000000u, 0xc0400000u, 0x04000400u, 0x7bfffbffu, 0x3f80bf80u};
  for (const auto &test : kDecodedCases) {
    SCOPED_TRACE(test.assembly);
    SCOPED_TRACE(test.arch);
    auto decoder = Decoder::create(test.arch);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, test.words));
    ASSERT_NE(instruction, nullptr);
    for (uint64_t exec : {uint64_t{0}, uint64_t{0x55555555}, uint64_t{0xffffffff}}) {
      std::vector<uint32_t> reference;
      for (bool force_scalar : {true, false}) {
        ComputeUnitCore::Config config{};
        config.arch = test.arch;
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 106;
        config.vgprs_per_wf = 256;
        config.lds_size_kb = 64;
        std::unique_ptr<ComputeUnitCore> target;
        ASSERT_NO_FATAL_FAILURE(rocjitsu::test::initialize_compute_unit(
            target, "simd_decoded_target", config, &memory, &l2));
        const uint32_t wave_size = test.arch == rocjitsu::IsaFamily::Cdna4 ? 64 : 32;
        auto *wave = target->dispatch_wf(0, 0, 106, 256, wave_size);
        ASSERT_NE(wave, nullptr);
        wave->set_mode_raw(3u << 6);
        wave->set_exec(exec);
        wave->set_vcc_raw(0x1234567887654321ULL);
        for (uint32_t reg = 0; reg < 10; ++reg) {
          for (uint32_t lane = 0; lane < wave_size; ++lane) {
            target->write_vgpr(wave->vgpr_alloc().base + reg, lane,
                               input[(reg + lane) % std::size(input)]);
          }
        }
        util::set_force_scalar_for_testing(force_scalar);
        ASSERT_TRUE(target->execute_instruction(instruction.get(), *wave).succeeded());
        std::vector<uint32_t> output;
        for (uint32_t reg = 0; reg < 10; ++reg) {
          for (uint32_t lane = 0; lane < wave_size; ++lane) {
            output.push_back(target->read_vgpr(wave->vgpr_alloc().base + reg, lane));
          }
        }
        output.push_back(uint32_t(wave->exec()));
        output.push_back(uint32_t(wave->exec() >> 32));
        output.push_back(uint32_t(wave->vcc()));
        output.push_back(uint32_t(wave->vcc() >> 32));
        output.push_back(target->read_sgpr(wave->sgpr_alloc().base + 8));
        output.push_back(target->read_sgpr(wave->sgpr_alloc().base + 9));
        if (force_scalar) {
          reference = output;
        } else {
          ASSERT_EQ(output.size(), reference.size());
          for (size_t i = 0; i < output.size(); ++i) {
            ASSERT_EQ(output[i], reference[i]) << "output index=" << i << " exec=" << exec;
          }
        }
      }
    }
  }
}

// Opt-in simulator host timing. Alternate scalar/SIMD order with matched inputs
// and warmups. These timings describe execution overhead, not GPU hardware.
TEST_F(SimdCoverage, DISABLED_InstructionTimings) {
  constexpr unsigned iterations = 5000;
  for (const auto &test : kDecodedCases) {
    if (std::string_view(test.assembly).find("fmac") != std::string_view::npos) {
      continue;
    }
    auto decoder = Decoder::create(test.arch);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, test.words));
    ASSERT_NE(instruction, nullptr);
    ComputeUnitCore::Config config{};
    config.arch = test.arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    std::unique_ptr<ComputeUnitCore> target;
    ASSERT_NO_FATAL_FAILURE(rocjitsu::test::initialize_compute_unit(target, "simd_benchmark_target",
                                                                    config, &memory, &l2));
    uint32_t wave_size = test.arch == rocjitsu::IsaFamily::Cdna4 ? 64 : 32;
    auto *wave = target->dispatch_wf(0, 0, 106, 256, wave_size);
    ASSERT_NE(wave, nullptr);
    wave->set_mode_raw(3u << 6);
    for (unsigned sample = 0; sample < 12; ++sample) {
      for (unsigned order = 0; order < 2; ++order) {
        bool scalar = (order ^ (sample & 1)) != 0;
        util::set_force_scalar_for_testing(scalar);
        const std::string_view name(test.assembly);
        const bool packed = name.find("pk_") != std::string_view::npos;
        const uint32_t input =
            packed ? (name.find("bf16") != std::string_view::npos ? 0x3f803f80u : 0x3c003c00u)
                   : 0x3f800000u;
        for (uint32_t reg = 0; reg < 10; ++reg) {
          for (uint32_t lane = 0; lane < wave_size; ++lane) {
            target->write_vgpr(wave->vgpr_alloc().base + reg, lane, input);
          }
        }
        const auto begin = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < iterations; ++i) {
          wave->set_exec(wave_size == 64 ? ~uint64_t{0} : 0xffffffffu);
          auto status = target->execute_instruction(instruction.get(), *wave);
          if (!status.succeeded()) {
            FAIL() << test.assembly;
          }
        }
        const auto elapsed =
            std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - begin)
                .count();
        if (sample >= 3) {
          std::printf("SIMD_TIMING\t%d\t%s\t%u\t%d\t%.3f\n", int(test.arch), test.assembly,
                      sample - 3, scalar, elapsed / iterations);
        }
      }
    }
  }
}

} // namespace
