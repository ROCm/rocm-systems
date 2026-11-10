// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_register_test.cpp
/// @brief Architectural register observation through execution plugins.
///
/// @details Register-observation tests in this file execute decoded/generated
/// instructions and assert that each callback names exactly the physical
/// registers, lanes, and bytes that the instruction architecturally accesses.
/// Machine-level preservation below RegisterAccess must not add callbacks.

#include "compute_unit_test_util.h"
#include "execution_plugin_test_util.h"
#include "mma_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_path_test_hooks.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "util/host_target.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop3.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::test::execution_plugin;
using mma_test::make_cdna4_mfma_scale_words;

struct ForceScalarOverride {
  explicit ForceScalarOverride(bool value) : old(util::force_scalar()) {
    util::set_force_scalar_for_testing(value);
  }
  ~ForceScalarOverride() { util::set_force_scalar_for_testing(old); }

  bool old;
};

struct Wave32PluginFixture {
  std::unique_ptr<amdgpu::GpuMemory> gpu_mem;
  std::unique_ptr<amdgpu::L2Cache> l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::shared_ptr<ExecutionPluginGroup> plugin_group;

  explicit Wave32PluginFixture(
      rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5,
      amdgpu::MemoryWaitDiagnostics wait_checks = amdgpu::MemoryWaitDiagnostics::Warn,
      uint32_t vgpr_slot_size = 256)
      : gpu_mem(std::make_unique<amdgpu::GpuMemory>("wave32_plugin_mem")),
        l2(std::make_unique<amdgpu::L2Cache>("wave32_plugin_l2")) {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 104;
    cfg.vgprs_per_wf = vgpr_slot_size;
    cfg.lds_size_kb = 64;
    cfg.memory_wait_diagnostics = wait_checks;
    rocjitsu::test::initialize_compute_unit(cu, "wave32_plugin_cu", cfg, gpu_mem.get(), l2.get());
    if (!cu)
      return;
  }

  OrderingPlugin *attach_ordering_plugin() {
    plugin_group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<OrderingPlugin>();
    auto *p = plugin.get();
    plugin_group->add(std::move(plugin));
    cu->set_plugin_group(plugin_group);
    plugin_group->onInit();
    return p;
  }
};

static_assert(!amdgpu::mma_f32_native_width_supported(16, 1));
static_assert(amdgpu::mma_f32_native_width_supported(16, 4));
static_assert(amdgpu::mma_f32_native_width_supported(16, 8));
static_assert(amdgpu::mma_f32_native_width_supported(16, 16));
static_assert(!amdgpu::mma_f32_native_width_supported(16, 32));

TEST(ExecutionPluginTest, Gfx11TfeRequiresTheWholeDestinationBeforeReadingAddressOperands) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    Wave32PluginFixture fixture(arch);
    auto *observer = fixture.attach_ordering_plugin();
    auto *wave = fixture.cu->dispatch_wf(0, 0, 104, 32, 32);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(1);
    fixture.cu->write_sgpr(wave->sgpr_alloc().base + 3, 0x20014000);
    auto decoder = Decoder::create(arch);
    for (bool typed : {false, true}) {
      for (uint8_t first : {30, 31}) {
        SCOPED_TRACE(testing::Message() << arch << " typed " << typed << " vdata " << +first);
        const auto words =
            typed ? rdna3::build_mtbuf(rdna3::kTbufferLoadFormatXMtbuf, {.format = 20,
                                                                         .vaddr = 4,
                                                                         .vdata = first,
                                                                         .tfe = 1,
                                                                         .offen = 1,
                                                                         .soffset = 12})
                  : rdna3::build_mubuf(
                        rdna3::kBufferLoadB32Mubuf,
                        {.vaddr = 4, .vdata = first, .tfe = 1, .offen = 1, .soffset = 12});
        auto inst = decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        observer->events.clear();
        ASSERT_TRUE(fixture.cu->execute_instruction(inst.value().get(), *wave).succeeded());
        std::set<uint32_t> scalar_reads, vector_reads;
        for (const auto &event : observer->events) {
          if (event.kind == HookEvent::READ_SGPR)
            scalar_reads.insert(event.physical_reg - wave->sgpr_alloc().base);
          if (event.kind == HookEvent::READ_VGPR)
            vector_reads.insert(event.physical_reg - wave->vgpr_alloc().base);
        }
        if (first == 30) {
          EXPECT_EQ(vector_reads, std::set<uint32_t>{4});
          EXPECT_EQ(scalar_reads, (std::set<uint32_t>{0, 1, 2, 3, 12}));
        } else {
          EXPECT_TRUE(vector_reads.empty());
          EXPECT_TRUE(scalar_reads.empty());
        }
      }
    }
  }
}

TEST(ExecutionPluginTest, ValuSimdReadObservationUsesActiveExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + 0, lane, lane);
      cu->write_vgpr(vb + 1, lane, lane * 3);
      cu->write_vgpr(vb + 2, lane, 0);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    uint32_t words[4] = {vop2_encode(/*opcode=*/52, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256), 0u, 0u,
                         0u};
    Instruction *inst = decode_valid(*decoder, words);
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(cu->execute_instruction(inst, *wf).succeeded());
    delete inst;

    expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {0, 1}, kPartialExecMask);
  }
}

TEST(ExecutionPluginTest, MultiplyClassificationPreservesActiveReadsAndExceptionFlags) {
  const std::pair<rj_code_arch_t, uint32_t> targets[] = {
      {ROCJITSU_CODE_ARCH_CDNA1, cdna1::kVMulF32Vop2},
      {ROCJITSU_CODE_ARCH_CDNA2, cdna2::kVMulF32Vop2},
      {ROCJITSU_CODE_ARCH_CDNA3, cdna3::kVMulF32Vop2},
      {ROCJITSU_CODE_ARCH_CDNA4, cdna4::kVMulF32Vop2},
      {ROCJITSU_CODE_ARCH_CDNA5, cdna5::kVMulF32Vop2},
      {ROCJITSU_CODE_ARCH_RDNA1, rdna1::kVMulF32Vop2},
      {ROCJITSU_CODE_ARCH_RDNA2, rdna2::kVMulF32Vop2},
      {ROCJITSU_CODE_ARCH_RDNA3, rdna3::kVMulF32Vop2},
      {ROCJITSU_CODE_ARCH_RDNA3_5, rdna3_5::kVMulF32Vop2},
      {ROCJITSU_CODE_ARCH_RDNA4, rdna4::kVMulF32Vop2},
  };
  constexpr uint64_t kExec = (1u << 1) | (1u << 7) | (1u << 21);
  for (const auto &[arch, opcode] : targets) {
    SCOPED_TRACE(static_cast<int>(arch));
    Wave32PluginFixture f(arch);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto *plugin = f.attach_ordering_plugin();
    auto *wf = f.cu->dispatch_wf(0, 0, 104, 32);
    ASSERT_NE(wf, nullptr);
    const uint32_t base = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      const uint32_t value =
          std::bit_cast<uint32_t>((kExec & (uint64_t{1} << lane)) ? 1.1f : 1e38f);
      f.cu->write_vgpr(base, lane, value);
      f.cu->write_vgpr(base + 1, lane, value);
    }
    f.cu->write_sgpr(wf->sgpr_alloc().base + 4, std::bit_cast<uint32_t>(1.1f));
    auto decoder = Decoder::create(arch);
    for (uint64_t exec : {uint64_t{0}, kExec}) {
      // Exercise vector, SGPR, literal, and inline-constant sources.
      for (uint32_t source : {256u, 4u, 255u, 240u}) {
        SCOPED_TRACE(testing::Message() << "exec " << exec << " source " << source);
        wf->set_exec(exec);
        wf->set_trapsts(0);
        uint32_t words[] = {vop2_encode(opcode, 2, 1, source), std::bit_cast<uint32_t>(1.1f)};
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
        ASSERT_NE(inst, nullptr);
        plugin->events.clear();
        ASSERT_TRUE(f.cu->execute_instruction(inst.get(), *wf).succeeded());
        EXPECT_EQ(wf->trapsts() & 0x7fu, exec && source != 240 ? 1u << 5 : 0u);
        std::map<uint32_t, uint64_t> reads;
        for (const auto &event : vgpr_read_events(*plugin)) {
          EXPECT_EQ(event.lane_mask & ~exec, 0u);
          EXPECT_EQ(event.byte_mask, ExecutionPlugin::kFullByteMask);
          reads[event.physical_reg - base] |= event.lane_mask;
        }
        std::map<uint32_t, uint64_t> expected;
        if (exec) {
          expected[1] = exec;
          if (source == 256)
            expected[0] = exec;
        }
        EXPECT_EQ(reads, expected);
      }
    }
  }
}

TEST(ExecutionPluginTest, AddUsesEachActiveSourceOnceForArithmeticAndStatus) {
  for (const auto &[arch, opcode, half] : {std::tuple{ROCJITSU_CODE_ARCH_CDNA1, 1u, false},
                                           {ROCJITSU_CODE_ARCH_RDNA3_5, 3u, false},
                                           {ROCJITSU_CODE_ARCH_RDNA1, 50u, true}}) {
    Wave32PluginFixture f(arch);
    ASSERT_TRUE(f.cu);
    auto *plugin = f.attach_ordering_plugin();
    auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, 64);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(arch);
    const uint32_t base = wave->vgpr_alloc().base;
    for (bool scalar : {false, true}) {
      ForceScalarOverride mode(scalar);
      for (uint64_t active : {0u, 1u, 0x11u}) {
        const uint32_t one = half ? 0x3c00u : 0x3f800000u;
        for (unsigned lane : {0u, 4u}) {
          f.cu->write_vgpr(base, lane, one);
          f.cu->write_vgpr(base + 1, lane, one);
        }
        wave->set_exec(active);
        wave->set_trapsts(0x40u);
        const std::array words{vop2_encode(opcode, /*vdst=*/0, /*vsrc1=*/1, /*src0=*/256)};
        auto inst = decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        plugin->events.clear();
        ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
        uint64_t observed[2]{};
        for (const auto &event : vgpr_read_events(*plugin)) {
          ASSERT_LT(event.physical_reg - base, 2u);
          uint64_t &seen = observed[event.physical_reg - base];
          EXPECT_EQ(seen & event.lane_mask, 0u);
          EXPECT_EQ(event.byte_mask, half ? 3u : 15u);
          seen |= event.lane_mask;
        }
        EXPECT_EQ(observed[0], active);
        EXPECT_EQ(observed[1], active);
        EXPECT_EQ(wave->trapsts(), 0x40u);
        for (unsigned lane : {0u, 4u})
          EXPECT_EQ(f.cu->read_vgpr(base, lane),
                    (active & (1ull << lane)) ? (half ? 0x4000u : 0x40000000u) : one);
      }
    }
    wave->halt();
  }
}

TEST(ExecutionPluginTest, SignalingCompareReadsOnlySelectedInputs) {
  for (auto arch :
       {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    Wave32PluginFixture f(arch);
    ASSERT_TRUE(f.cu);
    auto *plugin = f.attach_ordering_plugin();
    auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, 64);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(arch);
    const uint32_t vb = wave->vgpr_alloc().base;
    const bool gfx115 = arch == ROCJITSU_CODE_ARCH_RDNA3_5;
    const uint32_t eq = gfx115 ? 0xd402800a : 0xd4ca800a;
    const uint32_t never = gfx115 ? 0xd400800a : 0xd4c8800a;
    const uint32_t always = gfx115 ? 0xd40f800a : 0xd4ef800a;
    // The low half of v0 is a quiet NaN; its high half and both halves of v1 are 1.
    f.cu->write_vgpr(vb, 0, 0x3c007e00);
    f.cu->write_vgpr(vb + 1, 0, 0x3c003c00);
    struct Case {
      uint32_t word, result, invalid, mask;
    };
    const Case cases[] = {{eq, 0, 1, 3},
                          {eq | 0x800, unsigned(gfx115), unsigned(!gfx115), gfx115 ? 12u : 3u},
                          {never, 0, 1, 3},
                          {always, 1, 1, 3}};
    for (bool scalar : {false, true}) {
      ForceScalarOverride execution_mode(scalar);
      for (auto c : cases) {
        SCOPED_TRACE(testing::Message()
                     << arch << " word=" << std::hex << c.word << " scalar=" << scalar);
        wave->set_mode_raw(0x2f0);
        wave->set_exec(1);
        wave->set_trapsts(0x40);
        const std::array words{c.word, 0x02020300u};
        auto inst = decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        plugin->events.clear();
        ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
        EXPECT_EQ(wave->trapsts(), 0x40u | c.invalid);
        EXPECT_EQ(f.cu->read_sgpr(wave->sgpr_alloc().base + 10), c.result);
        const auto reads = vgpr_read_events(*plugin);
        ASSERT_EQ(reads.size(), 2u);
        EXPECT_EQ(reads[0].physical_reg, vb);
        EXPECT_EQ(reads[0].lane_mask, 1u);
        EXPECT_EQ(reads[0].byte_mask, c.mask);
        EXPECT_EQ(reads[1].physical_reg, vb + 1);
        EXPECT_EQ(reads[1].lane_mask, 1u);
        EXPECT_EQ(reads[1].byte_mask, 3u);
      }
    }
    wave->halt();
  }
}

TEST(ExecutionPluginTest, Gfx115CompareUsesAllocatedSourcesAndTheSimdPredicate) {
  Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA3_5);
  ASSERT_TRUE(f.cu);
  auto *observer = f.attach_ordering_plugin();
  auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, 32);
  ASSERT_NE(wave, nullptr);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
  const auto base = wave->vgpr_alloc().base;
  wave->set_mode_raw(0x2f0);
  wave->set_exec(0x11);
  for (bool scalar : {false, true}) {
    ForceScalarOverride mode(scalar);
    for (uint32_t reg : {0u, 20u, 40u}) {
      for (bool extended : {false, true}) {
        const std::array words = extended ? std::array{0xd412006au, 0x02020300u | reg}
                                          : std::array{0x7c240300u | reg, 0u};
        auto inst = decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        f.cu->write_vgpr(base, 0, 0x3f800000);
        f.cu->write_vgpr(base, 4, 0x7fa00000);
        f.cu->write_vgpr(base + 1, 0, 0x3f800000);
        f.cu->write_vgpr(base + 1, 4, 0x3f800000);
        f.cu->write_vgpr(base + 20, 0, 0x40000000);
        wave->set_trapsts(0);
        observer->events.clear();
        amdgpu::ScopedSimdFastPathTracker tracker;
        ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
        EXPECT_EQ(wave->vcc(), 1u);
        EXPECT_EQ(wave->trapsts() & 1u, 1u);
        if (scalar || !util::has_stdx_simd)
          EXPECT_TRUE(tracker.no_tracked_path_executed());
        else
          EXPECT_TRUE(
              tracker.only_tracked_path_executed(amdgpu::SimdFastPath::VOPC_COMPARE_STATUS));
        const auto reads = vgpr_read_events(*observer);
        ASSERT_EQ(reads.size(), 2u);
        EXPECT_EQ(reads[0].physical_reg, base);
        EXPECT_EQ(reads[1].physical_reg, base + 1);
        EXPECT_EQ(reads[0].lane_mask, 0x11u);
        EXPECT_EQ(reads[1].lane_mask, 0x11u);
      }
    }
  }
  // Compact high halves are normalized after remapping. A straddling F64
  // VALU source restarts at v[0:1]; the adjacent allocated v[14:15] stays intact.
  wave->set_exec(1);
  for (const auto &[word, second, half, first] : {std::tuple{0x7c040394u, 0u, true, 0u},
                                                  {0xd402086au, 0x02020314u, true, 0u},
                                                  {0x7c44050fu, 0u, false, 0u},
                                                  {0xd422006au, 0x0202050fu, false, 0u},
                                                  {0x7c44050eu, 0u, false, 14u},
                                                  {0xd422006au, 0x0202050eu, false, 14u}}) {
    f.cu->write_vgpr(base, 0, half ? 0x3c000000u : 0x40000000u);
    f.cu->write_vgpr(base + 1, 0, half ? 0x3c00u : 0x3ff00000u);
    f.cu->write_vgpr(base + 14, 0, 0x87654321u);
    f.cu->write_vgpr(base + 15, 0, 0x11111111u);
    f.cu->write_vgpr(base + 2, 0, first ? 0x87654321u : 0x40000000u);
    f.cu->write_vgpr(base + 3, 0, first ? 0x11111111u : 0x3ff00000u);
    auto inst = decoder->decode_window(std::array{word, second});
    ASSERT_TRUE(inst.succeeded());
    observer->events.clear();
    ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
    EXPECT_EQ(wave->vcc(), 1u);
    const auto reads = vgpr_read_events(*observer);
    ASSERT_EQ(reads.size(), half ? 2u : 4u);
    EXPECT_EQ(reads[0].physical_reg, base + first);
    if (half) {
      EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kHighHalfByteMask);
    } else {
      EXPECT_EQ(reads[1].physical_reg, base + first + 1);
      auto &state = wave->ensure_memory_wait_scoreboard();
      for (uint16_t reg : {0u, 1u, 14u, 15u}) {
        unsigned hazards = 0;
        state.clear();
        state.bind(0x200, &hazards, [](void *ctx, const auto &hazard) {
          if (!hazard.write)
            ++*static_cast<unsigned *>(ctx);
        });
        constexpr auto counter = amdgpu::WaitCounterKind::Load;
        state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, reg, 1}, counter, 0xf});
        state.check_instruction(*inst.value(), *wave);
        EXPECT_EQ(hazards != 0, reg == first || reg == first + 1);
        state.clear();
      }
    }
  }
  wave->halt();
}

TEST(ExecutionPluginTest, Gfx115ValuAllocationSuppressesTheIssueOrSelectsTheWholeSource) {
  for (unsigned slot : {16u, 32u}) {
    Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA3_5, amdgpu::MemoryWaitDiagnostics::Warn, slot);
    auto *observed = f.attach_ordering_plugin();
    auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, 32);
    ASSERT_NE(wave, nullptr);
    auto dec = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
    auto &state = wave->ensure_memory_wait_scoreboard();
    const auto base = wave->vgpr_alloc().base;
    for (bool scalar : {false, true}) {
      ForceScalarOverride mode(scalar);
      for (const auto &[words, dst, source, width, half] :
           {std::tuple{std::array{0x7e040314u, 0u}, 2u, 0u, 1u, false},
            std::tuple{std::array{0x7e2a0314u, 0u}, 21u, 0u, 1u, false},
            std::tuple{std::array{0xd7006a04u, 0x02020501u}, 4u, 1u, 1u, false},
            std::tuple{std::array{0xd7006a15u, 0x02020501u}, 21u, 1u, 1u, false},
            std::tuple{std::array{0x7f1e3994u, 0u}, 15u, 0u, 1u, true},
            std::tuple{std::array{0x7f2a3994u, 0u}, 21u, 0u, 1u, true},
            std::tuple{std::array{0xca100101u, 0x04040116u}, 5u, 2u, 1u, false},
            std::tuple{std::array{0xca100101u, 0x0a0a0117u}, 11u, 3u, 1u, false},
            std::tuple{std::array{0xca100115u, 0x06060102u}, 6u, 1u, 1u, false},
            std::tuple{std::array{0xd63d0004u, 0x043a1508u}, 4u, 0u, 4u, false},
            std::tuple{std::array{0xd63d0004u, 0x04321508u}, 4u, 12u, 4u, false},
            std::tuple{std::array{0xd63d000eu, 0x043a1508u}, 14u, 0u, 4u, false}}) {
        SCOPED_TRACE(testing::Message() << std::hex << words[0] << " " << words[1]
                                        << " slot=" << slot << " scalar=" << scalar);
        wave->set_exec(1);
        for (unsigned i = 0; i < slot; ++i)
          f.cu->write_vgpr(base + i, 0, ((0x100u + i) << 16) | (100u + i));
        if (width == 4)
          f.cu->write_vgpr(base + 10, 0,
                           0); // Zero reference bytes leave each accumulator unchanged.
        const bool carry = (words[0] >> 16) == 0xd700; // V_ADD_CO_U32 writes VCC and VDST.
        if (carry) {
          wave->set_exec(UINT32_MAX);
          wave->set_vcc_mask(0x11111111u);
          f.cu->write_vgpr(base + 1, 0, UINT32_MAX);
          f.cu->write_vgpr(base + 2, 0, 1);
        }
        auto decoded = dec->decode_window(words);
        ASSERT_TRUE(decoded.succeeded());
        auto &inst = *decoded.value();
        const bool suppressed = dst + width > 16;
        observed->events.clear();
        ASSERT_TRUE(f.cu->execute_instruction(&inst, *wave).succeeded());
        if (carry) {
          EXPECT_EQ(wave->vcc_mask(), suppressed ? 0x11111111u : 1u);
        }
        const auto reads = vgpr_read_events(*observed);
        if (suppressed) {
          EXPECT_TRUE(reads.empty());
        }
        for (unsigned i = 0; i < width && dst + i < slot; ++i) {
          const auto value = [](unsigned n) { return ((0x100u + n) << 16) | (100u + n); };
          const uint32_t expected = suppressed ? value(dst + i)
                                    : carry    ? 0u
                                    : half ? (value(source) & 0xffff0000u) | (value(dst) & 0xffffu)
                                           : value(source + i);
          EXPECT_EQ(f.cu->read_vgpr(base + dst + i, 0), expected);
        }
        for (uint16_t reg : {0u, 1u, 2u, 3u, 12u, 13u, 14u, 15u, 16u, 17u, 20u, 21u, 22u, 23u}) {
          const bool vopd = (words[0] >> 24) == 0xca;
          const bool other_source = (vopd && reg == (dst == 6 ? 2 : 1)) || (carry && reg == 2);
          const bool should_read =
              !suppressed && (other_source || (reg >= source && reg < source + width));
          EXPECT_EQ(std::ranges::any_of(
                        reads, [&](const auto &event) { return event.physical_reg == base + reg; }),
                    should_read)
              << reg;
          state.clear();
          unsigned warnings = 0;
          state.bind(0x200, &warnings, [](void *ctx, const auto &hazard) {
            if (!hazard.write)
              ++*static_cast<unsigned *>(ctx);
          });
          constexpr auto c = amdgpu::WaitCounterKind::Load;
          state.add({state.issue(c), 0x100, 1, {RegClass::VGPR, reg, 1}, c, 0xf});
          state.check_instruction(inst, *wave);
          EXPECT_EQ(warnings != 0, should_read) << reg;
        }
      }
    }
    wave->halt();
  }
}

TEST(ExecutionPluginTest, Gfx115M0RelativeDppReadsOnlyAnAdmittedDestination) {
  Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA3_5, amdgpu::MemoryWaitDiagnostics::Warn, 32);
  auto *observer = f.attach_ordering_plugin();
  auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, 32);
  ASSERT_NE(wave, nullptr);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
  const auto base = wave->vgpr_alloc().base;
  wave->set_exec(3);
  for (uint32_t opc : {0x7e088400u, 0xd5c20004u, 0x7e088800u, 0x7e089000u}) {
    for (bool dpp8 : {false, true}) {
      const uint32_t selector = dpp8 ? 0xe9u : 0xfau;
      const uint32_t ext = dpp8 ? 0xfac68803u : 0xff091103u;
      const auto words =
          opc == 0xd5c20004u ? std::array{opc, selector, ext} : std::array{opc | selector, ext, 0u};
      for (uint32_t offset : {11u, 12u, 16u}) {
        SCOPED_TRACE(testing::Message() << std::hex << opc << " dpp8=" << dpp8 << " m0=" << offset);
        const bool split = opc == 0x7e089000u;
        wave->set_m0(split ? (offset << 16) | 1u : offset);
        const auto source = opc == 0x7e088800u ? 3 + offset : split ? 4u : 3u;
        for (unsigned reg = 0; reg < 32; ++reg)
          for (unsigned lane = 0; lane < 2; ++lane)
            f.cu->write_vgpr(base + reg, lane, 100 * reg + lane);
        auto op = decoder->decode_window(words);
        ASSERT_TRUE(op.succeeded());
        observer->events.clear();
        ASSERT_TRUE(f.cu->execute_instruction(op.value().get(), *wave).succeeded());
        const auto reads = vgpr_read_events(*observer);
        EXPECT_EQ(!reads.empty(), offset == 11);
        const auto expected = offset == 11 ? 100 * source + unsigned(dpp8) : 100 * (4 + offset) + 1;
        EXPECT_EQ(f.cu->read_vgpr(base + 4 + offset, 1), expected);
        auto &state = wave->ensure_memory_wait_scoreboard();
        state.clear();
        unsigned warnings = 0;
        state.bind(0x200, &warnings, [](void *ctx, const auto &hazard) {
          if (!hazard.write)
            ++*static_cast<unsigned *>(ctx);
        });
        constexpr auto counter = amdgpu::WaitCounterKind::Load;
        state.add(
            {state.issue(counter), 0x100, 3, {RegClass::VGPR, uint16_t(source), 1}, counter, 0xf});
        state.check_instruction(*op.value(), *wave);
        EXPECT_EQ(warnings != 0, offset == 11);
      }
    }
  }
  wave->halt();
}

TEST(ExecutionPluginTest, Gfx115Wave64SkipsBothDualOperationsAndTheirWaits) {
  for (unsigned width : {32u, 64u}) {
    Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA3_5, amdgpu::MemoryWaitDiagnostics::Warn, 32);
    auto *observer = f.attach_ordering_plugin();
    auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, width);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
    auto inst = decoder->decode_window(std::array{0xca100101u, 0x04040116u});
    ASSERT_TRUE(inst.succeeded());
    wave->set_exec(1);
    auto &state = wave->ensure_memory_wait_scoreboard();
    unsigned hazards = 0;
    state.bind(0x200, &hazards, [](void *ctx, const auto &) { ++*static_cast<unsigned *>(ctx); });
    constexpr auto c = amdgpu::WaitCounterKind::Load;
    state.add({state.issue(c), 0x100, 1, {RegClass::VGPR, 2, 1}, c, 0xf});
    state.check_instruction(*inst.value(), *wave);
    EXPECT_EQ(hazards != 0, width == 32);
    observer->events.clear();
    ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
    EXPECT_EQ(!vgpr_read_events(*observer).empty(), width == 32);
    wave->halt();
  }
}

TEST(ExecutionPluginTest, Gfx115UniformDppAddChecksOnlyAnAdmittedLane) {
  Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA3_5, amdgpu::MemoryWaitDiagnostics::Warn, 32);
  auto *observer = f.attach_ordering_plugin();
  auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, 32);
  ASSERT_NE(wave, nullptr);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
  auto &state = wave->ensure_memory_wait_scoreboard();
  unsigned hazards = 0;
  state.bind(0x200, &hazards, [](void *ctx, const auto &) { ++*static_cast<unsigned *>(ctx); });
  constexpr auto counter = amdgpu::WaitCounterKind::Km;
  wave->set_exec(1);
  wave->debug_write_sgpr(4, 0x40000000u);
  const auto dst = wave->vgpr_alloc().base + 2;
  for (bool scalar : {false, true}) {
    ForceScalarOverride mode(scalar);
    for (bool clamp : {false, true}) {
      wave->set_mode_raw(0x2f0 | (uint32_t{clamp} << 8));
      for (bool bound : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "scalar=" << scalar << " clamp=" << clamp << " bound=" << bound);
        // v_add_f32 v2, v3, s4 row_shr:1. Lane zero has no predecessor.
        auto inst = decoder->decode_window(
            std::array{0xd5030002u, 0x000008fau, 0xff011103u | (uint32_t{bound} << 19)});
        ASSERT_TRUE(inst.succeeded());
        state.clear();
        hazards = 0;
        state.add(
            {state.issue(counter), 0x100, ~uint64_t{0}, {RegClass::SGPR, 4, 1}, counter, 0xf});
        state.check_instruction(*inst.value(), *wave);
        EXPECT_EQ(hazards != 0, bound);
        f.cu->write_vgpr(dst, 0, 0xdeadbeefu);
        observer->events.clear();
        ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
        EXPECT_EQ(f.cu->read_vgpr(dst, 0), bound ? 0x40000000u : 0xdeadbeefu);
        const bool read = std::ranges::any_of(observer->events, [&](const auto &event) {
          return event.kind == HookEvent::READ_SGPR &&
                 event.physical_reg == wave->sgpr_alloc().base + 4;
        });
        EXPECT_EQ(read, bound);
      }
    }
  }
  wave->halt();
}

TEST(ExecutionPluginTest, Gfx115WmmaUsesWholeTwoRegisterIu4Factors) {
  Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA3_5, amdgpu::MemoryWaitDiagnostics::Warn, 32);
  auto *observer = f.attach_ordering_plugin();
  auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, 32);
  ASSERT_NE(wave, nullptr);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
  auto &state = wave->ensure_memory_wait_scoreboard();
  const auto base = wave->vgpr_alloc().base;
  for (unsigned start : {14u, 15u}) {
    for (bool right : {false, true}) {
      SCOPED_TRACE(testing::Message() << "factor=" << (right ? 'B' : 'A') << " start=" << start);
      // v_wmma_i32_16x16x16_iu4 v[8:15], A, B, 0. The other factor is v[2:3].
      const unsigned a = right ? 2u : start;
      const unsigned b = right ? start : 2u;
      auto inst = decoder->decode_window(std::array{0xcc454008u, 0x1a020100u | (b << 9) | a});
      ASSERT_TRUE(inst.succeeded());
      for (unsigned reg : {0u, 1u, 2u, 3u, 14u, 15u, 16u})
        for (unsigned lane = 0; lane < 32; ++lane)
          f.cu->write_vgpr(base + reg, lane, reg < 4 ? 0x11111111u : 0x22222222u);
      observer->events.clear();
      ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
      EXPECT_EQ(f.cu->read_vgpr(base + 8, 0), start == 15 ? 16u : 32u);
      const auto reads = vgpr_read_events(*observer);
      for (uint16_t reg : {0u, 1u, 2u, 3u, 14u, 15u, 16u}) {
        const bool used = reg == 2 || reg == 3 || (start == 15 ? reg < 2 : reg == 14 || reg == 15);
        EXPECT_EQ(
            std::ranges::any_of(reads, [&](const auto &e) { return e.physical_reg == base + reg; }),
            used)
            << reg;
        state.clear();
        unsigned hazards = 0;
        state.bind(0x200, &hazards, [](void *ctx, const auto &hazard) {
          if (!hazard.write)
            ++*static_cast<unsigned *>(ctx);
        });
        constexpr auto counter = amdgpu::WaitCounterKind::Load;
        state.add(
            {state.issue(counter), 0x100, ~uint64_t{0}, {RegClass::VGPR, reg, 1}, counter, 0xf});
        state.check_instruction(*inst.value(), *wave);
        EXPECT_EQ(hazards != 0, used) << reg;
      }
    }
  }
  wave->halt();
}

TEST(ExecutionPluginTest, Gfx115WmmaUsesPerWaveAccumulatorAndDestinationTuples) {
  for (unsigned width : {32u, 64u}) {
    Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA3_5, amdgpu::MemoryWaitDiagnostics::Warn, 32);
    auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, width);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
    const auto base = wave->vgpr_alloc().base;
    const unsigned count = 256 / width;
    for (unsigned start : {0u, 8u, 12u, 14u}) {
      SCOPED_TRACE(testing::Message() << "wave=" << width << " accumulator=" << start);
      for (unsigned reg = 0; reg < 24; ++reg)
        for (unsigned lane = 0; lane < width; ++lane)
          f.cu->write_vgpr(base + reg, lane,
                           reg < 8 ? 0 : std::bit_cast<uint32_t>(float(100 + reg)));
      const unsigned dst = 16 - count;
      auto op = decoder->decode_window(std::array{0xcc404000u | dst, 0x1c020100u | (start << 18)});
      ASSERT_TRUE(op.succeeded());
      ASSERT_TRUE(f.cu->execute_instruction(op.value().get(), *wave).succeeded());
      for (unsigned i = 0; i < count; ++i)
        EXPECT_EQ(f.cu->read_vgpr(base + dst + i, 0),
                  start && start + count <= 16 ? std::bit_cast<uint32_t>(float(100 + start + i))
                                               : 0u);
    }
    // A and B keep their full eight-word extent at both wave sizes.
    for (uint32_t factors : {0x1a02010eu, 0x1a021d00u}) {
      for (unsigned reg = 0; reg < 24; ++reg)
        for (unsigned lane = 0; lane < width; ++lane)
          f.cu->write_vgpr(base + reg, lane, reg < 8 ? 0x3c003c00u : 0x40004000u);
      auto op = decoder->decode_window(std::array{0xcc404008u, factors});
      ASSERT_TRUE(op.succeeded());
      ASSERT_TRUE(f.cu->execute_instruction(op.value().get(), *wave).succeeded());
      EXPECT_EQ(f.cu->read_vgpr(base + 8, 0), std::bit_cast<uint32_t>(16.0f));
    }
    wave->halt();
  }
}

TEST(ExecutionPluginTest, Gfx115ConstantCompareAndDppAddObserveTheirPendingSourceLanes) {
  Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA3_5, amdgpu::MemoryWaitDiagnostics::Warn, 32);
  auto *observed = f.attach_ordering_plugin();
  auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, 32);
  ASSERT_NE(wave, nullptr);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
  const auto base = wave->vgpr_alloc().base;
  constexpr uint64_t active = 0x10003;
  wave->set_exec(active);
  for (bool scalar : {false, true}) {
    ForceScalarOverride mode(scalar);
    for (const auto &[words, add, fixed] :
         {std::tuple{std::array{0x7c200314u, 0u, 0u}, false, true}, // V_CMP_F_F32 e32
          std::tuple{std::array{0x7c3e0314u, 0u, 0u}, false, true}, // V_CMP_T_F32 e32
          std::tuple{std::array{0xd410006au, 0x02020314u, 0u}, false, true},
          std::tuple{std::array{0xd41f006au, 0x02020314u, 0u}, false, true},
          std::tuple{std::array{0x7c2002fau, 0x1f091114u, 0u}, false, true},
          std::tuple{std::array{0x7c2002e9u, 0xfac68814u, 0u}, false, true},
          std::tuple{std::array{0x7c240314u, 0u, 0u}, false, false}, // V_CMP_EQ_F32
          std::tuple{std::array{0x060628fau, 0x1f00e401u, 0u}, true, false},
          std::tuple{std::array{0xd5030003u, 0x000228fau, 0x1f00e401u}, true, false},
          std::tuple{std::array{0x060628fau, 0xff00e401u, 0u}, true, false}}) {
      auto inst = decoder->decode_window(words);
      ASSERT_TRUE(inst.succeeded());
      for (bool clamp : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << std::hex << words[0] << " clamp=" << clamp << " scalar=" << scalar);
        wave->set_mode_raw((wave->mode_raw() & ~0x100u) | (clamp ? 0x100u : 0u));
        wave->set_trapsts(0);
        wave->set_vcc_mask(0);
        for (unsigned lane : {0u, 1u, 16u}) {
          f.cu->write_vgpr(base, lane, add ? 0x40000000u : 0x7fa00000u);
          f.cu->write_vgpr(base + 1, lane, 0x3f800000u);
          f.cu->write_vgpr(base + 3, lane, 0xdeadbeefu);
        }
        observed->events.clear();
        ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
        uint64_t actual = 0;
        for (const auto &e : vgpr_read_events(*observed))
          if (e.physical_reg == base)
            actual |= e.lane_mask;
        if (add) {
          EXPECT_EQ(actual, active);
          EXPECT_EQ(f.cu->read_vgpr(base + 3, 0), 0x40400000u);
          const bool both_rows = words[1] == 0xff00e401u;
          EXPECT_EQ(f.cu->read_vgpr(base + 3, 16), both_rows ? 0x40400000u : 0xdeadbeefu);
        } else {
          EXPECT_EQ(actual != 0, !fixed || !clamp);
          EXPECT_EQ(wave->trapsts() & 1u, unsigned(!clamp));
        }
        auto &state = wave->ensure_memory_wait_scoreboard();
        for (unsigned lane : {0u, 1u, 16u}) {
          state.clear();
          unsigned warnings = 0;
          state.bind(0x200, &warnings, [](void *ctx, const auto &hazard) {
            if (!hazard.write)
              ++*static_cast<unsigned *>(ctx);
          });
          constexpr auto counter = amdgpu::WaitCounterKind::Load;
          state.add({state.issue(counter),
                     0x100,
                     uint64_t{1} << lane,
                     {RegClass::VGPR, 0, 1},
                     counter,
                     0xf});
          state.check_instruction(*inst.value(), *wave);
          EXPECT_EQ(warnings != 0, bool(actual & (uint64_t{1} << lane))) << lane;
        }
      }
    }
  }
  wave->halt();
}

TEST(ExecutionPluginTest, Gfx115ValuReadsAndWaitsUseTheSameAllocatedSource) {
  for (uint32_t slots : {16u, 32u}) {
    Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA3_5, amdgpu::MemoryWaitDiagnostics::Warn, slots);
    ASSERT_TRUE(f.cu);
    auto *observer = f.attach_ordering_plugin();
    auto *wave = f.cu->dispatch_wf(0, 0, 104, 16, 32);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
    const auto base = wave->vgpr_alloc().base;
    wave->set_exec(1);
    if (slots > 20)
      f.cu->write_vgpr(base + 20, 0, 0x40000000u);
    for (bool scalar : {false, true}) {
      ForceScalarOverride mode(scalar);
      for (const auto &[word, second, add, pair] : {std::tuple{0x06040314u, 0u, true, false},
                                                    {0xd5030002u, 0x02020314u, true, false},
                                                    {0x7c940314u, 0u, false, false},
                                                    {0xd44a006au, 0x02020314u, false, false},
                                                    {0x7c44050fu, 0u, false, true},
                                                    {0xd422006au, 0x0202050fu, false, true}}) {
        SCOPED_TRACE(testing::Message()
                     << std::hex << word << " slots=" << slots << " scalar=" << scalar);
        auto inst = decoder->decode_window(std::array{word, second});
        ASSERT_TRUE(inst.succeeded());
        for (uint32_t dx10 : {0u, 0x100u}) {
          for (uint32_t input : {0x3f800000u, 0x7fa00000u}) {
            wave->set_mode_raw(0x2f0u | dx10);
            wave->set_trapsts(0);
            f.cu->write_vgpr(base, 0, input);
            f.cu->write_vgpr(base + 1, 0, 0x3f800000u);
            f.cu->write_vgpr(base + 2, 0, input);
            f.cu->write_vgpr(base + 3, 0, 0x3f800000u);
            observer->events.clear();
            ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
            auto reads = vgpr_read_events(*observer);
            std::ranges::sort(reads, {}, [](const auto &read) { return read.physical_reg; });
            ASSERT_EQ(reads.size(), pair ? 4u : 2u);
            for (unsigned i = 0; i < reads.size(); ++i) {
              EXPECT_EQ(reads[i].physical_reg, base + i);
              EXPECT_EQ(reads[i].lane_mask, 1u);
            }
            if (add) {
              const auto result = f.cu->read_vgpr(base + 2, 0);
              const auto status = wave->trapsts();
              const auto reference_words =
                  second ? std::array{word, second & ~0xffu} : std::array{word & ~0xffu, 0u};
              auto reference = decoder->decode_window(reference_words);
              ASSERT_TRUE(reference.succeeded());
              wave->set_trapsts(0);
              ASSERT_TRUE(f.cu->execute_instruction(reference.value().get(), *wave).succeeded());
              EXPECT_EQ(result, f.cu->read_vgpr(base + 2, 0));
              EXPECT_EQ(status, wave->trapsts());
              if (input == 0x3f800000u) {
                EXPECT_EQ(result, 0x40000000u);
              }
              if (dx10 == 0) {
                EXPECT_EQ(status & 1u, uint32_t(input == 0x7fa00000u));
              }
            } else {
              EXPECT_EQ(wave->vcc(), uint64_t(pair || input == 0x3f800000u));
            }
          }
        }
        auto &state = wave->ensure_memory_wait_scoreboard();
        for (uint16_t reg : {0u, 1u, 15u, 20u}) {
          unsigned hazards = 0;
          state.clear();
          state.bind(0x200, &hazards, [](void *ctx, const auto &hazard) {
            if (!hazard.write)
              ++*static_cast<unsigned *>(ctx);
          });
          constexpr auto counter = amdgpu::WaitCounterKind::Load;
          state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, reg, 1}, counter, 0xf});
          state.check_instruction(*inst.value(), *wave);
          EXPECT_EQ(hazards != 0, reg < 2);
          state.clear();
        }
      }
    }
    wave->halt();
  }
}

TEST(ExecutionPluginTest, Integer16ComparesReadAndWaitForSelectedHalves) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
                    ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
    Wave32PluginFixture f(arch);
    ASSERT_NE(f.cu, nullptr);
    auto *observer = f.attach_ordering_plugin();
    auto *wave = f.cu->dispatch_wf(0, 0x100, 104, 16, 32);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(1);
    auto decoder = Decoder::create(arch);
    for (bool unsigned_compare : {false, true}) {
      for (unsigned variant = 0; variant < (gfx10 ? 2u : 5u); ++variant) {
        // Variant zero uses e32. The four e64 forms independently select SRC0/SRC1.
        const unsigned opsel = variant ? variant - 1 : 0;
        uint32_t encoding = gfx10 ? (unsigned_compare ? 0x7d540903u : 0x7d140903u)
                                  : (unsigned_compare ? 0x7c740903u : 0x7c640903u);
        if (variant) {
          encoding = gfx10 ? (unsigned_compare ? 0xd4aa000au : 0xd48a000au)
                           : (unsigned_compare ? 0xd43a000au : 0xd432000au);
          encoding |= opsel << 11;
        }
        auto inst = decoder->decode_window(std::array{encoding, 0x02020903u});
        ASSERT_TRUE(inst.succeeded());
        SCOPED_TRACE(testing::Message() << arch << " " << inst.value()->disassemble());
        observer->events.clear();
        ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
        const auto reads = vgpr_read_events(*observer);
        ASSERT_EQ(reads.size(), 2u);
        auto &state = wave->ensure_memory_wait_scoreboard();
        for (uint16_t operand : {0u, 1u}) {
          const uint16_t vgpr = 3 + operand;
          const uint8_t bytes = opsel & (1u << operand) ? 0xc : 0x3;
          auto read = std::ranges::find_if(reads, [&](const auto &event) {
            return event.physical_reg == wave->vgpr_alloc().base + vgpr;
          });
          ASSERT_NE(read, reads.end());
          EXPECT_EQ(read->byte_mask, bytes);
          for (uint8_t pending : {0x3, 0xc}) {
            unsigned reports = 0;
            state.clear();
            state.bind(0x200, &reports,
                       [](void *ctx, const auto &) { ++*static_cast<unsigned *>(ctx); });
            constexpr auto counter = amdgpu::WaitCounterKind::Load;
            state.add(
                {state.issue(counter), 0x100, 1, {RegClass::VGPR, vgpr, 1}, counter, pending});
            state.check_instruction(*inst.value(), *wave);
            EXPECT_EQ(reports, uint32_t(pending == bytes));
          }
        }
      }
    }
    wave->halt();
  }
}

TEST(ExecutionPluginTest, UniformDppCompareSourceWaitsOnlyWhenALaneReadsIt) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(arch);
    Wave32PluginFixture f(arch);
    ASSERT_NE(f.cu, nullptr);
    auto *observer = f.attach_ordering_plugin();
    auto *wave = f.cu->dispatch_wf(0, 0x100, 104, 16, 32);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(1);
    wave->set_mode_raw(0x2f0);
    wave->debug_write_sgpr(4, 0x7fa00000u);
    auto decoder = Decoder::create(arch);
    auto &state = wave->ensure_memory_wait_scoreboard();
    unsigned hazards = 0;
    state.bind(0x200, &hazards, [](void *ctx, const auto &) { ++*static_cast<unsigned *>(ctx); });
    constexpr auto counter = amdgpu::WaitCounterKind::Km;
    state.add({state.issue(counter), 0x100, ~uint64_t{0}, {RegClass::SGPR, 4, 1}, counter, 0xf});
    for (bool bound : {false, true}) {
      // v_cmp_eq_f32 vcc_lo, v3, s4 row_shr:1; only BOUND_CTRL admits lane zero.
      auto inst = decoder->decode_window(
          std::array{0xd412006au, 0x000008fau, 0xff011103u | (uint32_t{bound} << 19)});
      ASSERT_TRUE(inst.succeeded());
      state.check_instruction(*inst.value(), *wave);
      const bool reads_scalar = bound || arch == ROCJITSU_CODE_ARCH_RDNA4;
      EXPECT_EQ(hazards, uint32_t(reads_scalar));
      observer->events.clear();
      ASSERT_TRUE(f.cu->execute_instruction(inst.value().get(), *wave).succeeded());
      EXPECT_EQ(std::ranges::any_of(observer->events,
                                    [&](const auto &event) {
                                      return event.kind == HookEvent::READ_SGPR &&
                                             event.physical_reg == wave->sgpr_alloc().base + 4;
                                    }),
                reads_scalar);
    }
    wave->halt();
  }
}

TEST(ExecutionPluginTest, ValuSimdWriteObservationUsesActiveExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + 0, lane, lane);
      cu->write_vgpr(vb + 1, lane, lane * 3);
      cu->write_vgpr(vb + 2, lane, 0);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    uint32_t words[4] = {vop2_encode(/*opcode=*/52, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256), 0u, 0u,
                         0u};
    Instruction *inst = decode_valid(*decoder, words);
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(cu->execute_instruction(inst, *wf).succeeded());
    delete inst;

    expect_vgpr_read_set(vgpr_write_events(*plugin), vb, {2}, kPartialExecMask);
  }
}

// DPP observation tests exercise the interaction among EXEC, row/bank
// destination masks, source-lane permutation, BOUND_CTRL, fetch-inactive, and
// source/destination aliasing. Helper-level permutation coverage lives in
// shared_infra_test.cpp; these tests prove that decoded instructions carry the
// resulting source and destination lane sets through to plugin callbacks.
TEST(ExecutionPluginTest, DppObservationReportsExactSourceAndDestinationLanes) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    constexpr uint32_t kSrc = 2;
    constexpr uint32_t kDst = 5;
    constexpr uint32_t kSrcValue = 0x11223344u;
    constexpr uint32_t kOldDst = 0xAABBCCDDu;
    constexpr uint64_t kDppWriteMask = 0x0000'0000'0000'0F0FULL;
    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + kSrc, lane, kSrcValue);
      cu->write_vgpr(vb + kDst, lane, kOldDst);
    }

    // V_MOV_B32 with a partial row/bank mask. Only lane 0 survives EXEC and the
    // destination mask. Source observation is determined from every active
    // destination before row/bank commit filtering; dpp_ctrl=0 selects the
    // first source lane of each active quad.
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    uint32_t words[2] = {
        vop1_encode(/*opcode=*/1, kDst, amdgpu::SRC_DPP),
        vop1_dpp_word(kSrc, /*dpp_ctrl=*/0, /*row_mask=*/0x1, /*bank_mask=*/0x5),
    };
    for (bool force_scalar : {false, true}) {
      SCOPED_TRACE(force_scalar ? "scalar" : "simd");
      ForceScalarOverride force_scalar_override(force_scalar);
      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
        cu->write_vgpr(vb + kDst, lane, kOldDst);

      Instruction *inst = decode_valid(*decoder, words);
      ASSERT_NE(inst, nullptr);
      plugin->events.clear();
      EXPECT_TRUE(cu->execute_instruction(inst, *wf).succeeded());
      delete inst;

      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
        const uint64_t lane_bit = uint64_t{1} << lane;
        const uint32_t expected =
            (kPartialExecMask & kDppWriteMask & lane_bit) ? kSrcValue : kOldDst;
        EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, lane), expected) << "lane " << lane;
      }

      const auto reads = vgpr_read_events(*plugin);
      ASSERT_EQ(reads.size(), 1u);
      EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
      EXPECT_EQ(reads[0].lane_mask, 0x1111'1010'1111'1001ULL);
      EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kFullByteMask);

      const auto writes = vgpr_write_events(*plugin);
      ASSERT_EQ(writes.size(), 1u);
      EXPECT_EQ(writes[0].physical_reg, vb + kDst);
      EXPECT_EQ(writes[0].lane_mask, 1u);
      EXPECT_EQ(writes[0].byte_mask, ExecutionPlugin::kFullByteMask);
    }
  }
}

TEST(ExecutionPluginTest, DppOutOfBoundsObservationHonorsBoundCtrl) {
  ForceScalarOverride force_simd(false);
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  constexpr uint32_t kOldDst = 0xAABBCCDDu;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);

  auto run = [&](bool bound_ctrl) {
    cu->write_vgpr(vb + kSrc, 0, 0x11223344u);
    cu->write_vgpr(vb + kDst, 0, kOldDst);
    uint32_t words[2] = {
        vop1_encode(/*v_mov_b32 opcode=*/1, kDst, amdgpu::SRC_DPP),
        vop1_dpp_word(kSrc, amdgpu::dpp::ROW_SHR1, /*row_mask=*/0xF,
                      /*bank_mask=*/0xF, bound_ctrl),
    };
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    plugin->events.clear();
    EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
  };

  // Lane 0 has no source for row_shr:1. BOUND_CTRL=0 suppresses the
  // destination write entirely, so neither a source read nor destination write
  // is architectural.
  run(false);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), kOldDst);
  EXPECT_TRUE(vgpr_read_events(*plugin).empty());
  EXPECT_TRUE(vgpr_write_events(*plugin).empty());

  // BOUND_CTRL=1 turns the same missing source into zero. No VGPR source is
  // read, but the destination lane is now architecturally written.
  run(true);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0u);
  EXPECT_TRUE(vgpr_read_events(*plugin).empty());
  const auto writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, vb + kDst);
  EXPECT_EQ(writes[0].lane_mask, 1u);
  EXPECT_EQ(writes[0].byte_mask, ExecutionPlugin::kFullByteMask);
}

TEST(ExecutionPluginTest, DppSourceDestinationAliasStagesBeforeWriting) {
  ForceScalarOverride force_simd(false);
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xFu);

  constexpr uint32_t kReg = 5;
  constexpr std::array<uint32_t, 4> kValues{10u, 20u, 30u, 40u};
  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < kValues.size(); ++lane)
    cu->write_vgpr(vb + kReg, lane, kValues[lane]);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  uint32_t words[2] = {
      vop1_encode(/*v_mov_b32 opcode=*/1, kReg, amdgpu::SRC_DPP),
      vop1_dpp_word(kReg, /*quad_perm:[1,0,3,2]=*/0xB1, /*row_mask=*/0x1,
                    /*bank_mask=*/0x1, /*bound_ctrl=*/true),
  };
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  plugin->events.clear();
  EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());

  EXPECT_EQ(cu->read_vgpr_storage(vb + kReg, 0), kValues[1]);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kReg, 1), kValues[0]);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kReg, 2), kValues[3]);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kReg, 3), kValues[2]);

  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kReg);
  EXPECT_EQ(reads[0].lane_mask, 0xFu);
  const auto writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, vb + kReg);
  EXPECT_EQ(writes[0].lane_mask, 0xFu);
}

TEST(ExecutionPluginTest, Dpp8FetchInactiveControlsSourceObservation) {
  ForceScalarOverride force_simd(false);
  Wave32PluginFixture f;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(1u << 1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  constexpr uint32_t kValue = 0x11223344u;
  const uint32_t vb = wf->vgpr_alloc().base;
  const uint32_t lane_sel = (0u << 0u) | (0u << 3u) | (2u << 6u) | (3u << 9u) | (4u << 12u) |
                            (5u << 15u) | (6u << 18u) | (7u << 21u);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  auto run = [&](uint32_t src_marker) {
    f.cu->write_vgpr(vb + kSrc, 0, kValue);
    f.cu->write_vgpr(vb + kDst, 1, 0xAABBCCDDu);
    const uint32_t words[2] = {
        vop1_encode(/*v_mov_b32 opcode=*/1, kDst, src_marker),
        kSrc | (lane_sel << 8u),
    };
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    plugin->events.clear();
    EXPECT_TRUE(f.cu->execute_instruction(inst.get(), *wf).succeeded());
  };

  run(amdgpu::SRC_DPP8_FI_0);
  EXPECT_EQ(f.cu->read_vgpr_storage(vb + kDst, 1), 0u);
  EXPECT_TRUE(vgpr_read_events(*plugin).empty());
  auto writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].lane_mask, 1u << 1);

  run(amdgpu::SRC_DPP8_FI_1);
  EXPECT_EQ(f.cu->read_vgpr_storage(vb + kDst, 1), kValue);
  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
  EXPECT_EQ(reads[0].lane_mask, 1u);
  writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, vb + kDst);
  EXPECT_EQ(writes[0].lane_mask, 1u << 1);
}

// gfx1250 sub-dword and split-backend tests. The true16 cases cover both
// low-to-high and high-to-low half selection while checking that preservation
// of the other half is invisible to plugins. The 64-bit case separately pins
// write forwarding for both physical destination dwords.
TEST(ExecutionPluginTest, True16InstructionsReportSelectedSourceAndDestinationHalves) {
  Wave32PluginFixture f;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  struct Case {
    uint32_t word;
    uint32_t source_value;
    uint32_t expected_destination;
    uint8_t source_byte_mask;
    uint8_t destination_byte_mask;
  };
  constexpr std::array cases{
      Case{0x7F02D300u, 0xAABB00FFu, 0xFF005555u, ExecutionPlugin::kLowHalfByteMask,
           ExecutionPlugin::kHighHalfByteMask}, // v_not_b16_e32 v1.h, v0.l
      Case{0x7E02D380u, 0x00FFAABBu, 0xAAAAFF00u, ExecutionPlugin::kHighHalfByteMask,
           ExecutionPlugin::kLowHalfByteMask}, // v_not_b16_e32 v1.l, v0.h
  };

  const uint32_t vb = wf->vgpr_alloc().base;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  for (const Case &test_case : cases) {
    SCOPED_TRACE(std::format("instruction word 0x{:08x}", test_case.word));
    f.cu->write_vgpr(vb + 0, 0, test_case.source_value);
    f.cu->write_vgpr(vb + 1, 0, 0xAAAA5555u);
    f.cu->write_vgpr(vb + 129, 0, 0xDEADBEEFu);

    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &test_case.word));
    ASSERT_NE(inst, nullptr);
    plugin->events.clear();
    EXPECT_TRUE(f.cu->execute_instruction(inst.get(), *wf).succeeded());

    EXPECT_EQ(f.cu->read_vgpr_storage(vb + 1, 0), test_case.expected_destination);
    EXPECT_EQ(f.cu->read_vgpr_storage(vb + 129, 0), 0xDEADBEEFu);

    const auto reads = vgpr_read_events(*plugin);
    ASSERT_EQ(reads.size(), 1u);
    EXPECT_EQ(reads[0].physical_reg, vb + 0);
    EXPECT_EQ(reads[0].lane_mask, 1u);
    EXPECT_EQ(reads[0].byte_mask, test_case.source_byte_mask);
    const auto writes = vgpr_write_events(*plugin);
    ASSERT_EQ(writes.size(), 1u);
    EXPECT_EQ(writes[0].physical_reg, vb + 1);
    EXPECT_EQ(writes[0].lane_mask, 1u);
    EXPECT_EQ(writes[0].byte_mask, test_case.destination_byte_mask);
  }
}

TEST(ExecutionPluginTest, Rdna2F16TernaryPreservesDestinationWithoutObservingAnExtraRead) {
  if constexpr (!util::has_stdx_simd)
    GTEST_SKIP() << "<experimental/simd> unavailable";
  ForceScalarOverride force_simd(false);
  Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  constexpr uint64_t exec = 0x5;
  wf->set_exec(exec);
  const uint32_t base = wf->vgpr_alloc().base;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA2);
  ASSERT_NE(decoder, nullptr);

  for (auto opcode : {rdna2::kVFmaF16Vop3, rdna2::kVDivFixupF16Vop3}) {
    for (uint8_t dst : {0, 3}) {
      for (bool high : {false, true}) {
        SCOPED_TRACE(std::format("opcode={} dst={} high={}", opcode, dst, high));
        for (unsigned lane = 0; lane < 3; ++lane) {
          f.cu->write_vgpr(base, lane, 0xabcd4000); // 2
          f.cu->write_vgpr(base + 1, lane, 0x4200); // 3
          f.cu->write_vgpr(base + 2, lane, 0x4600); // 6
          f.cu->write_vgpr(base + 3, lane, 0xcece5555);
        }
        const auto words = rdna2::build_vop3(opcode, {.vdst = dst,
                                                      .op_sel = static_cast<uint8_t>(high ? 8 : 0),
                                                      .src0 = 256,
                                                      .src1 = 257,
                                                      .src2 = 258});
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
        ASSERT_NE(inst, nullptr);
        plugin->events.clear();
        amdgpu::ScopedSimdFastPathTracker tracker;
        ASSERT_TRUE(f.cu->execute_instruction(inst.get(), *wf).succeeded());
        const bool fma = opcode == rdna2::kVFmaF16Vop3;
        EXPECT_TRUE(tracker.only_tracked_path_executed(
            fma ? amdgpu::SimdFastPath::VOP3_FMA_FP16 : amdgpu::SimdFastPath::VOP3_TERNARY_FP16));

        std::array<uint64_t, 3> read_lanes{};
        for (const auto &read : vgpr_read_events(*plugin)) {
          ASSERT_GE(read.physical_reg, base);
          const uint32_t source = read.physical_reg - base;
          ASSERT_LT(source, read_lanes.size());
          EXPECT_EQ(read_lanes[source] & read.lane_mask, 0u);
          read_lanes[source] |= read.lane_mask;
        }
        EXPECT_EQ(read_lanes, (std::array{exec, exec, exec}));
        const auto writes = vgpr_write_events(*plugin);
        ASSERT_EQ(writes.size(), 1u);
        EXPECT_EQ(writes[0].physical_reg, base + dst);
        EXPECT_EQ(writes[0].lane_mask, exec);
        EXPECT_EQ(writes[0].byte_mask,
                  high ? ExecutionPlugin::kHighHalfByteMask : ExecutionPlugin::kLowHalfByteMask);

        const uint32_t old = dst == 0 ? 0xabcd4000u : 0xcece5555u;
        const uint32_t result = fma ? 0x4a00u : 0x4000u;
        const uint32_t expected =
            high ? (result << 16) | (old & 0xffffu) : (old & 0xffff0000u) | result;
        EXPECT_EQ(f.cu->read_vgpr_storage(base + dst, 0), expected);
        EXPECT_EQ(f.cu->read_vgpr_storage(base + dst, 1), old);
        EXPECT_EQ(f.cu->read_vgpr_storage(base + dst, 2), expected);
      }
    }
  }
}

TEST(ExecutionPluginTest, Gfx1250Simd64BitWriteReportsBothDestinationRegisters) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    Wave32PluginFixture f;
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto *plugin = f.attach_ordering_plugin();
    auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);

    constexpr uint32_t kSrc = 2;
    constexpr uint32_t kDst = 5;
    const uint32_t vb = wf->vgpr_alloc().base;
    f.cu->write_vgpr(vb + kSrc, 0, 0x11223344u);
    f.cu->write_vgpr(vb + kSrc + 1, 0, 0x55667788u);
    f.cu->write_vgpr(vb + kDst, 0, 0xAABBCCDDu);
    f.cu->write_vgpr(vb + kDst + 1, 0, 0xEEFF0011u);

    const uint32_t word =
        vop1_encode(/*v_mov_b64 opcode=*/29, kDst, /*generic VGPR source=*/256 + kSrc);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &word));
    ASSERT_NE(inst, nullptr);
    plugin->events.clear();
    EXPECT_TRUE(f.cu->execute_instruction(inst.get(), *wf).succeeded());

    EXPECT_EQ(f.cu->read_vgpr_storage(vb + kDst, 0), 0x11223344u);
    EXPECT_EQ(f.cu->read_vgpr_storage(vb + kDst + 1, 0), 0x55667788u);
    expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {kSrc, kSrc + 1}, 1u);
    expect_vgpr_read_set(vgpr_write_events(*plugin), vb, {kDst, kDst + 1}, 1u);
  }
}

TEST(ExecutionPluginTest, DppTrue16SourceReportsSelectedHalf) {
  ForceScalarOverride force_scalar(true);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  Wave32PluginFixture f;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 0;
  constexpr uint32_t kDst = 1;
  const uint32_t vb = wf->vgpr_alloc().base;
  f.cu->write_vgpr(vb + kSrc, 0, 0xAABB00FFu);
  f.cu->write_vgpr(vb + kDst, 0, 0xAAAA5555u);

  cdna5::Vop1VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = 0xE4; // identity quad permutation
  raw.fi = 1;
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;
  cdna5::VNotB16Vop1 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));

  plugin->events.clear();
  inst.execute_impl(*wf);

  EXPECT_EQ(f.cu->read_vgpr_storage(vb + kDst, 0), 0xAAAAFF00u);
  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
  EXPECT_EQ(reads[0].lane_mask, 1u);
  EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kLowHalfByteMask);
}

TEST(ExecutionPluginTest, Rdna4DppTrue16SourceReportsOpSelHalf) {
  ForceScalarOverride force_scalar(true);
  ScopedIsaExecutionBackend execution_backend_scope{&rdna4::execution_backend()};
  Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu.get();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x7);

  constexpr uint32_t kSrc0 = 0;
  constexpr uint32_t kSrc1 = 1;
  constexpr uint32_t kDst = 2;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (bool source_high : {false, true}) {
    SCOPED_TRACE(source_high ? "high source half" : "low source half");
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(
          vb + kSrc0, lane,
          (static_cast<uint32_t>(util::f32_to_f16(static_cast<float>(lane + 20))) << 16) |
              util::f32_to_f16(static_cast<float>(lane + 10)));
      const uint32_t half = util::f32_to_f16(0.5f);
      cu->write_vgpr(vb + kSrc1, lane, half | (half << 16));
      cu->write_vgpr(vb + kDst, lane, (0x7000u + lane) << 16 | (0x4000u + lane));
    }

    rdna4::Vop3VopDpp16MachineInst raw{};
    raw.vdst = kDst;
    raw.opsel = source_high ? 0xBu : 0u;
    raw.op = 0x132u;
    raw.encoding = 0x35u;
    raw.src0 = amdgpu::SRC_DPP;
    raw.src1 = 256u + kSrc1;
    raw.vsrc0 = kSrc0;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = 1;
    raw.bound_ctrl = 0;
    raw.bank_mask = 0xFu;
    raw.row_mask = 0x1u;

    rdna4::VAddF16Vop3 inst(reinterpret_cast<const rdna4::MachineInst *>(&raw));
    plugin->events.clear();
    inst.execute_impl(*wf);

    const float source_base = source_high ? 20.0f : 10.0f;
    const uint32_t lane1_result = util::f32_to_f16(source_base + 0.5f);
    const uint32_t lane2_result = util::f32_to_f16(source_base + 1.5f);
    EXPECT_EQ(cu->read_vgpr(vb + kDst, 1),
              source_high ? (lane1_result << 16) | 0x4001u : (0x7001u << 16) | lane1_result);
    EXPECT_EQ(cu->read_vgpr(vb + kDst, 2),
              source_high ? (lane2_result << 16) | 0x4002u : (0x7002u << 16) | lane2_result);

    std::vector<HookEvent> src0_reads;
    for (const auto &event : vgpr_read_events(*plugin))
      if (event.physical_reg == vb + kSrc0)
        src0_reads.push_back(event);
    ASSERT_EQ(src0_reads.size(), 1u);
    EXPECT_EQ(src0_reads[0].lane_mask, 0x3u);
    EXPECT_EQ(src0_reads[0].byte_mask,
              source_high ? ExecutionPlugin::kHighHalfByteMask : ExecutionPlugin::kLowHalfByteMask);
  }
}

TEST(ExecutionPluginTest, Dpp64BitSourceSimdStagesBothPhysicalDwords) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  ForceScalarOverride force_simd(false);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna4::execution_backend()};
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kSrc, 1, 0x11223344u);
  cu->write_vgpr(vb + kSrc + 1, 1, 0x55667788u);

  cdna4::Vop1VopDppMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHARE_BASE + 1; // row_newbcast:1
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;
  cdna4::VMovB64Vop1 inst(reinterpret_cast<const cdna4::MachineInst *>(&raw));

  plugin->events.clear();
  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0x11223344u);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst + 1, 0), 0x55667788u);
  expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {kSrc, kSrc + 1}, 1u << 1);
}

TEST(ExecutionPluginTest, DppInstructionReuseRestagesOriginalSource) {
  ForceScalarOverride force_scalar(true);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna4::execution_backend()};
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;

  cdna4::Vop1VopDppMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = 0xE4; // identity quad permutation
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;
  cdna4::VMovB32Vop1 inst(reinterpret_cast<const cdna4::MachineInst *>(&raw));

  cu->write_vgpr(vb + kSrc, 0, 0x11111111u);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0x11111111u);

  cu->write_vgpr(vb + kSrc, 0, 0x22222222u);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0x22222222u);
}

TEST(ExecutionPluginTest, SdwaFp8ConversionsIgnoreNumericalOutputModifiers) {
  // CDNA3 section 7.2 and CDNA4 section 7.3: these VOP1 conversions use only
  // the SDWA source register and byte selection; CLAMP and OMOD are ignored.
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    amdgpu::GpuMemory memory("sdwa_fp8_memory");
    amdgpu::L2Cache cache("sdwa_fp8_cache");
    cache.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    std::unique_ptr<amdgpu::ComputeUnitCore> cu;
    ASSERT_NO_FATAL_FAILURE(
        rocjitsu::test::initialize_compute_unit(cu, "sdwa_fp8", config, &memory, &cache));
    std::unique_ptr<Decoder> decoder = Decoder::create(arch);
    amdgpu::Wavefront *wave = cu->dispatch_wf(0, 0, 106, 256);
    ASSERT_NE(wave, nullptr);
    wave->set_mode_raw(0); // Enable ordinary OMOD so the opcode exception is exercised.
    const uint32_t base = wave->vgpr_alloc().base;
    for (uint32_t opcode : {84u, 85u}) {
      // Both encodings represent +2; CDNA3 uses the FNUZ biases.
      const uint32_t input =
          arch == ROCJITSU_CODE_ARCH_CDNA3 ? (opcode == 84 ? 0x48u : 0x44u) : 0x40u;
      for (uint32_t byte = 0; byte < 4; ++byte) {
        for (uint32_t modifiers = 0; modifiers < 8; ++modifiers) {
          SCOPED_TRACE(::testing::Message() << "arch=" << arch << " opcode=" << opcode
                                            << " byte=" << byte << " modifiers=" << modifiers);
          const std::array<uint32_t, 2> words{
              0x7e000000u | (6u << 17) | (opcode << 9) | amdgpu::SRC_SDWA,
              2u | (amdgpu::sdwa::DWORD << 8) | (modifiers << 13) | (byte << 16),
          };
          std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
          ASSERT_NE(instruction, nullptr);
          for (uint64_t exec : {uint64_t{0b101}, ~uint64_t{0}}) {
            wave->set_exec(exec);
            for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
              cu->write_vgpr(base + 2, lane, input << (8 * byte));
              cu->write_vgpr(base + 6, lane, 0xdeadbeefu);
            }
            ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
            for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
              EXPECT_EQ(cu->read_vgpr_storage(base + 6, lane),
                        (exec & (uint64_t{1} << lane)) ? 0x40000000u : 0xdeadbeefu);
          }
        }
      }
    }
    wave->halt();
  }
}

TEST(ExecutionPluginTest, Sdwa64BitDestinationWritesLegalConversionResult) {
  ForceScalarOverride force_scalar(true);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna3::execution_backend()};
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1, /*arch=*/"cdna3"));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint8_t kFp8Lo = 0x34u;
  constexpr uint8_t kFp8Hi = 0x12u;
  cu->write_vgpr(vb + kSrc, 0,
                 (static_cast<uint32_t>(kFp8Hi) << 24) | (static_cast<uint32_t>(kFp8Lo) << 16) |
                     0xABCDu);

  cdna3::Vop1VopSdwaMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.src0_sel = amdgpu::sdwa::WORD_1;
  raw.dst_sel = amdgpu::sdwa::DWORD;
  raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;
  cdna3::VCvtPkF32Fp8Vop1 inst(reinterpret_cast<const cdna3::MachineInst *>(&raw));

  plugin->events.clear();
  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0),
            std::bit_cast<uint32_t>(util::fp8_e4m3_fnuz_to_f32(kFp8Lo)));
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst + 1, 0),
            std::bit_cast<uint32_t>(util::fp8_e4m3_fnuz_to_f32(kFp8Hi)));

  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
  EXPECT_EQ(reads[0].lane_mask, 1u);
  EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kHighHalfByteMask);
  expect_vgpr_read_set(vgpr_write_events(*plugin), vb, {kDst, kDst + 1}, 1u);
}

TEST(ExecutionPluginTest, SdwaInstructionReuseRestagesOriginalSource) {
  ForceScalarOverride force_scalar(true);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna4::execution_backend()};
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;

  cdna4::Vop1VopSdwaMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.src0_sel = amdgpu::sdwa::BYTE_1;
  raw.dst_sel = amdgpu::sdwa::DWORD;
  raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;
  cdna4::VMovB32Vop1 inst(reinterpret_cast<const cdna4::MachineInst *>(&raw));

  cu->write_vgpr(vb + kSrc, 0, 0x11223344u);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0x33u);

  cu->write_vgpr(vb + kSrc, 0, 0xAABBCCDDu);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0xCCu);
}

TEST(ExecutionPluginTest, SdwaFloatingModifiersUseSemanticSourceWidth) {
  for (bool force_scalar : {true, false}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarOverride execution_mode(force_scalar);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);

    constexpr uint32_t kSrc0 = 2;
    constexpr uint32_t kSrc1 = 3;
    constexpr uint32_t kDst = 5;
    const uint32_t vb = wf->vgpr_alloc().base;

    cdna4::Vop2VopSdwaMachineInst add_f32{};
    add_f32.src0 = amdgpu::SRC_SDWA;
    add_f32.vsrc0 = kSrc0;
    add_f32.vsrc1 = kSrc1;
    add_f32.vdst = kDst;
    add_f32.op = cdna4::kVAddF32Vop2;
    add_f32.src0_sel = amdgpu::sdwa::DWORD;
    add_f32.src0_abs = 1;
    add_f32.src1_sel = amdgpu::sdwa::DWORD;
    add_f32.dst_sel = amdgpu::sdwa::DWORD;
    add_f32.dst_unused = amdgpu::sdwa::UNUSED_PAD;

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    std::unique_ptr<Instruction> f32_inst(
        decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&add_f32)));
    ASSERT_NE(f32_inst, nullptr);
    cu->write_vgpr(vb + kSrc0, 0, std::bit_cast<uint32_t>(-2.0f));
    cu->write_vgpr(vb + kSrc1, 0, std::bit_cast<uint32_t>(0.5f));
    EXPECT_TRUE(cu->execute_instruction(f32_inst.get(), *wf).succeeded());
    EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), std::bit_cast<uint32_t>(2.5f));

    cdna4::Vop1VopSdwaMachineInst cvt_bf16{};
    cvt_bf16.src0 = amdgpu::SRC_SDWA;
    cvt_bf16.vsrc0 = kSrc0;
    cvt_bf16.vdst = kDst;
    cvt_bf16.op = cdna4::kVCvtF32Bf16Vop1;
    cvt_bf16.encoding = cdna4::encoding::kVop1 >> 2;
    cvt_bf16.src0_sel = amdgpu::sdwa::WORD_0;
    cvt_bf16.src0_abs = 1;
    cvt_bf16.dst_sel = amdgpu::sdwa::DWORD;
    cvt_bf16.dst_unused = amdgpu::sdwa::UNUSED_PAD;

    std::unique_ptr<Instruction> bf16_inst(
        decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&cvt_bf16)));
    ASSERT_NE(bf16_inst, nullptr);
    ASSERT_EQ(std::string_view(bf16_inst->mnemonic()), "v_cvt_f32_bf16_sdwa");
    cu->write_vgpr(vb + kSrc0, 0, 0xCAFE'C000u);
    EXPECT_TRUE(cu->execute_instruction(bf16_inst.get(), *wf).succeeded());
    EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), std::bit_cast<uint32_t>(2.0f));

    for (uint32_t selection : {amdgpu::sdwa::WORD_0, amdgpu::sdwa::WORD_1}) {
      SCOPED_TRACE(selection);
      cdna4::Vop2VopSdwaMachineInst add_f16{};
      add_f16.src0 = amdgpu::SRC_SDWA;
      add_f16.vsrc0 = kSrc0;
      add_f16.vsrc1 = kSrc1;
      add_f16.vdst = kDst;
      add_f16.op = cdna4::kVAddF16Vop2;
      add_f16.src0_sel = selection;
      add_f16.src0_abs = 1;
      add_f16.src1_sel = selection;
      add_f16.dst_sel = amdgpu::sdwa::DWORD;
      add_f16.dst_unused = amdgpu::sdwa::UNUSED_PAD;

      std::unique_ptr<Instruction> f16_inst(
          decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&add_f16)));
      ASSERT_NE(f16_inst, nullptr);
      const uint32_t shift = selection == amdgpu::sdwa::WORD_1 ? 16u : 0u;
      const uint32_t selected_word_mask = uint32_t{0xFFFF} << shift;
      cu->write_vgpr(vb + kSrc0, 0,
                     (0xCAFE'BEEFu & ~selected_word_mask) | (uint32_t{0xC000} << shift));
      cu->write_vgpr(vb + kSrc1, 0,
                     (0x1234'5678u & ~selected_word_mask) | (uint32_t{0x3800} << shift));
      EXPECT_TRUE(cu->execute_instruction(f16_inst.get(), *wf).succeeded());
      EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), util::f32_to_f16(2.5f));
    }
  }
}

TEST(ExecutionPluginTest, SdwaVop2Src1SelectorReportsExactBytes) {
  ForceScalarOverride force_scalar(true);
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc0 = 2;
  constexpr uint32_t kSrc1 = 3;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kSrc0, 0, 10u);
  cu->write_vgpr(vb + kSrc1, 0, 0x11807F22u);

  cdna4::Vop2VopSdwaMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc0 = kSrc0;
  raw.vsrc1 = kSrc1;
  raw.vdst = kDst;
  raw.op = cdna4::kVAddU32Vop2;
  raw.src0_sel = amdgpu::sdwa::DWORD;
  raw.src1_sel = amdgpu::sdwa::BYTE_2;
  raw.src1_sext = 1;
  raw.dst_sel = amdgpu::sdwa::DWORD;
  raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  std::unique_ptr<Instruction> inst(
      decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
  ASSERT_NE(inst, nullptr);
  plugin->events.clear();
  EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());

  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0xFFFFFF8Au);

  std::vector<HookEvent> src1_reads;
  for (const auto &event : vgpr_read_events(*plugin))
    if (event.physical_reg == vb + kSrc1)
      src1_reads.push_back(event);
  ASSERT_EQ(src1_reads.size(), 1u);
  EXPECT_EQ(src1_reads[0].lane_mask, 1u);
  EXPECT_EQ(src1_reads[0].byte_mask, 0b0100);
}

TEST(ExecutionPluginTest, SdwaVop2ScalarSelectorsUseSgprs) {
  ForceScalarOverride force_scalar(true);
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kVectorSrc = 2;
  constexpr uint32_t kScalarSrc = 4;
  constexpr uint32_t kDst = 5;
  constexpr uint32_t kActiveLane = 3;
  constexpr uint64_t kActiveMask = uint64_t{1} << kActiveLane;
  const uint32_t vb = wf->vgpr_alloc().base;
  const uint32_t sb = wf->sgpr_alloc().base;
  wf->set_exec(kActiveMask);

  auto run = [&](bool scalar_src1) {
    SCOPED_TRACE(scalar_src1 ? "scalar src1" : "scalar src0");
    cu->write_vgpr(vb + kVectorSrc, kActiveLane, 10u);
    cu->write_sgpr(sb + kScalarSrc, 0x11807F22u);

    cdna4::Vop2VopSdwaMachineInst raw{};
    raw.src0 = amdgpu::SRC_SDWA;
    raw.vsrc0 = scalar_src1 ? kVectorSrc : kScalarSrc;
    raw.vsrc1 = scalar_src1 ? kScalarSrc : kVectorSrc;
    raw.vdst = kDst;
    raw.op = cdna4::kVAddU32Vop2;
    raw.src0_sel = scalar_src1 ? amdgpu::sdwa::DWORD : amdgpu::sdwa::BYTE_2;
    raw.src0_sext = scalar_src1 ? 0 : 1;
    raw.s0 = scalar_src1 ? 0 : 1;
    raw.src1_sel = scalar_src1 ? amdgpu::sdwa::BYTE_2 : amdgpu::sdwa::DWORD;
    raw.src1_sext = scalar_src1 ? 1 : 0;
    raw.s1 = scalar_src1 ? 1 : 0;
    raw.dst_sel = amdgpu::sdwa::DWORD;
    raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    std::unique_ptr<Instruction> inst(
        decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
    ASSERT_NE(inst, nullptr);
    plugin->events.clear();
    EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());

    EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, kActiveLane), 0xFFFFFF8Au);
    expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {kVectorSrc}, kActiveMask);

    uint32_t sgpr_reads = 0;
    for (const HookEvent &event : plugin->events)
      sgpr_reads += event.kind == HookEvent::READ_SGPR;
    EXPECT_EQ(sgpr_reads, 1u);
  };

  run(false);
  run(true);
}

TEST(ExecutionPluginTest, SgprWriteObservationUsesExplicitWavePhysicalBlock) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  constexpr uint32_t kRequestedSgprs = 40;
  constexpr uint32_t kLastAllocated = 47;
  auto *wf = cu->dispatch_wf(0, 0, kRequestedSgprs, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  // s_mov_b32 s47, 1 writes the last register in the rounded 48-SGPR allocation.
  constexpr uint32_t kMovTemporary = 0xBEAF0081u;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  auto decoded = decoder->decode(&kMovTemporary);
  ASSERT_TRUE(decoded.succeeded());
  std::unique_ptr<Instruction> inst = std::move(decoded).value();
  ASSERT_NE(inst, nullptr);

  plugin->events.clear();
  EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());

  std::vector<HookEvent> writes;
  for (const HookEvent &event : plugin->events)
    if (event.kind == HookEvent::WRITE_SGPR)
      writes.push_back(event);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, wf->sgpr_alloc().base + kLastAllocated);
  EXPECT_EQ(writes[0].wf_id, wf->wf_id());
  EXPECT_EQ(cu->read_sgpr_storage(writes[0].physical_reg), 1u);

  // The next register still has simulator backing, but no instruction write or
  // plugin event is allowed past the architectural allocation.
  constexpr uint32_t kMovOutside = 0xBEB00081u; // s_mov_b32 s48, 1
  auto outside = decoder->decode(&kMovOutside);
  ASSERT_TRUE(outside.succeeded());
  plugin->events.clear();
  ASSERT_TRUE(cu->execute_instruction(outside.value().get(), *wf).succeeded());
  EXPECT_FALSE(std::ranges::any_of(
      plugin->events, [](const HookEvent &event) { return event.kind == HookEvent::WRITE_SGPR; }));
  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + 48), 0u);
}

// SDWA observation tests cover byte, word, and dword source selectors plus
// preserve, pad, sign-extension, and clamp behavior.
// Selected source bytes must be the only bytes read. Preserved destination
// bytes are storage bookkeeping, while pad/sign-extension and clamp can widen
// the architectural write mask.
TEST(ExecutionPluginTest, SdwaObservationReportsExactSourceAndDestinationBytes) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    constexpr uint32_t kSrc = 2;
    constexpr uint32_t kDst = 5;
    constexpr uint32_t kSrcValue = 0x11223344u;
    constexpr uint32_t kOldDst = 0xAABBCCDDu;
    const uint32_t vb = wf->vgpr_alloc().base;

    struct Case {
      uint32_t dst_sel;
      uint32_t dst_unused;
      uint32_t src0_sel;
      uint32_t source_value;
      uint32_t expected_active;
      uint8_t architectural_byte_mask;
    };
    constexpr std::array cases{
        Case{amdgpu::sdwa::BYTE_1, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::BYTE_2, kSrcValue,
             0xAABB22DDu, 0b0010},
        Case{amdgpu::sdwa::BYTE_3, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::BYTE_3, kSrcValue,
             0x11BBCCDDu, 0b1000},
        Case{amdgpu::sdwa::WORD_0, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::DWORD, 0x12345678u,
             0xAABB5678u, 0b0011},
        Case{amdgpu::sdwa::WORD_1, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::DWORD, kSrcValue,
             0x3344CCDDu, 0b1100},
        Case{amdgpu::sdwa::BYTE_1, amdgpu::sdwa::UNUSED_PAD, amdgpu::sdwa::DWORD, kSrcValue,
             0x00004400u, ExecutionPlugin::kFullByteMask},
        Case{amdgpu::sdwa::BYTE_1, amdgpu::sdwa::UNUSED_SEXT, amdgpu::sdwa::BYTE_0, 0x00000080u,
             0xFFFF8000u, ExecutionPlugin::kFullByteMask},
        Case{amdgpu::sdwa::DWORD, amdgpu::sdwa::UNUSED_PAD, amdgpu::sdwa::DWORD, kSrcValue,
             kSrcValue, ExecutionPlugin::kFullByteMask},
    };

    // Selected sources and destinations carry their precise lane and byte
    // masks. Partial destinations currently use the scalar semantic path, so
    // they may report one exact write event per active lane.
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    for (const Case &test_case : cases) {
      SCOPED_TRACE(test_case.dst_sel);
      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
        cu->write_vgpr(vb + kSrc, lane, test_case.source_value);
        cu->write_vgpr(vb + kDst, lane, kOldDst);
      }

      uint32_t words[2] = {
          vop1_encode(/*opcode=*/1, kDst, amdgpu::SRC_SDWA),
          vop1_sdwa_word(kSrc, test_case.dst_sel, test_case.dst_unused, test_case.src0_sel),
      };
      Instruction *inst = decode_valid(*decoder, words);
      ASSERT_NE(inst, nullptr);
      plugin->events.clear();
      EXPECT_TRUE(cu->execute_instruction(inst, *wf).succeeded());
      delete inst;

      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
        const uint32_t expected =
            (kPartialExecMask & (uint64_t{1} << lane)) ? test_case.expected_active : kOldDst;
        EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, lane), expected) << "lane " << lane;
      }

      const auto writes = vgpr_write_events(*plugin);
      ASSERT_FALSE(writes.empty());
      uint64_t observed_write_lanes = 0;
      for (const auto &write : writes) {
        EXPECT_EQ(write.physical_reg, vb + kDst);
        EXPECT_EQ(write.byte_mask, test_case.architectural_byte_mask);
        observed_write_lanes |= write.lane_mask;
      }
      EXPECT_EQ(observed_write_lanes, kPartialExecMask);

      const auto reads = vgpr_read_events(*plugin);
      ASSERT_FALSE(reads.empty());
      uint64_t observed_read_lanes = 0;
      for (const auto &read : reads) {
        EXPECT_EQ(read.physical_reg, vb + kSrc);
        EXPECT_EQ(read.byte_mask, amdgpu::sdwa::sdwa_src_byte_mask(test_case.src0_sel));
        observed_read_lanes |= read.lane_mask;
      }
      EXPECT_EQ(observed_read_lanes, kPartialExecMask);
    }
  }
}

TEST(ExecutionPluginTest, SdwaClampIsAppliedInsideArchitecturalDestinationWrite) {
  ForceScalarOverride force_simd(false);
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kSrc, 0, std::bit_cast<uint32_t>(0.5f));
  cu->write_vgpr(vb + kDst, 0, 0xDEADBEEFu);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  uint32_t words[2] = {
      vop1_encode(/*v_rcp_f32 opcode=*/34, kDst, amdgpu::SRC_SDWA),
      vop1_sdwa_word(kSrc, amdgpu::sdwa::DWORD, amdgpu::sdwa::UNUSED_PAD, amdgpu::sdwa::DWORD,
                     /*clamp=*/true),
  };
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  plugin->events.clear();
  EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());

  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), std::bit_cast<uint32_t>(1.0f));
  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
  EXPECT_EQ(reads[0].lane_mask, 1u);
  EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kFullByteMask);
  const auto writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, vb + kDst);
  EXPECT_EQ(writes[0].lane_mask, 1u);
  EXPECT_EQ(writes[0].byte_mask, ExecutionPlugin::kFullByteMask);
}

TEST(ExecutionPluginTest, SdwaF16ClampPrecedesDestinationPlacement) {
  constexpr uint32_t kSrc0 = 2;
  constexpr uint32_t kSrc1 = 3;
  constexpr uint32_t kDst = 5;
  constexpr uint32_t kOldDst = 0xBEEFBEEFu;
  struct Case {
    uint16_t source;
    uint16_t expected;
    bool dx10_clamp;
  };
  constexpr std::array cases{
      Case{0x4000, 0x3C00, false}, // 2 + 2 saturates to one.
      Case{0xB800, 0, false},      // Negative result saturates to positive zero.
      Case{0x3000, 0x3400, false}, // Interior value stays unchanged.
      Case{0x8000, 0, false},      // Negative zero clamps to positive zero.
      Case{0x7C00, 0x3C00, false}, // Positive infinity saturates to one.
      Case{0xFC00, 0, false},      // Negative infinity saturates to zero.
      Case{0x7E55, 0x7E55, false}, // NaN payload survives without DX10_CLAMP.
      Case{0x7E55, 0, true},
  };
  for (bool force_scalar : {false, true}) {
    ForceScalarOverride scalar_override(force_scalar);
    PluginFixture fixture;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(/*num_wf_slots=*/1));
    OrderingPlugin *plugin = fixture.attach_ordering_plugin();
    amdgpu::ComputeUnitCore *compute_unit = fixture.cu();
    amdgpu::Wavefront *wavefront = compute_unit->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wavefront, nullptr);
    constexpr uint64_t kActiveLanes = 0b101;
    wavefront->set_exec(kActiveLanes);
    const uint32_t register_base = wavefront->vgpr_alloc().base;
    std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    for (uint32_t selection : {amdgpu::sdwa::WORD_0, amdgpu::sdwa::WORD_1, amdgpu::sdwa::DWORD}) {
      cdna4::Vop2VopSdwaMachineInst encoding{};
      encoding.src0 = amdgpu::SRC_SDWA;
      encoding.vsrc0 = kSrc0;
      encoding.vsrc1 = kSrc1;
      encoding.vdst = kDst;
      encoding.op = cdna4::kVAddF16Vop2;
      encoding.src0_sel = amdgpu::sdwa::WORD_0;
      encoding.src1_sel = amdgpu::sdwa::WORD_0;
      encoding.dst_sel = selection;
      encoding.dst_unused = amdgpu::sdwa::UNUSED_PRESERVE;
      encoding.clamp = 1;
      std::unique_ptr<Instruction> instruction(
          decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&encoding)));
      ASSERT_NE(instruction, nullptr);
      for (const Case &test_case : cases) {
        SCOPED_TRACE(test_case.source);
        SCOPED_TRACE(selection);
        wavefront->set_mode_raw(0xF0u |
                                (test_case.dx10_clamp ? amdgpu::Wavefront::DX10_CLAMP_BIT : 0u));
        for (uint32_t lane = 0; lane < wavefront->wf_size(); ++lane) {
          compute_unit->write_vgpr(register_base + kSrc0, lane, test_case.source);
          compute_unit->write_vgpr(register_base + kSrc1, lane, test_case.source);
          compute_unit->write_vgpr(register_base + kDst, lane, kOldDst);
        }
        plugin->events.clear();
        EXPECT_TRUE(compute_unit->execute_instruction(instruction.get(), *wavefront).succeeded());
        const uint32_t expected = selection == amdgpu::sdwa::DWORD ? test_case.expected
                                  : selection == amdgpu::sdwa::WORD_0
                                      ? 0xBEEF0000u | test_case.expected
                                      : (uint32_t{test_case.expected} << 16) | 0xBEEFu;
        for (uint32_t lane = 0; lane < wavefront->wf_size(); ++lane)
          EXPECT_EQ(compute_unit->read_vgpr_storage(register_base + kDst, lane),
                    (kActiveLanes & (uint64_t{1} << lane)) ? expected : kOldDst);
        const std::vector<HookEvent> writes = vgpr_write_events(*plugin);
        uint64_t written_lanes = 0;
        for (const HookEvent &write : writes) {
          EXPECT_EQ(write.physical_reg, register_base + kDst);
          EXPECT_EQ(write.byte_mask, selection == amdgpu::sdwa::DWORD    ? 0b1111
                                     : selection == amdgpu::sdwa::WORD_0 ? 0b0011
                                                                         : 0b1100);
          written_lanes |= write.lane_mask;
        }
        EXPECT_EQ(written_lanes, kActiveLanes);
      }
    }
  }
}

TEST(ExecutionPluginTest, SdwaPackedF16ClampTreatsEachHalfIndependently) {
  for (bool force_scalar : {false, true}) {
    ForceScalarOverride scalar_override(force_scalar);
    PluginFixture fixture;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(/*num_wf_slots=*/1));
    amdgpu::ComputeUnitCore *compute_unit = fixture.cu();
    amdgpu::Wavefront *wavefront = compute_unit->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wavefront, nullptr);
    wavefront->set_exec(1);
    wavefront->set_mode_raw(0xF0);
    const uint32_t register_base = wavefront->vgpr_alloc().base;
    std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    cdna4::Vop2VopSdwaMachineInst encoding{};
    encoding.src0 = amdgpu::SRC_SDWA;
    encoding.vsrc0 = 2;
    encoding.vsrc1 = 3;
    encoding.vdst = 5;
    encoding.op = cdna4::kVPkFmacF16Vop2;
    encoding.src0_sel = amdgpu::sdwa::DWORD;
    encoding.src1_sel = amdgpu::sdwa::DWORD;
    encoding.dst_sel = amdgpu::sdwa::DWORD;
    encoding.clamp = 1;
    std::unique_ptr<Instruction> instruction(
        decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&encoding)));
    ASSERT_NE(instruction, nullptr);
    // Each half computes its source times one plus zero. Saturate the low
    // result while preserving the high result, then exercise the reverse.
    for (bool high_saturates : {false, true}) {
      compute_unit->write_vgpr(register_base + 2, 0, high_saturates ? 0x40003800u : 0x38004000u);
      compute_unit->write_vgpr(register_base + 3, 0, 0x3C003C00u);
      compute_unit->write_vgpr(register_base + 5, 0, 0);
      EXPECT_TRUE(compute_unit->execute_instruction(instruction.get(), *wavefront).succeeded());
      EXPECT_EQ(compute_unit->read_vgpr_storage(register_base + 5, 0),
                high_saturates ? 0x3C003800u : 0x38003C00u);
    }
  }
}

TEST(ExecutionPluginTest, SdwaPackedConversionClampsBothHalvesBeforePlacement) {
  constexpr uint32_t kSrc0 = 2;
  constexpr uint32_t kSrc1 = 3;
  constexpr uint32_t kDst = 5;
  constexpr uint32_t kOldDst = 0xBEEFBEEFu;
  struct Case {
    float src0;
    float src1;
    uint32_t unclamped;
    uint32_t clamped;
  };
  constexpr std::array cases{
      Case{2.0f, -0.5f, 0xB8004000u, 0x00003C00u},
      Case{-0.5f, 2.0f, 0x4000B800u, 0x3C000000u},
      Case{2.0f, 0.5f, 0x38004000u, 0x38003C00u},
      Case{0.5f, 2.0f, 0x40003800u, 0x3C003800u},
  };
  for (const auto &[arch, arch_name] : {std::pair{ROCJITSU_CODE_ARCH_RDNA1, "rdna1"},
                                        std::pair{ROCJITSU_CODE_ARCH_RDNA2, "rdna2"}}) {
    SCOPED_TRACE(arch_name);
    for (bool force_scalar : {false, true}) {
      ForceScalarOverride scalar_override(force_scalar);
      PluginFixture fixture;
      ASSERT_NO_FATAL_FAILURE(
          fixture.initialize(/*num_wf_slots=*/1, arch_name, /*wavefront_size=*/32));
      OrderingPlugin *plugin = fixture.attach_ordering_plugin();
      amdgpu::ComputeUnitCore *cu = fixture.cu();
      amdgpu::Wavefront *wave = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
      ASSERT_NE(wave, nullptr);
      wave->set_mode_raw(0xF0u);
      const uint32_t base = wave->vgpr_alloc().base;
      std::unique_ptr<Decoder> decoder = Decoder::create(arch);
      for (bool clamp : {false, true}) {
        SCOPED_TRACE(clamp);
        for (uint32_t selection :
             {amdgpu::sdwa::DWORD, amdgpu::sdwa::WORD_0, amdgpu::sdwa::WORD_1}) {
          SCOPED_TRACE(selection);
          // RDNA1/2 V_CVT_PKRTZ_F16_F32 uses VOP2 opcode 47 and the same SDWA layout.
          const std::array<uint32_t, 2> words{
              vop2_encode(47, kDst, kSrc1, amdgpu::SRC_SDWA),
              vop1_sdwa_word(kSrc0, selection, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::DWORD,
                             clamp) |
                  (amdgpu::sdwa::DWORD << 24),
          };
          std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
          ASSERT_NE(instruction, nullptr);
          for (const Case &test_case : cases) {
            SCOPED_TRACE(test_case.unclamped);
            const uint32_t result = clamp ? test_case.clamped : test_case.unclamped;
            const uint32_t expected = selection == amdgpu::sdwa::DWORD ? result
                                      : selection == amdgpu::sdwa::WORD_0
                                          ? 0xBEEF0000u | (result & 0xFFFFu)
                                          : (result << 16) | 0xBEEFu;
            for (uint64_t exec : {uint64_t{0b101}, uint64_t{0xFFFFFFFFu}}) {
              wave->set_exec(exec);
              for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
                cu->write_vgpr(base + kSrc0, lane, std::bit_cast<uint32_t>(test_case.src0));
                cu->write_vgpr(base + kSrc1, lane, std::bit_cast<uint32_t>(test_case.src1));
                cu->write_vgpr(base + kDst, lane, kOldDst);
              }
              plugin->events.clear();
              ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
              for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
                EXPECT_EQ(cu->read_vgpr_storage(base + kDst, lane),
                          (exec & (uint64_t{1} << lane)) ? expected : kOldDst);
              uint64_t written_lanes = 0;
              for (const HookEvent &write : vgpr_write_events(*plugin)) {
                EXPECT_EQ(write.physical_reg, base + kDst);
                EXPECT_EQ(write.byte_mask, selection == amdgpu::sdwa::DWORD    ? 0b1111
                                           : selection == amdgpu::sdwa::WORD_0 ? 0b0011
                                                                               : 0b1100);
                written_lanes |= write.lane_mask;
              }
              EXPECT_EQ(written_lanes, exec);
              for (const HookEvent &read : vgpr_read_events(*plugin))
                EXPECT_NE(read.physical_reg, base + kDst);
            }
          }
        }
      }
    }
  }
}

TEST(ExecutionPluginTest, SdwaIntegerResultsDoNotUseFloatingClamp) {
  for (bool force_scalar : {false, true}) {
    ForceScalarOverride scalar_override(force_scalar);
    PluginFixture fixture;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(/*num_wf_slots=*/1));
    amdgpu::ComputeUnitCore *compute_unit = fixture.cu();
    amdgpu::Wavefront *wavefront = compute_unit->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wavefront, nullptr);
    wavefront->set_exec(1);
    wavefront->set_mode_raw(0xF0u | amdgpu::Wavefront::DX10_CLAMP_BIT);
    const uint32_t register_base = wavefront->vgpr_alloc().base;
    std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    struct Case {
      uint32_t encoding;
      uint32_t modifiers;
      uint32_t source;
      uint32_t result;
    };
    // LLVM gfx950 encodings: FREXP_EXP_I32_F32 with DWORD input and the
    // normalized I16/U16 conversions with WORD_0 input. Each has CLAMP set.
    constexpr std::array cases{
        Case{0x7E0A66F9u, 0x00062602u, std::bit_cast<uint32_t>(0.25f), 0xFFFFFFFFu},
        Case{0x7E0A66F9u, 0x00062602u, std::bit_cast<uint32_t>(0.125f), 0xFFFFFFFEu},
        Case{0x7E0A66F9u, 0x00062602u, std::bit_cast<uint32_t>(8.0f), 4u},
        Case{0x7E0A9AF9u, 0x00042602u, 0x3C00u, 0x7FFFu},
        Case{0x7E0A9CF9u, 0x00042602u, 0x3C00u, 0xFFFFu},
    };
    for (const Case &test_case : cases) {
      const uint32_t encoding[] = {test_case.encoding, test_case.modifiers};
      std::unique_ptr<Instruction> instruction(decode_valid(*decoder, encoding));
      ASSERT_NE(instruction, nullptr);
      compute_unit->write_vgpr(register_base + 2, 0, test_case.source);
      compute_unit->write_vgpr(register_base + 5, 0, 0xDEADBEEFu);
      EXPECT_TRUE(compute_unit->execute_instruction(instruction.get(), *wavefront).succeeded());
      // These integer bit patterns must not be interpreted as floating NaNs.
      EXPECT_EQ(compute_unit->read_vgpr_storage(register_base + 5, 0), test_case.result);
    }
  }
}

TEST(ExecutionPluginTest, SdwaClampHonorsDx10ClampMode) {
  ForceScalarOverride force_simd(false);
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  constexpr uint32_t kQuietNan = 0x7FC12345u;
  const uint32_t vb = wf->vgpr_alloc().base;

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  uint32_t words[2] = {
      vop1_encode(/*v_rcp_f32 opcode=*/34, kDst, amdgpu::SRC_SDWA),
      vop1_sdwa_word(kSrc, amdgpu::sdwa::DWORD, amdgpu::sdwa::UNUSED_PAD, amdgpu::sdwa::DWORD,
                     /*clamp=*/true),
  };
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);

  struct Case {
    uint32_t mode;
    uint32_t expected;
  };
  constexpr std::array cases{
      Case{0, kQuietNan},
      Case{amdgpu::Wavefront::DX10_CLAMP_BIT, std::bit_cast<uint32_t>(0.0f)},
  };
  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.mode);
    wf->set_mode_raw(test_case.mode);
    cu->write_vgpr(vb + kSrc, 0, kQuietNan);
    cu->write_vgpr(vb + kDst, 0, 0xDEADBEEFu);
    EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), test_case.expected);
  }
}

TEST(ExecutionPluginTest, SdwaPartialPreserveClampReportsOnlySelectedBytes) {
  ForceScalarOverride force_simd(false);
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kSrc, 0, std::bit_cast<uint32_t>(0.5f));
  // Clamp the reciprocal to 1.0f before BYTE_1 selects its low byte. The
  // original destination bytes must not become part of the numeric result.
  cu->write_vgpr(vb + kDst, 0, 0x4000CC00u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  uint32_t words[2] = {
      vop1_encode(/*v_rcp_f32 opcode=*/34, kDst, amdgpu::SRC_SDWA),
      vop1_sdwa_word(kSrc, amdgpu::sdwa::BYTE_1, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::DWORD,
                     /*clamp=*/true),
  };
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  plugin->events.clear();
  EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());

  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0x40000000u);
  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
  EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kFullByteMask);
  const auto writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, vb + kDst);
  EXPECT_EQ(writes[0].lane_mask, 1u);
  EXPECT_EQ(writes[0].byte_mask, 0b0010);
}

TEST(ExecutionPluginTest, SgprBlockOwnersTrackPhysicalAllocationAndSlotReuseAcrossTargets) {
  struct TargetCase {
    std::string_view arch;
    uint32_t wave_size;
    uint32_t sgprs_per_wf;
  };

  for (const TargetCase target : {TargetCase{"cdna4", 64, 104}, TargetCase{"cdna5", 32, 128}}) {
    SCOPED_TRACE(target.arch);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(
        f.initialize(/*num_wf_slots=*/3, target.arch, target.wave_size, target.sgprs_per_wf));
    auto *plugin = f.attach_ordering_plugin();
    const uint32_t first_sgpr_count = target.sgprs_per_wf / 2;
    auto *first = f.cu()->dispatch_wf(/*wg_id=*/10, /*pc=*/0, first_sgpr_count,
                                      /*vgprs=*/32);
    auto *second = f.cu()->dispatch_wf(/*wg_id=*/20, /*pc=*/0, target.sgprs_per_wf,
                                       /*vgprs=*/32);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(first->sgpr_alloc().base, 0u);
    ASSERT_EQ(second->sgpr_alloc().base, target.sgprs_per_wf);

    auto expect_owner = [&](const Wavefront &owner, uint32_t physical_reg) {
      plugin->events.clear();
      EXPECT_EQ(f.cu()->read_sgpr(physical_reg), 0u);
      ASSERT_EQ(plugin->events.size(), 1u);
      EXPECT_EQ(plugin->events[0].kind, HookEvent::READ_SGPR);
      EXPECT_EQ(plugin->events[0].wf_id, owner.wf_id());
      EXPECT_EQ(plugin->events[0].wg_id, owner.wg_id());
      EXPECT_EQ(plugin->events[0].physical_reg, physical_reg);
    };

    expect_owner(*first, first->sgpr_alloc().base + first_sgpr_count - 1);
    expect_owner(*second, second->sgpr_alloc().base);
    expect_owner(*first, first->sgpr_alloc().base + first_sgpr_count);

    const uint32_t reused_base = first->sgpr_alloc().base;
    const uint32_t reused_slot = first->wf_id();
    first->halt(Wavefront::CpCompletionNotice::Suppress);
    EXPECT_EQ(test::ComputeUnitTestAccess::sgpr_owner(*f.cu(), reused_base), nullptr);
    auto *reused = f.cu()->dispatch_wf(/*wg_id=*/30, /*pc=*/0, target.sgprs_per_wf,
                                       /*vgprs=*/32);
    ASSERT_NE(reused, nullptr);
    EXPECT_EQ(reused->wf_id(), reused_slot);
    EXPECT_EQ(reused->sgpr_alloc().base, reused_base);
    expect_owner(*reused, reused_base + target.sgprs_per_wf - 1);
    expect_owner(*second, second->sgpr_alloc().base + target.sgprs_per_wf - 1);
  }
}

TEST(ExecutionPluginTest, VopdIntegerSimdRejectsUnsupportedWaveAndModifiers) {
  ForceScalarOverride execution_mode(false);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  cdna5::Operand operand(32, cdna5::OperandType::OPR_VGPR, 0);
  struct Slot {
    uint16_t op = 8;
    Operand *dst;
    Operand *src0;
    Operand *src1;
    uint8_t neg = 0;
    bool has_src2_operand = false;
    bool src2_is_imm = false;
  } slot{8, &operand, &operand, &operand};
  const auto try_pair = [](Wavefront &wf, const Slot &x, const Slot &y) {
    return try_execute_vopd_integer_pair_simd<8, 16, 17>(wf, x, y);
  };

  PluginFixture wave64;
  ASSERT_NO_FATAL_FAILURE(wave64.initialize(/*num_wf_slots=*/1));
  auto *wide = wave64.cu()->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(wide, nullptr);
  ASSERT_EQ(wide->wf_size(), 64u);
  wide->set_exec(0);
  EXPECT_FALSE(try_pair(*wide, slot, slot));

  Wave32PluginFixture wave32;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *wf = wave32.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0);
  EXPECT_EQ(try_pair(*wf, slot, slot), util::has_stdx_simd);
  for (uint32_t modifier = 0; modifier < 3; ++modifier) {
    Slot unsupported = slot;
    unsupported.neg = modifier == 0 ? 1 : 0;
    unsupported.has_src2_operand = modifier == 1;
    unsupported.src2_is_imm = modifier == 2;
    EXPECT_FALSE(try_pair(*wf, unsupported, slot));
    EXPECT_FALSE(try_pair(*wf, slot, unsupported));
  }
}

TEST(ExecutionPluginTest, VopdIntegerPairsPreserveMasksAliasesAndObservation) {
  constexpr uint32_t kMov = 8;
  constexpr uint32_t kAdd = 16;
  constexpr uint32_t kShift = 17;
  constexpr uint32_t kDstX = 2;
  constexpr uint32_t kDstY = 3;
  constexpr uint32_t kInlineZero = 128;
  constexpr uint32_t kInlineThirtyTwo = 160;
  constexpr uint32_t kVgprSrcBase = 256;
  // Cover symmetric and one-sided aliases through each source position.
  struct Sources {
    uint32_t x0, x1, y0, y1;
  };
  const Sources sources[] = {
      {256 + kDstY, 4, 256 + kDstX, 5},      {256 + kDstY, 4, 256 + 7, 5},
      {256 + 6, 4, 256 + kDstX, 5},          {256 + 6, kDstY, 256 + 7, kDstX},
      {256 + 6, kDstY, 256 + 7, 5},          {256 + 6, 4, 256 + 7, kDstX},
      {kInlineZero, 4, kInlineThirtyTwo, 5},
  };
  const auto initial_value = [](uint32_t reg, uint32_t lane) {
    switch (reg) {
    case kDstX:
      return 0xFFFFFF00u + lane;
    case kDstY:
      return 40u + lane;
    case 4:
      return 0x01010101u * (lane + 1);
    case 5:
      return 0x80000000u + lane;
    default:
      return 3u * reg + lane;
    }
  };
  const auto evaluate = [](uint32_t op, uint32_t src0, uint32_t src1) {
    if (op == kMov)
      return src0;
    if (op == kAdd)
      return src0 + src1;
    return src1 << (src0 & 31u);
  };
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(static_cast<int>(arch));
    for (bool force_scalar : {false, true}) {
      SCOPED_TRACE(force_scalar);
      ForceScalarOverride execution_mode(force_scalar);
      Wave32PluginFixture f(arch);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      auto *plugin = f.attach_ordering_plugin();
      auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/32);
      ASSERT_NE(wf, nullptr);
      ASSERT_EQ(wf->wf_size(), 32u);
      const uint32_t base = wf->vgpr_alloc().base;
      auto decoder = Decoder::create(arch);
      ASSERT_NE(decoder, nullptr);
      for (bool vopd3 : {false, true}) {
        if (vopd3 && arch != ROCJITSU_CODE_ARCH_CDNA5)
          continue;
        SCOPED_TRACE(vopd3);
        for (uint32_t x_op : {kMov, kAdd, kShift}) {
          // Classic VOPD has no encoding for ADD/LSHL in the X slot.
          if (!vopd3 && x_op != kMov)
            continue;
          for (uint32_t y_op : {kMov, kAdd, kShift}) {
            SCOPED_TRACE(x_op);
            SCOPED_TRACE(y_op);
            for (const auto &src : sources) {
              SCOPED_TRACE(std::format("x=({}, {}) y=({}, {})", src.x0, src.x1, src.y0, src.y1));
              const uint32_t x_src0 = src.x0;
              const uint32_t y_src0 = src.y0;
              const std::array<uint32_t, 3> words =
                  vopd3
                      ? std::array<uint32_t, 3>{0xCF000000u | (x_op << 18) | (y_op << 12) | x_src0,
                                                y_src0 | (src.x1 << 16),
                                                kDstX | (src.y1 << 8) | (kDstY << 24)}
                      : std::array<uint32_t, 3>{
                            (0x32u << 26) | (x_op << 22) | (y_op << 17) | (src.x1 << 9) | x_src0,
                            y_src0 | (src.y1 << 9) | ((kDstY >> 1) << 17) | (kDstX << 24), 0};
              std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
              ASSERT_NE(inst, nullptr);
              for (uint64_t exec : {0u, 0xA5010081u, 0xFFFFFFFFu}) {
                SCOPED_TRACE(exec);
                wf->set_exec(exec);
                for (uint32_t reg = 0; reg < 8; ++reg)
                  for (uint32_t lane = 0; lane < 32; ++lane)
                    f.cu->write_vgpr(base + reg, lane, initial_value(reg, lane));
                plugin->events.clear();
                ASSERT_TRUE(f.cu->execute_instruction(inst.get(), *wf).succeeded());
                for (uint32_t lane = 0; lane < 32; ++lane) {
                  const uint32_t old_x = 0xFFFFFF00u + lane;
                  const uint32_t old_y = 40u + lane;
                  const auto source0_value = [&](uint32_t selector) {
                    return selector >= kVgprSrcBase ? initial_value(selector - kVgprSrcBase, lane)
                                                    : selector - kInlineZero;
                  };
                  const uint32_t expected_x =
                      (exec & (uint64_t{1} << lane))
                          ? evaluate(x_op, source0_value(src.x0), initial_value(src.x1, lane))
                          : old_x;
                  const uint32_t expected_y =
                      (exec & (uint64_t{1} << lane))
                          ? evaluate(y_op, source0_value(src.y0), initial_value(src.y1, lane))
                          : old_y;
                  EXPECT_EQ(f.cu->read_vgpr_storage(base + kDstX, lane), expected_x) << lane;
                  EXPECT_EQ(f.cu->read_vgpr_storage(base + kDstY, lane), expected_y) << lane;
                }
                std::map<uint32_t, uint64_t> reads;
                std::map<uint32_t, uint64_t> writes;
                uint32_t observed_read_lanes = 0;
                uint32_t observed_write_lanes = 0;
                for (const auto &event : vgpr_read_events(*plugin)) {
                  EXPECT_EQ(event.byte_mask, ExecutionPlugin::kFullByteMask);
                  if (util::has_stdx_simd && !force_scalar) {
                    EXPECT_EQ(event.lane_mask, exec);
                  }
                  reads[event.physical_reg - base] |= event.lane_mask;
                  observed_read_lanes += std::popcount(event.lane_mask);
                }
                for (const auto &event : vgpr_write_events(*plugin)) {
                  EXPECT_EQ(event.byte_mask, ExecutionPlugin::kFullByteMask);
                  if (util::has_stdx_simd && !force_scalar) {
                    EXPECT_EQ(event.lane_mask, exec);
                  }
                  writes[event.physical_reg - base] |= event.lane_mask;
                  observed_write_lanes += std::popcount(event.lane_mask);
                }
                std::map<uint32_t, uint64_t> expected_reads;
                std::map<uint32_t, uint64_t> expected_writes;
                if (exec != 0) {
                  if (src.x0 >= kVgprSrcBase)
                    expected_reads[src.x0 - kVgprSrcBase] = exec;
                  if (src.y0 >= kVgprSrcBase)
                    expected_reads[src.y0 - kVgprSrcBase] = exec;
                  if (x_op != kMov)
                    expected_reads[src.x1] = exec;
                  if (y_op != kMov)
                    expected_reads[src.y1] = exec;
                  expected_writes = {{kDstX, exec}, {kDstY, exec}};
                }
                if (util::has_stdx_simd && !force_scalar) {
                  // SIMD views report once per operand, not once per active lane.
                  EXPECT_EQ(vgpr_read_events(*plugin).size(), expected_reads.size());
                  EXPECT_EQ(vgpr_write_events(*plugin).size(), expected_writes.size());
                }
                EXPECT_EQ(observed_read_lanes, expected_reads.size() * std::popcount(exec));
                EXPECT_EQ(observed_write_lanes, expected_writes.size() * std::popcount(exec));
                EXPECT_EQ(reads, expected_reads);
                EXPECT_EQ(writes, expected_writes);
              }
            }
          }
        }
      }
    }
  }
}

TEST(ExecutionPluginTest, DisabledWaitCheckingPreservesRegisterHooksAcrossPluginReplacement) {
  Wave32PluginFixture f(ROCJITSU_CODE_ARCH_CDNA5, amdgpu::MemoryWaitDiagnostics::Off);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *wf = f.cu->dispatch_wf(0, 0, 104, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  const amdgpu::RegisterAccess regs(*wf);
  const uint32_t vbase = wf->vgpr_alloc().base + 4;
  const uint32_t sbase = wf->sgpr_alloc().base + 2;
  auto access = [&](uint32_t value) {
    auto write = regs.write_vgpr_region(vbase, 2, 1);
    write.set_lane(0, 0, value);
    write.set_lane(1, 0, value + 1);
    auto read = regs.read_vgpr_region(vbase, 2, 1);
    EXPECT_EQ(read.lane(0, 0), value);
    EXPECT_EQ(read.lane(1, 0), value + 1);
    const uint64_t pair = uint64_t{value} | (uint64_t{value + 1} << 32);
    ASSERT_TRUE(regs.write_sgpr64(sbase, pair).succeeded());
    EXPECT_EQ(regs.read_sgpr64(sbase), pair);
  };
  access(3);
  auto *plugin = f.attach_ordering_plugin();
  for (bool attached : {true, false, true}) {
    SCOPED_TRACE(attached);
    f.cu->set_plugin_group(attached ? f.plugin_group : nullptr);
    plugin->events.clear();
    access(7);
    if (!attached) {
      EXPECT_TRUE(plugin->events.empty());
      continue;
    }
    for (auto kind :
         {HookEvent::READ_VGPR, HookEvent::WRITE_VGPR, HookEvent::READ_SGPR, HookEvent::WRITE_SGPR})
      EXPECT_EQ(std::ranges::count(plugin->events, kind, &HookEvent::kind), 2);
  }
}

TEST(ExecutionPluginTest, RegisterObserverSnapshotsPreservePendingWaits) {
  class SnapshotPlugin : public ExecutionPlugin {
  public:
    SnapshotPlugin() : ExecutionPlugin("register_snapshot") {}
    void snapshot(const Wavefront *wf) {
      (void)wf->read_scc();
      ++snapshots;
    }
    void onAmdgpuReadScalarRegister(const Wavefront *wf, RegisterRef) override { snapshot(wf); }
    void onAmdgpuWriteScalarRegister(const Wavefront *wf, RegisterRef) override { snapshot(wf); }
    void onAmdgpuReadVgprLanes(const Wavefront *wf, uint32_t, uint64_t, uint8_t) override {
      snapshot(wf);
    }
    void onAmdgpuWriteVgprLanes(const Wavefront *wf, uint32_t, uint64_t, uint8_t) override {
      snapshot(wf);
    }
    void onAmdgpuTensorDmaMemoryAccess(const amdgpu::TensorDmaMemoryAccessObservation &) override {
      snapshot(tensor_wave);
    }
    bool observes_tensor_dma_memory_access() const override { return true; }
    const Wavefront *tensor_wave = nullptr;
    unsigned snapshots = 0;
  };
  Wave32PluginFixture f;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<SnapshotPlugin>();
  auto *snapshots = plugin.get();
  ASSERT_TRUE(group->add(std::move(plugin)));
  f.cu->set_plugin_group(group);
  auto *wf = f.cu->dispatch_wf(0, 0x200, 104, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  snapshots->tensor_wave = wf;
  const amdgpu::RegisterAccess regs(*wf);
  const uint32_t sbase = wf->sgpr_alloc().base;
  const uint32_t vbase = wf->vgpr_alloc().base;
  f.cu->write_sgpr(sbase, 0);
  f.cu->write_vgpr(vbase, 0, 0);
  auto &state = wf->ensure_memory_wait_scoreboard();
  unsigned hazards = 0;
  state.bind(wf->pc, &hazards, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
  for (unsigned hook = 0; hook < 6; ++hook) {
    SCOPED_TRACE(hook);
    state.clear();
    hazards = snapshots->snapshots = 0;
    state.add({state.issue(WaitCounterKind::Km),
               0x100,
               ~uint64_t{0},
               {RegClass::SCC, 0, 1},
               WaitCounterKind::Km,
               0xf});
    switch (hook) {
    case 0:
      (void)regs.read_sgpr(sbase);
      break;
    case 1:
      ASSERT_TRUE(regs.write_sgpr(sbase, 7).succeeded());
      break;
    case 2:
      (void)regs.read_vgpr_region(vbase, 1, 1).lane(0, 0);
      break;
    case 3:
      regs.write_vgpr_region(vbase, 1, 1).set_lane(0, 0, 7);
      break;
    case 4:
      f.cu->notify_scalar_lane_vgpr_write(wf, vbase, 1);
      break;
    case 5:
      wf->cu().report_tensor_dma_memory_access({});
      break;
    }
    EXPECT_EQ(snapshots->snapshots, 1u);
    EXPECT_EQ(hazards, 0u);
    EXPECT_FALSE(state.empty());
    const auto words = cdna5::build_sopp(cdna5::kSCbranchScc1Sopp);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    state.check_instruction(*decoded.value(), *wf);
    EXPECT_EQ(hazards, 1u); // The instruction's actual SCC use still diagnoses.
  }
}

TEST(ExecutionPluginTest, SpecialDestinationTrackingIncludesTheHighDword) {
  Wave32PluginFixture f;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *wf = f.cu->dispatch_wf(0, 0, 104, 32);
  ASSERT_NE(wf, nullptr);
  const amdgpu::RegisterAccess regs(*wf);
  const cdna5::Operand operand(64, cdna5::OperandType::OPR_VCC, 0);
  ASSERT_FALSE(operand.to_register_ref());
  const auto destination = regs.destination_register(operand);
  ASSERT_TRUE(destination);
  auto &state = wf->ensure_memory_wait_scoreboard();
  unsigned hazards = 0;
  state.bind(0x200, &hazards, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
  state.add({state.issue(WaitCounterKind::Ds), 0x100, ~uint64_t{0}, *destination,
             WaitCounterKind::Ds, 0xf});
  state.access({RegClass::VCC, 1, 1}, ~uint64_t{0}, 0xf, false);
  EXPECT_EQ(hazards, 1u);
}

// Records the host FP environment seen by each VGPR read callback.
class RoundingObserverPlugin : public ExecutionPlugin {
public:
  RoundingObserverPlugin() : ExecutionPlugin("rounding_observer") {}
  std::vector<int> read_roundings;
  std::vector<uint32_t> read_mxcsrs;
  std::vector<uint32_t> read_registers;

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *, uint32_t reg, uint64_t, uint8_t) override {
    read_registers.push_back(reg);
    read_roundings.push_back(std::fegetround());
#if RJ_HOST_ARCH_X86
    read_mxcsrs.push_back(_mm_getcsr());
#endif
  }
};

TEST(ExecutionPluginTest, ScalarRoundConvertAndPackObserversSeeCallerFpEnvironment) {
  class ScalarObserver final : public ExecutionPlugin {
  public:
    ScalarObserver() : ExecutionPlugin("scalar_fp_observer") {}
    void onAmdgpuReadScalarRegister(const Wavefront *, RegisterRef) override {
      reads.push_back(std::fegetround());
    }
    void onAmdgpuWriteScalarRegister(const Wavefront *, RegisterRef) override {
      writes.push_back(std::fegetround());
    }
    std::vector<int> reads, writes;
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    Wave32PluginFixture fixture(arch);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto owner = std::make_unique<ScalarObserver>();
    auto *observer = owner.get();
    fixture.plugin_group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    ASSERT_TRUE(fixture.plugin_group->add(std::move(owner)));
    fixture.cu->set_plugin_group(fixture.plugin_group);
    auto *wave = fixture.cu->dispatch_wf(0, 0, 104, 32);
    ASSERT_NE(wave, nullptr);
    fixture.cu->write_sgpr(wave->sgpr_alloc().base, 0x3fa00000);
    fixture.cu->write_sgpr(wave->sgpr_alloc().base + 1, 0x40600000);
    auto decoder = Decoder::create(arch);
    for (auto words :
         {rdna4::build_sop1(rdna4::kSCeilF32Sop1, {.ssrc0 = 0, .sdst = 2}),
          rdna4::build_sop1(rdna4::kSCvtF16F32Sop1, {.ssrc0 = 0, .sdst = 2}),
          rdna4::build_sop2(rdna4::kSCvtPkRtzF16F32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 2})}) {
      auto decoded = decoder->decode_window(words);
      ASSERT_TRUE(decoded.succeeded());
      observer->reads.clear();
      observer->writes.clear();
      std::fenv_t saved;
      ASSERT_EQ(std::fegetenv(&saved), 0);
      ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
      const bool succeeded =
          fixture.cu->execute_instruction(decoded.value().get(), *wave).succeeded();
      const int after = std::fegetround();
      std::fesetenv(&saved);
      EXPECT_TRUE(succeeded);
      EXPECT_EQ(after, FE_UPWARD);
      ASSERT_FALSE(observer->reads.empty());
      ASSERT_FALSE(observer->writes.empty());
      for (int rounding : observer->reads)
        EXPECT_EQ(rounding, FE_UPWARD);
      for (int rounding : observer->writes)
        EXPECT_EQ(rounding, FE_UPWARD);
    }
  }
}

// Guest rounding is independent of the default host controls seen by register
// observers. rj-arith may raise sticky flags while leaving those controls intact.
TEST(ExecutionPluginTest, F32ModeArithmeticObserversSeeCallerRounding) {
  struct Case {
    const char *name;
    uint16_t op;
    uint32_t src0;
    uint32_t src1;
    uint32_t round_up;
    uint32_t nearest;
  };
  constexpr std::array<Case, 3> cases = {{
      {"Add", rdna4::kVAddF32Vop2, 0x3f800000u, 0x30800000u, 0x3f800001u, 0x3f800000u},
      {"Sub", rdna4::kVSubF32Vop2, 0x3f800000u, 0xb0800000u, 0x3f800001u, 0x3f800000u},
      {"Mul", rdna4::kVMulF32Vop2, 0x3f800001u, 0x3f800001u, 0x3f800003u, 0x3f800002u},
  }};
  for (const Case &test : cases) {
    for (const uint32_t mode : {0xF1u, 0xF0u}) {
      for (bool force_scalar : {false, true}) {
        SCOPED_TRACE(std::format("{} mode={:#x} force_scalar={}", test.name, mode, force_scalar));
        ForceScalarOverride execution_mode(force_scalar);
        Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA4);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        auto plugin_owner = std::make_unique<RoundingObserverPlugin>();
        auto *plugin = plugin_owner.get();
        f.plugin_group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
        f.plugin_group->add(std::move(plugin_owner));
        f.cu->set_plugin_group(f.plugin_group);
        f.plugin_group->onInit();
        auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/32);
        ASSERT_NE(wf, nullptr);
        wf->set_exec(0xFFFFFFFFu);
        wf->set_mode_raw(mode);
        const uint32_t base = wf->vgpr_alloc().base;
        for (uint32_t lane = 0; lane < 32; ++lane) {
          f.cu->write_vgpr(base + 0, lane, test.src0);
          f.cu->write_vgpr(base + 1, lane, test.src1);
        }
        auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
        const auto words = rdna4::build_vop2(test.op, {.src0 = 256, .vsrc1 = 1, .vdst = 6});
        const std::array<uint32_t, 3> encoded = {words[0], 0u, 0u};
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, encoded.data()));
        ASSERT_NE(inst, nullptr);
        ASSERT_EQ(std::fegetround(), FE_TONEAREST);
#if RJ_HOST_ARCH_X86
        // Default host controls must stay visible to callbacks; arithmetic may
        // add sticky flags.
        const uint32_t saved_mxcsr = _mm_getcsr();
        _mm_setcsr((saved_mxcsr & ~_MM_EXCEPT_MASK) | _MM_EXCEPT_UNDERFLOW);
        const uint32_t host_mxcsr = _mm_getcsr();
#endif
        plugin->read_roundings.clear();
        plugin->read_mxcsrs.clear();
        bool succeeded = false;
        {
          amdgpu::ScopedSimdFastPathTracker tracker;
          succeeded = f.cu->execute_instruction(inst.get(), *wf).succeeded();
          // rj-arith does not use the legacy SIMD tracker.
          EXPECT_TRUE(tracker.no_tracked_path_executed());
        }
#if RJ_HOST_ARCH_X86
        const uint32_t after_mxcsr = _mm_getcsr();
        _mm_setcsr(saved_mxcsr);
        EXPECT_EQ(after_mxcsr & ~_MM_EXCEPT_MASK, host_mxcsr & ~_MM_EXCEPT_MASK);
        ASSERT_FALSE(plugin->read_mxcsrs.empty());
        for (const uint32_t seen : plugin->read_mxcsrs)
          EXPECT_EQ(seen & ~_MM_EXCEPT_MASK, host_mxcsr & ~_MM_EXCEPT_MASK);
#endif
        ASSERT_TRUE(succeeded);
        EXPECT_EQ(std::fegetround(), FE_TONEAREST);
        ASSERT_FALSE(plugin->read_roundings.empty());
        for (const int rounding : plugin->read_roundings)
          EXPECT_EQ(rounding, FE_TONEAREST);
        for (uint32_t lane = 0; lane < 32; ++lane)
          EXPECT_EQ(f.cu->read_vgpr(base + 6, lane), mode == 0xF1u ? test.round_up : test.nearest)
              << "lane " << lane;
      }
    }
  }
}

#if RJ_HOST_ARCH_X86
// Both VOPD slots expose unchanged default host controls to register observers.
// Migrated RDNA arithmetic may add sticky flags; legacy CDNA retains isolation.
TEST(ExecutionPluginTest, VopdArithmeticObserversSeeCallerFpState) {
  struct Case {
    rj_code_arch_t arch;
    std::array<uint32_t, 3> words;
    std::vector<std::pair<uint32_t, uint32_t>> sources;
    std::vector<std::pair<uint32_t, uint32_t>> expected;
  };
  const Case cases[] = {
      // v_dual_add_f32 v6, v0, v1 :: v_dual_add_f32 v7, v2, v3
      {ROCJITSU_CODE_ARCH_RDNA3,
       {0xc9080300u, 0x06060702u, 0u},
       {{0, 0x3f800000u}, {1, 0x33800001u}, {2, 0x3f800000u}, {3, 0x33800001u}},
       {{6, 0x3f800001u}, {7, 0x3f800001u}}},
      {ROCJITSU_CODE_ARCH_RDNA4,
       {0xc9080300u, 0x06060702u, 0u},
       {{0, 0x3f800000u}, {1, 0x33800001u}, {2, 0x3f800000u}, {3, 0x33800001u}},
       {{6, 0x3f800001u}, {7, 0x3f800001u}}},
      // VOPD3: v_dual_add_f64 v[8:9], v[0:1], v[4:5] :: v_dual_add_f32 v10, v2, v6.
      // F64 is supported only in X; each source port uses disjoint VGPR banks.
      {ROCJITSU_CODE_ARCH_CDNA5,
       {0xcf000000u | (33u << 18) | (4u << 12) | 256u, 258u | (4u << 16),
        8u | (6u << 8) | (10u << 24)},
       {{0, 0u}, {1, 0x3ff00000u}, {4, 1u}, {5, 0x3ca00000u}, {2, 0x3f800000u}, {6, 0x33800001u}},
       {{8, 1u}, {9, 0x3ff00000u}, {10, 0x3f800001u}}},
  };
  for (const Case &test : cases) {
    const bool uses_arith =
        test.arch == ROCJITSU_CODE_ARCH_RDNA3 || test.arch == ROCJITSU_CODE_ARCH_RDNA4;
    const uint32_t observed_bits = uses_arith ? ~uint32_t(_MM_EXCEPT_MASK) : UINT32_MAX;
    for (bool force_scalar : {false, true}) {
      SCOPED_TRACE(
          std::format("arch={} force_scalar={}", static_cast<int>(test.arch), force_scalar));
      ForceScalarOverride execution_mode(force_scalar);
      Wave32PluginFixture fixture(test.arch);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      auto observer_owner = std::make_unique<RoundingObserverPlugin>();
      auto *observer = observer_owner.get();
      fixture.plugin_group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
      fixture.plugin_group->add(std::move(observer_owner));
      fixture.cu->set_plugin_group(fixture.plugin_group);
      fixture.plugin_group->onInit();
      auto *wave = fixture.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/32);
      ASSERT_NE(wave, nullptr);
      wave->set_mode_raw(0xf0u);
      const uint32_t base = wave->vgpr_alloc().base;
      auto decoder = Decoder::create(test.arch);
      std::unique_ptr<Instruction> instruction(decode_valid(*decoder, test.words.data()));
      ASSERT_NE(instruction, nullptr);
      // Each slot adds 1 and just over half an ULP, rounding up and raising inexact.
      for (uint64_t exec : {uint64_t{1}, uint64_t{0xffffffffu}}) {
        SCOPED_TRACE(exec);
        wave->set_exec(exec);
        for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
          for (const auto &[reg, bits] : test.sources)
            fixture.cu->write_vgpr(base + reg, lane, bits);
          for (const auto &[reg, bits] : test.expected)
            fixture.cu->write_vgpr(base + reg, lane, 0xdeadbeefu);
        }
        observer->read_registers.clear();
        observer->read_roundings.clear();
        observer->read_mxcsrs.clear();
        {
          // Restore the original host environment even after a fatal assertion.
          fp_mode::ScopedEnvironment environment(0);
          _mm_setcsr((_mm_getcsr() & ~_MM_EXCEPT_MASK) | _MM_EXCEPT_UNDERFLOW);
          const uint32_t host_mxcsr = _mm_getcsr();
          ScopedSimdFastPathTracker tracker;
          const bool succeeded =
              fixture.cu->execute_instruction(instruction.get(), *wave).succeeded();
          const uint32_t restored_mxcsr = _mm_getcsr();
          ASSERT_TRUE(succeeded);
          EXPECT_EQ(restored_mxcsr & observed_bits, host_mxcsr & observed_bits);
          EXPECT_NE(restored_mxcsr & _MM_EXCEPT_UNDERFLOW, 0u);
          if (uses_arith || force_scalar || !util::has_stdx_simd)
            EXPECT_TRUE(tracker.no_tracked_path_executed());
          else
            EXPECT_TRUE(tracker.only_tracked_path_executed(SimdFastPath::VOPD));
          std::set<uint32_t> expected_reads;
          for (const auto &[reg, bits] : test.sources)
            expected_reads.insert(base + reg);
          EXPECT_EQ(
              std::set<uint32_t>(observer->read_registers.begin(), observer->read_registers.end()),
              expected_reads);
          for (uint32_t seen : observer->read_mxcsrs) {
            EXPECT_EQ(seen & observed_bits, host_mxcsr & observed_bits);
            EXPECT_NE(seen & _MM_EXCEPT_UNDERFLOW, 0u);
          }
          for (int seen : observer->read_roundings)
            EXPECT_EQ(seen, FE_TONEAREST);
        }
        for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
          for (const auto &[reg, bits] : test.expected)
            EXPECT_EQ(fixture.cu->read_vgpr_storage(base + reg, lane),
                      (exec & (uint64_t{1} << lane)) ? bits : 0xdeadbeefu);
      }
    }
  }
}
#endif

TEST(ExecutionPluginTest, Vop3CompareObservesOnlyArchitecturalDestinationReads) {
  Wave32PluginFixture f;
  ASSERT_NE(f.cu, nullptr);
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xFFFFFFFFULL);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kScalarDst = 12;
  uint32_t vbase = wf->vgpr_alloc().base;
  uint32_t sbase = wf->sgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    f.cu->write_vgpr(vbase + kSrc0, lane, lane);
    f.cu->write_vgpr(vbase + kSrc1, lane, lane);
  }
  f.cu->write_sgpr(sbase + kScalarDst, 0xFFFFFFFFu);
  f.cu->write_sgpr(sbase + kScalarDst + 1, 0u);

  auto destination_reads = [&] {
    std::vector<uint32_t> reads;
    for (const auto &event : plugin->events)
      if (event.kind == HookEvent::READ_SGPR)
        reads.push_back(event.physical_reg);
    return reads;
  };

  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  cdna5::Vop3MachineInst plain_raw{};
  plain_raw.vdst = kScalarDst;
  plain_raw.src0 = 256 + kSrc0;
  plain_raw.src1 = 256 + kSrc1;
  cdna5::VCmpEqU32Vop3 plain(reinterpret_cast<const cdna5::MachineInst *>(&plain_raw));
  plugin->events.clear();
  plain.execute_impl(*wf);
  EXPECT_TRUE(destination_reads().empty());

  cdna5::Vop3VopDpp16MachineInst dpp_raw{};
  dpp_raw.vdst = kScalarDst;
  dpp_raw.src0 = amdgpu::SRC_DPP;
  dpp_raw.src1 = 256 + kSrc1;
  dpp_raw.vsrc0 = kSrc0;
  dpp_raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  dpp_raw.fi = 1;
  dpp_raw.bound_ctrl = 0;
  dpp_raw.bank_mask = 0xF;
  dpp_raw.row_mask = 0xF;
  cdna5::VCmpEqU32Vop3 dpp(reinterpret_cast<const cdna5::MachineInst *>(&dpp_raw));
  plugin->events.clear();
  dpp.execute_impl(*wf);
  // BOUND_CTRL=0 forces invalid-source compare bits to zero; it does not
  // preserve or read the old lane-mask destination.
  EXPECT_TRUE(destination_reads().empty());

  // BOUND_CTRL=1 supplies zero for the OOB source and likewise requires no
  // old-destination read.
  dpp_raw.bound_ctrl = 1;
  cdna5::VCmpEqU32Vop3 zero_fill(reinterpret_cast<const cdna5::MachineInst *>(&dpp_raw));
  plugin->events.clear();
  zero_fill.execute_impl(*wf);
  EXPECT_TRUE(destination_reads().empty());
}

TEST(ExecutionPluginTest, Vop3DppCarryWritesScalarDestinationWithoutReadingIt) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }

  Wave32PluginFixture f;
  ASSERT_NE(f.cu, nullptr);
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xFFFFFFFFULL);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kDst = 10;
  constexpr uint32_t kScalarDst = 12;
  constexpr uint32_t kCarryIn = 20;
  uint32_t vbase = wf->vgpr_alloc().base;
  uint32_t sbase = wf->sgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    f.cu->write_vgpr(vbase + kSrc0, lane, lane);
    f.cu->write_vgpr(vbase + kSrc1, lane, 1u);
  }
  f.cu->write_sgpr(sbase + kCarryIn, 0u);

  cdna5::Vop3SdstEncVopDpp16MachineInst raw{};
  raw.vdst = kDst;
  raw.sdst = kScalarDst;
  raw.src0 = amdgpu::SRC_DPP;
  raw.src1 = 256 + kSrc1;
  raw.src2 = kCarryIn;
  raw.vsrc0 = kSrc0;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  raw.fi = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  auto destination_read_count = [&] {
    return std::ranges::count_if(plugin->events, [&](const HookEvent &event) {
      return event.kind == HookEvent::READ_SGPR && event.physical_reg == sbase + kScalarDst;
    });
  };

  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "SIMD");
    ForceScalarOverride force_scalar_scope(force_scalar);

    for (uint32_t bound_ctrl : {0u, 1u}) {
      raw.bound_ctrl = bound_ctrl;
      f.cu->write_sgpr(sbase + kScalarDst, 0xFFFFFFFFu);
      cdna5::VAddCoCiU32Vop3SdstEnc add(reinterpret_cast<const cdna5::MachineInst *>(&raw));
      plugin->events.clear();
      add.execute_impl(*wf);
      EXPECT_EQ(destination_read_count(), 0);
    }
  }
}

TEST(ExecutionPluginTest, F64SourceReadObservationReportsBothHalves) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + 0, lane, 0x0000'0000u);
      cu->write_vgpr(vb + 1, lane, 0x3ff0'0000u);
      cu->write_vgpr(vb + 2, lane, 0x0000'0000u);
      cu->write_vgpr(vb + 3, lane, 0x4000'0000u);
      cu->write_vgpr(vb + 4, lane, 0x0000'0000u);
      cu->write_vgpr(vb + 5, lane, 0x0000'0000u);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    uint32_t words[4] = {vop2_encode(/*opcode=*/4, /*vdst=*/4, /*vsrc1=*/2, /*src0=*/256), 0u, 0u,
                         0u};
    Instruction *inst = decode_valid(*decoder, words);
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(cu->execute_instruction(inst, *wf).succeeded());
    delete inst;

    const auto reads = vgpr_read_events(*plugin);
    if constexpr (RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS) {
      // The Clang/AVX-512 workaround executes F64 one lane at a time, so
      // require both halves of every source once per active lane.
      ASSERT_EQ(reads.size(), 6u * std::popcount(kPartialExecMask));
      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
        const uint64_t lane_mask = uint64_t{1} << lane;
        if (!(kPartialExecMask & lane_mask))
          continue;
        std::vector<HookEvent> lane_reads;
        for (const auto &read : reads)
          if (read.lane_mask == lane_mask)
            lane_reads.push_back(read);
        expect_vgpr_read_set(lane_reads, vb, {0, 1, 2, 3, 4, 5}, lane_mask);
      }
    } else {
      expect_vgpr_read_set(reads, vb, {0, 1, 2, 3, 4, 5}, kPartialExecMask);
    }
  }
}

TEST(ExecutionPluginTest, Vop3FmacSimdReadObservationReportsAccumulator) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    amdgpu::fp_mode::detail::ScopedFenv floating_environment(0);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);
    wf->set_mode_raw(0xf0u);

    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + 0, lane, 0x3f80'0000u);
      cu->write_vgpr(vb + 1, lane, 0x4000'0000u);
      cu->write_vgpr(vb + 4, lane, 0x0000'0000u);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(/*opcode=*/315, /*vdst=*/4, /*src0=*/256, /*src1=*/257, words);
    Instruction *inst = decode_valid(*decoder, words);
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(cu->execute_instruction(inst, *wf).succeeded());
    delete inst;

    expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {0, 1, 4}, kPartialExecMask);
  }
}

TEST(ExecutionPluginTest, MfmaReadObservationUsesLaneMasks) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t S0 = 0, S1 = 16, ACC = 32;

  amdgpu::observe_mfma_fast_path_reads(*cu, vb + S0, vb + S1, vb + ACC, /*M=*/16, /*N=*/16,
                                       /*K=*/32, /*B=*/1, /*data_bits=*/16, amdgpu::ACC_FROM_VGPR,
                                       wf->wf_size());

  expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                       {S0 + 0, S0 + 1, S0 + 2, S0 + 3, S1 + 0, S1 + 1, S1 + 2, S1 + 3, ACC + 0,
                        ACC + 1, ACC + 2, ACC + 3},
                       ~uint64_t{0});
}

TEST(ExecutionPluginTest, MfmaReadObservationSkipsConstantAccumulator) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t S0 = 0, S1 = 16, ACC = 32;

  amdgpu::observe_mfma_fast_path_reads(*cu, vb + S0, vb + S1, vb + ACC, /*M=*/16, /*N=*/16,
                                       /*K=*/32, /*B=*/1, /*data_bits=*/16,
                                       /*const_acc=*/0, wf->wf_size());

  expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                       {S0 + 0, S0 + 1, S0 + 2, S0 + 3, S1 + 0, S1 + 1, S1 + 2, S1 + 3},
                       ~uint64_t{0});
}

TEST(ExecutionPluginTest, Cdna4BlockScaleMfmaAbidZeroPrefixRejectsBeforeExecuteCallbacks) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  const auto scale = make_cdna4_mfma_scale_words(45, /*abid=*/0, vgpr_src(96), vgpr_src(97));
  const std::array<uint32_t, 5> code = {scale[0], scale[1], scale[2], scale[3], S_ENDPGM};

  f.run_kernel(code.data(), code.size());

  size_t before_count = 0;
  for (const auto &event : plugin->events)
    if (event.kind == HookEvent::BEFORE_INSTRUCTION)
      ++before_count;
  EXPECT_EQ(before_count, 0u);
}

TEST(ExecutionPluginTest, Cdna4BlockScaleMfmaCompoundAdvancesCallbacksByFullPrefixSize) {
  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  const auto scale = make_cdna4_mfma_scale_words(45, /*abid=*/1, vgpr_src(96), vgpr_src(97));
  const std::array<uint32_t, 5> code = {scale[0], scale[1], scale[2], scale[3], S_ENDPGM};

  f.run_kernel(code.data(), code.size());

  std::vector<HookEvent> before_events;
  for (const auto &event : plugin->events)
    if (event.kind == HookEvent::BEFORE_INSTRUCTION)
      before_events.push_back(event);
  ASSERT_EQ(before_events.size(), 2u);
  EXPECT_EQ(before_events[0].mnemonic, "v_mfma_scale_f32_16x16x128_f8f6f4");
  EXPECT_EQ(before_events[1].mnemonic, "s_endpgm");
  ASSERT_GE(before_events[1].pc, before_events[0].pc);
  EXPECT_EQ(before_events[1].pc - before_events[0].pc, 16u);
}

TEST(ExecutionPluginTest, Cdna4BlockScaleMfmaScaleReadsUseSelectedByteMask) {
  constexpr uint32_t kSrc0 = 0;
  constexpr uint32_t kSrc1 = 16;
  constexpr uint32_t kAccumulator = 32;
  constexpr uint32_t kScaleA = 96;
  constexpr uint32_t kScaleB = 97;

  for (uint32_t byte = 0; byte < 4; ++byte) {
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(~uint64_t{0});

    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      for (uint32_t reg = 0; reg < 8; ++reg) {
        cu->write_vgpr(vb + kSrc0 + reg, lane, 0x2222'2222u);
        cu->write_vgpr(vb + kSrc1 + reg, lane, 0x2222'2222u);
      }
      for (uint32_t reg = 0; reg < 4; ++reg)
        cu->write_vgpr(vb + kAccumulator + reg, lane, 0);
      cu->write_vgpr(vb + kScaleA, lane, 0x7F7F'7F7Fu);
      cu->write_vgpr(vb + kScaleB, lane, 0x7F7F'7F7Fu);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    const auto words = make_cdna4_mfma_scale_words(45, /*abid=*/1, vgpr_src(kScaleA),
                                                   vgpr_src(kScaleB), byte, byte);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());

    const uint8_t expected_mask = static_cast<uint8_t>(1u << byte);
    bool saw_scale_a = false;
    bool saw_scale_b = false;
    std::map<uint32_t, std::pair<uint64_t, uint8_t>> read_unions;
    const auto reads = vgpr_read_events(*plugin);
    for (const auto &event : reads) {
      auto &[lane_mask, byte_mask] = read_unions[event.physical_reg];
      lane_mask |= event.lane_mask;
      byte_mask |= event.byte_mask;
      if (event.physical_reg == vb + kScaleA) {
        EXPECT_EQ(event.byte_mask, expected_mask) << "A byte=" << byte;
        saw_scale_a = true;
      } else if (event.physical_reg == vb + kScaleB) {
        EXPECT_EQ(event.byte_mask, expected_mask) << "B byte=" << byte;
        saw_scale_b = true;
      }
    }
    EXPECT_TRUE(saw_scale_a) << "byte=" << byte;
    EXPECT_TRUE(saw_scale_b) << "byte=" << byte;

    std::map<uint32_t, std::pair<uint64_t, uint8_t>> expected_unions;
    for (uint32_t reg :
         {kSrc0, kSrc0 + 1, kSrc0 + 2, kSrc0 + 3, kSrc1, kSrc1 + 1, kSrc1 + 2, kSrc1 + 3,
          kAccumulator, kAccumulator + 1, kAccumulator + 2, kAccumulator + 3})
      expected_unions.emplace(vb + reg, std::pair{~uint64_t{0}, ExecutionPlugin::kFullByteMask});
    expected_unions.emplace(vb + kScaleA, std::pair{~uint64_t{0}, expected_mask});
    expected_unions.emplace(vb + kScaleB, std::pair{~uint64_t{0}, expected_mask});
    EXPECT_EQ(read_unions, expected_unions) << "byte=" << byte;
  }
}

TEST(ExecutionPluginTest, Cdna4BlockScaleMfmaInlineScalesDoNotReadVgprs) {
  constexpr uint32_t kSrc0 = 0;
  constexpr uint32_t kSrc1 = 16;
  constexpr uint32_t kAccumulator = 32;

  PluginFixture f;
  ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(~uint64_t{0});

  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    for (uint32_t reg = 0; reg < 4; ++reg) {
      cu->write_vgpr(vb + kSrc0 + reg, lane, 0x2222'2222u);
      cu->write_vgpr(vb + kSrc1 + reg, lane, 0x2222'2222u);
      cu->write_vgpr(vb + kAccumulator + reg, lane, 0);
    }
  }

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  const auto words = make_cdna4_mfma_scale_words(45, /*abid=*/1, /*+1.0f=*/242,
                                                 /*1/(2*pi)=*/248);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());

  std::set<uint32_t> actual_regs;
  for (const auto &event : vgpr_read_events(*plugin))
    actual_regs.insert(event.physical_reg - vb);
  std::set<uint32_t> expected_regs;
  for (uint32_t reg :
       {kSrc0, kSrc0 + 1, kSrc0 + 2, kSrc0 + 3, kSrc1, kSrc1 + 1, kSrc1 + 2, kSrc1 + 3,
        kAccumulator, kAccumulator + 1, kAccumulator + 2, kAccumulator + 3})
    expected_regs.insert(reg);
  EXPECT_EQ(actual_regs, expected_regs);
}

TEST(ExecutionPluginTest, WmmaReadObservationUsesWave32RegisterSet) {
  Wave32PluginFixture f;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t S0 = 0, S1 = 16, ACC = 32;

  amdgpu::observe_wmma_fast_path_reads</*M=*/16, /*N=*/16, /*K=*/32>(
      *f.cu, vb + S0, vb + S1, vb + ACC, /*data_bits=*/16, /*acc_bits=*/32, amdgpu::ACC_FROM_VGPR,
      wf->wf_size());

  expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                       {S0 + 0,  S0 + 1,  S0 + 2,  S0 + 3,  S0 + 4,  S0 + 5,  S0 + 6,  S0 + 7,
                        S1 + 0,  S1 + 1,  S1 + 2,  S1 + 3,  S1 + 4,  S1 + 5,  S1 + 6,  S1 + 7,
                        ACC + 0, ACC + 1, ACC + 2, ACC + 3, ACC + 4, ACC + 5, ACC + 6, ACC + 7},
                       0xFFFF'FFFFu);
}

TEST(ExecutionPluginTest, WmmaReadObservationSkipsConstantAccumulator) {
  Wave32PluginFixture f;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t S0 = 0, S1 = 16, ACC = 32;

  amdgpu::observe_wmma_fast_path_reads</*M=*/16, /*N=*/16, /*K=*/32>(
      *f.cu, vb + S0, vb + S1, vb + ACC, /*data_bits=*/16, /*acc_bits=*/32,
      /*const_acc=*/0, wf->wf_size());

  expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                       {S0 + 0, S0 + 1, S0 + 2, S0 + 3, S0 + 4, S0 + 5, S0 + 6, S0 + 7, S1 + 0,
                        S1 + 1, S1 + 2, S1 + 3, S1 + 4, S1 + 5, S1 + 6, S1 + 7},
                       0xFFFF'FFFFu);
}

TEST(ExecutionPluginTest, WmmaF32NativeWidthFastPathUsesRegionReads) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "stdx SIMD is unavailable";
  } else {
    constexpr uint32_t M = 16, N = 16, K = 4;
    constexpr uint32_t width = static_cast<uint32_t>(util::native<float>::size());
    if (!amdgpu::mma_f32_native_width_supported(N, width))
      GTEST_SKIP() << "f32 WMMA shape is not divisible by the native SIMD width";

    ForceScalarOverride force_simd(false);
    Wave32PluginFixture f;
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu.get();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(wf->wf_size(), 32u);

    const uint32_t vb = wf->vgpr_alloc().base;
    constexpr uint32_t S0 = 0, S1 = 16, ACC = 32, DST = 48;
    for (uint32_t reg = 0; reg < 64; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        cu->write_vgpr(vb + reg, lane, 0x3f80'0000u);

    amdgpu::exec_wmma_f32_f32_spec<M, N, K>(*cu, vb + DST, vb + S0, vb + S1, vb + ACC,
                                            amdgpu::ACC_FROM_VGPR);

    expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                         {S0 + 0, S0 + 1, S1 + 0, S1 + 1, ACC + 0, ACC + 1, ACC + 2, ACC + 3,
                          ACC + 4, ACC + 5, ACC + 6, ACC + 7},
                         0xFFFF'FFFFu);
  }
}

TEST(ExecutionPluginTest, WmmaF16NativeWidthPreservesObservedReadWriteOrder) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "stdx SIMD is unavailable";
  }
  ForceScalarOverride force_simd(false);
  constexpr uint32_t S0 = 0, S1 = 16, ACC = 32;
  for (uint32_t input : {0x3c00'3c00u, 0x7e11'7c33u}) {
    Wave32PluginFixture f;
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto *cu = f.cu.get();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t reg = 0; reg < 8; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane) {
        cu->write_vgpr(vb + S0 + reg, lane, input);
        cu->write_vgpr(vb + S1 + reg, lane, 0x3c00'3c00u);
        cu->write_vgpr(vb + ACC + reg, lane, 0x3f80'0000u);
      }
    auto *plugin = f.attach_ordering_plugin();
    // Use the same CU view instantiated by generated instruction execution.
    amdgpu::InstructionComputeUnitView view(*cu, *wf);
    amdgpu::exec_wmma_f32_16x16x32_f16(view, vb + ACC, vb + S0, vb + S1, vb + ACC);
    std::vector<uint32_t> expected_reads, expected_writes;
    for (uint32_t reg = 0; reg < 8; ++reg) {
      expected_reads.insert(expected_reads.end(), {S0 + reg, S1 + reg, ACC + reg});
      expected_writes.push_back(ACC + reg);
    }
    expect_vgpr_read_set(vgpr_read_events(*plugin), vb, expected_reads, 0xFFFF'FFFFu);
    expect_vgpr_read_set(vgpr_write_events(*plugin), vb, expected_writes, 0xFFFF'FFFFu);
    size_t last_read = 0, first_write = plugin->events.size();
    for (size_t i = 0; i < plugin->events.size(); ++i) {
      if (plugin->events[i].kind == HookEvent::READ_VGPR)
        last_read = i;
      if (plugin->events[i].kind == HookEvent::WRITE_VGPR)
        first_write = std::min(first_write, i);
    }
    EXPECT_LT(last_read, first_write);
  }
}

TEST(ExecutionPluginTest, WmmaBf16F32FastPathReadsBeforePackedWrites) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "stdx SIMD is unavailable";
  } else {
    if (util::native<float>::size() != 16)
      GTEST_SKIP() << "the BF16F32 fast path requires 16-lane native SIMD";

    ForceScalarOverride force_simd(false);
    Wave32PluginFixture f;
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu.get();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(wf->wf_size(), 32u);

    const uint32_t vb = wf->vgpr_alloc().base;
    constexpr uint32_t S0 = 0, S1 = 16, ACC = 32, DST = 48;
    for (uint32_t reg = 0; reg < 56; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        cu->write_vgpr(vb + reg, lane, 0);
    plugin->events.clear();

    amdgpu::exec_wmma_bf16f32_16x16x32_bf16(*cu, vb + DST, vb + S0, vb + S1, vb + ACC,
                                            amdgpu::ACC_FROM_VGPR,
                                            /*c_modifier=*/0);

    expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                         {S0 + 0,  S0 + 1,  S0 + 2,  S0 + 3,  S0 + 4,  S0 + 5,  S0 + 6,  S0 + 7,
                          S1 + 0,  S1 + 1,  S1 + 2,  S1 + 3,  S1 + 4,  S1 + 5,  S1 + 6,  S1 + 7,
                          ACC + 0, ACC + 1, ACC + 2, ACC + 3, ACC + 4, ACC + 5, ACC + 6, ACC + 7},
                         0xFFFF'FFFFu);
    expect_vgpr_read_set(vgpr_write_events(*plugin), vb, {DST + 0, DST + 1, DST + 2, DST + 3},
                         0xFFFF'FFFFu);

    size_t last_read = 0;
    size_t first_write = plugin->events.size();
    for (size_t i = 0; i < plugin->events.size(); ++i) {
      if (plugin->events[i].kind == HookEvent::READ_VGPR)
        last_read = i;
      if (plugin->events[i].kind == HookEvent::WRITE_VGPR)
        first_write = std::min(first_write, i);
    }
    ASSERT_LT(last_read, first_write);
  }
}

TEST(ExecutionPluginTest, MfmaF32NativeWidthFastPathUsesRegionReads) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "stdx SIMD is unavailable";
  } else {
    constexpr uint32_t M = 16, N = 16, K = 4, B = 1;
    constexpr uint32_t width = static_cast<uint32_t>(util::native<float>::size());
    if (!amdgpu::mma_f32_native_width_supported(N, width))
      GTEST_SKIP() << "f32 MFMA shape is not divisible by the native SIMD width";

    ForceScalarOverride force_simd(false);
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(wf->wf_size(), 64u);

    const uint32_t vb = wf->vgpr_alloc().base;
    constexpr uint32_t S0 = 0, S1 = 16, ACC = 32, DST = 48;
    for (uint32_t reg = 0; reg < 64; ++reg)
      for (uint32_t lane = 0; lane < 64; ++lane)
        cu->write_vgpr(vb + reg, lane, 0x3f80'0000u);

    amdgpu::exec_f32_mfma_f32_spec<M, N, K, B>(*cu, vb + DST, vb + S0, vb + S1, vb + ACC,
                                               amdgpu::ACC_FROM_VGPR,
                                               /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);

    expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                         {S0, S1, ACC + 0, ACC + 1, ACC + 2, ACC + 3}, ~uint64_t{0});
  }
}

TEST(ExecutionPluginTest, MfmaSnapshotsReadBeforeFullWaveWritesAndSkipConstantAccumulators) {
  if constexpr (!util::has_stdx_simd)
    GTEST_SKIP() << "<experimental/simd> unavailable";
  ForceScalarOverride force_simd(false);
  struct Case {
    const char *name;
    uint32_t source_regs, result_regs;
    void (*run)(ComputeUnitCore &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
  };
  const Case cases[] = {
      {"f64", 2, 8,
       [](ComputeUnitCore &cu, uint32_t d, uint32_t a, uint32_t b, uint32_t c, uint32_t acc) {
         exec_f64(cu, 16, 16, 4, 1, d, a, b, c, acc);
       }},
      {"fp8 chunked", 8, 4,
       [](ComputeUnitCore &cu, uint32_t d, uint32_t a, uint32_t b, uint32_t c, uint32_t acc) {
         exec_f32<16, 16, 128, 1>(cu, 8, d, a, b, c, extract_fp8, extract_fp8, acc);
       }},
      {"i8", 4, 4,
       [](ComputeUnitCore &cu, uint32_t d, uint32_t a, uint32_t b, uint32_t c, uint32_t acc) {
         exec_i32_i8(cu, 16, 16, 64, 1, d, a, b, c, acc);
       }},
  };
  constexpr uint32_t A = 3, B = 19, C = 35;
  for (const Case &test : cases)
    for (uint32_t acc : {ACC_FROM_VGPR, 0u})
      for (uint32_t destination : {67u, A, B, C})
        for (uint64_t lane_mask : {~uint64_t{0}, kPartialExecMask}) {
          SCOPED_TRACE(test.name);
          SCOPED_TRACE(acc);
          SCOPED_TRACE(destination);
          SCOPED_TRACE(lane_mask);
          PluginFixture f;
          ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
          auto *plugin = f.attach_ordering_plugin();
          auto *cu = f.cu();
          auto *wf = cu->dispatch_wf(0, 0, 104, 256);
          ASSERT_NE(wf, nullptr);
          const uint32_t vb = wf->vgpr_alloc().base;
          for (uint32_t reg = 0; reg < 80; ++reg)
            for (uint32_t lane = 0; lane < 64; ++lane)
              cu->write_vgpr(vb + reg, lane, 0x3f800000u);
          wf->set_exec(lane_mask);
          plugin->events.clear();
          test.run(*cu, vb + destination, vb + A, vb + B, vb + C, acc);
          std::vector<uint32_t> read_regs;
          for (uint32_t reg = 0; reg < test.source_regs; ++reg) {
            read_regs.push_back(A + reg);
            read_regs.push_back(B + reg);
          }
          if (acc == ACC_FROM_VGPR)
            for (uint32_t reg = 0; reg < test.result_regs; ++reg)
              read_regs.push_back(C + reg);
          expect_vgpr_read_set(vgpr_read_events(*plugin), vb, read_regs, ~uint64_t{0});
          const auto writes = vgpr_write_events(*plugin);
          ASSERT_EQ(writes.size(), test.result_regs);
          for (uint32_t reg = 0; reg < test.result_regs; ++reg) {
            EXPECT_EQ(writes[reg].physical_reg, vb + destination + reg);
            EXPECT_EQ(writes[reg].lane_mask, ~uint64_t{0});
            EXPECT_EQ(writes[reg].byte_mask, ExecutionPlugin::kFullByteMask);
          }
          bool saw_write = false;
          for (const auto &event : plugin->events) {
            if (event.kind == HookEvent::WRITE_VGPR)
              saw_write = true;
            if (event.kind == HookEvent::READ_VGPR) {
              EXPECT_FALSE(saw_write);
            }
          }
        }
}

// Register observers, such as the race detector, must see the same (register, lane) reads and
// writes from the snapshot path as from the per-element forced-scalar path, including when
// cbsz/abid or blgp broadcast part of a register. The snapshot path writes each destination
// register once.
TEST(ExecutionPluginTest, MfmaSnapshotObservationsMatchForcedScalar) {
  if constexpr (!util::has_stdx_simd)
    GTEST_SKIP() << "<experimental/simd> unavailable";
  ForceScalarOverride restore(util::force_scalar());
  constexpr uint32_t A = 0, B = 16, C = 32, D = 64;
  using Run = std::function<void(ComputeUnitCore &, uint32_t)>;
  struct Case {
    std::string name;
    Run run;
    bool uses_snapshot = true;
  };
  std::vector<Case> cases;
  for (uint32_t acc : {ACC_FROM_VGPR, 0u}) {
    for (const auto &[m, k, batch] : {std::tuple{16u, 4u, 1u}, std::tuple{4u, 4u, 4u}})
      cases.push_back({std::format("f64 {}x{}x{}x{} acc={}", m, m, k, batch, acc),
                       [=](ComputeUnitCore &cu, uint32_t vb) {
                         exec_f64(cu, m, m, k, batch, vb + D, vb + A, vb + B, vb + C, acc, 7);
                       }});
    for (const auto &[m, k, batch] :
         {std::tuple{16u, 64u, 1u}, std::tuple{32u, 32u, 1u}, std::tuple{32u, 16u, 1u},
          std::tuple{16u, 32u, 1u}, std::tuple{16u, 16u, 1u}, std::tuple{32u, 8u, 1u},
          std::tuple{32u, 4u, 2u}, std::tuple{16u, 4u, 4u}, std::tuple{4u, 4u, 16u}})
      for (uint32_t cbsz = 0; (1u << cbsz) <= batch; ++cbsz)
        for (uint32_t blgp = 0; blgp < 8; ++blgp)
          cases.push_back({std::format("i8 {}x{}x{}x{} acc={} cbsz={} blgp={}", m, m, k, batch, acc,
                                       cbsz, blgp),
                           [=](ComputeUnitCore &cu, uint32_t vb) {
                             exec_i32_i8(cu, m, m, k, batch, vb + D, vb + A, vb + B, vb + C, acc,
                                         cbsz, (1u << cbsz) - 1, blgp);
                           },
                           // 32 result registers exceed the snapshot.
                           /*uses_snapshot=*/batch != 2});
    for (const auto &[k, bits] : {std::pair{1u, 32u}, std::pair{2u, 16u}, std::pair{4u, 16u}})
      for (uint32_t cbsz = 0; cbsz <= 4; ++cbsz)
        for (uint32_t blgp = 0; blgp < 8; ++blgp)
          cases.push_back(
              {std::format("f32 4x4x{}x16 acc={} cbsz={} blgp={}", k, acc, cbsz, blgp),
               [=](ComputeUnitCore &cu, uint32_t vb) {
                 if (bits == 32)
                   exec_f32<4, 4, 1, 16>(cu, 32, vb + D, vb + A, vb + B, vb + C, extract_f32,
                                         extract_f32, acc, cbsz, 1, blgp);
                 else if (k == 2)
                   exec_f32<4, 4, 2, 16>(cu, 16, vb + D, vb + A, vb + B, vb + C, extract_bf16,
                                         extract_bf16, acc, cbsz, 1, blgp);
                 else
                   exec_f32<4, 4, 4, 16>(cu, 16, vb + D, vb + A, vb + B, vb + C, extract_bf16,
                                         extract_bf16, acc, cbsz, 1, blgp);
               }});
    // Four blocks of 4x4 outputs use only lanes 0-15 of each result register.
    cases.push_back({std::format("f32 4x4x4x4 acc={}", acc),
                     [=](ComputeUnitCore &cu, uint32_t vb) {
                       exec_f32<4, 4, 4, 4>(cu, 32, vb + D, vb + A, vb + B, vb + C, extract_f32,
                                            extract_f32, acc);
                     },
                     /*uses_snapshot=*/false});
    for (uint32_t m : {16u, 32u})
      for (uint32_t a_fmt = 0; a_fmt < 5; ++a_fmt)
        for (uint32_t b_fmt = 0; b_fmt < 5; ++b_fmt)
          cases.push_back(
              {std::format("f8f6f4 {}x{} acc={} a={} b={}", m, m, acc, a_fmt, b_fmt),
               [=](ComputeUnitCore &cu, uint32_t vb) {
                 ASSERT_TRUE(dispatch_matrix_fmt_pair(
                     a_fmt, b_fmt, [&](uint32_t a_bits, uint32_t b_bits, auto ea, auto eb) {
                       if (m == 16)
                         exec_f32_mixed<16, 16, 128, 1>(cu, a_bits, b_bits, vb + D, vb + A, vb + B,
                                                        vb + C, ea, eb, acc);
                       else
                         exec_f32_mixed<32, 32, 64, 1>(cu, a_bits, b_bits, vb + D, vb + A, vb + B,
                                                       vb + C, ea, eb, acc);
                     }));
               }});
  }
  using LaneSets = std::map<uint32_t, uint64_t>;
  using Counts = std::map<uint32_t, uint32_t>;
  auto observe = [](const Run &run, bool force_scalar, LaneSets &reads, LaneSets &writes,
                    Counts &write_counts) {
    PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(/*num_wf_slots=*/1));
    auto *wf = f.cu()->dispatch_wf(0, 0, 104, 256);
    ASSERT_NE(wf, nullptr);
    const uint32_t vb = wf->vgpr_alloc().base;
    auto *plugin = f.attach_ordering_plugin();
    util::set_force_scalar_for_testing(force_scalar);
    run(*f.cu(), vb);
    for (const HookEvent &event : vgpr_read_events(*plugin))
      reads[event.physical_reg - vb] |= event.lane_mask;
    for (const HookEvent &event : vgpr_write_events(*plugin)) {
      writes[event.physical_reg - vb] |= event.lane_mask;
      ++write_counts[event.physical_reg - vb];
    }
  };
  for (const Case &test : cases) {
    SCOPED_TRACE(test.name);
    LaneSets snapshot_reads, snapshot_writes, scalar_reads, scalar_writes;
    Counts snapshot_write_counts, scalar_write_counts;
    ASSERT_NO_FATAL_FAILURE(
        observe(test.run, false, snapshot_reads, snapshot_writes, snapshot_write_counts));
    ASSERT_NO_FATAL_FAILURE(
        observe(test.run, true, scalar_reads, scalar_writes, scalar_write_counts));
    EXPECT_EQ(snapshot_reads, scalar_reads);
    EXPECT_EQ(snapshot_writes, scalar_writes);
    ASSERT_FALSE(snapshot_write_counts.empty());
    for (const auto &[reg, count] : snapshot_write_counts) {
      if (test.uses_snapshot)
        EXPECT_EQ(count, 1u) << "v" << reg;
      else
        EXPECT_GT(count, 1u) << "v" << reg;
    }
    if (::testing::Test::HasFailure())
      return;
  }
}

TEST(ExecutionPluginTest, SdwaFloatingConversionsUseDestinationFormat) {
  struct Architecture {
    rj_code_arch_t arch;
    uint16_t cvt_f16_i16;
  };
  constexpr std::array architectures{
      Architecture{ROCJITSU_CODE_ARCH_CDNA4, cdna4::kVCvtF16I16Vop1},
      Architecture{ROCJITSU_CODE_ARCH_CDNA3, cdna3::kVCvtF16I16Vop1},
      Architecture{ROCJITSU_CODE_ARCH_CDNA2, cdna2::kVCvtF16I16Vop1},
      Architecture{ROCJITSU_CODE_ARCH_CDNA1, cdna1::kVCvtF16I16Vop1},
      Architecture{ROCJITSU_CODE_ARCH_RDNA2, rdna2::kVCvtF16I16Vop1},
      Architecture{ROCJITSU_CODE_ARCH_RDNA1, rdna1::kVCvtF16I16Vop1},
  };
  struct Case {
    uint16_t opcode;
    uint32_t source;
    uint32_t source_selection;
    uint32_t expected;
  };
  for (const Architecture &architecture : architectures) {
    const std::array cases{
        Case{cdna4::kVCvtF32I32Vop1, 2u, amdgpu::sdwa::DWORD, 0x3f800000u},
        Case{architecture.cvt_f16_i16, 2u, amdgpu::sdwa::WORD_0, 0x3c00u},
        Case{cdna4::kVCvtF16F32Vop1, 0x40000000u, amdgpu::sdwa::DWORD, 0x3c00u},
        Case{cdna4::kVCvtF32F16Vop1, 0x4000u, amdgpu::sdwa::WORD_0, 0x3f800000u},
        Case{cdna4::kVCvtF32Ubyte0Vop1, 2u, amdgpu::sdwa::DWORD, 0x3f800000u},
        // Integer destinations keep their conversion controls, without floating modifiers.
        Case{cdna4::kVCvtI32F32Vop1, 0x40000000u, amdgpu::sdwa::DWORD, 2u},
    };
    for (bool force_scalar : {false, true}) {
      ForceScalarOverride scalar_override(force_scalar);
      amdgpu::GpuMemory memory("sdwa_conversion_memory");
      amdgpu::L2Cache cache("sdwa_conversion_cache");
      amdgpu::ComputeUnitCore::Config config{};
      config.arch = architecture.arch;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 106;
      config.vgprs_per_wf = 256;
      config.lds_size_kb = 64;
      std::unique_ptr<amdgpu::ComputeUnitCore> compute_unit;
      ASSERT_NO_FATAL_FAILURE(rocjitsu::test::initialize_compute_unit(
          compute_unit, "sdwa_conversion", config, &memory, &cache));
      std::unique_ptr<Decoder> decoder = Decoder::create(architecture.arch);
      amdgpu::Wavefront *wave = compute_unit->dispatch_wf(0, 0, 106, 256);
      ASSERT_NE(wave, nullptr);
      wave->set_mode_raw(0);
      wave->set_exec(0b101);
      const uint32_t base = wave->vgpr_alloc().base;
      for (const Case &test : cases) {
        SCOPED_TRACE(::testing::Message() << "arch=" << architecture.arch << " op=" << test.opcode
                                          << " scalar=" << force_scalar);
        for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
          compute_unit->write_vgpr(base, lane, test.source);
          compute_unit->write_vgpr(base + 6, lane, 0xbeefbeefu);
        }
        // Common VOP1 opcodes except the per-family F16 integer conversion above.
        const std::array<uint32_t, 2> words{
            0x7e000000u | (6u << 17) | (uint32_t{test.opcode} << 9) | amdgpu::SRC_SDWA,
            (amdgpu::sdwa::DWORD << 8) | (1u << 13) | (test.source_selection << 16),
        };
        std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
        ASSERT_NE(instruction, nullptr);
        EXPECT_TRUE(compute_unit->execute_instruction(instruction.get(), *wave).succeeded());
        for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
          EXPECT_EQ(compute_unit->read_vgpr_storage(base + 6, lane),
                    (wave->exec() & (uint64_t{1} << lane)) ? test.expected : 0xbeefbeefu);
        }
      }
      wave->halt();
    }
  }
}

} // namespace
