// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/stream.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {
using Program = std::vector<uint32_t>;

template <typename T> void append(Program &program, T inst) {
  auto words = std::bit_cast<std::array<uint32_t, sizeof(T) / 4>>(inst);
  program.insert(program.end(), words.begin(), words.end());
}

uint32_t v_mov(uint32_t dst, uint32_t src) { return 0x7e000300u | (dst << 17) | src; }

void load(Program &program, uint8_t dst = 0) {
  rdna4::VglobalMachineInst inst{};
  inst.encoding = 0xee;
  inst.op = 20;
  inst.vdst = dst;
  inst.vaddr = 8;
  append(program, inst); // global_load_b32 vdst, v8, s[0:1].
}

void ds_load(Program &program, uint8_t dst = 0) {
  program.insert(program.end(), {0xd8d80000u, (uint32_t(dst) << 24) | 8u});
}

WaitcheckStreamReport analyze(const Program &program,
                              rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                              bool expert_scheduling = false, uint32_t wave_size = 64) {
  util::StringDiagnostic error;
  auto result = analyze_waitcheck_stream(
      program, arch, {.wave_size = wave_size, .expert_scheduling = expert_scheduling},
      error.emitter());
  EXPECT_TRUE(result.succeeded()) << error.message();
  if (result.failed())
    return {{}, WaitcheckStreamStop{0, "", error.message()}, 0};
  return std::move(result).value();
}

TEST(WaitcheckStream, ReportsProducerConsumerAndRequiredWait) {
  Program program;
  load(program);
  program.push_back(v_mov(1, 0));
  const auto report = analyze(program);
  ASSERT_FALSE(report.incomplete);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  const auto &diagnostic = report.diagnostics[0];
  EXPECT_EQ(diagnostic.producer_offset, 0u);
  EXPECT_EQ(diagnostic.consumer_offset, 12u);
  EXPECT_EQ(diagnostic.counter, WaitCounterKind::Load);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(diagnostic.access, WaitcheckAccess::Read);
  EXPECT_EQ(diagnostic.required_count, 0u);
  EXPECT_EQ(diagnostic.required_wait, "s_wait_loadcnt <= 0");
  EXPECT_TRUE(diagnostic.producer.starts_with("global_load_b32"));
  EXPECT_TRUE(diagnostic.consumer.starts_with("v_mov_b32"));
  EXPECT_EQ(report.instructions_analyzed, 2u);
}

TEST(WaitcheckStream, Gfx1251PackedArithmeticPreservesSubsequentMemoryChecks) {
  for (bool waited : {false, true}) {
    SCOPED_TRACE(waited);
    // v_pk_fma_f64 v[4:7], v[8:11], v[12:15], v[16:19].
    Program program{0xcc3b4004u, 0x1c421908u};
    load(program);
    if (waited)
      program.push_back(0xbfc00000u); // s_wait_loadcnt 0.
    program.push_back(v_mov(1, 0));

    // Architecture-only CDNA5 lookup selects gfx1250, which rejects the FMA.
    EXPECT_TRUE(analyze_waitcheck_stream(program, ROCJITSU_CODE_ARCH_CDNA5, {}).failed());
    EXPECT_TRUE(analyze_waitcheck_stream(program, default_isa_target_registry(),
                                         ROCJITSU_CODE_TARGET_GFX1250, {})
                    .failed());

    util::StringDiagnostic error;
    auto result = analyze_waitcheck_stream(program, default_isa_target_registry(),
                                           ROCJITSU_CODE_TARGET_GFX1251, {}, error.emitter());
    ASSERT_TRUE(result.succeeded()) << error.message();
    const auto &report = result.value();
    ASSERT_FALSE(report.incomplete);
    EXPECT_EQ(report.instructions_analyzed, waited ? 4u : 3u);
    ASSERT_EQ(report.diagnostics.size(), waited ? 0u : 1u);
    if (!waited) {
      const auto &diagnostic = report.diagnostics[0];
      EXPECT_EQ(diagnostic.producer_offset, 8u);
      EXPECT_EQ(diagnostic.consumer_offset, 20u);
      EXPECT_EQ(diagnostic.counter, WaitCounterKind::Load);
      EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VGPR, 0, 1}));
      EXPECT_EQ(diagnostic.access, WaitcheckAccess::Read);
      EXPECT_EQ(diagnostic.required_wait, "s_wait_loadcnt <= 0");
    }
  }
}

TEST(WaitcheckStream, ConcreteTargetsUseTheirArchitectureOptions) {
  for (auto target : {ROCJITSU_CODE_TARGET_GFX942, ROCJITSU_CODE_TARGET_GFX950,
                      ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_TARGET_GFX1251}) {
    SCOPED_TRACE(target);
    const bool cdna5 =
        target == ROCJITSU_CODE_TARGET_GFX1250 || target == ROCJITSU_CODE_TARGET_GFX1251;
    EXPECT_TRUE(
        analyze_waitcheck_stream({}, default_isa_target_registry(), target, {}).succeeded());
    EXPECT_EQ(analyze_waitcheck_stream({}, default_isa_target_registry(), target, {.wave_size = 32})
                  .succeeded(),
              cdna5);
    EXPECT_EQ(analyze_waitcheck_stream({}, default_isa_target_registry(), target,
                                       {.expert_scheduling = true})
                  .succeeded(),
              cdna5);
  }
}

TEST(WaitcheckStream, UnknownAndUnsupportedConcreteTargetsFail) {
  for (auto target : {ROCJITSU_CODE_TARGET_INVALID, ROCJITSU_CODE_TARGET_GFX90A}) {
    SCOPED_TRACE(target);
    util::StringDiagnostic error;
    EXPECT_TRUE(
        analyze_waitcheck_stream({}, default_isa_target_registry(), target, {}, error.emitter())
            .failed());
    EXPECT_FALSE(error.message().empty());
  }
  const std::array<IsaTargetDescriptor, 0> targets{};
  const IsaTargetRegistry empty_registry{targets};
  util::StringDiagnostic error;
  EXPECT_TRUE(analyze_waitcheck_stream({}, empty_registry, ROCJITSU_CODE_TARGET_GFX1251, {},
                                       error.emitter())
                  .failed());
  EXPECT_FALSE(error.message().empty());
}

TEST(WaitcheckStream, LoadWaitAndSentinelHaveDifferentEffects) {
  for (uint32_t count : {0u, 1u, 63u}) {
    Program program;
    load(program);
    program.push_back(0xbfc00000u | count);
    program.push_back(v_mov(1, 0));
    const auto report = analyze(program);
    ASSERT_FALSE(report.incomplete);
    EXPECT_EQ(report.diagnostics.size(), count == 0 ? 0u : 1u);
  }
}

TEST(WaitcheckStream, PartialWaitRetiresOnlyOlderLoads) {
  Program program;
  load(program, 0);
  load(program, 2);
  program.insert(program.end(), {0xbfc00001u, v_mov(4, 0), v_mov(5, 2)});
  const auto report = analyze(program);
  ASSERT_FALSE(report.incomplete);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].producer_offset, 12u);
  EXPECT_EQ(report.diagnostics[0].consumer_offset, 32u);
  EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
}

TEST(WaitcheckStream, CounterOnlyLoadsStillAgePendingResults) {
  Program program;
  load(program, 0);
  // global_inv contributes to LOAD_CNT without a register result.
  program.insert(program.end(), {0xee0ac07cu, 0, 0, 0xbfc00001u, v_mov(1, 0)});
  const auto report = analyze(program);
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckStream, CounterBoundsSaturateWithoutWrapping) {
  Program program;
  load(program, 0);
  for (unsigned i = 0; i < 300; ++i)
    program.insert(program.end(), {0xee0ac07cu, 0, 0}); // global_inv ages LOAD_CNT.
  program.insert(program.end(), {0xbfc0003eu, v_mov(4, 0)});
  const auto report = analyze(program);
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckStream, WrongCounterWaitDoesNotRetireLoad) {
  Program program;
  load(program);
  program.insert(program.end(), {0xbfc60000u, v_mov(1, 0)}); // s_wait_dscnt 0.
  const auto report = analyze(program);
  ASSERT_FALSE(report.incomplete);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
}

TEST(WaitcheckStream, EveryUnorderedPendingLoadProtectsItsDestination) {
  Program program;
  load(program);
  load(program);
  auto unwaited = analyze(program);
  EXPECT_FALSE(unwaited.incomplete);
  ASSERT_EQ(unwaited.diagnostics.size(), 1u);
  EXPECT_EQ(unwaited.diagnostics[0].access, WaitcheckAccess::Write);
  program.push_back(v_mov(0, 2));
  auto overwrite = analyze(program);
  ASSERT_EQ(overwrite.diagnostics.size(), 3u);
  EXPECT_EQ(overwrite.diagnostics[0].access, WaitcheckAccess::Write);
  EXPECT_EQ(overwrite.diagnostics[2].required_count, 0u);
}

TEST(WaitcheckStream, DsOverwritesRequireAWait) {
  Program program;
  ds_load(program);
  ds_load(program);
  auto report = analyze(program);
  ASSERT_FALSE(report.incomplete);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccess::Write);
}

TEST(WaitcheckStream, UnorderedVmemWritebackRequiresAWait) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool waited : {false, true}) {
      Program program;
      load(program);
      if (waited)
        program.push_back(0xbfc00000u);
      load(program);
      const auto report = analyze(program, arch);
      ASSERT_FALSE(report.incomplete);
      ASSERT_EQ(report.diagnostics.size(), waited ? 0u : 1u);
      if (!waited) {
        EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccess::Write);
        EXPECT_EQ(report.diagnostics[0].required_count, 0u);
      }
    }
  }
}

TEST(WaitcheckStream, LegacyVmemWritebackRemainsOrdered) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                    ROCJITSU_CODE_ARCH_RDNA3_5}) {
    const bool cdna = arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
    Program program = cdna ? Program{0xe0501000u, 0x80000008u, 0xe0501000u, 0x80000008u}
                           : Program{0xdc520000u, 8u, 0xdc520000u, 8u};
    const auto report = analyze(program, arch);
    ASSERT_FALSE(report.incomplete);
    EXPECT_TRUE(report.diagnostics.empty());
  }
}

TEST(WaitcheckStream, ScalarMemoryCannotUseAPartialWait) {
  Program program{0xf4000100u, 0xf8000000u,  // s_load_b32 s4, s[0:1], 0.
                  0xf4000180u, 0xf8000000u,  // s_load_b32 s6, s[0:1], 0.
                  0xbfc70001u, 0xbe880004u}; // s_wait_kmcnt 1; s_mov_b32 s8, s4.
  auto report = analyze(program);
  ASSERT_FALSE(report.incomplete);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  program[4] = 0xbfc70000u;
  report = analyze(program);
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckStream, UntrackedScalarMemoryDestinationsStopBeforeLosingDependencies) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    // Encodings assembled for gfx942: TTMP0, VCC, FLAT_SCRATCH and XNACK_MASK.
    for (const auto &[load, consumers] :
         std::array<std::pair<uint32_t, std::array<uint32_t, 3>>, 4>{{
             {0xc0021b00u, {0xbe84006cu, 0xbeec0004u, 0xbe84006cu}},
             {0xc0061a80u, {0xbe84016au, 0xbeea0104u, 0xbf860001u}},
             {0xc0061980u, {0xbe840166u, 0xbee60104u, 0xbe840166u}},
             {0xc0061a00u, {0xbe840168u, 0xbee80104u, 0xbe840168u}},
         }}) {
      for (uint32_t consumer : consumers) {
        SCOPED_TRACE(testing::Message()
                     << "arch=" << arch << " load=" << load << " consumer=" << consumer);
        const auto report = analyze({load, 0, consumer}, arch);
        ASSERT_TRUE(report.incomplete);
        EXPECT_EQ(report.incomplete->offset, 0u);
        EXPECT_EQ(report.instructions_analyzed, 0u);
        EXPECT_FALSE(report.incomplete->reason.empty());
      }
    }
  }
}

TEST(WaitcheckStream, TrapTemporaryUsesAndOverwritesAreIncomplete) {
  for (uint32_t instruction : {0xbe84006cu, 0xbeec0004u}) {
    const auto report = analyze({instruction}, ROCJITSU_CODE_ARCH_CDNA3);
    ASSERT_TRUE(report.incomplete);
    EXPECT_EQ(report.incomplete->offset, 0u);
    EXPECT_EQ(report.instructions_analyzed, 0u);
  }
}

TEST(WaitcheckStream, ScalarBufferStoresUseDecodedDataAndDescriptorWidths) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    for (uint32_t width : {1u, 2u}) {
      for (uint16_t loaded : {4u, 5u, 6u, 8u, 10u}) {
        // s_load_dword sN, s[0:1], 0; s_buffer_store_dword[x2] s4, s[8:11], 0.
        Program program{0xc0020000u | (loaded << 6), 0, width == 1 ? 0xc0620104u : 0xc0660104u, 0};
        const auto report = analyze(program, arch);
        ASSERT_FALSE(report.incomplete);
        const bool overlap = loaded < 4 + width || loaded >= 8;
        ASSERT_EQ(report.diagnostics.size(), overlap ? 1u : 0u);
        if (overlap) {
          EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, loaded, 1}));
          EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccess::Read);
        }
      }
    }
  }
}

TEST(WaitcheckStream, FlatLoadRequiresBothMemoryCounters) {
  Program program{0xec050000u, 0, 8, 0xbfc00000u, v_mov(1, 0)};
  auto report = analyze(program);
  ASSERT_FALSE(report.incomplete);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  program.insert(program.begin() + 4, 0xbfc60000u);
  report = analyze(program);
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckStream, ExpertModeProtectsVmemAddressSources) {
  Program program;
  load(program);
  program.push_back(v_mov(8, 10));
  const auto normal = analyze(program);
  EXPECT_FALSE(normal.incomplete);
  EXPECT_TRUE(normal.diagnostics.empty());
  auto report = analyze(program, ROCJITSU_CODE_ARCH_RDNA4, true);
  ASSERT_FALSE(report.incomplete);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].reg.index, 8u);
  program.insert(program.begin() + 3, 0xbf88ffe3u); // depctr_vm_vsrc(0).
  report = analyze(program, ROCJITSU_CODE_ARCH_RDNA4, true);
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckStream, MemoryWaitImpliesSourceLifetimeCompletion) {
  Program program;
  load(program);
  program.insert(program.end(), {0xbfc00000u, v_mov(8, 10)});
  const auto report = analyze(program, ROCJITSU_CODE_ARCH_RDNA4, true);
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckStream, SplitTargetsUseTheSameLoadCompletionPolicy) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    Program program;
    load(program);
    program.push_back(v_mov(1, 0));
    auto report = analyze(program, arch);
    ASSERT_FALSE(report.incomplete);
    ASSERT_EQ(report.diagnostics.size(), 1u);
    EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
    program.insert(program.begin() + 3, 0xbfc00000u);
    report = analyze(program, arch);
    EXPECT_FALSE(report.incomplete);
    EXPECT_TRUE(report.diagnostics.empty());
  }
}

TEST(WaitcheckStream, LegacyTargetsDecodeTheirOwnPackedWaitLayouts) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                    ROCJITSU_CODE_ARCH_RDNA3_5}) {
    const bool rdna = arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
    Program program = rdna ? Program{0xdc520000u, 8, v_mov(1, 0)}
                           : Program{0xe0501000u, 0x80000008u, v_mov(1, 0)};
    auto report = analyze(program, arch);
    ASSERT_FALSE(report.incomplete);
    ASSERT_EQ(report.diagnostics.size(), 1u);
    EXPECT_EQ(report.diagnostics[0].required_wait, "s_waitcnt vmcnt(0)");
    program.insert(program.begin() + 2, rdna ? 0xbf8903f7u : 0xbf8c0f70u);
    report = analyze(program, arch);
    EXPECT_FALSE(report.incomplete);
    EXPECT_TRUE(report.diagnostics.empty());
  }
}

TEST(WaitcheckStream, CdnaPreservesCommittedAndLiveInGenerations) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    for (uint32_t establish : {v_mov(4, 2), v_mov(5, 4)}) {
      Program program{establish,   0xd86c0000u, 0x04000000u, v_mov(5, 4),
                      0xbf8cc07fu, v_mov(4, 2), v_mov(5, 4)};
      auto report = analyze(program, arch);
      EXPECT_FALSE(report.incomplete);
      EXPECT_TRUE(report.diagnostics.empty());
    }
    const auto report = analyze({0xd86c0000u, 0x04000000u, v_mov(5, 4)}, arch);
    ASSERT_FALSE(report.incomplete);
    ASSERT_EQ(report.diagnostics.size(), 1u);
    EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  }
}

TEST(WaitcheckStream, CdnaWritesCannotReuseAnOldRegisterGeneration) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    for (bool old_value : {false, true}) {
      for (bool waited : {false, true}) {
        Program program;
        if (old_value)
          program.push_back(v_mov(0, 2));
        program.insert(program.end(), {0xe0501000u, 0x80000008u});
        if (waited)
          program.push_back(0xbf8c0f70u);
        program.push_back(v_mov(0, 2));
        const auto report = analyze(program, arch);
        ASSERT_FALSE(report.incomplete);
        ASSERT_EQ(report.diagnostics.size(), waited ? 0u : 1u);
        if (!waited) {
          EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccess::Write);
        }
      }
    }
  }
}

TEST(WaitcheckStream, EndProgramBoundsAnalysisBeforeUnrelatedBytes) {
  const auto report = analyze({0xbfb00000u, 0xffffffffu});
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
  EXPECT_EQ(report.instructions_analyzed, 1u);
}

TEST(WaitcheckStream, AbortTrapBoundsAnalysisButReturningTrapIsIncomplete) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    const auto abort = analyze({0xbf900002u, 0xffffffffu}, arch);
    EXPECT_FALSE(abort.incomplete);
    EXPECT_EQ(abort.instructions_analyzed, 1u);
    const auto returning = analyze({0xbf900001u, 0xbfc00000u}, arch);
    ASSERT_TRUE(returning.incomplete);
    EXPECT_EQ(returning.instructions_analyzed, 0u);
  }
}

TEST(WaitcheckStream, ControlFlowStopsBeforeAFallthroughWait) {
  Program program;
  load(program);
  program.insert(program.end(), {v_mov(1, 0), 0xbfa00001u, 0xbfc00000u, v_mov(2, 0)});
  const auto report = analyze(program);
  ASSERT_TRUE(report.incomplete);
  EXPECT_EQ(report.incomplete->offset, 16u);
  EXPECT_EQ(report.instructions_analyzed, 2u);
  EXPECT_EQ(report.diagnostics.size(), 1u);
}

TEST(WaitcheckStream, PcDestinationsStopBeforeAFallthroughWait) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    for (uint32_t transfer : {0xbe801f00u, 0xbe802e00u, 0x94800200u}) {
      // s_rfe_b64, s_cbranch_join, s_cbranch_g_fork have explicit PC destinations.
      Program program{0xe0501000u, 0x80000008u, transfer, 0xbf8c0f70u, v_mov(1, 0)};
      const auto report = analyze(program, arch);
      ASSERT_TRUE(report.incomplete);
      EXPECT_EQ(report.incomplete->offset, 8u);
      EXPECT_EQ(report.instructions_analyzed, 1u);
    }
  }
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    Program program;
    load(program);
    // s_rfe_b64 on RDNA4; s_rfe_i64 on CDNA5.
    program.insert(program.end(), {0xbe804a00u, 0xbfc00000u, v_mov(1, 0)});
    const auto report = analyze(program, arch);
    ASSERT_TRUE(report.incomplete);
    EXPECT_EQ(report.incomplete->offset, 12u);
    EXPECT_EQ(report.instructions_analyzed, 1u);
  }
}

TEST(WaitcheckStream, RegisterModeChangesAreIncomplete) {
  const auto report = analyze({0xbf860001u, v_mov(1, 0)}, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(report.incomplete);
  EXPECT_EQ(report.incomplete->offset, 0u);
  EXPECT_EQ(report.instructions_analyzed, 0u);
}

TEST(WaitcheckStream, PartialMemoryInstructionsAreIncomplete) {
  const auto report = analyze({0xda8c0000u, 0x01000001u}, ROCJITSU_CODE_ARCH_RDNA3_5);
  ASSERT_TRUE(report.incomplete);
  EXPECT_NE(report.incomplete->reason.find("partial"), std::string::npos);
  EXPECT_EQ(report.instructions_analyzed, 0u);
}

TEST(WaitcheckStream, DirectToLdsParityOrderingIsIncomplete) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    const auto report = analyze({0xe05d1000u, 0x80100008u}, arch);
    ASSERT_TRUE(report.incomplete);
    EXPECT_EQ(report.instructions_analyzed, 0u);
    EXPECT_NE(report.incomplete->reason.find("parity"), std::string::npos);
  }
}

TEST(WaitcheckStream, InvalidOptionsAndTruncatedInputFail) {
  for (auto options :
       {WaitcheckStreamOptions{.wave_size = 16}, WaitcheckStreamOptions{.wave_size = 128}}) {
    util::StringDiagnostic error;
    EXPECT_TRUE(
        analyze_waitcheck_stream({}, ROCJITSU_CODE_ARCH_RDNA4, options, error.emitter()).failed());
    EXPECT_FALSE(error.message().empty());
  }
  EXPECT_TRUE(analyze_waitcheck_stream({}, ROCJITSU_CODE_ARCH_CDNA3, {.wave_size = 32}).failed());
  EXPECT_TRUE(
      analyze_waitcheck_stream({}, ROCJITSU_CODE_ARCH_RDNA3, {.expert_scheduling = true}).failed());
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_RDNA1,
                    ROCJITSU_CODE_ARCH_RDNA2}) {
    SCOPED_TRACE(arch);
    EXPECT_TRUE(analyze_waitcheck_stream({}, arch, {.wave_size = 64}).failed());
  }
  for (const auto &words : {Program{0xee050000u}, Program{0xee050000u, 0}, Program{0xffffffffu}}) {
    util::StringDiagnostic error;
    EXPECT_TRUE(analyze_waitcheck_stream(words, ROCJITSU_CODE_ARCH_RDNA4, {.wave_size = 64},
                                         error.emitter())
                    .failed());
    EXPECT_FALSE(error.message().empty());
  }
}

TEST(WaitcheckStream, AtomicAndBarrierInstructionsAreIncomplete) {
  for (const auto &program : {Program{0xee0d4000u, 0x011c0000u, 1u}, Program{0xbf94ffffu}}) {
    const auto report = analyze(program);
    ASSERT_TRUE(report.incomplete);
    EXPECT_EQ(report.instructions_analyzed, 0u);
  }
}

TEST(WaitcheckStream, ScalarLoadsDoNotReuseOldRegisterGenerations) {
  const Program program{0xc0020100u, 0, 0xbf8cc07fu, 0xbe880004u, 0xc0020100u, 0, 0xbe880004u};
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    const auto report = analyze(program, arch);
    ASSERT_FALSE(report.incomplete);
    ASSERT_EQ(report.diagnostics.size(), 1u);
    EXPECT_EQ(report.diagnostics[0].consumer_offset, 24u);
    EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 4, 1}));
  }
}

TEST(WaitcheckStream, Gfx1250TranslationWaitProtectsAddressSources) {
  Program program;
  load(program);
  program.push_back(v_mov(8, 10));
  auto report = analyze(program, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_FALSE(report.incomplete);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  program.insert(program.begin() + 3, 0xbfc50000u);
  report = analyze(program, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckStream, Gfx1250TranslationGroupsDrainAtAGroupSwitch) {
  Program program;
  load(program);
  program.insert(program.end(), {0xf4000100u, 0xf8000000u, v_mov(8, 10)});
  const auto report = analyze(program, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckStream, Gfx1250VmemDestinationsImplicitlyWaitForAddressTranslations) {
  Program program;
  load(program);
  load(program, 8);
  const auto report = analyze(program, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckStream, BothWaveSizesAcceptOrdinaryMemoryStreams) {
  Program program;
  load(program);
  program.insert(program.end(), {0xbfc00000u, v_mov(1, 0)});
  for (uint32_t wave : {32u, 64u}) {
    const auto report = analyze(program, ROCJITSU_CODE_ARCH_RDNA4, false, wave);
    EXPECT_FALSE(report.incomplete);
    EXPECT_TRUE(report.diagnostics.empty());
  }
}

TEST(WaitcheckStream, ScratchAddressEnableDistinguishesV0FromNoAddress) {
  for (auto arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    const bool cdna = arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
    const bool split = arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
    for (bool enabled : {false, true}) {
      Program program;
      if (split)
        load(program);
      else
        program = cdna ? Program{0xe0501000u, 0x80000008u} : Program{0xdc520000u, 8};
      // scratch_load_{dword,b32} v1, {v0,off}, {off,s0}.
      const Program scratch = [&]() -> Program {
        if (split)
          return enabled ? Program{0xed05007cu, 0x00020001u, 0} : Program{0xed050000u, 1, 0};
        if (cdna)
          return enabled ? Program{0xdc506000u, 0x017f0000u} : Program{0xdc504000u, 0x01000000u};
        return enabled ? Program{0xdc510000u, 0x01fc0000u} : Program{0xdc510000u, 0x01000000u};
      }();
      const auto scratch_offset = program.size() * sizeof(uint32_t);
      program.insert(program.end(), scratch.begin(), scratch.end());
      const auto report = analyze(program, arch);
      ASSERT_FALSE(report.incomplete);
      ASSERT_EQ(report.diagnostics.size(), enabled ? 1u : 0u);
      if (enabled) {
        EXPECT_EQ(report.diagnostics[0].consumer_offset, scratch_offset);
        EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
      }
      if (split) {
        Program lifetime = scratch;
        lifetime.push_back(v_mov(0, 2));
        const auto source_report = analyze(lifetime, arch, true);
        ASSERT_FALSE(source_report.incomplete);
        if (enabled) {
          ASSERT_FALSE(source_report.diagnostics.empty());
          EXPECT_EQ(source_report.diagnostics[0].access, WaitcheckAccess::Write);
        } else {
          EXPECT_TRUE(source_report.diagnostics.empty());
        }
      }
    }
  }
}

TEST(WaitcheckStream, VectorSkipStopsBeforeCountingSkippedMemory) {
  const Program program{0xe0501000u, 0x80000008u, 0xbf108081u, // load v0; s_setvskip 1, 0.
                        0xe0501000u, 0x80000208u, 0xbf108080u, // load v2; s_setvskip 0, 0.
                        0xbf8c0f71u, v_mov(1, 0)};             // vmcnt(1); consume v0.
  const auto report = analyze(program, ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_TRUE(report.incomplete);
  EXPECT_EQ(report.incomplete->offset, 8u);
  EXPECT_EQ(report.instructions_analyzed, 1u);
}

TEST(WaitcheckStream, CdnaLoadAddressDoesNotMakeItsNewResultReady) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    // global_load_dword v0, v0, s[0:1]; v_mov_b32 v1, v0.
    const auto report = analyze({0xdc508000u, 0, v_mov(1, 0)}, arch);
    EXPECT_FALSE(report.incomplete);
    ASSERT_EQ(report.diagnostics.size(), 1u);
    EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  }
}

TEST(WaitcheckStream, AluSpacingDoesNotHideLaterMemoryDependencies) {
  const auto report = analyze({0xdc520000u, 8, 0xbf870001u, v_mov(1, 0)}, ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_FALSE(report.incomplete);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].consumer_offset, 12u);
}

TEST(WaitcheckStream, ControlFlowBoundaryChecksKnownScalarOperands) {
  for (bool write : {false, true}) {
    for (bool waited : {false, true}) {
      // s_load_b64 s[0:1] or s[4:5], s[2:3], 0.
      Program program{write ? 0xf4002101u : 0xf4002001u, 0xf8000000u};
      if (waited)
        program.push_back(0xbfc70000u); // s_wait_kmcnt 0.
      // s_setpc_b64 s[0:1] or s_swappc_b64 s[4:5], s[0:1].
      const auto boundary = program.size() * sizeof(uint32_t);
      program.push_back(write ? 0xbe844900u : 0xbe804800u);
      const auto report = analyze(program);
      ASSERT_TRUE(report.incomplete);
      EXPECT_EQ(report.incomplete->offset, boundary);
      EXPECT_EQ(report.instructions_analyzed, waited ? 2u : 1u);
      ASSERT_EQ(report.diagnostics.size(), waited ? 0u : 1u);
      if (!waited) {
        EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
        EXPECT_EQ(report.diagnostics[0].consumer_offset, boundary);
        EXPECT_EQ(report.diagnostics[0].access,
                  write ? WaitcheckAccess::Write : WaitcheckAccess::Read);
      }
    }
  }
}

TEST(WaitcheckStream, RelativeRegisterOperandsAreIncomplete) {
  // Effective operands use m0 rather than the encoded base register alone.
  for (uint32_t relative : {0x7e00d101u, 0x7e028700u, 0x7e008501u, 0xbe804004u, 0xbe804104u,
                            0xbe804204u, 0xbe804304u}) {
    Program program{0xbefd0084u}; // s_mov_b32 m0, 4.
    load(program, 4);
    program.insert(program.end(), {relative, 0xbfb00000u});
    const auto report = analyze(program);
    ASSERT_TRUE(report.incomplete);
    EXPECT_EQ(report.incomplete->offset, 16u);
    EXPECT_EQ(report.instructions_analyzed, 2u);
    EXPECT_NE(report.incomplete->reason.find("relative-register"), std::string::npos);
  }
}

TEST(WaitcheckStream, ScalarMaskDestinationsFollowWaveWidth) {
  // Carry outputs (add/subtract/multiply-add) and f32/f64 divide-scale define s2
  // in Wave32 and s[2:3] in Wave64. Encodings checked with LLVM's assembler.
  const std::array<std::array<uint32_t, 2>, 10> encodings{{{0xd7000200u, 0x02020501u},
                                                           {0xd7010200u, 0x02020501u},
                                                           {0xd7020200u, 0x02020501u},
                                                           {0xd5200200u, 0x00120501u},
                                                           {0xd5210200u, 0x00120501u},
                                                           {0xd5220200u, 0x00120501u},
                                                           {0xd6fc0200u, 0x040e0501u},
                                                           {0xd6fd0200u, 0x041a0902u},
                                                           {0xd6fe0200u, 0x040e0501u},
                                                           {0xd6ff0200u, 0x040e0501u}}};
  for (const std::array<uint32_t, 2> &encoding : encodings) {
    for (uint32_t wave : {32u, 64u}) {
      for (uint32_t loaded : {2u, 3u}) {
        const auto report =
            analyze({0xf4000002u | (loaded << 6), 0xf8000000u, encoding[0], encoding[1]},
                    ROCJITSU_CODE_ARCH_RDNA4, false, wave);
        ASSERT_FALSE(report.incomplete);
        const bool conflicts = wave == 64 || loaded == 2;
        ASSERT_EQ(report.diagnostics.size(), conflicts ? 1u : 0u);
        if (conflicts) {
          EXPECT_EQ(report.diagnostics[0].reg.index, loaded);
          EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccess::Write);
        }
      }
    }
  }
}

TEST(WaitcheckStream, LegacyMadScalarDestinationsFollowWaveWidth) {
  // RDNA3/3.5 use v_mad_{u64_u32,i64_i32}; RDNA4 adds _co to their names.
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    for (uint32_t opcode : {0xd6fe0200u, 0xd6ff0200u}) {
      for (uint32_t wave : {32u, 64u}) {
        for (uint32_t loaded : {2u, 3u}) {
          const auto report = analyze(
              {0xf4000002u | (loaded << 6), 0xf8000000u, opcode, 0x040e0501u}, arch, false, wave);
          ASSERT_FALSE(report.incomplete);
          EXPECT_EQ(report.diagnostics.size(), wave == 64 || loaded == 2 ? 1u : 0u);
        }
      }
    }
  }
}

TEST(WaitcheckStream, EmptyStreamHasNoHazards) {
  const auto report = analyze({});
  EXPECT_FALSE(report.incomplete);
  EXPECT_TRUE(report.diagnostics.empty());
  EXPECT_EQ(report.instructions_analyzed, 0u);
}
} // namespace
} // namespace rocjitsu
