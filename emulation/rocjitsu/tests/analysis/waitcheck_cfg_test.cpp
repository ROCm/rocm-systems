// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/cfg.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstring>

namespace rocjitsu {
namespace {
using Program = std::vector<uint32_t>;
constexpr auto kArch = ROCJITSU_CODE_ARCH_RDNA4;
constexpr auto kCdna = ROCJITSU_CODE_ARCH_CDNA4;
constexpr uint32_t kEnd = build_s_endpgm(kArch);
constexpr uint32_t kWait = 0xbfc00000u;
uint32_t mov(uint32_t dst, uint32_t src) { return 0x7e000300u | (dst << 17) | src; }
uint32_t branch(int16_t delta) { return build_s_branch(delta, kArch); }
uint32_t conditional(int16_t delta) { return 0xbfa10000u | static_cast<uint16_t>(delta); }
void load(Program &program, uint8_t dst = 0) {
  rdna4::VglobalMachineInst inst{};
  inst.encoding = 0xee;
  inst.op = 20;
  inst.vdst = dst;
  inst.vaddr = 8;
  auto words = std::bit_cast<std::array<uint32_t, sizeof(inst) / 4>>(inst);
  program.insert(program.end(), words.begin(), words.end());
}
class TestTextSection final : public Section {
public:
  explicit TestTextSection(const Program &program)
      : Section(".text", std::make_unique<char[]>(program.size() * 4)), size_(program.size() * 4) {
    if (!program.empty())
      std::memcpy(data_.get(), program.data(), size_);
  }
  size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  size_t size_;
};

class TestCodeObject final : public CodeObject {
public:
  explicit TestCodeObject(const Program &program) {
    sections_.push_back(std::make_unique<TestTextSection>(program));
    text_sections_.push_back(sections_.back().get());
  }
};

WaitcheckCfgReport analyze(const Program &program, std::vector<uint64_t> entries = {0},
                           WaitcheckCfgOptions options = {}, rj_code_arch_t arch = kArch) {
  TestCodeObject object(program);
  util::StringDiagnostic error;
  auto result = analyze_waitcheck_cfg(object, arch, entries, options, error.emitter());
  EXPECT_TRUE(result.succeeded()) << error.message();
  return result.succeeded() ? std::move(result).value() : WaitcheckCfgReport{};
}

TEST(WaitcheckCfg, CarriesDependenciesAcrossBranches) {
  Program program;
  load(program);
  program.insert(program.end(), {branch(1), 0xffffffffu, mov(1, 0), kEnd});
  auto report = analyze(program);
  EXPECT_TRUE(report.incomplete.empty());
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].producer_offset, 0u);
  EXPECT_EQ(report.diagnostics[0].consumer_offset, 20u);
  EXPECT_EQ(report.blocks_analyzed, 2u);
  EXPECT_EQ(report.instructions_analyzed, 4u);
  program.insert(program.begin() + 5, kWait);
  report = analyze(program);
  EXPECT_TRUE(report.incomplete.empty());
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckCfg, JoinsWaitedAndUnwaitedPaths) {
  for (bool both_wait : {false, true}) {
    Program program;
    load(program);
    program.insert(program.end(), {conditional(2), kWait, branch(1),
                                   both_wait ? kWait : build_s_nop(0, kArch), mov(1, 0), kEnd});
    const auto report = analyze(program);
    EXPECT_TRUE(report.incomplete.empty());
    EXPECT_EQ(report.diagnostics.size(), both_wait ? 0u : 1u);
  }
}

TEST(WaitcheckCfg, JoinsDistinctProducersWithoutDuplicatingDiagnostics) {
  Program program{conditional(4)};
  load(program);
  program.push_back(branch(3));
  load(program);
  program.insert(program.end(), {mov(1, 0), kEnd});
  const auto report = analyze(program);
  EXPECT_TRUE(report.incomplete.empty());
  ASSERT_EQ(report.diagnostics.size(), 2u);
  EXPECT_EQ(report.diagnostics[0].producer_offset, 4u);
  EXPECT_EQ(report.diagnostics[1].producer_offset, 20u);
  for (const auto &d : report.diagnostics)
    EXPECT_EQ(d.consumer_offset, 32u);
}

TEST(WaitcheckCfg, LoopCarriesProducerToAnEarlierConsumer) {
  Program program{mov(1, 0)};
  load(program);
  program.insert(program.end(), {conditional(-5), kEnd});
  const auto report = analyze(program);
  EXPECT_TRUE(report.incomplete.empty());
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].producer_offset, 4u);
  EXPECT_EQ(report.diagnostics[0].consumer_offset, 0u);
  EXPECT_EQ(report.instructions_analyzed, 4u);
}

TEST(WaitcheckCfg, LoopWaitEliminatesTheCarriedHazard) {
  Program program{kWait, mov(1, 0)};
  load(program);
  program.insert(program.end(), {conditional(-6), kEnd});
  const auto report = analyze(program);
  EXPECT_TRUE(report.incomplete.empty());
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckCfg, ExternalEntryWithBackedgeCannotInheritMustReadiness) {
  // First iteration has no committed v4. The loop tail commits one, but must
  // not make the first iteration's load appear to have an older generation.
  Program program{0xd86c0000u, 0x04000000u, mov(5, 4), 0xbf8cc07fu, build_s_branch(-5, kCdna)};
  const auto report = analyze(program, {0}, {}, kCdna);
  EXPECT_TRUE(report.incomplete.empty());
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].consumer_offset, 8u);
}

TEST(WaitcheckCfg, CdnaLiveInReadinessCrossesBlockBoundaries) {
  for (bool establish : {false, true}) {
    Program program{establish ? mov(5, 4) : build_s_nop(0, kCdna),
                    build_s_branch(0, kCdna),
                    0xd86c0000u,
                    0x04000000u,
                    mov(5, 4),
                    build_s_endpgm(kCdna)};
    auto report = analyze(program, {0}, {}, kCdna);
    EXPECT_TRUE(report.incomplete.empty());
    EXPECT_EQ(report.diagnostics.size(), establish ? 0u : 1u);
    // A fresh external entry at the load must intersect the inherited value.
    report = analyze(program, {0, 8}, {}, kCdna);
    EXPECT_EQ(report.diagnostics.size(), 1u);
  }
}

TEST(WaitcheckCfg, SameAddressLoadDoesNotInventAnOlderGeneration) {
  const Program program{0xdc508000u, 0, build_s_branch(0, kCdna), mov(1, 0), build_s_endpgm(kCdna)};
  const auto report = analyze(program, {0}, {}, kCdna);
  EXPECT_TRUE(report.incomplete.empty());
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].consumer_offset, 12u);
}

TEST(WaitcheckCfg, PostJoinWaitMakesACommonCompletedGenerationAvailable) {
  // Common ds_read on both paths, waited on one. The post-join wait completes
  // the other; a new ds_read may then expose the older v4 generation.
  Program program{
      0xd86c0000u,           0x04000000u, pack_sopp(5, 2), 0xbf8cc07fu, build_s_branch(1, kCdna),
      build_s_nop(0, kCdna), 0xbf8cc07fu, 0xd86c0000u,     0x04000000u, mov(5, 4),
      build_s_endpgm(kCdna)};
  const auto report = analyze(program, {0}, {}, kCdna);
  EXPECT_TRUE(report.incomplete.empty());
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckCfg, DisconnectedAndUnselectedCodeIsNotAnalyzed) {
  Program program{0xffffffffu};
  load(program);
  program.insert(program.end(), {mov(1, 0), kEnd, 0xffffffffu});
  const auto report = analyze(program, {4});
  EXPECT_TRUE(report.incomplete.empty());
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].producer_offset, 4u);
  EXPECT_EQ(report.blocks_analyzed, 1u);
  EXPECT_EQ(analyze(program, {}).instructions_analyzed, 0u);
}

TEST(WaitcheckCfg, DeferredPathDoesNotPoisonOrAnalyzeItsContinuation) {
  Program program{conditional(4), 0xbf94ffffu, mov(1, 0), branch(5), 0xffffffffu}; // s_barrier.
  load(program);
  program.insert(program.end(), {mov(1, 0), kEnd});
  const auto report = analyze(program);
  ASSERT_EQ(report.incomplete.size(), 1u);
  EXPECT_EQ(report.incomplete[0].offset, 4u);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].producer_offset, 20u);
  EXPECT_EQ(report.diagnostics[0].consumer_offset, 32u);
}

TEST(WaitcheckCfg, IndirectTransferChecksOperandsAndStops) {
  // s_load_b64 s[0:1], s[2:3], 0; s_setpc_b64 s[0:1].
  const Program program{0xf4002001u, 0xf8000000u, 0xbe804800u};
  const auto report = analyze(program);
  ASSERT_EQ(report.incomplete.size(), 1u);
  EXPECT_EQ(report.incomplete[0].offset, 8u);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
}

TEST(WaitcheckCfg, DirectCallsAreExplicitCoverageBoundaries) {
  const Program program{build_s_call_b64(0, 1, kArch), kEnd, kEnd}; // s_call_b64 s[0:1], 1.
  const auto report = analyze(program);
  ASSERT_EQ(report.incomplete.size(), 1u);
  EXPECT_EQ(report.incomplete[0].offset, 0u);
  EXPECT_EQ(report.blocks_analyzed, 1u);
}

TEST(WaitcheckCfg, MissingSuccessorIsIncompleteNotSuccess) {
  Program program;
  load(program);
  program.insert(program.end(), {mov(1, 0), branch(20)});
  const auto report = analyze(program);
  EXPECT_EQ(report.diagnostics.size(), 1u);
  ASSERT_EQ(report.incomplete.size(), 1u);
  EXPECT_EQ(report.incomplete[0].offset, 16u);
}

TEST(WaitcheckCfg, ExcludedTargetReportsIncompleteCoverage) {
  const std::array ranges{BasicBlock::CodeRange{0, 4}};
  const auto report = analyze({branch(0), kEnd}, {0}, {.permitted_ranges = ranges});
  ASSERT_EQ(report.incomplete.size(), 1u);
  EXPECT_EQ(report.instructions_analyzed, 1u);
}

TEST(WaitcheckCfg, ExpertSourceLifetimeSurvivesBranch) {
  for (bool expert : {false, true}) {
    Program program;
    load(program);
    program.insert(program.end(), {branch(0), mov(8, 10), kEnd});
    const auto report = analyze(program, {0}, {.entry_modes = {.expert_scheduling = expert}});
    EXPECT_TRUE(report.incomplete.empty());
    EXPECT_EQ(report.diagnostics.size(), expert ? 1u : 0u);
  }
}

TEST(WaitcheckCfg, CounterOnlyRequestsReleaseSourcesAcrossBlocks) {
  Program program;
  load(program);
  program.insert(program.end(), {branch(0), 0xee0ac07cu, 0, 0, kWait | 1u, mov(8, 10), kEnd});
  const auto report = analyze(program, {0}, {.entry_modes = {.expert_scheduling = true}});
  EXPECT_TRUE(report.incomplete.empty());
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckCfg, RelativeRegisterOperandsStopPropagation) {
  // v_movrels/reld_b32 and s_movrels/reld_b32/b64 index the encoded base with m0.
  for (uint32_t relative :
       {0x7e028700u, 0x7e008501u, 0xbe804004u, 0xbe804104u, 0xbe804204u, 0xbe804304u}) {
    Program program{0xbefd0084u}; // s_mov_b32 m0, 4.
    load(program, 4);
    program.insert(program.end(), {branch(0), relative, branch(0), mov(1, 4), kEnd});
    const auto report = analyze(program);
    ASSERT_EQ(report.incomplete.size(), 1u);
    EXPECT_EQ(report.incomplete[0].offset, 20u);
    EXPECT_EQ(report.blocks_analyzed, 2u);
    EXPECT_EQ(report.instructions_analyzed, 3u);
  }
}

TEST(WaitcheckCfg, WaveWidthControlsCrossBlockScalarMaskHazards) {
  // s_load_b32 s3, s[4:5], 0; branch; v_div_scale_f32 v0, s[2:3], v1, v2, v3.
  for (uint32_t wave : {32u, 64u}) {
    const Program program{0xf40000c2u, 0xf8000000u, branch(0), 0xd6fc0200u, 0x040e0501u, kEnd};
    const auto report = analyze(program, {0}, {.entry_modes = {.wave_size = wave}});
    EXPECT_TRUE(report.incomplete.empty());
    EXPECT_EQ(report.diagnostics.size(), wave == 32 ? 0u : 1u);
  }
}

TEST(WaitcheckCfg, Gfx1250TranslationLifetimeCrossesBranches) {
  for (bool waited : {false, true}) {
    Program program;
    load(program);
    program.push_back(build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5));
    if (waited)
      program.push_back(0xbfc50000u); // s_wait_xcnt 0.
    program.insert(program.end(), {mov(8, 10), kEnd});
    const auto report = analyze(program, {0}, {}, ROCJITSU_CODE_ARCH_CDNA5);
    EXPECT_TRUE(report.incomplete.empty());
    ASSERT_EQ(report.diagnostics.size(), waited ? 0u : 1u);
    if (!waited) {
      EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
    }
  }
}

TEST(WaitcheckCfg, Gfx1250GroupSwitchDrainsIncomingTranslations) {
  Program program;
  load(program);
  program.insert(program.end(), {build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5), 0xf4000100u,
                                 0xf8000000u, mov(8, 10), kEnd});
  const auto report = analyze(program, {0}, {}, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_TRUE(report.incomplete.empty());
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckCfg, ExplicitCalleeEntryChecksItsBodyAsAnIndependentRoot) {
  Program program{build_s_call_b64(0, 1, kArch), kEnd};
  load(program);
  program.insert(program.end(), {mov(1, 0), kEnd});
  auto report = analyze(program);
  EXPECT_EQ(report.incomplete.size(), 1u);
  EXPECT_EQ(report.blocks_analyzed, 1u);
  EXPECT_TRUE(report.diagnostics.empty());
  EXPECT_EQ(report.incomplete[0].reason, "calls and indirect control flow are not followed");
  report = analyze(program, {0, 8});
  EXPECT_EQ(report.incomplete.size(), 1u);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].producer_offset, 8u);
}

TEST(WaitcheckCfg, InvalidInputsAndBudgetExhaustionReturnFailure) {
  const std::array<uint64_t, 1> entries{0};
  for (auto program : {Program{0xffffffffu}, Program{branch(0), kEnd}}) {
    TestCodeObject object(program);
    util::StringDiagnostic error;
    const auto result =
        analyze_waitcheck_cfg(object, kArch, entries, {.max_block_visits = 1}, error.emitter());
    EXPECT_TRUE(result.failed());
    EXPECT_FALSE(error.message().empty());
  }
  TestCodeObject object({kEnd});
  for (auto options : {WaitcheckCfgOptions{.max_block_visits = 0},
                       WaitcheckCfgOptions{.entry_modes = {.wave_size = 16}}}) {
    util::StringDiagnostic error;
    EXPECT_TRUE(analyze_waitcheck_cfg(object, kArch, entries, options, error.emitter()).failed());
    EXPECT_FALSE(error.message().empty());
  }
}
} // namespace
} // namespace rocjitsu
