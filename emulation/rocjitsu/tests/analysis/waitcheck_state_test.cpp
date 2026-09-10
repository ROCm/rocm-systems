// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/state.h"
#include "rocjitsu/isa/decoder.h"

#include <array>
#include <memory>
#include <span>

#include <gtest/gtest.h>

namespace rocjitsu::waitcheck_detail {
namespace {
using Ops = WaitcheckStateOps;
const size_t kLoad = Ops::counter_index(WaitCounterKind::Load);

PendingEvent load(uint16_t reg, uint32_t younger, uint64_t offset = 0) {
  PendingEvent event;
  event.counter = WaitCounterKind::Load;
  event.kind = WaitEventKind::VmemNoSamplerLoad;
  event.regs.expand({RegClass::VGPR, reg, 1});
  event.produces_regs = true;
  event.min_younger = younger;
  event.section_offset = offset;
  return event;
}

TEST(WaitcheckState, PartialWaitRetiresOnlySufficientlyOldEvents) {
  PendingState state;
  state.pending[kLoad] = {load(0, 1), load(1, 0, 4)};
  Ops::apply_wait(state, WaitCounterKind::Load, 1);
  ASSERT_EQ(state.pending[kLoad].size(), 1u);
  EXPECT_TRUE(state.pending[kLoad][0].regs.contains({RegClass::VGPR, 1, 1}));
  EXPECT_TRUE(state.ready_regs.contains({RegClass::VGPR, 0, 1}));
  EXPECT_FALSE(state.ready_regs.contains({RegClass::VGPR, 1, 1}));
  Ops::apply_wait(state, WaitCounterKind::Load, 0);
  EXPECT_TRUE(state.pending[kLoad].empty());
  EXPECT_TRUE(state.ready_regs.contains({RegClass::VGPR, 1, 1}));
}

TEST(WaitcheckState, SharedGenerationBecomesReadyAfterAllCountersRetire) {
  PendingState state;
  auto event = load(0, 0);
  event.kind = WaitEventKind::FlatLoad;
  state.pending[kLoad].push_back(event);
  event.counter = WaitCounterKind::Ds;
  state.pending[Ops::counter_index(WaitCounterKind::Ds)].push_back(event);
  Ops::apply_wait(state, WaitCounterKind::Load, 0);
  EXPECT_FALSE(state.ready_regs.contains({RegClass::VGPR, 0, 1}));
  Ops::apply_wait(state, WaitCounterKind::Ds, 0);
  EXPECT_TRUE(state.ready_regs.contains({RegClass::VGPR, 0, 1}));
}

TEST(WaitcheckState, ZeroWaitClearsCounterOrderingFacts) {
  PendingState state;
  state.pending_smem[kLoad] = true;
  state.uncertain_order[kLoad] = true;
  state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::Smem)] = 0;
  Ops::apply_wait(state, WaitCounterKind::Load, 0);
  EXPECT_FALSE(state.pending_smem[kLoad]);
  EXPECT_FALSE(state.uncertain_order[kLoad]);
  EXPECT_EQ(state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::Smem)],
            kNoPendingEventAge);
}

TEST(WaitcheckState, PartialWaitCannotRetireAnOutOfOrderScalarCounter) {
  PendingState state;
  const size_t ds = Ops::counter_index(WaitCounterKind::Ds);
  auto event = load(0, 2);
  event.counter = WaitCounterKind::Ds;
  event.kind = WaitEventKind::Ds;
  state.pending[ds].push_back(event);
  state.pending_smem[ds] = true;
  ASSERT_TRUE(
      Ops::apply_memory_wait(state, WaitCounterKind::Ds, 1, ROCJITSU_CODE_ARCH_CDNA4).succeeded());
  EXPECT_EQ(state.pending[ds].size(), 1u);
  EXPECT_EQ(Ops::dependency_required_count(state, event, ROCJITSU_CODE_ARCH_CDNA4).value(), 0u);
  ASSERT_TRUE(
      Ops::apply_memory_wait(state, WaitCounterKind::Ds, 0, ROCJITSU_CODE_ARCH_CDNA4).succeeded());
  EXPECT_TRUE(state.pending[ds].empty());
}

TEST(WaitcheckState, MergeUsesTheLeastProgressAndIntersectsReadyRegisters) {
  std::vector<PendingState> outputs(2);
  outputs[0].pending[kLoad].push_back(load(0, 2));
  outputs[1].pending[kLoad].push_back(load(0, 1));
  outputs[0].ready_regs.expand({RegClass::VGPR, 8, 2});
  outputs[1].ready_regs.expand({RegClass::VGPR, 9, 2});
  const std::array<size_t, 2> predecessors{0, 1};
  const std::array<uint8_t, 2> initialized{1, 1};
  auto merged = Ops::merge_predecessors(predecessors, outputs, initialized);
  ASSERT_EQ(merged.pending[kLoad].size(), 1u);
  EXPECT_EQ(merged.pending[kLoad][0].min_younger, 1u);
  EXPECT_TRUE(merged.uncertain_order[kLoad]);
  EXPECT_EQ(merged.ready_regs.size(), 1u);
  EXPECT_TRUE(merged.ready_regs.contains({RegClass::VGPR, 9, 1}));
  auto repeated = merged;
  Ops::merge_into(repeated, merged);
  EXPECT_EQ(repeated, merged);
}

TEST(WaitcheckState, UnvisitedPredecessorDoesNotContributeAnEmptyState) {
  std::vector<PendingState> outputs(2);
  outputs[0].pending[kLoad].push_back(load(0, 1));
  outputs[0].ready_regs.expand({RegClass::VGPR, 8, 1});
  const std::array<size_t, 2> predecessors{0, 1};
  const std::array<uint8_t, 2> initialized{1, 0};
  EXPECT_EQ(Ops::merge_predecessors(predecessors, outputs, initialized), outputs[0]);
}

TEST(WaitcheckState, DisagreeingModesBecomeUnknownAtAJoin) {
  std::vector<PendingState> outputs(2);
  outputs[0].vgpr_msb.mode = 0;
  outputs[1].vgpr_msb.mode = 1;
  outputs[0].expert_scheduling.enabled = false;
  outputs[1].expert_scheduling.enabled = true;
  const std::array<size_t, 2> predecessors{0, 1};
  const std::array<uint8_t, 2> initialized{1, 1};
  const auto merged = Ops::merge_predecessors(predecessors, outputs, initialized);
  EXPECT_FALSE(merged.vgpr_msb.known);
  EXPECT_FALSE(merged.expert_scheduling.known);
  EXPECT_TRUE(merged.expert_scheduling.enabled);
}

std::unique_ptr<Instruction> decode_wait(uint32_t word, rj_code_arch_t arch) {
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  util::StringDiagnostic error;
  DecodeResult result = decoder->decode_window(std::span(&word, 1), 0, error.emitter());
  if (result.failed()) {
    ADD_FAILURE() << error.message();
    return nullptr;
  }
  return std::move(result).value();
}

TEST(WaitcheckState, InstructionWaitsRespectSentinelsAndArchitecture) {
  PendingState state;
  state.pending[kLoad] = {load(0, 63)};
  const PendingState before = state;
  std::unique_ptr<Instruction> sentinel = decode_wait(0xbfc0003fu, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(sentinel, nullptr);
  ASSERT_TRUE(Ops::apply_waitcnt(state, *sentinel, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_EQ(state, before);
  std::unique_ptr<Instruction> wait = decode_wait(0xbfc00000u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(wait, nullptr);
  ASSERT_TRUE(Ops::apply_waitcnt(state, *wait, ROCJITSU_CODE_ARCH_CDNA4).succeeded());
  EXPECT_EQ(state, before);
  Instruction missing_operand("s_wait_loadcnt", nullptr);
  ASSERT_TRUE(Ops::apply_waitcnt(state, missing_operand, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_EQ(state, before);
  ASSERT_TRUE(Ops::apply_waitcnt(state, *wait, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_TRUE(state.pending[kLoad].empty());
}

TEST(WaitcheckState, UnsupportedArchitectureFailsWithoutChangingState) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                              ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    PendingState state;
    state.pending[kLoad] = {load(0, 3)};
    state.va_vdst_hazards.hazards[0] = {2, false, {}};
    const PendingState before = state;
    const Instruction idle("s_wait_idle", nullptr);
    const Instruction ordinary("s_nop", nullptr);
    std::array<std::optional<uint32_t>, kCounterCount> fields;
    fields[kLoad] = 0;
    fields[Ops::counter_index(WaitCounterKind::VaVdst)] = 0;
    EXPECT_TRUE(Ops::apply_waitcnt(state, idle, arch).failed());
    EXPECT_EQ(state, before);
    EXPECT_TRUE(Ops::apply_embedded_waitcnt(state, ordinary, arch).failed());
    EXPECT_EQ(state, before);
    EXPECT_TRUE(Ops::apply_wait_fields(state, fields, arch).failed());
    EXPECT_EQ(state, before);
    for (uint32_t count : {0u, 1u}) {
      EXPECT_TRUE(Ops::apply_counter_wait(state, WaitCounterKind::Load, count, arch).failed());
      EXPECT_EQ(state, before);
      EXPECT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, count, arch).failed());
      EXPECT_EQ(state, before);
      EXPECT_TRUE(Ops::apply_kmcnt_wait(state, count, arch).failed());
      EXPECT_EQ(state, before);
    }
    EXPECT_TRUE(Ops::counter_out_of_order(state, WaitCounterKind::Load, arch).failed());
    EXPECT_TRUE(Ops::dependency_required_count(state, state.pending[kLoad][0], arch).failed());
  }
}

TEST(WaitcheckState, LegacyPackedWaitRetiresOnlyActiveFields) {
  PendingState state;
  state.pending[kLoad] = {load(0, 2), load(1, 1, 4)};
  const size_t ds = Ops::counter_index(WaitCounterKind::Ds);
  const size_t exp = Ops::counter_index(WaitCounterKind::Exp);
  state.pending[ds] = {load(2, 0)};
  state.pending[exp] = {load(3, 7)};
  std::unique_ptr<Instruction> wait = decode_wait(0xbf8c0072u, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(wait, nullptr);
  ASSERT_TRUE(Ops::apply_waitcnt(state, *wait, ROCJITSU_CODE_ARCH_CDNA4).succeeded());
  ASSERT_EQ(state.pending[kLoad].size(), 1u);
  EXPECT_TRUE(state.pending[kLoad][0].regs.contains({RegClass::VGPR, 1, 1}));
  EXPECT_TRUE(state.pending[ds].empty());
  EXPECT_EQ(state.pending[exp].size(), 1u);
}

TEST(WaitcheckState, DirectJoinIncludesModesReadinessAndSgprProgress) {
  PendingState left, right;
  left.ready_regs.expand({RegClass::VGPR, 0, 2});
  right.ready_regs.expand({RegClass::VGPR, 1, 2});
  left.previous_vm_vsrc_zero_wait = true;
  left.vgpr_msb.mode = 1;
  right.expert_scheduling.enabled = true;
  left.sgpr_hazards.consecutive_ds_nops = 3;
  right.sgpr_hazards.consecutive_ds_nops = 2;
  const std::vector<PendingState> outputs{left, right};
  const std::array<size_t, 2> predecessors{0, 1};
  const std::array<uint8_t, 2> initialized{1, 1};
  Ops::merge_into(left, right);
  EXPECT_EQ(left, Ops::merge_predecessors(predecessors, outputs, initialized));
  EXPECT_EQ(left.ready_regs.size(), 1u);
  EXPECT_FALSE(left.previous_vm_vsrc_zero_wait);
  EXPECT_FALSE(left.vgpr_msb.known);
  EXPECT_FALSE(left.expert_scheduling.known);
  EXPECT_EQ(left.sgpr_hazards.consecutive_ds_nops, 2u);
  const std::array<uint8_t, 2> first_only{1, 0};
  EXPECT_EQ(Ops::merge_predecessors(predecessors, outputs, first_only), outputs[0]);
}

TEST(WaitcheckState, DelayedWaitMustBeGuaranteedOnEveryIncomingPath) {
  PendingState left, right;
  left.sgpr_hazards.salu_hazards.set(0);
  right.sgpr_hazards.salu_hazards.set(0);
  left.delay_alu.push_back({1, DelayAluEffect::Salu});
  Ops::merge_into(left, right);
  EXPECT_TRUE(left.delay_alu.empty());
  EXPECT_TRUE(left.sgpr_hazards.salu_hazards.test(0));
  left.delay_alu = {
      {1, DelayAluEffect::Salu}, {3, DelayAluEffect::Salu}, {1, DelayAluEffect::Valu}};
  right.delay_alu = {{2, DelayAluEffect::Salu}};
  Ops::merge_into(left, right);
  ASSERT_EQ(left.delay_alu.size(), 1u);
  EXPECT_EQ(left.delay_alu[0].effect, DelayAluEffect::Salu);
  EXPECT_EQ(left.delay_alu[0].countdown, 2u);
}

TEST(WaitcheckState, CrossedIssueOrderRetainsEveryPossiblyPendingEvent) {
  PendingState left, right;
  left.pending[kLoad] = {load(0, 1, 0), load(1, 0, 4)};
  right.pending[kLoad] = {load(0, 0, 0), load(1, 1, 4)};
  Ops::merge_into(left, right);
  ASSERT_TRUE(
      Ops::apply_memory_wait(left, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_EQ(left.pending[kLoad].size(), 2u);
  ASSERT_TRUE(
      Ops::apply_memory_wait(left, WaitCounterKind::Load, 0, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_TRUE(left.pending[kLoad].empty());
}

TEST(WaitcheckState, JoinedAgeRemainsALowerBoundForEachEvent) {
  PendingState left, right;
  left.pending[kLoad] = {load(0, 2, 0), load(1, 1, 4), load(2, 0, 8)};
  right.pending[kLoad] = {load(0, 1, 0), load(1, 2, 4), load(2, 0, 8)};
  Ops::merge_into(left, right);
  ASSERT_TRUE(
      Ops::apply_memory_wait(left, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  ASSERT_EQ(left.pending[kLoad].size(), 1u);
  EXPECT_TRUE(left.pending[kLoad][0].regs.contains({RegClass::VGPR, 2, 1}));
}

TEST(WaitcheckState, CounterOnlySmemPreventsPartialXcntRetirement) {
  PendingState state;
  const size_t x = Ops::counter_index(WaitCounterKind::X);
  PendingEvent event = load(0, 2);
  event.counter = WaitCounterKind::X;
  state.pending[x].push_back(event);
  state.pending_event_ages[x].values[static_cast<size_t>(WaitEventKind::Smem)] = 3;
  state.pending_smem[x] = true;
  Ops::apply_xcnt_wait(state, 1);
  EXPECT_EQ(state.pending[x].size(), 1u);
  Ops::apply_xcnt_wait_implied_by_loadcnt(state, 1);
  EXPECT_EQ(state.pending[x].size(), 1u);
  Ops::apply_xcnt_wait_implied_by_kmcnt(state, 0);
  EXPECT_FALSE(state.pending_smem[x]);
  EXPECT_EQ(state.pending_event_ages[x].values[static_cast<size_t>(WaitEventKind::Smem)],
            kNoPendingEventAge);
  EXPECT_EQ(state.pending[x].size(), 1u);
  Ops::apply_xcnt_wait(state, 1);
  EXPECT_TRUE(state.pending[x].empty());
}

TEST(WaitcheckState, ImpliedWaitClearsMatchingCounterOnlyEventKinds) {
  PendingState state;
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  auto &ages = state.pending_event_ages[vm].values;
  ages[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 2;
  ages[static_cast<size_t>(WaitEventKind::Ds)] = 0;
  Ops::apply_implied_vm_vsrc_wait(state, WaitCounterKind::Load, 0);
  EXPECT_EQ(ages[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)], kNoPendingEventAge);
  EXPECT_EQ(ages[static_cast<size_t>(WaitEventKind::Ds)], 0u);
}

TEST(WaitcheckState, CounterOnlySmemRequiresZeroXcntEvenWithoutAnAge) {
  PendingState state;
  const size_t x = Ops::counter_index(WaitCounterKind::X);
  auto event = load(0, 3);
  event.counter = WaitCounterKind::X;
  state.pending[x].push_back(event);
  state.pending_smem[x] = true;
  EXPECT_TRUE(
      Ops::counter_out_of_order(state, WaitCounterKind::X, ROCJITSU_CODE_ARCH_CDNA5).value());
  EXPECT_EQ(Ops::dependency_required_count(state, event, ROCJITSU_CODE_ARCH_CDNA5).value(), 0u);
  Ops::apply_xcnt_wait(state, 1);
  EXPECT_EQ(state.pending[x].size(), 1u);
  Ops::apply_xcnt_wait(state, 0);
  EXPECT_TRUE(state.pending[x].empty());
  EXPECT_FALSE(state.pending_smem[x]);
}

TEST(WaitcheckState, ExplicitAndEmbeddedWaitsRetireVectorHazards) {
  PendingState state;
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  PendingEvent event = load(0, 0);
  event.counter = WaitCounterKind::VmVsrc;
  state.pending[vm].push_back(event);
  state.va_vdst_hazards.hazards[0] = {.age = 0, .trans_since = true, .producer = {}};
  std::unique_ptr<Instruction> alu = decode_wait(0xbf880fffu, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(alu, nullptr);
  ASSERT_TRUE(Ops::apply_waitcnt(state, *alu, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_TRUE(state.va_vdst_hazards.hazards.empty());
  EXPECT_EQ(state.pending[vm].size(), 1u);
  state.va_vdst_hazards.hazards[0] = {.age = 0, .trans_since = true, .producer = {}};
  std::unique_ptr<Instruction> dsdir = decode_wait(0xce100000u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(dsdir, nullptr);
  ASSERT_TRUE(Ops::apply_embedded_waitcnt(state, *dsdir, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_TRUE(state.va_vdst_hazards.hazards.empty());
  EXPECT_TRUE(state.pending[vm].empty());
}

TEST(WaitcheckState, Rdna3EmbeddedWaitRetiresOnlyProvenValuHazards) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    PendingState state;
    auto &hazards = state.va_vdst_hazards.hazards;
    hazards[0] = {.age = 2, .trans_since = false, .producer = {}};
    hazards[1] = {.age = 3, .trans_since = false, .producer = {}};
    hazards[2] = {.age = 3, .trans_since = true, .producer = {}};
    auto nop_wait = decode_wait(0xce1f0000u, arch);
    ASSERT_NE(nop_wait, nullptr);
    ASSERT_TRUE(Ops::apply_embedded_waitcnt(state, *nop_wait, arch).succeeded());
    EXPECT_EQ(hazards.size(), 3u);
    auto partial_wait = decode_wait(0xce130000u, arch);
    ASSERT_NE(partial_wait, nullptr);
    ASSERT_TRUE(Ops::apply_embedded_waitcnt(state, *partial_wait, arch).succeeded());
    EXPECT_TRUE(hazards.contains(0));
    EXPECT_FALSE(hazards.contains(1));
    EXPECT_TRUE(hazards.contains(2));
    auto zero_wait = decode_wait(0xce100000u, arch);
    ASSERT_NE(zero_wait, nullptr);
    ASSERT_TRUE(Ops::apply_embedded_waitcnt(state, *zero_wait, arch).succeeded());
    EXPECT_TRUE(hazards.empty());
  }
}

} // namespace
} // namespace rocjitsu::waitcheck_detail
