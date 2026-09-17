// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/state.h"
#include "rocjitsu/isa/decoder.h"

#include <algorithm>
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

TEST(WaitcheckState, StoreCounterOrderingControlsPartialRetirement) {
  struct Case {
    rj_code_arch_t arch;
    WaitCounterKind counter;
    WaitEventKind older;
    WaitEventKind younger;
    bool out_of_order;
  };
  const Case cases[] = {
      // Before VScnt, ordinary loads and stores are ordered on the shared counter.
      {ROCJITSU_CODE_ARCH_CDNA3, WaitCounterKind::Load, WaitEventKind::VmemStore,
       WaitEventKind::VmemNoSamplerLoad, false},
      {ROCJITSU_CODE_ARCH_CDNA4, WaitCounterKind::Load, WaitEventKind::VmemStore,
       WaitEventKind::VmemNoSamplerLoad, false},
      // Stores and writebacks share the same hardware event.
      {ROCJITSU_CODE_ARCH_RDNA3, WaitCounterKind::Store, WaitEventKind::VmemStore,
       WaitEventKind::GlobalWb, false},
      {ROCJITSU_CODE_ARCH_RDNA4, WaitCounterKind::Store, WaitEventKind::VmemStore,
       WaitEventKind::GlobalWb, false},
      // Generic FLAT stores prevent partial DS retirement on CDNA3/4, but
      // normalize to the same event as native DS accesses on RDNA.
      {ROCJITSU_CODE_ARCH_CDNA3, WaitCounterKind::Ds, WaitEventKind::FlatStore, WaitEventKind::Ds,
       true},
      {ROCJITSU_CODE_ARCH_CDNA4, WaitCounterKind::Ds, WaitEventKind::FlatStore, WaitEventKind::Ds,
       true},
      {ROCJITSU_CODE_ARCH_RDNA3, WaitCounterKind::Ds, WaitEventKind::FlatStore, WaitEventKind::Ds,
       false},
      {ROCJITSU_CODE_ARCH_RDNA4, WaitCounterKind::Ds, WaitEventKind::FlatStore, WaitEventKind::Ds,
       false},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.arch);
    SCOPED_TRACE(static_cast<int>(test.counter));
    PendingState state;
    const size_t idx = Ops::counter_index(test.counter);
    PendingEvent store;
    store.counter = test.counter;
    store.kind = test.older;
    store.check_memory_order = true;
    store.min_younger = 1;
    PendingEvent younger = store;
    younger.kind = test.younger;
    younger.section_offset = 4;
    younger.min_younger = 0;
    state.pending[idx] = {store, younger};
    auto &ages = state.pending_event_ages[idx].values;
    ages[static_cast<size_t>(test.older)] = 1;
    ages[static_cast<size_t>(test.younger)] = 0;
    auto out_of_order = Ops::counter_out_of_order(state, test.counter, test.arch);
    ASSERT_TRUE(out_of_order.succeeded());
    EXPECT_EQ(out_of_order.value(), test.out_of_order);
    auto required = Ops::dependency_required_count(state, store, test.arch);
    ASSERT_TRUE(required.succeeded());
    EXPECT_EQ(required.value(), test.out_of_order ? 0u : 1u);
    ASSERT_TRUE(Ops::apply_counter_wait(state, test.counter, 1, test.arch).succeeded());
    EXPECT_EQ(state.pending[idx], test.out_of_order ? (std::vector<PendingEvent>{store, younger})
                                                    : (std::vector<PendingEvent>{younger}));
    EXPECT_TRUE(state.ready_regs.none());
    ASSERT_TRUE(Ops::apply_counter_wait(state, test.counter, 0, test.arch).succeeded());
    EXPECT_TRUE(state.pending[idx].empty());
    EXPECT_EQ(state.pending_event_ages[idx], PendingEventAges{});
  }
}

TEST(WaitcheckState, LoadWaitDoesNotReleaseXcntSourcesWhileAStoreIsPending) {
  for (WaitEventKind kind : {WaitEventKind::VmemStore, WaitEventKind::FlatStore}) {
    SCOPED_TRACE(static_cast<int>(kind));
    PendingState state;
    const size_t x = Ops::counter_index(WaitCounterKind::X);
    auto source = load(0, 1);
    source.counter = WaitCounterKind::X;
    source.produces_regs = false;
    state.pending[x] = {source};
    // A store may be represented by the per-kind summary alone.
    state.pending_event_ages[x].values[static_cast<size_t>(kind)] = 0;
    const auto before = state;
    Ops::apply_xcnt_wait_implied_by_loadcnt(state, 0);
    EXPECT_EQ(state, before);
    Ops::apply_xcnt_wait(state, 0);
    EXPECT_TRUE(state.pending[x].empty());
    EXPECT_EQ(state.pending_event_ages[x], PendingEventAges{});
  }
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

TEST(WaitcheckState, OlderCommittedGenerationRemainsAvailableWithANewerProducer) {
  PendingState state;
  const RegisterRef reg{RegClass::VGPR, 0, 1};
  state.pending[kLoad] = {load(0, 1), load(0, 0, 4)};
  ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  ASSERT_EQ(state.pending[kLoad].size(), 1u);
  EXPECT_TRUE(state.ready_regs.contains(reg));
  EXPECT_TRUE(state.pending[kLoad][0].old_value_regs.contains(reg));
}

TEST(WaitcheckState, JoinedDifferentCountersCannotInventACommittedGeneration) {
  const RegisterRef reg{RegClass::VGPR, 0, 1};
  const size_t ds = Ops::counter_index(WaitCounterKind::Ds);
  PendingState left, right;
  left.pending[kLoad] = {load(0, 0)};
  auto ds_event = load(0, 0, 4);
  ds_event.counter = WaitCounterKind::Ds;
  ds_event.kind = WaitEventKind::Ds;
  right.pending[ds] = {ds_event};

  auto joined = left;
  Ops::merge_into(joined, right);
  ASSERT_TRUE(Ops::apply_memory_wait(joined, WaitCounterKind::Load, 0, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  EXPECT_TRUE(joined.pending[kLoad].empty());
  ASSERT_EQ(joined.pending[ds].size(), 1u);
  EXPECT_FALSE(joined.ready_regs.contains(reg));
  EXPECT_FALSE(joined.pending[ds][0].old_value_regs.contains(reg));

  // Waiting before joining must not establish readiness on the DS-only path.
  ASSERT_TRUE(
      Ops::apply_memory_wait(left, WaitCounterKind::Load, 0, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  ASSERT_TRUE(Ops::apply_memory_wait(right, WaitCounterKind::Load, 0, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  Ops::merge_into(left, right);
  EXPECT_EQ(joined.ready_regs, left.ready_regs);
  EXPECT_EQ(joined.pending[ds][0].old_value_regs, left.pending[ds][0].old_value_regs);
}

TEST(WaitcheckState, JoinedPartialLoadWaitCannotInventACommittedGeneration) {
  const RegisterRef reg{RegClass::VGPR, 0, 1};
  PendingState left, right;
  left.pending[kLoad] = {load(0, 1), load(0, 0, 4)};
  right.pending[kLoad] = {load(0, 0, 8)};

  auto joined = left;
  Ops::merge_into(joined, right);
  ASSERT_TRUE(Ops::apply_memory_wait(joined, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  ASSERT_EQ(joined.pending[kLoad].size(), 2u);
  EXPECT_FALSE(joined.ready_regs.contains(reg));
  for (const auto &event : joined.pending[kLoad])
    EXPECT_FALSE(event.old_value_regs.contains(reg));

  ASSERT_TRUE(
      Ops::apply_memory_wait(left, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  ASSERT_TRUE(Ops::apply_memory_wait(right, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  Ops::merge_into(left, right);
  EXPECT_EQ(joined.ready_regs, left.ready_regs);
  // The right-hand producer has no committed predecessor value on its path.
  EXPECT_EQ(joined.pending[kLoad][1].old_value_regs, left.pending[kLoad][1].old_value_regs);
}

TEST(WaitcheckState, JoinedCommonProducerEstablishesACommittedGeneration) {
  PendingState left, right;
  const RegisterRef reg{RegClass::VGPR, 0, 1};
  left.pending[kLoad] = {load(0, 1), load(0, 0, 4)};
  right.pending[kLoad] = {load(0, 1), load(0, 0, 8)};
  Ops::merge_into(left, right);
  ASSERT_TRUE(
      Ops::apply_memory_wait(left, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  ASSERT_EQ(left.pending[kLoad].size(), 2u);
  EXPECT_TRUE(left.ready_regs.contains(reg));
  for (const auto &event : left.pending[kLoad])
    EXPECT_TRUE(event.old_value_regs.contains(reg));
}

TEST(WaitcheckState, PendingAndCompletedProducerBecomesReadyAfterJoinedWait) {
  const RegisterRef reg{RegClass::VGPR, 0, 1};
  PendingState pending;
  pending.pending[kLoad] = {load(0, 0)};
  auto completed = pending;
  Ops::apply_wait(completed, WaitCounterKind::Load, 0);
  ASSERT_TRUE(completed.ready_regs.contains(reg));
  for (bool reverse : {false, true}) {
    SCOPED_TRACE(reverse);
    auto joined = reverse ? completed : pending;
    Ops::merge_into(joined, reverse ? pending : completed);
    EXPECT_FALSE(joined.ready_regs.contains(reg));
    // A later producer is still pending when the common older one retires.
    joined.pending[kLoad][0].min_younger = 1;
    joined.pending[kLoad].push_back(load(0, 0, 4));
    Ops::apply_wait(joined, WaitCounterKind::Load, 1);
    ASSERT_EQ(joined.pending[kLoad].size(), 1u);
    EXPECT_TRUE(joined.ready_regs.contains(reg));
    EXPECT_TRUE(joined.pending[kLoad][0].old_value_regs.contains(reg));
  }
}

TEST(WaitcheckState, CompletedPathCoverageIsPerRegisterAndIndependentOfJoinOrder) {
  auto producer = load(0, 0);
  producer.regs.expand({RegClass::VGPR, 1, 1});
  std::vector<PendingState> paths(3);
  paths[0].pending[kLoad] = {producer};
  paths[1] = paths[0];
  Ops::apply_wait(paths[1], WaitCounterKind::Load, 0);
  // This path has a committed v0 but has never produced v1.
  paths[2].ready_regs.expand({RegClass::VGPR, 0, 1});
  std::array<size_t, 3> order{0, 1, 2};
  const std::array<uint8_t, 3> initialized{1, 1, 1};
  const auto expected = Ops::merge_predecessors(order, paths, initialized).value();
  do {
    const auto result = Ops::merge_predecessors(order, paths, initialized);
    ASSERT_TRUE(result.succeeded());
    auto joined = result.value();
    EXPECT_EQ(joined, expected);
    auto rhs = paths[order[1]];
    Ops::merge_into(rhs, paths[order[2]]);
    auto associated = paths[order[0]];
    Ops::merge_into(associated, rhs);
    EXPECT_EQ(associated, joined);
    Ops::merge_into(joined, joined);
    EXPECT_EQ(joined, expected);
    Ops::apply_wait(joined, WaitCounterKind::Load, 0);
    EXPECT_TRUE(joined.ready_regs.contains({RegClass::VGPR, 0, 1}));
    EXPECT_FALSE(joined.ready_regs.contains({RegClass::VGPR, 1, 1}));
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST(WaitcheckState, ProducerAbsentOnOnePathStaysUnprovenAcrossFurtherJoins) {
  std::vector<PendingState> outputs(3);
  outputs[0].pending[kLoad] = {load(0, 0)};
  outputs[2].pending[kLoad] = {load(0, 0)};
  std::array<size_t, 3> predecessors{0, 1, 2};
  const std::array<uint8_t, 3> initialized{1, 1, 1};
  const auto expected = Ops::merge_predecessors(predecessors, outputs, initialized).value();
  do {
    auto joined = Ops::merge_predecessors(predecessors, outputs, initialized).value();
    EXPECT_EQ(joined, expected);
    auto repeated = joined;
    Ops::merge_into(repeated, joined);
    EXPECT_EQ(repeated, joined);
    Ops::apply_wait(joined, WaitCounterKind::Load, 0);
    EXPECT_TRUE(joined.pending[kLoad].empty());
    EXPECT_FALSE(joined.ready_regs.contains({RegClass::VGPR, 0, 1}));
  } while (std::next_permutation(predecessors.begin(), predecessors.end()));

  // An unvisited predecessor is not an initialized path lacking the event.
  const std::array<uint8_t, 3> middle_unvisited{1, 0, 1};
  auto joined = Ops::merge_predecessors(predecessors, outputs, middle_unvisited).value();
  Ops::apply_wait(joined, WaitCounterKind::Load, 0);
  EXPECT_TRUE(joined.ready_regs.contains({RegClass::VGPR, 0, 1}));

  // A producer issued after the join is present on every incoming path.
  joined = expected;
  joined.pending[kLoad].push_back(load(0, 0, 4));
  Ops::apply_wait(joined, WaitCounterKind::Load, 0);
  EXPECT_TRUE(joined.ready_regs.contains({RegClass::VGPR, 0, 1}));
}

TEST(WaitcheckState, MixedHardwareEventKindsPreventPartialCounterRetirement) {
  PendingState state;
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  auto vmem_event = load(0, 1);
  vmem_event.counter = WaitCounterKind::VmVsrc;
  auto ds_event = load(1, 0, 4);
  ds_event.counter = WaitCounterKind::VmVsrc;
  ds_event.kind = WaitEventKind::Ds;
  state.pending[vm] = {vmem_event, ds_event};
  auto &ages = state.pending_event_ages[vm].values;
  ages[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 1;
  ages[static_cast<size_t>(WaitEventKind::Ds)] = 0;
  const auto before = state;
  ASSERT_TRUE(Ops::apply_counter_wait(state, WaitCounterKind::VmVsrc, 1, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  EXPECT_EQ(state, before);
  ASSERT_TRUE(Ops::apply_counter_wait(state, WaitCounterKind::VmVsrc, 0, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  EXPECT_TRUE(state.pending[vm].empty());
  EXPECT_EQ(ages[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)], kNoPendingEventAge);
  EXPECT_EQ(ages[static_cast<size_t>(WaitEventKind::Ds)], kNoPendingEventAge);
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
  auto merged = Ops::merge_predecessors(predecessors, outputs, initialized).value();
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
  EXPECT_EQ(Ops::merge_predecessors(predecessors, outputs, initialized).value(), outputs[0]);
}

TEST(WaitcheckState, InvalidPredecessorFailsInsteadOfSkippingIncomingState) {
  const std::array<size_t, 2> predecessors{0, 1};
  std::vector<PendingState> outputs(2);
  outputs[0].ready_regs.expand({RegClass::VGPR, 0, 1});
  const std::array<uint8_t, 1> short_initialized{1};
  EXPECT_TRUE(Ops::merge_predecessors(predecessors, outputs, short_initialized).failed());

  outputs.resize(1);
  const std::array<uint8_t, 2> initialized{1, 1};
  EXPECT_TRUE(Ops::merge_predecessors(predecessors, outputs, initialized).failed());
  // Even an unvisited marker cannot make an out-of-range output index valid.
  const std::array<uint8_t, 2> unvisited{1, 0};
  EXPECT_TRUE(Ops::merge_predecessors(predecessors, outputs, unvisited).failed());
}

TEST(WaitcheckState, DisagreeingModesBecomeUnknownAtAJoin) {
  std::vector<PendingState> outputs(2);
  outputs[0].vgpr_msb.mode = 0;
  outputs[1].vgpr_msb.mode = 1;
  outputs[0].expert_scheduling.enabled = false;
  outputs[1].expert_scheduling.enabled = true;
  const std::array<size_t, 2> predecessors{0, 1};
  const std::array<uint8_t, 2> initialized{1, 1};
  const auto merged = Ops::merge_predecessors(predecessors, outputs, initialized).value();
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

TEST(WaitcheckState, DependencyWaitsRetireOnlySelectedScalarHazards) {
  struct Case {
    uint32_t immediate;
    bool salu;
    bool valu_sgpr;
    bool valu_vcc;
  };
  // Leave both vector fields at their no-wait sentinels. A nonzero VA_SDST
  // field must retain the hazard, even when it is below the no-wait value.
  const Case cases[] = {{0xffff, false, false, false}, {0xfffe, true, false, false},
                        {0xfffd, false, false, true},  {0xf1ff, false, true, false},
                        {0xf3ff, false, false, false}, {0xf1fc, true, true, true}};
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    for (const auto &test : cases) {
      SCOPED_TRACE(test.immediate);
      PendingState state;
      auto &hazards = state.sgpr_hazards;
      hazards.tracked_pairs.set(0);
      hazards.tracked_vcc = true;
      hazards.salu_hazards.set(0);
      hazards.valu_hazards.set(1);
      hazards.vcc_hazard = kSgprHazardSalu | kSgprHazardValu;
      hazards.salu_producers[0] = {};
      hazards.valu_producers[1] = {};
      hazards.salu_vcc_producer = SgprHazardProducer{};
      hazards.valu_vcc_producer = SgprHazardProducer{};
      hazards.consecutive_ds_nops = 2;
      auto source = load(0, 0);
      source.counter = WaitCounterKind::VmVsrc;
      source.produces_regs = false;
      state.pending[Ops::counter_index(WaitCounterKind::VmVsrc)].push_back(source);
      state.va_vdst_hazards.hazards[0] = {};

      auto expected = state;
      auto &remaining = expected.sgpr_hazards;
      if (test.salu) {
        remaining.salu_hazards.reset();
        remaining.salu_producers.clear();
        remaining.salu_vcc_producer.reset();
        remaining.vcc_hazard &= ~kSgprHazardSalu;
      }
      if (test.valu_sgpr) {
        remaining.valu_hazards.reset();
        remaining.valu_producers.clear();
      }
      if (test.valu_vcc) {
        remaining.valu_vcc_producer.reset();
        remaining.vcc_hazard &= ~kSgprHazardValu;
      }
      auto wait = decode_wait(0xbf880000u | test.immediate, arch);
      ASSERT_NE(wait, nullptr);
      ASSERT_TRUE(Ops::apply_waitcnt(state, *wait, arch).succeeded());
      EXPECT_EQ(state, expected);
    }
  }
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
  EXPECT_EQ(left, Ops::merge_predecessors(predecessors, outputs, initialized).value());
  EXPECT_EQ(left.ready_regs.size(), 1u);
  EXPECT_FALSE(left.previous_vm_vsrc_zero_wait);
  EXPECT_FALSE(left.vgpr_msb.known);
  EXPECT_FALSE(left.expert_scheduling.known);
  EXPECT_EQ(left.sgpr_hazards.consecutive_ds_nops, 2u);
  const std::array<uint8_t, 2> first_only{1, 0};
  EXPECT_EQ(Ops::merge_predecessors(predecessors, outputs, first_only).value(), outputs[0]);
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

PendingState load_before_global_inv() {
  PendingState state;
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  state.pending[kLoad] = {load(0, 1)};
  auto source = load(8, 0);
  source.counter = WaitCounterKind::VmVsrc;
  source.produces_regs = false;
  state.pending[vm] = {source};
  state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 1;
  state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::GlobalInv)] = 0;
  state.pending_event_ages[vm].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 0;
  return state;
}

TEST(WaitcheckState, LoadCompletionRetiresSourceFacetDespiteYoungerCounterOnlyRequest) {
  auto state = load_before_global_inv();
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  EXPECT_TRUE(state.pending[kLoad].empty());
  EXPECT_TRUE(state.pending[vm].empty());
  EXPECT_EQ(state.pending_event_ages[vm], PendingEventAges{});
  EXPECT_EQ(state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::GlobalInv)],
            0u);
}

TEST(WaitcheckState, PrimaryKindCompletionDoesNotRequireMaterializedEvents) {
  auto state = load_before_global_inv();
  state.pending[kLoad].clear();
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  state.pending[vm].clear();
  ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  EXPECT_EQ(state.pending_event_ages[vm], PendingEventAges{});
  EXPECT_EQ(state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::GlobalInv)],
            0u);
}

TEST(WaitcheckState, IndividualOperationCompletionDoesNotClearTheKindsSummary) {
  auto state = load_before_global_inv();
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  // A younger counter-only request of the same kind leaves its own source
  // summary pending even after the older materialized operation completes.
  state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 0;
  state.pending[vm][0].min_younger = 1;
  ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  EXPECT_TRUE(state.pending[vm].empty());
  EXPECT_EQ(
      state.pending_event_ages[vm].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)],
      0u);
}

TEST(WaitcheckState, PrimaryWaitDoesNotRetireAnUnrelatedSourceFacet) {
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  for (unsigned difference = 0; difference < 5; ++difference) {
    SCOPED_TRACE(difference);
    auto state = load_before_global_inv();
    auto &source = state.pending[vm][0];
    // A separate younger request prevents proving the whole kind complete.
    state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] =
        0;
    switch (difference) {
    case 0:
      source.section_offset = 4;
      break;
    case 1:
      source.file_offset = 4;
      break;
    case 2:
      source.section_name = ".other";
      break;
    case 3:
      source.instruction = "different load";
      break;
    case 4:
      source.kind = WaitEventKind::FlatLoad;
      break;
    }
    const auto expected = source;
    ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4)
                    .succeeded());
    EXPECT_TRUE(state.pending[kLoad].empty());
    EXPECT_EQ(state.pending[vm], (std::vector<PendingEvent>{expected}));
  }
}

TEST(WaitcheckState, PrimaryWaitNeedsCompletionOfEveryMatchingOperationVariant) {
  auto state = load_before_global_inv();
  // A younger instance of the same static operation uses a different lane.
  // Completing only its older variant cannot release all of its source facets.
  state.pending[kLoad].push_back(load(1, 0));
  state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 0;
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  const auto source = state.pending[vm];
  ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  ASSERT_EQ(state.pending[kLoad].size(), 1u);
  EXPECT_EQ(state.pending[vm], source);
}

TEST(WaitcheckState, PrimaryWaitRequiresCoverageAndOrderedCompletion) {
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  for (bool out_of_order : {false, true}) {
    SCOPED_TRACE(out_of_order);
    auto state = load_before_global_inv();
    if (out_of_order) {
      // CDNA4 FLAT completion is not ordered at a nonzero LOAD threshold.
      state.pending[kLoad][0].kind = WaitEventKind::FlatLoad;
      state.pending[vm][0].kind = WaitEventKind::FlatLoad;
      state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::FlatLoad)] = 1;
    } else {
      // Matching location alone does not prove completion on an absent path.
      state.pending[kLoad][0].present_on_all_paths = false;
    }
    state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] =
        0;
    const auto sources = state.pending[vm];
    ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_CDNA4)
                    .succeeded());
    EXPECT_EQ(state.pending[vm], sources);
    EXPECT_EQ(state.pending[kLoad].empty(), !out_of_order);
  }
}

TEST(WaitcheckState, PartialImpliedWaitUsesAgeInsteadOfCanonicalEventOrder) {
  PendingState state;
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  auto oldest = load(0, 3, 12);
  auto middle = load(1, 1, 4);
  auto youngest = load(2, 0, 0);
  auto ds = load(3, 2, 8);
  ds.kind = WaitEventKind::Ds;
  for (PendingEvent *event : {&oldest, &middle, &youngest, &ds}) {
    event->counter = WaitCounterKind::VmVsrc;
    event->produces_regs = false;
  }
  state.pending[vm] = {youngest, middle, ds, oldest};
  ASSERT_TRUE(std::ranges::is_sorted(state.pending[vm], Ops::event_identity_less));
  auto &ages = state.pending_event_ages[vm].values;
  ages[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 0;
  ages[static_cast<size_t>(WaitEventKind::Ds)] = 2;

  Ops::apply_implied_vm_vsrc_wait(state, WaitCounterKind::Load, 1);
  EXPECT_EQ(state.pending[vm], (std::vector<PendingEvent>{youngest, ds}));
  EXPECT_EQ(ages[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)], 0u);
  EXPECT_EQ(ages[static_cast<size_t>(WaitEventKind::Ds)], 2u);
  EXPECT_FALSE(state.uncertain_order[vm]);

  Ops::apply_implied_vm_vsrc_wait(state, WaitCounterKind::Load, 0);
  EXPECT_EQ(state.pending[vm], (std::vector<PendingEvent>{ds}));
  EXPECT_EQ(ages[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)], kNoPendingEventAge);
  EXPECT_EQ(ages[static_cast<size_t>(WaitEventKind::Ds)], 2u);
}

TEST(WaitcheckState, PartialImpliedWaitRetainsTiedCutoffEvents) {
  PendingState state;
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  auto first = load(0, 1, 0);
  auto second = load(1, 1, 4);
  first.counter = second.counter = WaitCounterKind::VmVsrc;
  first.produces_regs = second.produces_regs = false;
  state.pending[vm] = {first, second};
  state.pending_event_ages[vm].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 1;
  auto expected = state;
  expected.uncertain_order[vm] = true;

  Ops::apply_implied_vm_vsrc_wait(state, WaitCounterKind::Load, 1);
  EXPECT_EQ(state, expected);
  Ops::apply_implied_vm_vsrc_wait(state, WaitCounterKind::Load, 0);
  EXPECT_TRUE(state.pending[vm].empty());
  EXPECT_EQ(state.pending_event_ages[vm], PendingEventAges{});
  EXPECT_FALSE(state.uncertain_order[vm]);
}

TEST(WaitcheckState, PartialImpliedWaitRetainsEventsWithUncertainOrder) {
  PendingState state;
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  auto oldest = load(0, 1);
  auto youngest = load(1, 0, 4);
  oldest.counter = youngest.counter = WaitCounterKind::VmVsrc;
  oldest.produces_regs = youngest.produces_regs = false;
  state.pending[vm] = {oldest, youngest};
  state.pending_event_ages[vm].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 0;
  state.uncertain_order[vm] = true;
  const auto before = state;

  Ops::apply_implied_vm_vsrc_wait(state, WaitCounterKind::Load, 1);
  EXPECT_EQ(state, before);
  Ops::apply_implied_vm_vsrc_wait(state, WaitCounterKind::Load, 0);
  EXPECT_TRUE(state.pending[vm].empty());
  EXPECT_EQ(state.pending_event_ages[vm], PendingEventAges{});
  EXPECT_FALSE(state.uncertain_order[vm]);
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
