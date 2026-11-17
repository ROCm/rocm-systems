// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/waitcheck/state.h"
#include "rocjitsu/isa/decoder.h"

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <tuple>

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
  event.min_younger = younger;
  event.section_offset = offset;
  return event;
}

TEST(WaitcheckState, Gfx9D16WritebackRequiresKnownSramEccSetting) {
  EXPECT_TRUE(Ops::tracks_d16_subregisters(ROCJITSU_CODE_ARCH_CDNA1, EF_AMDGPU_MACH_AMDGCN_GFX900));
  EXPECT_TRUE(Ops::tracks_d16_subregisters(ROCJITSU_CODE_ARCH_CDNA1, EF_AMDGPU_MACH_AMDGCN_GFX90C));
  EXPECT_FALSE(Ops::tracks_d16_subregisters(ROCJITSU_CODE_ARCH_CDNA1));
  constexpr auto off = EF_AMDGPU_FEATURE_SRAMECC_OFF_V4;
  for (auto bits : {0u, 0x400u, off, 0xc00u}) {
    const auto flags = EF_AMDGPU_MACH_AMDGCN_GFX942 | bits;
    EXPECT_EQ(
        Ops::tracks_d16_subregisters(ROCJITSU_CODE_ARCH_CDNA3, flags, ELFABIVERSION_AMDGPU_HSA_V5),
        bits == off);
    EXPECT_FALSE(
        Ops::tracks_d16_subregisters(ROCJITSU_CODE_ARCH_CDNA3, flags, ELFABIVERSION_AMDGPU_HSA_V3));
  }
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA1);
  const std::array words{0xd8b60002u, 0x08000000u}; // ds_read_u16_d16_hi v8, v0 offset:2
  auto inst = decoder->decode_window(words, 0);
  ASSERT_TRUE(inst.succeeded());
  const auto result = Ops::partial_d16_load_def(*inst.value(), true);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->mask, kVgprHigh16Mask);
  EXPECT_FALSE(Ops::partial_d16_load_def(*inst.value(), false));
}

TEST(WaitcheckState, Rdna1And2D16LoadsWriteOnlyTheSelectedHalf) {
  constexpr RegisterRef dest{RegClass::VGPR, 8, 1};
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    auto decoder = Decoder::create(arch);
    for (auto [opcode, expected] : std::array<std::pair<uint32_t, uint8_t>, 3>{{
             {0xdc908000u, kVgprLow16Mask},  // global_load_short_d16 v8, v[0:1], off
             {0xdc948000u, kVgprHigh16Mask}, // global_load_short_d16_hi v8, v[0:1], off
             {0xdc308000u, kVgprFull32Mask}, // global_load_dword v8, v[0:1], off
         }}) {
      const std::array words{opcode, 0x087d0000u};
      auto inst = decoder->decode_window(words, 0);
      ASSERT_TRUE(inst.succeeded());
      const auto partial =
          Ops::partial_d16_load_def(*inst.value(), Ops::tracks_d16_subregisters(arch));
      if (expected == kVgprFull32Mask) {
        EXPECT_FALSE(partial);
      } else {
        ASSERT_TRUE(partial);
        EXPECT_EQ(partial->reg, dest);
        EXPECT_EQ(partial->mask, expected);
      }
    }
  }
}

TEST(WaitcheckState, Rdna1ScalarSourceLifetimeRequiresAZeroDependencyWait) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_RDNA1;
  constexpr auto counter = WaitCounterKind::VmVsrc;
  const size_t index = Ops::counter_index(counter);
  PendingState state;
  PendingEvent source;
  source.counter = counter;
  source.kind = WaitEventKind::Gfx10ScalarSource;
  source.min_younger = 5;
  state.pending[index].push_back(source);
  EXPECT_EQ(Ops::dependency_required_count(state, source, arch).value(), 0u);
  ASSERT_TRUE(Ops::apply_counter_wait(state, counter, 1, arch).succeeded());
  ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, 0, arch).succeeded());
  EXPECT_EQ(state.pending[index].size(), 1u);
  ASSERT_TRUE(Ops::apply_counter_wait(state, counter, 0, arch).succeeded());
  EXPECT_TRUE(state.pending[index].empty());
}

TEST(WaitcheckState, Rdna1FullDsWaitReleasesOnlySameOperationSource) {
  const auto source_index = Ops::counter_index(WaitCounterKind::VmVsrc);
  const auto ds_index = Ops::counter_index(WaitCounterKind::Ds);
  auto make_source = [](uint64_t pc) {
    PendingEvent source;
    source.counter = WaitCounterKind::VmVsrc;
    source.kind = WaitEventKind::Gfx10ScalarSource;
    source.section_name = ".text";
    source.section_offset = pc;
    source.instruction = "ds_read_b32 v0, v1";
    source.regs.expand({RegClass::EXEC, 0, 1});
    source.implies_active_exec = true;
    return source;
  };
  auto make_primary = [&](uint64_t pc, uint32_t younger) {
    auto event = make_source(pc);
    event.counter = WaitCounterKind::Ds;
    event.kind = WaitEventKind::Ds;
    event.min_younger = younger;
    return event;
  };
  const auto oldest = make_source(8);
  const auto newest = make_source(16);
  PendingState state;
  state.pending[source_index] = {oldest, newest};
  state.pending[ds_index] = {make_primary(8, 1), make_primary(16, 0)};
  ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Ds, 0,
                                     ROCJITSU_CODE_ARCH_RDNA1).succeeded());
  EXPECT_TRUE(state.pending[source_index].empty());

  PendingState partial;
  partial.pending[source_index] = {oldest};
  partial.pending[ds_index] = {make_primary(8, 1)};
  ASSERT_TRUE(Ops::apply_memory_wait(partial, WaitCounterKind::Ds, 1,
                                     ROCJITSU_CODE_ARCH_RDNA1).succeeded());
  EXPECT_EQ(partial.pending[source_index], std::vector{oldest});

  for (const auto [kind, same_operation, all_paths, active, arch] : {
           std::tuple{WaitEventKind::FlatLoad, true, true, true, ROCJITSU_CODE_ARCH_RDNA1},
           std::tuple{WaitEventKind::Ds, false, true, true, ROCJITSU_CODE_ARCH_RDNA1},
           std::tuple{WaitEventKind::Ds, true, false, true, ROCJITSU_CODE_ARCH_RDNA1},
           std::tuple{WaitEventKind::Ds, true, true, false, ROCJITSU_CODE_ARCH_RDNA1},
           std::tuple{WaitEventKind::Ds, true, true, true, ROCJITSU_CODE_ARCH_RDNA2}}) {
    auto control = PendingState{};
    control.pending[source_index] = {oldest};
    control.pending[source_index][0].implies_active_exec = active;
    auto primary = make_primary(same_operation ? 8 : 12, 1);
    primary.kind = kind;
    primary.present_on_all_paths = all_paths;
    control.pending[ds_index] = {primary};
    const auto expected_source = control.pending[source_index];
    ASSERT_TRUE(Ops::apply_memory_wait(control, WaitCounterKind::Ds, 0, arch).succeeded());
    EXPECT_EQ(control.pending[source_index], expected_source);
  }
}

TEST(WaitcheckState, PartialWaitRetiresOnlySufficientlyOldEvents) {
  PendingState state;
  state.pending[kLoad] = {load(0, 1), load(1, 0, 4)};
  Ops::apply_wait(state, WaitCounterKind::Load, 1);
  ASSERT_EQ(state.pending[kLoad].size(), 1u);
  EXPECT_TRUE(state.pending[kLoad][0].regs.contains({RegClass::VGPR, 1, 1}));
  Ops::apply_wait(state, WaitCounterKind::Load, 0);
  EXPECT_TRUE(state.pending[kLoad].empty());
}

TEST(WaitcheckState, Gfx1150ExportFifoProgressAtJoinRequiresBothPaths) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_RDNA3_5;
  const auto exp = Ops::counter_index(WaitCounterKind::Exp);
  PendingEvent position;
  position.counter = WaitCounterKind::Exp;
  position.kind = WaitEventKind::ExportPosition;
  position.regs.expand({RegClass::VGPR, 4, 1});
  position.min_younger = 2;
  PendingState left, right;
  left.pending[exp] = {position};
  position.min_younger = 1;
  right.pending[exp] = {position};
  Ops::merge_into(left, right);
  const auto required = Ops::dependency_required_count(left, left.pending[exp].front(), arch);
  ASSERT_TRUE(required.succeeded());
  EXPECT_EQ(required.value(), 1u);
  ASSERT_TRUE(Ops::apply_counter_wait(left, WaitCounterKind::Exp, 2, arch).succeeded());
  ASSERT_EQ(left.pending[exp].size(), 1u);
  ASSERT_TRUE(Ops::apply_counter_wait(left, WaitCounterKind::Exp, 1, arch).succeeded());
  EXPECT_TRUE(left.pending[exp].empty());
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
      // GFX11 and GFX12 FLAT always issue a Store position even on a pure LDS request.
      {ROCJITSU_CODE_ARCH_RDNA3, WaitCounterKind::Store, WaitEventKind::FlatStore,
       WaitEventKind::VmemStore, false},
      {ROCJITSU_CODE_ARCH_RDNA4, WaitCounterKind::Store, WaitEventKind::FlatStore,
       WaitEventKind::VmemStore, false},
      {ROCJITSU_CODE_ARCH_CDNA5, WaitCounterKind::Store, WaitEventKind::FlatStore,
       WaitEventKind::VmemStore, false},
      {ROCJITSU_CODE_ARCH_RDNA3, WaitCounterKind::Store, WaitEventKind::VmemStore,
       WaitEventKind::FlatStore, false},
      {ROCJITSU_CODE_ARCH_RDNA4, WaitCounterKind::Store, WaitEventKind::VmemStore,
       WaitEventKind::FlatStore, false},
      {ROCJITSU_CODE_ARCH_CDNA5, WaitCounterKind::Store, WaitEventKind::VmemStore,
       WaitEventKind::FlatStore, false},
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
    // Build the per-FIFO age using the same issue transfer as both frontends.
    store.min_younger = 0;
    PendingEvent younger = store;
    younger.kind = test.younger;
    younger.section_offset = 4;
    younger.min_younger = 0;
    state.pending[idx] = {store};
    Ops::age_events(state, younger, test.arch, 62);
    store = state.pending[idx].front();
    state.pending[idx].push_back(younger);
    auto &ages = state.pending_event_ages[idx].values;
    ages[static_cast<size_t>(test.older)] = 1;
    ages[static_cast<size_t>(test.younger)] = 0;
    auto out_of_order = Ops::counter_out_of_order(state, test.counter, test.arch);
    ASSERT_TRUE(out_of_order.succeeded());
    EXPECT_EQ(out_of_order.value(), test.out_of_order);
    auto required = Ops::dependency_required_count(state, store, test.arch);
    ASSERT_TRUE(required.succeeded());
    EXPECT_EQ(required.value(), store.min_younger);
    ASSERT_TRUE(Ops::apply_counter_wait(state, test.counter, 1, test.arch).succeeded());
    EXPECT_EQ(state.pending[idx], store.min_younger == 0
                                      ? (std::vector<PendingEvent>{store, younger})
                                      : (std::vector<PendingEvent>{younger}));
    ASSERT_TRUE(Ops::apply_counter_wait(state, test.counter, 0, test.arch).succeeded());
    EXPECT_TRUE(state.pending[idx].empty());
    EXPECT_EQ(state.pending_event_ages[idx], PendingEventAges{});
  }
}

TEST(WaitcheckState, LoadWaitReleasesLoadTranslationAndPreservesUnrelatedStore) {
  for (WaitEventKind kind : {WaitEventKind::VmemStore, WaitEventKind::FlatStore}) {
    SCOPED_TRACE(static_cast<int>(kind));
    PendingState state;
    const size_t x = Ops::counter_index(WaitCounterKind::X);
    auto source = load(0, 1);
    source.counter = WaitCounterKind::X;
    state.pending[x] = {source};
    // A younger store may be represented by the per-kind summary alone.
    auto &store_age = state.pending_event_ages[x].values[static_cast<size_t>(kind)];
    store_age = 0;
    Ops::apply_xcnt_wait_implied_by_memory(state, WaitCounterKind::Load, 0);
    EXPECT_TRUE(state.pending[x].empty());
    EXPECT_EQ(store_age, 0u);
    Ops::apply_xcnt_wait(state, 0);
    EXPECT_EQ(state.pending_event_ages[x], PendingEventAges{});
  }
}

TEST(WaitcheckState, SharedDestinationRemainsPendingUntilAllCountersRetire) {
  PendingState state;
  auto event = load(0, 0);
  event.kind = WaitEventKind::FlatLoad;
  state.pending[kLoad].push_back(event);
  event.counter = WaitCounterKind::Ds;
  const size_t ds = Ops::counter_index(WaitCounterKind::Ds);
  state.pending[ds].push_back(event);
  Ops::apply_wait(state, WaitCounterKind::Load, 0);
  EXPECT_TRUE(state.pending[kLoad].empty());
  EXPECT_EQ(state.pending[ds], (std::vector<PendingEvent>{event}));
  Ops::apply_wait(state, WaitCounterKind::Ds, 0);
  EXPECT_TRUE(state.pending[ds].empty());
}

TEST(WaitcheckState, PartialWaitKeepsNewerProducerOfTheSameRegisterUnchanged) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(arch);
    PendingState state;
    const auto newer = load(0, 0, 4);
    state.pending[kLoad] = {load(0, 1), newer};
    ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, 1, arch).succeeded());
    EXPECT_EQ(state.pending[kLoad], (std::vector<PendingEvent>{newer}));
  }
}

TEST(WaitcheckState, JoinedWaitDoesNotRetireAnotherCounter) {
  const size_t ds = Ops::counter_index(WaitCounterKind::Ds);
  PendingState left, right;
  left.pending[kLoad] = {load(0, 0)};
  auto ds_event = load(0, 0, 4);
  ds_event.counter = WaitCounterKind::Ds;
  ds_event.kind = WaitEventKind::Ds;
  right.pending[ds] = {ds_event};
  Ops::merge_into(left, right);
  const auto before = left.pending[ds];
  ASSERT_TRUE(
      Ops::apply_memory_wait(left, WaitCounterKind::Load, 0, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_TRUE(left.pending[kLoad].empty());
  EXPECT_EQ(left.pending[ds], before);
}

TEST(WaitcheckState, JoinedPartialWaitRetainsEveryNewerPhysicalDestination) {
  PendingState left, right;
  left.pending[kLoad] = {load(0, 1), load(0, 0, 4)};
  right.pending[kLoad] = {load(0, 0, 8)};
  Ops::merge_into(left, right);
  const std::vector<PendingEvent> newer{left.pending[kLoad][1], left.pending[kLoad][2]};
  ASSERT_TRUE(
      Ops::apply_memory_wait(left, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_EQ(left.pending[kLoad], newer);
}

TEST(WaitcheckState, JoinedCommonProducerDoesNotRetireLaterReplacements) {
  PendingState left, right;
  left.pending[kLoad] = {load(0, 1), load(0, 0, 4)};
  right.pending[kLoad] = {load(0, 1), load(0, 0, 8)};
  Ops::merge_into(left, right);
  const std::vector<PendingEvent> newer{left.pending[kLoad][1], left.pending[kLoad][2]};
  ASSERT_TRUE(
      Ops::apply_memory_wait(left, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_EQ(left.pending[kLoad], newer);
}

TEST(WaitcheckState, PendingAndCompletedPathsPreserveNewerProducerAfterWait) {
  PendingState pending;
  pending.pending[kLoad] = {load(0, 0)};
  auto completed = pending;
  Ops::apply_wait(completed, WaitCounterKind::Load, 0);
  for (bool reverse : {false, true}) {
    SCOPED_TRACE(reverse);
    auto joined = reverse ? completed : pending;
    Ops::merge_into(joined, reverse ? pending : completed);
    // A later producer remains pending when the common older one retires.
    joined.pending[kLoad][0].min_younger = 1;
    const auto newer = load(0, 0, 4);
    joined.pending[kLoad].push_back(newer);
    Ops::apply_wait(joined, WaitCounterKind::Load, 1);
    EXPECT_EQ(joined.pending[kLoad], (std::vector<PendingEvent>{newer}));
  }
}

TEST(WaitcheckState, PendingRegisterPayloadIsIndependentOfJoinOrder) {
  auto producer = load(0, 0);
  producer.regs.expand({RegClass::VGPR, 1, 1});
  std::vector<PendingState> paths(3);
  paths[0].pending[kLoad] = {producer};
  paths[1] = paths[0];
  Ops::apply_wait(paths[1], WaitCounterKind::Load, 0);
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
    ASSERT_EQ(joined.pending[kLoad].size(), 1u);
    EXPECT_EQ(joined.pending[kLoad][0].regs, producer.regs);
    EXPECT_FALSE(joined.pending[kLoad][0].present_on_all_paths);
    Ops::apply_wait(joined, WaitCounterKind::Load, 0);
    EXPECT_TRUE(joined.pending[kLoad].empty());
  } while (std::ranges::next_permutation(order).found);
}

TEST(WaitcheckState, ProducerAbsentOnOnePathStaysConditionalAcrossFurtherJoins) {
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
    EXPECT_FALSE(joined.pending[kLoad][0].present_on_all_paths);
    Ops::apply_wait(joined, WaitCounterKind::Load, 0);
    EXPECT_TRUE(joined.pending[kLoad].empty());
  } while (std::ranges::next_permutation(predecessors).found);

  // An unvisited predecessor is not an initialized path lacking the event.
  const std::array<uint8_t, 3> middle_unvisited{1, 0, 1};
  auto joined = Ops::merge_predecessors(predecessors, outputs, middle_unvisited).value();
  EXPECT_TRUE(joined.pending[kLoad][0].present_on_all_paths);

  // A producer issued after the join is present on every incoming path.
  joined = expected;
  joined.pending[kLoad].push_back(load(0, 0, 4));
  EXPECT_TRUE(joined.pending[kLoad][1].present_on_all_paths);
  Ops::apply_wait(joined, WaitCounterKind::Load, 0);
  EXPECT_TRUE(joined.pending[kLoad].empty());
}

TEST(WaitcheckState, MixedVmSourceFifosRetainTheirOwnPartialWaitBounds) {
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
  ASSERT_TRUE(Ops::apply_counter_wait(state, WaitCounterKind::VmVsrc, 1, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  ASSERT_EQ(state.pending[vm].size(), 1u);
  EXPECT_EQ(state.pending[vm].front().kind, WaitEventKind::Ds);
  EXPECT_EQ(ages[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)], kNoPendingEventAge);
  EXPECT_EQ(ages[static_cast<size_t>(WaitEventKind::Ds)], 0u);
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
  auto event = load(0, 0);
  event.counter = WaitCounterKind::Ds;
  event.kind = WaitEventKind::Smem;
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

TEST(WaitcheckState, MergeUsesTheLeastGuaranteedProgress) {
  std::vector<PendingState> outputs(2);
  outputs[0].pending[kLoad].push_back(load(0, 2));
  outputs[1].pending[kLoad].push_back(load(0, 1));
  const std::array<size_t, 2> predecessors{0, 1};
  const std::array<uint8_t, 2> initialized{1, 1};
  auto merged = Ops::merge_predecessors(predecessors, outputs, initialized).value();
  ASSERT_EQ(merged.pending[kLoad].size(), 1u);
  EXPECT_EQ(merged.pending[kLoad][0].min_younger, 1u);
  EXPECT_TRUE(merged.uncertain_order[kLoad]);
  auto repeated = merged;
  Ops::merge_into(repeated, merged);
  EXPECT_EQ(repeated, merged);
}

TEST(WaitcheckState, UnvisitedPredecessorDoesNotContributeAnEmptyState) {
  std::vector<PendingState> outputs(2);
  outputs[0].pending[kLoad].push_back(load(0, 1));
  const std::array<size_t, 2> predecessors{0, 1};
  const std::array<uint8_t, 2> initialized{1, 0};
  EXPECT_EQ(Ops::merge_predecessors(predecessors, outputs, initialized).value(), outputs[0]);
}

TEST(WaitcheckState, InvalidPredecessorFailsInsteadOfSkippingIncomingState) {
  const std::array<size_t, 2> predecessors{0, 1};
  std::vector<PendingState> outputs(2);
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
      state.pending[Ops::counter_index(WaitCounterKind::VmVsrc)].push_back(source);
      state.va_vdst_hazards.hazards[{0, false}] = {};

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
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RV32I, ROCJITSU_CODE_ARCH_RV64I, ROCJITSU_CODE_ARCH_INVALID}) {
    PendingState state;
    state.pending[kLoad] = {load(0, 3)};
    state.va_vdst_hazards.hazards[{0, false}] = {2, {}};
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

TEST(WaitcheckState, DirectJoinIncludesModesAndSgprProgress) {
  PendingState left, right;
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
  Ops::apply_xcnt_wait_implied_by_memory(state, WaitCounterKind::Load, 1);
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

PendingState load_with_younger_fifo_credit() {
  PendingState state;
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  // One guaranteed same-class issue follows the load. A pending invalidation
  // is unrelated to that credit.
  state.pending[kLoad] = {load(0, 1)};
  auto source = load(8, 0);
  source.counter = WaitCounterKind::VmVsrc;
  state.pending[vm] = {source};
  state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 1;
  state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::GlobalInv)] = 0;
  state.pending_event_ages[vm].values[static_cast<size_t>(WaitEventKind::VmemNoSamplerLoad)] = 0;
  return state;
}

TEST(WaitcheckState, LoadCompletionRetiresSourceFacetDespiteYoungerCounterOnlyRequest) {
  auto state = load_with_younger_fifo_credit();
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
  auto state = load_with_younger_fifo_credit();
  state.pending[kLoad].clear();
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  state.pending[vm].clear();
  ASSERT_TRUE(Ops::apply_memory_wait(state, WaitCounterKind::Load, 1, ROCJITSU_CODE_ARCH_RDNA4)
                  .succeeded());
  EXPECT_EQ(state.pending_event_ages[vm], PendingEventAges{});
  EXPECT_EQ(state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::GlobalInv)],
            0u);
}

TEST(WaitcheckState, FlatSourceRetirementRequiresBothCounterOnlyFacets) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4,
                    ROCJITSU_CODE_ARCH_CDNA5})
    for (bool store : {false, true})
      for (bool ds_first : {false, true}) {
        SCOPED_TRACE(arch);
        SCOPED_TRACE(store);
        SCOPED_TRACE(ds_first);
        PendingState state;
        const auto kind = store ? WaitEventKind::FlatStore : WaitEventKind::FlatLoad;
        const auto vmem = store ? WaitCounterKind::Store : WaitCounterKind::Load;
        const auto first = ds_first ? WaitCounterKind::Ds : vmem;
        const auto second = ds_first ? vmem : WaitCounterKind::Ds;
        const auto vm = Ops::counter_index(WaitCounterKind::VmVsrc);
        auto source = load(8, 1);
        source.counter = WaitCounterKind::VmVsrc;
        source.kind = kind;
        state.pending[vm] = {source};
        state.pending_event_ages[vm].values[static_cast<size_t>(kind)] = 1;
        // A younger counter-only request allows a partial wait to prove the
        // first route complete, but the second route can still use the source.
        state.pending_event_ages[Ops::counter_index(first)].values[static_cast<size_t>(kind)] = 1;
        state.pending_event_ages[Ops::counter_index(second)].values[static_cast<size_t>(kind)] = 0;
        ASSERT_TRUE(Ops::apply_memory_wait(state, first, 1, arch).succeeded());
        EXPECT_EQ(state.pending[vm], (std::vector<PendingEvent>{source}));
        EXPECT_EQ(state.pending_event_ages[vm].values[static_cast<size_t>(kind)], 1u);
        ASSERT_TRUE(Ops::apply_memory_wait(state, second, 0, arch).succeeded());
        EXPECT_TRUE(state.pending[vm].empty());
        EXPECT_EQ(state.pending_event_ages[vm], PendingEventAges{});
      }
}

TEST(WaitcheckState, LegacyFlatCannotProveOlderCompletionFromAnUnroutedCredit) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    auto nop = decode_wait(0xbf800000u, arch);
    ASSERT_TRUE(nop);
    for (auto kind : {WaitEventKind::FlatLoad, WaitEventKind::FlatStore})
      for (auto counter : {WaitCounterKind::Load, WaitCounterKind::Ds}) {
        SCOPED_TRACE(arch);
        SCOPED_TRACE(static_cast<int>(counter));
        SCOPED_TRACE(static_cast<int>(kind));
        const auto maximum = Ops::maximum_dependency_wait(arch, counter);
        ASSERT_TRUE(maximum.succeeded());
        EXPECT_FALSE(Ops::guaranteed_issue_order(counter, kind, arch, true));
        PendingState initial;
        auto older = load(8, maximum.value());
        older.counter = counter;
        older.kind = counter == WaitCounterKind::Ds ? WaitEventKind::Ds
                                                    : WaitEventKind::VmemNoSamplerLoad;
        const auto index = Ops::counter_index(counter);
        initial.pending[index] = {older};
        const std::array issued{ClassifiedEvent{counter, kind}};
        auto active = initial;
        active.exec_nonzero = true;
        Ops::apply_admission(active, *nop, issued, arch, false);
        EXPECT_EQ(active.pending[index], (std::vector<PendingEvent>{older}));
        Ops::apply_admission(active, *nop, issued, arch, true);
        EXPECT_EQ(active.pending[index], (std::vector<PendingEvent>{older}));
        auto empty = initial;
        empty.exec_nonzero = false;
        Ops::apply_admission(empty, *nop, issued, arch, false);
        Ops::apply_admission(empty, *nop, issued, arch, true);
        EXPECT_EQ(empty.pending[index], (std::vector<PendingEvent>{older}));

        // A native request on the same completion queue does provide the
        // capacity proof. Keep that useful case when FLAT routing is unknown.
        auto native = initial;
        native.exec_nonzero = true;
        const std::array native_issued{ClassifiedEvent{
            counter,
            counter == WaitCounterKind::Ds ? WaitEventKind::Ds : WaitEventKind::VmemNoSamplerLoad}};
        EXPECT_TRUE(Ops::guaranteed_issue_order(counter, native_issued[0].kind, arch, true));
        Ops::apply_admission(native, *nop, native_issued, arch, false);
        EXPECT_TRUE(native.pending[index].empty());
      }
  }
}

TEST(WaitcheckState, LegacyFlatCannotAgeOlderNativeResultForAPartialWait) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2})
    for (auto counter : {WaitCounterKind::Load, WaitCounterKind::Ds}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(static_cast<int>(counter));
      const auto maximum = Ops::maximum_dependency_wait(arch, counter);
      ASSERT_TRUE(maximum.succeeded());
      auto older = load(8, 0);
      older.counter = counter;
      older.kind =
          counter == WaitCounterKind::Ds ? WaitEventKind::Ds : WaitEventKind::VmemNoSamplerLoad;
      const size_t index = Ops::counter_index(counter);
      PendingState state;
      state.exec_nonzero = true;
      state.pending[index] = {older};
      auto incoming = older;
      incoming.kind = WaitEventKind::FlatLoad;
      Ops::age_events(state, incoming, arch, maximum.value());
      EXPECT_EQ(state.pending[index][0].min_younger, 0u);
      ASSERT_TRUE(Ops::apply_memory_wait(state, counter, 1, arch).succeeded());
      EXPECT_EQ(state.pending[index], (std::vector<PendingEvent>{older}));

      // A younger native request in the same queue has ordered completion.
      PendingState native;
      native.exec_nonzero = true;
      native.pending[index] = {older};
      incoming.kind = older.kind;
      Ops::age_events(native, incoming, arch, maximum.value());
      EXPECT_EQ(native.pending[index][0].min_younger, 1u);
      ASSERT_TRUE(Ops::apply_memory_wait(native, counter, 1, arch).succeeded());
      EXPECT_TRUE(native.pending[index].empty());
    }
}

TEST(WaitcheckState, OnlyCdna5FlatAdmissionPrecedesSourceReads) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    auto nop = decode_wait(0xbf800000u, arch);
    ASSERT_TRUE(nop);
    for (auto counter : {WaitCounterKind::Load, WaitCounterKind::Ds}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(static_cast<int>(counter));
      const auto maximum = Ops::maximum_dependency_wait(arch, counter);
      ASSERT_TRUE(maximum.succeeded());
      PendingState state;
      auto older = load(8, maximum.value());
      older.counter = counter;
      older.kind = counter == WaitCounterKind::Ds ? WaitEventKind::Ds
                                                  : WaitEventKind::VmemNoSamplerLoad;
      const auto index = Ops::counter_index(counter);
      state.pending[index] = {older};
      state.exec_nonzero = true;
      const std::array issued{ClassifiedEvent{counter, WaitEventKind::FlatLoad}};
      Ops::apply_admission(state, *nop, issued, arch, false);
      EXPECT_EQ(state.pending[index].empty(), arch == ROCJITSU_CODE_ARCH_CDNA5);
      Ops::apply_admission(state, *nop, issued, arch, true);
      EXPECT_TRUE(state.pending[index].empty());
    }
  }
}

TEST(WaitcheckState, JoinedFlatSourceNeedsCompletionOnEveryPath) {
  for (bool store : {false, true}) {
    SCOPED_TRACE(store);
    const auto kind = store ? WaitEventKind::FlatStore : WaitEventKind::FlatLoad;
    const auto vmem = store ? WaitCounterKind::Store : WaitCounterKind::Load;
    const auto vm = Ops::counter_index(WaitCounterKind::VmVsrc);
    PendingState left;
    auto source = load(8, 0);
    source.counter = WaitCounterKind::VmVsrc;
    source.kind = kind;
    left.pending[vm] = {source};
    for (auto counter : {vmem, WaitCounterKind::Ds, WaitCounterKind::VmVsrc})
      left.pending_event_ages[Ops::counter_index(counter)].values[static_cast<size_t>(kind)] = 0;
    auto right = left;
    ASSERT_TRUE(Ops::apply_memory_wait(left, vmem, 0, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
    ASSERT_TRUE(Ops::apply_memory_wait(right, WaitCounterKind::Ds, 0, ROCJITSU_CODE_ARCH_RDNA4)
                    .succeeded());
    Ops::merge_into(left, right);
    EXPECT_EQ(left.pending[vm], (std::vector<PendingEvent>{source}));
    ASSERT_TRUE(Ops::apply_memory_wait(left, vmem, 0, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
    EXPECT_EQ(left.pending[vm], (std::vector<PendingEvent>{source}));
    ASSERT_TRUE(
        Ops::apply_memory_wait(left, WaitCounterKind::Ds, 0, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
    EXPECT_TRUE(left.pending[vm].empty());
  }
}

TEST(WaitcheckState, IndividualOperationCompletionDoesNotClearTheKindsSummary) {
  auto state = load_with_younger_fifo_credit();
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
    auto state = load_with_younger_fifo_credit();
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
  auto state = load_with_younger_fifo_credit();
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
    auto state = load_with_younger_fifo_credit();
    if (out_of_order) {
      // CDNA4 FLAT completion is not ordered at a nonzero LOAD threshold.
      state.pending[kLoad][0].kind = WaitEventKind::FlatLoad;
      state.pending[kLoad][0].min_younger = 0;
      state.pending[vm][0].kind = WaitEventKind::FlatLoad;
      state.pending_event_ages[kLoad].values[static_cast<size_t>(WaitEventKind::FlatLoad)] = 0;
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

TEST(WaitcheckState, DependencyRequiredCountClampsTrackedAgeToWaitEncoding) {
  PendingState state;
  PendingEvent event = load(0, 7);
  event.counter = WaitCounterKind::VmVsrc;

  const auto required = Ops::dependency_required_count(state, event, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(required.succeeded());
  EXPECT_EQ(required.value(), 6u);
}

TEST(WaitcheckState, ExplicitAndEmbeddedWaitsRetireVectorHazards) {
  PendingState state;
  const size_t vm = Ops::counter_index(WaitCounterKind::VmVsrc);
  PendingEvent event = load(0, 0);
  event.counter = WaitCounterKind::VmVsrc;
  state.pending[vm].push_back(event);
  state.va_vdst_hazards.hazards[{0, true}] = {};
  std::unique_ptr<Instruction> alu = decode_wait(0xbf880fffu, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(alu, nullptr);
  ASSERT_TRUE(Ops::apply_waitcnt(state, *alu, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_TRUE(state.va_vdst_hazards.hazards.empty());
  EXPECT_EQ(state.pending[vm].size(), 1u);
  state.va_vdst_hazards.hazards[{0, true}] = {};
  std::unique_ptr<Instruction> dsdir = decode_wait(0xce100000u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(dsdir, nullptr);
  ASSERT_TRUE(Ops::apply_embedded_waitcnt(state, *dsdir, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_TRUE(state.va_vdst_hazards.hazards.empty());
  EXPECT_TRUE(state.pending[vm].empty());
}

TEST(WaitcheckState, VaVdstJoinIsIndependentOfPredecessorOrder) {
  // The same youngest age can recur with different producers on a backedge.
  std::array<VaVdstHazardState, 3> paths;
  for (size_t i = 0; i < paths.size(); ++i) {
    paths[i].hazards[{0, false}] = {
        static_cast<uint8_t>(i == 0 ? 2 : 1), {".text", 4 * i, 4 * i, "v_mov_b32"}};
    paths[i].hazards[{0, true}] = {
        static_cast<uint8_t>(10 - i), {".text", 16 + 4 * i, 16 + 4 * i, "v_sqrt_f32"}};
  }
  std::array<size_t, 3> order{0, 1, 2};
  do {
    auto joined = paths[order[0]];
    Ops::merge_va_vdst_hazards(joined, paths[order[1]]);
    Ops::merge_va_vdst_hazards(joined, paths[order[2]]);
    const auto &ordinary = joined.hazards.at({0, false});
    EXPECT_EQ(ordinary.age, 1u);
    EXPECT_EQ(ordinary.producer.section_offset, 4u);
    const auto [trans, required] = joined.requirement(0);
    ASSERT_NE(trans, nullptr);
    EXPECT_EQ(required, 0u);
    EXPECT_EQ(trans->age, 8u);
    EXPECT_EQ(trans->producer.section_offset, 24u);
    const auto stable = joined;
    for (size_t i : order)
      Ops::merge_va_vdst_hazards(joined, paths[i]);
    EXPECT_EQ(joined, stable);
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST(WaitcheckState, Rdna3EmbeddedWaitRetiresOnlyProvenValuHazards) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    PendingState state;
    auto &hazards = state.va_vdst_hazards.hazards;
    hazards[{0, false}] = {2, {}};
    hazards[{1, false}] = {3, {}};
    hazards[{2, true}] = {3, {}};
    auto nop_wait = decode_wait(0xce1f0000u, arch);
    ASSERT_NE(nop_wait, nullptr);
    ASSERT_TRUE(Ops::apply_embedded_waitcnt(state, *nop_wait, arch).succeeded());
    EXPECT_EQ(hazards.size(), 3u);
    auto partial_wait = decode_wait(0xce130000u, arch);
    ASSERT_NE(partial_wait, nullptr);
    ASSERT_TRUE(Ops::apply_embedded_waitcnt(state, *partial_wait, arch).succeeded());
    EXPECT_TRUE(hazards.contains({0, false}));
    EXPECT_FALSE(hazards.contains({1, false}));
    EXPECT_TRUE(hazards.contains({2, true}));
    auto zero_wait = decode_wait(0xce100000u, arch);
    ASSERT_NE(zero_wait, nullptr);
    ASSERT_TRUE(Ops::apply_embedded_waitcnt(state, *zero_wait, arch).succeeded());
    EXPECT_TRUE(hazards.empty());
  }
}

} // namespace
} // namespace rocjitsu::waitcheck_detail
