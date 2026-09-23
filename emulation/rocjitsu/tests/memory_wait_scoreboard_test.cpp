// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"

#include <fstream>
#include <iterator>
#include <thread>
#include <tuple>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::test::cdna5;

// Diagnostic tests opt in; an empty setting exercises the production default.
std::string memory_wait_test_config(std::string_view setting = "warn") {
  std::ifstream file(kGfx1250ConfigPath);
  std::string config((std::istreambuf_iterator<char>(file)), {});
  const auto cu = config.find("\"type\": \"compute_unit\"");
  const auto array = config.find('[', config.find("\"config\"", cu));
  if (!setting.empty())
    config.insert(
        array + 1,
        std::format("{{\"key\":\"memory_wait_diagnostics\",\"value\":\"{}\"}},", setting));
  return config;
}

struct MemoryWaitScoreboardTest : ::testing::Test {
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard state{shadow};
  std::vector<MemoryWaitScoreboard::Hazard> hazards;
  MemoryWaitScoreboardTest() {
    state.bind(0x200, &hazards, [](void *p, const auto &hazard) {
      static_cast<decltype(hazards) *>(p)->push_back(hazard);
    });
  }
  void load(unsigned reg, WaitCounterKind counter = WaitCounterKind::Load,
            uint64_t lanes = ~uint64_t{0}, uint8_t bytes = 0xf, bool unordered = false) {
    state.add({state.issue(counter, unordered),
               0x100,
               lanes,
               {RegClass::VGPR, static_cast<uint16_t>(reg), 1},
               counter,
               bytes});
  }
  void read(unsigned reg, uint64_t lanes = ~uint64_t{0}, uint8_t bytes = 0xf) {
    state.access({RegClass::VGPR, static_cast<uint16_t>(reg), 1}, lanes, bytes, false);
  }
};

TEST_F(MemoryWaitScoreboardTest, MissingWaitIdentifiesProducerAndConsumer) {
  load(5);
  read(6);
  EXPECT_TRUE(hazards.empty());
  read(5);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_EQ(hazards[0].producer.pc, 0x100u);
  EXPECT_EQ(hazards[0].consumer_pc, 0x200u);
  EXPECT_EQ(hazards[0].reg.index, 5u);
}

TEST_F(MemoryWaitScoreboardTest, PartialWaitReleasesOnlyTheOlderLoad) {
  load(5);
  load(6);
  state.wait(WaitCounterKind::Load, 1);
  read(5);
  EXPECT_TRUE(hazards.empty());
  read(6);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_EQ(hazards[0].reg.index, 6u);
}

TEST_F(MemoryWaitScoreboardTest, MixedCounterPartialWaitUsesOnlyOrderedYoungerOperations) {
  state.add({state.issue(WaitCounterKind::Ds, false, 1),
             0x100,
             1,
             {RegClass::VGPR, 5, 1},
             WaitCounterKind::Ds,
             0xf});
  state.issue(WaitCounterKind::Ds, true); // Unordered SMEM-like traffic.
  state.issue(WaitCounterKind::Ds, false, 1);
  state.wait(WaitCounterKind::Ds, 1);
  read(5);
  EXPECT_TRUE(hazards.empty());
}

TEST_F(MemoryWaitScoreboardTest, LdsChecksConflictsAcrossPathsAndAsyncWritesWithinOnePath) {
  using Kind = MemoryWaitScoreboard::LdsKind;
  for (auto older_kind : {Kind::Ds, Kind::Direct, Kind::Async})
    for (auto newer_kind : {Kind::Ds, Kind::Direct, Kind::Async})
      for (bool older_write : {false, true})
        for (bool newer_write : {false, true}) {
          state.clear();
          hazards.clear();
          const auto sequence = state.issue(WaitCounterKind::Async);
          state.add_lds({{sequence, 0x100, 0, {}, WaitCounterKind::Async, 0},
                         older_kind,
                         older_write,
                         {{16, 20}, {24, 28}}});
          state.access_lds({{20, 24}}, newer_kind, newer_write); // A hole is not an overlap.
          EXPECT_TRUE(hazards.empty());
          state.access_lds({{27, 29}}, newer_kind, newer_write);
          const bool expected = (older_write || newer_write) &&
                                (older_kind != newer_kind || older_kind == Kind::Async);
          ASSERT_EQ(hazards.size(), expected ? 1u : 0u);
          if (expected) {
            EXPECT_EQ(hazards[0].producer.pc, 0x100u);
            EXPECT_EQ(hazards[0].consumer_pc, 0x200u);
            EXPECT_EQ(hazards[0].lds_address, 27u);
            state.access_lds({{27, 29}}, newer_kind, newer_write);
            EXPECT_EQ(hazards.size(), 1u); // Recover without cascading reports.
          }
        }
}

TEST_F(MemoryWaitScoreboardTest, LdsPartialWaitAndAdmissionRetireOnlyProvenAccesses) {
  using Kind = MemoryWaitScoreboard::LdsKind;
  for (bool admission : {false, true}) {
    state.clear();
    hazards.clear();
    auto add = [&](uint32_t address) {
      state.add_lds({{state.issue(WaitCounterKind::Async), 0x100, 0, {}, WaitCounterKind::Async, 0},
                     Kind::Async,
                     true,
                     {{address, address + 4}}});
    };
    add(16);
    add(24);
    state.wait(WaitCounterKind::Ds, 0); // Wrong counter.
    ASSERT_EQ(state.lds_events().size(), 2u);
    if (admission) {
      for (unsigned i = 2; i < 63; ++i)
        state.issue(WaitCounterKind::Async);
      state.backpressure(WaitCounterKind::Async, 63);
    } else {
      state.wait(WaitCounterKind::Async, 1);
    }
    state.access_lds({{16, 20}}, Kind::Ds, false);
    EXPECT_TRUE(hazards.empty());
    state.access_lds({{24, 28}}, Kind::Ds, false);
    ASSERT_EQ(hazards.size(), 1u);
  }
}

TEST_F(MemoryWaitScoreboardTest, WrongCounterAndUnorderedPartialWaitDoNotProveReadiness) {
  load(5, WaitCounterKind::Km, ~uint64_t{0}, 0xf, true);
  load(6, WaitCounterKind::Km, ~uint64_t{0}, 0xf, true);
  state.wait(WaitCounterKind::Load, 0);
  state.wait(WaitCounterKind::Km, 1);
  read(5);
  EXPECT_EQ(hazards.size(), 1u);
  state.wait(WaitCounterKind::Km, 0);
  read(6);
  EXPECT_EQ(hazards.size(), 1u);
}

TEST_F(MemoryWaitScoreboardTest, CounterOnlyOperationsContributeToPartialWaits) {
  load(5);
  state.issue(WaitCounterKind::Load);
  state.wait(WaitCounterKind::Load, 1);
  read(5);
  EXPECT_TRUE(hazards.empty());
}

TEST_F(MemoryWaitScoreboardTest, AdmissionAtCapacityReleasesOnlyTheOldestOrderedResult) {
  for (uint32_t capacity : {7u, 15u, 31u, 63u}) {
    state.clear();
    load(5, WaitCounterKind::Ds);
    load(6, WaitCounterKind::Ds);
    for (uint32_t i = 2; i < capacity - 1; ++i)
      state.issue(WaitCounterKind::Ds);
    state.backpressure(WaitCounterKind::Ds, capacity);
    EXPECT_TRUE(shadow.test(5));
    state.issue(WaitCounterKind::Ds);
    // The queue can hold exactly capacity requests. Only admission of another
    // request forces the original result ready, before that request reads it.
    EXPECT_TRUE(shadow.test(5));
    state.backpressure(WaitCounterKind::Ds, capacity);
    EXPECT_FALSE(shadow.test(5));
    EXPECT_TRUE(shadow.test(6));
    read(5);
    EXPECT_TRUE(hazards.empty());
  }
}

TEST_F(MemoryWaitScoreboardTest, BackpressureKeepsCounterMembershipSeparateFromOrder) {
  load(5, WaitCounterKind::Ds);
  for (uint32_t i = 0; i < 30; ++i) {
    state.backpressure(WaitCounterKind::Ds, 15);
    state.issue(WaitCounterKind::Ds, true);
  }
  EXPECT_TRUE(shadow.test(5)); // Unordered younger operations prove no progress.
  for (uint32_t i = 1; i < 15; ++i)
    state.issue(WaitCounterKind::Ds);
  load(6, WaitCounterKind::Ds, 1, 0xf, true);
  state.backpressure(WaitCounterKind::Ds, 15);
  EXPECT_FALSE(shadow.test(5)); // A full ordered class must progress despite the mix.
  EXPECT_TRUE(shadow.test(6));
  state.wait(WaitCounterKind::Ds, 0);
  EXPECT_FALSE(shadow.test(6));
}

TEST_F(MemoryWaitScoreboardTest, BackpressureDoesNotAssumeFifoForScalarGdsOrLegacyFlat) {
  using namespace waitcheck_detail;
  for (auto kind : {WaitEventKind::Smem, WaitEventKind::Gds, WaitEventKind::FlatLoad}) {
    state.clear();
    const ClassifiedEvent event{WaitCounterKind::Ds, kind};
    const auto first = state.issue(event, ROCJITSU_CODE_ARCH_CDNA4);
    state.add({first, 0x100, 1, {RegClass::VGPR, 5, 1}, WaitCounterKind::Ds, 0xf});
    for (uint32_t i = 0; i < 32; ++i) {
      state.backpressure(WaitCounterKind::Ds, 15);
      state.issue(event, ROCJITSU_CODE_ARCH_CDNA4);
    }
    EXPECT_TRUE(shadow.test(5));
  }
}

TEST_F(MemoryWaitScoreboardTest, AsyncDirectionsAndLegacyImageTypesHaveSeparateCompletionOrder) {
  using namespace waitcheck_detail;
  for (const auto &[arch, counter, first_kind, other_kind] :
       {std::tuple{ROCJITSU_CODE_ARCH_CDNA5, WaitCounterKind::Async, WaitEventKind::AsyncLdsLoad,
                   WaitEventKind::AsyncLdsStore},
        {ROCJITSU_CODE_ARCH_RDNA1, WaitCounterKind::Load, WaitEventKind::VmemNoSamplerLoad,
         WaitEventKind::Sample}}) {
    state.clear();
    const ClassifiedEvent first{counter, first_kind}, other{counter, other_kind};
    state.add({state.issue(first, arch), 0x100, 1, {RegClass::VGPR, 5, 1}, counter, 0xf});
    for (unsigned i = 0; i < 64; ++i) {
      state.backpressure(counter, 63);
      state.issue(other, arch);
    }
    state.wait(counter, 1);
    EXPECT_TRUE(shadow.test(5)); // Neither admission nor the mixed wait proves this result.
    for (unsigned i = 1; i < 63; ++i)
      state.issue(first, arch);
    state.backpressure(counter, 63);
    EXPECT_FALSE(shadow.test(5));
  }
}

TEST_F(MemoryWaitScoreboardTest, CompletionBackpressureAlsoProvesReplayTranslation) {
  state.issue(WaitCounterKind::Load);
  state.add({state.issue_xcnt(WaitCounterKind::Load, false),
             0x100,
             1,
             {RegClass::VGPR, 5, 1},
             WaitCounterKind::X,
             0xf});
  for (unsigned i = 1; i < 63; ++i)
    state.issue(WaitCounterKind::Load);
  EXPECT_TRUE(shadow.test(5, true));
  state.backpressure(WaitCounterKind::Load, 63);
  EXPECT_FALSE(shadow.test(5, true));
  EXPECT_EQ(state.outstanding(WaitCounterKind::X), 0u);
}

TEST_F(MemoryWaitScoreboardTest, WaitThresholdLeavesExactlyTheRequestedQueueSuffix) {
  load(5);                            // A
  state.issue(WaitCounterKind::Load); // B, no register result
  load(6);                            // C
  state.issue(WaitCounterKind::Load); // D, no register result
  load(8, WaitCounterKind::Ds);
  state.wait(WaitCounterKind::Load, 2); // Retire A and B; leave C and D.
  EXPECT_EQ(state.outstanding(WaitCounterKind::Load), 2u);
  EXPECT_FALSE(shadow.test(5));
  EXPECT_TRUE(shadow.test(6));
  EXPECT_TRUE(shadow.test(8));
  load(7);                              // E
  state.wait(WaitCounterKind::Load, 2); // Retire C; leave D and E.
  EXPECT_FALSE(shadow.test(6));
  EXPECT_TRUE(shadow.test(7));
  state.wait(WaitCounterKind::Load, 3); // A weaker wait cannot undo completion.
  EXPECT_EQ(state.outstanding(WaitCounterKind::Load), 2u);
  EXPECT_FALSE(shadow.test(6));
  state.wait(WaitCounterKind::Ds, 0); // A different queue cannot retire E.
  EXPECT_TRUE(shadow.test(7));
  state.wait(WaitCounterKind::Load, 0);
  EXPECT_EQ(state.outstanding(WaitCounterKind::Load), 0u);
  read(5);
  read(6);
  read(7);
  read(8);
  EXPECT_TRUE(hazards.empty());
}

TEST_F(MemoryWaitScoreboardTest, DiagnosticReportsTheRequiredPartialWaitThreshold) {
  load(5);
  state.issue(WaitCounterKind::Load);
  state.issue(WaitCounterKind::Load);
  state.wait(WaitCounterKind::Load, 3); // Nothing retired; A needs count <= 2.
  read(5);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_EQ(hazards[0].required_wait, 2u);
  load(6, WaitCounterKind::Km, ~uint64_t{0}, 0xf, true);
  state.issue(WaitCounterKind::Km, true);
  read(6);
  ASSERT_EQ(hazards.size(), 2u);
  EXPECT_EQ(hazards[1].required_wait, 0u);
}

TEST_F(MemoryWaitScoreboardTest, OrderedMemoryWritesStillCheckOtherCounters) {
  load(5);
  state.access({RegClass::VGPR, 5, 1}, ~uint64_t{0}, 0xf, true, WaitCounterKind::Load);
  EXPECT_TRUE(hazards.empty());
  load(6, WaitCounterKind::Ds);
  state.access({RegClass::VGPR, 6, 1}, ~uint64_t{0}, 0xf, true, WaitCounterKind::Load);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_EQ(hazards[0].producer.counter, WaitCounterKind::Ds);
}

TEST_F(MemoryWaitScoreboardTest, UnorderedQueueStillChecksOverwritesOnTheSameCounter) {
  load(5, WaitCounterKind::Load, 1, 0xf, true);
  state.access({RegClass::VGPR, 5, 1}, 1, 0xf, true, WaitCounterKind::Load);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_TRUE(hazards.front().write);
  EXPECT_EQ(hazards.front().required_wait, 0u);

  state.wait(WaitCounterKind::Load, 0);
  hazards.clear();
  load(5, WaitCounterKind::Load, 1);
  state.access({RegClass::VGPR, 5, 1}, 1, 0xf, true, WaitCounterKind::Load);
  EXPECT_TRUE(hazards.empty());
}

TEST_F(MemoryWaitScoreboardTest, DisjointLanesAndRegisterHalvesAreIndependent) {
  load(300, WaitCounterKind::Load, 2, 0xc);
  read(300, 1, 0xf);
  read(300, 2, 0x3);
  EXPECT_TRUE(hazards.empty());
  state.access({RegClass::VGPR, 300, 1}, 2, 0xc, true);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_TRUE(hazards[0].write);
}

TEST_F(MemoryWaitScoreboardTest, ScopeDoesNotExposeIssuerStateToHelpers) {
  load(5);
  {
    ScopedMemoryWaitCheck scope(&state);
    EXPECT_EQ(active_memory_wait_check, &state);
    std::thread helper([&] {
      EXPECT_EQ(active_memory_wait_check, nullptr);
      EXPECT_TRUE(shadow.test(5));
      check_active_memory_wait({RegClass::VGPR, 5, 1}, 1, 0xf, false);
    });
    helper.join();
    EXPECT_TRUE(hazards.empty());
    {
      SuspendedMemoryWaitCheck observer;
      EXPECT_EQ(active_memory_wait_check, nullptr);
      check_active_memory_wait({RegClass::VGPR, 5, 1}, 1, 0xf, false);
    }
    EXPECT_EQ(active_memory_wait_check, &state);
    EXPECT_TRUE(hazards.empty());
  }
  EXPECT_EQ(active_memory_wait_check, nullptr);
  state.clear();
  ScopedMemoryWaitCheck scope(&state);
  EXPECT_EQ(active_memory_wait_check, nullptr);
}

TEST_F(MemoryWaitScoreboardTest, ShadowPreservesOverlappingDestinationsAfterRetirement) {
  state.add({state.issue(WaitCounterKind::Load),
             0x100,
             1,
             {RegClass::VGPR, 63, 2},
             WaitCounterKind::Load,
             0xf});
  load(64, WaitCounterKind::Ds, 2);
  state.wait(WaitCounterKind::Load, 0);
  EXPECT_FALSE(shadow.test(63));
  EXPECT_TRUE(shadow.test(64));
  read(64, 1);
  EXPECT_TRUE(hazards.empty());
  read(64, 2);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_EQ(hazards[0].producer.counter, WaitCounterKind::Ds);
  EXPECT_FALSE(shadow.test(64));
  load(1023);
  state.clear();
  EXPECT_FALSE(shadow.test(1023));
}

TEST_F(MemoryWaitScoreboardTest, HelpersCanProbeShadowWhileIssuerRetiresRecords) {
  std::atomic<bool> done{false};
  std::thread helper([&] {
    while (!done.load(std::memory_order_relaxed))
      if (shadow.test(5))
        check_active_memory_wait({RegClass::VGPR, 5, 1}, 1, 0xf, false);
  });
  {
    ScopedMemoryWaitCheck scope(&state);
    for (unsigned i = 0; i < 1000; ++i) {
      load(5);
      state.wait(WaitCounterKind::Load, 0);
    }
  }
  done.store(true, std::memory_order_relaxed);
  helper.join();
  EXPECT_TRUE(hazards.empty());
  EXPECT_FALSE(shadow.test(5));
}

TEST_F(MemoryWaitScoreboardTest, TargetWaitEncodingsRetireTheirCanonicalCounter) {
  struct Case {
    rj_code_arch_t arch;
    uint32_t word;
    WaitCounterKind counter;
  };
  // Synthetic pending destinations test wait decoding and canonical counter
  // wiring; they do not assert that every counter has a tracked producer.
  const Case cases[] = {
      {ROCJITSU_CODE_ARCH_CDNA1, 0xbf8c0f70u, WaitCounterKind::Load},
      {ROCJITSU_CODE_ARCH_CDNA1, 0xbf8cc07fu, WaitCounterKind::Ds},
      {ROCJITSU_CODE_ARCH_CDNA2, 0xbf8c0f70u, WaitCounterKind::Load},
      {ROCJITSU_CODE_ARCH_CDNA2, 0xbf8cc07fu, WaitCounterKind::Ds},
      {ROCJITSU_CODE_ARCH_RDNA1, 0xbf8c3f70u, WaitCounterKind::Load},
      {ROCJITSU_CODE_ARCH_RDNA1, 0xbf8cc07fu, WaitCounterKind::Ds},
      {ROCJITSU_CODE_ARCH_RDNA1, 0xbbfd0000u, WaitCounterKind::Store},
      {ROCJITSU_CODE_ARCH_RDNA2, 0xbf8c3f70u, WaitCounterKind::Load},
      {ROCJITSU_CODE_ARCH_RDNA2, 0xbf8cc07fu, WaitCounterKind::Ds},
      {ROCJITSU_CODE_ARCH_RDNA2, 0xbbfd0000u, WaitCounterKind::Store},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc20000u, WaitCounterKind::Sample},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc30000u, WaitCounterKind::Bvh},
      {ROCJITSU_CODE_ARCH_CDNA3, 0xbf8c0f70u, WaitCounterKind::Load},
      {ROCJITSU_CODE_ARCH_CDNA3, 0xbf8cc07fu, WaitCounterKind::Ds},
      {ROCJITSU_CODE_ARCH_CDNA4, 0xbf8c0f70u, WaitCounterKind::Load},
      {ROCJITSU_CODE_ARCH_CDNA4, 0xbf8cc07fu, WaitCounterKind::Ds},
      {ROCJITSU_CODE_ARCH_RDNA3, 0xbf8903f7u, WaitCounterKind::Load},
      {ROCJITSU_CODE_ARCH_RDNA3, 0xbf89fc07u, WaitCounterKind::Ds},
      {ROCJITSU_CODE_ARCH_RDNA3, 0xbc7c0000u, WaitCounterKind::Store},
      {ROCJITSU_CODE_ARCH_RDNA3_5, 0xbf8903f7u, WaitCounterKind::Load},
      {ROCJITSU_CODE_ARCH_RDNA3_5, 0xbf89fc07u, WaitCounterKind::Ds},
      {ROCJITSU_CODE_ARCH_RDNA3_5, 0xbc7c0000u, WaitCounterKind::Store},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc00000u, WaitCounterKind::Load},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc10000u, WaitCounterKind::Store},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc60000u, WaitCounterKind::Ds},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc70000u, WaitCounterKind::Km},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc40000u, WaitCounterKind::Exp},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfc00000u, WaitCounterKind::Load},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfc10000u, WaitCounterKind::Store},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfc60000u, WaitCounterKind::Ds},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfc70000u, WaitCounterKind::Km},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfca0000u, WaitCounterKind::Async},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfcb0000u, WaitCounterKind::Tensor},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(static_cast<int>(c.arch));
    SCOPED_TRACE(c.word);
    state.clear();
    auto decoder = Decoder::create(c.arch);
    ASSERT_NE(decoder, nullptr);
    util::StringDiagnostic error;
    auto decoded =
        decoder->decode_window(std::span<const uint32_t>(&c.word, 1), 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    load(5, c.counter);
    state.before(*decoded.value(), c.arch);
    EXPECT_TRUE(state.empty());
    EXPECT_FALSE(shadow.test(5));
  }
}

TEST_F(MemoryWaitScoreboardTest, Rdna4CompatibilityWaitDrainsAllCountersForEveryImmediate) {
  const auto counters = {WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds,
                         WaitCounterKind::Km,   WaitCounterKind::Exp,   WaitCounterKind::Sample,
                         WaitCounterKind::Bvh};
  for (uint32_t immediate : {0u, 0xfc07u, 0xffffu}) {
    SCOPED_TRACE(immediate);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
    const uint32_t word = 0xbf890000u | immediate;
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(std::span<const uint32_t>(&word, 1), 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    unsigned reg = 0;
    for (auto counter : counters)
      load(reg++, counter, ~uint64_t{0}, 0xf, true);
    state.before(*decoded.value(), ROCJITSU_CODE_ARCH_RDNA4);
    EXPECT_TRUE(state.empty());
    for (auto counter : counters)
      EXPECT_EQ(state.outstanding(counter), 0u);
    for (unsigned i = 0; i < counters.size(); ++i)
      read(i);
    EXPECT_TRUE(hazards.empty());
  }
}

TEST(MemoryWaitExecutionTest, WaitIdleAndRdna4CompatibilityWaitDrainEveryFunctionalCounter) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (uint32_t word : {0xbf890000u, 0xbf89fc07u, 0xbf89ffffu, 0xbf8a0000u}) {
      if (arch == ROCJITSU_CODE_ARCH_CDNA5 && word != 0xbf8a0000u)
        continue;
      SCOPED_TRACE(static_cast<unsigned>(arch));
      SCOPED_TRACE(word);
      GpuMemory memory("wait_memory");
      L2Cache l2("wait_l2");
      ComputeUnitCore::Config config{};
      config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
      config.arch = arch;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 128;
      config.vgprs_per_wf = 32;
      config.lds_size_kb = 64;
      auto cu = ComputeUnitCore::create("wait_cu", config, &memory, &l2);
      ASSERT_NE(cu, nullptr);
      auto *wf = cu->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
      ASSERT_NE(wf, nullptr);
      const auto counters = {WaitCounterType::LOADCNT, WaitCounterType::STORECNT,
                             WaitCounterType::DSCNT,   WaitCounterType::KMCNT,
                             WaitCounterType::EXPCNT,  WaitCounterType::TENSORCNT,
                             WaitCounterType::ASYNCCNT};
      for (auto counter : counters)
        wf->wait_counters().increment(counter);
      auto decoder = Decoder::create(arch);
      util::StringDiagnostic error;
      auto decoded =
          decoder->decode_window(std::span<const uint32_t>(&word, 1), 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      auto &inst = *decoded.value();
      inst.execute(inst, wf);
      EXPECT_EQ(wf->state(), WfState::WAITCNT);
      for (auto counter : counters) {
        EXPECT_FALSE(wf->wait_satisfied());
        wf->wait_counters().decrement(counter);
      }
      EXPECT_TRUE(wf->wait_satisfied());
    }
  }
}

TEST_F(MemoryWaitScoreboardTest, EveryCompletionCounterKeepsItsOwnOrderedSuffix) {
  for (auto counter : {WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds,
                       WaitCounterKind::Km, WaitCounterKind::Exp, WaitCounterKind::Sample,
                       WaitCounterKind::Bvh, WaitCounterKind::Async, WaitCounterKind::Tensor}) {
    SCOPED_TRACE(wait_counter_name(counter));
    state.clear();
    hazards.clear();
    load(5, counter);
    state.issue(counter); // A producer without a register result still counts.
    load(6, counter);
    state.wait(counter, 1);
    EXPECT_EQ(state.outstanding(counter), 1u);
    read(5);
    EXPECT_TRUE(hazards.empty());
    read(6);
    ASSERT_EQ(hazards.size(), 1u);
    EXPECT_EQ(hazards.front().required_wait, 0u);
  }
}

TEST_F(MemoryWaitScoreboardTest, MixedHardwareEventKindsRequireAZeroWait) {
  state.add({state.issue(WaitCounterKind::Exp, false, 1),
             0x100,
             1,
             {RegClass::VGPR, 5, 1},
             WaitCounterKind::Exp,
             0xf});
  state.issue(WaitCounterKind::Exp, false, 2);
  state.wait(WaitCounterKind::Exp, 1);
  read(5);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_EQ(hazards.front().required_wait, 0u);
  state.wait(WaitCounterKind::Exp, 0);
  state.add({state.issue(WaitCounterKind::Exp, false, 1),
             0x100,
             1,
             {RegClass::VGPR, 6, 1},
             WaitCounterKind::Exp,
             0xf});
  state.issue(WaitCounterKind::Exp, false, 1);
  state.wait(WaitCounterKind::Exp, 1);
  read(6);
  EXPECT_EQ(hazards.size(), 1u); // Zero wait reset the mixed-kind state.
}

TEST(MemoryWaitExecutionTest, ZeroExecTransposeResultsRequireDsWait) {
  GpuMemory memory("transpose_wait_memory");
  L2Cache l2("transpose_wait_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 32;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("transpose_wait_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0);
  auto decoder = Decoder::create(config.arch);
  for (auto opcode : {cdna5::kDsLoadTr4B64Vds, cdna5::kDsLoadTr6B96Vds, cdna5::kDsLoadTr16B128Vds,
                      cdna5::kDsLoadTr8B64Vds}) {
    for (bool waited : {false, true}) {
      for (bool write : {false, true}) {
        SCOPED_TRACE(opcode);
        SCOPED_TRACE(waited);
        SCOPED_TRACE(write);
        wf->ensure_memory_wait_scoreboard().clear();
        const auto before = cu->memory_wait_diagnostic_count();
        const auto words = cdna5::build_vds(opcode, {.addr = 0, .vdst = 2});
        util::StringDiagnostic error;
        auto decoded = decoder->decode_window(words, 0, error.emitter());
        ASSERT_TRUE(decoded.succeeded()) << error.message();
        auto &inst = *decoded.value();
        inst.execute(inst, wf);
        ASSERT_NE(inst.data(), nullptr);
        ASSERT_NE(inst.data_as<VectorMemState>()->exec_mask, 0u);
        cu->track_memory_wait(inst, *wf);
        auto &state = *wf->memory_wait_scoreboard();
        EXPECT_EQ(state.outstanding(WaitCounterKind::Ds), 1u);
        if (waited)
          state.wait(WaitCounterKind::Ds, 0);
        state.access({RegClass::VGPR, 2, 1}, 1, 0xf, write);
        EXPECT_EQ(cu->memory_wait_diagnostic_count() - before, waited ? 0u : 1u);
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, InlineDsResultAndCounterOnlyNopUseOrderedQueue) {
  for (unsigned threshold : {0u, 1u, 2u}) {
    SCOPED_TRACE(threshold);
    std::vector<uint32_t> code;
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 129, .vdst = 1}));
    append_instruction(code, cdna5::build_vds(cdna5::kDsSwizzleB32Vds, {.addr = 1, .vdst = 2}));
    append_instruction(code, cdna5::build_vds(cdna5::kDsNopVds));
    append_instruction(code, cdna5::build_sopp(cdna5::kSWaitDscntSopp,
                                               {.simm16 = static_cast<uint16_t>(threshold)}));
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 258, .vdst = 3}));
    append_instruction(code, S_ENDPGM_GFX12);
    Gfx1250Sim sim(memory_wait_test_config());
    auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32);
    test::AqlQueue queue(sim.memory, sim.cp());
    queue.dispatch(kernel, 32, 32);
    step_until_halted(*sim.engine, *sim.cu());
    ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
    EXPECT_EQ(sim.snapshot->snapshots().front().vgpr(3, 0), 1u);
    EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), threshold == 2 ? 1u : 0u);
  }
}

TEST(MemoryWaitExecutionTest, CounterAdmissionPrecedesTheIncomingInstructionsRegisterReads) {
  for (unsigned younger : {61u, 62u}) {
    std::vector<uint32_t> code;
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 129, .vdst = 1}));
    append_instruction(code, cdna5::build_vds(cdna5::kDsSwizzleB32Vds, {.addr = 1, .vdst = 2}));
    for (unsigned i = 0; i < younger; ++i)
      append_instruction(code, cdna5::build_vds(cdna5::kDsNopVds));
    append_instruction(code, cdna5::build_vds(cdna5::kDsSwizzleB32Vds, {.addr = 2, .vdst = 3}));
    append_instruction(code, S_ENDPGM_GFX12);
    Gfx1250Sim sim(memory_wait_test_config());
    auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32);
    test::AqlQueue queue(sim.memory, sim.cp());
    queue.dispatch(kernel, 32, 32);
    step_until_halted(*sim.engine, *sim.cu());
    ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
    EXPECT_EQ(sim.snapshot->snapshots().front().vgpr(3, 0), 1u);
    EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), younger == 61 ? 1u : 0u);
  }
}

TEST(MemoryWaitExecutionTest, ReturnMessagesHaveSeparateSendAndReturnCompletions) {
  for (unsigned threshold : {0u, 1u, 2u, 3u}) {
    SCOPED_TRACE(threshold);
    std::vector<uint32_t> code;
    append_instruction(code,
                       cdna5::build_sop1(cdna5::kSSendmsgRtnB32Sop1, {.ssrc0 = 0x80, .sdst = 4}));
    append_instruction(code,
                       cdna5::build_sop1(cdna5::kSSendmsgRtnB32Sop1, {.ssrc0 = 0x80, .sdst = 5}));
    append_instruction(code, cdna5::build_sopp(cdna5::kSWaitKmcntSopp,
                                               {.simm16 = static_cast<uint16_t>(threshold)}));
    append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 4, .sdst = 6}));
    append_instruction(code, S_ENDPGM_GFX12);
    Gfx1250Sim sim(memory_wait_test_config());
    auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32);
    test::AqlQueue queue(sim.memory, sim.cp());
    queue.dispatch(kernel, 32, 32);
    step_until_halted(*sim.engine, *sim.cu());
    EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), threshold == 3 ? 1u : 0u);
  }
}

TEST(MemoryWaitExecutionTest, MessageResultsCheckM0AndOnlyConsumedExecWords) {
  for (uint8_t destination : {125, 126, 127}) {
    // Explicit read/write, implicit EXEC use, unrelated scalar operation, and
    // full-pair scalar use. The producer returns zero, so implicit reads must
    // still observe pending EXEC_LO even though eager execution deactivates it.
    for (unsigned consumer : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
      for (unsigned wait : {0u, 1u, 2u}) {
        SCOPED_TRACE(destination);
        SCOPED_TRACE(consumer);
        SCOPED_TRACE(wait);
        std::vector<uint32_t> code;
        append_instruction(code, cdna5::build_sop1(cdna5::kSSendmsgRtnB32Sop1,
                                                   {.ssrc0 = 0x80, .sdst = destination}));
        if (wait)
          append_instruction(
              code, cdna5::build_sopp(wait == 1 ? cdna5::kSWaitKmcntSopp : cdna5::kSWaitDscntSopp,
                                      {.simm16 = 0}));
        if (consumer == 0)
          append_instruction(
              code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = destination, .sdst = 4}));
        else if (consumer == 1)
          append_instruction(
              code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = destination}));
        else if (consumer == 2)
          append_instruction(code, cdna5::build_sopp(cdna5::kSCbranchExeczSopp, {.simm16 = 0}));
        else if (consumer == 3)
          append_instruction(code, cdna5::build_sopp(cdna5::kSNopSopp, {.simm16 = 0}));
        else if (consumer == 4)
          append_instruction(
              code, cdna5::build_sop1(cdna5::kSAndSaveExecB64Sop1, {.ssrc0 = 193, .sdst = 4}));
        else if (consumer == 5)
          append_instruction(code,
                             cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 126}));
        else
          append_instruction(code,
                             cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 2}));
        append_instruction(code, S_ENDPGM_GFX12);
        Gfx1250Sim sim(memory_wait_test_config());
        auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32);
        test::AqlQueue queue(sim.memory, sim.cp());
        queue.dispatch(kernel, 32, 32);
        step_until_halted(*sim.engine, *sim.cu());
        const bool dependent = consumer <= 1 ||
                               ((consumer == 2 || consumer >= 5) && destination == 126) ||
                               (consumer == 4 && destination >= 126);
        EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), wait != 1 && dependent ? 1u : 0u);
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, CounterOnlyCacheOperationCountsWithZeroExecBehindPendingLoad) {
  using namespace rocr::llvm::amdhsa;
  for (bool zero_exec : {false, true}) {
    for (unsigned threshold : {1u, 2u}) {
      SCOPED_TRACE(zero_exec);
      SCOPED_TRACE(threshold);
      std::vector<uint32_t> code;
      append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
      append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                                    {.saddr = 0, .vdst = 2, .vaddr = 0}));
      if (zero_exec)
        append_instruction(code,
                           cdna5::build_sop1(cdna5::kSMovB64Sop1, {.ssrc0 = 128, .sdst = 126}));
      append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalInvVglobal));
      if (zero_exec)
        append_instruction(code,
                           cdna5::build_sop1(cdna5::kSMovB64Sop1, {.ssrc0 = 193, .sdst = 126}));
      append_instruction(code, cdna5::build_sopp(cdna5::kSWaitLoadcntSopp,
                                                 {.simm16 = static_cast<uint16_t>(threshold)}));
      append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 258, .vdst = 3}));
      append_instruction(code, S_ENDPGM_GFX12);
      uint32_t properties = 0;
      AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
      Gfx1250Sim sim(memory_wait_test_config());
      write_global_u32(*sim.memory, 0x400000, 0x12345678);
      auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32, 2, false, false,
                                     false, properties, 16);
      test::AqlQueue queue(sim.memory, sim.cp());
      queue.dispatch(kernel, 32, 32, 0x400000);
      step_until_halted(*sim.engine, *sim.cu());
      ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
      EXPECT_EQ(sim.snapshot->snapshots().front().vgpr(3, 0), 0x12345678u);
      EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), threshold == 2 ? 1u : 0u);
    }
  }
}

TEST(MemoryWaitExecutionTest, ScalarMissingWaitWarnsWithoutChangingTheResult) {
  using namespace rocr::llvm::amdhsa;
  for (bool wait : {false, true}) {
    SCOPED_TRACE(wait);
    std::vector<uint32_t> code;
    append_instruction(code, make_s_load_b32_scaled_imm(4, 0, 0));
    if (wait)
      append_instruction(code, S_WAIT_KMCNT_0_GFX12);
    append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 4, .sdst = 5}));
    append_instruction(code, S_ENDPGM_GFX12);
    uint32_t properties = 0;
    AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
    Gfx1250Sim sim(memory_wait_test_config());
    write_global_u32(*sim.memory, 0x400000, 0x12345678);
    auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32, 2, false, false,
                                   false, properties, 16);
    test::AqlQueue queue(sim.memory, sim.cp());
    queue.dispatch(kernel, 32, 32, 0x400000);
    step_until_halted(*sim.engine, *sim.cu());
    ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
    EXPECT_EQ(sim.snapshot->snapshots().front().sgpr(5), 0x12345678u);
    EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), wait ? 0u : 1u);
  }
}

TEST(MemoryWaitExecutionTest, FlatLanesHaveSeparateDependenciesAndKeepBothQueueEntries) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                    ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (uint64_t shared_lanes : {0u, 0b0101u, 0b1111u}) {
      SCOPED_TRACE(static_cast<unsigned>(arch));
      SCOPED_TRACE(shared_lanes);
      GpuMemory memory("flat_memory");
      L2Cache l2("flat_l2");
      ComputeUnitCore::Config config{};
      config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
      config.arch = arch;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 128;
      config.vgprs_per_wf = 32;
      config.lds_size_kb = 64;
      auto cu = ComputeUnitCore::create("flat_cu", config, &memory, &l2);
      ASSERT_NE(cu, nullptr);
      auto *wf = cu->dispatch_wf(0, 0x100, config.sgprs_per_wf, config.vgprs_per_wf);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(0b1111);
      auto data = std::make_unique<VectorMemState>(GLOBAL_MEM);
      data->is_load = true;
      data->exec_mask = 0b1111;
      data->lane_mask = 0b1111;
      data->elem_size = 4;
      data->num_elems = 1;
      data->dst_reg_base = wf->vgpr_alloc().base + 2;
      Instruction inst("flat_load_b32", nullptr);
      inst.set_data(std::move(data));
      cu->track_memory_wait(inst, *wf, shared_lanes);
      auto &state = *wf->memory_wait_scoreboard();
      EXPECT_EQ(state.outstanding(WaitCounterKind::Load), 1u);
      EXPECT_EQ(state.outstanding(WaitCounterKind::Ds), 1u);
      std::vector<MemoryWaitScoreboard::Hazard> hazards;
      state.bind(0x200, &hazards, [](void *p, const auto &hazard) {
        static_cast<decltype(hazards) *>(p)->push_back(hazard);
      });
      const RegisterRef result{RegClass::VGPR, 2, 1};
      state.wait(WaitCounterKind::Load, 0);
      state.access(result, 0b1111 & ~shared_lanes, 0xf, false);
      EXPECT_TRUE(hazards.empty());
      EXPECT_EQ(state.outstanding(WaitCounterKind::Ds), 1u);
      state.access(result, shared_lanes, 0xf, false);
      ASSERT_EQ(hazards.size(), shared_lanes ? 1u : 0u);
      if (shared_lanes) {
        EXPECT_EQ(hazards.front().producer.counter, WaitCounterKind::Ds);
      }
      state.wait(WaitCounterKind::Ds, 0);
      EXPECT_EQ(state.outstanding(WaitCounterKind::Ds), 0u);
      EXPECT_TRUE(state.empty());
    }
  }
}

TEST(MemoryWaitExecutionTest, IncomingFlatOverwriteUsesItsOwnOrderingAndRoutedLanes) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3}) {
    for (uint64_t shared_lanes : {0u, 0b0101u, 0b1111u}) {
      for (bool wait : {false, true}) {
        SCOPED_TRACE(static_cast<unsigned>(arch));
        SCOPED_TRACE(shared_lanes);
        SCOPED_TRACE(wait);
        GpuMemory memory("flat_overwrite_memory");
        L2Cache l2("flat_overwrite_l2");
        ComputeUnitCore::Config config{};
        config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
        config.arch = arch;
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 128;
        config.vgprs_per_wf = 32;
        config.lds_size_kb = 64;
        auto cu = ComputeUnitCore::create("flat_overwrite_cu", config, &memory, &l2);
        auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
        ASSERT_NE(wf, nullptr);
        wf->set_exec(0b1111);
        for (bool flat : {false, true}) {
          Instruction inst(flat ? "flat_load_dword" : "global_load_dword", nullptr);
          auto data = std::make_unique<VectorMemState>(GLOBAL_MEM);
          data->is_load = true;
          data->exec_mask = data->lane_mask = 0b1111;
          data->elem_size = 4;
          data->num_elems = 1;
          data->dst_reg_base = wf->vgpr_alloc().base + 2;
          inst.set_data(std::move(data));
          if (flat && wait)
            wf->memory_wait_scoreboard()->wait(WaitCounterKind::Load, 0);
          cu->track_memory_wait(inst, *wf, flat ? shared_lanes : 0);
          wf->pc += 8;
        }
        const bool ordered_global = arch == ROCJITSU_CODE_ARCH_RDNA3 && !shared_lanes;
        EXPECT_EQ(cu->memory_wait_diagnostic_count(), wait || ordered_global ? 0u : 1u);
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, BarrierObserversPreservePendingResults) {
  using namespace rocr::llvm::amdhsa;
  class BarrierSnapshot final : public ExecutionPlugin {
  public:
    BarrierSnapshot() : ExecutionPlugin("barrier_snapshot") {}
    void onAmdgpuBarrierResolved(std::span<Wavefront *> members) override {
      for (auto *wf : members) {
        EXPECT_EQ(RegisterAccess(*wf).read_sgpr(wf->sgpr_alloc().base + 4), 0x12345678u);
        ++snapshots;
      }
    }
    unsigned snapshots = 0;
  };
  for (bool wait : {false, true}) {
    SCOPED_TRACE(wait);
    Gfx1250Sim sim(memory_wait_test_config());
    auto observer = std::make_unique<BarrierSnapshot>();
    auto *snapshot = observer.get();
    ASSERT_TRUE(sim.plugin_group->add(std::move(observer)));
    sim.soc->set_plugin_group(sim.plugin_group);
    std::vector<uint32_t> code;
    append_instruction(code, make_s_load_b32_scaled_imm(4, 0, 0));
    append_instruction(code, cdna5::build_sop1(cdna5::kSBarrierSignalSop1, {.ssrc0 = 193}));
    append_instruction(code, cdna5::build_sopp(cdna5::kSBarrierWaitSopp, {.simm16 = 0xffff}));
    if (wait)
      append_instruction(code, S_WAIT_KMCNT_0_GFX12);
    append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 4, .sdst = 5}));
    append_instruction(code, S_ENDPGM_GFX12);
    uint32_t properties = 0;
    AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
    write_global_u32(*sim.memory, 0x400000, 0x12345678);
    auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32, 2, false, false,
                                   false, properties, 16);
    test::AqlQueue queue(sim.memory, sim.cp());
    queue.dispatch(kernel, 64, 64, 0x400000);
    step_until_halted(*sim.engine, *sim.cu());
    EXPECT_EQ(snapshot->snapshots, 2u);
    EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), wait ? 0u : 2u);
  }
}

TEST(MemoryWaitExecutionTest, BarrierSccReadAndOverwriteNeedTheKmWait) {
  for (unsigned group_size : {32u, 64u}) {
    SCOPED_TRACE(group_size);
    for (unsigned consumer : {0u, 1u, 2u, 3u, 4u}) {
      for (unsigned wait : {0u, 1u, 2u}) {
        SCOPED_TRACE(consumer);
        SCOPED_TRACE(wait);
        std::vector<uint32_t> code;
        append_instruction(code,
                           cdna5::build_sop1(cdna5::kSBarrierSignalIsfirstSop1, {.ssrc0 = 193}));
        if (wait)
          append_instruction(
              code, cdna5::build_sopp(wait == 1 ? cdna5::kSWaitKmcntSopp : cdna5::kSWaitDscntSopp,
                                      {.simm16 = 0}));
        if (consumer == 0)
          append_instruction(code, cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = 0}));
        else if (consumer == 1)
          append_instruction(code,
                             cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 253, .sdst = 4}));
        else if (consumer == 2)
          append_instruction(
              code, cdna5::build_sopc(cdna5::kSCmpEqU32Sopc, {.ssrc0 = 128, .ssrc1 = 129}));
        else
          append_instruction(code, cdna5::build_sopk(cdna5::kSGetregB32Sopk,
                                                     {.simm16 = static_cast<uint16_t>(
                                                          4 | ((consumer == 3 ? 9 : 10) << 6)),
                                                      .sdst = 4}));
        append_instruction(code, S_ENDPGM_GFX12);
        Gfx1250Sim sim(memory_wait_test_config());
        auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32);
        test::AqlQueue queue(sim.memory, sim.cp());
        queue.dispatch(kernel, group_size, group_size);
        step_until_halted(*sim.engine, *sim.cu());
        EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(),
                  wait == 1 || consumer == 4 || group_size == 32 ? 0u : 2u);
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, SccHwregWritesCheckOnlyPermittedOverlappingFields) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool write : {false, true}) {
      for (bool privileged : {false, true}) {
        for (bool scc_field : {false, true}) {
          SCOPED_TRACE(static_cast<unsigned>(arch));
          SCOPED_TRACE(write);
          SCOPED_TRACE(privileged);
          SCOPED_TRACE(scc_field);
          GpuMemory memory("hwreg_wait_memory");
          L2Cache l2("hwreg_wait_l2");
          ComputeUnitCore::Config config{};
          config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
          config.arch = arch;
          config.num_wf_slots = 1;
          config.sgprs_per_wf = 128;
          config.vgprs_per_wf = 32;
          config.lds_size_kb = 64;
          auto cu = ComputeUnitCore::create("hwreg_wait_cu", config, &memory, &l2);
          auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
          ASSERT_NE(wf, nullptr);
          wf->write_scc(true);
          wf->set_in_trap_handler(privileged);
          auto &state = wf->ensure_memory_wait_scoreboard();
          std::vector<MemoryWaitScoreboard::Hazard> hazards;
          state.bind(0x200, &hazards, [](void *p, const auto &hazard) {
            static_cast<decltype(hazards) *>(p)->push_back(hazard);
          });
          state.add({state.issue(WaitCounterKind::Km),
                     0x100,
                     ~uint64_t{0},
                     {RegClass::SCC, 0, 1},
                     WaitCounterKind::Km,
                     0xf});
          const bool modern = arch == ROCJITSU_CODE_ARCH_CDNA5;
          const uint16_t field = (modern ? 4 : 2) | (((modern ? 9 : 0) + (scc_field ? 0 : 1)) << 6);
          const bool permitted = !write || (privileged && arch != ROCJITSU_CODE_ARCH_RDNA3);
          uint32_t value = 0;
          HwregAccessResult result;
          {
            ScopedMemoryWaitCheck check(&state);
            result = write ? write_hwreg_field(*wf, field, 0) : read_hwreg_field(*wf, field, value);
          }
          EXPECT_EQ(result == HwregAccessResult::Success, permitted);
          EXPECT_EQ(hazards.size(), permitted && scc_field ? 1u : 0u);
          if (!hazards.empty()) {
            EXPECT_EQ(hazards.front().write, write);
          }
          if (!write && scc_field) {
            EXPECT_EQ(value, 1u);
          }
          EXPECT_EQ(wf->read_scc(), !(write && permitted && scc_field));
        }
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, Wave64MaskReadChecksPendingVccHighWord) {
  for (uint64_t lanes : {uint64_t{0}, uint64_t{1}, uint64_t{1} << 32, ~uint64_t{0}}) {
    SCOPED_TRACE(lanes);
    GpuMemory memory("vcc_high_memory");
    L2Cache l2("vcc_high_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = ROCJITSU_CODE_ARCH_CDNA3;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 128;
    config.vgprs_per_wf = 32;
    config.lds_size_kb = 64;
    auto cu = ComputeUnitCore::create("vcc_high_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(wf->wf_size(), 64u);
    auto &state = wf->ensure_memory_wait_scoreboard();
    std::vector<MemoryWaitScoreboard::Hazard> hazards;
    state.bind(0x200, &hazards, [](void *p, const auto &hazard) {
      static_cast<decltype(hazards) *>(p)->push_back(hazard);
    });
    state.add({state.issue(WaitCounterKind::Ds, true),
               0x100,
               ~uint64_t{0},
               {RegClass::VCC, 1, 1},
               WaitCounterKind::Ds,
               0xf});
    {
      ScopedMemoryWaitCheck check(&state);
      (void)wf->vcc_mask(lanes);
    }
    EXPECT_EQ(hazards.size(), lanes >> 32 ? 1u : 0u);
  }
}

TEST(MemoryWaitExecutionTest, ScalarVccLoadChecksOnlyTheConsumedOrWrittenWords) {
  using namespace rocr::llvm::amdhsa;
  for (unsigned destination : {106u, 107u}) {
    for (unsigned consumer : {0u, 1u, 2u, 3u, 4u}) {
      for (bool wait : {false, true}) {
        SCOPED_TRACE(destination);
        SCOPED_TRACE(consumer);
        SCOPED_TRACE(wait);
        std::vector<uint32_t> code;
        append_instruction(code, make_s_load_b32_scaled_imm(destination, 0, 0));
        if (wait)
          append_instruction(code, S_WAIT_KMCNT_0_GFX12);
        if (consumer == 0)
          append_instruction(code, cdna5::build_sopp(cdna5::kSCbranchVccnzSopp, {.simm16 = 0}));
        else if (consumer == 1)
          append_instruction(code, cdna5::build_vop2(cdna5::kVCndmaskB32Vop2,
                                                     {.src0 = 128, .vsrc1 = 0, .vdst = 2}));
        else if (consumer == 2)
          append_instruction(code,
                             cdna5::build_vopc(cdna5::kVCmpEqU32Vopc, {.src0 = 128, .vsrc1 = 0}));
        else if (consumer == 3)
          append_instruction(code,
                             cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 106}));
        else
          append_instruction(code,
                             cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 106, .sdst = 4}));
        append_instruction(code, S_ENDPGM_GFX12);
        uint32_t properties = 0;
        AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
        Gfx1250Sim sim(memory_wait_test_config());
        write_global_u32(*sim.memory, 0x400000, 1);
        auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32, 2, false, false,
                                       false, properties, 16);
        test::AqlQueue queue(sim.memory, sim.cp());
        queue.dispatch(kernel, 32, 32, 0x400000);
        step_until_halted(*sim.engine, *sim.cu());
        EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), !wait && destination == 106 ? 1u : 0u);
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, SdwaExplicitCompareDoesNotOverwriteItsTemporaryVcc) {
  for (bool explicit_destination : {false, true}) {
    for (RegClass pending_class : {RegClass::VCC, RegClass::SGPR}) {
      SCOPED_TRACE(explicit_destination);
      SCOPED_TRACE(static_cast<unsigned>(pending_class));
      GpuMemory memory("sdwa_wait_memory");
      L2Cache l2("sdwa_wait_l2");
      ComputeUnitCore::Config config{};
      config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
      config.arch = ROCJITSU_CODE_ARCH_CDNA4;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 128;
      config.vgprs_per_wf = 32;
      config.lds_size_kb = 64;
      auto cu = ComputeUnitCore::create("sdwa_wait_cu", config, &memory, &l2);
      auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(1);
      wf->set_vcc_raw(0x123456789abcdef0);
      auto &state = wf->ensure_memory_wait_scoreboard();
      std::vector<MemoryWaitScoreboard::Hazard> hazards;
      state.bind(0x200, &hazards, [](void *p, const auto &hazard) {
        static_cast<decltype(hazards) *>(p)->push_back(hazard);
      });
      state.add({state.issue(WaitCounterKind::Ds, true),
                 0x100,
                 ~uint64_t{0},
                 {pending_class, static_cast<uint16_t>(pending_class == RegClass::SGPR ? 4 : 0), 2},
                 WaitCounterKind::Ds,
                 0xf});
      cdna4::VopcVopSdwaSdstEncMachineInst raw{};
      raw.src0 = amdgpu::SRC_SDWA;
      raw.vsrc1 = 1;
      raw.op = cdna4::kVCmpEqF32Vopc;
      raw.encoding = 0x7c000000u >> 25;
      raw.sdst = 4;
      raw.sd = explicit_destination;
      raw.src0_sel = raw.src1_sel = 6; // DWORD
      const auto words = std::bit_cast<std::array<uint32_t, 2>>(raw);
      auto decoder = Decoder::create(config.arch);
      util::StringDiagnostic error;
      auto decoded = decoder->decode_window(words, 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      {
        ScopedMemoryWaitCheck check(&state);
        auto &inst = *decoded.value();
        inst.execute(inst, wf);
      }
      EXPECT_EQ(hazards.size(),
                explicit_destination == (pending_class == RegClass::SGPR) ? 1u : 0u);
      if (explicit_destination) {
        EXPECT_EQ(wf->vcc(), 0x123456789abcdef0u);
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, ScratchAddressChecksDoNotObserveGlobalOrInactiveLanes) {
  for (unsigned segment : {0u, 1u, 2u}) {
    for (bool private_address : {false, true}) {
      for (bool active : {false, true}) {
        SCOPED_TRACE(segment);
        SCOPED_TRACE(private_address);
        SCOPED_TRACE(active);
        GpuMemory memory("scratch_memory");
        L2Cache l2("scratch_l2");
        ComputeUnitCore::Config config{};
        config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
        config.arch = ROCJITSU_CODE_ARCH_CDNA3;
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 128;
        config.vgprs_per_wf = 32;
        config.lds_size_kb = 64;
        auto cu = ComputeUnitCore::create("scratch_cu", config, &memory, &l2);
        auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
        ASSERT_NE(wf, nullptr);
        wf->set_exec(active ? 1 : 0);
        wf->set_scratch_base(0x400000);
        wf->set_apertures(0, 0, uint64_t{2} << 32, uint64_t{3} << 32);
        cu->write_vgpr(wf->vgpr_alloc().base, 0, 0);
        cu->write_vgpr(wf->vgpr_alloc().base + 1, 0, private_address ? 2 : 0);
        auto &state = wf->ensure_memory_wait_scoreboard();
        std::vector<MemoryWaitScoreboard::Hazard> hazards;
        state.bind(0x200, &hazards, [](void *p, const auto &hazard) {
          static_cast<decltype(hazards) *>(p)->push_back(hazard);
        });
        state.add({state.issue(WaitCounterKind::Ds, true),
                   0x100,
                   ~uint64_t{0},
                   {RegClass::FLAT_SCRATCH, 0, 2},
                   WaitCounterKind::Ds,
                   0xf});
        struct FlatFields {
          unsigned seg, saddr = 0x7f, addr = 0, offset = 0, pad_12 = 0, lds = 1;
        } fields{segment};
        VectorMemState data(GLOBAL_MEM);
        {
          ScopedMemoryWaitCheck check(&state);
          addr_calc::flat_calculate_addresses(fields, *wf, data);
        }
        EXPECT_EQ(hazards.size(),
                  active && (segment == 1 || (segment == 0 && private_address)) ? 1u : 0u);
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, VopdMoveChecksOnlyItsConsumedSources) {
  // v_dual_mov_b32 v3, s3 :: v_dual_add_nc_u32 v2, s1, v6
  const uint32_t words[] = {0xca200003u, 0x03020c01u};
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4,
                    ROCJITSU_CODE_ARCH_CDNA5}) {
    for (uint16_t pending_reg : {0, 6}) {
      SCOPED_TRACE(static_cast<unsigned>(arch));
      SCOPED_TRACE(pending_reg);
      GpuMemory memory("vopd_memory");
      L2Cache l2("vopd_l2");
      ComputeUnitCore::Config config{};
      config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
      config.arch = arch;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 128;
      config.vgprs_per_wf = 32;
      config.lds_size_kb = 64;
      auto cu = ComputeUnitCore::create("vopd_cu", config, &memory, &l2);
      auto *wf = cu->dispatch_wf(0, 0x200, 128, 32, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(1);
      cu->write_vgpr(wf->vgpr_alloc().base + 6, 0, 17);
      auto &state = wf->ensure_memory_wait_scoreboard();
      std::vector<MemoryWaitScoreboard::Hazard> hazards;
      state.bind(wf->pc, &hazards, [](void *p, const auto &hazard) {
        static_cast<decltype(hazards) *>(p)->push_back(hazard);
      });
      state.add({state.issue(WaitCounterKind::Load),
                 0x100,
                 1,
                 {RegClass::VGPR, pending_reg, 1},
                 WaitCounterKind::Load,
                 0xf});
      auto decoder = Decoder::create(arch);
      util::StringDiagnostic error;
      auto decoded = decoder->decode_window(words, 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      {
        ScopedMemoryWaitCheck check(&state);
        auto &inst = *decoded.value();
        inst.execute(inst, wf);
      }
      EXPECT_EQ(hazards.size(), pending_reg == 6 ? 1u : 0u);
      EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 2, 0), 17u);
    }
  }
}

TEST(MemoryWaitExecutionTest, NarrowStoresCheckOnlyConsumedBytes) {
  for (unsigned opcode : {24u, 25u, 36u, 37u}) {
    const bool high = opcode >= 36;
    const unsigned bytes = (opcode & 1) ? 2 : 1;
    const uint8_t read_mask = ((1u << bytes) - 1) << (high ? 2 : 0);
    for (unsigned pending_byte = 0; pending_byte < 4; ++pending_byte) {
      SCOPED_TRACE(opcode);
      SCOPED_TRACE(pending_byte);
      GpuMemory memory("store_memory");
      L2Cache l2("store_l2");
      ComputeUnitCore::Config config{};
      config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
      config.arch = ROCJITSU_CODE_ARCH_RDNA3;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 128;
      config.vgprs_per_wf = 32;
      config.lds_size_kb = 64;
      auto cu = ComputeUnitCore::create("store_cu", config, &memory, &l2);
      auto *wf = cu->dispatch_wf(0, 0x200, 128, 32, 32);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(1);
      cu->write_vgpr(wf->vgpr_alloc().base + 17, 0, 0x44332211);
      auto &state = wf->ensure_memory_wait_scoreboard();
      std::vector<MemoryWaitScoreboard::Hazard> hazards;
      state.bind(wf->pc, &hazards, [](void *p, const auto &hazard) {
        static_cast<decltype(hazards) *>(p)->push_back(hazard);
      });
      state.add({state.issue(WaitCounterKind::Load),
                 0x100,
                 1,
                 {RegClass::VGPR, 17, 1},
                 WaitCounterKind::Load,
                 static_cast<uint8_t>(1u << pending_byte)});
      // FLAT store from v17 to v[2:3], low/high byte or halfword.
      const uint32_t words[] = {0xdc000000u | (opcode << 18), 0x007c1102u};
      auto decoder = Decoder::create(config.arch);
      util::StringDiagnostic error;
      auto decoded = decoder->decode_window(words, 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      auto &inst = *decoded.value();
      {
        ScopedMemoryWaitCheck check(&state);
        inst.execute(inst, wf);
      }
      EXPECT_EQ(hazards.size(), (read_mask & (1u << pending_byte)) ? 1u : 0u);
      const auto &data = *inst.data_as<VectorMemState>();
      EXPECT_EQ(data.elem_size, bytes);
      ASSERT_GE(data.store_data.size(), bytes);
      EXPECT_EQ(data.store_data[0], high ? 0x33 : 0x11);
      if (bytes == 2) {
        EXPECT_EQ(data.store_data[1], high ? 0x44 : 0x22);
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, GlobalFlatResultNeedsOnlyItsLoadCounter) {
  using namespace rocr::llvm::amdhsa;
  for (unsigned waits : {0u, 1u, 2u, 3u}) {
    SCOPED_TRACE(waits);
    std::vector<uint32_t> code;
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 0, .vdst = 0}));
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 1, .vdst = 1}));
    append_instruction(
        code, cdna5::build_vflat(cdna5::kFlatLoadB32Vflat, {.saddr = 124, .vdst = 2, .vaddr = 0}));
    if (waits & 1)
      append_instruction(code, cdna5::build_sopp(cdna5::kSWaitLoadcntSopp, {.simm16 = 0}));
    if (waits & 2)
      append_instruction(code, cdna5::build_sopp(cdna5::kSWaitDscntSopp, {.simm16 = 0}));
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 258, .vdst = 3}));
    append_instruction(code, S_ENDPGM_GFX12);
    uint32_t properties = 0;
    AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
    Gfx1250Sim sim(memory_wait_test_config());
    write_global_u32(*sim.memory, 0x400000, 0x12345678);
    auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32, 2, false, false,
                                   false, properties, 16);
    test::AqlQueue queue(sim.memory, sim.cp());
    queue.dispatch(kernel, 32, 32, 0x400000);
    step_until_halted(*sim.engine, *sim.cu());
    ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
    EXPECT_EQ(sim.snapshot->snapshots().front().vgpr(3, 0), 0x12345678u);
    EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), (waits & 1) ? 0u : 1u);
  }
}

TEST(MemoryWaitExecutionTest, EagerLdsTransfersStillDiagnoseMissingWaits) {
  // RAW through DS, WAR through DS, DS -> async RAW, and async WAW all
  // execute correctly with eager memory. Only an architectural wait proves
  // them safe. DS -> DS is an ordered control requiring no memory wait.
  for (unsigned pattern = 0; pattern < 5; ++pattern)
    for (unsigned mode = 0; mode < 4; ++mode) {
      SCOPED_TRACE(std::format("pattern={} mode={}", pattern, mode));
      std::vector<uint32_t> code;
      append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 255, .sdst = 0}));
      append_instruction(code, 0x400000u);
      append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 1}));
      append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
      append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 1}));
      append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 129, .vdst = 2}));
      auto async_load = [&] {
        append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadAsyncToLdsB32Vglobal,
                                                      {.saddr = 0, .vdst = 1, .vaddr = 0}));
      };
      auto async_store = [&] {
        append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalStoreAsyncFromLdsB32Vglobal,
                                                      {.saddr = 0, .vsrc = 1, .vaddr = 0}));
      };
      auto ds_store = [&] {
        append_instruction(code, cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 1, .data0 = 2}));
      };
      if (pattern == 1)
        async_store();
      else if (pattern == 2 || pattern == 4)
        ds_store();
      else
        async_load();
      if (mode == 1)
        append_instruction(code, cdna5::build_sopp(pattern == 2 || pattern == 4
                                                       ? cdna5::kSWaitDscntSopp
                                                       : cdna5::kSWaitAsynccntSopp,
                                                   {.simm16 = 0}));
      if (mode == 2)
        append_instruction(code, cdna5::build_sopp(cdna5::kSWaitLoadcntSopp, {.simm16 = 0}));
      if (pattern == 0 || pattern == 4)
        append_instruction(code, cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 1, .vdst = 3}));
      else if (pattern == 1)
        ds_store();
      else if (pattern == 2)
        async_store();
      else
        async_load();
      append_instruction(code, S_ENDPGM_GFX12);

      Gfx1250Sim sim(memory_wait_test_config(mode == 3 ? "off" : "warn"));
      write_global_u32(*sim.memory, 0x400000, 0x12345678);
      const auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32);
      hsa_kernel_dispatch_packet_t packet{};
      packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
      packet.setup = 1;
      packet.workgroup_size_x = packet.grid_size_x = 32;
      packet.workgroup_size_y = packet.workgroup_size_z = packet.grid_size_y = packet.grid_size_z =
          1;
      packet.group_segment_size = 256;
      packet.kernel_object = kernel;
      test::AqlQueue queue(sim.memory, sim.cp());
      queue.submit(packet);
      step_until_halted(*sim.engine, *sim.cu());
      ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
      if (pattern == 0 || pattern == 4)
        EXPECT_EQ(sim.snapshot->snapshots().front().vgpr(3, 0), pattern == 0 ? 0x12345678u : 1u);
      EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(),
                pattern != 4 && (mode == 0 || mode == 2) ? 1u : 0u);
    }
}

TEST(MemoryWaitExecutionTest, LdsFootprintsUseExecutedLanesAndExactBytes) {
  for (const auto [producer_exec, producer_address, consumer_address, expected] :
       {std::tuple{1u, 8u, 8u, 1u},
        {1u, 8u, 11u, 1u},
        {1u, 8u, 12u, 0u},
        {1u, 8u, 32u, 0u},
        {0u, 8u, 8u, 0u},
        {1u, 64u, 0u, 0u}}) {
    SCOPED_TRACE(std::format("exec={} source={} consumer={}", producer_exec, producer_address,
                             consumer_address));
    Gfx1250Sim sim(memory_wait_test_config());
    auto *cu = sim.cu();
    auto *wf = sim.dispatch_scratch_wf();
    cu->allocate_lds(256); // Check nonzero physical LDS allocation bases too.
    wf->set_lds_base(cu->allocate_lds(256));
    wf->set_lds_size(64);
    wf->set_exec(producer_exec);
    write_wave_sgpr(*cu, *wf, 0, 0x400000);
    write_wave_sgpr(*cu, *wf, 1, 0);
    cu->write_vgpr(wf->vgpr_alloc().base, 0, 0);
    cu->write_vgpr(wf->vgpr_alloc().base + 1, 0, producer_address);
    cu->write_vgpr(wf->vgpr_alloc().base + 1, 1, 32); // Inactive producer lane.
    auto producer = decode_gfx1250(cdna5::build_vglobal(cdna5::kGlobalLoadAsyncToLdsB32Vglobal,
                                                        {.saddr = 0, .vdst = 1, .vaddr = 0}),
                                   "global_load_async_to_lds_b32");
    ASSERT_NE(producer, nullptr);
    producer->execute(*producer, wf);
    cu->track_memory_wait(*producer, *wf);
    wf->set_exec(2); // A different lane can conflict with the same LDS bytes.
    cu->write_vgpr(wf->vgpr_alloc().base + 1, 1, consumer_address);
    const auto words = cdna5::build_vds(cdna5::kDsLoadU8Vds, {.addr = 1, .vdst = 3});
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    auto &consumer = decoded.value();
    consumer->execute(*consumer, wf);
    cu->track_memory_wait(*consumer, *wf);
    EXPECT_EQ(cu->memory_wait_diagnostic_count(), expected);
  }
}

TEST(MemoryWaitExecutionTest, VectorWaitDiagnosticsCanBeSilenced) {
  using namespace rocr::llvm::amdhsa;
  for (unsigned mode = 0; mode < 5; ++mode) {
    SCOPED_TRACE(mode);
    std::vector<uint32_t> code;
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
    append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                                  {.saddr = 0, .vdst = 2, .vaddr = 0}));
    if (mode == 1)
      append_instruction(code, cdna5::build_sopp(cdna5::kSWaitLoadcntSopp, {.simm16 = 0}));
    if (mode == 2)
      append_instruction(code, S_WAIT_KMCNT_0_GFX12); // Wrong counter.
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 258, .vdst = 3}));
    // A wait after the consumer must not hide the premature read (the corpus
    // inline-assembly failure pattern).
    if (mode == 4)
      append_instruction(code, cdna5::build_sopp(cdna5::kSWaitLoadcntSopp, {.simm16 = 0}));
    append_instruction(code, S_ENDPGM_GFX12);
    uint32_t properties = 0;
    AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
    Gfx1250Sim sim(memory_wait_test_config(mode == 3 ? "off" : "warn"));
    write_global_u32(*sim.memory, 0x400000, 0x12345678);
    auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32, 2, false, false,
                                   false, properties, 16);
    test::AqlQueue queue(sim.memory, sim.cp());
    queue.dispatch(kernel, 32, 32, 0x400000);
    step_until_halted(*sim.engine, *sim.cu());
    ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
    EXPECT_EQ(sim.snapshot->snapshots().front().vgpr(3, 0), 0x12345678u);
    EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(),
              mode == 0 || mode == 2 || mode == 4 ? 1u : 0u);
  }
}

TEST_F(MemoryWaitScoreboardTest, ReplaySourcesOnlySlowDownAndDiagnoseWrites) {
  load(5, WaitCounterKind::X);
  EXPECT_FALSE(shadow.test(5));
  EXPECT_TRUE(shadow.test(5, true));
  read(5);
  EXPECT_TRUE(hazards.empty());
  state.access({RegClass::VGPR, 5, 1}, 1, 0xf, true);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_EQ(hazards[0].producer.counter, WaitCounterKind::X);
  EXPECT_FALSE(shadow.test(5, true));
}

TEST_F(MemoryWaitScoreboardTest, ReplayAndResultShadowBitsSurviveIndependentRetirement) {
  for (bool replay_first : {false, true}) {
    state.clear();
    load(5, WaitCounterKind::X);
    load(5, WaitCounterKind::Ds);
    EXPECT_TRUE(shadow.test(5));
    state.wait(replay_first ? WaitCounterKind::X : WaitCounterKind::Ds, 0);
    EXPECT_EQ(shadow.test(5), replay_first);
    EXPECT_TRUE(shadow.test(5, true));
    state.wait(replay_first ? WaitCounterKind::Ds : WaitCounterKind::X, 0);
    EXPECT_FALSE(shadow.test(5, true));
  }
}

TEST_F(MemoryWaitScoreboardTest, ReplayPartialWaitRetainsTheRequestedSuffix) {
  state.xcnt_group(false);
  load(5, WaitCounterKind::X);
  state.issue(WaitCounterKind::X); // Counter-only instruction.
  load(6, WaitCounterKind::X);
  state.wait(WaitCounterKind::X, 2);
  EXPECT_EQ(state.outstanding(WaitCounterKind::X), 2u);
  EXPECT_FALSE(shadow.test(5, true));
  EXPECT_TRUE(shadow.test(6, true));
  state.access({RegClass::VGPR, 6, 1}, 1, 0xf, true);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_EQ(hazards[0].required_wait, 0u);
}

TEST_F(MemoryWaitScoreboardTest, ReplayImplicitVmemWriteRetiresOnlyTheRequiredPrefix) {
  state.xcnt_group(false);
  load(5, WaitCounterKind::X, 2, 0xc);
  load(6, WaitCounterKind::X);
  state.xcnt_ordered_write({RegClass::VGPR, 5, 1}, 1, 0xf);
  EXPECT_TRUE(shadow.test(5, true));
  state.xcnt_ordered_write({RegClass::VGPR, 5, 1}, 2, 0x3);
  EXPECT_TRUE(shadow.test(5, true));
  state.xcnt_ordered_write({RegClass::VGPR, 5, 1}, 2, 0xc);
  EXPECT_FALSE(shadow.test(5, true));
  EXPECT_TRUE(shadow.test(6, true));
  EXPECT_EQ(state.outstanding(WaitCounterKind::X), 1u);
}

TEST_F(MemoryWaitScoreboardTest, ReplayGroupSwitchesDrainAndSmemRequiresZero) {
  state.xcnt_group(false);
  load(5, WaitCounterKind::X);
  state.xcnt_group(true);
  EXPECT_FALSE(shadow.test(5, true));
  load(6, WaitCounterKind::X, ~uint64_t{0}, 0xf, true);
  load(7, WaitCounterKind::X, ~uint64_t{0}, 0xf, true);
  state.wait(WaitCounterKind::X, 1);
  EXPECT_TRUE(shadow.test(6, true));
  state.xcnt_ordered_write({RegClass::VGPR, 6, 1}, ~uint64_t{0}, 0xf);
  EXPECT_TRUE(shadow.test(6, true));
  state.xcnt_group(false);
  EXPECT_FALSE(shadow.test(6, true));
  EXPECT_FALSE(shadow.test(7, true));
}

TEST_F(MemoryWaitScoreboardTest, ReplayCompletionWaitsMapMixedCountersToTranslationPositions) {
  state.xcnt_group(true);
  load(5, WaitCounterKind::X, ~uint64_t{0}, 0xf, true);
  state.wait(WaitCounterKind::Km, 1);
  EXPECT_TRUE(shadow.test(5, true));
  state.wait(WaitCounterKind::Km, 0);
  EXPECT_FALSE(shadow.test(5, true));
  state.xcnt_group(false);
  auto issue = [&](WaitCounterKind counter, uint16_t reg) {
    state.issue(counter);
    state.add({state.issue_xcnt(counter, false),
               0x100,
               1,
               {RegClass::VGPR, reg, 1},
               WaitCounterKind::X,
               0xf});
  };
  issue(WaitCounterKind::Load, 5);      // X1, Load1
  issue(WaitCounterKind::Store, 6);     // X2, Store1
  state.issue(WaitCounterKind::Load);   // Counter-only invalidate, no X event.
  issue(WaitCounterKind::Load, 7);      // X3, Load3
  issue(WaitCounterKind::Store, 8);     // X4, Store2
  state.wait(WaitCounterKind::Load, 2); // Only Load1 completed.
  EXPECT_FALSE(shadow.test(5, true));
  EXPECT_TRUE(shadow.test(6, true));
  EXPECT_TRUE(shadow.test(7, true));
  state.wait(WaitCounterKind::Store, 1); // Store1 proves X2 translated.
  EXPECT_FALSE(shadow.test(6, true));
  EXPECT_TRUE(shadow.test(7, true));
  state.wait(WaitCounterKind::Store, 0); // Store2 proves the entire older X prefix.
  EXPECT_FALSE(shadow.test(7, true));
  EXPECT_FALSE(shadow.test(8, true));
  EXPECT_EQ(state.outstanding(WaitCounterKind::Load), 2u);
  EXPECT_EQ(state.outstanding(WaitCounterKind::X), 0u);
}

TEST_F(MemoryWaitScoreboardTest, ArchitecturalDrainsRetireReplayButNotResults) {
  std::vector<std::vector<uint32_t>> encodings;
  auto add = [&](auto words) { encodings.emplace_back(words.begin(), words.end()); };
  for (auto opcode : {cdna5::kSBranchSopp, cdna5::kSSetVgprMsbSopp, cdna5::kSSendmsgSopp,
                      cdna5::kSBarrierWaitSopp, cdna5::kSEndpgmSopp, cdna5::kSTrapSopp})
    add(cdna5::build_sopp(opcode, {.simm16 = 0}));
  add(cdna5::build_sopk(cdna5::kSGetregB32Sopk, {.simm16 = 1, .sdst = 0}));
  add(cdna5::build_sopk(cdna5::kSSetregB32Sopk, {.simm16 = 1, .sdst = 0}));
  add(cdna5::build_sop1(cdna5::kSBarrierSignalSop1, {.ssrc0 = 128, .sdst = 0}));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  for (const auto &words : encodings) {
    state.clear();
    util::StringDiagnostic error;
    auto inst = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(inst.succeeded()) << error.message();
    SCOPED_TRACE(inst.value()->mnemonic());
    load(5, WaitCounterKind::X);
    load(6);
    state.before(*inst.value(), ROCJITSU_CODE_ARCH_CDNA5);
    EXPECT_FALSE(shadow.test(5, true));
    EXPECT_TRUE(shadow.test(6));
  }
}

void enable_multi_group_replay(std::vector<uint32_t> &code) {
  append_instruction(code,
                     cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = 1u | (25u << 6)}));
  code.push_back(1);
}

std::array<uint64_t, 2> run_xcnt_kernel(std::vector<uint32_t> code,
                                        std::string_view setting = "warn", unsigned vgprs = 32) {
  using namespace rocr::llvm::amdhsa;
  append_instruction(code, S_WAIT_KMCNT_0_GFX12);
  append_instruction(code, cdna5::build_sopp(cdna5::kSWaitLoadcntSopp, {.simm16 = 0}));
  append_instruction(code, cdna5::build_sopp(cdna5::kSWaitStorecntSopp, {.simm16 = 0}));
  append_instruction(code, S_ENDPGM_GFX12);
  Gfx1250Sim sim(memory_wait_test_config(setting));
  write_global_u32(*sim.memory, 0x400000, 0x12345678);
  uint32_t properties = 0;
  AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, vgprs, 2, false, false,
                                 false, properties, 16);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel, 32, 32, 0x400000);
  step_until_halted(*sim.engine, *sim.cu());
  EXPECT_EQ(sim.snapshot->snapshots().size(), 1u);
  if (setting != "warn") {
    EXPECT_FALSE(sim.cu()->wf(0)->memory_wait_checks_enabled());
    EXPECT_EQ(sim.cu()->wf(0)->memory_wait_scoreboard(), nullptr);
  }
  return {sim.cu()->xcnt_diagnostic_count(), sim.cu()->memory_wait_diagnostic_count()};
}

TEST(XcntExecutionTest, ScalarAddressOverwriteNeedsZeroXOrKmWait) {
  for (unsigned wait = 0; wait < 7; ++wait) {
    SCOPED_TRACE(wait);
    std::vector<uint32_t> code;
    append_instruction(code, make_s_load_b32_scaled_imm(4, 0, 0));
    append_instruction(code, make_s_load_b32_scaled_imm(5, 0, 0));
    if (wait == 1 || wait == 2)
      append_instruction(code,
                         cdna5::build_sopp(cdna5::kSWaitXcntSopp,
                                           {.simm16 = static_cast<uint16_t>(wait == 1 ? 0 : 1)}));
    if (wait == 3 || wait == 4)
      append_instruction(code,
                         cdna5::build_sopp(cdna5::kSWaitKmcntSopp,
                                           {.simm16 = static_cast<uint16_t>(wait == 3 ? 0 : 1)}));
    if (wait == 5)
      append_instruction(code, cdna5::build_sopp(cdna5::kSWaitDscntSopp, {.simm16 = 0}));
    if (wait == 6)
      append_instruction(code, cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0}));
    append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 0}));
    const auto counts = run_xcnt_kernel(code);
    EXPECT_EQ(counts[0], wait == 1 || wait == 3 || wait == 6 ? 0u : 1u);
    EXPECT_EQ(counts[1], 0u);
  }
}

TEST(XcntExecutionTest, VectorAddressAndExecRespectPartialTranslationAndLoadWaits) {
  for (unsigned consumer = 0; consumer < 3; ++consumer)
    for (unsigned wait = 0; wait < 5; ++wait) {
      SCOPED_TRACE(consumer);
      SCOPED_TRACE(wait);
      std::vector<uint32_t> code;
      enable_multi_group_replay(code);
      for (unsigned v : {0u, 1u})
        append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1,
                                                   {.src0 = 128, .vdst = static_cast<uint8_t>(v)}));
      for (unsigned v : {0u, 1u})
        append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                                      {.saddr = 0,
                                                       .vdst = static_cast<uint8_t>(2 + v),
                                                       .vaddr = static_cast<uint8_t>(v)}));
      if (wait)
        append_instruction(
            code, cdna5::build_sopp(wait < 3 ? cdna5::kSWaitXcntSopp : cdna5::kSWaitLoadcntSopp,
                                    {.simm16 = static_cast<uint16_t>(wait % 2 == 0 ? 0 : 1)}));
      if (consumer == 0)
        append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
      else
        append_instruction(
            code, cdna5::build_sop1(
                      cdna5::kSMovB32Sop1,
                      {.ssrc0 = 128, .sdst = static_cast<uint8_t>(consumer == 1 ? 0 : 126)}));
      const auto counts = run_xcnt_kernel(code);
      EXPECT_EQ(counts[0], wait && (consumer == 0 || wait % 2 == 0) ? 0u : 1u);
      EXPECT_EQ(counts[1], 0u);
    }
}

TEST(XcntExecutionTest, StoreDataIsProtectedAndSourceReadsAreAllowed) {
  for (unsigned wait = 0; wait < 4; ++wait) {
    SCOPED_TRACE(wait);
    std::vector<uint32_t> code;
    enable_multi_group_replay(code);
    for (unsigned v : {0u, 1u})
      append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1,
                                                 {.src0 = 128, .vdst = static_cast<uint8_t>(v)}));
    append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalStoreB32Vglobal,
                                                  {.saddr = 0, .vsrc = 1, .vaddr = 0}));
    // A read of the replay source is legal and must not consume its protection.
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 2}));
    if (wait)
      append_instruction(code, cdna5::build_sopp(wait == 1   ? cdna5::kSWaitXcntSopp
                                                 : wait == 2 ? cdna5::kSWaitLoadcntSopp
                                                             : cdna5::kSWaitStorecntSopp,
                                                 {.simm16 = 0}));
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 129, .vdst = 1}));
    const auto counts = run_xcnt_kernel(code);
    EXPECT_EQ(counts[0], wait == 1 || wait == 3 ? 0u : 1u);
    EXPECT_EQ(counts[1], 0u);
  }
}

TEST(XcntExecutionTest, MemoryWaitSettingControlsCompletionAndReplayChecksTogether) {
  for (unsigned mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    std::vector<uint32_t> code;
    append_instruction(code, make_s_load_b32_scaled_imm(4, 0, 0));
    append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 4, .sdst = 5}));
    append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 0}));
    const auto counts = run_xcnt_kernel(code, mode == 1 ? "warn" : mode == 2 ? "off" : "");
    EXPECT_EQ(counts[0], mode == 1 ? 1u : 0u);
    EXPECT_EQ(counts[1], mode == 1 ? 1u : 0u);
  }
}

TEST(XcntExecutionTest, InvalidMemoryWaitSettingIsRejected) {
  EXPECT_THROW(Gfx1250Sim(memory_wait_test_config("no")), std::invalid_argument);
}

TEST(XcntExecutionTest, SingleGroupVmemIsOutsideQualifiedCoverage) {
  std::vector<uint32_t> code;
  append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
  append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                                {.saddr = 0, .vdst = 2, .vaddr = 0}));
  append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
  EXPECT_EQ(run_xcnt_kernel(code), (std::array<uint64_t, 2>{0, 0}));
}

TEST(XcntExecutionTest, GroupTransitionsAndVmemDestinationsProvideImplicitOrdering) {
  for (unsigned mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    std::vector<uint32_t> code;
    enable_multi_group_replay(code);
    append_instruction(code, cdna5::build_sop1(cdna5::kSMovB64Sop1, {.ssrc0 = 0, .sdst = 6}));
    for (uint8_t v : {0, 1})
      append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = v}));
    if (mode == 0)
      append_instruction(code, make_s_load_b32_scaled_imm(4, 3, 0));
    append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                                  {.saddr = 0, .vdst = 2, .vaddr = 0}));
    if (mode == 1)
      append_instruction(code, make_s_load_b32_scaled_imm(4, 3, 0));
    if (mode == 2)
      append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                                    {.saddr = 0, .vdst = 0, .vaddr = 1}));
    if (mode == 0)
      append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 6}));
    else if (mode == 1)
      append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
    else
      append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 1}));
    const auto counts = run_xcnt_kernel(code);
    EXPECT_EQ(counts[0], mode == 2 ? 1u : 0u);
    EXPECT_EQ(counts[1], 0u);
  }
}

TEST(XcntExecutionTest, ReplaySourcesUseTheExecutedHighVgprBank) {
  std::vector<uint32_t> code;
  enable_multi_group_replay(code);
  append_instruction(code, cdna5::build_sopp(cdna5::kSNopSopp, {.simm16 = 0}));
  append_instruction(code, cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x41}));
  append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
  append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                                {.saddr = 0, .vdst = 2, .vaddr = 0}));
  append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 129, .vdst = 0}));
  EXPECT_EQ(run_xcnt_kernel(code, "warn", 320), (std::array<uint64_t, 2>{1, 0}));
}

TEST(XcntExecutionTest, ReplayAddressFootprintsHonorScratchAndBufferEnableBits) {
  GpuMemory memory("replay_memory");
  L2Cache l2("replay_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 32;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("replay_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
  ASSERT_NE(wf, nullptr);
  const RegisterAccess regs(*wf);
  auto decoder = Decoder::create(config.arch);
  auto check = [&](auto words, unsigned width, unsigned address_operand = 0) {
    util::StringDiagnostic error;
    auto inst = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(inst.succeeded()) << error.message();
    const auto address = regs.source_register(*inst.value()->src_operand(address_operand));
    ASSERT_EQ(address.has_value(), width != 0);
    if (address) {
      EXPECT_EQ(address->cls, RegClass::VGPR);
      EXPECT_EQ(address->index, 4u);
      EXPECT_EQ(address->width, width);
    }
    const auto data = regs.source_register(*inst.value()->src_operand(1 - address_operand));
    ASSERT_TRUE(data.has_value());
    EXPECT_EQ(data->index, 8u);
    EXPECT_EQ(data->width, 2u);
  };
  for (uint8_t enabled : {0, 1})
    check(cdna5::build_vscratch(cdna5::kScratchStoreB64Vscratch,
                                {.saddr = 0, .sve = enabled, .vsrc = 8, .vaddr = 4}),
          enabled);
  for (uint8_t offen : {0, 1})
    for (uint8_t idxen : {0, 1})
      check(cdna5::build_vbuffer(
                cdna5::kBufferStoreB64Vbuffer,
                {.soffset = 124, .vdata = 8, .offen = offen, .idxen = idxen, .vaddr = 4}),
            offen + idxen, 1);
}

TEST(XcntExecutionTest, EmptyExecWithoutPendingTranslationsHasNoReplayDependency) {
  std::vector<uint32_t> code;
  enable_multi_group_replay(code);
  append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 126}));
  append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                                {.saddr = 0, .vdst = 2, .vaddr = 0}));
  append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 129, .sdst = 126}));
  EXPECT_EQ(run_xcnt_kernel(code), (std::array<uint64_t, 2>{0, 0}));
}

} // namespace
