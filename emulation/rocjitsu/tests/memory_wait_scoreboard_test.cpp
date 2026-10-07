// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "mma_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <fstream>
#include <iterator>
#include <thread>
#include <tuple>
#include <utility>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::test::cdna5;

// Diagnostic tests opt in; explicit empty settings exercise production defaults.
std::string memory_wait_test_config(std::string_view setting = "warn",
                                    std::string_view xcnt_setting = "") {
  std::ifstream file(kGfx1250ConfigPath);
  std::string config((std::istreambuf_iterator<char>(file)), {});
  const auto cu = config.find("\"type\": \"compute_unit\"");
  const auto array = config.find('[', config.find("\"config\"", cu));
  if (!setting.empty())
    config.insert(
        array + 1,
        std::format("{{\"key\":\"memory_wait_diagnostics\",\"value\":\"{}\"}},", setting));
  if (!xcnt_setting.empty())
    config.insert(array + 1,
                  std::format("{{\"key\":\"xcnt_diagnostics\",\"value\":\"{}\"}},", xcnt_setting));
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

TEST_F(MemoryWaitScoreboardTest, UnorderedProducerResultsRequireZeroWait) {
  using namespace waitcheck_detail;
  // Synthetic destinations isolate retirement policy for each unordered class.
  for (const auto &[counter, kind] : {std::pair{WaitCounterKind::Ds, WaitEventKind::Gds},
                                      {WaitCounterKind::Km, WaitEventKind::SqMessage},
                                      {WaitCounterKind::Km, WaitEventKind::SccWrite},
                                      {WaitCounterKind::Exp, WaitEventKind::Export},
                                      {WaitCounterKind::Load, WaitEventKind::GlobalInv},
                                      {WaitCounterKind::Ds, WaitEventKind::Unknown}}) {
    SCOPED_TRACE(static_cast<unsigned>(kind));
    state.clear();
    hazards.clear();
    const ClassifiedEvent event{counter, kind};
    for (uint16_t reg : {5, 6})
      state.add({state.issue(event, ROCJITSU_CODE_ARCH_CDNA4),
                 0x100,
                 1,
                 {RegClass::VGPR, reg, 1},
                 counter,
                 0xf});
    state.wait(counter, 1);
    read(5);
    ASSERT_EQ(hazards.size(), 1u);
    EXPECT_EQ(hazards.front().required_wait, 0u);
    EXPECT_TRUE(shadow.test(6));
    state.wait(counter, 0);
    read(6);
    EXPECT_EQ(hazards.size(), 1u);
  }
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

TEST_F(MemoryWaitScoreboardTest, AdmissionReservesEveryIncomingCounterUnit) {
  for (bool message : {false, true}) {
    const auto arch = message ? ROCJITSU_CODE_ARCH_RDNA3 : ROCJITSU_CODE_ARCH_CDNA4;
    const uint32_t capacity = message ? 63 : 15;
    auto decoder = Decoder::create(arch);
    for (bool wide : {false, true}) {
      std::vector<uint32_t> words;
      if (message && wide)
        append_instruction(
            words, rdna3::build_sop1(rdna3::kSSendmsgRtnB32Sop1, {.ssrc0 = 128, .sdst = 4}));
      else if (message)
        append_instruction(words, rdna3::build_sopp(rdna3::kSSendmsgSopp, {.simm16 = 1}));
      else
        append_instruction(
            words, cdna4::build_smem(wide ? cdna4::kSLoadDwordx2Smem : cdna4::kSLoadDwordSmem,
                                     {.sbase = 0, .sdata = 4, .imm = 1}));
      util::StringDiagnostic error;
      auto decoded = decoder->decode_window(words, 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      const auto &inst = *decoded.value();
      for (uint32_t outstanding : {capacity - 2, capacity - 1, capacity}) {
        SCOPED_TRACE(inst.mnemonic());
        SCOPED_TRACE(outstanding);
        state.clear();
        load(5, WaitCounterKind::Ds);
        load(6, WaitCounterKind::Ds);
        load(7, WaitCounterKind::Ds, 1, 0xf, true);
        for (uint32_t i = 2; i < outstanding; ++i)
          state.issue(WaitCounterKind::Ds);
        state.before(inst, arch);
        const uint32_t retired =
            outstanding + (wide ? 2 : 1) > capacity ? outstanding + (wide ? 2 : 1) - capacity : 0;
        EXPECT_EQ(shadow.test(5), retired < 1);
        EXPECT_EQ(shadow.test(6), retired < 2);
        // Admission cannot establish the unordered result's readiness.
        EXPECT_TRUE(shadow.test(7));
      }
    }
  }
}

TEST_F(MemoryWaitScoreboardTest, BackpressureMatchesPossibleCompletionOrders) {
  // Enumerate every legal FIFO completion prefix, plus an independent unordered
  // completion. An admission may release a result only if every possible state
  // that leaves room for the incoming units has completed that result.
  for (uint32_t capacity : {3u, 5u, 7u}) {
    for (uint32_t incoming : {1u, 2u}) {
      for (uint32_t a_units : {1u, 2u}) {
        for (uint32_t a_count = 0; a_count <= 4; ++a_count) {
          for (uint32_t b_count = 0; b_count <= 4; ++b_count) {
            for (int prior_wait : {-1, 0, 2}) {
              SCOPED_TRACE(std::format("capacity={} incoming={} a_units={} a_count={} b_count={} "
                                       "prior_wait={}",
                                       capacity, incoming, a_units, a_count, b_count, prior_wait));
              state.clear();
              struct Result {
                uint16_t reg;
                unsigned kind;
                unsigned end;
              };
              std::vector<Result> results;
              auto add = [&](unsigned kind, unsigned end, unsigned units = 1) {
                const auto reg = static_cast<uint16_t>(results.size());
                results.push_back({reg, kind, end});
                const auto sequence = state.issue(WaitCounterKind::Ds, kind == 0, kind, units);
                state.add({sequence, 0x100, 1, {RegClass::VGPR, reg, 1}, WaitCounterKind::Ds, 0xf});
              };
              for (unsigned i = 0; i < std::max(a_count, b_count); ++i) {
                if (i < a_count)
                  add(1, (i + 1) * a_units, a_units);
                if (i < b_count)
                  add(2, i + 1);
              }
              add(0, 1); // One unordered operation cannot borrow a FIFO guarantee.
              const unsigned old_a = a_count * a_units;
              if (prior_wait >= 0)
                state.wait(WaitCounterKind::Ds, prior_wait);
              add(1, old_a + 1); // New work after the wait must remain distinguishable.
              state.backpressure(WaitCounterKind::Ds, capacity, incoming);
              std::vector<bool> can_be_pending(results.size(), false);
              unsigned possibilities = 0;
              for (unsigned done_a = 0; done_a <= old_a + 1; ++done_a) {
                for (unsigned done_b = 0; done_b <= b_count; ++done_b) {
                  for (unsigned done_unordered : {0u, 1u}) {
                    const unsigned old_pending =
                        old_a - std::min(done_a, old_a) + b_count - done_b + 1 - done_unordered;
                    if (prior_wait >= 0 && old_pending > static_cast<unsigned>(prior_wait))
                      continue;
                    const unsigned pending =
                        old_a + 1 - done_a + b_count - done_b + 1 - done_unordered;
                    if (pending + incoming > capacity)
                      continue;
                    ++possibilities;
                    for (const auto &result : results) {
                      const unsigned done = result.kind == 1   ? done_a
                                            : result.kind == 2 ? done_b
                                                               : done_unordered;
                      can_be_pending[result.reg] = can_be_pending[result.reg] || result.end > done;
                    }
                  }
                }
              }
              ASSERT_GT(possibilities, 0u);
              for (const auto &result : results)
                EXPECT_EQ(shadow.test(result.reg), can_be_pending[result.reg]) << result.reg;
            }
          }
        }
      }
    }
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
  const auto order = state.events().front().order;
  state.access({RegClass::VGPR, 5, 1}, ~uint64_t{0}, 0xf, true, order);
  EXPECT_TRUE(hazards.empty());
  load(6, WaitCounterKind::Ds);
  state.access({RegClass::VGPR, 6, 1}, ~uint64_t{0}, 0xf, true, order);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_EQ(hazards[0].producer.counter, WaitCounterKind::Ds);
}

TEST_F(MemoryWaitScoreboardTest, UnorderedQueueStillChecksOverwritesOnTheSameCounter) {
  load(5, WaitCounterKind::Load, 1, 0xf, true);
  state.access({RegClass::VGPR, 5, 1}, 1, 0xf, true, state.events().front().order);
  ASSERT_EQ(hazards.size(), 1u);
  EXPECT_TRUE(hazards.front().write);
  EXPECT_EQ(hazards.front().required_wait, 0u);

  state.wait(WaitCounterKind::Load, 0);
  hazards.clear();
  load(5, WaitCounterKind::Load, 1);
  state.access({RegClass::VGPR, 5, 1}, 1, 0xf, true, state.events().front().order);
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

TEST_F(MemoryWaitScoreboardTest, SupersedingResultsRequiresTheSameOrderedCompletionClass) {
  using namespace waitcheck_detail;
  for (auto kind : {WaitEventKind::VmemNoSamplerLoad, WaitEventKind::Sample, WaitEventKind::Bvh,
                    WaitEventKind::Smem}) {
    SCOPED_TRACE(static_cast<unsigned>(kind));
    state.clear();
    const ClassifiedEvent first{WaitCounterKind::Load, kind == WaitEventKind::Smem
                                                           ? kind
                                                           : WaitEventKind::VmemNoSamplerLoad};
    const ClassifiedEvent second{WaitCounterKind::Load, kind};
    for (const auto &event : {first, second})
      state.add({state.issue(event, ROCJITSU_CODE_ARCH_RDNA1),
                 0x100,
                 1,
                 {RegClass::VGPR, 5, 1},
                 WaitCounterKind::Load,
                 0xf});
    const bool same_ordered_class = kind == WaitEventKind::VmemNoSamplerLoad;
    EXPECT_EQ(state.events().size(), same_ordered_class ? 1u : 2u);
    // Completing the younger class must not erase the older class's result.
    state.issue(second, ROCJITSU_CODE_ARCH_RDNA1);
    state.wait(WaitCounterKind::Load, 1);
    EXPECT_EQ(shadow.test(5), !same_ordered_class);
    state.wait(WaitCounterKind::Load, 0);
    EXPECT_FALSE(shadow.test(5));
  }
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

TEST_F(MemoryWaitScoreboardTest, ConcurrentWavesKeepIndependentPendingRegisters) {
  load(5);
  std::thread other_wave([] {
    MemoryWaitShadow other_shadow;
    MemoryWaitScoreboard other_state(other_shadow);
    for (unsigned i = 0; i < 1000; ++i) {
      other_state.add({other_state.issue(WaitCounterKind::Load),
                       0x100,
                       1,
                       {RegClass::VGPR, 5, 1},
                       WaitCounterKind::Load,
                       0xf});
      other_state.wait(WaitCounterKind::Load, 0);
    }
    EXPECT_FALSE(other_shadow.test(5));
  });
  for (unsigned i = 0; i < 1000; ++i)
    EXPECT_TRUE(shadow.test(5));
  other_wave.join();
  EXPECT_TRUE(hazards.empty());
  read(5);
  ASSERT_EQ(hazards.size(), 1u);
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

TEST(MemoryWaitExecutionTest, FormattedLoadsTrackRegisterLayoutInsteadOfMemoryFootprint) {
  GpuMemory memory("format_wait_memory");
  L2Cache l2("format_wait_l2");
  ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA4;
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 32;
  auto cu = ComputeUnitCore::create("format_wait_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  auto decoder = Decoder::create(config.arch);
  struct Case {
    uint16_t opcode;
    std::array<uint8_t, 4> bytes;
  };
  const Case cases[] = {
      {rdna4::kBufferLoadFormatXVbuffer, {0xf, 0, 0, 0}},
      {rdna4::kBufferLoadFormatXyzwVbuffer, {0xf, 0xf, 0xf, 0xf}},
      {rdna4::kTbufferLoadFormatXyzwVbuffer, {0xf, 0xf, 0xf, 0xf}},
      {rdna4::kBufferLoadD16FormatXVbuffer, {0x3, 0, 0, 0}},
      {rdna4::kBufferLoadD16HiFormatXVbuffer, {0xc, 0, 0, 0}},
      {rdna4::kBufferLoadD16FormatXyVbuffer, {0xf, 0, 0, 0}},
      {rdna4::kBufferLoadD16FormatXyzVbuffer, {0xf, 0x3, 0, 0}},
      {rdna4::kBufferLoadD16FormatXyzwVbuffer, {0xf, 0xf, 0, 0}},
  };
  for (const auto &test : cases) {
    // The two formats read four and sixteen bytes, independently of the VGPR count.
    for (uint32_t format : {46u, 62u}) {
      for (uint8_t destination : {8, 31}) {
        SCOPED_TRACE(test.opcode);
        SCOPED_TRACE(format);
        SCOPED_TRACE(destination);
        const uint32_t scalar = wf->sgpr_alloc().base;
        cu->write_sgpr(scalar + 4, 0x1000);
        cu->write_sgpr(scalar + 5, 0);
        cu->write_sgpr(scalar + 6, 64);
        cu->write_sgpr(scalar + 7,
                       4 | (5 << 3) | (6 << 6) | (7 << 9) | (format << 12) | (1u << 28));
        const auto words =
            rdna4::build_vbuffer(test.opcode, {.soffset = rdna4::OPR_SREG_M0_NULL,
                                               .vdata = destination,
                                               .rsrc = 4,
                                               .format = static_cast<uint8_t>(format)});
        util::StringDiagnostic error;
        auto decoded = decoder->decode_window(words, 0, error.emitter());
        ASSERT_TRUE(decoded.succeeded()) << error.message();
        auto &inst = *decoded.value();
        const bool valid = destination == 8 || test.bytes[1] == 0;
        auto &state = wf->ensure_memory_wait_scoreboard();
        // Probe overwrite footprints before execute_impl has created a payload.
        for (unsigned offset = 0; offset < test.bytes.size(); ++offset)
          for (uint8_t bytes : {0x3, 0xc})
            for (uint64_t lanes : {1, 2}) {
              state.clear();
              unsigned reports = 0;
              state.bind(0x200, &reports,
                         [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
              state.add({state.issue(WaitCounterKind::Ds),
                         0x100,
                         lanes,
                         {RegClass::VGPR, static_cast<uint16_t>(destination + offset), 1},
                         WaitCounterKind::Ds,
                         bytes});
              state.check_instruction(inst, *wf);
              EXPECT_EQ(reports, valid && lanes == 1 && (test.bytes[offset] & bytes) ? 1u : 0u)
                  << "offset=" << offset << " bytes=" << unsigned(bytes);
            }
        state.clear();
        inst.execute(inst, wf);
        ASSERT_NE(inst.data(), nullptr);
        for (unsigned offset = 0; offset < test.bytes.size(); ++offset) {
          for (uint8_t bytes : {0x3, 0xc}) {
            for (bool waited : {false, true}) {
              for (bool write : {false, true}) {
                SCOPED_TRACE(offset);
                SCOPED_TRACE(bytes);
                SCOPED_TRACE(waited);
                SCOPED_TRACE(write);
                state.clear();
                cu->track_memory_wait(inst, *wf);
                unsigned reports = 0;
                state.bind(0x200, &reports,
                           [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
                if (waited)
                  state.wait(WaitCounterKind::Load, 0);
                const RegisterRef reg{RegClass::VGPR, static_cast<uint16_t>(destination + offset),
                                      1};
                state.access(reg, 2, bytes, write); // Inactive lane never needs a wait.
                EXPECT_EQ(reports, 0u);
                state.access(reg, 1, bytes, write);
                EXPECT_EQ(reports, valid && !waited && (test.bytes[offset] & bytes) ? 1u : 0u);
              }
            }
          }
        }
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, LdsStackTracksOnePointerWordAlongsideTwoPoppedNodes) {
  GpuMemory memory("stack_wait_memory");
  L2Cache l2("stack_wait_l2");
  ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA4;
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 32;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("stack_wait_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  auto decoder = Decoder::create(config.arch);
  for (uint8_t pointer : {16, 31}) {
    const auto words =
        rdna4::build_vds(rdna4::kDsBvhStackPush8Pop2RtnB64Vds,
                         {.offset0 = 16, .addr = pointer, .data0 = 0, .data1 = 4, .vdst = 12});
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    auto &inst = *decoded.value();
    inst.execute(inst, wf);
    ASSERT_NE(inst.data(), nullptr);
    for (uint16_t reg : {12, 13, 14, 16, 17, 31, 32}) {
      for (bool waited : {false, true}) {
        auto &state = wf->ensure_memory_wait_scoreboard();
        state.clear();
        cu->track_memory_wait(inst, *wf);
        unsigned reports = 0;
        state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
        if (waited)
          state.wait(WaitCounterKind::Ds, 0);
        state.access({RegClass::VGPR, reg, 1}, 1, 0xf, false);
        EXPECT_EQ(reports, !waited && (reg == 12 || reg == 13 || reg == pointer) ? 1u : 0u)
            << "pointer=" << unsigned(pointer) << " register=" << reg;
      }
    }
  }
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
        auto &precheck = wf->ensure_memory_wait_scoreboard();
        unsigned reports = 0;
        precheck.bind(0x200, &reports,
                      [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
        precheck.add({precheck.issue(WaitCounterKind::Load),
                      0x100,
                      uint64_t{1} << 7,
                      {RegClass::VGPR, 2, 1},
                      WaitCounterKind::Load,
                      0xf});
        precheck.check_instruction(inst, *wf);
        EXPECT_EQ(reports, 1u);
        precheck.clear();
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

TEST(MemoryWaitExecutionTest, ReturnMessageResultsRequireZeroWait) {
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
    // Each message has send and return units, but neither a younger send nor a
    // younger return proves that the first message's result has completed.
    EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), threshold != 0 ? 1u : 0u);
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

TEST(MemoryWaitExecutionTest, CounterOnlyCacheOperationDoesNotProveOlderLoadComplete) {
  using namespace rocr::llvm::amdhsa;
  for (bool zero_exec : {false, true}) {
    for (unsigned threshold : {0u, 1u, 2u}) {
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
      // Invalidation contributes to LOADcnt even with zero EXEC, but does not
      // share the load's completion order. Only a zero wait proves it ready.
      EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), threshold != 0 ? 1u : 0u);
    }
  }
}

TEST(MemoryWaitExecutionTest, ReusedInstructionResolvesCurrentSourceAndDestinationBanks) {
  GpuMemory memory("bank_wait_memory");
  L2Cache l2("bank_wait_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 1024;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("bank_wait_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 128, 1024, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  auto decoder = Decoder::create(config.arch);
  const auto words = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 2});
  util::StringDiagnostic error;
  auto decoded = decoder->decode_window(words, 0, error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  auto &state = wf->ensure_memory_wait_scoreboard();
  std::vector<MemoryWaitScoreboard::Hazard> hazards;
  state.bind(0x200, &hazards, [](void *p, const auto &hazard) {
    static_cast<decltype(hazards) *>(p)->push_back(hazard);
  });
  // Reuse the same decoded object across masks and bank changes, as the
  // instruction cache does. Full EXEC exercises the cached access plan.
  for (uint64_t exec : {uint64_t{1}, uint64_t{0xffffffff}, uint64_t{1}}) {
    wf->set_exec(exec);
    SCOPED_TRACE(exec);
    for (uint16_t mode : {0u, 0x81u, 0x43u, 0u}) {
      const auto set_words = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = mode});
      auto set = decoder->decode_window(set_words, 0, error.emitter());
      ASSERT_TRUE(set.succeeded()) << error.message();
      ASSERT_TRUE(cu->execute_instruction(set.value().get(), *wf).succeeded());
      ASSERT_EQ(wf->vgpr_msb_mode(), mode);
      for (bool destination : {false, true}) {
        const unsigned bank = destination ? (mode >> 6) & 3 : mode & 3;
        for (unsigned pending_bank = 0; pending_bank < 4; ++pending_bank) {
          SCOPED_TRACE(mode);
          SCOPED_TRACE(destination);
          SCOPED_TRACE(pending_bank);
          state.clear();
          hazards.clear();
          const uint16_t reg = pending_bank * 256 + (destination ? 2 : 1);
          state.add({state.issue(WaitCounterKind::Load),
                     0x100,
                     1,
                     {RegClass::VGPR, reg, 1},
                     WaitCounterKind::Load,
                     0xf});
          state.check_instruction(*decoded.value(), *wf);
          EXPECT_EQ(hazards.size(), pending_bank == bank ? 1u : 0u);
          if (!hazards.empty()) {
            EXPECT_EQ(hazards.front().write, destination);
          }
        }
      }
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

// Counter-class tests need decoded register identities, without an executable
// image implementation or a memory payload. Use the same pre-execution API as
// decoded instructions in the core issuer.
class TestMemoryLoad final : public Instruction {
public:
  TestMemoryLoad(std::string_view mnemonic, uint16_t reg) : Instruction(mnemonic, nullptr) {
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
    const auto words = rdna3::build_flat(
        rdna3::kFlatLoadB32Flat, {.addr = 0, .saddr = 124, .vdst = static_cast<uint8_t>(reg)});
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    if (!decoded.succeeded())
      throw std::runtime_error(error.message());
    backing_ = std::move(decoded.value());
    flags_ |= MEMORY_WAIT_PRODUCER;
    dst_operands_[0] = const_cast<Operand *>(backing_->dst_operand(0));
    num_dst_ = 1;
    src_operands_[0] = const_cast<Operand *>(backing_->src_operand(0));
    num_src_ = 1;
  }
  void amdgpu_register_modifiers(RegisterModifiers &modifiers) const override {
    modifiers.flat_address = src_operand(0);
  }

private:
  std::unique_ptr<Instruction> backing_;
};

void set_test_flat_domains(ComputeUnitCore &cu, Wavefront &wf, uint64_t shared_lanes) {
  constexpr uint64_t shared = uint64_t{2} << 32;
  cu.set_apertures(shared, shared + UINT32_MAX, 0, 0);
  wf.set_apertures(shared, shared + UINT32_MAX, 0, 0);
  for (unsigned lane = 0; lane < wf.wf_size(); ++lane) {
    cu.write_vgpr(wf.vgpr_alloc().base, lane, 0);
    cu.write_vgpr(wf.vgpr_alloc().base + 1, lane, shared_lanes & (uint64_t{1} << lane) ? 2 : 8);
  }
}

TEST(MemoryWaitExecutionTest, FlatLanesHaveSeparateDependenciesAndCounterParticipation) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                    ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (uint64_t shared_lanes : {0u, 0b0101u, 0b1111u}) {
      for (auto waited : {WaitCounterKind::Load, WaitCounterKind::Ds}) {
        SCOPED_TRACE(static_cast<unsigned>(waited));
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
        set_test_flat_domains(*cu, *wf, shared_lanes);
        TestMemoryLoad inst("flat_load_b32", 2);
        cu->track_memory_wait(inst, *wf);
        auto &state = *wf->memory_wait_scoreboard();
        EXPECT_EQ(state.outstanding(WaitCounterKind::Load), shared_lanes != 0b1111 ? 1u : 0u);
        EXPECT_EQ(state.outstanding(WaitCounterKind::Ds), shared_lanes != 0 ? 1u : 0u);
        std::vector<MemoryWaitScoreboard::Hazard> hazards;
        state.bind(0x200, &hazards, [](void *p, const auto &hazard) {
          static_cast<decltype(hazards) *>(p)->push_back(hazard);
        });
        const RegisterRef result{RegClass::VGPR, 2, 1};
        const auto pending =
            waited == WaitCounterKind::Load ? WaitCounterKind::Ds : WaitCounterKind::Load;
        const uint64_t ready_lanes =
            waited == WaitCounterKind::Ds ? shared_lanes : 0b1111 & ~shared_lanes;
        const uint64_t pending_lanes = 0b1111 & ~ready_lanes;
        state.wait(waited, 0);
        state.access(result, ready_lanes, 0xf, false);
        EXPECT_TRUE(hazards.empty());
        EXPECT_EQ(state.outstanding(pending), pending_lanes ? 1u : 0u);
        state.access(result, pending_lanes, 0xf, false);
        ASSERT_EQ(hazards.size(), pending_lanes ? 1u : 0u);
        if (pending_lanes) {
          EXPECT_EQ(hazards.front().producer.counter, pending);
          EXPECT_EQ(hazards.front().producer.lanes, pending_lanes);
        }
        state.wait(pending, 0);
        EXPECT_EQ(state.outstanding(pending), 0u);
        EXPECT_TRUE(state.empty());
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, DecodedFlatPlansMatchExecutedAddressesBeforeAnyRegisterAccess) {
  class Observer final : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("flat_planning_observer") {}
    unsigned reads = 0;
    void onAmdgpuReadVgprLanes(const Wavefront *, uint32_t, uint64_t, uint8_t) override { ++reads; }
    void onAmdgpuReadSgpr(const Wavefront *, uint32_t) override { ++reads; }
  };
  constexpr uint64_t shared = uint64_t{2} << 32;
  constexpr uint64_t private_base = uint64_t{3} << 32;
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA1,
                    ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                    ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(static_cast<unsigned>(arch));
    GpuMemory memory("flat_plan_memory");
    L2Cache l2("flat_plan_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 32;
    config.vgprs_per_wf = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 1024 : 256;
    config.lds_size_kb = 64;
    auto cu = ComputeUnitCore::create("flat_plan_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 32, config.vgprs_per_wf, 32);
    ASSERT_NE(wf, nullptr);
    cu->set_apertures(shared, shared + UINT32_MAX, private_base, private_base + UINT32_MAX);
    wf->set_apertures(shared, shared + UINT32_MAX, private_base, private_base + UINT32_MAX);
    wf->set_scratch_base(0x400000);
    wf->set_scratch_lane_size(256);
    const bool cdna5 = arch == ROCJITSU_CODE_ARCH_CDNA5;
    const bool legacy_cdna = arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
    const bool null127 =
        legacy_cdna || arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
    wf->set_vgpr_msb_mode(cdna5 ? 0x81 : 0); // src0 bank 1, destination bank 2.
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto observer = std::make_unique<Observer>();
    auto *observed = observer.get();
    group->add(std::move(observer));
    cu->set_plugin_group(group);
    group->onInit();
    auto decoder = Decoder::create(arch);
    for (unsigned scalar_mode : {0u, 1u, 2u}) {
      for (int offset : {0, 4, -4}) {
        for (uint64_t exec : {uint64_t{0}, uint64_t{1}, uint64_t{0b1010}, uint64_t{0xf}}) {
          SCOPED_TRACE(scalar_mode);
          SCOPED_TRACE(offset);
          SCOPED_TRACE(exec);
          wf->set_exec(exec);
          const uint8_t saddr = scalar_mode == 0   ? (null127 ? 127 : 124)
                                : scalar_mode == 1 ? 0
                                                   : 96;
          RegisterAccess(*wf).write_sgpr64(wf->sgpr_alloc().base, shared);
          const uint64_t addresses[] = {
              shared + 0x100, shared + (uint64_t{1} << 31) + 0x100, uint64_t{8} << 32,
              cdna5 ? 0x400080 | (uint64_t{3} << 52) : private_base + 0x80};
          for (unsigned lane = 0; lane < 4; ++lane) {
            const uint64_t value = addresses[lane] - (scalar_mode && !legacy_cdna ? shared : 0);
            const unsigned base = wf->vgpr_alloc().base + (cdna5 ? 256 : 0);
            cu->write_vgpr(base, lane, static_cast<uint32_t>(value));
            cu->write_vgpr(base + 1, lane, value >> 32);
          }
          std::vector<uint32_t> words;
          if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
            append_instruction(words,
                               cdna3::build_flat(cdna3::kFlatLoadDwordFlat,
                                                 {.offset = static_cast<uint16_t>(offset & 0xfff),
                                                  .addr = 0,
                                                  .saddr = saddr,
                                                  .vdst = 2}));
          else if (arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2)
            append_instruction(words,
                               rdna1::build_flat(rdna1::kFlatLoadDwordFlat,
                                                 {.offset = static_cast<uint16_t>(offset & 0xfff),
                                                  .addr = 0,
                                                  .saddr = saddr,
                                                  .vdst = 2}));
          else if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
            append_instruction(words,
                               rdna3::build_flat(rdna3::kFlatLoadB32Flat,
                                                 {.offset = static_cast<uint16_t>(offset & 0x1fff),
                                                  .addr = 0,
                                                  .saddr = saddr,
                                                  .vdst = 2}));
          else if (arch == ROCJITSU_CODE_ARCH_RDNA4)
            append_instruction(
                words, rdna4::build_vflat(rdna4::kFlatLoadB32Vflat,
                                          {.saddr = saddr,
                                           .vdst = 2,
                                           .vaddr = 0,
                                           .ioffset = static_cast<uint32_t>(offset) & 0xffffff}));
          else
            append_instruction(
                words, cdna5::build_vflat(cdna5::kFlatLoadB32Vflat,
                                          {.saddr = saddr,
                                           .vdst = 2,
                                           .vaddr = 0,
                                           .ioffset = static_cast<uint32_t>(offset) & 0xffffff}));
          util::StringDiagnostic error;
          auto decoded = decoder->decode_window(words, 0, error.emitter());
          ASSERT_TRUE(decoded.succeeded()) << error.message();
          auto &inst = *decoded.value();
          ASSERT_TRUE(inst.is_memory_wait_producer());
          auto &state = wf->ensure_memory_wait_scoreboard();
          state.clear();
          observed->reads = 0;
          const auto planned =
              MemoryWaitScoreboard::flat_lanes(inst, *wf, shared, shared + UINT32_MAX);
          state.check_instruction(inst, *wf);
          cu->track_memory_wait(inst, *wf);
          EXPECT_EQ(observed->reads, 0u);
          EXPECT_EQ(state.outstanding(WaitCounterKind::Ds), planned.shared ? 1u : 0u);
          EXPECT_EQ(state.outstanding(WaitCounterKind::Load),
                    planned.requests & ~planned.shared ? 1u : 0u);
          inst.execute(inst, wf);
          ASSERT_NE(inst.data(), nullptr);
          const auto &data = *inst.data_as<VectorMemState>();
          EXPECT_EQ(planned.requests, data.lane_mask);
          uint64_t shared_lanes = 0;
          for (unsigned lane = 0; lane < wf->wf_size(); ++lane) {
            const uint64_t bit = uint64_t{1} << lane;
            if ((data.lane_mask & bit) &&
                !(data.scratch_swizzle && (data.scratch_lane_mask & bit)) &&
                data.per_lane_addr[lane] >= shared &&
                data.per_lane_addr[lane] <= shared + UINT32_MAX)
              shared_lanes |= bit;
          }
          EXPECT_EQ(planned.shared, shared_lanes);
          std::vector<MemoryWaitScoreboard::Hazard> hazards;
          state.bind(0x200, &hazards, [](void *p, const auto &hazard) {
            static_cast<decltype(hazards) *>(p)->push_back(hazard);
          });
          const RegisterRef destination{RegClass::VGPR, static_cast<uint16_t>(cdna5 ? 514 : 2), 1};
          state.wait(WaitCounterKind::Ds, 0);
          state.access(destination, shared_lanes, 0xf, false);
          EXPECT_TRUE(hazards.empty());
          state.access(destination, planned.requests & ~shared_lanes, 0xf, false);
          EXPECT_EQ(hazards.size(), (planned.requests & ~shared_lanes) ? 1u : 0u);
        }
      }
    }
    group->onShutdown();
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
        set_test_flat_domains(*cu, *wf, shared_lanes);
        for (bool flat : {false, true}) {
          TestMemoryLoad inst(flat ? "flat_load_dword" : "global_load_dword", 2);
          if (flat && wait)
            wf->memory_wait_scoreboard()->wait(WaitCounterKind::Load, 0);
          wf->ensure_memory_wait_scoreboard().check_instruction(inst, *wf);
          cu->track_memory_wait(inst, *wf);
          wf->pc += 8;
        }
        const bool ordered_global = arch == ROCJITSU_CODE_ARCH_RDNA3 && !shared_lanes;
        EXPECT_EQ(cu->memory_wait_diagnostic_count(), wait || ordered_global ? 0u : 1u);
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, LegacyImageOverwritesCheckTheIncomingCompletionClass) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
                    ROCJITSU_CODE_ARCH_RDNA3_5})
    for (const auto &[first, second] : {std::pair{"global_load_dword", "image_sample"},
                                        {"global_load_dword", "image_bvh_intersect_ray"},
                                        {"image_sample", "global_load_dword"},
                                        {"image_sample", "image_bvh_intersect_ray"},
                                        {"image_sample", "image_sample"},
                                        {"global_load_dword", "global_load_dword"}})
      for (bool wait : {false, true}) {
        SCOPED_TRACE(static_cast<unsigned>(arch));
        SCOPED_TRACE(first);
        SCOPED_TRACE(second);
        SCOPED_TRACE(wait);
        GpuMemory memory("image_overwrite_memory");
        L2Cache l2("image_overwrite_l2");
        ComputeUnitCore::Config config{};
        config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
        config.arch = arch;
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 128;
        config.vgprs_per_wf = 32;
        config.lds_size_kb = 64;
        auto cu = ComputeUnitCore::create("image_overwrite_cu", config, &memory, &l2);
        auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
        ASSERT_NE(wf, nullptr);
        wf->set_exec(1);
        // Exercise the runtime tracking path with decoded destinations; this
        // does not depend on functional support for image sampling or BVH.
        auto track = [&](std::string_view mnemonic, unsigned reg) {
          TestMemoryLoad inst(mnemonic, reg);
          wf->ensure_memory_wait_scoreboard().check_instruction(inst, *wf);
          cu->track_memory_wait(inst, *wf);
          wf->pc += 8;
        };
        // An unrelated load makes the image/image control a mixed counter.
        track("global_load_dword", 3);
        track(first, 2);
        if (wait)
          wf->memory_wait_scoreboard()->wait(WaitCounterKind::Load, 0);
        track(second, 2);
        const bool same_class = std::string_view(first) == second;
        EXPECT_EQ(cu->memory_wait_diagnostic_count(), wait || same_class ? 0u : 1u);
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
            const auto words =
                modern ? cdna5::build_sopk(write ? cdna5::kSSetregB32Sopk : cdna5::kSGetregB32Sopk,
                                           {.simm16 = field, .sdst = 4})
                       : cdna3::build_sopk(write ? cdna3::kSSetregB32Sopk : cdna3::kSGetregB32Sopk,
                                           {.simm16 = field, .sdst = 4});
            auto decoder = Decoder::create(arch);
            util::StringDiagnostic error;
            auto decoded = decoder->decode_window(words, 0, error.emitter());
            ASSERT_TRUE(decoded.succeeded()) << error.message();
            state.check_instruction(*decoded.value(), *wf);
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
      wf->set_exec(lanes);
      const auto words =
          cdna3::build_vop2(cdna3::kVCndmaskB32Vop2, {.src0 = 128, .vsrc1 = 0, .vdst = 2});
      auto decoder = Decoder::create(config.arch);
      util::StringDiagnostic error;
      auto decoded = decoder->decode_window(words, 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      state.check_instruction(*decoded.value(), *wf);
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
        auto &inst = *decoded.value();
        state.check_instruction(inst, *wf);
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
  class AddressObserver final : public ExecutionPlugin {
  public:
    AddressObserver() : ExecutionPlugin("scratch_planning_observer") {}
    unsigned reads = 0;
    void onAmdgpuReadVgprLanes(const Wavefront *, uint32_t, uint64_t, uint8_t) override { ++reads; }
  };
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
        auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
        auto observer = std::make_unique<AddressObserver>();
        auto *observed = observer.get();
        group->add(std::move(observer));
        cu->set_plugin_group(group);
        group->onInit();
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
          const auto words =
              cdna3::build_flat(cdna3::kFlatLoadDwordFlat, {.lds = 1,
                                                            .seg = static_cast<uint8_t>(segment),
                                                            .addr = 0,
                                                            .saddr = 0x7f,
                                                            .vdst = 2});
          auto decoder = Decoder::create(config.arch);
          util::StringDiagnostic error;
          auto decoded = decoder->decode_window(words, 0, error.emitter());
          ASSERT_TRUE(decoded.succeeded()) << error.message();
          state.check_instruction(*decoded.value(), *wf);
          EXPECT_EQ(observed->reads, 0u);
          addr_calc::flat_calculate_addresses(fields, *wf, data);
          EXPECT_EQ(observed->reads != 0, active);
        }
        group->onShutdown();
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
        auto &inst = *decoded.value();
        state.check_instruction(inst, *wf);
        inst.execute(inst, wf);
      }
      EXPECT_EQ(hazards.size(), pending_reg == 6 ? 1u : 0u);
      EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base + 2, 0), 17u);
    }
  }
}

TEST(MemoryWaitExecutionTest, VopdCndmaskChecksPendingVccOnlyForActiveLanes) {
  // v_dual_cndmask_b32 v0, v4, v6 :: v_dual_mov_b32 v1, v5
  const uint32_t words[] = {0xca500d04u, 0x00000105u};
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4,
                    ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool active : {false, true}) {
      for (bool waited : {false, true}) {
        SCOPED_TRACE(static_cast<unsigned>(arch));
        SCOPED_TRACE(active);
        SCOPED_TRACE(waited);
        GpuMemory memory("vopd_vcc_memory");
        L2Cache l2("vopd_vcc_l2");
        ComputeUnitCore::Config config{};
        config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
        config.arch = arch;
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 128;
        config.vgprs_per_wf = 32;
        config.lds_size_kb = 64;
        auto cu = ComputeUnitCore::create("vopd_vcc_cu", config, &memory, &l2);
        auto *wf = cu->dispatch_wf(0, 0x200, 128, 32, 32);
        ASSERT_NE(wf, nullptr);
        wf->set_exec(active ? 1 : 0);
        wf->set_vcc(1);
        cu->write_vgpr(wf->vgpr_alloc().base + 6, 0, 17);
        auto &state = wf->ensure_memory_wait_scoreboard();
        std::vector<MemoryWaitScoreboard::Hazard> hazards;
        state.bind(wf->pc, &hazards, [](void *p, const auto &hazard) {
          static_cast<decltype(hazards) *>(p)->push_back(hazard);
        });
        state.add({state.issue(WaitCounterKind::Ds, true),
                   0x100,
                   ~uint64_t{0},
                   {RegClass::VCC, 0, 1},
                   WaitCounterKind::Ds,
                   0xf});
        if (waited)
          state.wait(WaitCounterKind::Ds, 0);
        auto decoder = Decoder::create(arch);
        util::StringDiagnostic error;
        auto decoded = decoder->decode_window(words, 0, error.emitter());
        ASSERT_TRUE(decoded.succeeded()) << error.message();
        {
          auto &inst = *decoded.value();
          state.check_instruction(inst, *wf);
          inst.execute(inst, wf);
        }
        EXPECT_EQ(hazards.size(), active && !waited ? 1u : 0u);
        if (active) {
          EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, 0), 17u);
        }
      }
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
        state.check_instruction(inst, *wf);
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

TEST_F(MemoryWaitScoreboardTest, UnmappedReplayEntriesRetainTheirPlaceInTheTranslationQueue) {
  for (bool wait_for_completion : {false, true}) {
    state.clear();
    state.issue(WaitCounterKind::Load);
    state.add({state.issue_xcnt(WaitCounterKind::Load, false),
               0x100,
               1,
               {RegClass::VGPR, 5, 1},
               WaitCounterKind::X,
               0xf});
    state.add({state.issue_xcnt(std::nullopt, false),
               0x104,
               1,
               {RegClass::EXEC, 0, 1},
               WaitCounterKind::X,
               0xf});
    state.wait(WaitCounterKind::Store, 0);
    EXPECT_TRUE(shadow.test(5, true));
    EXPECT_TRUE(shadow.pending({RegClass::EXEC, 0, 1}, true));
    state.wait(wait_for_completion ? WaitCounterKind::Load : WaitCounterKind::X,
               wait_for_completion ? 0 : 1);
    EXPECT_FALSE(shadow.test(5, true));
    EXPECT_TRUE(shadow.pending({RegClass::EXEC, 0, 1}, true));
    EXPECT_EQ(state.outstanding(WaitCounterKind::X), 1u);
    // A later mapped instruction's completion proves the remaining X prefix.
    state.issue(WaitCounterKind::Store);
    state.issue_xcnt(WaitCounterKind::Store, false);
    state.wait(WaitCounterKind::Store, 0);
    EXPECT_FALSE(shadow.pending({RegClass::EXEC, 0, 1}, true));
    EXPECT_EQ(state.outstanding(WaitCounterKind::X), 0u);
  }
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
                                        std::string_view xcnt_setting = "warn", unsigned vgprs = 32,
                                        std::string_view memory_setting = "warn") {
  using namespace rocr::llvm::amdhsa;
  append_instruction(code, S_WAIT_KMCNT_0_GFX12);
  append_instruction(code, cdna5::build_sopp(cdna5::kSWaitLoadcntSopp, {.simm16 = 0}));
  append_instruction(code, cdna5::build_sopp(cdna5::kSWaitStorecntSopp, {.simm16 = 0}));
  append_instruction(code, S_ENDPGM_GFX12);
  Gfx1250Sim sim(memory_wait_test_config(memory_setting, xcnt_setting));
  write_global_u32(*sim.memory, 0x400000, 0x12345678);
  uint32_t properties = 0;
  AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, vgprs, 2, false, false,
                                 false, properties, 16);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel, 32, 32, 0x400000);
  step_until_halted(*sim.engine, *sim.cu());
  EXPECT_EQ(sim.snapshot->snapshots().size(), 1u);
  if (memory_setting != "warn" && xcnt_setting != "warn") {
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

TEST(XcntExecutionTest, CompletionAndReplaySettingsAreIndependent) {
  for (std::string_view memory : {"", "warn", "off"})
    for (std::string_view xcnt : {"", "warn", "off"}) {
      SCOPED_TRACE(std::format("memory={} xcnt={}", memory, xcnt));
      std::vector<uint32_t> code;
      append_instruction(code, make_s_load_b32_scaled_imm(4, 0, 0));
      append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 4, .sdst = 5}));
      append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 0}));
      const auto counts = run_xcnt_kernel(code, xcnt, 32, memory);
      EXPECT_EQ(counts[0], xcnt == "warn" ? 1u : 0u);
      EXPECT_EQ(counts[1], memory == "warn" ? 1u : 0u);
    }
}

TEST(XcntExecutionTest, ReplayOnlyCheckingRetainsCompletionOrdering) {
  for (unsigned wait = 0; wait < 4; ++wait) {
    SCOPED_TRACE(wait);
    std::vector<uint32_t> code;
    enable_multi_group_replay(code);
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
    append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                                  {.saddr = 0, .vdst = 2, .vaddr = 0}));
    if (wait)
      append_instruction(code, cdna5::build_sopp(wait == 1   ? cdna5::kSWaitXcntSopp
                                                 : wait == 2 ? cdna5::kSWaitLoadcntSopp
                                                             : cdna5::kSWaitStorecntSopp,
                                                 {.simm16 = 0}));
    append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 129, .vdst = 0}));
    EXPECT_EQ(run_xcnt_kernel(code, "warn", 32, "off"),
              (std::array<uint64_t, 2>{wait == 1 || wait == 2 ? 0u : 1u, 0}));
  }
}

TEST(XcntExecutionTest, InvalidDiagnosticSettingsAreRejected) {
  EXPECT_THROW(Gfx1250Sim(memory_wait_test_config("no")), std::invalid_argument);
  EXPECT_THROW(Gfx1250Sim(memory_wait_test_config("", "no")), std::invalid_argument);
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

TEST(XcntExecutionTest, EmptyExecStoreCannotUseAnOlderCompletionToDrainReplaySources) {
  for (bool previous_store : {false, true})
    for (unsigned wait = 0; wait < 3; ++wait) {
      SCOPED_TRACE(previous_store);
      SCOPED_TRACE(wait);
      std::vector<uint32_t> code;
      enable_multi_group_replay(code);
      append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
      if (previous_store) {
        append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalStoreB32Vglobal,
                                                      {.saddr = 0, .vsrc = 0, .vaddr = 0}));
        append_instruction(code, cdna5::build_sopp(cdna5::kSWaitStorecntSopp, {.simm16 = 0}));
      }
      append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                                    {.saddr = 0, .vdst = 2, .vaddr = 0}));
      // This first overwrite diagnoses EXEC, leaving the address sources pending.
      append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 126}));
      append_instruction(code, cdna5::build_vglobal(cdna5::kGlobalStoreB32Vglobal,
                                                    {.saddr = 0, .vsrc = 0, .vaddr = 0}));
      append_instruction(code, cdna5::build_sopp(cdna5::kSWaitStorecntSopp, {.simm16 = 0}));
      if (wait)
        append_instruction(
            code, cdna5::build_sopp(wait == 1 ? cdna5::kSWaitXcntSopp : cdna5::kSWaitLoadcntSopp,
                                    {.simm16 = 0}));
      // Scalar sources can be overwritten even with EXEC zero. Only a real load
      // completion or an X wait proves that this older address is safe to change.
      append_instruction(code, cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 0}));
      EXPECT_EQ(run_xcnt_kernel(code), (std::array<uint64_t, 2>{wait ? 1u : 2u, 0}));
    }
}

// The observer is an independent test oracle. Production checking stays in core
// and computes the footprint before executing the instruction.
class MemoryWaitFootprintObserver final : public ExecutionPlugin {
public:
  MemoryWaitFootprintObserver() : ExecutionPlugin("wait_footprint_test") {}
  struct Access {
    std::array<uint64_t, 4> read{};
    std::array<uint64_t, 4> write{};
  };
  std::array<Access, 1024> accesses{};
  unsigned callbacks = 0;
  void onAmdgpuReadVgprLanes(const Wavefront *wf, uint32_t reg, uint64_t lanes,
                             uint8_t bytes) override {
    record(wf, reg, lanes, bytes, false);
  }
  void onAmdgpuWriteVgprLanes(const Wavefront *wf, uint32_t reg, uint64_t lanes,
                              uint8_t bytes) override {
    record(wf, reg, lanes, bytes, true);
  }
  void record(const Wavefront *wf, uint32_t reg, uint64_t lanes, uint8_t bytes, bool write) {
    if (!wf || reg < wf->vgpr_alloc().base)
      return;
    reg -= wf->vgpr_alloc().base;
    ASSERT_LT(reg, accesses.size());
    ++callbacks;
    auto &masks = write ? accesses[reg].write : accesses[reg].read;
    for (unsigned byte = 0; byte < 4; ++byte)
      if (bytes & (1u << byte))
        masks[byte] |= lanes;
  }
};

void check_vector_footprints(rj_code_arch_t arch, unsigned wave_size, unsigned num_vgprs,
                             const std::vector<std::vector<uint32_t>> &cases, uint32_t m0 = 0,
                             uint8_t msb = 0,
                             const std::function<void(Wavefront &)> &prepare = {}) {
  GpuMemory memory("footprint_memory");
  L2Cache l2("footprint_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = arch;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = num_vgprs;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("footprint_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 104, num_vgprs, wave_size);
  ASSERT_NE(wf, nullptr);
  wf->set_m0(m0);
  wf->set_vgpr_msb_mode(msb);
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<MemoryWaitFootprintObserver>();
  auto *observed = observer.get();
  group->add(std::move(observer));
  cu->set_plugin_group(group);
  group->onInit();
  auto decoder = Decoder::create(config.arch);
  for (const auto &words : cases) {
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    auto &inst = *decoded.value();
    SCOPED_TRACE(inst.mnemonic());
    const uint64_t all = wave_size == 64 ? ~uint64_t{0} : uint64_t{0xffffffff};
    for (uint64_t exec : {all, uint64_t{1}, uint64_t{0xa5a5f0f012348001} & all, uint64_t{0}}) {
      SCOPED_TRACE(exec);
      wf->set_exec_raw(exec);
      if (prepare)
        prepare(*wf);
      observed->accesses = {};
      ASSERT_TRUE(cu->execute_instruction(&inst, *wf).succeeded());
      const auto expected = observed->accesses;
      if (prepare)
        prepare(*wf);
      const auto callbacks_before = observed->callbacks;
      // These operations do not change EXEC or their lane-select operand.
      ASSERT_EQ(wf->exec_raw(), exec);
      auto &state = wf->ensure_memory_wait_scoreboard();
      unsigned reports = 0;
      state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
      unsigned mismatches = 0;
      for (unsigned reg = 0; reg < num_vgprs; ++reg) {
        for (unsigned lane = 0; lane < wave_size; ++lane)
          for (unsigned byte = 0; byte < 4; ++byte)
            for (bool replay : {false, true}) {
              state.clear();
              reports = 0;
              const auto counter = replay ? WaitCounterKind::X : WaitCounterKind::Load;
              state.add({state.issue(counter),
                         0x100,
                         uint64_t{1} << lane,
                         {RegClass::VGPR, static_cast<uint16_t>(reg), 1},
                         counter,
                         static_cast<uint8_t>(1u << byte)});
              state.check_instruction(inst, *wf);
              const auto lanes =
                  expected[reg].write[byte] | (replay ? 0 : expected[reg].read[byte]);
              const bool hazard = lanes & (uint64_t{1} << lane);
              if ((reports != 0) != hazard) {
                ADD_FAILURE() << "v" << reg << " lane=" << lane << " byte=" << byte
                              << " replay=" << replay << " expected=" << hazard
                              << " reported=" << reports;
                if (++mismatches == 8)
                  goto next_mask;
              }
            }
      }
    next_mask:
      state.clear();
      EXPECT_EQ(observed->callbacks, callbacks_before);
    }
  }
  group->onShutdown();
}

TEST(MemoryWaitFootprintTest, NarrowDsAndBufferStoresCheckOnlyConsumedBytes) {
  std::vector<std::vector<uint32_t>> cases;
  for (auto opcode : {rdna3::kDsStoreB8Ds, rdna3::kDsStoreB16Ds, rdna3::kDsStoreB8D16HiDs,
                      rdna3::kDsStoreB16D16HiDs}) {
    const auto words = rdna3::build_ds(opcode, {.addr = 0, .data0 = 4});
    cases.emplace_back(words.begin(), words.end());
  }
  for (auto opcode : {rdna3::kBufferStoreB8Mubuf, rdna3::kBufferStoreB16Mubuf,
                      rdna3::kBufferStoreD16HiB8Mubuf, rdna3::kBufferStoreD16HiB16Mubuf}) {
    const auto words = rdna3::build_mubuf(opcode, {.vaddr = 0, .vdata = 4, .srsrc = 0, .offen = 1});
    cases.emplace_back(words.begin(), words.end());
  }
  check_vector_footprints(ROCJITSU_CODE_ARCH_RDNA3, 32, 16, cases);
}

TEST(MemoryWaitFootprintTest, Cdna4HazardsMatchObservedVectorAccesses) {
  std::vector<std::vector<uint32_t>> cases;
  auto add = [&](auto words) { cases.emplace_back(words.begin(), words.end()); };
  add(cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 256, .vdst = 16}));
  add(cdna4::build_vop2(cdna4::kVFmacF32Vop2, {.src0 = 256, .vsrc1 = 4, .vdst = 16}));
  add(cdna4::build_vop3(cdna4::kVAddU32Vop3, {.vdst = 16, .src0 = 256, .src1 = 260}));
  add(cdna4::build_vop3p_mfma(cdna4::kVMfmaF3216x16x32F16Vop3pMfma,
                              {.vdst = 16, .src0 = 256, .src1 = 260, .src2 = 264}));
  add(cdna4::build_vop3p_mfma(
      cdna4::kVMfmaF3216x16x32F16Vop3pMfma,
      {.vdst = 16, .acc_cd = 1, .src0 = 256, .src1 = 260, .src2 = 264, .acc = 3}));
  add(cdna4::build_vop3(cdna4::kVReadlaneB32Vop3, {.vdst = 16, .src0 = 256, .src1 = 135}));
  add(cdna4::build_vop3(cdna4::kVWritelaneB32Vop3, {.vdst = 16, .src0 = 128, .src1 = 135}));
  add(cdna4::build_vop1(cdna4::kVReadfirstlaneB32Vop1, {.src0 = 256, .vdst = 16}));
  add(cdna4::build_vop3p(cdna4::kVMadMixF32Vop3p,
                         {.vdst = 16, .src0 = 256, .src1 = 260, .src2 = 264}));
  check_vector_footprints(ROCJITSU_CODE_ARCH_CDNA4, 64, 512, cases);
}

TEST(MemoryWaitFootprintTest, DualFmacHazardsMatchObservedVectorAccesses) {
  std::vector<std::vector<uint32_t>> cases;
  // VOPD X is FMAC. Exercise Y's FMAC, MOV, integer ADD and shift slots,
  // including the destination reads tied to the FMAC slots.
  for (unsigned opy : {0u, 8u, 16u, 17u})
    cases.push_back({(0x32u << 26) | (opy << 17) | (4u << 9) | 256u,
                     (16u << 24) | (8u << 17) | (12u << 9) | 264u});
  check_vector_footprints(ROCJITSU_CODE_ARCH_RDNA3, 32, 256, cases);
  check_vector_footprints(ROCJITSU_CODE_ARCH_CDNA5, 32, 512, cases);
}

TEST(MemoryWaitFootprintTest, ValuPermutationsCheckConsumedLanesBeforeExecution) {
  struct Case {
    std::vector<uint32_t> words;
    uint64_t exec;
    std::array<uint64_t, 3> reads; // v1, v2, v4
    std::array<uint64_t, 3> writes;
    rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5;
  };
  std::vector<Case> cases;
  auto add = [&](auto words, uint64_t exec, std::array<uint64_t, 3> reads,
                 std::array<uint64_t, 3> writes) {
    cases.push_back({{words.begin(), words.end()}, exec, reads, writes});
  };
  add(rdna3::build_vop1(rdna3::kVPermlane64B32Vop1, {.src0 = 258, .vdst = 1}), 1,
      {0, uint64_t{1} << 32, 0}, {1, 0, 0});
  cases.back().arch = ROCJITSU_CODE_ARCH_RDNA3;
  for (unsigned control : {0u, 1u, 2u, 3u}) {
    for (bool cross : {false, true}) {
      const auto source = uint64_t{1} << (cross ? 21 : 5);
      const auto reads = control & 1 ? source : 0;
      const auto writes = control ? 1u : 0u;
      add(cdna5::build_vop3(cross ? cdna5::kVPermlanex16B32Vop3 : cdna5::kVPermlane16B32Vop3,
                            {.vdst = 1,
                             .opsel = static_cast<uint8_t>(control),
                             .src0 = 258,
                             .src1 = 4,
                             .src2 = 5}),
          1, {0, reads, 0}, {writes, 0, 0});
      add(cdna5::build_vop3(
              cross ? cdna5::kVPermlanex16VarB32Vop3 : cdna5::kVPermlane16VarB32Vop3,
              {.vdst = 1, .opsel = static_cast<uint8_t>(control), .src0 = 258, .src1 = 260}),
          1, {0, reads, 1}, {writes, 0, 0});
    }
  }
  for (auto opcode :
       {cdna5::kVPermlaneBcastB32Vop3, cdna5::kVPermlaneDownB32Vop3, cdna5::kVPermlaneXorB32Vop3})
    add(cdna5::build_vop3(opcode, {.vdst = 1, .src0 = 258, .src1 = 133, .src2 = 136}), 1,
        {0, 1u << 5, 0}, {1, 0, 0});
  add(cdna5::build_vop3(cdna5::kVPermlaneUpB32Vop3,
                        {.vdst = 1, .src0 = 258, .src1 = 133, .src2 = 136}),
      1u << 7, {0, 1u << 2, 0}, {1u << 7, 0, 0});
  add(cdna5::build_vop3(cdna5::kVPermlaneDownB32Vop3,
                        {.vdst = 1, .src0 = 258, .src1 = 133, .src2 = 136}),
      1u << 7, {0, 0, 0}, {1u << 7, 0, 0});
  add(cdna5::build_vop1(cdna5::kVPermlane16SwapB32Vop1, {.src0 = 258, .vdst = 1}), 1,
      {1u << 16, 0, 0}, {0, 1, 0});
  add(cdna5::build_vop1(cdna5::kVPermlane16SwapB32Vop1, {.src0 = 258, .vdst = 1}), 1u << 16,
      {0, 1, 0}, {1u << 16, 0, 0});

  for (const auto &test : cases) {
    GpuMemory memory("permlane_memory");
    L2Cache l2("permlane_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = test.arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 128;
    config.vgprs_per_wf = 32;
    config.lds_size_kb = 64;
    auto cu = ComputeUnitCore::create("permlane_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 104, 32, test.arch == ROCJITSU_CODE_ARCH_RDNA3 ? 64 : 32);
    ASSERT_NE(wf, nullptr);
    RegisterAccess(*wf).write_sgpr(wf->sgpr_alloc().base + 4, 0x55555555);
    RegisterAccess(*wf).write_sgpr(wf->sgpr_alloc().base + 5, 0x55555555);
    for (unsigned lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(wf->vgpr_alloc().base + 4, lane, 5);
    auto decoder = Decoder::create(config.arch);
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(test.words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    auto &inst = *decoded.value();
    SCOPED_TRACE(inst.mnemonic());
    SCOPED_TRACE(test.exec);
    wf->set_exec(test.exec);
    for (bool replay : {false, true})
      for (unsigned operand = 0; operand < 3; ++operand)
        for (unsigned lane = 0; lane < wf->wf_size(); ++lane) {
          const unsigned reg = operand == 0 ? 1 : operand == 1 ? 2 : 4;
          auto &state = wf->ensure_memory_wait_scoreboard();
          state.clear();
          unsigned reports = 0;
          state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          const auto counter = replay ? WaitCounterKind::X : WaitCounterKind::Ds;
          state.add({state.issue(counter),
                     0x100,
                     uint64_t{1} << lane,
                     {RegClass::VGPR, static_cast<uint16_t>(reg), 1},
                     counter,
                     0xf});
          state.check_instruction(inst, *wf);
          const uint64_t expected = test.writes[operand] | (replay ? 0 : test.reads[operand]);
          EXPECT_EQ(reports != 0, bool(expected & (uint64_t{1} << lane)))
              << "reg=" << reg << " lane=" << lane << " replay=" << replay;
        }
  }
}

TEST(MemoryWaitFootprintTest, MixedPrecisionChecksOnlyConsumedAndWrittenHalves) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    std::vector<std::vector<uint32_t>> cases;
    auto add = [&](auto words) { cases.emplace_back(words.begin(), words.end()); };
    for (uint8_t selection : {0, 1, 7})
      for (uint8_t half_sources : {0, 3})
        for (unsigned output : {0u, 1u, 2u}) {
          if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
            const uint16_t opcodes[] = {cdna4::kVMadMixF32Vop3p, cdna4::kVMadMixloF16Vop3p,
                                        cdna4::kVMadMixhiF16Vop3p};
            add(cdna4::build_vop3p(opcodes[output],
                                   {.vdst = 8,
                                    .op_sel = selection,
                                    .op_sel_hi_2 = static_cast<uint8_t>(half_sources != 0),
                                    .src0 = 256,
                                    .src1 = 258,
                                    .src2 = 260,
                                    .op_sel_hi = half_sources}));
          } else if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
            const uint16_t opcodes[] = {rdna3::kVFmaMixF32Vop3p, rdna3::kVFmaMixloF16Vop3p,
                                        rdna3::kVFmaMixhiF16Vop3p};
            add(rdna3::build_vop3p(opcodes[output],
                                   {.vdst = 8,
                                    .op_sel = selection,
                                    .op_sel_hi_2 = static_cast<uint8_t>(half_sources != 0),
                                    .src0 = 256,
                                    .src1 = 258,
                                    .src2 = 260,
                                    .op_sel_hi = half_sources}));
          } else {
            const uint16_t opcodes[] = {cdna5::kVFmaMixF32Vop3p, cdna5::kVFmaMixloF16Vop3p,
                                        cdna5::kVFmaMixhiF16Vop3p};
            add(cdna5::build_vop3p(opcodes[output],
                                   {.vdst = 8,
                                    .opsel = selection,
                                    .opsel_hi_2 = static_cast<uint8_t>(half_sources != 0),
                                    .src0 = 256,
                                    .src1 = 258,
                                    .src2 = 260,
                                    .opsel_hi = half_sources}));
          }
        }
    check_vector_footprints(arch, arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32, 16, cases);
  }
}

TEST(MemoryWaitFootprintTest, PackedArithmeticUsesSelectedHalvesIncludingFmacDestination) {
  std::vector<std::vector<uint32_t>> cases;
  for (unsigned source_half : {0u, 128u})
    for (unsigned destination_half : {0u, 128u})
      for (auto opcode : {rdna3::kVAddF16Vop2, rdna3::kVMulF16Vop2, rdna3::kVFmacF16Vop2}) {
        const auto words =
            rdna3::build_vop2(opcode, {.src0 = static_cast<uint16_t>(258 + source_half),
                                       .vsrc1 = static_cast<uint8_t>(4 + source_half),
                                       .vdst = static_cast<uint8_t>(8 + destination_half)});
        cases.emplace_back(words.begin(), words.end());
      }
  check_vector_footprints(ROCJITSU_CODE_ARCH_RDNA3, 32, 16, cases);
}

TEST(MemoryWaitFootprintTest, MatrixIndexingLeavesAccumulatorBankUnchanged) {
  std::vector<std::vector<uint32_t>> cases;
  for (bool accumulators : {false, true}) {
    const auto words = cdna3::build_vop3p_mfma(cdna3::kVMfmaF3216x16x8Xf32Vop3pMfma,
                                               {.vdst = 16,
                                                .acc_cd = accumulators,
                                                .src0 = 256,
                                                .src1 = 260,
                                                .src2 = 264,
                                                .acc = static_cast<uint8_t>(accumulators ? 3 : 0)});
    cases.emplace_back(words.begin(), words.end());
  }
  check_vector_footprints(ROCJITSU_CODE_ARCH_CDNA3, 64, 512, cases, 0xf004, 0,
                          [](Wavefront &wf) { wf.set_mode_raw(wf.mode_raw() | (1u << 27)); });
}

TEST(MemoryWaitFootprintTest, True16MovesMatchObservedBytes) {
  std::vector<std::vector<uint32_t>> cases;
  auto add = [&](auto words) { cases.emplace_back(words.begin(), words.end()); };
  for (unsigned src_hi : {0u, 1u})
    for (unsigned dst_hi : {0u, 1u}) {
      add(cdna5::build_vop1(cdna5::kVMovB16Vop1,
                            {.src0 = static_cast<uint16_t>(256 + src_hi * 128),
                             .vdst = static_cast<uint8_t>(16 + dst_hi * 128)}));
      add(cdna5::build_vop3(
          cdna5::kVMovB16Vop3,
          {.vdst = 16, .opsel = static_cast<uint8_t>(src_hi | (dst_hi << 3)), .src0 = 256}));
    }
  check_vector_footprints(ROCJITSU_CODE_ARCH_CDNA5, 32, 512, cases);
}

TEST(MemoryWaitFootprintTest, ScalarAndInlineSourcesAreNotVectorRegisterNumbers) {
  std::vector<std::vector<uint32_t>> cases;
  for (uint16_t selector : {1, 128, 129, 160, 193, 240, 241, 242, 243}) {
    const auto words =
        cdna5::build_vop2(cdna5::kVAndB32Vop2, {.src0 = selector, .vsrc1 = 4, .vdst = 8});
    cases.emplace_back(words.begin(), words.end());
  }
  check_vector_footprints(ROCJITSU_CODE_ARCH_CDNA5, 32, 512, cases);
}

TEST(MemoryWaitFootprintTest, DppUsesPhysicalSourceLanesAndMaskedDestinations) {
  for (const auto arch :
       {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    std::vector<std::vector<uint32_t>> cases;
    for (uint32_t control : {0u, 0xb1u, 0x111u, 0x142u})
      for (bool bound : {false, true})
        for (uint32_t destination : {2u, 5u}) {
          const uint32_t extension =
              2u | (control << 8) | (uint32_t{bound} << 19) | (5u << 24) | (1u << 28);
          cases.push_back({(0x3fu << 25) | (destination << 17) | (1u << 9) | 250u, extension});
        }
    // gfx9 row-broadcast encodings are not accepted on newer targets.
    if (arch != ROCJITSU_CODE_ARCH_CDNA4)
      cases.resize(12);
    check_vector_footprints(arch, arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32, 256, cases);
  }
}

TEST(MemoryWaitFootprintTest, Dpp8InactiveFetchDoesNotUseDestinationExecAsSourceMask) {
  std::vector<std::vector<uint32_t>> cases;
  constexpr uint32_t selection =
      (2u << 6) | (3u << 9) | (4u << 12) | (5u << 15) | (6u << 18) | (7u << 21);
  for (uint32_t marker : {233u, 234u})
    for (uint32_t destination : {2u, 5u})
      cases.push_back(
          {(0x3fu << 25) | (destination << 17) | (1u << 9) | marker, 2u | (selection << 8)});
  check_vector_footprints(ROCJITSU_CODE_ARCH_CDNA5, 32, 256, cases);
}

TEST(MemoryWaitFootprintTest, SdwaChecksSelectedBytesWithoutPreservationReads) {
  std::vector<std::vector<uint32_t>> cases;
  for (uint32_t source_selection = 0; source_selection < 7; ++source_selection)
    for (uint32_t destination_selection = 0; destination_selection < 7; ++destination_selection) {
      const uint32_t extension =
          2u | (destination_selection << 8) | (2u << 11) | (source_selection << 16);
      cases.push_back({(0x3fu << 25) | (5u << 17) | (1u << 9) | 249u, extension});
    }
  check_vector_footprints(ROCJITSU_CODE_ARCH_CDNA4, 64, 256, cases);
}

TEST(MemoryWaitFootprintTest, RelativeMovesAndSwapsUseM0AfterBankSelection) {
  std::vector<std::vector<uint32_t>> cases;
  for (uint16_t opcode :
       {cdna5::kVMovrelsB32Vop1, cdna5::kVMovreldB32Vop1, cdna5::kVMovrelsdB32Vop1,
        cdna5::kVMovrelsd2B32Vop1, cdna5::kVSwaprelB32Vop1}) {
    const auto words = cdna5::build_vop1(opcode, {.src0 = 257, .vdst = 10});
    cases.emplace_back(words.begin(), words.end());
  }
  for (uint32_t m0 : {3u, (4u << 16) | 2u, 1023u, 1024u}) {
    check_vector_footprints(ROCJITSU_CODE_ARCH_CDNA5, 32, 1024, cases, m0);
    check_vector_footprints(ROCJITSU_CODE_ARCH_CDNA5, 32, 1024, cases, m0, 0x81);
  }
}

} // namespace

TEST(MemoryWaitFootprintTest, WaveMaskDestinationUsesWaveWidth) {
  for (unsigned wave_size : {32u, 64u}) {
    GpuMemory memory("mask_wait_memory");
    L2Cache l2("mask_wait_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = ROCJITSU_CODE_ARCH_RDNA3;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 128;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    auto cu = ComputeUnitCore::create("mask_wait_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 104, 256, wave_size);
    ASSERT_NE(wf, nullptr);
    const auto words =
        rdna3::build_vop3(rdna3::kVCmpGtI32Vop3, {.vdst = 1, .src0 = 266, .src1 = 292});
    auto decoder = Decoder::create(config.arch);
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    auto &state = wf->ensure_memory_wait_scoreboard();
    std::vector<MemoryWaitScoreboard::Hazard> hazards;
    state.bind(0x200, &hazards, [](void *p, const auto &hazard) {
      static_cast<decltype(hazards) *>(p)->push_back(hazard);
    });
    for (unsigned pending : {0u, 1u, 2u, 3u}) {
      state.clear();
      hazards.clear();
      state.add({state.issue(WaitCounterKind::Km),
                 0x100,
                 ~uint64_t{0},
                 {RegClass::SGPR, static_cast<uint16_t>(pending), 1},
                 WaitCounterKind::Km,
                 0xf});
      state.check_instruction(*decoded.value(), *wf);
      EXPECT_EQ(hazards.size(), pending >= 1 && pending < 1 + wave_size / 32 ? 1u : 0u)
          << wave_size << " " << pending;
    }
  }
}

TEST(MemoryWaitFootprintTest, WaitImmediateIsNotAScalarRegister) {
  GpuMemory memory("immediate_wait_memory");
  L2Cache l2("immediate_wait_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("immediate_wait_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 104, 256, 32);
  ASSERT_NE(wf, nullptr);
  auto decoder = Decoder::create(config.arch);
  util::StringDiagnostic error;
  for (uint16_t value : {0u, 1u, 31u, 63u}) {
    const auto words = cdna5::build_sopp(cdna5::kSWaitXcntSopp, {.simm16 = value});
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    auto &state = wf->ensure_memory_wait_scoreboard();
    unsigned hazards = 0;
    state.bind(0x200, &hazards, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    state.clear();
    state.add({state.issue(WaitCounterKind::Km),
               0x100,
               ~uint64_t{0},
               {RegClass::SGPR, value, 1},
               WaitCounterKind::Km,
               0xf});
    state.check_instruction(*decoded.value(), *wf);
    EXPECT_EQ(hazards, 0u) << value;
  }
}

TEST(MemoryWaitShadowTest, RangeQueriesMatchIndividualBitsAtWordAndBankBoundaries) {
  MemoryWaitShadow shadow;
  for (auto cls : {RegClass::VGPR, RegClass::SGPR, RegClass::TTMP, RegClass::EXEC}) {
    for (unsigned pending : {0u, 31u, 63u, 64u, 127u, 255u, 256u, 511u, 512u, 1023u}) {
      const auto bit = MemoryWaitShadow::index({cls, static_cast<uint16_t>(pending), 1});
      if (bit == MemoryWaitShadow::kRegisters)
        continue;
      for (uint8_t kind : {MemoryWaitShadow::kResult, MemoryWaitShadow::kReplaySource}) {
        shadow.reset();
        shadow.set(bit, kind);
        for (unsigned start :
             {0u,   1u,   31u,  32u,  62u,  63u,  64u,  65u,  126u,  127u,  128u,
              254u, 255u, 256u, 257u, 510u, 511u, 512u, 513u, 1022u, 1023u, 1024u}) {
          for (uint8_t width : {0, 1, 2, 3, 4, 16, 32, 63, 64, 65, 127, 128, 255}) {
            for (bool write : {false, true}) {
              bool expected = false;
              for (unsigned i = 0; i < width; ++i)
                expected |= shadow.test(
                    MemoryWaitShadow::index({cls, static_cast<uint16_t>(start + i), 1}), write);
              EXPECT_EQ(shadow.pending({cls, static_cast<uint16_t>(start), width}, write), expected)
                  << static_cast<unsigned>(cls) << " pending=" << pending << " start=" << start
                  << " width=" << unsigned(width) << " write=" << write;
              if (write && cls == RegClass::VGPR && start + width <= REGISTER_SET_MAX_VGPRS) {
                EXPECT_EQ(shadow.pending_vgpr(static_cast<uint16_t>(start), width), expected);
              }
            }
          }
        }
        shadow.clear(bit);
        EXPECT_FALSE(shadow.test(bit, true));
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, WaterfallKeepsPendingResultsInTheirIssuingLanesAndBank) {
  using namespace rocr::llvm::amdhsa;
  for (bool rmw : {false, true})
    for (unsigned bank : {0u, 1u, 3u})
      for (int wait : {-1, 0, 1})
        for (uint32_t consume_mask : {0x55555555u, 0xaaaaaaaau, 0xffffffffu}) {
          SCOPED_TRACE(rmw);
          SCOPED_TRACE(bank);
          SCOPED_TRACE(wait);
          SCOPED_TRACE(consume_mask);
          std::vector<uint32_t> code;
          auto emit = [&](auto words) { append_instruction(code, words); };
          auto bank_mode = [&](unsigned mode) {
            emit(cdna5::build_sopp(cdna5::kSSetVgprMsbSopp,
                                   {.simm16 = static_cast<uint16_t>(mode)}));
          };
          // Two divergent resource IDs, one for even lanes and one for odd lanes.
          emit(
              cdna5::build_vop3(cdna5::kVMbcntLoU32B32Vop3, {.vdst = 1, .src0 = 193, .src1 = 128}));
          emit(cdna5::build_vop2(cdna5::kVAndB32Vop2, {.src0 = 129, .vsrc1 = 1, .vdst = 1}));
          emit(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
          emit(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 193, .sdst = 8}));
          const size_t loop = code.size();
          bank_mode(0);
          emit(cdna5::build_vop1(cdna5::kVReadfirstlaneB32Vop1, {.src0 = 257, .vdst = 4}));
          emit(cdna5::build_vop3(cdna5::kVCmpEqU32Vop3, {.vdst = 6, .src0 = 4, .src1 = 257}));
          emit(cdna5::build_sop1(cdna5::kSAndSaveExecB32Sop1, {.ssrc0 = 6, .sdst = 10}));
          bank_mode(bank << 6);
          emit(cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                    {.saddr = 0, .vdst = 2, .vaddr = 0}));
          emit(cdna5::build_sop2(cdna5::kSAndNot1B32Sop2, {.ssrc0 = 8, .ssrc1 = 126, .sdst = 8}));
          emit(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 8, .sdst = 126}));
          bank_mode(rmw ? bank << 6 : bank);
          // The remaining lanes have not issued this load yet. Reading the same
          // physical VGPR there must not diagnose either waterfall iteration.
          if (rmw)
            emit(cdna5::build_vop2(cdna5::kVFmacF32Vop2, {.src0 = 128, .vsrc1 = 0, .vdst = 2}));
          else
            emit(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 258, .vdst = 3}));
          const auto displacement = static_cast<int16_t>(static_cast<int64_t>(loop) -
                                                         static_cast<int64_t>(code.size()) - 1);
          emit(cdna5::build_sopp(cdna5::kSCbranchExecnzSopp,
                                 {.simm16 = static_cast<uint16_t>(displacement)}));
          if (wait >= 0)
            emit(cdna5::build_sopp(cdna5::kSWaitLoadcntSopp,
                                   {.simm16 = static_cast<uint16_t>(wait)}));
          emit(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 255, .sdst = 126}));
          code.push_back(consume_mask);
          if (rmw)
            emit(cdna5::build_vop2(cdna5::kVFmacF32Vop2, {.src0 = 128, .vsrc1 = 0, .vdst = 2}));
          else
            emit(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 258, .vdst = 3}));
          emit(S_ENDPGM_GFX12);
          uint32_t properties = 0;
          AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
          Gfx1250Sim sim(memory_wait_test_config());
          write_global_u32(*sim.memory, 0x400000, 0x12345678);
          const auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 1024, 2,
                                               false, false, false, properties, 16);
          test::AqlQueue queue(sim.memory, sim.cp());
          queue.dispatch(kernel, 32, 32, 0x400000);
          step_until_halted(*sim.engine, *sim.cu());
          ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
          const auto &snapshot = sim.snapshot->snapshots().front();
          for (unsigned lane = 0; lane < 32; ++lane) {
            EXPECT_EQ(snapshot.vgpr(bank * 256 + 2, lane), 0x12345678u);
            if (!rmw && (consume_mask & (uint32_t{1} << lane))) {
              EXPECT_EQ(snapshot.vgpr(3, lane), 0x12345678u);
            }
          }
          const unsigned expected = wait == 0
                                        ? 0
                                        : unsigned(wait == -1 && (consume_mask & 0x55555555u)) +
                                              unsigned(consume_mask & 0xaaaaaaaau ? 1 : 0);
          EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), expected);
        }
}

TEST_F(MemoryWaitScoreboardTest, PendingBitsMatchLiveEventsThroughMixedRetirement) {
  uint32_t random_state = 0x1250;
  auto random = [&] {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
  };
  auto validate = [&] {
    std::array<uint8_t, MemoryWaitShadow::kRegisters> expected{};
    for (const auto &event : state.events())
      for (unsigned offset = 0; offset < event.reg.width; ++offset) {
        auto reg = event.reg;
        reg.index += offset;
        const auto slot = MemoryWaitShadow::index(reg);
        if (slot < expected.size())
          expected[slot] |= event.counter == WaitCounterKind::X ? 2 : 1;
      }
    for (unsigned slot = 0; slot < expected.size(); ++slot) {
      ASSERT_EQ(shadow.test(slot), bool(expected[slot] & 1)) << slot;
      ASSERT_EQ(shadow.test(slot, true), bool(expected[slot])) << slot;
    }
  };
  constexpr WaitCounterKind counters[] = {WaitCounterKind::Load, WaitCounterKind::Ds,
                                          WaitCounterKind::Km, WaitCounterKind::X};
  for (unsigned step = 0; step < 4096; ++step) {
    SCOPED_TRACE(step);
    const unsigned choice = random() >> 24;
    const auto counter = counters[(random() >> 24) & 3];
    const bool scalar = ((random() >> 24) & 3) == 0;
    const RegisterRef reg{scalar ? RegClass::SGPR : RegClass::VGPR,
                          static_cast<uint16_t>((random() >> 24) & 15),
                          static_cast<uint8_t>(1 + ((random() >> 24) & 3))};
    const uint64_t lanes = uint64_t{1} << ((random() >> 24) & 3);
    const uint8_t bytes = 1u << ((random() >> 24) & 3);
    if (step % 64 == 0) {
      // Exercise reuse after both full retirement and explicit wave reset.
      if (step & 64)
        state.clear();
      else
        for (auto kind : counters)
          state.wait(kind, 0);
    } else if (choice < 160) {
      state.add({state.issue(counter, scalar), step, lanes, reg, counter, bytes});
    } else if (choice < 208) {
      state.wait(counter, (random() >> 24) & 3);
    } else {
      state.access(reg, lanes, bytes, choice & 1);
    }
    validate();
  }
}

TEST(MemoryWaitExecutionTest, MemoryProducerOverwriteIsCheckedBeforeExecution) {
  using namespace rocr::llvm::amdhsa;
  class BeforeObserver final : public ExecutionPlugin {
  public:
    BeforeObserver(uint64_t pc, ComputeUnitCore *cu)
        : ExecutionPlugin("wait_before_test"), target(pc), cu(cu) {}
    void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &,
                                          Wavefront &wf) override {
      if (pc == target) {
        ++visits;
        diagnostics = cu->memory_wait_diagnostic_count();
        if (const auto *state = wf.memory_wait_scoreboard())
          planned_result = std::ranges::any_of(state->events(),
                                               [pc](const auto &event) { return event.pc == pc; });
      }
    }
    uint64_t target;
    ComputeUnitCore *cu;
    unsigned visits = 0;
    uint64_t diagnostics = 0;
    bool planned_result = false;
  };
  for (bool scalar : {false, true})
    for (bool waited : {false, true}) {
      SCOPED_TRACE(scalar);
      SCOPED_TRACE(waited);
      std::vector<uint32_t> code;
      auto emit = [&](auto words) { append_instruction(code, words); };
      emit(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
      if (scalar)
        emit(make_s_load_b32_scaled_imm(4, 0, 0));
      else
        emit(cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                  {.saddr = 0, .vdst = 2, .vaddr = 0}));
      if (waited)
        emit(cdna5::build_sopp(scalar ? cdna5::kSWaitKmcntSopp : cdna5::kSWaitLoadcntSopp,
                               {.simm16 = 0}));
      const uint64_t consumer = 0x10000 + sizeof(kernel_descriptor_t) + code.size() * 4;
      if (scalar)
        emit(make_s_load_b32_scaled_imm(4, 0, 0));
      else
        emit(cdna5::build_vds(cdna5::kDsLoadU16Vds, {.addr = 0, .vdst = 2}));
      emit(S_ENDPGM_GFX12);
      Gfx1250Sim sim(memory_wait_test_config());
      auto observer = std::make_unique<BeforeObserver>(consumer, sim.cu());
      auto *observed = observer.get();
      sim.plugin_group->add(std::move(observer));
      sim.soc->set_plugin_group(sim.plugin_group);
      uint32_t properties = 0;
      AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
      write_global_u32(*sim.memory, 0x400000, 0x12345678);
      const auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32, 2, false,
                                           false, false, properties, 16);
      test::AqlQueue queue(sim.memory, sim.cp());
      queue.dispatch(kernel, 32, 32, 0x400000);
      step_until_halted(*sim.engine, *sim.cu());
      EXPECT_EQ(observed->visits, 1u);
      EXPECT_TRUE(observed->planned_result);
      EXPECT_EQ(observed->diagnostics, waited ? 0u : 1u);
      EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), waited ? 0u : 1u);
    }
}

TEST(MemoryWaitExecutionTest, TransposePrecheckUsesExpandedExecutionLanes) {
  GpuMemory memory("transpose_precheck_memory");
  L2Cache l2("transpose_precheck_l2");
  ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 32;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("transpose_precheck_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 106, 32);
  ASSERT_NE(wf, nullptr);
  auto decoder = Decoder::create(config.arch);
  for (uint64_t exec : {0ull, 1ull, 0x80000000ull}) {
    wf->set_exec(exec);
    for (unsigned pending : {0, 8}) {
      SCOPED_TRACE(exec);
      SCOPED_TRACE(pending);
      const auto words = cdna5::build_vglobal(cdna5::kGlobalLoadTr4B64Vglobal,
                                              {.saddr = 0, .vdst = 8, .vaddr = 0});
      util::StringDiagnostic error;
      auto decoded = decoder->decode_window(words, 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      auto &state = wf->ensure_memory_wait_scoreboard();
      state.clear();
      unsigned reports = 0;
      state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
      state.add({state.issue(WaitCounterKind::Ds),
                 0x100,
                 uint64_t{1} << 7,
                 {RegClass::VGPR, static_cast<uint16_t>(pending), 1},
                 WaitCounterKind::Ds,
                 0xf});
      state.check_instruction(*decoded.value(), *wf);
      EXPECT_EQ(reports, exec ? 1u : 0u);
    }
  }
}

TEST(MemoryWaitExecutionTest, BufferTypeSuppressesResultsButKeepsEncodedSourceDependencies) {
  class Observer final : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("planning_observer") {}
    void onAmdgpuReadScalarRegister(const Wavefront *, RegisterRef) override { ++reads; }
    unsigned reads = 0;
  };
  GpuMemory memory("buffer_type_precheck_memory");
  L2Cache l2("buffer_type_precheck_l2");
  ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 32;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("buffer_type_precheck_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 106, 32);
  ASSERT_NE(wf, nullptr);
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Observer>();
  auto *observed = observer.get();
  group->add(std::move(observer));
  cu->set_plugin_group(group);
  group->onInit();
  auto decoder = Decoder::create(config.arch);
  for (uint8_t selector : {4, 104, 108}) {
    const RegisterRef descriptor = selector == 4     ? RegisterRef{RegClass::SGPR, 7, 1}
                                   : selector == 104 ? RegisterRef{RegClass::VCC, 1, 1}
                                                     : RegisterRef{RegClass::TTMP, 3, 1};
    for (uint32_t type : {0, 1, 2, 3}) {
      cu->write_sgpr(wf->sgpr_alloc().base + 7, type << 30);
      wf->set_vcc_raw(uint64_t{type} << 62);
      wf->set_ttmp(3, type << 30);
      for (uint64_t exec : {0ull, 1ull}) {
        wf->set_exec(exec);
        for (RegisterRef pending :
             {RegisterRef{RegClass::VGPR, 8, 1}, RegisterRef{RegClass::VGPR, 0, 1}, descriptor}) {
          SCOPED_TRACE(selector);
          SCOPED_TRACE(type);
          SCOPED_TRACE(exec);
          SCOPED_TRACE(static_cast<unsigned>(pending.cls));
          SCOPED_TRACE(pending.index);
          const auto words = cdna5::build_vbuffer(
              cdna5::kBufferLoadB32Vbuffer,
              {.soffset = 124, .vdata = 8, .rsrc = selector, .offen = 1, .vaddr = 0});
          util::StringDiagnostic error;
          auto decoded = decoder->decode_window(words, 0, error.emitter());
          ASSERT_TRUE(decoded.succeeded()) << error.message();
          auto &state = wf->ensure_memory_wait_scoreboard();
          state.clear();
          unsigned reports = 0;
          state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          state.add(
              {state.issue(WaitCounterKind::Ds), 0x100, 1, pending, WaitCounterKind::Ds, 0xf});
          const auto reads_before = observed->reads;
          state.check_instruction(*decoded.value(), *wf);
          EXPECT_EQ(reports,
                    pending.cls != RegClass::VGPR || (exec && (pending.index == 0 || !type)) ? 1u
                                                                                             : 0u);
          EXPECT_EQ(observed->reads, reads_before);
        }
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, DecodedScalarIssueUsesArchitecturalWaitDomain) {
  using namespace waitcheck_detail;
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(static_cast<unsigned>(arch));
    GpuMemory memory("scalar_issue_memory");
    L2Cache l2("scalar_issue_l2");
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 32;
    config.lds_size_kb = 64;
    auto cu = ComputeUnitCore::create("scalar_issue_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 106, 32);
    ASSERT_NE(wf, nullptr);
    const auto words = arch == ROCJITSU_CODE_ARCH_CDNA5 ? make_s_load_b32_scaled_imm(4, 0, 0)
                       : arch == ROCJITSU_CODE_ARCH_CDNA4
                           ? cdna4::build_smem(cdna4::kSLoadDwordSmem, {.sdata = 4})
                           : rdna3::build_smem(rdna3::kSLoadB32Smem, {.sdata = 4});
    auto decoder = Decoder::create(arch);
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    auto &inst = *decoded.value();
    inst.execute(inst, wf);
    ASSERT_NE(inst.data(), nullptr);
    const auto expected = WaitcheckTarget::classify_events(inst, arch);
    ASSERT_TRUE(expected.succeeded());
    ASSERT_FALSE(expected.value().empty());
    const auto &event = expected.value().front();
    cu->track_memory_wait(inst, *wf);
    auto &state = wf->ensure_memory_wait_scoreboard();
    ASSERT_FALSE(state.events().empty());
    EXPECT_EQ(state.events().front().counter, event.counter);
    EXPECT_EQ(state.outstanding(event.counter),
              MemoryWaitScoreboard::issue_units(inst, event, arch));
    state.wait(event.counter, 0);
    EXPECT_TRUE(state.empty());
  }
}

TEST(MemoryWaitFootprintTest, ScalarRelativeAccessesMatchExecution) {
  class ScalarObserver final : public ExecutionPlugin {
  public:
    ScalarObserver() : ExecutionPlugin("scalar_relative_footprint") {}
    std::vector<RegisterRef> reads, writes;
    void onAmdgpuReadScalarRegister(const Wavefront *, RegisterRef reg) override {
      reads.push_back(reg);
    }
    void onAmdgpuWriteScalarRegister(const Wavefront *, RegisterRef reg) override {
      writes.push_back(reg);
    }
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(static_cast<unsigned>(arch));
    GpuMemory memory("scalar_relative_memory");
    L2Cache l2("scalar_relative_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 32;
    config.lds_size_kb = 64;
    auto cu = ComputeUnitCore::create("scalar_relative_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 106, 32, arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64);
    ASSERT_NE(wf, nullptr);
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto observer = std::make_unique<ScalarObserver>();
    auto *observed = observer.get();
    group->add(std::move(observer));
    cu->set_plugin_group(group);
    group->onInit();
    auto decoder = Decoder::create(arch);
    for (unsigned op = 0; op < (arch == ROCJITSU_CODE_ARCH_CDNA4 ? 4u : 5u); ++op) {
      SCOPED_TRACE(op);
      const auto words =
          arch == ROCJITSU_CODE_ARCH_CDNA4
              ? cdna4::build_sop1(cdna4::kSMovrelsB32Sop1 + op, {.ssrc0 = 4, .sdst = 32})
          : arch == ROCJITSU_CODE_ARCH_RDNA3
              ? rdna3::build_sop1(rdna3::kSMovrelsB32Sop1 + op, {.ssrc0 = 4, .sdst = 32})
              : cdna5::build_sop1(cdna5::kSMovrelsB32Sop1 + op, {.ssrc0 = 4, .sdst = 32});
      util::StringDiagnostic error;
      auto decoded = decoder->decode_window(words, 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      for (uint32_t m0 : {0u, 3u, 35u, 37u, 74u, 92u, 102u, 104u, 0x2303u}) {
        if ((op == 1 && (m0 & 0xffu) > 100) || (op == 3 && (m0 & 0xffu) >= 48) ||
            (op == 2 && (m0 & 0xffu) >= 96))
          continue;
        SCOPED_TRACE(m0);
        wf->set_exec_raw(~uint64_t{0});
        wf->set_m0(m0);
        observed->reads.clear();
        observed->writes.clear();
        ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wf).succeeded());
        auto reads = observed->reads;
        auto writes = observed->writes;
        // Named-register getters/setters do not issue scalar-observer callbacks.
        // Supplement those aliases from the documented scalar selector map.
        const unsigned width = op == 1 || op == 3 ? 2 : 1;
        const unsigned src = 4 + (op < 2 || op == 4 ? (m0 & 0xffu) * width : 0);
        const unsigned dst = 32 + (op == 4   ? (m0 >> 8) & 0xffu
                                   : op >= 2 ? (m0 & 0xffu) * width
                                             : 0);
        auto named = [&](unsigned selector, auto &accesses) {
          for (unsigned word = 0; word < width; ++word) {
            const auto current = selector + word;
            if (current == 106 || current == 107)
              accesses.push_back({RegClass::VCC, static_cast<uint16_t>(current - 106), 1});
            else if (current == 126 || current == 127)
              accesses.push_back({RegClass::EXEC, static_cast<uint16_t>(current - 126), 1});
            else if (arch == ROCJITSU_CODE_ARCH_CDNA4 && (current == 102 || current == 103))
              accesses.push_back({RegClass::FLAT_SCRATCH, static_cast<uint16_t>(current - 102), 1});
            else if (current == (arch == ROCJITSU_CODE_ARCH_CDNA4 ? 124u : 125u))
              accesses.push_back({RegClass::M0, 0, 1});
          }
        };
        named(src, reads);
        named(dst, writes);
        wf->set_exec_raw(~uint64_t{0});
        wf->set_m0(m0);
        auto &state = wf->ensure_memory_wait_scoreboard();
        unsigned reports = 0;
        state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
        for (auto cls : {RegClass::SGPR, RegClass::TTMP, RegClass::VCC, RegClass::EXEC,
                         RegClass::M0, RegClass::FLAT_SCRATCH, RegClass::SCC}) {
          const unsigned count =
              cls == RegClass::SGPR                                                            ? 106
              : cls == RegClass::TTMP                                                          ? 16
              : cls == RegClass::VCC || cls == RegClass::EXEC || cls == RegClass::FLAT_SCRATCH ? 2
                                                                                               : 1;
          for (unsigned index = 0; index < count; ++index)
            for (bool replay : {false, true}) {
              const RegisterRef reg{cls, static_cast<uint16_t>(index), 1};
              auto overlaps = [&](const auto &accesses) {
                return std::ranges::any_of(accesses, [&](RegisterRef access) {
                  return access.cls == cls && index >= access.index &&
                         index < unsigned(access.index + access.width);
                });
              };
              const bool expected =
                  overlaps(writes) || (!replay && (overlaps(reads) || cls == RegClass::M0));
              state.clear();
              reports = 0;
              const auto counter = replay ? WaitCounterKind::X : WaitCounterKind::Km;
              state.add({state.issue(counter), 0x80, ~uint64_t{0}, reg, counter, 0xf});
              const auto read_count = observed->reads.size();
              const auto write_count = observed->writes.size();
              state.check_instruction(*decoded.value(), *wf);
              EXPECT_EQ(reports != 0, expected) << "class=" << static_cast<unsigned>(cls)
                                                << " index=" << index << " replay=" << replay;
              EXPECT_EQ(observed->reads.size(), read_count);
              EXPECT_EQ(observed->writes.size(), write_count);
            }
        }
        state.clear();
      }
    }
    group->onShutdown();
  }
}

TEST(MemoryWaitExecutionTest, ScalarStoresReadTheirDataAndHaveNoRegisterResult) {
  GpuMemory memory("scalar_store_plan_memory");
  L2Cache l2("scalar_store_plan_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_CDNA4;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 32;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("scalar_store_plan_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 128, 32);
  ASSERT_NE(wf, nullptr);
  auto decoder = Decoder::create(config.arch);
  const auto words = cdna4::build_smem(cdna4::kSStoreDwordSmem, {.sdata = 4});
  util::StringDiagnostic error;
  auto decoded = decoder->decode_window(words, 0, error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  auto &inst = *decoded.value();
  auto &state = wf->ensure_memory_wait_scoreboard();
  for (bool waited : {false, true}) {
    state.clear();
    unsigned reads = 0, writes = 0;
    std::pair<unsigned *, unsigned *> counts{&reads, &writes};
    state.bind(0x100, &counts, [](void *p, const auto &hazard) {
      auto &[reads, writes] = *static_cast<decltype(counts) *>(p);
      ++*(hazard.write ? writes : reads);
    });
    state.add({state.issue(WaitCounterKind::Ds, true),
               0x80,
               ~uint64_t{0},
               {RegClass::SGPR, 4, 1},
               WaitCounterKind::Ds,
               0xf});
    if (waited)
      state.wait(WaitCounterKind::Ds, 0);
    state.check_instruction(inst, *wf);
    EXPECT_EQ(reads, waited ? 0u : 1u);
    EXPECT_EQ(writes, 0u);
  }
  state.clear();
  cu->track_memory_wait(inst, *wf);
  EXPECT_TRUE(state.events().empty());
  EXPECT_EQ(state.outstanding(WaitCounterKind::Ds), 1u);
}

TEST(MemoryWaitExecutionTest, DirectLdsResultsUseWholeQuadsAndValidateM0BeforeExecution) {
  GpuMemory memory("lds_direct_plan_memory");
  L2Cache l2("lds_direct_plan_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_RDNA3;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 32;
  config.vgprs_per_wf = 32;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("lds_direct_plan_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 32, 32, 32);
  ASSERT_NE(wf, nullptr);
  auto decoder = Decoder::create(config.arch);
  const auto words = rdna3::build_ldsdir(rdna3::kLdsDirectLoadLdsdir, {.vdst = 4});
  util::StringDiagnostic error;
  auto decoded = decoder->decode_window(words, 0, error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  auto &inst = *decoded.value();
  auto &state = wf->ensure_memory_wait_scoreboard();
  for (unsigned type = 0; type < 8; ++type)
    for (unsigned address : {0u, 1u})
      for (uint64_t exec : {uint64_t{0}, uint64_t{1}, uint64_t{0x10}}) {
        SCOPED_TRACE(type);
        SCOPED_TRACE(address);
        SCOPED_TRACE(exec);
        const bool valid = address == 0 && type != 3 && type < 6;
        wf->set_m0((type << 16) | address);
        wf->set_exec(exec);
        state.clear();
        unsigned reports = 0;
        state.bind(0x100, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
        state.add({state.issue(WaitCounterKind::Load),
                   0x80,
                   4,
                   {RegClass::VGPR, 4, 1},
                   WaitCounterKind::Load,
                   0xf});
        state.check_instruction(inst, *wf);
        EXPECT_EQ(reports, valid && exec == 1 ? 1u : 0u);
        state.clear();
        cu->track_memory_wait(inst, *wf);
        const uint64_t expected = !valid || !exec ? 0 : exec == 1 ? 0xf : 0xf0;
        uint64_t recorded = 0;
        for (const auto &event : state.events()) {
          EXPECT_EQ(event.counter, WaitCounterKind::Exp);
          EXPECT_EQ(event.reg.index, 4u);
          recorded |= event.lanes;
        }
        EXPECT_EQ(recorded, expected);
      }
}

TEST(MemoryWaitExecutionTest, DecodedProducerResultsMatchResolvedPipelineDestinations) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(static_cast<unsigned>(arch));
    std::vector<std::vector<uint32_t>> cases;
    auto add = [&](auto words) { cases.emplace_back(words.begin(), words.end()); };
    if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
      for (unsigned th : {0u, 4u}) {
        add(cdna5::build_vbuffer(
            cdna5::kBufferAtomicCmpswapB32Vbuffer,
            {.soffset = 124, .vdata = 4, .rsrc = 0, .th = static_cast<uint8_t>(th)}));
        add(cdna5::build_vbuffer(
            cdna5::kBufferAtomicCmpswapB64Vbuffer,
            {.soffset = 124, .vdata = 4, .rsrc = 0, .th = static_cast<uint8_t>(th)}));
        add(cdna5::build_vglobal(
            cdna5::kGlobalAtomicCmpswapB64Vglobal,
            {.saddr = 124, .vdst = 16, .th = static_cast<uint8_t>(th), .vsrc = 4, .vaddr = 0}));
      }
      for (auto op :
           {cdna5::kDsAddRtnU32Vds, cdna5::kDsCmpstoreRtnB64Vds, cdna5::kDsStorexchg2addrRtnB64Vds,
            cdna5::kDsConsumeVds, cdna5::kDsAppendVds, cdna5::kDsAtomicBarrierArriveRtnB64Vds})
        add(cdna5::build_vds(op, {.addr = 0, .data0 = 4, .data1 = 8, .vdst = 16}));
      for (uint8_t dst : {16, 30})
        add(cdna5::build_vds(cdna5::kDsLoad2addrB64Vds, {.addr = 0, .vdst = dst}));
    } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
      for (uint8_t lds : {0, 1})
        add(cdna4::build_mubuf(cdna4::kBufferLoadDwordMubuf,
                               {.lds = lds, .vdata = 16, .srsrc = 0, .soffset = 128}));
      for (uint8_t glc : {0, 1}) {
        add(cdna4::build_mubuf(cdna4::kBufferAtomicCmpswapMubuf,
                               {.sc0 = glc, .vdata = 4, .srsrc = 0, .soffset = 128}));
        add(cdna4::build_mubuf(cdna4::kBufferAtomicCmpswapX2Mubuf,
                               {.sc0 = glc, .vdata = 4, .srsrc = 0, .soffset = 128}));
      }
      for (auto op : {cdna4::kDsAddRtnU32Ds, cdna4::kDsCmpstRtnB64Ds, cdna4::kDsWrxchg2RtnB64Ds,
                      cdna4::kDsConsumeDs, cdna4::kDsAppendDs})
        add(cdna4::build_ds(op, {.addr = 0, .data0 = 4, .data1 = 8, .vdst = 16}));
      for (uint8_t dst : {16, 30})
        add(cdna4::build_ds(cdna4::kDsRead2B64Ds, {.addr = 0, .vdst = dst}));
    } else {
      for (uint8_t glc : {0, 1}) {
        add(rdna3::build_mubuf(rdna3::kBufferAtomicCmpswapB32Mubuf,
                               {.glc = glc, .vdata = 4, .srsrc = 0, .soffset = 128}));
        add(rdna3::build_mubuf(rdna3::kBufferAtomicCmpswapB64Mubuf,
                               {.glc = glc, .vdata = 4, .srsrc = 0, .soffset = 128}));
      }
      for (auto op : {rdna3::kDsAddRtnU32Ds, rdna3::kDsCmpstoreRtnB64Ds,
                      rdna3::kDsStorexchg2addrRtnB64Ds, rdna3::kDsConsumeDs, rdna3::kDsAppendDs})
        add(rdna3::build_ds(op, {.addr = 0, .data0 = 4, .data1 = 8, .vdst = 16}));
      for (uint8_t dst : {16, 30}) {
        add(rdna3::build_ds(rdna3::kDsLoad2addrB64Ds, {.addr = 0, .vdst = dst}));
        add(rdna3::build_ds(rdna3::kDsBvhStackRtnB32Ds,
                            {.offset0 = 2, .addr = 0, .data0 = 4, .data1 = 8, .vdst = dst}));
      }
    }
    GpuMemory memory("producer_shape_memory");
    L2Cache l2("producer_shape_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 32;
    config.lds_size_kb = 64;
    auto cu = ComputeUnitCore::create("producer_shape_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 106, 32, arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32);
    ASSERT_NE(wf, nullptr);
    auto decoder = Decoder::create(arch);
    for (const auto &words : cases) {
      util::StringDiagnostic error;
      auto decoded = decoder->decode_window(words, 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      auto &inst = *decoded.value();
      SCOPED_TRACE(inst.mnemonic());
      ASSERT_TRUE(inst.is_memory_wait_producer());
      for (uint64_t exec : {0ull, 1ull, 2ull, 0xffffffffull}) {
        SCOPED_TRACE(exec);
        wf->set_exec(exec);
        auto &state = wf->ensure_memory_wait_scoreboard();
        state.clear();
        cu->track_memory_wait(inst, *wf);
        const auto planned = state.events();
        state.clear();
        ASSERT_TRUE(cu->execute_instruction(&inst, *wf).succeeded());
        ASSERT_NE(inst.data(), nullptr);
        const auto &payload = *inst.data_as<VectorMemState>();
        std::vector<MemoryWaitScoreboard::Event> expected;
        const unsigned width = payload.destination_vgpr_count();
        const unsigned second_width = payload.ds2_active ? payload.ds2_destination_vgpr_count() : 0;
        if (payload.is_load && !payload.lds_dst &&
            cu->owns_vgpr_range(*wf, payload.dst_reg_base, width) &&
            (!payload.ds2_active ||
             cu->owns_vgpr_range(*wf, payload.ds2_dst_reg_base, second_width))) {
          auto append = [&](unsigned base, unsigned width) {
            expected.push_back(
                {0,
                 0,
                 payload.exec_mask,
                 {RegClass::VGPR, static_cast<uint16_t>(base - wf->vgpr_alloc().base),
                  static_cast<uint8_t>(width)},
                 WaitCounterKind::Ds,
                 0xf});
          };
          append(payload.dst_reg_base, width);
          if (payload.ds2_active)
            append(payload.ds2_dst_reg_base, second_width);
        }
        for (unsigned reg = 0; reg < 32; ++reg)
          for (uint64_t lane : {1ull, 2ull, 0x80000000ull})
            for (unsigned byte = 0; byte < 4; ++byte) {
              state.clear();
              unsigned reports = 0;
              state.bind(0x200, &reports, [](void *p, const auto &hazard) {
                if (hazard.write)
                  ++*static_cast<unsigned *>(p);
              });
              state.add({state.issue(WaitCounterKind::Ds),
                         0x80,
                         lane,
                         {RegClass::VGPR, static_cast<uint16_t>(reg), 1},
                         WaitCounterKind::Ds,
                         static_cast<uint8_t>(1u << byte)});
              state.check_instruction(inst, *wf);
              const auto overlaps = [&](const auto &event) {
                return event.reg.cls == RegClass::VGPR && reg >= event.reg.index &&
                       reg < unsigned(event.reg.index + event.reg.width) && (lane & event.lanes) &&
                       (event.bytes & (1u << byte));
              };
              const bool written = std::ranges::any_of(expected, overlaps);
              EXPECT_EQ(std::ranges::any_of(planned, overlaps), written)
                  << "planned v" << reg << " lane=" << lane << " byte=" << byte;
              EXPECT_EQ(reports != 0, written)
                  << "v" << reg << " lane=" << lane << " byte=" << byte;
            }
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, DualDsStoresSnapshotActiveLanesAndAliasedSources) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(static_cast<unsigned>(arch));
    GpuMemory memory("dual_store_memory");
    L2Cache l2("dual_store_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 32;
    auto cu = ComputeUnitCore::create("dual_store_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 106, 32, arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32);
    ASSERT_NE(wf, nullptr);
    const auto value = [](unsigned reg, unsigned lane) { return reg * 0x10000u + lane * 32u; };
    for (unsigned reg = 0; reg < 4; ++reg)
      for (unsigned lane = 0; lane < wf->wf_size(); ++lane)
        cu->write_vgpr(wf->vgpr_alloc().base + reg, lane, value(reg, lane));
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto observer = std::make_unique<MemoryWaitFootprintObserver>();
    auto *observed = observer.get();
    group->add(std::move(observer));
    cu->set_plugin_group(group);
    group->onInit();
    auto decoder = Decoder::create(arch);
    for (unsigned width : {1u, 2u})
      for (bool stride64 : {false, true})
        for (uint64_t exec : {0ull, 0x80000005ull, ~0ull}) {
          SCOPED_TRACE(width);
          SCOPED_TRACE(stride64);
          SCOPED_TRACE(exec);
          wf->set_exec(exec);
          const uint16_t opcode = (width == 1 ? 14 : 78) + stride64;
          // Address and first data operand deliberately alias.
          const auto words =
              arch == ROCJITSU_CODE_ARCH_CDNA5
                  ? cdna5::build_vds(
                        opcode, {.offset0 = 1, .offset1 = 3, .addr = 0, .data0 = 0, .data1 = 2})
              : arch == ROCJITSU_CODE_ARCH_CDNA4
                  ? cdna4::build_ds(opcode,
                                    {.offset0 = 1, .offset1 = 3, .addr = 0, .data0 = 0, .data1 = 2})
                  : rdna3::build_ds(
                        opcode, {.offset0 = 1, .offset1 = 3, .addr = 0, .data0 = 0, .data1 = 2});
          util::StringDiagnostic error;
          auto decoded = decoder->decode_window(words, 0, error.emitter());
          ASSERT_TRUE(decoded.succeeded()) << error.message();
          observed->callbacks = 0;
          observed->accesses = {};
          ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wf).succeeded());
          const auto *data = decoded.value()->data_as<VectorMemState>();
          ASSERT_NE(data, nullptr);
          ASSERT_TRUE(data->ds2_active);
          EXPECT_EQ(observed->callbacks, exec ? 1 + 2 * width : 0);
          const unsigned step = width * 4 * (stride64 ? 64 : 1);
          const auto active = exec & (wf->wf_size() == 64 ? ~0ull : 0xffffffffull);
          for (unsigned reg : {0u, 2u})
            EXPECT_EQ(observed->accesses[reg].read[0], active);
          for (unsigned lane = 0; lane < wf->wf_size(); ++lane) {
            const bool enabled = active & (1ull << lane);
            if (enabled) {
              EXPECT_EQ(data->per_lane_addr[lane], value(0, lane) + step + wf->lds_base());
              EXPECT_EQ(data->ds2_per_lane_addr[lane], value(0, lane) + step * 3 + wf->lds_base());
            }
            for (unsigned word = 0; word < width; ++word) {
              uint32_t first, second;
              std::memcpy(&first, data->store_data.data() + (lane * width + word) * 4, 4);
              std::memcpy(&second, data->ds2_store_data.data() + (lane * width + word) * 4, 4);
              EXPECT_EQ(first, enabled ? value(word, lane) : 0u);
              EXPECT_EQ(second, enabled ? value(2 + word, lane) : 0u);
            }
          }
        }
    group->onShutdown();
  }
}

TEST(MemoryWaitExecutionTest, DualDsExchangeExecutesBothDecodedResultsAcrossArchitectures) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(static_cast<unsigned>(arch));
    GpuMemory memory("dual_ds_memory");
    L2Cache l2("dual_ds_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 32;
    config.lds_size_kb = 64;
    auto cu = ComputeUnitCore::create("dual_ds_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 106, 32, arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);
    wf->set_lds_base(cu->allocate_lds(4096));
    auto decoder = Decoder::create(arch);
    for (unsigned width : {1u, 2u})
      for (bool stride64 : {false, true}) {
        SCOPED_TRACE(width);
        SCOPED_TRACE(stride64);
        const uint16_t opcode = (width == 1 ? 46 : 110) + stride64;
        const auto words =
            arch == ROCJITSU_CODE_ARCH_CDNA5
                ? cdna5::build_vds(
                      opcode,
                      {.offset0 = 1, .offset1 = 3, .addr = 0, .data0 = 4, .data1 = 8, .vdst = 16})
            : arch == ROCJITSU_CODE_ARCH_CDNA4
                ? cdna4::build_ds(
                      opcode,
                      {.offset0 = 1, .offset1 = 3, .addr = 0, .data0 = 4, .data1 = 8, .vdst = 16})
                : rdna3::build_ds(
                      opcode,
                      {.offset0 = 1, .offset1 = 3, .addr = 0, .data0 = 4, .data1 = 8, .vdst = 16});
        const unsigned step = width * 4 * (stride64 ? 64 : 1);
        const auto first_address = wf->lds_base() + 32 + step;
        const auto second_address = wf->lds_base() + 32 + step * 3;
        const auto vb = wf->vgpr_alloc().base;
        cu->write_vgpr(vb, 0, 32);
        for (unsigned word = 0; word < width; ++word) {
          cu->write_vgpr(vb + 4 + word, 0, 0x11111111u + word);
          cu->write_vgpr(vb + 8 + word, 0, 0x22222222u + word);
          cu->lds().write32(first_address + word * 4, 0x33333333u + word);
          cu->lds().write32(second_address + word * 4, 0x44444444u + word);
        }
        util::StringDiagnostic error;
        auto decoded = decoder->decode_window(words, 0, error.emitter());
        ASSERT_TRUE(decoded.succeeded()) << error.message();
        ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wf).succeeded());
        auto *data = decoded.value()->data_as<VectorMemState>();
        ASSERT_NE(data, nullptr);
        ASSERT_TRUE(data->ds2_active);
        EXPECT_EQ(data->per_lane_addr[0], first_address);
        EXPECT_EQ(data->ds2_per_lane_addr[0], second_address);
        LocalMemPipeline pipeline;
        pipeline.issue(decoded.value().release(), *wf);
        for (unsigned word = 0; word < width; ++word) {
          EXPECT_EQ(cu->lds().read32(first_address + word * 4), 0x11111111u + word);
          EXPECT_EQ(cu->lds().read32(second_address + word * 4), 0x22222222u + word);
          EXPECT_EQ(cu->read_vgpr(vb + 16 + word, 0), 0x33333333u + word);
          EXPECT_EQ(cu->read_vgpr(vb + 16 + width + word, 0), 0x44444444u + word);
        }
      }
  }
}

TEST(MemoryWaitExecutionTest,
     BufferResourceAliasesAndEncodedAddressInputsArePlannedBeforeExecution) {
  class Observer final : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("buffer_alias_observer") {}
    unsigned reads = 0;
    void onAmdgpuReadScalarRegister(const Wavefront *, RegisterRef) override { ++reads; }
    void onAmdgpuReadVgprLanes(const Wavefront *, uint32_t, uint64_t, uint8_t) override { ++reads; }
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5})
    for (unsigned sgprs : {16u, 106u}) {
      SCOPED_TRACE(static_cast<unsigned>(arch));
      SCOPED_TRACE(sgprs);
      GpuMemory memory("buffer_alias_memory");
      L2Cache l2("buffer_alias_l2");
      ComputeUnitCore::Config config{};
      config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
      config.arch = arch;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = sgprs;
      config.vgprs_per_wf = 32;
      config.lds_size_kb = 64;
      auto cu = ComputeUnitCore::create("buffer_alias_cu", config, &memory, &l2);
      auto *wf = cu->dispatch_wf(0, 0x100, sgprs, 32);
      ASSERT_NE(wf, nullptr);
      auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
      auto observer = std::make_unique<Observer>();
      auto *observed = observer.get();
      group->add(std::move(observer));
      cu->set_plugin_group(group);
      group->onInit();
      auto decoder = Decoder::create(arch);
      std::vector<uint32_t> words;
      if (arch == ROCJITSU_CODE_ARCH_CDNA5)
        append_instruction(
            words,
            cdna5::build_vbuffer(cdna5::kBufferLoadB32Vbuffer,
                                 {.soffset = 4, .vdata = 8, .rsrc = 104, .offen = 1, .vaddr = 0}));
      else
        append_instruction(
            words,
            rdna3::build_mubuf(rdna3::kBufferLoadB32Mubuf,
                               {.vaddr = 0, .vdata = 8, .srsrc = 26, .offen = 1, .soffset = 4}));
      util::StringDiagnostic error;
      auto decoded = decoder->decode_window(words, 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      for (uint64_t exec : {0ull, 1ull})
        for (RegisterRef pending :
             {RegisterRef{RegClass::VGPR, 8, 1}, RegisterRef{RegClass::VGPR, 0, 1},
              RegisterRef{RegClass::SGPR, 4, 1}, RegisterRef{RegClass::SGPR, 104, 1},
              RegisterRef{RegClass::VCC, 1, 1}}) {
          SCOPED_TRACE(exec);
          SCOPED_TRACE(static_cast<unsigned>(pending.cls));
          SCOPED_TRACE(pending.index);
          wf->set_exec(exec);
          auto &state = wf->ensure_memory_wait_scoreboard();
          state.clear();
          unsigned reports = 0;
          state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          state.add({state.issue(WaitCounterKind::Ds), 0x80, 1, pending, WaitCounterKind::Ds, 0xf});
          const auto reads_before = observed->reads;
          state.check_instruction(*decoded.value(), *wf);
          // Invalid backing does not establish an ISA exemption from source
          // waits. Address/offset operands keep their encoded dependencies.
          const bool expected = pending.cls == RegClass::VGPR ? bool(exec)
                                : pending.cls == RegClass::SGPR && pending.index == 4
                                    ? true
                                    : sgprs == 106;
          EXPECT_EQ(reports != 0, expected);
          EXPECT_EQ(observed->reads, reads_before);
        }
      group->onShutdown();
    }
}

TEST(XcntExecutionTest, BufferResourceCrossingVccProtectsBothAliasWords) {
  GpuMemory memory("xcnt_resource_memory");
  L2Cache l2("xcnt_resource_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.xcnt_diagnostics = MemoryWaitDiagnostics::Warn;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 32;
  config.lds_size_kb = 64;
  auto cu = ComputeUnitCore::create("xcnt_resource_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 106, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  wf->set_mode_raw(wf->mode_raw() | (1u << 25));
  auto decoder = Decoder::create(config.arch);
  const auto words =
      cdna5::build_vbuffer(cdna5::kBufferLoadB32Vbuffer,
                           {.soffset = 128, .vdata = 8, .rsrc = 104, .offen = 1, .vaddr = 0});
  util::StringDiagnostic error;
  auto decoded = decoder->decode_window(words, 0, error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  for (bool waited : {false, true})
    for (uint16_t half : {0, 1}) {
      auto &state = wf->ensure_memory_wait_scoreboard();
      state.clear();
      cu->track_memory_wait(*decoded.value(), *wf);
      unsigned reports = 0;
      state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
      if (waited)
        state.wait(WaitCounterKind::X, 0);
      state.access({RegClass::VCC, half, 1}, ~uint64_t{0}, 0xf, true);
      EXPECT_EQ(reports, waited ? 0u : 1u) << half;
    }
}

TEST(MemoryWaitFootprintTest, DsLanePermutationsUseConsumedSourceLanes) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    const auto prepare = [](Wavefront &wf) {
      for (unsigned lane = 0; lane < wf.wf_size(); ++lane) {
        RegisterAccess(wf).write_vgpr(wf.vgpr_alloc().base, lane, 4 * ((lane * 3 + 17) % 64));
        RegisterAccess(wf).write_vgpr(wf.vgpr_alloc().base + 4, lane, 0x1000 + lane);
      }
    };
    std::vector<std::vector<uint32_t>> cases;
    for (unsigned operation = 0; operation < 4; ++operation) {
      if (operation == 3 && arch != ROCJITSU_CODE_ARCH_CDNA5)
        continue;
      for (unsigned offset : {0u, 4u, 0x801bu, 0x401fu}) {
        for (unsigned destination : {0u, 8u}) {
          auto add = [&](const auto &words) { cases.emplace_back(words.begin(), words.end()); };
          if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
            const uint16_t ops[] = {cdna5::kDsPermuteB32Vds, cdna5::kDsBpermuteB32Vds,
                                    cdna5::kDsSwizzleB32Vds, cdna5::kDsBpermuteFiB32Vds};
            add(cdna5::build_vds(ops[operation], {.offset0 = static_cast<uint8_t>(offset),
                                                  .offset1 = static_cast<uint8_t>(offset >> 8),
                                                  .addr = 0,
                                                  .data0 = 4,
                                                  .vdst = static_cast<uint8_t>(destination)}));
          } else if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
            const uint16_t ops[] = {rdna3::kDsPermuteB32Ds, rdna3::kDsBpermuteB32Ds,
                                    rdna3::kDsSwizzleB32Ds};
            add(rdna3::build_ds(ops[operation], {.offset0 = static_cast<uint8_t>(offset),
                                                 .offset1 = static_cast<uint8_t>(offset >> 8),
                                                 .addr = 0,
                                                 .data0 = 4,
                                                 .vdst = static_cast<uint8_t>(destination)}));
          } else {
            const uint16_t ops[] = {cdna4::kDsPermuteB32Ds, cdna4::kDsBpermuteB32Ds,
                                    cdna4::kDsSwizzleB32Ds};
            add(cdna4::build_ds(ops[operation], {.offset0 = static_cast<uint8_t>(offset),
                                                 .offset1 = static_cast<uint8_t>(offset >> 8),
                                                 .addr = 0,
                                                 .data0 = 4,
                                                 .vdst = static_cast<uint8_t>(destination)}));
          }
        }
      }
    }
    check_vector_footprints(arch, arch == ROCJITSU_CODE_ARCH_CDNA5 ? 32 : 64, 16, cases, 0, 0,
                            prepare);
    if (arch == ROCJITSU_CODE_ARCH_RDNA3)
      check_vector_footprints(arch, 32, 16, cases, 0, 0, prepare);
  }
}

TEST(MemoryWaitFootprintTest, BasicSwizzleModesMatchPublishedLaneExamples) {
  for (unsigned lane = 0; lane < 64; ++lane) {
    EXPECT_EQ(ds_swizzle_lane(lane, 0x801b), (lane & ~3u) + 3 - (lane & 3u));
    EXPECT_EQ(ds_swizzle_lane(lane, 0x401f), lane ^ 16u);
    EXPECT_EQ(ds_swizzle_lane(lane, 7u << 5), (lane & ~31u) + 7);
  }
}

TEST(MemoryWaitFootprintTest, RestrictedScalarSelectorsAndZeroAliasesKeepDependencies) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3}) {
    GpuMemory memory("selector_wait_memory");
    L2Cache l2("selector_wait_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 128;
    config.vgprs_per_wf = 32;
    auto cu = ComputeUnitCore::create("selector_wait_cu", config, &memory, &l2);
    const unsigned wave_size = arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32;
    auto *wf = cu->dispatch_wf(0, 0x100, 104, 32, wave_size);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);
    auto decoder = Decoder::create(arch);
    for (unsigned selector : {106u, 107u, 251u, 252u, 253u})
      for (bool vector : {false, true}) {
        // GFX11 removed these computed source selectors.
        if (arch == ROCJITSU_CODE_ARCH_RDNA3 && (selector == 251 || selector == 252))
          continue;
        SCOPED_TRACE(selector);
        SCOPED_TRACE(vector);
        std::vector<uint32_t> words;
        if (vector) {
          if (arch == ROCJITSU_CODE_ARCH_CDNA4)
            append_instruction(
                words, cdna4::build_vop3(cdna4::kVMovB32Vop3,
                                         {.vdst = 4, .src0 = static_cast<uint16_t>(selector)}));
          else
            append_instruction(
                words, rdna3::build_vop3(rdna3::kVMovB32Vop3,
                                         {.vdst = 4, .src0 = static_cast<uint16_t>(selector)}));
        } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
          append_instruction(
              words, cdna4::build_sop1(cdna4::kSMovB32Sop1,
                                       {.ssrc0 = static_cast<uint8_t>(selector), .sdst = 4}));
        } else {
          append_instruction(
              words, rdna3::build_sop1(rdna3::kSMovB32Sop1,
                                       {.ssrc0 = static_cast<uint8_t>(selector), .sdst = 4}));
        }
        util::StringDiagnostic error;
        auto decoded = decoder->decode_window(words, 0, error.emitter());
        ASSERT_TRUE(decoded.succeeded()) << error.message();
        SCOPED_TRACE(decoded.value()->mnemonic());
        SCOPED_TRACE(selector);
        for (auto kind : {RegClass::VCC, RegClass::EXEC, RegClass::SCC})
          for (uint16_t word : {0, 1})
            for (bool waited : {false, true}) {
              if (kind == RegClass::SCC && word)
                continue;
              auto &state = wf->ensure_memory_wait_scoreboard();
              state.clear();
              unsigned reports = 0;
              state.bind(0x200, &reports,
                         [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
              state.add({state.issue(WaitCounterKind::Ds, true),
                         0x100,
                         ~0ull,
                         {kind, word, 1},
                         WaitCounterKind::Ds,
                         0xf});
              if (waited)
                state.wait(WaitCounterKind::Ds, 0);
              state.check_instruction(*decoded.value(), *wf);
              bool consumes = selector == 106   ? kind == RegClass::VCC && word == 0
                              : selector == 107 ? kind == RegClass::VCC && word == 1
                              : selector == 251 ? kind == RegClass::VCC && word < wave_size / 32
                              : selector == 252 ? kind == RegClass::EXEC && word < wave_size / 32
                                                : kind == RegClass::SCC;
              consumes |= vector && kind == RegClass::EXEC && word < wave_size / 32;
              EXPECT_EQ(reports != 0, consumes && !waited)
                  << static_cast<unsigned>(kind) << ":" << word;
            }
      }
  }
}

TEST(MemoryWaitFootprintTest, BarrierIdSelectorsReadPendingMessageResultsInM0) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4}) {
    GpuMemory memory("barrier_selector_memory");
    L2Cache l2("barrier_selector_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 32;
    auto cu = ComputeUnitCore::create("barrier_selector_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 104, 32, 32);
    ASSERT_NE(wf, nullptr);
    auto decoder = Decoder::create(arch);
    auto encode = [&](uint16_t opcode, uint8_t source, uint8_t destination) {
      return arch == ROCJITSU_CODE_ARCH_CDNA5
                 ? cdna5::build_sop1(opcode, {.ssrc0 = source, .sdst = destination})
                 : rdna4::build_sop1(opcode, {.ssrc0 = source, .sdst = destination});
    };
    util::StringDiagnostic error;
    const auto producer_words = encode(cdna5::kSSendmsgRtnB32Sop1, 128, 125);
    auto producer = decoder->decode_window(producer_words, 0, error.emitter());
    ASSERT_TRUE(producer.succeeded()) << error.message();
    for (uint16_t opcode : {cdna5::kSBarrierSignalSop1, cdna5::kSBarrierSignalIsfirstSop1,
                            cdna5::kSGetBarrierStateSop1, cdna5::kSBarrierInitSop1,
                            cdna5::kSBarrierJoinSop1, cdna5::kSWakeupBarrierSop1}) {
      // RDNA4 implements the two signal forms; the other named-barrier
      // instructions belong to CDNA5.
      if (arch == ROCJITSU_CODE_ARCH_RDNA4 && opcode > rdna4::kSBarrierSignalIsfirstSop1)
        continue;
      for (uint8_t source : {125, 128, 193}) {
        const auto words = encode(opcode, source, 4);
        auto decoded = decoder->decode_window(words, 0, error.emitter());
        ASSERT_TRUE(decoded.succeeded()) << error.message();
        for (bool waited : {false, true}) {
          auto &state = wf->ensure_memory_wait_scoreboard();
          state.clear();
          cu->track_memory_wait(*producer.value(), *wf);
          ASSERT_EQ(state.events().size(), 1u);
          ASSERT_EQ(state.events()[0].reg.cls, RegClass::M0);
          unsigned reports = 0;
          state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          if (waited)
            state.wait(WaitCounterKind::Km, 0);
          state.check_instruction(*decoded.value(), *wf);
          // INIT always reads the member count from M0, even for a constant ID.
          const bool reads_m0 = source == 125 || opcode == cdna5::kSBarrierInitSop1;
          EXPECT_EQ(reports != 0, reads_m0 && !waited)
              << decoded.value()->mnemonic() << " source=" << unsigned(source);
        }
      }
    }
  }
}

TEST(MemoryWaitExecutionTest, BarrierM0ReadChecksMessageReturnBeforeExecution) {
  for (bool waited : {false, true}) {
    std::vector<uint32_t> code;
    append_instruction(code,
                       cdna5::build_sop1(cdna5::kSSendmsgRtnB32Sop1, {.ssrc0 = 128, .sdst = 125}));
    if (waited)
      append_instruction(code, cdna5::build_sopp(cdna5::kSWaitKmcntSopp, {.simm16 = 0}));
    append_instruction(code, cdna5::build_sop1(cdna5::kSBarrierSignalSop1, {.ssrc0 = 125}));
    append_instruction(code, S_ENDPGM_GFX12);
    Gfx1250Sim sim(memory_wait_test_config());
    auto kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32);
    test::AqlQueue queue(sim.memory, sim.cp());
    queue.dispatch(kernel, 32, 32);
    step_until_halted(*sim.engine, *sim.cu());
    EXPECT_EQ(sim.cu()->memory_wait_diagnostic_count(), waited ? 0u : 1u);
  }
}

TEST(MemoryWaitFootprintTest, PermutationControlsShareExecutionSelectorValues) {
  GpuMemory memory("lane_control_memory");
  L2Cache l2("lane_control_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_RDNA3;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 32;
  auto cu = ComputeUnitCore::create("lane_control_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 104, 32, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec_raw(1ull << 32);
  wf->set_vcc_raw(1ull << 32);
  wf->set_apertures(3ull << 32, 4ull << 32, 5ull << 32, 6ull << 32);
  wf->set_exec(1);
  for (unsigned lane = 0; lane < 32; ++lane)
    cu->write_vgpr(wf->vgpr_alloc().base + 8, lane, lane);
  auto decoder = Decoder::create(config.arch);
  for (const auto &[selector, expected] :
       {std::pair{131u, 3u}, {193u, 15u}, {235u, 3u}, {236u, 4u}, {237u, 5u}, {238u, 6u}}) {
    const auto words =
        rdna3::build_vop3(rdna3::kVPermlane16B32Vop3, {.vdst = 4,
                                                       .op_sel = 1,
                                                       .src0 = 264,
                                                       .src1 = static_cast<uint16_t>(selector),
                                                       .src2 = 128});
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message() << " selector=" << selector;
    ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wf).succeeded());
    EXPECT_EQ(RegisterAccess(*wf).read_vgpr(wf->vgpr_alloc().base + 4, 0), expected);
    for (unsigned lane = 0; lane < 32; ++lane) {
      auto &state = wf->ensure_memory_wait_scoreboard();
      state.clear();
      unsigned reports = 0;
      state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
      state.add({state.issue(WaitCounterKind::Load),
                 0x100,
                 1ull << lane,
                 {RegClass::VGPR, 8, 1},
                 WaitCounterKind::Load,
                 0xf});
      state.check_instruction(*decoded.value(), *wf);
      EXPECT_EQ(reports != 0, lane == expected) << "selector=" << selector << " lane=" << lane;
    }
  }
}

TEST(MemoryWaitFootprintTest, WordwiseMemorySourcesKeepTheBackedPrefix) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA5}) {
    GpuMemory memory("wordwise_memory");
    L2Cache l2("wordwise_l2");
    ComputeUnitCore::Config config{};
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 128;
    config.vgprs_per_wf = 32;
    auto cu = ComputeUnitCore::create("wordwise_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 104, 32, 32);
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(cu->vgpr_allocation_block_size(), 32u);
    cu->write_vgpr(wf->vgpr_alloc().base + 31, 0, 0x12345678);
    auto decoder = Decoder::create(arch);
    for (bool second : {false, true}) {
      const auto words =
          arch == ROCJITSU_CODE_ARCH_RDNA3
              ? rdna3::build_ds(78, {.offset1 = 1,
                                     .addr = 0,
                                     .data0 = static_cast<uint8_t>(second ? 8 : 31),
                                     .data1 = static_cast<uint8_t>(second ? 31 : 8)})
              : cdna5::build_vds(78, {.offset1 = 1,
                                      .addr = 0,
                                      .data0 = static_cast<uint8_t>(second ? 8 : 31),
                                      .data1 = static_cast<uint8_t>(second ? 31 : 8)});
      util::StringDiagnostic error;
      auto decoded = decoder->decode_window(words, 0, error.emitter());
      ASSERT_TRUE(decoded.succeeded()) << error.message();
      wf->set_exec(1);
      ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wf).succeeded());
      const auto *data = decoded.value()->data_as<VectorMemState>();
      ASSERT_NE(data, nullptr);
      const auto &bytes = second ? data->ds2_store_data : data->store_data;
      uint64_t value = 0;
      std::memcpy(&value, bytes.data(), 8);
      EXPECT_EQ(value, 0x12345678ull);
      for (uint64_t exec : {0ull, 1ull, 2ull})
        for (bool waited : {false, true}) {
          wf->set_exec(exec);
          auto &state = wf->ensure_memory_wait_scoreboard();
          state.clear();
          unsigned reports = 0;
          state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          state.add({state.issue(WaitCounterKind::Load),
                     0x100,
                     1,
                     {RegClass::VGPR, 31, 1},
                     WaitCounterKind::Load,
                     0xf});
          if (waited)
            state.wait(WaitCounterKind::Load, 0);
          state.check_instruction(*decoded.value(), *wf);
          EXPECT_EQ(reports != 0, bool(exec & 1) && !waited);
        }
    }
  }
}

TEST(MemoryWaitFootprintTest, SparseWmmaChecksAllLanesIncludingZeroExec) {
  GpuMemory memory("swmma_memory");
  L2Cache l2("swmma_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_RDNA4;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 40;
  auto cu = ComputeUnitCore::create("swmma_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 104, 40, 32);
  ASSERT_NE(wf, nullptr);
  auto decoder = Decoder::create(config.arch);
  const auto words = rdna4::build_vop3p(rdna4::kVSwmmacF3216x16x32F16Vop3p,
                                        {.vdst = 0, .src0 = 272, .src1 = 280, .src2 = 288});
  util::StringDiagnostic error;
  auto decoded = decoder->decode_window(words, 0, error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  // The ISA consumes complete encoded matrix operands, independently of EXEC
  // and the subset of dense B elements chosen by a particular sparsity pattern.
  for (uint64_t exec : {0ull, 1ull, 0xffffffffull}) {
    wf->set_exec(exec);
    for (uint16_t reg = 0; reg < 40; ++reg)
      for (unsigned lane = 0; lane < 32; ++lane)
        for (bool replay : {false, true}) {
          auto &state = wf->ensure_memory_wait_scoreboard();
          state.clear();
          unsigned reports = 0;
          state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          const auto counter = replay ? WaitCounterKind::X : WaitCounterKind::Load;
          state.add(
              {state.issue(counter), 0x100, 1ull << lane, {RegClass::VGPR, reg, 1}, counter, 0xf});
          state.check_instruction(*decoded.value(), *wf);
          const bool output = reg < 8;
          const bool input = (reg >= 16 && reg < 20) || (reg >= 24 && reg < 33);
          EXPECT_EQ(reports != 0, output || (!replay && input))
              << "reg=" << reg << " lane=" << lane << " EXEC=" << exec;
        }
  }
}

TEST(MemoryWaitFootprintTest, ScaledMfmaChecksAllSourcesAndLanesRegardlessOfExec) {
  GpuMemory memory("scaled_mfma_memory");
  L2Cache l2("scaled_mfma_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_CDNA4;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 104;
  config.vgprs_per_wf = 104;
  auto cu = ComputeUnitCore::create("scaled_mfma_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 104, 104, 64);
  ASSERT_NE(wf, nullptr);
  auto decoder = Decoder::create(config.arch);
  for (unsigned opcode : {45u, 46u}) {
    const auto words = mma_test::make_cdna4_mfma_scale_words(opcode, 1, 256 + 96, 256 + 97);
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    ASSERT_TRUE(decoded.value()->is_mfma());
    // FP4 A and B each occupy four VGPRs in both shapes. C/D have
    // M*N/64 dwords per lane; the two scales are independent sources.
    const unsigned result_width = opcode == 45 ? 4 : 16;
    for (uint64_t exec : {0ull, 1ull, ~0ull}) {
      wf->set_exec(exec);
      for (uint16_t reg = 0; reg < 104; ++reg)
        for (unsigned lane : {0u, 31u, 63u})
          for (bool waited : {false, true}) {
            auto &state = wf->ensure_memory_wait_scoreboard();
            state.clear();
            unsigned reports = 0;
            state.bind(0x200, &reports,
                       [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
            state.add({state.issue(WaitCounterKind::Load),
                       0x100,
                       1ull << lane,
                       {RegClass::VGPR, reg, 1},
                       WaitCounterKind::Load,
                       0xf});
            if (waited)
              state.wait(WaitCounterKind::Load, 0);
            state.check_instruction(*decoded.value(), *wf);
            const bool accessed = reg < 4 || (reg >= 16 && reg < 20) ||
                                  (reg >= 32 && reg < 32 + result_width) ||
                                  (reg >= 64 && reg < 64 + result_width) || reg == 96 || reg == 97;
            EXPECT_EQ(reports != 0, accessed && !waited)
                << "opcode=" << opcode << " reg=" << reg << " lane=" << lane << " EXEC=" << exec;
          }
    }
  }
}

TEST(MemoryWaitExecutionTest, DiagnosticBudgetsAndCountersAreIndependentAcrossCUs) {
  GpuMemory memory("budget_memory");
  L2Cache l2("budget_l2");
  std::vector<std::unique_ptr<ComputeUnitCore>> units;
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_RDNA3;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 32;
  for (unsigned i = 0; i < 3; ++i) {
    auto cu = ComputeUnitCore::create(std::format("budget_cu{}", i), config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, 0x100, 104, 32, 32);
    ASSERT_NE(wf, nullptr);
    MemoryWaitScoreboard::Hazard hazard{
        {1, 0x100, 1, {RegClass::VGPR, 0, 1}, WaitCounterKind::Load, 0xf},
        0x200,
        {RegClass::VGPR, 0, 1},
        false,
        0};
    testing::internal::CaptureStderr();
    for (unsigned report = 0; report < 20; ++report)
      ComputeUnitCore::report_memory_wait(wf, hazard);
    const auto output = testing::internal::GetCapturedStderr();
    unsigned lines = 0;
    for (size_t pos = 0; (pos = output.find("memory-wait:", pos)) != std::string::npos; ++pos)
      ++lines;
    EXPECT_EQ(lines, 17u);
    EXPECT_NE(output.find("further diagnostics on this CU are suppressed"), std::string::npos);
    EXPECT_EQ(cu->memory_wait_diagnostic_count(), 20u);
    EXPECT_EQ(cu->xcnt_diagnostic_count(), 0u);
    units.push_back(std::move(cu));
  }
  for (const auto &cu : units)
    EXPECT_EQ(cu->memory_wait_diagnostic_count(), 20u);
}

TEST(MemoryWaitFootprintTest, DirectAluFamiliesMatchObservedAccesses) {
  std::vector<std::vector<uint32_t>> cases;
  auto add = [&](auto words) { cases.emplace_back(words.begin(), words.end()); };
  add(cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 256, .vdst = 12}));
  add(cdna4::build_vop1(cdna4::kVCvtF32U32Vop1, {.src0 = 256, .vdst = 12}));
  for (auto opcode : {cdna4::kVMinF32Vop2, cdna4::kVMaxF32Vop2, cdna4::kVCndmaskB32Vop2,
                      cdna4::kVAddCoU32Vop2, cdna4::kVFmacF32Vop2})
    add(cdna4::build_vop2(opcode, {.src0 = 256, .vsrc1 = 4, .vdst = 12}));
  add(cdna4::build_vop3(cdna4::kVFmaF32Vop3, {.vdst = 12, .src0 = 256, .src1 = 260, .src2 = 264}));
  add(cdna4::build_vopc(cdna4::kVCmpLtF32Vopc, {.src0 = 256, .vsrc1 = 4}));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  for (const auto &words : cases) {
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    EXPECT_TRUE(decoded.value()->flags() & DIRECT_REGISTER_ACCESSES) << decoded.value()->mnemonic();
  }
  check_vector_footprints(ROCJITSU_CODE_ARCH_CDNA4, 64, 16, cases);
}

TEST(MemoryWaitFootprintTest, RestrictedPackedAndScalarBranchSourcesKeepVccWords) {
  GpuMemory memory("restricted_memory");
  L2Cache l2("restricted_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_CDNA4;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 32;
  auto cu = ComputeUnitCore::create("restricted_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 104, 32, 64);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  auto decoder = Decoder::create(config.arch);
  for (bool scalar : {false, true}) {
    std::vector<uint32_t> words;
    if (scalar)
      append_instruction(
          words, cdna4::build_sop2(cdna4::kSCbranchGForkSop2, {.ssrc0 = 106, .ssrc1 = 128}));
    else
      append_instruction(words,
                         cdna4::build_vop3p(cdna4::kVPkMadI16Vop3p,
                                            {.vdst = 4, .src0 = 128, .src1 = 106, .src2 = 107}));
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    for (uint16_t word : {0, 1}) {
      auto &state = wf->ensure_memory_wait_scoreboard();
      state.clear();
      unsigned reports = 0;
      state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
      state.add({state.issue(WaitCounterKind::Ds, true),
                 0x100,
                 ~0ull,
                 {RegClass::VCC, word, 1},
                 WaitCounterKind::Ds,
                 0xf});
      state.check_instruction(*decoded.value(), *wf);
      EXPECT_NE(reports, 0u) << decoded.value()->mnemonic() << " word=" << word;
    }
  }
}

TEST(MemoryWaitFootprintTest, DirectLdsChecksItsExecDependency) {
  GpuMemory memory("lds_exec_memory");
  L2Cache l2("lds_exec_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_RDNA3;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 32;
  config.vgprs_per_wf = 32;
  auto cu = ComputeUnitCore::create("lds_exec_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 32, 32, 32);
  ASSERT_NE(wf, nullptr);
  auto decoder = Decoder::create(config.arch);
  for (auto opcode : {rdna3::kLdsDirectLoadLdsdir, rdna3::kLdsParamLoadLdsdir}) {
    const auto words = rdna3::build_ldsdir(opcode, {.vdst = 4});
    util::StringDiagnostic error;
    auto decoded = decoder->decode_window(words, 0, error.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error.message();
    for (uint64_t exec : {0ull, 1ull}) {
      wf->set_exec(exec);
      auto &state = wf->ensure_memory_wait_scoreboard();
      state.clear();
      unsigned reports = 0;
      state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
      // Synthetic readiness checks the implicit dependency; SMEM cannot target EXEC.
      state.add({state.issue(WaitCounterKind::Ds, true),
                 0x100,
                 ~0ull,
                 {RegClass::EXEC, 0, 1},
                 WaitCounterKind::Ds,
                 0xf});
      state.check_instruction(*decoded.value(), *wf);
      EXPECT_NE(reports, 0u) << decoded.value()->mnemonic();
    }
  }
}

TEST(XcntExecutionTest, WordwiseBufferDataRetainsItsBackedReplaySource) {
  GpuMemory memory("buffer_prefix_memory");
  L2Cache l2("buffer_prefix_l2");
  ComputeUnitCore::Config config{};
  config.memory_wait_diagnostics = MemoryWaitDiagnostics::Warn;
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.xcnt_diagnostics = MemoryWaitDiagnostics::Warn;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 128;
  config.vgprs_per_wf = 32;
  auto cu = ComputeUnitCore::create("buffer_prefix_cu", config, &memory, &l2);
  auto *wf = cu->dispatch_wf(0, 0x100, 104, 32, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(cu->vgpr_allocation_block_size(), 32u);
  wf->set_mode_raw(1u << 25);
  auto decoder = Decoder::create(config.arch);
  const auto words =
      cdna5::build_vbuffer(cdna5::kBufferStoreB64Vbuffer, {.soffset = 124, .vdata = 31});
  util::StringDiagnostic error;
  auto decoded = decoder->decode_window(words, 0, error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  for (uint64_t exec : {0ull, 1ull, 2ull}) {
    wf->set_exec(exec);
    auto &state = wf->ensure_memory_wait_scoreboard();
    state.clear();
    unsigned reports = 0;
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    state.add({state.issue(WaitCounterKind::Load),
               0x80,
               1,
               {RegClass::VGPR, 31, 1},
               WaitCounterKind::Load,
               0xf});
    state.check_instruction(*decoded.value(), *wf);
    EXPECT_EQ(reports != 0, bool(exec & 1));
    state.clear();
    cu->track_memory_wait(*decoded.value(), *wf);
    uint64_t recorded = 0;
    for (const auto &event : state.events())
      if (event.counter == WaitCounterKind::X && event.reg.cls == RegClass::VGPR) {
        EXPECT_EQ(event.reg.index, 31u);
        EXPECT_EQ(event.reg.width, 1u);
        recorded |= event.lanes;
      }
    EXPECT_EQ(recorded, exec);
  }
}
