// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/plugins/race_detector/plugin.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <optional>
#include <span>

namespace rocjitsu::test {
namespace {
using namespace amdgpu;
using namespace plugins::race_detector;

// A contract shared by both dynamic detectors. Keep expected counter families
// explicit instead of using the production conversion as the test oracle.
// Sample/BVH have no race-plugin memory-pipeline representation.
struct CounterDomain {
  WaitCounterKind core;
  std::optional<WaitCounterType> race;
};

constexpr CounterDomain kLoad{WaitCounterKind::Load, WaitCounterType::LOADCNT};
constexpr CounterDomain kDs{WaitCounterKind::Ds, WaitCounterType::DSCNT};

// Decode once per wait and give each adapter the actual ISA instruction.
// Event seeding below intentionally bypasses producer classification, so these
// tests isolate the wait/readiness contract from instruction execution coverage.
template <typename Consume> void with_wait(rj_code_arch_t arch, uint32_t word, Consume consume) {
  auto decoder = Decoder::create(arch);
  ASSERT_NE(decoder, nullptr);
  util::StringDiagnostic error;
  auto decoded = decoder->decode_window(std::span<const uint32_t>(&word, 1), 0, error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  ASSERT_TRUE(decoded.value()->is_waitcnt());
  consume(*decoded.value());
}

class CoreWaitBackend {
public:
  explicit CoreWaitBackend(rj_code_arch_t arch) : arch_(arch) {
    state_.bind(0x1000, &reports_,
                [](void *context, const auto &) { ++*static_cast<unsigned *>(context); });
  }
  bool ready() const { return true; }
  static bool supports(CounterDomain) { return true; }
  void load(CounterDomain domain, unsigned reg, bool unordered = false, uint64_t lanes = 1,
            uint8_t bytes = 0xf) {
    state_.add({state_.issue(domain.core, unordered),
                0x100 + reg * 4,
                lanes,
                {RegClass::VGPR, static_cast<uint16_t>(reg), 1},
                domain.core,
                bytes});
  }
  void counter_only(CounterDomain domain) { state_.issue(domain.core); }
  void wait(uint32_t word) {
    with_wait(arch_, word, [&](const Instruction &inst) { state_.before(inst, arch_); });
  }
  void expect_drained() const {
    EXPECT_TRUE(state_.empty());
    for (size_t i = 0; i < kWaitCounterCount; ++i)
      EXPECT_EQ(state_.outstanding(static_cast<WaitCounterKind>(i)), 0u);
    for (size_t i = 0; i < MemoryWaitShadow::kRegisters; ++i)
      EXPECT_FALSE(shadow_.test(i, true));
  }
  bool reports(unsigned reg, uint64_t lanes = 1, uint8_t bytes = 0xf, bool write = false) {
    const auto before = reports_;
    state_.access({RegClass::VGPR, static_cast<uint16_t>(reg), 1}, lanes, bytes, write);
    return reports_ != before;
  }

private:
  rj_code_arch_t arch_;
  MemoryWaitShadow shadow_;
  MemoryWaitScoreboard state_{shadow_};
  unsigned reports_ = 0;
};

class RaceWaitBackend {
public:
  explicit RaceWaitBackend(rj_code_arch_t arch) {
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    config.memory_wait_diagnostics = MemoryWaitDiagnostics::Off;
    cu_ = ComputeUnitCore::create("shared_wait_cu", config, &memory_, &l2_);
    if (!cu_)
      return;
    PluginSinkConfig sinks;
    sink_ = &sinks.emplace<StringSink>();
    plugins_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    auto plugin = std::make_unique<RaceDetectorPlugin>();
    auto *observer = plugin.get();
    EXPECT_TRUE(plugins_->add(std::move(plugin)));
    cu_->set_plugin_group(plugins_);
    plugins_->onInit();
    wave_ = cu_->dispatch_wf(0, 0x100, 104, 256);
    if (!wave_)
      return;
    wave_->set_exec(1);
    std::array<Wavefront *, 1> waves{wave_};
    plugins_->onAmdgpuWorkgroupDispatched(1, 0, 256, 104, waves);
    state_ =
        static_cast<RaceWavefrontState *>(wave_->plugin_state(observer->slot_index()))->race_state;
  }
  bool ready() const { return state_ != nullptr; }
  static bool supports(CounterDomain domain) { return domain.race.has_value(); }
  void load(CounterDomain domain, unsigned reg, bool unordered = false, uint64_t lanes = 1,
            uint8_t bytes = 0xf) {
    state_->registerEvent(0x100 + reg * 4, MemoryEventType::GLOBAL_TO_VGPR, {reg}, lanes, bytes,
                          *domain.race,
                          unordered ? MemoryOrderClass::UNORDERED : MemoryOrderClass::VMEM);
  }
  void counter_only(CounterDomain domain) {
    state_->registerEvent(0x800, MemoryEventType::VGPR_TO_GLOBAL, {}, 1, 0xf, *domain.race,
                          MemoryOrderClass::VMEM);
  }
  void wait(uint32_t word) {
    with_wait(cu_->arch(), word, [&](Instruction &inst) {
      EXPECT_TRUE(cu_->execute_instruction(&inst, *wave_).succeeded());
      plugins_->onAmdgpuAfterExecuteInstruction(wave_->pc, inst, *wave_);
    });
  }
  void expect_drained() const { EXPECT_TRUE(state_->getWaveMemoryEvents().empty()); }
  bool reports(unsigned reg, uint64_t lanes = 1, uint8_t bytes = 0xf, bool write = false) {
    const auto before = sink_->str().size();
    wave_->pc = 0x1000 + reg * 4;
    if (write)
      plugins_->onAmdgpuWriteVgprLanes(wave_, wave_->vgpr_alloc().base + reg, lanes, bytes);
    else
      plugins_->onAmdgpuReadVgprLanes(wave_, wave_->vgpr_alloc().base + reg, lanes, bytes);
    return sink_->str().size() != before;
  }

private:
  GpuMemory memory_{"shared_wait_memory"};
  L2Cache l2_{"shared_wait_l2"};
  std::shared_ptr<ExecutionPluginGroup> plugins_;
  std::unique_ptr<ComputeUnitCore> cu_;
  Wavefront *wave_ = nullptr;
  WaveRaceState *state_ = nullptr;
  StringSink *sink_ = nullptr;
};

template <typename Backend> class WaitHazardTest : public ::testing::Test {};
using WaitBackends = ::testing::Types<CoreWaitBackend, RaceWaitBackend>;
TYPED_TEST_SUITE(WaitHazardTest, WaitBackends);

TYPED_TEST(WaitHazardTest, TargetWaitEncodingsRetireTheirCounter) {
  struct Case {
    rj_code_arch_t arch;
    uint32_t word;
    CounterDomain counter;
  };
  const Case cases[] = {
      {ROCJITSU_CODE_ARCH_CDNA1, 0xbf8c0f70u, {WaitCounterKind::Load, WaitCounterType::VMCNT}},
      {ROCJITSU_CODE_ARCH_CDNA1, 0xbf8cc07fu, {WaitCounterKind::Ds, WaitCounterType::LGKMCNT}},
      {ROCJITSU_CODE_ARCH_CDNA2, 0xbf8c0f70u, {WaitCounterKind::Load, WaitCounterType::VMCNT}},
      {ROCJITSU_CODE_ARCH_CDNA2, 0xbf8cc07fu, {WaitCounterKind::Ds, WaitCounterType::LGKMCNT}},
      {ROCJITSU_CODE_ARCH_RDNA1, 0xbf8c3f70u, {WaitCounterKind::Load, WaitCounterType::VMCNT}},
      {ROCJITSU_CODE_ARCH_RDNA1, 0xbf8cc07fu, {WaitCounterKind::Ds, WaitCounterType::LGKMCNT}},
      {ROCJITSU_CODE_ARCH_RDNA1, 0xbbfd0000u, {WaitCounterKind::Store, WaitCounterType::VSCNT}},
      {ROCJITSU_CODE_ARCH_RDNA2, 0xbf8c3f70u, {WaitCounterKind::Load, WaitCounterType::VMCNT}},
      {ROCJITSU_CODE_ARCH_RDNA2, 0xbf8cc07fu, {WaitCounterKind::Ds, WaitCounterType::LGKMCNT}},
      {ROCJITSU_CODE_ARCH_RDNA2, 0xbbfd0000u, {WaitCounterKind::Store, WaitCounterType::VSCNT}},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc20000u, {WaitCounterKind::Sample, std::nullopt}},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc30000u, {WaitCounterKind::Bvh, std::nullopt}},
      {ROCJITSU_CODE_ARCH_CDNA3, 0xbf8c0f70u, {WaitCounterKind::Load, WaitCounterType::VMCNT}},
      {ROCJITSU_CODE_ARCH_CDNA3, 0xbf8cc07fu, {WaitCounterKind::Ds, WaitCounterType::LGKMCNT}},
      {ROCJITSU_CODE_ARCH_CDNA4, 0xbf8c0f70u, {WaitCounterKind::Load, WaitCounterType::VMCNT}},
      {ROCJITSU_CODE_ARCH_CDNA4, 0xbf8cc07fu, {WaitCounterKind::Ds, WaitCounterType::LGKMCNT}},
      {ROCJITSU_CODE_ARCH_RDNA3, 0xbf8903f7u, {WaitCounterKind::Load, WaitCounterType::LOADCNT}},
      {ROCJITSU_CODE_ARCH_RDNA3, 0xbf89fc07u, {WaitCounterKind::Ds, WaitCounterType::DSCNT}},
      {ROCJITSU_CODE_ARCH_RDNA3, 0xbc7c0000u, {WaitCounterKind::Store, WaitCounterType::STORECNT}},
      {ROCJITSU_CODE_ARCH_RDNA3_5, 0xbf8903f7u, {WaitCounterKind::Load, WaitCounterType::LOADCNT}},
      {ROCJITSU_CODE_ARCH_RDNA3_5, 0xbf89fc07u, {WaitCounterKind::Ds, WaitCounterType::DSCNT}},
      {ROCJITSU_CODE_ARCH_RDNA3_5,
       0xbc7c0000u,
       {WaitCounterKind::Store, WaitCounterType::STORECNT}},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc00000u, {WaitCounterKind::Load, WaitCounterType::LOADCNT}},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc10000u, {WaitCounterKind::Store, WaitCounterType::STORECNT}},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc60000u, {WaitCounterKind::Ds, WaitCounterType::DSCNT}},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc70000u, {WaitCounterKind::Km, WaitCounterType::KMCNT}},
      {ROCJITSU_CODE_ARCH_RDNA4, 0xbfc40000u, {WaitCounterKind::Exp, WaitCounterType::EXPCNT}},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfc00000u, {WaitCounterKind::Load, WaitCounterType::LOADCNT}},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfc10000u, {WaitCounterKind::Store, WaitCounterType::STORECNT}},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfc60000u, {WaitCounterKind::Ds, WaitCounterType::DSCNT}},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfc70000u, {WaitCounterKind::Km, WaitCounterType::KMCNT}},
      {ROCJITSU_CODE_ARCH_CDNA5, 0xbfca0000u, {WaitCounterKind::Async, WaitCounterType::ASYNCCNT}},
      {ROCJITSU_CODE_ARCH_CDNA5,
       0xbfcb0000u,
       {WaitCounterKind::Tensor, WaitCounterType::TENSORCNT}},
  };
  for (const auto &c : cases) {
    if (!TypeParam::supports(c.counter))
      continue;
    SCOPED_TRACE(static_cast<unsigned>(c.arch));
    SCOPED_TRACE(c.word);
    TypeParam backend(c.arch);
    ASSERT_TRUE(backend.ready());
    backend.load(c.counter, 5);
    backend.wait(c.word);
    backend.expect_drained();
    EXPECT_FALSE(backend.reports(5));
  }
}

TYPED_TEST(WaitHazardTest, FullDrainWaitsRetireEverySupportedCounter) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (uint32_t word : {0xbf890000u, 0xbf89fc07u, 0xbf89ffffu, 0xbf8a0000u}) {
      if (arch == ROCJITSU_CODE_ARCH_CDNA5 && word != 0xbf8a0000u)
        continue;
      SCOPED_TRACE(static_cast<unsigned>(arch));
      SCOPED_TRACE(word);
      TypeParam backend(arch);
      ASSERT_TRUE(backend.ready());
      const CounterDomain counters[] = {kLoad,
                                        {WaitCounterKind::Store, WaitCounterType::STORECNT},
                                        kDs,
                                        {WaitCounterKind::Km, WaitCounterType::KMCNT},
                                        {WaitCounterKind::Exp, WaitCounterType::EXPCNT},
                                        {WaitCounterKind::Sample, std::nullopt},
                                        {WaitCounterKind::Bvh, std::nullopt},
                                        {WaitCounterKind::Async, WaitCounterType::ASYNCCNT},
                                        {WaitCounterKind::Tensor, WaitCounterType::TENSORCNT}};
      unsigned reg = 0;
      for (const auto counter : counters) {
        if (!TypeParam::supports(counter))
          continue;
        if (arch == ROCJITSU_CODE_ARCH_RDNA4 &&
            (counter.core == WaitCounterKind::Async || counter.core == WaitCounterKind::Tensor))
          continue;
        if (arch == ROCJITSU_CODE_ARCH_CDNA5 &&
            (counter.core == WaitCounterKind::Sample || counter.core == WaitCounterKind::Bvh))
          continue;
        backend.load(counter, reg++, true);
      }
      backend.wait(word);
      backend.expect_drained();
      for (unsigned i = 0; i < reg; ++i)
        EXPECT_FALSE(backend.reports(i)) << "register " << i;
    }
  }
}

TYPED_TEST(WaitHazardTest, PartialWaitCountsOperationsWithoutRegisterResults) {
  TypeParam backend(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(backend.ready());
  backend.load(kLoad, 5);
  backend.counter_only(kLoad);
  backend.load(kLoad, 6);
  backend.wait(0xbfc00002u); // s_wait_loadcnt 2 releases the first of three operations.
  EXPECT_FALSE(backend.reports(5));
  EXPECT_TRUE(backend.reports(6));
}

TYPED_TEST(WaitHazardTest, UnrelatedWaitDoesNotReuseAnEarlierThreshold) {
  TypeParam backend(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(backend.ready());
  backend.wait(0xbfc00000u); // The wave's stored LOADCNT target is now zero.
  backend.load(kLoad, 5);
  backend.load(kDs, 6);
  backend.wait(0xbfc60000u); // s_wait_dscnt 0 must not apply the old LOADCNT target.
  EXPECT_TRUE(backend.reports(5));
  EXPECT_FALSE(backend.reports(6));
}

TYPED_TEST(WaitHazardTest, NoWaitSentinelPreservesPendingResults) {
  for (const uint32_t word : {0xbfc0003fu, 0xbfc83f3fu}) {
    SCOPED_TRACE(word);
    TypeParam backend(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_TRUE(backend.ready());
    backend.load(kLoad, 5);
    backend.wait(word);
    EXPECT_TRUE(backend.reports(5));
  }
}

TYPED_TEST(WaitHazardTest, UnorderedResultRequiresAZeroWait) {
  for (const uint32_t threshold : {0u, 1u}) {
    SCOPED_TRACE(threshold);
    TypeParam backend(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_TRUE(backend.ready());
    const CounterDomain scalar{WaitCounterKind::Km, WaitCounterType::KMCNT};
    backend.load(scalar, 5, true);
    backend.counter_only(scalar);
    backend.wait(0xbfc70000u | threshold);
    EXPECT_EQ(backend.reports(5), threshold != 0);
  }
}

TYPED_TEST(WaitHazardTest, RegisterLanesAndBytesKeepIndependentReadiness) {
  TypeParam backend(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(backend.ready());
  backend.load(kLoad, 5, false, 1, 0x3);
  EXPECT_FALSE(backend.reports(5, 2, 0xf));
  EXPECT_FALSE(backend.reports(5, 1, 0xc, true));
  EXPECT_TRUE(backend.reports(5, 1, 0x3));
}

} // namespace
} // namespace rocjitsu::test
