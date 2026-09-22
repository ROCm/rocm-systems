// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/cpu_dispatch_pool.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "throwing_instruction_test_util.h"

#include <gtest/gtest.h>

#include <array>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocjitsu::amdgpu {

class CpuDispatchPoolTestAccess {
public:
  static void construct_with_failure(uint32_t threads, uint32_t fail_after) {
    CpuDispatchPool pool(threads, fail_after);
  }
};

} // namespace rocjitsu::amdgpu

namespace {

using namespace rocjitsu;

constexpr uint32_t kSNop = 0xBF800000u;
constexpr uint32_t kSMovB32 = 0xBE800000u; // s_mov_b32 s0, s0
constexpr uint32_t kSSetvskip = 0xBF100000u;
constexpr uint64_t kProgramBase = 0x100000;

struct DispatchPoolFixture {
  explicit DispatchPoolFixture(uint32_t cu_count, uint32_t functional_quantum = 1) : l2("pool_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 104;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    cfg.functional_quantum = functional_quantum;

    for (uint32_t i = 0; i < 256; ++i)
      memory.write32(kProgramBase + i * sizeof(uint32_t), kSNop);

    cus.reserve(cu_count);
    tasks.reserve(cu_count);
    wfs.reserve(cu_count);
    for (uint32_t i = 0; i < cu_count; ++i) {
      auto cu = amdgpu::ComputeUnitCore::create("pool_cu" + std::to_string(i), cfg, &memory, &l2);
      auto *wf = cu->dispatch_wf(/*wg_id=*/i, kProgramBase, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
      EXPECT_NE(wf, nullptr);
      tasks.push_back(cu.get());
      wfs.push_back(wf);
      cus.push_back(std::move(cu));
    }
  }

  amdgpu::GpuMemory memory{"pool_memory"};
  amdgpu::L2Cache l2;
  std::vector<std::unique_ptr<amdgpu::ComputeUnitCore>> cus;
  std::vector<amdgpu::ComputeUnitCore *> tasks;
  std::vector<amdgpu::Wavefront *> wfs;
};

// Hold CUs at an observable instruction callback so tests can count overlapping
// execution and keep submissions open until every waiter is explicitly released.
class GatedInstructionPlugin final : public ExecutionPlugin {
public:
  GatedInstructionPlugin() : ExecutionPlugin("gated_instruction") {}

  void onAmdgpuReadSgpr(const amdgpu::Wavefront *, uint32_t) override {
    std::unique_lock lock(mutex_);
    ++entered_;
    changed_.notify_all();
    changed_.wait(lock, [this] { return released_; });
  }

  // Checks after launching gated work must stay nonfatal: even a timeout must
  // be followed by release() before a future is destroyed and joins its task.
  bool wait_for_entries(size_t count, std::chrono::milliseconds timeout = std::chrono::seconds(2)) {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, timeout, [&] { return entered_ >= count; });
  }

  void release() {
    std::lock_guard lock(mutex_);
    released_ = true;
    changed_.notify_all();
  }

private:
  std::mutex mutex_;
  std::condition_variable changed_;
  size_t entered_ = 0;
  bool released_ = false;
};

GatedInstructionPlugin *gate_fixture(DispatchPoolFixture &fixture) {
  fixture.memory.write32(kProgramBase, kSMovB32);
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<GatedInstructionPlugin>();
  auto *gate = plugin.get();
  if (!group->add(std::move(plugin)))
    throw std::runtime_error("failed to install gated instruction plugin");
  for (auto &cu : fixture.cus)
    cu->set_plugin_group(group);
  return gate;
}

TEST(CpuDispatchPoolTest, ConcurrentSubmissionsUseWorkersAndCompleteIndependently) {
  DispatchPoolFixture first(/*cu_count=*/2), second(/*cu_count=*/2);
  auto *first_gate = gate_fixture(first);
  auto *second_gate = gate_fixture(second);
  amdgpu::CpuDispatchPool pool(/*threads=*/4);
  std::array<amdgpu::FunctionalQuantumResult, 2> first_results{}, second_results{};
  auto first_run = std::async(std::launch::async,
                              [&] { return pool.run(first.tasks, /*threads=*/2, first_results); });
  // One CU blocks its caller and one blocks a worker, leaving other pool workers free.
  EXPECT_TRUE(first_gate->wait_for_entries(2));
  auto second_run = std::async(
      std::launch::async, [&] { return pool.run(second.tasks, /*threads=*/2, second_results); });
  EXPECT_TRUE(second_gate->wait_for_entries(2));
  second_gate->release();
  const auto second_status = second_run.wait_for(std::chrono::seconds(2));
  first_gate->release();
  EXPECT_EQ(second_status, std::future_status::ready);
  EXPECT_TRUE(first_run.get().ran);
  EXPECT_TRUE(second_run.get().ran);
  for (const auto &result : first_results)
    EXPECT_EQ(result.iterations, 1u);
  for (const auto &result : second_results)
    EXPECT_EQ(result.iterations, 1u);
}

TEST(CpuDispatchPoolTest, ConcurrentFailureStaysWithItsSubmission) {
  DispatchPoolFixture failing(/*cu_count=*/3), succeeding(/*cu_count=*/4);
  auto *gate = gate_fixture(failing);
  auto throwing = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(throwing->add(std::make_unique<test::ThrowingInstructionPlugin>()));
  failing.cus[0]->set_plugin_group(throwing);
  amdgpu::CpuDispatchPool pool(/*threads=*/4);
  auto failed_run = std::async(std::launch::async, [&] { pool.run(failing.tasks, /*threads=*/2); });
  // With two executors and three CUs, reaching both gates proves one executor
  // has already caught the first CU's exception and advanced to another task.
  // The exception stays pending until these two remaining CUs are released.
  EXPECT_TRUE(gate->wait_for_entries(2));
  auto successful_run =
      std::async(std::launch::async, [&] { pool.run(succeeding.tasks, /*threads=*/2); });
  const auto successful_status = successful_run.wait_for(std::chrono::seconds(2));
  // Complete the unrelated join while the failing caller cannot yet consume
  // its exception. A shared exception slot would leak or lose that exception.
  if (successful_status == std::future_status::ready) {
    EXPECT_NO_THROW(successful_run.get());
  }
  gate->release();
  EXPECT_EQ(successful_status, std::future_status::ready);
  EXPECT_THROW(failed_run.get(), std::runtime_error);
  if (successful_run.valid()) {
    EXPECT_NO_THROW(successful_run.get());
  }
  for (auto *wf : succeeding.wfs)
    EXPECT_EQ(wf->pc, kProgramBase + sizeof(uint32_t));
  EXPECT_NO_THROW(pool.run(succeeding.tasks, /*threads=*/4));
}

TEST(CpuDispatchPoolTest, SubmissionsRespectRequestedWidthWithIdleWorkers) {
  amdgpu::CpuDispatchPool pool(/*threads=*/4);
  for (uint32_t width : {1u, 2u}) {
    DispatchPoolFixture fixture(/*cu_count=*/8);
    auto *gate = gate_fixture(fixture);
    auto run = std::async(std::launch::async, [&] { pool.run(fixture.tasks, /*threads=*/width); });
    EXPECT_TRUE(gate->wait_for_entries(width));
    // All entered CUs remain blocked. Extra entries would require exceeding the
    // requested width, even though the pool has more idle workers available.
    const bool exceeded_width = gate->wait_for_entries(width + 1, std::chrono::milliseconds(100));
    gate->release();
    EXPECT_FALSE(exceeded_width) << "requested width " << width;
    EXPECT_NO_THROW(run.get());
    for (auto *wf : fixture.wfs)
      EXPECT_EQ(wf->pc, kProgramBase + sizeof(uint32_t));
  }
}

TEST(CpuDispatchPoolTest, ConcurrentReusedSubmissionsRunEveryCuOnce) {
  constexpr size_t kSubmitters = 8;
  constexpr uint32_t kRounds = 32;
  std::array<std::unique_ptr<DispatchPoolFixture>, kSubmitters> fixtures;
  for (auto &fixture : fixtures)
    fixture = std::make_unique<DispatchPoolFixture>(/*cu_count=*/8);
  amdgpu::CpuDispatchPool pool(/*threads=*/4);
  std::barrier start(static_cast<std::ptrdiff_t>(kSubmitters));
  std::vector<std::future<void>> runs;
  for (size_t i = 0; i < kSubmitters; ++i) {
    runs.push_back(std::async(std::launch::async, [&, i] {
      start.arrive_and_wait();
      for (uint32_t round = 0; round < kRounds; ++round)
        pool.run(fixtures[i]->tasks, /*threads=*/1 + (round + i) % 4);
    }));
  }
  for (auto &run : runs)
    EXPECT_NO_THROW(run.get());
  for (const auto &fixture : fixtures) {
    for (auto *wf : fixture->wfs) {
      EXPECT_EQ(wf->trace_inst_count_, kRounds);
      EXPECT_EQ(wf->pc, kProgramBase + kRounds * sizeof(uint32_t));
    }
  }
}

TEST(CpuDispatchPoolTest, ReusedBatchesRunEachCuOnceAtRequestedThreadCounts) {
  DispatchPoolFixture fixture(/*cu_count=*/8);
  amdgpu::CpuDispatchPool pool(/*threads=*/8);

  constexpr std::array<uint32_t, 6> kThreadCounts = {1, 2, 8, 3, 8, 1};
  uint32_t expected_quanta = 0;
  for (uint32_t repeat = 0; repeat < 8; ++repeat) {
    for (uint32_t thread_count : kThreadCounts) {
      pool.run(std::span<amdgpu::ComputeUnitCore *>(fixture.tasks), thread_count);
      ++expected_quanta;

      for (auto *wf : fixture.wfs) {
        EXPECT_EQ(wf->trace_inst_count_, expected_quanta);
        EXPECT_EQ(wf->pc, kProgramBase + expected_quanta * sizeof(uint32_t));
      }
    }
  }
}

TEST(CpuDispatchPoolTest, ZeroThreadsFallsBackToCallingThread) {
  DispatchPoolFixture fixture(/*cu_count=*/1);
  amdgpu::CpuDispatchPool pool(/*threads=*/0);

  EXPECT_EQ(pool.thread_count(), 1u);
  pool.run(std::span<amdgpu::ComputeUnitCore *>(fixture.tasks), /*threads=*/0);
  EXPECT_EQ(fixture.wfs.front()->trace_inst_count_, 1u);
}

TEST(CpuDispatchPoolTest, ZeroFunctionalQuantumRunsUntilWavefrontHalts) {
  constexpr uint32_t kSEndpgm = 0xBF810000u;
  constexpr uint64_t kSecondProgramBase = kProgramBase + 128 * sizeof(uint32_t);
  DispatchPoolFixture fixture(/*cu_count=*/2, /*functional_quantum=*/0);
  fixture.memory.write32(kProgramBase + 2 * sizeof(uint32_t), kSEndpgm);
  fixture.memory.write32(kSecondProgramBase + 3 * sizeof(uint32_t), kSEndpgm);
  fixture.wfs[1]->pc = kSecondProgramBase;

  amdgpu::CpuDispatchPool pool(/*threads=*/2);
  std::array<amdgpu::FunctionalQuantumResult, 2> per_cu{};
  auto result = pool.run(std::span<amdgpu::ComputeUnitCore *>(fixture.tasks), /*threads=*/2,
                         std::span<amdgpu::FunctionalQuantumResult>(per_cu));

  EXPECT_TRUE(result.ran);
  EXPECT_FALSE(result.yielded);
  EXPECT_EQ(result.iterations, 3u);
  EXPECT_TRUE(per_cu[0].ran);
  EXPECT_FALSE(per_cu[0].yielded);
  EXPECT_EQ(per_cu[0].iterations, 2u);
  EXPECT_TRUE(per_cu[1].ran);
  EXPECT_FALSE(per_cu[1].yielded);
  EXPECT_EQ(per_cu[1].iterations, 3u);
  EXPECT_TRUE(fixture.cus[0]->is_idle());
  EXPECT_TRUE(fixture.cus[1]->is_idle());

  std::array<amdgpu::FunctionalQuantumResult, 1> wrong_size{};
  EXPECT_THROW(pool.run(std::span<amdgpu::ComputeUnitCore *>(fixture.tasks), /*threads=*/2,
                        std::span<amdgpu::FunctionalQuantumResult>(wrong_size)),
               std::invalid_argument);
}

TEST(CpuDispatchPoolTest, WorkerExceptionsRethrowAndPoolRemainsReusable) {
  DispatchPoolFixture fixture(/*cu_count=*/64);
  amdgpu::CpuDispatchPool pool(/*threads=*/8);
  fixture.memory.write32(kProgramBase, kSMovB32);

  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::make_unique<test::ThrowingInstructionPlugin>()));
  for (auto &cu : fixture.cus)
    cu->set_plugin_group(group);
  EXPECT_THROW(pool.run(std::span<amdgpu::ComputeUnitCore *>(fixture.tasks), /*threads=*/8),
               std::exception);

  for (auto &cu : fixture.cus)
    cu->set_plugin_group(nullptr);
  for (auto *wf : fixture.wfs)
    wf->pc = kProgramBase;
  EXPECT_NO_THROW(pool.run(std::span<amdgpu::ComputeUnitCore *>(fixture.tasks), /*threads=*/8));
  for (auto *wf : fixture.wfs)
    EXPECT_EQ(wf->pc, kProgramBase + sizeof(uint32_t));
}

TEST(CpuDispatchPoolTest, OneThreadFinishesBatchBeforeRethrowing) {
  DispatchPoolFixture fixture(/*cu_count=*/4);
  amdgpu::CpuDispatchPool pool(/*threads=*/4);
  fixture.memory.write32(kProgramBase, kSMovB32);
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::make_unique<test::ThrowingInstructionPlugin>()));
  fixture.cus[0]->set_plugin_group(group);

  EXPECT_THROW(pool.run(std::span<amdgpu::ComputeUnitCore *>(fixture.tasks), /*threads=*/1),
               std::exception);
  for (size_t i = 1; i < fixture.wfs.size(); ++i)
    EXPECT_EQ(fixture.wfs[i]->pc, kProgramBase + sizeof(uint32_t));
}

TEST(CpuDispatchPoolTest, UnimplementedInstructionsHaltWithoutThrowing) {
  DispatchPoolFixture fixture(/*cu_count=*/8);
  amdgpu::CpuDispatchPool pool(/*threads=*/4);
  fixture.memory.write32(kProgramBase, kSSetvskip);

  EXPECT_NO_THROW(pool.run(std::span<amdgpu::ComputeUnitCore *>(fixture.tasks), /*threads=*/4));
  for (const auto &cu : fixture.cus)
    EXPECT_TRUE(cu->is_idle());
}

TEST(CpuDispatchPoolTest, DestroyJoinsParkedWorkers) {
  for (uint32_t i = 0; i < 100; ++i) {
    amdgpu::CpuDispatchPool pool(/*threads=*/8);
    EXPECT_EQ(pool.thread_count(), 8u);
  }
}

TEST(CpuDispatchPoolTest, PartialConstructionJoinsParkedWorkers) {
  EXPECT_THROW(amdgpu::CpuDispatchPoolTestAccess::construct_with_failure(/*threads=*/8,
                                                                         /*fail_after=*/2),
               std::runtime_error);
}

// A caller must not steal pending work from another submission: that work can
// be waiting for the caller's own submission to complete.
TEST(CpuDispatchPoolTest, SubmittingThreadRunsOnlyItsOwnSubmission) {
  DispatchPoolFixture foreign(/*cu_count=*/3), own(/*cu_count=*/1);
  auto *gate = gate_fixture(foreign);
  amdgpu::CpuDispatchPool pool(/*threads=*/2);
  auto foreign_run = std::async(std::launch::async, [&] { pool.run(foreign.tasks, 2); });
  // Occupy the worker and foreign caller while one foreign CU remains pending.
  EXPECT_TRUE(gate->wait_for_entries(2));
  auto own_run = std::async(std::launch::async, [&] { pool.run(own.tasks, 1); });
  const auto own_status = own_run.wait_for(std::chrono::seconds(2));
  gate->release();
  EXPECT_EQ(own_status, std::future_status::ready);
  EXPECT_NO_THROW(own_run.get());
  EXPECT_NO_THROW(foreign_run.get());
  EXPECT_EQ(own.wfs.front()->trace_inst_count_, 1u);
  for (const auto *wf : foreign.wfs)
    EXPECT_EQ(wf->trace_inst_count_, 1u);
}

// Measures the sparse-XCD shape that motivated concurrent submissions: eight
// command processors share one SoC pool, but each has only a few runnable CUs.
// This is deliberately excluded from the default CTest run with the other
// *Benchmark* tests. Invoke it directly and compare identical builds/configs.
TEST(CpuDispatchPoolBenchmark, SparseConcurrentSubmissions) {
  constexpr uint32_t kSubmissions = 8;
  constexpr uint32_t kPoolThreads = 32;
  constexpr uint32_t kFunctionalQuantum = 100000;
  constexpr uint32_t kWarmupRounds = 1;
  constexpr uint32_t kMeasuredRounds = 3;
  constexpr uint32_t kSBranchSelf = 0xBF82FFFFu; // s_branch -1

  for (uint32_t cus_per_submission : {1u, 2u, 4u, 8u, 16u, 32u}) {
    amdgpu::CpuDispatchPool pool(kPoolThreads);
    std::vector<std::unique_ptr<DispatchPoolFixture>> fixtures;
    fixtures.reserve(kSubmissions);
    for (uint32_t i = 0; i < kSubmissions; ++i) {
      auto fixture = std::make_unique<DispatchPoolFixture>(cus_per_submission, kFunctionalQuantum);
      fixture->memory.write32(kProgramBase, kSBranchSelf);
      fixtures.push_back(std::move(fixture));
    }

    std::barrier start_round(static_cast<std::ptrdiff_t>(kSubmissions + 1));
    std::barrier finish_round(static_cast<std::ptrdiff_t>(kSubmissions + 1));
    std::vector<std::jthread> submitters;
    submitters.reserve(kSubmissions);
    for (uint32_t i = 0; i < kSubmissions; ++i) {
      submitters.emplace_back([&, i]() {
        for (uint32_t round = 0; round < kWarmupRounds + kMeasuredRounds; ++round) {
          start_round.arrive_and_wait();
          pool.run(std::span<amdgpu::ComputeUnitCore *>(fixtures[i]->tasks), kPoolThreads);
          finish_round.arrive_and_wait();
        }
      });
    }

    start_round.arrive_and_wait();
    finish_round.arrive_and_wait();
    const auto begin = std::chrono::steady_clock::now();
    for (uint32_t round = 0; round < kMeasuredRounds; ++round) {
      start_round.arrive_and_wait();
      finish_round.arrive_and_wait();
    }
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
    const uint64_t instructions = static_cast<uint64_t>(kSubmissions) * cus_per_submission *
                                  kFunctionalQuantum * kMeasuredRounds;
    std::printf("  submissions=%u cus/submission=%u pool_threads=%u elapsed_ms=%.3f "
                "throughput_minst_s=%.3f\n",
                kSubmissions, cus_per_submission, kPoolThreads, elapsed_ms,
                static_cast<double>(instructions) / (elapsed_ms * 1000.0));

    const uint64_t expected_instructions =
        static_cast<uint64_t>(kFunctionalQuantum) * (kWarmupRounds + kMeasuredRounds);
    for (const auto &fixture : fixtures)
      for (const auto *wf : fixture->wfs)
        EXPECT_EQ(wf->trace_inst_count_, expected_instructions);
  }
}

} // namespace
