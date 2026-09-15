// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/cpu_dispatch_pool.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "throwing_instruction_test_util.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
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

class SubmissionOverlapPlugin final : public ExecutionPlugin {
public:
  explicit SubmissionOverlapPlugin(uint32_t expected_overlap)
      : ExecutionPlugin("submission_overlap"), expected_overlap_(expected_overlap) {}

  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &,
                                        amdgpu::Wavefront &) override {
    std::unique_lock<std::mutex> lock(mutex_);
    ++active_callbacks_;
    max_active_callbacks_ = std::max(max_active_callbacks_, active_callbacks_);
    if (active_callbacks_ >= expected_overlap_) {
      overlap_reached_ = true;
      cv_.notify_all();
    } else if (!cv_.wait_for(lock, std::chrono::seconds(2),
                             [this]() { return overlap_reached_; })) {
      timed_out_ = true;
      overlap_reached_ = true;
      cv_.notify_all();
    }
    --active_callbacks_;
  }

  bool timed_out() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return timed_out_;
  }

  uint32_t max_active_callbacks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_active_callbacks_;
  }

private:
  const uint32_t expected_overlap_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  uint32_t active_callbacks_ = 0;
  uint32_t max_active_callbacks_ = 0;
  bool overlap_reached_ = false;
  bool timed_out_ = false;
};

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

TEST(CpuDispatchPoolTest, ConcurrentSubmissionsShareWorkerCapacity) {
  constexpr uint32_t kSubmissionCount = 8;
  constexpr uint32_t kPoolThreads = 4;
  DispatchPoolFixture fixture(kSubmissionCount);
  amdgpu::CpuDispatchPool pool(kPoolThreads);

  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto overlap_plugin = std::make_unique<SubmissionOverlapPlugin>(kPoolThreads);
  auto *overlap = overlap_plugin.get();
  ASSERT_TRUE(group->add(std::move(overlap_plugin)));
  for (auto &cu : fixture.cus)
    cu->set_plugin_group(group);

  std::barrier start(static_cast<std::ptrdiff_t>(kSubmissionCount + 1));
  std::array<std::exception_ptr, kSubmissionCount> errors{};
  std::vector<std::jthread> submitters;
  submitters.reserve(kSubmissionCount);
  for (uint32_t i = 0; i < kSubmissionCount; ++i) {
    submitters.emplace_back([&, i]() {
      start.arrive_and_wait();
      try {
        pool.run(std::span<amdgpu::ComputeUnitCore *>(&fixture.tasks[i], 1), kPoolThreads);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  start.arrive_and_wait();
  submitters.clear();

  for (const auto &error : errors)
    EXPECT_EQ(error, nullptr);
  EXPECT_FALSE(overlap->timed_out());
  EXPECT_EQ(overlap->max_active_callbacks(), kPoolThreads);
  for (const auto *wf : fixture.wfs)
    EXPECT_EQ(wf->trace_inst_count_, 1u);
}

TEST(CpuDispatchPoolTest, ConcurrentSubmissionsKeepExceptionsIndependent) {
  DispatchPoolFixture fixture(/*cu_count=*/2);
  amdgpu::CpuDispatchPool pool(/*threads=*/2);
  fixture.memory.write32(kProgramBase, kSMovB32);

  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::make_unique<test::ThrowingInstructionPlugin>()));
  fixture.cus[0]->set_plugin_group(group);

  std::barrier start(3);
  std::array<std::exception_ptr, 2> errors{};
  std::array<amdgpu::FunctionalQuantumResult, 2> results{};
  std::vector<std::jthread> submitters;
  submitters.reserve(2);
  for (size_t i = 0; i < 2; ++i) {
    submitters.emplace_back([&, i]() {
      start.arrive_and_wait();
      try {
        pool.run(std::span<amdgpu::ComputeUnitCore *>(&fixture.tasks[i], 1), /*threads=*/2,
                 std::span<amdgpu::FunctionalQuantumResult>(&results[i], 1));
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  start.arrive_and_wait();
  submitters.clear();

  EXPECT_NE(errors[0], nullptr);
  EXPECT_EQ(errors[1], nullptr);
  EXPECT_FALSE(results[0].ran);
  EXPECT_TRUE(results[1].ran);
  EXPECT_EQ(fixture.wfs[1]->pc, kProgramBase + sizeof(uint32_t));
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
