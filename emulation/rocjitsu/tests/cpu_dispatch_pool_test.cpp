// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/cpu_dispatch_pool.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
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
  DispatchPoolFixture fixture(/*cu_count=*/1, /*functional_quantum=*/0);
  fixture.memory.write32(kProgramBase + 2 * sizeof(uint32_t), kSEndpgm);

  auto result = fixture.cus.front()->run_quantum();

  EXPECT_TRUE(result.ran);
  EXPECT_TRUE(fixture.cus.front()->is_idle());
}

TEST(CpuDispatchPoolTest, WorkerExceptionsRethrowAndPoolRemainsReusable) {
  DispatchPoolFixture fixture(/*cu_count=*/64);
  amdgpu::CpuDispatchPool pool(/*threads=*/8);

  fixture.memory.write32(kProgramBase, kSSetvskip);
  EXPECT_THROW(pool.run(std::span<amdgpu::ComputeUnitCore *>(fixture.tasks), /*threads=*/8),
               std::exception);

  fixture.memory.write32(kProgramBase, kSNop);
  for (auto *wf : fixture.wfs)
    wf->pc = kProgramBase;
  EXPECT_NO_THROW(pool.run(std::span<amdgpu::ComputeUnitCore *>(fixture.tasks), /*threads=*/8));
  for (auto *wf : fixture.wfs)
    EXPECT_EQ(wf->pc, kProgramBase + sizeof(uint32_t));
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

} // namespace
