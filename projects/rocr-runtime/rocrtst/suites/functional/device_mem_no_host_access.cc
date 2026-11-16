/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <setjmp.h>
#include <signal.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "suites/functional/device_mem_no_host_access.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

constexpr size_t kAllocSize = 2 * 1024 * 1024;
constexpr uint32_t kPattern = 0xA5A5A5A5u;

// In-process SIGSEGV detection. fork() cannot be used: libhsakmt maps BOs with
// MADV_DONTFORK, so a forked child would fault even in the default mode.
sigjmp_buf g_jmp_env;
volatile sig_atomic_t g_in_probe = 0;

void SegvHandler(int sig, siginfo_t*, void*) {
  if (g_in_probe) siglongjmp(g_jmp_env, 1);
  // Not ours: restore default action and re-raise.
  signal(sig, SIG_DFL);
  raise(sig);
}

// Perform one CPU write and one CPU read of *p. Returns true if a fault was
// taken. If no fault, *value_out holds the value read back.
bool CpuTouchFaults(volatile uint32_t* p, uint32_t* value_out) {
  struct sigaction sa, old_segv, old_bus;
  memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = SegvHandler;
  sa.sa_flags = SA_SIGINFO | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, &old_segv);
  sigaction(SIGBUS, &sa, &old_bus);

  bool faulted = false;
  if (sigsetjmp(g_jmp_env, 1) == 0) {
    g_in_probe = 1;
    *p = kPattern;
    *value_out = *p;
    g_in_probe = 0;
  } else {
    g_in_probe = 0;
    faulted = true;
  }

  sigaction(SIGSEGV, &old_segv, nullptr);
  sigaction(SIGBUS, &old_bus, nullptr);
  return faulted;
}

bool NoHostAccessRequested() {
  const char* v = getenv("HSA_DEVICE_MEM_NO_HOST_ACCESS");
  if (v == nullptr || *v == '\0') return false;
  std::string s(v);
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  // Same spellings as ROCr's Flag::IsEnvVarTruthy
  return !(s == "0" || s == "false" || s == "no" || s == "off" || s == "n" || s == "f");
}

// Select the coarse-grain global pool of a GPU agent (device-local memory).
hsa_status_t FindCoarseDevicePool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment;
  hsa_status_t err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (err != HSA_STATUS_SUCCESS) return err;
  if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

  uint32_t flags = 0;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (err != HSA_STATUS_SUCCESS) return err;
  if (!(flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)) return HSA_STATUS_SUCCESS;

  bool alloc_allowed = false;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                     &alloc_allowed);
  if (err != HSA_STATUS_SUCCESS) return err;
  if (!alloc_allowed) return HSA_STATUS_SUCCESS;

  *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
  return HSA_STATUS_INFO_BREAK;
}

}  // namespace

DeviceMemNoHostAccess::DeviceMemNoHostAccess() : TestBase() {
  set_num_iteration(1);
  set_title("RocR Device Memory No-Host-Access Test");
  set_description(
      "Validates HSA_DEVICE_MEM_NO_HOST_ACCESS: when set, CPU access to GPU "
      "device-local memory faults while GPU access keeps working; when unset, "
      "CPU access to host-visible device memory works.");
}

DeviceMemNoHostAccess::~DeviceMemNoHostAccess() {}

void DeviceMemNoHostAccess::SetUp(void) {
  TestBase::SetUp();  // Applies platform filtering; must be called.
  if (test_skipped_) return;

  hsa_status_t err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
}

void DeviceMemNoHostAccess::Run(void) {
  if (!rocrtst::CheckProfile(this)) return;
  TestBase::Run();
}

void DeviceMemNoHostAccess::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void DeviceMemNoHostAccess::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) return;
}

void DeviceMemNoHostAccess::Close(void) { TestBase::Close(); }

void DeviceMemNoHostAccess::CpuAccessTest(void) {
  hsa_status_t err;
  const bool no_host_access = NoHostAccessRequested();

  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(cpus.size(), 0u);

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  if (gpus.empty()) {
    std::cout << "[ SKIPPED ] No GPU agents." << std::endl;
    return;
  }

  const hsa_agent_t cpu = cpus[0];
  for (const hsa_agent_t gpu : gpus) {
    hsa_amd_memory_pool_t pool = {};
    err = hsa_amd_agent_iterate_memory_pools(gpu, FindCoarseDevicePool, &pool);
    if (err != HSA_STATUS_INFO_BREAK) {
      std::cout << "[ SKIPPED ] GPU agent has no coarse-grain device pool." << std::endl;
      continue;
    }

    hsa_amd_memory_pool_access_t access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    err = hsa_amd_agent_memory_pool_get_info(cpu, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                             &access);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    bool signal_capable = false;
    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIGNAL_CAPABLE,
                                       &signal_capable);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    if (no_host_access) {
      EXPECT_EQ(access, HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED)
          << "CPU must not have access to device pool when HSA_DEVICE_MEM_NO_HOST_ACCESS is set";
      EXPECT_FALSE(signal_capable)
          << "Device pool must not be signal capable when HSA_DEVICE_MEM_NO_HOST_ACCESS is set";
    } else if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      std::cout << "[ SKIPPED ] Device pool is not CPU accessible (small BAR)." << std::endl;
      continue;
    }

    uint32_t* buf = nullptr;
    err = hsa_amd_memory_pool_allocate(pool, kAllocSize, 0, reinterpret_cast<void**>(&buf));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    ASSERT_NE(buf, nullptr);

    // GPU-side access must always keep working: fill on the device, copy to
    // host memory and check the value.
    {
      hsa_amd_memory_pool_t sys_pool = {};
      err = hsa_amd_agent_iterate_memory_pools(cpu, rocrtst::FindStandardPool, &sys_pool);
      ASSERT_EQ(err, HSA_STATUS_INFO_BREAK);

      uint32_t* staging = nullptr;
      err =
          hsa_amd_memory_pool_allocate(sys_pool, kAllocSize, 0, reinterpret_cast<void**>(&staging));
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      memset(staging, 0, kAllocSize);
      err = hsa_amd_agents_allow_access(1, &gpu, nullptr, staging);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);

      err = hsa_amd_memory_fill(buf, kPattern, kAllocSize / sizeof(uint32_t));
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);

      hsa_signal_t copy_signal;
      err = hsa_signal_create(1, 0, nullptr, &copy_signal);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      err = hsa_amd_memory_async_copy(staging, cpu, buf, gpu, kAllocSize, 0, nullptr, copy_signal);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      while (hsa_signal_wait_acquire(copy_signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)(-1),
                                     HSA_WAIT_STATE_ACTIVE)) {
      }
      EXPECT_EQ(staging[0], kPattern);
      EXPECT_EQ(staging[kAllocSize / sizeof(uint32_t) - 1], kPattern);

      hsa_signal_destroy(copy_signal);
      hsa_amd_memory_pool_free(staging);
    }

    // CPU access: one write + one read, detecting a fault in-process.
    uint32_t readback = 0;
    const bool faulted = CpuTouchFaults(buf, &readback);
    if (no_host_access) {
      EXPECT_TRUE(faulted) << "CPU access to device memory should fault with "
                              "HSA_DEVICE_MEM_NO_HOST_ACCESS set";
    } else {
      EXPECT_FALSE(faulted) << "CPU access to host-visible device memory faulted";
      if (!faulted) EXPECT_EQ(readback, kPattern);
    }

    err = hsa_amd_memory_pool_free(buf);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
}
