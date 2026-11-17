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
constexpr const char* kEnvName = "HSA_DEVICE_MEM_NO_HOST_ACCESS";

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

// Install the fault handler for one probe; returns the previous handlers.
void InstallHandlers(struct sigaction* old_segv, struct sigaction* old_bus) {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = SegvHandler;
  sa.sa_flags = SA_SIGINFO | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, old_segv);
  sigaction(SIGBUS, &sa, old_bus);
}

void RestoreHandlers(const struct sigaction& old_segv, const struct sigaction& old_bus) {
  sigaction(SIGSEGV, &old_segv, nullptr);
  sigaction(SIGBUS, &old_bus, nullptr);
}

// Perform one CPU write of value to *p. Returns true if a fault was taken.
bool CpuWriteFaults(volatile uint32_t* p, uint32_t value) {
  struct sigaction old_segv, old_bus;
  InstallHandlers(&old_segv, &old_bus);

  bool faulted = false;
  if (sigsetjmp(g_jmp_env, 1) == 0) {
    g_in_probe = 1;
    *p = value;
    g_in_probe = 0;
  } else {
    g_in_probe = 0;
    faulted = true;
  }

  RestoreHandlers(old_segv, old_bus);
  return faulted;
}

// Perform one CPU read of *p. Returns true if a fault was taken; otherwise
// *value_out holds the value read.
bool CpuReadFaults(volatile uint32_t* p, uint32_t* value_out) {
  struct sigaction old_segv, old_bus;
  InstallHandlers(&old_segv, &old_bus);

  bool faulted = false;
  if (sigsetjmp(g_jmp_env, 1) == 0) {
    g_in_probe = 1;
    *value_out = *p;
    g_in_probe = 0;
  } else {
    g_in_probe = 0;
    faulted = true;
  }

  RestoreHandlers(old_segv, old_bus);
  return faulted;
}

struct PoolInfo {
  hsa_amd_memory_pool_t pool;
  const char* kind;
};

// Collect all global pools of an agent that allow runtime allocation.
hsa_status_t CollectDevicePools(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment;
  hsa_status_t err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (err != HSA_STATUS_SUCCESS) return err;
  if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

  bool alloc_allowed = false;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                     &alloc_allowed);
  if (err != HSA_STATUS_SUCCESS) return err;
  if (!alloc_allowed) return HSA_STATUS_SUCCESS;

  uint32_t flags = 0;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (err != HSA_STATUS_SUCCESS) return err;

  const char* kind = "other";
  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
    kind = "coarse-grained";
  else if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED)
    kind = "extended-scope fine-grained";
  else if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
    kind = "fine-grained";

  static_cast<std::vector<PoolInfo>*>(data)->push_back({pool, kind});
  return HSA_STATUS_SUCCESS;
}

}  // namespace

DeviceMemNoHostAccess::DeviceMemNoHostAccess(bool no_host_access)
    : TestBase(), no_host_access_(no_host_access), env_changed_(false), saved_env_was_set_(false) {
  set_num_iteration(1);
  if (no_host_access_) {
    set_title("RocR Device Memory No-Host-Access Test (enabled)");
    set_description(
        "Sets HSA_DEVICE_MEM_NO_HOST_ACCESS=1 and validates that CPU access to GPU "
        "device-local memory faults while GPU access keeps working.");
  } else {
    set_title("RocR Device Memory No-Host-Access Test (default)");
    set_description(
        "Unsets HSA_DEVICE_MEM_NO_HOST_ACCESS and validates that CPU access to "
        "host-mapped GPU device-local memory works.");
  }
}

// Restore the environment here rather than in Close(): when the test is
// skipped, Close() is never called.
DeviceMemNoHostAccess::~DeviceMemNoHostAccess() {
  if (!env_changed_) return;
  if (saved_env_was_set_)
    setenv(kEnvName, saved_env_value_.c_str(), 1);
  else
    unsetenv(kEnvName);
  env_changed_ = false;
}

void DeviceMemNoHostAccess::SetUp(void) {
  // ROCr reads the variable in hsa_init(), invoked by TestBase::SetUp(), so the
  // environment must be set up first.
  const char* cur = getenv(kEnvName);
  saved_env_was_set_ = (cur != nullptr);
  if (cur != nullptr) saved_env_value_ = cur;
  env_changed_ = true;
  if (no_host_access_)
    setenv(kEnvName, "1", 1);
  else
    unsetenv(kEnvName);

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
  const bool no_host_access = no_host_access_;

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
    std::vector<PoolInfo> pools;
    err = hsa_amd_agent_iterate_memory_pools(gpu, CollectDevicePools, &pools);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    if (pools.empty()) {
      std::cout << "[ SKIPPED ] GPU agent has no testable device pools." << std::endl;
      continue;
    }

    for (const PoolInfo& pi : pools) {
      const hsa_amd_memory_pool_t pool = pi.pool;
      if (verbosity() > 0) {
        std::cout << "Testing " << pi.kind << " device pool ("
                  << (no_host_access ? "enabled" : "default") << " mode)" << std::endl;
      }

      // a. Pool-level queries.
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
            << pi.kind
            << " pool: CPU must not have access to device pool when "
               "HSA_DEVICE_MEM_NO_HOST_ACCESS is set (the variable may not have taken effect if "
               "the runtime was already initialized)";
        EXPECT_FALSE(signal_capable)
            << pi.kind
            << " pool: device pool must not be signal capable when "
               "HSA_DEVICE_MEM_NO_HOST_ACCESS is set (the variable may not have taken effect if "
               "the runtime was already initialized)";
      }

      // b. Allocate.
      uint32_t* buf = nullptr;
      err = hsa_amd_memory_pool_allocate(pool, kAllocSize, 0, reinterpret_cast<void**>(&buf));
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      ASSERT_NE(buf, nullptr);

      // c. Is the allocation host mapped?
      hsa_amd_pointer_info_t info;
      memset(&info, 0, sizeof(info));
      info.size = sizeof(info);
      err = hsa_amd_pointer_info(buf, &info, nullptr, nullptr, nullptr);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      const bool host_access =
          (info.alloc_flags & HSA_AMD_POINTER_INFO_ALLOC_FLAG_HOST_ACCESS) != 0;
      if (no_host_access) {
        EXPECT_FALSE(host_access) << pi.kind
                                  << " pool: allocation must not have HOST_ACCESS set when "
                                     "HSA_DEVICE_MEM_NO_HOST_ACCESS is set";
      } else if (!host_access) {
        // E.g. small-BAR dGPU: pool is not host mapped even though the pool ACCESS enum reports
        // DISALLOWED_BY_DEFAULT rather than NEVER_ALLOWED.
        std::cout << "[ SKIPPED ] " << pi.kind << " device pool is not host mapped (small BAR)."
                  << std::endl;
        err = hsa_amd_memory_pool_free(buf);
        ASSERT_EQ(err, HSA_STATUS_SUCCESS);
        continue;
      }

      // d. GPU-side access must always keep working: fill on the device, copy to host memory and
      // check the value.
      {
        hsa_amd_memory_pool_t sys_pool = {};
        err = hsa_amd_agent_iterate_memory_pools(cpu, rocrtst::FindStandardPool, &sys_pool);
        ASSERT_EQ(err, HSA_STATUS_INFO_BREAK);

        uint32_t* staging = nullptr;
        err = hsa_amd_memory_pool_allocate(sys_pool, kAllocSize, 0,
                                           reinterpret_cast<void**>(&staging));
        ASSERT_EQ(err, HSA_STATUS_SUCCESS);
        memset(staging, 0, kAllocSize);
        err = hsa_amd_agents_allow_access(1, &gpu, nullptr, staging);
        ASSERT_EQ(err, HSA_STATUS_SUCCESS);

        err = hsa_amd_memory_fill(buf, kPattern, kAllocSize / sizeof(uint32_t));
        ASSERT_EQ(err, HSA_STATUS_SUCCESS);

        hsa_signal_t copy_signal;
        err = hsa_signal_create(1, 0, nullptr, &copy_signal);
        ASSERT_EQ(err, HSA_STATUS_SUCCESS);
        err =
            hsa_amd_memory_async_copy(staging, cpu, buf, gpu, kAllocSize, 0, nullptr, copy_signal);
        ASSERT_EQ(err, HSA_STATUS_SUCCESS);
        // A failed copy sets the signal negative, which also satisfies LT 1.
        hsa_signal_wait_scacquire(copy_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                  HSA_WAIT_STATE_BLOCKED);
        const hsa_signal_value_t copy_result = hsa_signal_load_scacquire(copy_signal);
        EXPECT_EQ(0, copy_result) << pi.kind << " pool: async copy failed";
        if (copy_result == 0) {
          EXPECT_EQ(staging[0], kPattern);
          EXPECT_EQ(staging[kAllocSize / sizeof(uint32_t) - 1], kPattern);
        }

        hsa_signal_destroy(copy_signal);
        hsa_amd_memory_pool_free(staging);
        ASSERT_EQ(0, copy_result);
      }

      // e. CPU probes: one write, one read, each guarded independently.
      const bool write_faulted = CpuWriteFaults(buf, kPattern);
      uint32_t readback = 0;
      const bool read_faulted = CpuReadFaults(buf, &readback);
      if (no_host_access) {
        EXPECT_TRUE(write_faulted) << pi.kind
                                   << " pool: CPU write should fault with "
                                      "HSA_DEVICE_MEM_NO_HOST_ACCESS set";
        EXPECT_TRUE(read_faulted) << pi.kind
                                  << " pool: CPU read should fault with "
                                     "HSA_DEVICE_MEM_NO_HOST_ACCESS set";
      } else {
        EXPECT_FALSE(write_faulted)
            << pi.kind << " pool: CPU write to host-mapped device memory faulted";
        EXPECT_FALSE(read_faulted)
            << pi.kind << " pool: CPU read of host-mapped device memory faulted";
        if (!write_faulted && !read_faulted) EXPECT_EQ(readback, kPattern);
      }
      if (verbosity() > 0) {
        std::cout << "  " << pi.kind << ": write " << (write_faulted ? "faulted" : "ok")
                  << ", read " << (read_faulted ? "faulted" : "ok") << std::endl;
      }

      // f. Free.
      err = hsa_amd_memory_pool_free(buf);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    }
  }
}
