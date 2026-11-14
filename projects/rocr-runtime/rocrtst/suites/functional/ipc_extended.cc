/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

//
// Extended IPC tests to cover socket-server restart, free-before-detach,
// and repeated handle creation scenarios.
//

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <vector>
#include <atomic>

#include "suites/functional/ipc_extended.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "common/platform_filter.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

// Phase values for multi-stage synchronization
enum SyncPhase {
  PHASE_INIT = 0,
  PHASE_EXPORT_1_READY = 1,
  PHASE_ATTACH_1_DONE = 2,
  PHASE_DETACH_1_DONE = 3,
  PHASE_FREE_1_DONE = 4,
  PHASE_EXPORT_2_READY = 5,
  PHASE_ATTACH_2_DONE = 6,
  PHASE_TEST_COMPLETE = 7,
  PHASE_ERROR = -1
};

// Wrap printf to add first or second process indicator
#define PROCESS_LOG(format, ...)                                                                   \
  {                                                                                                \
    if (verbosity() >= VERBOSE_STANDARD || !parentProcess_) {                                      \
      fprintf(stdout, "line:%d P%u: " format, __LINE__, static_cast<int>(!parentProcess_),         \
              ##__VA_ARGS__);                                                                      \
    }                                                                                              \
  }

// Fork safe ASSERT_EQ
#define MSG(y, msg, ...) msg
#define Y(y, ...) y

#define FORK_ASSERT_EQ(x, ...)                                                                     \
  if ((x) != (Y(__VA_ARGS__))) {                                                                   \
    if ((x) != (Y(__VA_ARGS__))) {                                                                 \
      std::cout << MSG(__VA_ARGS__, "");                                                           \
      if (parentProcess_) {                                                                        \
        shared_->parent_status = -1;                                                               \
      } else {                                                                                     \
        shared_->child_status = -1;                                                                \
      }                                                                                            \
      ASSERT_EQ(x, Y(__VA_ARGS__));                                                                \
    }                                                                                              \
  }

static void ClearExtendedShared(IPCExtendedShared* s) {
  s->token = 0;
  s->count = 0;
  s->size = 0;
  s->child_status = 0;
  s->parent_status = 0;
  s->phase = PHASE_INIT;
  memset(&s->handle.handle, 0, sizeof(hsa_amd_ipc_memory_t));
  memset(&s->handle2.handle, 0, sizeof(hsa_amd_ipc_memory_t));
  memset(&s->signal_handle, 0, sizeof(hsa_amd_ipc_signal_t));
}

// Wait for phase with timeout
static bool WaitForPhase(IPCExtendedShared* shared, int expected_phase, int timeout_ms = 30000) {
  auto start = std::chrono::steady_clock::now();
  while (shared->phase != expected_phase && shared->phase != PHASE_ERROR) {
    sched_yield();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    if (elapsed > timeout_ms) {
      return false;
    }
  }
  return shared->phase == expected_phase;
}

// =============================================================================
// IPCServerRestartTest Implementation
// =============================================================================

IPCServerRestartTest::IPCServerRestartTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("IPC Server Restart Test");
  set_description(
      "Tests that the IPC socket server correctly restarts "
      "when all exports are freed and a new export is created.");
}

IPCServerRestartTest::~IPCServerRestartTest(void) {}

void IPCServerRestartTest::SetUp(void) {
  if (!checkPlatformFiltering()) return;

#ifdef ROCRTST_ASAN
  std::cout << "Skipping IPC test under ASAN (fork unsupported)." << std::endl;
  test_skipped_ = true;
  return;
#endif

  hsa_status_t err;

  shared_ = reinterpret_cast<IPCExtendedShared*>(mmap(nullptr, sizeof(IPCExtendedShared),
                                                      PROT_READ | PROT_WRITE,
                                                      MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(shared_, MAP_FAILED) << "mmap failed to allocate shared memory";

  ClearExtendedShared(shared_);

  child_ = 0;
  child_ = fork();
  ASSERT_NE(-1, child_) << "fork failed";

  if (child_ != 0) {
    parentProcess_ = true;
    // Simple handshake
    shared_->token = 1;
    while (shared_->token == 1) {
      sched_yield();
    }
    shared_->token = 1;
    while (shared_->token == 1) {
      sched_yield();
    }
  } else {
    parentProcess_ = false;
    set_verbosity(0);
    while (shared_->token == 0) {
      sched_yield();
    }
    shared_->token = 0;
    while (shared_->token == 0) {
      sched_yield();
    }
    shared_->token = 0;
  }

  TestBase::SetUp();
  if (test_skipped_) return;

  err = rocrtst::SetDefaultAgents(this);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (rocrtst::isEmuModeEnabled()) {
    gpu_mem_granule = 4;
  } else {
    err = hsa_amd_memory_pool_get_info(
        device_pool(), HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &gpu_mem_granule);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }
}

void IPCServerRestartTest::ParentProcessImpl() {
  hsa_status_t err;
  uint32_t* gpuBuf1 = NULL;
  uint32_t* gpuBuf2 = NULL;
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  PROCESS_LOG("Parent: Starting server restart test\n");

  // Phase 1: Allocate and export first buffer
  err = hsa_amd_memory_pool_allocate(device_pool(), gpu_mem_granule, 0,
                                     reinterpret_cast<void**>(&gpuBuf1));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to allocate buffer 1\n");
  PROCESS_LOG("Parent: Allocated buffer 1 at %p\n", gpuBuf1);

  err = hsa_amd_agents_allow_access(2, ag_list, NULL, gpuBuf1);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  shared_->size = gpu_mem_granule;

  err = hsa_amd_ipc_memory_create(gpuBuf1, gpu_mem_granule,
                                  const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to create IPC handle 1\n");
  PROCESS_LOG("Parent: Created IPC handle for buffer 1\n");

  // Signal child that export 1 is ready
  shared_->phase = PHASE_EXPORT_1_READY;

  // Wait for child to attach and detach
  ASSERT_TRUE(WaitForPhase(shared_, PHASE_DETACH_1_DONE));
  PROCESS_LOG("Parent: Child detached from buffer 1\n");

  // Free buffer 1 - this should clear ipc_sock_server_conns_ entry
  err = hsa_amd_memory_pool_free(gpuBuf1);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to free buffer 1\n");
  PROCESS_LOG("Parent: Freed buffer 1 (ipc_sock_server_conns_ should be empty)\n");

  shared_->phase = PHASE_FREE_1_DONE;

  // Phase 2: Allocate and export second buffer - this triggers server restart
  err = hsa_amd_memory_pool_allocate(device_pool(), gpu_mem_granule, 0,
                                     reinterpret_cast<void**>(&gpuBuf2));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to allocate buffer 2\n");
  PROCESS_LOG("Parent: Allocated buffer 2 at %p\n", gpuBuf2);

  err = hsa_amd_agents_allow_access(2, ag_list, NULL, gpuBuf2);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Fill buffer 2 with a pattern to verify
  err = hsa_amd_memory_fill(gpuBuf2, 0xDEADBEEF, gpu_mem_granule / sizeof(uint32_t));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_ipc_memory_create(gpuBuf2, gpu_mem_granule,
                                  const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle2));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err,
                 "Parent: Failed to create IPC handle 2 (server restart failed?)\n");
  PROCESS_LOG("Parent: Created IPC handle for buffer 2 (server restart succeeded)\n");

  // Signal child that export 2 is ready
  shared_->phase = PHASE_EXPORT_2_READY;

  // Wait for child to attach to buffer 2
  ASSERT_TRUE(WaitForPhase(shared_, PHASE_ATTACH_2_DONE));
  PROCESS_LOG("Parent: Child attached to buffer 2\n");

  // Cleanup
  err = hsa_amd_memory_pool_free(gpuBuf2);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  shared_->phase = PHASE_TEST_COMPLETE;
  PROCESS_LOG("Parent: Server restart test PASSED\n");

  int exit_status = 0;
  waitpid(child_, &exit_status, 0);
  munmap(shared_, sizeof(IPCExtendedShared));
}

void IPCServerRestartTest::ChildProcessImpl() {
  hsa_status_t err;
  void* ipc_ptr1;
  void* ipc_ptr2;
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  PROCESS_LOG("Child: Waiting for export 1\n");

  // Wait for export 1 to be ready
  ASSERT_TRUE(WaitForPhase(shared_, PHASE_EXPORT_1_READY));

  // Attach to buffer 1
  err = hsa_amd_ipc_memory_attach(const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle),
                                  shared_->size, 1, ag_list, &ipc_ptr1);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: Failed to attach to buffer 1\n");
  PROCESS_LOG("Child: Attached to buffer 1 at %p\n", ipc_ptr1);

  shared_->phase = PHASE_ATTACH_1_DONE;

  // Detach from buffer 1
  err = hsa_amd_ipc_memory_detach(ipc_ptr1);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: Failed to detach from buffer 1\n");
  PROCESS_LOG("Child: Detached from buffer 1\n");

  shared_->phase = PHASE_DETACH_1_DONE;

  // Wait for parent to free buffer 1 and export buffer 2
  ASSERT_TRUE(WaitForPhase(shared_, PHASE_EXPORT_2_READY));

  // Attach to buffer 2 - this validates server restart worked
  err = hsa_amd_ipc_memory_attach(const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle2),
                                  shared_->size, 1, ag_list, &ipc_ptr2);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err,
                 "Child: Failed to attach to buffer 2 (server restart failed?)\n");
  PROCESS_LOG("Child: Attached to buffer 2 at %p (server restart validated)\n", ipc_ptr2);

  // Detach from buffer 2
  err = hsa_amd_ipc_memory_detach(ipc_ptr2);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  shared_->phase = PHASE_ATTACH_2_DONE;

  // Wait for test completion
  ASSERT_TRUE(WaitForPhase(shared_, PHASE_TEST_COMPLETE));

  PROCESS_LOG("Child: Server restart test PASSED\n");
}

uint32_t IPCServerRestartTest::RealIterationNum(void) { return num_iteration() * 1.2 + 1; }

void IPCServerRestartTest::PrintVerboseMesg(void) {
  if (verbosity() >= VERBOSE_STANDARD) {
    hsa_status_t err;
    char name1[64] = {0};
    char name2[64] = {0};
    err = hsa_agent_get_info(*cpu_device(), HSA_AGENT_INFO_NAME, name1);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    err = hsa_agent_get_info(*gpu_device1(), HSA_AGENT_INFO_NAME, name2);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    fprintf(stdout, "Using: %s and %s\n", name1, name2);
  }
}

void IPCServerRestartTest::Run(void) {
  TestBase::Run();

  if (verbosity() >= VERBOSE_STANDARD) {
    PrintVerboseMesg();
  }

  if (parentProcess_) {
    ParentProcessImpl();
  } else {
    ChildProcessImpl();
    hsa_shut_down();
    exit(0);
  }
}

void IPCServerRestartTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void IPCServerRestartTest::DisplayResults(void) const { TestBase::DisplayResults(); }

void IPCServerRestartTest::Close() { TestBase::Close(); }

// =============================================================================
// IPCFreeBeforeDetachTest Implementation
// =============================================================================

IPCFreeBeforeDetachTest::IPCFreeBeforeDetachTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("IPC Free Before Detach Test");
  set_description(
      "Tests that freeing exported memory before importer detaches "
      "correctly cleans up bookkeeping and allows subsequent exports.");
}

IPCFreeBeforeDetachTest::~IPCFreeBeforeDetachTest(void) {}

void IPCFreeBeforeDetachTest::SetUp(void) {
  if (!checkPlatformFiltering()) return;

#ifdef ROCRTST_ASAN
  std::cout << "Skipping IPC test under ASAN (fork unsupported)." << std::endl;
  test_skipped_ = true;
  return;
#endif

  hsa_status_t err;

  shared_ = reinterpret_cast<IPCExtendedShared*>(mmap(nullptr, sizeof(IPCExtendedShared),
                                                      PROT_READ | PROT_WRITE,
                                                      MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(shared_, MAP_FAILED) << "mmap failed to allocate shared memory";

  ClearExtendedShared(shared_);

  child_ = 0;
  child_ = fork();
  ASSERT_NE(-1, child_) << "fork failed";

  if (child_ != 0) {
    parentProcess_ = true;
    shared_->token = 1;
    while (shared_->token == 1) {
      sched_yield();
    }
    shared_->token = 1;
    while (shared_->token == 1) {
      sched_yield();
    }
  } else {
    parentProcess_ = false;
    set_verbosity(0);
    while (shared_->token == 0) {
      sched_yield();
    }
    shared_->token = 0;
    while (shared_->token == 0) {
      sched_yield();
    }
    shared_->token = 0;
  }

  TestBase::SetUp();
  if (test_skipped_) return;

  err = rocrtst::SetDefaultAgents(this);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (rocrtst::isEmuModeEnabled()) {
    gpu_mem_granule = 4;
  } else {
    err = hsa_amd_memory_pool_get_info(
        device_pool(), HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &gpu_mem_granule);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }
}

void IPCFreeBeforeDetachTest::ParentProcessImpl() {
  hsa_status_t err;
  uint32_t* gpuBuf1 = NULL;
  uint32_t* gpuBuf2 = NULL;
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  PROCESS_LOG("Parent: Starting free-before-detach test\n");

  // Phase 1: Allocate and export first buffer
  err = hsa_amd_memory_pool_allocate(device_pool(), gpu_mem_granule, 0,
                                     reinterpret_cast<void**>(&gpuBuf1));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  PROCESS_LOG("Parent: Allocated buffer 1 at %p\n", gpuBuf1);

  err = hsa_amd_agents_allow_access(2, ag_list, NULL, gpuBuf1);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  shared_->size = gpu_mem_granule;

  err = hsa_amd_memory_fill(gpuBuf1, 0x12345678, gpu_mem_granule / sizeof(uint32_t));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_ipc_memory_create(gpuBuf1, gpu_mem_granule,
                                  const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  PROCESS_LOG("Parent: Created IPC handle for buffer 1\n");

  shared_->phase = PHASE_EXPORT_1_READY;

  // Wait for child to attach
  ASSERT_TRUE(WaitForPhase(shared_, PHASE_ATTACH_1_DONE));
  PROCESS_LOG("Parent: Child attached to buffer 1\n");

  // Free buffer 1 BEFORE child detaches - this is the key scenario
  err = hsa_amd_memory_pool_free(gpuBuf1);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  PROCESS_LOG("Parent: Freed buffer 1 while child still attached\n");

  shared_->phase = PHASE_FREE_1_DONE;

  // Now try to export a new buffer - should succeed if bookkeeping is correct
  err = hsa_amd_memory_pool_allocate(device_pool(), gpu_mem_granule, 0,
                                     reinterpret_cast<void**>(&gpuBuf2));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  PROCESS_LOG("Parent: Allocated buffer 2 at %p\n", gpuBuf2);

  err = hsa_amd_agents_allow_access(2, ag_list, NULL, gpuBuf2);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_fill(gpuBuf2, 0xABCDEF01, gpu_mem_granule / sizeof(uint32_t));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_ipc_memory_create(gpuBuf2, gpu_mem_granule,
                                  const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle2));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to export buffer 2 (stale entry?)\n");
  PROCESS_LOG("Parent: Created IPC handle for buffer 2 (no stale entries)\n");

  shared_->phase = PHASE_EXPORT_2_READY;

  // Wait for child to validate
  ASSERT_TRUE(WaitForPhase(shared_, PHASE_ATTACH_2_DONE));

  err = hsa_amd_memory_pool_free(gpuBuf2);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  shared_->phase = PHASE_TEST_COMPLETE;
  PROCESS_LOG("Parent: Free-before-detach test PASSED\n");

  int exit_status = 0;
  waitpid(child_, &exit_status, 0);
  munmap(shared_, sizeof(IPCExtendedShared));
}

void IPCFreeBeforeDetachTest::ChildProcessImpl() {
  hsa_status_t err;
  void* ipc_ptr1;
  void* ipc_ptr2;
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  PROCESS_LOG("Child: Waiting for export 1\n");

  ASSERT_TRUE(WaitForPhase(shared_, PHASE_EXPORT_1_READY));

  err = hsa_amd_ipc_memory_attach(const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle),
                                  shared_->size, 1, ag_list, &ipc_ptr1);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  PROCESS_LOG("Child: Attached to buffer 1 at %p\n", ipc_ptr1);

  shared_->phase = PHASE_ATTACH_1_DONE;

  // Wait for parent to free buffer 1
  ASSERT_TRUE(WaitForPhase(shared_, PHASE_FREE_1_DONE));

  // Now detach from buffer 1 (after parent freed it)
  err = hsa_amd_ipc_memory_detach(ipc_ptr1);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  PROCESS_LOG("Child: Detached from buffer 1 (after parent freed it)\n");

  // Wait for export 2
  ASSERT_TRUE(WaitForPhase(shared_, PHASE_EXPORT_2_READY));

  err = hsa_amd_ipc_memory_attach(const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle2),
                                  shared_->size, 1, ag_list, &ipc_ptr2);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: Failed to attach to buffer 2\n");
  PROCESS_LOG("Child: Attached to buffer 2 at %p\n", ipc_ptr2);

  err = hsa_amd_ipc_memory_detach(ipc_ptr2);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  shared_->phase = PHASE_ATTACH_2_DONE;

  ASSERT_TRUE(WaitForPhase(shared_, PHASE_TEST_COMPLETE));
  PROCESS_LOG("Child: Free-before-detach test PASSED\n");
}

uint32_t IPCFreeBeforeDetachTest::RealIterationNum(void) { return num_iteration() * 1.2 + 1; }

void IPCFreeBeforeDetachTest::PrintVerboseMesg(void) {
  if (verbosity() >= VERBOSE_STANDARD) {
    hsa_status_t err;
    char name1[64] = {0};
    char name2[64] = {0};
    err = hsa_agent_get_info(*cpu_device(), HSA_AGENT_INFO_NAME, name1);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    err = hsa_agent_get_info(*gpu_device1(), HSA_AGENT_INFO_NAME, name2);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    fprintf(stdout, "Using: %s and %s\n", name1, name2);
  }
}

void IPCFreeBeforeDetachTest::Run(void) {
  TestBase::Run();

  if (verbosity() >= VERBOSE_STANDARD) {
    PrintVerboseMesg();
  }

  if (parentProcess_) {
    ParentProcessImpl();
  } else {
    ChildProcessImpl();
    hsa_shut_down();
    exit(0);
  }
}

void IPCFreeBeforeDetachTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void IPCFreeBeforeDetachTest::DisplayResults(void) const { TestBase::DisplayResults(); }

void IPCFreeBeforeDetachTest::Close() { TestBase::Close(); }

// =============================================================================
// IPCRepeatedHandleTest Implementation
// =============================================================================

IPCRepeatedHandleTest::IPCRepeatedHandleTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("IPC Repeated Handle Test");
  set_description(
      "Tests repeated IPC handle creation and attach/detach cycles "
      "to verify BO handle management doesn't leak.");
}

IPCRepeatedHandleTest::~IPCRepeatedHandleTest(void) {}

void IPCRepeatedHandleTest::SetUp(void) {
  if (!checkPlatformFiltering()) return;

#ifdef ROCRTST_ASAN
  std::cout << "Skipping IPC test under ASAN (fork unsupported)." << std::endl;
  test_skipped_ = true;
  return;
#endif

  hsa_status_t err;

  shared_ = reinterpret_cast<IPCExtendedShared*>(mmap(nullptr, sizeof(IPCExtendedShared),
                                                      PROT_READ | PROT_WRITE,
                                                      MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(shared_, MAP_FAILED) << "mmap failed to allocate shared memory";

  ClearExtendedShared(shared_);

  child_ = 0;
  child_ = fork();
  ASSERT_NE(-1, child_) << "fork failed";

  if (child_ != 0) {
    parentProcess_ = true;
    shared_->token = 1;
    while (shared_->token == 1) {
      sched_yield();
    }
    shared_->token = 1;
    while (shared_->token == 1) {
      sched_yield();
    }
  } else {
    parentProcess_ = false;
    set_verbosity(0);
    while (shared_->token == 0) {
      sched_yield();
    }
    shared_->token = 0;
    while (shared_->token == 0) {
      sched_yield();
    }
    shared_->token = 0;
  }

  TestBase::SetUp();
  if (test_skipped_) return;

  err = rocrtst::SetDefaultAgents(this);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (rocrtst::isEmuModeEnabled()) {
    gpu_mem_granule = 4;
  } else {
    err = hsa_amd_memory_pool_get_info(
        device_pool(), HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &gpu_mem_granule);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }
}

void IPCRepeatedHandleTest::ParentProcessImpl() {
  hsa_status_t err;
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  PROCESS_LOG("Parent: Starting repeated handle test (%d cycles)\n", kRepeatCount);

  shared_->size = gpu_mem_granule;

  for (int cycle = 0; cycle < kRepeatCount; cycle++) {
    PROCESS_LOG("Parent: Starting cycle %d\n", cycle + 1);

    // Allocate buffer
    uint32_t* gpuBuf = NULL;
    err = hsa_amd_memory_pool_allocate(device_pool(), gpu_mem_granule, 0,
                                       reinterpret_cast<void**>(&gpuBuf));
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    err = hsa_amd_agents_allow_access(2, ag_list, NULL, gpuBuf);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    // Fill with cycle-specific pattern
    err = hsa_amd_memory_fill(gpuBuf, 0x10000000 + cycle, gpu_mem_granule / sizeof(uint32_t));
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    // Create IPC handle
    err = hsa_amd_ipc_memory_create(gpuBuf, gpu_mem_granule,
                                    const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle));
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    PROCESS_LOG("Parent: Created IPC handle for cycle %d\n", cycle + 1);

    // Create handle AGAIN for same buffer - should not leak
    hsa_amd_ipc_memory_t handle_dup;
    err = hsa_amd_ipc_memory_create(gpuBuf, gpu_mem_granule, &handle_dup);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    (void)handle_dup;  // Intentionally unused - testing duplicate creation doesn't leak
    PROCESS_LOG("Parent: Created duplicate IPC handle (testing no leak)\n");

    // Signal child
    shared_->count = cycle;
    shared_->phase = PHASE_EXPORT_1_READY;

    // Wait for child to complete attach/detach cycle
    ASSERT_TRUE(WaitForPhase(shared_, PHASE_ATTACH_1_DONE));

    // Free buffer
    err = hsa_amd_memory_pool_free(gpuBuf);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    PROCESS_LOG("Parent: Freed buffer for cycle %d\n", cycle + 1);

    shared_->phase = PHASE_FREE_1_DONE;

    if (cycle < kRepeatCount - 1) {
      // Wait for child to be ready for next cycle
      ASSERT_TRUE(WaitForPhase(shared_, PHASE_INIT));
    }
  }

  shared_->phase = PHASE_TEST_COMPLETE;
  PROCESS_LOG("Parent: Repeated handle test PASSED\n");

  int exit_status = 0;
  waitpid(child_, &exit_status, 0);
  munmap(shared_, sizeof(IPCExtendedShared));
}

void IPCRepeatedHandleTest::ChildProcessImpl() {
  hsa_status_t err;
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  PROCESS_LOG("Child: Starting repeated handle test\n");

  for (int cycle = 0; cycle < kRepeatCount; cycle++) {
    PROCESS_LOG("Child: Waiting for cycle %d export\n", cycle + 1);

    ASSERT_TRUE(WaitForPhase(shared_, PHASE_EXPORT_1_READY));
    FORK_ASSERT_EQ(cycle, shared_->count.load());

    // Attach to buffer
    void* ipc_ptr;
    err = hsa_amd_ipc_memory_attach(const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle),
                                    shared_->size, 1, ag_list, &ipc_ptr);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    PROCESS_LOG("Child: Attached to buffer for cycle %d\n", cycle + 1);

    // Attach AGAIN - should work with handle lifetime fixes
    void* ipc_ptr2;
    err = hsa_amd_ipc_memory_attach(const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle),
                                    shared_->size, 1, ag_list, &ipc_ptr2);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    PROCESS_LOG("Child: Second attach succeeded for cycle %d\n", cycle + 1);

    // Detach both
    err = hsa_amd_ipc_memory_detach(ipc_ptr2);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    err = hsa_amd_ipc_memory_detach(ipc_ptr);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    PROCESS_LOG("Child: Detached from buffer for cycle %d\n", cycle + 1);

    shared_->phase = PHASE_ATTACH_1_DONE;

    if (cycle < kRepeatCount - 1) {
      // Wait for parent to free (except on last cycle)
      ASSERT_TRUE(WaitForPhase(shared_, PHASE_FREE_1_DONE));
      shared_->phase = PHASE_INIT;
    }
  }

  // On last cycle, parent goes directly to PHASE_TEST_COMPLETE
  ASSERT_TRUE(WaitForPhase(shared_, PHASE_TEST_COMPLETE));
  PROCESS_LOG("Child: Repeated handle test PASSED\n");
}

uint32_t IPCRepeatedHandleTest::RealIterationNum(void) { return num_iteration() * 1.2 + 1; }

void IPCRepeatedHandleTest::PrintVerboseMesg(void) {
  if (verbosity() >= VERBOSE_STANDARD) {
    hsa_status_t err;
    char name1[64] = {0};
    char name2[64] = {0};
    err = hsa_agent_get_info(*cpu_device(), HSA_AGENT_INFO_NAME, name1);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    err = hsa_agent_get_info(*gpu_device1(), HSA_AGENT_INFO_NAME, name2);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    fprintf(stdout, "Using: %s and %s\n", name1, name2);
  }
}

void IPCRepeatedHandleTest::Run(void) {
  TestBase::Run();

  if (verbosity() >= VERBOSE_STANDARD) {
    PrintVerboseMesg();
  }

  if (parentProcess_) {
    ParentProcessImpl();
  } else {
    ChildProcessImpl();
    hsa_shut_down();
    exit(0);
  }
}

void IPCRepeatedHandleTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void IPCRepeatedHandleTest::DisplayResults(void) const { TestBase::DisplayResults(); }

void IPCRepeatedHandleTest::Close() { TestBase::Close(); }

#undef PROCESS_LOG
#undef FORK_ASSERT_EQ
#undef MSG
#undef Y
