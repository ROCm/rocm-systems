/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// IPC functional test that exercises SDMA (copy engine) on imported IPC memory.
// Complements ipc.cc, which only validates the buffer via generic async copies
// that may use a blit kernel instead of SDMA
// 1. Process A: Allocate and export IPC memory handle (parent)
// 2. Process B: Import the IPC memory handle (child)
// 3. Process B: Attempt SDMA copy operation
//    - Copy FROM the IPC memory to another buffer (read test)
//    - Copy TO the IPC memory from another buffer (write test)
// 4. Verify the SDMA operations succeed and data is correct



#include <sys/mman.h>
#include <sys/wait.h>

#include <cstring>

#include "suites/functional/ipc_sdma.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa_ext_amd.h"

namespace {

constexpr uint32_t kPatternA = 0xA501C001u;
constexpr uint32_t kPatternB = 0xB602D002u;

#define PROCESS_LOG(format, ...)                                                          \
  do {                                                                                    \
    if (verbosity() >= VERBOSE_STANDARD || !parentProcess_) {                             \
      fprintf(stdout, "line:%d P%u: " format, __LINE__, static_cast<int>(!parentProcess_), \
              ##__VA_ARGS__);                                                             \
    }                                                                                     \
  } while (0)

#define MSG(y, msg, ...) msg
#define Y(y, ...) y

#define FORK_ASSERT_EQ(x, ...)                                                             \
  do {                                                                                     \
    if ((x) != (Y(__VA_ARGS__))) {                                                         \
      std::cout << MSG(__VA_ARGS__, "");                                                   \
      if (parentProcess_) {                                                                \
        shared_->parent_status = -1;                                                       \
      } else {                                                                             \
        shared_->child_status = -1;                                                        \
      }                                                                                    \
      ASSERT_EQ(x, Y(__VA_ARGS__));                                                        \
    }                                                                                      \
  } while (0)

static int CheckAndSetToken(std::atomic<int>* token, int newVal) {
  if (*token == -1) {
    return -1;
  }
  *token = newVal;
  return 0;
}

static hsa_status_t SelectSdmaEngine(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                                     hsa_amd_sdma_engine_id_t* engine_id) {
  uint32_t preferred_mask = 0;
  uint32_t engine_ids_mask = 0;
  hsa_status_t err = hsa_amd_memory_get_preferred_copy_engine(dst_agent, src_agent, &preferred_mask);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  err = hsa_amd_memory_copy_engine_status(dst_agent, src_agent, &engine_ids_mask);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }
  preferred_mask = preferred_mask ? (preferred_mask & engine_ids_mask) : engine_ids_mask;
  engine_ids_mask = preferred_mask ? preferred_mask : engine_ids_mask;
  if (engine_ids_mask == 0) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  int bit = __builtin_ffs(static_cast<int>(engine_ids_mask));
  *engine_id = static_cast<hsa_amd_sdma_engine_id_t>(1u << (static_cast<unsigned>(bit) - 1u));
  return HSA_STATUS_SUCCESS;
}

}  // namespace

IPCSDMATest::IPCSDMATest() : TestBase() {
  set_num_iteration(1);
  set_title("IPC SDMA copy test");
  set_description(
      "Forked IPC test: child imports GPU memory and copies using "
      "hsa_amd_memory_async_copy_on_engine (SDMA). Catches regressions where "
      "imported memory is shader-accessible but not SDMA-accessible.");
}

IPCSDMATest::~IPCSDMATest() = default;

static void ClearShared(SharedIpcSdma* s) {
  s->token = 0;
  s->child_status = 0;
  s->parent_status = 0;
  s->size = 0;
  memset(&s->handle.handle, 0, sizeof(hsa_amd_ipc_memory_t));
}

void IPCSDMATest::SetUp(void) {
  if (!checkPlatformFiltering()) return;
  
  #ifdef ROCRTST_ASAN
  // IPC test uses fork() which is unsupported under ASAN...
  std::cout << "Skipping IPC test under ASAN (fork unsupported)." <<
  std::endl;
  test_skipped_ = true;
  return;
  #endif

  shared_ = reinterpret_cast<SharedIpcSdma*>(
      mmap(nullptr, sizeof(SharedIpcSdma), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(shared_, MAP_FAILED) << "mmap failed for shared control block";
  ClearShared(shared_);

  child_ = fork();
  ASSERT_NE(-1, child_) << "fork failed";
  std::atomic<int>* token = &shared_->token;
  if (child_ != 0) {
    parentProcess_ = true;
    *token = 1;
    while (*token == 1) {
      sched_yield();
    }
    PROCESS_LOG("Second process observed, handshake...\n");
    *token = 1;
    while (*token == 1) {
      sched_yield();
    }
  } else {
    parentProcess_ = false;
    set_verbosity(0);
    PROCESS_LOG("Second process running.\n");
    while (*token == 0) {
      sched_yield();
    }
    int ret = CheckAndSetToken(token, 0);
    ASSERT_EQ(0, ret) << "Error detected in child process\n";
    while (*token == 0) {
      sched_yield();
    }
    ret = CheckAndSetToken(token, 0);
    ASSERT_EQ(0, ret) << "Error detected in child process\n";
  }

  TestBase::SetUp();

  hsa_status_t err = rocrtst::SetDefaultAgents(this);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (rocrtst::isEmuModeEnabled()) {
    gpu_mem_granule_ = 4;
  } else {
    err = hsa_amd_memory_pool_get_info(device_pool(), HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                       &gpu_mem_granule_);
    FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }
}

void IPCSDMATest::ChildProcessImpl() {
  while (shared_->token == 0) {
    sched_yield();
  }
  if (shared_->token != 1) {
    shared_->token = -1;
  }
  FORK_ASSERT_EQ(1, shared_->token, "Child: bad handshake token\n");
  PROCESS_LOG("Child: woke on parent signal\n");

  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};
  void* ipc_ptr = nullptr;
  hsa_status_t err =
      hsa_amd_ipc_memory_attach(const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle), shared_->size, 1,
                                ag_list, &ipc_ptr);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: hsa_amd_ipc_memory_attach failed\n");
  PROCESS_LOG("Child: attached IPC buffer at %p\n", ipc_ptr);

  hsa_amd_sdma_engine_id_t engine_d2h{};
  err = SelectSdmaEngine(*cpu_device(), *gpu_device1(), &engine_d2h);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: no SDMA engine for D2H copy\n");

  uint32_t* sysBuf = nullptr;
  err = hsa_amd_memory_pool_allocate(cpu_pool(), shared_->size, 0, reinterpret_cast<void**>(&sysBuf));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: failed to allocate host staging buffer\n");
  err = hsa_amd_agents_allow_access(2, ag_list, nullptr, sysBuf);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: agents_allow_access on staging failed\n");

  hsa_signal_t copy_signal{};
  err = hsa_signal_create(1, 0, nullptr, &copy_signal);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_signal_store_relaxed(copy_signal, 1);
  err = hsa_amd_memory_async_copy_on_engine(sysBuf, *cpu_device(), ipc_ptr, *gpu_device1(),
                                            shared_->size, 0, nullptr, copy_signal, engine_d2h, false);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: SDMA D2H from IPC memory failed\n");
  hsa_signal_value_t sig = hsa_signal_wait_relaxed(copy_signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1),
                                                   HSA_WAIT_STATE_BLOCKED);
  FORK_ASSERT_EQ(0, sig, "Child: D2H completion signal wrong value\n");

  const size_t dword_count = shared_->size / sizeof(uint32_t);
  for (size_t i = 0; i < dword_count; i++) {
    FORK_ASSERT_EQ(kPatternA, sysBuf[i], "Child: SDMA read wrong data from IPC buffer\n");
    sysBuf[i] = kPatternB;
  }

  hsa_amd_sdma_engine_id_t engine_h2d{};
  err = SelectSdmaEngine(*gpu_device1(), *cpu_device(), &engine_h2d);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: no SDMA engine for H2D copy\n");

  hsa_signal_store_relaxed(copy_signal, 1);
  err = hsa_amd_memory_async_copy_on_engine(ipc_ptr, *gpu_device1(), sysBuf, *cpu_device(),
                                            shared_->size, 0, nullptr, copy_signal, engine_h2d, false);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: SDMA H2D to IPC memory failed\n");
  sig = hsa_signal_wait_relaxed(copy_signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1),
                              HSA_WAIT_STATE_BLOCKED);
  FORK_ASSERT_EQ(0, sig, "Child: H2D completion signal wrong value\n");

  err = hsa_signal_destroy(copy_signal);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = hsa_amd_memory_pool_free(sysBuf);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_ipc_memory_detach(ipc_ptr);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: ipc_memory_detach failed\n");

  shared_->token = 2;
  PROCESS_LOG("Child: IPC SDMA test passed\n");
}

void IPCSDMATest::ParentProcessImpl() {

  // Ignoring the first allocation to exercise fragment allocation.
  hsa_status_t err;
  uint32_t* discard = nullptr;
  err = hsa_amd_memory_pool_allocate(device_pool(), gpu_mem_granule_, 0, reinterpret_cast<void**>(&discard));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: discard alloc failed\n");


  // Allocate some VRAM that is used to test IPC
  uint32_t* gpuBuf = nullptr;
  err = hsa_amd_memory_pool_allocate(device_pool(), gpu_mem_granule_, 0, reinterpret_cast<void**>(&gpuBuf));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: gpu alloc failed\n");

  // Free the test allocation of memory block
  err = hsa_amd_memory_pool_free(discard);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: free discard failed\n");

  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};
  err = hsa_amd_agents_allow_access(2, ag_list, nullptr, gpuBuf);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: agents_allow_access failed\n");

  shared_->size = gpu_mem_granule_;
  const size_t dword_count = gpu_mem_granule_ / sizeof(uint32_t);
  err = hsa_amd_memory_fill(gpuBuf, kPatternA, dword_count);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: memory_fill failed\n");

  // Create an IPC memory handle. IPC handle value is shared with
  // child process via a shared data structure
  err = hsa_amd_ipc_memory_create(gpuBuf, gpu_mem_granule_, const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: ipc_memory_create failed\n");
  PROCESS_LOG("Parent: IPC handle ready\n");

  CheckAndSetToken(&shared_->token, 1);

  while (shared_->token < 2) {
    if (shared_->child_status == -1) {
      err = hsa_amd_memory_pool_free(gpuBuf);
      FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      int exit_status = 0;
      pid_t waited_pid = waitpid(child_, &exit_status, 0);
      ASSERT_EQ(child_, waited_pid);

      munmap(shared_, sizeof(SharedIpcSdma));
      FAIL() << "Parent: child process reported failure";
    }
    sched_yield();
  }

  uint32_t* host_verify = nullptr;
  err = hsa_amd_memory_pool_allocate(cpu_pool(), gpu_mem_granule_, 0, reinterpret_cast<void**>(&host_verify));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: host staging alloc failed\n");
  err = hsa_amd_agents_allow_access(2, ag_list, nullptr, host_verify);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_signal_t copy_signal{};
  err = hsa_signal_create(1, 0, nullptr, &copy_signal);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_async_copy(host_verify, *cpu_device(), gpuBuf, *gpu_device1(), gpu_mem_granule_, 0,
                                  nullptr, copy_signal);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: async_copy for verify failed\n");
  hsa_signal_value_t sig =
      hsa_signal_wait_relaxed(copy_signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1), HSA_WAIT_STATE_BLOCKED);
  FORK_ASSERT_EQ(0, sig);

  for (size_t i = 0; i < dword_count; i++) {
    FORK_ASSERT_EQ(kPatternB, host_verify[i],
                   "Parent: GPU buffer not updated by child SDMA\n");
  }

  err = hsa_signal_destroy(copy_signal);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = hsa_amd_memory_pool_free(host_verify);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_pool_free(gpuBuf);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  int exit_status = 0;
  pid_t waited_pid = waitpid(child_, &exit_status, 0);
  ASSERT_EQ(child_, waited_pid);
  ASSERT_TRUE(WIFEXITED(exit_status));
  ASSERT_EQ(0, WEXITSTATUS(exit_status));
  munmap(shared_, sizeof(SharedIpcSdma));
  PROCESS_LOG("Parent: IPC SDMA test passed\n");
}

void IPCSDMATest::Run(void) {
  TestBase::Run();
  if (parentProcess_) {
    ParentProcessImpl();
  } else {
    ChildProcessImpl();
    hsa_shut_down();
    exit(0);
  }
}

void IPCSDMATest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void IPCSDMATest::DisplayResults(void) const {
  TestBase::DisplayResults();
}

void IPCSDMATest::Close() {
  TestBase::Close();
}

#undef PROCESS_LOG
#undef FORK_ASSERT_EQ
#undef MSG
#undef Y
