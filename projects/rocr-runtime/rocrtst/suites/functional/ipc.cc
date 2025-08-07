/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

//
//        Parent Process
//  Allocate a block of gpu-local memory
//  Print log message about allocation
//  Acquire access to gpu-local memory
//    This step may not be needed
//  Obtain a IPC handle for gpu-local memory
//  Print log message about getting IPC handle
//  Initialize DWords of gpu-local memory with 0x01
//  Print log message about updating gpu-local memory
//  Create a Signal that is capable of IPC
//  Obtain a IPC handle to signal
//  Print log message about signalling Child process
//  Signal Child process that it can proceed
//  Print log message about waiting for signal from Child process
//  Wait for Child processes signal
//  Verify Child has updated DWords of gpu-local memory to 0x02
//  Print log message about validation of gpu-local memory
//  Set the DWords of gpu-local memory with 0x03
//  Signal Child process that it can proceed  by setting signal to 3
//  Wait for Child processes signal
//  Verify Child has updated DWords of gpu-local memory to 0x04
//  Print log message that IPC test passed
//
//        Child Process
//  Print log message about waiting for signal from Parent process
//  Wait/Yield for Parent process signal
//  Validate Parent process signal is per expectation
//  Attach to IPC memory handle shared by Parent process
//  Print log message about successful acquisition of IPC memory handle
//  Print log message about successful acquisition of IPC signal handle
//  Verify Parent process has updated every DWord of Gpu buffer to 0x01
//  Update every DWord of Gpu buffer with 0x02 value
//  Print log message about validation of Gpu buffer state i.e every DWord has 0x01
//  Register a callback using hsa_amd_signal_async_handler on the ipc signal
//    - the callback function will update gpu-local memory DWords to 0x04
//    - and update a local token to indicate that the callback happened.
//  Signal the parent process that it can proceed by setting signal to 2
//  Wait for callback function to update the local token.
//  Signal the parent process that it can proceed by setting signal to 4
//  Wait for parent to set signal to 0 to indicate that it can clean-up and exit.
//
// The comments provided below are focused more on the use of common rocrtst
// utilities and boilerplate code, rather than the example app. itself.
//

#include <sys/mman.h>

#include <algorithm>
#include <vector>
#include <atomic>

#include "suites/functional/ipc.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const uint32_t kNumBufferElements = 256;

struct callback_args {
  hsa_agent_t host;
  hsa_agent_t device;
  hsa_amd_memory_pool_t cpu_pool;
  hsa_amd_memory_pool_t gpu_pool;
  size_t gpu_mem_granule;
};

// Wrap printf to add first or second process indicator
#define PROCESS_LOG(format, ...)  { \
    if (verbosity() >= VERBOSE_STANDARD || !parentProcess_) { \
      fprintf(stdout, "line:%d P%u: " format, \
                   __LINE__, static_cast<int>(!parentProcess_), ##__VA_ARGS__); \
    } \
}

// Fork safe ASSERT_EQ.
#define MSG(y, msg, ...) msg
#define Y(y, ...) y

#define FORK_ASSERT_EQ(x, ...)                                                    \
  if ((x) != (Y(__VA_ARGS__))) {                                                  \
    if ((x) != (Y(__VA_ARGS__))) {                                                \
      std::cout << MSG(__VA_ARGS__, "");                                          \
      if (parentProcess_) {                                                       \
        shared_->parent_status = -1;                                              \
      } else {                                                                    \
        shared_->child_status = -1;                                               \
      }                                                                           \
      ASSERT_EQ(x, Y(__VA_ARGS__));                                               \
    }                                                                             \
  }

#define USR_TRIGGERED_FAILURE(x, y, z)                                            \
  if (usr_fail_val_ == (z)) {                                                     \
    std::cout << "Env value is: " << z << std::endl;                              \
    std::cout << "Return value before: " << x << std::endl;                       \
    std::cout << "Return value  after: " << y << std::endl << std::flush;         \
    (x) = (y);                                                                    \
  }

IPCTest::IPCTest(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("IPC Test");
  set_description("IPCTest verifies that the IPC feature of RocR is "
      "functioning as expected. The test first forks off second process. The "
      "2 processes share pointers to RocR allocated memory and also share "
      "signal handles");
}

IPCTest::~IPCTest(void) {
}

// See if the other process wrote an error value to the token; if not, write
// the newVal to the token.
static int CheckAndSetToken(std::atomic<int> *token, int newVal) {
  if (*token == -1) {
    return -1;
  } else {
    *token = newVal;
  }

  return 0;
}

static void ClearShared(Shared *s) {
  s->token = 0;
  s->count = 0;
  s->size = 0;
  s->child_status = 0;
  s->parent_status = 0;
  memset(&s->handle.handle, 0, sizeof(hsa_amd_ipc_memory_t));
  memset(&s->signal_handle, 0, sizeof(hsa_amd_ipc_signal_t));
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void IPCTest::SetUp(void) {
  hsa_status_t err;

  // Allow user to trigger a failure
  const char* env_val = getenv("ROCR_IPC_FAIL_KEY");
  if (env_val != NULL) {
    usr_fail_val_ = atoi(env_val);
  }

  // We must fork process before doing HSA stuff, specifically, hsa_init, as
  // each process needs to do this.
  // Allocate linux shared_ memory.
  shared_ = reinterpret_cast<Shared*>(
      mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                                          MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(shared_, MAP_FAILED) << "mmap failed to allocated shared_ memory";

  // Initialize shared control block to zeros. The field "token"
  // is used to signal state changes between the 2 processes.
  ClearShared(shared_);

  // Spawn second process and verify communication
  child_ = 0;
  child_ = fork();
  ASSERT_NE(-1, child_) << "fork failed";
  std::atomic<int> * token = &shared_->token;
  if (child_ != 0) {
    parentProcess_ = true;

    // Signal to other process we are waiting, and then wait...
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

    int ret;
    ret = CheckAndSetToken(token, 0);
    ASSERT_EQ(0, ret) << "Error detected in child process\n";
    // Wait for handshake
    while (*token == 0) {
      sched_yield();
    }
    ret = CheckAndSetToken(token, 0);
    ASSERT_EQ(0, ret) << "Error detected in child process\n";
  }
  // TestBase::SetUp() will set HSA_ENABLE_INTERRUPT if enable_interrupt() is
  // true, and call hsa_init(). It also prints the SetUp header.
  TestBase::SetUp();

  // SetDefaultAgents(this) will assign the first CPU and GPU found on
  // iterating through the agents and assign them to cpu_device_ and
  // gpu_device1_, respectively (cpu_device() and gpu_device1()). These
  // BaseRocR member variables are used in some utilities. Additionally,
  // SetDefaultAgents() checks the profile of the gpu and compares this
  // to any required profile.
  err = rocrtst::SetDefaultAgents(this);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Find and assign HSA_AMD_SEGMENT_GLOBAL pools for cpu, gpu and a kern_arg
  // pool
  err = rocrtst::SetPoolsTypical(this);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

// Update the size granularity for allocations
#ifdef ROCRTST_EMULATOR_BUILD
  gpu_mem_granule = 4;
#else
  err = hsa_amd_memory_pool_get_info(device_pool(), HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                     &gpu_mem_granule);
#endif

  return;
}

// Do a few extra iterations as we toss out some of the inital and final
// iterations when calculating statistics
uint32_t IPCTest::RealIterationNum(void) {
  return num_iteration() * 1.2 + 1;
}

/*
 * if the hsa_signal_value_t value matches sig_value, and
 * then set destination to
 * new value.
 */
struct signal_cb_handler_data {
  IPCTest *obj;
  hsa_signal_value_t exp_sig_value;
  uint32_t exp_value;
  uint32_t *destination;
  uint32_t new_value;
  std::atomic<int> token;
};

bool SignalCallbackHandler(hsa_signal_value_t value, void* arg) {
  signal_cb_handler_data* cb_data = reinterpret_cast<signal_cb_handler_data*>(arg);
  if (cb_data->exp_sig_value != value)
    return false;

  cb_data->obj->CheckAndFillBuffer(cb_data->destination, cb_data->exp_value, cb_data->new_value);
  cb_data->token++;

  /* return false to stop monitoring this callback */
  return false;
}

void IPCTest::ChildProcessImpl() {

  // Yield until shared token value changes i.e. is updated by parent.
  // Validate parent's update is per expectation
  PROCESS_LOG("Child: Waiting for parent process to signal\n");
  while (shared_->token == 0) {
    sched_yield();
  }
  if (shared_->token != 1) {
    shared_->token = -1;
  }
  FORK_ASSERT_EQ(1, shared_->token, "Child: Error detected in signaling token\n");
  PROCESS_LOG("Child: Waking upon signal from parent process\n");

  // List of devices involved in test. Gpu device is used
  // to allocate buffer and signal that are part of an IPC
  // transaction. Cpu is used in support of initialization
  // of Gpu buffer
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  // Attach to IPC memory handle shared by parent process
  void* ipc_ptr;
  hsa_status_t err;
  err = hsa_amd_ipc_memory_attach(const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle),
                                  shared_->size, 1, ag_list, &ipc_ptr);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 200);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: Failure in attaching to IPC memory handle\n");
  PROCESS_LOG("Child: Attached to IPC buffer shared by parent process\n");
  PROCESS_LOG("Child: Address of buffer enabled for IPC: %p\n", ipc_ptr);

  // Attach to IPC signal handle shared by parent process
  hsa_signal_t ipc_signal;
  err = hsa_amd_ipc_signal_attach(const_cast<hsa_amd_ipc_signal_t*>(&shared_->signal_handle),
                                  &ipc_signal);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 201);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: Failure in attaching to IPC signal handle\n");
  PROCESS_LOG("Child: Attached to IPC signal shared by parent process\n");

  // Validate Gpu buffer is filled per expectation i.e. if so update
  // per previously agreed upon value (first_val_ and second_val_)
  CheckAndFillBuffer(reinterpret_cast<uint32_t*>(ipc_ptr), first_val_, second_val_);
  PROCESS_LOG("Child: Confirmed DWord's of IPC buffer has: %d\n", first_val_);
  PROCESS_LOG("Child: Updated DWord's of IPC buffer to: %d\n", second_val_);

  // Register an async handler, we wait for parent process to set buffer value to
  // third_val_. During the callback, SignalCallbackHandler  will set cb_result
  // to fourth_val_ and increment cb_data->token
  struct signal_cb_handler_data child_cb_data;
  child_cb_data.obj = this;
  child_cb_data.exp_sig_value = 3;
  child_cb_data.exp_value = third_val_;
  child_cb_data.destination = reinterpret_cast<uint32_t*>(ipc_ptr);
  child_cb_data.new_value = fourth_val_;
  child_cb_data.token = 0;

  err = hsa_amd_signal_async_handler(ipc_signal, HSA_SIGNAL_CONDITION_GTE, 3, &SignalCallbackHandler, &child_cb_data);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 202);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: Failure registering async_handler to ipc_signal\n");
  PROCESS_LOG("Child: [pid:%d] Attached async handler to IPC signal shared by parent process\n", getpid());

  // Signal parent process to wake up and continue.
  // The next time parent process updates ipc_signal, SignalCallbackHandler will
  // be called
  hsa_signal_store_release(ipc_signal, 2);

  // Wait for SignalCallbackHandler to be called
  while (child_cb_data.token <= 0)
    sched_yield();

  PROCESS_LOG("Child: Confirmed DWord's of IPC buffer has: %d\n", third_val_);
  PROCESS_LOG("Child: Updated DWord's of IPC buffer to: %d\n", fourth_val_);

  // Signal parent process to wake up and continue
  hsa_signal_store_release(ipc_signal, 4);

  hsa_signal_value_t ret = 1;
  while(true) {
    ret = hsa_signal_wait_acquire(ipc_signal, HSA_SIGNAL_CONDITION_LT, 0, timeout_, HSA_WAIT_STATE_BLOCKED);
    if (shared_->child_status == -1) {
      exit(0);
    }
    if (ret < 0) {
      break;
    }
  }
  USR_TRIGGERED_FAILURE(ret, HSA_STATUS_ERROR, 203);
  FORK_ASSERT_EQ(-1, ret, "Child: Expected signal value of 0, but got " << ret << "\n");

  // Detach IPC memory that was used to test
  err = hsa_amd_ipc_memory_detach(ipc_ptr);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 204);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: Failure in detaching IPC memory handle\n");
  PROCESS_LOG("Child: Detached IPC memory handle\n");

  // Reset the signal object and release acquired resources
  err = hsa_signal_destroy(ipc_signal);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 205);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Child: Failure in destroying IPC signal handle\n");
  PROCESS_LOG("Child: IPC test PASSED\n");
}

void IPCTest::ParentProcessImpl() {

  // Ignoring the first allocation to exercise fragment allocation.
  hsa_status_t err;
  uint32_t* discard = NULL;
  err = hsa_amd_memory_pool_allocate(device_pool(), gpu_mem_granule, 0,
                                     reinterpret_cast<void**>(&discard));
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 100);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to allocate gpu memory\n");

  // Allocate some VRAM that is used to test IPC
  uint32_t* gpuBuf = NULL;
  err = hsa_amd_memory_pool_allocate(device_pool(), gpu_mem_granule, 0,
                                     reinterpret_cast<void**>(&gpuBuf));
  PROCESS_LOG("Parent: Allocated framebuffer of size: %zu\n", gpu_mem_granule);
  PROCESS_LOG("Parent: Address of allocated framebuffer: %p\n", gpuBuf);

  // Free the test allocation of memory block
  err = hsa_amd_memory_pool_free(discard);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 101);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to free gpu memory\n");

  // List of devices involved in test. Gpu device is used
  // to allocate buffer and signal that are part of an IPC
  // transaction. Cpu is used in support of initialization
  // of Gpu buffer
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  // Grant access to buffer to participating devices
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, gpuBuf);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 102);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to get access to gpu memory\n");

  // Update shared data structure's buffer related parameters
  shared_->size = gpu_mem_granule;
  shared_->count = gpu_mem_granule / sizeof(uint32_t);

  // Initialize every DWord of IPC buffer with a value per previous
  // agreement i.e. first_val_
  err = hsa_amd_memory_fill(gpuBuf, first_val_, shared_->count);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 103);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to initialize gpu memory\n");
  PROCESS_LOG("Parent: Initialized Dword's of framebuffer with: %d\n", first_val_);

  // Create an IPC memory handle. IPC handle value is shared with
  // child process via a shared data structure
  err = hsa_amd_ipc_memory_create(gpuBuf, gpu_mem_granule,
                                  const_cast<hsa_amd_ipc_memory_t*>(&shared_->handle));
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 104);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to create IPC memory handle\n");
  PROCESS_LOG("Parent: Created IPC handle for framebuffer: %p\n", gpuBuf);

  // Create a signal that is capable of IPC. Also obtain a IPC handle
  // which is shared with child process via a shared data structure
  hsa_signal_t ipc_signal;
  err = hsa_amd_signal_create(1, 0, NULL, HSA_AMD_SIGNAL_IPC, &ipc_signal);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 105);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to create IPC signal\n");
  err = hsa_amd_ipc_signal_create(ipc_signal,
                                  const_cast<hsa_amd_ipc_signal_t*>(&shared_->signal_handle));
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 106);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to create IPC signal handle\n");
  PROCESS_LOG("Parent: Created IPC handle associated with ipc_signal\n");

  // Signal child process that the gpu buffer is ready to read.
  PROCESS_LOG("Parent: Signalling child proces process\n");
  CheckAndSetToken(&shared_->token, 1);
  PROCESS_LOG("Parent: Waiting for signal from child process\n");

  // Wait for child processs to signal. Child will update signal object
  // value to TWO (2). Check signal value is per expectation
  hsa_signal_value_t ret = 1;
  while(true) {
    ret = hsa_signal_wait_acquire(ipc_signal, HSA_SIGNAL_CONDITION_GTE, 2, timeout_, HSA_WAIT_STATE_BLOCKED);
    if (shared_->child_status == -1) {
      exit(0);
    }
    if (ret >= 2) {
      break;
    }
  }
  USR_TRIGGERED_FAILURE(ret, HSA_STATUS_ERROR, 107);
  FORK_ASSERT_EQ(2, ret, "Parent: Expected signal value of 2, but got " << ret << "\n");

  // Verify child process has updated all DWords of buffer per
  // previously agreed upon values (second_val_ and third_val_)
  CheckAndFillBuffer(gpuBuf, second_val_, third_val_);
  PROCESS_LOG("Parent: Confirmed DWord's of frambuffer has: %d\n", second_val_);
  PROCESS_LOG("Parent: Updated DWord's of framebuffer to: %d\n", third_val_);

  hsa_signal_store_relaxed(ipc_signal, 3);

  while(true) {
    ret = hsa_signal_wait_acquire(ipc_signal, HSA_SIGNAL_CONDITION_GTE, 4, timeout_, HSA_WAIT_STATE_BLOCKED);
    if (shared_->child_status == -1) {
      exit(0);
    }
    if (ret >= 4) {
      break;
    }
  }

  CheckAndFillBuffer(gpuBuf, fourth_val_, 0);
  PROCESS_LOG("Parent: Confirmed DWord's of frambuffer has: %d\n", fourth_val_);

  USR_TRIGGERED_FAILURE(ret, HSA_STATUS_ERROR, 108);
  FORK_ASSERT_EQ(4, ret, "Parent: Expected signal value of 4, but got " << ret << "\n");

  // Reset the signal object and release acquired resources
  hsa_signal_store_relaxed(ipc_signal, -1);
  err = hsa_signal_destroy(ipc_signal);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 109);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failure in destroying IPC signal\n");
  err = hsa_amd_memory_pool_free(gpuBuf);
  USR_TRIGGERED_FAILURE(err, HSA_STATUS_ERROR, 110);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "Parent: Failed to free gpu memory\n");
  PROCESS_LOG("Parent: IPC test PASSED\n");

  // Wait for child process to terminate before exiting
  int exit_status = 0;
  waitpid(child_, &exit_status, 0);
  munmap(shared_, sizeof(Shared));
}

void IPCTest::PrintVerboseMesg(void) {
  // Collect names of GPU's
  hsa_status_t err;
  char name1[64] = {0};
  char name2[64] = {0};
  err = hsa_agent_get_info(*cpu_device(), HSA_AGENT_INFO_NAME, name1);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "hsa_agent_get_info() failed\n");
  err = hsa_agent_get_info(*gpu_device1(), HSA_AGENT_INFO_NAME, name2);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err, "hsa_agent_get_info() failed\n");

  // Collect BDF information of GPU's
  uint32_t loc1, loc2;
  err = hsa_agent_get_info(*cpu_device(), (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &loc1);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = hsa_agent_get_info(*gpu_device1(), (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &loc2);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Print the name and BDF info about the devices
  fprintf(stdout, "Using: %s (%d) and %s (%d)\n", name1, loc1, name2, loc2);
}

void IPCTest::CheckAndFillBuffer(void* gpu_src_ptr, uint32_t exp_cur_val, uint32_t new_val) {
  uint32_t* sysBuf;
  hsa_status_t err;
  hsa_signal_value_t sig;
  hsa_signal_t copy_signal;

  // Bind the size granularity of allocation
  size_t sz = gpu_mem_granule;

  // Allocate a signal to track copy progress
  err = hsa_signal_create(1, 0, NULL, &copy_signal);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Allocate buffer in system memory to validate
  err = hsa_amd_memory_pool_allocate(cpu_pool(), sz, 0, reinterpret_cast<void**>(&sysBuf));
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Enable access to buffer in system memory
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, sysBuf);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Copy data to buffer in system memory
  err = hsa_amd_memory_async_copy(sysBuf, *cpu_device(), gpu_src_ptr, *gpu_device1(), sz, 0, NULL,
                                  copy_signal);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Wait for copy to complete
  sig = hsa_signal_wait_relaxed(copy_signal,
                   HSA_SIGNAL_CONDITION_LT, 1, -1, HSA_WAIT_STATE_BLOCKED);
  FORK_ASSERT_EQ(0, sig, "Expected signal 0, but got " << sig << "\n");

  // Validate buffer has expected data
  uint32_t count = sz / sizeof(uint32_t);
  for (uint32_t idx = 0; idx < count; idx++) {
    if (exp_cur_val != sysBuf[idx]) {
      PROCESS_LOG("Validation failed: expected: %d observed: %d at index: %d\n",
                  exp_cur_val, sysBuf[idx], idx);
      FORK_ASSERT_EQ(exp_cur_val, sysBuf[idx]);
    }
    sysBuf[idx] = new_val;
  }

  // Reset copy signal and update buffer in Gpu with new value
  hsa_signal_store_relaxed(copy_signal, 1);
  err = hsa_amd_memory_async_copy(gpu_src_ptr, *gpu_device1(), sysBuf, *cpu_device(), sz, 0, NULL,
                                  copy_signal);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Wait for copy to complete
  sig = hsa_signal_wait_relaxed(copy_signal, HSA_SIGNAL_CONDITION_LT, 1, -1, HSA_WAIT_STATE_BLOCKED);
  FORK_ASSERT_EQ(sig, 0, "Expected signal 0, but got " << sig << "\n");

  // Release resources allocated by this method
  err = hsa_signal_destroy(copy_signal);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = hsa_amd_memory_pool_free(sysBuf);
  FORK_ASSERT_EQ(HSA_STATUS_SUCCESS, err);
}

void IPCTest::Run(void) {
  TestBase::Run();

  // Collect and print debug information
  if (verbosity() >= VERBOSE_STANDARD) {
    PrintVerboseMesg();
  }

  // Note: Close() (and hsa_shut_down()) will be called from main()
  // processOne is true for parent process, false for child process
  if (parentProcess_) {
    ParentProcessImpl();
  } else {
    ChildProcessImpl();
    exit(0);
  }

  return;
}

void IPCTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void IPCTest::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void IPCTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

#undef PROCESS_LOG
#undef FORK_ASSERT_EQ
#undef MSG
#undef Y
