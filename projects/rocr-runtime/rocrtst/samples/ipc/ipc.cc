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

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/wait.h>

#include <cassert>
#include <iostream>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

static const uint32_t kShmemID = 1594685;

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    const char* msg = 0; \
    hsa_status_string(err, &msg); \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
                          __FILE__ << ". Call returned " << err << std::endl; \
    std::cout << msg << std::endl; \
    return (err); \
  } \
}

struct callback_args {
  hsa_agent_t host;
  hsa_agent_t device;
  hsa_amd_memory_pool_t cpu_pool;
  hsa_amd_memory_pool_t gpu_pool;
  size_t gpu_mem_granule;
};

// This function will test whether the provided memory pool is 1) in the
// GLOBAL segment, 2) allows allocation and 3) is accessible by the provided
// agent. If the provided pool meets these criteria, HSA_STATUS_INFO_BREAK is
// returned
static hsa_status_t
FindPool(hsa_amd_memory_pool_t in_pool, hsa_agent_t agent) {
  hsa_amd_segment_t segment;
  hsa_status_t err;

  err = hsa_amd_memory_pool_get_info(in_pool,
                                  HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  RET_IF_HSA_ERR(err);
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  bool canAlloc;
  err = hsa_amd_memory_pool_get_info(in_pool,
                   HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &canAlloc);
  RET_IF_HSA_ERR(err);
  if (!canAlloc) {
     return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_pool_access_t access =
                                     HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  err = hsa_amd_agent_memory_pool_get_info(agent, in_pool,
                              HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
  RET_IF_HSA_ERR(err);

  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    return HSA_STATUS_SUCCESS;
  }

  return HSA_STATUS_INFO_BREAK;
}

// Callback function for hsa_amd_agent_iterate_memory_pools(). If the provided
// pool is suitable (see comments for FindPool()), HSA_STATUS_INFO_BREAK is
// returned. The input parameter "data" should point to memory for a "struct
// callback_args", which includes a gpu pool and a granule field.  These fields
// will be filled in by this function if the provided pool meets all the
// requirements.
static hsa_status_t FindDevicePool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_status_t err;

  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  struct callback_args *args = (struct callback_args *)data;

  err = FindPool(pool, args->device);

  if (err == HSA_STATUS_INFO_BREAK) {
    args->gpu_pool = pool;


#ifdef ROCRTST_EMULATOR_BUILD
  args->gpu_mem_granule = 4;
#else
    err = hsa_amd_memory_pool_get_info(args->gpu_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &args->gpu_mem_granule);
    RET_IF_HSA_ERR(err);
#endif

    // We found what we were looking for, so return HSA_STATUS_INFO_BREAK
    return HSA_STATUS_INFO_BREAK;
  }

  return HSA_STATUS_SUCCESS;
}

// Callback function for hsa_amd_agent_iterate_memory_pools(). If the provided
// pool is suitable (see comments for FindPool()), HSA_STATUS_INFO_BREAK is
// returned. The input parameter "data" should point to memory for a "struct
// callback_args", which includes a cpu pool. This field will be filled in by
// this function if the provided pool meets all the requirements.
static hsa_status_t FindCPUPool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_status_t err;

  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  struct callback_args *args = (struct callback_args *)data;

  err = FindPool(pool, args->host);

  if (err == HSA_STATUS_INFO_BREAK) {
    args->cpu_pool = pool;
  }
  return err;
}


// This function is meant to be a call-back to hsa_iterate_agents. Find the
// first GPU agent that has memory accessible by CPU
// Return values:
//  HSA_STATUS_INFO_BREAK -- 2 GPU agents have been found and stored. Iterator
//    should stop iterating
//  HSA_STATUS_SUCCESS -- 2 GPU agents have not yet been found; iterator
//    should keep iterating
//  Other -- Some error occurred
static hsa_status_t FindGpu(hsa_agent_t agent, void *data) {
  if (data == NULL) {
     return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_device_type_t hsa_device_type;
  hsa_status_t err = hsa_agent_get_info(agent,
                                     HSA_AGENT_INFO_DEVICE, &hsa_device_type);
  RET_IF_HSA_ERR(err);

  if (hsa_device_type != HSA_DEVICE_TYPE_GPU) {
    return HSA_STATUS_SUCCESS;
  }

  struct callback_args *args = (struct callback_args *)data;

  // Make sure GPU device has pool host can access
  args->device = agent;
  err = hsa_amd_agent_iterate_memory_pools(agent, FindDevicePool, args);

  if (err == HSA_STATUS_INFO_BREAK) {
    // We were looking for, so return HSA_STATUS_INFO_BREAK
    return HSA_STATUS_INFO_BREAK;
  } else {
    args->device = {0};
  }

  RET_IF_HSA_ERR(err);

  // Returning HSA_STATUS_SUCCESS tells the calling iterator to keep iterating
  return HSA_STATUS_SUCCESS;
}

// This function is meant to be a call-back to hsa_iterate_agents. For each
// input agent the iterator provides as input, this function will check to
// see if the input agent is a CPU. If so, it will update the callback_args
// structure pointed to by the input parameter "data".

// Return values:
//  HSA_STATUS_INFO_BREAK -- CPU agent has been found and stored. Iterator
//    should stop iterating
//  HSA_STATUS_SUCCESS -- CPU agent has not yet been found; iterator
//    should keep iterating
//  Other -- Some error occurred
static hsa_status_t FindCPUDevice(hsa_agent_t agent, void *data) {
  if (data == NULL) {
     return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_device_type_t hsa_device_type;
  hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE,
                                                            &hsa_device_type);
  RET_IF_HSA_ERR(err);

  if (hsa_device_type == HSA_DEVICE_TYPE_CPU) {
    struct callback_args *args = (struct callback_args *)data;

    args->host = agent;

    err = hsa_amd_agent_iterate_memory_pools(agent, FindCPUPool, args);

    if (err == HSA_STATUS_INFO_BREAK) {  // we found what we were looking for
      return HSA_STATUS_INFO_BREAK;
    } else {
      args->host = {0};
      return err;
    }
  }

  // Returning HSA_STATUS_SUCCESS tells the calling iterator to keep iterating
  return HSA_STATUS_SUCCESS;
}

// This function will test whether the gpu-local buffer has been filled
// with an expected value and return an error if not. The expected value is
// also replaced with a new value.
// Implementation notes: We create a buffer in system memory and copy
// the gpu-local data buffer to be tested to this system memory buffer.
// We also write the system memory buffer with the new value, and then copy
// it back the gpu-local buffer.
static hsa_status_t
CheckAndFillBuffer(struct callback_args *args, void *gpu_src_ptr,
                                     uint32_t exp_cur_val, uint32_t new_val) {
  hsa_signal_t copy_signal;
  size_t sz = args->gpu_mem_granule;
  hsa_agent_t cpu_ag = args->host;
  hsa_agent_t gpu_ag = args->device;
  hsa_status_t err;

  err = hsa_signal_create(1, 0, NULL, &copy_signal);
  RET_IF_HSA_ERR(err);

  uint32_t *sysBuf;

  err = hsa_amd_memory_pool_allocate(args->cpu_pool, sz, 0,
                                          reinterpret_cast<void **>(&sysBuf));
  RET_IF_HSA_ERR(err);

  hsa_agent_t ag_list[2] = {args->device, args->host};
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, sysBuf);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_async_copy(sysBuf, cpu_ag, gpu_src_ptr, gpu_ag,
                                                    sz, 0, NULL, copy_signal);
  RET_IF_HSA_ERR(err);

  if (hsa_signal_wait_relaxed(copy_signal, HSA_SIGNAL_CONDITION_LT,
                                       1, -1, HSA_WAIT_STATE_BLOCKED) != 0) {
    printf("Async copy returned error value.\n");
    return HSA_STATUS_ERROR;
  }

  uint32_t count = sz/sizeof(uint32_t);

  for (uint32_t i = 0; i < count; ++i) {
    if (sysBuf[i] != exp_cur_val) {
      fprintf(stdout, "Expected %d but got %d in buffer.\n",
                                                      exp_cur_val, sysBuf[i]);
      err = HSA_STATUS_ERROR;
      break;
    }
    sysBuf[i] = new_val;
  }

  hsa_signal_store_relaxed(copy_signal, 1);

  err = hsa_amd_memory_async_copy(gpu_src_ptr, gpu_ag, sysBuf, cpu_ag,
                                                    sz, 0, NULL, copy_signal);
  RET_IF_HSA_ERR(err);

  if (hsa_signal_wait_relaxed(copy_signal, HSA_SIGNAL_CONDITION_LT,
                                       1, -1, HSA_WAIT_STATE_BLOCKED) != 0) {
    printf("Async copy returned error value.\n");
    return HSA_STATUS_ERROR;
  }

  err = hsa_signal_destroy(copy_signal);
  RET_IF_HSA_ERR(err);

  err = hsa_amd_memory_pool_free(sysBuf);
  RET_IF_HSA_ERR(err);

  return HSA_STATUS_SUCCESS;
}

// See if the other process wrote an error value to the token; if not, write
// the newVal to the token.
static void CheckAndSetToken(volatile int *token, int newVal) {
  if (*token == -1) {
    printf("Error in other process. Exiting.\n");
    exit(-1);
  } else {
    *token = newVal;
  }
}

// Summary of this IPC Sample:
// This program demonstrates the IPC apis. Run it by executing 2 instances
// of the program.
// The first process will allocate some gpu-local memory and fill it with
// 1's. This HSA buffer will be made shareable with hsa_amd_ipc_memory_create()
// The 2nd process will access this shared buffer with
// hsa_amd_ipc_memory_attach(), verify that 1's were written, and then fill
// the buffer with 2's. Finally, the first process will then read the
// gpu-local buffer and verify that the 2's were indeed written. The main
// point is to show how hsa memory buffer handles can be shared among
// processes.
//
// Implementation Notes:
// -Standard linux shared memory is used in this sample program as a way
// of sharing info and  synchronizing the 2 processes. This is independent
// of RocR IPC and should not be confused with it.
int main(int argc, char** argv) {
  // IPC test
  struct Shared {
    volatile int token;
    volatile int count;
    volatile size_t size;
    volatile hsa_amd_ipc_memory_t handle;
    volatile hsa_amd_ipc_signal_t signal_handle;
  };

  // Allocate linux shared memory.
  Shared* shared = (Shared*)mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (shared == MAP_FAILED) {
    fprintf(stdout, "Unable to allocate shared memory. Exiting.\n");
    return -1;
  }

  // "token" is used to signal state changes between the 2 processes.
  volatile int* token = &shared->token;
  *token = 0;
  bool processOne;

  // Spawn second process and verify communication
  int child = fork();
  if (child == -1) {
    printf("fork failed.  Exiting.\n");
    return -1;
  }
  if (child != 0) {
    processOne = true;

    // Signal to other process we are waiting, and then wait...
    *token = 1;
    while (*token == 1) {
      sched_yield();
    }

    fprintf(stdout, "Second process observed, handshake...\n");
    *token = 1;
    while (*token == 1) {
      sched_yield();
    }
  } else {
    processOne = false;
    fprintf(stdout, "Second process running.\n");

    while (*token == 0) {
      sched_yield();
    }

    CheckAndSetToken(token, 0);
    // Wait for handshake
    while (*token == 0) {
      sched_yield();
    }
    CheckAndSetToken(token, 0);
    fprintf(stdout, "Handshake complete.\n");
  }

  hsa_status_t err;

  err = hsa_init();
  RET_IF_HSA_ERR(err);

  struct callback_args args = {0, 0, 0};

  err = hsa_iterate_agents(FindCPUDevice, &args);
  assert(err == HSA_STATUS_INFO_BREAK);
  if (err != HSA_STATUS_INFO_BREAK) {
    return -1;
  }

  err = hsa_iterate_agents(FindGpu, &args);

  if (err != HSA_STATUS_INFO_BREAK) {
    printf(
     "No GPU with accessible VRAM required for this program found. Exiting\n");
    return -1;
  }

  // Print out name of the device.
  char name1[64] = {0};
  char name2[64] = {0};
  err = hsa_agent_get_info(args.host, HSA_AGENT_INFO_NAME, name1);
  RET_IF_HSA_ERR(err);
  err = hsa_agent_get_info(args.device, HSA_AGENT_INFO_NAME, name2);
  RET_IF_HSA_ERR(err);
  uint16_t loc1, loc2;
  err = hsa_agent_get_info(args.host,
                           (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &loc1);
  RET_IF_HSA_ERR(err);
  err = hsa_agent_get_info(args.device,
                           (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &loc2);
  RET_IF_HSA_ERR(err);
  fprintf(stdout, "Using: %s (%d) and %s (%d)\n", name1, loc1, name2, loc2);

  // Get signal for async copy
  hsa_signal_t copy_signal;
  err = hsa_signal_create(1, 0, NULL, &copy_signal);
  RET_IF_HSA_ERR(err);

// Wrap printf to add first or second process indicator
#define PROCESS_LOG(format, ...) \
    fprintf(stdout, "line:%d P%u: " format, \
                      __LINE__, static_cast<int>(!processOne), ##__VA_ARGS__);

  hsa_agent_t ag_list[2] = {args.device, args.host};

  if (processOne) {
    // Allocate some VRAM and fill it with 1's
    uint32_t* gpuBuf = NULL;
    err = hsa_amd_memory_pool_allocate(args.gpu_pool, args.gpu_mem_granule, 0,
                                            reinterpret_cast<void**>(&gpuBuf));
    RET_IF_HSA_ERR(err);

    PROCESS_LOG("Allocated local memory buffer at %p\n", gpuBuf);

    err = hsa_amd_agents_allow_access(2, ag_list, NULL, gpuBuf);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_ipc_memory_create(gpuBuf, args.gpu_mem_granule,
                          const_cast<hsa_amd_ipc_memory_t*>(&shared->handle));
    PROCESS_LOG(
    "Created IPC handle associated with gpu-local buffer at P0 address %p\n",
                                                                      gpuBuf);

    RET_IF_HSA_ERR(err);

    uint32_t count = args.gpu_mem_granule/sizeof(uint32_t);
    shared->size = args.gpu_mem_granule;
    shared->count = count;

    err = hsa_amd_memory_fill(gpuBuf, 1, count);
    RET_IF_HSA_ERR(err);

    // Get IPC capable signal
    hsa_signal_t ipc_signal;
    err = hsa_amd_signal_create(1, 0, NULL, HSA_AMD_SIGNAL_IPC, &ipc_signal);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_ipc_signal_create(ipc_signal,
                                    const_cast<hsa_amd_ipc_signal_t*>(&shared->signal_handle));
    PROCESS_LOG("Created IPC handle associated with ipc_signal\n");
    RET_IF_HSA_ERR(err);

    // Signal Process 2 that the gpu buffer is ready to read.
    CheckAndSetToken(token, 1);

    PROCESS_LOG("Allocated buffer and filled it with 1's. Wait for P1...\n");
    hsa_signal_value_t ret =
        hsa_signal_wait_acquire(ipc_signal, HSA_SIGNAL_CONDITION_NE, 1, -1, HSA_WAIT_STATE_BLOCKED);

    if (ret != 2) {
      hsa_signal_store_release(ipc_signal, -1);
      return -1;
    }

    err = CheckAndFillBuffer(&args, gpuBuf, 2, 0);
    RET_IF_HSA_ERR(err);
    PROCESS_LOG("Confirmed P1 filled buffer with 2\n")
    PROCESS_LOG("PASSED on P0\n");

    hsa_signal_store_relaxed(ipc_signal, 0);
    
    err = hsa_signal_destroy(ipc_signal);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_free(gpuBuf);
    RET_IF_HSA_ERR(err);

    waitpid(child, nullptr, 0);

  } else {  // "ProcessTwo"
    PROCESS_LOG("Waiting for process 0 to write 1 to token...\n");
    while (*token == 0) {
      sched_yield();
    }
    if (*token != 1) {
      *token = -1;
      return -1;
    }

    // Attach shared VRAM
    void* ptr;
    err = hsa_amd_ipc_memory_attach(
      const_cast<hsa_amd_ipc_memory_t*>(&shared->handle), shared->size, 1,
                                                               ag_list, &ptr);
    RET_IF_HSA_ERR(err);

    PROCESS_LOG(
     "Attached to IPC handle; P1 buffer address gpu-local memory is %p\n",
                                                                         ptr);

    // Attach shared signal
    hsa_signal_t ipc_signal;
    err = hsa_amd_ipc_signal_attach(const_cast<hsa_amd_ipc_signal_t*>(&shared->signal_handle),
                                    &ipc_signal);
    RET_IF_HSA_ERR(err);

    PROCESS_LOG("Attached to signal IPC handle\n");

    err = CheckAndFillBuffer(&args, reinterpret_cast<uint32_t *>(ptr), 1, 2);
    RET_IF_HSA_ERR(err);

    PROCESS_LOG(
      "Confirmed P0 filled buffer with 1; P1 re-filled buffer with 2\n");
    PROCESS_LOG("PASSED on P1\n");

    hsa_signal_store_release(ipc_signal, 2);

    err = hsa_amd_ipc_memory_detach(ptr);
    RET_IF_HSA_ERR(err);

    hsa_signal_wait_relaxed(ipc_signal, HSA_SIGNAL_CONDITION_NE, 2, -1, HSA_WAIT_STATE_BLOCKED);

    err = hsa_signal_destroy(ipc_signal);
    RET_IF_HSA_ERR(err);
  }

  err = hsa_signal_destroy(copy_signal);
  RET_IF_HSA_ERR(err);

  munmap(shared, sizeof(Shared));

  err = hsa_shut_down();
  RET_IF_HSA_ERR(err);

#undef PROCESS_LOG
  return 0;
}
