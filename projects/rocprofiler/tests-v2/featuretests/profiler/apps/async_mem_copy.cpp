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
#include <cassert>
#include <iostream>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define RET_IF_HSA_ERR(err)                                                                        \
  {                                                                                                \
    if ((err) != HSA_STATUS_SUCCESS) {                                                             \
      const char* msg = 0;                                                                         \
      hsa_status_string(err, &msg);                                                                \
      std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << __FILE__           \
                << ". Call returned " << err << std::endl;                                         \
      std::cout << msg << std::endl;                                                               \
      return (err);                                                                                \
    }                                                                                              \
  }

static const uint32_t kTestFillValue1 = 0xabcdef12;
static const uint32_t kTestFillValue2 = 0xba5eba11;
static const uint32_t kTestFillValue3 = 0xfeed5a1e;
static const uint32_t kTestInitValue = 0xbaadf00d;

// This structure holds an agent pointer and associated memory pool to be used
// for this test program.
struct async_mem_cpy_agent {
  hsa_agent_t dev;
  hsa_amd_memory_pool_t pool;
  size_t granule;
  void* ptr;
};
struct async_mem_cpy_pool_query {
  async_mem_cpy_agent* pool_info;
  hsa_agent_t peer_device;
};
struct callback_args {
  struct async_mem_cpy_agent cpu;
  struct async_mem_cpy_agent gpu1;
  struct async_mem_cpy_agent gpu2;
};
// Find the least common multiple of 2 numbers
static uint32_t lcm(uint32_t a, uint32_t b) {
  int tmp_a;
  int tmp_b;
  tmp_a = a;
  tmp_b = b;
  while (tmp_a != tmp_b) {
    if (tmp_a < tmp_b) {
      tmp_a = tmp_a + a;
    } else {
      tmp_b = tmp_b + b;
    }
  }
  return tmp_a;
}
// This function is a callback for hsa_amd_agent_iterate_memory_pools()
// and will test whether the provided memory pool is 1) in the GLOBAL
// segment, 2) allows allocation and 3) is accessible by the provided
// agent. The "data" input parameter is assumed to be pointing to a
// struct async_mem_cpy_agent. If the provided pool meets these criteria,
// HSA_STATUS_INFO_BREAK is returned.
static hsa_status_t FindPool(hsa_amd_memory_pool_t in_pool, void* data) {
  hsa_amd_segment_t segment;
  hsa_status_t err;
  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  struct async_mem_cpy_pool_query* args = (struct async_mem_cpy_pool_query*)data;
  err = hsa_amd_memory_pool_get_info(in_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  RET_IF_HSA_ERR(err);
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }
  bool canAlloc;
  err = hsa_amd_memory_pool_get_info(in_pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                     &canAlloc);
  RET_IF_HSA_ERR(err);
  if (!canAlloc) {
    return HSA_STATUS_SUCCESS;
  }
  if (args->peer_device.handle != 0) {
    hsa_amd_memory_pool_access_t access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    err = hsa_amd_agent_memory_pool_get_info(args->peer_device, in_pool,
                                             HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
    RET_IF_HSA_ERR(err);
    if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      return HSA_STATUS_SUCCESS;
    }
  }
  err = hsa_amd_memory_pool_get_info(in_pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                     &args->pool_info->granule);
  RET_IF_HSA_ERR(err);
  args->pool_info->pool = in_pool;
  return HSA_STATUS_INFO_BREAK;
}
// This function is meant to be a callback to hsa_iterate_agents. For each
// input agent the iterator provides as input, this function will check to
// see if the input agent is a CPU agent. If so, it will update the
// async_mem_cpy_agent structure pointed to by the input parameter "data".
// Return values:
//  HSA_STATUS_INFO_BREAK -- CPU agent has been found and stored. Iterator
//    should stop iterating
//  HSA_STATUS_SUCCESS -- CPU agent has not yet been found; iterator
//    should keep iterating
//  Other -- Some error occurred
static hsa_status_t FindCPUDevice(hsa_agent_t agent, void* data) {
  if (data == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  hsa_device_type_t hsa_device_type;
  hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
  RET_IF_HSA_ERR(err);
  if (hsa_device_type == HSA_DEVICE_TYPE_CPU) {
    struct async_mem_cpy_agent* args = (struct async_mem_cpy_agent*)data;
    args->dev = agent;
    async_mem_cpy_pool_query pool_query;
    pool_query.peer_device.handle = 0;
    pool_query.pool_info = args;
    err = hsa_amd_agent_iterate_memory_pools(agent, FindPool, &pool_query);
    if (err == HSA_STATUS_INFO_BREAK) {  // we found what we were looking for
      return HSA_STATUS_INFO_BREAK;
    } else {
      args->dev = {0};
      return err;
    }
  }
  // Returning HSA_STATUS_SUCCESS tells the calling iterator to keep iterating
  return HSA_STATUS_SUCCESS;
}
// This function is meant to be a callback to hsa_iterate_agents. It will
// attempt to find 2, or at least 1 GPU agent suitable for our test. The data
// input parameter should point to a callback_args struct. The 2 GPU fields
// will be updated as GPUs are discovered.
// Return values:
//  HSA_STATUS_INFO_BREAK -- 2 GPU agents have been found and stored. Iterator
//    should stop iterating
//  HSA_STATUS_SUCCESS -- 2 GPU agents have not yet been found; 0 or 1 may
//    have been found; iterator function should keep iterating
//  Other -- Some error occurred
static hsa_status_t FindGPUs(hsa_agent_t agent, void* data) {
  if (data == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  hsa_device_type_t hsa_device_type;
  hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
  RET_IF_HSA_ERR(err);
  if (hsa_device_type != HSA_DEVICE_TYPE_GPU) {
    return HSA_STATUS_SUCCESS;
  }
  struct callback_args* args = (struct callback_args*)data;
  struct async_mem_cpy_agent* gpu;
  async_mem_cpy_pool_query pool_query = {0, 0};
  if (args->gpu1.dev.handle == 0) {
    gpu = &args->gpu1;
  } else {
    gpu = &args->gpu2;
    // Check that gpu1 has peer access into the selected pool.
    pool_query.peer_device = args->gpu1.dev;
  }
  // Make sure GPU device has pool host can access
  gpu->dev = agent;
  pool_query.pool_info = gpu;
  err = hsa_amd_agent_iterate_memory_pools(agent, FindPool, &pool_query);
  if (err == HSA_STATUS_INFO_BREAK) {
    if (gpu == &args->gpu2) {
      // We found 2 gpu's
      return HSA_STATUS_INFO_BREAK;
    } else {
      // Keep looking for another gpu
      return HSA_STATUS_SUCCESS;
    }
  } else {
    gpu->dev = {0};
  }
  RET_IF_HSA_ERR(err);
  // Returning HSA_STATUS_SUCCESS tells the calling iterator to keep iterating
  return HSA_STATUS_SUCCESS;
}
// This is the main test, showing various paths of async. copy. Source and
// destination agents and their respective pools should already be discovered.
// Additionally, buffer from the pools should already be allocated and availble
// from the input parameters.
static hsa_status_t AsyncCpyTest(async_mem_cpy_agent* dst, async_mem_cpy_agent* src,
                                 callback_args* args, size_t sz, uint32_t val) {
  hsa_status_t err;
  hsa_signal_t copy_signal;
  // Initialize the system and destination buffers with a value so we can later
  // validate it has been overwritten
  void* sysPtr = args->cpu.ptr;
  err = hsa_amd_memory_fill(sysPtr, kTestInitValue, sz / sizeof(uint32_t));
  RET_IF_HSA_ERR(err);
  if (dst->ptr != sysPtr) {
    err = hsa_amd_memory_fill(dst->ptr, kTestInitValue, sz / sizeof(uint32_t));
    RET_IF_HSA_ERR(err);
  }
  // Fill the source buffer with the provided uint32_t value
  err = hsa_amd_memory_fill(src->ptr, val, sz / sizeof(uint32_t));
  RET_IF_HSA_ERR(err);
  // Make sure the target and destination agents have access to the buffer.
  hsa_agent_t ag_list[2] = {dst->dev, src->dev};
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, dst->ptr);
  RET_IF_HSA_ERR(err);
  // Create a signal that will be used to inform us when the copy is done
  err = hsa_signal_create(1, 0, NULL, &copy_signal);
  RET_IF_HSA_ERR(err);
  // Do the copy...
  err = hsa_amd_memory_async_copy(dst->ptr, dst->dev, src->ptr, src->dev, sz, 0, NULL, copy_signal);
  RET_IF_HSA_ERR(err);
  // Here we do a blocking wait. Alternatively, we could also use a
  // non-blocking wait in a loop, and do other work while waiting.
  if (hsa_signal_wait_relaxed(copy_signal, HSA_SIGNAL_CONDITION_LT, 1, -1,
                              HSA_WAIT_STATE_BLOCKED) != 0) {
    printf("Async copy returned error value.\n");
    return HSA_STATUS_ERROR;
  }
  // Verify the copy was successful; copy from the dst buffer to the sysBuf,
  // (if the result is not already in sys. mem.) and check the sysBuf values
  if (dst->ptr != sysPtr) {
    if (src->ptr != sysPtr) {
      // In this case, we need to give the gpu dev that owns dst->ptr access
      // to the system memory we are going to copy to.
      hsa_agent_t ag_list_ck[2] = {dst->dev, args->cpu.dev};
      err = hsa_amd_agents_allow_access(2, ag_list_ck, NULL, sysPtr);
      RET_IF_HSA_ERR(err);
    }
    // Reset signal to 1
    hsa_signal_store_screlease(copy_signal, 1);
    err = hsa_amd_memory_async_copy(sysPtr, args->cpu.dev, dst->ptr, dst->dev, sz, 0, NULL,
                                    copy_signal);
    RET_IF_HSA_ERR(err);
    if (hsa_signal_wait_relaxed(copy_signal, HSA_SIGNAL_CONDITION_LT, 1, -1,
                                HSA_WAIT_STATE_BLOCKED) != 0) {
      printf("Async copy returned error value.\n");
      return HSA_STATUS_ERROR;
    }
  }
  // Check that the contents of the buffer are what is expected.
  for (uint32_t i = 0; i < sz / sizeof(uint32_t); ++i) {
    if (reinterpret_cast<uint32_t*>(sysPtr)[i] != val) {
      fprintf(stdout, "Expected 0x%x but got 0x%x in buffer at index %d.\n", val,
              reinterpret_cast<uint32_t*>(sysPtr)[i], i);
      return HSA_STATUS_ERROR;
    }
  }
  return HSA_STATUS_SUCCESS;
}
// This program illustrates the usage of the asynchronous copy capability of
// the RocR runtime library. The program will create a system memory buffer and
// a local buffer for each GPU, up to 2 GPUs, if the system has at least 2
// GPUs. The program will copy data to/from the host from/to the GPU. If 2
// GPUs are available, the program will also copy data from one to the other.
int main() {
  hsa_status_t err;
  struct callback_args args;
  bool twoGPUs = false;
  err = hsa_init();
  RET_IF_HSA_ERR(err);
  // First, find the cpu agent and associated pool
  args.cpu = {0, 0, 0};
  err = hsa_iterate_agents(FindCPUDevice, reinterpret_cast<void*>(&args.cpu));
  assert(err == HSA_STATUS_INFO_BREAK);
  if (err != HSA_STATUS_INFO_BREAK) {
    return -1;
  }
  // Now, find 1 or 2 (if possible) GPUs and associated pool(s) for our test
  args.gpu1 = {0, 0, 0};
  args.gpu2 = {0, 0, 0};
  err = hsa_iterate_agents(FindGPUs, &args);
  if (err == HSA_STATUS_INFO_BREAK) {
    twoGPUs = true;
  } else {
    // See if we at least have 1 GPU
    if (args.gpu1.dev.handle == 0) {
      fprintf(stdout, "GPU with accessible VRAM not found; at least 1 required. Exiting\n");
      return -1;
    }
    fprintf(stdout,
            "Only 1 GPU found with required VRAM. "
            "Peer-to-Peer copy will be skipped.\n");
  }
  // We will use the smallest amount of allocatable memory that works for all
  // potential sources and destinations of the copy
  size_t sz = lcm(args.cpu.granule, args.gpu1.granule);
  // Allocate memory on each source/destination
  if (twoGPUs) {
    sz = lcm(sz, args.gpu2.granule);
    err = hsa_amd_memory_pool_allocate(args.gpu2.pool, sz, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG,
                                       reinterpret_cast<void**>(&args.gpu2.ptr));
    RET_IF_HSA_ERR(err);
  }
  err = hsa_amd_memory_pool_allocate(args.cpu.pool, sz, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, reinterpret_cast<void**>(&args.cpu.ptr));
  RET_IF_HSA_ERR(err);
  err =
      hsa_amd_memory_pool_allocate(args.gpu1.pool, sz, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, reinterpret_cast<void**>(&args.gpu1.ptr));
  RET_IF_HSA_ERR(err);
  char name[64];
  err = hsa_agent_get_info(args.cpu.dev, HSA_AGENT_INFO_NAME, &name);
  fprintf(stdout, "CPU is \"%s\"\n", name);
  err = hsa_agent_get_info(args.gpu1.dev, HSA_AGENT_INFO_NAME, &name);
  fprintf(stdout, "GPU1 is \"%s\"\n", name);
  if (twoGPUs) {
    err = hsa_agent_get_info(args.gpu2.dev, HSA_AGENT_INFO_NAME, &name);
    fprintf(stdout, "GPU2 is \"%s\"\n", name);
  }
  fprintf(stdout, "Copying %lu bytes from gpu1 memory to system memory...\n", sz);
  err = AsyncCpyTest(&args.cpu, &args.gpu1, &args, sz, kTestFillValue1);
  RET_IF_HSA_ERR(err);
  fprintf(stdout, "Success!\n");
  fprintf(stdout, "Copying %lu bytes from system memory to gpu1 memory...\n", sz);
  err = AsyncCpyTest(&args.gpu1, &args.cpu, &args, sz, kTestFillValue2);
  RET_IF_HSA_ERR(err);
  fprintf(stdout, "Success!\n");
  if (twoGPUs) {
    fprintf(stdout, "Copying %lu bytes from gpu1 memory to gpu2 memory...\n", sz);
    err = AsyncCpyTest(&args.gpu2, &args.gpu1, &args, sz, kTestFillValue3);
    RET_IF_HSA_ERR(err);
    fprintf(stdout, "Success!\n");
  }
  // Clean up
  err = hsa_amd_memory_pool_free(args.cpu.ptr);
  RET_IF_HSA_ERR(err);
  err = hsa_amd_memory_pool_free(args.gpu1.ptr);
  RET_IF_HSA_ERR(err);
  if (twoGPUs) {
    err = hsa_amd_memory_pool_free(args.gpu2.ptr);
    RET_IF_HSA_ERR(err);
  }
}
