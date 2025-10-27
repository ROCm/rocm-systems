/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
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


#include <sys/mman.h>
#include <fcntl.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <sys/socket.h>

#include "suites/functional/svm_memory.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
//#include "common/concurrent_utils.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

// Wrap printf to add first or second process indicator
#define PROCESS_LOG(format, ...)                                                                   \
  {                                                                                                \
    if (verbosity() >= VERBOSE_STANDARD || !parentProcess_) {                                      \
      fprintf(stdout, "line:%d P%u: " format, __LINE__, static_cast<int>(!parentProcess_),         \
              ##__VA_ARGS__);                                                                      \
    }                                                                                              \
  }

// Fork safe ASSERT_EQ.
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

static const char kSubTestSeparator[] = "  **************************";

static void PrintMemorySubtestHeader(const char* header) {
  std::cout << "  *** Virtual Memory Functional Subtest: " << header << " ***" << std::endl;
}

SvmMemoryTestBasic::SvmMemoryTestBasic(void) : TestBase() {
  set_title("ROCr SVM Memory Basic Tests");
  set_description(" Tests SVM memory API functions");
}

SvmMemoryTestBasic::~SvmMemoryTestBasic(void) {}

// Test to check that GPU can read and write to SVM memory.
void SvmMemoryTestBasic::TestCreateDestroy(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  hsa_agent_t* agents_accessible;
  hsa_amd_pointer_info_t ptrInfo = {};
  uint32_t num_agents_accessible = 0;
  std::vector<hsa_agent_t> gpus;
  rocrtst::pool_info_t pool_i;
  hsa_device_type_t ag_type;
  char ag_name[64];
  void* addressRange;
  hsa_status_t err;
  hsa_agent_t cpu_agent;

  typedef struct __attribute__((aligned(16))) args_t {
    int* a;
    int* b;
    int* c;
  } args;
  args* kernArgs = NULL;

  static const int kMemoryAllocSize = 1024;

  ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type));
  ASSERT_SUCCESS(hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NEAREST_CPU, &cpu_agent));

  ASSERT_SUCCESS(rocrtst::AcquirePoolInfo(pool, &pool_i));

  if (ag_type != HSA_DEVICE_TYPE_GPU || !pool_i.alloc_allowed) return;

  hsa_queue_t* queue = NULL;  // command queue
  hsa_signal_t signal = {0};  // completion signal

  /* Create a queue to enqueue kernel */
  // get queue size
  uint32_t queue_size = 0;
  ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size));

  // create queue
  ASSERT_SUCCESS(
      hsa_queue_create(agent, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue));

  // Find a memory pool that supports kernel arguments.
  hsa_amd_memory_pool_t kernarg_pool;
  ASSERT_SUCCESS(
      hsa_amd_agent_iterate_memory_pools(cpu_agent, rocrtst::GetKernArgMemoryPool, &kernarg_pool));

  struct host_data_t {
    int data[kMemoryAllocSize * 4];
    int dup_data[kMemoryAllocSize * 4];
    int result[kMemoryAllocSize * 4];
  };

  struct dev_data_t {
    int result[kMemoryAllocSize * 4];
  };

  struct host_data_t* host_data = NULL;
  struct dev_data_t* dev_data = NULL;

  /* Set up host_data */
  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve((void**)&host_data, sizeof(host_data_t), 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER));
  ASSERT_NE(host_data, nullptr);

  /* Verify that pointer info for unmapped VA's return expected values */
  ptrInfo.size = sizeof(ptrInfo);
  ASSERT_SUCCESS(hsa_amd_pointer_info(host_data, &ptrInfo, nullptr, nullptr, nullptr));
  ASSERT_EQ(ptrInfo.type, HSA_EXT_POINTER_TYPE_RESERVED_ADDR);
  ASSERT_EQ(ptrInfo.hostBaseAddress, host_data);
  /* For unmapped VA, then size is equal to size of address reservation */
  ASSERT_EQ(ptrInfo.sizeInBytes, sizeof(host_data_t));
  ASSERT_EQ(num_agents_accessible, 0);

  ptrInfo.size = sizeof(ptrInfo);
  ASSERT_SUCCESS(hsa_amd_pointer_info(&host_data->result, &ptrInfo, nullptr, nullptr, nullptr));
  ASSERT_EQ(ptrInfo.type, HSA_EXT_POINTER_TYPE_RESERVED_ADDR);
  ASSERT_EQ(ptrInfo.hostBaseAddress, host_data);
  /* For unmapped VA, then size is equal to size of address reservation */
  ASSERT_EQ(ptrInfo.sizeInBytes, sizeof(host_data_t));
  ASSERT_EQ(num_agents_accessible, 0);
   if (verbosity() > 0) {
    std::cout << "    Pointer info on reserved address OK" << std::endl;
  }

  std::vector<hsa_amd_svm_attribute_pair_t> host_attrs;
  host_attrs.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, cpu_agent.handle});
  host_attrs.push_back({HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, agent.handle});
  ASSERT_SUCCESS(hsa_amd_svm_attributes_set(host_data, sizeof(host_data_t), host_attrs.data(), host_attrs.size()));

  /* Set up dev_data */
  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve((void**)&dev_data, sizeof(dev_data_t), 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER));
  ASSERT_NE(dev_data, nullptr);

  std::vector<hsa_amd_svm_attribute_pair_t> dev_attrs;
  dev_attrs.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, agent.handle});
  dev_attrs.push_back({HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, agent.handle});

  ASSERT_SUCCESS(hsa_amd_svm_attributes_set(dev_data, sizeof(dev_data_t), dev_attrs.data(), dev_attrs.size()));

  // initialize the host buffers
  for (int i = 0; i < kMemoryAllocSize; ++i) {
    unsigned int seed = time(NULL);
    host_data->data[i] = 1 + rand_r(&seed) % 1;
    host_data->dup_data[i] = host_data->data[i];
  }

  memset(host_data->result, 0, sizeof(host_data->result));
  memset(dev_data->result, 0, sizeof(dev_data->result));

  // Allocate the kernel argument buffer from the kernarg_pool.
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(args_t), 0,
                                              reinterpret_cast<void**>(&kernArgs)));

  ASSERT_SUCCESS(hsa_amd_agents_allow_access(1, &agent, NULL, kernArgs));
  kernArgs->a = host_data->data;
  kernArgs->b = host_data->result;  // system memory passed to gpu for write
  kernArgs->c = dev_data->result;   // gpu memory to verify that gpu read system data

  // Create the executable, get symbol by name and load the code object
  set_kernel_file_name("gpuReadWrite_kernels.hsaco");
  set_kernel_name("gpuReadWrite");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, &agent));

  ASSERT_SUCCESS(hsa_signal_create(1, 0, NULL, &signal));

  // create aql packet
  hsa_kernel_dispatch_packet_t aql;
  memset(&aql, 0, sizeof(aql));

  // initialize aql packet
  aql.workgroup_size_x = 256;
  aql.workgroup_size_y = 1;
  aql.workgroup_size_z = 1;
  aql.grid_size_x = kMemoryAllocSize;
  aql.grid_size_y = 1;
  aql.grid_size_z = 1;
  aql.private_segment_size = 0;
  aql.group_segment_size = 0;
  aql.kernel_object = kernel_object();  // kernel_code;
  aql.kernarg_address = kernArgs;
  aql.completion_signal = signal;

  const uint32_t queue_mask = queue->size - 1;

  // write to command queue
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);
  hsa_queue_store_write_index_relaxed(queue, index + 1);

  rocrtst::WriteAQLToQueueLoc(queue, index, &aql);

  hsa_kernel_dispatch_packet_t* q_base_addr =
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address);
  rocrtst::AtomicSetPacketHeader(
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
          (1 << HSA_PACKET_HEADER_BARRIER) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE),
      (1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS),
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(&q_base_addr[index & queue_mask]));

  // ringdoor bell
  hsa_signal_store_relaxed(queue->doorbell_signal, index);
  // wait for the signal and reset it for future use
  while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                   HSA_WAIT_STATE_ACTIVE)) {
  }

  // compare device and host side results
  if (verbosity() > 0) {
    std::cout << "    Check GPU has read the host memory" << std::endl;
  }
  for (int i = 0; i < kMemoryAllocSize; ++i) {
    // printf("Verifying data at index[%d]\n", i);
    ASSERT_EQ(dev_data->result[i], host_data->dup_data[i]);
  }

  if (verbosity() > 0) {
    std::cout << "    GPU has read the host memory successfully" << std::endl;
    std::cout << "    Check GPU has written to host memory" << std::endl;
  }
  for (int i = 0; i < kMemoryAllocSize; ++i) {
    ASSERT_EQ(host_data->result[i], i);
  }

  if (verbosity() > 0) {
    std::cout << "    GPU has written to host memory successfully" << std::endl;
  }

  if (kernArgs) {
    hsa_memory_free(kernArgs);
  }

  if (signal.handle) {
    hsa_signal_destroy(signal);
  }
  if (queue) {
    hsa_queue_destroy(queue);
  }
}

void SvmMemoryTestBasic::TestCreateDestroy(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  if (verbosity() > 0) {
    PrintMemorySubtestHeader("CreateDestroy Test");
  }

  ASSERT_SUCCESS(rocrtst::GetAgentPools(&agent_pools));

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      TestCreateDestroy(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "    Subtest finished" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void SvmMemoryTestBasic::TestSVMPrefetch(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  if (verbosity() > 0) {
    PrintMemorySubtestHeader("SVMPrefetch Test");
  }

  ASSERT_SUCCESS(rocrtst::GetAgentPools(&agent_pools));

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      TestSVMPrefetch(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "    Subtest finished" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void SvmMemoryTestBasic::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  return;
}

void SvmMemoryTestBasic::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void SvmMemoryTestBasic::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void SvmMemoryTestBasic::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void SvmMemoryTestBasic::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


// Test to check that GPU can prefetch SVM memory from specific agent.
void SvmMemoryTestBasic::TestSVMPrefetch(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  hsa_amd_pointer_info_t ptrInfo = {};
  rocrtst::pool_info_t pool_i;
  hsa_device_type_t ag_type;
  hsa_agent_t cpu_agent;

  static const int kMemoryAllocSize = 1024;

  ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type));
  if (ag_type != HSA_DEVICE_TYPE_GPU) return;

  ASSERT_SUCCESS(hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NEAREST_CPU, &cpu_agent));

  ASSERT_SUCCESS(rocrtst::AcquirePoolInfo(pool, &pool_i));

  if(!pool_i.alloc_allowed) return;

  hsa_queue_t* queue = NULL;  // command queue
  hsa_signal_t signal = {0};  // completion signal

  /* Create a queue to enqueue kernel */
  // get queue size
  uint32_t queue_size = 0;
  ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size));

  // create queue
  ASSERT_SUCCESS(
      hsa_queue_create(agent, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue));

  struct host_data_t {
    int data[kMemoryAllocSize * 4];
    int dup_data[kMemoryAllocSize * 4];
    int result[kMemoryAllocSize * 4];
  };

  struct dev_data_t {
    int result[kMemoryAllocSize * 4];
  };

  struct host_data_t* host_data = NULL;
  struct dev_data_t* dev_data = NULL;

  /* Set up host_data */
  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve((void**)&host_data, sizeof(host_data_t), 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER));
  ASSERT_NE(host_data, nullptr);

  /* Verify that pointer info for unmapped VA's return expected values */
  ptrInfo.size = sizeof(ptrInfo);
  ASSERT_SUCCESS(hsa_amd_pointer_info(host_data, &ptrInfo, nullptr, nullptr, nullptr));
  ASSERT_EQ(ptrInfo.type, HSA_EXT_POINTER_TYPE_RESERVED_ADDR);
  ASSERT_EQ(ptrInfo.hostBaseAddress, host_data);
  /* For unmapped VA, then size is equal to size of address reservation */
  ASSERT_EQ(ptrInfo.sizeInBytes, sizeof(host_data_t));

  if (verbosity() > 0) {
    std::cout << "    Pointer info on reserved address OK" << std::endl;
  }

  std::vector<hsa_amd_svm_attribute_pair_t> host_attrs;
  host_attrs.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, cpu_agent.handle});
  host_attrs.push_back({HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, agent.handle});
  ASSERT_SUCCESS(hsa_amd_svm_attributes_set(host_data, sizeof(host_data_t), host_attrs.data(), host_attrs.size()));

  /* Set up dev_data */
  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve((void**)&dev_data, sizeof(dev_data_t), 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER));
  ASSERT_NE(dev_data, nullptr);

  std::vector<hsa_amd_svm_attribute_pair_t> dev_attrs;
  dev_attrs.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, agent.handle});
  dev_attrs.push_back({HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, agent.handle});

  ASSERT_SUCCESS(hsa_amd_svm_attributes_set(dev_data, sizeof(dev_data_t), dev_attrs.data(), dev_attrs.size()));
  ASSERT_SUCCESS(hsa_signal_create(1, 0, NULL, &signal));

  ASSERT_SUCCESS(hsa_amd_svm_prefetch_async(dev_data, sizeof(dev_data_t), agent, 0, nullptr, signal));

  // wait for the signal and reset it for future use
  while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                   HSA_WAIT_STATE_ACTIVE)) {
  }

  hsa_amd_svm_attributes_get(dev_data, sizeof(dev_data_t), dev_attrs.data() , dev_attrs.size());

  // Check if mem location is sourced from the expected agent
  ASSERT_EQ(dev_attrs[0].value, agent.handle);

  //verify the agent owner
  if (verbosity() > 0) {
    std::cout << "    GPU has prefetched the preferred agent memory successfully" << std::endl;
  }
  if (signal.handle) {
    hsa_signal_destroy(signal);
  }
  if (queue) {
    hsa_queue_destroy(queue);
  }

}
