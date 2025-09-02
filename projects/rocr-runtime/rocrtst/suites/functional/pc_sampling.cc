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

#include <iostream>
#include <vector>
#include <sstream>
#include <unistd.h>  // for usleep and sleep
#include <cstdint>

#include "common/base_rocr.h"

#include <algorithm>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "suites/functional/pc_sampling.h"

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

static const char kSubTestSeparator[] = "  **************************";

static void PrintAgentPropsSubtestHeader(const char *header) {
  std::cout << "  *** " << header << " ***" << std::endl;
}

const size_t MIN_INTERVAL = 100;
const size_t MAX_INTERVAL = 10000;

std::vector<size_t> buffer_sizes = {
                          64 * sizeof(perf_sample_hosttrap_v1_t),
                          256 * sizeof(perf_sample_hosttrap_v1_t),
                          1024 * sizeof(perf_sample_hosttrap_v1_t)
                        };

PcSamplingTest::PcSamplingTest(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("  *** PC Sampling test ***");
  set_description("  *** Test PC Sampling on GPU agent on a system ***");
}

PcSamplingTest::~PcSamplingTest(void) {
}

typedef struct {
  size_t buffer_size;
  uint8_t* buffer;
  size_t bytes_received;
  size_t bytes_max;
  int *finish; //increment finish to when bytes_max > bytes_received to stop the kernel
} pc_sampling_copy_data_t;

hsa_status_t pc_sampling_info_cb(const hsa_ven_amd_pcs_configuration_t* configuration,
                                       void* _test_list) {
  std::vector<pc_sampling_test_t>* test_list = (std::vector<pc_sampling_test_t>*)_test_list;

  std::cout << "  Supported PC sampling configuration:" << std::endl;
  std::cout << "    method:" << configuration->method << std::endl;
  std::cout << "    units:" << configuration->units << std::endl;
  std::cout << "    min_interval:" << configuration->min_interval << std::endl;
  std::cout << "    max_interval:" << configuration->max_interval << std::endl;
  std::cout << std::endl;

  for (auto size: buffer_sizes) {
    pc_sampling_test_t test = {};
    test.method = configuration->method;
    test.units = configuration->units;
    test.interval = std::max(configuration->min_interval, MIN_INTERVAL);
    test.interval = std::min(test.interval, MAX_INTERVAL);
    test.buffer_size = size;

    test_list->push_back(test);
  }
  return HSA_STATUS_SUCCESS;
}

void pc_sampling_data_ready_callback(void* client_callback_data, size_t data_size,
                                     size_t lost_sample_count,
                                     hsa_ven_amd_pcs_data_copy_callback_t data_copy_callback,
                                     void* hsa_callback_data) {

  pc_sampling_copy_data_t* results = (pc_sampling_copy_data_t*)client_callback_data;

  //@Elena please delete or comment out after bug that I mentioned is fixed otherwise this will log too much.
  // This is to show that we do not receive any PC Sampling data during the second test
  std::cout << "DEBUG:  bytes received:" << results->bytes_received << " max:" << results->bytes_max << std::endl;


  results->bytes_received += data_size;
  if (data_copy_callback(hsa_callback_data, /* size_to_copy */ data_size,
                         /* buffer */ results->buffer) != HSA_STATUS_SUCCESS) {
    std::cout << "Failed to data_copy_callback" << std::endl;
    return;
  }
  return;
}

/* Launches a busySpin kernel */
void PcSamplingTest::PrepareKernArgs(hsa_agent_t agent, void** kernargs) {
  typedef struct __attribute__((aligned(16))) args_t {
    int* count;
    int* output;
  } args;

  args* kernArgs = NULL;
  hsa_agent_t cpu_agent;

  ASSERT_SUCCESS(hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NEAREST_CPU, &cpu_agent));

  hsa_amd_memory_pool_t kernarg_pool;
  ASSERT_SUCCESS(
      hsa_amd_agent_iterate_memory_pools(cpu_agent, rocrtst::GetKernArgMemoryPool, &kernarg_pool));

  // Allocate the kernel argument buffer from the kernarg_pool.
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(args_t), 0,
                                              reinterpret_cast<void**>(&kernArgs)));

  ASSERT_SUCCESS(hsa_amd_agents_allow_access(1, &agent, NULL, kernArgs));

  kernArgs->count = &test_data->count;
  kernArgs->output =test_data->output;

  // Create the executable, get symbol by name and load the code object
  set_kernel_file_name("busySpin_kernels.hsaco");
  set_kernel_name("busySpin");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, &agent));

  *kernargs = kernArgs;
  return;
}

void PcSamplingTest::ExecKernel(hsa_agent_t agent, void* kernargs, hsa_queue_t* queue, hsa_signal_t signal) {
  // Fill the dispatch packet with
  // workgroup_size, grid_size, kernelArgs and completion signal
  // Put it on the queue and launch the kernel by ringing the doorbell

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
  aql.kernel_object = kernel_object();
  aql.kernarg_address = kernargs;
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

  //std::cout << "Starting kernel" << std::endl;

  // ringdoor bell
  hsa_signal_store_relaxed(queue->doorbell_signal, index);
  // wait for the signal and reset it for future use
  while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                   HSA_WAIT_STATE_ACTIVE)) {
  }
  //std::cout << "Kernel finished" << std::endl;
  hsa_signal_store_relaxed(signal, 1);


}

void PcSamplingTest::RunPcSamplingTest(hsa_agent_t agent, pc_sampling_test_t test) {
  hsa_status_t err;
  pc_sampling_copy_data_t results = {};
  results.buffer_size = test.buffer_size;
  results.bytes_received = 0;
  results.bytes_max = (test.buffer_size * 10000) / test.interval;

  std::cout << "  Testing Method: " << (test.method == HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1 ? "Hosttrap" :
                              test.method == HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1 ? "Stochastic" : "Unknown") << std::endl;
  std::cout << "    Interval: " << test.interval << std::endl;
  std::cout << "    Buffer size: " << test.buffer_size << std::endl;
  std::cout << "    Bytes Max: " << results.bytes_max << std::endl;

  results.buffer = static_cast<uint8_t*>(malloc(results.buffer_size));
  if (!results.buffer) {
    std::cout << "Failed to allocate local buffer" << std::endl;
    return;
  }
  std::cout << "    Calling hsa_ven_amd_pcs_create" << std::endl;

  // pc_sampling_data_ready_callback will increment finish when
  // results.bytes_received >= results.bytes_max

  err = hsa_ven_amd_pcs_create(agent, test.method, test.units, test.interval,
                                test.latency, test.buffer_size,
                                &pc_sampling_data_ready_callback, &results, &test.handle);

  std::cout << "    Calling hsa_ven_amd_pcs_start" << std::endl;
  ASSERT_SUCCESS(hsa_ven_amd_pcs_start(test.handle));

  //Launch the kernel the kernel will busy wait until finish is set to non-zero;
  hsa_queue_t* queue = {};
  hsa_signal_t signal = {};  // completion signal
  void *kernargs;
  ASSERT_SUCCESS(hsa_signal_create(1, 0, NULL, &signal));

  ASSERT_SUCCESS(rocrtst::CreateQueue(agent, &queue));

  hsa_amd_memory_pool_t global_pool;
  ASSERT_SUCCESS(hsa_amd_agent_iterate_memory_pools(agent, rocrtst::GetGlobalMemoryPool, &global_pool));
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(global_pool, sizeof(test_data_t), 0, reinterpret_cast<void**>(&test_data)));
  test_data->count = test.interval * 100000;

  PrepareKernArgs(agent, &kernargs);

  while(results.bytes_received < results.bytes_max) {
    ExecKernel(agent, kernargs, queue, signal);
  }

  std::cout << "    Calling hsa_ven_amd_pcs_stop" << std::endl;
  ASSERT_SUCCESS(hsa_ven_amd_pcs_stop(test.handle));

  sleep(3);
  std::cout << "    Calling hsa_ven_amd_pcs_destroy" << std::endl;
  ASSERT_SUCCESS(hsa_ven_amd_pcs_destroy(test.handle));

  ASSERT_SUCCESS(hsa_memory_free(kernargs));

  ASSERT_SUCCESS(hsa_memory_free(test_data));
  free(results.buffer);

  ASSERT_SUCCESS(hsa_queue_destroy(queue));

  std::cout << kSubTestSeparator << std::endl;
  return;
}

void PcSamplingTest::PcSampling(hsa_agent_t agent) {
  std::vector<pc_sampling_test_t> test_list;

  // Build the test list based on supported sessions
  ASSERT_SUCCESS(hsa_ven_amd_pcs_iterate_configuration(agent, &pc_sampling_info_cb, &test_list));

  for (auto test:test_list) {
    RunPcSamplingTest(agent, test);
  }

}

void PcSamplingTest::PcSampling() {
  hsa_status_t err;
  if (verbosity() > 0) {
    PrintAgentPropsSubtestHeader("Test PC Sampling");
  }

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  if (!gpus.size()) {
    if (verbosity() > 0) {
      std::cout << "  *** Execution skipped - no GPUs found "
                << " ***" << std::endl;
    }
    return;
  }

  for (auto gpu_agent: gpus)
    PcSampling(gpu_agent);
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void PcSamplingTest::SetUp(void) {
  TestBase::SetUp();
}

void PcSamplingTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void PcSamplingTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void PcSamplingTest::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void PcSamplingTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

#undef RET_IF_HSA_ERR
