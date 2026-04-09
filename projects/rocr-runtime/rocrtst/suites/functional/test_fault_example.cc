/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// This test is based on TestExample but intentionally passes nullptr for
// the kernel array arguments to trigger a GPU fault. This is useful for
// testing GPU core dump functionality and fault handling.
//
// Key differences from TestExample:
// * No memory allocation for src_buffer or dst_buffer
// * Kernel arguments set with nullptr for array pointers
// * No result verification (we expect a fault)
// * Test is DISABLED by default to prevent running in CI

#include <algorithm>
#include <iostream>
#include <vector>

#include "suites/functional/test_fault_example.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const uint32_t kNumBufferElements = rocrtst::isEmuModeEnabled() ? 4 : 256;

TestFaultExample::TestFaultExample(void) :
    TestBase() {
  set_num_iteration(1);  // Only need one iteration to trigger the fault
  set_title("Test Fault Example");
  set_description("This test intentionally passes nullptr for kernel array "
      "arguments to trigger a GPU fault. This is useful for testing GPU "
      "core dump functionality and fault handling mechanisms. "
      "NOTE: This test is DISABLED by default and should be run manually.");

  set_kernel_file_name("test_case_template_kernels.hsaco");
  set_kernel_name("square");  // kernel function name
}

TestFaultExample::~TestFaultExample(void) {
}

// Setup the test environment - similar to TestExample but without buffer allocation
void TestFaultExample::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_agent_t* gpu_dev = gpu_device1();

  // Find and assign HSA_AMD_SEGMENT_GLOBAL pools for cpu, gpu and a kern_arg pool
  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Create a queue
  hsa_queue_t* q = nullptr;
  rocrtst::CreateQueue(*gpu_dev, &q);
  ASSERT_NE(q, nullptr);
  set_main_queue(q);

  err = rocrtst::LoadKernelFromObjFile(this, gpu_dev);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // NOTE: We do NOT allocate src_buffer or dst_buffer
  // We will pass nullptr to the kernel to trigger a fault

  // Set up Kernel arguments with nullptr for array pointers
  struct __attribute__((aligned(16))) local_args_t {
    uint32_t* dstArray;
    uint32_t* srcArray;
    uint32_t size;
    uint32_t pad;
    uint64_t global_offset_x;
    uint64_t global_offset_y;
    uint64_t global_offset_z;
    uint64_t printf_buffer;
    uint64_t default_queue;
    uint64_t completion_action;
  } local_args;

  // Intentionally set array pointers to nullptr to cause a fault
  local_args.dstArray = nullptr;
  local_args.srcArray = nullptr;
  local_args.size = kNumBufferElements;
  local_args.global_offset_x = 0;
  local_args.global_offset_y = 0;
  local_args.global_offset_z = 0;
  local_args.printf_buffer = 0;
  local_args.default_queue = 0;
  local_args.completion_action = 0;

  err = rocrtst::AllocAndSetKernArgs(this, &local_args, sizeof(local_args));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  return;
}

// This wrapper atomically writes the provided header and setup to the
// provided AQL packet. The provided AQL packet address should be in the
// queue memory space.
static inline void AtomicSetPacketHeader(uint16_t header, uint16_t setup,
                                  hsa_kernel_dispatch_packet_t* queue_packet) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(queue_packet),
                   header | (setup << 16), __ATOMIC_RELEASE);
}

void TestFaultExample::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();

  // Override whatever we need to...
  aql().workgroup_size_x = kNumBufferElements;
  aql().grid_size_x = kNumBufferElements;

  hsa_kernel_dispatch_packet_t *queue_aql_packet;
  uint64_t index;

  if (verbosity() >= VERBOSE_STANDARD) {
    std::cout << "Dispatching kernel with nullptr arrays - expecting GPU fault..." << std::endl;
  }

  // This function simply copies the data we've collected so far into our
  // local AQL packet, except the the setup and header fields.
  queue_aql_packet = WriteAQLToQueue(this, &index);
  ASSERT_EQ(queue_aql_packet,
            reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                                    (main_queue()->base_address) + index);
  uint32_t aql_header = HSA_PACKET_TYPE_KERNEL_DISPATCH;

  aql_header |= HSA_FENCE_SCOPE_SYSTEM <<
                HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  aql_header |= HSA_FENCE_SCOPE_SYSTEM <<
                HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

  ::AtomicSetPacketHeader(aql_header, aql().setup, queue_aql_packet);

  hsa_signal_store_screlease(main_queue()->doorbell_signal, index);

  // Wait on the dispatch signal until the kernel is finished (or faults).
  // Note: This may trigger a GPU fault/exception
  while (hsa_signal_wait_scacquire(aql().completion_signal,
       HSA_SIGNAL_CONDITION_LT, 1, (uint64_t) - 1, HSA_WAIT_STATE_ACTIVE)) {
  }

  if (verbosity() >= VERBOSE_STANDARD) {
    std::cout << "Kernel dispatch completed (fault may have occurred)" << std::endl;
  }

  hsa_signal_store_screlease(aql().completion_signal, 1);

  // NOTE: We do NOT verify results since we expect a fault
}

void TestFaultExample::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestFaultExample::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::DisplayResults();
  std::cout << "Test completed. Check for GPU core dump if fault handling is enabled." << std::endl;
  return;
}

void TestFaultExample::Close() {
  // NOTE: We do NOT free src_buffer or dst_buffer since we never allocated them

  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

#undef RET_IF_HSA_ERR
