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

// The purpose of this test is to provide an example of the use of the
// common RocrTest classes and utilities that are used in many examples.
// It can be used as a template to start off with when writing new tests.
// In many cases, the existing boilerplate code will be sufficient as is.
// Otherwise, the boilerplate code can be either supplemented or replaced
// by your own code in your example, as necessary.
//
// The comments provided are focused more on the use of the common rocrtst
// utilities and boilerplate code, rather than the example app. itself.
//
// The boilerplate code includes code for:
// * hsa initialization and clean up
// * code to load pre-built kernels
// * creating queues
// * populating AQL packets
// * checking for required profiles
// * finding cpu and gpu agents (callbacks for common use cases)
// * finding pools (having common requirements)
// * allocating and setting kernel arguments
// * somewhat standardized output
// * handling additional command line arguments, beyond google-test arguments
// * support for various level of verbosity, controlled from command line arg
// * support for building OpenCL kernels
// * timer support
//
// Overview of RocrTst code organization:
// Classes:
// * class BaseRocR (base_rocr.h) -- base class for all rocrtst examples and
//   tests. Most of the rocrtst common utilities act on BaseRocR objects
//
// * TestBase (test_base.h)  -- derives from BaseRocR and is the base class
//   for all tests under <rocrtst root>/suites. The implementation in TestBase
//   methods are typically actions that are required for most/all tests and
//   should therefore be called from the derived implementions of the methods.
//
// Utilities:
// * <rocrtst root>/common/base_rocr_utils.<cc/h> contains a set of utilities
//   that act on BaseRocR objects.
//
// * <rocrtst root>/common/common.<cc/h> contain other non-BaseRocR utilities
//
// Special Files:
// * main.cc -- The main google test file from which the tests are invoked.
//     There should be an entry for each test to be run there.
//
// * kernels -- OpenCL kernel source files should go in the kernels directory
//
// * CMakeLists.txt -- Host code (*.cc and *.h files) should build without
//     modifying the CMakeList.txt file, if the files are place in the
//     "performance" directory. However, an entry for OpenCL kernels. For
//     each kernel to be built, the bitcode libraries must be indicated before
//     the call to "build_kernel()" is made. See existing code for examples.

#include <algorithm>
#include <iostream>
#include <vector>

#include "suites/test_common/test_case_template.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

#ifdef ROCRTST_EMULATOR_BUILD
static const uint32_t kNumBufferElements = 4;
#else
static const uint32_t kNumBufferElements = 256;
#endif

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

// Many test cases want to perform an operation on memory sizes of various
// granularities.
#if 0
static const int kNumGranularity = 20;
const char* Str[kNumGranularity] = {"1k", "2K", "4K", "8K", "16K", "32K",
    "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M",
                                               "64M", "128M", "256M", "512M"};

const size_t Size[kNumGranularity] = {
    1024, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024,
    256*1024, 512*1024, 1024*1024, 2048*1024, 4096*1024, 8*1024*1024,
    16*1024*1024, 32*1024*1024, 64*1024*1024, 128*1024*1024, 256*1024*1024,
    512*1024*1024};

static const int kMaxCopySize = Size[kNumGranularity - 1];
#endif
TestExample::TestExample(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("Test Case Example");
  set_description("Put a description of the test case here. Line breaks "
      "will be taken care of on output, not here.");

  set_kernel_file_name("test_case_template_kernels.hsaco");
  set_kernel_name("square");  // kernel function name

#if 0
  // Set required profile to HSA_PROFILE_FULL or HSA_PROFILE_BASE if it
  // matters for this test. If either profile is fine, then leave with
  // default
  set_requires_profile(<value>);
#endif
}

TestExample::~TestExample(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void TestExample::SetUp(void) {
  hsa_status_t err;

  // TestBase::SetUp() will set HSA_ENABLE_INTERRUPT if enable_interrupt() is
  // true, and call hsa_init(). It also prints the SetUp header.
  TestBase::SetUp();

  // SetDefaultAgents(this) will assign the first CPU and GPU found on
  // iterating through the agents and assign them to cpu_device_ and
  // gpu_device1_, respectively (cpu_device() and gpu_device1()). These
  // BaseRocR member variables are used in some utilities. Additionally,
  // SetDefaultAgents() checks the profile of the gpu and compares this
  // to any required profile.
  //
  // If SetDefaultAgents() is not used, if the profile of the target GPU
  // matters for this test, it should be set with set_profile() and
  // CheckProfileAndInform() should be called to check if it is the
  // required profile
  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_agent_t* gpu_dev = gpu_device1();

  // Find and assign HSA_AMD_SEGMENT_GLOBAL pools for cpu, gpu and a kern_arg
  // pool
  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Create a queue
  hsa_queue_t* q = nullptr;
  rocrtst::CreateQueue(*gpu_dev, &q);
  ASSERT_NE(q, nullptr);
  set_main_queue(q);

  err = rocrtst::LoadKernelFromObjFile(this, gpu_dev);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Fill up the kernel packet (except header) with some values we've
  // collected so far, and some reasonable default values; this should be after
  // LoadKernelFromObjFile(). AllocAndSetKernArgs() will fill in the kern_args
  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  // Allocate a few buffers for our example
  err = hsa_amd_memory_pool_allocate(cpu_pool(),
                                   kNumBufferElements*sizeof(uint32_t),
                                   0, reinterpret_cast<void**>(&src_buffer_));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_agents_allow_access(2, ag_list, NULL, src_buffer_);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Initialize the source buffer
  for (uint32_t i = 0; i < kNumBufferElements; ++i) {
    reinterpret_cast<uint32_t *>(src_buffer_)[i] = i;
  }

  err = hsa_amd_memory_pool_allocate(cpu_pool(),
                                   kNumBufferElements*sizeof(uint32_t),
                                   0, reinterpret_cast<void**>(&dst_buffer_));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_agents_allow_access(2, ag_list, NULL, dst_buffer_);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Set up Kernel arguments
  // See the meta-data for the compiled OpenCL kernel code to ascertain
  // the sizes, padding and alignment required for kernel arguments.
  // This can be seen by executing
  // $ amdgcn-amd-amdhsa-readelf -aw ./binary_search_kernels.hsaco
  // The kernel code will expect the following arguments aligned as shown.
//  typedef uint32_t uint4[4];
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

  local_args.dstArray = reinterpret_cast<uint32_t *>(dst_buffer_);
  local_args.srcArray = reinterpret_cast<uint32_t *>(src_buffer_);
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

// Do a few extra iterations as we toss out some of the inital and final
// iterations when calculating statistics
uint32_t TestExample::RealIterationNum(void) {
  return num_iteration() * 1.2 + 1;
}

static bool VerifyResult(uint32_t *ar, size_t sz) {
  for (size_t i = 0; i < sz; ++i) {
    if (i*i != ar[i]) {
      return false;
    }
  }
  return true;
}
void TestExample::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();

  // Override whatever we need to...
  aql().workgroup_size_x = kNumBufferElements;
  aql().grid_size_x = kNumBufferElements;

  std::vector<double> timer;

  int it = RealIterationNum();
  hsa_kernel_dispatch_packet_t *queue_aql_packet;

  rocrtst::PerfTimer p_timer;
  uint64_t index;

  for (int i = 0; i < it; i++) {
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

    // Create and start a timer for this iteration
    int id = p_timer.CreateTimer();
    p_timer.StartTimer(id);

    ::AtomicSetPacketHeader(aql_header, aql().setup, queue_aql_packet);

    hsa_signal_store_screlease(main_queue()->doorbell_signal, index);

    // Wait on the dispatch signal until the kernel is finished.
    while (hsa_signal_wait_scacquire(aql().completion_signal,
         HSA_SIGNAL_CONDITION_LT, 1, (uint64_t) - 1, HSA_WAIT_STATE_ACTIVE)) {
    }

    // Stop the timer
    p_timer.StopTimer(id);

    // Store time for later analysis
    timer.push_back(p_timer.ReadTimer(id));
    hsa_signal_store_screlease(aql().completion_signal, 1);

    ASSERT_TRUE(VerifyResult(reinterpret_cast<uint32_t *>(dst_buffer_),
                                                         kNumBufferElements));

    // Pay attention to verbosity level for things like progress output
    if (verbosity() >= VERBOSE_PROGRESS) {
      std::cout << ".";
      fflush(stdout);
    }
  }

  if (verbosity() >= VERBOSE_PROGRESS) {
    std::cout << std::endl;
  }

  // Abandon the first result and after sort, delete the last 2% value
  timer.erase(timer.begin());
  std::sort(timer.begin(), timer.end());
  timer.erase(timer.begin() + num_iteration(), timer.end());

  time_mean_ = rocrtst::CalcMean(timer);
}

void TestExample::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestExample::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::DisplayResults();
  std::cout << "The average time was: " << time_mean_ * 1e6 <<
                                                           " uS" << std::endl;
  return;
}

void TestExample::Close() {
  hsa_status_t err;

  err = hsa_amd_memory_pool_free(src_buffer_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_pool_free(dst_buffer_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


#undef RET_IF_HSA_ERR
