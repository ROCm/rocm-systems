/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <iostream>
#include <vector>

#include "suites/functional/metadata_prefetch.h"

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

static const char kSubTestSeparator[] = "  **************************";

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

MetadataPrefetch::MetadataPrefetch(void) :
    TestBase(){
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("Metadata prefetch example");
  set_description("Example to fill in the metadata prefetch packet");

  set_kernel_file_name("test_case_template_kernels_argpreload.hsaco");

  set_kernel_name("square");  // kernel function name
}

MetadataPrefetch::~MetadataPrefetch(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void MetadataPrefetch::SetUp(void) {
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
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));

  hsa_agent_t* gpu_dev = gpu_device1();

  // Find and assign HSA_AMD_SEGMENT_GLOBAL pools for cpu, gpu and a kern_arg
  // pool
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  // Create a queue
  hsa_queue_t* q = nullptr;
  rocrtst::CreateQueue(*gpu_dev, &q);
  ASSERT_NE(q, nullptr);
  set_main_queue(q);

  err = rocrtst::EnableMetadataPrefetch(this, q);
  if (err == HSA_STATUS_ERROR_INVALID_QUEUE) {
    markAsSkip();
    if (verbosity() > 0) {
      std::cout<< "Test not applicable as GPU does not support metadata prefetch."
                  "Skipping."<< std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

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

  ASSERT_SUCCESS(rocrtst::InitializeMetadataPrefetchPacket(this, &local_args, &aql(), &metadata_prefetch()));

  return;
}

// This wrapper atomically writes the provided header and setup to the
// provided AQL packet. The provided AQL packet address should be in the
// queue memory space.
static inline void AtomicSetAqlPacketHeader(uint16_t header, uint16_t setup,
                                  hsa_kernel_dispatch_packet_t* queue_packet) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(queue_packet),
                   header | (setup << 16), __ATOMIC_RELEASE);
}

// This wrapper writes the provided header and setup to the provided
// Metadata-Prefetch packet. The provided Metadata-Prefetch packet address
// should be in the queue memory space. This does not have to be atomic, but
// this needs to be set before the AQL packet header.
static inline void SetMetadataPacketHeader(uint32_t header,
  hsa_amd_metadata_kernel_dispatch_packet_t* queue_packet) {
    queue_packet->header0 = header;
    queue_packet->header1 = header;
    queue_packet->header2 = header;
    queue_packet->header3 = header;
}

// Do a few extra iterations as we toss out some of the inital and final
// iterations when calculating statistics
uint32_t MetadataPrefetch::RealIterationNum(void) {
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
void MetadataPrefetch::Run(void) {
  if (Skip()) return;
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
  hsa_amd_metadata_kernel_dispatch_packet_t *queue_metadata_packet;

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

    uint32_t metadata_header = HSA_PACKET_TYPE_KERNEL_DISPATCH;

    metadata_header |= metadata_prefetch_dispatch_version_major() <<
                        HSA_AMD_METADATA_PACKET_HEADER_VERSION_MAJOR;
    metadata_header |= metadata_prefetch_dispatch_version_minor() <<
                        HSA_AMD_METADATA_PACKET_HEADER_VERSION_MINOR;

    queue_metadata_packet = WriteMetadataToQueue(this, index);
    ASSERT_EQ(queue_metadata_packet,
              reinterpret_cast<hsa_amd_metadata_kernel_dispatch_packet_t *>
              (metadata_prefetch_queue_base()) + index);

    /* Set the MetadataPacket header before the AQL header - no need to be atomic */
    ::SetMetadataPacketHeader(metadata_header, queue_metadata_packet);

    ::AtomicSetAqlPacketHeader(aql_header, aql().setup, queue_aql_packet);

    // Create and start a timer for this iteration
    int id = p_timer.CreateTimer();
    p_timer.StartTimer(id);

    hsa_signal_store_screlease(main_queue()->doorbell_signal, index);

    // Wait on the dispatch signal until the kernel is finished.
    while (hsa_signal_wait_scacquire(aql().completion_signal,
         HSA_SIGNAL_CONDITION_LT, 1, (uint64_t) - 1, HSA_WAIT_STATE_ACTIVE)) {
    }

    // Stop the timer
    p_timer.StopTimer(id);

    // Verify the metadata packet headers have been invalidated
    ASSERT_EQ(queue_metadata_packet->header0, HSA_PACKET_TYPE_INVALID);
    ASSERT_EQ(queue_metadata_packet->header1, HSA_PACKET_TYPE_INVALID);
    ASSERT_EQ(queue_metadata_packet->header2, HSA_PACKET_TYPE_INVALID);
    ASSERT_EQ(queue_metadata_packet->header3, HSA_PACKET_TYPE_INVALID);

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

void MetadataPrefetch::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MetadataPrefetch::DisplayResults(void) const {
  if (Skip()) return;
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

void MetadataPrefetch::Close() {
  if (!Skip()) {
    hsa_status_t err;

    err = hsa_amd_memory_pool_free(src_buffer_);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    err = hsa_amd_memory_pool_free(dst_buffer_);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }

  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


#undef RET_IF_HSA_ERR
