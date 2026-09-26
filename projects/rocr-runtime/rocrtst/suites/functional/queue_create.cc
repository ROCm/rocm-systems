/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <unistd.h>

#include <atomic>
#include <cstring>

#include "suites/functional/queue_create.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa.h"
#include "common/base_rocr_utils.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"

static constexpr uint32_t kQueueSizePackets = 1024;
static constexpr uint32_t kQueueSizeBytes =
    kQueueSizePackets * sizeof(hsa_kernel_dispatch_packet_t);
static constexpr uint32_t kSdmaQueueSizeBytes = 4096;

namespace {

// SDMA write-linear ("write untiled") packet. Declared here because this is the
// only packet rocrtst builds by hand for an SDMA queue.
struct SdmaWriteLinearPacket {
  uint32_t header;
  uint32_t dst_addr_lo;
  uint32_t dst_addr_hi;
  uint32_t count;
  uint32_t data0;
};

constexpr uint32_t kSdmaOpWrite = 2;
constexpr uint32_t kSdmaSubOpWriteLinear = 0;

// Bytes withheld from the ring when submitting the malformed packet, matching
// the packetSizeOffset KFDNegativeTest uses to provoke an SDMA queue reset.
constexpr uint32_t kSdmaBadPacketByteOffset = 6;

// Records what the queue's asynchronous error callback observed.
struct SdmaErrorCallbackState {
  std::atomic<uint32_t> invocations{0};
  hsa_status_t status{HSA_STATUS_SUCCESS};
  hsa_queue_t* source{nullptr};
};

void SdmaErrorCallback(hsa_status_t status, hsa_queue_t* source, void* data) {
  auto* state = reinterpret_cast<SdmaErrorCallbackState*>(data);
  state->status = status;
  state->source = source;
  state->invocations.fetch_add(1, std::memory_order_release);
}

}  // namespace

static bool VerifySquareResult(uint32_t* ar, size_t sz) {
  for (size_t i = 0; i < sz; ++i) {
    if (i * i != ar[i]) {
      return false;
    }
  }
  return true;
}

QueueCreateTest::QueueCreateTest() : TestBase() {
  set_title("RocR Queue Create API Test");
  set_description(
      "Tests hsa_amd_queue_create with various device-memory placement flags: "
      "system memory (default) and device-resident ring buffer.");
}

QueueCreateTest::~QueueCreateTest() {}

void QueueCreateTest::SetUp() {
  TestBase::SetUp();
  if (test_skipped_) return;
}

void QueueCreateTest::Run() {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void QueueCreateTest::Close() {
  TestBase::Close();
}

void QueueCreateTest::DisplayResults() const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::DisplayResults();
}

void QueueCreateTest::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

void QueueCreateTest::DispatchAndVerify(hsa_queue_t* queue,
                                        const char* test_label) {
  ASSERT_NE(queue, nullptr) << test_label << ": queue is null";

  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  void* src_buffer = nullptr;
  ASSERT_SUCCESS(
      hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0, &src_buffer));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, src_buffer));

  for (uint32_t i = 0; i < 256; ++i) {
    reinterpret_cast<uint32_t*>(src_buffer)[i] = i;
  }

  void* dst_buffer = nullptr;
  ASSERT_SUCCESS(
      hsa_amd_memory_pool_allocate(cpu_pool(), 256 * sizeof(uint32_t), 0, &dst_buffer));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, dst_buffer));

  hsa_signal_t completion_signal;
  ASSERT_SUCCESS(hsa_signal_create(1, 0, nullptr, &completion_signal));

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

  local_args.dstArray = reinterpret_cast<uint32_t*>(dst_buffer);
  local_args.srcArray = reinterpret_cast<uint32_t*>(src_buffer);
  local_args.size = 256;
  local_args.global_offset_x = 0;
  local_args.global_offset_y = 0;
  local_args.global_offset_z = 0;
  local_args.printf_buffer = 0;
  local_args.default_queue = 0;
  local_args.completion_action = 0;

  void* kernarg_address = nullptr;
  ASSERT_SUCCESS(
      hsa_amd_memory_pool_allocate(kern_arg_pool(), sizeof(local_args), 0, &kernarg_address));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, kernarg_address));
  memcpy(kernarg_address, &local_args, sizeof(local_args));

  const int kIterations = 10;
  const uint32_t queue_mask = queue->size - 1;

  for (int i = 0; i < kIterations; i++) {
    uint64_t index = hsa_queue_add_write_index_relaxed(queue, 1);

    hsa_kernel_dispatch_packet_t* pkt =
        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
            queue->base_address))[index & queue_mask];

    pkt->setup = 1;
    pkt->workgroup_size_x = 256;
    pkt->workgroup_size_y = 1;
    pkt->workgroup_size_z = 1;
    pkt->grid_size_x = 256;
    pkt->grid_size_y = 1;
    pkt->grid_size_z = 1;
    pkt->private_segment_size = 0;
    pkt->group_segment_size = 0;
    pkt->kernel_object = kernel_object();
    pkt->kernarg_address = kernarg_address;
    pkt->completion_signal = completion_signal;

    uint32_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    __atomic_store_n(reinterpret_cast<uint16_t*>(&pkt->header), header,
                     __ATOMIC_RELEASE);

    hsa_signal_store_screlease(queue->doorbell_signal, index);

    hsa_signal_value_t val = hsa_signal_wait_scacquire(
        completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
        5000000000ULL /* 5s timeout */, HSA_WAIT_STATE_ACTIVE);
    ASSERT_EQ(val, 0) << test_label << ": iteration " << i
                       << " timed out waiting for completion signal";
    hsa_signal_store_screlease(completion_signal, 1);

    ASSERT_TRUE(VerifySquareResult(reinterpret_cast<uint32_t*>(dst_buffer), 256))
        << test_label << ": iteration " << i;
  }

  ASSERT_SUCCESS(hsa_amd_memory_pool_free(kernarg_address));
  ASSERT_SUCCESS(hsa_signal_destroy(completion_signal));
  ASSERT_SUCCESS(hsa_amd_memory_pool_free(src_buffer));
  ASSERT_SUCCESS(hsa_amd_memory_pool_free(dst_buffer));
}

void QueueCreateTest::SystemMemQueueTest() {
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  set_kernel_file_name("test_case_template_kernels.hsaco");
  set_kernel_name("square");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, gpu_device1()));

  hsa_amd_queue_create_desc_t desc = {};
  desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
  desc.queue_size_bytes = kQueueSizeBytes;
  desc.engine.compute.type = HSA_QUEUE_TYPE_MULTI;
  desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  desc.flags = static_cast<hsa_amd_queue_create_flag_t>(HSA_AMD_QUEUE_CREATE_SYSTEM_MEM);
  desc.engine.compute.private_segment_size = HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT;
  desc.callback = nullptr;
  desc.callback_data = nullptr;

  ASSERT_SUCCESS(hsa_amd_queue_create(*gpu_device1(), &desc, 1));
  ASSERT_NE(desc.queue, nullptr);

  printf("  SystemMemQueueTest: queue=%p base=%p size=%u\n",
         desc.queue, desc.queue->base_address, desc.queue->size);

  DispatchAndVerify(desc.queue, "SystemMemQueue");

  ASSERT_SUCCESS(hsa_queue_destroy(desc.queue));
}

void QueueCreateTest::DeviceMemRingBufQueueTest() {
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  set_kernel_file_name("test_case_template_kernels.hsaco");
  set_kernel_name("square");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, gpu_device1()));

  hsa_amd_queue_create_desc_t desc = {};
  desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
  desc.queue_size_bytes = kQueueSizeBytes;
  desc.engine.compute.type = HSA_QUEUE_TYPE_MULTI;
  desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  desc.flags = static_cast<hsa_amd_queue_create_flag_t>(
      HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF);
  desc.engine.compute.private_segment_size = HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT;
  desc.callback = nullptr;
  desc.callback_data = nullptr;

  hsa_status_t status = hsa_amd_queue_create(*gpu_device1(), &desc, 1);
  if (status == HSA_STATUS_ERROR_INVALID_ARGUMENT ||
      status == HSA_STATUS_ERROR_INVALID_QUEUE_CREATION) {
    printf("  DeviceMemRingBufQueueTest: SKIPPED (agent does not support "
           "device-memory ring buffers or Large BAR is unavailable)\n");
    return;
  }
  ASSERT_SUCCESS(status);
  ASSERT_NE(desc.queue, nullptr);

  printf("  DeviceMemRingBufQueueTest: queue=%p base=%p size=%u\n",
         desc.queue, desc.queue->base_address, desc.queue->size);

  DispatchAndVerify(desc.queue, "DeviceMemRingBuf");

  ASSERT_SUCCESS(hsa_queue_destroy(desc.queue));
}

void QueueCreateTest::BatchQueueCreateTest() {
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  set_kernel_file_name("test_case_template_kernels.hsaco");
  set_kernel_name("square");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, gpu_device1()));

  constexpr int kNumQueues = 2;
  hsa_amd_queue_create_desc_t descs[kNumQueues] = {};

  // Queue 0: system memory
  descs[0].version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
  descs[0].queue_size_bytes = kQueueSizeBytes;
  descs[0].engine.compute.type = HSA_QUEUE_TYPE_MULTI;
  descs[0].priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  descs[0].flags = static_cast<hsa_amd_queue_create_flag_t>(HSA_AMD_QUEUE_CREATE_SYSTEM_MEM);
  descs[0].engine.compute.private_segment_size = HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT;

  // Queue 1: device-memory ring buffer
  descs[1].version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
  descs[1].queue_size_bytes = kQueueSizeBytes;
  descs[1].engine.compute.type = HSA_QUEUE_TYPE_MULTI;
  descs[1].priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  descs[1].flags = static_cast<hsa_amd_queue_create_flag_t>(
      HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF);
  descs[1].engine.compute.private_segment_size = HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT;

  hsa_status_t status = hsa_amd_queue_create(*gpu_device1(), descs, kNumQueues);
  if (status == HSA_STATUS_ERROR_INVALID_ARGUMENT ||
      status == HSA_STATUS_ERROR_INVALID_QUEUE_CREATION) {
    for (int i = 0; i < kNumQueues; i++) {
      if (descs[i].queue != nullptr) {
        hsa_queue_destroy(descs[i].queue);
      }
    }
    printf("  BatchQueueCreateTest: SKIPPED (agent does not support "
           "device-memory flags or Large BAR is unavailable)\n");
    return;
  }
  ASSERT_SUCCESS(status);

  const char* labels[] = {"Batch[0] SysMem", "Batch[1] DevMemRB"};
  for (int i = 0; i < kNumQueues; i++) {
    ASSERT_NE(descs[i].queue, nullptr) << labels[i] << ": queue is null";
    printf("  %s: queue=%p base=%p size=%u\n",
           labels[i], descs[i].queue,
           descs[i].queue->base_address, descs[i].queue->size);
    DispatchAndVerify(descs[i].queue, labels[i]);
  }

  for (int i = 0; i < kNumQueues; i++) {
    ASSERT_SUCCESS(hsa_queue_destroy(descs[i].queue));
  }
}

void QueueCreateTest::SdmaQueueCreateDestroyTest() {
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  // HSA_AMD_QUEUE_INFO_READ_POINTER / _WRITE_POINTER are HSA enum constants added
  // in hsa_ext_amd.h interface 1.31.
#if HSA_AMD_INTERFACE_VERSION_MINOR >= 31
  auto create_and_destroy = [&](uint16_t flags, const char* label) {
    hsa_amd_queue_create_desc_t desc = {};
    desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
    desc.engine_type = HSA_AMD_QUEUE_ENGINE_SDMA;
    desc.queue_size_bytes = kSdmaQueueSizeBytes;
    desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    desc.flags = static_cast<hsa_amd_queue_create_flag_t>(flags);
    desc.engine.sdma.sdma_engine_id = HSA_AMD_SDMA_ENGINE_ID_ANY;

    hsa_status_t status = hsa_amd_queue_create(*gpu_device1(), &desc, 1);
    if (status == HSA_STATUS_ERROR_INVALID_QUEUE_CREATION) {
      printf("  SdmaQueueCreateDestroyTest: SKIPPED %s (agent, driver, or Large BAR "
             "does not support requested SDMA queue placement)\n",
             label);
      return;
    }

    ASSERT_SUCCESS(status) << label;
    ASSERT_NE(desc.queue, nullptr) << label;
    EXPECT_EQ(desc.queue->type, HSA_QUEUE_TYPE_SINGLE) << label;
    EXPECT_EQ(desc.queue->features, 0) << label;
    EXPECT_EQ(desc.queue->size, kSdmaQueueSizeBytes) << label;
    EXPECT_NE(desc.queue->base_address, nullptr) << label;
    EXPECT_NE(desc.queue->doorbell_signal.handle, 0) << label;

    uint64_t read_pointer = 0;
    uint64_t write_pointer = 0;
    uint64_t doorbell_pointer = 0;
    ASSERT_SUCCESS(
        hsa_amd_queue_get_info(desc.queue, HSA_AMD_QUEUE_INFO_READ_POINTER, &read_pointer));
    ASSERT_SUCCESS(
        hsa_amd_queue_get_info(desc.queue, HSA_AMD_QUEUE_INFO_WRITE_POINTER, &write_pointer));
    ASSERT_SUCCESS(
        hsa_amd_queue_get_info(desc.queue, HSA_AMD_QUEUE_INFO_DOORBELL_ID, &doorbell_pointer));
    ASSERT_NE(read_pointer, 0u);
    ASSERT_NE(write_pointer, 0u);
    ASSERT_NE(doorbell_pointer, 0u);

    auto* raw_read_pointer = reinterpret_cast<volatile uint64_t*>(read_pointer);
    auto* raw_write_pointer = reinterpret_cast<volatile uint64_t*>(write_pointer);
    EXPECT_EQ(hsa_queue_load_read_index_relaxed(desc.queue), *raw_read_pointer);

    hsa_queue_store_write_index_relaxed(desc.queue, 64);
    EXPECT_EQ(hsa_queue_load_write_index_relaxed(desc.queue), 64u);
    EXPECT_EQ(*raw_write_pointer, 64u);
    EXPECT_EQ(hsa_queue_add_write_index_relaxed(desc.queue, 32), 64u);
    EXPECT_EQ(*raw_write_pointer, 96u);
    EXPECT_EQ(hsa_queue_cas_write_index_relaxed(desc.queue, 96, 128), 96u);
    EXPECT_EQ(*raw_write_pointer, 128u);
    hsa_queue_store_write_index_relaxed(desc.queue, 0);

    ASSERT_SUCCESS(hsa_queue_destroy(desc.queue)) << label;
  };

  create_and_destroy(HSA_AMD_QUEUE_CREATE_SYSTEM_MEM, "system memory");
  create_and_destroy(HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF, "device ring buffer");
  create_and_destroy(HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR, "device queue descriptor");
  create_and_destroy(HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF |
                         HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR,
                     "device ring buffer and descriptor");
#else
  // The gtest vendored under rocrtst/ predates GTEST_SKIP(); use the standard
  // [ SKIPPED ] marker instead.
  fprintf(stdout, "[ SKIPPED ] SdmaQueueCreateDestroyTest: HSA headers predate "
                  "queue read/write pointer queries\n");
#endif
}

void QueueCreateTest::SdmaQueueErrorCallbackTest() {
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  // Destination of the malformed write. The engine never retires the packet, so
  // the contents are never inspected.
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};
  void* dst_buffer = nullptr;
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(cpu_pool(), sizeof(uint32_t), 0, &dst_buffer));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, ag_list, NULL, dst_buffer));

  SdmaErrorCallbackState state;

  hsa_amd_queue_create_desc_t desc = {};
  desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
  desc.engine_type = HSA_AMD_QUEUE_ENGINE_SDMA;
  desc.queue_size_bytes = kSdmaQueueSizeBytes;
  desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
  desc.flags = static_cast<hsa_amd_queue_create_flag_t>(HSA_AMD_QUEUE_CREATE_SYSTEM_MEM);
  desc.engine.sdma.sdma_engine_id = HSA_AMD_SDMA_ENGINE_ID_ANY;
  desc.callback = SdmaErrorCallback;
  desc.callback_data = &state;

  hsa_status_t status = hsa_amd_queue_create(*gpu_device1(), &desc, 1);
  if (status == HSA_STATUS_ERROR_INVALID_QUEUE_CREATION) {
    printf("  SdmaQueueErrorCallbackTest: SKIPPED (agent or driver does not support "
           "SDMA queue creation)\n");
    ASSERT_SUCCESS(hsa_amd_memory_pool_free(dst_buffer));
    return;
  }
  ASSERT_SUCCESS(status);
  ASSERT_NE(desc.queue, nullptr);

  hsa_queue_t* queue = desc.queue;
  const uint64_t dst_address = reinterpret_cast<uint64_t>(dst_buffer);

  SdmaWriteLinearPacket pkt = {};
  pkt.header = kSdmaOpWrite | (kSdmaSubOpWriteLinear << 8);
  pkt.dst_addr_lo = static_cast<uint32_t>(dst_address);
  pkt.dst_addr_hi = static_cast<uint32_t>(dst_address >> 32);
  // The count field holds dwords minus one, so 0 requests a single dword.
  pkt.count = 0;
  pkt.data0 = 0;

  memcpy(queue->base_address, &pkt, sizeof(pkt));

  // Publish fewer bytes than the packet the engine will parse. The engine is
  // left reading a truncated packet and hangs, which is what makes the driver
  // reset the queue once it is destroyed. SDMA write and read pointers are byte
  // offsets, so the doorbell value is a byte count.
  const uint32_t published_bytes = sizeof(pkt) - kSdmaBadPacketByteOffset;
  hsa_signal_store_screlease(queue->doorbell_signal, published_bytes);

  // Give the engine time to fetch the malformed packet and hang on it.
  usleep(50 * 1000);

  ASSERT_SUCCESS(hsa_queue_destroy(queue));

  // The queue reset is triggered by the destroy above, so the notification is
  // expected to arrive after the queue is already gone.
  const int kCallbackTimeoutMs = 10000;
  const int kPollIntervalMs = 10;
  for (int waited_ms = 0;
       state.invocations.load(std::memory_order_acquire) == 0 && waited_ms < kCallbackTimeoutMs;
       waited_ms += kPollIntervalMs) {
    usleep(kPollIntervalMs * 1000);
  }

  ASSERT_EQ(state.invocations.load(std::memory_order_acquire), 1u)
      << "No error callback within " << kCallbackTimeoutMs
      << " ms of destroying the queue. This requires a driver that supports "
         "per-SDMA-queue reset.";
  EXPECT_EQ(state.status, HSA_STATUS_ERROR_EXCEPTION);
  EXPECT_EQ(state.source, queue);

  printf("  SdmaQueueErrorCallbackTest: error callback reported status %d for queue %p\n",
         state.status, state.source);

  ASSERT_SUCCESS(hsa_amd_memory_pool_free(dst_buffer));
}

void QueueCreateTest::InvalidArgsTest() {
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  auto make_valid_desc = []() {
    hsa_amd_queue_create_desc_t desc = {};
    desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
    desc.queue_size_bytes = kQueueSizeBytes;
    desc.engine.compute.type = HSA_QUEUE_TYPE_MULTI;
    desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    desc.flags = static_cast<hsa_amd_queue_create_flag_t>(HSA_AMD_QUEUE_CREATE_SYSTEM_MEM);
    desc.engine.compute.private_segment_size = HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT;
    return desc;
  };

  hsa_amd_queue_create_desc_t desc = make_valid_desc();
  auto make_valid_sdma_desc = []() {
    hsa_amd_queue_create_desc_t desc = {};
    desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
    desc.engine_type = HSA_AMD_QUEUE_ENGINE_SDMA;
    desc.queue_size_bytes = kSdmaQueueSizeBytes;
    desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    desc.flags = static_cast<hsa_amd_queue_create_flag_t>(HSA_AMD_QUEUE_CREATE_SYSTEM_MEM);
    desc.engine.sdma.sdma_engine_id = HSA_AMD_SDMA_ENGINE_ID_ANY;
    return desc;
  };

  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), nullptr, 1),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 0),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);

  uint32_t cu_mask = 0x1;

  desc = make_valid_desc();
  desc.engine.compute.cu_mask_count = 1;
  desc.engine.compute.cu_mask = nullptr;
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_desc();
  desc.engine.compute.cu_mask_count = 0;
  desc.engine.compute.cu_mask = &cu_mask;
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_desc();
  desc.engine.compute.cu_mask_count = 1;
  desc.engine.compute.cu_mask = &cu_mask;
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_desc();
  desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION + 1;
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_desc();
  desc.queue_size_bytes = 0;
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_desc();
  desc.queue_size_bytes = 3;
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_desc();
  desc.queue_size_bytes = sizeof(hsa_kernel_dispatch_packet_t) / 2;
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_desc();
  desc.engine.compute.type = static_cast<hsa_queue_type32_t>(HSA_QUEUE_TYPE_COOPERATIVE + 1);
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_desc();
  desc.priority = static_cast<hsa_amd_queue_priority_t>(UINT32_MAX);
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_desc();
  desc.reserved[0] = 1;
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_sdma_desc();
  desc.engine.sdma.reserved[0] = 1;
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), &desc, 1), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(desc.queue, nullptr);

  desc = make_valid_sdma_desc();
  EXPECT_EQ(hsa_amd_queue_create(*cpu_device(), &desc, 1), HSA_STATUS_ERROR_INVALID_AGENT);
  EXPECT_EQ(desc.queue, nullptr);

  hsa_amd_queue_create_desc_t descs[2] = {make_valid_desc(), make_valid_desc()};
  descs[1].queue_size_bytes = 3;
  EXPECT_EQ(hsa_amd_queue_create(*gpu_device1(), descs, 2),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_NE(descs[0].queue, nullptr);
  EXPECT_EQ(descs[1].queue, nullptr);
  ASSERT_SUCCESS(hsa_queue_destroy(descs[0].queue));
}
