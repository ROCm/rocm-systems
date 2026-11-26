/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/scratch_memory.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

constexpr uint32_t kWorkgroupSize = 256;
constexpr uint32_t kQueuePackets = 1024;
constexpr size_t kSmallGrid = 1024;

// A bounded wait so that a queue that never recovers from a scratch request
// fails the test instead of hanging it.
constexpr uint64_t kWaitSeconds = 60;

bool WaitForZero(hsa_signal_t signal) {
  uint64_t freq = 0;
  if (hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq) != HSA_STATUS_SUCCESS)
    return false;
  return hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, freq * kWaitSeconds,
                                   HSA_WAIT_STATE_BLOCKED) == 0;
}

// The value stored in the input at @p index. Kept small so the square fits an int.
int InputValue(size_t index) { return static_cast<int>(index % 1000); }

// Destroys a queue when the test scope ends, including on an early ASSERT return.
class QueueGuard {
 public:
  explicit QueueGuard(hsa_queue_t* queue = nullptr) : queue_(queue) {}
  ~QueueGuard() {
    if (queue_ != nullptr) hsa_queue_destroy(queue_);
  }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
  hsa_queue_t** out() { return &queue_; }
  hsa_queue_t* get() const { return queue_; }

 private:
  hsa_queue_t* queue_;
};

// Puts the async scratch threshold back when the test scope ends.
class ScratchLimitGuard {
 public:
  ScratchLimitGuard(hsa_agent_t agent, uint64_t value) : agent_(agent), value_(value) {}
  ~ScratchLimitGuard() { hsa_amd_agent_set_async_scratch_limit(agent_, value_); }

 private:
  hsa_agent_t agent_;
  uint64_t value_;
};

}  // namespace

ScratchMemoryTest::ScratchMemoryTest() : TestBase() {
  set_title("RocR Scratch Memory Test");
  set_description(
      "Dispatches with a range of private segment sizes and grid sizes so that "
      "ROCr has to allocate, grow, switch and reclaim queue scratch memory.");
}

ScratchMemoryTest::~ScratchMemoryTest() { FreeInput(); }

void ScratchMemoryTest::SetUp() {
  TestBase::SetUp();
  if (test_skipped_) return;

  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  set_kernel_file_name("test_case_template_kernels.hsaco");
  set_kernel_name("square");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, gpu_device1()));
}

void ScratchMemoryTest::Run() {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void ScratchMemoryTest::Close() {
  FreeInput();
  TestBase::Close();
}

void ScratchMemoryTest::DisplayResults() const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::DisplayResults();
}

void ScratchMemoryTest::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

uint32_t ScratchMemoryTest::ComputeUnits() {
  uint32_t cus = 0;
  EXPECT_EQ(hsa_agent_get_info(*gpu_device1(),
                               static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
                               &cus),
            HSA_STATUS_SUCCESS);
  return std::max<uint32_t>(cus, 1);
}

size_t ScratchMemoryTest::FullDeviceElements() {
  // More than 32 wave64 slots per CU, so every scratch slot of the device is used.
  return static_cast<size_t>(ComputeUnits()) * 4096;
}

hsa_status_t ScratchMemoryTest::CreateQueue(uint32_t private_segment_size, hsa_queue_t** queue) {
  uint32_t max_size = 0;
  hsa_status_t err = hsa_agent_get_info(*gpu_device1(), HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_size);
  if (err != HSA_STATUS_SUCCESS) return err;
  return hsa_queue_create(*gpu_device1(), std::min(max_size, kQueuePackets), HSA_QUEUE_TYPE_MULTI,
                          nullptr, nullptr, private_segment_size, UINT32_MAX, queue);
}

void ScratchMemoryTest::AllocateInput(size_t max_elements) {
  FreeInput();
  const hsa_agent_t agents[2] = {*gpu_device1(), *cpu_device()};
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(cpu_pool(), max_elements * sizeof(int), 0, &src_));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(2, agents, nullptr, src_));
  int* values = static_cast<int*>(src_);
  for (size_t i = 0; i < max_elements; ++i) values[i] = InputValue(i);
  src_elements_ = max_elements;
}

void ScratchMemoryTest::FreeInput() {
  if (src_ != nullptr) {
    hsa_amd_memory_pool_free(src_);
    src_ = nullptr;
  }
  src_elements_ = 0;
}

bool ScratchMemoryTest::Submit(Job* job, hsa_queue_t* queue, size_t elements,
                               uint32_t private_segment_size, const char* label) {
  if (elements > src_elements_) {
    ADD_FAILURE() << label << ": input holds " << src_elements_ << " elements, need " << elements;
    return false;
  }
  const hsa_agent_t agents[2] = {*gpu_device1(), *cpu_device()};

  job->elements = elements;
  if (hsa_amd_memory_pool_allocate(cpu_pool(), elements * sizeof(int), 0, &job->dst) !=
          HSA_STATUS_SUCCESS ||
      hsa_amd_agents_allow_access(2, agents, nullptr, job->dst) != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << label << ": cannot allocate the output buffer";
    return false;
  }
  memset(job->dst, 0xFF, elements * sizeof(int));

  struct __attribute__((aligned(16))) Args {
    int* dst;
    const int* src;
    int size;
    int pad;
    uint64_t global_offset_x;
    uint64_t global_offset_y;
    uint64_t global_offset_z;
    uint64_t printf_buffer;
    uint64_t default_queue;
    uint64_t completion_action;
  } args = {};
  args.dst = static_cast<int*>(job->dst);
  args.src = static_cast<const int*>(src_);
  args.size = static_cast<int>(elements);

  if (hsa_amd_memory_pool_allocate(kern_arg_pool(), sizeof(args), 0, &job->kernarg) !=
          HSA_STATUS_SUCCESS ||
      hsa_amd_agents_allow_access(2, agents, nullptr, job->kernarg) != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << label << ": cannot allocate the kernel arguments";
    return false;
  }
  memcpy(job->kernarg, &args, sizeof(args));

  if (hsa_signal_create(1, 0, nullptr, &job->done) != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << label << ": cannot create the completion signal";
    return false;
  }

  const uint64_t grid = ((elements + kWorkgroupSize - 1) / kWorkgroupSize) * kWorkgroupSize;
  const uint64_t index = hsa_queue_add_write_index_relaxed(queue, 1);
  auto* packets = static_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address);
  hsa_kernel_dispatch_packet_t* pkt = &packets[index & (queue->size - 1)];

  memset(reinterpret_cast<uint8_t*>(pkt) + sizeof(pkt->header), 0,
         sizeof(*pkt) - sizeof(pkt->header));
  pkt->setup = 1;
  pkt->workgroup_size_x = kWorkgroupSize;
  pkt->workgroup_size_y = 1;
  pkt->workgroup_size_z = 1;
  pkt->grid_size_x = static_cast<uint32_t>(grid);
  pkt->grid_size_y = 1;
  pkt->grid_size_z = 1;
  pkt->private_segment_size = private_segment_size;
  pkt->group_segment_size = 0;
  pkt->kernel_object = kernel_object();
  pkt->kernarg_address = job->kernarg;
  pkt->completion_signal = job->done;

  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(reinterpret_cast<uint16_t*>(&pkt->header), header, __ATOMIC_RELEASE);
  hsa_signal_store_screlease(queue->doorbell_signal, index);
  return true;
}

bool ScratchMemoryTest::Finish(Job* job, const char* label, bool dispatched) {
  bool ok = true;
  if (job->done.handle != 0) {
    if (dispatched && !WaitForZero(job->done)) {
      ADD_FAILURE() << label << ": dispatch did not complete within " << kWaitSeconds << " s";
      // The queue is still using the buffers. Leak them rather than free them under the GPU.
      return false;
    }
    hsa_signal_destroy(job->done);
    job->done.handle = 0;
  }

  if (job->dst != nullptr) {
    const int* out = static_cast<const int*>(job->dst);
    for (size_t i = 0; dispatched && i < job->elements; ++i) {
      const int expected = InputValue(i) * InputValue(i);
      if (out[i] != expected) {
        ADD_FAILURE() << label << ": element " << i << " is " << out[i] << ", expected " << expected;
        ok = false;
        break;
      }
    }
    hsa_amd_memory_pool_free(job->dst);
    job->dst = nullptr;
  }
  if (job->kernarg != nullptr) {
    hsa_amd_memory_pool_free(job->kernarg);
    job->kernarg = nullptr;
  }
  return ok;
}

bool ScratchMemoryTest::DispatchSquare(hsa_queue_t* queue, size_t elements,
                                       uint32_t private_segment_size, const char* label) {
  Job job;
  if (!Submit(&job, queue, elements, private_segment_size, label)) {
    Finish(&job, label, false);
    return false;
  }
  return Finish(&job, label);
}

void ScratchMemoryTest::DynamicScratchGrowthTest() {
  AllocateInput(kSmallGrid);
  ASSERT_NE(src_, nullptr);

  // Created with a private segment size of zero, so the queue starts with no scratch.
  QueueGuard queue;
  ASSERT_SUCCESS(CreateQueue(0, queue.out()));
  ASSERT_NE(queue.get(), nullptr);

  // Bytes of scratch per work item. The sequence grows past what the queue holds,
  // then asks for less than it holds, then for none at all.
  const uint32_t sizes[] = {0, 64, 256, 4096, 16384, 256, 0, 8192};
  for (uint32_t size : sizes) {
    char label[64];
    snprintf(label, sizeof(label), "private segment %u bytes", size);
    ASSERT_TRUE(DispatchSquare(queue.get(), kSmallGrid, size, label));
  }
}

void ScratchMemoryTest::ScratchFullDeviceAndSmallGridTest() {
  const size_t full = FullDeviceElements();
  AllocateInput(full);
  ASSERT_NE(src_, nullptr);

  QueueGuard queue;
  ASSERT_SUCCESS(CreateQueue(0, queue.out()));
  ASSERT_NE(queue.get(), nullptr);

  // A small grid needs few scratch slots, a full-device grid needs all of them.
  struct Step {
    size_t elements;
    uint32_t private_segment_size;
    const char* label;
  };
  const Step steps[] = {
      {kSmallGrid, 256, "small grid, 256 bytes"},
      {full, 256, "full-device grid, 256 bytes"},
      {kSmallGrid, 256, "small grid again, 256 bytes"},
      {kSmallGrid, 1024, "small grid, 1024 bytes"},
      {full, 128, "full-device grid, 128 bytes"},
      {full, 256, "full-device grid again, 256 bytes"},
  };
  for (const Step& step : steps) {
    ASSERT_TRUE(DispatchSquare(queue.get(), step.elements, step.private_segment_size, step.label));
  }
}

void ScratchMemoryTest::QueueCreateScratchHintTest() {
  AllocateInput(kSmallGrid);
  ASSERT_NE(src_, nullptr);

  // No scratch, the "unknown size" value, and a hint of 256 bytes per work item.
  const uint32_t hints[] = {0, UINT32_MAX, 256};
  for (uint32_t hint : hints) {
    QueueGuard queue;
    ASSERT_SUCCESS(CreateQueue(hint, queue.out())) << "hint " << hint;
    ASSERT_NE(queue.get(), nullptr);

    // One dispatch within the hint, and one that is beyond it.
    char label[64];
    snprintf(label, sizeof(label), "hint %u, 64 bytes", hint);
    ASSERT_TRUE(DispatchSquare(queue.get(), kSmallGrid, 64, label));
    snprintf(label, sizeof(label), "hint %u, 1024 bytes", hint);
    ASSERT_TRUE(DispatchSquare(queue.get(), kSmallGrid, 1024, label));
  }
}

void ScratchMemoryTest::MultiQueueScratchTest() {
  AllocateInput(kSmallGrid);
  ASSERT_NE(src_, nullptr);

  constexpr int kQueues = 4;
  QueueGuard queues[kQueues];
  Job jobs[kQueues];
  for (int i = 0; i < kQueues; ++i) {
    ASSERT_SUCCESS(CreateQueue(0, queues[i].out()));
    ASSERT_NE(queues[i].get(), nullptr);
  }

  // Every queue asks for a different amount of scratch before any has finished.
  bool submitted[kQueues] = {};
  bool all_submitted = true;
  for (int i = 0; i < kQueues; ++i) {
    submitted[i] = Submit(&jobs[i], queues[i].get(), kSmallGrid, 256u << i, "multi queue");
    all_submitted = all_submitted && submitted[i];
  }

  bool all_ok = true;
  for (int i = 0; i < kQueues; ++i) {
    char label[64];
    snprintf(label, sizeof(label), "multi queue %d", i);
    all_ok = Finish(&jobs[i], label, submitted[i]) && all_ok;
  }
  ASSERT_TRUE(all_submitted);
  ASSERT_TRUE(all_ok);

  // Queues that are destroyed return their scratch. The survivors keep working.
  ASSERT_SUCCESS(hsa_queue_destroy(queues[0].get()));
  *queues[0].out() = nullptr;
  ASSERT_TRUE(DispatchSquare(queues[1].get(), kSmallGrid, 512, "after destroying a queue"));
}

void ScratchMemoryTest::AsyncScratchLimitTest() {
  const hsa_agent_t gpu = *gpu_device1();

  // Only GPU agents have scratch.
  EXPECT_EQ(hsa_amd_agent_set_async_scratch_limit(*cpu_device(), 1 << 20),
            HSA_STATUS_ERROR_INVALID_AGENT);

  uint64_t max_limit = 0;
  uint64_t current = 0;
  ASSERT_SUCCESS(hsa_agent_get_info(
      gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX), &max_limit));
  ASSERT_SUCCESS(hsa_agent_get_info(
      gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT), &current));
  ASSERT_GT(max_limit, 0u);

  // Async reclaim is a firmware capability that can also be disabled by the environment.
  hsa_status_t err = hsa_amd_agent_set_async_scratch_limit(gpu, current);
  if (err == HSA_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("  AsyncScratchLimitTest: SKIPPED (async scratch reclaim is not enabled)\n");
    return;
  }
  ASSERT_SUCCESS(err);
  ScratchLimitGuard restore(gpu, current);

  // The limit cannot exceed what the device can hold.
  EXPECT_EQ(hsa_amd_agent_set_async_scratch_limit(gpu, max_limit + 1),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);

  const size_t full = FullDeviceElements();
  AllocateInput(full);
  ASSERT_NE(src_, nullptr);

  QueueGuard queue;
  ASSERT_SUCCESS(CreateQueue(0, queue.out()));
  ASSERT_NE(queue.get(), nullptr);

  // Take scratch, then drop the limit below what the queue holds.
  ASSERT_TRUE(DispatchSquare(queue.get(), full, 256, "scratch before lowering the limit"));

  const uint64_t low_limit = std::min<uint64_t>(1 << 20, max_limit);
  ASSERT_SUCCESS(hsa_amd_agent_set_async_scratch_limit(gpu, low_limit));
  uint64_t reported = 0;
  ASSERT_SUCCESS(hsa_agent_get_info(
      gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT), &reported));
  EXPECT_EQ(reported, low_limit);

  // With a low limit a large dispatch is served by use-once scratch.
  ASSERT_TRUE(DispatchSquare(queue.get(), full, 256, "full-device grid under a low limit"));
  ASSERT_TRUE(DispatchSquare(queue.get(), kSmallGrid, 64, "small grid under a low limit"));

  // Raise it again. The queue keeps working.
  ASSERT_SUCCESS(hsa_amd_agent_set_async_scratch_limit(gpu, current));
  ASSERT_SUCCESS(hsa_agent_get_info(
      gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT), &reported));
  EXPECT_EQ(reported, current);
  ASSERT_TRUE(DispatchSquare(queue.get(), full, 256, "scratch after restoring the limit"));
}
