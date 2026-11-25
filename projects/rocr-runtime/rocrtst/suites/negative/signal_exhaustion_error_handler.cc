/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/negative/signal_exhaustion_error_handler.h"

#include <unistd.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

namespace {
// KFD gives each process at most 4096 events and hsa_signal_create() consumes one per interrupt
// signal, so this is more than enough to run out of them. No public HSA API reports whether a
// signal got a KFD event, and the runtime does not return OUT_OF_RESOURCES when it runs out, so
// the test cannot observe exhaustion directly. It relies on the KFD limit staying at 4096: if the
// limit is raised above kTargetSignalCount, every signal gets an event and the test takes the
// ordinary path while still passing. Keep kTargetSignalCount above the KFD limit when changing it.
constexpr uint32_t kMinSignalCount = 4096;
constexpr uint32_t kTargetSignalCount = 8192;
constexpr uint32_t kQueueSize = 1024;
// A kernel object address that is never mapped (it is below vm.mmap_min_addr), so the packet
// processor takes a page fault when it fetches the kernel descriptor.
constexpr uint64_t kUnmappedKernelObject = 0x1000;
// How long to wait for the exception to be reported before giving up.
constexpr int kCallbackTimeoutMs = 30000;
}  // namespace

SignalExhaustionErrorHandlerTest::SignalExhaustionErrorHandlerTest()
    : TestBase(), rlimit_(), system_handler_registered_(false) {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("RocR System Error Handler With Exhausted Signal Events");
  set_description(
      "This test allocates more signals than KFD has events, subscribes to the system event "
      "handler, enqueues an invalid AQL dispatch packet and verifies that the resulting GPU "
      "exception is still reported to the system event handler. It guards against the async "
      "events thread no longer waiting on KFD events once some signals have no event.");

  memset(&aql(), 0, sizeof(hsa_kernel_dispatch_packet_t));
  set_kernel_file_name("dispatch_time_kernels.hsaco");
  set_kernel_name("empty_kernel");
}

SignalExhaustionErrorHandlerTest::~SignalExhaustionErrorHandlerTest() {}

void SignalExhaustionErrorHandlerTest::SetUp(void) {
  hsa_status_t err;

  // Interrupt signals need HSA_ENABLE_INTERRUPT, which TestBase::SetUp sets from this. Without
  // it the runtime does not register the system event handler and the signals take no KFD event.
  set_enable_interrupt(true);
  TestBase::SetUp();
  if (test_skipped_) return;

  // The GPU exception can trigger a coredump. Set the limit to 0 to disable it. Only the soft
  // limit is lowered so that Close() is allowed to restore it.
  if (getrlimit(RLIMIT_CORE, &rlimit_)) {
    perror("Could not get system rlimit\n");
  } else {
    struct rlimit rlimit_set = rlimit_;
    rlimit_set.rlim_cur = 0;

    // Do not error if system does not allow disabling limit
    if (setrlimit(RLIMIT_CORE, &rlimit_set)) perror("Could not set core file size\n");
  }

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
}

void SignalExhaustionErrorHandlerTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void SignalExhaustionErrorHandlerTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void SignalExhaustionErrorHandlerTest::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
}

void SignalExhaustionErrorHandlerTest::Close() {
  // Restore rlimit to initial value before test - do not error if fails
  if (setrlimit(RLIMIT_CORE, &rlimit_)) perror("Could not set core file size\n");

  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

hsa_status_t SignalExhaustionErrorHandlerTest::SystemEventCallback(const hsa_amd_event_t* event,
                                                                   void* data) {
  CallbackData* cb_data = reinterpret_cast<CallbackData*>(data);

  // Ignore the other events, e.g. the shutdown event. Returning an error for them leaves the
  // runtime to its default action, which aborts the process. This is intentional: an unexpected
  // GPU event fails the whole run rather than being masked.
  if (event->event_type != HSA_AMD_GPU_MEMORY_FAULT_EVENT) return HSA_STATUS_ERROR;

  cb_data->fault_address = event->memory_fault.virtual_address;
  cb_data->fault_agent = event->memory_fault.agent.handle;
  cb_data->fault_reported = true;

  // The fault is handled here: the runtime must not abort the process.
  return HSA_STATUS_SUCCESS;
}

void SignalExhaustionErrorHandlerTest::RunOnAgent(hsa_agent_t gpu_agent) {
  // Create the executable, get symbol by name and load the code object. This is done before the
  // KFD events run out because freezing the executable needs a signal with an event.
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, &gpu_agent));

  uint32_t queue_max = 0;
  ASSERT_SUCCESS(hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max));
  queue_max = (queue_max < kQueueSize) ? queue_max : kQueueSize;

  // The signals and the queue are released on every exit path so that a failed assertion does
  // not leave them behind. The guard is armed before the first allocation.
  std::vector<hsa_signal_t> signals;
  hsa_queue_t* queue = nullptr;
  MAKE_SCOPE_GUARD([&]() {
    if (queue != nullptr) hsa_queue_destroy(queue);
    for (auto& signal : signals) hsa_signal_destroy(signal);
    clear_code_object();
  });

  // 1) Use up the KFD events. The runtime falls back to signals without an event when it runs
  // out, so creation keeps succeeding.
  signals.reserve(kTargetSignalCount);
  for (uint32_t i = 0; i < kTargetSignalCount; ++i) {
    hsa_signal_t signal;
    hsa_status_t status = hsa_signal_create(1, 0, nullptr, &signal);
    if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES) break;
    ASSERT_SUCCESS(status);
    ASSERT_NE(signal.handle, 0u);
    signals.push_back(signal);
  }
  if (verbosity() > 0) {
    std::cout << "Allocated " << signals.size() << " signals" << std::endl;
  }

  // A system that cannot create this many signals cannot run out of KFD events, so there is
  // nothing to test. Skip instead of failing.
  if (signals.size() < kMinSignalCount) {
    rocrtst::SkipCurrentTest("Not enough signals could be created to run out of KFD events");
    return;
  }

  // 2) Subscribe to the system event handler. It cannot be unregistered, so do it once.
  if (!system_handler_registered_) {
    ASSERT_SUCCESS(hsa_amd_register_system_event_handler(SystemEventCallback, &cb_data_));
    system_handler_registered_ = true;
  }
  cb_data_.fault_reported = false;

  ASSERT_SUCCESS(hsa_queue_create(gpu_agent, queue_max, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                  0, &queue));

  // 3) Generate an exception with an invalid AQL dispatch packet: its kernel object does not point
  // to a mapped kernel descriptor, so the packet processor takes a page fault.
  hsa_kernel_dispatch_packet_t packet = {};
  packet.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet.workgroup_size_x = 1;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 1;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.kernel_object = kUnmappedKernelObject;

  const uint32_t queue_mask = queue->size - 1;
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);
  hsa_queue_store_write_index_relaxed(queue, index + 1);

  rocrtst::WriteAQLToQueueLoc(queue, index, &packet);

  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

  rocrtst::AtomicSetPacketHeader(header, packet.setup,
                                 &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
                                     queue->base_address))[index & queue_mask]);

  // ring doorbell
  hsa_signal_store_relaxed(queue->doorbell_signal, index);

  // 4) Confirm that the system event handler is called
  for (int waited_ms = 0; waited_ms < kCallbackTimeoutMs && !cb_data_.fault_reported;
       waited_ms += 10) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_TRUE(cb_data_.fault_reported)
      << "System event handler was not called within " << kCallbackTimeoutMs
      << " ms after the exception was generated";
  ASSERT_EQ(cb_data_.fault_address.load(), kUnmappedKernelObject);
  ASSERT_EQ(cb_data_.fault_agent.load(), gpu_agent.handle);
}

/*
 * Test Name: TestSystemErrorHandlerAfterSignalExhaustion
 * Scope: Negative
 *
 * Purpose: Verifies that the runtime keeps listening for KFD events (and therefore reports GPU
 * exceptions to the system event handler) after the process has run out of KFD events.
 *
 * Test Description:
 * 1) Allocate more than 4095 signals so that KFD runs out of events
 * 2) Subscribe to the system event handler
 * 3) Generate an exception by enqueueing an invalid AQL dispatch packet
 * 4) Confirm that the system event handler is called
 *
 * Expected Results: The system event handler is called with a memory fault event for the GPU and
 * for the address of the invalid kernel object.
 */
void SignalExhaustionErrorHandlerTest::TestSystemErrorHandlerAfterSignalExhaustion(void) {
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));
  ASSERT_FALSE(gpus.empty());

  // The runtime removes its memory fault handler after the first fault, so only one fault can be
  // reported per process: use a single GPU.
  RunOnAgent(gpus[0]);
}
