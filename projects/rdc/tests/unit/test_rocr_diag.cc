/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// The ROCr checks of `rdci diag` (the compute queue check and the system memory
// check) and the module that runs them, against the link-time HSA fake in
// fake_hsa.cc: this binary builds the module's own sources, so it needs neither
// ROCr nor a GPU. Every case runs in a forked child of its own, which works in a
// temporary directory the parent makes, so that a write past a block or a
// sanitizer's report fails that case alone and a hang costs one deadline.
//
// It holds the fake's own positive controls, which every case that rests on the
// fake needs, and then the module's case groups, each of which comes with the
// fix it covers: the arguments of the entry points, the results by position and
// the texts, and the mapping of an entity to its agent with the mapping
// failures. The groups still to come, each as its own task, are:
// - the order in which a check creates and releases what it uses, the clean-up
//   failures, and the abandon of a kernel or a copy that does not end;
// - the kernel argument buffer, the dispatched packet and the grid;
// - the pools a check chooses, the accesses it asks for and the sub-tests it
//   skips;
// - the kernel files it searches for and the ones it refuses to open;
// - what a run writes to stdout and stderr.

#define RDC_FAKE_HSA_HARNESS 1

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "fake_hsa.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "rdc_lib/DiagText.h"
#include "rdc_lib/RdcDiagnosticLibInterface.h"

#if RDC_ROCR_TEST_HOOKS
#include "rdc_modules/rdc_rocr/RocrTestHooks.h"
#endif

namespace {

// The pools the fake lists for the CPU agent: a standard pool, a kernel argument
// pool and a coarse-grained one.
constexpr uint32_t kCpuAgent = 0;
constexpr uint32_t kStandardPool = 0;
constexpr uint32_t kKernArgPool = 1;

// A status ROCr has no string for. It is inside hsa_status_t's range (hsa.h's largest enumerator
// is HSA_STATUS_ERROR_FATAL, 0x1026) and no enumerator uses it, so a value out of that range,
// which is undefined behaviour to load, is not needed to make a status with no text.
constexpr int32_t kStatusWithNoString = 0x101F;

hsa_amd_memory_pool_t pool_handle(uint32_t pool) {
  hsa_amd_memory_pool_t handle = {};
  handle.handle = rdc_fake_hsa_pool_handle(pool);
  return handle;
}

hsa_agent_t agent_handle(uint32_t agent) {
  hsa_agent_t handle = {};
  handle.handle = rdc_fake_hsa_agent_handle(agent);
  return handle;
}

// What the queue error callback of the case below recorded.
hsa_status_t g_error_status = HSA_STATUS_SUCCESS;
void* g_error_data = nullptr;

void queue_error(hsa_status_t status, hsa_queue_t*, void* data) {
  g_error_status = status;
  g_error_data = data;
}

}  // namespace

// An hsa_shut_down() to zero that leaves a queue alive latches ROCr for the rest
// of the process, as it does on the hardware (f22/exp/leaks_1.txt).
TEST(FakeHsaControl, AShutdownWithALiveQueueLatchesTheNextInit) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    hsa_queue_t* queue = nullptr;
    ASSERT_EQ(hsa_queue_create(agent_handle(1), 128, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0, 0,
                               &queue),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_latched(), 1);
    EXPECT_EQ(hsa_init(), HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
    EXPECT_EQ(rdc_fake_hsa_load_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_failed_load_count(), 1u);
  });
}

// A signal, a block or an executable left alive is not what arms it.
TEST(FakeHsaControl, AShutdownWithOtherObjectsAliveDoesNotLatch) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    hsa_signal_t signal = {};
    ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
    void* block = nullptr;
    ASSERT_EQ(hsa_amd_memory_pool_allocate(pool_handle(kStandardPool), 64, 0, &block),
              HSA_STATUS_SUCCESS);
    hsa_executable_t executable = {};
    ASSERT_EQ(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                        nullptr, &executable),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);
    EXPECT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_load_count(), 2u);
  });
}

// A shutdown that does not reach zero never arms it, which is what F3's run-wide
// reference and an abandoned object's unreleased one rest on.
TEST(FakeHsaControl, AShutdownThatKeepsAReferenceDoesNotLatch) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    hsa_queue_t* queue = nullptr;
    ASSERT_EQ(hsa_queue_create(agent_handle(1), 128, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0, 0,
                               &queue),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 1u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);
    EXPECT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_load_count(), 1u);
  });
}

// A block freed twice is counted, not a crash: the mapping stays after the free.
TEST(FakeHsaControl, ABlockFreedTwiceIsReported) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    void* block = nullptr;
    ASSERT_EQ(hsa_amd_memory_pool_allocate(pool_handle(kStandardPool), 128, 0, &block),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_amd_memory_pool_free(block), HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_double_frees(), 0u);
    EXPECT_EQ(hsa_memory_free(block), HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_double_frees(), 1u);
    EXPECT_EQ(rdc_fake_hsa_alloc_frees(0), 2u);
    EXPECT_EQ(rdc_fake_hsa_live_blocks(), 0u);
  });
}

// A free of a pointer the fake never issued is recorded and refused.
TEST(FakeHsaControl, AFreeOfAPointerItNeverIssuedIsSeen) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    int own = 0;
    EXPECT_EQ(hsa_amd_memory_pool_free(&own), HSA_STATUS_ERROR_INVALID_ALLOCATION);
    EXPECT_EQ(rdc_fake_hsa_unissued_frees(), 1u);
    EXPECT_EQ(rdc_fake_hsa_double_frees(), 0u);
  });
}

// A fresh block, and a fresh queue's ring, hold 0x5A rather than the zeroes a
// new mapping would read as, so that what a check does not write shows.
TEST(FakeHsaControl, AFreshBlockAndANewRingRead0x5A) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    void* block = nullptr;
    ASSERT_EQ(hsa_amd_memory_pool_allocate(pool_handle(kStandardPool), 256, 0, &block),
              HSA_STATUS_SUCCESS);
    const unsigned char* bytes = static_cast<const unsigned char*>(block);
    for (size_t i = 0; i < 256; i++) {
      ASSERT_EQ(bytes[i], 0x5A) << "byte " << i;
    }

    hsa_queue_t* queue = nullptr;
    ASSERT_EQ(hsa_queue_create(agent_handle(1), 128, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0, 0,
                               &queue),
              HSA_STATUS_SUCCESS);
    ASSERT_NE(queue, nullptr);
    const unsigned char* ring = static_cast<const unsigned char*>(queue->base_address);
    for (size_t i = 0; i < 128 * sizeof(hsa_kernel_dispatch_packet_t); i++) {
      ASSERT_EQ(ring[i], 0x5A) << "ring byte " << i;
    }
  });
}

// hsa_status_string leaves its output untouched for a status it has no string
// for, as ROCr does, so that a text built from it carries no dirt.
TEST(FakeHsaControl, AStatusItHasNoStringForGetsNone) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    const char* known = nullptr;
    EXPECT_EQ(hsa_status_string(HSA_STATUS_ERROR_OUT_OF_RESOURCES, &known), HSA_STATUS_SUCCESS);
    ASSERT_NE(known, nullptr);
    EXPECT_STRNE(known, "");

    const char* kept = "the text the caller had";
    const char* text = kept;
    EXPECT_EQ(hsa_status_string(static_cast<hsa_status_t>(kStatusWithNoString), &text),
              HSA_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(text, kept);
  });
}

// An allocation above the pool's maximum is refused as ROCr refuses it, and
// hands out no block.
TEST(FakeHsaControl, AnAllocationAboveThePoolsMaximumIsRefused) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    void* block = reinterpret_cast<void*>(0x1);
    EXPECT_EQ(hsa_amd_memory_pool_allocate(pool_handle(kStandardPool), (64 * 1024) + 1, 0, &block),
              HSA_STATUS_ERROR_INVALID_ALLOCATION);
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);
    EXPECT_EQ(block, reinterpret_cast<void*>(0x1));
  });
}

// Every entry point called with no live reference is refused and recorded, as
// ROCr's are.
TEST(FakeHsaControl, ACallWithNoLiveReferenceIsRefused) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    void* block = nullptr;
    EXPECT_EQ(hsa_amd_memory_pool_allocate(pool_handle(kStandardPool), 64, 0, &block),
              HSA_STATUS_ERROR_NOT_INITIALIZED);
    EXPECT_EQ(hsa_iterate_agents(nullptr, nullptr), HSA_STATUS_ERROR_NOT_INITIALIZED);
    EXPECT_EQ(rdc_fake_hsa_refused_calls(), 2u);
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);
  });
}

// A handle the fake never issued is refused without being read.
TEST(FakeHsaControl, AHandleItNeverIssuedIsRefused) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
    EXPECT_EQ(hsa_agent_get_info(agent_handle(99), HSA_AGENT_INFO_DEVICE, &type),
              HSA_STATUS_ERROR_INVALID_AGENT);
    hsa_signal_t signal = {};
    signal.handle = 0x999;
    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_ERROR_INVALID_SIGNAL);
    EXPECT_EQ(hsa_queue_destroy(reinterpret_cast<hsa_queue_t*>(0x1)),
              HSA_STATUS_ERROR_INVALID_QUEUE);
    hsa_executable_t executable = {};
    executable.handle = 0x999;
    EXPECT_EQ(hsa_executable_destroy(executable), HSA_STATUS_ERROR_INVALID_EXECUTABLE);
    hsa_code_object_reader_t reader = {};
    reader.handle = 0x999;
    EXPECT_EQ(hsa_code_object_reader_destroy(reader), HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER);
    EXPECT_EQ(rdc_fake_hsa_foreign_handles(), 5u);

    hsa_amd_memory_pool_t pool = {};
    pool.handle = 0x999;
    void* block = nullptr;
    EXPECT_EQ(hsa_amd_memory_pool_allocate(pool, 64, 0, &block), HSA_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rdc_fake_hsa_foreign_pools(), 1u);
  });
}

// A write past a block lands in the redzone, which the free checks.
TEST(FakeHsaControl, AWritePastABlockIsSeenByItsRedzone) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    void* block = nullptr;
    ASSERT_EQ(hsa_amd_memory_pool_allocate(pool_handle(kStandardPool), 16, 0, &block),
              HSA_STATUS_SUCCESS);
    unsigned char* bytes = static_cast<unsigned char*>(block);
    bytes[16] = 0;  // one byte past the block, as a check's off-by-one would be
    EXPECT_EQ(hsa_amd_memory_pool_free(block), HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_redzone_failures(), 1u);
  });
}

// A wait ends at its hint and returns the signal's value, and the knob that
// completes a dispatch at its third wait leaves the two before it at 1.
TEST(FakeHsaControl, AWaitEndsAtItsHintAndTheThirdOneCompletesTheDispatch) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    // A thousand ticks a second, so that a hint of 50 is 50 ms.
    rdc_fake_hsa_set_timestamp_frequency(1000);
    rdc_fake_hsa_set_completion_wait(3);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

    hsa_queue_t* queue = nullptr;
    ASSERT_EQ(hsa_queue_create(agent_handle(1), 128, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0, 0,
                               &queue),
              HSA_STATUS_SUCCESS);
    hsa_signal_t signal = {};
    ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
    void* arguments = nullptr;
    ASSERT_EQ(hsa_amd_memory_pool_allocate(pool_handle(kKernArgPool), 80, 0, &arguments),
              HSA_STATUS_SUCCESS);

    hsa_kernel_dispatch_packet_t packet = {};
    packet.setup = 1;
    packet.workgroup_size_x = 64;
    packet.grid_size_x = 64;
    packet.kernarg_address = arguments;
    packet.completion_signal = signal;
    std::memcpy(queue->base_address, &packet, sizeof(packet));
    hsa_signal_store_relaxed(queue->doorbell_signal, 0);

    ASSERT_EQ(rdc_fake_hsa_dispatch_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_dispatch_grid(0), 64u);
    EXPECT_EQ(rdc_fake_hsa_dispatch_kernarg_size(0), 80u);
    EXPECT_EQ(rdc_fake_hsa_dispatch_kernarg_address(0), reinterpret_cast<uint64_t>(arguments));

    EXPECT_EQ(
        hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, 50, HSA_WAIT_STATE_BLOCKED),
        1);
    EXPECT_EQ(
        hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, 50, HSA_WAIT_STATE_BLOCKED),
        1);
    EXPECT_EQ(
        hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, 50, HSA_WAIT_STATE_BLOCKED),
        0);
    EXPECT_EQ(rdc_fake_hsa_wait_count(), 3u);
    ASSERT_NE(rdc_fake_hsa_dispatch_signal(0), UINT32_MAX);
    EXPECT_EQ(rdc_fake_hsa_signal_waits(rdc_fake_hsa_dispatch_signal(0)), 3u);
    ASSERT_EQ(rdc_fake_hsa_wait_hint_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_wait_hint(0), 50u);
  });
}

// A gate holds the call of the thread it names until the case releases it, which
// is what orders the cases that run two checks at once.
TEST(FakeHsaControl, AGateHoldsACallUntilItIsReleased) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    rdc_fake_hsa_gate("hsa_agent_get_info", 7, 0);

    bool returned = false;
    std::thread caller([&returned]() {
      rdc_fake_hsa_set_caller(7);
      hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
      hsa_agent_get_info(agent_handle(0), HSA_AGENT_INFO_DEVICE, &type);
      returned = true;
    });

    EXPECT_EQ(rdc_fake_hsa_gate_wait("hsa_agent_get_info", 1, 5000), 1);
    EXPECT_FALSE(returned);
    EXPECT_EQ(rdc_fake_hsa_gate_held("hsa_agent_get_info"), 1u);
    rdc_fake_hsa_gate_release("hsa_agent_get_info");
    caller.join();
    EXPECT_TRUE(returned);
    EXPECT_EQ(rdc_fake_hsa_gate_held("hsa_agent_get_info"), 0u);
  });
}

// A queue error reaches the queue's callback with the data the queue was made
// with; a queue without one, as F9's queues are, has the error recorded and is
// stopped. hsa_queue_destroy joins the thread that raised it, as ~AqlQueue waits
// for its error handler.
TEST(FakeHsaControl, AQueueErrorReachesTheCallbackAndIsJoinedByTheDestroy) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    int block = 0;
    hsa_queue_t* with_callback = nullptr;
    ASSERT_EQ(hsa_queue_create(agent_handle(1), 128, HSA_QUEUE_TYPE_MULTI, queue_error, &block, 0,
                               0, &with_callback),
              HSA_STATUS_SUCCESS);
    hsa_queue_t* without_callback = nullptr;
    ASSERT_EQ(hsa_queue_create(agent_handle(1), 128, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0, 0,
                               &without_callback),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_queue_error_data(0), &block);

    rdc_fake_hsa_raise_queue_error(0, HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_EQ(hsa_queue_destroy(with_callback), HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_error_status, HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_EQ(g_error_data, &block);
    EXPECT_EQ(rdc_fake_hsa_queue_errors_recorded(), 0u);

    rdc_fake_hsa_raise_queue_error(1, HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_EQ(hsa_queue_destroy(without_callback), HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_queue_errors_recorded(), 1u);
    EXPECT_EQ(rdc_fake_hsa_queue_destroyed(0), 1);
    EXPECT_EQ(rdc_fake_hsa_queue_destroyed(1), 1);
  });
}

// The knob that maps the second page of an allocation onto its first, so that a
// check which writes past the end of its own data reads back what it did not
// write, and the one that hands out a kernel argument block that is not at the
// start of its page.
TEST(FakeHsaControl, TheAliasedPagesAndTheKernelArgumentOffsetAreHandedOut) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    rdc_fake_hsa_set_alias_pages(1);
    void* block = nullptr;
    ASSERT_EQ(hsa_amd_memory_pool_allocate(pool_handle(kStandardPool), 8192, 0, &block),
              HSA_STATUS_SUCCESS);
    unsigned char* bytes = static_cast<unsigned char*>(block);
    bytes[0] = 0x11;
    EXPECT_EQ(bytes[4096], 0x11);
    bytes[4096 + 7] = 0x22;
    EXPECT_EQ(bytes[7], 0x22);
    rdc_fake_hsa_set_alias_pages(0);

    rdc_fake_hsa_set_kernarg_offset(8);
    void* arguments = nullptr;
    ASSERT_EQ(hsa_amd_memory_pool_allocate(pool_handle(kKernArgPool), 80, 0, &arguments),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(arguments) % 4096, 8u);
    void* next = nullptr;
    ASSERT_EQ(hsa_amd_memory_pool_allocate(pool_handle(kKernArgPool), 80, 0, &next),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(next) % 4096, 0u);
  });
}

// A failure switch fires on the call it names and nothing else, a failed
// hsa_init() takes no reference, and a throw armed on a call leaves the fake
// through it, which is the guard of the catch of what no ROCr call raises.
TEST(FakeHsaControl, AFailureFiresOnTheCallItNamesAndAThrowLeavesTheFake) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail("hsa_init", 2, HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_init(), HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 2u);
    EXPECT_EQ(rdc_fake_hsa_call_count("hsa_init"), 3u);
    ASSERT_EQ(rdc_fake_hsa_call_log_size(), 3u);
    EXPECT_STREQ(rdc_fake_hsa_call_log_entry(0), "hsa_init");

    rdc_fake_hsa_clear_failures();
    rdc_fake_hsa_throw_int("hsa_agent_get_info", 0, 7);
    int caught = 0;
    try {
      hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
      hsa_agent_get_info(agent_handle(0), HSA_AGENT_INFO_DEVICE, &type);
    } catch (int value) {
      caught = value;
    }
    EXPECT_EQ(caught, 7);
  });
}

// The module's entry points are what the cases to come call; this one holds the
// test binary to the interface they are declared with.
TEST(FakeHsaControl, TheModulesEntryPointsAreLinkedIn) {
  RDC_ROCR_CASE({
    EXPECT_EQ(rdc_diag_init(0), RDC_ST_OK);
    EXPECT_EQ(rdc_diag_destroy(), RDC_ST_OK);
  });
}

namespace {

// A check's result is half a megabyte, so a case keeps it on the heap.
using ResultPtr = std::unique_ptr<rdc_diag_test_result_t>;

ResultPtr new_result() { return ResultPtr(new rdc_diag_test_result_t()); }

std::vector<uint32_t> flat_ids(uint32_t count) {
  std::vector<uint32_t> ids;
  for (uint32_t id = 0; id < count; id++) {
    ids.push_back(id);
  }
  return ids;
}

// The entity id of a partition instance, as RDC encodes it.
uint32_t partition_id(uint32_t socket, uint32_t instance) {
  rdc_entity_info_t info = {};
  info.device_index = socket;
  info.instance_index = instance;
  info.entity_role = RDC_DEVICE_ROLE_PARTITION_INSTANCE;
  info.device_type = RDC_DEVICE_TYPE_GPU;
  return rdc_get_entity_index_from_info(info);
}

std::string text_of(const char* text) { return std::string(text); }

// What the module's progress callback was given, and the call that throws (counting from 1).
std::vector<std::string> g_progress_lines;
size_t g_progress_throws_at = 0;

void record_progress(void*, void* message) {
  g_progress_lines.push_back(static_cast<const char*>(message));
  if (g_progress_throws_at != 0 && g_progress_lines.size() == g_progress_throws_at) {
    throw std::runtime_error("the caller's callback failed");
  }
}

// The kernel file a check looks for, written into the case's own working directory, which the
// module searches first. What is in it does not matter: the fake records the identity of the
// file the reader was made from and reports the kernel's own sizes.
void write_kernel_file(const char* name) {
  std::FILE* file = std::fopen(name, "wb");
  ASSERT_NE(file, nullptr) << name << ": " << std::strerror(errno);
  ASSERT_GE(std::fputs("hsaco", file), 0) << name;
  ASSERT_EQ(std::fclose(file), 0) << name;
}

// The directories the module searches for a kernel file, where the build has the hook. In a
// build without it the module keeps its own list, which a case that writes its files into the
// working directory reaches through the "./" entry F22 drops.
void search_these_directories(const std::vector<std::string>& directories) {
#if RDC_ROCR_TEST_HOOKS
  std::vector<const char*> named;
  for (const std::string& directory : directories) {
    named.push_back(directory.c_str());
  }
  rdc_rocr_test_set_kernel_directories(named.data(), named.size());
#else
  (void)directories;
#endif
}

// Both checks' kernel files, so that a case reaches a dispatch, and the working directory named
// as the place to find them: F22 does not search the working directory of its own accord.
void write_kernel_files() {
  write_kernel_file("binary_search_kernels.hsaco");
  write_kernel_file("gpuReadWrite_kernels.hsaco");
  search_these_directories({"."});
}

// The file a path names, as the fake records the file each code object reader was made from.
struct FileIdentity {
  uint64_t device = 0;
  uint64_t inode = 0;
};

FileIdentity identity_of(const char* path) {
  struct stat about = {};
  FileIdentity identity;
  EXPECT_EQ(::stat(path, &about), 0) << path << ": " << std::strerror(errno);
  identity.device = static_cast<uint64_t>(about.st_dev);
  identity.inode = static_cast<uint64_t>(about.st_ino);
  return identity;
}

bool a_reader_was_made_from(const FileIdentity& identity) {
  for (uint32_t reader = 0; reader < rdc_fake_hsa_reader_count(); reader++) {
    if (rdc_fake_hsa_reader_device(reader) == identity.device &&
        rdc_fake_hsa_reader_inode(reader) == identity.inode) {
      return true;
    }
  }
  return false;
}

// What one GPU's run of a check answered: the outcome, which is that GPU's when the group holds
// one, and the GPU's own text.
struct OneGpu {
  rdc_diag_result_t status = RDC_DIAG_RESULT_PASS;
  std::string text;
};

OneGpu run_one_gpu(rdc_diag_test_cases_t check) {
  uint32_t ids[1] = {0};
  ResultPtr result = new_result();
  OneGpu answer;
  EXPECT_EQ(rdc_diag_test_case_run(check, ids, 1, nullptr, 0, result.get(), nullptr), RDC_ST_OK);
  answer.status = result->status;
  if (result->per_gpu_result_count == 1) {
    answer.text = result->gpu_results[0].gpu_result.msg;
  }
  return answer;
}

rdc_diag_callback_t progress_callback() {
  rdc_diag_callback_t callback = {};
  callback.callback = record_progress;
  callback.cookie = &g_progress_lines;
  return callback;
}

}  // namespace

// --- the arguments of the entry points ------------------------------------------------------

// A null result is refused and nothing is checked (M: T4's mutation drops the null result from
// the entry point's argument check).
TEST(RocrArguments, ANullResultIsRefused) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    uint32_t ids[1] = {0};
    EXPECT_EQ(rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, nullptr, nullptr),
              RDC_ST_BAD_PARAMETER);
    EXPECT_EQ(rdc_fake_hsa_call_log_size(), 0u);
    EXPECT_EQ(rdc_fake_hsa_node_id_calls(), 0u);
  });
}

// F: F9 reads through a null gpu_index.
TEST(RocrArguments, ANullGpuIndexIsRefused) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ResultPtr result = new_result();
    EXPECT_EQ(rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, nullptr, 2, nullptr, 0, result.get(),
                                     nullptr),
              RDC_ST_BAD_PARAMETER);
    EXPECT_EQ(rdc_fake_hsa_call_log_size(), 0u);
    EXPECT_EQ(result->per_gpu_result_count, 0u);
  });
}

// F: F9 writes the test cases through a null array.
TEST(RocrArguments, AQueryWithoutAnArrayIsRefused) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    uint32_t count = 7;
    EXPECT_EQ(rdc_diag_test_cases_query(nullptr, &count), RDC_ST_BAD_PARAMETER);
    EXPECT_EQ(count, 0u);

    rdc_diag_test_cases_t cases[MAX_TEST_CASES] = {};
    EXPECT_EQ(rdc_diag_test_cases_query(cases, &count), RDC_ST_OK);
    EXPECT_EQ(count, 2u);
  });
}

// A call that fails inside the second of three GPUs' checks is that GPU's failure, with the
// exception's text, and the first and the third GPU keep their results (F: F9 overwrites one
// status with the next GPU's).
TEST(RocrArguments, AFailedCallIsThatGpusFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    // The first and the third GPU carry a node no agent has, so that only the second one reaches
    // the call that fails.
    write_kernel_files();
    rdc_fake_hsa_set_node_id(0, 9);
    rdc_fake_hsa_set_node_id(2, 9);
    rdc_fake_hsa_fail("hsa_signal_create", 0, HSA_STATUS_ERROR_OUT_OF_RESOURCES);

    uint32_t ids[3] = {0, 1, 2};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 3, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 3u);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    EXPECT_EQ(result->gpu_results[0].gpu_index, 0u);
    EXPECT_EQ(result->gpu_results[1].gpu_index, 1u);
    EXPECT_EQ(result->gpu_results[2].gpu_index, 2u);
    EXPECT_EQ(text_of(result->gpu_results[0].gpu_result.msg), "ROCr does not list the GPU.");
    EXPECT_EQ(text_of(result->gpu_results[2].gpu_result.msg), "ROCr does not list the GPU.");
    const std::string failed = result->gpu_results[1].gpu_result.msg;
    EXPECT_NE(failed.find("Fail to create signal."), std::string::npos) << failed;
    EXPECT_NE(failed.find("4104"), std::string::npos) << failed;
  });
}

// The same with an exception that is not a std::exception, which no ROCr call throws: it is that
// GPU's failure too, with the sentence that says nothing more (F: F9's helpers catch
// std::exception only, so it leaves the entry point).
TEST(RocrArguments, AnExceptionOfAnotherKindIsThatGpusFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_set_node_id(0, 9);
    rdc_fake_hsa_set_node_id(2, 9);
    rdc_fake_hsa_throw_int("hsa_signal_create", 0, 7);

    uint32_t ids[3] = {0, 1, 2};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 3, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 3u);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    EXPECT_EQ(text_of(result->gpu_results[1].gpu_result.msg),
              "An unknown error stopped the check.");
    EXPECT_EQ(text_of(result->gpu_results[0].gpu_result.msg), "ROCr does not list the GPU.");
    EXPECT_EQ(text_of(result->gpu_results[2].gpu_result.msg), "ROCr does not list the GPU.");
  });
}

// --- the results by position ------------------------------------------------------------------

// Each GPU's row is the next one written, carrying its entity id, and the count covers the rows
// written (F: F9 leaves the count at 0 and sets no gpu_index).
TEST(RocrResults, ResultsAreWrittenByPosition) {
  RDC_ROCR_CASE({
    // No GPU can be mapped, so every row is written without an HSA call.
    for (uint32_t count : {1u, 2u, 128u}) {
      rdc_fake_hsa_reset();
      rdc_fake_hsa_fail_node_id(UINT32_MAX, AMDSMI_STATUS_NOT_FOUND);
      std::vector<uint32_t> ids = flat_ids(count);
      ResultPtr result = new_result();
      ASSERT_EQ(rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids.data(), count, nullptr, 0,
                                       result.get(), nullptr),
                RDC_ST_OK);
      ASSERT_EQ(result->per_gpu_result_count, count);
      for (uint32_t i = 0; i < count; i++) {
        EXPECT_EQ(result->gpu_results[i].gpu_index, ids[i]) << "row " << i;
        EXPECT_EQ(text_of(result->gpu_results[i].gpu_result.msg), "RDC cannot identify the GPU.")
            << "row " << i;
      }
      EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP);
      EXPECT_EQ(rdc_fake_hsa_call_log_size(), 0u);
    }

    // A group of a flat id and a partition id, which are the ids the rows carry.
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail_node_id(UINT32_MAX, AMDSMI_STATUS_NOT_FOUND);
    uint32_t mixed[2] = {3, partition_id(1, 1)};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, mixed, 2, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 2u);
    EXPECT_EQ(result->gpu_results[0].gpu_index, mixed[0]);
    EXPECT_EQ(result->gpu_results[1].gpu_index, mixed[1]);
  });
}

// A group larger than the result's array is clamped to it, and nothing is written past the
// struct (F: F9 indexes gpu_results[] by the entity id, so ids 128 to 199 land past it).
TEST(RocrResults, AGroupLargerThanTheResultArrayIsClamped) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail_node_id(UINT32_MAX, AMDSMI_STATUS_NOT_FOUND);

    // One block that holds the result and the 72 rows F9's indexing would reach past it, so that
    // its reads and writes stay inside memory this case owns.
    const size_t kRowsPastTheStruct = 72;
    const size_t block_size =
        sizeof(rdc_diag_test_result_t) + kRowsPastTheStruct * sizeof(rdc_diag_per_gpu_result_t);
    std::vector<unsigned char> block(block_size, 0xAB);
    std::memset(block.data(), 0, sizeof(rdc_diag_test_result_t));
    // Row 128's message falls in the struct's info, which the zeroing above covered; each of rows
    // 129 to 199 gets a NUL where its message starts, as F9 reads a row's message as a string
    // before it writes the row.
    unsigned char* rows = block.data() + offsetof(rdc_diag_test_result_t, gpu_results);
    for (size_t row = 129; row < 200; row++) {
      rows[(row * sizeof(rdc_diag_per_gpu_result_t)) +
           offsetof(rdc_diag_per_gpu_result_t, gpu_result)] = 0;
    }
    const std::vector<unsigned char> tail(block.begin() + sizeof(rdc_diag_test_result_t),
                                          block.end());

    std::vector<uint32_t> ids = flat_ids(200);
    rdc_diag_test_result_t* result = reinterpret_cast<rdc_diag_test_result_t*>(block.data());
    ASSERT_EQ(rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids.data(), 200, nullptr, 0, result,
                                     nullptr),
              RDC_ST_OK);
    EXPECT_EQ(result->per_gpu_result_count, 128u);
    for (uint32_t i = 0; i < RDC_MAX_NUM_DEVICES; i++) {
      EXPECT_EQ(result->gpu_results[i].gpu_index, i) << "row " << i;
    }
    const std::vector<unsigned char> after(block.begin() + sizeof(rdc_diag_test_result_t),
                                           block.end());
    EXPECT_EQ(after, tail);
  });
}

// The check's status is the highest of its GPUs', whichever GPU comes first (F: F9 keeps the
// last GPU's status).
TEST(RocrResults, TheStatusIsTheHighestOfItsGpus) {
  RDC_ROCR_CASE({
    for (int failing_first = 1; failing_first >= 0; failing_first--) {
      rdc_fake_hsa_reset();
      write_kernel_files();
      rdc_fake_hsa_set_node_id(0, failing_first != 0 ? 1u : 9u);
      rdc_fake_hsa_set_node_id(1, failing_first != 0 ? 9u : 1u);
      rdc_fake_hsa_fail("hsa_signal_create", 0, HSA_STATUS_ERROR_OUT_OF_RESOURCES);

      uint32_t ids[2] = {0, 1};
      ResultPtr result = new_result();
      ASSERT_EQ(
          rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 2, nullptr, 0, result.get(), nullptr),
          RDC_ST_OK);
      ASSERT_EQ(result->per_gpu_result_count, 2u);
      EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL) << "failing GPU first: " << failing_first;
      const uint32_t failed = failing_first != 0 ? 0u : 1u;
      const uint32_t skipped = failing_first != 0 ? 1u : 0u;
      EXPECT_NE(text_of(result->gpu_results[failed].gpu_result.msg).find("4104"),
                std::string::npos);
      EXPECT_EQ(text_of(result->gpu_results[skipped].gpu_result.msg),
                "ROCr does not list the GPU.");
    }
  });
}

// A group with no GPU is the check's skip, with the result it leaves naming its test case (F: F9
// returns RDC_ST_BAD_PARAMETER and writes no result).
TEST(RocrResults, AnEmptyGroupIsSkipped) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ResultPtr result = new_result();
    std::memset(result.get(), 0xAB, sizeof(rdc_diag_test_result_t));
    uint32_t ids[1] = {0};
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 0, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP);
    EXPECT_EQ(result->test_case, RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(result->per_gpu_result_count, 0u);
    EXPECT_EQ(text_of(result->info), "The group has no GPU.");
    EXPECT_EQ(rdc_fake_hsa_call_log_size(), 0u);
  });
}

// --- the texts ----------------------------------------------------------------------------------

// A text too long for one result is cut after a whole unit and says how many units it does not
// show (F: F9 cuts its text mid-word at 4095 bytes).
TEST(RocrTexts, TextsAreCutAtWholeUnits) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail_node_id(UINT32_MAX, AMDSMI_STATUS_NOT_FOUND);
    std::vector<uint32_t> ids = flat_ids(128);
    ResultPtr result = new_result();
    ASSERT_EQ(rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids.data(), 128, nullptr, 0,
                                     result.get(), nullptr),
              RDC_ST_OK);

    const std::string info = result->info;
    ASSERT_LE(info.size(), static_cast<size_t>(MAX_DIAG_MSG_LENGTH - 1));
    const size_t suffix = info.rfind(" (");
    ASSERT_NE(suffix, std::string::npos) << info;

    // What is kept ends after a whole unit, and the suffix counts the units it does not show.
    const std::string kept = info.substr(0, suffix);
    const size_t units = static_cast<size_t>(std::count(kept.begin(), kept.end(), ':'));
    EXPECT_GT(units, 0u);
    EXPECT_LT(units, 128u);
    EXPECT_EQ(info.substr(suffix), " (" + std::to_string(128 - units) + " more not shown)") << info;
    EXPECT_EQ(kept.substr(kept.size() - 4), "GPU.") << kept.substr(kept.size() - 40);
    EXPECT_EQ(kept.substr(0, 36), "GPU 0: RDC cannot identify the GPU. ") << kept.substr(0, 60);
    EXPECT_EQ(text_of(result->gpu_results[0].gpu_result.msg), "RDC cannot identify the GPU.");
  });
}

// The text's own cases: a fixed start that leaves no room for a unit is cut at the last space
// inside the room the suffix leaves, so that the suffix always fits whole (M: T4's
// mutations restore DiagText's two defects and cut the start at its last space).
TEST(RocrTexts, AFixedStartWithSpacesIsCutAtASpace) {
  RDC_ROCR_CASE({
    // The budget is MAX_DIAG_MSG_LENGTH - 1 bytes and the suffix below takes 19 of them, so
    // the start has room for 4076: the space at 4000 is inside that room and the one at
    // 4085 is past it.
    const size_t inside_the_room = 4000;
    const size_t past_the_room = 4085;
    std::string start(4090, 'a');
    start[inside_the_room] = ' ';
    start[past_the_room] = ' ';
    amd::rdc::DiagText text(start);
    text.add(std::string(50, 'b'));
    text.add(std::string(50, 'c'));

    char out[MAX_DIAG_MSG_LENGTH];
    std::memset(out, 0xAB, sizeof(out));
    text.copy_to(out);
    const std::string written(out);

    const std::string suffix = " (2 more not shown)";
    ASSERT_LE(written.size(), static_cast<size_t>(MAX_DIAG_MSG_LENGTH - 1));
    ASSERT_GT(written.size(), suffix.size());
    EXPECT_EQ(written.substr(written.size() - suffix.size()), suffix);
    const std::string kept = written.substr(0, written.size() - suffix.size());
    EXPECT_EQ(kept, start.substr(0, kept.size()));
    ASSERT_LT(kept.size(), start.size());
    EXPECT_EQ(start[kept.size()], ' ');
    // The cut took the last space inside the room, not the last space of the start:
    // cutting at that one would leave no room for the whole suffix.
    EXPECT_EQ(kept.size(), inside_the_room);
  });
}

// The same with a start that has no space in it: it is cut at the budget less the suffix (M).
TEST(RocrTexts, AFixedStartOfOneWordIsCutToTheBudget) {
  RDC_ROCR_CASE({
    const std::string start(4090, 'a');
    amd::rdc::DiagText text(start);
    text.add(std::string(50, 'b'));

    char out[MAX_DIAG_MSG_LENGTH];
    std::memset(out, 0xAB, sizeof(out));
    text.copy_to(out);
    const std::string written(out);

    const std::string suffix = " (1 more not shown)";
    EXPECT_EQ(written.size(), static_cast<size_t>(MAX_DIAG_MSG_LENGTH - 1));
    EXPECT_EQ(written.substr(written.size() - suffix.size()), suffix);
    EXPECT_EQ(written.substr(0, written.size() - suffix.size()),
              start.substr(0, written.size() - suffix.size()));
  });
}

// A unit longer than the whole budget is counted and none of it is shown (M: T4's mutation
// keeps a unit no budget can hold).
TEST(RocrTexts, AUnitTooLongForTheBudgetIsOnlyCounted) {
  RDC_ROCR_CASE({
    amd::rdc::DiagText text;
    text.add(std::string(MAX_DIAG_MSG_LENGTH + 100, 'x'));

    char out[MAX_DIAG_MSG_LENGTH];
    std::memset(out, 0xAB, sizeof(out));
    text.copy_to(out);
    EXPECT_EQ(text_of(out), "(1 more not shown)");
  });
}

// An empty unit says nothing, so it is neither kept nor counted (M).
TEST(RocrTexts, EmptyUnitsAreNeitherKeptNorCounted) {
  RDC_ROCR_CASE({
    amd::rdc::DiagText text("A start.");
    for (int i = 0; i < 1000000; i++) {
      text.add("");
    }

    char out[MAX_DIAG_MSG_LENGTH];
    std::memset(out, 0xAB, sizeof(out));
    text.copy_to(out);
    EXPECT_EQ(text_of(out), "A start.");
  });
}

// The check's info states each GPU's outcome: its first failure, else its first skip, in a
// sentence that reads on from the GPU's name (F: F9 names a position and keeps one text).
TEST(RocrTexts, InfoStatesEachGpusOutcome) {
  RDC_ROCR_CASE({
    // A GPU whose pools are checked and whose next sub-test is stopped.
    rdc_fake_hsa_reset();
    rdc_fake_hsa_throw_int("hsa_amd_agents_allow_access", 0, 7);
    uint32_t stopped[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, stopped, 1, nullptr, 0, result.get(),
                                     nullptr),
              RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 1u);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    EXPECT_EQ(text_of(result->info), "GPU 0: an unknown error stopped the check.");
    const std::string msg = result->gpu_results[0].gpu_result.msg;
    EXPECT_EQ(msg.find("Memory pool 1: allocated and freed 64 KiB. An unknown error stopped "
                       "the check."),
              0u)
        << msg;

    // A GPU ROCr does not list keeps its sentence's capital, as it is a name.
    rdc_fake_hsa_reset();
    rdc_fake_hsa_set_node_id(5, 9);
    uint32_t unlisted[1] = {5};
    result = new_result();
    ASSERT_EQ(rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, unlisted, 1, nullptr, 0, result.get(),
                                     nullptr),
              RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 1u);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP);
    EXPECT_EQ(text_of(result->info), "GPU 5: ROCr does not list the GPU.");
    EXPECT_EQ(text_of(result->gpu_results[0].gpu_result.msg), "ROCr does not list the GPU.");
  });
}

// --- the progress callback ------------------------------------------------------------------

// An exception from the caller's progress callback is the check's failure, and the GPUs checked
// before it keep their rows (F: on F9 it leaves the entry point).
TEST(RocrProgress, AProgressCallbackThatThrowsStopsTheCheck) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    // Every GPU carries a node no agent has, so each costs one ROCr load and leaves nothing.
    rdc_fake_hsa_set_node_id(0, 9);
    rdc_fake_hsa_set_node_id(1, 9);
    rdc_fake_hsa_set_node_id(2, 9);
    g_progress_lines.clear();
    g_progress_throws_at = 2;

    uint32_t ids[3] = {0, 1, 2};
    rdc_diag_callback_t callback = progress_callback();
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 3, nullptr, 0, result.get(), &callback),
        RDC_ST_OK);
    g_progress_throws_at = 0;

    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    EXPECT_EQ(text_of(result->info), "The check could not finish.");
    ASSERT_EQ(result->per_gpu_result_count, 1u);
    EXPECT_EQ(result->gpu_results[0].gpu_index, 0u);
    EXPECT_EQ(text_of(result->gpu_results[0].gpu_result.msg), "ROCr does not list the GPU.");
    EXPECT_EQ(result->gpu_results[1].gpu_index, 0u);
    EXPECT_EQ(text_of(result->gpu_results[1].gpu_result.msg), "");
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
    EXPECT_EQ(rdc_fake_hsa_unbalanced_shutdowns(), 0u);

    // ROCr is loaded afresh by the run after it, so nothing was left holding it.
    const uint32_t loads = rdc_fake_hsa_load_count();
    ResultPtr again = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, again.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(rdc_fake_hsa_load_count(), loads + 1);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
  });
}

// The callback names each GPU by its entity id, flat or partition (F: F9 sends its position in
// the group).
TEST(RocrProgress, TheProgressCallbackNamesEachEntity) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail_node_id(UINT32_MAX, AMDSMI_STATUS_NOT_FOUND);
    g_progress_lines.clear();

    uint32_t ids[2] = {3, partition_id(1, 1)};
    rdc_diag_callback_t callback = progress_callback();
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 2, nullptr, 0, result.get(), &callback),
        RDC_ST_OK);

    ASSERT_EQ(g_progress_lines.size(), 2u);
    EXPECT_EQ(g_progress_lines[0], "ROCR test on GPU 3");
    EXPECT_EQ(g_progress_lines[1], "ROCR test on GPU " + std::to_string(ids[1]));
  });
}

// A test case this module does not implement leaves the result as it was, so that the
// dispatcher's own skip stands (M: T4's mutation zeroes the result first).
TEST(RocrProgress, ATestCaseTheModuleDoesNotImplementWritesNothing) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ResultPtr result = new_result();
    result->status = RDC_DIAG_RESULT_SKIP;
    result->test_case = RDC_DIAG_NODE_TOPOLOGY;
    result->per_gpu_result_count = 0;
    const char kUntouched[] = "The module does not run this test.";
    std::memcpy(result->info, kUntouched, sizeof(kUntouched));

    uint32_t ids[1] = {0};
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_NODE_TOPOLOGY, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP);
    EXPECT_EQ(result->test_case, RDC_DIAG_NODE_TOPOLOGY);
    EXPECT_EQ(text_of(result->info), kUntouched);
    EXPECT_EQ(rdc_fake_hsa_call_log_size(), 0u);
  });
}

// --- the mapping of an entity to its agent ------------------------------------------------------

// A GPU's work is done on the agent that carries its KFD node, whatever that agent's place in
// HSA's list (F: F9 takes the agent at the GPU's own position).
TEST(RocrMapping, TheAgentsAreFoundByTheirKfdNode) {
  RDC_ROCR_CASE({
    // RDC's GPU 0 is the agent HSA lists second and its GPU 1 the one HSA lists first, as they
    // are on this host (f22/exp/amdsmi_list.txt).
    for (uint32_t entity = 0; entity < 2; entity++) {
      const uint32_t node = entity == 0 ? 2u : 1u;
      const uint32_t agent = node;  // the fake lists the CPU, then node 1, then node 2
      rdc_fake_hsa_reset();
      write_kernel_files();
      rdc_fake_hsa_set_node_id(entity, node);

      uint32_t ids[1] = {entity};
      ResultPtr result = new_result();
      ASSERT_EQ(
          rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
          RDC_ST_OK);
      ASSERT_EQ(rdc_fake_hsa_queue_count(), 1u) << "entity " << entity;
      EXPECT_EQ(rdc_fake_hsa_queue_agent(0), agent) << "entity " << entity;
      ASSERT_EQ(result->per_gpu_result_count, 1u);
      EXPECT_EQ(result->gpu_results[0].gpu_index, entity);
      EXPECT_EQ(text_of(result->info).find("GPU " + std::to_string(entity) + ": "), 0u)
          << result->info;
    }
  });
}

// A group whose GPUs AMD SMI cannot map costs no HSA call at all (F: F9 asks AMD SMI nothing and
// goes straight to ROCr).
TEST(RocrMapping, AGroupThatCannotBeMappedMakesNoHsaCall) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail_node_id(UINT32_MAX, AMDSMI_STATUS_NOT_FOUND);

    uint32_t ids[2] = {0, 1};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 2, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(rdc_fake_hsa_call_log_size(), 0u);
    EXPECT_EQ(rdc_fake_hsa_node_id_calls(), 2u);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP);
    ASSERT_EQ(result->per_gpu_result_count, 2u);
    EXPECT_EQ(text_of(result->gpu_results[0].gpu_result.msg), "RDC cannot identify the GPU.");
    EXPECT_EQ(text_of(result->gpu_results[1].gpu_result.msg), "RDC cannot identify the GPU.");
  });
}

// A GPU whose KFD node no agent carries is that GPU's skip, for the one ROCr load the lookup
// costs (F: F9 reads the agent at the GPU's position).
TEST(RocrMapping, ANodeNoAgentCarriesIsSkipped) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_set_node_id(0, 9);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 1u);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP);
    EXPECT_EQ(text_of(result->gpu_results[0].gpu_result.msg), "ROCr does not list the GPU.");
    EXPECT_EQ(rdc_fake_hsa_load_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
    EXPECT_EQ(rdc_fake_hsa_refused_calls(), 0u);
    EXPECT_EQ(rdc_fake_hsa_queue_count(), 0u);
  });
}

// A read of an agent's KFD node that fails is that GPU's failure, naming the call that failed
// and its status: ROCr listed agents, so the GPU is not one ROCr does not list, and the GPU's
// outcome says which call could not be made (F: F9 takes the agent at the GPU's position and
// reads no agent's node; M: T4's mutation ignores the read's status, after which the node stays
// 0 and the GPU is skipped as one ROCr does not list).
TEST(RocrMapping, AFailedNodeReadIsThatGpusFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail_agent_info(UINT32_MAX, HSA_AMD_AGENT_INFO_DRIVER_NODE_ID,
                                 HSA_STATUS_ERROR_INVALID_ARGUMENT);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 1u);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    const std::string text = result->gpu_results[0].gpu_result.msg;
    EXPECT_NE(text.find("Fail to read the KFD node of an agent."), std::string::npos) << text;
    EXPECT_NE(text.find("4097"), std::string::npos) << text;
    // The failure came before anything was made for the GPU, and the run gave ROCr up.
    EXPECT_EQ(rdc_fake_hsa_queue_count(), 0u);
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);
    EXPECT_EQ(rdc_fake_hsa_live_blocks(), 0u);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
  });
}

// Each cause of a GPU without an agent has its own sentence (F in an ASan build of F9's sources,
// where entity 2 reads past the agents it lists; M otherwise).
TEST(RocrMapping, EachMappingFailureHasItsOwnText) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail_node_id(0, AMDSMI_STATUS_NOT_FOUND);
    rdc_fake_hsa_fail_node_id(1, AMDSMI_STATUS_NOT_SUPPORTED);
    rdc_fake_hsa_set_node_id(2, 9);  // the fake lists two GPU agents, with nodes 1 and 2

    uint32_t ids[3] = {0, 1, 2};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 3, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 3u);
    EXPECT_EQ(text_of(result->gpu_results[0].gpu_result.msg), "RDC cannot identify the GPU.");
    EXPECT_EQ(text_of(result->gpu_results[1].gpu_result.msg),
              "AMD SMI reports no KFD node for the GPU.");
    EXPECT_EQ(text_of(result->gpu_results[2].gpu_result.msg), "ROCr does not list the GPU.");
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP);
  });
}

namespace {

// The order in which the fake released what a check made, with its hsa_shut_down() calls.
std::vector<std::string> destroy_log() {
  std::vector<std::string> entries;
  for (uint32_t i = 0; i < rdc_fake_hsa_destroy_log_size(); i++) {
    entries.push_back(rdc_fake_hsa_destroy_log_entry(i));
  }
  return entries;
}

std::string joined(const std::vector<std::string>& entries) {
  std::string text;
  for (const std::string& entry : entries) {
    text += entry;
    text += "; ";
  }
  return text;
}

// Where the first hsa_shut_down() of the log is, so that a case can say what was released before
// a test object gave its reference back.
size_t first_shutdown(const std::vector<std::string>& entries) {
  for (size_t i = 0; i < entries.size(); i++) {
    if (entries[i] == "shut_down") {
      return i;
    }
  }
  return entries.size();
}

// The calls the fake saw, which a case that states a check made none of them reads.
std::vector<std::string> call_log() {
  std::vector<std::string> entries;
  for (uint32_t i = 0; i < rdc_fake_hsa_call_log_size(); i++) {
    entries.push_back(rdc_fake_hsa_call_log_entry(i));
  }
  return entries;
}

size_t released(const std::vector<std::string>& entries, const std::string& kind) {
  size_t count = 0;
  for (const std::string& entry : entries) {
    if (entry.compare(0, kind.size(), kind) == 0) {
      count++;
    }
  }
  return count;
}

// The descriptors this process holds, so that a check that opens a kernel file closes it again.
size_t open_descriptors() {
  size_t count = 0;
  DIR* directory = ::opendir("/proc/self/fd");
  if (directory == nullptr) {
    return 0;
  }
  while (::readdir(directory) != nullptr) {
    count++;
  }
  ::closedir(directory);
  return count;
}

// 1 MB of stack filled with 0xA5, so that a value a check reads before it writes it shows. It is
// not instrumented: ASan's redzones would leave the top of the next frame clean.
__attribute__((noinline, no_sanitize_address)) void dirty_stack() {
  volatile unsigned char block[1024 * 1024];
  for (size_t i = 0; i < sizeof(block); i++) {
    block[i] = 0xA5;
  }
}

// Every object the fake issued for a run was released: its queues and signals destroyed, its
// blocks freed, its executables and their readers destroyed.
void expect_everything_released() {
  const std::vector<std::string> log = destroy_log();
  EXPECT_EQ(rdc_fake_hsa_live_blocks(), 0u) << joined(log);
  EXPECT_EQ(released(log, "queue"), rdc_fake_hsa_queue_count()) << joined(log);
  // Every signal but the doorbell each queue carries, which ROCr destroys with its queue.
  EXPECT_EQ(released(log, "signal"), rdc_fake_hsa_signal_count() - rdc_fake_hsa_queue_count())
      << joined(log);
  EXPECT_EQ(released(log, "executable"), rdc_fake_hsa_executable_count()) << joined(log);
  EXPECT_EQ(released(log, "reader"), rdc_fake_hsa_reader_count()) << joined(log);
  EXPECT_EQ(rdc_fake_hsa_double_frees(), 0u);
  EXPECT_EQ(rdc_fake_hsa_unissued_frees(), 0u);
}

// A third GPU agent with a pool of its own, so that a case can check a group of three GPUs that
// ROCr lists.
void add_third_gpu() {
  const uint32_t agent = rdc_fake_hsa_add_agent(1, 3, "gfx1030");
  rdc_fake_hsa_add_pool(agent, HSA_AMD_SEGMENT_GLOBAL,
                        HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, 64 * 1024, 64 * 1024, 1,
                        4096, 4096, 1);
}

// The deadline a case lowers through the hook, so that work that cannot be stopped is abandoned
// after a few hints rather than after the module's ten seconds.
constexpr uint64_t kShortDeadlineMs = 300;
constexpr uint64_t kModuleDeadlineMs = 10 * 1000;

// One more GPU agent, with a pool of its own, so that a case can check more GPUs than the fake
// lists by default.
uint32_t add_gpu_with_node(uint32_t node) {
  const uint32_t agent = rdc_fake_hsa_add_agent(1, node, "gfx1030");
  rdc_fake_hsa_add_pool(agent, HSA_AMD_SEGMENT_GLOBAL,
                        HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, 64 * 1024, 64 * 1024, 1,
                        4096, 4096, 1);
  return agent;
}

// What a GPU whose kernel file is not installed says: the file and the agent it is not
// installed for, and no errno, which says nothing to the user.
void expect_missing_kernel(const char* text) {
  EXPECT_NE(std::string(text).find("No binary_search_kernels.hsaco is installed for"),
            std::string::npos)
      << text;
}

}  // namespace

// --- the lifetimes of what a check makes ------------------------------------------------------

// A kernel that is not installed is that GPU's skip, and nothing of that check is left alive:
// under F22 it has made nothing by then, as the case after this one states, so what this one
// says is that no descriptor and no reference is kept and that the next run is normal (F: F9
// makes its queue, its signal and its buffers first and leaves them alive, so its hsa_shut_down()
// latches ROCr and the next run's hsa_init() fails).
TEST(RocrLifetimes, AMissingKernelReleasesWhatItMadeInOrder) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    const size_t descriptors = open_descriptors();

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP);
    expect_missing_kernel(result->gpu_results[0].gpu_result.msg);

    expect_everything_released();
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
    EXPECT_EQ(rdc_fake_hsa_unbalanced_shutdowns(), 0u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);
    EXPECT_EQ(open_descriptors(), descriptors);

    // Nothing is left alive, so the next run loads ROCr again and checks the GPU as this one did.
    ResultPtr second = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, second.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(second->status, RDC_DIAG_RESULT_SKIP);
    EXPECT_EQ(rdc_fake_hsa_load_count(), 2u);
    EXPECT_EQ(rdc_fake_hsa_failed_load_count(), 0u);
  });
}

// A check whose kernel is not installed makes nothing at all: it finds and loads its kernel
// before it creates a queue, a signal or a buffer, so a GPU without that kernel is a skip that
// cost no queue, and so no scratch ROCr reserves at one (M: F9 makes all three first; T4's
// mutation puts the load back after them).
TEST(RocrLifetimes, AMissingKernelMakesNothingAtAll) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();

    const OneGpu gpu = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_SKIP) << gpu.text;
    EXPECT_NE(gpu.text.find("No binary_search_kernels.hsaco is installed for gfx1030."),
              std::string::npos)
        << gpu.text;

    EXPECT_EQ(rdc_fake_hsa_signal_count(), 0u);
    EXPECT_EQ(rdc_fake_hsa_queue_count(), 0u);
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);
    EXPECT_EQ(rdc_fake_hsa_executable_count(), 0u);
    EXPECT_EQ(rdc_fake_hsa_reader_count(), 0u);
  });
}

// A failure at each step of a check, the first HSA call after the test object's hsa_init() among
// them, and an exception of another kind: everything made by then is released, no descriptor is
// kept, and the next run is normal (F: on F9 whatever was made stays alive, so its shutdown
// latches ROCr and every later hsa_init() fails).
TEST(RocrLifetimes, AThrowAtEachStepReleasesWhatWasMade) {
  RDC_ROCR_CASE({
    const char* const steps[] = {"hsa_iterate_agents",
                                 "hsa_signal_create",
                                 "hsa_queue_create",
                                 "hsa_amd_agent_iterate_memory_pools",
                                 "hsa_amd_memory_pool_allocate",
                                 "hsa_amd_agents_allow_access"};
    for (const char* step : steps) {
      for (int thrown = 0; thrown < 2; thrown++) {
        rdc_fake_hsa_reset();
        write_kernel_files();
        const size_t descriptors = open_descriptors();
        if (thrown == 0) {
          rdc_fake_hsa_fail(step, 0, HSA_STATUS_ERROR_OUT_OF_RESOURCES);
        } else {
          rdc_fake_hsa_throw_int(step, 0, 11);
        }

        uint32_t ids[1] = {0};
        ResultPtr result = new_result();
        ASSERT_EQ(rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(),
                                         nullptr),
                  RDC_ST_OK)
            << step;
        EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL) << step;

        expect_everything_released();
        EXPECT_EQ(rdc_fake_hsa_live_references(), 0u) << step;
        EXPECT_EQ(rdc_fake_hsa_unbalanced_shutdowns(), 0u) << step;
        EXPECT_EQ(rdc_fake_hsa_latched(), 0) << step;
        EXPECT_EQ(open_descriptors(), descriptors) << step;

        // The last step comes after the check made every kind of object, so the order of the
        // log is F3's: the queue, the signal, the executable and its reader, the buffers last.
        if (std::string(step) == "hsa_amd_agents_allow_access") {
          const std::vector<std::string> log = destroy_log();
          const size_t shutdown = first_shutdown(log);
          ASSERT_GT(shutdown, 4u) << joined(log);
          EXPECT_EQ(log[0], "queue 0") << joined(log);
          EXPECT_EQ(log[1], "signal 0") << joined(log);
          EXPECT_EQ(log[2], "executable 0") << joined(log);
          EXPECT_EQ(log[3], "reader 0") << joined(log);
          for (size_t i = 4; i < shutdown; i++) {
            EXPECT_EQ(log[i].compare(0, 5, "block"), 0) << joined(log);
          }
        }

        rdc_fake_hsa_clear_failures();
        ResultPtr second = new_result();
        ASSERT_EQ(rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, second.get(),
                                         nullptr),
                  RDC_ST_OK)
            << step;
        EXPECT_NE(second->status, RDC_DIAG_RESULT_FAIL) << step;
        EXPECT_EQ(rdc_fake_hsa_failed_load_count(), 0u) << step;
      }
    }
  });
}

// A check loads ROCr once, whatever the number of GPUs it looks at and of test objects it makes
// (F: F9 loads and unloads ROCr per test object).
TEST(RocrLifetimes, TheRunLoadsRocrOnceForTheCheck) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    add_third_gpu();
    uint32_t ids[3] = {0, 1, 2};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 3, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(result->per_gpu_result_count, 3u);
    EXPECT_EQ(rdc_fake_hsa_load_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);

    // A group of which only one GPU can be mapped costs the same one load.
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail_node_id(1, AMDSMI_STATUS_NOT_FOUND);
    uint32_t mixed[2] = {0, 1};
    ResultPtr second = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, mixed, 2, nullptr, 0, second.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(rdc_fake_hsa_load_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
  });
}

// A reference another module of the process holds, as the profiler module's, costs the checks no
// load of their own and is still there after them, with nothing of theirs left alive (F: on F9
// each run leaves its queue and its blocks alive, a leak per run).
TEST(RocrLifetimes, AReferenceHeldAcrossTwoRunsCostsNoLoad) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    ASSERT_EQ(rdc_fake_hsa_load_count(), 1u);

    uint32_t ids[2] = {0, 1};
    for (int run = 0; run < 2; run++) {
      for (rdc_diag_test_cases_t check : {RDC_DIAG_COMPUTE_QUEUE, RDC_DIAG_SYS_MEM_CHECK}) {
        ResultPtr result = new_result();
        ASSERT_EQ(rdc_diag_test_case_run(check, ids, 2, nullptr, 0, result.get(), nullptr),
                  RDC_ST_OK)
            << "run " << run;
        EXPECT_EQ(result->per_gpu_result_count, 2u) << "run " << run;
      }
      // No load of their own, nothing left alive, and this case's reference still held.
      EXPECT_EQ(rdc_fake_hsa_load_count(), 1u) << "run " << run;
      EXPECT_EQ(rdc_fake_hsa_live_references(), 1u) << "run " << run;
      EXPECT_EQ(rdc_fake_hsa_latched(), 0) << "run " << run;
      for (uint32_t queue = 0; queue < rdc_fake_hsa_queue_count(); queue++) {
        EXPECT_EQ(rdc_fake_hsa_queue_destroyed(queue), 1) << "run " << run << " queue " << queue;
      }
      EXPECT_EQ(rdc_fake_hsa_live_blocks(), 0u) << "run " << run;
    }
  });
}

// The run's own hsa_init() failing is every GPU's failure, with no second try and no other HSA
// call, and the next run tries again (F: F9 calls hsa_init() per test object, writes no per-GPU
// result and reports another text).
TEST(RocrLifetimes, AFailedRunWideInitFailsEveryGpuWithoutRetrying) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    add_third_gpu();
    rdc_fake_hsa_fail("hsa_init", 1, HSA_STATUS_ERROR_OUT_OF_RESOURCES);

    uint32_t ids[3] = {0, 1, 2};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 3, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 3u);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    for (uint32_t row = 0; row < 3; row++) {
      const std::string text = result->gpu_results[row].gpu_result.msg;
      EXPECT_NE(text.find("hsa_init"), std::string::npos) << text;
      EXPECT_NE(text.find("4104"), std::string::npos) << text;
    }
    // One call, and no other than the text of the status each GPU's sentence names.
    EXPECT_EQ(rdc_fake_hsa_call_count("hsa_init"), 1u);
    for (const std::string& call : call_log()) {
      EXPECT_TRUE(call == "hsa_init" || call == "hsa_status_string") << call;
    }
    EXPECT_EQ(rdc_fake_hsa_failed_load_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
    EXPECT_EQ(rdc_fake_hsa_queue_count(), 0u);
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);

    rdc_fake_hsa_clear_failures();
    ResultPtr second = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 3, nullptr, 0, second.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(second->status, RDC_DIAG_RESULT_SKIP);
    EXPECT_EQ(rdc_fake_hsa_load_count(), 1u);
  });
}

// A test object's own hsa_init() failing is that GPU's failure, it makes nothing, and the GPUs
// after it are checked as usual (F: F9 ignores that status and goes on to call HSA without a
// reference, and its destructor releases one it never took).
TEST(RocrLifetimes, AFailedTestObjectInitIsThatGpusFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    add_third_gpu();
    // The run's own reference is the first call; the first GPU's test object makes the second.
    rdc_fake_hsa_fail("hsa_init", 2, HSA_STATUS_ERROR_OUT_OF_RESOURCES);

    uint32_t ids[3] = {0, 1, 2};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 3, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 3u);
    const std::string failed = result->gpu_results[0].gpu_result.msg;
    EXPECT_NE(failed.find("hsa_init"), std::string::npos) << failed;
    EXPECT_NE(failed.find("4104"), std::string::npos) << failed;
    EXPECT_EQ(text_of(result->gpu_results[1].gpu_result.msg), "A kernel ran on a compute queue.");
    EXPECT_EQ(text_of(result->gpu_results[2].gpu_result.msg), "A kernel ran on a compute queue.");

    // The first GPU made nothing; the other two were checked on their own agents.
    EXPECT_EQ(rdc_fake_hsa_call_count("hsa_init"), 4u);
    ASSERT_EQ(rdc_fake_hsa_queue_count(), 2u);
    EXPECT_EQ(rdc_fake_hsa_queue_agent(0), 2u);
    EXPECT_EQ(rdc_fake_hsa_queue_agent(1), 3u);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
    EXPECT_EQ(rdc_fake_hsa_unbalanced_shutdowns(), 0u);
    expect_everything_released();
  });
}

// A clean-up call that fails is that GPU's failure, with the call and its status, the rest is
// released all the same and the process goes on (F: on F9 a failed clean-up throws from
// TestBase's destructor, which ends the process).
TEST(RocrLifetimes, AFailedCleanUpCallIsThatGpusFailureAndTheRestIsReleased) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_fail("hsa_amd_memory_pool_free", 1, HSA_STATUS_ERROR_INVALID_ALLOCATION);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    const std::string text = result->gpu_results[0].gpu_result.msg;
    EXPECT_NE(text.find("hsa_amd_memory_pool_free"), std::string::npos) << text;
    EXPECT_NE(text.find("4099"), std::string::npos) << text;

    // The queue, the signal and the blocks the free did not refuse are released all the same.
    const std::vector<std::string> log = destroy_log();
    EXPECT_EQ(released(log, "queue"), rdc_fake_hsa_queue_count()) << joined(log);
    EXPECT_EQ(released(log, "signal"), rdc_fake_hsa_signal_count() - rdc_fake_hsa_queue_count())
        << joined(log);
    EXPECT_EQ(rdc_fake_hsa_live_blocks(), 1u) << joined(log);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);
  });
}

// A test object's hsa_shut_down() failing is that GPU's failure, that GPU keeps its result and
// the GPU after it is checked as usual (F: on F9 that failure throws from a destructor).
TEST(RocrLifetimes, AFailedShutdownIsThatGpusFailureAndTheNextGpuIsChecked) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    add_third_gpu();
    // This check makes one test object per GPU: the second GPU's is the second shutdown.
    rdc_fake_hsa_fail("hsa_shut_down", 2, HSA_STATUS_ERROR_OUT_OF_RESOURCES);

    uint32_t ids[3] = {0, 1, 2};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 3, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 3u);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    const std::string text = result->gpu_results[1].gpu_result.msg;
    EXPECT_NE(text.find("hsa_shut_down"), std::string::npos) << text;
    EXPECT_NE(text.find("4104"), std::string::npos) << text;
    expect_missing_kernel(result->gpu_results[2].gpu_result.msg);
    EXPECT_EQ(rdc_fake_hsa_unbalanced_shutdowns(), 0u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);
    // The reference that shutdown did not release is the one left.
    EXPECT_EQ(rdc_fake_hsa_live_references(), 1u);
  });
}

// A queue that cannot be destroyed abandons the rest: nothing the work on it can touch is freed
// or destroyed, that test object keeps its reference to ROCr and no shutdown of its own takes
// ROCr's count to zero, so ROCr is not latched (F: on F9 the shutdown with that queue alive makes
// every later hsa_init() fail). The GPU is recorded, as it is at a deadline, so a later run
// reports it and skips the others without an HSA call of its own.
TEST(RocrLifetimes, AFailedQueueDestroyAbandonsTheRest) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_fail("hsa_queue_destroy", 0, HSA_STATUS_ERROR_INVALID_QUEUE);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    const std::string text = result->gpu_results[0].gpu_result.msg;
    EXPECT_NE(text.find("hsa_queue_destroy"), std::string::npos) << text;
    EXPECT_NE(text.find("4103"), std::string::npos) << text;

    // Nothing the work on that queue could still touch is released: its blocks and the signal
    // it dispatched with are kept, where the memory copy's own signal was destroyed long before
    // the queue's destroy failed.
    const std::vector<std::string> log = destroy_log();
    EXPECT_EQ(released(log, "block"), 0u) << joined(log);
    EXPECT_EQ(released(log, "signal 0"), 0u) << joined(log);
    EXPECT_EQ(rdc_fake_hsa_live_blocks(), rdc_fake_hsa_alloc_count()) << joined(log);
    EXPECT_EQ(rdc_fake_hsa_queue_destroyed(0), 0);
    // Only the run's own reference is given back, so ROCr's count never reaches zero.
    EXPECT_EQ(rdc_fake_hsa_call_count("hsa_shut_down"), 1u);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 1u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);

    // The record disables the checks: that GPU is a failure in a later run, the other is
    // skipped, and the run makes no call the fake sees.
    const uint32_t calls = rdc_fake_hsa_call_log_size();
    uint32_t later[2] = {0, 1};
    ResultPtr second = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, later, 2, nullptr, 0, second.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(second->per_gpu_result_count, 2u);
    EXPECT_NE(std::string(second->gpu_results[0].gpu_result.msg)
                  .find("An earlier check could not stop its work on this GPU"),
              std::string::npos)
        << second->gpu_results[0].gpu_result.msg;
    EXPECT_NE(std::string(second->gpu_results[1].gpu_result.msg).find("on GPU 0;"),
              std::string::npos)
        << second->gpu_results[1].gpu_result.msg;
    EXPECT_EQ(rdc_fake_hsa_call_log_size(), calls);
  });
}

// The run's own hsa_shut_down(), the last of a check, failing is logged and nothing else: every
// GPU keeps its result and the check returns as it would have (M: T4's mutation makes that
// destructor throw, which would end the process).
TEST(RocrLifetimes, TheRunWideShutdownFailureIsLoggedOnly) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    // This check makes two test objects per GPU, so the run's own shutdown is the third of a
    // group of one GPU.
    rdc_fake_hsa_fail("hsa_shut_down", 3, HSA_STATUS_ERROR_OUT_OF_RESOURCES);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 1u);
    EXPECT_EQ(result->gpu_results[0].gpu_index, 0u);
    EXPECT_NE(result->status, RDC_DIAG_RESULT_FAIL);
    EXPECT_EQ(rdc_fake_hsa_call_count("hsa_shut_down"), 3u);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 1u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);
  });
}

// A check leaves HSA_ENABLE_INTERRUPT as it found it and seeds no random sequence of the process
// (F: F9 writes that variable after every test object and calls srand()).
TEST(RocrLifetimes, TheInterruptFlagAndTheRandomSequenceAreLeftAlone) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(::setenv("HSA_ENABLE_INTERRUPT", "1", 1), 0);
    ::srand(1);
    const int expected = ::rand();
    ::srand(1);

    uint32_t ids[1] = {0};
    for (rdc_diag_test_cases_t check : {RDC_DIAG_COMPUTE_QUEUE, RDC_DIAG_SYS_MEM_CHECK}) {
      ResultPtr result = new_result();
      ASSERT_EQ(rdc_diag_test_case_run(check, ids, 1, nullptr, 0, result.get(), nullptr),
                RDC_ST_OK);
    }

    const char* const value = ::getenv("HSA_ENABLE_INTERRUPT");
    ASSERT_NE(value, nullptr);
    EXPECT_STREQ(value, "1");
    EXPECT_EQ(::rand(), expected);
  });
}

// A status ROCr has no string for is reported as a number, and the check goes on (F: F9 builds a
// std::string from the null pointer hsa_status_string left, which ends the process).
TEST(RocrLifetimes, AStatusWithNoTextIsStillReported) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_fail("hsa_signal_create", 0, kStatusWithNoString);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    const std::string text = result->gpu_results[0].gpu_result.msg;
    EXPECT_NE(text.find(std::to_string(kStatusWithNoString)), std::string::npos) << text;
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
  });
}

// --- the checks' callbacks ---------------------------------------------------------------------

// A host that lists no CPU agent is that GPU's failure in both checks (F: F9 reads cpus[0] of an
// empty vector).
TEST(RocrCallbacks, NoCpuAgentIsThatGpusFailure) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : {RDC_DIAG_COMPUTE_QUEUE, RDC_DIAG_SYS_MEM_CHECK}) {
      rdc_fake_hsa_reset();
      rdc_fake_hsa_clear_agents();
      const uint32_t gpu = rdc_fake_hsa_add_agent(1, 1, "gfx1030");
      rdc_fake_hsa_add_pool(gpu, HSA_AMD_SEGMENT_GLOBAL,
                            HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, 64 * 1024, 64 * 1024, 1,
                            4096, 4096, 1);

      uint32_t ids[1] = {0};
      ResultPtr result = new_result();
      ASSERT_EQ(rdc_diag_test_case_run(check, ids, 1, nullptr, 0, result.get(), nullptr),
                RDC_ST_OK);
      ASSERT_EQ(result->per_gpu_result_count, 1u);
      EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL) << check;
      const std::string text = result->gpu_results[0].gpu_result.msg;
      EXPECT_NE(text.find("No CPU agent."), std::string::npos) << text;
      expect_everything_released();
    }
  });
}

// An HSA call that fails inside an agent iteration is that GPU's failure, with the call's status
// in its text, and the check allocates nothing (F: F9's callback throws through ROCr, which turns
// it into an error of its own, so the reason is lost).
TEST(RocrCallbacks, AFailureInsideAnAgentIterationIsThatGpusFailure) {
  RDC_ROCR_CASE({
    dirty_stack();
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail_agent_info(UINT32_MAX, HSA_AGENT_INFO_DEVICE, HSA_STATUS_ERROR_INVALID_AGENT);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    const std::string text = result->gpu_results[0].gpu_result.msg;
    EXPECT_NE(text.find("4100"), std::string::npos) << text;
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);
    EXPECT_EQ(rdc_fake_hsa_queue_count(), 0u);
    expect_everything_released();
  });
}

// The same inside a pool iteration (F: F9's pool callbacks throw too, and FindPools turns every
// status but INFO_BREAK into HSA_STATUS_ERROR, so the reason is lost twice).
TEST(RocrCallbacks, AFailureInsideAPoolIterationIsThatGpusFailure) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : {RDC_DIAG_COMPUTE_QUEUE, RDC_DIAG_SYS_MEM_CHECK}) {
      dirty_stack();
      rdc_fake_hsa_reset();
      write_kernel_files();
      rdc_fake_hsa_fail_pool_info(UINT32_MAX, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                  HSA_STATUS_ERROR_INVALID_ARGUMENT);
      // The System memory check's own read of each pool fails the same way, so that neither
      // check allocates from a pool it could not read.
      rdc_fake_hsa_fail_pool_info(UINT32_MAX, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                  HSA_STATUS_ERROR_INVALID_ARGUMENT);

      uint32_t ids[1] = {0};
      ResultPtr result = new_result();
      ASSERT_EQ(rdc_diag_test_case_run(check, ids, 1, nullptr, 0, result.get(), nullptr),
                RDC_ST_OK);
      EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL) << check;
      const std::string text = result->gpu_results[0].gpu_result.msg;
      EXPECT_NE(text.find("4097"), std::string::npos) << text;
      EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u) << check;
      EXPECT_EQ(rdc_fake_hsa_foreign_pools(), 0u) << check;
      expect_everything_released();
    }
  });
}

// An iteration that ends without a match is that GPU's skip, and one that fails is its failure,
// with the call and the status (F: F9 reports both as "Fail to find pools.").
TEST(RocrCallbacks, NoPoolIsASkipAndAFailingIterationIsAFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_clear_pools(kCpuAgent);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP);
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);
    expect_everything_released();

    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_fail("hsa_amd_agent_iterate_memory_pools", 0, HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    ResultPtr second = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, second.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(second->status, RDC_DIAG_RESULT_FAIL);
    const std::string text = second->gpu_results[0].gpu_result.msg;
    EXPECT_NE(text.find("hsa_amd_agent_iterate_memory_pools"), std::string::npos) << text;
    EXPECT_NE(text.find("4104"), std::string::npos) << text;
    expect_everything_released();
  });
}

// A pool that does not allow allocation is not chosen: a sub-test that keeps the last matching
// pool takes the one before it (F: F9 takes the last match and fails at its first allocation;
// M: T4's mutations drop that rule from GetGlobalMemoryPool and from GetKernArgMemoryPool).
TEST(RocrCallbacks, ThePoolsThatDoNotAllowAllocationAreNotChosen) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    // The kernel files are written so that the sub-test that reads and writes GPU memory,
    // which is the one that chooses a global pool and a kernel argument pool, reaches its
    // choice: without them it skips at its load, which comes before its allocations.
    write_kernel_files();
    const uint32_t gpu_agent = 1;
    const uint32_t gpu_pool = rdc_fake_hsa_pool_of_agent(gpu_agent, 0);
    const uint32_t cpu_global = rdc_fake_hsa_pool_of_agent(kCpuAgent, 0);
    const uint32_t cpu_kernarg = rdc_fake_hsa_pool_of_agent(kCpuAgent, 1);
    const uint32_t cpu_coarse = rdc_fake_hsa_pool_of_agent(kCpuAgent, 2);

    // One more pool of each kind, after the ones above, allocating from none of them.
    const uint32_t gpu_last = rdc_fake_hsa_add_pool(gpu_agent, HSA_AMD_SEGMENT_GLOBAL,
                                                    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                                                    64 * 1024, 64 * 1024, 0, 4096, 4096, 1);
    const uint32_t cpu_last = rdc_fake_hsa_add_pool(kCpuAgent, HSA_AMD_SEGMENT_GLOBAL,
                                                    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                                                    64 * 1024, 64 * 1024, 0, 4096, 4096, 1);
    const uint32_t kernarg_last = rdc_fake_hsa_add_pool(
        kCpuAgent, HSA_AMD_SEGMENT_GLOBAL, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT, 64 * 1024,
        64 * 1024, 0, 4096, 4096, 1);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_NE(result->status, RDC_DIAG_RESULT_FAIL);

    ASSERT_GT(rdc_fake_hsa_alloc_count(), 0u);
    for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
      const uint32_t pool = rdc_fake_hsa_alloc_pool(block);
      EXPECT_NE(pool, gpu_last) << "block " << block;
      EXPECT_NE(pool, cpu_last) << "block " << block;
      EXPECT_NE(pool, kernarg_last) << "block " << block;
      EXPECT_TRUE(pool == gpu_pool || pool == cpu_global || pool == cpu_kernarg ||
                  pool == cpu_coarse)
          << "block " << block << " came from pool " << pool;
    }
    expect_everything_released();
  });
}

// The same where a check stops at the first pool that matches: the first that allows allocation
// is the one it takes (F: F9 takes the first match and fails at its first allocation).
TEST(RocrCallbacks, TheFirstPoolThatAllowsAllocationIsChosen) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_clear_pools(kCpuAgent);
    const uint32_t refused_standard = rdc_fake_hsa_add_pool(
        kCpuAgent, HSA_AMD_SEGMENT_GLOBAL, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
        64 * 1024, 64 * 1024, 0, 4096, 4096, 1);
    const uint32_t standard = rdc_fake_hsa_add_pool(kCpuAgent, HSA_AMD_SEGMENT_GLOBAL,
                                                    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                                                    64 * 1024, 64 * 1024, 1, 4096, 4096, 1);
    const uint32_t refused_kernarg = rdc_fake_hsa_add_pool(
        kCpuAgent, HSA_AMD_SEGMENT_GLOBAL, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT, 64 * 1024,
        64 * 1024, 0, 4096, 4096, 1);
    rdc_fake_hsa_add_pool(kCpuAgent, HSA_AMD_SEGMENT_GLOBAL,
                          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT, 64 * 1024, 64 * 1024, 1,
                          4096, 4096, 1);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_NE(result->status, RDC_DIAG_RESULT_FAIL);

    ASSERT_GT(rdc_fake_hsa_alloc_count(), 0u);
    for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
      const uint32_t pool = rdc_fake_hsa_alloc_pool(block);
      EXPECT_NE(pool, refused_standard) << "block " << block;
      EXPECT_NE(pool, refused_kernarg) << "block " << block;
    }
    EXPECT_EQ(rdc_fake_hsa_alloc_pool(0), standard);
    expect_everything_released();
  });
}

namespace {

#if RDC_ROCR_TEST_HOOKS
// One GPU's run with the module's debug lines captured, which is what says where it looked for
// its kernel file. The logger reads its level from the environment when the first line of this
// process is written, which is inside the run below.
std::string run_at_debug_level(rdc_diag_test_cases_t check, OneGpu* answer) {
  EXPECT_EQ(::setenv("RDC_LOG", "DEBUG", 1), 0);
  ::testing::internal::CaptureStdout();
  const OneGpu gpu = run_one_gpu(check);
  const std::string log = ::testing::internal::GetCapturedStdout();
  if (answer != nullptr) {
    *answer = gpu;
  }
  return log;
}

// The kernel files the module looked at, in the order of the debug lines that name them.
std::vector<std::string> candidates_in(const std::string& log) {
  const std::string marker = "Kernel file candidate ";
  std::vector<std::string> candidates;
  size_t at = log.find(marker);
  while (at != std::string::npos) {
    const size_t start = at + marker.size();
    const size_t end = log.find('\n', start);
    candidates.push_back(log.substr(start, end - start));
    at = log.find(marker, start);
  }
  return candidates;
}
#endif

// Both checks, so that a case states a rule both of them follow.
const rdc_diag_test_cases_t kBothChecks[] = {RDC_DIAG_COMPUTE_QUEUE, RDC_DIAG_SYS_MEM_CHECK};

// The kernel symbol the search check loads, which the fake keys a kernel's reported sizes by.
constexpr char kSearchKernel[] = "binarySearch.kd";

// The block the fake issued at that address, or its count when it issued none there.
uint32_t block_of(uint64_t address) {
  for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
    if (reinterpret_cast<uint64_t>(rdc_fake_hsa_alloc_address(block)) == address) {
      return block;
    }
  }
  return rdc_fake_hsa_alloc_count();
}

// The one block of that many bytes, so that a case reads what a check left in it: a block stays
// mapped after it is freed.
const uint32_t* block_of_size(uint64_t bytes) {
  const uint32_t* found = nullptr;
  for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
    if (rdc_fake_hsa_alloc_size(block) == bytes) {
      EXPECT_EQ(found, nullptr) << "more than one block of " << bytes << " bytes";
      found = static_cast<const uint32_t*>(rdc_fake_hsa_alloc_address(block));
    }
  }
  return found;
}

// The pool the CPU-access sub-test's 8 KiB block came from, or UINT32_MAX when it made none.
uint32_t cpu_access_pool() {
  for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
    if (rdc_fake_hsa_alloc_size(block) == 8 * 1024) {
      return rdc_fake_hsa_alloc_pool(block);
    }
  }
  return UINT32_MAX;
}

// 1 MB of 0xA5 on the stack is a control only where a frame of the depth the check runs at reads
// it back: this probe is not instrumented either, as ASan's redzones would leave it clean.
__attribute__((noinline, no_sanitize_address)) bool stack_dirt_is_visible() {
  unsigned char probe[4096];
  // The probe is read as the frame was found, which is the point of it. The pointer is hidden
  // from the compiler, so that it reads the memory rather than reason about a variable it
  // knows nothing was written to.
  unsigned char* bytes = probe;
  __asm__ volatile("" : "+r"(bytes));
  size_t dirty = 0;
  for (size_t i = 0; i < sizeof(probe); i++) {
    if (bytes[i] == 0xA5) {
      dirty++;
    }
  }
  return dirty > sizeof(probe) / 2;
}

}  // namespace

// --- the waits and the errors of a queue ---------------------------------------------------------

namespace {

// The error a case raises on a check's queue, and the number ROCr gives it, which the failure of
// that GPU names.
constexpr hsa_status_t kQueueError = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
constexpr char kQueueErrorNumber[] = "4104";

// Where each check allocates its kernel arguments, counted among the allocations it makes: it is
// the last of them, and it comes after hsa_queue_create, as both checks load their kernel before
// they make a queue. A case raises a queue error there, which the check must see at the doorbell.
constexpr uint32_t kComputeQueueKernArgAllocation = 5;
constexpr uint32_t kSysMemKernArgAllocation = 7;

#if RDC_ROCR_TEST_HOOKS
// Whether the address lies outside the stack of the calling thread, which is where a queue's
// status block must be: ROCr may write into it long after the frame that made the queue is gone.
bool outside_this_stack(const void* address) {
  pthread_attr_t attributes;
  EXPECT_EQ(::pthread_getattr_np(::pthread_self(), &attributes), 0);
  void* base = nullptr;
  size_t size = 0;
  EXPECT_EQ(::pthread_attr_getstack(&attributes, &base, &size), 0);
  EXPECT_EQ(::pthread_attr_destroy(&attributes), 0);
  const char* const at = static_cast<const char*>(address);
  return at < static_cast<const char*>(base) || at >= static_cast<const char*>(base) + size;
}
#endif

// Where that call first appears in the fake's log, or the log's size when it is not in it.
size_t first_call(const std::vector<std::string>& calls, const std::string& name) {
  for (size_t i = 0; i < calls.size(); i++) {
    if (calls[i] == name) {
      return i;
    }
  }
  return calls.size();
}

// Raises an error on that queue from a thread of the fake's and waits until the module's
// callback has stored it in the queue's status block, so that a check the case holds at a gate
// sees the error when it is let go.
void raise_queue_error_and_wait(uint32_t queue, hsa_status_t status) {
  const std::atomic<hsa_status_t>* const block =
      static_cast<const std::atomic<hsa_status_t>*>(rdc_fake_hsa_queue_error_data(queue));
  ASSERT_NE(block, nullptr);
  rdc_fake_hsa_raise_queue_error(queue, status);
  for (int spin = 0; spin < 5000 && block->load() != status; spin++) {
    struct timespec pause = {0, 1000 * 1000};
    ::nanosleep(&pause, nullptr);
  }
  ASSERT_EQ(block->load(), status);
}

// A run of a check on a thread of its own, so that two runs overlap and so that a gate can hold
// one run while the case does something the run must see. The thread names itself to the fake.
class Running {
 public:
  Running(uint64_t caller, rdc_diag_test_cases_t check, const std::vector<uint32_t>& ids)
      : ids_(ids), result_(new_result()) {
    thread_ = std::thread([this, caller, check]() {
      rdc_fake_hsa_set_caller(caller);
      status_ = rdc_diag_test_case_run(check, ids_.data(), static_cast<uint32_t>(ids_.size()),
                                       nullptr, 0, result_.get(), nullptr);
    });
  }

  void join() { thread_.join(); }
  rdc_status_t status() const { return status_; }
  const rdc_diag_test_result_t& result() const { return *result_; }
  std::string text(uint32_t row) const {
    return std::string(result_->gpu_results[row].gpu_result.msg);
  }

 private:
  std::vector<uint32_t> ids_;
  ResultPtr result_;
  rdc_status_t status_ = RDC_ST_UNKNOWN_ERROR;
  std::thread thread_;
};

// The tokens the runs of a case name themselves with.
constexpr uint64_t kFirstRun = 1;
constexpr uint64_t kSecondRun = 2;

}  // namespace

// ROCr reports an operation that failed by a negative value on its signal, which satisfies the
// wait: the value is that GPU's failure, everything the check made is released and no GPU is
// recorded, so the next run checks it as it would have (F: on F9 the throw at the copy leaves
// its signal and skips the clean-up, so the queue is alive at the shutdown and the next run's
// hsa_init() fails).
TEST(RocrWaits, ACopyWhoseSignalReportsAFailureIsThatGpusFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_set_copy_mode(RDC_FAKE_COPY_FAILED);

    const OneGpu answer = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(answer.status, RDC_DIAG_RESULT_FAIL);
    EXPECT_NE(answer.text.find("-1"), std::string::npos) << answer.text;
    EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u);
    expect_everything_released();
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);

    rdc_fake_hsa_set_copy_mode(RDC_FAKE_COPY_NORMAL);
    const OneGpu again = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(again.status, RDC_DIAG_RESULT_PASS) << again.text;
  });
}

// A dispatch's signal below zero, a value ROCr does not give a dispatch, ends its wait the same
// way (F: F9's loop ends only on 0, so it spins until the case's time limit kills the child).
TEST(RocrWaits, ADispatchSignalBelowZeroIsThatGpusFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_set_dispatch_signal_value(-1);

    const OneGpu answer = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(answer.status, RDC_DIAG_RESULT_FAIL);
    EXPECT_NE(answer.text.find("-1"), std::string::npos) << answer.text;
    expect_everything_released();
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);

    rdc_fake_hsa_set_dispatch_signal_value(0);
    const OneGpu again = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(again.status, RDC_DIAG_RESULT_PASS) << again.text;
  });
}

// An error ROCr raises on the queue while the check waits for a dispatch that never completes:
// the wait reads the queue's status block at its next hint, so the error is that GPU's failure
// within one hint, its queue is destroyed, no GPU is recorded, and the GPU after it is checked
// with a status block of its own (F: F9's queue has no error callback and its wait never ends,
// so the case's time limit kills the child).
TEST(RocrWaits, AQueueErrorDuringADispatchIsThatGpusFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    add_gpu_with_node(3);

    // The second GPU is the one whose kernel never completes, and the third is checked with the
    // kernel it has: each is held at the kernel it loads, which both checks do before they make
    // their queue.
    rdc_fake_hsa_gate("hsa_code_object_reader_create_from_file", 0, 2);
    rdc_fake_hsa_gate("hsa_executable_create_alt", 0, 3);

    uint32_t ids[3] = {0, 1, 2};
    ResultPtr result = new_result();
    rdc_status_t status = RDC_ST_UNKNOWN_ERROR;
    std::thread checking([&ids, &result, &status]() {
      status =
          rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 3, nullptr, 0, result.get(), nullptr);
    });

    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_code_object_reader_create_from_file", 1, 30000), 1);
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_NEVER);
    // The second GPU is held at the first wait it makes, which is the wait of its dispatch: the
    // error is raised there and waited for, so that the check reads it at the next hint of a
    // wait it is already in, rather than whenever a thread of the fake's happens to store it.
    rdc_fake_hsa_gate("hsa_signal_wait_scacquire", 0, 0);
    rdc_fake_hsa_gate_release("hsa_code_object_reader_create_from_file");

    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_signal_wait_scacquire", 1, 30000), 1);
    ASSERT_EQ(rdc_fake_hsa_queue_count(), 2u);
    raise_queue_error_and_wait(1, kQueueError);
    rdc_fake_hsa_gate_release("hsa_signal_wait_scacquire");

    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_executable_create_alt", 1, 30000), 1);
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_NORMAL);
    rdc_fake_hsa_gate_release("hsa_executable_create_alt");
    checking.join();

    EXPECT_EQ(status, RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 3u);
    const std::string first = result->gpu_results[0].gpu_result.msg;
    const std::string second = result->gpu_results[1].gpu_result.msg;
    const std::string third = result->gpu_results[2].gpu_result.msg;
    EXPECT_NE(first.find("The GPU read and wrote 4 KiB of the CPU's memory."), std::string::npos)
        << first;
    EXPECT_NE(second.find(kQueueErrorNumber), std::string::npos) << second;
    // The third GPU was checked, so no slot was claimed: a run that found one makes no HSA call.
    EXPECT_NE(third.find("The GPU read and wrote 4 KiB of the CPU's memory."), std::string::npos)
        << third;

    // The suspended queue is destroyed, and every queue was given a block of the module's
    // own. The block of a queue goes with it, so two of these addresses may be equal; that each
    // queue has a block of its own is shown where one of them is still alive, in
    // EveryQueueHasItsOwnStatusBlockOutsideTheStacks.
    ASSERT_EQ(rdc_fake_hsa_queue_count(), 3u);
    EXPECT_EQ(rdc_fake_hsa_queue_destroyed(1), 1);
    for (uint32_t queue = 0; queue < rdc_fake_hsa_queue_count(); queue++) {
      EXPECT_NE(rdc_fake_hsa_queue_error_data(queue), nullptr) << "queue " << queue;
    }
    expect_everything_released();
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
  });
}

// An error ROCr reported earlier suspended the queue, so nothing put on it would complete: the
// check reads the queue's status block before it makes a packet ready, so that GPU's failure
// names the status and no packet is dispatched (M: T4's mutation drops the check before the
// doorbell).
TEST(RocrWaits, AQueueErrorIsSeenBeforeTheDoorbell) {
  RDC_ROCR_CASE({
    struct Check {
      rdc_diag_test_cases_t check;
      uint32_t kernel_arguments;
    };
    const Check checks[] = {{RDC_DIAG_COMPUTE_QUEUE, kComputeQueueKernArgAllocation},
                            {RDC_DIAG_SYS_MEM_CHECK, kSysMemKernArgAllocation}};

    for (const Check& one : checks) {
      rdc_fake_hsa_reset();
      write_kernel_files();
      // The check is held at the allocation of its kernel arguments, which follows
      // hsa_queue_create in both checks, and the error reaches its queue while it is held.
      rdc_fake_hsa_gate("hsa_amd_memory_pool_allocate", kFirstRun, one.kernel_arguments);
      Running run(kFirstRun, one.check, {0});
      ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_amd_memory_pool_allocate", 1, 30000), 1);
      ASSERT_EQ(rdc_fake_hsa_queue_count(), 1u);
      raise_queue_error_and_wait(0, kQueueError);
      rdc_fake_hsa_gate_release("hsa_amd_memory_pool_allocate");
      run.join();

      EXPECT_EQ(run.status(), RDC_ST_OK);
      ASSERT_EQ(run.result().per_gpu_result_count, 1u);
      EXPECT_EQ(run.result().status, RDC_DIAG_RESULT_FAIL) << run.text(0);
      EXPECT_NE(run.text(0).find(kQueueErrorNumber), std::string::npos) << run.text(0);
      EXPECT_EQ(rdc_fake_hsa_alloc_count(), one.kernel_arguments);
      EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u);
      expect_everything_released();

      // No GPU was recorded: the next run checks it as it would have.
      const OneGpu again = run_one_gpu(one.check);
      EXPECT_EQ(again.status, RDC_DIAG_RESULT_PASS) << again.text;
    }
  });
}

// The memory copy's wait does not read the queue's status block: the copy runs on ROCr's own
// path, so an error raised while it is copying ends neither the copy nor its wait, and the check
// gives that GPU's failure before the doorbell, once the copy is done (M: T4's mutation makes
// the copy's wait read the block).
TEST(RocrWaits, AQueueErrorDuringTheMemoryCopyIsSeenAfterIt) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    // The copy completes inside the third wait on its signal, so that the wait it is waited
    // for in has passes of its own: a wait that read the queue's status block would throw at
    // the first of them, where this one waits the copy out.
    rdc_fake_hsa_set_completion_wait(3);
    // The check is held at the memory copy it makes, and the error reaches its queue there, so
    // that the copy and its wait run with the error already reported.
    rdc_fake_hsa_gate("hsa_amd_memory_async_copy", kFirstRun, 0);
    Running run(kFirstRun, RDC_DIAG_COMPUTE_QUEUE, {0});
    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_amd_memory_async_copy", 1, 30000), 1);
    ASSERT_EQ(rdc_fake_hsa_queue_count(), 1u);
    raise_queue_error_and_wait(0, kQueueError);
    rdc_fake_hsa_gate_release("hsa_amd_memory_async_copy");
    run.join();

    EXPECT_EQ(run.status(), RDC_ST_OK);
    EXPECT_EQ(run.result().status, RDC_DIAG_RESULT_FAIL) << run.text(0);
    EXPECT_NE(run.text(0).find(kQueueErrorNumber), std::string::npos) << run.text(0);
    // The copy ended on its own signal, with the error already in the queue's block, and only
    // then was it this GPU's failure, at the check before the doorbell.
    EXPECT_EQ(rdc_fake_hsa_copy_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u);
    // The copy's signal is the only signal the check waited on, it was waited on three
    // times, and it was satisfied: the copy completed, and only then was this GPU's
    // failure given.
    uint32_t waited_on = 0;
    uint32_t signals_waited_on = 0;
    for (uint32_t signal = 0; signal < rdc_fake_hsa_signal_count(); signal++) {
      if (rdc_fake_hsa_signal_waits(signal) != 0) {
        waited_on = signal;
        signals_waited_on++;
      }
    }
    ASSERT_EQ(signals_waited_on, 1u);
    EXPECT_EQ(rdc_fake_hsa_signal_waits(waited_on), 3u);
    EXPECT_EQ(rdc_fake_hsa_signal_value(waited_on), 0);
    EXPECT_EQ(rdc_fake_hsa_wait_count(), 3u);

    // The copy was made and waited for before anything of the check's was released.
    const std::vector<std::string> calls = call_log();
    const size_t copied = first_call(calls, "hsa_amd_memory_async_copy");
    EXPECT_LT(copied, first_call(calls, "hsa_signal_destroy"));
    EXPECT_LT(copied, first_call(calls, "hsa_amd_memory_pool_free"));
    expect_everything_released();
  });
}

// Each queue's error callback is given a block of the module's own, outside the stack of the
// thread that made the queue, and the block of a queue a check abandoned is never freed: an
// error ROCr reports after that run returned writes into memory that is still allocated, and
// claims no second slot (M: T4's two mutations of the block, kept in the test object and freed
// at an abandon, fail it, the first by the stack check and the second in the ASan leg).
TEST(RocrWaits, EveryQueueHasItsOwnStatusBlockOutsideTheStacks) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();

    EXPECT_EQ(run_one_gpu(RDC_DIAG_COMPUTE_QUEUE).status, RDC_DIAG_RESULT_PASS);
    EXPECT_EQ(run_one_gpu(RDC_DIAG_SYS_MEM_CHECK).status, RDC_DIAG_RESULT_PASS);
    ASSERT_EQ(rdc_fake_hsa_queue_count(), 2u);
    for (uint32_t queue = 0; queue < rdc_fake_hsa_queue_count(); queue++) {
      EXPECT_EQ(rdc_fake_hsa_queue_has_callback(queue), 1) << "queue " << queue;
      const void* const block = rdc_fake_hsa_queue_error_data(queue);
      ASSERT_NE(block, nullptr) << "queue " << queue;
      EXPECT_TRUE(outside_this_stack(block)) << "queue " << queue;
    }

    // The blocks of those two queues went with them, so whether their addresses differ says
    // nothing. Two runs at once show that each queue has a block of its own: the queue of the
    // run held at its doorbell keeps its block while the module takes one for the queue of the
    // other run, and two blocks that are alive together are at different addresses.
    rdc_fake_hsa_gate("hsa_signal_store_relaxed", 1, 0);
    Running held(1, RDC_DIAG_SYS_MEM_CHECK, {0});
    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_signal_store_relaxed", 1, 30000), 1);
    Running other(2, RDC_DIAG_SYS_MEM_CHECK, {1});
    other.join();
    rdc_fake_hsa_gate_release("hsa_signal_store_relaxed");
    held.join();
    EXPECT_EQ(held.status(), RDC_ST_OK);
    EXPECT_EQ(other.status(), RDC_ST_OK);
    EXPECT_EQ(held.result().status, RDC_DIAG_RESULT_PASS) << held.text(0);
    EXPECT_EQ(other.result().status, RDC_DIAG_RESULT_PASS) << other.text(0);
    ASSERT_EQ(rdc_fake_hsa_queue_count(), 4u);
    EXPECT_NE(rdc_fake_hsa_queue_error_data(2), rdc_fake_hsa_queue_error_data(3));

    // A check that could not stop its kernel keeps its queue and the block with it.
    EXPECT_EQ(rdc_rocr_test_set_wait_deadline_ms(kShortDeadlineMs), kModuleDeadlineMs);
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_NEVER);
    const OneGpu abandoned = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(abandoned.status, RDC_DIAG_RESULT_FAIL) << abandoned.text;
    ASSERT_EQ(rdc_fake_hsa_queue_count(), 5u);
    EXPECT_EQ(rdc_fake_hsa_queue_destroyed(4), 0);

    // The error the fake raises on it now reaches the module's callback, which writes into that
    // block: a block that had been freed would be an ASan report.
    const std::atomic<hsa_status_t>* const kept =
        static_cast<const std::atomic<hsa_status_t>*>(rdc_fake_hsa_queue_error_data(4));
    ASSERT_NE(kept, nullptr);
    EXPECT_EQ(kept->load(), HSA_STATUS_SUCCESS);
    raise_queue_error_and_wait(4, kQueueError);

    // That error claimed no slot: the record still names one GPU.
    uint32_t ids[2] = {0, 1};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 2, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 2u);
    EXPECT_NE(std::string(result->gpu_results[0].gpu_result.msg)
                  .find("An earlier check could not stop its work on this GPU"),
              std::string::npos)
        << result->gpu_results[0].gpu_result.msg;
    EXPECT_NE(std::string(result->gpu_results[1].gpu_result.msg).find("on GPU 0;"),
              std::string::npos)
        << result->gpu_results[1].gpu_result.msg;
  });
#else
  GTEST_SKIP() << "needs the deadline hook of RocrTestHooks.h";
#endif
}

// The frequency a wait's hint is counted in is read once per test object and before anything is
// dispatched or copied: a read that fails, and a frequency of 0, are that GPU's failure, with
// nothing dispatched, no copy started and everything the check made released before its
// hsa_shut_down(); a frequency below ten ticks a second still leaves a hint of one tick (M).
TEST(RocrWaits, AFailedTimestampFrequencyIsThatGpusFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_fail("hsa_system_get_info", 0, HSA_STATUS_ERROR_INVALID_ARGUMENT);

    const OneGpu failed = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(failed.status, RDC_DIAG_RESULT_FAIL) << failed.text;
    EXPECT_NE(failed.text.find("hsa_system_get_info"), std::string::npos) << failed.text;
    EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u);
    expect_everything_released();
    EXPECT_EQ(destroy_log().back(), "shut_down") << joined(destroy_log());

    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_set_timestamp_frequency(0);

    const OneGpu zero = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(zero.status, RDC_DIAG_RESULT_FAIL) << zero.text;
    EXPECT_NE(zero.text.find("frequency of 0"), std::string::npos) << zero.text;
    EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u);
    expect_everything_released();
    EXPECT_EQ(destroy_log().back(), "shut_down") << joined(destroy_log());

    // The Compute Queue check reads the hint before it makes the copy's signal and starts the
    // copy, so a frequency it cannot read is this GPU's failure with nothing in flight: a throw
    // while the copy ran would unwind over the buffer the copy writes into.
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_fail("hsa_system_get_info", 0, HSA_STATUS_ERROR_INVALID_ARGUMENT);

    const OneGpu copying = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(copying.status, RDC_DIAG_RESULT_FAIL) << copying.text;
    EXPECT_NE(copying.text.find("hsa_system_get_info"), std::string::npos) << copying.text;
    EXPECT_EQ(rdc_fake_hsa_call_count("hsa_amd_memory_async_copy"), 0u);
    EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u);
    expect_everything_released();

    // A frequency of fewer than ten ticks a second leaves a hint of one tick rather than none:
    // a hint of 0 is a wait that returns at once and spins to the deadline.
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_set_timestamp_frequency(5);

    const OneGpu slow = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(slow.status, RDC_DIAG_RESULT_PASS) << slow.text;
    ASSERT_EQ(rdc_fake_hsa_wait_hint_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_wait_hint(0), 1u);
  });
}

// --- the queues the checks create -------------------------------------------------------------

// Each check creates a queue of the packets it needs, with no private and no group segment
// reserved (F: the Compute Queue check asks for UINT32_MAX of each, which on an APU reserves
// 2048 bytes per lane for every scratch slot of every compute unit, and the read and write
// sub-test asks for the largest queue the agent has, 8 MiB of ring on ROCr).
TEST(RocrQueues, TheQueuesHoldThePacketsTheCheckNeedsAndNoScratch) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : kBothChecks) {
      rdc_fake_hsa_reset();
      write_kernel_files();

      const OneGpu gpu = run_one_gpu(check);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_PASS) << gpu.text;
      ASSERT_GT(rdc_fake_hsa_queue_count(), 0u);
      for (uint32_t queue = 0; queue < rdc_fake_hsa_queue_count(); queue++) {
        EXPECT_EQ(rdc_fake_hsa_queue_size(queue), 128u) << "queue " << queue;
        EXPECT_EQ(rdc_fake_hsa_queue_private_segment(queue), 0u) << "queue " << queue;
        EXPECT_EQ(rdc_fake_hsa_queue_group_segment(queue), 0u) << "queue " << queue;
      }
      expect_everything_released();
    }
  });
}

// An agent whose largest queue is ROCr's own minimum gets a queue of that size (F).
TEST(RocrQueues, AnAgentWhoseMaximumIs64GetsThatQueue) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : kBothChecks) {
      rdc_fake_hsa_reset();
      write_kernel_files();
      rdc_fake_hsa_set_agent_queue_max(1, 64);

      const OneGpu gpu = run_one_gpu(check);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_PASS) << gpu.text;
      ASSERT_GT(rdc_fake_hsa_queue_count(), 0u);
      for (uint32_t queue = 0; queue < rdc_fake_hsa_queue_count(); queue++) {
        EXPECT_EQ(rdc_fake_hsa_queue_size(queue), 64u) << "queue " << queue;
      }
    }
  });
}

// A failed query of that maximum is that GPU's failure, naming the call, with no queue of its
// own, while the GPUs before and after it are checked as they would have been (F: F9's Compute
// Queue check never asks, and its read and write sub-test throws and writes no per-GPU row).
TEST(RocrQueues, AFailedQueueMaximumQueryIsThatGpusFailure) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : kBothChecks) {
      rdc_fake_hsa_reset();
      add_third_gpu();
      write_kernel_files();
      const uint32_t second = 2;  // the agent the second GPU of this group is checked on
      rdc_fake_hsa_fail_agent_info(second, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                                   HSA_STATUS_ERROR_OUT_OF_RESOURCES);

      uint32_t ids[3] = {0, 1, 2};
      ResultPtr result = new_result();
      ASSERT_EQ(rdc_diag_test_case_run(check, ids, 3, nullptr, 0, result.get(), nullptr),
                RDC_ST_OK);
      ASSERT_EQ(result->per_gpu_result_count, 3u);
      EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);

      const std::string text = result->gpu_results[1].gpu_result.msg;
      EXPECT_NE(text.find("hsa_agent_get_info"), std::string::npos) << text;
      EXPECT_NE(text.find("4104"), std::string::npos) << text;
      for (uint32_t queue = 0; queue < rdc_fake_hsa_queue_count(); queue++) {
        EXPECT_NE(rdc_fake_hsa_queue_agent(queue), second) << "queue " << queue;
      }
      expect_everything_released();
    }
  });
}

// --- the packets the checks dispatch ----------------------------------------------------------

// Every dispatched packet carries the private and group segment sizes the kernel reports, so
// that a kernel rebuilt with scratch or LDS is dispatched with what it asks for: ROCr gives a
// queue scratch only for a dispatch that asks (F: the read and write sub-test dispatches 0).
TEST(RocrPackets, TheDispatchedPacketsCarryTheKernelsSegments) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : kBothChecks) {
      rdc_fake_hsa_reset();
      write_kernel_files();

      const OneGpu gpu = run_one_gpu(check);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_PASS) << gpu.text;
      ASSERT_EQ(rdc_fake_hsa_dispatch_count(), 1u);
      EXPECT_EQ(rdc_fake_hsa_dispatch_private_segment(0), 16u);
      EXPECT_EQ(rdc_fake_hsa_dispatch_group_segment(0), 64u);
    }
  });
}

// The fields the AQL protocol reserves reach the ring as the zeros it asks for, where ROCr
// leaves a new ring's packet contents uninitialized and the fake fills it with 0x5A: the 0xA5
// stack runs first, so that a staging packet left uncleared would carry dirt of its own (F).
TEST(RocrPackets, TheDispatchedPacketsReservedFieldsAreZero) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : kBothChecks) {
      dirty_stack();
      ASSERT_TRUE(stack_dirt_is_visible()) << "the stack the check runs on is not dirty";
      rdc_fake_hsa_reset();
      write_kernel_files();

      const OneGpu gpu = run_one_gpu(check);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_PASS) << gpu.text;
      ASSERT_EQ(rdc_fake_hsa_dispatch_count(), 1u);
      EXPECT_EQ(rdc_fake_hsa_dispatch_reserved0(0), 0u);
      EXPECT_EQ(rdc_fake_hsa_dispatch_reserved2(0), 0u);
    }
  });
}

// --- the data the search dispatches -----------------------------------------------------------

// The input the search dispatches is the same sequence on every run and holds the value to find,
// at input[54], and the run dispatches once: the kernel answers with the sub-division that holds
// it (F: F9's input is random, it writes 2048 bytes into a 16-byte block, and it dispatches a
// grid of 64 at an array sized for 2 sub-divisions).
TEST(RocrSearch, TheInputHoldsTheValueAndIsDispatchedOnce) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();

    const OneGpu gpu = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_PASS) << gpu.text;

    const uint32_t* input = block_of_size(512 * sizeof(uint32_t));
    ASSERT_NE(input, nullptr);
    EXPECT_EQ(input[54], 108u);
    for (uint32_t i = 0; i < 512; i++) {
      ASSERT_EQ(input[i], 2 * i) << "element " << i;
    }

    ASSERT_EQ(rdc_fake_hsa_dispatch_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_dispatch_grid(0), 64u);
    EXPECT_EQ(rdc_fake_hsa_grid_violations(), 0u) << rdc_fake_hsa_last_grid_violation();
    expect_everything_released();
  });
}

// A sub-division that is inside the input but does not hold the value is that GPU's failure
// (F: F9 logs the miss at debug level and passes).
TEST(RocrSearch, AnAnswerThatDoesNotHoldTheValueIsAFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_OUTPUT);
    rdc_fake_hsa_set_search_output(5, 1);

    const OneGpu gpu = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL);
    EXPECT_NE(gpu.text.find("The kernel's search did not find the value it was given."),
              std::string::npos)
        << gpu.text;
    EXPECT_EQ(rdc_fake_hsa_redzone_failures(), 0u);
    expect_everything_released();
  });
}

// An answer outside the run's sub-divisions, the sentinel a kernel that wrote nothing leaves
// among them, is that GPU's failure: the comparison comes before the multiply, which would wrap
// 0x20000000 back into the input (F).
TEST(RocrSearch, AnAnswerOutsideTheSubDivisionsIsAFailure) {
  RDC_ROCR_CASE({
    struct Answer {
      const char* name;
      int mode;
      uint32_t element0;
    };
    const Answer answers[] = {{"UINT32_MAX", RDC_FAKE_KERNEL_OUTPUT, UINT32_MAX},
                              {"0x20000000", RDC_FAKE_KERNEL_OUTPUT, 0x20000000},
                              {"nothing", RDC_FAKE_KERNEL_NOTHING, 0}};
    for (const Answer& answer : answers) {
      rdc_fake_hsa_reset();
      write_kernel_files();
      rdc_fake_hsa_set_kernel_mode(answer.mode);
      rdc_fake_hsa_set_search_output(answer.element0, 1);

      const OneGpu gpu = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << answer.name;
      EXPECT_NE(gpu.text.find("The kernel returned an index out of range."), std::string::npos)
          << answer.name << ": " << gpu.text;
      EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 1u) << answer.name;
      EXPECT_EQ(rdc_fake_hsa_redzone_failures(), 0u) << answer.name;
      expect_everything_released();
    }
  });
}

// --- the kernel arguments ---------------------------------------------------------------------

// Every byte of each argument segment the fake recorded is the image the kernel expects: the
// members at their offsets and zeros from there to the segment's end, with the 0xA5 stack and
// its control first, so that padding the check did not write would show (PLAN2 rule (b); F: F9
// copies a structure whose `pad` and trailing padding it never sets, into a segment it sizes
// from that structure rather than from the kernel).
TEST(RocrKernelArgs, EachSegmentIsTheImageTheKernelExpects) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : kBothChecks) {
      dirty_stack();
      ASSERT_TRUE(stack_dirt_is_visible()) << "the stack the check runs on is not dirty";
      rdc_fake_hsa_reset();
      write_kernel_files();

      const OneGpu gpu = run_one_gpu(check);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_PASS) << gpu.text;
      ASSERT_EQ(rdc_fake_hsa_dispatch_count(), 1u);
      ASSERT_EQ(rdc_fake_hsa_dispatch_kernarg_size(0), 80u);
      const unsigned char* segment = rdc_fake_hsa_dispatch_kernarg(0);
      ASSERT_NE(segment, nullptr);

      // Each pointer the kernel is given is a block the fake issued, and the search kernel's
      // third argument is the value it is to find.
      const size_t pointers = (check == RDC_DIAG_COMPUTE_QUEUE) ? 2 : 3;
      for (size_t i = 0; i < pointers; i++) {
        uint64_t address = 0;
        std::memcpy(&address, segment + (8 * i), sizeof(address));
        EXPECT_LT(block_of(address), rdc_fake_hsa_alloc_count()) << "argument " << i;
      }
      size_t written = 8 * pointers;
      if (check == RDC_DIAG_COMPUTE_QUEUE) {
        uint32_t find_me = 0;
        std::memcpy(&find_me, segment + written, sizeof(find_me));
        EXPECT_EQ(find_me, 108u);
        written += sizeof(find_me);
      }
      for (size_t i = written; i < 80; i++) {
        ASSERT_EQ(segment[i], 0u) << "byte " << i;
      }
    }
  });
}

// A kernel whose argument segment is longer than its arguments is dispatched, with the bytes
// past them zero (F: F9 allocates and dispatches what its own structure measures).
TEST(RocrKernelArgs, ALongerSegmentIsZeroedToItsEnd) {
  RDC_ROCR_CASE({
    dirty_stack();
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_set_kernel_info(kSearchKernel, 96, 16, 16, 64);

    const OneGpu gpu = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_PASS) << gpu.text;
    ASSERT_EQ(rdc_fake_hsa_dispatch_count(), 1u);
    ASSERT_EQ(rdc_fake_hsa_dispatch_kernarg_size(0), 96u);
    const unsigned char* segment = rdc_fake_hsa_dispatch_kernarg(0);
    ASSERT_NE(segment, nullptr);
    for (size_t i = 72; i < 96; i++) {
      ASSERT_EQ(segment[i], 0u) << "byte " << i;
    }
    EXPECT_EQ(rdc_fake_hsa_grid_violations(), 0u) << rdc_fake_hsa_last_grid_violation();
  });
}

// A kernel whose argument segment is shorter than its arguments is that GPU's failure, decided
// from the loaded symbol before any kernel argument block is allocated, so that nothing is
// dispatched, and the next run is normal (F: F9's check of that size is an assert the shipped
// builds compile out, so it dispatches a segment the kernel reads past).
TEST(RocrKernelArgs, ASegmentShorterThanTheArgumentsIsAFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_set_kernel_info(kSearchKernel, 64, 16, 16, 64);
    const uint32_t kernarg_pool = rdc_fake_hsa_pool_of_agent(kCpuAgent, kKernArgPool);

    const OneGpu gpu = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL);
    EXPECT_NE(gpu.text.find("The kernel's argument segment is 64 bytes, shorter than its "
                            "arguments."),
              std::string::npos)
        << gpu.text;
    EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u);
    for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
      EXPECT_NE(rdc_fake_hsa_alloc_pool(block), kernarg_pool) << "block " << block;
    }
    expect_everything_released();

    rdc_fake_hsa_reset();
    write_kernel_files();
    const OneGpu second = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(second.status, RDC_DIAG_RESULT_PASS) << second.text;
  });
}

// An argument segment alignment of 0 is the 16 the loader gives a kernel that reports none, and
// is dispatched, as it is today (F: on F9 the case dies of a segmentation fault, as the kernel
// argument cases around it do).
TEST(RocrKernelArgs, AnAlignmentOfZeroIsTakenAsSixteen) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : kBothChecks) {
      rdc_fake_hsa_reset();
      write_kernel_files();
      rdc_fake_hsa_set_kernel_info(nullptr, 80, 0, 16, 64);

      const OneGpu gpu = run_one_gpu(check);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_PASS) << gpu.text;
      EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 1u);
    }
  });
}

// An alignment that is not a power of two, and one above what the pool promises, are that GPU's
// failure, naming the value, with no kernel argument block and no dispatch (F: F9's checks of
// both are asserts the shipped builds compile out, and its read and write sub-test checks
// neither).
TEST(RocrKernelArgs, AnAlignmentTheCheckCannotMeetIsAFailure) {
  RDC_ROCR_CASE({
    const uint32_t alignments[] = {3, 8192};
    for (uint32_t alignment : alignments) {
      for (rdc_diag_test_cases_t check : kBothChecks) {
        rdc_fake_hsa_reset();
        write_kernel_files();
        rdc_fake_hsa_set_kernel_info(nullptr, 80, alignment, 16, 64);
        const uint32_t kernarg_pool = rdc_fake_hsa_pool_of_agent(kCpuAgent, kKernArgPool);

        const OneGpu gpu = run_one_gpu(check);
        EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << alignment;
        EXPECT_NE(gpu.text.find("aligned to " + std::to_string(alignment) + " bytes"),
                  std::string::npos)
            << gpu.text;
        EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u) << alignment;
        for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
          EXPECT_NE(rdc_fake_hsa_alloc_pool(block), kernarg_pool) << "block " << block;
        }
        expect_everything_released();
      }
    }
  });
}

// A kernel argument block the pool hands out 8 bytes into its page is that GPU's failure, with
// no dispatch (F: F9's Compute Queue check aligns the address itself, out of a block it asked
// twice the alignment more of, and its read and write sub-test dispatches the address as it is).
TEST(RocrKernelArgs, AMisalignedBlockIsAFailure) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : kBothChecks) {
      rdc_fake_hsa_reset();
      write_kernel_files();
      rdc_fake_hsa_set_kernarg_offset(8);

      const OneGpu gpu = run_one_gpu(check);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL);
      EXPECT_NE(gpu.text.find("The kernel argument buffer is not aligned to 16 bytes."),
                std::string::npos)
          << gpu.text;
      EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u);
      expect_everything_released();
    }
  });
}

// The System memory check on two GPUs from two threads, the first held at its doorbell, which
// comes after it allocated its kernel argument block and before it frees it, until the second
// has returned: each dispatch reads a block its own thread allocated, each block is freed once,
// and ROCr is loaded once for the two runs (F: F9 keeps that pointer in a file-scope variable,
// so the second run frees the first run's block and never frees its own).
TEST(RocrKernelArgs, TwoThreadsEachUseTheirOwnKernelArgumentBlock) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_gate("hsa_signal_store_relaxed", 1, 1);

    rdc_diag_result_t first = RDC_DIAG_RESULT_FAIL;
    rdc_diag_result_t second = RDC_DIAG_RESULT_FAIL;
    std::string first_text;
    std::string second_text;

    std::thread held([&first, &first_text]() {
      rdc_fake_hsa_set_caller(1);
      uint32_t ids[1] = {0};
      ResultPtr result = new_result();
      if (rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 1, nullptr, 0, result.get(),
                                 nullptr) == RDC_ST_OK &&
          result->per_gpu_result_count == 1) {
        first = result->status;
        first_text = result->gpu_results[0].gpu_result.msg;
      }
    });
    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_signal_store_relaxed", 1, 30000), 1);

    std::thread other([&second, &second_text]() {
      rdc_fake_hsa_set_caller(2);
      uint32_t ids[1] = {1};
      ResultPtr result = new_result();
      if (rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 1, nullptr, 0, result.get(),
                                 nullptr) == RDC_ST_OK &&
          result->per_gpu_result_count == 1) {
        second = result->status;
        second_text = result->gpu_results[0].gpu_result.msg;
      }
    });
    other.join();
    rdc_fake_hsa_gate_release("hsa_signal_store_relaxed");
    held.join();

    EXPECT_EQ(first, RDC_DIAG_RESULT_PASS) << first_text;
    EXPECT_EQ(second, RDC_DIAG_RESULT_PASS) << second_text;

    ASSERT_EQ(rdc_fake_hsa_dispatch_count(), 2u);
    EXPECT_NE(rdc_fake_hsa_dispatch_kernarg_address(0), rdc_fake_hsa_dispatch_kernarg_address(1));
    EXPECT_LT(block_of(rdc_fake_hsa_dispatch_kernarg_address(0)), rdc_fake_hsa_alloc_count());
    EXPECT_LT(block_of(rdc_fake_hsa_dispatch_kernarg_address(1)), rdc_fake_hsa_alloc_count());
    for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
      EXPECT_EQ(rdc_fake_hsa_alloc_frees(block), 1u) << "block " << block;
    }
    EXPECT_EQ(rdc_fake_hsa_load_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_refused_calls(), 0u);
    EXPECT_EQ(rdc_fake_hsa_double_frees(), 0u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);
  });
}

// --- the blocks the checks write ---------------------------------------------------------------

// Every block a check writes stays inside it, and the grid of each dispatch fits the buffers its
// arguments point to (F: F9 writes 2048 bytes into the search's 16-byte input and dispatches a
// grid of 64 at a 16-byte array, and its read and write sub-test writes 4 KiB into each 1 KiB
// buffer and its CPU-access sub-test one element past its own).
TEST(RocrBlocks, EveryBlockStaysInsideItsRedzone) {
  RDC_ROCR_CASE({
    for (rdc_diag_test_cases_t check : kBothChecks) {
      rdc_fake_hsa_reset();
      write_kernel_files();

      const OneGpu gpu = run_one_gpu(check);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_PASS) << gpu.text;
      EXPECT_EQ(rdc_fake_hsa_redzone_failures(), 0u);
      EXPECT_EQ(rdc_fake_hsa_grid_violations(), 0u) << rdc_fake_hsa_last_grid_violation();
      expect_everything_released();
    }
  });
}

// --- the pools the memory sub-tests choose ------------------------------------------------------

// A pool handle and an access value start as zeros, so that an iteration that matched nothing
// leaves no garbage handle for ROCr to dereference and no branch on dirt: the 0xA5 stack runs
// first, and the fake saw no pool handle it never issued (F).
TEST(RocrPools, AnIterationThatMatchedNothingLeavesNoGarbageHandle) {
  RDC_ROCR_CASE({
    // The CPU lists no pool, so the Compute Queue check has none to allocate from.
    dirty_stack();
    ASSERT_TRUE(stack_dirt_is_visible()) << "the stack the check runs on is not dirty";
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_clear_pools(kCpuAgent);
    OneGpu gpu = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_SKIP) << gpu.text;
    EXPECT_EQ(rdc_fake_hsa_foreign_pools(), 0u);
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);

    // The GPU lists no pool, so both access sub-tests skip, each saying which pool it wanted.
    dirty_stack();
    rdc_fake_hsa_reset();
    rdc_fake_hsa_clear_pools(1);
    gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_NE(gpu.text.find("The GPU has no coarse-grained memory pool to check."),
              std::string::npos)
        << gpu.text;
    EXPECT_NE(gpu.text.find("The GPU has no memory pool to check."), std::string::npos) << gpu.text;
    EXPECT_EQ(rdc_fake_hsa_foreign_pools(), 0u);
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);
    expect_everything_released();

    // A query that fails inside a pool iteration is that GPU's failure, and no handle it never
    // issued reaches the fake.
    dirty_stack();
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail_pool_info(UINT32_MAX, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_EQ(rdc_fake_hsa_foreign_pools(), 0u);
    expect_everything_released();

    // The read of an agent's access to a pool is the CPU-access sub-test's own, the first of
    // the check: a failure there is that GPU's failure, with its status, rather than a branch
    // on the value the call did not write.
    dirty_stack();
    rdc_fake_hsa_reset();
    rdc_fake_hsa_fail("hsa_amd_agent_memory_pool_get_info", 1, HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_NE(gpu.text.find("4104"), std::string::npos) << gpu.text;
    EXPECT_EQ(cpu_access_pool(), UINT32_MAX) << gpu.text;
    expect_everything_released();
  });
}

// The pool the CPU-access sub-test writes and reads is the first of the GPU's that it can use:
// global and coarse-grained, allowing allocation, large enough for what it writes and reachable
// by the CPU, whatever pool comes after it (F: F9 keeps the last global pool of the GPU, coarse
// or fine, and runs only where that pool's flags are exactly COARSE_GRAINED).
TEST(RocrPools, TheCpuAccessSubTestTakesTheFirstPoolItCanUse) {
  RDC_ROCR_CASE({
    const uint32_t gpu_agent = 1;

    // A coarse-grained pool, then a fine-grained one: the coarse-grained pool is written.
    rdc_fake_hsa_reset();
    rdc_fake_hsa_clear_pools(gpu_agent);
    uint32_t wanted = rdc_fake_hsa_add_pool(gpu_agent, HSA_AMD_SEGMENT_GLOBAL,
                                            HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                                            64 * 1024, 64 * 1024, 1, 4096, 4096, 1);
    rdc_fake_hsa_add_pool(gpu_agent, HSA_AMD_SEGMENT_GLOBAL,
                          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED, 64 * 1024, 64 * 1024, 1,
                          4096, 4096, 1);
    OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_NE(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_EQ(cpu_access_pool(), wanted) << gpu.text;

    // A coarse-grained pool that carries the kernel argument flag as well is still coarse-grained
    // memory the CPU can write: the flag is read as the bit it is.
    rdc_fake_hsa_reset();
    rdc_fake_hsa_clear_pools(gpu_agent);
    wanted = rdc_fake_hsa_add_pool(gpu_agent, HSA_AMD_SEGMENT_GLOBAL,
                                   HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED |
                                       HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT,
                                   64 * 1024, 64 * 1024, 1, 4096, 4096, 1);
    gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_NE(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_EQ(cpu_access_pool(), wanted) << gpu.text;

    // Two coarse-grained pools whose first the CPU cannot reach: the sub-test runs on the second
    // (M, not an F: F9 keeps the last matching pool, which here is the one it can use).
    rdc_fake_hsa_reset();
    rdc_fake_hsa_clear_pools(gpu_agent);
    const uint32_t refused = rdc_fake_hsa_add_pool(gpu_agent, HSA_AMD_SEGMENT_GLOBAL,
                                                   HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                                                   64 * 1024, 64 * 1024, 1, 4096, 4096, 1);
    wanted = rdc_fake_hsa_add_pool(gpu_agent, HSA_AMD_SEGMENT_GLOBAL,
                                   HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, 64 * 1024,
                                   64 * 1024, 1, 4096, 4096, 1);
    rdc_fake_hsa_set_pool_access(kCpuAgent, refused, HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED);
    gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_NE(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_EQ(cpu_access_pool(), wanted) << gpu.text;
    expect_everything_released();
  });
}

// A GPU with no pool that sub-test can use is its skip, which says whether a pool was passed
// over for the CPU's access alone, and nothing is allocated from it (F: F9 allocates half of
// whatever pool it kept and reads one element past it).
TEST(RocrPools, AGpuWithNoPoolTheCpuCanWriteIsASkipThatSaysWhy) {
  RDC_ROCR_CASE({
    struct Listed {
      const char* name;
      uint32_t flags;
      uint64_t max_size;
      uint32_t access;
      const char* sentence;
    };
    const Listed listed[] = {
        {"fine-grained", HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED, 64 * 1024,
         HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT,
         "The GPU has no coarse-grained memory pool to check."},
        {"a 4 KiB maximum", HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, 4 * 1024,
         HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT,
         "The GPU has no coarse-grained memory pool to check."},
        {"out of the CPU's reach", HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, 64 * 1024,
         HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED, "The CPU cannot access the GPU's memory pool."}};

    for (const Listed& one : listed) {
      rdc_fake_hsa_reset();
      const uint32_t gpu_agent = 1;
      rdc_fake_hsa_clear_pools(gpu_agent);
      const uint32_t pool = rdc_fake_hsa_add_pool(gpu_agent, HSA_AMD_SEGMENT_GLOBAL, one.flags,
                                                  64 * 1024, one.max_size, 1, 4096, 4096, 1);
      rdc_fake_hsa_set_pool_access(kCpuAgent, pool, one.access);

      const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
      EXPECT_NE(gpu.text.find(one.sentence), std::string::npos) << one.name << ": " << gpu.text;
      EXPECT_EQ(cpu_access_pool(), UINT32_MAX) << one.name;
      expect_everything_released();
    }
  });
}

// Each of the two accesses the read and write sub-test needs has a skip of its own, decided
// before it creates a queue, a signal or a block (F: F9 reads only the CPU's access to the GPU's
// pool, grants the GPU's access to the CPU's without reading it, and appends a pass either way).
TEST(RocrPools, EachAccessTheReadAndWriteSubTestNeedsHasItsOwnSkip) {
  RDC_ROCR_CASE({
    struct Refused {
      const char* name;
      uint32_t agent;
      uint32_t pool_of;  // the agent whose pool the access is refused on
      uint32_t nth_pool;
      const char* sentence;
    };
    const Refused refused[] = {{"the CPU to the GPU's pool", kCpuAgent, 1, 0,
                                "The CPU cannot read the GPU's memory "
                                "pool."},
                               {"the GPU to the CPU's pool", 1, kCpuAgent, 2,
                                "The GPU cannot access the CPU's memory "
                                "pool."}};

    for (const Refused& one : refused) {
      rdc_fake_hsa_reset();
      write_kernel_files();
      rdc_fake_hsa_set_pool_access(one.agent, rdc_fake_hsa_pool_of_agent(one.pool_of, one.nth_pool),
                                   HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED);

      const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
      EXPECT_NE(gpu.text.find(one.sentence), std::string::npos) << one.name << ": " << gpu.text;
      EXPECT_EQ(rdc_fake_hsa_queue_count(), 0u) << one.name;
      EXPECT_EQ(rdc_fake_hsa_signal_count(), 0u) << one.name;
      EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u) << one.name;
      expect_everything_released();
    }
  });
}

// A grant of access that fails is that GPU's failure, with every block the fake issued freed
// (F: F9 leaks the host buffer and the pool block of its CPU-access sub-test).
TEST(RocrPools, AFailedGrantIsThatGpusFailure) {
  RDC_ROCR_CASE({
    // The first grant is the CPU-access sub-test's, and the sixth the read and write sub-test's
    // grant of the kernel argument block.
    const uint32_t grants[] = {1, 6};
    for (uint32_t nth : grants) {
      rdc_fake_hsa_reset();
      write_kernel_files();
      rdc_fake_hsa_fail("hsa_amd_agents_allow_access", nth, HSA_STATUS_ERROR_OUT_OF_RESOURCES);

      const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << "grant " << nth << ": " << gpu.text;
      expect_everything_released();
    }
  });
}

// --- what the memory sub-tests read back --------------------------------------------------------

// A read-back shifted by one element is that GPU's failure, where F9's data, all 1s, hides it
// (F).
TEST(RocrData, AReadBackShiftedByOneElementIsAFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_SHIFTED);

    const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_NE(gpu.text.find("The GPU read other values from the system memory than the CPU wrote."),
              std::string::npos)
        << gpu.text;
    expect_everything_released();
  });
}

// The two pages of the CPU-access buffer are distinct memory: where the fake maps the second
// onto the first, the read-back finds other values than the CPU wrote, from element 0 on, and
// that is the GPU's failure (F: F9's buffer is 8 bytes in one page, and a mismatch is a line on
// stdout rather than an outcome).
TEST(RocrData, AliasedPagesAreSeenByTheReadBack) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_set_alias_pages(1);

    const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_NE(gpu.text.find("The CPU read other values from the GPU's memory than it wrote."),
              std::string::npos)
        << gpu.text;
    expect_everything_released();
  });
}

// --- the pools the System memory check counts ---------------------------------------------------

namespace {

// A coarse-grained pool of the GPU of that size, which is also its largest allocation.
uint32_t add_gpu_pool(uint64_t size, uint64_t max_size, int alloc_allowed = 1) {
  return rdc_fake_hsa_add_pool(1, HSA_AMD_SEGMENT_GLOBAL,
                               HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, size, max_size,
                               alloc_allowed, 4096, 4096, 1);
}

}  // namespace

// Each pool of the GPU that allows allocation is allocated from and freed, 64 MiB of it or its
// whole maximum where that is smaller, and the GPU's text says so pool by pool (F: F9 searches
// for the largest allocation each pool can give, which its own loop cuts short at half the pool,
// 8.6 GB on this host's gfx1030, and says "Pool N test pass." whether it allocated or not).
TEST(RocrSysMem, EachPoolThatAllowsAllocationIsAllocatedAndFreed) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_clear_pools(1);
    const uint32_t small = add_gpu_pool(64 * 1024, 64 * 1024);
    const uint32_t large = add_gpu_pool(1024ull * 1024 * 1024, 1024ull * 1024 * 1024);

    const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_NE(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_NE(gpu.text.find("Memory pool 1: allocated and freed 64 KiB."), std::string::npos)
        << gpu.text;
    EXPECT_NE(gpu.text.find("Memory pool 2: allocated and freed 64 MiB."), std::string::npos)
        << gpu.text;

    // The whole of the pool that is smaller than 64 MiB, and 64 MiB of the pool that is larger.
    bool whole_pool = false;
    bool part_of_pool = false;
    for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
      if (rdc_fake_hsa_alloc_pool(block) == small && rdc_fake_hsa_alloc_size(block) == 64 * 1024) {
        whole_pool = true;
      }
      if (rdc_fake_hsa_alloc_pool(block) == large) {
        EXPECT_EQ(rdc_fake_hsa_alloc_size(block), 64ull * 1024 * 1024) << "block " << block;
        part_of_pool = true;
      }
    }
    EXPECT_TRUE(whole_pool);
    EXPECT_TRUE(part_of_pool);
    expect_everything_released();
  });
}

// A pool whose largest allocation is below what a sub-test writes is not the pool it takes: ROCr
// refuses the request before it allocates anything, so a GPU with no pool that can hold the
// sub-test's buffers is its skip (F: F9 allocates from it and reports the refusal as a failure).
TEST(RocrSysMem, APoolTooSmallForASubTestIsNotTheOneItTakes) {
  RDC_ROCR_CASE({
    // The GPU's only pool cannot hand out the 4 KiB the read and write sub-test gives the kernel.
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_fake_hsa_clear_pools(1);
    const uint32_t gpu_pool = add_gpu_pool(64 * 1024, 2048);

    OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_NE(gpu.text.find("The GPU has no memory pool to check."), std::string::npos) << gpu.text;
    for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
      if (rdc_fake_hsa_alloc_pool(block) == gpu_pool) {
        // The one block from it is the pool sub-test's own, of the pool's whole maximum.
        EXPECT_EQ(rdc_fake_hsa_alloc_size(block), 2048u) << "block " << block;
      }
    }
    expect_everything_released();

    // The same of the CPU's pools, from which that sub-test takes its host buffers.
    rdc_fake_hsa_reset();
    write_kernel_files();
    for (uint32_t n = 0; n < rdc_fake_hsa_pool_count(kCpuAgent); n++) {
      rdc_fake_hsa_set_pool_max(rdc_fake_hsa_pool_of_agent(kCpuAgent, n), 2048);
    }

    gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_NE(gpu.text.find("The CPU has no memory pool for the check."), std::string::npos)
        << gpu.text;
    EXPECT_EQ(rdc_fake_hsa_queue_count(), 0u);
    expect_everything_released();
  });
}

// A read of one pool's own attributes that fails is that pool's failure, naming the call, and
// the pools after it are checked all the same (F: F9's AcquirePoolInfo throws, so the GPU's
// later pools go unchecked and the check writes no per-GPU row).
TEST(RocrSysMem, AFailedPoolQueryIsThatPoolsFailure) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_clear_pools(1);
    add_gpu_pool(64 * 1024, 64 * 1024);
    const uint32_t refused = add_gpu_pool(64 * 1024, 64 * 1024);
    add_gpu_pool(64 * 1024, 64 * 1024);
    rdc_fake_hsa_fail_pool_info(refused, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                HSA_STATUS_ERROR_INVALID_ARGUMENT);

    const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_NE(gpu.text.find("Fail to call hsa_amd_memory_pool_get_info() for memory pool 2."),
              std::string::npos)
        << gpu.text;
    EXPECT_NE(gpu.text.find("4097"), std::string::npos) << gpu.text;
    EXPECT_NE(gpu.text.find("Memory pool 1: allocated and freed 64 KiB."), std::string::npos)
        << gpu.text;
    EXPECT_NE(gpu.text.find("Memory pool 3: allocated and freed 64 KiB."), std::string::npos)
        << gpu.text;
    expect_everything_released();
  });
}

// A pool whose block cannot be given back says that it was allocated, is not counted, and is
// that GPU's failure through the holder; the pools after it are checked all the same (F: on F9
// the case dies of a segmentation fault, as the pool cases around it do).
TEST(RocrSysMem, APoolWhoseBlockIsNotFreedSaysOnlyThatItWasAllocated) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_clear_pools(1);
    add_gpu_pool(64 * 1024, 64 * 1024);
    add_gpu_pool(64 * 1024, 64 * 1024);
    // The first free of the run is the first pool's: this sub-test runs before the two access
    // sub-tests, and allocates nothing else.
    rdc_fake_hsa_fail("hsa_amd_memory_pool_free", 1, HSA_STATUS_ERROR_INVALID_ALLOCATION);

    const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_NE(gpu.text.find("Memory pool 1: allocated 64 KiB."), std::string::npos) << gpu.text;
    EXPECT_EQ(gpu.text.find("Memory pool 1: allocated and freed"), std::string::npos) << gpu.text;
    EXPECT_NE(gpu.text.find("Fail to call hsa_amd_memory_pool_free()."), std::string::npos)
        << gpu.text;
    EXPECT_NE(gpu.text.find("4099"), std::string::npos) << gpu.text;
    // The pool after it is checked and says what it did, and the block that was not freed is the
    // only one the run left behind.
    EXPECT_NE(gpu.text.find("Memory pool 2: allocated and freed 64 KiB."), std::string::npos)
        << gpu.text;
    EXPECT_EQ(rdc_fake_hsa_live_blocks(), 1u);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);
  });
}

// A GPU whose only pool could not take its block back has no pool that was checked to its end:
// its text says so, and carries the release's own failure and the pool's own sentence (F: on F9
// the case dies of a segmentation fault, as the pool cases around it do; M: T4's mutation counts
// a pool whose free failed).
TEST(RocrSysMem, TheOnlyPoolWhoseBlockIsNotFreedLeavesNothingChecked) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_clear_pools(1);
    add_gpu_pool(64 * 1024, 64 * 1024);
    // The first free of the run is that pool's: this sub-test runs before the two access
    // sub-tests, and nothing else allocates from a pool this small.
    rdc_fake_hsa_fail("hsa_amd_memory_pool_free", 1, HSA_STATUS_ERROR_INVALID_ALLOCATION);

    const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_NE(gpu.text.find("No memory pool could be checked."), std::string::npos) << gpu.text;
    EXPECT_NE(gpu.text.find("Fail to call hsa_amd_memory_pool_free()."), std::string::npos)
        << gpu.text;
    EXPECT_NE(gpu.text.find("Memory pool 1: allocated 64 KiB."), std::string::npos) << gpu.text;
    // The block that was not freed is the only one the run left behind.
    EXPECT_EQ(rdc_fake_hsa_live_blocks(), 1u);
    EXPECT_EQ(rdc_fake_hsa_live_references(), 0u);
  });
}

// A pool with no free memory is that pool's skip, which says so; any other status is that GPU's
// failure, naming the call (F: F9 makes the pool's status the check's and writes no per-GPU row,
// so a GPU whose memory another process is using fails the check).
TEST(RocrSysMem, APoolWithNoFreeMemoryIsASkipAndAnyOtherErrorIsAFailure) {
  RDC_ROCR_CASE({
    struct Answer {
      int32_t status;
      bool failure;
      const char* sentence;
    };
    const Answer answers[] = {
        {HSA_STATUS_ERROR_OUT_OF_RESOURCES, false, "Memory pool 1 had no free memory."},
        {HSA_STATUS_ERROR_INVALID_ALLOCATION, true,
         "Fail to call hsa_amd_memory_pool_allocate() for memory pool 1."},
        {HSA_STATUS_ERROR_INVALID_ARGUMENT, true,
         "Fail to call hsa_amd_memory_pool_allocate() for memory pool 1."}};

    for (const Answer& answer : answers) {
      rdc_fake_hsa_reset();
      rdc_fake_hsa_clear_pools(1);
      add_gpu_pool(64 * 1024, 64 * 1024);
      // The first allocation of the run is the pool sub-test's, which runs before the others.
      rdc_fake_hsa_fail("hsa_amd_memory_pool_allocate", 1, answer.status);

      const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
      EXPECT_NE(gpu.text.find(answer.sentence), std::string::npos)
          << answer.status << ": " << gpu.text;
      EXPECT_EQ(gpu.status == RDC_DIAG_RESULT_FAIL, answer.failure)
          << answer.status << ": " << gpu.text;
      if (!answer.failure) {
        // A pool that had no memory to give was not checked, and no other pool was either.
        EXPECT_NE(gpu.text.find("No memory pool could be checked."), std::string::npos) << gpu.text;
      }
      expect_everything_released();
    }
  });
}

// A pool that does not allow allocation is not counted, and a GPU with no pool that could be
// checked is a skip that says so, in the GPU's text and in the check's info (F: F9 says " for
// GPU n Pass." and writes no per-GPU text when it checked nothing).
TEST(RocrSysMem, NoCountedPoolIsASkipThatSaysSo) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_clear_pools(1);
    add_gpu_pool(64 * 1024, 64 * 1024, 0);

    uint32_t ids[1] = {0};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 1, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 1u);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_SKIP);
    const std::string text = result->gpu_results[0].gpu_result.msg;
    EXPECT_NE(text.find("No memory pool could be checked."), std::string::npos) << text;
    EXPECT_EQ(text_of(result->info), "GPU 0: no memory pool could be checked.") << result->info;
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);
    expect_everything_released();
  });
}

// A pool that allows allocation but reports a maximum of 0 is not counted either: ROCr refuses a
// request of no bytes, so there is nothing to check (M: T4's mutation drops the rule, after which
// the check asks for 0 bytes and reports the refusal).
TEST(RocrSysMem, APoolWhoseMaximumIsZeroIsNotCounted) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_clear_pools(1);
    add_gpu_pool(64 * 1024, 0);

    const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_NE(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_NE(gpu.text.find("No memory pool could be checked."), std::string::npos) << gpu.text;
    EXPECT_EQ(rdc_fake_hsa_alloc_count(), 0u);
    expect_everything_released();
  });
}

// --- the work a check could not stop -------------------------------------------------------------

namespace {

// What the Compute Queue check says of a GPU it checked to the end, and what the read and write
// sub-test of the System memory check says: a case that states one GPU kept its result reads it.
constexpr char kGpuAccessDone[] = "The GPU read and wrote 4 KiB of the CPU's memory.";

#if RDC_ROCR_TEST_HOOKS
// The entity ids a skip names, in the order the record holds them.
std::vector<uint32_t> named_gpus(const std::string& text) {
  std::vector<uint32_t> entities;
  const std::string marker = "its work on GPU";
  size_t at = text.find(marker);
  if (at == std::string::npos) {
    return entities;
  }
  for (at += marker.size(); at < text.size() && text[at] != ';'; at++) {
    if (std::isdigit(static_cast<unsigned char>(text[at])) == 0) {
      continue;
    }
    uint32_t entity = 0;
    while (at < text.size() && std::isdigit(static_cast<unsigned char>(text[at])) != 0) {
      entity = (entity * 10) + static_cast<uint32_t>(text[at] - '0');
      at++;
    }
    entities.push_back(entity);
  }
  return entities;
}

// How often the text holds that line.
size_t times_in(const std::string& text, const std::string& line) {
  size_t found = 0;
  for (size_t at = text.find(line); at != std::string::npos; at = text.find(line, at + 1)) {
    found++;
  }
  return found;
}
#endif

}  // namespace

// A kernel that never completes is abandoned at the deadline: nothing it can touch is released,
// the test object keeps its reference to ROCr rather than shutting it down under a live queue,
// the GPU is recorded, and the GPU after it in the same run is skipped. The module pins itself
// first, and here, where its code is part of the test binary, the dlopen finds nothing and the
// pin writes its one line, which carries no text from the error another call left pending (M).
TEST(RocrAbandon, AKernelThatNeverCompletesIsAbandonedAndRecorded) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    add_gpu_with_node(3);
    EXPECT_EQ(rdc_rocr_test_set_wait_deadline_ms(kShortDeadlineMs), kModuleDeadlineMs);

    // An error an earlier call left pending and nobody read, as a loader's work leaves one.
    (void)::dlsym(RTLD_DEFAULT, "rdc_a_symbol_no_process_carries");

    // The second GPU is the one whose kernel never completes: its check is held at the kernel it
    // loads, which both checks do before they make a queue.
    rdc_fake_hsa_gate("hsa_code_object_reader_create_from_file", 0, 2);

    ::testing::internal::CaptureStdout();
    Running run(kFirstRun, RDC_DIAG_SYS_MEM_CHECK, {0, 1, 2});
    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_code_object_reader_create_from_file", 1, 30000), 1);
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_NEVER);
    rdc_fake_hsa_gate_release("hsa_code_object_reader_create_from_file");
    run.join();
    std::cout << "the log still works" << std::endl;
    const std::string log = ::testing::internal::GetCapturedStdout();

    EXPECT_EQ(run.status(), RDC_ST_OK);
    ASSERT_EQ(run.result().per_gpu_result_count, 3u);
    EXPECT_NE(run.text(0).find(kGpuAccessDone), std::string::npos) << run.text(0);
    EXPECT_NE(run.text(1).find("The kernel did not complete in 300 ms."), std::string::npos)
        << run.text(1);
    EXPECT_NE(run.text(2).find("The ROCr checks are disabled"), std::string::npos) << run.text(2);
    EXPECT_EQ(named_gpus(run.text(2)), std::vector<uint32_t>({1})) << run.text(2);

    // Nothing the kernel can touch was released: its queue, its signal, its executable and its
    // reader are the ones that stayed, with the five blocks that dispatch reads and writes.
    const std::vector<std::string> destroys = destroy_log();
    EXPECT_EQ(rdc_fake_hsa_queue_destroyed(1), 0) << joined(destroys);
    EXPECT_EQ(released(destroys, "queue"), 1u) << joined(destroys);
    EXPECT_EQ(released(destroys, "signal"), 1u) << joined(destroys);
    EXPECT_EQ(released(destroys, "executable"), 1u) << joined(destroys);
    EXPECT_EQ(released(destroys, "reader"), 1u) << joined(destroys);
    EXPECT_EQ(rdc_fake_hsa_live_blocks(), 5u) << joined(destroys);

    // The abandoned object skipped its Close(), so ROCr keeps a reference and is not latched.
    EXPECT_EQ(rdc_fake_hsa_live_references(), 1u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);

    // The pin's one line is its own text and nothing else.
    const size_t pin = log.find("Cannot pin ");
    ASSERT_NE(pin, std::string::npos) << log;
    EXPECT_EQ(times_in(log, "Cannot pin "), 1u) << log;
    const size_t ended = log.find('\n', pin);
    ASSERT_NE(ended, std::string::npos) << log;
    const std::string line = log.substr(pin, ended - pin);
    EXPECT_NE(line.find("the record of GPU 1 lasts while the module stays mapped."),
              std::string::npos)
        << line;
    EXPECT_EQ(line.find("rdc_a_symbol_no_process_carries"), std::string::npos) << line;

    // The log still works: a line written after the pin arrives whole.
    EXPECT_TRUE(std::cout.good());
    EXPECT_NE(log.find("the log still works\n"), std::string::npos) << log;
  });
#else
  GTEST_SKIP() << "needs the deadline hook of RocrTestHooks.h";
#endif
}

// A memory copy that never completes is abandoned by the same rule: its signal and the buffers
// it reads and writes are kept, with the queue (M: T4's deadline group, whose two mutations are
// killed by the kernel case above; this is the copy's half of the one rule, and mut.py's covered
// list carries that reason).
TEST(RocrAbandon, ACopyThatNeverCompletesIsAbandoned) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    EXPECT_EQ(rdc_rocr_test_set_wait_deadline_ms(kShortDeadlineMs), kModuleDeadlineMs);
    rdc_fake_hsa_set_copy_mode(RDC_FAKE_COPY_NEVER);

    const OneGpu answer = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(answer.status, RDC_DIAG_RESULT_FAIL) << answer.text;
    EXPECT_NE(answer.text.find("A memory copy did not complete in 300 ms."), std::string::npos)
        << answer.text;

    const std::vector<std::string> destroys = destroy_log();
    EXPECT_EQ(released(destroys, "signal"), 0u) << joined(destroys);
    EXPECT_EQ(released(destroys, "block"), 0u) << joined(destroys);
    EXPECT_EQ(rdc_fake_hsa_queue_destroyed(0), 0) << joined(destroys);
    EXPECT_EQ(rdc_fake_hsa_live_blocks(), rdc_fake_hsa_alloc_count());
    EXPECT_EQ(rdc_fake_hsa_live_references(), 1u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);
  });
#else
  GTEST_SKIP() << "needs the deadline hook of RocrTestHooks.h";
#endif
}

// A later run reads the record before it maps an entity or calls ROCr: the GPU in it is that
// run's failure, every other GPU is skipped with the recorded GPU named, and the fake sees no
// call at all (M).
TEST(RocrAbandon, ALaterRunFailsTheRecordedGpuAndSkipsTheOthers) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_rocr_test_set_wait_deadline_ms(kShortDeadlineMs);
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_NEVER);
    EXPECT_EQ(run_one_gpu(RDC_DIAG_SYS_MEM_CHECK).status, RDC_DIAG_RESULT_FAIL);

    rdc_fake_hsa_reset();
    uint32_t ids[2] = {0, 1};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_COMPUTE_QUEUE, ids, 2, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
    ASSERT_EQ(result->per_gpu_result_count, 2u);
    EXPECT_NE(std::string(result->gpu_results[0].gpu_result.msg)
                  .find("An earlier check could not stop its work on this GPU; the ROCr checks "
                        "are disabled until the process running them restarts."),
              std::string::npos)
        << result->gpu_results[0].gpu_result.msg;
    EXPECT_NE(std::string(result->gpu_results[1].gpu_result.msg)
                  .find("The ROCr checks are disabled: an earlier check could not stop its work "
                        "on GPU 0; restart the process that runs them."),
              std::string::npos)
        << result->gpu_results[1].gpu_result.msg;
    EXPECT_EQ(rdc_fake_hsa_call_log_size(), 0u);
    EXPECT_EQ(rdc_fake_hsa_node_id_calls(), 0u);
  });
#else
  GTEST_SKIP() << "needs the deadline hook of RocrTestHooks.h";
#endif
}

// A run that is in flight reads the record before each of its GPUs, so one set while it waits on
// its first GPU skips its next one; once both runs have returned, the only reference left is the
// abandoned object's (M).
TEST(RocrAbandon, ARunInFlightSeesTheRecordAtItsNextGpu) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    add_gpu_with_node(3);
    rdc_rocr_test_set_wait_deadline_ms(kShortDeadlineMs);

    // The first run is held inside the wait of its first GPU, whose dispatch has completed.
    rdc_fake_hsa_gate("hsa_signal_wait_scacquire", kFirstRun, 0);
    Running first(kFirstRun, RDC_DIAG_SYS_MEM_CHECK, {0, 1});
    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_signal_wait_scacquire", 1, 30000), 1);

    // The other run's waits run to the deadline and record its GPU.
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_NEVER);
    Running second(kSecondRun, RDC_DIAG_SYS_MEM_CHECK, {2});
    second.join();
    EXPECT_EQ(second.status(), RDC_ST_OK);
    EXPECT_NE(second.text(0).find("The kernel did not complete in 300 ms."), std::string::npos)
        << second.text(0);

    rdc_fake_hsa_gate_release("hsa_signal_wait_scacquire");
    first.join();
    EXPECT_EQ(first.status(), RDC_ST_OK);
    ASSERT_EQ(first.result().per_gpu_result_count, 2u);
    EXPECT_NE(first.text(0).find(kGpuAccessDone), std::string::npos) << first.text(0);
    EXPECT_NE(first.text(1).find("The ROCr checks are disabled"), std::string::npos)
        << first.text(1);
    EXPECT_EQ(named_gpus(first.text(1)), std::vector<uint32_t>({2})) << first.text(1);

    EXPECT_EQ(rdc_fake_hsa_live_references(), 1u);
    EXPECT_EQ(rdc_fake_hsa_latched(), 0);
  });
#else
  GTEST_SKIP() << "needs the deadline hook of RocrTestHooks.h";
#endif
}

// Three runs held until all three have dispatched abandon their GPUs together: each of them is a
// failure in a later run, and the GPU that was not checked is skipped with all three named
// (M: T4's mutation lets a claim try the first slot alone, which leaves two of the three
// unrecorded).
TEST(RocrAbandon, ThreeRunsThatAbandonTheirGpusAreAllNamed) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    add_gpu_with_node(3);
    add_gpu_with_node(4);
    rdc_rocr_test_set_wait_deadline_ms(kShortDeadlineMs);
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_NEVER);

    rdc_fake_hsa_gate("hsa_signal_wait_scacquire", 0, 0);
    Running first(kFirstRun, RDC_DIAG_SYS_MEM_CHECK, {0});
    Running second(kSecondRun, RDC_DIAG_SYS_MEM_CHECK, {1});
    Running third(3, RDC_DIAG_SYS_MEM_CHECK, {2});
    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_signal_wait_scacquire", 3, 30000), 1);
    rdc_fake_hsa_gate_release("hsa_signal_wait_scacquire");
    first.join();
    second.join();
    third.join();
    EXPECT_EQ(first.status(), RDC_ST_OK);
    EXPECT_EQ(second.status(), RDC_ST_OK);
    EXPECT_EQ(third.status(), RDC_ST_OK);
    EXPECT_NE(first.text(0).find("The kernel did not complete in 300 ms."), std::string::npos)
        << first.text(0);
    EXPECT_NE(second.text(0).find("The kernel did not complete in 300 ms."), std::string::npos)
        << second.text(0);
    EXPECT_NE(third.text(0).find("The kernel did not complete in 300 ms."), std::string::npos)
        << third.text(0);

    uint32_t ids[4] = {0, 1, 2, 3};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 4, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 4u);
    for (uint32_t row = 0; row < 3; row++) {
      EXPECT_NE(std::string(result->gpu_results[row].gpu_result.msg)
                    .find("An earlier check could not stop its work on this GPU"),
                std::string::npos)
          << result->gpu_results[row].gpu_result.msg;
    }
    const std::string skipped = result->gpu_results[3].gpu_result.msg;
    std::vector<uint32_t> named = named_gpus(skipped);
    std::sort(named.begin(), named.end());
    EXPECT_EQ(named, std::vector<uint32_t>({0, 1, 2})) << skipped;
    EXPECT_NE(skipped.find("GPUs "), std::string::npos) << skipped;
    EXPECT_NE(skipped.find(" and "), std::string::npos) << skipped;
  });
#else
  GTEST_SKIP() << "needs the deadline hook of RocrTestHooks.h";
#endif
}

// Two runs that abandon their work on the same GPU spend one slot, and the texts name it once
// (M: no mutation of its own. The scan that finds a GPU already in the record is one of T4's
// declared exceptions, because the claim loop's own return on that GPU's slot reaches the same
// end, so what this case holds cannot be reverted on its own; mut.py's covered list says so).
TEST(RocrAbandon, TwoRunsThatAbandonTheSameGpuSpendOneSlot) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    rdc_rocr_test_set_wait_deadline_ms(kShortDeadlineMs);
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_NEVER);

    // The second run is held at its own dispatch until the first run's claim is published.
    rdc_fake_hsa_gate("hsa_signal_wait_scacquire", kSecondRun, 0);
    Running second(kSecondRun, RDC_DIAG_SYS_MEM_CHECK, {0});
    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_signal_wait_scacquire", 1, 30000), 1);
    Running first(kFirstRun, RDC_DIAG_SYS_MEM_CHECK, {0});
    first.join();
    EXPECT_EQ(first.status(), RDC_ST_OK);
    EXPECT_NE(first.text(0).find("The kernel did not complete in 300 ms."), std::string::npos)
        << first.text(0);
    rdc_fake_hsa_gate_release("hsa_signal_wait_scacquire");
    second.join();
    EXPECT_EQ(second.status(), RDC_ST_OK);
    EXPECT_NE(second.text(0).find("The kernel did not complete in 300 ms."), std::string::npos)
        << second.text(0);

    uint32_t ids[2] = {0, 1};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 2, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 2u);
    const std::string skipped = result->gpu_results[1].gpu_result.msg;
    EXPECT_EQ(named_gpus(skipped), std::vector<uint32_t>({0})) << skipped;
    EXPECT_NE(skipped.find("on GPU 0;"), std::string::npos) << skipped;
  });
#else
  GTEST_SKIP() << "needs the deadline hook of RocrTestHooks.h";
#endif
}

// Nine runs released together onto kernels that never complete: eight GPUs are recorded, the one
// claim that finds every slot taken is logged at error level, and all nine are failures (M).
TEST(RocrAbandon, AnAbandonWithNoSlotLeftIsLoggedAndIsStillAFailure) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    for (uint32_t node = 3; node <= 9; node++) {
      add_gpu_with_node(node);
    }
    ASSERT_EQ(rdc_fake_hsa_agent_count(), 10u);  // the CPU agent and nine GPUs
    rdc_rocr_test_set_wait_deadline_ms(kShortDeadlineMs);
    rdc_fake_hsa_set_kernel_mode(RDC_FAKE_KERNEL_NEVER);
    rdc_fake_hsa_gate("hsa_signal_wait_scacquire", 0, 0);

    ::testing::internal::CaptureStdout();
    std::vector<std::unique_ptr<Running>> runs;
    for (uint32_t gpu = 0; gpu < 9; gpu++) {
      runs.push_back(std::unique_ptr<Running>(new Running(gpu + 1, RDC_DIAG_SYS_MEM_CHECK, {gpu})));
    }
    ASSERT_EQ(rdc_fake_hsa_gate_wait("hsa_signal_wait_scacquire", 9, 50000), 1);
    rdc_fake_hsa_gate_release("hsa_signal_wait_scacquire");
    for (const std::unique_ptr<Running>& run : runs) {
      run->join();
    }
    const std::string log = ::testing::internal::GetCapturedStdout();

    for (uint32_t gpu = 0; gpu < 9; gpu++) {
      EXPECT_EQ(runs[gpu]->status(), RDC_ST_OK) << "GPU " << gpu;
      EXPECT_NE(runs[gpu]->text(0).find("The kernel did not complete in 300 ms."),
                std::string::npos)
          << "GPU " << gpu << ": " << runs[gpu]->text(0);
    }
    EXPECT_EQ(times_in(log, "no slot left to record it"), 1u) << log;

    // Eight of the nine GPUs are recorded: the ninth, whose claim found no slot, is the one a
    // later run skips, naming the eight.
    uint32_t ids[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    ResultPtr result = new_result();
    ASSERT_EQ(
        rdc_diag_test_case_run(RDC_DIAG_SYS_MEM_CHECK, ids, 9, nullptr, 0, result.get(), nullptr),
        RDC_ST_OK);
    ASSERT_EQ(result->per_gpu_result_count, 9u);
    size_t failures = 0;
    std::vector<uint32_t> named;
    for (uint32_t row = 0; row < 9; row++) {
      const std::string text = result->gpu_results[row].gpu_result.msg;
      if (text.find("An earlier check could not stop its work on this GPU") != std::string::npos) {
        failures++;
        continue;
      }
      EXPECT_NE(text.find("The ROCr checks are disabled"), std::string::npos) << text;
      named = named_gpus(text);
    }
    EXPECT_EQ(failures, 8u);
    std::sort(named.begin(), named.end());
    named.erase(std::unique(named.begin(), named.end()), named.end());
    EXPECT_EQ(named.size(), 8u);
  });
#else
  GTEST_SKIP() << "needs the deadline hook of RocrTestHooks.h";
#endif
}

// A dispatch the fake completes inside the third wait on its signal: the check waits three
// times, each with a hint of a tenth of the timestamp frequency and each of them blocked,
// and passes (F: F9 waits once, with a hint that never runs out, so the case's time limit
// kills the child; M: T4's mutation waits in the active state, which burns the CPU for the
// whole hint).
TEST(RocrAbandon, ADispatchCompletedAtTheThirdWaitPassesAfterThreeWaits) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    write_kernel_files();
    // A thousand ticks a second, so that a tenth of a second is a hint of 100 of them.
    rdc_fake_hsa_set_timestamp_frequency(1000);
    rdc_fake_hsa_set_completion_wait(3);

    const OneGpu answer = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_EQ(answer.status, RDC_DIAG_RESULT_PASS) << answer.text;
    EXPECT_EQ(rdc_fake_hsa_wait_count(), 3u);
    ASSERT_EQ(rdc_fake_hsa_wait_hint_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_wait_hint(0), 100u);
    // Every wait the module made is blocked: one state, and that state.
    ASSERT_EQ(rdc_fake_hsa_wait_state_count(), 1u);
    EXPECT_EQ(rdc_fake_hsa_wait_state(0), HSA_WAIT_STATE_BLOCKED);
  });
}

// The deadline the module waits to, before a test has replaced it, is ten seconds (M: T4's
// mutation of the constant).
TEST(RocrAbandon, TheDeadlineTheModuleWaitsToIsTenSeconds) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    // No case of this child has called the hook before this one, so what it returns is the
    // module's own deadline.
    EXPECT_EQ(rdc_rocr_test_set_wait_deadline_ms(kShortDeadlineMs), kModuleDeadlineMs);
    EXPECT_EQ(rdc_rocr_test_set_wait_deadline_ms(kModuleDeadlineMs), kShortDeadlineMs);
  });
#else
  GTEST_SKIP() << "needs the deadline hook of RocrTestHooks.h";
#endif
}

// --- the kernel files the checks load -----------------------------------------------------------

// A kernel file that is not installed for the agent is that sub-test's skip, which names the
// file and the agent, while the sub-tests that need no kernel keep their results; the sub-test
// creates no queue and allocates nothing of its own (F: F9 creates its queue and its buffers
// first, and its text is "fail to open <file>" with an errno).
TEST(RocrKernelFiles, AMissingKernelIsThatSubTestsSkip) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    // Only the other check's kernel is installed, so this one's is the file that is missing.
    write_kernel_file("binary_search_kernels.hsaco");
    search_these_directories({"."});

    const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    EXPECT_NE(gpu.text.find("No gpuReadWrite_kernels.hsaco is installed for gfx1030."),
              std::string::npos)
        << gpu.text;
    // A skip beside two sub-tests that passed is that GPU's pass, and each of them said what it
    // did.
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_PASS) << gpu.text;
    EXPECT_NE(gpu.text.find("Memory pool 1: allocated and freed 64 KiB."), std::string::npos)
        << gpu.text;
    EXPECT_NE(gpu.text.find("The CPU wrote and read back 8 KiB of GPU memory."), std::string::npos)
        << gpu.text;

    EXPECT_EQ(rdc_fake_hsa_queue_count(), 0u);
    EXPECT_EQ(rdc_fake_hsa_signal_count(), 0u);
    EXPECT_EQ(rdc_fake_hsa_dispatch_count(), 0u);
    for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
      // The sub-test's own buffers are 4 KiB; the blocks here are the other sub-tests'.
      EXPECT_NE(rdc_fake_hsa_alloc_size(block), 4096u) << "block " << block;
    }
    expect_everything_released();
  });
}

// A file of the kernel's name in the working directory is not loaded: that directory is not
// searched, and the check is the skip of a kernel that is not installed (F: F9 searches "./"
// first, so a file left in the daemon's working directory is loaded and run on the GPU).
TEST(RocrKernelFiles, ADecoyInTheWorkingDirectoryIsNotLoaded) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    // No directory is named to the module, so it searches the list of its own, in both builds.
    write_kernel_file("binary_search_kernels.hsaco");
    const FileIdentity decoy = identity_of("binary_search_kernels.hsaco");

    const OneGpu gpu = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_SKIP) << gpu.text;
    EXPECT_NE(gpu.text.find("No binary_search_kernels.hsaco is installed for gfx1030."),
              std::string::npos)
        << gpu.text;
    EXPECT_FALSE(a_reader_was_made_from(decoy));
    EXPECT_EQ(rdc_fake_hsa_reader_count(), 0u);
  });
}

// The file the check loads is the file the search opened, in the directory it was told to search
// and not a file of that name elsewhere (M: F9's sources have no such directory).
TEST(RocrKernelFiles, TheFileInTheConfiguredDirectoryIsTheOneLoaded) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(::mkdir("kernels", 0700), 0) << std::strerror(errno);
    write_kernel_file("kernels/binary_search_kernels.hsaco");
    write_kernel_file("binary_search_kernels.hsaco");
    search_these_directories({"kernels"});
    const FileIdentity configured = identity_of("kernels/binary_search_kernels.hsaco");
    const FileIdentity elsewhere = identity_of("binary_search_kernels.hsaco");

    const OneGpu gpu = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_NE(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    ASSERT_EQ(rdc_fake_hsa_reader_count(), 1u) << gpu.text;
    EXPECT_TRUE(a_reader_was_made_from(configured)) << gpu.text;
    EXPECT_FALSE(a_reader_was_made_from(elsewhere)) << gpu.text;
  });
#else
  GTEST_SKIP() << "the module of this build has no hook to name a directory with";
#endif
}

// The places the module looks for a kernel file, in the order it tries them: the program's
// directory, the module's own and the three trees under it, the last of them the build tree's
// kernels directory. The module is linked into this binary, so it finds no librdc_rocr.so and
// every entry but the first is passed over for want of a directory (M: F9's list is another one,
// and T4's mutations reorder it and drop its build-tree entry).
TEST(RocrKernelFiles, TheSearchListIsTriedInItsOrder) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    OneGpu gpu;
    const std::string log = run_at_debug_level(RDC_DIAG_COMPUTE_QUEUE, &gpu);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_SKIP) << gpu.text;

    const std::vector<std::string> tried = candidates_in(log);
    ASSERT_EQ(tried.size(), 5u) << log;
    EXPECT_EQ(tried[0].compare(0, 1, "/"), 0) << tried[0];
    EXPECT_NE(tried[0].find("/binary_search_kernels.hsaco:"), std::string::npos) << tried[0];
    EXPECT_EQ(tried[1],
              "<the module's directory>/binary_search_kernels.hsaco: its directory is not known.")
        << log;
    EXPECT_EQ(tried[2],
              "<the module's directory>/rdc/hsaco/gfx1030/binary_search_kernels.hsaco: its "
              "directory is not known.")
        << log;
    EXPECT_EQ(tried[3],
              "<the module's directory>/hsaco/gfx1030/binary_search_kernels.hsaco: its directory "
              "is not known.")
        << log;
    EXPECT_EQ(tried[4],
              "<the module's directory>/../../rdc_libs/rdc_modules/kernels/hsaco/gfx1030/"
              "binary_search_kernels.hsaco: its directory is not known.")
        << log;
    for (const std::string& candidate : tried) {
      EXPECT_NE(candidate.compare(0, 2, "./"), 0) << candidate;
    }
  });
#else
  GTEST_SKIP() << "the module of this build searches another list";
#endif
}

// A FIFO of the kernel file's name is passed over rather than read: the check is the skip of a
// kernel that is not installed, within this case's limit (F: F9 opens it and blocks there until
// the case is killed; T4's mutations drop O_NONBLOCK, which blocks, and the check that the file
// is a regular file, after which a reader is made from a FIFO).
TEST(RocrKernelFiles, AFifoNamedLikeTheKernelFileIsPassedOver) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(::mkfifo("binary_search_kernels.hsaco", 0600), 0) << std::strerror(errno);
    // Where the module can be told where to look, it is told to look here; where it cannot, its
    // own list has the working directory in it.
    search_these_directories({"."});

    const OneGpu gpu = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_SKIP) << gpu.text;
    EXPECT_NE(gpu.text.find("No binary_search_kernels.hsaco is installed for gfx1030."),
              std::string::npos)
        << gpu.text;
    EXPECT_EQ(rdc_fake_hsa_reader_count(), 0u);
  });
}

// A symlink that points at itself is passed over with a debug line that names the reason, and
// the kernel file in the next directory is the one loaded; the loop on its own is that GPU's
// failure, which names the reason as well (M: T4's mutations stop the search at the first errno
// that is not a name that is missing, and drop the errno the failure names).
TEST(RocrKernelFiles, ASymlinkLoopIsPassedOverAndOnItsOwnIsAFailure) {
#if RDC_ROCR_TEST_HOOKS
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    ASSERT_EQ(::mkdir("first", 0700), 0) << std::strerror(errno);
    ASSERT_EQ(::mkdir("second", 0700), 0) << std::strerror(errno);
    ASSERT_EQ(::symlink("binary_search_kernels.hsaco", "first/binary_search_kernels.hsaco"), 0)
        << std::strerror(errno);
    write_kernel_file("second/binary_search_kernels.hsaco");
    search_these_directories({"first", "second"});

    OneGpu gpu;
    const std::string log = run_at_debug_level(RDC_DIAG_COMPUTE_QUEUE, &gpu);
    EXPECT_NE(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    EXPECT_TRUE(a_reader_was_made_from(identity_of("second/binary_search_kernels.hsaco")))
        << gpu.text;
    EXPECT_NE(log.find("first/binary_search_kernels.hsaco: errno " + std::to_string(ELOOP)),
              std::string::npos)
        << log;

    // The loop alone is a file that is there and cannot be read, which is a failure, not a
    // kernel that was never installed.
    rdc_fake_hsa_reset();
    search_these_directories({"first"});
    OneGpu alone;
    // The debug lines of this run are captured as well: the logger of this child is at debug
    // level from here on, and what it writes is this case's, not the binary's output.
    run_at_debug_level(RDC_DIAG_COMPUTE_QUEUE, &alone);
    EXPECT_EQ(alone.status, RDC_DIAG_RESULT_FAIL) << alone.text;
    EXPECT_NE(alone.text.find("errno " + std::to_string(ELOOP)), std::string::npos) << alone.text;
    EXPECT_EQ(rdc_fake_hsa_reader_count(), 0u);
  });
#else
  GTEST_SKIP() << "the module of this build has no hook to name a directory with";
#endif
}

// A kernel file that is there and cannot be opened is that GPU's failure, which names the reason,
// and each check writes one error line for it (F: F9 reports the errno as an HSA status, which
// reads as another error altogether; M: T4's two mutations, one per check, write that line at
// debug level).
TEST(RocrKernelFiles, AKernelFileThatCannotBeOpenedIsThatGpusFailure) {
  RDC_ROCR_CASE({
    struct Checked {
      rdc_diag_test_cases_t check;
      const char* kernel_file;
    };
    const Checked checks[] = {{RDC_DIAG_COMPUTE_QUEUE, "binary_search_kernels.hsaco"},
                              {RDC_DIAG_SYS_MEM_CHECK, "gpuReadWrite_kernels.hsaco"}};

    for (const Checked& one : checks) {
      rdc_fake_hsa_reset();
      write_kernel_files();

      // The check runs twice with the descriptors free first, once to its Pass and once to a
      // failure of its own, and the fake is reset after them, so that the run below is the first
      // of a fresh registry. UndefinedBehaviorSanitizer reads an object's vtable through a pipe
      // to see whether the memory is there, which it cannot open once every descriptor is taken,
      // and calls the vptr invalid; a check it has already made it answers from its type cache,
      // reading nothing. The failing run is there for the check the module's catch of a
      // std::exception makes, which a run that passes never reaches.
      const OneGpu warmed = run_one_gpu(one.check);
      EXPECT_EQ(warmed.status, RDC_DIAG_RESULT_PASS) << warmed.text;
      rdc_fake_hsa_reset();
      write_kernel_files();
      rdc_fake_hsa_fail("hsa_signal_create", 0, HSA_STATUS_ERROR_OUT_OF_RESOURCES);
      const OneGpu threw = run_one_gpu(one.check);
      EXPECT_EQ(threw.status, RDC_DIAG_RESULT_FAIL) << threw.text;
      rdc_fake_hsa_reset();
      write_kernel_files();

      // A kernel file that is there and cannot be opened is an error of the run, which the
      // module writes one line for at error level, within every log level. The capture is
      // started before the descriptors are taken and read after they are given back, as reading
      // it opens a file of its own.
      ::testing::internal::CaptureStdout();

      // Every descriptor this process may open is taken, so that the open of the kernel file
      // fails with EMFILE even where the case runs as root. The limit is lowered first, so that
      // holding them takes a bounded number of descriptors.
      struct rlimit limit = {};
      ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &limit), 0) << std::strerror(errno);
      const struct rlimit original = limit;
      limit.rlim_cur = std::min<rlim_t>(limit.rlim_cur, 128);
      ASSERT_EQ(::setrlimit(RLIMIT_NOFILE, &limit), 0) << std::strerror(errno);
      std::vector<int> held;
      for (;;) {
        const int descriptor = ::open("/dev/null", O_RDONLY);
        if (descriptor < 0) {
          break;
        }
        held.push_back(descriptor);
      }
      const int refused = errno;

      const OneGpu gpu = run_one_gpu(one.check);

      for (int descriptor : held) {
        ::close(descriptor);
      }
      ASSERT_EQ(::setrlimit(RLIMIT_NOFILE, &original), 0) << std::strerror(errno);
      const std::string log = ::testing::internal::GetCapturedStdout();

      EXPECT_EQ(refused, EMFILE);
      EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
      EXPECT_NE(gpu.text.find("errno " + std::to_string(EMFILE)), std::string::npos) << gpu.text;
      EXPECT_EQ(rdc_fake_hsa_reader_count(), 0u);

      // One error line, which names the file and the errno; the kernel that is not installed is
      // a debug line, which this level does not carry.
      const std::string marker = std::string("Fail to load the kernel file ") + one.kernel_file;
      size_t error_lines = 0;
      for (size_t at = log.find(marker); at != std::string::npos; at = log.find(marker, at + 1)) {
        error_lines++;
      }
      EXPECT_EQ(error_lines, 1u) << log;
      EXPECT_NE(log.find("errno " + std::to_string(EMFILE)), std::string::npos) << log;
      expect_everything_released();
    }
  });
}

// --- the agent each sentence names --------------------------------------------------------------

// An agent whose name fills its 64 bytes with no NUL, as ROCr writes one that long, is read as
// those 64 characters and no more, from a stack the 0xA5 helper filled first, so that a read
// past the name finds no NUL: both checks' skips name those 64 characters and nothing after them
// (F: F22's sentences name the agent, where F9's Compute Queue sentence names none and its
// System memory check reads an uninitialized char[64]).
TEST(RocrNames, AnAgentNameThatFillsItsBytesIsKeptToItsLength) {
  RDC_ROCR_CASE({
    // The name ROCr wrote: the agent's name over and over, with no NUL in the 64 bytes.
    std::string name;
    while (name.size() < 64) {
      name += "gfx1030";
    }
    name.resize(64);

    struct Check {
      rdc_diag_test_cases_t check;
      const char* file;
    };
    const Check checks[] = {{RDC_DIAG_COMPUTE_QUEUE, "binary_search_kernels.hsaco"},
                            {RDC_DIAG_SYS_MEM_CHECK, "gpuReadWrite_kernels.hsaco"}};

    for (const Check& one : checks) {
      dirty_stack();
      ASSERT_TRUE(stack_dirt_is_visible()) << "the stack the check runs on is not dirty";
      rdc_fake_hsa_reset();
      rdc_fake_hsa_set_agent_name(1, "gfx1030", 0);

      const OneGpu gpu = run_one_gpu(one.check);
      EXPECT_NE(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
      // The sentence ends after the name, so nothing the stack held was read as part of it.
      EXPECT_NE(gpu.text.find("No " + std::string(one.file) + " is installed for " + name + "."),
                std::string::npos)
          << gpu.text;
      expect_everything_released();
    }
  });
}

// --- what a run writes to the console --------------------------------------------------------

// Both checks run by a daemon, which has no console: at the default log level nothing of the
// module reaches stdout or stderr, whatever the run found. This one finds no kernel file
// installed for its agent, so each check's kernel sub-test is a skip, and the fake's aliased
// pages, which map the second page of the CPU-access buffer onto the first, so that the sub-test
// reads back other values than it wrote from element 0 on: that is the GPU's failure, in its
// text, and the element is named in a debug line, which only a run at debug level writes and
// which carries the logger's own header (F: F9 logs a kernel file it cannot open at error
// level, which is within every level, and prints a mismatch with fprintf(stdout)).
TEST(RocrOutput, NothingIsWrittenToStdoutOrStderrAtTheDefaultLevel) {
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    // No kernel file is written anywhere the module looks, and the two pages of the CPU-access
    // buffer are one.
    rdc_fake_hsa_set_alias_pages(1);
    ASSERT_EQ(::unsetenv("RDC_LOG"), 0);

    ::testing::internal::CaptureStdout();
    ::testing::internal::CaptureStderr();
    const OneGpu queue_check = run_one_gpu(RDC_DIAG_COMPUTE_QUEUE);
    const size_t first_check_entries = rdc_fake_hsa_destroy_log_size();
    const OneGpu memory_check = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    const std::string out = ::testing::internal::GetCapturedStdout();
    const std::string err = ::testing::internal::GetCapturedStderr();

    EXPECT_EQ(out, "");
    EXPECT_EQ(err, "");

    // The run found both of the things F9 writes about.
    EXPECT_EQ(queue_check.status, RDC_DIAG_RESULT_SKIP) << queue_check.text;
    EXPECT_NE(queue_check.text.find("No binary_search_kernels.hsaco is installed for gfx1030."),
              std::string::npos)
        << queue_check.text;
    EXPECT_EQ(memory_check.status, RDC_DIAG_RESULT_FAIL) << memory_check.text;
    EXPECT_NE(
        memory_check.text.find("The CPU read other values from the GPU's memory than it wrote."),
        std::string::npos)
        << memory_check.text;

    // The sub-test's 8 KiB block went back to its pool before the test object gave its reference
    // to ROCr back, failure or not.
    uint32_t eight_kib = rdc_fake_hsa_alloc_count();
    for (uint32_t block = 0; block < rdc_fake_hsa_alloc_count(); block++) {
      if (rdc_fake_hsa_alloc_size(block) == 8 * 1024) {
        eight_kib = block;
      }
    }
    ASSERT_LT(eight_kib, rdc_fake_hsa_alloc_count());
    const std::vector<std::string> both = destroy_log();
    ASSERT_LE(first_check_entries, both.size());
    const std::vector<std::string> log(both.begin() + first_check_entries, both.end());
    const std::string entry = "block " + std::to_string(eight_kib);
    const std::vector<std::string>::const_iterator freed = std::find(log.begin(), log.end(), entry);
    ASSERT_NE(freed, log.end()) << joined(log);
    EXPECT_NE(std::find(freed, log.end(), "shut_down"), log.end()) << joined(log);
    expect_everything_released();
  });

  // The same run at debug level: the mismatch is in the logger's lines and nowhere else.
  RDC_ROCR_CASE({
    rdc_fake_hsa_reset();
    rdc_fake_hsa_set_alias_pages(1);
    ASSERT_EQ(::setenv("RDC_LOG", "DEBUG", 1), 0);

    ::testing::internal::CaptureStdout();
    const OneGpu gpu = run_one_gpu(RDC_DIAG_SYS_MEM_CHECK);
    const std::string log = ::testing::internal::GetCapturedStdout();

    EXPECT_EQ(gpu.status, RDC_DIAG_RESULT_FAIL) << gpu.text;
    // The first element that did not read back is element 0, as the pages are aliased from there.
    EXPECT_NE(log.find("at element 0,"), std::string::npos) << log;
    // Every line that states the mismatch is one the logger wrote, with its level in its header.
    const std::string mismatch = "The GPU\'s memory holds ";
    for (size_t at = log.find(mismatch); at != std::string::npos;
         at = log.find(mismatch, at + mismatch.size())) {
      const size_t start = log.rfind('\n', at);
      const std::string line =
          log.substr(start == std::string::npos ? 0 : start + 1, log.find('\n', at) - (start + 1));
      EXPECT_NE(line.find(" DEBUG "), std::string::npos) << line;
      EXPECT_LT(line.find(" DEBUG "), line.find(mismatch)) << line;
    }
  });
}
