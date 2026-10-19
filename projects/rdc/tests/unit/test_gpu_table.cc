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

// The flat GPU table in SmiUtils.cc and RDC's AMD SMI session, against the fake
// AMD SMI functions below: a slot per GPU, a placeholder for a processor AMD SMI
// cannot identify, the retries that fill placeholders in place, and the
// session's reference count. SmiUtils.cc is compiled into this binary, so it
// needs neither librdc nor a GPU.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace {

using amd::rdc::get_flat_gpu_table;
using amd::rdc::get_kfd_node_id;
using amd::rdc::get_kfd_partition_id;
using amd::rdc::get_processor_handle_from_id;
using amd::rdc::reset_flat_gpu_table;
using amd::rdc::set_flat_gpu_table_retry_period;
using amd::rdc::SmiSession;
using std::chrono::milliseconds;

// A processor of the fake system; its handle is its id.
struct FakeProcessor {
  uintptr_t id;
  processor_type_t type;
  bool type_fails;
};

// g_sockets[s]: the processors of socket s, in AMD SMI's order.
std::vector<std::vector<FakeProcessor>> g_sockets;
bool g_socket_list_fails = false;
bool g_socket_list_throws = false;
int g_listings = 0;  // amdsmi_get_socket_handles() count queries
int g_inits = 0;
int g_shutdowns = 0;
amdsmi_status_t g_init_status = AMDSMI_STATUS_SUCCESS;
bool g_init_fails_with_cpus = false;
uint64_t g_init_flags = 0;          // of the last amdsmi_init() call
uint32_t g_profile_partitions = 0;  // the partition count AMD SMI reports; 0: not supported
// What amdsmi_get_gpu_kfd_info answers: a node id per processor id (its own id plus 100 when the
// case names none), and a status it fails with.
std::map<uintptr_t, uint32_t> g_kfd_nodes;
amdsmi_status_t g_kfd_status = AMDSMI_STATUS_SUCCESS;

FakeProcessor gpu(uintptr_t id) { return {id, AMDSMI_PROCESSOR_TYPE_AMD_GPU, false}; }
FakeProcessor cpu(uintptr_t id) { return {id, AMDSMI_PROCESSOR_TYPE_AMD_CPU, false}; }
FakeProcessor unknown(uintptr_t id) { return {id, AMDSMI_PROCESSOR_TYPE_AMD_GPU, true}; }

// One GPU per socket, ids 1..n.
std::vector<std::vector<FakeProcessor>> gpus(uintptr_t n) {
  std::vector<std::vector<FakeProcessor>> sockets;
  for (uintptr_t i = 1; i <= n; i++) {
    sockets.push_back({gpu(i)});
  }
  return sockets;
}

const FakeProcessor* find_processor(amdsmi_processor_handle handle) {
  for (const auto& socket : g_sockets) {
    for (const auto& proc : socket) {
      if (reinterpret_cast<amdsmi_processor_handle>(proc.id) == handle) {
        return &proc;
      }
    }
  }
  return nullptr;
}

amdsmi_processor_handle handle_of(uintptr_t id) {
  return reinterpret_cast<amdsmi_processor_handle>(id);
}

// The encoded entity index of a partition instance (socket, proc).
uint32_t PartitionEntityIndex(uint32_t socket, uint32_t proc) {
  rdc_entity_info_t info = {};
  info.device_index = socket;
  info.instance_index = proc;
  info.entity_role = RDC_DEVICE_ROLE_PARTITION_INSTANCE;
  info.device_type = RDC_DEVICE_TYPE_GPU;
  return rdc_get_entity_index_from_info(info);
}

// The handle RDC GPU `index` resolves to, or nullptr with the status in *status.
amdsmi_processor_handle lookup(uint32_t index, amdsmi_status_t* status = nullptr) {
  amdsmi_processor_handle handle = nullptr;
  amdsmi_status_t ret = get_processor_handle_from_id(index, &handle);
  if (status != nullptr) {
    *status = ret;
  }
  return ret == AMDSMI_STATUS_SUCCESS ? handle : nullptr;
}

std::vector<bool> availability() {
  std::vector<bool> out;
  for (const auto& entry : *get_flat_gpu_table()) {
    out.push_back(entry.available);
  }
  return out;
}

class GpuTable : public ::testing::Test {
 protected:
  void SetUp() override {
    g_socket_list_fails = false;
    g_socket_list_throws = false;
    g_listings = 0;
    g_inits = 0;
    g_shutdowns = 0;
    g_init_status = AMDSMI_STATUS_SUCCESS;
    g_init_fails_with_cpus = false;
    g_init_flags = 0;
    g_profile_partitions = 0;
    g_kfd_nodes.clear();
    g_kfd_status = AMDSMI_STATUS_SUCCESS;
    set_flat_gpu_table_retry_period(milliseconds(0));  // retry at every use
    reset_flat_gpu_table();
  }
  void TearDown() override { reset_flat_gpu_table(); }
};

}  // namespace

// Fakes for the AMD SMI calls in SmiUtils.cc.
amdsmi_status_t amdsmi_get_socket_handles(uint32_t* socket_count,
                                          amdsmi_socket_handle* socket_handles) {
  if (g_socket_list_throws) {
    throw std::runtime_error("listing failed");
  }
  if (g_socket_list_fails) {
    return AMDSMI_STATUS_UNKNOWN_ERROR;
  }
  if (socket_handles == nullptr) {
    g_listings++;
    *socket_count = static_cast<uint32_t>(g_sockets.size());
    return AMDSMI_STATUS_SUCCESS;
  }
  uint32_t n = std::min<uint32_t>(*socket_count, static_cast<uint32_t>(g_sockets.size()));
  for (uint32_t s = 0; s < n; s++) {
    socket_handles[s] = reinterpret_cast<amdsmi_socket_handle>(static_cast<uintptr_t>(s + 1));
  }
  *socket_count = n;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle socket_handle,
                                             uint32_t* processor_count,
                                             amdsmi_processor_handle* processor_handles) {
  size_t s = reinterpret_cast<uintptr_t>(socket_handle) - 1;
  if (s >= g_sockets.size()) {
    return AMDSMI_STATUS_INVAL;
  }
  const auto& procs = g_sockets[s];
  if (processor_handles == nullptr) {
    *processor_count = static_cast<uint32_t>(procs.size());
    return AMDSMI_STATUS_SUCCESS;
  }
  uint32_t n = std::min<uint32_t>(*processor_count, static_cast<uint32_t>(procs.size()));
  for (uint32_t p = 0; p < n; p++) {
    processor_handles[p] = handle_of(procs[p].id);
  }
  *processor_count = n;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle processor_handle,
                                          processor_type_t* processor_type) {
  const FakeProcessor* proc = find_processor(processor_handle);
  if (proc == nullptr) {
    return AMDSMI_STATUS_NOT_FOUND;
  }
  if (proc->type_fails) {
    return AMDSMI_STATUS_UNKNOWN_ERROR;
  }
  *processor_type = proc->type;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_init(uint64_t init_flags) {
  g_inits++;
  g_init_flags = init_flags;
  if (g_init_fails_with_cpus && (init_flags & AMDSMI_INIT_AMD_CPUS) != 0) {
    return AMDSMI_STATUS_INIT_ERROR;
  }
  return g_init_status;
}

amdsmi_status_t amdsmi_shut_down() {
  g_shutdowns++;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_accelerator_partition_profile(
    amdsmi_processor_handle, amdsmi_accelerator_partition_profile_t* profile,
    uint32_t* partition_id) {
  if (g_profile_partitions == 0) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  *profile = {};
  profile->num_partitions = g_profile_partitions;
  *partition_id = 0;
  return AMDSMI_STATUS_SUCCESS;
}

// The KFD information of a processor, which get_kfd_node_id() and get_kfd_partition_id() read.
amdsmi_status_t amdsmi_get_gpu_kfd_info(amdsmi_processor_handle processor_handle,
                                        amdsmi_kfd_info_t* info) {
  if (g_kfd_status != AMDSMI_STATUS_SUCCESS) {
    return g_kfd_status;
  }
  const FakeProcessor* proc = find_processor(processor_handle);
  if (proc == nullptr) {
    return AMDSMI_STATUS_NOT_FOUND;
  }
  const auto known = g_kfd_nodes.find(proc->id);
  *info = {};
  info->kfd_id = proc->id;
  info->node_id =
      known == g_kfd_nodes.end() ? static_cast<uint32_t>(proc->id) + 100 : known->second;
  info->current_partition_id = 0;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_get_gpu_metrics_info(amdsmi_processor_handle, amdsmi_gpu_metrics_t*) {
  return AMDSMI_STATUS_NOT_SUPPORTED;
}

TEST_F(GpuTable, UnidentifiedGpuKeepsItsIndex) {
  g_sockets = gpus(4);
  g_sockets[1][0].type_fails = true;
  EXPECT_EQ(availability(), (std::vector<bool>{true, false, true, true}));
  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
  EXPECT_EQ(lookup(1, &status), nullptr);
  EXPECT_EQ(status, AMDSMI_STATUS_NOT_FOUND);
  EXPECT_EQ(lookup(0), handle_of(1));
  EXPECT_EQ(lookup(2), handle_of(3));
  EXPECT_EQ(lookup(3), handle_of(4));
}

TEST_F(GpuTable, UnidentifiedFirstGpuKeepsItsIndex) {
  g_sockets = gpus(4);
  g_sockets[0][0].type_fails = true;
  EXPECT_EQ(availability(), (std::vector<bool>{false, true, true, true}));
  EXPECT_EQ(lookup(0), nullptr);
  EXPECT_EQ(lookup(1), handle_of(2));
  EXPECT_EQ(lookup(3), handle_of(4));
}

TEST_F(GpuTable, UnidentifiedProcessorNextToCpusIsLeftOut) {
  // Socket 1 is a CPU socket; socket 2's only processor could be anything.
  g_sockets = {{gpu(1)}, {cpu(10), unknown(11)}, {unknown(20)}, {gpu(2)}};
  EXPECT_EQ(availability(), (std::vector<bool>{true, true}));
  EXPECT_EQ(lookup(0), handle_of(1));
  EXPECT_EQ(lookup(1), handle_of(2));
}

TEST_F(GpuTable, UnidentifiedPartitionOfAGpuIsAPlaceholder) {
  g_sockets = {{gpu(1), unknown(2), gpu(3)}, {cpu(10)}};
  EXPECT_EQ(availability(), (std::vector<bool>{true, false, true}));
  EXPECT_EQ(lookup(2), handle_of(3));
}

TEST_F(GpuTable, RetryFillsThePlaceholderInPlace) {
  g_sockets = gpus(4);
  g_sockets[1][0].type_fails = true;
  ASSERT_EQ(availability(), (std::vector<bool>{true, false, true, true}));
  g_sockets[1][0].type_fails = false;
  EXPECT_EQ(lookup(1), handle_of(2));
  EXPECT_EQ(availability(), (std::vector<bool>(4, true)));
  EXPECT_EQ(lookup(2), handle_of(3));
}

TEST_F(GpuTable, RetryStopsForAProcessorOfAnotherKind) {
  g_sockets = gpus(3);
  g_sockets[1][0].type_fails = true;
  ASSERT_EQ(availability(), (std::vector<bool>{true, false, true}));
  g_sockets[1][0] = cpu(2);
  lookup(0);  // retries: the processor is a CPU
  int listings = g_listings;
  lookup(0);
  lookup(1);
  EXPECT_EQ(g_listings, listings) << "the table is listed again with nothing left to fill";
  EXPECT_EQ(availability(), (std::vector<bool>{true, false, true}));
}

TEST_F(GpuTable, GpuNoLongerListedBecomesUnavailable) {
  g_sockets = gpus(4);
  g_sockets[0][0].type_fails = true;  // keeps the table retrying
  ASSERT_EQ(availability(), (std::vector<bool>{false, true, true, true}));
  g_sockets.pop_back();
  EXPECT_EQ(lookup(3), nullptr);
  EXPECT_EQ(availability(), (std::vector<bool>{false, true, true, false}));
}

TEST_F(GpuTable, RetryNeverGivesOneHandleToTwoIndexes) {
  g_sockets = {{gpu(1)}, {unknown(2)}, {gpu(3)}};
  ASSERT_EQ(availability(), (std::vector<bool>{true, false, true}));
  g_sockets[1][0] = gpu(3);  // the listing now names GPU 3 at GPU 1's place
  EXPECT_EQ(lookup(1), nullptr);
  EXPECT_EQ(lookup(2), handle_of(3));
}

TEST_F(GpuTable, RetriesAtMostOncePerPeriod) {
  set_flat_gpu_table_retry_period(milliseconds(60000));
  g_sockets = gpus(4);
  g_sockets[1][0].type_fails = true;
  ASSERT_EQ(availability(), (std::vector<bool>{true, false, true, true}));
  int listings = g_listings;
  for (uint32_t i = 0; i < 10; i++) {
    lookup(i % 4);
  }
  EXPECT_EQ(g_listings, listings);
  set_flat_gpu_table_retry_period(milliseconds(0));
  lookup(0);
  EXPECT_EQ(g_listings, listings + 1);
}

TEST_F(GpuTable, FailedFirstListingIsRetried) {
  g_sockets = gpus(2);
  g_socket_list_fails = true;
  EXPECT_TRUE(get_flat_gpu_table()->empty());
  g_socket_list_fails = false;
  EXPECT_EQ(lookup(1), handle_of(2));
  EXPECT_EQ(get_flat_gpu_table()->size(), 2u);
}

TEST_F(GpuTable, AtMostMaxDevicesSlots) {
  g_sockets = gpus(RDC_MAX_NUM_DEVICES + 2);
  EXPECT_EQ(get_flat_gpu_table()->size(), static_cast<size_t>(RDC_MAX_NUM_DEVICES));
}

TEST_F(GpuTable, GpuIdOfAHandleIsItsSlot) {
  // A CPU socket, a processor that could be a CPU (left out), then two GPU sockets.
  g_sockets = {{cpu(10)}, {unknown(20)}, {gpu(1)}, {gpu(2), unknown(3)}};
  ASSERT_EQ(availability(), (std::vector<bool>{true, true, false}));
  uint32_t index = 99;
  ASSERT_EQ(amd::rdc::get_gpu_id_from_processor_handle(handle_of(1), &index),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(index, 0u);
  ASSERT_EQ(amd::rdc::get_gpu_id_from_processor_handle(handle_of(2), &index),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(index, 1u);
  EXPECT_EQ(amd::rdc::get_gpu_id_from_processor_handle(handle_of(10), &index),
            AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS);
  EXPECT_EQ(amd::rdc::get_gpu_id_from_processor_handle(handle_of(3), &index),
            AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS);
}

TEST_F(GpuTable, PartitionCountFallbackCountsPlaceholders) {
  // With no valid count from AMD SMI, a GPU has as many partitions as its socket has slots.
  g_sockets = {{gpu(1), unknown(2), gpu(3)}, {gpu(4)}};
  g_profile_partitions = UINT32_MAX;
  uint16_t partitions = 0;
  ASSERT_EQ(amd::rdc::get_num_partition(0, &partitions), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(partitions, 3);
  ASSERT_EQ(amd::rdc::get_num_partition(3, &partitions), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(partitions, 1);
}

TEST_F(GpuTable, SessionCountsItsReferences) {
  g_sockets = gpus(2);
  {
    SmiSession first;
    EXPECT_EQ(g_inits, 1);
    {
      SmiSession second;
      EXPECT_EQ(g_inits, 1);
    }
    EXPECT_EQ(g_shutdowns, 0);
  }
  EXPECT_EQ(g_shutdowns, 1);
}

TEST_F(GpuTable, SessionInitFailureThrowsAndLeavesNoReference) {
  g_sockets = gpus(2);
  g_init_status = AMDSMI_STATUS_INIT_ERROR;
  EXPECT_THROW(SmiSession session, amd::rdc::RdcException);
  EXPECT_EQ(g_shutdowns, 0);
  g_init_status = AMDSMI_STATUS_SUCCESS;
  int inits = g_inits;
  {
    SmiSession session;
    EXPECT_EQ(g_inits, inits + 1) << "the failed session left a reference behind";
  }
  EXPECT_EQ(g_shutdowns, 1);
}

TEST_F(GpuTable, SessionFallsBackToGpusOnly) {
  // Without the CPU library, RDC asks for GPUs only; with it, it retries without CPUs.
  g_sockets = gpus(2);
  g_init_fails_with_cpus = true;
  {
    SmiSession session;
    EXPECT_EQ(g_init_flags & AMDSMI_INIT_AMD_CPUS, 0u);
    EXPECT_EQ(lookup(1), handle_of(2));
  }
  EXPECT_EQ(g_shutdowns, 1);
}

TEST_F(GpuTable, SessionReleasesAmdSmiWhenListingThrows) {
  g_sockets = gpus(2);
  g_socket_list_throws = true;
  EXPECT_THROW(SmiSession session, std::runtime_error);
  EXPECT_EQ(g_shutdowns, 1) << "the reference amdsmi_init() took leaked";
  g_socket_list_throws = false;
  int inits = g_inits;
  {
    SmiSession session;
    EXPECT_EQ(g_inits, inits + 1);
    EXPECT_EQ(lookup(1), handle_of(2));
  }
  EXPECT_EQ(g_shutdowns, 2);
}

TEST_F(GpuTable, SessionListsTheGpusAgain) {
  g_sockets = gpus(2);
  ASSERT_EQ(lookup(1), handle_of(2));
  // AMD SMI was shut down and initialized again: every handle changed.
  g_sockets = {{gpu(101)}, {gpu(102)}};
  {
    SmiSession session;
    EXPECT_EQ(lookup(1), handle_of(102));
  }
  g_sockets = {{gpu(201)}, {gpu(202)}};
  EXPECT_EQ(lookup(1), handle_of(202)) << "the table outlived the last session";
}

// The KFD node of a GPU, which is what the ROCr diagnostic module finds its agent by.
TEST_F(GpuTable, KfdNodeOfAFlatId) {
  g_sockets = gpus(3);
  g_kfd_nodes[2] = 7;
  uint32_t node = 0;
  EXPECT_EQ(get_kfd_node_id(1, &node), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(node, 7u);
}

TEST_F(GpuTable, KfdNodeOfAPartitionId) {
  g_sockets = {{gpu(1), gpu(2)}, {gpu(3)}};
  g_kfd_nodes[2] = 11;
  uint32_t node = 0;
  EXPECT_EQ(get_kfd_node_id(PartitionEntityIndex(0, 1), &node), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(node, 11u);
}

TEST_F(GpuTable, PartitionsOfOneSocketHaveTheirOwnNodes) {
  g_sockets = {{gpu(1), gpu(2)}};
  g_kfd_nodes[1] = 4;
  g_kfd_nodes[2] = 5;
  uint32_t first = 0;
  uint32_t second = 0;
  EXPECT_EQ(get_kfd_node_id(PartitionEntityIndex(0, 0), &first), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(get_kfd_node_id(PartitionEntityIndex(0, 1), &second), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(first, 4u);
  EXPECT_EQ(second, 5u);
}

// A GPU AMD SMI cannot identify is one RDC cannot name a node for.
TEST_F(GpuTable, AnUnavailableGpuHasNoKfdNode) {
  g_sockets = gpus(3);
  g_sockets[1][0].type_fails = true;
  uint32_t node = 9;
  EXPECT_EQ(get_kfd_node_id(1, &node), AMDSMI_STATUS_NOT_FOUND);
  EXPECT_EQ(node, 9u);
}

TEST_F(GpuTable, AFlatIdPastTheTableHasNoKfdNode) {
  g_sockets = gpus(2);
  uint32_t node = 0;
  EXPECT_EQ(get_kfd_node_id(5, &node), AMDSMI_STATUS_NOT_FOUND);
}

TEST_F(GpuTable, APartitionIdOutOfRangeHasNoKfdNode) {
  g_sockets = {{gpu(1), gpu(2)}};
  uint32_t node = 0;
  EXPECT_EQ(get_kfd_node_id(PartitionEntityIndex(3, 0), &node), AMDSMI_STATUS_NOT_FOUND);
  EXPECT_EQ(get_kfd_node_id(PartitionEntityIndex(0, 4), &node), AMDSMI_STATUS_NOT_FOUND);
}

// An AMD SMI that cannot report KFD information is told apart from an id it does not know, so
// that each gets its own sentence in the check's result.
TEST_F(GpuTable, KfdInfoThatIsNotSupportedIsNotSupported) {
  g_sockets = gpus(2);
  g_kfd_status = AMDSMI_STATUS_NOT_SUPPORTED;
  uint32_t node = 0;
  EXPECT_EQ(get_kfd_node_id(0, &node), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST_F(GpuTable, AFailingKfdInfoCallGivesItsStatus) {
  g_sockets = gpus(2);
  g_kfd_status = AMDSMI_STATUS_UNKNOWN_ERROR;
  uint32_t node = 0;
  EXPECT_EQ(get_kfd_node_id(0, &node), AMDSMI_STATUS_UNKNOWN_ERROR);
}

TEST_F(GpuTable, AKfdInfoCallThatFindsNothingIsNotSupported) {
  g_sockets = gpus(2);
  g_kfd_status = AMDSMI_STATUS_NOT_FOUND;
  uint32_t node = 0;
  EXPECT_EQ(get_kfd_node_id(0, &node), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST_F(GpuTable, ANodeIdAmdSmiCannotReadIsNotSupported) {
  g_sockets = gpus(2);
  g_kfd_nodes[1] = 0xFFFFFFFF;
  uint32_t node = 0;
  EXPECT_EQ(get_kfd_node_id(0, &node), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(node, 0u);
}

TEST_F(GpuTable, ANullNodeIdIsRefused) {
  g_sockets = gpus(2);
  EXPECT_EQ(get_kfd_node_id(0, nullptr), AMDSMI_STATUS_INVAL);
}

TEST_F(GpuTable, ANullPartitionIdIsRefused) {
  g_sockets = gpus(2);
  EXPECT_EQ(get_kfd_partition_id(handle_of(1), nullptr), AMDSMI_STATUS_INVAL);
}
