/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// abce_hsa.h's queue-ring adapter over a stubbed HSA queue table: the adapter must read as
// initialized only once every fallible query has succeeded.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "abce_hsa.h"

namespace {

// An SDMA queue in plain memory: the ring buffer and the hardware words that
// direct_queue_words drives.
struct FakeSdmaQueue {
  alignas(64) char ring[4096] = {};
  uint64_t read_word = 0;
  uint64_t write_word = 0;
  uint64_t doorbell_word = 0;
  hsa_queue_t queue = {};

  FakeSdmaQueue() {
    queue.base_address = ring;
    queue.size = sizeof(ring);
  }
};

// The queue table carries no user data, so the stub reads the queue and the injected failure
// from here.
struct StubState {
  FakeSdmaQueue* fake_queue = nullptr;
  bool fail_query = false;
  hsa_queue_info_attribute_t failing_attribute = HSA_AMD_QUEUE_INFO_ENGINE_TYPE;
};
StubState stub_state;

hsa_status_t StubQueueGetInfo(hsa_queue_t* /*queue*/, hsa_queue_info_attribute_t attribute,
                              void* value) {
  if (stub_state.fail_query && attribute == stub_state.failing_attribute)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  FakeSdmaQueue& fake_queue = *stub_state.fake_queue;
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_ENGINE_TYPE:
      *static_cast<hsa_amd_queue_engine_t*>(value) = HSA_AMD_QUEUE_ENGINE_SDMA;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_QUEUE_INFO_SDMA_ENGINE_ID:
      *static_cast<uint32_t*>(value) = 3;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_QUEUE_INFO_READ_POINTER:
      *static_cast<uint64_t*>(value) = reinterpret_cast<uintptr_t>(&fake_queue.read_word);
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_QUEUE_INFO_WRITE_POINTER:
      *static_cast<uint64_t*>(value) = reinterpret_cast<uintptr_t>(&fake_queue.write_word);
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_QUEUE_INFO_DOORBELL_ID:
      *static_cast<uint64_t*>(value) = reinterpret_cast<uintptr_t>(&fake_queue.doorbell_word);
      return HSA_STATUS_SUCCESS;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

uint64_t StubLoadIndex(const hsa_queue_t* /*queue*/) { return 0; }

void StubSignalStoreRelease(hsa_signal_t /*signal*/, hsa_signal_value_t /*value*/) {}

abce_hsa_queue_api_t StubApi() {
  abce_hsa_queue_api_t api;
  api.queue_get_info = StubQueueGetInfo;
  api.load_read_index = StubLoadIndex;
  api.load_write_index = StubLoadIndex;
  api.signal_store_release = StubSignalStoreRelease;
  return api;
}

abce_hsa_ring_options_t DirectWordOptions() {
  abce_hsa_ring_options_t options;
  abce_hsa_ring_options_initialize(&options);
  options.direct_queue_words = true;
  return options;
}

TEST(HsaQueueRing, AFailedDirectWordQueryLeavesTheAdapterUninitialized) {
  FakeSdmaQueue fake_queue;
  stub_state.fake_queue = &fake_queue;
  const abce_hsa_queue_api_t api = StubApi();
  const abce_hsa_ring_options_t options = DirectWordOptions();

  for (const hsa_queue_info_attribute_t failing_attribute :
       {HSA_AMD_QUEUE_INFO_READ_POINTER, HSA_AMD_QUEUE_INFO_WRITE_POINTER,
        HSA_AMD_QUEUE_INFO_DOORBELL_ID}) {
    SCOPED_TRACE(static_cast<int>(failing_attribute));
    stub_state.fail_query = true;
    stub_state.failing_attribute = failing_attribute;
    abce_hsa_queue_ring_t adapter;
    std::memset(&adapter, 0, sizeof(adapter));

    EXPECT_EQ(abce_hsa_queue_ring_initialize(&adapter, &fake_queue.queue, &api, &options),
              HSA_STATUS_ERROR_INVALID_QUEUE);
    EXPECT_FALSE(abce_hsa_queue_ring_initialized(&adapter));
    EXPECT_EQ(abce_hsa_queue_ring_ring(&adapter), nullptr);
    EXPECT_EQ(abce_hsa_queue_ring_drain(&adapter), ABCE_STATUS_INVALID_ARGUMENT);
  }
  stub_state = StubState();
}

TEST(HsaQueueRing, ADirectWordAdapterInitializesOnceEveryQuerySucceeds) {
  FakeSdmaQueue fake_queue;
  stub_state.fake_queue = &fake_queue;
  const abce_hsa_queue_api_t api = StubApi();
  const abce_hsa_ring_options_t options = DirectWordOptions();
  abce_hsa_queue_ring_t adapter;
  std::memset(&adapter, 0, sizeof(adapter));

  ASSERT_EQ(abce_hsa_queue_ring_initialize(&adapter, &fake_queue.queue, &api, &options),
            HSA_STATUS_SUCCESS);
  EXPECT_TRUE(abce_hsa_queue_ring_initialized(&adapter));
  EXPECT_NE(abce_hsa_queue_ring_ring(&adapter), nullptr);
  EXPECT_EQ(adapter.engine_id, 3u);
  EXPECT_EQ(abce_hsa_queue_ring_drain(&adapter), ABCE_STATUS_OK);
  stub_state = StubState();
}

}  // namespace
