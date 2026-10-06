/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/signal_create_v2.h"

#include <cstring>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"

SignalCreateV2Test::SignalCreateV2Test() : TestBase() {
  set_num_iteration(1);
  set_title("RocR hsa_amd_signal_create_v2 Test");
  set_description("Checks hsa_amd_signal_create_v2's descriptor validation, partial batch "
                  "contract and device resident value word.");
}

SignalCreateV2Test::~SignalCreateV2Test(void) {}

void SignalCreateV2Test::SetUp(void) {
  TestBase::SetUp();
  if (test_skipped_) return;
  ASSERT_EQ(HSA_STATUS_SUCCESS, rocrtst::SetDefaultAgents(this));
}

void SignalCreateV2Test::Run(void) { TestBase::Run(); }

void SignalCreateV2Test::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void SignalCreateV2Test::DisplayResults(void) const {}

void SignalCreateV2Test::Close() { TestBase::Close(); }

bool SignalCreateV2Test::DeviceResidentSupported() {
  bool supported = false;
  const hsa_status_t err = hsa_agent_get_info(
      *gpu_device1(),
      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ORDERING_EDGE_SIGNAL_SUPPORTED),
      &supported);
  return err == HSA_STATUS_SUCCESS && supported;
}

hsa_amd_signal_create_desc_t SignalCreateV2Test::DeviceDesc() {
  hsa_amd_signal_create_desc_t desc;
  memset(&desc, 0, sizeof(desc));
  desc.version = HSA_AMD_SIGNAL_CREATE_DESC_VERSION;
  desc.flags = HSA_AMD_SIGNAL_CREATE_DEVICE_MEM_VALUE_WORD;
  desc.initial_value = 1;
  desc.num_consumers = 1;
  desc.consumers = gpu_device1();
  return desc;
}

void SignalCreateV2Test::TestRejections(void) {
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, hsa_amd_signal_create_v2(nullptr, 1));
  hsa_amd_signal_create_desc_t desc = DeviceDesc();
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, hsa_amd_signal_create_v2(&desc, 0));

  struct Case {
    const char* name;
    void (*mutate)(hsa_amd_signal_create_desc_t*, hsa_agent_t*);
    hsa_status_t expect;
  };
  const Case cases[] = {
      {"unknown version", [](auto* d, auto*) { d->version += 1; },
       HSA_STATUS_ERROR_INVALID_ARGUMENT},
      {"reserved_header", [](auto* d, auto*) { d->reserved_header[0] = 1; },
       HSA_STATUS_ERROR_INVALID_ARGUMENT},
      {"reserved_count", [](auto* d, auto*) { d->reserved_count = 1; },
       HSA_STATUS_ERROR_INVALID_ARGUMENT},
      {"reserved", [](auto* d, auto*) { d->reserved[15] = 1; },
       HSA_STATUS_ERROR_INVALID_ARGUMENT},
      {"undefined flag", [](auto* d, auto*) { d->flags |= 0x8000; },
       HSA_STATUS_ERROR_INVALID_ARGUMENT},
      {"undefined attribute", [](auto* d, auto*) { d->attributes |= 1ull << 63; },
       HSA_STATUS_ERROR_INVALID_ARGUMENT},
      {"IPC", [](auto* d, auto*) { d->attributes |= HSA_AMD_SIGNAL_IPC; },
       HSA_STATUS_ERROR_INVALID_ARGUMENT},
      {"no consumer", [](auto* d, auto*) { d->num_consumers = 0; },
       HSA_STATUS_ERROR_INVALID_ARGUMENT},
      {"two consumers", [](auto* d, auto*) { d->num_consumers = 2; },
       HSA_STATUS_ERROR_INVALID_ARGUMENT},
      {"CPU consumer", [](auto* d, auto* cpu) { d->consumers = cpu; },
       HSA_STATUS_ERROR_INVALID_AGENT},
  };
  for (const auto& c : cases) {
    desc = DeviceDesc();
    c.mutate(&desc, cpu_device());
    desc.signal.handle = 0xdead;
    EXPECT_EQ(c.expect, hsa_amd_signal_create_v2(&desc, 1)) << c.name;
    EXPECT_EQ(0u, desc.signal.handle) << c.name;
  }
}

void SignalCreateV2Test::TestPartialBatch(void) {
  if (!DeviceResidentSupported()) {
    rocrtst::SkipCurrentTest("GPU cannot host a device resident signal value word");
    return;
  }
  hsa_amd_signal_create_desc_t descs[3] = {DeviceDesc(), DeviceDesc(), DeviceDesc()};
  descs[1].version += 1;
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, hsa_amd_signal_create_v2(descs, 3));
  EXPECT_NE(0u, descs[0].signal.handle);
  EXPECT_EQ(0u, descs[1].signal.handle);
  EXPECT_NE(0u, descs[2].signal.handle);
  for (const auto& d : descs) {
    if (d.signal.handle != 0) EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_destroy(d.signal));
  }
}

void SignalCreateV2Test::TestDeviceResident(void) {
  if (!DeviceResidentSupported()) {
    rocrtst::SkipCurrentTest("GPU cannot host a device resident signal value word");
    return;
  }
  hsa_amd_signal_create_desc_t desc = DeviceDesc();
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_signal_create_v2(&desc, 1));
  const hsa_signal_t signal = desc.signal;
  ASSERT_NE(0u, signal.handle);

  // The handle is the address of the ABI block, which must be owned by the consumer GPU.
  hsa_amd_pointer_info_t info;
  memset(&info, 0, sizeof(info));
  info.size = sizeof(info);
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_pointer_info(reinterpret_cast<void*>(signal.handle), &info,
                                                     nullptr, nullptr, nullptr));
  EXPECT_EQ(gpu_device1()->handle, info.agentOwner.handle);

  EXPECT_EQ(1, hsa_signal_load_relaxed(signal));
  hsa_signal_store_relaxed(signal, 7);
  EXPECT_EQ(7, hsa_signal_load_scacquire(signal));
  hsa_signal_store_screlease(signal, 0);
  EXPECT_EQ(0, hsa_signal_load_relaxed(signal));

  // The pointer's documented use is a host atomic update, which is not atomic on this word.
  volatile hsa_signal_value_t* ptr = nullptr;
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_SIGNAL, hsa_amd_signal_value_pointer(signal, &ptr));

  EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_signal_destroy(signal));
}
