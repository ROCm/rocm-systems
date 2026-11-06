// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>

#include "amd_smi/amdsmi.h"

namespace {

// A CPU-only init needs no AMD hardware: without the HSMP driver it initializes
// with no processors, which is enough to reach the argument checks.
TEST(SystemUnit, ProcessorCountFromHandlesRejectsNullOutputs) {
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_CPUS), AMDSMI_STATUS_SUCCESS);

  amdsmi_processor_handle handles[1] = {};
  uint32_t count = 0;
  uint32_t sockets = 7;
  uint32_t cores = 7;
  uint32_t gpus = 7;
  EXPECT_EQ(amdsmi_get_processor_count_from_handles(handles, &count, nullptr, &cores, &gpus),
            AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_processor_count_from_handles(handles, &count, &sockets, nullptr, &gpus),
            AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_processor_count_from_handles(handles, &count, &sockets, &cores, nullptr),
            AMDSMI_STATUS_INVAL);
  EXPECT_EQ(sockets, 7u);
  EXPECT_EQ(cores, 7u);
  EXPECT_EQ(gpus, 7u);

  // With every output present, an empty handle list classifies to zero of each type.
  EXPECT_EQ(amdsmi_get_processor_count_from_handles(handles, &count, &sockets, &cores, &gpus),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(sockets, 0u);
  EXPECT_EQ(cores, 0u);
  EXPECT_EQ(gpus, 0u);

  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

}  // namespace
