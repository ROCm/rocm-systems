// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// amdsmi_uuid_gen() must not synthesize a plausible-looking UUID out of a
// serial the driver could not supply (HSA-KMT's UniqueID sentinel). No GPU
// required.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_uuid.h"

namespace {

TEST(GpuUnit, UuidGenRejectsSentinelSerial) {
  char uuid[AMDSMI_GPU_UUID_SIZE] = {};
  amdsmi_status_t status =
      amdsmi_uuid_gen(uuid, std::numeric_limits<uint64_t>::max(), 0x1586, 0xff);
  EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, UuidGenAcceptsRealSerial) {
  char uuid[AMDSMI_GPU_UUID_SIZE] = {};
  amdsmi_status_t status = amdsmi_uuid_gen(uuid, 0x1234, 0x1586, 0xff);
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(uuid, "00ff1586-0000-1000-8000-000000001234");
}

}  // namespace
