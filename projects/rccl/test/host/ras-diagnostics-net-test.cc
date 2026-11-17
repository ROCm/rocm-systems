/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for `src/ras/diagnostics_net.cc`.

#include <string>

#include <gtest/gtest.h>

#include "comm.h"
#include "ras/diagnostics_checks_common.h"

#include RAS_DIAGNOSTICS_NET_CC_PATH

namespace {

std::string FormatDeviceType(int type) {
  struct rasDiagnosticsNetDeviceData netData {};
  netData.netDeviceType = (int8_t)type;
  char buffer[256];
  rasDiagnosticsNetFormatProps(buffer, sizeof(buffer), &netData);
  return buffer;
}

TEST(RasDiagnosticsNetMicrotest, FormatProps_NamesEveryNetDeviceTypeIncludingRcclGinBackends) {
  EXPECT_NE(FormatDeviceType(NCCL_NET_DEVICE_HOST).find("net device type=Host,"), std::string::npos);
  EXPECT_NE(FormatDeviceType(NCCL_NET_DEVICE_GIN_EFA_GDA).find("net device type=GIN EFA GDA,"), std::string::npos);
  EXPECT_NE(FormatDeviceType(NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA).find("net device type=GIN rocSHMEM GDA,"),
            std::string::npos);
  EXPECT_NE(FormatDeviceType(NCCL_NET_DEVICE_GIN_ANVIL_SDMA).find("net device type=GIN Anvil SDMA,"),
            std::string::npos);
  EXPECT_NE(FormatDeviceType(NCCL_NET_DEVICE_GIN_ANVIL_SDMA + 1).find("net device type=unknown,"), std::string::npos);
  EXPECT_NE(FormatDeviceType(-1).find("net device type=unknown,"), std::string::npos);
}

}  // namespace
