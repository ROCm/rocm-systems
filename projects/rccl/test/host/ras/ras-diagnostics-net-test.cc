/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for `src/ras/diagnostics_net.cc`.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include "comm.h"
#include "fakes/os_fakes.h"
#include "fakes/ras_diagnostics_test_support.h"
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

// NCCL 2.32.6 (53f436bb6d39): the speed stays the plugin's. On HW multiplane NICs the sysfs "rate" reports one plane.
TEST(RasDiagnosticsNetMicrotest, ReadPortAttrs_KeepsPluginSpeedOverSysfsRate) {
  char root[] = "/tmp/rccl-ras-net-XXXXXX";
  ASSERT_NE(nullptr, mkdtemp(root));
  const std::string ports = std::string(root) + "/mlx5_0/ports";
  ASSERT_EQ(0, mkdir((std::string(root) + "/mlx5_0").c_str(), 0700));
  ASSERT_EQ(0, mkdir(ports.c_str(), 0700));
  ASSERT_EQ(0, mkdir((ports + "/1").c_str(), 0700));
  FILE* state = fopen((ports + "/1/state").c_str(), "w");  // rasDiagnosticsNetHasOnePort() probes it with access()
  ASSERT_NE(nullptr, state);
  fclose(state);
  const auto savedSysfs = g_ncclOsTopoGetStrFromSys;
  g_ncclOsTopoGetStrFromSys = [](const char*, const char* file, char* value, int maxLen) {
    const char* answer = strcmp(file, "link_layer") == 0   ? "InfiniBand"
                         : strcmp(file, "state") == 0      ? "4: ACTIVE"
                         : strcmp(file, "phys_state") == 0 ? "5: LinkUp"
                         : strcmp(file, "rate") == 0       ? "200 Gb/sec (4X HDR)"
                                                           : "";
    snprintf(value, maxLen, "%s", answer);
    return ncclSuccess;
  };

  struct rasDiagnosticsNetDeviceData netData {};
  snprintf(netData.name, sizeof(netData.name), "mlx5_0");
  netData.speed = 400000;
  rasDiagnosticsNetReadPortAttrs(&netData, root);
  g_ncclOsTopoGetStrFromSys = savedSysfs;
  unlink((ports + "/1/state").c_str());
  rmdir((ports + "/1").c_str());
  rmdir(ports.c_str());
  rmdir((std::string(root) + "/mlx5_0").c_str());
  rmdir(root);

  EXPECT_EQ(1, netData.port);
  EXPECT_EQ(RAS_DIAG_NET_LINK_INFINIBAND, netData.linkLayer);
  EXPECT_EQ(4, netData.portState);
  EXPECT_EQ(400000, netData.speed) << "the sysfs rate replaced the plugin's port speed";
}

// Two nodes, one rank each, both reporting a healthy mlx5_0; mlx5_1 is healthy too, on one node or on both.
std::vector<std::string> SummarizeTwoNodes(bool secondDeviceOnBothNodes) {
  std::vector<std::pair<rasDiagnosticsRankHeader, rasDiagnosticsNetDeviceData>> records;
  for (int rank = 0; rank < 2; rank++) {
    for (int dev = 0; dev < (rank == 0 || secondDeviceOnBothNodes ? 2 : 1); dev++) {
      rasDiagnosticsRankHeader header{};
      header.commId.commHash = 0x4000;
      header.commRank = rank;
      header.commNRanks = 2;
      rasDiagnosticsNetDeviceData netData{};
      netData.hostHash = 0x100 + rank;
      netData.speed = 400000;
      netData.port = 1;
      netData.linkLayer = RAS_DIAG_NET_LINK_INFINIBAND;
      netData.portState = 4;
      netData.physState = 5;
      snprintf(netData.name, sizeof(netData.name), "mlx5_%d", dev);
      records.emplace_back(header, netData);
    }
  }
  const std::vector<char> data = ras_test::BuildRecords(records);
  ras_test::ReporterState state;
  const rasDiagnosticsReporter reporter = ras_test::MakeReporter(&state);
  rasDiagnosticsContext ctx{};
  EXPECT_EQ(ncclSuccess, rasDiagnosticsNetDeviceSummarize(&ctx, &reporter, data.data(), (int)data.size()));
  return state.lines;
}

// NCCL 2.32.6 (6db1ae38879e): healthy NICs go to NCCL_DEBUG_SUBSYS=DIAG; the report holds one summary line.
TEST(RasDiagnosticsNetMicrotest, Summarize_HealthyNicsYieldOneSummaryLine) {
  const std::vector<std::string> lines = SummarizeTwoNodes(/*secondDeviceOnBothNodes=*/true);
  ASSERT_EQ(1u, lines.size()) << (lines.empty() ? "" : lines[0]);
  EXPECT_EQ("[OK]   Net device: NIC configuration consistent across 2 nodes in comm 0x4000", lines[0]);
}

// A failing NIC is reported on its own; the healthy one and the "consistent" summary are not.
TEST(RasDiagnosticsNetMicrotest, Summarize_ReportsOnlyFailingNics) {
  const std::vector<std::string> lines = SummarizeTwoNodes(/*secondDeviceOnBothNodes=*/false);
  ASSERT_EQ(1u, lines.size()) << (lines.empty() ? "" : lines[0]);
  EXPECT_EQ(0u, lines[0].find("[INFO] Net device mlx5_1:1: ")) << lines[0];
  EXPECT_NE(std::string::npos, lines[0].find("reported on only 1 of 2 nodes")) << lines[0];
}

}  // namespace
