/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for `src/ras/diagnostics_net.cc`.

#include <string>
#include <vector>

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

std::vector<std::string> g_lines;

ncclResult_t RecordingEmit(void* /*target*/, const char* line) {
  g_lines.emplace_back(line);
  return ncclSuccess;
}

bool AnyLineContains(const std::string& needle) {
  for (const std::string& line : g_lines)
    if (line.find(needle) != std::string::npos) return true;
  return false;
}

struct NetRecord {
  int rank;
  uint64_t hostHash;
  const char* name;
  int speed;
};

// One record per entry, all in one 4-rank comm, every port InfiniBand and ACTIVE/LinkUp.
std::vector<char> MakeRecords(const std::vector<NetRecord>& entries) {
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNetDeviceData));
  std::vector<char> records(entries.size() * stride, 0);
  for (size_t i = 0; i < entries.size(); i++) {
    auto* header = reinterpret_cast<struct rasDiagnosticsRankHeader*>(records.data() + i * stride);
    header->commId = {0x1234, 0x10, 0x20};
    header->commRank = entries[i].rank;
    header->commNRanks = 4;
    auto* netData = reinterpret_cast<struct rasDiagnosticsNetDeviceData*>(records.data() + i * stride + sizeof(*header));
    netData->hostHash = entries[i].hostHash;
    netData->speed = entries[i].speed;
    netData->port = 1;
    netData->linkLayer = RAS_DIAG_NET_LINK_INFINIBAND;
    netData->portState = 4;
    netData->physState = 5;
    snprintf(netData->name, sizeof(netData->name), "%s", entries[i].name);
  }
  return records;
}

ncclResult_t Summarize(const std::vector<char>& records) {
  rasDiagnosticsReporter reporter{};
  reporter.emit = RecordingEmit;
  g_lines.clear();
  return rasDiagnosticsNetDeviceSummarize(nullptr, &reporter, records.data(), (int)records.size());
}

TEST(RasDiagnosticsNetMicrotest, Summarize_ReportsConsistentNicsAcrossNodes) {
  ASSERT_EQ(ncclSuccess, Summarize(MakeRecords({{0, 1, "mlx5_0", 400000}, {1, 1, "mlx5_0", 400000},
                                                {2, 2, "mlx5_0", 400000}, {3, 2, "mlx5_0", 400000}})));
  EXPECT_TRUE(AnyLineContains("[OK]   Net device: NIC configuration consistent across 2 nodes in comm 0x1234"));
  EXPECT_FALSE(AnyLineContains("differ"));
}

TEST(RasDiagnosticsNetMicrotest, Summarize_FlagsSpeedMismatchAcrossAndWithinNodes) {
  ASSERT_EQ(ncclSuccess, Summarize(MakeRecords({{0, 1, "mlx5_0", 400000}, {1, 1, "mlx5_0", 400000},
                                                {2, 2, "mlx5_0", 400000}, {3, 2, "mlx5_0", 200000}})));
  EXPECT_TRUE(AnyLineContains("Net device mlx5_0:1: rank(s) {3} differ from rank 0"));
  EXPECT_TRUE(AnyLineContains("speed=400000 Mbps, link layer=InfiniBand, state=ACTIVE/LinkUp"));
  EXPECT_TRUE(AnyLineContains("differs within 1 of 2 nodes in comm 0x1234"));
  EXPECT_FALSE(AnyLineContains("[OK]"));
}

TEST(RasDiagnosticsNetMicrotest, Summarize_FlagsDeviceMissingOnANode) {
  ASSERT_EQ(ncclSuccess, Summarize(MakeRecords({{0, 1, "mlx5_0", 400000}, {0, 1, "mlx5_1", 400000},
                                                {1, 1, "mlx5_0", 400000}, {2, 2, "mlx5_0", 400000},
                                                {3, 2, "mlx5_0", 400000}})));
  EXPECT_TRUE(AnyLineContains("Net device mlx5_1:1: reported on only 1 of 2 nodes in comm 0x1234"));
}

TEST(RasDiagnosticsNetMicrotest, Prepare_ExpandsMergedDevicesIntoUniqueMembers) {
  std::vector<ncclTopoNetPropertiesSnapshot> props(4);
  const char* names[] = {"mlx5_0", "mlx5_1", "mlx5_0+mlx5_1", "mlx5_9"};
  for (int i = 0; i < 4; i++) {
    props[i].propertiesValid = i != 3;  // mlx5_9 has no valid properties
    props[i].vProps.ndevs = 1;
    props[i].vProps.devs[0] = i;
    props[i].speed = 100000 * (i + 1);
    props[i].port = 1;
    props[i].netDeviceType = NCCL_NET_DEVICE_HOST;
    snprintf(props[i].name, sizeof(props[i].name), "%s", names[i]);
  }
  props[2].vProps.ndevs = 2;  // merged device: its members are devices 0 and 1, already listed
  props[2].vProps.devs[0] = 0;
  props[2].vProps.devs[1] = 1;

  std::vector<struct rasDiagnosticsNetDeviceData> out(4);
  ASSERT_EQ(2, rasDiagnosticsNetDevicePrepare(props.data(), 4, 0xabc, reinterpret_cast<char*>(out.data()),
                                              sizeof(out[0]), 4));
  EXPECT_STREQ("mlx5_0", out[0].name);
  EXPECT_STREQ("mlx5_1", out[1].name);
  EXPECT_EQ(200000, out[1].speed);
  EXPECT_EQ(0xabcu, out[1].hostHash);
  EXPECT_EQ(1, out[1].port);
}

}  // namespace
