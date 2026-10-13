/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for the NIC counter collector (collector.cu).
// No GPU, MPI or NIC required.

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "collector.cu"

static NetworkCounterSnapshot Snap(std::map<std::string, uint64_t> c, int64_t tsUs) {
  NetworkCounterSnapshot s;
  memset(s.nic_name, 0, sizeof(s.nic_name));
  memset(s.ib_device, 0, sizeof(s.ib_device));
  strcpy(s.nic_name, "benic1p1");
  strcpy(s.ib_device, "bnxt_re0");
  s.counters = c;
  s.timestamp_us = tsUs;
  s.nic_type = NIC_BNXT_RE;
  return s;
}

static std::vector<std::string> Names(const std::vector<CounterDescriptor>& d) {
  std::vector<std::string> n;
  for (const auto& x : d) n.push_back(x.name);
  return n;
}

// Downstream parsers read these columns by position.
TEST(Collector, Bnxt_ColumnOrderKept) {
  std::vector<std::string> expect = {
      "rx_pfc_ena_frames_pri", "tx_pfc_ena_frames_pri", "pfc_pri3_rx_transitions", "pfc_pri3_tx_transitions",
      "rx_cnp_pkts",           "tx_cnp_pkts",           "rx_roce_discards",        "rx_stat_discards",
      "to_retransmits",        "max_retry_exceeded",    "oos_drop_count",          "seq_err_naks_rcvd",
      "pfc_pri*_rx_transitions", "pfc_pri*_tx_transitions"};
  EXPECT_EQ(Names(NetCounterGetCounterList(NIC_BNXT_RE)), expect);
}

TEST(Collector, Bnxt_RetransmitCountersReadFromSysfs) {
  for (const auto& d : NetCounterGetCounterList(NIC_BNXT_RE)) {
    if (d.name == "to_retransmits" || d.name == "max_retry_exceeded" || d.name == "oos_drop_count" ||
        d.name == "seq_err_naks_rcvd") {
      EXPECT_EQ(d.source, COUNTER_SRC_IB_HW) << d.name;
    }
  }
}

TEST(Collector, Mlx5_TableSelected) {
  auto n = Names(NetCounterGetCounterList(NIC_MLX5));
  EXPECT_FALSE(n.empty());
  EXPECT_NE(std::find(n.begin(), n.end(), "rp_cnp_handled"), n.end());
}

TEST(Collector, Delta_MissingCounterIsNotZero) {
  CounterDescriptor d{"to_retransmits", COUNTER_SRC_IB_HW, false, ""};
  uint64_t delta = 99;
  EXPECT_FALSE(NetCounterComputeDelta(Snap({}, 0), Snap({}, 1), d, &delta));
  EXPECT_FALSE(NetCounterComputeDelta(Snap({{"to_retransmits", 5}}, 0), Snap({}, 1), d, &delta));
  EXPECT_TRUE(NetCounterComputeDelta(Snap({{"to_retransmits", 5}}, 0), Snap({{"to_retransmits", 5}}, 1), d, &delta));
  EXPECT_EQ(delta, 0u);
  EXPECT_TRUE(NetCounterComputeDelta(Snap({{"to_retransmits", 5}}, 0), Snap({{"to_retransmits", 9}}, 1), d, &delta));
  EXPECT_EQ(delta, 4u);
}

TEST(Collector, Delta_PriorityPatternSumsAllPriorities) {
  CounterDescriptor d{"pfc_pri*_rx_transitions", COUNTER_SRC_ETHTOOL, true, ""};
  std::map<std::string, uint64_t> before = {{"pfc_pri0_rx_transitions", 1},
                                            {"pfc_pri3_rx_transitions", 10},
                                            {"pfc_pri3_rx_duration_us", 100},
                                            {"pfc_pri3_tx_transitions", 7}};
  std::map<std::string, uint64_t> after = {{"pfc_pri0_rx_transitions", 2},
                                           {"pfc_pri3_rx_transitions", 15},
                                           {"pfc_pri3_rx_duration_us", 900},
                                           {"pfc_pri3_tx_transitions", 70}};
  uint64_t delta = 0;
  EXPECT_TRUE(NetCounterComputeDelta(Snap(before, 0), Snap(after, 1), d, &delta));
  EXPECT_EQ(delta, 6u);
}

TEST(Collector, Delta_PlainPrefix) {
  CounterDescriptor d{"rx_pfc_ena_frames_pri", COUNTER_SRC_ETHTOOL, true, ""};
  uint64_t delta = 0;
  EXPECT_TRUE(NetCounterComputeDelta(Snap({{"rx_pfc_ena_frames_pri0", 1}, {"rx_pfc_ena_frames_pri3", 1}}, 0),
                                     Snap({{"rx_pfc_ena_frames_pri0", 3}, {"rx_pfc_ena_frames_pri3", 11}}, 1), d,
                                     &delta));
  EXPECT_EQ(delta, 12u);
}

TEST(Collector, Table_NotAvailablePrintsNA) {
  std::vector<CounterDescriptor> sel = {{"rx_cnp_pkts", COUNTER_SRC_IB_HW, false, ""},
                                        {"to_retransmits", COUNTER_SRC_IB_HW, false, ""}};
  testing::internal::CaptureStdout();
  NetCounterPrintTable({"benic1p1"}, {Snap({{"rx_cnp_pkts", 1}}, 0)}, {Snap({{"rx_cnp_pkts", 4}}, 1000000)}, 0, sel);
  std::string out = testing::internal::GetCapturedStdout();
  std::istringstream lines(out);
  std::string line, row;
  while (std::getline(lines, line))
    if (line.rfind("bnxt_re0(benic1p1)", 0) == 0) row = line;
  ASSERT_FALSE(row.empty()) << out;
  std::istringstream tok(row);
  std::vector<std::string> t;
  for (std::string w; tok >> w;) t.push_back(w);
  // label, then count + rate per counter
  ASSERT_EQ(t.size(), 5u) << row;
  EXPECT_EQ(t[1], "3");
  EXPECT_EQ(t[2], "3.00");
  EXPECT_EQ(t[3], "N/A");
  EXPECT_EQ(t[4], "N/A");
}

TEST(Collector, IbHca_PrefixExactExcludePort) {
  std::vector<std::string> devs = {"bnxt_re10", "bnxt_re1", "bnxt_re0", "mlx5_0"};
  using V = std::vector<std::string>;
  EXPECT_EQ(NetCounterFilterIbHca("bnxt_re", devs), (V{"bnxt_re0", "bnxt_re1", "bnxt_re10"}));
  EXPECT_EQ(NetCounterFilterIbHca("bnxt_re1", devs), (V{"bnxt_re1", "bnxt_re10"}));
  EXPECT_EQ(NetCounterFilterIbHca("=bnxt_re1", devs), (V{"bnxt_re1"}));
  EXPECT_EQ(NetCounterFilterIbHca("^bnxt_re", devs), (V{"mlx5_0"}));
  EXPECT_EQ(NetCounterFilterIbHca("^=bnxt_re1,mlx5_0", devs), (V{"bnxt_re0", "bnxt_re10"}));
  EXPECT_EQ(NetCounterFilterIbHca("=bnxt_re0:1, mlx5_0:2", devs), (V{"bnxt_re0:1", "mlx5_0:2"}));
  EXPECT_EQ(NetCounterFilterIbHca("nosuchdev", devs), V{});
  EXPECT_EQ(NetCounterFilterIbHca("", devs), V{});
}
