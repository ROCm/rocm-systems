/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for the NIC counter collector (collector.cu).
// No GPU, MPI or NIC required.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "collector.cu"

// The counter filter is read once; keep it from narrowing the tables.
static const int kUnsetFilter = unsetenv("RCCL_TESTS_NIC_COUNTER_LIST");

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
      "pfc_rx_transitions",    "pfc_tx_transitions"};
  EXPECT_EQ(Names(NetCounterGetCounterList(NIC_BNXT_RE)), expect);
}

TEST(Collector, Bnxt_RetransmitCountersReadFromSysfs) {
  for (const auto& d : NetCounterGetCounterList(NIC_BNXT_RE)) {
    if (d.name == "to_retransmits" || d.name == "max_retry_exceeded" || d.name == "oos_drop_count" ||
        d.name == "seq_err_naks_rcvd") {
      EXPECT_EQ(d.source, COUNTER_SRC_IB_HW) << d.name;
    }
    if (d.name == "rx_stat_discards") {
      EXPECT_EQ(d.source, COUNTER_SRC_ETHTOOL);
    }
    if (d.name == "oos_drop_count") {  // same order as RCCL telemetry
      EXPECT_EQ(d.key, "res_oos_drop_count");
      EXPECT_EQ(d.fallback_name, "oos_drop_count");
    }
  }
}

TEST(Collector, Tables_DisplayNamesHaveNoPattern) {
  for (NicType t : {NIC_BNXT_RE, NIC_IONIC, NIC_MLX5, NIC_UNKNOWN}) {
    for (const auto& d : NetCounterGetCounterList(t)) {
      EXPECT_EQ(d.name.find('*'), std::string::npos) << d.name;
    }
  }
}

TEST(Collector, Unknown_UsesBnxtTable) {
  EXPECT_EQ(Names(NetCounterGetCounterList(NIC_UNKNOWN)), Names(NetCounterGetCounterList(NIC_BNXT_RE)));
}

TEST(Collector, Ethtool_FallbackKeyAndPerQueueLines) {
  std::vector<CounterDescriptor> sel = {{"rx_stat_discards", COUNTER_SRC_ETHTOOL, false, "rx_stat_discard"},
                                        {"rx_discards", COUNTER_SRC_ETHTOOL, false, ""},
                                        {"pfc_rx_transitions", COUNTER_SRC_ETHTOOL, true, "",
                                         "pfc_pri*_rx_transitions"}};
  std::map<std::string, uint64_t> out;
  NetCounterParseEthtool("NIC statistics:\n"
                         "     [0]: rx_discards: 9\n"
                         "     rx_stat_discard: 5\n"
                         "     pfc_pri3_rx_transitions: 4\n"
                         "     pfc_pri3_rx_duration_us: 100\n",
                         sel, out);
  EXPECT_EQ(out["rx_stat_discards"], 5u);
  EXPECT_EQ(out.count("rx_discards"), 0u);
  EXPECT_EQ(out.count("pfc_pri3_rx_transitions"), 1u);
  EXPECT_EQ(out.count("pfc_pri3_rx_duration_us"), 0u);
}

TEST(Collector, Ethtool_PrimaryWinsOverFallback) {
  std::vector<CounterDescriptor> sel = {{"rx_stat_discards", COUNTER_SRC_ETHTOOL, false, "rx_stat_discard"}};
  std::map<std::string, uint64_t> out;
  NetCounterParseEthtool("     rx_stat_discard: 5\n     rx_stat_discards: 7\n", sel, out);
  EXPECT_EQ(out["rx_stat_discards"], 7u);
}

TEST(Collector, Mlx5_PausePatternSkipsDuration) {
  std::vector<CounterDescriptor> sel;
  for (const auto& d : NetCounterGetCounterList(NIC_MLX5)) {
    if (d.name == "rx_prio_pause") { sel.push_back(d); }
  }
  ASSERT_EQ(sel.size(), 1u);
  std::map<std::string, uint64_t> out;
  NetCounterParseEthtool("     rx_prio0_pause: 1\n     rx_prio3_pause: 2\n     rx_prio3_pause_duration: 9\n", sel, out);
  EXPECT_EQ(out.size(), 2u);
  EXPECT_EQ(out.count("rx_prio3_pause_duration"), 0u);
}

TEST(Collector, Debugfs_ExactKeyOnly) {
  std::map<std::string, uint64_t> out;
  NetCounterParseDebugfs("\tres_oos_drop_count: 3\n\toos_drop_count: 8\n\tto_retransmits = 2\n",
                         {"oos_drop_count", "to_retransmits"}, out);
  EXPECT_EQ(out["oos_drop_count"], 8u);
  EXPECT_EQ(out["to_retransmits"], 2u);
  EXPECT_EQ(out.size(), 2u);
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
  CounterDescriptor d{"pfc_rx_transitions", COUNTER_SRC_ETHTOOL, true, "", "pfc_pri*_rx_transitions"};
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

TEST(Collector, Table_NoData) {
  testing::internal::CaptureStdout();
  NetCounterPrintTable({}, {}, {}, 3, {});
  EXPECT_NE(testing::internal::GetCapturedStdout().find("rank=3 status=NO_DATA"), std::string::npos);
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

TEST(Collector, IbHca_RcclEdgeCases) {
  std::vector<std::string> devs = {"bnxt_re1", "bnxt_re0", "mlx5_0"};
  using V = std::vector<std::string>;
  // Entry order is kept.
  EXPECT_EQ(NetCounterFilterIbHca("=bnxt_re1,bnxt_re0", devs), (V{"bnxt_re1", "bnxt_re0"}));
  // Excluding another port keeps the collected port 1.
  EXPECT_EQ(NetCounterFilterIbHca("^mlx5_0:2", devs), (V{"bnxt_re0", "bnxt_re1", "mlx5_0"}));
  EXPECT_EQ(NetCounterFilterIbHca("^mlx5_0:1", devs), (V{"bnxt_re0", "bnxt_re1"}));
  // Both ports of one device.
  EXPECT_EQ(NetCounterFilterIbHca("mlx5_0:1,mlx5_0:2", devs), (V{"mlx5_0:1", "mlx5_0:2"}));
  // Port 0 or a non-number matches no port; rail/plane fields are ignored.
  EXPECT_EQ(NetCounterFilterIbHca("mlx5_0:0,bnxt_re0:x", devs), V{});
  EXPECT_EQ(NetCounterFilterIbHca("mlx5_0:1:3:0", devs), (V{"mlx5_0:1"}));
  // An empty list selects every device; '^' alone selects none.
  EXPECT_EQ(NetCounterFilterIbHca(",", devs), (V{"bnxt_re0", "bnxt_re1", "mlx5_0"}));
  EXPECT_EQ(NetCounterFilterIbHca("^", devs), V{});
}
