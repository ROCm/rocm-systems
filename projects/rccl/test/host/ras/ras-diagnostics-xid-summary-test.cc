/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for `src/ras/diagnostics_xid.cc`'s summarizer.

#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// The file's out-of-unit calls (the host identity, the dmesg runner) and rasDiagnosticsInit, which ras-test.cc fakes,
// are renamed so this unit links into rccl-UnitTestsMicro on its own. The summarizer never reaches them.
#define getHostHash XidTestGetHostHash
#define getHostName XidTestGetHostName
#define ncclDiagChildRun XidTestDiagChildRun
#define rasDiagnosticsInit XidTestRasDiagnosticsInit
#define rasDiagnosticsXidCollectLocal XidTestCollectLocal
#define rasDiagnosticsXidSummarize XidTestSummarize

#include RAS_DIAGNOSTICS_XID_CC_PATH

uint64_t getHostHash() { return 0; }
ncclResult_t getHostName(char* hostname, int maxlen, const char /*delim*/) {
  if (maxlen > 0) hostname[0] = '\0';
  return ncclSuccess;
}
int ncclDiagChildRun(const char*, int, char*, int, bool*) { return -1; }

namespace {

std::vector<std::string> g_lines;

ncclResult_t RecordingEmit(void* /*target*/, const char* line) {
  g_lines.emplace_back(line);
  return ncclSuccess;
}

struct HostScan {
  uint64_t hostHash;
  const char* hostName;
  bool scanned;
  bool truncated;
};

std::vector<std::string> Summarize(const std::vector<HostScan>& hosts) {
  std::vector<rasDiagnosticsXidData> records(hosts.size());
  memset(records.data(), 0, records.size() * sizeof(records[0]));
  for (size_t i = 0; i < hosts.size(); i++) {
    records[i].kind = RAS_DIAG_XID_RECORD_SCAN_STATUS;
    records[i].scan.hostHash = hosts[i].hostHash;
    snprintf(records[i].scan.hostName, sizeof(records[i].scan.hostName), "%s", hosts[i].hostName);
    records[i].scan.scanned = hosts[i].scanned;
    records[i].scan.truncated = hosts[i].truncated;
  }
  rasDiagnosticsReporter reporter{};
  reporter.emit = RecordingEmit;
  struct rasDiagnosticsContext ctx{};
  g_lines.clear();
  EXPECT_EQ(ncclSuccess, rasDiagnosticsXidSummarize(&ctx, &reporter, reinterpret_cast<const char*>(records.data()),
                                                    static_cast<int>(records.size() * sizeof(records[0]))));
  return g_lines;
}

int CountContaining(const std::vector<std::string>& lines, const std::string& needle) {
  int n = 0;
  for (const auto& l : lines) n += l.find(needle) != std::string::npos;
  return n;
}

}  // namespace

// NCCL 2.32.6 (2284a209cc59): hosts without a kernel-log source are reported as one N/M count, not a line per host.
TEST(RasDiagnosticsXidSummaryMicrotest, Summarize_HostsWithoutLogSourceReportOneCount) {
  auto lines = Summarize({{0x10, "hostA", false, false}, {0x20, "hostB", true, false}, {0x30, "hostC", false, false}});
  EXPECT_EQ(1, CountContaining(lines, "no kernel-log source"));
  EXPECT_EQ(0, CountContaining(lines, "hostA"));
  EXPECT_EQ(0, CountContaining(lines, "hostC"));
  EXPECT_EQ(1, CountContaining(lines, "[INFO] Xid/SXid: no kernel-log source was available on 2/3 hosts"));
  // An incomplete scan never reports the all-clear.
  EXPECT_EQ(0, CountContaining(lines, "no events requiring immediate attention"));
}

TEST(RasDiagnosticsXidSummaryMicrotest, Summarize_SingleHostWithoutLogSourceIsSingular) {
  auto lines = Summarize({{0x10, "hostA", false, false}});
  ASSERT_EQ(1u, lines.size());
  EXPECT_EQ("[INFO] Xid/SXid: no kernel-log source was available on 1/1 host", lines[0]);
}

// Unchanged by the summary: all-clear across scanned hosts, and truncation still reported per host.
TEST(RasDiagnosticsXidSummaryMicrotest, Control_ScannedHostsReportAllClear) {
  auto lines = Summarize({{0x10, "hostA", true, false}, {0x20, "hostB", true, false}});
  ASSERT_EQ(1u, lines.size());
  EXPECT_EQ("[OK]   Xid/SXid: no events requiring immediate attention across 2 hosts", lines[0]);
}

TEST(RasDiagnosticsXidSummaryMicrotest, Control_TruncatedScanReportedPerHost) {
  auto lines = Summarize({{0x10, "hostA", true, true}, {0x20, "hostB", true, true}});
  EXPECT_EQ(1, CountContaining(lines, "[INFO] Xid/SXid: recent-event scan was truncated on host hostA"));
  EXPECT_EQ(1, CountContaining(lines, "[INFO] Xid/SXid: recent-event scan was truncated on host hostB"));
  EXPECT_EQ(0, CountContaining(lines, "kernel-log source"));
  EXPECT_EQ(0, CountContaining(lines, "no events requiring immediate attention"));
}
