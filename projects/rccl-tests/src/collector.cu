/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*************************************************************************
 * Network Counter Collector – implementation
 *
 * See collector.h for the public API and environment variable reference.
 *************************************************************************/
#include "collector.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <set>
#include <unistd.h>

// =====================================================================
// Per-NIC-type counter tables
// =====================================================================
//
// Each NIC type has its own table with actual hardware key names.
// name          : counter key (used in output headers and primary sysfs path)
// source        : where to read the value
// is_prefix     : true → prefix-match (expands to name0..name7)
// fallback_name : alternative sysfs key to try if 'name' is not found (or NULL)

struct CounterTableEntry {
  const char*   name;
  CounterSource source;
  bool          is_prefix;
  const char*   fallback_name;  // NULL = no fallback
};

// ----- IONIC (AINIC) counter table -----
static const CounterTableEntry ionic_table[] = {
  // ethtool – PFC frame counters (prefix → expands to name0..name7)
  {"frames_rx_pripause",       COUNTER_SRC_ETHTOOL,  true,  NULL},
  {"frames_tx_pripause",       COUNTER_SRC_ETHTOOL,  true,  NULL},
  // IB hw_counters – CNP
  {"rx_rdma_cnp_pkts",         COUNTER_SRC_IB_HW,    false, NULL},
  {"tx_rdma_cnp_pkts",         COUNTER_SRC_IB_HW,    false, NULL},
  // IB hw_counters – errors & retransmits
  {"rx_rdma_mtu_discard_pkts", COUNTER_SRC_IB_HW,    false, NULL},
  {"resp_rx_outof_buf",        COUNTER_SRC_IB_HW,    false, NULL},
  {"tx_rdma_ack_timeout",      COUNTER_SRC_IB_HW,    false, NULL},
  {"req_tx_retry_excd_err",    COUNTER_SRC_IB_HW,    false, NULL},
  {"resp_rx_outouf_seq",       COUNTER_SRC_IB_HW,    false, NULL},
  {"req_rx_pkt_seq_err",       COUNTER_SRC_IB_HW,    false, NULL},
  // IB hw_counters – RDMA throughput
  {"tx_rdma_ucast_bytes",      COUNTER_SRC_IB_HW,    false, NULL},
  {"rx_rdma_ucast_bytes",      COUNTER_SRC_IB_HW,    false, NULL},
  {"tx_rdma_ucast_pkts",       COUNTER_SRC_IB_HW,    false, NULL},
  {"rx_rdma_ucast_pkts",       COUNTER_SRC_IB_HW,    false, NULL},
};
static const int ionic_table_size = (int)(sizeof(ionic_table) / sizeof(ionic_table[0]));

// ----- BNXT_RE (Thor2) counter table -----
static const CounterTableEntry bnxt_re_table[] = {
  // ethtool – PFC frame counters (prefix → expands to name0..name7)
  {"rx_pfc_ena_frames_pri",   COUNTER_SRC_ETHTOOL,  true,  NULL},
  {"tx_pfc_ena_frames_pri",   COUNTER_SRC_ETHTOOL,  true,  NULL},
  // ethtool – PFC transition counters (exact)
  {"pfc_pri3_rx_transitions", COUNTER_SRC_ETHTOOL,  false, NULL},
  {"pfc_pri3_tx_transitions", COUNTER_SRC_ETHTOOL,  false, NULL},
  // IB hw_counters – CNP (try rx_cnp_pkts first, fall back to rp_cnp_handled)
  {"rx_cnp_pkts",             COUNTER_SRC_IB_HW,    false, "rp_cnp_handled"},
  {"tx_cnp_pkts",             COUNTER_SRC_IB_HW,    false, "np_cnp_sent"},
  {"rx_roce_discards",        COUNTER_SRC_IB_HW,    false, NULL},
  // debugfs (bnxt_re info)
  {"rx_stat_discards",        COUNTER_SRC_DEBUGFS,  false, NULL},
  {"to_retransmits",          COUNTER_SRC_DEBUGFS,  false, NULL},
  {"max_retry_exceeded",      COUNTER_SRC_DEBUGFS,  false, NULL},
  {"oos_drop_count",          COUNTER_SRC_DEBUGFS,  false, NULL},
  {"seq_err_naks_rcvd",       COUNTER_SRC_DEBUGFS,  false, NULL},
};
static const int bnxt_re_table_size = (int)(sizeof(bnxt_re_table) / sizeof(bnxt_re_table[0]));

static void GetTable(NicType nic_type,
                     const CounterTableEntry*& table, int& size) {
  switch (nic_type) {
    case NIC_IONIC:
      table = ionic_table;
      size = ionic_table_size;
      break;
    case NIC_BNXT_RE:
      table = bnxt_re_table;
      size = bnxt_re_table_size;
      break;
    case NIC_UNKNOWN:
      fprintf(stderr,
              "# Warning: NIC type unknown, defaulting to bnxt_re counter set\n");
      table = bnxt_re_table;
      size = bnxt_re_table_size;
      break;
  }
}

// =====================================================================
// NIC prefix helpers (for fallback interface discovery)
// =====================================================================

static const char* NicPrefix() {
  static const char* prefix = NULL;
  static bool checked = false;
  if (!checked) {
    prefix = getenv("RCCL_TESTS_NET_COUNTER_NIC_PREFIX");
    checked = true;
  }
  return prefix;
}

// =====================================================================
// Public helpers
// =====================================================================

static const std::set<std::string>& RequestedCounterFilter() {
  static std::set<std::string> filter;
  static bool checked = false;
  if (!checked) {
    const char* env = getenv("RCCL_TESTS_NIC_COUNTER_LIST");
    if (env && strlen(env) > 0) {
      std::string list(env);
      size_t pos = 0;
      while (pos < list.size()) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) { comma = list.size(); }
        std::string token = list.substr(pos, comma - pos);
        while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) {
          token.erase(token.begin());
        }
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) {
          token.pop_back();
        }
        if (!token.empty()) { filter.insert(token); }
        pos = comma + 1;
      }
    }
    checked = true;
  }
  return filter;
}

static bool CounterMatchesRequestedFilter(const CounterTableEntry& entry) {
  const std::set<std::string>& filter = RequestedCounterFilter();
  if (filter.empty()) { return true; }
  if (filter.count(entry.name) > 0) { return true; }
  return entry.fallback_name != NULL && filter.count(entry.fallback_name) > 0;
}

bool NetCounterIsEnabled() {
  static int enabled = -1;
  if (enabled == -1) {
    const char* env = getenv("RCCL_TESTS_NET_COUNTER_ENABLE");
    enabled = (env && atoi(env) > 0) ? 1 : 0;
  }
  return enabled == 1;
}

std::vector<CounterDescriptor> NetCounterGetCounterList(NicType nic_type) {
  std::vector<CounterDescriptor> result;
  const CounterTableEntry* table;
  int size;
  GetTable(nic_type, table, size);
  for (int i = 0; i < size; i++) {
    if (!CounterMatchesRequestedFilter(table[i])) continue;
    result.push_back({table[i].name, table[i].source, table[i].is_prefix,
                      table[i].fallback_name ? table[i].fallback_name : ""});
  }
  return result;
}

std::vector<std::string> NetCounterParseIbHcaList() {
  std::vector<std::string> result;

  const char* env = getenv("NCCL_IB_HCA");
  if (!env || strlen(env) == 0) { return result; }

  std::string list(env);

  size_t pos = 0;
  while (pos < list.size()) {
    size_t comma = list.find(',', pos);
    if (comma == std::string::npos) { comma = list.size(); }

    std::string token = list.substr(pos, comma - pos);

    while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) {
      token.erase(token.begin());
    }

    while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) {
      token.pop_back();
    }

    if (!token.empty()) { result.push_back(token); }

    pos = comma + 1;
  }
  return result;
}

// =====================================================================
// IB device name parsing
// =====================================================================

// Split "rdma0:2" -> base="rdma0", port=2.  Default port is 1.
static void ParseIbDevice(const std::string& ib_device,
                           std::string& base, int& port) {
  size_t colon = ib_device.find(':');
  if (colon == std::string::npos) {
    base = ib_device;
    port = 1;
  } else {
    base = ib_device.substr(0, colon);
    port = atoi(ib_device.c_str() + colon + 1);
    if (port <= 0) port = 1;
  }
}

// =====================================================================
// Device lookup
// =====================================================================

static std::string RunShellOneLiner(const char* cmd) {
  FILE* fp = popen(cmd, "r");
  if (!fp) { return ""; }
  char buf[256] = {0};
  if (fgets(buf, sizeof(buf), fp)) {
    size_t len = strlen(buf);
    if (len > 0 && buf[len-1] == '\n') {
      buf[len-1] = '\0';
    }
  }
  pclose(fp);
  return std::string(buf);
}

std::string NetCounterFindIbDeviceForNic(const std::string& nic) {
  char cmd[512] = {0};
  snprintf(cmd, sizeof(cmd),
           "ls /sys/class/net/%s/device/infiniband 2>/dev/null | head -1",
           nic.c_str());
  return RunShellOneLiner(cmd);
}

std::string NetCounterFindNicForIbDevice(const std::string& ib_device) {
  std::string ib_base;
  int port;
  ParseIbDevice(ib_device, ib_base, port);
  (void)port;
  char cmd[512] = {0};
  snprintf(cmd, sizeof(cmd),
           "ls /sys/class/infiniband/%s/device/net 2>/dev/null | head -1",
           ib_base.c_str());
  return RunShellOneLiner(cmd);
}

void NetCounterGetNetworkInterfaces(std::vector<std::string>& interfaces) {
  FILE* fp = popen("ls /sys/class/net 2>/dev/null", "r");
  if (fp == NULL) { return; }

  const char* prefix = NicPrefix();

  char path[256] = {0};
  while (fgets(path, sizeof(path), fp) != NULL) {
    if (path[strlen(path)-1] == '\n') {
      path[strlen(path)-1] = '\0';
    }
    if (prefix) {
      if (strncmp(path, prefix, strlen(prefix)) == 0) {
        interfaces.push_back(std::string(path));
      }
    } else {
      if (strcmp(path, "lo") != 0 &&
          strncmp(path, "veth", 4) != 0 &&
          strncmp(path, "fenic", 5) != 0) {
        interfaces.push_back(std::string(path));
      }
    }
  }
  pclose(fp);
}

// =====================================================================
// NIC type detection
// =====================================================================

NicType NetCounterDetectNicType(const std::string& ib_device) {
  if (ib_device.empty()) return NIC_UNKNOWN;

  std::string ib_base;
  int port;
  ParseIbDevice(ib_device, ib_base, port);
  (void)port;

  if (ib_base.compare(0, 7, "bnxt_re") == 0) return NIC_BNXT_RE;

  char link[4096] = {0};
  std::string driver_path =
      "/sys/class/infiniband/" + ib_base + "/device/driver";
  ssize_t len = readlink(driver_path.c_str(), link, sizeof(link) - 1);
  if (len <= 0) return NIC_UNKNOWN;
  link[len] = '\0';

  const char* drv = strrchr(link, '/');
  drv = drv ? drv + 1 : link;

  if (strcmp(drv, "bnxt_re") == 0 || strcmp(drv, "bnxt_en") == 0)
    return NIC_BNXT_RE;
  if (strcmp(drv, "ionic") == 0)
    return NIC_IONIC;
  return NIC_UNKNOWN;
}

const char* NicTypeStr(NicType t) {
  switch (t) {
    case NIC_BNXT_RE: return "BNXT_RE";
    case NIC_IONIC:   return "IONIC";
    default:          return "UNKNOWN";
  }
}

NicType DetectNicTypeFromSnapshots(
    const std::vector<NetworkCounterSnapshot>& snaps, bool& mixed) {
  NicType nic_type = NIC_UNKNOWN;
  mixed = false;
  for (size_t i = 0; i < snaps.size(); i++) {
    if (snaps[i].nic_type == NIC_UNKNOWN) continue;
    if (nic_type == NIC_UNKNOWN) {
      nic_type = snaps[i].nic_type;
    } else if (snaps[i].nic_type != nic_type) {
      mixed = true;
      break;
    }
  }
  return nic_type;
}

// =====================================================================
// Per-source collection
// =====================================================================

static void CollectEthtoolCounters(const std::string& nic,
                                   const std::vector<CounterDescriptor>& selected,
                                   std::map<std::string, uint64_t>& out) {
  std::set<std::string> exact;
  std::vector<std::string> prefixes;
  for (const auto& d : selected) {
    if (d.source != COUNTER_SRC_ETHTOOL) { continue; }
    if (d.is_prefix) {
      prefixes.push_back(d.name);
    } else {
      exact.insert(d.name);
    }
  }
  if (exact.empty() && prefixes.empty()) { return; }

  char command[512] = {0};
  snprintf(command, sizeof(command), "ethtool -S %s 2>/dev/null", nic.c_str());

  FILE* fp = popen(command, "r");
  if (!fp) { return; }

  char line[256] = {0};
  while (fgets(line, sizeof(line), fp)) {
    char key[256] = {0};
    unsigned long long tmp = 0;
    if (sscanf(line, "%255[^:]: %llu", key, &tmp) == 2) {
      uint64_t value = (uint64_t)tmp;
      char* start = key;
      while (*start == ' ' || *start == '\t') { start++; }
      std::string k(start);
      if (exact.count(k)) {
        out[k] = value;
        continue;
      }
      for (const auto& pfx : prefixes) {
        if (k.compare(0, pfx.size(), pfx) == 0) {
          out[k] = value;
          break;
        }
      }
    }
  }
  pclose(fp);
}

static void CollectIbHwCounters(const std::string& ib_device,
                                const std::vector<CounterDescriptor>& selected,
                                std::map<std::string, uint64_t>& out) {
  if (ib_device.empty()) { return; }

  std::string ib_base;
  int ib_port;
  ParseIbDevice(ib_device, ib_base, ib_port);
  char port_str[16];
  snprintf(port_str, sizeof(port_str), "%d", ib_port);
  std::string base_port =
      "/sys/class/infiniband/" + ib_base + "/ports/" + port_str + "/hw_counters/";
  std::string base_dev =
      "/sys/class/infiniband/" + ib_base + "/hw_counters/";

  for (const auto& d : selected) {
    if (d.source != COUNTER_SRC_IB_HW) continue;
    std::string path = base_port + d.name;
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp) fp = fopen((base_dev + d.name).c_str(), "r");
    // If primary name not found, try fallback
    if (!fp && !d.fallback_name.empty()) {
      fp = fopen((base_port + d.fallback_name).c_str(), "r");
      if (!fp) fp = fopen((base_dev + d.fallback_name).c_str(), "r");
    }
    if (fp) {
      unsigned long long tmp = 0;
      if (fscanf(fp, "%llu", &tmp) == 1) out[d.name] = (uint64_t)tmp;
      fclose(fp);
    }
  }
}

static void CollectDebugCounters(const std::string& ib_device,
                                 const std::vector<CounterDescriptor>& selected,
                                 std::map<std::string, uint64_t>& out) {
  if (ib_device.empty()) return;

  std::set<std::string> wanted;
  for (const auto& d : selected) {
    if (d.source == COUNTER_SRC_DEBUGFS) {
      wanted.insert(d.name);
    }
  }
  if (wanted.empty()) { return; }

  std::string ib_base;
  int port;
  ParseIbDevice(ib_device, ib_base, port);
  (void)port;
  std::string info_path =
      "/sys/kernel/debug/bnxt_re/" + ib_base + "/info";
  FILE* fp = fopen(info_path.c_str(), "r");
  if (!fp) { return; }

  char line[512] = {0};
  while (fgets(line, sizeof(line), fp)) {
    for (const auto& name : wanted) {
      if (strstr(line, name.c_str())) {
        unsigned long long tmp = 0;
        char key[256] = {0};
        if (sscanf(line, " %255[^:=] %*[:=] %llu", key, &tmp) == 2 ||
            sscanf(line, " %255s %llu", key, &tmp) == 2) {
          uint64_t value = (uint64_t)tmp;
          char* start = key;
          while (*start == ' ' || *start == '\t') { start++; }
          char* end = start + strlen(start) - 1;
          while (end > start && (*end == ' ' || *end == '\t')) { *end-- = '\0'; }
          if (wanted.count(std::string(start))) {
            out[std::string(start)] = value;
          }
        }
        break;
      }
    }
  }
  fclose(fp);
}

// =====================================================================
// Snapshot collection
// =====================================================================

NetworkCounterSnapshot NetCounterCollectSnapshot(
    const std::string& nic,
    const std::string& ib_dev,
    const std::vector<CounterDescriptor>& selected) {
  NetworkCounterSnapshot snap;

  memset(snap.nic_name, 0, sizeof(snap.nic_name));
  memset(snap.ib_device, 0, sizeof(snap.ib_device));

  if (!nic.empty()) {
    strncpy(snap.nic_name, nic.c_str(), sizeof(snap.nic_name) - 1);
  }

  if (!ib_dev.empty()) {
    strncpy(snap.ib_device, ib_dev.c_str(), sizeof(snap.ib_device) - 1);
  }

  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  snap.timestamp_us = (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
  snap.nic_type = NetCounterDetectNicType(ib_dev);

  CollectEthtoolCounters(nic, selected, snap.counters);
  CollectIbHwCounters(ib_dev, selected, snap.counters);
  CollectDebugCounters(ib_dev, selected, snap.counters);

  return snap;
}

NetworkCounterSnapshot NetCounterCollectSnapshot(
    const std::string& nic,
    const std::vector<CounterDescriptor>& selected) {
  return NetCounterCollectSnapshot(nic, NetCounterFindIbDeviceForNic(nic),
                                   selected);
}

// =====================================================================
// Delta computation
// =====================================================================

uint64_t NetCounterComputeDelta(
    const NetworkCounterSnapshot& before,
    const NetworkCounterSnapshot& after,
    const CounterDescriptor& desc) {
  uint64_t delta = 0;
  if (desc.is_prefix) {
    for (const auto& entry : after.counters) {
      if (entry.first.compare(0, desc.name.size(), desc.name) == 0) {
        uint64_t a = entry.second;
        uint64_t b = 0;
        auto it = before.counters.find(entry.first);
        if (it != before.counters.end()) { b = it->second; }
        delta += (a - b);
      }
    }
  } else {
    uint64_t a = 0, b = 0;
    auto it = before.counters.find(desc.name);
    if (it != before.counters.end()) { b = it->second; }
    it = after.counters.find(desc.name);
    if (it != after.counters.end()) { a = it->second; }
    delta = a - b;
  }
  return delta;
}

// =====================================================================
// Table printer
// =====================================================================

void NetCounterPrintTable(
    const std::vector<std::string>& nic_names,
    const std::vector<NetworkCounterSnapshot>& before,
    const std::vector<NetworkCounterSnapshot>& after,
    int rank,
    const std::vector<CounterDescriptor>& selected) {
  size_t num_nics     = nic_names.size();
  size_t num_counters = selected.size();

  char hostname[256] = {0};

  if (gethostname(hostname, sizeof(hostname)) != 0) {
    strncpy(hostname, "unknown", sizeof(hostname));
  }

  hostname[sizeof(hostname)-1] = '\0';

  if (num_counters == 0 || num_nics == 0) {
    printf("NET_COUNTER_TABLE: node=%s rank=%d status=NO_DATA\n",
           hostname, rank);
    fflush(stdout);
    return;
  }

  bool mixed = false;
  NicType nic_type = DetectNicTypeFromSnapshots(before, mixed);

  // Pre-compute deltas, rates, durations
  std::vector<std::vector<uint64_t>> deltas(
      num_nics, std::vector<uint64_t>(num_counters, 0));

  std::vector<std::vector<double>> rates(
      num_nics, std::vector<double>(num_counters, 0.0));

  std::vector<double> dur_sec(num_nics);

  for (size_t n = 0; n < num_nics; n++) {
    dur_sec[n] = (after[n].timestamp_us - before[n].timestamp_us) / 1e6;
    for (size_t c = 0; c < num_counters; c++) {
      deltas[n][c] = NetCounterComputeDelta(before[n], after[n], selected[c]);
      rates[n][c] = (dur_sec[n] > 0.0)
                        ? (double)deltas[n][c] / dur_sec[n]
                        : 0.0;
    }
  }

  // Display labels: prefer "IB(NIC)" when both are known
  std::vector<std::string> labels(num_nics);

  for (size_t n = 0; n < num_nics; n++) {
    if (before[n].ib_device[0] != '\0' && before[n].nic_name[0] != '\0') {
      labels[n] = std::string(before[n].ib_device)
                  + "(" + before[n].nic_name + ")";
    } else if (before[n].ib_device[0] != '\0') {
      labels[n] = before[n].ib_device;
    } else {
      labels[n] = before[n].nic_name;
    }
  }

  // Column widths
  int nic_w = 6; // strlen("Device")
  for (const auto& lbl : labels) {
    nic_w = std::max(nic_w, (int)lbl.size());
  }

  nic_w += 2;

  std::vector<int> cnt_w(num_counters, 5);
  std::vector<int> rt_w(num_counters, 6);

  for (size_t c = 0; c < num_counters; c++) {
    for (size_t n = 0; n < num_nics; n++) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%llu", (unsigned long long)deltas[n][c]);
      cnt_w[c] = std::max(cnt_w[c], (int)strlen(buf));
      snprintf(buf, sizeof(buf), "%.2f", rates[n][c]);
      rt_w[c] = std::max(rt_w[c], (int)strlen(buf));
    }
    int name_len = (int)selected[c].name.size();
    int pair_w   = cnt_w[c] + 2 + rt_w[c];
    if (name_len > pair_w) {
      rt_w[c] += (name_len - pair_w);
    }
  }

  // Separator
  std::string sep(nic_w, '-');
  for (size_t c = 0; c < num_counters; c++) {
    sep += "--";
    sep += std::string(cnt_w[c], '-');
    sep += "--";
    sep += std::string(rt_w[c], '-');
  }

  // Print
  printf("\nNET_COUNTER_TABLE: node=%s  rank=%d  duration_sec=%.3f  nic_type=%s\n",
         hostname, rank, dur_sec[0], mixed ? "MIXED" : NicTypeStr(nic_type));
  printf("%s\n", sep.c_str());

  printf("%-*s", nic_w, "Device");
  for (size_t c = 0; c < num_counters; c++) {
    int pair_w = cnt_w[c] + 2 + rt_w[c];
    printf("  %-*s", pair_w, selected[c].name.c_str());
  }
  printf("\n");

  printf("%-*s", nic_w, "");
  for (size_t c = 0; c < num_counters; c++) {
    printf("  %*s  %*s", cnt_w[c], "count", rt_w[c], "rate/s");
  }
  printf("\n");

  printf("%s\n", sep.c_str());

  for (size_t n = 0; n < num_nics; n++) {
    printf("%-*s", nic_w, labels[n].c_str());
    for (size_t c = 0; c < num_counters; c++) {
      printf("  %*llu  %*.2f", cnt_w[c], (unsigned long long)deltas[n][c], rt_w[c], rates[n][c]);
    }
    printf("\n");
  }

  printf("%s\n\n", sep.c_str());
  fflush(stdout);
}
