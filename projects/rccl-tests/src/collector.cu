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
// Counter registry
// =====================================================================

struct CounterRegistryEntry {
  const char*   name;
  CounterSource source;
  bool          is_prefix;
};

static const CounterRegistryEntry counter_registry[] = {
  // ethtool – PFC frame counters (prefix → expands to name0..name7)
  {"rx_pfc_ena_frames_pri",   COUNTER_SRC_ETHTOOL,  true},
  {"tx_pfc_ena_frames_pri",   COUNTER_SRC_ETHTOOL,  true},
  // ethtool – PFC transition counters (exact)
  {"pfc_pri3_rx_transitions", COUNTER_SRC_ETHTOOL,  false},
  {"pfc_pri3_tx_transitions", COUNTER_SRC_ETHTOOL,  false},
  // IB hw_counters
  {"rx_cnp_pkts",             COUNTER_SRC_IB_HW,    false},
  {"tx_cnp_pkts",             COUNTER_SRC_IB_HW,    false},
  {"rx_roce_discards",        COUNTER_SRC_IB_HW,    false},
  // debugfs (bnxt_re info)
  {"rx_stat_discards",        COUNTER_SRC_DEBUGFS,   false},
  {"to_retransmits",          COUNTER_SRC_DEBUGFS,   false},
  {"max_retry_exceeded",      COUNTER_SRC_DEBUGFS,   false},
  {"oos_drop_count",          COUNTER_SRC_DEBUGFS,   false},
  {"seq_err_naks_rcvd",       COUNTER_SRC_DEBUGFS,   false},
};

static const int counter_registry_size =
    (int)(sizeof(counter_registry) / sizeof(counter_registry[0]));

static const CounterRegistryEntry* FindInRegistry(const std::string& name) {
  for (int i = 0; i < counter_registry_size; i++) {
    if (name == counter_registry[i].name) return &counter_registry[i];
  }
  return NULL;
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

bool NetCounterIsEnabled() {
  static int enabled = -1;
  if (enabled == -1) {
    const char* env = getenv("RCCL_TESTS_NET_COUNTER_ENABLE");
    enabled = (env && atoi(env) > 0) ? 1 : 0;
  }
  return enabled == 1;
}

std::vector<CounterDescriptor> NetCounterParseCounterList() {
  std::vector<CounterDescriptor> result;

  const char* env = getenv("RCCL_TESTS_NIC_COUNTER_LIST");
  if (!env || strlen(env) == 0) {
    for (int i = 0; i < counter_registry_size; i++) {
      result.push_back({counter_registry[i].name,
                        counter_registry[i].source,
                        counter_registry[i].is_prefix});
    }
    return result;
  }

  std::string list(env);

  size_t pos = 0;
  while (pos < list.size()) {
    size_t comma = list.find(',', pos);
    if (comma == std::string::npos) {
      comma = list.size();
    }

    std::string token = list.substr(pos, comma - pos);

    while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) {
      token.erase(token.begin());
    }

    while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) {
      token.pop_back();
    }

    if (!token.empty()) {
      const CounterRegistryEntry* entry = FindInRegistry(token);
      if (entry) {
        result.push_back({entry->name, entry->source, entry->is_prefix});
      } else {
        fprintf(stderr,
                "# Warning: counter \"%s\" not in registry, assuming ethtool\n",
                token.c_str());
        result.push_back({token, COUNTER_SRC_ETHTOOL, false});
      }
    }
    pos = comma + 1;
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
  char cmd[512] = {0};
  snprintf(cmd, sizeof(cmd),
           "ls /sys/class/infiniband/%s/device/net 2>/dev/null | head -1",
           ib_device.c_str());
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
    uint64_t value = 0;
    if (sscanf(line, "%255[^:]: %lu", key, &value) == 2) {
      char* start = key;
      while (*start == ' ' || *start == '\t') { start++; }
      std::string k(start);
      if (exact.count(k)) { out[k] = value; continue; }
      for (const auto& pfx : prefixes) {
        if (k.compare(0, pfx.size(), pfx) == 0) { out[k] = value; break; }
      }
    }
  }
  pclose(fp);
}

static void CollectIbHwCounters(const std::string& ib_device,
                                const std::vector<CounterDescriptor>& selected,
                                std::map<std::string, uint64_t>& out) {
  if (ib_device.empty()) { return; }

  std::string base =
      "/sys/class/infiniband/" + ib_device + "/ports/1/hw_counters/";

  for (const auto& d : selected) {
    if (d.source != COUNTER_SRC_IB_HW) { continue; }
    std::string path = base + d.name;
    FILE* fp = fopen(path.c_str(), "r");
    if (fp) {
      uint64_t value = 0;
      if (fscanf(fp, "%lu", &value) == 1) {
        out[d.name] = value;
      }
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
    if (d.source == COUNTER_SRC_DEBUGFS) { wanted.insert(d.name); }
  }
  if (wanted.empty()) { return; }

  std::string info_path =
      "/sys/kernel/debug/bnxt_re/" + ib_device + "/info";
  FILE* fp = fopen(info_path.c_str(), "r");
  if (!fp) { return; }

  char line[512] = {0};
  while (fgets(line, sizeof(line), fp)) {
    for (const auto& name : wanted) {
      if (strstr(line, name.c_str())) {
        uint64_t value = 0;
        char key[256] = {0};
        if (sscanf(line, " %255[^:=] %*[:=] %lu", key, &value) == 2 ||
            sscanf(line, " %255s %lu", key, &value) == 2) {
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

  snap.timestamp = time(NULL);

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

  // Pre-compute deltas, rates, durations
  std::vector<std::vector<uint64_t>> deltas(
      num_nics, std::vector<uint64_t>(num_counters, 0));

  std::vector<std::vector<double>> rates(
      num_nics, std::vector<double>(num_counters, 0.0));

  std::vector<long> durations(num_nics);

  for (size_t n = 0; n < num_nics; n++) {
    durations[n] = after[n].timestamp - before[n].timestamp;
    for (size_t c = 0; c < num_counters; c++) {
      deltas[n][c] = NetCounterComputeDelta(before[n], after[n], selected[c]);
      rates[n][c] = (durations[n] > 0)
                        ? (double)deltas[n][c] / durations[n]
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
      snprintf(buf, sizeof(buf), "%lu", deltas[n][c]);
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
  printf("\nNET_COUNTER_TABLE: node=%s  rank=%d  duration_sec=%ld\n",
         hostname, rank, durations[0]);
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
      printf("  %*lu  %*.2f", cnt_w[c], deltas[n][c], rt_w[c], rates[n][c]);
    }
    printf("\n");
  }

  printf("%s\n\n", sep.c_str());
  fflush(stdout);
}
