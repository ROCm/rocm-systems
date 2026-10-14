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
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <set>
#include <dirent.h>
#include <unistd.h>

// =====================================================================
// Per-NIC-type counter tables
// =====================================================================
//
// Each NIC type has its own table with actual hardware key names.
// name          : column name; also the lookup key unless 'key' is set
// source        : where to read the value
// is_prefix     : true → sum of all keys starting with the key; a '*' in the
//                 key matches a priority number (pfc_pri*_rx_transitions)
// fallback_name : alternative key to try if the key is not found (or NULL)
// key           : lookup key or pattern when it differs from name (or NULL)

struct CounterTableEntry {
  const char*   name;
  CounterSource source;
  bool          is_prefix;
  const char*   fallback_name;  // NULL = no fallback
  const char*   key = nullptr;  // NULL = name
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
  {"rx_stat_discards",        COUNTER_SRC_ETHTOOL,  false, "rx_stat_discard"},
  {"to_retransmits",          COUNTER_SRC_IB_HW,    false, NULL},
  {"max_retry_exceeded",      COUNTER_SRC_IB_HW,    false, NULL},
  {"oos_drop_count",          COUNTER_SRC_IB_HW,    false, "oos_drop_count", "res_oos_drop_count"},
  {"seq_err_naks_rcvd",       COUNTER_SRC_IB_HW,    false, NULL},
  // Appended to keep the earlier columns in place.
  {"pfc_rx_transitions",      COUNTER_SRC_ETHTOOL,  true,  NULL, "pfc_pri*_rx_transitions"},
  {"pfc_tx_transitions",      COUNTER_SRC_ETHTOOL,  true,  NULL, "pfc_pri*_tx_transitions"},
};
static const int bnxt_re_table_size = (int)(sizeof(bnxt_re_table) / sizeof(bnxt_re_table[0]));

// ----- MLX5 (ConnectX) counter table -----
static const CounterTableEntry mlx5_table[] = {
  {"rx_prio_pause",              COUNTER_SRC_ETHTOOL,  true,  NULL, "rx_prio*_pause"},
  {"tx_prio_pause",              COUNTER_SRC_ETHTOOL,  true,  NULL, "tx_prio*_pause"},
  {"rp_cnp_handled",             COUNTER_SRC_IB_HW,    false, NULL},
  {"np_cnp_sent",                COUNTER_SRC_IB_HW,    false, NULL},
  {"np_ecn_marked_roce_packets", COUNTER_SRC_IB_HW,    false, NULL},
  {"out_of_buffer",              COUNTER_SRC_IB_HW,    false, NULL},
  {"out_of_sequence",            COUNTER_SRC_IB_HW,    false, NULL},
  {"packet_seq_err",             COUNTER_SRC_IB_HW,    false, NULL},
  {"local_ack_timeout_err",      COUNTER_SRC_IB_HW,    false, NULL},
  {"rnr_nak_retry_err",          COUNTER_SRC_IB_HW,    false, NULL},
  {"rx_discards_phy",            COUNTER_SRC_ETHTOOL,  false, NULL},
};
static const int mlx5_table_size = (int)(sizeof(mlx5_table) / sizeof(mlx5_table[0]));

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
    case NIC_MLX5:
      table = mlx5_table;
      size = mlx5_table_size;
      break;
    case NIC_UNKNOWN:
    default:
      fprintf(stderr,
              "# Warning: NIC type unknown, defaulting to bnxt_re counter set\n");
      table = bnxt_re_table;
      size = bnxt_re_table_size;
      break;
  }
}

// Match an ethtool/sysfs key against a prefix pattern ('*' = digits).
static bool PrefixMatches(const std::string& pattern, const std::string& key) {
  size_t star = pattern.find('*');
  if (star == std::string::npos) { return key.compare(0, pattern.size(), pattern) == 0; }
  std::string pre = pattern.substr(0, star), suf = pattern.substr(star + 1);
  if (key.size() <= pre.size() + suf.size()) { return false; }
  if (key.compare(0, pre.size(), pre) != 0) { return false; }
  if (key.compare(key.size() - suf.size(), suf.size(), suf) != 0) { return false; }
  for (size_t i = pre.size(); i < key.size() - suf.size(); i++) {
    if (!isdigit((unsigned char)key[i])) { return false; }
  }
  return true;
}

static const std::string& LookupKey(const CounterDescriptor& d) {
  return d.key.empty() ? d.name : d.key;
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
  if (entry.key != NULL && filter.count(entry.key) > 0) { return true; }
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
                      table[i].fallback_name ? table[i].fallback_name : "",
                      table[i].key ? table[i].key : ""});
  }
  return result;
}

// RCCL-like NCCL_IB_HCA matching: a leading '^' excludes, then '=' matches
// names exactly, otherwise each entry is a prefix. Entries are name[:port].
// Port 1 is collected unless an include entry names another port; an empty
// list selects every device ('^' alone selects none).
std::vector<std::string> NetCounterFilterIbHca(const char* env, std::vector<std::string> devs) {
  std::vector<std::string> result;
  if (!env || strlen(env) == 0) { return result; }

  bool exclude = false, exact = false;
  if (*env == '^') { exclude = true; env++; }
  if (*env == '=') { exact = true; env++; }

  struct Entry { std::string name; int port; };
  std::vector<Entry> entries;
  std::string list(env);
  size_t pos = 0;
  while (pos < list.size()) {
    size_t comma = list.find(',', pos);
    if (comma == std::string::npos) { comma = list.size(); }
    std::string token = list.substr(pos, comma - pos);
    pos = comma + 1;
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);
    size_t colon = token.find(':');
    std::string name = token.substr(0, colon);
    if (name.empty()) { continue; }
    int port = -1;
    if (colon != std::string::npos && colon + 1 < token.size() && token[colon + 1] != ':') {
      port = std::max(0, atoi(token.c_str() + colon + 1));  // 0 matches no port
    }
    entries.push_back({name, port});
  }

  std::sort(devs.begin(), devs.end(), [](const std::string& a, const std::string& b) {
    return strverscmp(a.c_str(), b.c_str()) < 0;
  });
  auto matches = [&](const std::string& dev, const Entry& en) {
    return exact ? dev == en.name : dev.compare(0, en.name.size(), en.name) == 0;
  };

  if (exclude) {
    if (entries.empty()) { return result; }
    for (const auto& dev : devs) {
      bool hit = false;
      for (const auto& en : entries) { hit |= matches(dev, en) && (en.port == -1 || en.port == 1); }
      if (!hit) { result.push_back(dev); }
    }
    return result;
  }
  if (entries.empty()) { return devs; }
  for (const auto& en : entries) {
    if (en.port == 0) { continue; }
    for (const auto& dev : devs) {
      if (!matches(dev, en)) { continue; }
      std::string sel = en.port > 0 ? dev + ":" + std::to_string(en.port) : dev;
      if (std::find(result.begin(), result.end(), sel) == result.end()) { result.push_back(sel); }
    }
  }
  return result;
}

std::vector<std::string> NetCounterParseIbHcaList() {
  const char* env = getenv("NCCL_IB_HCA");
  if (!env || strlen(env) == 0) { return {}; }
  std::vector<std::string> devs;
  DIR* dir = opendir("/sys/class/infiniband");
  if (dir) {
    for (struct dirent* e; (e = readdir(dir)) != NULL;)
      if (e->d_name[0] != '.') devs.push_back(e->d_name);
    closedir(dir);
  }
  return NetCounterFilterIbHca(env, devs);
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
  if (strcmp(drv, "mlx5_core") == 0)
    return NIC_MLX5;
  return NIC_UNKNOWN;
}

const char* NicTypeStr(NicType t) {
  switch (t) {
    case NIC_BNXT_RE: return "BNXT_RE";
    case NIC_IONIC:   return "IONIC";
    case NIC_MLX5:    return "MLX5";
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

void NetCounterParseEthtool(const std::string& text,
                            const std::vector<CounterDescriptor>& selected,
                            std::map<std::string, uint64_t>& out) {
  std::map<std::string, uint64_t> stats;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) { eol = text.size(); }
    std::string line = text.substr(pos, eol - pos);
    pos = eol + 1;
    char key[256] = {0};
    unsigned long long tmp = 0;
    // Per-queue lines ("[0]: rx_discards: 0") do not parse and are skipped.
    if (sscanf(line.c_str(), "%255[^:]: %llu", key, &tmp) == 2) {
      char* start = key;
      while (*start == ' ' || *start == '\t') { start++; }
      stats[start] = (uint64_t)tmp;
    }
  }

  for (const auto& d : selected) {
    if (d.source != COUNTER_SRC_ETHTOOL) { continue; }
    if (d.is_prefix) {
      for (const auto& kv : stats) {
        if (PrefixMatches(LookupKey(d), kv.first)) { out[kv.first] = kv.second; }
      }
      continue;
    }
    auto it = stats.find(LookupKey(d));
    if (it == stats.end() && !d.fallback_name.empty()) { it = stats.find(d.fallback_name); }
    if (it != stats.end()) { out[d.name] = it->second; }
  }
}

static void CollectEthtoolCounters(const std::string& nic,
                                   const std::vector<CounterDescriptor>& selected,
                                   std::map<std::string, uint64_t>& out) {
  bool wanted = false;
  for (const auto& d : selected) { wanted |= (d.source == COUNTER_SRC_ETHTOOL); }
  if (!wanted || nic.empty()) { return; }

  char command[512] = {0};
  snprintf(command, sizeof(command), "ethtool -S %s 2>/dev/null", nic.c_str());

  FILE* fp = popen(command, "r");
  if (!fp) { return; }
  std::string text;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) { text.append(buf, n); }
  pclose(fp);
  NetCounterParseEthtool(text, selected, out);
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
    if (d.source != COUNTER_SRC_IB_HW) { continue; }
    FILE* fp = fopen((base_port + LookupKey(d)).c_str(), "r");
    if (!fp) { fp = fopen((base_dev + LookupKey(d)).c_str(), "r"); }
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

void NetCounterParseDebugfs(const std::string& text, const std::set<std::string>& wanted,
                            std::map<std::string, uint64_t>& out) {
  size_t pos = 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) { eol = text.size(); }
    std::string line = text.substr(pos, eol - pos);
    pos = eol + 1;
    for (const auto& name : wanted) {
      if (line.find(name) == std::string::npos) { continue; }
      unsigned long long tmp = 0;
      char key[256] = {0};
      if (sscanf(line.c_str(), " %255[^:=] %*[:=] %llu", key, &tmp) == 2 ||
          sscanf(line.c_str(), " %255s %llu", key, &tmp) == 2) {
        char* start = key;
        while (*start == ' ' || *start == '\t') { start++; }
        char* end = start + strlen(start) - 1;
        while (end > start && (*end == ' ' || *end == '\t')) { *end-- = '\0'; }
        if (wanted.count(std::string(start))) { out[std::string(start)] = (uint64_t)tmp; }
      }
      break;
    }
  }
}

// Fills counters still missing after sysfs/ethtool from bnxt_re debugfs,
// which needs root.
static void CollectDebugCounters(const std::string& ib_device,
                                 const std::vector<CounterDescriptor>& selected,
                                 std::map<std::string, uint64_t>& out) {
  if (ib_device.empty()) { return; }

  std::set<std::string> wanted;
  for (const auto& d : selected) {
    if (!d.is_prefix && out.count(d.name) == 0) { wanted.insert(d.name); }
  }
  if (wanted.empty()) { return; }

  std::string ib_base;
  int port;
  ParseIbDevice(ib_device, ib_base, port);
  (void)port;
  std::string info_path = "/sys/kernel/debug/bnxt_re/" + ib_base + "/info";
  FILE* fp = fopen(info_path.c_str(), "r");
  if (!fp) { return; }
  std::string text;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) { text.append(buf, n); }
  fclose(fp);
  NetCounterParseDebugfs(text, wanted, out);
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
  if (snap.nic_type == NIC_BNXT_RE) { CollectDebugCounters(ib_dev, selected, snap.counters); }

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

bool NetCounterComputeDelta(
    const NetworkCounterSnapshot& before,
    const NetworkCounterSnapshot& after,
    const CounterDescriptor& desc,
    uint64_t* delta) {
  bool found = false;
  *delta = 0;
  if (desc.is_prefix) {
    for (const auto& entry : after.counters) {
      if (!PrefixMatches(LookupKey(desc), entry.first)) { continue; }
      auto it = before.counters.find(entry.first);
      if (it == before.counters.end()) { continue; }
      *delta += entry.second - it->second;
      found = true;
    }
  } else {
    auto b = before.counters.find(desc.name);
    auto a = after.counters.find(desc.name);
    if (b != before.counters.end() && a != after.counters.end()) {
      *delta = a->second - b->second;
      found = true;
    }
  }
  return found;
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

  std::vector<std::vector<bool>> valid(
      num_nics, std::vector<bool>(num_counters, false));

  std::vector<std::vector<double>> rates(
      num_nics, std::vector<double>(num_counters, 0.0));

  std::vector<double> dur_sec(num_nics);

  for (size_t n = 0; n < num_nics; n++) {
    dur_sec[n] = (after[n].timestamp_us - before[n].timestamp_us) / 1e6;
    for (size_t c = 0; c < num_counters; c++) {
      valid[n][c] = NetCounterComputeDelta(before[n], after[n], selected[c], &deltas[n][c]);
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
      if (valid[n][c]) {
        printf("  %*llu  %*.2f", cnt_w[c], (unsigned long long)deltas[n][c], rt_w[c], rates[n][c]);
      } else {
        printf("  %*s  %*s", cnt_w[c], "N/A", rt_w[c], "N/A");
      }
    }
    printf("\n");
  }

  printf("%s\n\n", sep.c_str());
  fflush(stdout);
}
