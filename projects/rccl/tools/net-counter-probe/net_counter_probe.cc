/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * net-counter-probe: how fresh are a NIC's counters, and how much does one
 * read cost?
 *
 * RCCL telemetry and the accl fabric report read PFC pause time, ECN/CNP and
 * byte counters from ethtool and IB sysfs. Some drivers refresh some of these
 * only every so often (e.g. ~1 s), and a counter that moves in 1 s jumps cannot
 * be differenced across a 200 us collective. This reads the counters in a
 * tight loop for a few seconds while traffic is running and reports, per
 * counter, how often its value actually changed, plus the cost of each read.
 *
 * Run it on a node while an rccl-tests job drives the same NIC.
 */

#include <errno.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <dirent.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

static int64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

struct Counter {
  std::string name;
  std::string source;      // "ethtool" or "sysfs"
  std::string path;        // sysfs only
  int ethIdx = -1;         // ethtool only
  bool haveLast = false;
  uint64_t last = 0;
  int64_t lastChangeNs = 0;
  std::vector<double> gapsMs;   // time between consecutive value changes
  std::vector<double> costUs;   // sysfs per-read cost
  uint64_t first = 0, final = 0;
};

static void observe(Counter& c, uint64_t v, int64_t t) {
  if (!c.haveLast) {
    c.haveLast = true;
    c.first = v;
    c.last = v;
    c.lastChangeNs = 0;  // first change after start has no previous change to measure from
  } else if (v != c.last) {
    if (c.lastChangeNs != 0) c.gapsMs.push_back((t - c.lastChangeNs) / 1e6);
    c.lastChangeNs = t;
    c.last = v;
  }
  c.final = v;
}

static double quantile(std::vector<double> v, double q) {
  if (v.empty()) return -1;
  std::sort(v.begin(), v.end());
  size_t i = (size_t)(q * (v.size() - 1));
  return v[i];
}

static bool readSysfs(const std::string& path, uint64_t* out) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  char buf[64];
  ssize_t n = read(fd, buf, sizeof(buf) - 1);
  close(fd);
  if (n <= 0) return false;
  buf[n] = '\0';
  *out = strtoull(buf, NULL, 10);
  return true;
}

static std::vector<std::string> split(const char* s) {
  std::vector<std::string> out;
  std::string cur;
  for (; s && *s; s++) {
    if (*s == ',') {
      if (!cur.empty()) out.push_back(cur);
      cur.clear();
    } else {
      cur += *s;
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

static bool contains(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

// Default ethtool keys: PFC pause duration under each supported driver's naming
// (mlx5 rx_prioN_pause_duration, AINIC rx_pripause_N_1us_count, bnxt
// pfc_priN_rx_duration_us) plus the mlx5 RoCE byte counters telemetry uses.
static bool defaultEthKey(const std::string& k) {
  return contains(k, "pause_duration") || contains(k, "_1us_count") ||
         contains(k, "duration_us") || k == "tx_vport_rdma_unicast_bytes" ||
         k == "rx_vport_rdma_unicast_bytes";
}

// Default sysfs counters: anything ECN/CNP-related, plus the IB port byte counters.
static bool defaultSysfsKey(const std::string& k) {
  return contains(k, "ecn") || contains(k, "cnp") || k == "port_xmit_data" || k == "port_rcv_data";
}

static void usage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s --eth IFACE [--ib DEV] [options]\n"
          "  --eth IFACE       netdev of the RoCE NIC (ethtool counters), e.g. enp12s0np0\n"
          "  --ib DEV          RDMA device (IB sysfs counters), e.g. mlx5_0\n"
          "  --port N          IB port for sysfs counters (default 1)\n"
          "  --keys a,b        ethtool keys to watch (default: PFC pause duration + RoCE bytes)\n"
          "  --sysfs a,b       sysfs counters under ports/N/{hw_counters,counters}/\n"
          "                    (default: every ECN/CNP counter + port_xmit_data/port_rcv_data)\n"
          "  --seconds S       how long to watch (default 10)\n"
          "  --sleep-us U      pause between read rounds (default 100)\n"
          "Run it while traffic (e.g. rccl-tests) is driving the NIC.\n",
          argv0);
}

int main(int argc, char** argv) {
  const char* eth = NULL;
  const char* ib = NULL;
  const char* keysArg = NULL;
  const char* sysfsArg = NULL;
  int port = 1;
  double seconds = 10;
  int sleepUs = 100;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    const char* v = (i + 1 < argc) ? argv[i + 1] : NULL;
    if (a == "--eth" && v) { eth = v; i++; }
    else if (a == "--ib" && v) { ib = v; i++; }
    else if (a == "--port" && v) { port = atoi(v); i++; }
    else if (a == "--keys" && v) { keysArg = v; i++; }
    else if (a == "--sysfs" && v) { sysfsArg = v; i++; }
    else if (a == "--seconds" && v) { seconds = atof(v); i++; }
    else if (a == "--sleep-us" && v) { sleepUs = atoi(v); i++; }
    else { usage(argv[0]); return 2; }
  }
  if (!eth && !ib) { usage(argv[0]); return 2; }

  std::vector<Counter> counters;

  // ---- ethtool setup: names once, then one GSTATS per round ----
  int fd = -1;
  unsigned int nStats = 0;
  struct ethtool_stats* stats = NULL;
  struct ifreq ifr;
  char driver[32] = "?";
  if (eth) {
    if (strlen(eth) >= IFNAMSIZ) { fprintf(stderr, "interface name too long\n"); return 1; }
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    memset(&ifr, 0, sizeof(ifr));
    memcpy(ifr.ifr_name, eth, strlen(eth));
    struct ethtool_drvinfo drv;
    memset(&drv, 0, sizeof(drv));
    drv.cmd = ETHTOOL_GDRVINFO;
    ifr.ifr_data = (char*)&drv;
    if (fd < 0 || ioctl(fd, SIOCETHTOOL, &ifr) != 0) {
      fprintf(stderr, "ethtool GDRVINFO on %s failed: %s\n", eth, strerror(errno));
      return 1;
    }
    snprintf(driver, sizeof(driver), "%s", drv.driver);
    nStats = drv.n_stats;
    struct ethtool_gstrings* strs =
        (struct ethtool_gstrings*)calloc(1, sizeof(*strs) + (size_t)nStats * ETH_GSTRING_LEN);
    strs->cmd = ETHTOOL_GSTRINGS;
    strs->string_set = ETH_SS_STATS;
    strs->len = nStats;
    ifr.ifr_data = (char*)strs;
    if (ioctl(fd, SIOCETHTOOL, &ifr) != 0) {
      fprintf(stderr, "ethtool GSTRINGS on %s failed: %s\n", eth, strerror(errno));
      return 1;
    }
    std::vector<std::string> wanted = split(keysArg);
    for (unsigned int s = 0; s < nStats; s++) {
      char name[ETH_GSTRING_LEN + 1];
      memcpy(name, strs->data + (size_t)s * ETH_GSTRING_LEN, ETH_GSTRING_LEN);
      name[ETH_GSTRING_LEN] = '\0';
      std::string n = name;
      bool want = keysArg ? std::find(wanted.begin(), wanted.end(), n) != wanted.end() : defaultEthKey(n);
      if (!want) continue;
      Counter c;
      c.name = n;
      c.source = "ethtool";
      c.ethIdx = (int)s;
      counters.push_back(c);
    }
    free(strs);
    // Slack: GSTATS writes as many values as the driver has *now*.
    stats = (struct ethtool_stats*)calloc(1, sizeof(*stats) + (size_t)(nStats + 256) * sizeof(uint64_t));
  }

  // ---- sysfs setup ----
  if (ib) {
    std::vector<std::string> wanted = split(sysfsArg);
    const char* subdirs[] = {"hw_counters", "counters"};
    for (const char* sub : subdirs) {
      char dir[512];
      snprintf(dir, sizeof(dir), "/sys/class/infiniband/%s/ports/%d/%s", ib, port, sub);
      DIR* d = opendir(dir);
      if (!d) continue;
      struct dirent* e;
      while ((e = readdir(d)) != NULL) {
        std::string n = e->d_name;
        if (n[0] == '.') continue;
        bool want = sysfsArg ? std::find(wanted.begin(), wanted.end(), n) != wanted.end() : defaultSysfsKey(n);
        if (!want) continue;
        Counter c;
        c.name = n;
        c.source = "sysfs";
        c.path = std::string(dir) + "/" + n;
        counters.push_back(c);
      }
      closedir(d);
    }
  }
  if (counters.empty()) {
    fprintf(stderr, "No matching counters found. List them with `ethtool -S %s` or "
                    "`ls /sys/class/infiniband/%s/ports/%d/hw_counters`, then pass --keys/--sysfs.\n",
            eth ? eth : "IFACE", ib ? ib : "DEV", port);
    return 1;
  }

  // ---- watch ----
  std::vector<double> gstatsUs;
  int64_t start = nowNs(), end = start + (int64_t)(seconds * 1e9);
  long rounds = 0, statsChanged = 0;
  while (nowNs() < end) {
    if (fd >= 0) {
      stats->cmd = ETHTOOL_GSTATS;
      stats->n_stats = nStats;
      ifr.ifr_data = (char*)stats;
      int64_t t0 = nowNs();
      int rc = ioctl(fd, SIOCETHTOOL, &ifr);
      int64_t t1 = nowNs();
      if (rc == 0) {
        gstatsUs.push_back((t1 - t0) / 1e3);
        if (stats->n_stats != nStats) statsChanged++;
        for (auto& c : counters)
          if (c.source == "ethtool" && (unsigned)c.ethIdx < stats->n_stats) observe(c, stats->data[c.ethIdx], t1);
      }
    }
    for (auto& c : counters) {
      if (c.source != "sysfs") continue;
      uint64_t v;
      int64_t t0 = nowNs();
      bool ok = readSysfs(c.path, &v);
      int64_t t1 = nowNs();
      if (!ok) continue;
      c.costUs.push_back((t1 - t0) / 1e3);
      observe(c, v, t1);
    }
    rounds++;
    if (sleepUs > 0) usleep(sleepUs);
  }

  // ---- report ----
  printf("net-counter-probe: eth=%s (driver %s) ib=%s port=%d, %.1f s, %ld read rounds\n",
         eth ? eth : "-", driver, ib ? ib : "-", port, seconds, rounds);
  if (!gstatsUs.empty())
    printf("ethtool GSTATS (%u stats): read cost median %.1f us, p99 %.1f us, max %.1f us\n", nStats,
           quantile(gstatsUs, 0.5), quantile(gstatsUs, 0.99), quantile(gstatsUs, 1.0));
  if (statsChanged)
    printf("WARNING: the ethtool stat count changed during the run (%ld rounds)\n", statsChanged);
  printf("\n%-34s %-7s %8s %8s  %-26s %s\n", "counter", "source", "changes", "read us",
         "change every ms (p50/min/max)", "verdict");
  for (auto& c : counters) {
    size_t changes = c.gapsMs.size() + (c.lastChangeNs ? 1 : 0);
    double cost = c.source == "sysfs" ? quantile(c.costUs, 0.5) : quantile(gstatsUs, 0.5);
    char gap[64] = "-";
    const char* verdict;
    double p50 = quantile(c.gapsMs, 0.5);
    if (changes < 3) {
      verdict = "did not move: no traffic or no pause on it during the probe";
    } else if (p50 <= 2) {
      verdict = "live: usable per collective";
    } else if (p50 <= 50) {
      verdict = "near-live: per collective only if it runs >> this";
    } else {
      verdict = "batched: use time windows, not per collective";
    }
    if (!c.gapsMs.empty())
      snprintf(gap, sizeof(gap), "%.2f / %.2f / %.2f", p50, quantile(c.gapsMs, 0.0), quantile(c.gapsMs, 1.0));
    printf("%-34s %-7s %8zu %8.1f  %-26s %s\n", c.name.c_str(), c.source.c_str(), changes, cost, gap, verdict);
  }
  printf("\nA counter that moves in fixed-size steps (e.g. every ~1000 ms under steady\n"
         "traffic) is refreshed by the driver, not live. Compare the 'change every'\n"
         "column with your collective durations.\n");
  free(stats);
  if (fd >= 0) close(fd);
  return 0;
}
