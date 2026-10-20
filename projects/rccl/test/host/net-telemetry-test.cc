/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_telemetry.cc and src/include/net_telemetry.h.

#include <dirent.h>
#include <fcntl.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "../common/ProcessIsolatedTestRunner.hpp"
#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/libc_fakes.h"

int NetTelemetryTestClockGettime(clockid_t clk, struct timespec* ts);
int NetTelemetryTestPosixMemalign(void** ptr, size_t alignment, size_t size);
int NetTelemetryTestPthreadCreate(pthread_t* thread, const pthread_attr_t* attr, void* (*fn)(void*), void* arg);
int NetTelemetryTestNanosleep(const struct timespec* req, struct timespec* rem);

#define clock_gettime NetTelemetryTestClockGettime
#define posix_memalign NetTelemetryTestPosixMemalign
#define pthread_create NetTelemetryTestPthreadCreate
#define nanosleep NetTelemetryTestNanosleep
#define getenv micro_getenv

#include "fakes/libc_seam.h"
#include NET_TELEMETRY_CC_PATH
#include "fakes/libc_seam_undef.h"

#undef getenv
#undef nanosleep
#undef pthread_create
#undef posix_memalign
#undef clock_gettime

namespace {

int64_t g_nowNs = 1000000000000;

int DefaultPosixMemalign(void** ptr, size_t alignment, size_t size) { return ::posix_memalign(ptr, alignment, size); }
void* RealCalloc(size_t count, size_t size) { return std::calloc(count, size); }
int DefaultPthreadCreate(pthread_t* thread, const pthread_attr_t* attr, void* (*fn)(void*), void* arg) {
  return ::pthread_create(thread, attr, fn, arg);
}
int DefaultNanosleep(const struct timespec* req, struct timespec* rem) { return ::nanosleep(req, rem); }

std::function<int(void**, size_t, size_t)> g_posixMemalign = DefaultPosixMemalign;
std::function<int(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*)> g_pthreadCreate = DefaultPthreadCreate;
std::function<int(const struct timespec*, struct timespec*)> g_nanosleep = DefaultNanosleep;

}  // namespace

int NetTelemetryTestClockGettime(clockid_t, struct timespec* ts) {
  ts->tv_sec = g_nowNs / 1000000000;
  ts->tv_nsec = g_nowNs % 1000000000;
  return 0;
}
int NetTelemetryTestPosixMemalign(void** ptr, size_t alignment, size_t size) {
  return g_posixMemalign(ptr, alignment, size);
}
int NetTelemetryTestPthreadCreate(pthread_t* thread, const pthread_attr_t* attr, void* (*fn)(void*), void* arg) {
  return g_pthreadCreate(thread, attr, fn, arg);
}
int NetTelemetryTestNanosleep(const struct timespec* req, struct timespec* rem) { return g_nanosleep(req, rem); }

namespace {

constexpr const char* kIb = "/sys/class/infiniband/";

// In-memory sysfs, procfs, environment and ethtool behind the libc seam.
struct FakeSys {
  struct Output {
    char* buf = nullptr;
    size_t len = 0;
  };
  struct Dir {
    std::vector<dirent> entries;
    size_t pos = 0;
  };

  std::map<std::string, std::string> files;
  std::map<std::string, std::string> links;
  std::map<std::string, std::vector<std::string>> dirs;
  std::map<std::string, std::vector<std::pair<std::string, uint64_t>>> nics;
  uint32_t failEthtoolCmd = 0;
  bool failWrite = false;

  std::map<int, std::string> fds;
  int nextFd = 300;
  std::vector<std::string> opened;
  std::vector<std::string> fopened;
  std::deque<std::string> readBufs;
  std::map<std::string, Output> outputs;
  std::deque<Dir> openDirs;
  int ethtoolCalls = 0;

  void Reset() {
    for (auto& kv : outputs) free(kv.second.buf);
    *this = FakeSys();
  }

  std::string Written(const std::string& path) {
    auto it = outputs.find(path);
    return it == outputs.end() || it->second.buf == nullptr ? "" : std::string(it->second.buf, it->second.len);
  }
};
FakeSys g_sys;

int FakeOpen(const char* path, int, mode_t) {
  g_sys.opened.push_back(path);
  auto it = g_sys.files.find(path);
  if (it == g_sys.files.end()) {
    errno = ENOENT;
    return -1;
  }
  int fd = g_sys.nextFd++;
  g_sys.fds[fd] = it->second;
  return fd;
}

ssize_t FakeRead(int fd, void* buf, size_t count) {
  auto it = g_sys.fds.find(fd);
  if (it == g_sys.fds.end()) {
    errno = EBADF;
    return -1;
  }
  size_t n = std::min(count, it->second.size());
  std::memcpy(buf, it->second.data(), n);
  it->second.erase(0, n);
  return static_cast<ssize_t>(n);
}

int FakeClose(int fd) {
  g_closedFds.push_back(fd);
  g_sys.fds.erase(fd);
  return 0;
}

FILE* FakeFopen(const char* path, const char* mode) {
  g_sys.fopened.push_back(path);
  if (mode[0] == 'w') {
    if (g_sys.failWrite) {
      errno = EACCES;
      return nullptr;
    }
    FakeSys::Output& out = g_sys.outputs[path];
    free(out.buf);
    out = FakeSys::Output();
    return open_memstream(&out.buf, &out.len);
  }
  auto it = g_sys.files.find(path);
  if (it == g_sys.files.end()) {
    errno = ENOENT;
    return nullptr;
  }
  if (it->second.empty()) return ::fopen("/dev/null", "r");
  g_sys.readBufs.push_back(it->second);
  return fmemopen(&g_sys.readBufs.back()[0], g_sys.readBufs.back().size(), "r");
}

DIR* FakeOpendir(const char* path) {
  auto it = g_sys.dirs.find(path);
  if (it == g_sys.dirs.end()) {
    errno = ENOENT;
    return nullptr;
  }
  FakeSys::Dir dir;
  for (const std::string& name : it->second) {
    dirent entry{};
    std::snprintf(entry.d_name, sizeof(entry.d_name), "%s", name.c_str());
    dir.entries.push_back(entry);
  }
  g_sys.openDirs.push_back(dir);
  return reinterpret_cast<DIR*>(&g_sys.openDirs.back());
}

struct dirent* FakeReaddir(DIR* dirp) {
  auto* dir = reinterpret_cast<FakeSys::Dir*>(dirp);
  return dir->pos < dir->entries.size() ? &dir->entries[dir->pos++] : nullptr;
}

int FakeClosedir(DIR*) { return 0; }

ssize_t FakeReadlink(const char* path, char* buf, size_t len) {
  auto it = g_sys.links.find(path);
  if (it == g_sys.links.end()) {
    errno = ENOENT;
    return -1;
  }
  size_t n = std::min(len, it->second.size());
  std::memcpy(buf, it->second.data(), n);
  return static_cast<ssize_t>(n);
}

int FakeIoctl(int, unsigned long request, void* arg) {
  auto* ifr = static_cast<struct ifreq*>(arg);
  auto nic = g_sys.nics.find(ifr->ifr_name);
  if (request != SIOCETHTOOL || nic == g_sys.nics.end()) {
    errno = ENODEV;
    return -1;
  }
  g_sys.ethtoolCalls++;
  uint32_t cmd = *reinterpret_cast<uint32_t*>(ifr->ifr_data);
  if (cmd == g_sys.failEthtoolCmd) {
    errno = EIO;
    return -1;
  }
  const auto& stats = nic->second;
  if (cmd == ETHTOOL_GDRVINFO) {
    reinterpret_cast<struct ethtool_drvinfo*>(ifr->ifr_data)->n_stats = static_cast<uint32_t>(stats.size());
  } else if (cmd == ETHTOOL_GSTRINGS) {
    auto* strings = reinterpret_cast<struct ethtool_gstrings*>(ifr->ifr_data);
    for (size_t i = 0; i < stats.size(); i++) {
      std::strncpy(reinterpret_cast<char*>(strings->data) + i * ETH_GSTRING_LEN, stats[i].first.c_str(),
                   ETH_GSTRING_LEN);
    }
  } else if (cmd == ETHTOOL_GSTATS) {
    auto* values = reinterpret_cast<struct ethtool_stats*>(ifr->ifr_data);
    for (size_t i = 0; i < stats.size(); i++) values->data[i] = stats[i].second;
  }
  return 0;
}

int FakeGethostname(char* name, size_t len) {
  std::snprintf(name, len, "testhost");
  return 0;
}

void FreeTelemetryStorage() {
  for (RcclDeviceStats& dev : rcclTelemetryDevs) {
    if (dev.channel_blocks == nullptr) continue;
    for (int b = 0; b < RCCL_TELEMETRY_BLOCKS; b++) {
      auto* block = static_cast<RcclChannelStats*>(dev.channel_blocks[b]);
      if (block == nullptr) continue;
      int entries = (1 << dev.channel_block0_log2) << b;
      for (int e = 0; e < entries; e++) {
        if (block[e].qp_blocks == nullptr) continue;
        for (int q = 0; q < RCCL_TELEMETRY_BLOCKS; q++) free(block[e].qp_blocks[q]);
        free(block[e].qp_blocks);
      }
      free(block);
    }
    free(dev.channel_blocks);
  }
}

void ResetTelemetryState() {
  if (rcclTelemetrySamplerRunning) rcclTelemetrySamplerStop();
  FreeTelemetryStorage();
  std::memset(rcclTelemetryDevs, 0, sizeof(rcclTelemetryDevs));
  std::memset(&rcclTelemetryCfg, 0, sizeof(rcclTelemetryCfg));
  rcclTelemetryEnabled = 0;
  rcclTelemetryNumDevs = 0;
  rcclTelemetryLatencySampleMask = 0;
  rcclTelemetryLatencySampleN = 1;
  rcclTelemetryStartTime[0] = '\0';
  rcclTelemetryProcessName[0] = '\0';
  free(rcclTelemetrySamples);
  rcclTelemetrySamples = nullptr;
  rcclTelemetryNumSamples = 0;
  rcclTelemetrySampleIntervalMs = 0;
  rcclTelemetrySamplerStopFlag = 0;
  rcclTelemetryInitOnceControl = PTHREAD_ONCE_INIT;
  rcclTelemetryInitStatus = 0;
}

int StderrWrites() {
  return static_cast<int>(std::count(g_fprintfCalls.begin(), g_fprintfCalls.end(), stderr));
}

class NetTelemetryMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetTelemetryState();
    g_sys.Reset();
    g_nowNs = 1000000000000;
    g_open = FakeOpen;
    g_read = FakeRead;
    g_close = FakeClose;
    g_fopen = FakeFopen;
    g_opendir = FakeOpendir;
    g_readdir = FakeReaddir;
    g_closedir = FakeClosedir;
    g_readlink = FakeReadlink;
    g_ioctl = FakeIoctl;
    g_gethostname = FakeGethostname;
  }

  void TearDown() override {
    ResetTelemetryState();
    g_sys.Reset();
    ResetLibcFakes();
    ResetEnvFakes();
    g_posixMemalign = DefaultPosixMemalign;
    g_pthreadCreate = DefaultPthreadCreate;
    g_nanosleep = DefaultNanosleep;
  }

  static void Enable() {
    SetMicroEnv("RCCL_TELEMETRY_ENABLE", "1");
    ASSERT_EQ(rcclTelemetryInit(), 0);
    ASSERT_TRUE(rcclTelemetryOn());
  }

  static void AddNic(const std::string& roce, const std::string& eth, const std::string& driver) {
    g_sys.dirs[kIb + roce + "/device/net"] = {".", "..", eth};
    g_sys.links[kIb + roce + "/device/driver"] = "../../../bus/pci/drivers/" + driver;
  }

  static void SetHwCounter(const std::string& roce, const std::string& name, int64_t value) {
    g_sys.files[kIb + roce + "/hw_counters/" + name] = std::to_string(value) + "\n";
  }

  static void SetEthtool(const std::string& eth, const std::string& name, uint64_t value) {
    auto& stats = g_sys.nics[eth];
    for (auto& stat : stats) {
      if (stat.first == name) {
        stat.second = value;
        return;
      }
    }
    stats.emplace_back(name, value);
  }

  // Registered and set up on one channel, so the HW-counter baseline is taken.
  static RcclDeviceStats* UsedDevice(int idx, const std::string& roce, const std::string& eth,
                                     const std::string& driver) {
    AddNic(roce, eth, driver);
    EXPECT_EQ(rcclTelemetryRegisterDevice(idx, roce.c_str(), "IB"), idx);
    int slots = 0;
    EXPECT_EQ(rcclTelemetrySetupChannel(idx, 0, 1, &slots), 0);
    return &rcclTelemetryDevs[idx];
  }

  static int CounterIndex(const RcclHwConfig* hw, const char* jsonName) {
    for (int c = 0; c < hw->num_counters; c++) {
      if (std::strcmp(hw->counters[c].json_name, jsonName) == 0) return c;
    }
    ADD_FAILURE() << "no counter " << jsonName << " in " << hw->name;
    return 0;
  }

  static RcclQpStats* SetupQps(int dev, int ch, int numQps) {
    int slots = 0;
    int first = rcclTelemetrySetupChannel(dev, ch, numQps, &slots);
    EXPECT_EQ(slots, numQps);
    return rcclTelemetryResolveQp(dev, ch, first);
  }

  static std::string WriteJson() {
    char* buf = nullptr;
    size_t len = 0;
    FILE* fp = open_memstream(&buf, &len);
    rcclTelemetryWriteJson(fp);
    std::fclose(fp);
    std::string out(buf, len);
    free(buf);
    return out;
  }

  static std::string FlushPath() {
    return "/out/rccl_telemetry_testhost_" + std::to_string(getuid()) + "_" + std::to_string(getpid()) + ".json";
  }
};

// ---------------------------------------------------------------------------

using NetTelemetryInitMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryInitMicrotest, Init_EnableUnset_StaysDisabled) {
  EXPECT_EQ(rcclTelemetryInit(), 0);

  EXPECT_FALSE(rcclTelemetryOn());
  EXPECT_TRUE(g_atexitFns.empty());
  EXPECT_STREQ(rcclTelemetryCfg.output_dir, "");
}

TEST_F(NetTelemetryInitMicrotest, Init_EnableNotOne_StaysDisabled) {
  SetMicroEnv("RCCL_TELEMETRY_ENABLE", "true");

  EXPECT_EQ(rcclTelemetryInit(), 0);

  EXPECT_FALSE(rcclTelemetryOn());
}

TEST_F(NetTelemetryInitMicrotest, Init_Enabled_AppliesDefaultsAndRegistersExitFlush) {
  rcclTelemetryNumDevs = 4;
  rcclTelemetryDevs[3].tx_bytes = 9;
  std::snprintf(rcclTelemetryCfg.hw_counter_list, sizeof(rcclTelemetryCfg.hw_counter_list), "stale");

  Enable();

  EXPECT_STREQ(rcclTelemetryCfg.output_dir, "/tmp");
  EXPECT_EQ(rcclTelemetryCfg.histogram_max_buckets, 5);
  EXPECT_EQ(rcclTelemetryCfg.histogram_bucket_interval_ns, 30000);
  EXPECT_EQ(rcclTelemetryCfg.histogram_recip, (uint64_t{1} << 63) / 30000 + 1);
  EXPECT_STREQ(rcclTelemetryCfg.hw_counter_list, "");
  EXPECT_EQ(rcclTelemetryLatencySampleN, 1);
  EXPECT_EQ(rcclTelemetryLatencySampleMask, 0u);
  EXPECT_EQ(rcclTelemetryNumDevs, 0);
  EXPECT_EQ(rcclTelemetryDevs[3].tx_bytes, 0u);
  EXPECT_FALSE(rcclTelemetrySamplerRunning);
  EXPECT_EQ(g_atexitFns, std::vector<void (*)(void)>{rcclTelemetryFlushAtExit});
  EXPECT_TRUE(std::regex_match(rcclTelemetryStartTime, std::regex(R"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")));
  EXPECT_EQ(StderrWrites(), 0);
}

TEST_F(NetTelemetryInitMicrotest, Init_Enabled_SeedsEveryDeviceCounterAsUnavailable) {
  Enable();

  for (const RcclDeviceStats& dev : rcclTelemetryDevs) {
    for (int c = 0; c < RCCL_TELEMETRY_MAX_HWC; c++) {
      ASSERT_EQ(dev.hw_counters[c], -1);
      ASSERT_EQ(dev.snap_init_hw_counters[c], -1);
    }
    for (int p = 0; p < RCCL_TELEMETRY_NUM_PFC_PRIO; p++) {
      ASSERT_EQ(dev.pfc_rx_frames[p], -1);
      ASSERT_EQ(dev.pfc_tx_frames[p], -1);
      ASSERT_EQ(dev.pfc_rx_pause_us[p], -1);
      ASSERT_EQ(dev.pfc_tx_pause_us[p], -1);
      ASSERT_EQ(dev.snap_init_pfc_rx_frames[p], -1);
      ASSERT_EQ(dev.snap_init_pfc_tx_frames[p], -1);
      ASSERT_EQ(dev.snap_init_pfc_rx_pause_us[p], -1);
      ASSERT_EQ(dev.snap_init_pfc_tx_pause_us[p], -1);
    }
    ASSERT_EQ(dev.snap_init_tx_bytes, -1);
    ASSERT_EQ(dev.snap_init_rx_bytes, -1);
    ASSERT_EQ(dev.snap_init_tx_packets, -1);
    ASSERT_EQ(dev.snap_init_rx_packets, -1);
    ASSERT_EQ(dev.delta_tx_bytes, -1);
    ASSERT_EQ(dev.delta_rx_bytes, -1);
    ASSERT_EQ(dev.delta_tx_packets, -1);
    ASSERT_EQ(dev.delta_rx_packets, -1);
  }
}

TEST_F(NetTelemetryInitMicrotest, Init_EnvOverrides_AreApplied) {
  SetMicroEnv("RCCL_TELEMETRY_OUTPUT_DIR", "/out");
  SetMicroEnv("RCCL_TELEMETRY_HISTOGRAM_BUCKETS", "16");
  SetMicroEnv("RCCL_TELEMETRY_HISTOGRAM_INTERVAL_NS", "1000");
  SetMicroEnv("RCCL_TELEMETRY_HW_COUNTERS", "cnp_sent, ecn_marked_pkts");

  Enable();

  EXPECT_STREQ(rcclTelemetryCfg.output_dir, "/out");
  EXPECT_EQ(rcclTelemetryCfg.histogram_max_buckets, 16);
  EXPECT_EQ(rcclTelemetryCfg.histogram_bucket_interval_ns, 1000);
  EXPECT_EQ(rcclTelemetryCfg.histogram_recip, (uint64_t{1} << 63) / 1000 + 1);
  EXPECT_STREQ(rcclTelemetryCfg.hw_counter_list, "cnp_sent, ecn_marked_pkts");
}

TEST_F(NetTelemetryInitMicrotest, Init_EmptyOverrides_KeepDefaults) {
  SetMicroEnv("RCCL_TELEMETRY_OUTPUT_DIR", "");
  SetMicroEnv("RCCL_TELEMETRY_LATENCY_SAMPLE", "");
  SetMicroEnv("RCCL_TELEMETRY_HW_COUNTERS", "");

  Enable();

  EXPECT_STREQ(rcclTelemetryCfg.output_dir, "/tmp");
  EXPECT_EQ(rcclTelemetryLatencySampleN, 1);
  EXPECT_STREQ(rcclTelemetryCfg.hw_counter_list, "");
  EXPECT_EQ(StderrWrites(), 0);
}

TEST_F(NetTelemetryInitMicrotest, Init_OutputDirAtLimit_IsAccepted) {
  std::string dir(sizeof(rcclTelemetryCfg.output_dir) - 1, 'd');
  SetMicroEnv("RCCL_TELEMETRY_OUTPUT_DIR", dir.c_str());

  Enable();

  EXPECT_EQ(rcclTelemetryCfg.output_dir, dir);
}

TEST_F(NetTelemetryInitMicrotest, Init_OutputDirTooLong_StaysDisabledAndReturnsError) {
  SetMicroEnv("RCCL_TELEMETRY_ENABLE", "1");
  SetMicroEnv("RCCL_TELEMETRY_OUTPUT_DIR", std::string(sizeof(rcclTelemetryCfg.output_dir), 'd').c_str());

  EXPECT_EQ(rcclTelemetryInit(), -1);

  EXPECT_FALSE(rcclTelemetryOn());
  EXPECT_TRUE(g_atexitFns.empty());
  EXPECT_EQ(StderrWrites(), 1);
}

TEST_F(NetTelemetryInitMicrotest, Init_HistogramBucketsOutOfRange_KeepsDefault) {
  for (const char* value : {"0", "-1", "17"}) {
    SCOPED_TRACE(value);
    ResetTelemetryState();
    SetMicroEnv("RCCL_TELEMETRY_HISTOGRAM_BUCKETS", value);

    Enable();

    EXPECT_EQ(rcclTelemetryCfg.histogram_max_buckets, 5);
  }
}

TEST_F(NetTelemetryInitMicrotest, Init_HistogramBucketsAtLowerBound_IsApplied) {
  SetMicroEnv("RCCL_TELEMETRY_HISTOGRAM_BUCKETS", "1");

  Enable();

  EXPECT_EQ(rcclTelemetryCfg.histogram_max_buckets, 1);
}

TEST_F(NetTelemetryInitMicrotest, Init_IntervalNotPositive_KeepsDefault) {
  for (const char* value : {"0", "-5"}) {
    SCOPED_TRACE(value);
    ResetTelemetryState();
    SetMicroEnv("RCCL_TELEMETRY_HISTOGRAM_INTERVAL_NS", value);

    Enable();

    EXPECT_EQ(rcclTelemetryCfg.histogram_bucket_interval_ns, 30000);
  }
}

TEST_F(NetTelemetryInitMicrotest, Init_IntervalOfOne_IsApplied) {
  SetMicroEnv("RCCL_TELEMETRY_HISTOGRAM_INTERVAL_NS", "1");

  Enable();

  EXPECT_EQ(rcclTelemetryCfg.histogram_bucket_interval_ns, 1);
}

TEST_F(NetTelemetryInitMicrotest, Init_LatencySamplePowerOfTwo_UsedSilently) {
  for (int n : {1, 8, 1 << 24}) {
    SCOPED_TRACE(n);
    ResetTelemetryState();
    g_fprintfCalls.clear();
    SetMicroEnv("RCCL_TELEMETRY_LATENCY_SAMPLE", std::to_string(n).c_str());

    Enable();

    EXPECT_EQ(rcclTelemetryLatencySampleN, n);
    EXPECT_EQ(rcclTelemetryLatencySampleMask, static_cast<uint64_t>(n - 1));
    EXPECT_EQ(StderrWrites(), 0);
  }
}

TEST_F(NetTelemetryInitMicrotest, Init_LatencySampleNotPowerOfTwo_RoundsUpAndWarns) {
  SetMicroEnv("RCCL_TELEMETRY_LATENCY_SAMPLE", "5");

  Enable();

  EXPECT_EQ(rcclTelemetryLatencySampleN, 8);
  EXPECT_EQ(rcclTelemetryLatencySampleMask, 7u);
  EXPECT_EQ(StderrWrites(), 1);
}

TEST_F(NetTelemetryInitMicrotest, Init_LatencySampleAboveMax_ClampsAndWarns) {
  SetMicroEnv("RCCL_TELEMETRY_LATENCY_SAMPLE", std::to_string((1 << 24) + 1).c_str());

  Enable();

  EXPECT_EQ(rcclTelemetryLatencySampleN, 1 << 24);
  EXPECT_EQ(StderrWrites(), 1);
}

TEST_F(NetTelemetryInitMicrotest, Init_LatencySampleNotPositive_KeepsOneAndWarns) {
  for (const char* value : {"0", "-4", "abc"}) {
    SCOPED_TRACE(value);
    ResetTelemetryState();
    g_fprintfCalls.clear();
    SetMicroEnv("RCCL_TELEMETRY_LATENCY_SAMPLE", value);

    Enable();

    EXPECT_EQ(rcclTelemetryLatencySampleN, 1);
    EXPECT_EQ(rcclTelemetryLatencySampleMask, 0u);
    EXPECT_EQ(StderrWrites(), 1);
  }
}

TEST_F(NetTelemetryInitMicrotest, Init_HwCounterListAtLimit_IsKeptWhole) {
  std::string list(sizeof(rcclTelemetryCfg.hw_counter_list) - 1, 'c');
  SetMicroEnv("RCCL_TELEMETRY_HW_COUNTERS", list.c_str());

  Enable();

  EXPECT_EQ(rcclTelemetryCfg.hw_counter_list, list);
  EXPECT_EQ(StderrWrites(), 0);
}

TEST_F(NetTelemetryInitMicrotest, Init_HwCounterListTooLong_TruncatesAndWarns) {
  std::string list(sizeof(rcclTelemetryCfg.hw_counter_list), 'c');
  SetMicroEnv("RCCL_TELEMETRY_HW_COUNTERS", list.c_str());

  Enable();

  EXPECT_EQ(rcclTelemetryCfg.hw_counter_list, list.substr(1));
  EXPECT_EQ(StderrWrites(), 1);
}

TEST_F(NetTelemetryInitMicrotest, Init_ProcessCommReadable_RecordsNameWithoutNewline) {
  g_sys.files["/proc/" + std::to_string(getpid()) + "/comm"] = "trainer\n";

  Enable();

  EXPECT_STREQ(rcclTelemetryProcessName, "trainer");
}

TEST_F(NetTelemetryInitMicrotest, Init_ProcessCommWithoutNewline_KeepsWholeName) {
  g_sys.files["/proc/" + std::to_string(getpid()) + "/comm"] = "trainer";

  Enable();

  EXPECT_STREQ(rcclTelemetryProcessName, "trainer");
}

TEST_F(NetTelemetryInitMicrotest, Init_ProcessCommMissingOrEmpty_LeavesNameEmpty) {
  for (bool present : {false, true}) {
    SCOPED_TRACE(present);
    ResetTelemetryState();
    std::snprintf(rcclTelemetryProcessName, sizeof(rcclTelemetryProcessName), "stale");
    if (present) g_sys.files["/proc/" + std::to_string(getpid()) + "/comm"] = "";

    Enable();

    EXPECT_STREQ(rcclTelemetryProcessName, "");
  }
}

TEST_F(NetTelemetryInitMicrotest, Init_ExitHandlerUnavailable_StaysDisabledWithoutSampler) {
  SetMicroEnv("RCCL_TELEMETRY_ENABLE", "1");
  SetMicroEnv("RCCL_TELEMETRY_SAMPLE_MS", "5");
  ScopedHook atexitHook(g_atexit, [](void (*)(void)) { return -1; });

  EXPECT_EQ(rcclTelemetryInit(), -1);

  EXPECT_FALSE(rcclTelemetryOn());
  EXPECT_FALSE(rcclTelemetrySamplerRunning);
}

TEST_F(NetTelemetryInitMicrotest, Init_SampleMsPositive_StartsSampler) {
  SetMicroEnv("RCCL_TELEMETRY_SAMPLE_MS", "5");

  Enable();

  EXPECT_EQ(rcclTelemetrySampleIntervalMs, 5);
  EXPECT_TRUE(rcclTelemetrySamplerRunning);
}

TEST_F(NetTelemetryInitMicrotest, Init_SampleMsNotPositive_LeavesSamplerOff) {
  for (const char* value : {"0", "-3", ""}) {
    SCOPED_TRACE(value);
    ResetTelemetryState();
    SetMicroEnv("RCCL_TELEMETRY_SAMPLE_MS", value);

    Enable();

    EXPECT_EQ(rcclTelemetrySampleIntervalMs, 0);
    EXPECT_FALSE(rcclTelemetrySamplerRunning);
  }
}

TEST_F(NetTelemetryInitMicrotest, Init_SecondCall_ReturnsFirstResultWithoutRereadingEnv) {
  SetMicroEnv("RCCL_TELEMETRY_ENABLE", "1");
  SetMicroEnv("RCCL_TELEMETRY_OUTPUT_DIR", std::string(sizeof(rcclTelemetryCfg.output_dir), 'd').c_str());
  ASSERT_EQ(rcclTelemetryInit(), -1);
  SetMicroEnv("RCCL_TELEMETRY_OUTPUT_DIR", "/out");

  EXPECT_EQ(rcclTelemetryInit(), -1);

  EXPECT_FALSE(rcclTelemetryOn());
  EXPECT_TRUE(g_atexitFns.empty());
}

TEST_F(NetTelemetryInitMicrotest, GetTimestamp_FormatsLocalDateAndTime) {
  char buf[64];

  rcclTelemetryGetTimestamp(buf, sizeof(buf));

  EXPECT_TRUE(std::regex_match(buf, std::regex(R"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"))) << buf;
}

TEST_F(NetTelemetryInitMicrotest, ConfigDeriveHistogram_PositiveInterval_DerivesReciprocal) {
  for (int64_t d : {int64_t{1}, int64_t{3}, int64_t{30000}}) {
    SCOPED_TRACE(d);
    rcclTelemetryCfg.histogram_bucket_interval_ns = d;

    rcclTelemetryConfigDeriveHistogram();

    EXPECT_EQ(rcclTelemetryCfg.histogram_recip, (uint64_t{1} << 63) / d + 1);
    EXPECT_EQ(rcclTelemetryCfg.histogram_recip_max_ns, (uint64_t{1} << 63) / d);
  }
}

TEST_F(NetTelemetryInitMicrotest, ConfigDeriveHistogram_IntervalNotPositive_ClearsReciprocal) {
  for (int64_t d : {int64_t{0}, int64_t{-7}}) {
    SCOPED_TRACE(d);
    rcclTelemetryCfg.histogram_recip = 5;
    rcclTelemetryCfg.histogram_recip_max_ns = 5;
    rcclTelemetryCfg.histogram_bucket_interval_ns = d;

    rcclTelemetryConfigDeriveHistogram();

    EXPECT_EQ(rcclTelemetryCfg.histogram_recip, 0u);
    EXPECT_EQ(rcclTelemetryCfg.histogram_recip_max_ns, 0u);
  }
}

// ---------------------------------------------------------------------------

using NetTelemetryRegisterMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_Disabled_ReturnsMinusOne) {
  AddNic("rdma0", "eth0", "ionic");

  EXPECT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), -1);

  EXPECT_EQ(rcclTelemetryNumDevs, 0);
  EXPECT_STREQ(rcclTelemetryDevs[0].roce_device, "");
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_IdOutOfRange_ReturnsMinusOne) {
  Enable();

  EXPECT_EQ(rcclTelemetryRegisterDevice(-1, "rdma0", "IB"), -1);
  EXPECT_EQ(rcclTelemetryRegisterDevice(RCCL_TELEMETRY_MAX_DEVS, "rdma0", "IB"), -1);

  EXPECT_EQ(rcclTelemetryNumDevs, 0);
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_IdOutOfRangeTwice_WarnsOnce) {
  RUN_ISOLATED_TEST("NetTelemetry_RegisterDevice_IdOutOfRangeTwice_WarnsOnce", []() {
    Enable();

    rcclTelemetryRegisterDevice(-1, "rdma0", "IB");
    rcclTelemetryRegisterDevice(RCCL_TELEMETRY_MAX_DEVS, "rdma0", "IB");

    EXPECT_EQ(StderrWrites(), 1);
  });
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_ValidId_RecordsNamesAndResolvesHardware) {
  Enable();
  AddNic("rdma0", "eth4", "ionic");
  rcclTelemetryDevs[2].hw_counters[0] = 7;
  rcclTelemetryDevs[2].delta_tx_bytes = 7;

  EXPECT_EQ(rcclTelemetryRegisterDevice(2, "rdma0", "IB-CAST"), 2);

  const RcclDeviceStats& dev = rcclTelemetryDevs[2];
  EXPECT_EQ(dev.device_id, 2);
  EXPECT_STREQ(dev.roce_device, "rdma0");
  EXPECT_STREQ(dev.eth_device, "eth4");
  EXPECT_STREQ(dev.transport, "IB-CAST");
  EXPECT_EQ(dev.hw_config, &rcclHwConfigAinic);
  EXPECT_EQ(dev.hw_counters[0], -1);
  EXPECT_EQ(dev.delta_tx_bytes, -1);
  EXPECT_EQ(rcclTelemetryNumDevs, 3);
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_ReregisteredDevice_ResetsEveryBaselineToUnavailable) {
  Enable();
  RcclDeviceStats& dev = rcclTelemetryDevs[0];
  for (int c = 0; c < RCCL_TELEMETRY_MAX_HWC; c++) dev.hw_counters[c] = dev.snap_init_hw_counters[c] = 3;
  for (int p = 0; p < RCCL_TELEMETRY_NUM_PFC_PRIO; p++) {
    dev.pfc_rx_frames[p] = dev.snap_init_pfc_rx_frames[p] = 3;
    dev.pfc_tx_frames[p] = dev.snap_init_pfc_tx_frames[p] = 3;
    dev.pfc_rx_pause_us[p] = dev.snap_init_pfc_rx_pause_us[p] = 3;
    dev.pfc_tx_pause_us[p] = dev.snap_init_pfc_tx_pause_us[p] = 3;
  }
  dev.snap_init_tx_bytes = dev.snap_init_rx_bytes = dev.snap_init_tx_packets = dev.snap_init_rx_packets = 3;
  dev.delta_tx_bytes = dev.delta_rx_bytes = dev.delta_tx_packets = dev.delta_rx_packets = 3;

  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);

  for (int c = 0; c < RCCL_TELEMETRY_MAX_HWC; c++) {
    EXPECT_EQ(dev.hw_counters[c], -1);
    EXPECT_EQ(dev.snap_init_hw_counters[c], -1);
  }
  for (int p = 0; p < RCCL_TELEMETRY_NUM_PFC_PRIO; p++) {
    EXPECT_EQ(dev.pfc_rx_frames[p], -1);
    EXPECT_EQ(dev.snap_init_pfc_rx_frames[p], -1);
    EXPECT_EQ(dev.pfc_tx_frames[p], -1);
    EXPECT_EQ(dev.snap_init_pfc_tx_frames[p], -1);
    EXPECT_EQ(dev.pfc_rx_pause_us[p], -1);
    EXPECT_EQ(dev.snap_init_pfc_rx_pause_us[p], -1);
    EXPECT_EQ(dev.pfc_tx_pause_us[p], -1);
    EXPECT_EQ(dev.snap_init_pfc_tx_pause_us[p], -1);
  }
  EXPECT_EQ(dev.snap_init_tx_bytes, -1);
  EXPECT_EQ(dev.snap_init_rx_bytes, -1);
  EXPECT_EQ(dev.snap_init_tx_packets, -1);
  EXPECT_EQ(dev.snap_init_rx_packets, -1);
  EXPECT_EQ(dev.delta_tx_bytes, -1);
  EXPECT_EQ(dev.delta_rx_bytes, -1);
  EXPECT_EQ(dev.delta_tx_packets, -1);
  EXPECT_EQ(dev.delta_rx_packets, -1);
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_KnownDrivers_ResolveTheirCounterTables) {
  Enable();
  std::vector<std::pair<std::string, const RcclHwConfig*>> drivers = {{"ionic", &rcclHwConfigAinic},
                                                                       {"mlx5_core", &rcclHwConfigMlx5},
                                                                       {"bnxt_re", &rcclHwConfigThor2},
                                                                       {"bnxt_en", &rcclHwConfigThor2},
                                                                       {"e1000e", nullptr}};
  for (size_t i = 0; i < drivers.size(); i++) {
    SCOPED_TRACE(drivers[i].first);
    std::string roce = "rdma" + std::to_string(i);
    AddNic(roce, "eth0", drivers[i].first);

    ASSERT_EQ(rcclTelemetryRegisterDevice(static_cast<int>(i), roce.c_str(), "IB"), static_cast<int>(i));

    EXPECT_EQ(rcclTelemetryDevs[i].hw_config, drivers[i].second);
  }
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_DriverLinkMissingOrWithoutSlash_IsUnsupported) {
  Enable();
  g_sys.links[std::string(kIb) + "rdma1/device/driver"] = "ionic";

  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);

  EXPECT_EQ(rcclTelemetryDevs[0].hw_config, nullptr);
  EXPECT_EQ(rcclTelemetryDevs[1].hw_config, nullptr);
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_LowerIdAfterHigher_KeepsHighWaterMark) {
  Enable();
  ASSERT_EQ(rcclTelemetryRegisterDevice(5, "rdma5", "IB"), 5);

  ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);

  EXPECT_EQ(rcclTelemetryNumDevs, 6);
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_NullNames_KeepPreviousNames) {
  Enable();
  std::snprintf(rcclTelemetryDevs[0].transport, RCCL_TELEMETRY_TRANSPORT_MAX, "OLD");

  ASSERT_EQ(rcclTelemetryRegisterDevice(0, nullptr, nullptr), 0);

  EXPECT_STREQ(rcclTelemetryDevs[0].roce_device, "");
  EXPECT_STREQ(rcclTelemetryDevs[0].eth_device, "");
  EXPECT_STREQ(rcclTelemetryDevs[0].transport, "OLD");
  EXPECT_TRUE(g_sys.openDirs.empty());
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_LongNames_AreTruncatedToFit) {
  Enable();
  std::string roce(RCCL_TELEMETRY_DEV_NAME_MAX + 4, 'r');
  std::string eth(RCCL_TELEMETRY_DEV_NAME_MAX + 4, 'e');
  std::string transport(RCCL_TELEMETRY_TRANSPORT_MAX + 4, 't');
  g_sys.dirs[kIb + roce.substr(0, RCCL_TELEMETRY_DEV_NAME_MAX - 1) + "/device/net"] = {eth};

  ASSERT_EQ(rcclTelemetryRegisterDevice(0, roce.c_str(), transport.c_str()), 0);

  EXPECT_EQ(rcclTelemetryDevs[0].roce_device, roce.substr(0, RCCL_TELEMETRY_DEV_NAME_MAX - 1));
  EXPECT_EQ(rcclTelemetryDevs[0].eth_device, eth.substr(0, RCCL_TELEMETRY_DEV_NAME_MAX - 1));
  EXPECT_EQ(rcclTelemetryDevs[0].transport, transport.substr(0, RCCL_TELEMETRY_TRANSPORT_MAX - 1));
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_NoNetdevEntry_LeavesEthDeviceEmpty) {
  Enable();
  std::snprintf(rcclTelemetryDevs[1].eth_device, RCCL_TELEMETRY_DEV_NAME_MAX, "stale");
  g_sys.dirs[std::string(kIb) + "rdma0/device/net"] = {".", ".."};

  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);

  EXPECT_STREQ(rcclTelemetryDevs[0].eth_device, "");
  EXPECT_STREQ(rcclTelemetryDevs[1].eth_device, "");
}

TEST_F(NetTelemetryRegisterMicrotest, RegisterDevice_DebugEnv_LogsRegistration) {
  Enable();
  SetMicroEnv("RCCL_TELEMETRY_DEBUG", "1");

  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  ASSERT_EQ(rcclTelemetryRegisterDevice(1, nullptr, nullptr), 1);

  EXPECT_EQ(StderrWrites(), 2);
}

// ---------------------------------------------------------------------------

using NetTelemetryChannelMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_Disabled_ReservesNothing) {
  int slots = 7;

  EXPECT_EQ(rcclTelemetrySetupChannel(0, 0, 2, &slots), -1);

  EXPECT_EQ(slots, 0);
  EXPECT_EQ(rcclTelemetryDevs[0].channel_blocks, nullptr);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_InvalidDeviceOrQpCount_ReservesNothing) {
  Enable();
  for (auto args : {std::make_pair(-1, 2), std::make_pair(RCCL_TELEMETRY_MAX_DEVS, 2), std::make_pair(0, 0),
                    std::make_pair(0, -1)}) {
    SCOPED_TRACE(args.first);
    int slots = 7;

    EXPECT_EQ(rcclTelemetrySetupChannel(args.first, 0, args.second, &slots), -1);

    EXPECT_EQ(slots, 0);
  }
  EXPECT_EQ(rcclTelemetryDevs[0].channel_blocks, nullptr);
  EXPECT_EQ(rcclTelemetryDevs[0].snap_taken, 0);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_FirstCall_ReservesSlotsFromZero) {
  Enable();
  int slots = 0;

  EXPECT_EQ(rcclTelemetrySetupChannel(0, 3, 2, &slots), 0);

  EXPECT_EQ(slots, 2);
  RcclChannelStats* ch = rcclTelemetryChannel(0, 3);
  ASSERT_NE(ch, nullptr);
  EXPECT_EQ(ch->id, 3);
  EXPECT_EQ(ch->num_qps, 2);
  EXPECT_EQ(rcclTelemetryQp(ch, 0)->id, 0);
  EXPECT_EQ(rcclTelemetryQp(ch, 1)->id, 1);
  EXPECT_EQ(rcclTelemetryQp(ch, 2), nullptr);
  EXPECT_EQ(rcclTelemetryDevs[0].num_channels, 4);
  EXPECT_EQ(rcclTelemetryDevs[0].snap_taken, 1);
  EXPECT_EQ(rcclTelemetryDevs[0].num_qp_untracked, 0);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_SameChannelAgain_AppendsAfterExistingSlots) {
  Enable();
  int slots = 0;
  ASSERT_EQ(rcclTelemetrySetupChannel(0, 0, 3, &slots), 0);

  EXPECT_EQ(rcclTelemetrySetupChannel(0, 0, 20, &slots), 3);

  EXPECT_EQ(slots, 20);
  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);
  EXPECT_EQ(ch->num_qps, 23);
  for (int q = 0; q < 23; q++) EXPECT_EQ(rcclTelemetryQp(ch, q)->id, q) << q;
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_NullSlotCount_StillReserves) {
  Enable();

  EXPECT_EQ(rcclTelemetrySetupChannel(0, 0, 2, nullptr), 0);

  EXPECT_EQ(rcclTelemetryChannel(0, 0)->num_qps, 2);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_LowerChannelLater_KeepsChannelCount) {
  Enable();
  int slots = 0;
  ASSERT_EQ(rcclTelemetrySetupChannel(0, 5, 1, &slots), 0);

  ASSERT_EQ(rcclTelemetrySetupChannel(0, 2, 1, &slots), 0);

  EXPECT_EQ(rcclTelemetryDevs[0].num_channels, 6);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_ChannelOutOfRange_ChargesDeviceUntracked) {
  Enable();
  for (int chIdx : {-1, RCCL_TELEMETRY_MAX_CHANNELS}) {
    SCOPED_TRACE(chIdx);
    int slots = 7;

    EXPECT_EQ(rcclTelemetrySetupChannel(0, chIdx, 3, &slots), -1);

    EXPECT_EQ(slots, 0);
  }
  EXPECT_EQ(rcclTelemetryDevs[0].num_qp_untracked, 6);
  EXPECT_EQ(rcclTelemetryDevs[0].channel_blocks, nullptr);
  EXPECT_EQ(rcclTelemetryDevs[0].snap_taken, 1);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_LastValidChannel_IsTracked) {
  Enable();
  int slots = 0;

  EXPECT_EQ(rcclTelemetrySetupChannel(0, RCCL_TELEMETRY_MAX_CHANNELS - 1, 1, &slots), 0);

  EXPECT_EQ(slots, 1);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_ChannelTableAllocFails_ChargesDeviceUntracked) {
  Enable();
  ScopedHook callocHook(g_calloc, [](size_t, size_t) -> void* { return nullptr; });
  int slots = 7;

  EXPECT_EQ(rcclTelemetrySetupChannel(0, 0, 2, &slots), -1);

  EXPECT_EQ(slots, 0);
  EXPECT_EQ(rcclTelemetryDevs[0].num_qp_untracked, 2);
  EXPECT_EQ(rcclTelemetryDevs[0].num_channels, 0);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_ChannelBlockAllocFails_ChargesDeviceUntracked) {
  Enable();
  ScopedHook memalignHook(g_posixMemalign, [](void**, size_t, size_t) { return ENOMEM; });
  int slots = 7;

  EXPECT_EQ(rcclTelemetrySetupChannel(0, 0, 2, &slots), -1);

  EXPECT_EQ(rcclTelemetryDevs[0].num_qp_untracked, 2);
  EXPECT_EQ(rcclTelemetryDevs[0].channel_capacity, 0);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_QpTableAllocFails_ChargesChannelAndDeviceUntracked) {
  Enable();
  int calls = 0;
  ScopedHook callocHook(g_calloc, [&](size_t n, size_t size) { return calls++ == 0 ? RealCalloc(n, size) : nullptr; });
  int slots = 7;

  EXPECT_EQ(rcclTelemetrySetupChannel(0, 0, 2, &slots), -1);

  EXPECT_EQ(slots, 0);
  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);
  ASSERT_NE(ch, nullptr);
  EXPECT_EQ(ch->num_qps, 0);
  EXPECT_EQ(ch->num_qp_untracked, 2);
  EXPECT_EQ(rcclTelemetryDevs[0].num_qp_untracked, 2);
  EXPECT_EQ(rcclTelemetryDevs[0].num_channels, 1);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_QpBlockAllocFailsMidway_GrantsWhatFits) {
  Enable();
  int slots = 0;
  ASSERT_EQ(rcclTelemetrySetupChannel(0, 0, 3, &slots), 0);
  ScopedHook memalignHook(g_posixMemalign, [](void**, size_t, size_t) { return ENOMEM; });

  EXPECT_EQ(rcclTelemetrySetupChannel(0, 0, 10, &slots), 3);

  EXPECT_EQ(slots, 5);
  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);
  EXPECT_EQ(ch->num_qps, 8);
  EXPECT_EQ(rcclTelemetryQp(ch, 7)->id, 7);
  EXPECT_EQ(ch->num_qp_untracked, 5);
  EXPECT_EQ(rcclTelemetryDevs[0].num_qp_untracked, 5);
}

TEST_F(NetTelemetryChannelMicrotest, SetupChannel_FullQpBlock_GrantsNothing) {
  Enable();
  int slots = 0;
  ASSERT_EQ(rcclTelemetrySetupChannel(0, 0, 8, &slots), 0);
  ScopedHook memalignHook(g_posixMemalign, [](void**, size_t, size_t) { return ENOMEM; });

  EXPECT_EQ(rcclTelemetrySetupChannel(0, 0, 2, &slots), -1);

  EXPECT_EQ(slots, 0);
  EXPECT_EQ(rcclTelemetryChannel(0, 0)->num_qps, 8);
  EXPECT_EQ(rcclTelemetryChannel(0, 0)->num_qp_untracked, 2);
}

TEST_F(NetTelemetryChannelMicrotest, BlocksGrow_FirstRequest_SizesBlockZeroToTheRequest) {
  for (auto c : {std::make_pair(1, 3), std::make_pair(8, 3), std::make_pair(9, 4), std::make_pair(20, 5)}) {
    SCOPED_TRACE(c.first);
    void* blocks[RCCL_TELEMETRY_BLOCKS] = {};
    int log2 = 0, capacity = 0;

    EXPECT_EQ(rcclTelemetryBlocksGrow(blocks, &log2, &capacity, c.first, sizeof(RcclQpStats)), 1 << c.second);

    EXPECT_EQ(log2, c.second);
    EXPECT_EQ(capacity, 1 << c.second);
    EXPECT_NE(blocks[0], nullptr);
    EXPECT_EQ(blocks[1], nullptr);
    free(blocks[0]);
  }
}

TEST_F(NetTelemetryChannelMicrotest, BlocksGrow_LargerRequest_AppendsDoublingBlocks) {
  void* blocks[RCCL_TELEMETRY_BLOCKS] = {};
  int log2 = 0, capacity = 0;
  ASSERT_EQ(rcclTelemetryBlocksGrow(blocks, &log2, &capacity, 8, sizeof(RcclQpStats)), 8);

  EXPECT_EQ(rcclTelemetryBlocksGrow(blocks, &log2, &capacity, 30, sizeof(RcclQpStats)), 56);
  EXPECT_EQ(rcclTelemetryBlocksGrow(blocks, &log2, &capacity, 56, sizeof(RcclQpStats)), 56);

  EXPECT_EQ(log2, 3);
  EXPECT_EQ(capacity, 56);
  EXPECT_NE(blocks[1], nullptr);
  EXPECT_NE(blocks[2], nullptr);
  EXPECT_EQ(blocks[3], nullptr);
  for (void* block : blocks) free(block);
}

TEST_F(NetTelemetryChannelMicrotest, AlignedCalloc_Allocates_ZeroedCacheLineAlignedBlock) {
  ScopedHook memalignHook(g_posixMemalign, [](void** p, size_t alignment, size_t size) {
    EXPECT_EQ(alignment, static_cast<size_t>(RCCL_TELEMETRY_CACHELINE));
    int ret = DefaultPosixMemalign(p, alignment, size);
    std::memset(*p, 0xa5, size);
    return ret;
  });

  auto* block = static_cast<unsigned char*>(rcclTelemetryAlignedCalloc(3, 64));

  ASSERT_NE(block, nullptr);
  EXPECT_TRUE(std::all_of(block, block + 192, [](unsigned char b) { return b == 0; }));
  free(block);
}

TEST_F(NetTelemetryChannelMicrotest, AlignedCalloc_AllocFails_ReturnsNull) {
  ScopedHook memalignHook(g_posixMemalign, [](void** p, size_t, size_t) {
    *p = nullptr;
    return 0;
  });

  EXPECT_EQ(rcclTelemetryAlignedCalloc(1, 64), nullptr);
}

// ---------------------------------------------------------------------------

using NetTelemetryHooksMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryHooksMicrotest, GetNs_ReadsMonotonicClockInNanoseconds) {
  g_nowNs = 12345678901234;

  EXPECT_EQ(rcclTelemetryGetNs(), 12345678901234);
}

TEST_F(NetTelemetryHooksMicrotest, BlockIndex_MapsIndexToDoublingBlocks) {
  struct Case {
    int idx, log2;
    unsigned block, offset;
  };
  for (Case c : {Case{0, 3, 0, 0}, Case{7, 3, 0, 7}, Case{8, 3, 1, 0}, Case{23, 3, 1, 15}, Case{24, 3, 2, 0},
                 Case{55, 3, 2, 31}, Case{16, 4, 1, 0}}) {
    SCOPED_TRACE(c.idx);
    unsigned block = 99, offset = 99;

    rcclTelemetryBlockIndex(c.idx, c.log2, &block, &offset);

    EXPECT_EQ(block, c.block);
    EXPECT_EQ(offset, c.offset);
  }
}

TEST_F(NetTelemetryHooksMicrotest, Channel_DisabledOrOutOfRange_ReturnsNull) {
  EXPECT_EQ(rcclTelemetryChannel(0, 0), nullptr);
  Enable();
  int slots = 0;
  ASSERT_EQ(rcclTelemetrySetupChannel(0, 0, 1, &slots), 0);

  EXPECT_EQ(rcclTelemetryChannel(-1, 0), nullptr);
  EXPECT_EQ(rcclTelemetryChannel(RCCL_TELEMETRY_MAX_DEVS, 0), nullptr);
  EXPECT_EQ(rcclTelemetryChannel(0, -1), nullptr);
  EXPECT_EQ(rcclTelemetryChannel(0, rcclTelemetryDevs[0].channel_capacity), nullptr);
  EXPECT_NE(rcclTelemetryChannel(0, rcclTelemetryDevs[0].channel_capacity - 1), nullptr);
}

TEST_F(NetTelemetryHooksMicrotest, QpSlot_BoundedByCapacityWhileQpBoundedBySlotsHandedOut) {
  Enable();
  int slots = 0;
  ASSERT_EQ(rcclTelemetrySetupChannel(0, 0, 2, &slots), 0);
  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);

  EXPECT_NE(rcclTelemetryQpSlot(ch, 7), nullptr);
  EXPECT_EQ(rcclTelemetryQpSlot(ch, 8), nullptr);
  EXPECT_EQ(rcclTelemetryQpSlot(ch, -1), nullptr);
  EXPECT_EQ(rcclTelemetryQp(ch, 2), nullptr);
  EXPECT_EQ(rcclTelemetryQp(ch, -1), nullptr);
  EXPECT_EQ(rcclTelemetryQp(ch, 1), rcclTelemetryQpSlot(ch, 1));
}

TEST_F(NetTelemetryHooksMicrotest, Resolve_ValidIndices_ReturnSlots) {
  Enable();
  RcclQpStats* qp = SetupQps(0, 1, 2);

  EXPECT_EQ(rcclTelemetryResolveQp(0, 1, 0), qp);
  EXPECT_EQ(rcclTelemetryResolveChannel(0, 1), rcclTelemetryChannel(0, 1));
  EXPECT_EQ(rcclTelemetryResolveQp(0, 1, 2), nullptr);
  EXPECT_EQ(rcclTelemetryResolveQp(0, 5, 0), nullptr);
}

TEST_F(NetTelemetryHooksMicrotest, PostTs_SampleEverything_ReturnsClock) {
  EXPECT_EQ(rcclTelemetryPostTs(nullptr), g_nowNs);
}

TEST_F(NetTelemetryHooksMicrotest, PostTs_OneInFour_SamplesEveryFourthPosting) {
  Enable();
  RcclQpStats* qp = SetupQps(0, 0, 1);
  rcclTelemetryLatencySampleMask = 3;

  std::vector<int64_t> ts;
  for (int i = 0; i < 8; i++) ts.push_back(rcclTelemetryPostTs(qp));

  int64_t u = RCCL_TELEMETRY_TS_UNSAMPLED;
  EXPECT_EQ(ts, (std::vector<int64_t>{u, u, u, g_nowNs, u, u, u, g_nowNs}));
  EXPECT_EQ(qp->latency_sample_seq, 8u);
  EXPECT_EQ(rcclTelemetryPostTs(nullptr), u);
}

TEST_F(NetTelemetryHooksMicrotest, LatencyBucket_DividesByIntervalAndClampsToLastBucket) {
  rcclTelemetryCfg.histogram_max_buckets = 5;
  rcclTelemetryCfg.histogram_bucket_interval_ns = 30000;
  for (bool derived : {false, true}) {
    SCOPED_TRACE(derived);
    if (derived) rcclTelemetryConfigDeriveHistogram();
    for (auto c : {std::make_pair(int64_t{0}, 0), std::make_pair(int64_t{29999}, 0), std::make_pair(int64_t{30000}, 1),
                   std::make_pair(int64_t{149999}, 4), std::make_pair(int64_t{150000}, 4),
                   std::make_pair(int64_t{1} << 62, 4)}) {
      EXPECT_EQ(rcclTelemetryLatencyBucket(c.first), c.second) << c.first;
    }
  }
}

class NetTelemetryQpHooksMicrotest : public NetTelemetryMicrotest {
 protected:
  RcclQpStats* qp_ = nullptr;

  void SetUp() override {
    NetTelemetryMicrotest::SetUp();
    Enable();
    qp_ = SetupQps(0, 0, 1);
  }
};

TEST_F(NetTelemetryQpHooksMicrotest, QpWqeComplete_UnmatchedCompletion_CountsReceivedOnly) {
  rcclTelemetryQpWqeComplete(qp_, 0);

  EXPECT_EQ(qp_->num_wqe_rcvd, 1u);
  EXPECT_EQ(qp_->num_wqe_completed, 0u);
}

TEST_F(NetTelemetryQpHooksMicrotest, QpWqeComplete_UnsampledCompletion_CountsWithoutLatency) {
  rcclTelemetryQpWqeComplete(qp_, RCCL_TELEMETRY_TS_UNSAMPLED);

  EXPECT_EQ(qp_->num_wqe_rcvd, 1u);
  EXPECT_EQ(qp_->num_wqe_completed, 1u);
  EXPECT_EQ(rcclTelemetryQpSampledCount(qp_), 0u);
  EXPECT_EQ(qp_->wqe_completion_ns_min, 0);
}

TEST_F(NetTelemetryQpHooksMicrotest, QpWqeComplete_NonPositiveLatency_CountsWithoutLatency) {
  rcclTelemetryQpWqeComplete(qp_, g_nowNs);
  rcclTelemetryQpWqeComplete(qp_, g_nowNs + 5);

  EXPECT_EQ(qp_->num_wqe_completed, 2u);
  EXPECT_EQ(rcclTelemetryQpSampledCount(qp_), 0u);
  EXPECT_EQ(qp_->wqe_completion_ns_max, 0);
}

TEST_F(NetTelemetryQpHooksMicrotest, QpWqeComplete_SampledCompletions_TrackHistogramMinAndMax) {
  rcclTelemetryQpWqeComplete(qp_, g_nowNs - 40000);
  rcclTelemetryQpWqeComplete(qp_, g_nowNs - 20000);
  rcclTelemetryQpWqeComplete(qp_, g_nowNs - 90000);
  rcclTelemetryQpWqeComplete(qp_, g_nowNs - 1);

  EXPECT_EQ(qp_->num_wqe_completed, 4u);
  EXPECT_EQ(qp_->wqe_completion_ns_min, 1);
  EXPECT_EQ(qp_->wqe_completion_ns_max, 90000);
  EXPECT_EQ(qp_->wqe_completion_histogram[0], 2u);
  EXPECT_EQ(qp_->wqe_completion_histogram[1], 1u);
  EXPECT_EQ(qp_->wqe_completion_histogram[3], 1u);
}

TEST_F(NetTelemetryQpHooksMicrotest, QpWqeComplete_BucketPastHistogram_IsNotCounted) {
  rcclTelemetryCfg.histogram_max_buckets = RCCL_TELEMETRY_HISTOGRAM_SIZE + 4;

  rcclTelemetryQpWqeComplete(qp_, g_nowNs - int64_t{30000} * (RCCL_TELEMETRY_HISTOGRAM_SIZE + 1));

  EXPECT_EQ(rcclTelemetryQpSampledCount(qp_), 0u);
  EXPECT_EQ(qp_->wqe_completion_ns_max, int64_t{30000} * (RCCL_TELEMETRY_HISTOGRAM_SIZE + 1));
}

TEST_F(NetTelemetryQpHooksMicrotest, QpWqeComplete_LastHistogramBucket_IsCounted) {
  rcclTelemetryCfg.histogram_max_buckets = RCCL_TELEMETRY_HISTOGRAM_SIZE;

  rcclTelemetryQpWqeComplete(qp_, g_nowNs - int64_t{30000} * (RCCL_TELEMETRY_HISTOGRAM_SIZE - 1));

  EXPECT_EQ(qp_->wqe_completion_histogram[RCCL_TELEMETRY_HISTOGRAM_SIZE - 1], 1u);
}

TEST_F(NetTelemetryQpHooksMicrotest, QpHooks_NullHandle_AreNoOps) {
  rcclTelemetryQpWqeComplete(nullptr, g_nowNs - 1);
  rcclTelemetryQpSendPosted(nullptr, 1);
  rcclTelemetryQpRecvPosted(nullptr);
  rcclTelemetryQpCtsSent(nullptr, 1);
  rcclTelemetryQpSlotMiss(nullptr);
  rcclTelemetryChRequestCompleted(nullptr);

  EXPECT_EQ(rcclTelemetryQpSampledCount(nullptr), 0u);
}

TEST_F(NetTelemetryQpHooksMicrotest, QpSendPosted_CountsPostingAndOpcode) {
  rcclTelemetryQpSendPosted(qp_, 0);
  rcclTelemetryQpSendPosted(qp_, 1);
  rcclTelemetryQpSendPosted(qp_, 1);

  EXPECT_EQ(qp_->num_wqe_sent, 3u);
  EXPECT_EQ(qp_->num_write_wqe, 1u);
  EXPECT_EQ(qp_->num_write_imm_wqe, 2u);
}

TEST_F(NetTelemetryQpHooksMicrotest, QpRecvPostedSlotMissAndCts_CountOnTheirCounters) {
  rcclTelemetryQpRecvPosted(qp_);
  rcclTelemetryQpSlotMiss(qp_);
  rcclTelemetryQpCtsSent(qp_, 1);
  rcclTelemetryQpCtsSent(qp_, 0);
  rcclTelemetryQpCtsSent(qp_, 0);

  EXPECT_EQ(qp_->num_recv_wqe, 1u);
  EXPECT_EQ(qp_->num_slot_miss, 1u);
  EXPECT_EQ(qp_->num_cts_sent, 3u);
  EXPECT_EQ(qp_->num_cts_sent_signalled, 1u);
  EXPECT_EQ(qp_->num_cts_sent_unsignalled, 2u);
}

TEST_F(NetTelemetryQpHooksMicrotest, WqePosted_IsSend_CountsOnlyItsOwnCounter) {
  rcclTelemetryWqePosted(0, 0, 0, 1);
  rcclTelemetryWqePosted(0, 0, 0, 1);

  EXPECT_EQ(qp_->num_wqe_sent, 2u);
  EXPECT_EQ(qp_->num_recv_wqe, 0u);
}

TEST_F(NetTelemetryQpHooksMicrotest, IndexHooks_ResolveTheSlotAndCount) {
  rcclTelemetryWqeComplete(0, 0, 0, g_nowNs - 40000);
  rcclTelemetryWqePosted(0, 0, 0, 1);
  rcclTelemetryWqePosted(0, 0, 0, 0);
  rcclTelemetryRequestCompleted(0, 0);
  rcclTelemetrySlotMiss(0, 0, 0);
  rcclTelemetryCtsSent(0, 0, 0, 1);
  rcclTelemetryWriteWqe(0, 0, 0, 0);
  rcclTelemetryWriteWqe(0, 0, 0, 1);

  EXPECT_EQ(qp_->num_wqe_completed, 1u);
  EXPECT_EQ(qp_->wqe_completion_histogram[1], 1u);
  EXPECT_EQ(qp_->num_wqe_sent, 1u);
  EXPECT_EQ(qp_->num_recv_wqe, 1u);
  EXPECT_EQ(rcclTelemetryChannel(0, 0)->num_req_completed, 1u);
  EXPECT_EQ(qp_->num_slot_miss, 1u);
  EXPECT_EQ(qp_->num_cts_sent_signalled, 1u);
  EXPECT_EQ(qp_->num_write_wqe, 1u);
  EXPECT_EQ(qp_->num_write_imm_wqe, 1u);
}

TEST_F(NetTelemetryQpHooksMicrotest, IndexHooks_UnknownSlot_AreNoOps) {
  rcclTelemetryWqePosted(0, 0, 1, 1);
  rcclTelemetryWriteWqe(0, 0, 1, 1);
  rcclTelemetryWqePosted(0, 9, 0, 1);

  EXPECT_EQ(qp_->num_wqe_sent, 0u);
  EXPECT_EQ(qp_->num_write_imm_wqe, 0u);
}

TEST_F(NetTelemetryQpHooksMicrotest, SetQpRole_CountsDataAndCtsQps) {
  ASSERT_EQ(rcclTelemetrySetupChannel(0, 0, 2, nullptr), 1);

  rcclTelemetrySetQpRole(0, 0, 0, 5);
  rcclTelemetrySetQpRole(0, 0, 1, 0);
  rcclTelemetrySetQpRole(0, 0, 2, 0);
  rcclTelemetrySetQpRole(0, 0, 3, 1);
  rcclTelemetrySetQpRole(RCCL_TELEMETRY_MAX_DEVS, 0, 0, 1);

  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);
  EXPECT_EQ(rcclTelemetryQp(ch, 0)->is_data_qp, 1);
  EXPECT_EQ(rcclTelemetryQp(ch, 1)->is_data_qp, 0);
  EXPECT_EQ(ch->num_data_qp, 1);
  EXPECT_EQ(ch->num_cts_qp, 2);
}

using NetTelemetryDeviceHooksMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryDeviceHooksMicrotest, DeviceHooks_Disabled_AreNoOps) {
  rcclTelemetryBytes(0, 1, 5);
  rcclTelemetryCqError(0);
  rcclTelemetryAddCqPolls(0, 5);
  rcclTelemetryWqeSize(0, 5);
  rcclTelemetryMarkChannelsUnknown(0);

  const RcclDeviceStats& dev = rcclTelemetryDevs[0];
  EXPECT_EQ(dev.tx_bytes, 0u);
  EXPECT_EQ(dev.num_cq_errors, 0u);
  EXPECT_EQ(dev.cq_poll_count, 0u);
  EXPECT_EQ(dev.wqe_size_histogram[3], 0u);
  EXPECT_EQ(dev.channels_unknown, 0);
}

TEST_F(NetTelemetryDeviceHooksMicrotest, DeviceHooks_DeviceOutOfRange_AreNoOps) {
  Enable();
  for (int devIdx : {-1, RCCL_TELEMETRY_MAX_DEVS}) {
    rcclTelemetryBytes(devIdx, 1, 5);
    rcclTelemetryCqError(devIdx);
    rcclTelemetryAddCqPolls(devIdx, 5);
    rcclTelemetryWqeSize(devIdx, 5);
    rcclTelemetryMarkChannelsUnknown(devIdx);
  }

  RcclTelemetrySwSnapshot snap[RCCL_TELEMETRY_MAX_DEVS];
  rcclTelemetryNumDevs = RCCL_TELEMETRY_MAX_DEVS;
  ASSERT_EQ(rcclTelemetrySwCapture(snap, RCCL_TELEMETRY_MAX_DEVS), RCCL_TELEMETRY_MAX_DEVS);
  for (const RcclTelemetrySwSnapshot& s : snap) {
    EXPECT_EQ(s.tx_bytes, 0u);
    EXPECT_EQ(s.num_cq_errors, 0u);
  }
}

TEST_F(NetTelemetryDeviceHooksMicrotest, DeviceHooks_LastDevice_Count) {
  Enable();
  const int last = RCCL_TELEMETRY_MAX_DEVS - 1;

  rcclTelemetryBytes(last, 1, 5);
  rcclTelemetryBytes(last, 0, 7);
  rcclTelemetryCqError(last);
  rcclTelemetryAddCqPolls(last, 9);
  rcclTelemetryAddCqPolls(last, 0);
  rcclTelemetryMarkChannelsUnknown(last);

  const RcclDeviceStats& dev = rcclTelemetryDevs[last];
  EXPECT_EQ(dev.tx_bytes, 5u);
  EXPECT_EQ(dev.rx_bytes, 7u);
  EXPECT_EQ(dev.num_cq_errors, 1u);
  EXPECT_EQ(dev.cq_poll_count, 9u);
  EXPECT_EQ(dev.channels_unknown, 1);
}

TEST_F(NetTelemetryDeviceHooksMicrotest, WqeSize_BucketsByPowerOfTwo) {
  Enable();
  for (uint64_t bytes : {uint64_t{0}, uint64_t{1}, uint64_t{2}, uint64_t{3}, uint64_t{4}, uint64_t{1} << 30,
                         uint64_t{1} << 31, UINT64_MAX}) {
    rcclTelemetryWqeSize(0, bytes);
  }

  const uint64_t* h = rcclTelemetryDevs[0].wqe_size_histogram;
  EXPECT_EQ(h[0], 1u);
  EXPECT_EQ(h[1], 1u);
  EXPECT_EQ(h[2], 2u);
  EXPECT_EQ(h[3], 1u);
  EXPECT_EQ(h[30], 0u);
  EXPECT_EQ(h[31], 3u);
}

TEST_F(NetTelemetryDeviceHooksMicrotest, ChannelAggregate_SumsEveryQp) {
  Enable();
  RcclQpStats* a = SetupQps(0, 0, 2);
  RcclQpStats* b = rcclTelemetryResolveQp(0, 0, 1);
  rcclTelemetryQpSendPosted(a, 0);
  rcclTelemetryQpSendPosted(b, 0);
  rcclTelemetryQpRecvPosted(b);
  rcclTelemetryQpWqeComplete(a, 0);
  rcclTelemetryQpWqeComplete(b, g_nowNs - 10);
  rcclTelemetryQpWqeComplete(b, RCCL_TELEMETRY_TS_UNSAMPLED);
  rcclTelemetryQpCtsSent(a, 1);

  RcclChannelAggregate agg;
  rcclTelemetryChannelAggregate(rcclTelemetryChannel(0, 0), &agg);

  EXPECT_EQ(agg.num_wqe_sent, 2u);
  EXPECT_EQ(agg.num_recv_wqe, 1u);
  EXPECT_EQ(agg.num_wqe_rcvd, 3u);
  EXPECT_EQ(agg.num_wqe_completed, 2u);
  EXPECT_EQ(agg.num_wqe_sampled, 1u);
  EXPECT_EQ(agg.num_cts_sent, 1u);
}

TEST_F(NetTelemetryDeviceHooksMicrotest, ChannelAggregate_NullChannel_IsZero) {
  RcclChannelAggregate agg;
  std::memset(&agg, 0xff, sizeof(agg));

  rcclTelemetryChannelAggregate(nullptr, &agg);

  EXPECT_EQ(agg.num_wqe_sent, 0u);
  EXPECT_EQ(agg.num_recv_wqe, 0u);
  EXPECT_EQ(agg.num_wqe_rcvd, 0u);
  EXPECT_EQ(agg.num_wqe_completed, 0u);
  EXPECT_EQ(agg.num_wqe_sampled, 0u);
  EXPECT_EQ(agg.num_cts_sent, 0u);
}

// ---------------------------------------------------------------------------

using NetTelemetryCaptureMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryCaptureMicrotest, SwCapture_DisabledOrNoRoom_CapturesNothing) {
  RcclTelemetrySwSnapshot snap[2];
  rcclTelemetryNumDevs = 1;
  EXPECT_EQ(rcclTelemetrySwCapture(snap, 2), 0);

  Enable();
  rcclTelemetryNumDevs = 1;
  EXPECT_EQ(rcclTelemetrySwCapture(nullptr, 2), 0);
  EXPECT_EQ(rcclTelemetrySwCapture(snap, 0), 0);
}

TEST_F(NetTelemetryCaptureMicrotest, SwCapture_FewerSlotsThanDevices_FillsOnlyThoseSlots) {
  Enable();
  ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);
  rcclTelemetryBytes(0, 1, 4);
  RcclTelemetrySwSnapshot snap[2];
  std::memset(snap, 0xff, sizeof(snap));

  EXPECT_EQ(rcclTelemetrySwCapture(snap, 1), 1);

  EXPECT_EQ(snap[0].tx_bytes, 4u);
  EXPECT_EQ(snap[1].tx_bytes, UINT64_MAX);
}

TEST_F(NetTelemetryCaptureMicrotest, SwCapture_SumsChannelsAndQpsPerDevice) {
  Enable();
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);
  RcclQpStats* a = SetupQps(1, 0, 1);
  RcclQpStats* b = SetupQps(1, 2, 2);
  RcclQpStats* c = rcclTelemetryResolveQp(1, 2, 1);
  rcclTelemetryBytes(1, 1, 100);
  rcclTelemetryBytes(1, 0, 50);
  rcclTelemetryCqError(1);
  rcclTelemetryQpSendPosted(a, 0);
  rcclTelemetryQpRecvPosted(b);
  rcclTelemetryQpWqeComplete(a, 0);
  rcclTelemetryQpWqeComplete(b, g_nowNs - 500);
  rcclTelemetryQpWqeComplete(c, g_nowNs - 300);
  rcclTelemetryQpWqeComplete(c, g_nowNs - 70000);
  RcclTelemetrySwSnapshot snap[RCCL_TELEMETRY_MAX_DEVS];

  ASSERT_EQ(rcclTelemetrySwCapture(snap, RCCL_TELEMETRY_MAX_DEVS), 2);

  const RcclTelemetrySwSnapshot& s = snap[1];
  EXPECT_EQ(s.device_id, 1);
  EXPECT_EQ(s.tx_bytes, 100u);
  EXPECT_EQ(s.rx_bytes, 50u);
  EXPECT_EQ(s.num_cq_errors, 1u);
  EXPECT_EQ(s.wqe_sent, 1u);
  EXPECT_EQ(s.recv_wqe, 1u);
  EXPECT_EQ(s.wqe_rcvd, 4u);
  EXPECT_EQ(s.wqe_completed, 3u);
  EXPECT_EQ(s.wqe_sampled, 3u);
  EXPECT_EQ(s.wqe_completion_ns_min, 300);
  EXPECT_EQ(s.wqe_completion_ns_max, 70000);
  EXPECT_EQ(s.wqe_completion_histogram[0], 2u);
  EXPECT_EQ(s.wqe_completion_histogram[2], 1u);
  EXPECT_EQ(snap[0].wqe_sent, 0u);
  EXPECT_EQ(snap[0].wqe_completion_ns_min, 0);
}

TEST_F(NetTelemetryCaptureMicrotest, SwCapture_MinComesFromAnyQpWithSamples) {
  Enable();
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  RcclQpStats* a = SetupQps(0, 0, 3);
  rcclTelemetryQpWqeComplete(a, g_nowNs - 300);
  rcclTelemetryQpWqeComplete(rcclTelemetryResolveQp(0, 0, 2), g_nowNs - 200);
  RcclTelemetrySwSnapshot snap[1];

  ASSERT_EQ(rcclTelemetrySwCapture(snap, 1), 1);

  EXPECT_EQ(snap[0].wqe_completion_ns_min, 200);
  EXPECT_EQ(snap[0].wqe_completion_ns_max, 300);
}

// ---------------------------------------------------------------------------

using NetTelemetryCountersMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryCountersMicrotest, ReadSysfsCounter_ParsesDecimalValue) {
  g_sys.files["/c"] = "12345\n";

  EXPECT_EQ(rcclTelemetryReadSysfsCounter("/c"), 12345);

  EXPECT_EQ(g_closedFds.size(), 1u);
}

TEST_F(NetTelemetryCountersMicrotest, ReadSysfsCounter_MissingOrEmpty_ReturnsMinusOne) {
  g_sys.files["/empty"] = "";

  EXPECT_EQ(rcclTelemetryReadSysfsCounter("/missing"), -1);
  EXPECT_EQ(rcclTelemetryReadSysfsCounter("/empty"), -1);
}

TEST_F(NetTelemetryCountersMicrotest, ReadSysfsCounter_LongContent_ReadsAtMost63Bytes) {
  g_sys.files["/long"] = std::string(63, '1') + "9";

  EXPECT_EQ(rcclTelemetryReadSysfsCounter("/long"), LLONG_MAX);
  EXPECT_EQ(g_sys.fds.size(), 0u);
}

TEST_F(NetTelemetryCountersMicrotest, ReadHwCounter_TriesDeviceThenPortOneThenPortTwo) {
  g_sys.files[std::string(kIb) + "a/hw_counters/x"] = "1";
  g_sys.files[std::string(kIb) + "a/ports/1/hw_counters/x"] = "2";
  g_sys.files[std::string(kIb) + "b/ports/1/hw_counters/x"] = "3";
  g_sys.files[std::string(kIb) + "b/ports/2/hw_counters/x"] = "4";
  g_sys.files[std::string(kIb) + "c/ports/2/hw_counters/x"] = "5";

  EXPECT_EQ(rcclTelemetryReadHwCounter("a", "x"), 1);
  EXPECT_EQ(rcclTelemetryReadHwCounter("b", "x"), 3);
  EXPECT_EQ(rcclTelemetryReadHwCounter("c", "x"), 5);
  EXPECT_EQ(rcclTelemetryReadHwCounter("d", "x"), -1);
}

TEST_F(NetTelemetryCountersMicrotest, ReadHwCounter_ZeroValue_IsValid) {
  g_sys.files[std::string(kIb) + "a/hw_counters/x"] = "0";
  g_sys.files[std::string(kIb) + "a/ports/1/hw_counters/x"] = "9";

  EXPECT_EQ(rcclTelemetryReadHwCounter("a", "x"), 0);
}

TEST_F(NetTelemetryCountersMicrotest, IsCounterEnabled_EmptyList_EnablesEverything) {
  EXPECT_TRUE(rcclTelemetryIsCounterEnabled("anything"));
}

TEST_F(NetTelemetryCountersMicrotest, IsCounterEnabled_List_MatchesWholeNamesOnly) {
  std::snprintf(rcclTelemetryCfg.hw_counter_list, sizeof(rcclTelemetryCfg.hw_counter_list),
                "cnp_sent, ecn ,rx_errors");

  EXPECT_TRUE(rcclTelemetryIsCounterEnabled("cnp_sent"));
  EXPECT_TRUE(rcclTelemetryIsCounterEnabled("ecn"));
  EXPECT_TRUE(rcclTelemetryIsCounterEnabled("rx_errors"));
  EXPECT_FALSE(rcclTelemetryIsCounterEnabled("cnp"));
  EXPECT_FALSE(rcclTelemetryIsCounterEnabled("cnp_sent_x"));
  EXPECT_FALSE(rcclTelemetryIsCounterEnabled("errors"));
  EXPECT_FALSE(rcclTelemetryIsCounterEnabled("sent"));
}

TEST_F(NetTelemetryCountersMicrotest, GetDriverName_ResolvesLastPathComponent) {
  g_sys.links[std::string(kIb) + "a/device/driver"] = "../../bus/pci/drivers/mlx5_core";
  char name[RCCL_TELEMETRY_DEV_NAME_MAX];

  rcclTelemetryGetDriverName("a", name, sizeof(name));

  EXPECT_STREQ(name, "mlx5_core");
}

TEST_F(NetTelemetryCountersMicrotest, GetDriverName_LongName_IsTruncated) {
  g_sys.links[std::string(kIb) + "a/device/driver"] = "/drivers/abcdefgh";
  char name[5];

  rcclTelemetryGetDriverName("a", name, sizeof(name));

  EXPECT_STREQ(name, "abcd");
}

TEST_F(NetTelemetryCountersMicrotest, GetDriverName_NoLink_IsEmpty) {
  char name[8] = "stale";

  rcclTelemetryGetDriverName("a", name, sizeof(name));

  EXPECT_STREQ(name, "");
}

TEST_F(NetTelemetryCountersMicrotest, ResolveHw_UnknownOrEmptyDriver_IsUnsupported) {
  EXPECT_EQ(rcclTelemetryResolveHw(nullptr), nullptr);
  EXPECT_EQ(rcclTelemetryResolveHw(""), nullptr);
  EXPECT_EQ(rcclTelemetryResolveHw("ionic2"), nullptr);
}

class NetTelemetryEthtoolMicrotest : public NetTelemetryMicrotest {
 protected:
  int64_t a_ = -1, b_ = -1, c_ = -1;
  RcclEthtoolWantedEx wanted_[3];

  void SetUp() override {
    NetTelemetryMicrotest::SetUp();
    Want(0, "stat_a", &a_);
    Want(1, "stat_b", &b_);
    Want(2, "stat_c", &c_);
    SetEthtool("eth0", "other", 1);
    SetEthtool("eth0", "stat_b", 20);
    SetEthtool("eth0", "stat_a", 10);
  }

  void Want(int i, const char* key, int64_t* target) {
    std::snprintf(wanted_[i].key, sizeof(wanted_[i].key), "%s", key);
    wanted_[i].target = target;
  }
};

TEST_F(NetTelemetryEthtoolMicrotest, CollectEthtoolBatch_MatchingStats_FillTargets) {
  std::string ifname;
  ScopedHook ioctlHook(g_ioctl, [&](int fd, unsigned long request, void* arg) {
    EXPECT_EQ(fd, 42);
    ifname = static_cast<struct ifreq*>(arg)->ifr_name;
    return FakeIoctl(fd, request, arg);
  });

  rcclTelemetryCollectEthtoolBatch("eth0", wanted_, 3);

  EXPECT_EQ(a_, 10);
  EXPECT_EQ(b_, 20);
  EXPECT_EQ(c_, -1);
  EXPECT_EQ(ifname, "eth0");
  EXPECT_EQ(g_closedFds, std::vector<int>{42});
}

TEST_F(NetTelemetryEthtoolMicrotest, CollectEthtoolBatch_TargetAlreadySet_KeepsFirstValue) {
  Want(2, "stat_a", &a_);
  SetEthtool("eth0", "stat_c", 30);
  a_ = 5;

  rcclTelemetryCollectEthtoolBatch("eth0", wanted_, 3);

  EXPECT_EQ(a_, 5);
}

TEST_F(NetTelemetryEthtoolMicrotest, CollectEthtoolBatch_TwoKeysForOneTarget_FirstListedStatWins) {
  Want(1, "stat_b", &a_);

  rcclTelemetryCollectEthtoolBatch("eth0", wanted_, 3);

  EXPECT_EQ(a_, 20);
}

TEST_F(NetTelemetryEthtoolMicrotest, CollectEthtoolBatch_FullWidthStatName_Matches) {
  std::string name(ETH_GSTRING_LEN, 'n');
  Want(2, name.c_str(), &c_);
  SetEthtool("eth0", name, 30);

  rcclTelemetryCollectEthtoolBatch("eth0", wanted_, 3);

  EXPECT_EQ(c_, 30);
}

TEST_F(NetTelemetryEthtoolMicrotest, CollectEthtoolBatch_NoInterfaceOrNothingWanted_OpensNoSocket) {
  ScopedHook socketHook(g_socket, [](int, int, int) { return 42; });

  rcclTelemetryCollectEthtoolBatch("", wanted_, 3);
  rcclTelemetryCollectEthtoolBatch("eth0", wanted_, 0);
  rcclTelemetryCollectEthtoolBatch(std::string(IFNAMSIZ, 'e').c_str(), wanted_, 3);

  EXPECT_EQ(socketHook.calls, 0);
}

TEST_F(NetTelemetryEthtoolMicrotest, CollectEthtoolBatch_LongestInterfaceName_IsQueried) {
  std::string eth(IFNAMSIZ - 1, 'e');
  SetEthtool(eth, "stat_a", 3);

  rcclTelemetryCollectEthtoolBatch(eth.c_str(), wanted_, 3);

  EXPECT_EQ(a_, 3);
}

TEST_F(NetTelemetryEthtoolMicrotest, CollectEthtoolBatch_SocketFails_LeavesTargets) {
  g_nextSocketFd = -1;
  ScopedHook ioctlHook(g_ioctl, FakeIoctl);

  rcclTelemetryCollectEthtoolBatch("eth0", wanted_, 3);

  EXPECT_EQ(ioctlHook.calls, 0);
  EXPECT_EQ(a_, -1);
}

TEST_F(NetTelemetryEthtoolMicrotest, CollectEthtoolBatch_IoctlStepFails_ClosesSocketAndLeavesTargets) {
  for (uint32_t cmd : {ETHTOOL_GDRVINFO, ETHTOOL_GSTRINGS, ETHTOOL_GSTATS}) {
    SCOPED_TRACE(cmd);
    g_sys.failEthtoolCmd = cmd;
    g_closedFds.clear();

    rcclTelemetryCollectEthtoolBatch("eth0", wanted_, 3);

    EXPECT_EQ(a_, -1);
    EXPECT_EQ(g_closedFds, std::vector<int>{42});
  }
}

TEST_F(NetTelemetryEthtoolMicrotest, CollectEthtoolBatch_NoStats_ClosesSocket) {
  g_sys.nics["eth0"].clear();
  ScopedHook callocHook(g_calloc, RealCalloc);

  rcclTelemetryCollectEthtoolBatch("eth0", wanted_, 3);

  EXPECT_EQ(callocHook.calls, 0);
  EXPECT_EQ(g_closedFds, std::vector<int>{42});
}

TEST_F(NetTelemetryEthtoolMicrotest, CollectEthtoolBatch_BufferAllocFails_ClosesSocketWithoutQuery) {
  for (int failAt : {0, 1}) {
    SCOPED_TRACE(failAt);
    int calls = 0;
    ScopedHook callocHook(g_calloc,
                          [&](size_t n, size_t size) { return calls++ == failAt ? nullptr : RealCalloc(n, size); });
    g_sys.ethtoolCalls = 0;
    g_closedFds.clear();

    rcclTelemetryCollectEthtoolBatch("eth0", wanted_, 3);

    EXPECT_EQ(g_sys.ethtoolCalls, 1);
    EXPECT_EQ(a_, -1);
    EXPECT_EQ(g_closedFds, std::vector<int>{42});
  }
}

using NetTelemetryDebugfsMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryDebugfsMicrotest, CollectDebugfs_InfoFile_ParsesEachWantedKey) {
  g_sys.files["/sys/kernel/debug/drv/rdma0/info"] = "alpha: 5\nbeta = 7\ngamma 9\ndelta:";
  int64_t hwc[4] = {-1, -1, 3, -1};
  RcclDebugfsWanted wanted[] = {{"alpha", 0}, {"beta", 1}, {"gamma", 2}, {"delta", 3}};

  rcclTelemetryCollectDebugfs(hwc, "rdma0", "drv", wanted, 4);

  EXPECT_EQ(hwc[0], 5);
  EXPECT_EQ(hwc[1], 7);
  EXPECT_EQ(hwc[2], 3);
  EXPECT_EQ(hwc[3], -1);
}

TEST_F(NetTelemetryDebugfsMicrotest, CollectDebugfs_NoDriverOrNothingWanted_OpensNothing) {
  int64_t hwc[1] = {-1};
  RcclDebugfsWanted wanted[] = {{"alpha", 0}};

  rcclTelemetryCollectDebugfs(hwc, "rdma0", "", wanted, 1);
  rcclTelemetryCollectDebugfs(hwc, "rdma0", "drv", wanted, 0);
  rcclTelemetryCollectDebugfs(hwc, "rdma0", "drv", wanted, 1);

  EXPECT_EQ(g_sys.fopened, std::vector<std::string>{"/sys/kernel/debug/drv/rdma0/info"});
  EXPECT_EQ(hwc[0], -1);
}

TEST_F(NetTelemetryDebugfsMicrotest, CollectDebugfs_RepeatedKey_KeepsFirstValue) {
  g_sys.files["/sys/kernel/debug/drv/rdma0/info"] = "alpha: 5\nalpha: 6\n";
  int64_t hwc[1] = {-1};
  RcclDebugfsWanted wanted[] = {{"alpha", 0}};

  rcclTelemetryCollectDebugfs(hwc, "rdma0", "drv", wanted, 1);

  EXPECT_EQ(hwc[0], 5);
}

class NetTelemetryReadCountersMicrotest : public NetTelemetryMicrotest {
 protected:
  int64_t hwc_[RCCL_TELEMETRY_MAX_HWC];
  int64_t pfc_[4][RCCL_TELEMETRY_NUM_PFC_PRIO];
  int64_t delta_[4];

  void SetUp() override {
    NetTelemetryMicrotest::SetUp();
    std::fill(&hwc_[0], &hwc_[0] + RCCL_TELEMETRY_MAX_HWC, 7);
    std::fill(&pfc_[0][0], &pfc_[0][0] + 4 * RCCL_TELEMETRY_NUM_PFC_PRIO, 7);
    std::fill(delta_, delta_ + 4, 7);
  }

  RcclDeviceStats* Device(const RcclHwConfig* hw) {
    RcclDeviceStats* dev = &rcclTelemetryDevs[0];
    std::snprintf(dev->roce_device, sizeof(dev->roce_device), "rdma0");
    std::snprintf(dev->eth_device, sizeof(dev->eth_device), "eth0");
    dev->hw_config = hw;
    return dev;
  }

  void Read(RcclDeviceStats* dev) {
    rcclTelemetryReadCounters(dev, hwc_, pfc_[0], pfc_[1], pfc_[2], pfc_[3], &delta_[0], &delta_[1], &delta_[2],
                              &delta_[3]);
  }
};

TEST_F(NetTelemetryReadCountersMicrotest, ReadCounters_NoRoceDeviceOrHardware_LeavesOutputs) {
  RcclDeviceStats* dev = Device(nullptr);
  Read(dev);
  dev->hw_config = &rcclHwConfigAinic;
  dev->roce_device[0] = '\0';
  Read(dev);

  EXPECT_EQ(hwc_[0], 7);
  EXPECT_EQ(delta_[0], 7);
  EXPECT_TRUE(g_sys.opened.empty());
}

TEST_F(NetTelemetryReadCountersMicrotest, ReadCounters_NothingReadable_ResetsEveryOutputToUnavailable) {
  Read(Device(&rcclHwConfigAinic));

  for (int64_t v : hwc_) EXPECT_EQ(v, -1);
  for (auto& row : pfc_)
    for (int64_t v : row) EXPECT_EQ(v, -1);
  for (int64_t v : delta_) EXPECT_EQ(v, -1);
}

TEST_F(NetTelemetryReadCountersMicrotest, ReadCounters_Ainic_ReadsSysfsEthtoolPfcAndSysfsDeltas) {
  const RcclHwConfig* hw = &rcclHwConfigAinic;
  SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 11);
  SetHwCounter("rdma0", "tx_rdma_ucast_bytes", 100);
  SetHwCounter("rdma0", "rx_rdma_ucast_bytes", 200);
  SetHwCounter("rdma0", "tx_rdma_ucast_pkts", 3);
  SetHwCounter("rdma0", "rx_rdma_ucast_pkts", 4);
  SetEthtool("eth0", "hw_rx_dropped", 21);
  SetEthtool("eth0", "frames_rx_pri_3", 31);
  SetEthtool("eth0", "frames_tx_pri_0", 32);
  SetEthtool("eth0", "rx_pripause_7_1us_count", 33);
  SetEthtool("eth0", "tx_pripause_5_1us_count", 34);

  Read(Device(hw));

  EXPECT_EQ(hwc_[CounterIndex(hw, "cnp_sent")], 11);
  EXPECT_EQ(hwc_[CounterIndex(hw, "hw_rx_dropped")], 21);
  EXPECT_EQ(hwc_[CounterIndex(hw, "cnp_rcvd")], -1);
  EXPECT_EQ(hwc_[hw->num_counters], -1);
  EXPECT_EQ(pfc_[0][3], 31);
  EXPECT_EQ(pfc_[1][0], 32);
  EXPECT_EQ(pfc_[2][7], 33);
  EXPECT_EQ(pfc_[3][5], 34);
  EXPECT_EQ(pfc_[0][0], -1);
  EXPECT_EQ(delta_[0], 100);
  EXPECT_EQ(delta_[1], 200);
  EXPECT_EQ(delta_[2], 3);
  EXPECT_EQ(delta_[3], 4);
}

TEST_F(NetTelemetryReadCountersMicrotest, ReadCounters_Mlx5_ReadsDeltasAndFallbackKeysFromEthtool) {
  const RcclHwConfig* hw = &rcclHwConfigMlx5;
  SetEthtool("eth0", "tx_vport_rdma_unicast_bytes", 100);
  SetEthtool("eth0", "rx_vport_rdma_unicast_bytes", 200);
  SetEthtool("eth0", "tx_vport_rdma_unicast_packets", 3);
  SetEthtool("eth0", "rx_vport_rdma_unicast_packets", 4);
  SetEthtool("eth0", "rx_pause", 41);
  SetEthtool("eth0", "tx_pause_ctrl_phy", 42);
  SetEthtool("eth0", "rx_prio2_pause", 43);
  SetHwCounter("rdma0", "tx_vport_rdma_unicast_bytes", 999);

  Read(Device(hw));

  EXPECT_EQ(hwc_[CounterIndex(hw, "pfc_rx_pause_frames")], 41);
  EXPECT_EQ(hwc_[CounterIndex(hw, "pfc_tx_pause_frames")], 42);
  EXPECT_EQ(pfc_[0][2], 43);
  EXPECT_EQ(delta_[0], 100);
  EXPECT_EQ(delta_[1], 200);
  EXPECT_EQ(delta_[2], 3);
  EXPECT_EQ(delta_[3], 4);
}

TEST_F(NetTelemetryReadCountersMicrotest, ReadCounters_SysfsPrimaryMissing_UsesFallbackKey) {
  const RcclHwConfig* hw = &rcclHwConfigThor2;
  SetHwCounter("rdma0", "np_ecn_marked_roce_packets", 51);
  SetHwCounter("rdma0", "res_oos_drop_count", 52);
  SetHwCounter("rdma0", "oos_drop_count", 99);

  Read(Device(hw));

  EXPECT_EQ(hwc_[CounterIndex(hw, "ecn_marked_pkts")], 51);
  EXPECT_EQ(hwc_[CounterIndex(hw, "oos_drop_count")], 52);
}

TEST_F(NetTelemetryReadCountersMicrotest, ReadCounters_CounterFilter_ReadsOnlyListedCounters) {
  const RcclHwConfig* hw = &rcclHwConfigAinic;
  std::snprintf(rcclTelemetryCfg.hw_counter_list, sizeof(rcclTelemetryCfg.hw_counter_list), "cnp_sent");
  SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 11);
  SetHwCounter("rdma0", "rx_rdma_cnp_pkts", 12);
  SetEthtool("eth0", "hw_rx_dropped", 21);
  SetEthtool("eth0", "frames_rx_pri_3", 31);
  SetHwCounter("rdma0", "tx_rdma_ucast_bytes", 100);

  Read(Device(hw));

  EXPECT_EQ(hwc_[CounterIndex(hw, "cnp_sent")], 11);
  EXPECT_EQ(hwc_[CounterIndex(hw, "cnp_rcvd")], -1);
  EXPECT_EQ(hwc_[CounterIndex(hw, "hw_rx_dropped")], -1);
  EXPECT_EQ(pfc_[0][3], 31);
  EXPECT_EQ(delta_[0], 100);
}

TEST_F(NetTelemetryReadCountersMicrotest, ReadCounters_NoNetdev_StillReadsSysfs) {
  RcclDeviceStats* dev = Device(&rcclHwConfigThor2);
  dev->eth_device[0] = '\0';
  SetHwCounter("rdma0", "tx_bytes", 100);
  ScopedHook socketHook(g_socket, [](int, int, int) { return 42; });

  Read(dev);

  EXPECT_EQ(delta_[0], 100);
  EXPECT_EQ(socketHook.calls, 0);
}

TEST_F(NetTelemetryReadCountersMicrotest, ReadCounters_DebugfsRows_ReadDriverInfoFile) {
  static const RcclHwCounterDesc counters[] = {HWC("first", HWC_DEBUGFS, "alpha"), HWC("second", HWC_DEBUGFS, "beta"),
                                               HWC("third", HWC_DEBUGFS, "gamma")};
  RcclHwConfig hw = rcclHwConfigAinic;
  hw.counters = counters;
  hw.num_counters = 3;
  g_sys.links[std::string(kIb) + "rdma0/device/driver"] = "/drivers/ionic";
  g_sys.files["/sys/kernel/debug/ionic/rdma0/info"] = "alpha: 5\nbeta: 6\ngamma: 7\n";
  std::snprintf(rcclTelemetryCfg.hw_counter_list, sizeof(rcclTelemetryCfg.hw_counter_list), "first,third");

  Read(Device(&hw));

  EXPECT_EQ(hwc_[0], 5);
  EXPECT_EQ(hwc_[1], -1);
  EXPECT_EQ(hwc_[2], 7);
}

TEST_F(NetTelemetryReadCountersMicrotest, ReadCounters_NoDebugfsRowEnabled_OpensNoDebugfsFile) {
  static const RcclHwCounterDesc counters[] = {HWC("first", HWC_DEBUGFS, "alpha")};
  RcclHwConfig hw = rcclHwConfigAinic;
  hw.counters = counters;
  hw.num_counters = 1;
  std::snprintf(rcclTelemetryCfg.hw_counter_list, sizeof(rcclTelemetryCfg.hw_counter_list), "second");

  Read(Device(&hw));

  EXPECT_TRUE(g_sys.fopened.empty());
}

using NetTelemetrySnapshotMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetrySnapshotMicrotest, FirstChannelSetup_TakesBaselineOnce) {
  Enable();
  SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 11);
  SetHwCounter("rdma0", "tx_rdma_ucast_bytes", 100);
  RcclDeviceStats* dev = UsedDevice(0, "rdma0", "eth0", "ionic");
  SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 15);
  size_t reads = g_sys.opened.size();

  ASSERT_EQ(rcclTelemetrySetupChannel(0, 1, 1, nullptr), 0);

  EXPECT_EQ(dev->snap_taken, 1);
  EXPECT_EQ(dev->snap_init_hw_counters[CounterIndex(&rcclHwConfigAinic, "cnp_sent")], 11);
  EXPECT_EQ(dev->snap_init_tx_bytes, 100);
  EXPECT_EQ(g_sys.opened.size(), reads);
}

TEST_F(NetTelemetrySnapshotMicrotest, FirstChannelSetup_NoNetdevOrHardware_MarksBaselineTakenWithoutReading) {
  Enable();
  g_sys.links[std::string(kIb) + "rdma0/device/driver"] = "/drivers/ionic";
  AddNic("rdma1", "eth1", "e1000e");
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);

  ASSERT_EQ(rcclTelemetrySetupChannel(0, 0, 1, nullptr), 0);
  ASSERT_EQ(rcclTelemetrySetupChannel(1, 0, 1, nullptr), 0);

  EXPECT_EQ(rcclTelemetryDevs[0].snap_taken, 1);
  EXPECT_EQ(rcclTelemetryDevs[1].snap_taken, 1);
  EXPECT_TRUE(g_sys.opened.empty());
}

TEST_F(NetTelemetrySnapshotMicrotest, EnsureSnapshot_DeviceOutOfRange_IsNoOp) {
  rcclTelemetryEnsureSnapshot(-1);
  rcclTelemetryEnsureSnapshot(RCCL_TELEMETRY_MAX_DEVS);

  EXPECT_EQ(rcclTelemetryDevs[0].snap_taken, 0);
}

using NetTelemetryCollectMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryCollectMicrotest, CollectHwCounters_BothEndsRead_ReportsDeltas) {
  Enable();
  SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 11);
  SetHwCounter("rdma0", "tx_rdma_ucast_bytes", 100);
  SetHwCounter("rdma0", "rx_rdma_ucast_bytes", 200);
  SetHwCounter("rdma0", "tx_rdma_ucast_pkts", 3);
  SetHwCounter("rdma0", "rx_rdma_ucast_pkts", 4);
  SetEthtool("eth0", "frames_rx_pri_1", 10);
  SetEthtool("eth0", "frames_tx_pri_1", 10);
  SetEthtool("eth0", "rx_pripause_1_1us_count", 10);
  SetEthtool("eth0", "tx_pripause_1_1us_count", 10);
  RcclDeviceStats* dev = UsedDevice(0, "rdma0", "eth0", "ionic");
  SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 15);
  SetHwCounter("rdma0", "tx_rdma_ucast_bytes", 160);
  SetHwCounter("rdma0", "rx_rdma_ucast_bytes", 270);
  SetHwCounter("rdma0", "tx_rdma_ucast_pkts", 8);
  SetHwCounter("rdma0", "rx_rdma_ucast_pkts", 10);
  SetEthtool("eth0", "frames_rx_pri_1", 11);
  SetEthtool("eth0", "frames_tx_pri_1", 12);
  SetEthtool("eth0", "rx_pripause_1_1us_count", 13);
  SetEthtool("eth0", "tx_pripause_1_1us_count", 14);

  rcclTelemetryCollectHwCounters(dev);

  EXPECT_EQ(dev->hw_counters[CounterIndex(&rcclHwConfigAinic, "cnp_sent")], 4);
  EXPECT_EQ(dev->hw_counters[CounterIndex(&rcclHwConfigAinic, "cnp_rcvd")], -1);
  EXPECT_EQ(dev->delta_tx_bytes, 60);
  EXPECT_EQ(dev->delta_rx_bytes, 70);
  EXPECT_EQ(dev->delta_tx_packets, 5);
  EXPECT_EQ(dev->delta_rx_packets, 6);
  EXPECT_EQ(dev->pfc_rx_frames[1], 1);
  EXPECT_EQ(dev->pfc_tx_frames[1], 2);
  EXPECT_EQ(dev->pfc_rx_pause_us[1], 3);
  EXPECT_EQ(dev->pfc_tx_pause_us[1], 4);
  EXPECT_EQ(dev->pfc_rx_frames[0], -1);
}

TEST_F(NetTelemetryCollectMicrotest, CollectHwCounters_EitherEndUnavailable_ReportsUnavailable) {
  Enable();
  SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 11);
  SetHwCounter("rdma0", "tx_rdma_ucast_bytes", 100);
  SetHwCounter("rdma0", "tx_rdma_ucast_pkts", 3);
  SetEthtool("eth0", "frames_rx_pri_1", 10);
  SetEthtool("eth0", "rx_pripause_1_1us_count", 10);
  RcclDeviceStats* dev = UsedDevice(0, "rdma0", "eth0", "ionic");
  g_sys.files.clear();
  SetHwCounter("rdma0", "rx_rdma_ucast_bytes", 270);
  SetHwCounter("rdma0", "rx_rdma_ucast_pkts", 10);
  SetHwCounter("rdma0", "rx_rdma_cnp_pkts", 3);
  g_sys.nics["eth0"].clear();
  SetEthtool("eth0", "frames_tx_pri_1", 12);
  SetEthtool("eth0", "tx_pripause_1_1us_count", 14);

  rcclTelemetryCollectHwCounters(dev);

  EXPECT_EQ(dev->hw_counters[CounterIndex(&rcclHwConfigAinic, "cnp_sent")], -1);
  EXPECT_EQ(dev->hw_counters[CounterIndex(&rcclHwConfigAinic, "cnp_rcvd")], -1);
  EXPECT_EQ(dev->delta_tx_bytes, -1);
  EXPECT_EQ(dev->delta_rx_bytes, -1);
  EXPECT_EQ(dev->delta_tx_packets, -1);
  EXPECT_EQ(dev->delta_rx_packets, -1);
  EXPECT_EQ(dev->pfc_rx_frames[1], -1);
  EXPECT_EQ(dev->pfc_tx_frames[1], -1);
  EXPECT_EQ(dev->pfc_rx_pause_us[1], -1);
  EXPECT_EQ(dev->pfc_tx_pause_us[1], -1);
}

TEST_F(NetTelemetryCollectMicrotest, CollectHwCounters_ZeroBaseline_IsADelta) {
  Enable();
  SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 0);
  SetHwCounter("rdma0", "tx_rdma_ucast_bytes", 0);
  SetEthtool("eth0", "frames_rx_pri_1", 0);
  RcclDeviceStats* dev = UsedDevice(0, "rdma0", "eth0", "ionic");
  SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 6);
  SetHwCounter("rdma0", "tx_rdma_ucast_bytes", 7);
  SetEthtool("eth0", "frames_rx_pri_1", 8);

  rcclTelemetryCollectHwCounters(dev);

  EXPECT_EQ(dev->hw_counters[CounterIndex(&rcclHwConfigAinic, "cnp_sent")], 6);
  EXPECT_EQ(dev->delta_tx_bytes, 7);
  EXPECT_EQ(dev->pfc_rx_frames[1], 8);
}

TEST_F(NetTelemetryCollectMicrotest, CollectHwCounters_NoRoceDeviceOrHardware_LeavesCounters) {
  RcclDeviceStats* dev = &rcclTelemetryDevs[0];
  dev->hw_counters[0] = 7;
  dev->delta_tx_bytes = 7;
  std::snprintf(dev->roce_device, sizeof(dev->roce_device), "rdma0");
  rcclTelemetryCollectHwCounters(dev);
  dev->roce_device[0] = '\0';
  dev->hw_config = &rcclHwConfigAinic;
  rcclTelemetryCollectHwCounters(dev);

  EXPECT_EQ(dev->hw_counters[0], 7);
  EXPECT_EQ(dev->delta_tx_bytes, 7);
}

// ---------------------------------------------------------------------------

class NetTelemetrySamplerMicrotest : public NetTelemetryMicrotest {
 protected:
  std::vector<struct timespec> sleeps_;

  // Each sleep ends the loop, so one call of the sampler body takes one round of samples.
  void StopOnSleep() {
    g_nanosleep = [this](const struct timespec* req, struct timespec*) {
      sleeps_.push_back(*req);
      rcclTelemetrySamplerStopFlag = 1;
      return 0;
    };
  }
};

TEST_F(NetTelemetrySamplerMicrotest, SampledTableIdx_ResolvesNameThenAlternate) {
  EXPECT_EQ(rcclTelemetrySampledTableIdx(&rcclHwConfigAinic, 2), CounterIndex(&rcclHwConfigAinic, "cnp_rcvd"));
  EXPECT_EQ(rcclTelemetrySampledTableIdx(&rcclHwConfigMlx5, 2), CounterIndex(&rcclHwConfigMlx5, "cnp_handled"));
  EXPECT_EQ(rcclTelemetrySampledTableIdx(&rcclHwConfigAinic, 7), CounterIndex(&rcclHwConfigAinic, "to_retransmits"));
  EXPECT_EQ(rcclTelemetrySampledTableIdx(&rcclHwConfigMlx5, 7),
            CounterIndex(&rcclHwConfigMlx5, "local_ack_timeout_err"));
  EXPECT_EQ(rcclTelemetrySampledTableIdx(&rcclHwConfigThor2, 3), -1);
}

TEST_F(NetTelemetrySamplerMicrotest, SampledTableIdx_AlternateMissingToo_ReturnsMinusOne) {
  static const RcclHwCounterDesc counters[] = {HWC("cnp_sent", HWC_IB_SYSFS, "x")};
  RcclHwConfig hw = rcclHwConfigAinic;
  hw.counters = counters;
  hw.num_counters = 1;

  EXPECT_EQ(rcclTelemetrySampledTableIdx(&hw, 2), -1);
  EXPECT_EQ(rcclTelemetrySampledTableIdx(&hw, 1), 0);
}

TEST_F(NetTelemetrySamplerMicrotest, SampleRow_PrimaryMissing_ReadsFallback) {
  RcclDeviceStats dev{};
  std::snprintf(dev.roce_device, sizeof(dev.roce_device), "rdma0");
  RcclHwCounterDesc row = HWC_FB("x", HWC_IB_SYSFS, "primary", "fallback");
  RcclHwCounterDesc noFallback = HWC("x", HWC_IB_SYSFS, "primary");
  SetHwCounter("rdma0", "fallback", 9);

  EXPECT_EQ(rcclTelemetrySampleRow(&dev, &row), 9);
  EXPECT_EQ(rcclTelemetrySampleRow(&dev, &noFallback), -1);
  SetHwCounter("rdma0", "primary", 4);
  EXPECT_EQ(rcclTelemetrySampleRow(&dev, &row), 4);
}

TEST_F(NetTelemetrySamplerMicrotest, WarnUnresolvedSampled_EveryEvaluatedDeviceMissing_WarnsOncePerName) {
  int idx[2][RCCL_TEL_NUM_SAMPLED];
  int warned[RCCL_TEL_NUM_SAMPLED] = {};
  for (int c = 0; c < RCCL_TEL_NUM_SAMPLED; c++) {
    idx[0][c] = c == 0 ? -1 : 0;
    idx[1][c] = c == 0 ? -1 : -2;
  }
  idx[0][1] = -2;

  rcclTelemetryWarnUnresolvedSampled(idx, 2, warned);
  rcclTelemetryWarnUnresolvedSampled(idx, 2, warned);

  EXPECT_EQ(StderrWrites(), 1);
  EXPECT_EQ(warned[0], 1);
  EXPECT_EQ(warned[1], 0);
  EXPECT_EQ(warned[2], 0);
}

TEST_F(NetTelemetrySamplerMicrotest, WarnUnresolvedSampled_ResolvedOnAnyDevice_DoesNotWarn) {
  int idx[2][RCCL_TEL_NUM_SAMPLED];
  int warned[RCCL_TEL_NUM_SAMPLED] = {};
  for (int c = 0; c < RCCL_TEL_NUM_SAMPLED; c++) {
    idx[0][c] = -1;
    idx[1][c] = c;
  }

  rcclTelemetryWarnUnresolvedSampled(idx, 2, warned);

  EXPECT_EQ(StderrWrites(), 0);
}

TEST_F(NetTelemetrySamplerMicrotest, SamplerMain_OneRound_SamplesEachUsableDeviceThenSleeps) {
  Enable();
  AddNic("rdma0", "eth0", "ionic");
  AddNic("rdma2", "eth2", "bnxt_re");
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);
  ASSERT_EQ(rcclTelemetryRegisterDevice(2, "rdma2", "IB"), 2);
  rcclTelemetryBytes(0, 1, 100);
  rcclTelemetryBytes(0, 0, 50);
  SetHwCounter("rdma0", "rx_rdma_ecn_pkts", 7);
  SetHwCounter("rdma0", "tx_rdma_ack_timeout", 8);
  rcclTelemetrySamples = static_cast<RcclHwSample*>(::calloc(RCCL_TEL_MAX_SAMPLES, sizeof(RcclHwSample)));
  rcclTelemetrySampleIntervalMs = 1500;
  g_nowNs = 7000123456;
  StopOnSleep();

  EXPECT_EQ(rcclTelemetrySamplerMain(nullptr), nullptr);

  ASSERT_EQ(rcclTelemetryNumSamples, 2);
  const RcclHwSample& s = rcclTelemetrySamples[0];
  EXPECT_EQ(s.ts_us, 7000123);
  EXPECT_EQ(s.dev_idx, 0);
  EXPECT_EQ(s.tx_bytes, 100u);
  EXPECT_EQ(s.rx_bytes, 50u);
  EXPECT_EQ(s.cong[0], 7);
  EXPECT_EQ(s.cong[7], 8);
  EXPECT_EQ(s.cong[1], -1);
  EXPECT_EQ(rcclTelemetrySamples[1].dev_idx, 2);
  EXPECT_EQ(rcclTelemetrySamples[1].cong[3], -1);
  ASSERT_EQ(sleeps_.size(), 1u);
  EXPECT_EQ(sleeps_[0].tv_sec, 1);
  EXPECT_EQ(sleeps_[0].tv_nsec, 500000000);
  EXPECT_EQ(StderrWrites(), 0);
}

TEST_F(NetTelemetrySamplerMicrotest, SamplerMain_StopRequested_ReturnsWithoutSampling) {
  Enable();
  rcclTelemetrySamplerStopFlag = 1;
  StopOnSleep();

  EXPECT_EQ(rcclTelemetrySamplerMain(nullptr), nullptr);

  EXPECT_TRUE(sleeps_.empty());
}

TEST_F(NetTelemetrySamplerMicrotest, SamplerMain_BufferFull_StopsSampling) {
  Enable();
  AddNic("rdma0", "eth0", "ionic");
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  rcclTelemetrySamples = static_cast<RcclHwSample*>(::calloc(RCCL_TEL_MAX_SAMPLES, sizeof(RcclHwSample)));
  rcclTelemetryNumSamples = RCCL_TEL_MAX_SAMPLES;
  StopOnSleep();

  EXPECT_EQ(rcclTelemetrySamplerMain(nullptr), nullptr);

  EXPECT_TRUE(sleeps_.empty());
  EXPECT_EQ(rcclTelemetryNumSamples, RCCL_TEL_MAX_SAMPLES);
}

TEST_F(NetTelemetrySamplerMicrotest, SamplerMain_LastFreeSlot_IsFilled) {
  Enable();
  AddNic("rdma0", "eth0", "ionic");
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  rcclTelemetrySamples = static_cast<RcclHwSample*>(::calloc(RCCL_TEL_MAX_SAMPLES, sizeof(RcclHwSample)));
  rcclTelemetryNumSamples = RCCL_TEL_MAX_SAMPLES - 1;
  StopOnSleep();

  rcclTelemetrySamplerMain(nullptr);

  EXPECT_EQ(rcclTelemetryNumSamples, RCCL_TEL_MAX_SAMPLES);
}

TEST_F(NetTelemetrySamplerMicrotest, SamplerStart_NoInterval_StartsNothing) {
  ScopedHook createHook(g_pthreadCreate, DefaultPthreadCreate);

  rcclTelemetrySamplerStart();

  EXPECT_EQ(createHook.calls, 0);
  EXPECT_EQ(rcclTelemetrySamples, nullptr);
}

TEST_F(NetTelemetrySamplerMicrotest, SamplerStart_BufferAllocFails_StartsNothing) {
  rcclTelemetrySampleIntervalMs = 5;
  ScopedHook callocHook(g_calloc, [](size_t, size_t) -> void* { return nullptr; });
  ScopedHook createHook(g_pthreadCreate, DefaultPthreadCreate);

  rcclTelemetrySamplerStart();

  EXPECT_EQ(createHook.calls, 0);
  EXPECT_FALSE(rcclTelemetrySamplerRunning);
}

TEST_F(NetTelemetrySamplerMicrotest, SamplerStart_ThreadCreateFails_StaysStopped) {
  rcclTelemetrySampleIntervalMs = 5;
  ScopedHook createHook(g_pthreadCreate,
                        [](pthread_t*, const pthread_attr_t*, void* (*)(void*), void*) { return EAGAIN; });

  rcclTelemetrySamplerStart();

  EXPECT_FALSE(rcclTelemetrySamplerRunning);
}

TEST_F(NetTelemetrySamplerMicrotest, SamplerStartStop_RunsThreadUntilStopped) {
  rcclTelemetrySampleIntervalMs = 1;
  rcclTelemetrySamplerStopFlag = 1;

  rcclTelemetrySamplerStart();
  EXPECT_TRUE(rcclTelemetrySamplerRunning);
  EXPECT_EQ(static_cast<int>(rcclTelemetrySamplerStopFlag), 0);
  rcclTelemetrySamplerStop();

  EXPECT_FALSE(rcclTelemetrySamplerRunning);
  EXPECT_EQ(static_cast<int>(rcclTelemetrySamplerStopFlag), 1);
}

TEST_F(NetTelemetrySamplerMicrotest, SamplerStop_NotRunning_LeavesStopFlag) {
  rcclTelemetrySamplerStop();

  EXPECT_EQ(static_cast<int>(rcclTelemetrySamplerStopFlag), 0);
}

// ---------------------------------------------------------------------------

using NetTelemetryJsonMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryJsonMicrotest, WriteJson_ActiveDevice_WritesFullReport) {
  g_sys.files["/proc/" + std::to_string(getpid()) + "/comm"] = "proc\n";
  Enable();
  rcclTelemetryCfg.histogram_max_buckets = 2;
  AddNic("rdma0", "eth0", "e1000e");
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  RcclQpStats* data = SetupQps(0, 0, 2);
  RcclQpStats* cts = rcclTelemetryResolveQp(0, 0, 1);
  rcclTelemetrySetQpRole(0, 0, 0, 1);
  rcclTelemetrySetQpRole(0, 0, 1, 0);
  rcclTelemetryBytes(0, 1, 100);
  rcclTelemetryBytes(0, 0, 50);
  rcclTelemetryCqError(0);
  rcclTelemetryAddCqPolls(0, 7);
  rcclTelemetryWqeSize(0, 0);
  rcclTelemetryWqeSize(0, 3);
  rcclTelemetryWqeSize(0, 2);
  rcclTelemetryQpSendPosted(data, 0);
  rcclTelemetryQpRecvPosted(data);
  rcclTelemetryQpWqeComplete(data, g_nowNs - 40000);
  rcclTelemetryQpCtsSent(cts, 1);
  rcclTelemetryQpSlotMiss(cts);
  rcclTelemetryChRequestCompleted(rcclTelemetryChannel(0, 0));

  std::string json =
      std::regex_replace(WriteJson(), std::regex(R"("(start|end)_time": "[^"]*")"), "\"$1_time\": \"T\"");

  std::string qpTail = R"(              "wqe_completion_histogram": [
                {"latency_ns": 30000, "num_wqe": 0},
                {"latency_ns": 60000, "num_wqe": )";
  std::string expected = R"({
  "version": "1.0",
  "host_name": "testhost",
  "process_name": "proc",
  "process_id": ")" + std::to_string(getpid()) + R"(",
  "start_time": "T",
  "end_time": "T",
  "transport": "IB",
  "devices": [
    {
      "device_id": 0,
      "roce_device": "rdma0",
      "eth_device": "eth0",
      "hw_type": "unsupported",
      "tx_bytes": 100,
      "rx_bytes": 50,
      "num_cq_errors": 1,
      "cq_poll_count": 7,
      "num_channels": 1,
      "active_channels": 1,
      "num_qp_untracked": 0,
      "wqe_size_stats": [
        {"max_wqe_size": 0, "num_wqe": 1},
        {"max_wqe_size": 3, "num_wqe": 2}
      ],
      "channels": [
        {
          "id": 0,
          "num_wqe_sent": 1,
          "num_recv_wqe": 1,
          "num_wqe_rcvd": 1,
          "num_wqe_completed": 1,
          "num_cts_sent": 1,
          "num_req_completed": 1,
          "num_data_qp": 1,
          "num_cts_qp": 1,
          "num_qp_untracked": 0,
          "queue_pairs": [
            {
              "id": 0,
              "data_qp": true,
              "num_wqe_sent": 1,
              "num_recv_wqe": 1,
              "num_wqe_rcvd": 1,
              "num_wqe_completed": 1,
              "num_slot_miss": 0,
              "num_cts_sent": 0,
              "num_cts_sent_signalled": 0,
              "num_cts_sent_unsignalled": 0,
              "num_write_wqe": 1,
              "num_write_imm_wqe": 0,
              "wqe_completion_ns_min": 40000,
              "wqe_completion_ns_max": 40000,
)" + qpTail + R"(1}
              ]
            },
            {
              "id": 1,
              "data_qp": false,
              "num_wqe_sent": 0,
              "num_recv_wqe": 0,
              "num_wqe_rcvd": 0,
              "num_wqe_completed": 0,
              "num_slot_miss": 1,
              "num_cts_sent": 1,
              "num_cts_sent_signalled": 1,
              "num_cts_sent_unsignalled": 0,
              "num_write_wqe": 0,
              "num_write_imm_wqe": 0,
              "wqe_completion_ns_min": 0,
              "wqe_completion_ns_max": 0,
)" + qpTail + R"(0}
              ]
            }
          ]
        }
      ],
      "hw_counters": {
        "delta_tx_bytes": -1,
        "delta_rx_bytes": -1,
        "delta_tx_packets": -1,
        "delta_rx_packets": -1
      }
    }
  ]
}
)";
  EXPECT_EQ(json, expected);
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_NoDevices_WritesEmptyListAndDefaultTransport) {
  Enable();

  std::string json = WriteJson();

  EXPECT_NE(json.find("  \"transport\": \"IB-CAST\",\n  \"devices\": [\n  ]\n}\n"), std::string::npos) << json;
  EXPECT_EQ(json.find("latency_sample_interval"), std::string::npos);
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_FirstDeviceWithoutTransport_UsesDefaultTransport) {
  Enable();
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", nullptr), 0);

  EXPECT_NE(WriteJson().find("\"transport\": \"IB-CAST\""), std::string::npos);
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_HostnameUnavailable_WritesUnknown) {
  Enable();
  ScopedHook hostHook(g_gethostname, [](char*, size_t) { return -1; });

  EXPECT_NE(WriteJson().find("\"host_name\": \"unknown\""), std::string::npos);
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_IdleDevice_IsSkipped) {
  Enable();
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);
  rcclTelemetryBytes(1, 1, 1);

  std::string json = WriteJson();

  EXPECT_EQ(json.find("\"device_id\": 0"), std::string::npos);
  EXPECT_NE(json.find("  \"devices\": [\n    {\n      \"device_id\": 1,"), std::string::npos) << json;
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_AnySingleActivity_ReportsDevice) {
  std::vector<std::function<void()>> activities = {
      [] { rcclTelemetryBytes(0, 1, 1); },
      [] { rcclTelemetryBytes(0, 0, 1); },
      [] { rcclTelemetryCqError(0); },
      [] { rcclTelemetrySetupChannel(0, 0, 1, nullptr); },
  };
  for (size_t i = 0; i < activities.size(); i++) {
    SCOPED_TRACE(i);
    ResetTelemetryState();
    Enable();
    ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
    activities[i]();

    EXPECT_NE(WriteJson().find("\"device_id\": 0"), std::string::npos);
  }
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_ChannelWithUntrackedQpsOnly_IsActive) {
  Enable();
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  int calls = 0;
  ScopedHook memalignHook(g_posixMemalign,
                          [&](void** p, size_t a, size_t n) {
                            return calls++ == 0 ? DefaultPosixMemalign(p, a, n) : ENOMEM;
                          });
  ASSERT_EQ(rcclTelemetrySetupChannel(0, 1, 2, nullptr), -1);

  std::string json = WriteJson();

  EXPECT_NE(json.find("\"active_channels\": 1,"), std::string::npos) << json;
  EXPECT_NE(json.find("          \"id\": 1,\n"), std::string::npos);
  EXPECT_NE(json.find("          \"num_qp_untracked\": 2,\n          \"queue_pairs\": [\n          ]\n"),
            std::string::npos);
  EXPECT_EQ(json.find("          \"id\": 0,"), std::string::npos);
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_SeveralDevicesAndChannels_AreCommaSeparated) {
  Enable();
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);
  SetupQps(0, 0, 1);
  SetupQps(0, 1, 1);
  rcclTelemetryBytes(1, 1, 1);

  std::string json = WriteJson();

  EXPECT_NE(json.find("          ]\n        },\n        {\n          \"id\": 1,"), std::string::npos) << json;
  EXPECT_NE(json.find("    },\n    {\n      \"device_id\": 1,"), std::string::npos);
  EXPECT_NE(json.find("      \"channels\": [\n      ],"), std::string::npos);
  EXPECT_NE(json.find("      \"wqe_size_stats\": [],"), std::string::npos);
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_ChannelsUnknown_WritesNullIds) {
  Enable();
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  SetupQps(0, 0, 1);
  rcclTelemetryMarkChannelsUnknown(0);

  std::string json = WriteJson();

  EXPECT_NE(json.find("\"num_channels\": null,"), std::string::npos);
  EXPECT_NE(json.find("\"id\": null,"), std::string::npos);
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_LatencySampled_ReportsIntervalAndSampledCounts) {
  Enable();
  rcclTelemetryLatencySampleN = 4;
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  RcclQpStats* qp = SetupQps(0, 0, 1);
  rcclTelemetryQpWqeComplete(qp, g_nowNs - 10);
  rcclTelemetryQpWqeComplete(qp, RCCL_TELEMETRY_TS_UNSAMPLED);

  std::string json = WriteJson();

  EXPECT_NE(json.find("  \"latency_sample_interval\": 4,\n"), std::string::npos);
  EXPECT_NE(json.find("          \"num_wqe_completed\": 2,\n          \"num_wqe_sampled\": 1,\n"), std::string::npos)
      << json;
  EXPECT_NE(json.find("              \"num_wqe_sampled\": 1,\n"), std::string::npos);
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_MoreBucketsThanHistogram_ClampsToHistogramSize) {
  Enable();
  rcclTelemetryCfg.histogram_max_buckets = RCCL_TELEMETRY_HISTOGRAM_SIZE + 4;
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  SetupQps(0, 0, 1);

  std::string json = WriteJson();

  std::string last =
      "{\"latency_ns\": " + std::to_string(30000 * RCCL_TELEMETRY_HISTOGRAM_SIZE) + ", \"num_wqe\": 0}\n";
  EXPECT_NE(json.find(last), std::string::npos) << json;
  EXPECT_EQ(json.find(std::to_string(30000 * (RCCL_TELEMETRY_HISTOGRAM_SIZE + 1))), std::string::npos);
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_SupportedHardware_WritesEveryTableCounterAndPfcArrays) {
  Enable();
  AddNic("rdma0", "eth0", "mlx5_core");
  ASSERT_EQ(rcclTelemetryRegisterDevice(0, "rdma0", "IB"), 0);
  rcclTelemetryBytes(0, 1, 1);
  rcclTelemetryDevs[0].hw_counters[CounterIndex(&rcclHwConfigMlx5, "cnp_sent")] = 12;
  rcclTelemetryDevs[0].pfc_tx_pause_us[7] = 5;

  std::string json = WriteJson();

  EXPECT_NE(json.find("      \"hw_type\": \"mlx5\",\n"), std::string::npos);
  EXPECT_NE(json.find("      \"hw_counters\": {\n        \"ecn_marked_pkts\": -1,\n        \"cnp_sent\": 12,\n"),
            std::string::npos)
      << json;
  EXPECT_NE(json.find("        \"pfc_tx_pause_frames\": -1,\n"
                      "        \"pfc_rx_frames\": [-1, -1, -1, -1, -1, -1, -1, -1],\n"
                      "        \"pfc_tx_frames\": [-1, -1, -1, -1, -1, -1, -1, -1],\n"
                      "        \"pfc_rx_pause_us\": [-1, -1, -1, -1, -1, -1, -1, -1],\n"
                      "        \"pfc_tx_pause_us\": [-1, -1, -1, -1, -1, -1, -1, 5],\n"
                      "        \"delta_tx_bytes\": -1,\n"),
            std::string::npos);
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_Samples_AppendsTimeSeries) {
  Enable();
  ASSERT_EQ(rcclTelemetryRegisterDevice(3, "rdma3", "IB"), 3);
  rcclTelemetrySamples = static_cast<RcclHwSample*>(::calloc(2, sizeof(RcclHwSample)));
  for (int i = 0; i < 2; i++) {
    RcclHwSample& s = rcclTelemetrySamples[i];
    s.ts_us = 100 + i;
    s.dev_idx = 3;
    s.tx_bytes = 10;
    s.rx_bytes = 20;
    for (int c = 0; c < RCCL_TEL_NUM_SAMPLED; c++) s.cong[c] = c;
  }
  rcclTelemetryNumSamples = 2;

  std::string json = WriteJson();

  std::string row = "\"device_id\": 3, \"roce_device\": \"rdma3\", \"tx_bytes\": 10, \"rx_bytes\": 20, "
                    "\"ecn_marked_pkts\": 0, \"cnp_sent\": 1, \"cnp_rcvd\": 2, \"out_of_buffer\": 3, "
                    "\"oos_drop_count\": 4, \"seq_err_naks_rcvd\": 5, \"rnr_retry_err\": 6, "
                    "\"local_ack_timeout_err\": 7}";
  EXPECT_NE(json.find("  ],\n  \"hw_samples\": [\n    {\"ts_us\": 100, " + row + ",\n    {\"ts_us\": 101, " + row +
                      "\n  ]\n}\n"),
            std::string::npos)
      << json;
}

TEST_F(NetTelemetryJsonMicrotest, WriteJson_SampleCountWithoutBuffer_WritesNoTimeSeries) {
  Enable();
  rcclTelemetryNumSamples = 3;

  EXPECT_EQ(WriteJson().find("hw_samples"), std::string::npos);
}

// ---------------------------------------------------------------------------

using NetTelemetryFlushMicrotest = NetTelemetryMicrotest;

TEST_F(NetTelemetryFlushMicrotest, Flush_Disabled_WritesNothing) {
  EXPECT_EQ(rcclTelemetryFlush(), 0);

  EXPECT_TRUE(g_sys.fopened.empty());
}

TEST_F(NetTelemetryFlushMicrotest, Flush_Enabled_WritesReportOnce) {
  RUN_ISOLATED_TEST("NetTelemetry_Flush_Enabled_WritesReportOnce", [this]() {
    SetMicroEnv("RCCL_TELEMETRY_OUTPUT_DIR", "/out");
    Enable();
    g_sys.fopened.clear();

    EXPECT_EQ(rcclTelemetryFlush(), 0);
    EXPECT_EQ(rcclTelemetryFlush(), 0);

    EXPECT_EQ(g_sys.fopened, std::vector<std::string>{FlushPath()});
    EXPECT_EQ(g_sys.Written(FlushPath()).rfind("{\n  \"version\": \"1.0\",\n", 0), 0u);
  });
}

TEST_F(NetTelemetryFlushMicrotest, Flush_ExitHandler_WritesReport) {
  RUN_ISOLATED_TEST("NetTelemetry_Flush_ExitHandler_WritesReport", [this]() {
    SetMicroEnv("RCCL_TELEMETRY_OUTPUT_DIR", "/out");
    Enable();
    ASSERT_EQ(g_atexitFns.size(), 1u);

    g_atexitFns[0]();

    EXPECT_FALSE(g_sys.Written(FlushPath()).empty());
  });
}

TEST_F(NetTelemetryFlushMicrotest, Flush_OutputNotWritable_ReturnsError) {
  RUN_ISOLATED_TEST("NetTelemetry_Flush_OutputNotWritable_ReturnsError", []() {
    Enable();
    g_sys.failWrite = true;

    EXPECT_EQ(rcclTelemetryFlush(), -1);

    EXPECT_EQ(StderrWrites(), 1);
  });
}

TEST_F(NetTelemetryFlushMicrotest, Flush_HostnameUnavailable_NamesFileUnknown) {
  RUN_ISOLATED_TEST("NetTelemetry_Flush_HostnameUnavailable_NamesFileUnknown", []() {
    Enable();
    g_sys.fopened.clear();
    ScopedHook hostHook(g_gethostname, [](char*, size_t) { return -1; });

    EXPECT_EQ(rcclTelemetryFlush(), 0);

    EXPECT_EQ(g_sys.fopened, std::vector<std::string>{"/tmp/rccl_telemetry_unknown_" + std::to_string(getuid()) +
                                                      "_" + std::to_string(getpid()) + ".json"});
  });
}

TEST_F(NetTelemetryFlushMicrotest, Flush_CollectsCountersOnlyForUsedDevices) {
  RUN_ISOLATED_TEST("NetTelemetry_Flush_CollectsCountersOnlyForUsedDevices", []() {
    Enable();
    SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 11);
    SetHwCounter("rdma1", "tx_rdma_cnp_pkts", 11);
    RcclDeviceStats* used = UsedDevice(0, "rdma0", "eth0", "ionic");
    AddNic("rdma1", "eth1", "ionic");
    ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);
    SetHwCounter("rdma0", "tx_rdma_cnp_pkts", 14);
    SetHwCounter("rdma1", "tx_rdma_cnp_pkts", 14);

    ASSERT_EQ(rcclTelemetryFlush(), 0);

    int idx = CounterIndex(&rcclHwConfigAinic, "cnp_sent");
    EXPECT_EQ(used->hw_counters[idx], 3);
    EXPECT_EQ(rcclTelemetryDevs[1].hw_counters[idx], -1);
  });
}

TEST_F(NetTelemetryFlushMicrotest, Flush_SamplerRunning_StopsSamplerFirst) {
  RUN_ISOLATED_TEST("NetTelemetry_Flush_SamplerRunning_StopsSamplerFirst", []() {
    SetMicroEnv("RCCL_TELEMETRY_SAMPLE_MS", "1");
    Enable();
    ASSERT_TRUE(rcclTelemetrySamplerRunning);

    ASSERT_EQ(rcclTelemetryFlush(), 0);

    EXPECT_FALSE(rcclTelemetrySamplerRunning);
  });
}

TEST_F(NetTelemetryFlushMicrotest, Flush_DebugEnv_LogsPerDeviceTotals) {
  RUN_ISOLATED_TEST("NetTelemetry_Flush_DebugEnv_LogsPerDeviceTotals", []() {
    Enable();
    ASSERT_EQ(rcclTelemetryRegisterDevice(1, "rdma1", "IB"), 1);
    SetupQps(1, 1, 1);
    SetMicroEnv("RCCL_TELEMETRY_DEBUG", "1");
    g_fprintfCalls.clear();

    ASSERT_EQ(rcclTelemetryFlush(), 0);

    EXPECT_EQ(StderrWrites(), 3);
  });
}

}  // namespace
