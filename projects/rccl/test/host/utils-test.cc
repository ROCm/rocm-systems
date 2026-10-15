/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/utils.cc: bus-id and host/pid hashing, interface lists, the
// memory stack's spill path and the intrusive address map. HIP, env and libc calls are seamed here.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

#include "nccl.h"
#include "core.h"
#include "debug.h"
#include "os.h"
#include "utils.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;

struct UtilsLog {
  ncclDebugLogLevel level;
  std::string text;
};
static std::vector<UtilsLog> g_utilsLog;
static bool g_utilsLogToStderr = false;  // death tests match the abort message on stderr

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_utilsLog.push_back({level, buf});
  if (g_utilsLogToStderr) std::fprintf(stderr, "%s\n", buf);
}

static std::map<std::string, std::string> g_utilsEnv;
const char* ncclGetEnv(const char* name) {
  auto it = g_utilsEnv.find(name);
  return it == g_utilsEnv.end() ? nullptr : it->second.c_str();
}
static uint64_t g_utilsPid = 4242;
uint64_t ncclOsGetPid() { return g_utilsPid; }
char* ncclOsStrSep(char** stringp, const char* delim) { return strsep(stringp, delim); }

static std::map<std::string, int> g_hipCalls;
static std::map<std::string, int> g_hipFailAt;  // fn -> 1-based call that fails
static int g_hipDevice = 2;
static std::map<int, int> g_hipAttr;
static std::vector<int> g_hipAttrDevices;
static std::string g_hipBusId = "0000:c1:00.0";
static int g_hipBusIdLen = 0;
static hipError_t HipResult(const char* fn) {
  return ++g_hipCalls[fn] == g_hipFailAt[fn] ? hipErrorInvalidValue : hipSuccess;
}
hipError_t hipGetDevice(int* device) {
  *device = g_hipDevice;
  return HipResult("hipGetDevice");
}
hipError_t hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attr, int device) {
  *value = g_hipAttr[attr];
  g_hipAttrDevices.push_back(device);
  return HipResult("hipDeviceGetAttribute");
}
hipError_t hipDeviceGetPCIBusId(char* busId, int len, int device) {
  g_hipBusIdLen = len;
  snprintf(busId, len, "%s", g_hipBusId.c_str());
  return HipResult("hipDeviceGetPCIBusId");
}
hipError_t hipGetLastError() { return hipSuccess; }
const char* hipGetErrorString(hipError_t) { return "fake hip error"; }

// libc seams: allocations are tracked so leaks and frees are visible.
static std::set<void*> g_live;
static std::vector<size_t> g_mallocSizes;
static int g_allocCalls = 0;
static int g_allocFailAt = 0;
static bool g_strdupFails = false;
static void* Track(void* p) {
  if (p) g_live.insert(p);
  return p;
}
static void* UtilsMalloc(size_t size) {
  g_mallocSizes.push_back(size);
  return ++g_allocCalls == g_allocFailAt ? nullptr : Track(std::malloc(size));
}
static void* UtilsCalloc(size_t n, size_t size) {
  return ++g_allocCalls == g_allocFailAt ? nullptr : Track(std::calloc(n, size));
}
static char* UtilsStrdup(const char* s) { return g_strdupFails ? nullptr : (char*)Track(::strdup(s)); }
static void UtilsFree(void* p) {
  g_live.erase(p);
  std::free(p);
}

static std::string g_hostname = "node7";
static int g_gethostnameResult = 0;
static int g_gethostnameCalls = 0;
static int UtilsGethostname(char* name, size_t len) {
  g_gethostnameCalls++;
  if (g_gethostnameResult != 0) return g_gethostnameResult;
  std::memcpy(name, g_hostname.c_str(), std::min(len, g_hostname.size() + 1));  // unterminated if it fills len
  return 0;
}
static std::string g_bootIdPath;  // file served for the boot id; empty: fopen fails
static std::string g_fopenPath;
static FILE* UtilsFopen(const char* path, const char* mode) {
  g_fopenPath = path;
  return g_bootIdPath.empty() ? nullptr : std::fopen(g_bootIdPath.c_str(), mode);
}
static std::string g_pidNs = "pid:[4026531836]";
static int g_readlinkResult = 0;  // 0: copy g_pidNs
static std::string g_readlinkPath;
static size_t g_readlinkLen = 0;
static ssize_t UtilsReadlink(const char* path, char* buf, size_t len) {
  g_readlinkPath = path;
  g_readlinkLen = len;
  if (g_readlinkResult != 0) {
    buf[0] = 'X';  // contents are unspecified on failure
    return g_readlinkResult;
  }
  std::memcpy(buf, g_pidNs.data(), g_pidNs.size());  // readlink does not terminate
  return g_pidNs.size();
}
static int g_sysconfCalls = 0;
static long UtilsSysconf(int name) {
  g_sysconfCalls++;
  return name == _SC_PAGESIZE ? 4096 : -1;
}

#define malloc(n) UtilsMalloc(n)
#define calloc(n, s) UtilsCalloc(n, s)
#define strdup(s) UtilsStrdup(s)
#define free(p) UtilsFree(p)
#define gethostname(n, l) UtilsGethostname(n, l)
#define fopen(p, m) UtilsFopen(p, m)
#define readlink(p, b, l) UtilsReadlink(p, b, l)
#define sysconf(n) UtilsSysconf(n)

#include UTILS_CC_PATH

#undef malloc
#undef calloc
#undef strdup
#undef free
#undef gethostname
#undef fopen
#undef readlink
#undef sysconf

namespace {

constexpr size_t kHunk = 64 << 10;

bool Logged(ncclDebugLogLevel level, const char* needle) {
  for (const auto& m : g_utilsLog) {
    if (m.level == level && m.text.find(needle) != std::string::npos) return true;
  }
  return false;
}

uintptr_t Addr(const void* p) { return reinterpret_cast<uintptr_t>(p); }

struct Entry {
  void* key;
  Entry* next;
};
using EntryMap = ncclIntruAddressMap<Entry, void*, &Entry::key, &Entry::next>;

class UtilsMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_utilsLog.clear();
    g_utilsLogToStderr = false;
    g_utilsEnv.clear();
    g_utilsPid = 4242;
    g_hipCalls.clear();
    g_hipFailAt.clear();
    g_hipDevice = 2;
    g_hipAttr.clear();
    g_hipAttrDevices.clear();
    g_hipBusId = "0000:c1:00.0";
    g_live.clear();
    g_mallocSizes.clear();
    g_allocCalls = 0;
    g_allocFailAt = 0;
    g_strdupFails = false;
    g_hostname = "node7";
    g_gethostnameResult = 0;
    g_bootIdPath.clear();
    g_fopenPath.clear();
    g_readlinkResult = 0;
    ncclMemoryStackConstruct(&stack_);
  }
  void TearDown() override {
    if (!stackDestructed_) ncclMemoryStackDestruct(&stack_);
    if (map_.base.table) std::free(map_.base.table);
    if (!bootIdFile_.empty()) std::remove(bootIdFile_.c_str());
  }

  void WriteBootId(const std::string& id) {
    bootIdFile_ = "/tmp/utils-test-boot-id-" + std::to_string(getpid());
    FILE* f = std::fopen(bootIdFile_.c_str(), "w");
    std::fputs(id.c_str(), f);
    std::fclose(f);
    g_bootIdPath = bootIdFile_;
  }
  uint64_t HostHash() {
    hostHashValue = 0;
    getHostHashOnce();
    return hostHashValue;
  }
  void* Alloc(size_t size, size_t align = 1) { return ncclMemoryStack::allocate(&stack_, size, align); }
  // Bump-allocates until exactly `left` bytes remain in the top hunk.
  void FillTo(size_t left) { Alloc(stack_.topFrame.end - stack_.topFrame.bumper - left); }
  void DestructStack() {
    ncclMemoryStackDestruct(&stack_);
    stackDestructed_ = true;
  }
  ncclResult_t Insert(Entry* e) { return ncclIntruAddressMapInsert(&map_, e->key, e); }
  Entry* Find(void* key) {
    Entry* e = reinterpret_cast<Entry*>(0x1);
    EXPECT_EQ(ncclSuccess, ncclIntruAddressMapFind(&map_, key, &e));
    return e;
  }

  ncclMemoryStack stack_;
  bool stackDestructed_ = false;
  EntryMap map_{};
  std::string bootIdFile_;
};

// Device queries

TEST_F(UtilsMicrotest, CudaCompCap_CombinesMajorAndMinorOfTheCurrentDevice) {
  g_hipAttr[hipDeviceAttributeComputeCapabilityMajor] = 9;
  g_hipAttr[hipDeviceAttributeComputeCapabilityMinor] = 4;
  EXPECT_EQ(94, ncclCudaCompCap());
  EXPECT_EQ((std::vector<int>{2, 2}), g_hipAttrDevices);
}

TEST_F(UtilsMicrotest, CudaCompCap_AnyHipFailureIsZero) {
  g_hipAttr[hipDeviceAttributeComputeCapabilityMajor] = 9;
  g_hipFailAt["hipGetDevice"] = 1;
  EXPECT_EQ(0, ncclCudaCompCap());
  for (int nth : {1, 2}) {
    g_hipCalls.clear();
    g_hipFailAt = {{"hipDeviceGetAttribute", nth}};
    EXPECT_EQ(0, ncclCudaCompCap()) << nth;
  }
}

TEST_F(UtilsMicrotest, BusId_RoundTripsThroughInt64) {
  char busId[32];
  ASSERT_EQ(ncclSuccess, int64ToBusId(0x12c11f3, busId));
  EXPECT_STREQ("0012:c1:1f.3", busId);
  ASSERT_EQ(ncclSuccess, int64ToBusId(0x00000c100b, busId));
  EXPECT_STREQ("0000:c1:00.b", busId);
  int64_t id = 0;
  ASSERT_EQ(ncclSuccess, busIdToInt64("0000:C1:00.3", &id));
  EXPECT_EQ(0xc1003, id);
}

// Separators are skipped, parsing stops at the first non-hex character, and at most 16 digits are kept.
TEST_F(UtilsMicrotest, BusIdToInt64_KeepsUpToSixteenHexDigits) {
  int64_t id = -1;
  ASSERT_EQ(ncclSuccess, busIdToInt64("0001:af:00.0/0000:02:00.0", &id));
  EXPECT_EQ(0x1af000, id);
  ASSERT_EQ(ncclSuccess, busIdToInt64("00000000000000001", &id));
  EXPECT_EQ(0, id);
  ASSERT_EQ(ncclSuccess, busIdToInt64("z", &id));
  EXPECT_EQ(0, id);
}

TEST_F(UtilsMicrotest, PciPathToInt64_ParsesTheLastPathComponent) {
  char withSlash[] = "/sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/";
  char withoutSlash[] = "/sys/class/pci0000:00/0000:00:02.0/0000:02:00.1";
  int64_t id = -1;
  ASSERT_EQ(ncclSuccess, pciPathToInt64(withSlash, &id));
  EXPECT_EQ(0x2000, id);
  ASSERT_EQ(ncclSuccess, pciPathToInt64(withoutSlash, &id));
  EXPECT_EQ(0x2001, id);
}

TEST_F(UtilsMicrotest, GetBusId_ParsesTheDevicesPciBusId) {
  int64_t id = -1;
  ASSERT_EQ(ncclSuccess, getBusId(1, &id));
  EXPECT_EQ(0xc1000, id);
  EXPECT_EQ(17, g_hipBusIdLen);
  g_hipFailAt["hipDeviceGetPCIBusId"] = 2;
  EXPECT_EQ(ncclUnhandledCudaError, getBusId(1, &id));
}

// Host and process identity

TEST_F(UtilsMicrotest, GetHostName_CutsAtTheDelimiter) {
  g_hostname = "node7.cluster.local";
  char name[64];
  ASSERT_EQ(ncclSuccess, getHostName(name, sizeof(name), '.'));
  EXPECT_STREQ("node7", name);
  ASSERT_EQ(ncclSuccess, getHostName(name, sizeof(name), '\0'));
  EXPECT_STREQ("node7.cluster.local", name);
}

TEST_F(UtilsMicrotest, GetHostName_TerminatesATruncatedName) {
  g_hostname = "abcdefgh";
  char name[8];
  ASSERT_EQ(ncclSuccess, getHostName(name, sizeof(name), '.'));
  EXPECT_STREQ("abcdefg", name);
}

TEST_F(UtilsMicrotest, GetHostName_FailureReportsUnknown) {
  g_gethostnameResult = -1;
  char name[64];
  EXPECT_EQ(ncclSystemError, getHostName(name, sizeof(name), '.'));
  EXPECT_STREQ("unknown", name);
}

TEST_F(UtilsMicrotest, HostHash_IsHostnamePlusBootId) {
  WriteBootId("8a3c-boot\nignored");
  EXPECT_EQ(getHash("node78a3c-boot", 14), HostHash());
  EXPECT_EQ("/proc/sys/kernel/random/boot_id", g_fopenPath);
}

TEST_F(UtilsMicrotest, HostHash_FallsBackToTheHostnameAlone) {
  EXPECT_EQ(getHash("node7", 5), HostHash());  // boot id unreadable
  WriteBootId("");
  EXPECT_EQ(getHash("node7", 5), HostHash());  // boot id empty
}

TEST_F(UtilsMicrotest, HostHash_NcclHostIdReplacesHostnameAndBootId) {
  WriteBootId("8a3c-boot");
  g_utilsEnv["NCCL_HOSTID"] = "my-host-id";
  EXPECT_EQ(getHash("my-host-id", 10), HostHash());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "NCCL_HOSTID set by environment to my-host-id"));
  EXPECT_EQ("", g_fopenPath);
}

// The id string is capped at 1023 bytes either way.
TEST_F(UtilsMicrotest, HostHash_TruncatesToTheBuffer) {
  g_hostname = std::string(1020, 'h');
  WriteBootId("bootid");
  std::string expected = g_hostname + "boo";
  EXPECT_EQ(getHash(expected.c_str(), expected.size()), HostHash());

  g_utilsEnv["NCCL_HOSTID"] = std::string(2000, 'x');
  EXPECT_EQ(getHash(std::string(1023, 'x').c_str(), 1023), HostHash());
}

TEST_F(UtilsMicrotest, GetHostHash_IsComputedOnce) {
  uint64_t first = getHostHash();
  int calls = g_gethostnameCalls;
  g_hostname = "other";
  EXPECT_EQ(first, getHostHash());
  EXPECT_EQ(calls, g_gethostnameCalls);
}

TEST_F(UtilsMicrotest, HashCombine_IsOrderSensitive) {
  uint64_t acc[2] = {1, 1};
  uint64_t base = 7, value = 9;
  eatHash(acc, &base);
  eatHash(acc, &value);
  EXPECT_EQ(digestHash(acc), hashCombine(7, 9));
  EXPECT_NE(hashCombine(7, 9), hashCombine(9, 7));
}

TEST_F(UtilsMicrotest, PidHash_IsPidPlusPidNamespace) {
  EXPECT_EQ(getHash("4242pid:[4026531836]", 20), getPidHash());
  EXPECT_EQ("/proc/self/ns/pid", g_readlinkPath);
  EXPECT_EQ(1024u - 1 - 4, g_readlinkLen);
  g_readlinkResult = -1;
  g_utilsPid = 17;
  EXPECT_EQ(getHash("17", 2), getPidHash());
}

// Interface lists

TEST_F(UtilsMicrotest, ParseStringList_ReadsPrefixPortRailAndPlane) {
  netIf list[4];
  ASSERT_EQ(4, parseStringList("mlx5_0:1:2:3,,ib::5,eth:1::2,eth", list, 4));
  EXPECT_STREQ("mlx5_0", list[0].prefix);
  EXPECT_EQ(1, list[0].port);
  EXPECT_EQ(2, list[0].rail);
  EXPECT_EQ(3, list[0].plane);
  EXPECT_STREQ("ib", list[1].prefix);
  EXPECT_EQ(-1, list[1].port);
  EXPECT_EQ(5, list[1].rail);
  EXPECT_EQ(-1, list[1].plane);
  EXPECT_EQ(1, list[2].port);
  EXPECT_EQ(-1, list[2].rail);
  EXPECT_EQ(2, list[2].plane);
  EXPECT_STREQ("eth", list[3].prefix);
  EXPECT_EQ(-1, list[3].port);
  EXPECT_TRUE(g_live.empty());
}

TEST_F(UtilsMicrotest, ParseStringList_SkipsEmptyPrefixesAndStopsAtMaxList) {
  netIf list[2];
  EXPECT_EQ(1, parseStringList(":1,eth0", list, 2));
  EXPECT_STREQ("eth0", list[0].prefix);
  EXPECT_EQ(2, parseStringList("a,b,c", list, 2));
  EXPECT_STREQ("b", list[1].prefix);
  EXPECT_EQ(0, parseStringList(nullptr, list, 2));
  g_strdupFails = true;
  EXPECT_EQ(0, parseStringList("a", list, 2));
}

TEST_F(UtilsMicrotest, MatchIfList_EmptyListMatchesEverything) {
  int ifId = 5;
  EXPECT_TRUE(matchIfList("eth0", 1, nullptr, 0, false, &ifId));
  EXPECT_EQ(-1, ifId);
}

TEST_F(UtilsMicrotest, MatchIfList_PrefixOrExactAndPort) {
  netIf list[2];
  ASSERT_EQ(2, parseStringList("ib:2,eth", list, 2));
  int ifId = 5;
  EXPECT_TRUE(matchIfList("ib0", 2, list, 2, false, &ifId));
  EXPECT_EQ(0, ifId);
  EXPECT_TRUE(matchIfList("ib0", -1, list, 2, false, &ifId));  // any port
  EXPECT_FALSE(matchIfList("ib0", 1, list, 2, false, &ifId));
  EXPECT_EQ(-1, ifId);
  EXPECT_TRUE(matchIfList("eth1", 7, list, 2, false, &ifId));  // list entry has no port
  EXPECT_EQ(1, ifId);
  EXPECT_FALSE(matchIfList("eth1", 7, list, 2, true, nullptr));
  EXPECT_TRUE(matchIfList("eth", 7, list, 2, true, &ifId));
  EXPECT_FALSE(matchIfList("et", 7, list, 2, true, &ifId));
  EXPECT_FALSE(matchIfList("et", 7, list, 2, false, &ifId));
}

// Memory stack spill path

TEST_F(UtilsMicrotest, MemoryStack_FirstSpillAllocatesA64KiBHunk) {
  void* obj = Alloc(100, 8);
  ncclMemoryStack::Hunk* hunk = stack_.topFrame.hunk;
  EXPECT_EQ((std::vector<size_t>{kHunk}), g_mallocSizes);
  EXPECT_EQ(hunk, stack_.stub.above);
  EXPECT_EQ(kHunk, hunk->size);
  EXPECT_EQ(nullptr, hunk->above);
  EXPECT_EQ(Addr(hunk) + sizeof(*hunk), Addr(obj));
  EXPECT_EQ(Addr(hunk) + kHunk, stack_.topFrame.end);
  EXPECT_EQ(Addr(obj) + 100, stack_.topFrame.bumper);
}

TEST_F(UtilsMicrotest, MemoryStack_NextHunkGrowsBy64KiBAndIsAligned) {
  Alloc(100);
  ncclMemoryStack::Hunk* first = stack_.topFrame.hunk;
  FillTo(8191);
  void* obj = Alloc(9000, 64);
  ncclMemoryStack::Hunk* second = stack_.topFrame.hunk;
  EXPECT_EQ((std::vector<size_t>{kHunk, 2 * kHunk}), g_mallocSizes);
  EXPECT_EQ(second, first->above);
  EXPECT_EQ(2 * kHunk, second->size);
  EXPECT_EQ(0u, Addr(obj) % 64);
  EXPECT_GE(Addr(obj), Addr(second) + sizeof(*second));
  EXPECT_LT(Addr(obj), Addr(second) + sizeof(*second) + 64);
}

// With 8 KiB or more still free, an object that does not fit goes out of hunk behind an in-hunk proxy.
TEST_F(UtilsMicrotest, MemoryStack_RoomLeftSendsBigObjectsOutOfHunk) {
  Alloc(100);
  void* obj = Alloc(70000);
  EXPECT_EQ((std::vector<size_t>{kHunk, 70000}), g_mallocSizes);
  ASSERT_NE(nullptr, stack_.topFrame.unhunks);
  EXPECT_EQ(obj, stack_.topFrame.unhunks->obj);
  EXPECT_EQ(0u, Addr(stack_.topFrame.unhunks) % alignof(ncclMemoryStack::Unhunk));
  EXPECT_EQ(Addr(stack_.topFrame.unhunks) + sizeof(ncclMemoryStack::Unhunk), stack_.topFrame.bumper);
  void* second = Alloc(70000);
  EXPECT_EQ(second, stack_.topFrame.unhunks->obj);
  EXPECT_EQ(obj, stack_.topFrame.unhunks->next->obj);
  DestructStack();
  EXPECT_TRUE(g_live.empty());
}

TEST_F(UtilsMicrotest, MemoryStack_OutOfHunkFrom8KiBLeft) {
  Alloc(100);
  FillTo(8192);
  void* obj = Alloc(9000);
  EXPECT_EQ((std::vector<size_t>{kHunk, 9000}), g_mallocSizes);
  EXPECT_EQ(obj, stack_.topFrame.unhunks->obj);
}

TEST_F(UtilsMicrotest, MemoryStack_ObjectTooBigForTheNextHunkGoesOutOfHunk) {
  Alloc(100);
  FillTo(4000);
  void* obj = Alloc(200000);
  EXPECT_EQ((std::vector<size_t>{kHunk, 200000}), g_mallocSizes);
  EXPECT_EQ(obj, stack_.topFrame.unhunks->obj);
}

TEST_F(UtilsMicrotest, MemoryStack_ReusesTheEmptyHunkLeftAboveByPop) {
  Alloc(100);
  FillTo(4000);
  ncclMemoryStackPush(&stack_);
  Alloc(5000);
  ncclMemoryStack::Hunk* above = stack_.topFrame.hunk;
  ncclMemoryStackPop(&stack_);
  ASSERT_NE(above, stack_.topFrame.hunk);

  void* obj = Alloc(5000, 64);
  EXPECT_EQ(2u, g_mallocSizes.size());
  EXPECT_EQ(above, stack_.topFrame.hunk);
  EXPECT_EQ(0u, Addr(obj) % 64);
  EXPECT_GE(Addr(obj), Addr(above) + sizeof(*above));
  EXPECT_LT(Addr(obj), Addr(above) + sizeof(*above) + 64);
  EXPECT_EQ(Addr(above) + above->size, stack_.topFrame.end);
}

// Observed: when the empty hunk above is too small and no proxy fits below it, a new hunk replaces
// it in the chain and the old one is never freed.
TEST_F(UtilsMicrotest, MemoryStack_ReplacingAnUnusableHunkAboveLeaksIt) {
  Alloc(100);
  FillTo(8);
  ncclMemoryStackPush(&stack_);  // the frame itself spills into a second hunk
  ncclMemoryStack::Hunk* above = stack_.topFrame.hunk;
  ncclMemoryStackPop(&stack_);

  void* obj = Alloc(200000);
  EXPECT_EQ((std::vector<size_t>{kHunk, 2 * kHunk, 2 * kHunk, 200000}), g_mallocSizes);
  EXPECT_NE(above, stack_.stub.above->above);
  EXPECT_EQ(obj, stack_.topFrame.unhunks->obj);
  DestructStack();
  EXPECT_EQ((std::set<void*>{above}), g_live);
  std::free(above);
}

TEST_F(UtilsMicrotest, MemoryStack_DestructFreesOutOfHunkObjectsOfEveryFrame) {
  Alloc(100);
  Alloc(70000);
  ncclMemoryStackPush(&stack_);
  Alloc(70000);
  ncclMemoryStackPush(&stack_);
  Alloc(70000);
  EXPECT_EQ(4u, g_live.size());
  DestructStack();
  EXPECT_TRUE(g_live.empty());
}

TEST_F(UtilsMicrotest, MemoryStackDeathTest_MallocFailureAborts) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  EXPECT_DEATH(
      {
        g_utilsLogToStderr = true;
        g_allocFailAt = 1;
        Alloc(100);
      },
      "malloc\\(size=65536\\) returned null");
  Alloc(100);
  EXPECT_DEATH(
      {
        g_utilsLogToStderr = true;
        g_allocFailAt = g_allocCalls + 1;
        Alloc(70000);
      },
      "malloc\\(size=70000\\) returned null");
}

// Bit strings

const char* BitName(int bit) {
  static const char* kNames[] = {"INIT", "COLL", "P2P", "SHM"};
  return kNames[bit];
}

TEST_F(UtilsMicrotest, BitsToString_JoinsSetBitsOrPrintsTheWildcard) {
  char buf[64];
  ASSERT_EQ(ncclSuccess, ncclBitsToString(0x5, 0xf, BitName, buf, sizeof(buf), "ALL"));
  EXPECT_STREQ("INIT|P2P", buf);
  ASSERT_EQ(ncclSuccess, ncclBitsToString(0x1f, 0xf, BitName, buf, sizeof(buf), "ALL"));
  EXPECT_STREQ("ALL", buf);
  ASSERT_EQ(ncclSuccess, ncclBitsToString(0xf, 0xf, BitName, buf, sizeof(buf), nullptr));
  EXPECT_STREQ("INIT|COLL|P2P|SHM", buf);
  ASSERT_EQ(ncclSuccess, ncclBitsToString(0x10, 0xf, BitName, buf, sizeof(buf), "ALL"));
  EXPECT_STREQ("INIT|COLL|P2P|SHM", buf);  // observed: no set bits leaves buf untouched
  EXPECT_EQ(ncclInvalidArgument, ncclBitsToString(1, 1, BitName, nullptr, 8, nullptr));
  EXPECT_EQ(ncclInvalidArgument, ncclBitsToString(1, 1, BitName, buf, 0, nullptr));
}

// Observed: pos runs past bufLen once a name is truncated, so the next name is written past it.
TEST_F(UtilsMicrotest, BitsToString_TruncationWritesPastBufLen) {
  char buf[64];
  std::memset(buf, '#', sizeof(buf));
  ASSERT_EQ(ncclSuccess, ncclBitsToString(0x3, 0xf, BitName, buf, 4, nullptr));
  EXPECT_STREQ("INI", buf);
  EXPECT_EQ('#', buf[4]);
  EXPECT_EQ('C', buf[5]);  // how much of "COLL" lands is up to libc
}

// Page alignment

TEST_F(UtilsMicrotest, PageSize_IsCachedAfterTheFirstQuery) {
  EXPECT_EQ(4096u, get_sc_page_size());
  int calls = g_sysconfCalls;
  EXPECT_EQ(4096u, get_sc_page_size());
  EXPECT_EQ(calls, g_sysconfCalls);
  EXPECT_LE(calls, 1);
}

TEST_F(UtilsMicrotest, AlignedPtrAndSize_CoverWholePages) {
  void* ptr = nullptr;
  size_t size = 0;
  get_aligned_ptr_and_size(reinterpret_cast<void*>(0x1234), 10, &ptr, &size);
  EXPECT_EQ(0x1000u, Addr(ptr));
  EXPECT_EQ(0x1000u, size);
  get_aligned_ptr_and_size(reinterpret_cast<void*>(0x1ff0), 0x20, &ptr, &size);
  EXPECT_EQ(0x1000u, Addr(ptr));
  EXPECT_EQ(0x2000u, size);
  get_aligned_ptr_and_size(reinterpret_cast<void*>(0x3000), 0x1000, &ptr, &size);
  EXPECT_EQ(0x3000u, Addr(ptr));
  EXPECT_EQ(0x1000u, size);
  get_aligned_ptr_and_size(ptr, 1, nullptr, &size);
  get_aligned_ptr_and_size(ptr, 1, &ptr, nullptr);
  EXPECT_EQ(0x1000u, size);
}

// Intrusive address map

TEST_F(UtilsMicrotest, HashPointer_MixesTheKeyIntoTheTopBits) {
  EXPECT_EQ(7u, ncclHashPointer(4, reinterpret_cast<void*>(0x1000)));
  EXPECT_EQ(0u, ncclHashPointer(8, reinterpret_cast<void*>(0xdeadbeef)));
  EXPECT_EQ(54560u, ncclHashPointer(16, reinterpret_cast<void*>(0x7fff12345678)));
  EXPECT_EQ(1u, ncclHashPointer(1, reinterpret_cast<void*>(0x1)));
}

TEST_F(UtilsMicrotest, AddressMap_InsertFindRemove) {
  Entry a{reinterpret_cast<void*>(0x1000), nullptr};
  Entry b{reinterpret_cast<void*>(0x2000), nullptr};
  EXPECT_EQ(nullptr, Find(a.key));
  ASSERT_EQ(ncclSuccess, Insert(&a));
  EXPECT_EQ(4, map_.base.hbits);
  EXPECT_EQ(1, map_.base.count);
  ASSERT_EQ(ncclSuccess, Insert(&b));
  EXPECT_EQ(&a, Find(a.key));
  EXPECT_EQ(&b, Find(b.key));
  EXPECT_EQ(nullptr, Find(reinterpret_cast<void*>(0x3000)));

  ASSERT_EQ(ncclSuccess, ncclIntruAddressMapRemove(&map_, reinterpret_cast<void*>(0x3000)));
  EXPECT_EQ(2, map_.base.count);
  ASSERT_EQ(ncclSuccess, ncclIntruAddressMapRemove(&map_, a.key));
  EXPECT_EQ(nullptr, Find(a.key));
  EXPECT_EQ(nullptr, a.next);
  EXPECT_EQ(1, map_.base.count);
  ASSERT_EQ(ncclSuccess, ncclIntruAddressMapRemove(&map_, b.key));
  EXPECT_EQ(0, map_.base.hbits);  // the last removal frees the table
  EXPECT_EQ(nullptr, map_.base.table);
  EXPECT_TRUE(g_live.empty());
  EXPECT_EQ(ncclSuccess, ncclIntruAddressMapRemove(&map_, b.key));
}

// Keys sharing a bucket chain; removal unlinks from the head, the middle and the tail.
TEST_F(UtilsMicrotest, AddressMap_CollidingKeysChainInOneBucket) {
  std::vector<Entry> entries;
  for (uintptr_t k = 1; entries.size() < 4; k++) {
    if (ncclHashPointer(4, reinterpret_cast<void*>(k << 12)) == 3) {
      entries.push_back({reinterpret_cast<void*>(k << 12), nullptr});
    }
  }
  for (auto& e : entries) ASSERT_EQ(ncclSuccess, Insert(&e));
  for (auto& e : entries) EXPECT_EQ(&e, Find(e.key));
  for (int i : {1, 3, 0}) {
    ASSERT_EQ(ncclSuccess, ncclIntruAddressMapRemove(&map_, entries[i].key));
    EXPECT_EQ(nullptr, Find(entries[i].key));
    EXPECT_EQ(nullptr, entries[i].next);
  }
  EXPECT_EQ(&entries[2], Find(entries[2].key));
  EXPECT_EQ(1, map_.base.count);
}

// More than two entries per bucket doubles the table and rehashes every chain.
TEST_F(UtilsMicrotest, AddressMap_GrowsAtTwoEntriesPerBucket) {
  std::vector<Entry> entries(33);
  for (size_t i = 0; i < entries.size(); i++) entries[i].key = reinterpret_cast<void*>((i + 1) << 12);
  for (size_t i = 0; i < 32; i++) ASSERT_EQ(ncclSuccess, Insert(&entries[i]));
  EXPECT_EQ(4, map_.base.hbits);
  ASSERT_EQ(ncclSuccess, Insert(&entries[32]));
  EXPECT_EQ(5, map_.base.hbits);
  EXPECT_EQ(33, map_.base.count);
  for (auto& e : entries) EXPECT_EQ(&e, Find(e.key));
  for (int b = 0; b < 32; b++) {
    for (Entry* e = static_cast<Entry*>(map_.base.table[b]); e; e = e->next) {
      EXPECT_EQ((uint64_t)b, ncclHashPointer(5, e->key));
    }
  }
  EXPECT_EQ(1u, g_live.size());
}

TEST_F(UtilsMicrotest, AddressMap_AllocationFailuresAreSystemErrors) {
  Entry e{reinterpret_cast<void*>(0x1000), nullptr};
  g_allocFailAt = 1;
  EXPECT_EQ(ncclSystemError, Insert(&e));
  EXPECT_EQ(0, map_.base.hbits);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "calloc(16 entries) returned null"));

  std::vector<Entry> entries(33);
  for (size_t i = 0; i < entries.size(); i++) entries[i].key = reinterpret_cast<void*>((i + 1) << 12);
  for (size_t i = 0; i < 32; i++) ASSERT_EQ(ncclSuccess, Insert(&entries[i]));
  g_allocFailAt = g_allocCalls + 1;
  EXPECT_EQ(ncclSystemError, Insert(&entries[32]));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "malloc(32 entries) returned null"));
  EXPECT_EQ(4, map_.base.hbits);
  EXPECT_EQ(32, map_.base.count);
  EXPECT_EQ(&entries[31], Find(entries[31].key));
}

TEST_F(UtilsMicrotest, AddressMap_InsertingALinkedObjectIsReported) {
  Entry other{nullptr, nullptr};
  Entry e{reinterpret_cast<void*>(0x1000), &other};
  ASSERT_EQ(ncclSuccess, Insert(&e));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "non-NULL next pointer"));
  EXPECT_EQ(nullptr, e.next);
}

TEST_F(UtilsMicrotest, AddressMap_ShortKeysCompareOnlyTheirBytes) {
  struct Short {
    uint32_t key;
    uint32_t pad;
    Short* next;
  } s{0x1234, 0xffffffff, nullptr};
  const int keyOff = offsetof(Short, key), nextOff = offsetof(Short, next);
  ASSERT_EQ(ncclSuccess, ncclIntruAddressMapInsert_untyped(&map_.base, 4, keyOff, nextOff, 0x1234, &s));
  void* found = nullptr;
  ASSERT_EQ(ncclSuccess, ncclIntruAddressMapFind_untyped(&map_.base, 4, keyOff, nextOff, 0x1234, &found));
  EXPECT_EQ(&s, found);
  ASSERT_EQ(ncclSuccess, ncclIntruAddressMapRemove_untyped(&map_.base, 4, keyOff, nextOff, 0x1234));
  EXPECT_EQ(0, map_.base.count);
}

TEST_F(UtilsMicrotest, AddressMap_RejectsBadArguments) {
  Entry e{reinterpret_cast<void*>(0x1000), nullptr};
  void* found = nullptr;
  ncclIntruAddressMap_untyped* map = &map_.base;
  EXPECT_EQ(ncclInvalidUsage, ncclIntruAddressMapInsert_untyped(nullptr, 8, 0, 8, 1, &e));
  EXPECT_EQ(ncclInvalidUsage, ncclIntruAddressMapInsert_untyped(map, 8, 0, 8, 1, nullptr));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "Object pointer is NULL"));
  EXPECT_EQ(ncclInvalidUsage, ncclIntruAddressMapFind_untyped(nullptr, 8, 0, 8, 1, &found));
  EXPECT_EQ(ncclInvalidUsage, ncclIntruAddressMapFind_untyped(map, 8, 0, 8, 1, nullptr));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "Output object pointer is NULL"));
  EXPECT_EQ(ncclInvalidUsage, ncclIntruAddressMapRemove_untyped(nullptr, 8, 0, 8, 1));
  for (int keySize : {0, 9}) {
    SCOPED_TRACE(keySize);
    g_utilsLog.clear();
    EXPECT_EQ(ncclInvalidUsage, ncclIntruAddressMapInsert_untyped(map, keySize, 0, 8, 1, &e));
    EXPECT_EQ(ncclInvalidUsage, ncclIntruAddressMapFind_untyped(map, keySize, 0, 8, 1, &found));
    EXPECT_EQ(ncclInvalidUsage, ncclIntruAddressMapRemove_untyped(map, keySize, 0, 8, 1));
    EXPECT_TRUE(Logged(NCCL_LOG_WARN, "Invalid key size"));
  }
  for (int keySize : {1, 8}) EXPECT_EQ(ncclSuccess, ncclIntruAddressMapFind_untyped(map, keySize, 0, 8, 1, &found));
  EXPECT_EQ(0, map_.base.count);
}

}  // namespace
