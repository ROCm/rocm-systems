/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/net.cc: the NCCL_NET_PLUGIN list, the v12-to-v6 version probe, per-comm
// plugin init/assign/finalize, and the device-count and GDR queries. Library open/close and the version getters
// are seamed in this TU; logging, env, params and rcclUseAinic come from the shared fakes. The per-version shims
// and plugin_open.cc have their own test TUs.

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../common/LogCapture.hpp"
#include "../common/ProcessIsolatedTestRunner.hpp"
#include "ExpectedCrash.h"
#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/transport_stubs.h"
#include "nccl.h"
#include "comm.h"
#include "debug.h"
#include "net.h"
#include "net_telemetry.h"
#include "plugin.h"

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

namespace {
int g_telemetryInitResult = 0;
}  // namespace

// No shared owner for these two.
int rcclTelemetryInit(void) { return g_telemetryInitResult; }
ncclResult_t ncclProfilerCallback(void**, int, void*, int64_t, void*) { return ncclSuccess; }

namespace {

struct NetScript {
  ncclResult_t initResult = ncclSuccess;
  ncclResult_t devicesResult = ncclSuccess;
  int ndev = 2;
  ncclResult_t propsResult = ncclSuccess;
  ncclNetDeviceType deviceType = NCCL_NET_DEVICE_HOST;
  int deviceVersion = 0;
  unsigned cudaDevMask = 0;  // bit d: device d reports NCCL_PTR_CUDA
  ncclResult_t finalizeResult = ncclSuccess;
  int initCalls = 0;
  int devicesCalls = 0;
  int propsCalls = 0;
  int finalizeCalls = 0;
  uint64_t lastCommId = 0;
  int lastTrafficClass = 0;
  ncclProfilerCallback_t lastProfFn = nullptr;
  void* lastFinalizeCtx = nullptr;
};

struct CollScript {
  ncclResult_t initResult = ncclSuccess;
  ncclResult_t devicesResult = ncclSuccess;
  int ndev = 1;
  ncclResult_t finalizeResult = ncclSuccess;
  int initCalls = 0;
  int devicesCalls = 0;
  int finalizeCalls = 0;
};

enum { kExtA, kExtB, kIb, kSocket, kIbCast, kNumNets };
NetScript g_net[kNumNets];
CollScript g_coll[2];

// Plugin entry points carry no instance pointer, so each fake plugin gets its own instantiation.
template <int Id>
struct FakeNet {
  static ncclResult_t Init(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t,
                           ncclProfilerCallback_t profFn) {
    NetScript& s = g_net[Id];
    ++s.initCalls;
    s.lastCommId = commId;
    s.lastTrafficClass = config->trafficClass;
    s.lastProfFn = profFn;
    if (s.initResult == ncclSuccess) *ctx = &s;
    return s.initResult;
  }
  static ncclResult_t Devices(int* ndev) {
    NetScript& s = g_net[Id];
    ++s.devicesCalls;
    *ndev = s.ndev;
    return s.devicesResult;
  }
  static ncclResult_t GetProperties(int dev, ncclNetProperties_t* props) {
    NetScript& s = g_net[Id];
    ++s.propsCalls;
    std::memset(props, 0, sizeof(*props));
    props->netDeviceType = s.deviceType;
    props->netDeviceVersion = s.deviceVersion;
    props->ptrSupport = NCCL_PTR_HOST | (((s.cudaDevMask >> dev) & 1) ? NCCL_PTR_CUDA : 0);
    return s.propsResult;
  }
  static ncclResult_t Finalize(void* ctx) {
    NetScript& s = g_net[Id];
    ++s.finalizeCalls;
    s.lastFinalizeCtx = ctx;
    return s.finalizeResult;
  }
  static ncclNet_t Make(const char* name) {
    ncclNet_t net{};
    net.name = name;
    net.init = Init;
    net.devices = Devices;
    net.getProperties = GetProperties;
    net.finalize = Finalize;
    return net;
  }
};

template <int Id>
struct FakeCollNet {
  static ncclResult_t Init(void** ctx, uint64_t, ncclDebugLogger_t) {
    CollScript& s = g_coll[Id];
    ++s.initCalls;
    if (s.initResult == ncclSuccess) *ctx = &s;
    return s.initResult;
  }
  static ncclResult_t Devices(int* ndev) {
    CollScript& s = g_coll[Id];
    ++s.devicesCalls;
    *ndev = s.ndev;
    return s.devicesResult;
  }
  static ncclResult_t Finalize(void*) {
    ++g_coll[Id].finalizeCalls;
    return g_coll[Id].finalizeResult;
  }
  static ncclCollNet_t Make(const char* name) {
    ncclCollNet_t net{};
    net.name = name;
    net.init = Init;
    net.devices = Devices;
    net.finalize = Finalize;
    return net;
  }
};

ncclNet_t g_extNet[2] = {FakeNet<kExtA>::Make("ExtA"), FakeNet<kExtB>::Make("ExtB")};
ncclCollNet_t g_extCollNet[2] = {FakeCollNet<0>::Make("CollA"), FakeCollNet<1>::Make("CollB")};

// What a fake plugin library exports: the ncclNetPlugin_vN / ncclCollNetPlugin_vN versions it has (0 = none).
struct FakeLib {
  int netVersion = 12;
  int collNetVersion = 0;
  int slot = 0;  // which g_extNet / g_extCollNet it exports, and its handle
};
std::map<std::string, FakeLib> g_libs;
std::vector<std::string> g_opened;
std::vector<std::pair<void*, ncclPluginType>> g_closed;
ncclResult_t g_closeResult = ncclSuccess;
std::vector<int> g_netProbes;
std::vector<int> g_collNetProbes;

void* const kLibHandles[2] = {reinterpret_cast<void*>(0xa0), reinterpret_cast<void*>(0xb0)};

const FakeLib& LibFor(void* handle) {
  for (auto& lib : g_libs)
    if (kLibHandles[lib.second.slot] == handle) return lib.second;
  static const FakeLib none{0, 0, 0};
  return none;
}

}  // namespace

void* FakeOpenNetPluginLib(const char* name) {
  g_opened.push_back(name);
  auto it = g_libs.find(name);
  return it == g_libs.end() ? nullptr : kLibHandles[it->second.slot];
}

ncclResult_t FakeClosePluginLib(void* handle, ncclPluginType type) {
  g_closed.push_back({handle, type});
  return g_closeResult;
}

#include "fakes/param_redirect.h"
#define ncclOpenNetPluginLib FakeOpenNetPluginLib
#define ncclClosePluginLib FakeClosePluginLib
// The GPU-registration probe after the AMD-only break is dead on HIP; keep its allocations off the HIP runtime.
#undef ncclCudaMalloc
#define ncclCudaMalloc(...) ncclInternalError
#define ncclCudaFree(...) ncclInternalError

#include PLUGIN_NET_CC_PATH

#undef ncclCudaFree
#undef ncclCudaMalloc
#undef ncclClosePluginLib
#undef ncclOpenNetPluginLib

ncclNet_t ncclNetIb = FakeNet<kIb>::Make("IB");
ncclNet_t ncclNetSocket = FakeNet<kSocket>::Make("Socket");
ncclNet_t netIbCast = FakeNet<kIbCast>::Make("IB-CAST");

namespace {

template <int Index>
ncclNet_t* FakeGetNet(void* lib) {
  g_netProbes.push_back(ncclNetVersion[Index]);
  return LibFor(lib).netVersion == ncclNetVersion[Index] ? &g_extNet[LibFor(lib).slot] : nullptr;
}

template <int Index>
ncclCollNet_t* FakeGetCollNet(void* lib) {
  g_collNetProbes.push_back(ncclNetVersion[Index]);
  return LibFor(lib).collNetVersion == ncclNetVersion[Index] ? &g_extCollNet[LibFor(lib).slot] : nullptr;
}

template <int... I>
void InstallFakeGetters(std::integer_sequence<int, I...>) {
  ((getNcclNet[I] = FakeGetNet<I>), ...);
  ((getNcclCollNet[I] = FakeGetCollNet<I>), ...);
}

class PluginNetMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    // ncclNetInit must not reparse the env behind a test's back; tests call initPluginLibsOnceFunc() directly.
    std::call_once(initPluginLibsOnceFlag, [] {});
    std::memcpy(savedNet_, getNcclNet, sizeof(savedNet_));
    std::memcpy(savedCollNet_, getNcclCollNet, sizeof(savedCollNet_));
    InstallFakeGetters(std::make_integer_sequence<int, NCCL_NET_VERSION_COUNT>{});
    savedLibPath_ = ncclPluginLibPaths[ncclPluginTypeNet];
    ncclPluginLibPaths[ncclPluginTypeNet] = nullptr;
    std::memset(netPluginLibs, 0, sizeof(netPluginLibs));
    pluginCount = 0;
    for (auto& s : g_net) s = NetScript{};
    for (auto& s : g_coll) s = CollScript{};
    ResetEnvFakes();
    ResetTransportStubs();
    g_telemetryInitResult = 0;
    g_libs.clear();
    g_opened.clear();
    g_closed.clear();
    g_closeResult = ncclSuccess;
    g_netProbes.clear();
    g_collNetProbes.clear();
    comm_ = std::make_unique<ncclComm>();
    comm_->config.netName = nullptr;
    comm_->config.trafficClass = NCCL_CONFIG_UNDEF_INT;
    comm_->commHash = 0xc0ffee;
  }

  void TearDown() override {
    std::memcpy(getNcclNet, savedNet_, sizeof(savedNet_));
    std::memcpy(getNcclCollNet, savedCollNet_, sizeof(savedCollNet_));
    ncclPluginLibPaths[ncclPluginTypeNet] = savedLibPath_;
    std::memset(netPluginLibs, 0, sizeof(netPluginLibs));
    pluginCount = 0;
  }

  // Rebuilds the plugin list as ncclNetInit's call_once would from the current env.
  void BuildList(const char* netPlugin) {
    if (netPlugin) SetMicroEnv("NCCL_NET_PLUGIN", netPlugin);
    initPluginLibsOnceFunc();
  }

  netPluginLib_t& Lib(int i) { return netPluginLibs[i]; }

  getNcclNet_t* savedNet_[NCCL_NET_VERSION_COUNT];
  getNcclCollNet_t* savedCollNet_[NCCL_NET_VERSION_COUNT];
  char* savedLibPath_ = nullptr;
  std::unique_ptr<ncclComm> comm_;
  int64_t netPluginRefCount_ = 0;  // NCCL_NET_PLUGIN_REF_COUNT
  ScopedHook<int64_t(const char*, int64_t)> params_{g_loadParam, [this](const char* env, int64_t deftVal) {
    return std::strcmp(env, "NET_PLUGIN_REF_COUNT") == 0 ? netPluginRefCount_ : deftVal;
  }};
  RcclUnitTesting::ScopedDebugLogging logAll_{-1, ~uint64_t(0)};  // nccl_fakes.cc mutes INFO by default
};

// Plugin list

TEST_F(PluginNetMicrotest, ListDefaultsToLibrcclNetThenIbAndSocket) {
  netPluginRefCount_ = 3;
  BuildList(nullptr);

  ASSERT_EQ(pluginCount, 3);
  EXPECT_STREQ(Lib(0).name, "librccl-net.so");
  EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateLoadReady);
  EXPECT_EQ(Lib(0).ncclNetPluginRefCount, 3);
  EXPECT_EQ(Lib(0).ncclNet, nullptr);
  for (int i : {1, 2}) {
    EXPECT_EQ(Lib(i).ncclNet, i == 1 ? &ncclNetIb : &ncclNetSocket);
    EXPECT_EQ(Lib(i).ncclNetVer, 0);
    EXPECT_EQ(Lib(i).ncclNetPluginState, ncclNetPluginStateInitReady);
    EXPECT_EQ(Lib(i).ncclCollNetPluginState, ncclNetPluginStateDisabled);
    EXPECT_EQ(Lib(i).ncclNetPluginRefCount, 0);
    EXPECT_EQ(Lib(i).netPhysDevs, NCCL_UNDEF_DEV_COUNT);
    EXPECT_EQ(Lib(i).netVirtDevs, NCCL_UNDEF_DEV_COUNT);
    EXPECT_EQ(Lib(i).collNetPhysDevs, NCCL_UNDEF_DEV_COUNT);
    EXPECT_EQ(Lib(i).collNetVirtDevs, NCCL_UNDEF_DEV_COUNT);
  }
}

TEST_F(PluginNetMicrotest, ListKeepsNetPluginOrderAheadOfInternals) {
  netPluginRefCount_ = 1;
  BuildList("first,second");

  ASSERT_EQ(pluginCount, 4);
  EXPECT_STREQ(Lib(0).name, "first");
  EXPECT_STREQ(Lib(1).name, "second");
  EXPECT_EQ(Lib(1).ncclNetPluginState, ncclNetPluginStateLoadReady);
  EXPECT_EQ(Lib(1).ncclNetPluginRefCount, 1);
  EXPECT_EQ(Lib(2).ncclNet, &ncclNetIb);
  EXPECT_EQ(Lib(3).ncclNet, &ncclNetSocket);
}

TEST_F(PluginNetMicrotest, ListNoneOrEmptyLeavesOnlyInternals) {
  for (const char* value : {"none", "NoNe", ""}) {
    std::string log = CaptureLog([&] { BuildList(value); });
    ASSERT_EQ(pluginCount, 2) << value;
    EXPECT_EQ(Lib(0).ncclNet, &ncclNetIb) << value;
    EXPECT_TRUE(LogHas(log, (std::string("NCCL_NET_PLUGIN set by environment to ") + value).c_str())) << log;
  }
}

// NCCL_NET_MAX_PLUGINS less the two internals leaves room for 14, but the log claims 13.
// Pins AICOMRCCL-2986; a fix flips this pin.
TEST_F(PluginNetMicrotest, List_KnownBug_KeepsFourteenExternalsButLogsThirteen) {
  constexpr int kExternals = NCCL_NET_MAX_PLUGINS - 2;
  std::string list;
  for (int i = 0; i < NCCL_NET_MAX_PLUGINS; i++) list += (i ? ",p" : "p") + std::to_string(i);
  std::string log = CaptureLog([&] { BuildList(list.c_str()); });

  ASSERT_EQ(pluginCount, NCCL_NET_MAX_PLUGINS);
  EXPECT_EQ(std::string(Lib(kExternals - 1).name), "p" + std::to_string(kExternals - 1));
  EXPECT_EQ(Lib(kExternals).ncclNet, &ncclNetIb);
  EXPECT_EQ(Lib(kExternals + 1).ncclNet, &ncclNetSocket);
  EXPECT_TRUE(LogHas(log, ("NCCL_NET_PLUGIN list contains more than " + std::to_string(kExternals - 1) +
                           " plugins, ignoring the rest").c_str()))
      << log;
}

TEST_F(PluginNetMicrotest, ListSkipsNamesThatDoNotFitWithTheirTerminator) {
  const std::string fits(MAX_STR_LEN - 1, 'a');
  const std::string tooLong(MAX_STR_LEN, 'b');
  std::string log = CaptureLog([&] { BuildList((tooLong + "," + fits).c_str()); });

  ASSERT_EQ(pluginCount, 3);
  EXPECT_EQ(std::string(Lib(0).name), fits);
  EXPECT_TRUE(LogHas(log, ("longer than " + std::to_string(MAX_STR_LEN) + " characters, ignoring it").c_str()))
      << log;
}

TEST_F(PluginNetMicrotest, ListPicksIbCastForAinicOrNcclNetIbCast) {
  struct Case {
    const char* net;
    const char* netPlugin;
    bool ainic;
    ncclNet_t* expected;
  } cases[] = {
      {nullptr, nullptr, true, &netIbCast},          {nullptr, nullptr, false, &ncclNetIb},
      {"IB-CAST", nullptr, false, &netIbCast},       {"rocm-ib", nullptr, false, &netIbCast},
      {"IB", nullptr, true, &ncclNetIb},             {"IB-CAST", "ext", false, &ncclNetIb},
      {nullptr, "ext", true, &ncclNetIb},            {nullptr, "none", true, &netIbCast},
  };
  for (const Case& c : cases) {
    ClearMicroEnv();
    if (c.net) SetMicroEnv("NCCL_NET", c.net);
    g_rcclUseAinicValue = c.ainic;
    BuildList(c.netPlugin);
    EXPECT_EQ(Lib(pluginCount - 2).ncclNet, c.expected)
        << "NCCL_NET=" << (c.net ? c.net : "-") << " NCCL_NET_PLUGIN=" << (c.netPlugin ? c.netPlugin : "-")
        << " ainic=" << c.ainic;
    EXPECT_EQ(Lib(pluginCount - 1).ncclNet, &ncclNetSocket);
  }
}

// Library load and version probe

TEST_F(PluginNetMicrotest, LoadOpenFailureMarksBothPluginsLoadFailed) {
  BuildList("missing");
  ASSERT_EQ(ncclNetPluginLoad(&Lib(0)), ncclSuccess);

  EXPECT_EQ(g_opened, std::vector<std::string>{"missing"});
  EXPECT_EQ(Lib(0).dlHandle, nullptr);
  EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateLoadFailed);
  EXPECT_EQ(Lib(0).ncclCollNetPluginState, ncclNetPluginStateLoadFailed);
  EXPECT_TRUE(g_netProbes.empty());
  EXPECT_TRUE(g_closed.empty());
}

TEST_F(PluginNetMicrotest, LoadProbesNewestVersionFirstAndStopsAtTheFirstHit) {
  g_libs["ext"] = {9, 7};
  BuildList("ext");
  ASSERT_EQ(ncclNetPluginLoad(&Lib(0)), ncclSuccess);

  EXPECT_EQ(g_netProbes, (std::vector<int>{12, 11, 10, 9}));
  EXPECT_EQ(g_collNetProbes, (std::vector<int>{12, 11, 10, 9, 8, 7}));
  EXPECT_EQ(Lib(0).dlHandle, kLibHandles[0]);
  EXPECT_EQ(Lib(0).ncclNet, &g_extNet[0]);
  EXPECT_EQ(Lib(0).ncclNetVer, 9);
  EXPECT_EQ(Lib(0).ncclCollNet, &g_extCollNet[0]);
  EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateInitReady);
  EXPECT_EQ(Lib(0).ncclCollNetPluginState, ncclNetPluginStateInitReady);
  EXPECT_TRUE(g_closed.empty());
}

TEST_F(PluginNetMicrotest, LoadFindsTheOldestSupportedVersion) {
  g_libs["ext"] = {6, 6};
  BuildList("ext");
  ASSERT_EQ(ncclNetPluginLoad(&Lib(0)), ncclSuccess);

  EXPECT_EQ(g_netProbes, (std::vector<int>{12, 11, 10, 9, 8, 7, 6}));
  EXPECT_EQ(Lib(0).ncclNetVer, 6);
  EXPECT_EQ(Lib(0).ncclCollNet, &g_extCollNet[0]);
}

TEST_F(PluginNetMicrotest, LoadWithoutCollNetLeavesNetUsable) {
  g_libs["ext"] = {12, 0};
  ncclPluginLibPaths[ncclPluginTypeNet] = const_cast<char*>("/opt/lib/ext.so");
  BuildList("ext");
  std::string log = CaptureLog([&] { ASSERT_EQ(ncclNetPluginLoad(&Lib(0)), ncclSuccess); });

  EXPECT_EQ(g_netProbes, std::vector<int>{12});
  EXPECT_EQ(Lib(0).ncclNetVer, 12);
  EXPECT_EQ(Lib(0).ncclCollNet, nullptr);
  EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateInitReady);
  EXPECT_EQ(Lib(0).ncclCollNetPluginState, ncclNetPluginStateLoadFailed);
  EXPECT_TRUE(LogHas(log, "INFO Successfully loaded external network plugin /opt/lib/ext.so")) << log;
}

TEST_F(PluginNetMicrotest, LoadWithNoKnownVersionClosesTheLibrary) {
  g_libs["ext"] = {5, 0};
  std::string log = CaptureLog([&] {
    initPluginLibsOnceFunc();  // NCCL_NET_PLUGIN unset: the unsupported library is only INFO-logged
    std::strcpy(Lib(0).name, "ext");
    ASSERT_EQ(ncclNetPluginLoad(&Lib(0)), ncclSuccess);
  });

  EXPECT_EQ(g_netProbes.size(), size_t(NCCL_NET_VERSION_COUNT));
  ASSERT_EQ(g_closed.size(), 1u);
  EXPECT_EQ(g_closed[0].first, kLibHandles[0]);
  EXPECT_EQ(g_closed[0].second, ncclPluginTypeNet);
  EXPECT_EQ(Lib(0).dlHandle, nullptr);
  EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateLoadFailed);
  EXPECT_EQ(Lib(0).ncclCollNetPluginState, ncclNetPluginStateLoadFailed);
  EXPECT_TRUE(LogHas(log, "INFO External network plugin ext is unsupported")) << log;
}

// An explicitly requested plugin that is unusable is an ATTN, naming the resolved path when there is one.
TEST_F(PluginNetMicrotest, LoadWithNoKnownVersionAttnsWhenNetPluginIsSet) {
  g_libs["ext"] = {5, 0};
  BuildList("ext");
  std::string log = CaptureLog([&] { ASSERT_EQ(ncclNetPluginLoad(&Lib(0)), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "ATTN External network plugin ext is unsupported")) << log;

  Lib(0).ncclNetPluginState = ncclNetPluginStateLoadReady;
  ncclPluginLibPaths[ncclPluginTypeNet] = const_cast<char*>("/opt/lib/ext.so");
  log = CaptureLog([&] { ASSERT_EQ(ncclNetPluginLoad(&Lib(0)), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "ATTN External network plugin /opt/lib/ext.so is unsupported")) << log;
}

TEST_F(PluginNetMicrotest, LoadPropagatesACloseFailure) {
  g_libs["ext"] = {5, 0};
  g_closeResult = ncclSystemError;
  BuildList("ext");
  EXPECT_EQ(ncclNetPluginLoad(&Lib(0)), ncclSystemError);
}

// Device version check

TEST_F(PluginNetMicrotest, CheckDeviceVersionAcceptsHostAndMatchingUnpack) {
  ncclNet_t net = FakeNet<kExtA>::Make("ExtA");
  EXPECT_EQ(ncclNetCheckDeviceVersion(comm_.get(), &net, 0), ncclSuccess);

  g_net[kExtA].deviceType = NCCL_NET_DEVICE_UNPACK;
  g_net[kExtA].deviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
  std::string log = CaptureLog([&] { EXPECT_EQ(ncclNetCheckDeviceVersion(comm_.get(), &net, 0), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "Using NCCL_NET_DEVICE_UNPACK net plugin version 7")) << log;
}

TEST_F(PluginNetMicrotest, CheckDeviceVersionRejectsOtherUnpackVersionsAndUnknownTypes) {
  ncclNet_t net = FakeNet<kExtA>::Make("ExtA");
  g_net[kExtA].deviceType = NCCL_NET_DEVICE_UNPACK;
  g_net[kExtA].deviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION + 1;
  std::string log =
      CaptureLog([&] { EXPECT_EQ(ncclNetCheckDeviceVersion(comm_.get(), &net, 0), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "WARN NCCL_DEVICE_UNPACK plugin has incompatible version 8")) << log;

  g_net[kExtA].deviceType = static_cast<ncclNetDeviceType>(0x42);
  log = CaptureLog([&] { EXPECT_EQ(ncclNetCheckDeviceVersion(comm_.get(), &net, 0), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "WARN Unknown device code index 66")) << log;

  g_net[kExtA].propsResult = ncclSystemError;
  EXPECT_EQ(ncclNetCheckDeviceVersion(comm_.get(), &net, 0), ncclSystemError);
}

// Per-comm plugin init

TEST_F(PluginNetMicrotest, PluginInitProbesDevicesOnFirstInit) {
  BuildList(nullptr);
  comm_->config.trafficClass = 5;
  ASSERT_EQ(ncclNetPluginInit(comm_.get(), &Lib(1)), ncclSuccess);

  const NetScript& ib = g_net[kIb];
  EXPECT_EQ(ib.initCalls, 1);
  EXPECT_EQ(ib.lastCommId, 0xc0ffeeu);
  EXPECT_EQ(ib.lastTrafficClass, 5);
  EXPECT_EQ(ib.lastProfFn, &ncclProfilerCallback);
  EXPECT_EQ(comm_->netContext, &g_net[kIb]);
  EXPECT_EQ(ib.devicesCalls, 1);
  EXPECT_EQ(Lib(1).netPhysDevs, 2);
  EXPECT_EQ(Lib(1).netVirtDevs, NCCL_UNDEF_DEV_COUNT);
  EXPECT_EQ(Lib(1).ncclNetPluginState, ncclNetPluginStateEnabled);
}

TEST_F(PluginNetMicrotest, PluginInitMapsAnUnsetTrafficClassToTheNetUndef) {
  BuildList(nullptr);
  ASSERT_EQ(ncclNetPluginInit(comm_.get(), &Lib(1)), ncclSuccess);
  EXPECT_EQ(g_net[kIb].lastTrafficClass, NCCL_NET_TRAFFIC_CLASS_UNDEF);
}

// Every comm re-inits for its own context, but devices are only counted once.
TEST_F(PluginNetMicrotest, PluginInitOnAnEnabledPluginSkipsDeviceProbe) {
  BuildList(nullptr);
  ASSERT_EQ(ncclNetPluginInit(comm_.get(), &Lib(1)), ncclSuccess);
  g_net[kIb].ndev = 7;
  ASSERT_EQ(ncclNetPluginInit(comm_.get(), &Lib(1)), ncclSuccess);

  EXPECT_EQ(g_net[kIb].initCalls, 2);
  EXPECT_EQ(g_net[kIb].devicesCalls, 1);
  EXPECT_EQ(Lib(1).netPhysDevs, 2);
}

TEST_F(PluginNetMicrotest, PluginInitFailureDisablesWithoutFinalize) {
  BuildList(nullptr);
  g_net[kIb].initResult = ncclSystemError;
  std::string log = CaptureLog([&] { ASSERT_EQ(ncclNetPluginInit(comm_.get(), &Lib(1)), ncclSuccess); });

  EXPECT_EQ(g_net[kIb].finalizeCalls, 0);
  EXPECT_EQ(g_net[kIb].devicesCalls, 0);
  EXPECT_EQ(Lib(1).ncclNetPluginState, ncclNetPluginStateDisabled);
  EXPECT_EQ(Lib(1).ncclCollNetPluginState, ncclNetPluginStateDisabled);
  EXPECT_TRUE(LogHas(log, "ATTN Failed to initialize NET plugin IB")) << log;
}

TEST_F(PluginNetMicrotest, PluginInitWithoutDevicesFinalizesAndDisables) {
  for (bool devicesFails : {true, false}) {
    BuildList(nullptr);
    g_net[kIb] = NetScript{};
    if (devicesFails) g_net[kIb].devicesResult = ncclSystemError;
    else g_net[kIb].ndev = 0;
    Lib(1).collNetPhysDevs = Lib(1).collNetVirtDevs = 4;
    ASSERT_EQ(ncclNetPluginInit(comm_.get(), &Lib(1)), ncclSuccess);

    EXPECT_EQ(g_net[kIb].finalizeCalls, 1) << devicesFails;
    EXPECT_EQ(g_net[kIb].lastFinalizeCtx, &g_net[kIb]);
    EXPECT_EQ(Lib(1).ncclNetPluginState, ncclNetPluginStateDisabled);
    EXPECT_EQ(Lib(1).ncclCollNetPluginState, ncclNetPluginStateDisabled);
    EXPECT_EQ(Lib(1).netPhysDevs, NCCL_UNDEF_DEV_COUNT);
    EXPECT_EQ(Lib(1).netVirtDevs, NCCL_UNDEF_DEV_COUNT);
    EXPECT_EQ(Lib(1).collNetPhysDevs, NCCL_UNDEF_DEV_COUNT);
    EXPECT_EQ(Lib(1).collNetVirtDevs, NCCL_UNDEF_DEV_COUNT);
  }
}

TEST_F(PluginNetMicrotest, PluginInitEnablesCollNetWithDevices) {
  g_libs["ext"] = {12, 12};
  g_coll[0].ndev = 3;
  BuildList("ext");
  ASSERT_EQ(ncclNetPluginLoad(&Lib(0)), ncclSuccess);
  ASSERT_EQ(ncclNetPluginInit(comm_.get(), &Lib(0)), ncclSuccess);

  EXPECT_EQ(g_coll[0].initCalls, 1);
  EXPECT_EQ(comm_->collNetContext, &g_coll[0]);
  EXPECT_EQ(Lib(0).ncclCollNetPluginState, ncclNetPluginStateEnabled);
  EXPECT_EQ(Lib(0).collNetPhysDevs, 3);
  EXPECT_EQ(Lib(0).collNetVirtDevs, NCCL_UNDEF_DEV_COUNT);

  ASSERT_EQ(ncclNetPluginInit(comm_.get(), &Lib(0)), ncclSuccess);
  EXPECT_EQ(g_coll[0].initCalls, 2);
  EXPECT_EQ(g_coll[0].devicesCalls, 1);
}

TEST_F(PluginNetMicrotest, PluginInitCollNetFailuresOnlyDisableCollNet) {
  struct Case {
    ncclResult_t initResult;
    ncclResult_t devicesResult;
    int ndev;
    int devicesCalls;
  } cases[] = {{ncclSystemError, ncclSuccess, 1, 0}, {ncclSuccess, ncclSystemError, 1, 1}, {ncclSuccess, ncclSuccess, 0, 1}};
  for (const Case& c : cases) {
    g_libs["ext"] = {12, 12};
    BuildList("ext");
    g_coll[0] = CollScript{};
    g_coll[0].initResult = c.initResult;
    g_coll[0].devicesResult = c.devicesResult;
    g_coll[0].ndev = c.ndev;
    ASSERT_EQ(ncclNetPluginLoad(&Lib(0)), ncclSuccess);
    ASSERT_EQ(ncclNetPluginInit(comm_.get(), &Lib(0)), ncclSuccess);

    EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateEnabled);
    EXPECT_EQ(Lib(0).ncclCollNetPluginState, ncclNetPluginStateDisabled);
    EXPECT_EQ(g_coll[0].devicesCalls, c.devicesCalls);
  }
}

// ncclNetInit

TEST_F(PluginNetMicrotest, NetInitAssignsTheFirstWorkingExternalAndDisablesTheRest) {
  g_libs["a"] = {11, 11};
  g_libs["b"] = {12, 0, 1};
  SetMicroEnv("NCCL_NET_PLUGIN", "a,b,c");
  initPluginLibsOnceFunc();
  comm_->ncclCollNet = &g_extCollNet[1];
  std::string log = CaptureLog([&] { ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess); });

  EXPECT_EQ(comm_->ncclNet, &g_extNet[0]);
  EXPECT_EQ(comm_->ncclNetVer, 11);
  EXPECT_EQ(comm_->netPluginIndex, 0);
  EXPECT_EQ(comm_->ncclCollNet, &g_extCollNet[0]);
  EXPECT_EQ(Lib(0).ncclNetPluginRefCount, 1);
  EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateEnabled);
  EXPECT_EQ(Lib(1).ncclNetPluginState, ncclNetPluginStateDisabled);
  EXPECT_EQ(Lib(2).ncclNetPluginState, ncclNetPluginStateDisabled);
  EXPECT_EQ(Lib(3).ncclNetPluginState, ncclNetPluginStateInitReady);  // internals stay available
  EXPECT_EQ(g_opened, std::vector<std::string>{"a"});
  EXPECT_EQ(g_net[kIb].initCalls, 0);
  EXPECT_TRUE(LogHas(log, "Disabling external plugins: b, c")) << log;
  EXPECT_TRUE(LogHas(log, "Assigned NET plugin ExtA to comm")) << log;
}

TEST_F(PluginNetMicrotest, NetInitClearsAStaleCollNetWhenTheAssignedPluginHasNone) {
  BuildList("missing");
  comm_->ncclCollNet = &g_extCollNet[1];
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ncclNet, &ncclNetIb);
  EXPECT_EQ(comm_->netPluginIndex, 1);
  EXPECT_EQ(comm_->ncclCollNet, nullptr);
}

// The disable sweep only runs for an external winner: falling back to an internal leaves failed externals as is.
TEST_F(PluginNetMicrotest, NetInitFallsBackToIbWithoutTouchingFailedExternals) {
  g_libs["b"] = {12, 0, 1};
  g_net[kExtB].initResult = ncclSystemError;
  BuildList("a,b");
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ncclNet, &ncclNetIb);
  EXPECT_EQ(comm_->netPluginIndex, 2);
  EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateLoadFailed);
  EXPECT_EQ(Lib(1).ncclNetPluginState, ncclNetPluginStateDisabled);
  EXPECT_EQ(Lib(2).ncclNetPluginRefCount, 1);
}

TEST_F(PluginNetMicrotest, NetInitHonoursNetNameCaseInsensitively) {
  BuildList("none");
  comm_->config.netName = "socket";
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ncclNet, &ncclNetSocket);
  EXPECT_EQ(comm_->netPluginIndex, 1);
  EXPECT_EQ(g_net[kIb].initCalls, 0);
}

TEST_F(PluginNetMicrotest, NetInitWithoutAUsablePluginIsInvalidUsage) {
  BuildList("none");
  comm_->config.netName = "nope";
  std::string log = CaptureLog([&] { EXPECT_EQ(ncclNetInit(comm_.get()), ncclInvalidUsage); });
  EXPECT_TRUE(LogHas(log, "WARN Failed to initialize any NET plugin")) << log;
}

TEST_F(PluginNetMicrotest, NetInitContinuesWhenTelemetryCannotStart) {
  BuildList("none");
  g_telemetryInitResult = -1;
  std::string log = CaptureLog([&] { EXPECT_EQ(ncclNetInit(comm_.get()), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "Telemetry was requested but could not start")) << log;
}

TEST_F(PluginNetMicrotest, NetInitPropagatesALoadFailure) {
  g_libs["ext"] = {5, 0};
  g_closeResult = ncclSystemError;
  BuildList("ext");
  EXPECT_EQ(ncclNetInit(comm_.get()), ncclSystemError);
}

TEST_F(PluginNetMicrotest, NetInitSecondCommReusesTheEnabledPlugin) {
  g_libs["ext"] = {12, 0};
  BuildList("ext");
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);
  auto other = std::make_unique<ncclComm>();
  other->config.trafficClass = NCCL_CONFIG_UNDEF_INT;
  ASSERT_EQ(ncclNetInit(other.get()), ncclSuccess);

  EXPECT_EQ(other->ncclNet, &g_extNet[0]);
  EXPECT_EQ(g_opened.size(), 1u);
  EXPECT_EQ(g_net[kExtA].initCalls, 2);
  EXPECT_EQ(g_net[kExtA].devicesCalls, 1);
  EXPECT_EQ(Lib(0).ncclNetPluginRefCount, 2);
}

// A rejected plugin's finalize drops a refcount it never took, so the library is never unloaded.
// Pins AICOMRCCL-2983; a fix flips this pin.
TEST_F(PluginNetMicrotest, NetInit_KnownBug_RejectedPluginLeavesANegativeRefCount) {
  g_libs["ext"] = {12, 0};
  g_net[kExtA].deviceType = static_cast<ncclNetDeviceType>(0x42);
  BuildList("ext");
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ncclNet, &ncclNetIb);
  EXPECT_EQ(g_net[kExtA].finalizeCalls, 1);
  EXPECT_EQ(Lib(0).ncclNetPluginRefCount, -1);
  EXPECT_EQ(Lib(0).dlHandle, kLibHandles[0]);
  EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateEnabled);
  EXPECT_TRUE(g_closed.empty());
}

// Finalize

TEST_F(PluginNetMicrotest, FinalizeUnloadsAnExternalPluginAtZeroRefCount) {
  g_libs["ext"] = {12, 12};
  BuildList("ext");
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);
  std::string log = CaptureLog([&] { ASSERT_EQ(ncclNetFinalize(comm_.get()), ncclSuccess); });

  EXPECT_EQ(g_net[kExtA].finalizeCalls, 1);
  EXPECT_EQ(g_net[kExtA].lastFinalizeCtx, &g_net[kExtA]);
  EXPECT_EQ(g_coll[0].finalizeCalls, 1);
  ASSERT_EQ(g_closed.size(), 1u);
  EXPECT_EQ(g_closed[0].first, kLibHandles[0]);
  EXPECT_STREQ(Lib(0).name, "ext");
  EXPECT_EQ(Lib(0).dlHandle, nullptr);
  EXPECT_EQ(Lib(0).ncclNet, nullptr);
  EXPECT_EQ(Lib(0).ncclNetVer, 0);
  EXPECT_EQ(Lib(0).ncclCollNet, nullptr);
  EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateLoadReady);
  EXPECT_EQ(Lib(0).ncclCollNetPluginState, ncclNetPluginStateLoadReady);
  EXPECT_EQ(Lib(0).ncclNetPluginRefCount, 0);
  EXPECT_EQ(Lib(0).netPhysDevs, NCCL_UNDEF_DEV_COUNT);
  EXPECT_EQ(Lib(0).netVirtDevs, NCCL_UNDEF_DEV_COUNT);
  EXPECT_EQ(Lib(0).collNetPhysDevs, NCCL_UNDEF_DEV_COUNT);
  EXPECT_EQ(Lib(0).collNetVirtDevs, NCCL_UNDEF_DEV_COUNT);
  EXPECT_TRUE(LogHas(log, "Unloading plugin ext")) << log;

  // The preserved name lets the next comm reload it.
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);
  EXPECT_EQ(g_opened, (std::vector<std::string>{"ext", "ext"}));
  EXPECT_EQ(comm_->ncclNet, &g_extNet[0]);
}

TEST_F(PluginNetMicrotest, FinalizeKeepsTheLibraryWhileNetPluginRefCountPinsIt) {
  g_libs["ext"] = {12, 0};
  netPluginRefCount_ = 1;
  BuildList("ext");
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);
  ASSERT_EQ(ncclNetFinalize(comm_.get()), ncclSuccess);

  EXPECT_EQ(Lib(0).ncclNetPluginRefCount, 1);
  EXPECT_EQ(Lib(0).dlHandle, kLibHandles[0]);
  EXPECT_TRUE(g_closed.empty());
}

TEST_F(PluginNetMicrotest, FinalizeNeverUnloadsAnInternalPlugin) {
  BuildList("none");
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);
  ASSERT_EQ(ncclNetFinalize(comm_.get()), ncclSuccess);

  EXPECT_EQ(g_net[kIb].finalizeCalls, 1);
  EXPECT_EQ(Lib(0).ncclNetPluginRefCount, 0);
  EXPECT_EQ(Lib(0).ncclNet, &ncclNetIb);
  EXPECT_EQ(Lib(0).ncclNetPluginState, ncclNetPluginStateEnabled);
}

TEST_F(PluginNetMicrotest, FinalizeSkipsACollNetThatIsNotEnabled) {
  g_libs["ext"] = {12, 12};
  g_coll[0].ndev = 0;
  BuildList("ext");
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->ncclCollNet, nullptr);
  ASSERT_EQ(ncclNetFinalize(comm_.get()), ncclSuccess);
  EXPECT_EQ(g_coll[0].finalizeCalls, 0);
}

// A failing net or collnet finalize returns before the refcount drop, so the comm's reference is never released
// and the library stays loaded for good.
// Pins AICOMRCCL-3079; a fix flips this pin.
TEST_F(PluginNetMicrotest, Finalize_KnownBug_AFailedPluginFinalizeKeepsTheRef) {
  g_libs["ext"] = {12, 12};
  BuildList("ext");
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);

  g_net[kExtA].finalizeResult = ncclSystemError;
  EXPECT_EQ(ncclNetFinalize(comm_.get()), ncclSystemError);
  EXPECT_EQ(g_coll[0].finalizeCalls, 0);
  EXPECT_EQ(Lib(0).ncclNetPluginRefCount, 1);

  g_net[kExtA].finalizeResult = ncclSuccess;
  g_coll[0].finalizeResult = ncclRemoteError;
  EXPECT_EQ(ncclNetFinalize(comm_.get()), ncclRemoteError);
  EXPECT_EQ(Lib(0).ncclNetPluginRefCount, 1);
  EXPECT_EQ(Lib(0).dlHandle, kLibHandles[0]);
  EXPECT_TRUE(g_closed.empty());
}

TEST_F(PluginNetMicrotest, FinalizePropagatesAnUnloadFailure) {
  g_libs["ext"] = {12, 12};
  BuildList("ext");
  ASSERT_EQ(ncclNetInit(comm_.get()), ncclSuccess);
  g_closeResult = ncclSystemError;
  EXPECT_EQ(ncclNetFinalize(comm_.get()), ncclSystemError);
  EXPECT_EQ(Lib(0).ncclNetPluginRefCount, 0);
  EXPECT_EQ(g_closed.size(), 1u);
}

// Comms split with shared resources

TEST_F(PluginNetMicrotest, InitFromParentInheritsThePlugin) {
  auto parent = std::make_unique<ncclComm>();
  int netCtx, collCtx;
  parent->netContext = &netCtx;
  parent->collNetContext = &collCtx;
  parent->ncclNet = &ncclNetIb;
  parent->ncclCollNet = &g_extCollNet[0];
  parent->netPluginIndex = 3;
  parent->ncclNetVer = 9;
  parent->config.netName = "IB";
  parent->config.trafficClass = 1;
  comm_->config.netName = "ib";
  std::string log = CaptureLog([&] { ASSERT_EQ(ncclNetInitFromParent(comm_.get(), parent.get()), ncclSuccess); });
  EXPECT_FALSE(LogHas(log, "trafficClass")) << log;  // an unset child traffic class just inherits

  EXPECT_EQ(comm_->netContext, &netCtx);
  EXPECT_EQ(comm_->collNetContext, &collCtx);
  EXPECT_EQ(comm_->ncclNet, &ncclNetIb);
  EXPECT_EQ(comm_->ncclCollNet, &g_extCollNet[0]);
  EXPECT_EQ(comm_->netPluginIndex, 3);
  EXPECT_EQ(comm_->ncclNetVer, 9);
}

TEST_F(PluginNetMicrotest, InitFromParentRejectsADifferentNetNameButOnlyLogsTrafficClass) {
  auto parent = std::make_unique<ncclComm>();
  parent->config.netName = "IB";
  parent->config.trafficClass = 1;
  comm_->config.netName = "Socket";
  std::string log =
      CaptureLog([&] { EXPECT_EQ(ncclNetInitFromParent(comm_.get(), parent.get()), ncclInvalidUsage); });
  EXPECT_TRUE(LogHas(log, "WARN Comm config netName (Socket) does not match the parent (IB)")) << log;

  comm_->config.netName = nullptr;
  comm_->config.trafficClass = 2;
  log = CaptureLog([&] { EXPECT_EQ(ncclNetInitFromParent(comm_.get(), parent.get()), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "INFO Comm config trafficClass (2) does not match the parent (1)")) << log;

  comm_->config.trafficClass = 1;
  log = CaptureLog([&] { EXPECT_EQ(ncclNetInitFromParent(comm_.get(), parent.get()), ncclSuccess); });
  EXPECT_FALSE(LogHas(log, "trafficClass")) << log;
}

// A child with a netName under a parent without one passes NULL to strcasecmp.
// Pins AICOMRCCL-2984; a fix flips this pin.
TEST_F(PluginNetMicrotest, InitFromParent_KnownBug_CrashesWhenOnlyTheChildHasANetName) {
  auto parent = std::make_unique<ncclComm>();
  comm_->config.netName = "IB";
  ScopedDeathTestStyle style("threadsafe");
  EXPECT_EXIT(
      {
        DisableCoreDumpsForExpectedCrash();
        (void)ncclNetInitFromParent(comm_.get(), parent.get());
      },
      DEATH_BY_SEGV, "");
}

// Device counts

TEST_F(PluginNetMicrotest, DevCountsRequireAnEnabledProbedPlugin) {
  BuildList("none");
  int phys = -7, virt = -7;
  std::string log = CaptureLog([&] {
    EXPECT_EQ(ncclNetGetDevCount(0, &phys, &virt), ncclInternalError);
    EXPECT_EQ(ncclCollNetGetDevCount(0, &phys, &virt), ncclInternalError);
  });
  EXPECT_EQ(phys, -7);
  EXPECT_TRUE(LogHas(log, "WARN trying to access the number of devices of an uninitialized netPlugin[0]")) << log;

  // Enabled but never probed (a later comm whose first init failed) is rejected too.
  Lib(0).ncclNetPluginState = Lib(0).ncclCollNetPluginState = ncclNetPluginStateEnabled;
  EXPECT_EQ(ncclNetGetDevCount(0, &phys, &virt), ncclInternalError);
  EXPECT_EQ(ncclCollNetGetDevCount(0, &phys, &virt), ncclInternalError);

  Lib(0).netPhysDevs = 4;
  Lib(0).netVirtDevs = 6;
  Lib(0).collNetPhysDevs = 2;
  Lib(0).collNetVirtDevs = 3;
  ASSERT_EQ(ncclNetGetDevCount(0, &phys, &virt), ncclSuccess);
  EXPECT_EQ(phys, 4);
  EXPECT_EQ(virt, 6);
  ASSERT_EQ(ncclCollNetGetDevCount(0, &phys, &virt), ncclSuccess);
  EXPECT_EQ(phys, 2);
  EXPECT_EQ(virt, 3);
}

// netPluginIndex only comes from ncclNetInit's loop (or the parent comm), below pluginCount: the last slot is the bound.
TEST_F(PluginNetMicrotest, DevCountsReachTheLastSlot) {
  constexpr int kLast = NCCL_NET_MAX_PLUGINS - 1;
  BuildList("none");
  int phys = -7, virt = -7;
  EXPECT_EQ(ncclNetGetDevCount(kLast, &phys, &virt), ncclInternalError);  // past pluginCount: never loaded
  EXPECT_EQ(ncclNetSetVirtDevCount(kLast, 1), ncclInternalError);

  std::string list;
  for (int i = 0; i < NCCL_NET_MAX_PLUGINS; i++) list += (i ? ",p" : "p") + std::to_string(i);
  BuildList(list.c_str());
  ASSERT_EQ(pluginCount, NCCL_NET_MAX_PLUGINS);
  ASSERT_EQ(ncclNetPluginInit(comm_.get(), &Lib(kLast)), ncclSuccess);
  ASSERT_EQ(ncclNetSetVirtDevCount(kLast, 3), ncclSuccess);
  ASSERT_EQ(ncclNetGetDevCount(kLast, &phys, &virt), ncclSuccess);
  EXPECT_EQ(phys, 2);
  EXPECT_EQ(virt, 3);
  EXPECT_EQ(Lib(kLast - 1).netVirtDevs, NCCL_UNDEF_DEV_COUNT);
  EXPECT_EQ(ncclCollNetGetDevCount(kLast, &phys, &virt), ncclInternalError);  // internals have no collnet
  EXPECT_EQ(ncclCollNetSetVirtDevCount(kLast, 1), ncclInternalError);
}

TEST_F(PluginNetMicrotest, SetVirtDevCountRequiresEnabledAndNonNegative) {
  BuildList("none");
  std::string log = CaptureLog([&] {
    EXPECT_EQ(ncclNetSetVirtDevCount(1, 2), ncclInternalError);
    EXPECT_EQ(ncclCollNetSetVirtDevCount(1, 2), ncclInternalError);
  });
  EXPECT_TRUE(LogHas(log, "WARN ncclNetSetVirtDevCount: failed to set the number of devices for netPlugin[1] to 2"))
      << log;
  EXPECT_TRUE(LogHas(log, "WARN ncclCollNetSetVirtDevCount: failed")) << log;

  Lib(1).ncclNetPluginState = Lib(1).ncclCollNetPluginState = ncclNetPluginStateEnabled;
  EXPECT_EQ(ncclNetSetVirtDevCount(1, -1), ncclInternalError);
  EXPECT_EQ(ncclCollNetSetVirtDevCount(1, -1), ncclInternalError);
  EXPECT_EQ(Lib(1).netVirtDevs, NCCL_UNDEF_DEV_COUNT);

  ASSERT_EQ(ncclNetSetVirtDevCount(1, 0), ncclSuccess);
  ASSERT_EQ(ncclCollNetSetVirtDevCount(1, 5), ncclSuccess);
  EXPECT_EQ(Lib(1).netVirtDevs, 0);
  EXPECT_EQ(Lib(1).collNetVirtDevs, 5);
}

// GDR support. The answer is cached per cudaDev in a function-local static, so each case runs in a fresh process.

class PluginNetGdrMicrotestIsolated : public PluginNetMicrotest {
 protected:
  void SetUp() override {
    PluginNetMicrotest::SetUp();
    net_ = FakeNet<kIb>::Make("IB");
    comm_->ncclNet = &net_;
    comm_->cudaDev = 3;
  }
  ncclNet_t net_;
};

TEST_F(PluginNetGdrMicrotestIsolated, StopsAtTheFirstCudaCapableDeviceAndCaches) {
  RUN_ISOLATED_TEST("PluginNetGdr_StopsAtTheFirstCudaCapableDeviceAndCaches", [this]() {
    g_net[kIb].ndev = 4;
    g_net[kIb].cudaDevMask = 0b0110;
    int gdr = -1;
    ASSERT_EQ(ncclGpuGdrSupport(comm_.get(), &gdr), ncclSuccess);
    EXPECT_EQ(gdr, 1);
    EXPECT_EQ(g_net[kIb].devicesCalls, 1);
    EXPECT_EQ(g_net[kIb].propsCalls, 2);

    g_net[kIb].cudaDevMask = 0;
    ASSERT_EQ(ncclGpuGdrSupport(comm_.get(), &gdr), ncclSuccess);
    EXPECT_EQ(gdr, 1);
    EXPECT_EQ(g_net[kIb].devicesCalls, 1);
  });
}

TEST_F(PluginNetGdrMicrotestIsolated, NoCudaCapableDeviceMeansNoGdr) {
  RUN_ISOLATED_TEST("PluginNetGdr_NoCudaCapableDeviceMeansNoGdr", [this]() {
    g_net[kIb].ndev = 3;
    int gdr = -1;
    ASSERT_EQ(ncclGpuGdrSupport(comm_.get(), &gdr), ncclSuccess);
    EXPECT_EQ(gdr, 0);
    EXPECT_EQ(g_net[kIb].propsCalls, 3);

    // Another device has its own cache slot.
    comm_->cudaDev = 4;
    g_net[kIb].cudaDevMask = 0b1;
    ASSERT_EQ(ncclGpuGdrSupport(comm_.get(), &gdr), ncclSuccess);
    EXPECT_EQ(gdr, 1);
  });
}

TEST_F(PluginNetGdrMicrotestIsolated, ADevicesFailureIsRetried) {
  RUN_ISOLATED_TEST("PluginNetGdr_ADevicesFailureIsRetried", [this]() {
    int gdr = -1;
    g_net[kIb].cudaDevMask = 0b1;
    g_net[kIb].devicesResult = ncclSystemError;
    EXPECT_EQ(ncclGpuGdrSupport(comm_.get(), &gdr), ncclSystemError);
    EXPECT_EQ(gdr, -1);

    g_net[kIb].devicesResult = ncclSuccess;
    ASSERT_EQ(ncclGpuGdrSupport(comm_.get(), &gdr), ncclSuccess);
    EXPECT_EQ(gdr, 1);
    EXPECT_EQ(g_net[kIb].devicesCalls, 2);
  });
}

// The slot is zeroed before the scan, so one getProperties failure caches "no GDR".
// Pins AICOMRCCL-2985; a fix (#11701) flips this pin.
TEST_F(PluginNetGdrMicrotestIsolated, KnownBug_GetPropertiesFailureCachesNoGdr) {
  RUN_ISOLATED_TEST("PluginNetGdr_KnownBug_GetPropertiesFailureCachesNoGdr", [this]() {
    int gdr = -1;
    g_net[kIb].cudaDevMask = 0b1;
    g_net[kIb].propsResult = ncclRemoteError;
    EXPECT_EQ(ncclGpuGdrSupport(comm_.get(), &gdr), ncclRemoteError);

    g_net[kIb].propsResult = ncclSuccess;
    ASSERT_EQ(ncclGpuGdrSupport(comm_.get(), &gdr), ncclSuccess);
    EXPECT_EQ(gdr, 0);
    EXPECT_EQ(g_net[kIb].devicesCalls, 1);
  });
}

}  // namespace
