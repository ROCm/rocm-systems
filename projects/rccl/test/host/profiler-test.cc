/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/profiler.cc: plugin load/unload, the event start/stop/record wrappers and
// the KernelCh profiler thread, against a fake plugin behind the plugin-open and dlsym seams defined here.

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nccl.h"
#include "comm.h"
#include "debug.h"
#include "info.h"
#include "plugin.h"
#include "profiler.h"
#include "proxy.h"
#include "sym_kernels.h"
#include "transport.h"
#include "ce_coll.h"

// This binary links no shared nccl fakes. The adapter TUs (profiler-v*-test.cc) also resolve their
// ncclProfiler_vN symbol through ncclOsDlsym below, where a library handle is a FakeLib of exported symbols.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = "";
thread_local int ncclGroupDepth = 0;

using FakeLib = std::map<std::string, void*>;

namespace {
struct LoggedMessage {
  ncclDebugLogLevel level;
  std::string text;
};
std::mutex g_logMutex;  // the profiler thread logs too
std::vector<LoggedMessage> g_logged;

bool Logged(ncclDebugLogLevel level, const char* needle) {
  std::lock_guard<std::mutex> lock(g_logMutex);
  for (const auto& m : g_logged)
    if (m.level == level && m.text.find(needle) != std::string::npos) return true;
  return false;
}
std::vector<std::string> TakeLog() {
  std::lock_guard<std::mutex> lock(g_logMutex);
  std::vector<std::string> out;
  for (const auto& m : g_logged) out.push_back(m.text);
  g_logged.clear();
  return out;
}

std::map<std::string, std::string> g_env;
std::function<void*(const char*)> g_openProfilerLib;
std::vector<std::string> g_openedNames;
void* g_netPluginLib = nullptr;
std::vector<void*> g_closedLibs;
int64_t g_enableProxyTrace = 0;
const char* g_dladdrFname = nullptr;
uint64_t g_pid = 4242;
hipError_t g_setDeviceResult = hipSuccess;
std::atomic<int> g_setDeviceArg{-1};
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> lock(g_logMutex);
  g_logged.push_back({level, buf});
}

const char* ncclGetEnv(const char* name) {
  auto it = g_env.find(name);
  return it == g_env.end() ? nullptr : it->second.c_str();
}
int64_t rcclParamEnableProxyTrace() { return g_enableProxyTrace; }

char* ncclPluginLibPaths[ncclPluginTypeEnv + 1] = {};
void* ncclOpenProfilerPluginLib(const char* name) {
  g_openedNames.push_back(name ? name : "(null)");
  return g_openProfilerLib ? g_openProfilerLib(name) : nullptr;
}
void* ncclGetNetPluginLib(enum ncclPluginType type) {
  return type == ncclPluginTypeProfiler ? g_netPluginLib : nullptr;
}
ncclResult_t ncclClosePluginLib(void* handle, enum ncclPluginType type) {
  if (type == ncclPluginTypeProfiler) g_closedLibs.push_back(handle);
  return ncclSuccess;
}

void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol) {
  const FakeLib* lib = static_cast<const FakeLib*>(handle);
  auto it = lib->find(symbol);
  return it == lib->end() ? nullptr : it->second;
}
uint64_t ncclOsGetPid() { return g_pid; }

static void* testDlsym(void* handle, const char* symbol) { return ncclOsDlsym(handle, symbol); }
static int testDladdr(const void*, Dl_info* info) {
  if (g_dladdrFname == nullptr) return 0;
  info->dli_fname = g_dladdrFname;
  return 1;
}

hipError_t hipSetDevice(int device) {
  g_setDeviceArg.store(device);
  return g_setDeviceResult;
}
const char* hipGetErrorString(hipError_t) { return "fake hip error"; }
hipError_t hipGetLastError() { return hipSuccess; }

static const char* const kFuncNames[] = {"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7",
                                         "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15"};
const char* ncclFuncToString(ncclFunc_t fn) { return kFuncNames[fn & 15]; }
const char* ncclDatatypeToString(ncclDataType_t type) { return type == ncclFloat32 ? "ncclFloat32" : "ncclInt8"; }
const char* ncclAlgoToString(int algo) { return algo == NCCL_ALGO_RING ? "RING" : "TREE"; }
const char* ncclProtoToString(int proto) { return proto == NCCL_PROTO_LL ? "LL" : "SIMPLE"; }
const char* ncclSymkKernelIdToString(int kernelId) { return kernelId == 1 ? "AGxLL_R" : "NoUnderscore"; }

static ncclResult_t netSendProgress(struct ncclProxyState*, struct ncclProxyArgs*) { return ncclSuccess; }
static struct ncclTransport g_netTransport = {"NET", nullptr, {}, {}};
struct ncclTransport* ncclTransports[NTRANSPORTS] = {nullptr, nullptr, &g_netTransport, nullptr};

#define dlsym testDlsym
#define dladdr testDladdr
#include PROFILER_CC_PATH
#undef dlsym
#undef dladdr

namespace {

constexpr auto kWait = std::chrono::milliseconds(5000);

template <typename Pred>
bool waitUntil(Pred pred, std::chrono::milliseconds budget = kWait) {
  const auto deadline = std::chrono::steady_clock::now() + budget;
  while (!pred()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return true;
}

void* H(uintptr_t v) { return reinterpret_cast<void*>(v); }

struct StartCall {
  void* ctx;
  ncclProfilerEventDescr_t d;
  void* handle;
};
struct RecordCall {
  void* handle;
  ncclProfilerEventState_t state;
  bool hasArgs;
  ncclProfilerEventStateArgs_t args;
};

// The v7 (native) plugin. Called from the profiler thread as well, so every list is mutex-guarded.
struct FakePlugin {
  std::mutex mu;
  std::vector<StartCall> starts;
  std::vector<void*> stops;
  std::vector<RecordCall> records;
  std::vector<void*> finalized;
  int initCalls = 0;
  int initMask = 0;
  ncclResult_t initResult = ncclSuccess;
  void* ctx = H(0xC0);
  uint64_t initCommId = 0;
  std::string initCommName;
  int initNNodes = -1, initNRanks = -1, initRank = -1;
  ncclDebugLogger_t initLogfn = nullptr;
  uintptr_t nextHandle = 0x1000;
  uint64_t nullHandleType = 0;  // starts of this type hand back a null handle
  std::function<void()> onStart;
  ncclProfiler_t vtable{"FakeProfiler", &Init, &Start, &Stop, &Record, &Finalize};

  static FakePlugin*& cur() {
    static FakePlugin* p = nullptr;
    return p;
  }

  static ncclResult_t Init(void** context, uint64_t commId, int* mask, const char* name, int nNodes, int nRanks,
                           int rank, ncclDebugLogger_t logfn) {
    FakePlugin& f = *cur();
    f.initCalls++;
    f.initCommId = commId;
    f.initCommName = name ? name : "";
    f.initNNodes = nNodes;
    f.initNRanks = nRanks;
    f.initRank = rank;
    f.initLogfn = logfn;
    *mask = f.initMask;
    *context = f.ctx;
    return f.initResult;
  }
  static ncclResult_t Start(void* ctx, void** eHandle, ncclProfilerEventDescr_t* d) {
    FakePlugin& f = *cur();
    if (f.onStart) f.onStart();
    std::lock_guard<std::mutex> lock(f.mu);
    *eHandle = d->type == f.nullHandleType ? nullptr : H(f.nextHandle++);
    f.starts.push_back({ctx, *d, *eHandle});
    return ncclSuccess;
  }
  static ncclResult_t Stop(void* eHandle) {
    FakePlugin& f = *cur();
    std::lock_guard<std::mutex> lock(f.mu);
    f.stops.push_back(eHandle);
    return ncclSuccess;
  }
  static ncclResult_t Record(void* eHandle, ncclProfilerEventState_t s, ncclProfilerEventStateArgs_t* a) {
    FakePlugin& f = *cur();
    std::lock_guard<std::mutex> lock(f.mu);
    RecordCall r{eHandle, s, a != nullptr, {}};
    if (a) r.args = *a;
    f.records.push_back(r);
    return ncclSuccess;
  }
  static ncclResult_t Finalize(void* context) {
    FakePlugin& f = *cur();
    std::lock_guard<std::mutex> lock(f.mu);
    f.finalized.push_back(context);
    return ncclSuccess;
  }

  std::vector<StartCall> StartsOf(uint64_t type) {
    std::lock_guard<std::mutex> lock(mu);
    std::vector<StartCall> out;
    for (auto& s : starts)
      if (s.d.type == type) out.push_back(s);
    return out;
  }
  size_t NumStops() {
    std::lock_guard<std::mutex> lock(mu);
    return stops.size();
  }
  std::vector<void*> Stops() {
    std::lock_guard<std::mutex> lock(mu);
    return stops;
  }
  std::vector<RecordCall> Records() {
    std::lock_guard<std::mutex> lock(mu);
    return records;
  }
  void Clear() {
    std::lock_guard<std::mutex> lock(mu);
    starts.clear();
    stops.clear();
    records.clear();
  }
};

// Resets the statics profiler.cc keeps across comms; the failed-load state is otherwise sticky for the process.
void ResetProfilerStatics() {
  profilerPluginStatus = profilerPluginLoadReady;
  profilerPluginRefCount = 0;
  profilerPluginLib = nullptr;
  ncclProfiler = nullptr;
  ncclProfilerProxyTraceDumpFn = nullptr;
  pid = 0;
  ncclProfilerEventMask = 0;
  ncclProfilerRasOverrideActive = false;
  ncclProfilerRasOverrideMask = 0;
  ncclProfilerApiState = {};
  ncclGroupDepth = 0;
}

class ProfilerMicrotest : public ::testing::Test {
 protected:
  FakePlugin fake_;
  FakeLib lib_;
  std::unique_ptr<ncclComm> comm_;

  void SetUp() override {
    ResetProfilerStatics();
    TakeLog();
    g_env.clear();
    g_openedNames.clear();
    g_closedLibs.clear();
    g_netPluginLib = nullptr;
    g_enableProxyTrace = 0;
    g_dladdrFname = "/opt/rocm/lib/librccl.so";
    g_setDeviceResult = hipSuccess;
    g_setDeviceArg.store(-1);
    for (auto& p : ncclPluginLibPaths) p = nullptr;
    FakePlugin::cur() = &fake_;
    lib_ = {{"ncclProfiler_v7", &fake_.vtable}};
    g_openProfilerLib = [this](const char*) -> void* { return &lib_; };
    comm_ = NewComm(H(0xC0));
  }

  void TearDown() override {
    g_openProfilerLib = nullptr;
    ResetProfilerStatics();
    FakePlugin::cur() = nullptr;
  }

  std::unique_ptr<ncclComm> NewComm(void* ctx) {
    auto comm = std::make_unique<ncclComm>();
    comm->rank = 3;
    comm->nRanks = 8;
    comm->nNodes = 2;
    comm->commHash = 0xABCD;
    comm->config.commName = "c0";
    comm->cudaDev = 5;
    comm->profilerContext = ctx;
    return comm;
  }

  void Load(int mask) {
    fake_.initMask = mask;
    ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
    ASSERT_TRUE(ncclProfilerPluginLoaded());
    fake_.Clear();
  }

  ncclInfo MakeInfo(ncclFunc_t coll) {
    ncclInfo info{};
    info.coll = coll;
    info.count = 42;
    info.datatype = ncclFloat32;
    info.root = 6;
    info.comm = comm_.get();
    info.stream = reinterpret_cast<hipStream_t>(0x55);
    info.collConfig.userProfilerTag = 9;
    return info;
  }
};

// Plugin load and unload

TEST_F(ProfilerMicrotest, PluginInit_EnvNone_NeverOpensALibAndStaysUnloaded) {
  g_env["NCCL_PROFILER_PLUGIN"] = "none";
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_TRUE(g_openedNames.empty());
  EXPECT_FALSE(ncclProfilerPluginLoaded());
  EXPECT_EQ(fake_.initCalls, 0);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "NCCL_PROFILER_PLUGIN set by environment to none"));
}

TEST_F(ProfilerMicrotest, PluginInit_NoLibAnywhere_FailsOnceAndDoesNotRetry) {
  g_openProfilerLib = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_EQ(g_openedNames, std::vector<std::string>{"(null)"});
  EXPECT_FALSE(ncclProfilerPluginLoaded());
  EXPECT_TRUE(g_closedLibs.empty());
}

TEST_F(ProfilerMicrotest, PluginInit_V7Plugin_InitsWithTheCommIdentity) {
  fake_.initMask = ncclProfileColl;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_TRUE(ncclProfilerPluginLoaded());
  EXPECT_EQ(ncclProfiler, &fake_.vtable);
  EXPECT_EQ(fake_.initCalls, 1);
  EXPECT_EQ(fake_.initCommId, 0xABCDu);
  EXPECT_EQ(fake_.initCommName, "c0");
  EXPECT_EQ(fake_.initNNodes, 2);
  EXPECT_EQ(fake_.initNRanks, 8);
  EXPECT_EQ(fake_.initRank, 3);
  EXPECT_EQ(fake_.initLogfn, &ncclDebugLog);
  EXPECT_EQ(comm_->profilerContext, fake_.ctx);
  EXPECT_EQ(ncclProfilerEventMask, ncclProfileColl);
  // No name was requested, so there is nothing to report as loaded.
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "Successfully loaded external profiler plugin"));

  ncclProxyOp op{};
  ASSERT_EQ(ncclSuccess, ncclProfilerAddPidToProxyOp(&op));
  EXPECT_EQ(op.pid, 4242);
}

TEST_F(ProfilerMicrotest, PluginInit_NamedPlugin_LogsTheNameOrTheResolvedPath) {
  g_env["NCCL_PROFILER_PLUGIN"] = "myprof";
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_EQ(g_openedNames, std::vector<std::string>{"myprof"});
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "Successfully loaded external profiler plugin myprof"));
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginFinalize(comm_.get()));

  char path[] = "/x/libnccl-profiler-myprof.so";
  ncclPluginLibPaths[ncclPluginTypeProfiler] = path;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "Successfully loaded external profiler plugin /x/libnccl-profiler-myprof.so"));
}

TEST_F(ProfilerMicrotest, PluginInit_SecondComm_SharesTheLibUntilTheLastFinalize) {
  auto other = NewComm(nullptr);
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(other.get()));
  EXPECT_EQ(g_openedNames.size(), 1u);
  EXPECT_EQ(fake_.initCalls, 2);

  ASSERT_EQ(ncclSuccess, ncclProfilerPluginFinalize(comm_.get()));
  EXPECT_TRUE(g_closedLibs.empty());
  EXPECT_TRUE(ncclProfilerPluginLoaded());

  ASSERT_EQ(ncclSuccess, ncclProfilerPluginFinalize(other.get()));
  EXPECT_EQ(g_closedLibs, std::vector<void*>{&lib_});
  EXPECT_FALSE(ncclProfilerPluginLoaded());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "Closing profiler plugin FakeProfiler"));

  // Fully unloaded, so the next comm loads afresh.
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_EQ(g_openedNames.size(), 2u);
}

TEST_F(ProfilerMicrotest, PluginLoad_PicksTheNewestVersionTheLibExports) {
  static const char* const kNames[] = {nullptr, "v1", "v2", "v3", "v4", "v5", "v6"};
  ncclProfiler_v1_t v1{kNames[1]};
  ncclProfiler_v2_t v2{kNames[2]};
  ncclProfiler_v3_t v3{kNames[3]};
  ncclProfiler_v4_t v4{kNames[4]};
  ncclProfiler_v5_t v5{kNames[5]};
  ncclProfiler_v6_t v6{kNames[6]};
  void* const vtables[] = {nullptr, &v1, &v2, &v3, &v4, &v5, &v6};
  for (int newest = 1; newest <= 6; newest++) {
    SCOPED_TRACE(newest);
    lib_.clear();
    for (int v = 1; v <= newest; v++) lib_["ncclProfiler_v" + std::to_string(v)] = vtables[v];
    ASSERT_EQ(ncclSuccess, ncclProfilerPluginLoad());
    ASSERT_TRUE(ncclProfilerPluginLoaded());
    EXPECT_STREQ(ncclProfiler->name, kNames[newest]);
    ASSERT_EQ(ncclSuccess, ncclProfilerPluginUnload());
  }
  lib_ = {{"ncclProfiler_v1", &v1}, {"ncclProfiler_v7", &fake_.vtable}};
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginLoad());
  EXPECT_EQ(ncclProfiler, &fake_.vtable);
}

TEST_F(ProfilerMicrotest, PluginLoad_NoKnownSymbol_ClosesTheLibAndSaysWhyOnlyIfNamed) {
  lib_.clear();
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_FALSE(ncclProfilerPluginLoaded());
  EXPECT_EQ(g_closedLibs, std::vector<void*>{&lib_});
  EXPECT_FALSE(Logged(NCCL_LOG_ATTN, "is unsupported"));
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "is unsupported"));

  ResetProfilerStatics();
  char path[] = "/x/libauto.so";
  ncclPluginLibPaths[ncclPluginTypeProfiler] = path;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "External profiler plugin /x/libauto.so is unsupported"));
  EXPECT_FALSE(Logged(NCCL_LOG_ATTN, "is unsupported"));

  ResetProfilerStatics();
  ncclPluginLibPaths[ncclPluginTypeProfiler] = nullptr;
  g_env["NCCL_PROFILER_PLUGIN"] = "libbad.so";
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_TRUE(Logged(NCCL_LOG_ATTN, "External profiler plugin libbad.so is unsupported"));
}

TEST_F(ProfilerMicrotest, PluginLoad_NoProfilerLib_FallsBackToTheNetPluginLib) {
  g_env["NCCL_PROFILER_PLUGIN"] = "missing";
  g_openProfilerLib = nullptr;
  g_netPluginLib = &lib_;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_TRUE(ncclProfilerPluginLoaded());
  // The net plugin's lib is not the one that was named, so its name is not reported.
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "Successfully loaded external profiler plugin"));
}

TEST_F(ProfilerMicrotest, PluginLoad_ProxyTraceEnabled_OpensTheSiblingNextToRccl) {
  g_enableProxyTrace = 1;
  g_openProfilerLib = [this](const char* name) -> void* {
    return name && std::strstr(name, "librccl-profiler-proxytrace.so") ? &lib_ : nullptr;
  };
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_EQ(g_openedNames,
            (std::vector<std::string>{"(null)", "/opt/rocm/lib/librccl-profiler-proxytrace.so"}));
  EXPECT_TRUE(ncclProfilerPluginLoaded());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "Loaded proxy-trace plugin from /opt/rocm/lib/librccl-profiler-proxytrace.so"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO,
                     "Successfully loaded external profiler plugin /opt/rocm/lib/librccl-profiler-proxytrace.so"));
}

TEST_F(ProfilerMicrotest, PluginLoad_ProxyTraceSiblingUnavailable_FallsThroughToTheNetPluginLib) {
  g_openProfilerLib = nullptr;
  g_netPluginLib = &lib_;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_EQ(g_openedNames.size(), 1u);  // proxy trace off: no sibling probe

  ResetProfilerStatics();
  g_openedNames.clear();
  g_enableProxyTrace = 1;
  g_dladdrFname = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_EQ(g_openedNames.size(), 1u);  // own path unknown

  ResetProfilerStatics();
  g_openedNames.clear();
  g_dladdrFname = "/opt/rocm/lib/librccl.so";
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_EQ(g_openedNames.size(), 2u);  // sibling probed but absent
  EXPECT_TRUE(ncclProfilerPluginLoaded());
}

int g_dumpCalls = 0;
void* g_dumpCtx = nullptr;
ncclDebugLogger_t g_dumpLogfn = nullptr;
int g_dumpNoWarn = -1;
void FakeProxyTraceDump(void* ctx, ncclDebugLogger_t logfn) {
  g_dumpCalls++;
  g_dumpCtx = ctx;
  g_dumpLogfn = logfn;
  g_dumpNoWarn = ncclDebugNoWarn;
  std::snprintf(ncclLastError, sizeof(ncclLastError), "dump clobbered this");
}

TEST_F(ProfilerMicrotest, ProxyTraceDump_Exported_RunsWithWarningsOnAndRestoresTheCallerState) {
  g_dumpCalls = 0;
  lib_["ncclProfilerProxyTraceDump"] = reinterpret_cast<void*>(&FakeProxyTraceDump);
  Load(0);
  ncclDebugNoWarn = NCCL_DESTROY;
  std::snprintf(ncclLastError, sizeof(ncclLastError), "earlier error");

  ncclProfilerProxyTraceDumpIfAny(nullptr);
  EXPECT_EQ(g_dumpCalls, 0);

  ncclProfilerProxyTraceDumpIfAny(H(0xD0));
  EXPECT_EQ(g_dumpCalls, 1);
  EXPECT_EQ(g_dumpCtx, H(0xD0));
  EXPECT_EQ(g_dumpLogfn, &ncclDebugLog);
  EXPECT_EQ(g_dumpNoWarn, 0);
  EXPECT_EQ(ncclDebugNoWarn, NCCL_DESTROY);
  EXPECT_STREQ(ncclLastError, "earlier error");
  ncclDebugNoWarn = 0;

  ASSERT_EQ(ncclSuccess, ncclProfilerPluginFinalize(comm_.get()));
  ncclProfilerProxyTraceDumpIfAny(H(0xD0));
  EXPECT_EQ(g_dumpCalls, 1);
}

TEST_F(ProfilerMicrotest, ProxyTraceDump_NotExported_IsANoOp) {
  g_dumpCalls = 0;
  Load(0);
  ncclProfilerProxyTraceDumpIfAny(H(0xD0));
  EXPECT_EQ(g_dumpCalls, 0);
}

TEST_F(ProfilerMicrotest, PluginInit_PluginInitFails_UnloadsAndContinuesWithoutProfiler) {
  fake_.initResult = ncclInternalError;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_TRUE(Logged(NCCL_LOG_ATTN, "Profiler init failed with error '3'"));
  EXPECT_FALSE(ncclProfilerPluginLoaded());
  EXPECT_EQ(g_closedLibs, std::vector<void*>{&lib_});
}

// Bug: PluginInit already unloads on a failed plugin init(), and commFree calls PluginFinalize for every comm,
// so the failed comm unloads twice and closes the lib under a comm that is still profiling.
TEST_F(ProfilerMicrotest, PluginInit_FailedInitThenFinalize_ClosesTheLibUnderAnotherLiveComm) {
  auto failing = NewComm(nullptr);
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  fake_.initResult = ncclInternalError;
  fake_.ctx = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(failing.get()));
  EXPECT_TRUE(ncclProfilerPluginLoaded());

  ASSERT_EQ(ncclSuccess, ncclProfilerPluginFinalize(failing.get()));
  EXPECT_FALSE(ncclProfilerPluginLoaded());
  EXPECT_EQ(g_closedLibs, std::vector<void*>{&lib_});

  // The surviving comm's context never reaches the plugin's finalize.
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginFinalize(comm_.get()));
  EXPECT_TRUE(fake_.finalized.empty());
  EXPECT_EQ(profilerPluginRefCount, -1);
}

TEST_F(ProfilerMicrotest, PluginFinalize_FinalizesTheCommContextUnlessItIsNull) {
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginFinalize(comm_.get()));
  EXPECT_EQ(fake_.finalized, std::vector<void*>{H(0xC0)});

  fake_.ctx = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginFinalize(comm_.get()));
  EXPECT_EQ(fake_.finalized.size(), 1u);
  EXPECT_EQ(g_closedLibs.size(), 2u);
}

// Event mask

TEST_F(ProfilerMicrotest, PluginInit_KernelPhaseWithoutKernelCh_TurnsKernelChOn) {
  fake_.initMask = ncclProfileKernelPhase;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_EQ(ncclProfilerEventMask, ncclProfileKernelPhase | ncclProfileKernelCh);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "enabling ncclProfileKernelCh implicitly (plugin set ncclProfileKernelPhase"));

  TakeLog();
  fake_.initMask = ncclProfileKernelPhase | ncclProfileKernelCh;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "implicitly"));
}

TEST_F(ProfilerMicrotest, PrintEventMask_NamesEachEnabledBit) {
  const char* const kNames[] = {"Group",    "Coll",      "P2p",     "ProxyOp",  "ProxyStep",   "ProxyCtrl",
                                "KernelCh", "NetPlugin", "GroupApi", "CollApi", "P2pApi",      "KernelLaunch",
                                "CeColl",   "CeSync",    "CeBatch", "KernelPhase", "ProxyDiag"};
  for (int bit = 0; bit <= 16; bit++) {
    printProfilerEventMask(1 << bit);
    char want[128];
    std::snprintf(want, sizeof(want), "Profiler event mask: 0x%x (%d) - Enabled: %s ", 1 << bit, 1 << bit,
                  kNames[bit]);
    EXPECT_EQ(TakeLog(), std::vector<std::string>{want});
  }
  printProfilerEventMask(0);
  printProfilerEventMask(0x1ffff);
  auto log = TakeLog();
  ASSERT_EQ(log.size(), 1u);
  EXPECT_NE(log[0].find("Enabled: Group Coll P2p ProxyOp ProxyStep ProxyCtrl KernelCh KernelPhase NetPlugin "
                        "GroupApi CollApi P2pApi KernelLaunch CeColl CeSync CeBatch ProxyDiag "),
            std::string::npos);
}

TEST_F(ProfilerMicrotest, SetRasOverride_SurvivesALaterPluginInit) {
  ncclProfilerSetRasOverride(ncclProfileColl);
  EXPECT_EQ(ncclProfilerEventMask, ncclProfileColl);
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "set out-of-band via RAS"));

  fake_.initMask = ncclProfileProxyOp;
  ASSERT_EQ(ncclSuccess, ncclProfilerPluginInit(comm_.get()));
  EXPECT_EQ(ncclProfilerEventMask, ncclProfileColl);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "re-applied RAS override 0x2"));
}

TEST_F(ProfilerMicrotest, SetRasOverride_KernelPhase_AddsKernelChAndLogsWhenLoaded) {
  Load(0);
  ncclProfilerSetRasOverride(ncclProfileKernelPhase);
  EXPECT_EQ(ncclProfilerEventMask, ncclProfileKernelPhase | ncclProfileKernelCh);
  EXPECT_EQ(ncclProfilerRasOverrideMask, ncclProfileKernelPhase | ncclProfileKernelCh);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "(RAS override set ncclProfileKernelPhase"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "set out-of-band via RAS"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "Enabled: KernelCh KernelPhase "));
}

// API events

TEST_F(ProfilerMicrotest, StartGroupApi_OutermostGroup_StartsAndRecordsTheDepth) {
  Load(ncclProfileGroupApi);
  ncclGroupDepth = 2;
  ncclInfo info = MakeInfo(ncclFuncAllReduce);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartGroupApiEvent(&info, true));
  auto starts = fake_.StartsOf(ncclProfileGroupApi);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0].ctx, H(0xC0));
  EXPECT_EQ(starts[0].d.rank, 3);
  EXPECT_TRUE(starts[0].d.groupApi.graphCaptured);
  EXPECT_EQ(starts[0].d.groupApi.groupDepth, 2);
  EXPECT_EQ(ncclProfilerApiState.groupApiEventHandle, starts[0].handle);
  EXPECT_EQ(ncclProfilerApiState.profilerGroupDepth, 2);
  EXPECT_EQ(ncclProfilerApiState.state, ncclProfilerGroupApiStartStateStarted);

  // Nested: only the outermost group is profiled.
  ASSERT_EQ(ncclSuccess, ncclProfilerStartGroupApiEvent(&info, true));
  EXPECT_EQ(fake_.StartsOf(ncclProfileGroupApi).size(), 1u);
}

TEST_F(ProfilerMicrotest, StartGroupApi_OnlyApiAndTaskBitsStartIt) {
  const int kStarts[] = {ncclProfileGroupApi, ncclProfileP2pApi,    ncclProfileCollApi,   ncclProfileKernelLaunch,
                         ncclProfileGroup,    ncclProfileColl,      ncclProfileP2p,       ncclProfileProxyOp,
                         ncclProfileProxyStep, ncclProfileKernelCh, ncclProfileNetPlugin, ncclProfileCeColl,
                         ncclProfileCeSync,   ncclProfileCeBatch};
  const int kNoStart[] = {ncclProfileProxyCtrl, ncclProfileKernelPhase, ncclProfileProxyDiag};
  Load(0);
  ncclInfo info = MakeInfo(ncclFuncAllReduce);
  for (int bit : kStarts) {
    SCOPED_TRACE(bit);
    ncclProfilerApiState = {};
    fake_.Clear();
    ncclProfilerEventMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartGroupApiEvent(&info, false));
    EXPECT_EQ(fake_.StartsOf(ncclProfileGroupApi).size(), 1u);
  }
  for (int bit : kNoStart) {
    SCOPED_TRACE(bit);
    ncclProfilerApiState = {};
    fake_.Clear();
    ncclProfilerEventMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartGroupApiEvent(&info, false));
    EXPECT_TRUE(fake_.StartsOf(ncclProfileGroupApi).empty());
    EXPECT_EQ(ncclProfilerApiState.eActivationMask, bit);
  }
}

TEST_F(ProfilerMicrotest, StartGroupApi_NoPlugin_OnlySnapshotsTheMask) {
  ncclProfilerEventMask = ncclProfileGroupApi;
  ncclInfo info = MakeInfo(ncclFuncAllReduce);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartGroupApiEvent(&info, false));
  EXPECT_EQ(ncclProfilerApiState.eActivationMask, ncclProfileGroupApi);
  EXPECT_EQ(ncclProfilerApiState.groupApiEventHandle, nullptr);
  EXPECT_EQ(ncclProfilerApiState.state, ncclProfilerGroupApiStartStateReset);
}

TEST_F(ProfilerMicrotest, StopGroupApi_OnlyAtDepthZero_StopsAndClearsTheHandle) {
  Load(ncclProfileGroupApi);
  ncclProfilerApiState.groupApiEventHandle = H(0x71);
  ncclProfilerApiState.profilerGroupDepth = 1;
  ASSERT_EQ(ncclSuccess, ncclProfilerStopGroupApiEvent());
  EXPECT_EQ(fake_.NumStops(), 0u);
  EXPECT_EQ(ncclProfilerApiState.groupApiEventHandle, H(0x71));

  ncclProfilerApiState.profilerGroupDepth = 0;
  ASSERT_EQ(ncclSuccess, ncclProfilerStopGroupApiEvent());
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{H(0x71)});
  EXPECT_EQ(ncclProfilerApiState.groupApiEventHandle, nullptr);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopGroupApiEvent());
  EXPECT_EQ(fake_.NumStops(), 1u);
}

TEST_F(ProfilerMicrotest, RecordGroupApiState_FollowsStartStopThenEndStart) {
  Load(ncclProfileGroupApi);
  ncclProfilerApiState.groupApiEventHandle = H(0x71);
  ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStarted;

  ASSERT_EQ(ncclSuccess, ncclProfilerRecordGroupApiEventState(ncclProfilerGroupEndApiStart));  // out of order
  EXPECT_TRUE(fake_.Records().empty());
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));
  EXPECT_EQ(ncclProfilerApiState.state, ncclProfilerGroupApiStartStateStopped);
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));  // repeat
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordGroupApiEventState(ncclProfilerGroupEndApiStart));
  EXPECT_EQ(ncclProfilerApiState.state, ncclProfilerGroupApiStartStateReset);
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordGroupApiEventState(ncclProfilerGroupEndApiStart));  // repeat

  auto records = fake_.Records();
  ASSERT_EQ(records.size(), 2u);
  EXPECT_EQ(records[0].handle, H(0x71));
  EXPECT_EQ(records[0].state, ncclProfilerGroupStartApiStop);
  EXPECT_FALSE(records[0].hasArgs);
  EXPECT_EQ(records[1].state, ncclProfilerGroupEndApiStart);
}

TEST_F(ProfilerMicrotest, RecordGroupApiState_NoHandle_AdvancesTheStateWithoutRecording) {
  Load(ncclProfileGroupApi);
  ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStarted;
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));
  EXPECT_EQ(ncclProfilerApiState.state, ncclProfilerGroupApiStartStateStopped);
  EXPECT_TRUE(fake_.Records().empty());
}

TEST_F(ProfilerMicrotest, StartP2pApi_FillsTheDescriptorUnderTheGroupApiParent) {
  Load(ncclProfileP2pApi);
  ncclProfilerApiState.eActivationMask = ncclProfileP2pApi;
  ncclProfilerApiState.groupApiEventHandle = H(0x77);
  ncclInfo info = MakeInfo(ncclFuncSend);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartP2pApiEvent(&info, false));
  auto starts = fake_.StartsOf(ncclProfileP2pApi);
  ASSERT_EQ(starts.size(), 1u);
  const auto& d = starts[0].d;
  EXPECT_EQ(d.rank, 3);
  EXPECT_EQ(d.parentObj, H(0x77));
  EXPECT_STREQ(d.p2pApi.func, ncclFuncToString(ncclFuncSend));
  EXPECT_EQ(d.p2pApi.count, 42u);
  EXPECT_STREQ(d.p2pApi.datatype, "ncclFloat32");
  EXPECT_EQ(d.p2pApi.stream, H(0x55));
  EXPECT_FALSE(d.p2pApi.graphCaptured);
  EXPECT_EQ(d.p2pApi.userTag, 9u);
  EXPECT_EQ(ncclProfilerApiState.p2pApiEventHandle, starts[0].handle);

  ASSERT_EQ(ncclSuccess, ncclProfilerStopP2pApiEvent());
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{starts[0].handle});
  EXPECT_EQ(ncclProfilerApiState.p2pApiEventHandle, nullptr);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopP2pApiEvent());
  EXPECT_EQ(fake_.NumStops(), 1u);
}

TEST_F(ProfilerMicrotest, StartP2pApi_OnlyP2pRelatedBitsStartIt) {
  const int kStarts[] = {ncclProfileP2pApi, ncclProfileP2p, ncclProfileProxyOp, ncclProfileProxyStep,
                         ncclProfileKernelCh, ncclProfileNetPlugin};
  const int kNoStart[] = {ncclProfileCollApi, ncclProfileColl, ncclProfileGroupApi, ncclProfileCeColl};
  Load(0);
  ncclInfo info = MakeInfo(ncclFuncSend);
  for (int bit : kStarts) {
    fake_.Clear();
    ncclProfilerApiState.eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartP2pApiEvent(&info, false));
    EXPECT_EQ(fake_.StartsOf(ncclProfileP2pApi).size(), 1u) << bit;
  }
  for (int bit : kNoStart) {
    fake_.Clear();
    ncclProfilerApiState.eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartP2pApiEvent(&info, false));
    EXPECT_TRUE(fake_.StartsOf(ncclProfileP2pApi).empty()) << bit;
  }
}

TEST_F(ProfilerMicrotest, StartApiEvents_GraphCapturedWithGraphUsageModeZero_LogsTheHangRisk) {
  ncclInfo info = MakeInfo(ncclFuncSend);
  comm_->config.graphUsageMode = 1;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartP2pApiEvent(&info, true));
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCollApiEvent(&info, true));
  comm_->config.graphUsageMode = 0;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartP2pApiEvent(&info, false));
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCollApiEvent(&info, false));
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "graphUsageMode"));

  ASSERT_EQ(ncclSuccess, ncclProfilerStartP2pApiEvent(&info, true));
  auto log = TakeLog();
  ASSERT_EQ(log.size(), 1u);
  EXPECT_NE(log[0].find("graphUsageMode is set to 0 but the user is capturing graphs"), std::string::npos);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCollApiEvent(&info, true));
  EXPECT_EQ(TakeLog().size(), 1u);
}

TEST_F(ProfilerMicrotest, StartCollApi_FillsTheDescriptorIncludingRoot) {
  Load(ncclProfileCollApi);
  ncclProfilerApiState.eActivationMask = ncclProfileCollApi;
  ncclProfilerApiState.groupApiEventHandle = H(0x77);
  ncclInfo info = MakeInfo(ncclFuncBroadcast);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCollApiEvent(&info, true));
  auto starts = fake_.StartsOf(ncclProfileCollApi);
  ASSERT_EQ(starts.size(), 1u);
  const auto& d = starts[0].d;
  EXPECT_EQ(d.rank, 3);
  EXPECT_EQ(d.parentObj, H(0x77));
  EXPECT_STREQ(d.collApi.func, ncclFuncToString(ncclFuncBroadcast));
  EXPECT_EQ(d.collApi.count, 42u);
  EXPECT_STREQ(d.collApi.datatype, "ncclFloat32");
  EXPECT_EQ(d.collApi.stream, H(0x55));
  EXPECT_EQ(d.collApi.root, 6);
  EXPECT_TRUE(d.collApi.graphCaptured);
  EXPECT_EQ(d.collApi.userTag, 9u);
  EXPECT_EQ(ncclProfilerApiState.collApiEventHandle, starts[0].handle);

  ASSERT_EQ(ncclSuccess, ncclProfilerStopCollApiEvent());
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{starts[0].handle});
  EXPECT_EQ(ncclProfilerApiState.collApiEventHandle, nullptr);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopCollApiEvent());
  EXPECT_EQ(fake_.NumStops(), 1u);
}

TEST_F(ProfilerMicrotest, StartCollApi_OnlyCollRelatedBitsStartIt) {
  const int kStarts[] = {ncclProfileCollApi,  ncclProfileColl,      ncclProfileProxyOp,
                         ncclProfileProxyStep, ncclProfileKernelCh, ncclProfileNetPlugin,
                         ncclProfileCeColl,   ncclProfileCeSync,    ncclProfileCeBatch};
  const int kNoStart[] = {ncclProfileP2pApi, ncclProfileP2p, ncclProfileGroupApi};
  Load(0);
  ncclInfo info = MakeInfo(ncclFuncAllReduce);
  for (int bit : kStarts) {
    fake_.Clear();
    ncclProfilerApiState.eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartCollApiEvent(&info, false));
    EXPECT_EQ(fake_.StartsOf(ncclProfileCollApi).size(), 1u) << bit;
  }
  for (int bit : kNoStart) {
    fake_.Clear();
    ncclProfilerApiState.eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartCollApiEvent(&info, false));
    EXPECT_TRUE(fake_.StartsOf(ncclProfileCollApi).empty()) << bit;
  }
}

// Plan-level events

class ProfilerPlanMicrotest : public ProfilerMicrotest {
 protected:
  std::unique_ptr<ncclKernelPlan> plan_ = std::make_unique<ncclKernelPlan>();
  std::vector<std::unique_ptr<ncclTaskColl>> colls_;
  std::vector<std::unique_ptr<ncclTaskP2p>> p2ps_;

  void SetUp() override {
    ProfilerMicrotest::SetUp();
    plan_->comm = comm_.get();
  }
  ncclTaskColl* AddColl(int mask) {
    colls_.push_back(std::make_unique<ncclTaskColl>());
    ncclTaskColl* t = colls_.back().get();
    t->eActivationMask = mask;
    ncclIntruQueueEnqueue(&plan_->collTaskQueue, t);
    return t;
  }
  ncclTaskP2p* AddP2p(int mask) {
    p2ps_.push_back(std::make_unique<ncclTaskP2p>());
    ncclTaskP2p* t = p2ps_.back().get();
    t->eActivationMask = mask;
    ncclIntruQueueEnqueue(&plan_->p2pTaskQueue, t);
    return t;
  }
};

TEST_F(ProfilerPlanMicrotest, StartKernelLaunch_FirstMaskedCollTaskGivesTheParent) {
  Load(ncclProfileKernelLaunch);
  AddColl(0)->groupApiEventHandle = H(0x90);
  AddColl(ncclProfileKernelLaunch)->groupApiEventHandle = H(0x91);
  AddP2p(ncclProfileKernelLaunch)->groupApiEventHandle = H(0x92);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartKernelLaunchEvent(plan_.get(), reinterpret_cast<hipStream_t>(0x56)));
  auto starts = fake_.StartsOf(ncclProfileKernelLaunch);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0].d.rank, 3);
  EXPECT_EQ(starts[0].d.parentObj, H(0x91));
  EXPECT_EQ(starts[0].d.kernelLaunch.stream, H(0x56));
  EXPECT_EQ(plan_->kernelLaunchEventHandle, starts[0].handle);
}

TEST_F(ProfilerPlanMicrotest, StartKernelLaunch_NoMaskedCollTask_FallsBackToP2p) {
  Load(ncclProfileKernelLaunch);
  AddColl(0);
  AddP2p(0)->groupApiEventHandle = H(0x92);
  AddP2p(ncclProfileKernelLaunch)->groupApiEventHandle = H(0x93);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartKernelLaunchEvent(plan_.get(), nullptr));
  auto starts = fake_.StartsOf(ncclProfileKernelLaunch);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0].d.parentObj, H(0x93));
}

TEST_F(ProfilerPlanMicrotest, StartKernelLaunch_FirstMaskedTaskLacksKernelLaunch_StartsNothing) {
  Load(ncclProfileKernelLaunch);
  AddColl(ncclProfileColl);
  AddColl(ncclProfileKernelLaunch);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartKernelLaunchEvent(plan_.get(), nullptr));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileKernelLaunch).empty());

  plan_ = std::make_unique<ncclKernelPlan>();
  plan_->comm = comm_.get();
  AddP2p(ncclProfileP2p);
  AddP2p(ncclProfileKernelLaunch);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartKernelLaunchEvent(plan_.get(), nullptr));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileKernelLaunch).empty());
}

// Unlike the group and task stops, the KernelLaunch stop leaves its handle set, so a second stop repeats it.
TEST_F(ProfilerPlanMicrotest, StopKernelLaunch_StopsButKeepsTheHandle) {
  Load(ncclProfileKernelLaunch);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopKernelLaunchEvent(plan_.get()));
  EXPECT_EQ(fake_.NumStops(), 0u);
  plan_->kernelLaunchEventHandle = H(0x94);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopKernelLaunchEvent(plan_.get()));
  ASSERT_EQ(ncclSuccess, ncclProfilerStopKernelLaunchEvent(plan_.get()));
  EXPECT_EQ(fake_.Stops(), (std::vector<void*>{H(0x94), H(0x94)}));
  EXPECT_EQ(plan_->kernelLaunchEventHandle, H(0x94));
}

TEST_F(ProfilerPlanMicrotest, StartGroup_OnlyTaskLevelBitsStartIt) {
  const int kStarts[] = {ncclProfileGroup,     ncclProfileColl,     ncclProfileP2p,      ncclProfileProxyOp,
                         ncclProfileProxyStep, ncclProfileKernelCh, ncclProfileNetPlugin};
  const int kNoStart[] = {ncclProfileGroupApi, ncclProfileCollApi, ncclProfileKernelLaunch, ncclProfileProxyCtrl};
  Load(0);
  ncclTaskColl* ct = AddColl(0);
  for (int bit : kStarts) {
    fake_.Clear();
    ct->eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartGroupEvent(plan_.get()));
    auto starts = fake_.StartsOf(ncclProfileGroup);
    ASSERT_EQ(starts.size(), 1u) << bit;
    EXPECT_EQ(plan_->groupEventHandle, starts[0].handle);
    EXPECT_EQ(starts[0].ctx, H(0xC0));
  }
  for (int bit : kNoStart) {
    fake_.Clear();
    ct->eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartGroupEvent(plan_.get()));
    EXPECT_TRUE(fake_.StartsOf(ncclProfileGroup).empty()) << bit;
  }
}

TEST_F(ProfilerPlanMicrotest, StartGroup_FirstMaskedTaskDecides_P2pWhenNoCollIsMasked) {
  Load(0);
  AddColl(0);
  AddP2p(0);
  AddP2p(ncclProfileP2p);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartGroupEvent(plan_.get()));
  EXPECT_EQ(fake_.StartsOf(ncclProfileGroup).size(), 1u);

  fake_.Clear();
  colls_[0]->eActivationMask = ncclProfileCollApi;  // masked but no group bit: wins over the p2p task
  ASSERT_EQ(ncclSuccess, ncclProfilerStartGroupEvent(plan_.get()));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileGroup).empty());

  fake_.Clear();
  colls_[0]->eActivationMask = 0;
  p2ps_[0]->eActivationMask = ncclProfileP2pApi;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartGroupEvent(plan_.get()));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileGroup).empty());
}

TEST_F(ProfilerPlanMicrotest, StopGroup_StopsOnceAndClearsTheHandle) {
  Load(ncclProfileGroup);
  plan_->groupEventHandle = H(0x95);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopGroupEvent(plan_.get()));
  ASSERT_EQ(ncclSuccess, ncclProfilerStopGroupEvent(plan_.get()));
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{H(0x95)});
  EXPECT_EQ(plan_->groupEventHandle, nullptr);
}

TEST_F(ProfilerPlanMicrotest, StartTaskEvents_CollDescriptorCarriesTheTaskAndBumpsSeqNumber) {
  Load(ncclProfileColl);
  plan_->groupEventHandle = H(0x30);
  comm_->seqNumber[ncclFuncAllReduce] = 7;
  ncclTaskColl* ct = AddColl(ncclProfileColl);
  ct->func = ncclFuncAllReduce;
  ct->sendbuff = H(0x100);
  ct->recvbuff = H(0x200);
  ct->count = 1000;
  ct->root = 2;
  ct->datatype = ncclFloat32;
  ct->nChannels = 4;
  ct->nWarps = 8;
  ct->algorithm = NCCL_ALGO_RING;
  ct->protocol = NCCL_PROTO_LL;
  ct->profilerTag = 11;
  ct->collApiEventHandle = H(0x31);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
  auto starts = fake_.StartsOf(ncclProfileColl);
  ASSERT_EQ(starts.size(), 1u);
  const auto& d = starts[0].d;
  EXPECT_EQ(d.coll.parentGroup, H(0x30));
  EXPECT_EQ(d.parentObj, H(0x31));
  EXPECT_EQ(d.rank, 3);
  EXPECT_EQ(d.coll.seqNumber, 7u);
  EXPECT_STREQ(d.coll.func, ncclFuncToString(ncclFuncAllReduce));
  EXPECT_EQ(d.coll.sendBuff, H(0x100));
  EXPECT_EQ(d.coll.recvBuff, H(0x200));
  EXPECT_EQ(d.coll.count, 1000u);
  EXPECT_EQ(d.coll.root, 2);
  EXPECT_STREQ(d.coll.datatype, "ncclFloat32");
  EXPECT_EQ(d.coll.nChannels, 4);
  EXPECT_EQ(d.coll.nWarps, 8);
  EXPECT_STREQ(d.coll.algo, "RING");
  EXPECT_STREQ(d.coll.proto, "LL");
  EXPECT_EQ(d.coll.kernelVariant, nullptr);
  EXPECT_FALSE(d.coll.isSymColl);
  EXPECT_EQ(d.coll.userTag, 11u);
  EXPECT_EQ(ct->eventHandle, starts[0].handle);
  EXPECT_EQ(comm_->seqNumber[ncclFuncAllReduce], 8u);
}

TEST_F(ProfilerPlanMicrotest, StartTaskEvents_OnlyCollTaskBitsStartACollEvent) {
  const int kStarts[] = {ncclProfileColl, ncclProfileProxyOp, ncclProfileProxyStep, ncclProfileKernelCh,
                         ncclProfileNetPlugin};
  const int kNoStart[] = {ncclProfileGroup, ncclProfileP2p, ncclProfileCollApi};
  Load(0);
  ncclTaskColl* ct = AddColl(0);
  for (int bit : kStarts) {
    fake_.Clear();
    ct->eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
    EXPECT_EQ(fake_.StartsOf(ncclProfileColl).size(), 1u) << bit;
  }
  for (int bit : kNoStart) {
    fake_.Clear();
    ct->eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
    EXPECT_TRUE(fake_.StartsOf(ncclProfileColl).empty()) << bit;
  }
}

TEST_F(ProfilerPlanMicrotest, StartTaskEvents_SymPlan_AdvertisesTheKernelVariantAndPlanChannels) {
  Load(ncclProfileColl);
  plan_->isSymColl = true;
  plan_->channelMask.masks[0] = 0xB;  // 3 channels
  ncclTaskColl* head = AddColl(ncclProfileColl);
  head->devFuncId = 1;  // "AGxLL_R"
  ncclTaskColl* tail = AddColl(ncclProfileColl);
  tail->devFuncId = 2;  // "NoUnderscore"
  tail->nChannels = 5;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
  auto starts = fake_.StartsOf(ncclProfileColl);
  ASSERT_EQ(starts.size(), 2u);
  EXPECT_EQ(starts[0].d.coll.nChannels, 3);
  EXPECT_TRUE(starts[0].d.coll.isSymColl);
  EXPECT_STREQ(starts[0].d.coll.kernelVariant, "AGxLL_R");
  EXPECT_STREQ(starts[0].d.coll.algo, "R");
  EXPECT_STREQ(starts[0].d.coll.proto, "");
  EXPECT_EQ(starts[1].d.coll.nChannels, 5);  // only the head task takes the plan's channel count
  EXPECT_STREQ(starts[1].d.coll.algo, "NoUnderscore");
}

// Bug: ncclTaskColl::nChannels was widened to 16 bits for MAXCHANNELS=256, but the v7 descriptor's
// coll.nChannels is uint8_t, so a 256-channel collective is advertised to the plugin as 0 channels.
TEST_F(ProfilerPlanMicrotest, StartTaskEvents_CollOn256Channels_AdvertisesZeroChannels) {
  Load(ncclProfileColl);
  AddColl(ncclProfileColl)->nChannels = 256;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
  auto starts = fake_.StartsOf(ncclProfileColl);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0].d.coll.nChannels, 0);
}

TEST_F(ProfilerPlanMicrotest, StartTaskEvents_GraphCapturedPlan_CountsSeqNumberOnlyWithKernelChEvents) {
  ncclTaskColl* ct = AddColl(ncclProfileKernelCh);
  ct->func = ncclFuncAllGather;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));  // no plugin, not captured
  EXPECT_EQ(comm_->seqNumber[ncclFuncAllGather], 1u);

  plan_->persistent = true;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));  // no plugin
  EXPECT_EQ(comm_->seqNumber[ncclFuncAllGather], 1u);

  Load(ncclProfileKernelCh);
  ct->eventHandle = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));  // neither parent handle
  EXPECT_EQ(comm_->seqNumber[ncclFuncAllGather], 1u);
  plan_->groupEventHandle = H(0x30);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
  EXPECT_EQ(comm_->seqNumber[ncclFuncAllGather], 2u);
  plan_->groupEventHandle = nullptr;
  ct->collApiEventHandle = H(0x31);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
  EXPECT_EQ(comm_->seqNumber[ncclFuncAllGather], 3u);
  ct->eActivationMask = ncclProfileColl;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
  EXPECT_EQ(comm_->seqNumber[ncclFuncAllGather], 3u);
}

TEST_F(ProfilerPlanMicrotest, StartTaskEvents_P2pDescriptorAdvertisesItsChannels) {
  Load(ncclProfileP2p);
  plan_->groupEventHandle = H(0x30);
  ncclTaskP2p* pt = AddP2p(ncclProfileP2p);
  pt->func = ncclFuncRecv;
  pt->buff = H(0x300);
  pt->count = 77;
  pt->datatype = ncclFloat32;
  pt->root = 4;
  pt->nChannels = 2;
  pt->channelMask = 0x7;
  pt->profilerTag = 12;
  pt->p2pApiEventHandle = H(0x32);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
  auto starts = fake_.StartsOf(ncclProfileP2p);
  ASSERT_EQ(starts.size(), 1u);
  const auto& d = starts[0].d;
  EXPECT_EQ(d.p2p.parentGroup, H(0x30));
  EXPECT_EQ(d.parentObj, H(0x32));
  EXPECT_EQ(d.rank, 3);
  EXPECT_STREQ(d.p2p.func, ncclFuncToString(ncclFuncRecv));
  EXPECT_EQ(d.p2p.buff, H(0x300));
  EXPECT_EQ(d.p2p.count, 77u);
  EXPECT_STREQ(d.p2p.datatype, "ncclFloat32");
  EXPECT_EQ(d.p2p.peer, 4);
  EXPECT_EQ(d.p2p.nChannels, 2);
  EXPECT_EQ(d.p2p.userTag, 12u);
  EXPECT_EQ(pt->eventHandle, starts[0].handle);

  // With KernelCh the count is the per-direction channel mask.
  fake_.Clear();
  pt->eActivationMask = ncclProfileP2p | ncclProfileKernelCh;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
  starts = fake_.StartsOf(ncclProfileP2p);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0].d.p2p.nChannels, 3);
}

TEST_F(ProfilerPlanMicrotest, StartTaskEvents_OnlyP2pTaskBitsStartAP2pEvent) {
  const int kStarts[] = {ncclProfileP2p, ncclProfileProxyOp, ncclProfileProxyStep, ncclProfileKernelCh,
                         ncclProfileNetPlugin};
  const int kNoStart[] = {ncclProfileColl, ncclProfileP2pApi, ncclProfileGroup};
  Load(0);
  ncclTaskP2p* pt = AddP2p(0);
  for (int bit : kStarts) {
    fake_.Clear();
    pt->eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
    EXPECT_EQ(fake_.StartsOf(ncclProfileP2p).size(), 1u) << bit;
  }
  for (int bit : kNoStart) {
    fake_.Clear();
    pt->eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartTaskEvents(plan_.get()));
    EXPECT_TRUE(fake_.StartsOf(ncclProfileP2p).empty()) << bit;
  }
}

TEST_F(ProfilerPlanMicrotest, StopTaskEvents_StopsEachStartedTaskOnce) {
  AddColl(0)->eventHandle = H(0x41);
  AddColl(0);
  AddP2p(0)->eventHandle = H(0x42);
  AddP2p(0);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopTaskEvents(plan_.get()));  // no plugin
  EXPECT_EQ(colls_[0]->eventHandle, H(0x41));

  Load(ncclProfileColl);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopTaskEvents(plan_.get()));
  ASSERT_EQ(ncclSuccess, ncclProfilerStopTaskEvents(plan_.get()));
  EXPECT_EQ(fake_.Stops(), (std::vector<void*>{H(0x41), H(0x42)}));
  EXPECT_EQ(colls_[0]->eventHandle, nullptr);
  EXPECT_EQ(p2ps_[0]->eventHandle, nullptr);
}

// Proxy events

class ProfilerProxyMicrotest : public ProfilerMicrotest {
 protected:
  std::unique_ptr<ncclProxyArgs> args_ = std::make_unique<ncclProxyArgs>();
  ncclProxySubArgs* sub_ = &args_->subs[1];

  void SetUp() override {
    ProfilerMicrotest::SetUp();
    g_netTransport.send.proxyProgress = netSendProgress;
    args_->sliceSteps = 2;
    args_->chunkSize = 100;
    args_->opCount = 12;
    args_->progress = netSendProgress;
    sub_->eActivationMask = ncclProfileProxyOp;
    sub_->taskEventHandle = H(0x41);
    sub_->rank = 3;
    sub_->pid = 99;
    sub_->channelId = 7;
    sub_->peer = 5;
    sub_->nsteps = 5;
    sub_->nbytes = 4096;
    sub_->profilerContext = H(0xC1);
    sub_->commHash = 0xAB;
    sub_->profExtras = {1, 2, 3, 4, 5};
  }
};

TEST_F(ProfilerProxyMicrotest, StartProxyOp_ScalesStepsBySliceStepsAndZeroesDiagFields) {
  Load(ncclProfileProxyOp);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartProxyOpEvent(1, args_.get()));
  auto starts = fake_.StartsOf(ncclProfileProxyOp);
  ASSERT_EQ(starts.size(), 1u);
  const auto& d = starts[0].d;
  EXPECT_EQ(starts[0].ctx, H(0xC1));
  EXPECT_EQ(d.parentObj, H(0x41));
  EXPECT_EQ(d.rank, 3);
  EXPECT_EQ(d.proxyOp.pid, 99);
  EXPECT_EQ(d.proxyOp.channelId, 7);
  EXPECT_EQ(d.proxyOp.peer, 5);
  EXPECT_EQ(d.proxyOp.nSteps, 3);  // DIVUP(5, 2)
  EXPECT_EQ(d.proxyOp.chunkSize, 200);
  EXPECT_EQ(d.proxyOp.isSend, 1);
  EXPECT_EQ(d.proxyOp.commHash, 0u);
  EXPECT_EQ(d.proxyOp.opCount, 0);
  EXPECT_EQ(d.proxyOp.traceFuncIdx, 0);
  EXPECT_EQ(d.proxyOp.traceTotalBytes, 0u);
  EXPECT_EQ(d.proxyOp.proxyNbytes, 0u);
  EXPECT_EQ(d.proxyOp.rawProxyNsteps, 0);
  EXPECT_EQ(sub_->opEventHandle, starts[0].handle);

  fake_.Clear();
  args_->progress = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartProxyOpEvent(1, args_.get()));
  EXPECT_EQ(fake_.StartsOf(ncclProfileProxyOp)[0].d.proxyOp.isSend, 0);
}

TEST_F(ProfilerProxyMicrotest, StartProxyOp_ProxyDiag_CarriesTheTraceFields) {
  Load(ncclProfileProxyDiag);
  sub_->eActivationMask = ncclProfileProxyDiag;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartProxyOpEvent(1, args_.get()));
  auto starts = fake_.StartsOf(ncclProfileProxyOp);
  ASSERT_EQ(starts.size(), 1u);
  const auto& p = starts[0].d.proxyOp;
  EXPECT_EQ(p.commHash, 0xABu);
  EXPECT_EQ(p.opCount, 12);
  EXPECT_EQ(p.traceFuncIdx, 1);
  EXPECT_EQ(p.traceProtocol, 2);
  EXPECT_EQ(p.tracePattern, 3);
  EXPECT_EQ(p.traceTotalBytes, 4u);
  EXPECT_EQ(p.proxyNbytes, 4096u);
  EXPECT_EQ(p.rawProxyNsteps, 5);
}

TEST_F(ProfilerProxyMicrotest, StartProxyOp_OnlyProxyBitsStartIt) {
  const int kStarts[] = {ncclProfileProxyOp, ncclProfileProxyStep, ncclProfileNetPlugin, ncclProfileProxyDiag};
  const int kNoStart[] = {ncclProfileColl, ncclProfileKernelCh, ncclProfileProxyCtrl};
  ASSERT_EQ(ncclSuccess, ncclProfilerStartProxyOpEvent(1, args_.get()));  // no plugin
  Load(0);
  for (int bit : kStarts) {
    fake_.Clear();
    sub_->eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartProxyOpEvent(1, args_.get()));
    EXPECT_EQ(fake_.StartsOf(ncclProfileProxyOp).size(), 1u) << bit;
  }
  for (int bit : kNoStart) {
    fake_.Clear();
    sub_->eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartProxyOpEvent(1, args_.get()));
    EXPECT_TRUE(fake_.StartsOf(ncclProfileProxyOp).empty()) << bit;
  }
}

TEST_F(ProfilerProxyMicrotest, StopProxyOp_StopsOnceAndClearsTheHandle) {
  Load(ncclProfileProxyOp);
  sub_->opEventHandle = H(0x43);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopProxyOpEvent(1, args_.get()));
  ASSERT_EQ(ncclSuccess, ncclProfilerStopProxyOpEvent(1, args_.get()));
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{H(0x43)});
  EXPECT_EQ(sub_->opEventHandle, nullptr);
}

TEST_F(ProfilerProxyMicrotest, StartProxyStep_SendAndRecv_FileTheHandleUnderTheSliceStep) {
  using StartFn = ncclResult_t (*)(int, ncclProxyArgs*, int);
  Load(ncclProfileProxyStep);
  sub_->eActivationMask = ncclProfileProxyStep;
  sub_->opEventHandle = H(0x43);
  for (StartFn start : {&ncclProfilerStartSendProxyStepEvent, &ncclProfilerStartRecvProxyStepEvent}) {
    fake_.Clear();
    std::memset(sub_->pHandles, 0, sizeof(sub_->pHandles));
    const int stepId = 2 * (NCCL_STEPS + 1);  // DIVUP by sliceSteps 2 -> step NCCL_STEPS + 1 -> slot 1
    ASSERT_EQ(ncclSuccess, start(1, args_.get(), stepId));
    auto starts = fake_.StartsOf(ncclProfileProxyStep);
    ASSERT_EQ(starts.size(), 1u);
    EXPECT_EQ(starts[0].ctx, H(0xC1));
    EXPECT_EQ(starts[0].d.parentObj, H(0x43));
    EXPECT_EQ(starts[0].d.rank, 3);
    EXPECT_EQ(starts[0].d.proxyStep.step, NCCL_STEPS + 1);
    EXPECT_EQ(sub_->pHandles[1].stepEventHandle, starts[0].handle);
    EXPECT_EQ(sub_->pHandles[1].subArgPtr, sub_);

    ASSERT_EQ(ncclSuccess, start(1, args_.get(), 3));  // DIVUP(3, 2) = 2
    EXPECT_EQ(fake_.StartsOf(ncclProfileProxyStep).back().d.proxyStep.step, 2);
    EXPECT_NE(sub_->pHandles[2].stepEventHandle, nullptr);
  }
}

TEST_F(ProfilerProxyMicrotest, StartProxyStep_OnlyStepBitsStartIt_ButTheSubArgIsAlwaysLinked) {
  const int kStarts[] = {ncclProfileProxyStep, ncclProfileNetPlugin, ncclProfileProxyDiag};
  ASSERT_EQ(ncclSuccess, ncclProfilerStartSendProxyStepEvent(1, args_.get(), 2));  // no plugin
  EXPECT_EQ(sub_->pHandles[1].subArgPtr, sub_);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartRecvProxyStepEvent(1, args_.get(), 4));
  EXPECT_EQ(sub_->pHandles[2].subArgPtr, sub_);
  Load(0);
  for (int bit : kStarts) {
    fake_.Clear();
    sub_->eActivationMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartSendProxyStepEvent(1, args_.get(), 2));
    ASSERT_EQ(ncclSuccess, ncclProfilerStartRecvProxyStepEvent(1, args_.get(), 2));
    EXPECT_EQ(fake_.StartsOf(ncclProfileProxyStep).size(), 2u) << bit;
  }
  fake_.Clear();
  sub_->eActivationMask = ncclProfileProxyOp;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartSendProxyStepEvent(1, args_.get(), 2));
  ASSERT_EQ(ncclSuccess, ncclProfilerStartRecvProxyStepEvent(1, args_.get(), 2));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileProxyStep).empty());
}

TEST_F(ProfilerProxyMicrotest, StopProxyStep_StopsTheSlotOnceAndClearsIt) {
  sub_->pHandles[2].stepEventHandle = H(0x44);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopProxyStepEvent(1, args_.get(), 3));  // no plugin
  EXPECT_EQ(sub_->pHandles[2].stepEventHandle, H(0x44));
  Load(ncclProfileProxyStep);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopProxyStepEvent(1, args_.get(), 3));
  ASSERT_EQ(ncclSuccess, ncclProfilerStopProxyStepEvent(1, args_.get(), 3));
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{H(0x44)});
  EXPECT_EQ(sub_->pHandles[2].stepEventHandle, nullptr);
}

TEST_F(ProfilerProxyMicrotest, ProxyCtrl_FollowsTheGlobalMask) {
  void* h = H(0xBAD);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartProxyCtrlEvent(H(0xC2), &h));  // no plugin
  EXPECT_EQ(h, nullptr);
  Load(ncclProfileProxyOp);
  h = H(0xBAD);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartProxyCtrlEvent(H(0xC2), &h));
  EXPECT_EQ(h, nullptr);
  EXPECT_TRUE(fake_.StartsOf(ncclProfileProxyCtrl).empty());

  ncclProfilerEventMask = ncclProfileProxyCtrl;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartProxyCtrlEvent(H(0xC2), &h));
  auto starts = fake_.StartsOf(ncclProfileProxyCtrl);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0].ctx, H(0xC2));
  EXPECT_EQ(h, starts[0].handle);

  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyCtrlEventState(h, 6, ncclProfilerProxyCtrlAppend));
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyCtrlEventState(nullptr, 6, ncclProfilerProxyCtrlAppend));
  ASSERT_EQ(ncclSuccess, ncclProfilerStopProxyCtrlEvent(h));
  ASSERT_EQ(ncclSuccess, ncclProfilerStopProxyCtrlEvent(nullptr));
  ncclProfilerEventMask = ncclProfileProxyOp;
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyCtrlEventState(h, 7, ncclProfilerProxyCtrlAppend));

  auto records = fake_.Records();
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records[0].handle, h);
  EXPECT_EQ(records[0].state, ncclProfilerProxyCtrlAppend);
  EXPECT_EQ(records[0].args.proxyCtrl.appendedProxyOps, 6);
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{h});
}

TEST_F(ProfilerProxyMicrotest, RecordProxyOpAndStep_NeedTheirHandles) {
  Load(ncclProfileProxyStep);
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyOpEventState(1, args_.get(), ncclProfilerProxyOpInProgress_v4));
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyStepEventState(1, args_.get(), 3, ncclProfilerProxyStepSendWait));
  EXPECT_TRUE(fake_.Records().empty());

  sub_->opEventHandle = H(0x43);
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyStepEventState(1, args_.get(), 3, ncclProfilerProxyStepSendWait));
  EXPECT_TRUE(fake_.Records().empty());  // no step handle in slot 2

  sub_->pHandles[2].stepEventHandle = H(0x44);
  sub_->transSize = 8192;
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyOpEventState(1, args_.get(), ncclProfilerProxyOpInProgress_v4));
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyStepEventState(1, args_.get(), 3, ncclProfilerProxyStepSendWait));
  auto records = fake_.Records();
  ASSERT_EQ(records.size(), 2u);
  EXPECT_EQ(records[0].handle, H(0x43));
  EXPECT_EQ(records[0].state, ncclProfilerProxyOpInProgress_v4);
  EXPECT_TRUE(records[0].hasArgs);
  EXPECT_EQ(records[1].handle, H(0x44));
  EXPECT_EQ(records[1].state, ncclProfilerProxyStepSendWait);
  EXPECT_EQ(records[1].args.proxyStep.transSize, 8192u);
}

TEST_F(ProfilerProxyMicrotest, ProxyDiag_RecordsCountersOnlyWhenEnabledAndStarted) {
  EXPECT_FALSE(ncclProfilerProxyDiagEnabled());
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyDiagState(1, args_.get(), 1, 2, 3, 4));  // no plugin
  Load(ncclProfileProxyOp);
  EXPECT_FALSE(ncclProfilerProxyDiagEnabled());
  ncclProfilerEventMask = ncclProfileProxyDiag;
  EXPECT_TRUE(ncclProfilerProxyDiagEnabled());

  sub_->opEventHandle = H(0x43);
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyDiagState(1, args_.get(), 1, 2, 3, 4));  // sub not diag
  sub_->eActivationMask = ncclProfileProxyDiag;
  sub_->opEventHandle = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyDiagState(1, args_.get(), 1, 2, 3, 4));  // not started
  EXPECT_TRUE(fake_.Records().empty());

  sub_->opEventHandle = H(0x43);
  ASSERT_EQ(ncclSuccess, ncclProfilerRecordProxyDiagState(1, args_.get(), 9, -5, 1LL << 40, 3));
  auto records = fake_.Records();
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records[0].handle, H(0x43));
  EXPECT_EQ(records[0].state, ncclProfilerProxyDiagUpdate);
  EXPECT_EQ(records[0].args.proxyDiag.counterKind, 9);
  EXPECT_EQ(records[0].args.proxyDiag.value, -5);
  EXPECT_EQ(records[0].args.proxyDiag.value2, 1LL << 40);
  EXPECT_EQ(records[0].args.proxyDiag.flags, 3);
}

TEST_F(ProfilerProxyMicrotest, NetPluginCallback_StartStopUpdate) {
  ncclProxyEventHandle p{H(0x44), sub_};
  void* h = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProfilerCallback(&h, ncclProfilerNetEventStart, &p, 17, H(0xE0)));  // no plugin
  Load(ncclProfileNetPlugin);
  ASSERT_EQ(ncclSuccess, ncclProfilerCallback(&h, ncclProfilerNetEventStart, &p, 17, H(0xE0)));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileNetPlugin).empty());  // the sub is not NetPlugin-enabled

  sub_->eActivationMask = ncclProfileNetPlugin;
  ASSERT_EQ(ncclSuccess, ncclProfilerCallback(&h, ncclProfilerNetEventStart, &p, 17, H(0xE0)));
  auto starts = fake_.StartsOf(ncclProfileNetPlugin);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0].ctx, H(0xC1));
  EXPECT_EQ(starts[0].d.parentObj, H(0x44));
  EXPECT_EQ(starts[0].d.rank, 3);
  EXPECT_EQ(starts[0].d.netPlugin.id, 17);
  EXPECT_EQ(starts[0].d.netPlugin.data, H(0xE0));
  EXPECT_EQ(h, starts[0].handle);

  ASSERT_EQ(ncclSuccess, ncclProfilerCallback(&h, ncclProfilerNetEventUpdate, nullptr, 0, H(0xE1)));
  ASSERT_EQ(ncclSuccess, ncclProfilerCallback(&h, ncclProfilerNetEventStop, nullptr, 0, nullptr));
  ASSERT_EQ(ncclSuccess, ncclProfilerCallback(&h, ncclProfilerNetEventUpdateAndStop, nullptr, 0, H(0xE2)));
  auto records = fake_.Records();
  ASSERT_EQ(records.size(), 2u);
  EXPECT_EQ(records[0].state, ncclProfilerNetPluginUpdate);
  EXPECT_EQ(records[0].args.netPlugin.data, H(0xE1));
  EXPECT_EQ(records[1].args.netPlugin.data, H(0xE2));
  EXPECT_EQ(fake_.Stops(), (std::vector<void*>{h, h}));
}

TEST_F(ProfilerMicrotest, KernelCh_StartsUnderTheTaskAndStopRecordsTheStopTime) {
  ncclProfilerWorkOp op{};
  op.eActivationMask = ncclProfileKernelCh;
  op.taskEventHandle = H(0x51);
  op.channelId = 9;
  op.profilerContext = H(0xC3);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartKernelChEvent(&op, 111));  // no plugin
  ASSERT_EQ(ncclSuccess, ncclProfilerStopKernelChEvent(&op, 222));
  Load(ncclProfileKernelCh);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopKernelChEvent(&op, 222));  // not started
  op.eActivationMask = ncclProfileColl;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartKernelChEvent(&op, 111));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileKernelCh).empty());

  op.eActivationMask = ncclProfileKernelCh;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartKernelChEvent(&op, 111));
  auto starts = fake_.StartsOf(ncclProfileKernelCh);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0].ctx, H(0xC3));
  EXPECT_EQ(starts[0].d.parentObj, H(0x51));
  EXPECT_EQ(starts[0].d.kernelCh.channelId, 9);
  EXPECT_EQ(starts[0].d.kernelCh.pTimer, 111u);
  EXPECT_EQ(op.kernelEventHandle, starts[0].handle);

  ASSERT_EQ(ncclSuccess, ncclProfilerStopKernelChEvent(&op, 222));
  auto records = fake_.Records();
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records[0].handle, op.kernelEventHandle);
  EXPECT_EQ(records[0].state, ncclProfilerKernelChStop);
  EXPECT_EQ(records[0].args.kernelCh.pTimer, 222u);
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{op.kernelEventHandle});
}

TEST_F(ProfilerMicrotest, DeviceMode_FollowsKernelChAndKernelPhase) {
  EXPECT_EQ(ncclProfilerDeviceMode(ncclProfileKernelCh), ncclDevProfilerModeNone);  // no plugin
  Load(0);
  EXPECT_EQ(ncclProfilerDeviceMode(ncclProfileColl), ncclDevProfilerModeNone);
  EXPECT_EQ(ncclProfilerDeviceMode(ncclProfileKernelCh), ncclDevProfilerModeKernelCh);
  EXPECT_EQ(ncclProfilerDeviceMode(ncclProfileKernelPhase),
            ncclDevProfilerModeKernelCh | ncclDevProfilerModeKernelPhase);
}

// CE events

class ProfilerCeMicrotest : public ProfilerMicrotest {
 protected:
  ncclCeCollArgs args_{};
  void SetUp() override {
    ProfilerMicrotest::SetUp();
    args_.func = ncclFuncAllGather;
    args_.rootRank = 1;
    args_.datatype = ncclFloat32;
    args_.nElts = 64;
    args_.sendBuff = reinterpret_cast<uint8_t*>(0x100);
    args_.recvBuff = reinterpret_cast<uint8_t*>(0x200);
    args_.collApiEventHandle = H(0x31);
    args_.userTag = 13;
    comm_->ceColl.ceSeqNum = 21;
    comm_->ceColl.useCompletePtr = true;
  }
};

TEST_F(ProfilerCeMicrotest, StartCeColl_FillsTheDescriptorWhenAnyCeBitIsSet) {
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeCollEvent(comm_.get(), &args_, nullptr));  // no plugin
  Load(ncclProfileColl);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeCollEvent(comm_.get(), &args_, nullptr));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileCeColl).empty());

  for (int bit : {ncclProfileCeColl, ncclProfileCeSync, ncclProfileCeBatch}) {
    fake_.Clear();
    ncclProfilerEventMask = bit;
    ASSERT_EQ(ncclSuccess, ncclProfilerStartCeCollEvent(comm_.get(), &args_, reinterpret_cast<hipStream_t>(0x57)));
    EXPECT_EQ(fake_.StartsOf(ncclProfileCeColl).size(), 1u) << bit;
  }
  auto starts = fake_.StartsOf(ncclProfileCeColl);
  const auto& d = starts[0].d;
  EXPECT_EQ(d.parentObj, H(0x31));
  EXPECT_EQ(d.rank, 3);
  EXPECT_EQ(d.ceColl.seqNumber, 21u);
  EXPECT_STREQ(d.ceColl.func, ncclFuncToString(ncclFuncAllGather));
  EXPECT_EQ(d.ceColl.sendBuff, H(0x100));
  EXPECT_EQ(d.ceColl.recvBuff, H(0x200));
  EXPECT_EQ(d.ceColl.count, 64u);
  EXPECT_EQ(d.ceColl.root, 1);
  EXPECT_STREQ(d.ceColl.datatype, "ncclFloat32");
  EXPECT_STREQ(d.ceColl.syncStrategy, "UC");
  EXPECT_FALSE(d.ceColl.intraBatchSync);
  EXPECT_EQ(d.ceColl.batchSize, 0u);
  EXPECT_EQ(d.ceColl.numBatches, 0u);
  EXPECT_EQ(d.ceColl.ceSeqNum, 21u);
  EXPECT_EQ(d.ceColl.stream, H(0x57));
  EXPECT_EQ(d.ceColl.userTag, 13u);
  EXPECT_EQ(args_.ceCollProfHandle, starts[0].handle);

  comm_->symkState.hasLsaMultimem = true;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeCollEvent(comm_.get(), &args_, nullptr));
  EXPECT_STREQ(fake_.StartsOf(ncclProfileCeColl).back().d.ceColl.syncStrategy, "MC");
}

TEST_F(ProfilerCeMicrotest, StopCeColl_StopsTheStartedHandleAndToleratesNullArgs) {
  ASSERT_EQ(ncclSuccess, ncclProfilerStopCeCollEvent(comm_.get(), &args_, nullptr));  // no plugin
  Load(ncclProfileCeColl);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopCeCollEvent(comm_.get(), nullptr, nullptr));
  ASSERT_EQ(ncclSuccess, ncclProfilerStopCeCollEvent(comm_.get(), &args_, nullptr));
  EXPECT_EQ(fake_.NumStops(), 0u);
  args_.ceCollProfHandle = H(0x45);
  ASSERT_EQ(ncclSuccess, ncclProfilerStopCeCollEvent(comm_.get(), &args_, nullptr));
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{H(0x45)});
}

TEST_F(ProfilerCeMicrotest, CeSync_NeedsTheCeSyncBitAndAStartedParent) {
  void* h = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeSyncEvent(comm_.get(), &args_, nullptr, &h));  // no plugin
  Load(ncclProfileCeSync);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeSyncEvent(comm_.get(), nullptr, nullptr, &h));
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeSyncEvent(comm_.get(), &args_, nullptr, &h));  // no parent
  args_.ceCollProfHandle = H(0x45);
  ncclProfilerEventMask = ncclProfileCeColl;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeSyncEvent(comm_.get(), &args_, nullptr, &h));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileCeSync).empty());

  ncclProfilerEventMask = ncclProfileCeSync;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeSyncEvent(comm_.get(), &args_, nullptr, &h));
  auto starts = fake_.StartsOf(ncclProfileCeSync);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0].d.parentObj, H(0x45));
  EXPECT_EQ(starts[0].d.rank, 3);
  EXPECT_TRUE(starts[0].d.ceCollSync.isComplete);
  EXPECT_EQ(starts[0].d.ceCollSync.nRanks, 8);
  EXPECT_EQ(h, starts[0].handle);

  ASSERT_EQ(ncclSuccess, ncclProfilerStopCeSyncEvent(comm_.get(), nullptr, nullptr));
  ASSERT_EQ(ncclSuccess, ncclProfilerStopCeSyncEvent(comm_.get(), h, nullptr));
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{h});
}

TEST_F(ProfilerCeMicrotest, CeBatch_SumsTheOpSizes) {
  size_t sizes[] = {100, 200, 300};
  ncclCeBatchOpsParams params{};
  params.sizes = sizes;
  params.numOps = 3;
  params.intraBatchSync = true;
  void* h = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeBatchEvent(comm_.get(), &args_, &params, nullptr, &h));  // no plugin
  Load(ncclProfileCeBatch);
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeBatchEvent(comm_.get(), nullptr, &params, nullptr, &h));
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeBatchEvent(comm_.get(), &args_, &params, nullptr, &h));  // no parent
  args_.ceCollProfHandle = H(0x45);
  ncclProfilerEventMask = ncclProfileCeSync;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeBatchEvent(comm_.get(), &args_, &params, nullptr, &h));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileCeBatch).empty());

  ncclProfilerEventMask = ncclProfileCeBatch;
  ASSERT_EQ(ncclSuccess, ncclProfilerStartCeBatchEvent(comm_.get(), &args_, &params, nullptr, &h));
  auto starts = fake_.StartsOf(ncclProfileCeBatch);
  ASSERT_EQ(starts.size(), 1u);
  EXPECT_EQ(starts[0].d.parentObj, H(0x45));
  EXPECT_EQ(starts[0].d.rank, 3);
  EXPECT_EQ(starts[0].d.ceCollBatch.numOps, 3);
  EXPECT_EQ(starts[0].d.ceCollBatch.totalBytes, 600u);
  EXPECT_TRUE(starts[0].d.ceCollBatch.useIntraSync);
  EXPECT_EQ(h, starts[0].handle);

  ASSERT_EQ(ncclSuccess, ncclProfilerStopCeBatchEvent(comm_.get(), nullptr, nullptr));
  ASSERT_EQ(ncclSuccess, ncclProfilerStopCeBatchEvent(comm_.get(), h, nullptr));
  EXPECT_EQ(fake_.Stops(), std::vector<void*>{h});
}

// KernelCh profiler thread

class ProfilerThreadMicrotest : public ProfilerPlanMicrotest {
 protected:
  std::vector<ncclDevProfiler> started_ = std::vector<ncclDevProfiler>(MAXCHANNELS);
  std::vector<ncclDevProfiler> completed_ = std::vector<ncclDevProfiler>(MAXCHANNELS);
  std::vector<ncclDevProfiler> symStarted_ = std::vector<ncclDevProfiler>(MAXCHANNELS);
  std::vector<ncclDevProfiler> symCompleted_ = std::vector<ncclDevProfiler>(MAXCHANNELS);
  std::vector<ncclDevProfilerPhases> phases_ = std::vector<ncclDevProfilerPhases>(MAXCHANNELS);
  uint32_t abortFlag_ = 0;
  std::vector<ncclComm*> live_;  // comms whose thread ref TearDown still has to drop

  void SetUp() override {
    ProfilerPlanMicrotest::SetUp();
    Wire(comm_.get());
  }
  void TearDown() override {
    for (auto it = live_.rbegin(); it != live_.rend(); ++it) ncclProfilerThreadDestroy(*it);
    live_.clear();
    ProfilerPlanMicrotest::TearDown();
  }
  void Wire(ncclComm* comm) {
    comm->abortFlag = &abortFlag_;
    comm->profiler.workStarted = started_.data();
    comm->profiler.workCompleted = completed_.data();
    comm->profiler.symWorkStarted = symStarted_.data();
    comm->profiler.symWorkCompleted = symCompleted_.data();
    comm->profiler.symWorkPhases = phases_.data();
  }
  void CreateThread(ncclComm* comm, ncclComm* parent = nullptr) {
    ASSERT_EQ(ncclSuccess, ncclProfilerThreadCreate(comm, parent));
    ASSERT_NE(comm->profiler.profilerThread, nullptr);
    live_.push_back(comm);
  }
  void Destroy(ncclComm* comm) {
    ASSERT_EQ(ncclSuccess, ncclProfilerThreadDestroy(comm));
    live_.erase(std::find(live_.begin(), live_.end(), comm));
  }
  // What the device writes when the kernel for work wc on ch begins / ends.
  static void Stamp(std::vector<ncclDevProfiler>& buf, int ch, uint64_t wc, uint64_t ts) {
    auto& slot = buf[ch].data[wc % MAX_PROFILER_EVENTS_PER_CHANNEL];
    __atomic_store_n(&slot.timestamp, ts, __ATOMIC_RELAXED);
    __atomic_store_n(&slot.counter, wc, __ATOMIC_RELEASE);
  }
  // The thread walks `active` outside the mutex while iterationActive is set, so the lists are only stable
  // under the mutex once that clears.
  template <typename F>
  void WithIdleThread(ncclComm* comm, F f) {
    ncclProfilerThread* pt = comm->profiler.profilerThread;
    for (;;) {
      std::lock_guard<std::mutex> lock(pt->mutex);
      if (!pt->iterationActive) return f(pt);
    }
  }
  size_t OpsOn(ncclComm* comm, int ch) {
    size_t n = 0;
    WithIdleThread(comm, [&](ncclProfilerThread* pt) {
      for (ncclProfilerWorkOp* l : {pt->pending, pt->active})
        for (ncclProfilerWorkOp* op = l; op; op = op->next) n += op->channelId == ch;
    });
    return n;
  }
  size_t Inflight(ncclComm* comm) {
    std::lock_guard<std::mutex> lock(comm->profiler.profilerThread->mutex);
    return comm->profiler.profilerThread->inflight;
  }
  ncclTaskColl* AddKernelChColl(int lo, int hi, void* eventHandle) {
    ncclTaskColl* ct = AddColl(ncclProfileKernelCh);
    ct->channelLo = lo;
    ct->channelHi = hi;
    ct->eventHandle = eventHandle;
    return ct;
  }
};

TEST_F(ProfilerThreadMicrotest, ThreadCreate_NoPlugin_StartsNoThread) {
  ASSERT_EQ(ncclSuccess, ncclProfilerThreadCreate(comm_.get(), nullptr));
  EXPECT_EQ(comm_->profiler.profilerThread, nullptr);
  ASSERT_EQ(ncclSuccess, ncclProfilerThreadDestroy(comm_.get()));
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
}

TEST_F(ProfilerThreadMicrotest, ThreadCreate_BindsTheCommDeviceAndLogsItsCap) {
  Load(ncclProfileKernelCh);
  g_setDeviceResult = hipErrorInvalidDevice;
  CreateThread(comm_.get());
  ASSERT_TRUE(waitUntil([] { return Logged(NCCL_LOG_INFO, "[Profiler Thread] started"); }));
  EXPECT_EQ(g_setDeviceArg.load(), 5);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "SetDevice(5) failed: fake hip error"));
  char want[96];
  std::snprintf(want, sizeof(want), "started (cudaDev=5, maxInflight=%d)",
                MAXCHANNELS * MAX_PROFILER_EVENTS_PER_CHANNEL * 4);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, want));
  Destroy(comm_.get());
  EXPECT_EQ(comm_->profiler.profilerThread, nullptr);
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "[Profiler Thread] exiting"));  // nothing was ever queued
}

TEST_F(ProfilerThreadMicrotest, ThreadCreate_SharedResourcesChild_ReusesTheParentsThread) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  auto child = NewComm(H(0xC9));
  Wire(child.get());
  ASSERT_EQ(ncclSuccess, ncclProfilerThreadCreate(child.get(), comm_.get()));  // parent not sharing
  EXPECT_NE(child->profiler.profilerThread, comm_->profiler.profilerThread);
  ASSERT_EQ(ncclSuccess, ncclProfilerThreadDestroy(child.get()));

  comm_->shareResources = true;
  CreateThread(child.get(), comm_.get());
  ncclProfilerThread* pt = comm_->profiler.profilerThread;
  EXPECT_EQ(child->profiler.profilerThread, pt);
  EXPECT_EQ(pt->refCount, 2);
  Destroy(child.get());
  EXPECT_EQ(child->profiler.profilerThread, nullptr);
  EXPECT_EQ(pt->refCount, 1);

  // The parent's thread still runs.
  Stamp(started_, 0, 1, 10);
  Stamp(completed_, 0, 1, 20);
  AddKernelChColl(0, 0, H(0x51));
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  EXPECT_TRUE(waitUntil([this] { return fake_.NumStops() == 1; }));
}

TEST_F(ProfilerThreadMicrotest, PostPlanWork_CollTask_DrivesKernelChStartAndStopPerChannel) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  for (int ch : {1, 2}) {
    Stamp(started_, ch, 1, 100 + ch);
    Stamp(completed_, ch, 1, 200 + ch);
  }
  AddKernelChColl(1, 2, H(0x51));
  AddColl(ncclProfileColl);  // no KernelCh: posts nothing
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  EXPECT_EQ(comm_->profiler.workCounter[0], 0u);
  EXPECT_EQ(comm_->profiler.workCounter[1], 1u);
  EXPECT_EQ(comm_->profiler.workCounter[2], 1u);
  EXPECT_EQ(comm_->profiler.workCounter[3], 0u);

  ASSERT_TRUE(waitUntil([this] { return fake_.NumStops() == 2; }));
  auto starts = fake_.StartsOf(ncclProfileKernelCh);
  ASSERT_EQ(starts.size(), 2u);
  for (const auto& s : starts) {
    EXPECT_EQ(s.ctx, H(0xC0));
    EXPECT_EQ(s.d.parentObj, H(0x51));
    EXPECT_EQ(s.d.kernelCh.pTimer, 100u + s.d.kernelCh.channelId);
  }
  auto records = fake_.Records();
  ASSERT_EQ(records.size(), 2u);
  for (const auto& r : records) {
    EXPECT_EQ(r.state, ncclProfilerKernelChStop);
    EXPECT_TRUE(r.args.kernelCh.pTimer == 201u || r.args.kernelCh.pTimer == 202u);
  }
  ASSERT_TRUE(waitUntil([this] { return Inflight(comm_.get()) == 0; }));
}

// Only sym kernels publish phases, so a regular task carrying the global KernelPhase bit must not wait for them.
TEST_F(ProfilerThreadMicrotest, PostPlanWork_RegularTaskWithKernelPhase_CompletesWithoutPhases) {
  Load(ncclProfileKernelCh | ncclProfileKernelPhase);
  CreateThread(comm_.get());
  Stamp(started_, 0, 1, 10);
  Stamp(completed_, 0, 1, 20);
  AddKernelChColl(0, 0, H(0x51))->eActivationMask = ncclProfileKernelCh | ncclProfileKernelPhase;
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  EXPECT_TRUE(waitUntil([this] { return fake_.NumStops() == 1; }));
  EXPECT_TRUE(fake_.StartsOf(ncclProfileKernelPhase).empty());
}

TEST_F(ProfilerThreadMicrotest, PostPlanWork_KernelStillRunning_HoldsTheStopUntilItCompletes) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  Stamp(started_, 0, 1, 10);
  AddKernelChColl(0, 0, H(0x51));
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  ASSERT_TRUE(waitUntil([this] { return fake_.StartsOf(ncclProfileKernelCh).size() == 1; }));
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(fake_.NumStops(), 0u);
  EXPECT_EQ(Inflight(comm_.get()), 1u);

  // A slot the device has already reused for a later work still counts as completed.
  Stamp(completed_, 0, 1 + MAX_PROFILER_EVENTS_PER_CHANNEL, 20);
  ASSERT_TRUE(waitUntil([this] { return fake_.NumStops() == 1; }));
  EXPECT_EQ(fake_.StartsOf(ncclProfileKernelCh).size(), 1u);
}

TEST_F(ProfilerThreadMicrotest, PostPlanWork_SlotAlreadyReused_StartsTheKernelCh) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  Stamp(started_, 0, 1 + MAX_PROFILER_EVENTS_PER_CHANNEL, 10);
  AddKernelChColl(0, 0, H(0x51));
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  EXPECT_TRUE(waitUntil([this] { return fake_.StartsOf(ncclProfileKernelCh).size() == 1; }));
}

TEST_F(ProfilerThreadMicrotest, PostPlanWork_P2pPair_SharesOneCounterBumpPerChannel) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  for (int ch : {0, 1, 2}) {
    Stamp(started_, ch, 1, 10);
    Stamp(completed_, ch, 1, 20);
  }
  ncclTaskP2p* send = AddP2p(ncclProfileKernelCh);
  send->p2pPairId = 1;
  send->channelMask = 0x3;
  send->eventHandle = H(0x61);
  ncclTaskP2p* recv = AddP2p(ncclProfileKernelCh);
  recv->p2pPairId = 1;
  recv->channelMask = 0x6;
  recv->eventHandle = H(0x62);
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  for (int ch : {0, 1, 2}) EXPECT_EQ(comm_->profiler.workCounter[ch], 1u) << ch;
  EXPECT_EQ(comm_->profiler.workCounter[3], 0u);

  // Channel 1 carries both halves of the pair on the same device slot.
  ASSERT_TRUE(waitUntil([this] { return fake_.NumStops() == 4; }));
  int onCh1 = 0;
  for (const auto& s : fake_.StartsOf(ncclProfileKernelCh))
    if (s.d.kernelCh.channelId == 1) onCh1++;
  EXPECT_EQ(onCh1, 2);
}

TEST_F(ProfilerThreadMicrotest, PostPlanWork_P2pOutsideAPair_BumpsPerTask) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  for (uint16_t pid : {0, 0, 3, 4}) {
    ncclTaskP2p* t = AddP2p(ncclProfileKernelCh);
    t->p2pPairId = pid;
    t->channelMask = 0x1;
  }
  ncclTaskP2p* lone = AddP2p(ncclProfileKernelCh);  // pair partner without KernelCh still shares the bump
  lone->p2pPairId = 5;
  lone->channelMask = 0x1;
  ncclTaskP2p* partner = AddP2p(ncclProfileP2p);
  partner->p2pPairId = 5;
  partner->channelMask = 0x1;
  AddP2p(ncclProfileP2p)->channelMask = 0x1;  // no KernelCh at all
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  EXPECT_EQ(comm_->profiler.workCounter[0], 5u);
  EXPECT_EQ(OpsOn(comm_.get(), 0), 5u);
}

// Bug: the p2p post loop walks c < MAXCHANNELS (256) testing `1ull << c` against the 64-bit channelMask. For
// c >= 64 the shift is undefined; x86 wraps the count, so channel c aliases channel c % 64 and every aliased
// channel gets a spurious workCounter bump and an op that never completes.
TEST_F(ProfilerThreadMicrotest, PostPlanWork_P2pOnChannel0_AlsoBumpsTheAliasedChannelsAbove63) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  ncclTaskP2p* t = AddP2p(ncclProfileKernelCh);
  t->channelMask = 0x1;
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  for (int c = 0; c < MAXCHANNELS; c++) EXPECT_EQ(comm_->profiler.workCounter[c], c % 64 == 0 ? 1u : 0u) << c;
  EXPECT_EQ(Inflight(comm_.get()), size_t(MAXCHANNELS / 64));
}

TEST_F(ProfilerThreadMicrotest, ReserveSymCounters_WritesTheNextSymCounterPerPlanChannel) {
  alignas(16) static ncclSymkDevWorkArgs4K buf;
  std::memset(&buf, 0, sizeof(buf));
  buf.args.profilerMode = 1;
  plan_->kernelSymArgs = &buf.args;
  plan_->isSymColl = true;
  plan_->channelMask.masks[0] = 0x16;  // 3 channels
  ncclTaskColl* ct = AddColl(ncclProfileKernelCh);
  uint64_t* counters = buf.args.getProfilerCounters();

  ncclProfilerReserveSymCounters(comm_.get(), plan_.get());  // no plugin
  Load(ncclProfileKernelCh);
  ncclProfilerReserveSymCounters(comm_.get(), plan_.get());  // no thread
  CreateThread(comm_.get());
  plan_->isSymColl = false;
  ncclProfilerReserveSymCounters(comm_.get(), plan_.get());
  plan_->isSymColl = true;
  ct->eActivationMask = ncclProfileColl;
  ncclProfilerReserveSymCounters(comm_.get(), plan_.get());
  ct->eActivationMask = ncclProfileKernelCh;
  buf.args.profilerMode = 0;
  ncclProfilerReserveSymCounters(comm_.get(), plan_.get());
  EXPECT_EQ(counters[0], 0u);
  EXPECT_EQ(comm_->profiler.symWorkCounter[0], 0u);

  buf.args.profilerMode = 1;
  ncclProfilerReserveSymCounters(comm_.get(), plan_.get());
  ncclProfilerReserveSymCounters(comm_.get(), plan_.get());
  for (int c = 0; c < 3; c++) {
    EXPECT_EQ(counters[c], 2u) << c;
    EXPECT_EQ(comm_->profiler.symWorkCounter[c], 2u) << c;
  }
  EXPECT_EQ(counters[3], 0u);
  EXPECT_EQ(comm_->profiler.workCounter[0], 0u);  // regular counters are a separate sequence
}

TEST_F(ProfilerThreadMicrotest, ReserveSymCounters_EmptyPlan_ReservesNothing) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  plan_->isSymColl = true;
  ncclProfilerReserveSymCounters(comm_.get(), plan_.get());
  EXPECT_EQ(comm_->profiler.symWorkCounter[0], 0u);
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  EXPECT_EQ(Inflight(comm_.get()), 0u);
}

class ProfilerSymThreadMicrotest : public ProfilerThreadMicrotest {
 protected:
  alignas(16) ncclSymkDevWorkArgs4K buf_;
  ncclTaskColl* ct_ = nullptr;

  void SetUp() override {
    ProfilerThreadMicrotest::SetUp();
    std::memset(&buf_, 0, sizeof(buf_));
    buf_.args.profilerMode = 1;
    plan_->kernelSymArgs = &buf_.args;
    plan_->isSymColl = true;
    plan_->channelMask.masks[0] = 0x1;
    ct_ = AddColl(ncclProfileKernelCh | ncclProfileKernelPhase);
    ct_->eventHandle = H(0x52);
  }
  void SetPhases(int ch, uint64_t wc, std::initializer_list<uint64_t> ts) {
    auto& slot = phases_[ch].data[wc % MAX_PROFILER_EVENTS_PER_CHANNEL];
    int i = 0;
    for (uint64_t t : ts) slot.timestamps[i++] = t;
    __atomic_store_n(&slot.counter, wc, __ATOMIC_RELEASE);
  }
  // Runs one sym kernel on channel 0 and returns the phase events it produced.
  std::vector<StartCall> RunSym() {
    Stamp(symStarted_, 0, 1, 5);
    Stamp(symCompleted_, 0, 1, 50);
    ncclProfilerReserveSymCounters(comm_.get(), plan_.get());
    EXPECT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
    EXPECT_TRUE(waitUntil([this] { return Inflight(comm_.get()) == 0; }));
    return fake_.StartsOf(ncclProfileKernelPhase);
  }
};

TEST_F(ProfilerSymThreadMicrotest, PostPlanWork_SymKernel_FiresThePhasesBeforeTheKernelChStop) {
  Load(ncclProfileKernelCh | ncclProfileKernelPhase);
  CreateThread(comm_.get());
  SetPhases(0, 1, {10, 20, 30, 40});
  auto phases = RunSym();
  EXPECT_EQ(comm_->profiler.workCounter[0], 0u);
  auto kch = fake_.StartsOf(ncclProfileKernelCh);
  ASSERT_EQ(kch.size(), 1u);
  EXPECT_EQ(kch[0].d.parentObj, H(0x52));
  EXPECT_EQ(kch[0].d.kernelCh.pTimer, 5u);

  ASSERT_EQ(phases.size(), 3u);
  const char* const kNames[] = {"initial_sync", "compute", "final_sync"};
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(phases[i].ctx, H(0xC0));
    EXPECT_EQ(phases[i].d.parentObj, kch[0].handle);
    EXPECT_EQ(phases[i].d.kernelPhase.channelId, 0);
    EXPECT_EQ(phases[i].d.kernelPhase.phaseId, i);
    EXPECT_STREQ(phases[i].d.kernelPhase.phaseName, kNames[i]);
    EXPECT_EQ(phases[i].d.kernelPhase.pTimer, 10u * (i + 1));
  }
  auto records = fake_.Records();
  ASSERT_EQ(records.size(), 4u);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(records[i].handle, phases[i].handle);
    EXPECT_EQ(records[i].state, ncclProfilerKernelPhaseStop);
    EXPECT_EQ(records[i].args.kernelCh.pTimer, 10u * (i + 2));
  }
  EXPECT_EQ(records[3].state, ncclProfilerKernelChStop);
  EXPECT_EQ(records[3].args.kernelCh.pTimer, 50u);
  EXPECT_EQ(fake_.Stops(), (std::vector<void*>{phases[0].handle, phases[1].handle, phases[2].handle, kch[0].handle}));
  for (uint64_t t : phases_[0].data[1].timestamps) EXPECT_EQ(t, 0u);
}

TEST_F(ProfilerSymThreadMicrotest, PhaseEvents_MissingInnerBoundaries_FoldOntoTheOuterOnes) {
  Load(ncclProfileKernelCh | ncclProfileKernelPhase);
  CreateThread(comm_.get());
  SetPhases(0, 1, {10, 0, 30, 40});  // no AFTER_OPEN: initial_sync collapses to zero length
  auto phases = RunSym();
  ASSERT_EQ(phases.size(), 2u);
  EXPECT_EQ(phases[0].d.kernelPhase.phaseId, 1);
  EXPECT_EQ(phases[0].d.kernelPhase.pTimer, 10u);
  EXPECT_EQ(phases[1].d.kernelPhase.phaseId, 2);

  fake_.Clear();
  SetPhases(0, 2, {10, 20, 0, 40});  // no BEFORE_CLOSE: final_sync collapses
  Stamp(symStarted_, 0, 2, 5);
  Stamp(symCompleted_, 0, 2, 50);
  ncclProfilerReserveSymCounters(comm_.get(), plan_.get());
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  ASSERT_TRUE(waitUntil([this] { return Inflight(comm_.get()) == 0; }));
  phases = fake_.StartsOf(ncclProfileKernelPhase);
  ASSERT_EQ(phases.size(), 2u);
  EXPECT_EQ(phases[0].d.kernelPhase.phaseId, 0);
  EXPECT_EQ(phases[1].d.kernelPhase.phaseId, 1);
  auto records = fake_.Records();
  EXPECT_EQ(records[1].args.kernelCh.pTimer, 40u);
}

TEST_F(ProfilerSymThreadMicrotest, PhaseEvents_NoInnerBoundaries_FiresNoPhases) {
  Load(ncclProfileKernelCh | ncclProfileKernelPhase);
  CreateThread(comm_.get());
  SetPhases(0, 1, {10, 0, 0, 40});
  EXPECT_TRUE(RunSym().empty());
  EXPECT_EQ(fake_.NumStops(), 1u);
  for (uint64_t t : phases_[0].data[1].timestamps) EXPECT_EQ(t, 0u);
}

TEST_F(ProfilerSymThreadMicrotest, PhaseEvents_UnstampedOuterBoundaries_AreNotFolded) {
  Load(ncclProfileKernelCh | ncclProfileKernelPhase);
  CreateThread(comm_.get());
  SetPhases(0, 1, {0, 0, 30, 0});
  EXPECT_TRUE(RunSym().empty());
}

TEST_F(ProfilerSymThreadMicrotest, PhaseEvents_PluginDeclinesAPhase_NoStopForIt) {
  Load(ncclProfileKernelCh | ncclProfileKernelPhase);
  CreateThread(comm_.get());
  fake_.nullHandleType = ncclProfileKernelPhase;
  SetPhases(0, 1, {10, 20, 30, 40});
  EXPECT_EQ(RunSym().size(), 3u);
  EXPECT_EQ(fake_.Records().size(), 1u);  // only the KernelCh stop
  EXPECT_EQ(fake_.NumStops(), 1u);
}

TEST_F(ProfilerSymThreadMicrotest, PostPlanWork_PhasesNotPublishedYet_WaitsBeforeStopping) {
  Load(ncclProfileKernelCh | ncclProfileKernelPhase);
  CreateThread(comm_.get());
  Stamp(symStarted_, 0, 1, 5);
  Stamp(symCompleted_, 0, 1, 50);
  ncclProfilerReserveSymCounters(comm_.get(), plan_.get());
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  ASSERT_TRUE(waitUntil([this] { return fake_.StartsOf(ncclProfileKernelCh).size() == 1; }));
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(fake_.NumStops(), 0u);
  SetPhases(0, 1, {10, 20, 30, 40});
  EXPECT_TRUE(waitUntil([this] { return fake_.NumStops() == 4; }));
}

TEST_F(ProfilerSymThreadMicrotest, PostPlanWork_KernelChOnly_IgnoresThePhasesCounter) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  ct_->eActivationMask = ncclProfileKernelCh;
  EXPECT_TRUE(RunSym().empty());
  EXPECT_EQ(fake_.NumStops(), 1u);
}

TEST_F(ProfilerSymThreadMicrotest, PostPlanWork_SymWithoutProfilerArgsOrKernelCh_PostsNothing) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  buf_.args.profilerMode = 0;
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  buf_.args.profilerMode = 1;
  ct_->eActivationMask = ncclProfileColl;
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  EXPECT_EQ(Inflight(comm_.get()), 0u);

  ct_->eActivationMask = ncclProfileKernelCh;
  ncclProfilerThread* pt = comm_->profiler.profilerThread;
  comm_->profiler.profilerThread = nullptr;
  profilerPostPlanWorkSym(comm_.get(), plan_.get());
  comm_->profiler.profilerThread = pt;
  EXPECT_EQ(Inflight(comm_.get()), 0u);
}

TEST_F(ProfilerThreadMicrotest, ThreadDestroy_SharedThread_PurgesOnlyTheLeavingCommsOps) {
  Load(ncclProfileKernelCh);
  comm_->shareResources = true;
  CreateThread(comm_.get());
  auto child = NewComm(H(0xC9));
  Wire(child.get());
  CreateThread(child.get(), comm_.get());

  AddKernelChColl(0, 1, H(0x51));
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));  // never completes
  plan_->comm = child.get();
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(child.get(), plan_.get()));
  EXPECT_EQ(Inflight(comm_.get()), 4u);

  Destroy(child.get());
  WithIdleThread(comm_.get(), [](ncclProfilerThread* pt) {
    EXPECT_EQ(pt->inflight, 2u);
    for (ncclProfilerWorkOp* l : {pt->pending, pt->active})
      for (ncclProfilerWorkOp* op = l; op; op = op->next) EXPECT_EQ(op->profilerContext, H(0xC0));
  });
}

// A child leaving a shared thread does not join it, so only the iterationActive wait keeps its context alive
// across a plugin callback that is still running against it.
TEST_F(ProfilerThreadMicrotest, ThreadDestroy_WaitsOutAPluginCallbackInFlight) {
  Load(ncclProfileKernelCh);
  std::atomic<bool> inCallback{false}, release{false};
  fake_.onStart = [&] {
    inCallback = true;
    while (!release) std::this_thread::yield();
  };
  comm_->shareResources = true;
  CreateThread(comm_.get());
  auto child = NewComm(H(0xC9));
  Wire(child.get());
  CreateThread(child.get(), comm_.get());
  Stamp(started_, 0, 1, 10);
  AddKernelChColl(0, 0, H(0x51));
  plan_->comm = child.get();
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(child.get(), plan_.get()));
  ASSERT_TRUE(waitUntil([&] { return inCallback.load(); }));

  std::atomic<bool> destroyed{false};
  std::thread destroyer([&] {
    ncclProfilerThreadDestroy(child.get());
    destroyed = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_FALSE(destroyed.load());
  release = true;
  destroyer.join();
  EXPECT_TRUE(destroyed.load());
  live_.pop_back();
}

TEST_F(ProfilerThreadMicrotest, ProfilerThread_Abort_DrainsStuckOpsAndExits) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  AddKernelChColl(0, 0, H(0x51));
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  ASSERT_TRUE(waitUntil([this] {
    std::lock_guard<std::mutex> lock(comm_->profiler.profilerThread->mutex);
    return comm_->profiler.profilerThread->active != nullptr;
  }));
  __atomic_store_n(&abortFlag_, 1, __ATOMIC_RELAXED);
  ASSERT_TRUE(waitUntil([] { return Logged(NCCL_LOG_INFO, "exiting: maxInflightSeen=1, droppedOps=0"); }));
  EXPECT_EQ(Inflight(comm_.get()), 0u);
}

TEST_F(ProfilerThreadMicrotest, ProfilerThread_AbortWhileIdle_Exits) {
  Load(ncclProfileKernelCh);
  __atomic_store_n(&abortFlag_, 1, __ATOMIC_RELAXED);
  CreateThread(comm_.get());
  Destroy(comm_.get());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "[Profiler Thread] started"));
}

TEST_F(ProfilerThreadMicrotest, ThreadDestroy_LastRefWithAnotherContextsOpsQueued_DrainsAndJoins) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  AddKernelChColl(0, 2, H(0x51));
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  ASSERT_TRUE(waitUntil([this] {
    std::lock_guard<std::mutex> lock(comm_->profiler.profilerThread->mutex);
    return comm_->profiler.profilerThread->pending == nullptr;
  }));
  comm_->profilerContext = H(0xC8);  // the queued ops no longer match this comm, so the purge leaves them
  Destroy(comm_.get());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "exiting: maxInflightSeen=3, droppedOps=0"));
}

TEST_F(ProfilerThreadMicrotest, PostPlanWork_PastTheSoftCap_LogsOnPowersOfTwo) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  ASSERT_TRUE(waitUntil([] { return Logged(NCCL_LOG_INFO, "[Profiler Thread] started"); }));
  {
    std::lock_guard<std::mutex> lock(comm_->profiler.profilerThread->mutex);
    comm_->profiler.profilerThread->maxInflight = 1;
  }
  AddKernelChColl(0, 0, H(0x51));
  for (int i = 0; i < 4; i++) ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "inflight=1, cap=1"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "exceeded soft cap (inflight=2, cap=1)"));
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "inflight=3, cap=1"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "exceeded soft cap (inflight=4, cap=1)"));
  EXPECT_EQ(comm_->profiler.workCounter[0], 4u);
}

TEST_F(ProfilerThreadMicrotest, PostPlanWork_ManyOpsInFlight_CompleteInOrderOfTheDevice) {
  Load(ncclProfileKernelCh);
  CreateThread(comm_.get());
  AddKernelChColl(0, 0, H(0x51));
  for (int i = 0; i < 3; i++) ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  // The device finishes works 1 and 3; 2 stays outstanding behind them.
  for (uint64_t wc : {1, 3}) {
    Stamp(started_, 0, wc, wc);
    Stamp(completed_, 0, wc, 10 * wc);
  }
  ASSERT_TRUE(waitUntil([this] { return fake_.NumStops() == 2; }));
  EXPECT_EQ(Inflight(comm_.get()), 1u);
  Stamp(started_, 0, 2, 2);
  Stamp(completed_, 0, 2, 20);
  ASSERT_TRUE(waitUntil([this] { return Inflight(comm_.get()) == 0; }));
  EXPECT_EQ(fake_.NumStops(), 3u);

  // Work posted while older work is still active queues behind it.
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));  // wc 4, stays outstanding
  ASSERT_TRUE(waitUntil([this] {
    bool spliced = false;
    WithIdleThread(comm_.get(), [&](ncclProfilerThread* pt) { spliced = !pt->pending && pt->activeTail; });
    return spliced;
  }));
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));  // wc 5
  Stamp(started_, 0, 5, 5);
  Stamp(completed_, 0, 5, 50);
  ASSERT_TRUE(waitUntil([this] { return fake_.NumStops() == 4; }));
  Stamp(started_, 0, 4, 4);
  Stamp(completed_, 0, 4, 40);
  ASSERT_TRUE(waitUntil([this] { return Inflight(comm_.get()) == 0; }));
  EXPECT_EQ(fake_.NumStops(), 5u);

  // The list was fully recycled, so a later post appends cleanly.
  Stamp(started_, 0, 6, 6);
  Stamp(completed_, 0, 6, 60);
  ASSERT_EQ(ncclSuccess, ncclProfilerPostPlanWork(comm_.get(), plan_.get()));
  EXPECT_TRUE(waitUntil([this] { return fake_.NumStops() == 6; }));
}

}  // namespace
