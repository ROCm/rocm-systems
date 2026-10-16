/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/amdsmi_wrap.cc across its WSL2, amd_smi_lib and ARSMI
// backends. FakeSmi stands in for amd_smi_lib through the file's own pfn_* table; Reset() clears
// the file's process-wide caches, and the dlopen tests fork because the handle is static.

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "fakes/hip_fakes.h"

#include "nccl.h"
#include "debug.h"
#include "amdsmi_wrap.h"
#include "alt_rsmi.h"
#include "core.h"
#include "utils.h"

// --- Logging and params: this binary links no shared nccl fakes.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;  // utils.cc reads it

namespace {
struct LoggedMessage {
  ncclDebugLogLevel level;
  std::string text;
};
std::vector<LoggedMessage> g_logged;
std::map<std::string, int64_t> g_params;
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_logged.push_back({level, buf});
}

std::function<int64_t(const char*, int64_t)> g_loadParam = [](const char* env, int64_t deftVal) {
  auto it = g_params.find(env);
  return it == g_params.end() ? deftVal : it->second;
};

#include "fakes/param_redirect.h"

// --- ARSMI (src/misc/alt_rsmi.cc): scriptable stand-ins.
namespace {
struct FakeArsmi {
  int initResult = 0;
  int numDevicesResult = 0;
  uint32_t numDevices = 0;
  std::vector<uint64_t> bdfIds;  // ARSMI_dev_pci_id_get, by index
  int pciIdResult = 0;
  ARSMI_linkInfo link{};
  int linkResult = 0;
  uint64_t fwVersion = 0;
  int fwResult = 0;
  std::map<uint32_t, ARSMI_fabricInfo> fabric;  // devices missing here report ENODEV
  int initCalls = 0;
};
FakeArsmi g_arsmi;
}  // namespace

int ARSMI_init(void) {
  ++g_arsmi.initCalls;
  return g_arsmi.initResult;
}
int ARSMI_get_num_devices(uint32_t* n) {
  if (g_arsmi.numDevicesResult == 0) *n = g_arsmi.numDevices;
  return g_arsmi.numDevicesResult;
}
int ARSMI_dev_pci_id_get(uint32_t i, uint64_t* id) {
  if (g_arsmi.pciIdResult != 0) return g_arsmi.pciIdResult;
  *id = i < g_arsmi.bdfIds.size() ? g_arsmi.bdfIds[i] : 0;
  return 0;
}
int ARSMI_topo_get_link_info(uint32_t, uint32_t, ARSMI_linkInfo* info) {
  *info = g_arsmi.link;
  return g_arsmi.linkResult;
}
int ARSMI_get_fw_version(uint32_t, uint64_t* v) {
  *v = g_arsmi.fwVersion;
  return g_arsmi.fwResult;
}
int ARSMI_get_fabric_info(uint32_t i, struct ARSMI_fabricInfo* info) {
  auto it = g_arsmi.fabric.find(i);
  if (it == g_arsmi.fabric.end()) return 19;  // ENODEV
  *info = it->second;
  return 0;
}
const char* ARSMI_fabric_telem_id_to_string(uint64_t) { return "ARSMI_NAME"; }

// --- Seams scoped to amdsmi_wrap.cc's own function bodies.
namespace {
std::function<int(const char*, int)> g_access;
std::function<void*(const char*, int)> g_dlopen;
std::map<std::string, void*> g_dlsymTable;
std::function<hipError_t(int*, const char*)> g_hipDeviceGetByPCIBusId;
}  // namespace

static int AmdsmiTestAccess(const char* path, int mode) { return g_access(path, mode); }
static void* AmdsmiTestDlopen(const char* name, int flags) { return g_dlopen(name, flags); }
static void* AmdsmiTestDlsym(void*, const char* name) {
  auto it = g_dlsymTable.find(name);
  return it == g_dlsymTable.end() ? nullptr : it->second;
}
static hipError_t AmdsmiTestDeviceGetByPCIBusId(int* dev, const char* busId) {
  return g_hipDeviceGetByPCIBusId(dev, busId);
}

#define access AmdsmiTestAccess
#define dlopen AmdsmiTestDlopen
#define dlsym AmdsmiTestDlsym
#define hipDeviceGetByPCIBusId AmdsmiTestDeviceGetByPCIBusId
#include AMDSMI_WRAP_CC_PATH
#undef hipDeviceGetByPCIBusId
#undef dlsym
#undef dlopen
#undef access

namespace {

// Write extent of each runtime layout amdSmiDetectFabricRuntimeLayout recognizes.
constexpr size_t kExtent8Gpu = kAmdSmiFabricInfo8GpuSize;
constexpr size_t kExtent16Gpu = kAmdSmiFabricInfo16GpuSize;
constexpr size_t kExtentExtendedUnion = kAmdSmiFabricV1PayloadEnd;
// The extent a library matching this build's header writes, so the typed path is taken.
constexpr size_t kCompiledExtent =
  amdSmiFabricLayoutIs8Gpu ? kExtent8Gpu : (amdSmiFabricLayoutIs16Gpu ? kExtent16Gpu : kExtentExtendedUnion);

// The typed amd_smi fabric path is only ever taken for the 8- and 16-GPU layouts; the extended
// union always falls back to sysfs, so tests of the typed path skip on such a build.
constexpr bool kTypedFabricLayout = amdSmiFabricLayoutIs8Gpu || amdSmiFabricLayoutIs16Gpu;

template <typename FabricInfoT>
void SetFabricVersion(FabricInfoT& info, uint32_t version) {
  if constexpr (amdSmiFabricInfoIsFlat<FabricInfoT>::value) {
    info.fabric_version = version;
  } else {
    info.fabric_info.version = version;
  }
}

// One amd_smi_lib processor. Handles are this struct's address, so a test can
// tell which processor a call was made with.
struct FakeProc {
  bool gpu = true;
  uint32_t hipId = 0;
  uint64_t bdfId = 0;
  uint64_t fwVersion = 0;
  uint8_t numFwInfo = 1;
  bool fabricOk = true;
  uint32_t fabricVersion = AMDSMI_FABRIC_INFO_CURRENT_VERSION;
  amdsmi_fabric_type_t fabricType = AMDSMI_FABRIC_TYPE_UALOE;
  amdsmi_fabric_accelerator_vpod_state_t accelState = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE;
  uint32_t accelId = 0;
  uint32_t bandwidth = 0;
  uint32_t latency = 0;
  uint32_t vpodId = 0;
  uint32_t vpodSize = 0;
  uint32_t ppodSize = 0;
  uint8_t ppodId[16] = {};
};

// The amd_smi library as a set of sockets, each a list of processors.
struct FakeSmi {
  std::vector<std::vector<FakeProc>> sockets;
  amdsmi_status_t initStatus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t versionStatus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t shutdownStatus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t socketsStatus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t procsStatus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t enumStatus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t bdfStatus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t fromBdfStatus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t fwStatus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t telemStatus = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t fabricProbeStatus = AMDSMI_STATUS_SUCCESS;
  size_t fabricExtent = kCompiledExtent;  // bytes the library writes, from offset 0
  uint32_t libMajor = 27;
  amdsmi_link_type_t linkType = AMDSMI_LINK_TYPE_PCIE;
  uint64_t linkHops = 3;
  uint64_t minBw = 0;
  uint64_t maxBw = 0;
  int initCalls = 0;
  int shutdownCalls = 0;
  int procsCalls = 0;
  int typeCalls = 0;
  int enumCalls = 0;
  int fabricInfoCalls = 0;
  int telemAllocCalls = 0;
  int telemGetCalls = 0;
  int telemFreeCalls = 0;
  amdsmi_bdf_t lastBdf{};
  amdsmi_processor_handle lastLinkSrc = nullptr;
  amdsmi_processor_handle lastLinkDst = nullptr;
  amdsmi_processor_handle lastTelemHandle = nullptr;
  uint32_t lastTelemMask = 0;
  const char* telemName = "OUTPARAM_NAME";
  amdsmi_status_t telemNameStatus = AMDSMI_STATUS_SUCCESS;
};
FakeSmi g_smi;

FakeProc* Proc(amdsmi_processor_handle h) { return static_cast<FakeProc*>(h); }

amdsmi_status_t FakeInit(uint64_t) {
  ++g_smi.initCalls;
  return g_smi.initStatus;
}
amdsmi_status_t FakeShutDown() {
  ++g_smi.shutdownCalls;
  return g_smi.shutdownStatus;
}
amdsmi_status_t FakeStatusString(amdsmi_status_t, const char** s) {
  *s = "FAKE_STATUS";
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t FakeLibVersion(amdsmi_version_t* v) {
  if (g_smi.versionStatus != AMDSMI_STATUS_SUCCESS) return g_smi.versionStatus;
  std::memset(v, 0, sizeof(*v));
  v->major = g_smi.libMajor;
  v->minor = 1;
  v->release = 2;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t FakeSocketHandles(uint32_t* count, amdsmi_socket_handle* handles) {
  if (g_smi.socketsStatus != AMDSMI_STATUS_SUCCESS) return g_smi.socketsStatus;
  if (handles != nullptr) {
    for (uint32_t i = 0; i < *count && i < g_smi.sockets.size(); ++i) handles[i] = &g_smi.sockets[i];
  }
  *count = static_cast<uint32_t>(g_smi.sockets.size());
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t FakeProcessorHandles(amdsmi_socket_handle socket, uint32_t* count, amdsmi_processor_handle* handles) {
  ++g_smi.procsCalls;
  if (g_smi.procsStatus != AMDSMI_STATUS_SUCCESS) return g_smi.procsStatus;
  auto& procs = *static_cast<std::vector<FakeProc>*>(socket);
  if (handles != nullptr) {
    for (uint32_t i = 0; i < *count && i < procs.size(); ++i) handles[i] = &procs[i];
  }
  *count = static_cast<uint32_t>(procs.size());
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t FakeProcessorType(amdsmi_processor_handle h, processor_type_t* type) {
  ++g_smi.typeCalls;
  *type = Proc(h)->gpu ? AMDSMI_PROCESSOR_TYPE_AMD_GPU : AMDSMI_PROCESSOR_TYPE_AMD_CPU;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t FakeEnumerationInfo(amdsmi_processor_handle h, amdsmi_enumeration_info_t* info) {
  ++g_smi.enumCalls;
  if (g_smi.enumStatus != AMDSMI_STATUS_SUCCESS) return g_smi.enumStatus;
  std::memset(info, 0, sizeof(*info));
  info->hip_id = Proc(h)->hipId;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t FakeBdfId(amdsmi_processor_handle h, uint64_t* id) {
  if (g_smi.bdfStatus != AMDSMI_STATUS_SUCCESS) return g_smi.bdfStatus;
  *id = Proc(h)->bdfId;
  return AMDSMI_STATUS_SUCCESS;
}
// Looks the BDF up the way the library does: the first processor whose bdfId matches.
amdsmi_status_t FakeHandleFromBdf(amdsmi_bdf_t bdf, amdsmi_processor_handle* h) {
  g_smi.lastBdf = bdf;
  if (g_smi.fromBdfStatus != AMDSMI_STATUS_SUCCESS) return g_smi.fromBdfStatus;
  for (auto& s : g_smi.sockets) {
    for (auto& p : s) {
      if (p.bdfId == bdf.as_uint) {
        *h = &p;
        return AMDSMI_STATUS_SUCCESS;
      }
    }
  }
  return AMDSMI_STATUS_NOT_FOUND;
}
amdsmi_status_t FakeLinkType(amdsmi_processor_handle src, amdsmi_processor_handle dst, uint64_t* hops,
                             amdsmi_link_type_t* type) {
  g_smi.lastLinkSrc = src;
  g_smi.lastLinkDst = dst;
  *hops = g_smi.linkHops;
  *type = g_smi.linkType;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t FakeLinkWeight(amdsmi_processor_handle, amdsmi_processor_handle, uint64_t* weight) {
  *weight = 0;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t FakeMinMaxBw(amdsmi_processor_handle, amdsmi_processor_handle, uint64_t* minBw, uint64_t* maxBw) {
  *minBw = g_smi.minBw;
  *maxBw = g_smi.maxBw;
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t FakeFwInfo(amdsmi_processor_handle h, amdsmi_fw_info_t* info) {
  if (g_smi.fwStatus != AMDSMI_STATUS_SUCCESS) return g_smi.fwStatus;
  info->num_fw_info = Proc(h)->numFwInfo;
  info->fw_info_list[0].fw_version = Proc(h)->fwVersion;
  return AMDSMI_STATUS_SUCCESS;
}
// Writes like a whole-object library assignment: the first fabricExtent bytes, zeroed, then the v1
// payload. Reserved bytes stay zero, which is what the layout probe keys on.
amdsmi_status_t FakeFabricInfo(amdsmi_processor_handle h, amdsmi_fabric_info_t* info) {
  ++g_smi.fabricInfoCalls;
  const FakeProc& p = *Proc(h);
  if (!p.fabricOk) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (g_smi.fabricInfoCalls == 1 && g_smi.fabricProbeStatus != AMDSMI_STATUS_SUCCESS) {
    return g_smi.fabricProbeStatus;
  }
  std::memset(info, 0, std::min(g_smi.fabricExtent, sizeof(amdSmiFabricInfoBuffer)));
  SetFabricVersion(*info, p.fabricVersion);
  auto* v1 = const_cast<std::remove_const_t<std::remove_pointer_t<decltype(amdSmiFabricInfoV1(*info))>>*>(
    amdSmiFabricInfoV1(*info));
  v1->fabric_type = p.fabricType;
  v1->accel_state = p.accelState;
  v1->accelerator_id = p.accelId;
  v1->bandwidth = p.bandwidth;
  v1->latency = p.latency;
  v1->vpod_id = p.vpodId;
  v1->vpod_size = p.vpodSize;
  v1->ppod_size = p.ppodSize;
  std::memcpy(v1->ppod_id, p.ppodId, sizeof(v1->ppod_id));
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t FakeAllocTelem(amdsmi_processor_handle h, uint32_t mask, amdsmi_fabric_telemetry_t** t) {
  ++g_smi.telemAllocCalls;
  g_smi.lastTelemHandle = h;
  g_smi.lastTelemMask = mask;
  *t = reinterpret_cast<amdsmi_fabric_telemetry_t*>(0x1000);
  return g_smi.telemStatus;
}
amdsmi_status_t FakeGetTelem(amdsmi_processor_handle h, amdsmi_fabric_telemetry_t*) {
  ++g_smi.telemGetCalls;
  g_smi.lastTelemHandle = h;
  return g_smi.telemStatus;
}
amdsmi_status_t FakeFreeTelem(amdsmi_processor_handle h, amdsmi_fabric_telemetry_t*) {
  ++g_smi.telemFreeCalls;
  g_smi.lastTelemHandle = h;
  return g_smi.telemStatus;
}
amdsmi_status_t FakeTelemIdToString(uint64_t, const char** name) {
  if (g_smi.telemNameStatus != AMDSMI_STATUS_SUCCESS) return g_smi.telemNameStatus;
  *name = g_smi.telemName;
  return AMDSMI_STATUS_SUCCESS;
}
const char* g_legacyTelemName = "LEGACY_NAME";
const char* FakeLegacyTelemIdToString(uint64_t) { return g_legacyTelemName; }

// Every amd_smi entry point amdsmi_wrap.cc resolves, by the name it dlsym()s, with its pfn_* slot and
// this file's fake. One list drives installing the fakes, clearing them, and the dlopen test's dlsym.
struct SmiSymbol {
  const char* name;
  void** slot;
  void* fake;
};
#define SMI_SYMBOL(fn, fake) SmiSymbol{#fn, reinterpret_cast<void**>(&pfn_##fn), reinterpret_cast<void*>(fake)}
const std::vector<SmiSymbol>& SmiSymbols() {
  static const std::vector<SmiSymbol> symbols = {
    SMI_SYMBOL(amdsmi_init, FakeInit),
    SMI_SYMBOL(amdsmi_shut_down, FakeShutDown),
    SMI_SYMBOL(amdsmi_status_code_to_string, FakeStatusString),
    SMI_SYMBOL(amdsmi_get_lib_version, FakeLibVersion),
    SMI_SYMBOL(amdsmi_get_socket_handles, FakeSocketHandles),
    SMI_SYMBOL(amdsmi_get_processor_handles, FakeProcessorHandles),
    SMI_SYMBOL(amdsmi_get_processor_type, FakeProcessorType),
    SMI_SYMBOL(amdsmi_get_processor_handle_from_bdf, FakeHandleFromBdf),
    SMI_SYMBOL(amdsmi_get_gpu_enumeration_info, FakeEnumerationInfo),
    SMI_SYMBOL(amdsmi_get_gpu_bdf_id, FakeBdfId),
    SMI_SYMBOL(amdsmi_topo_get_link_type, FakeLinkType),
    SMI_SYMBOL(amdsmi_topo_get_link_weight, FakeLinkWeight),
    SMI_SYMBOL(amdsmi_get_minmax_bandwidth_between_processors, FakeMinMaxBw),
    SMI_SYMBOL(amdsmi_get_gpu_fabric_info, FakeFabricInfo),
    SMI_SYMBOL(amdsmi_alloc_fabric_telemetry, FakeAllocTelem),
    SMI_SYMBOL(amdsmi_get_fabric_telemetry_data, FakeGetTelem),
    SMI_SYMBOL(amdsmi_free_fabric_telemetry, FakeFreeTelem),
    SMI_SYMBOL(amdsmi_fabric_telem_id_to_string, FakeTelemIdToString),
    SMI_SYMBOL(amdsmi_get_fw_info, FakeFwInfo),
  };
  return symbols;
}
#undef SMI_SYMBOL

void InstallFakeSmiTable() {
  for (const auto& sym : SmiSymbols()) *sym.slot = sym.fake;
}

void ClearSmiTable() {
  for (const auto& sym : SmiSymbols()) *sym.slot = nullptr;
}

class AmdsmiWrapMicrotest : public ::testing::Test {
protected:
  void SetUp() override { Reset(); }
  void TearDown() override {
    ResetHipFakes();
    ClearSmiTable();
  }

  // Every seam back to a bare-metal host on the default ARSMI backend, and amdsmi_wrap.cc's
  // process-wide state back to "never initialized".
  void Reset() {
    ResetHipFakes();
    g_logged.clear();
    g_params.clear();
    g_arsmi = FakeArsmi{};
    g_smi = FakeSmi{};
    g_access = [](const char*, int) { return -1; };  // no /dev/dxg: not WSL2
    g_dlopen = [](const char*, int) -> void* { return nullptr; };
    g_dlsymTable.clear();
    g_hipDeviceGetByPCIBusId = [](int*, const char*) { return hipErrorInvalidValue; };
    g_legacyTelemName = "LEGACY_NAME";
    InstallFakeSmiTable();

    is_wsl2 = -1;
    amdSmiInitCalled = false;
    amdSmiInitResult = ncclSuccess;
    amdSmiLibInitialized = false;
    amdSmiLibMajor.store(kAmdSmiLibVersionUnknown);
    fabricInitialized = false;
    threadFabricInitialized = false;
    fabricInitResult = ncclSuccess;
    amdsmiFabricDeviceCount = 0;
    std::memset(amdsmiFabricDevices, 0, sizeof(amdsmiFabricDevices));
  }

  static void UseLib() { g_params["RCCL_USE_AMD_SMI_LIB"] = 1; }
  static void UseWsl2() {
    g_access = [](const char* path, int) { return std::strcmp(path, "/dev/dxg") == 0 ? 0 : -1; };
  }

  // Runs the one-time init so is_wsl2 and the backend are settled, then clears the log.
  static void InitFor(bool lib) {
    if (lib) UseLib();
    ASSERT_EQ(amd_smi_init(), ncclSuccess);
    g_logged.clear();
  }

  // One socket with GPUs whose hip ids are 0..n-1, plus a CPU first so lookups must skip it.
  static void TwoGpusBehindACpu() {
    FakeProc cpu;
    cpu.gpu = false;
    cpu.hipId = 1;  // would match hip id 1 if the type check were skipped
    FakeProc g0, g1;
    g0.hipId = 0;
    g0.bdfId = 0x1234000000c1ull;
    g1.hipId = 1;
    g1.bdfId = 0x1234000000d2ull;
    g_smi.sockets = {{cpu, g0, g1}};
  }

  static bool Logged(ncclDebugLogLevel level, const char* needle) {
    for (const auto& m : g_logged) {
      if (m.level == level && m.text.find(needle) != std::string::npos) return true;
    }
    return false;
  }
};

}  // namespace

// amd_smi_init / amd_smi_shutdown

TEST_F(AmdsmiWrapMicrotest, Init_Wsl2SkipsEveryBackend) {
  UseWsl2();
  UseLib();
  EXPECT_EQ(amd_smi_init(), ncclSuccess);
  EXPECT_EQ(is_wsl2, 1);
  EXPECT_EQ(g_smi.initCalls, 0);
  EXPECT_EQ(g_arsmi.initCalls, 0);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "WSL2 environment detected"));
}

TEST_F(AmdsmiWrapMicrotest, Init_DefaultBackendIsArsmi) {
  EXPECT_EQ(amd_smi_init(), ncclSuccess);
  EXPECT_EQ(is_wsl2, 0);
  EXPECT_EQ(g_arsmi.initCalls, 1);
  EXPECT_EQ(g_smi.initCalls, 0);
}

TEST_F(AmdsmiWrapMicrotest, Init_ArsmiFailureIsInternalError) {
  g_arsmi.initResult = 5;
  EXPECT_EQ(amd_smi_init(), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_ERROR, "ARSMI failure: 5"));
}

TEST_F(AmdsmiWrapMicrotest, Init_LibInitsAndRecordsTheLibraryMajor) {
  UseLib();
  g_smi.libMajor = 26;
  EXPECT_EQ(amd_smi_init(), ncclSuccess);
  EXPECT_EQ(g_smi.initCalls, 1);
  EXPECT_TRUE(amdSmiLibInitialized);
  EXPECT_EQ(amdSmiLibMajor.load(), 26u);
  EXPECT_EQ(g_arsmi.initCalls, 0);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "amdsmi_lib: version 26.1.2"));
}

TEST_F(AmdsmiWrapMicrotest, Init_LibInitFailureLeavesTheLibraryUninitialized) {
  UseLib();
  g_smi.initStatus = AMDSMI_STATUS_INVAL;
  EXPECT_EQ(amd_smi_init(), ncclInternalError);
  EXPECT_FALSE(amdSmiLibInitialized);
  EXPECT_TRUE(Logged(NCCL_LOG_ERROR, "AMD SMI failure: FAKE_STATUS"));
}

// The library is up even though the version query failed, so shutdown must still shut it down.
TEST_F(AmdsmiWrapMicrotest, Init_VersionFailureAfterInitStillCountsAsInitialized) {
  UseLib();
  g_smi.versionStatus = AMDSMI_STATUS_INVAL;
  EXPECT_EQ(amd_smi_init(), ncclInternalError);
  EXPECT_TRUE(amdSmiLibInitialized);
  EXPECT_EQ(amdSmiLibMajor.load(), kAmdSmiLibVersionUnknown);
  EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
  EXPECT_EQ(g_smi.shutdownCalls, 1);
}

// The result is cached, failures included, until shutdown resets it.
TEST_F(AmdsmiWrapMicrotest, Init_CachesItsResultUntilShutdown) {
  g_arsmi.initResult = 1;
  EXPECT_EQ(amd_smi_init(), ncclInternalError);
  g_arsmi.initResult = 0;
  EXPECT_EQ(amd_smi_init(), ncclInternalError);
  EXPECT_EQ(g_arsmi.initCalls, 1);

  EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
  EXPECT_EQ(amd_smi_init(), ncclSuccess);
  EXPECT_EQ(g_arsmi.initCalls, 2);
}

TEST_F(AmdsmiWrapMicrotest, Shutdown_WithoutInitDoesNothing) {
  EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
  EXPECT_EQ(g_smi.shutdownCalls, 0);
}

TEST_F(AmdsmiWrapMicrotest, Shutdown_ShutsTheLibraryDownOnceAndForgetsItsMajor) {
  UseLib();
  ASSERT_EQ(amd_smi_init(), ncclSuccess);
  EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
  EXPECT_EQ(g_smi.shutdownCalls, 1);
  EXPECT_FALSE(amdSmiLibInitialized);
  EXPECT_EQ(amdSmiLibMajor.load(), kAmdSmiLibVersionUnknown);
  EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
  EXPECT_EQ(g_smi.shutdownCalls, 1);
}

// The ARSMI backend has nothing to shut down.
TEST_F(AmdsmiWrapMicrotest, Shutdown_ArsmiBackendNeverCallsTheLibrary) {
  ASSERT_EQ(amd_smi_init(), ncclSuccess);
  EXPECT_EQ(amd_smi_shutdown(), ncclSuccess);
  EXPECT_EQ(g_smi.shutdownCalls, 0);
}

TEST_F(AmdsmiWrapMicrotest, Shutdown_LibraryFailureIsReturned) {
  UseLib();
  ASSERT_EQ(amd_smi_init(), ncclSuccess);
  g_smi.shutdownStatus = AMDSMI_STATUS_INVAL;
  EXPECT_EQ(amd_smi_shutdown(), ncclInternalError);
}

// dlopen runs once per process (function-local static handle), so these two run in forked children.
// The child bodies are plain functions: their lambdas' commas would split EXPECT_EXIT's arguments.
namespace {
int DlopenFailureChild() {
  ClearSmiTable();
  g_params["RCCL_USE_AMD_SMI_LIB"] = 1;
  static std::string opened;
  g_dlopen = [](const char* name, int) -> void* {
    opened = name;
    return nullptr;
  };
  ncclResult_t r = amd_smi_init();
  bool warned = false;
  for (const auto& m : g_logged) {
    warned |= m.level == NCCL_LOG_WARN && m.text.find("Failed to open") != std::string::npos;
  }
  return r == ncclInternalError && opened.rfind("libamd_smi.so", 0) == 0 && warned ? 0 : 1;
}

int DlopenResolvesChild() {
  ClearSmiTable();
  g_params["RCCL_USE_AMD_SMI_LIB"] = 1;
  static int handle;
  g_dlopen = [](const char*, int) -> void* { return &handle; };
  for (const auto& sym : SmiSymbols()) g_dlsymTable[sym.name] = sym.fake;
  if (amd_smi_init() != ncclSuccess || g_smi.initCalls != 1) return 1;
  // Every slot must hold the fake dlsym returned for its own name: a misspelled name leaves it null.
  for (const auto& sym : SmiSymbols()) {
    if (*sym.slot != sym.fake) {
      std::fprintf(stderr, "pfn slot for %s not resolved\n", sym.name);
      return 2;
    }
  }
  return 0;
}
}  // namespace

TEST_F(AmdsmiWrapMicrotest, Init_DlopenFailureIsInternalError) {
  EXPECT_EXIT(std::exit(DlopenFailureChild()), ::testing::ExitedWithCode(0), "");
}

TEST_F(AmdsmiWrapMicrotest, Init_DlopenResolvesEveryEntryPointByName) {
  EXPECT_EXIT(std::exit(DlopenResolvesChild()), ::testing::ExitedWithCode(0), "");
}

// getProcessorHandle (via amd_smi_getFirmwareVersion, its only plain caller)

TEST_F(AmdsmiWrapMicrotest, ProcessorHandle_IsAmdSmiLibOnly) {
  amdsmi_processor_handle h = nullptr;
  EXPECT_EQ(getProcessorHandle(0, &h), ncclSystemError);
}

TEST_F(AmdsmiWrapMicrotest, ProcessorHandle_MatchesTheHipIdAndSkipsNonGpus) {
  UseLib();
  TwoGpusBehindACpu();
  amdsmi_processor_handle h = nullptr;
  ASSERT_EQ(getProcessorHandle(1, &h), ncclSuccess);
  EXPECT_EQ(h, &g_smi.sockets[0][2]);
  EXPECT_EQ(getProcessorHandle(7, &h), ncclInternalError);
}

TEST_F(AmdsmiWrapMicrotest, ProcessorHandle_SearchesEverySocket) {
  UseLib();
  FakeProc a, b;
  a.hipId = 0;
  b.hipId = 1;
  g_smi.sockets = {{a}, {b}};
  amdsmi_processor_handle h = nullptr;
  ASSERT_EQ(getProcessorHandle(1, &h), ncclSuccess);
  EXPECT_EQ(h, &g_smi.sockets[1][0]);
}

TEST_F(AmdsmiWrapMicrotest, ProcessorHandle_LibraryFailuresAreInternalError) {
  UseLib();
  TwoGpusBehindACpu();
  amdsmi_processor_handle h = nullptr;
  // Each failure returns at once: nothing past the failing call runs.
  g_smi.socketsStatus = AMDSMI_STATUS_INVAL;
  EXPECT_EQ(getProcessorHandle(0, &h), ncclInternalError);
  EXPECT_EQ(g_smi.procsCalls, 0);
  g_smi.socketsStatus = AMDSMI_STATUS_SUCCESS;
  g_smi.enumStatus = AMDSMI_STATUS_INVAL;
  EXPECT_EQ(getProcessorHandle(0, &h), ncclInternalError);
  EXPECT_EQ(g_smi.enumCalls, 1);  // the first GPU's query failed; the second GPU was never asked
  EXPECT_EQ(g_smi.typeCalls, 2);  // the CPU and that first GPU
  EXPECT_EQ(h, nullptr);
}

#if !AMDSMI_DIRECT
// Built without the amd_smi header, an unresolved entry point is a quiet failure, not a crash.
TEST_F(AmdsmiWrapMicrotest, ProcessorHandle_MissingSymbolFailsQuietly) {
  UseLib();
  TwoGpusBehindACpu();
  pfn_amdsmi_get_socket_handles = nullptr;
  amdsmi_processor_handle h = nullptr;
  EXPECT_EQ(getProcessorHandle(0, &h), ncclInternalError);
  EXPECT_FALSE(Logged(NCCL_LOG_ERROR, "AMD SMI failure"));
}
#endif

// amd_smi_getNumDevice

TEST_F(AmdsmiWrapMicrotest, NumDevice_Wsl2AsksHip) {
  UseWsl2();
  InitFor(/*lib=*/false);
  g_hipGetDeviceCount = [](int* n) {
    *n = 3;
    return hipSuccess;
  };
  uint32_t n = 0;
  EXPECT_EQ(amd_smi_getNumDevice(&n), ncclSuccess);
  EXPECT_EQ(n, 3u);
}

// Counts every processor handle in every socket, CPUs included; a fix flips this pin.
TEST_F(AmdsmiWrapMicrotest, NumDevice_KnownBug_LibCountsCpuProcessors) {
  InitFor(/*lib=*/true);
  g_smi.sockets = {{FakeProc{}, FakeProc{}}, {FakeProc{}}};
  g_smi.sockets[1][0].gpu = false;
  uint32_t n = 0;
  EXPECT_EQ(amd_smi_getNumDevice(&n), ncclSuccess);
  EXPECT_EQ(n, 3u);
}

TEST_F(AmdsmiWrapMicrotest, NumDevice_LibFailureIsInternalError) {
  InitFor(/*lib=*/true);
  g_smi.sockets = {{FakeProc{}}};
  g_smi.procsStatus = AMDSMI_STATUS_INVAL;
  uint32_t n = 0;
  EXPECT_EQ(amd_smi_getNumDevice(&n), ncclInternalError);
}

TEST_F(AmdsmiWrapMicrotest, NumDevice_ArsmiReportsItsCount) {
  InitFor(/*lib=*/false);
  g_arsmi.numDevices = 8;
  uint32_t n = 0;
  EXPECT_EQ(amd_smi_getNumDevice(&n), ncclSuccess);
  EXPECT_EQ(n, 8u);
  g_arsmi.numDevicesResult = 2;
  EXPECT_EQ(amd_smi_getNumDevice(&n), ncclInternalError);
}

// amd_smi_getDevicePciBusIdString

TEST_F(AmdsmiWrapMicrotest, PciBusIdString_Wsl2AsksHip) {
  UseWsl2();
  InitFor(/*lib=*/false);
  g_hipDeviceGetPCIBusId = [](char* busId, int len, int dev) {
    std::snprintf(busId, len, "wsl-%d", dev);
    return hipSuccess;
  };
  char busId[32] = {};
  EXPECT_EQ(amd_smi_getDevicePciBusIdString(2, busId, sizeof(busId)), ncclSuccess);
  EXPECT_STREQ(busId, "wsl-2");
}

// Domain from bits [63:32], bus [15:8], device [7:3], function [2:0]; the source comment says
// the domain is [63:16].
TEST_F(AmdsmiWrapMicrotest, PciBusIdString_ArsmiFormatsTheBdf) {
  InitFor(/*lib=*/false);
  g_arsmi.bdfIds = {0, 0x0000002A0000C3DBull};
  char busId[32] = {};
  EXPECT_EQ(amd_smi_getDevicePciBusIdString(1, busId, sizeof(busId)), ncclSuccess);
  EXPECT_STREQ(busId, "002a:c3:1b.3");
  g_arsmi.pciIdResult = 1;
  EXPECT_EQ(amd_smi_getDevicePciBusIdString(1, busId, sizeof(busId)), ncclInternalError);
}

TEST_F(AmdsmiWrapMicrotest, PciBusIdString_LibFindsTheGpuByHipId) {
  InitFor(/*lib=*/true);
  TwoGpusBehindACpu();
  char busId[32] = {};
  ASSERT_EQ(amd_smi_getDevicePciBusIdString(1, busId, sizeof(busId)), ncclSuccess);
  EXPECT_STREQ(busId, "1234:00:1a.2");  // bdfId 0x1234000000d2 of the GPU, not the CPU
}

TEST_F(AmdsmiWrapMicrotest, PciBusIdString_LibDeviceNotFoundIsInternalError) {
  InitFor(/*lib=*/true);
  TwoGpusBehindACpu();
  char busId[32] = {};
  EXPECT_EQ(amd_smi_getDevicePciBusIdString(5, busId, sizeof(busId)), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_ERROR, "device index 5 not found"));
}

TEST_F(AmdsmiWrapMicrotest, PciBusIdString_LibBdfQueryFailureIsInternalError) {
  InitFor(/*lib=*/true);
  TwoGpusBehindACpu();
  g_smi.bdfStatus = AMDSMI_STATUS_INVAL;
  char busId[32] = {};
  EXPECT_EQ(amd_smi_getDevicePciBusIdString(0, busId, sizeof(busId)), ncclInternalError);
}

// The result is truncated to the caller's buffer.
TEST_F(AmdsmiWrapMicrotest, PciBusIdString_HonorsTheBufferLength) {
  InitFor(/*lib=*/false);
  g_arsmi.bdfIds = {0x0000002A0000C3DBull};
  char busId[6];
  std::memset(busId, 'x', sizeof(busId));
  EXPECT_EQ(amd_smi_getDevicePciBusIdString(0, busId, sizeof(busId)), ncclSuccess);
  EXPECT_STREQ(busId, "002a:");
}

// amd_smi_getDeviceIndexByPciBusId

TEST_F(AmdsmiWrapMicrotest, IndexByPciBusId_Wsl2AsksHip) {
  UseWsl2();
  InitFor(/*lib=*/false);
  std::string seen;
  g_hipDeviceGetByPCIBusId = [&seen](int* dev, const char* busId) {
    seen = busId;
    *dev = 4;
    return hipSuccess;
  };
  uint32_t idx = 0;
  EXPECT_EQ(amd_smi_getDeviceIndexByPciBusId("0000:c3:00.0", &idx), ncclSuccess);
  EXPECT_EQ(idx, 4u);
  EXPECT_EQ(seen, "0000:c3:00.0");
}

// ARSMI: the NCCL bus id is converted to the sysfs BDF encoding and matched against every device.
TEST_F(AmdsmiWrapMicrotest, IndexByPciBusId_ArsmiMatchesTheConvertedBdf) {
  InitFor(/*lib=*/false);
  // "0000:c3:1b.3" -> busIdToInt64 0xc31b3 -> ((0)<<12) + (0xc3000>>4) + (0x1b0>>1) + 3 = 0xc300 + 0xd8 + 3
  g_arsmi.numDevices = 3;
  g_arsmi.bdfIds = {0x1111, 0xc3db, 0x2222};
  uint32_t idx = 99;
  EXPECT_EQ(amd_smi_getDeviceIndexByPciBusId("0000:c3:1b.3", &idx), ncclSuccess);
  EXPECT_EQ(idx, 1u);
}

TEST_F(AmdsmiWrapMicrotest, IndexByPciBusId_ArsmiNoMatchIsInternalError) {
  InitFor(/*lib=*/false);
  g_arsmi.numDevices = 2;
  g_arsmi.bdfIds = {0x1111, 0x2222};
  uint32_t idx = 99;
  EXPECT_EQ(amd_smi_getDeviceIndexByPciBusId("0000:c3:1b.3", &idx), ncclInternalError);
  EXPECT_EQ(idx, 99u);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "device index not found"));
}

// The lib backend splits NCCL's bus-id integer into amd_smi's BDF fields itself.
TEST_F(AmdsmiWrapMicrotest, IndexByPciBusId_LibPassesTheBdfFieldsAndReturnsTheHipId) {
  InitFor(/*lib=*/true);
  FakeProc gpu;
  gpu.hipId = 6;
  g_smi.sockets = {{gpu}};
  // busIdToInt64("002a:c3:0b.3") = 0x2ac30b3 -> function 3, device 0x0b, bus 0xc3, domain 0x2a
  amdsmi_bdf_t expected{};
  expected.function_number = 3;
  expected.device_number = 0x0b;
  expected.bus_number = 0xc3;
  expected.domain_number = 0x2a;
  g_smi.sockets[0][0].bdfId = expected.as_uint;
  uint32_t idx = 0;
  ASSERT_EQ(amd_smi_getDeviceIndexByPciBusId("002a:c3:0b.3", &idx), ncclSuccess);
  EXPECT_EQ(idx, 6u);
  EXPECT_EQ(g_smi.lastBdf.function_number, 3u);
  EXPECT_EQ(g_smi.lastBdf.device_number, 0x0bu);
  EXPECT_EQ(g_smi.lastBdf.bus_number, 0xc3u);
  EXPECT_EQ(g_smi.lastBdf.domain_number, 0x2au);
}

// `(busid & 0xff) >> 4` keeps only the device number's low hex digit (0x1b becomes 0x0b); a fix flips this pin.
TEST_F(AmdsmiWrapMicrotest, IndexByPciBusId_KnownBug_LibDropsDeviceHighDigit) {
  InitFor(/*lib=*/true);
  g_smi.fromBdfStatus = AMDSMI_STATUS_NOT_FOUND;
  uint32_t idx = 0;
  EXPECT_EQ(amd_smi_getDeviceIndexByPciBusId("002a:c3:1b.3", &idx), ncclInternalError);
  EXPECT_EQ(g_smi.lastBdf.device_number, 0x0bu);
  EXPECT_EQ(g_smi.lastBdf.bus_number, 0xc3u);
}

TEST_F(AmdsmiWrapMicrotest, IndexByPciBusId_LibNonGpuIsInternalError) {
  InitFor(/*lib=*/true);
  FakeProc cpu;
  cpu.gpu = false;
  g_smi.sockets = {{cpu}};
  amdsmi_bdf_t bdf{};
  bdf.bus_number = 0xc3;
  g_smi.sockets[0][0].bdfId = bdf.as_uint;
  uint32_t idx = 0;
  EXPECT_EQ(amd_smi_getDeviceIndexByPciBusId("0000:c3:00.0", &idx), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_ERROR, "device index not found"));
}

TEST_F(AmdsmiWrapMicrotest, IndexByPciBusId_LibLookupFailureIsInternalError) {
  InitFor(/*lib=*/true);
  g_smi.fromBdfStatus = AMDSMI_STATUS_NOT_FOUND;
  uint32_t idx = 0;
  EXPECT_EQ(amd_smi_getDeviceIndexByPciBusId("0000:c3:00.0", &idx), ncclInternalError);
}

// amd_smi_getLinkInfo

TEST_F(AmdsmiWrapMicrotest, LinkInfo_Wsl2IsOnePcieHop) {
  UseWsl2();
  InitFor(/*lib=*/false);
  amdsmi_link_type_t type = AMDSMI_LINK_TYPE_XGMI;
  int hops = 0, count = 0;
  EXPECT_EQ(amd_smi_getLinkInfo(0, 1, &type, &hops, &count), ncclSuccess);
  EXPECT_EQ(type, AMDSMI_LINK_TYPE_PCIE);
  EXPECT_EQ(hops, 1);
  EXPECT_EQ(count, 1);
}

// ARSMI's link-type enum is not value-compatible with amdsmi's, so each is mapped explicitly.
TEST_F(AmdsmiWrapMicrotest, LinkInfo_ArsmiMapsTheLinkType) {
  InitFor(/*lib=*/false);
  struct Case {
    ARSMI_IO_LINK_TYPE in;
    amdsmi_link_type_t out;
  };
  for (const Case& c : {Case{ARSMI_IOLINK_TYPE_PCIEXPRESS, AMDSMI_LINK_TYPE_PCIE},
                        Case{ARSMI_IOLINK_TYPE_XGMI, AMDSMI_LINK_TYPE_XGMI},
                        Case{ARSMI_IOLINK_TYPE_UNDEFINED, AMDSMI_LINK_TYPE_UNKNOWN}}) {
    SCOPED_TRACE(static_cast<int>(c.in));
    g_arsmi.link = ARSMI_linkInfo{};
    g_arsmi.link.type = c.in;
    amdsmi_link_type_t type = AMDSMI_LINK_TYPE_NOT_APPLICABLE;
    int hops = 0, count = 0;
    EXPECT_EQ(amd_smi_getLinkInfo(0, 1, &type, &hops, &count), ncclSuccess);
    EXPECT_EQ(type, c.out);
    EXPECT_EQ(hops, 2);
    EXPECT_EQ(count, 1);
  }
}

// An XGMI link with one of the known direct-link weights is one hop, its count the bandwidth ratio.
TEST_F(AmdsmiWrapMicrotest, LinkInfo_ArsmiDirectXgmiWeightsAreOneHop) {
  InitFor(/*lib=*/false);
  for (uint64_t weight : {15ull, 41ull, 13ull}) {
    SCOPED_TRACE(weight);
    g_arsmi.link = ARSMI_linkInfo{};
    g_arsmi.link.type = ARSMI_IOLINK_TYPE_XGMI;
    g_arsmi.link.weight = weight;
    g_arsmi.link.min_bandwidth = 50;
    g_arsmi.link.max_bandwidth = 200;
    amdsmi_link_type_t type;
    int hops = 0, count = 0;
    EXPECT_EQ(amd_smi_getLinkInfo(0, 1, &type, &hops, &count), ncclSuccess);
    EXPECT_EQ(hops, 1);
    EXPECT_EQ(count, 4);
  }
  // Any other weight is two hops, count 1; and a zero bandwidth leaves count at 1.
  g_arsmi.link.weight = 16;
  amdsmi_link_type_t type;
  int hops = 0, count = 0;
  EXPECT_EQ(amd_smi_getLinkInfo(0, 1, &type, &hops, &count), ncclSuccess);
  EXPECT_EQ(hops, 2);
  EXPECT_EQ(count, 1);
  g_arsmi.link.weight = 15;
  g_arsmi.link.min_bandwidth = 0;
  EXPECT_EQ(amd_smi_getLinkInfo(0, 1, &type, &hops, &count), ncclSuccess);
  EXPECT_EQ(hops, 1);
  EXPECT_EQ(count, 1);
  // A direct-link weight only counts on XGMI: a PCIe link with weight 15 stays two hops, count 1.
  g_arsmi.link.type = ARSMI_IOLINK_TYPE_PCIEXPRESS;
  g_arsmi.link.min_bandwidth = 50;
  hops = 0;
  count = 0;
  EXPECT_EQ(amd_smi_getLinkInfo(0, 1, &type, &hops, &count), ncclSuccess);
  EXPECT_EQ(type, AMDSMI_LINK_TYPE_PCIE);
  EXPECT_EQ(hops, 2);
  EXPECT_EQ(count, 1);
}

TEST_F(AmdsmiWrapMicrotest, LinkInfo_ArsmiFailureIsInternalError) {
  InitFor(/*lib=*/false);
  g_arsmi.linkResult = 3;
  amdsmi_link_type_t type;
  int hops = 0, count = 0;
  EXPECT_EQ(amd_smi_getLinkInfo(0, 1, &type, &hops, &count), ncclInternalError);
}

TEST_F(AmdsmiWrapMicrotest, LinkInfo_LibQueriesTheSrcAndDstGpus) {
  InitFor(/*lib=*/true);
  TwoGpusBehindACpu();
  FakeProc trailingCpu;  // after GPU1 too, so a last-match lookup without the type check picks it
  trailingCpu.gpu = false;
  trailingCpu.hipId = 1;
  g_smi.sockets[0].push_back(trailingCpu);
  g_smi.linkType = AMDSMI_LINK_TYPE_PCIE;
  g_smi.linkHops = 3;
  g_smi.minBw = 64;  // only XGMI uses the bandwidth ratio, so a PCIe link's count stays 1
  g_smi.maxBw = 448;
  amdsmi_link_type_t type;
  int hops = 0, count = 0;
  ASSERT_EQ(amd_smi_getLinkInfo(1, 0, &type, &hops, &count), ncclSuccess);
  EXPECT_EQ(g_smi.lastLinkSrc, &g_smi.sockets[0][2]);
  EXPECT_EQ(g_smi.lastLinkDst, &g_smi.sockets[0][1]);
  EXPECT_EQ(type, AMDSMI_LINK_TYPE_PCIE);
  EXPECT_EQ(hops, 3);
  EXPECT_EQ(count, 1);
}

// XGMI links report weight 0, so their count comes from the max/min bandwidth ratio instead.
TEST_F(AmdsmiWrapMicrotest, LinkInfo_LibXgmiCountIsTheBandwidthRatio) {
  InitFor(/*lib=*/true);
  TwoGpusBehindACpu();
  g_smi.linkType = AMDSMI_LINK_TYPE_XGMI;
  g_smi.minBw = 64;
  g_smi.maxBw = 448;
  amdsmi_link_type_t type;
  int hops = 0, count = 0;
  ASSERT_EQ(amd_smi_getLinkInfo(0, 1, &type, &hops, &count), ncclSuccess);
  EXPECT_EQ(count, 7);
  g_smi.minBw = 0;
  ASSERT_EQ(amd_smi_getLinkInfo(0, 1, &type, &hops, &count), ncclSuccess);
  EXPECT_EQ(count, 1);
}

// An unknown index is logged, then queried as a null handle and reported as success; a fix flips this pin.
TEST_F(AmdsmiWrapMicrotest, LinkInfo_KnownBug_LibUnknownIndexStillSucceeds) {
  InitFor(/*lib=*/true);
  TwoGpusBehindACpu();
  amdsmi_link_type_t type;
  int hops = 0, count = 0;
  EXPECT_EQ(amd_smi_getLinkInfo(0, 9, &type, &hops, &count), ncclSuccess);
  EXPECT_TRUE(Logged(NCCL_LOG_ERROR, "could not find processor handle for dstIndex: 9"));
  EXPECT_FALSE(Logged(NCCL_LOG_ERROR, "srcIndex"));
  EXPECT_EQ(g_smi.lastLinkDst, nullptr);
}

// amd_smi_getFirmwareVersion

TEST_F(AmdsmiWrapMicrotest, FirmwareVersion_InitFailureIsReturned) {
  g_arsmi.initResult = 1;
  uint64_t fw = 7;
  EXPECT_EQ(amd_smi_getFirmwareVersion(0, &fw), ncclInternalError);
  EXPECT_EQ(fw, 7u);
}

TEST_F(AmdsmiWrapMicrotest, FirmwareVersion_Wsl2IsZero) {
  UseWsl2();
  g_arsmi.fwVersion = 0x55;  // so only the WSL2 early return can produce 0
  uint64_t fw = 7;
  EXPECT_EQ(amd_smi_getFirmwareVersion(0, &fw), ncclSuccess);
  EXPECT_EQ(fw, 0u);
}

TEST_F(AmdsmiWrapMicrotest, FirmwareVersion_ArsmiReportsItsVersion) {
  g_arsmi.fwVersion = 0x55;
  uint64_t fw = 0;
  EXPECT_EQ(amd_smi_getFirmwareVersion(0, &fw), ncclSuccess);
  EXPECT_EQ(fw, 0x55u);
  g_arsmi.fwResult = 2;
  EXPECT_EQ(amd_smi_getFirmwareVersion(0, &fw), ncclInternalError);
}

TEST_F(AmdsmiWrapMicrotest, FirmwareVersion_LibReportsTheFirstEntry) {
  UseLib();
  TwoGpusBehindACpu();
  g_smi.sockets[0][2].fwVersion = 0x99;
  uint64_t fw = 0;
  EXPECT_EQ(amd_smi_getFirmwareVersion(1, &fw), ncclSuccess);
  EXPECT_EQ(fw, 0x99u);
}

TEST_F(AmdsmiWrapMicrotest, FirmwareVersion_LibWithNoEntriesIsZero) {
  UseLib();
  TwoGpusBehindACpu();
  g_smi.sockets[0][1].numFwInfo = 0;
  g_smi.sockets[0][1].fwVersion = 0x99;
  uint64_t fw = 7;
  EXPECT_EQ(amd_smi_getFirmwareVersion(0, &fw), ncclSuccess);
  EXPECT_EQ(fw, 0u);
}

TEST_F(AmdsmiWrapMicrotest, FirmwareVersion_LibFailures) {
  UseLib();
  TwoGpusBehindACpu();
  uint64_t fw = 0;
  EXPECT_EQ(amd_smi_getFirmwareVersion(5, &fw), ncclInternalError);  // no such GPU
  g_smi.fwStatus = AMDSMI_STATUS_INVAL;
  EXPECT_EQ(amd_smi_getFirmwareVersion(0, &fw), ncclInternalError);
  g_smi.fwStatus = AMDSMI_STATUS_SUCCESS;
  pfn_amdsmi_get_fw_info = nullptr;
  EXPECT_EQ(amd_smi_getFirmwareVersion(0, &fw), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_ERROR, "amdsmi_get_fw_info symbol not loaded"));
}

// amd_smi_ensureFabricInitialized and the fabric accessors

namespace {
ARSMI_fabricInfo ArsmiFabric(int supported, uint32_t accelId, uint32_t bw) {
  ARSMI_fabricInfo f{};
  f.supported = supported;
  f.fabric_type = ARSMI_FABRIC_TYPE_UALLINK;
  f.accel_state = ARSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE;
  f.accel_id = accelId;
  f.bandwidth = bw;
  f.latency = 900;
  f.vpod_id = 3;
  f.vpod_size = 8;
  f.ppod_size = 64;
  for (int i = 0; i < 16; ++i) f.ppod_id[i] = static_cast<uint8_t>(i + 1);
  return f;
}
}  // namespace

TEST_F(AmdsmiWrapMicrotest, Fabric_InitFailureIsCachedForTheThread) {
  g_arsmi.initResult = 1;
  EXPECT_EQ(amd_smi_ensureFabricInitialized(), ncclInternalError);
  g_arsmi.initResult = 0;
  EXPECT_EQ(amd_smi_ensureFabricInitialized(), ncclInternalError);
}

TEST_F(AmdsmiWrapMicrotest, Fabric_Wsl2HasNoFabric) {
  UseWsl2();
  EXPECT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  EXPECT_EQ(amdsmiFabricDeviceCount, 0);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "skipped: WSL2"));
}

TEST_F(AmdsmiWrapMicrotest, Fabric_NoDevicesIsSuccessWithAnEmptyCache) {
  EXPECT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  EXPECT_EQ(amdsmiFabricDeviceCount, 0);
}

TEST_F(AmdsmiWrapMicrotest, Fabric_TooManyDevicesIsInternalError) {
  g_arsmi.numDevices = amdsmiFabricMaxDevices + 1;
  EXPECT_EQ(amd_smi_ensureFabricInitialized(), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "exceeds internal maximum"));
  Reset();  // the boundary itself fits
  g_arsmi.numDevices = amdsmiFabricMaxDevices;
  EXPECT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  EXPECT_EQ(amdsmiFabricDeviceCount, amdsmiFabricMaxDevices);
}

TEST_F(AmdsmiWrapMicrotest, Fabric_SysfsPathCopiesEachDevicesFields) {
  g_arsmi.numDevices = 2;
  g_arsmi.fabric[0] = ArsmiFabric(1, 11, 400000);  // device 1 has no ualink dir
  ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  ASSERT_EQ(amdsmiFabricDeviceCount, 2);
  const auto& d0 = amdsmiFabricDevices[0];
  EXPECT_TRUE(d0.fabricSupported);
  EXPECT_EQ(d0.fabricType, AMDSMI_FABRIC_TYPE_UALLINK);
  EXPECT_EQ(d0.state, AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE);
  EXPECT_EQ(d0.acceleratorId, 11u);
  EXPECT_EQ(d0.bandwidth, 400000u);
  EXPECT_EQ(d0.latency, 900u);
  EXPECT_EQ(d0.cliqueId, 3u);
  EXPECT_EQ(d0.vpodSize, 8u);
  EXPECT_EQ(d0.ppodSize, 64u);
  EXPECT_EQ(d0.clusterUuid[0], 1);
  EXPECT_EQ(d0.clusterUuid[15], 16);
  EXPECT_FALSE(amdsmiFabricDevices[1].fabricSupported);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "UALLink fabric detected (sysfs)"));
}

// Discovery runs once; later calls, on any thread, read the cache.
TEST_F(AmdsmiWrapMicrotest, Fabric_DiscoveryRunsOnce) {
  g_arsmi.numDevices = 1;
  g_arsmi.fabric[0] = ArsmiFabric(1, 11, 100);
  ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  g_arsmi.fabric[0] = ArsmiFabric(1, 22, 200);
  ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  std::thread other([] { EXPECT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess); });
  other.join();
  EXPECT_EQ(amdsmiFabricDevices[0].acceleratorId, 11u);
}

TEST_F(AmdsmiWrapMicrotest, Fabric_LibPathReadsTheTypedStruct) {
  if (!kTypedFabricLayout) {
    GTEST_SKIP() << "this build's amd_smi header uses the extended-union layout, which always reads sysfs";
  }
  UseLib();
  TwoGpusBehindACpu();
  g_smi.sockets[0].erase(g_smi.sockets[0].begin());  // GPUs only: getNumDevice counts every processor
  FakeProc& g1 = g_smi.sockets[0][1];
  g1.accelId = 21;
  g1.bandwidth = 800000;
  g1.latency = 700;
  g1.vpodId = 5;
  g1.vpodSize = 16;
  g1.ppodSize = 128;
  g1.ppodId[0] = 0xAB;
  g1.fabricType = AMDSMI_FABRIC_TYPE_UALLINK;
  g_smi.sockets[0][0].accelState = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ERROR;  // not usable
  ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  ASSERT_EQ(amdsmiFabricDeviceCount, 2);
  EXPECT_FALSE(amdsmiFabricDevices[0].fabricSupported);
  const auto& d1 = amdsmiFabricDevices[1];
  EXPECT_TRUE(d1.fabricSupported);
  EXPECT_EQ(d1.fabricType, AMDSMI_FABRIC_TYPE_UALLINK);
  EXPECT_EQ(d1.acceleratorId, 21u);
  EXPECT_EQ(d1.bandwidth, 800000u);
  EXPECT_EQ(d1.latency, 700u);
  EXPECT_EQ(d1.cliqueId, 5u);
  EXPECT_EQ(d1.vpodSize, 16u);
  EXPECT_EQ(d1.ppodSize, 128u);
  EXPECT_EQ(d1.clusterUuid[0], 0xAB);
  EXPECT_EQ(g_arsmi.fabric.size(), 0u);
  EXPECT_FALSE(Logged(NCCL_LOG_WARN, "falling back to sysfs"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "GPU 1: UALLink fabric detected - "));
}

// A version the library claims but RCCL cannot read is refused; the unreported sentinel reads as v1.
TEST_F(AmdsmiWrapMicrotest, Fabric_LibVersionGate) {
  if (!kTypedFabricLayout) GTEST_SKIP() << "extended-union layout reads sysfs";
  UseLib();
  FakeProc a, b;
  a.hipId = 0;
  a.fabricVersion = 7;
  b.hipId = 1;
  b.fabricVersion = kAmdSmiFabricVersionUnreported;
  g_smi.sockets = {{a, b}};
  ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  EXPECT_FALSE(amdsmiFabricDevices[0].fabricSupported);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "unexpected fabric info version 7 for device 0"));
  EXPECT_TRUE(amdsmiFabricDevices[1].fabricSupported);
}

TEST_F(AmdsmiWrapMicrotest, Fabric_LibPerDeviceFailureMarksOnlyThatDeviceUnsupported) {
  if (!kTypedFabricLayout) GTEST_SKIP() << "extended-union layout reads sysfs";
  UseLib();
  FakeProc a, b;
  a.hipId = 0;
  b.hipId = 1;
  b.fabricOk = false;
  g_smi.sockets = {{a, b}};
  ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  EXPECT_TRUE(amdsmiFabricDevices[0].fabricSupported);
  EXPECT_FALSE(amdsmiFabricDevices[1].fabricSupported);
}

// Each way the ABI probe can leave the library's layout unconfirmed falls back to sysfs.
TEST_F(AmdsmiWrapMicrotest, Fabric_LibProbeFallsBackToSysfs) {
  struct Case {
    const char* name;
    std::function<void()> arrange;
    const char* warn;
  };
  const size_t otherExtent = amdSmiFabricLayoutIs8Gpu ? kExtent16Gpu : kExtent8Gpu;
  const std::vector<Case> cases = {
    {"probe call fails", [] { g_smi.fabricProbeStatus = AMDSMI_STATUS_NOT_SUPPORTED; }, "the probe call returned"},
    // 300 overruns the 8-GPU object but stops short of the 16-GPU one: no known library writes that.
    {"unknown extent", [] { g_smi.fabricExtent = 300; }, "matches no layout RCCL knows"},
    {"extended union", [] { g_smi.fabricExtent = kExtentExtendedUnion; }, "wrote only the v1 payload"},
    {"other layout", [otherExtent] { g_smi.fabricExtent = otherExtent; }, "ABI mismatch"},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    if (std::strcmp(c.name, "other layout") == 0 && !kTypedFabricLayout) continue;
    Reset();
    UseLib();
    FakeProc a;
    g_smi.sockets = {{a}};
    g_arsmi.fabric[0] = ArsmiFabric(1, 33, 100);
    c.arrange();
    ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
    EXPECT_TRUE(Logged(NCCL_LOG_WARN, c.warn));
    EXPECT_TRUE(Logged(NCCL_LOG_WARN, "falling back to sysfs"));
    EXPECT_EQ(amdsmiFabricDevices[0].acceleratorId, 33u);  // came from ARSMI
  }
}

TEST_F(AmdsmiWrapMicrotest, Fabric_LibWithoutAProbeHandleFallsBackToSysfs) {
  UseLib();
  FakeProc a;
  a.hipId = 4;  // no GPU answers to hip id 0
  g_smi.sockets = {{a}};
  g_arsmi.fabric[0] = ArsmiFabric(1, 44, 100);
  ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "no processor handle to probe"));
  EXPECT_EQ(amdsmiFabricDevices[0].acceleratorId, 44u);
}

// Without the fabric entry point the probe is skipped and each device is marked unsupported.
TEST_F(AmdsmiWrapMicrotest, Fabric_LibWithoutTheFabricSymbolMarksDevicesUnsupported) {
  UseLib();
  FakeProc a;
  g_smi.sockets = {{a}};
  g_arsmi.fabric[0] = ArsmiFabric(1, 44, 100);
  pfn_amdsmi_get_gpu_fabric_info = nullptr;
  ASSERT_EQ(amd_smi_ensureFabricInitialized(), ncclSuccess);
  EXPECT_FALSE(amdsmiFabricDevices[0].fabricSupported);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "fabric functions not loaded for device 0"));
}

TEST_F(AmdsmiWrapMicrotest, FabricAccessors_ReadTheCacheWithBoundsChecks) {
  g_arsmi.numDevices = 2;
  g_arsmi.fabric[0] = ArsmiFabric(1, 11, 400000);
  g_arsmi.fabric[1] = ArsmiFabric(0, 12, 300000);  // present but not usable

  bool supported = false;
  ASSERT_EQ(amd_smi_isFabricSupported(0, &supported), ncclSuccess);
  EXPECT_TRUE(supported);
  ASSERT_EQ(amd_smi_isFabricSupported(1, &supported), ncclSuccess);
  EXPECT_FALSE(supported);
  // Past the device count but populated, so only the bounds checks keep index 2 out.
  amdsmiFabricDevices[2].fabricSupported = true;
  amdsmiFabricDevices[2].bandwidth = 500000;
  supported = true;
  ASSERT_EQ(amd_smi_isFabricSupported(2, &supported), ncclSuccess);
  EXPECT_FALSE(supported);

  amdsmiFabricDeviceInfo info{};
  ASSERT_EQ(amd_smi_getFabricDeviceInfo(0, &info), ncclSuccess);
  EXPECT_EQ(info.acceleratorId, 11u);
  EXPECT_EQ(amd_smi_getFabricDeviceInfo(2, &info), ncclInvalidArgument);

  uint32_t bw = 7;
  ASSERT_EQ(amd_smi_getFabricBandwidth(0, &bw), ncclSuccess);
  EXPECT_EQ(bw, 400000u);
  ASSERT_EQ(amd_smi_getFabricBandwidth(1, &bw), ncclSuccess);  // unsupported: 0, not its bandwidth
  EXPECT_EQ(bw, 0u);
  bw = 7;
  ASSERT_EQ(amd_smi_getFabricBandwidth(2, &bw), ncclSuccess);
  EXPECT_EQ(bw, 0u);
}

TEST_F(AmdsmiWrapMicrotest, FabricAccessors_PropagateAnInitFailure) {
  g_arsmi.initResult = 1;
  bool supported = true;
  EXPECT_EQ(amd_smi_isFabricSupported(0, &supported), ncclInternalError);
  amdsmiFabricDeviceInfo info{};
  EXPECT_EQ(amd_smi_getFabricDeviceInfo(0, &info), ncclInternalError);
  uint32_t bw = 0;
  EXPECT_EQ(amd_smi_getFabricBandwidth(0, &bw), ncclInternalError);
}

// Fabric telemetry

TEST_F(AmdsmiWrapMicrotest, Telemetry_IsAmdSmiLibOnly) {
  amdsmi_fabric_telemetry_t* t = nullptr;
  auto* dummy = reinterpret_cast<amdsmi_fabric_telemetry_t*>(0x2000);
  EXPECT_EQ(amd_smi_allocFabricTelemetry(0, 1, &t), ncclSystemError);
  EXPECT_EQ(amd_smi_getFabricTelemetryData(0, dummy), ncclSystemError);
  EXPECT_EQ(amd_smi_freeFabricTelemetry(0, dummy), ncclSystemError);
  EXPECT_EQ(g_smi.telemAllocCalls + g_smi.telemGetCalls + g_smi.telemFreeCalls, 0);
}

TEST_F(AmdsmiWrapMicrotest, Telemetry_NullBufferIsRejected) {
  UseLib();
  TwoGpusBehindACpu();
  EXPECT_EQ(amd_smi_getFabricTelemetryData(0, nullptr), ncclSystemError);
  EXPECT_EQ(amd_smi_freeFabricTelemetry(0, nullptr), ncclSystemError);
  EXPECT_EQ(g_smi.telemGetCalls + g_smi.telemFreeCalls, 0);
}

TEST_F(AmdsmiWrapMicrotest, Telemetry_CallsTheLibraryWithTheDevicesHandle) {
  UseLib();
  TwoGpusBehindACpu();
  amdsmi_fabric_telemetry_t* t = nullptr;
  ASSERT_EQ(amd_smi_allocFabricTelemetry(1, 0x6, &t), ncclSuccess);
  EXPECT_EQ(t, reinterpret_cast<amdsmi_fabric_telemetry_t*>(0x1000));
  EXPECT_EQ(g_smi.lastTelemHandle, &g_smi.sockets[0][2]);
  EXPECT_EQ(g_smi.lastTelemMask, 0x6u);
  g_smi.lastTelemHandle = nullptr;
  ASSERT_EQ(amd_smi_getFabricTelemetryData(1, t), ncclSuccess);
  EXPECT_EQ(g_smi.lastTelemHandle, &g_smi.sockets[0][2]);
  g_smi.lastTelemHandle = nullptr;
  ASSERT_EQ(amd_smi_freeFabricTelemetry(1, t), ncclSuccess);
  EXPECT_EQ(g_smi.lastTelemHandle, &g_smi.sockets[0][2]);
}

TEST_F(AmdsmiWrapMicrotest, Telemetry_FailuresAreReturned) {
  UseLib();
  TwoGpusBehindACpu();
  amdsmi_fabric_telemetry_t* t = nullptr;
  auto* dummy = reinterpret_cast<amdsmi_fabric_telemetry_t*>(0x2000);
  EXPECT_EQ(amd_smi_allocFabricTelemetry(9, 1, &t), ncclInternalError);  // no such GPU
  EXPECT_EQ(amd_smi_getFabricTelemetryData(9, dummy), ncclInternalError);
  EXPECT_EQ(amd_smi_freeFabricTelemetry(9, dummy), ncclInternalError);
  g_smi.telemStatus = AMDSMI_STATUS_INVAL;
  EXPECT_EQ(amd_smi_allocFabricTelemetry(0, 1, &t), ncclInternalError);
  EXPECT_EQ(amd_smi_getFabricTelemetryData(0, dummy), ncclInternalError);
  EXPECT_EQ(amd_smi_freeFabricTelemetry(0, dummy), ncclInternalError);
}

// amd_smi_fabricTelemIdToString

TEST_F(AmdsmiWrapMicrotest, TelemIdToString_WithoutTheSymbolAsksArsmi) {
  pfn_amdsmi_fabric_telem_id_to_string = nullptr;
  EXPECT_STREQ(amd_smi_fabricTelemIdToString(5), "ARSMI_NAME");
}

// amd_smi 27+ and an unknown version use the out-parameter form.
TEST_F(AmdsmiWrapMicrotest, TelemIdToString_OutParamForm) {
  for (uint32_t major : {kAmdSmiLibVersionUnknown, 27u, 30u}) {
    SCOPED_TRACE(major);
    amdSmiLibMajor.store(major);
    g_smi.telemNameStatus = AMDSMI_STATUS_SUCCESS;
    g_smi.telemName = "OUTPARAM_NAME";
    EXPECT_STREQ(amd_smi_fabricTelemIdToString(5), "OUTPARAM_NAME");
    g_smi.telemNameStatus = AMDSMI_STATUS_INVAL;
    EXPECT_STREQ(amd_smi_fabricTelemIdToString(5), "UNKNOWN");
    g_smi.telemNameStatus = AMDSMI_STATUS_SUCCESS;
    g_smi.telemName = nullptr;
    EXPECT_STREQ(amd_smi_fabricTelemIdToString(5), "UNKNOWN");
  }
}

// Before 27 the symbol returns the name, so it is called through that signature instead.
TEST_F(AmdsmiWrapMicrotest, TelemIdToString_LegacyFormBeforeMajor27) {
  amdSmiLibMajor.store(26);
  pfn_amdsmi_fabric_telem_id_to_string =
    reinterpret_cast<decltype(pfn_amdsmi_fabric_telem_id_to_string)>(FakeLegacyTelemIdToString);
  g_legacyTelemName = "LEGACY_NAME";
  EXPECT_STREQ(amd_smi_fabricTelemIdToString(5), "LEGACY_NAME");
  g_legacyTelemName = nullptr;
  EXPECT_STREQ(amd_smi_fabricTelemIdToString(5), "UNKNOWN");
}
