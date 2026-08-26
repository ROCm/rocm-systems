/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the init-only fake seams. See init_fakes.h.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // RTLD_NEXT
#endif
#include <dlfcn.h>

#include "init_fakes.h"

#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>

#include "recorder.h"

namespace {
// A nullopt entry means "absent": it masks whatever the real environment holds. Unmapped names fall through.
std::unordered_map<std::string, std::optional<std::string>>& microEnvMap() {
  static std::unordered_map<std::string, std::optional<std::string>> m;
  return m;
}
// Resolved past our interposing definition below so the map-miss fallback doesn't recurse into ourselves.
char* real_getenv(const char* name) {
  using Fn = char* (*)(const char*);
  static Fn next = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "getenv"));
  return next ? next(name) : nullptr;
}
}  // namespace

const char* micro_getenv(const char* name) {
  if (name != nullptr) {
    auto& m = microEnvMap();
    auto it = m.find(name);
    if (it != m.end()) {
      return it->second ? it->second->c_str() : nullptr;
    }
  }
  return real_getenv(name);
}

// Link-level override rather than a scoped macro, so both getenv() and std::getenv() in the UUT are caught.
extern "C" char* getenv(const char* name) {
  return const_cast<char*>(micro_getenv(name));
}

// A null value means "absent", NOT "leave unmapped" -- leaving it unmapped would fall through to the real getenv.
void SetMicroEnv(const char* name, const char* value) {
  if (name == nullptr) return;
  if (value == nullptr) microEnvMap()[name] = std::nullopt;
  else microEnvMap()[name] = value;
}

void SetMicroEnvAbsent(const char* name) { SetMicroEnv(name, nullptr); }

void ClearMicroEnv() { microEnvMap().clear(); }

// Arming gethostname failure + reaching fillInfo latches getHostName's hostHash call_once, poisoning later tests.
namespace {
bool g_gethostnameFail = false;
bool g_dladdrFail = false;
size_t g_lastGethostnameLen = 0;
}  // namespace

void SetGethostnameFail(bool fail) { g_gethostnameFail = fail; }
void SetDladdrFail(bool fail) { g_dladdrFail = fail; }
size_t LastGethostnameLen() { return g_lastGethostnameLen; }

extern "C" int gethostname(char* name, size_t len) {
  using Fn = int (*)(char*, size_t);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "gethostname"));
  g_lastGethostnameLen = len;
  if (g_gethostnameFail) {
    errno = ENAMETOOLONG;
    return -1;
  }
  return real ? real(name, len) : -1;
}

extern "C" int dladdr(const void* addr, Dl_info* info) {
  using Fn = int (*)(const void*, Dl_info*);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "dladdr"));
  if (g_dladdrFail) return 0;  // dladdr reports failure as 0, not -1
  return real ? real(addr, info) : 0;
}

const char* ncclGetEnv(const char* name) { return micro_getenv(name); }

// ncclParam* referenced by init.cc but not declared inside it, so the redirected NCCL_PARAM does not cover them.
int64_t ncclParamLaunchOrderImplicit() { return g_loadParam("LAUNCH_ORDER_IMPLICIT", 0); }
int64_t ncclParamNvlsEnable() { return g_loadParam("NVLS_ENABLE", 2); }
int64_t ncclParamNvtxDisable() { return g_loadParam("NVTX_DISABLE", 0); }
int64_t ncclParamPatEnable() { return g_loadParam("PAT_ENABLE", 2); }
int64_t ncclParamSingleProcMemRegEnable() { return g_loadParam("SINGLE_PROC_MEM_REG_ENABLE", 1); }

// Recorder is pure instrumentation -> no-op fake.
namespace rccl {
Recorder::Recorder() {}
Recorder::~Recorder() {}
Recorder& Recorder::instance() {
  static Recorder inst;
  return inst;
}
void Recorder::record(const char*) {}
void Recorder::record(ncclComm_t*, int, const int*) {}
ncclResult_t Recorder::record(rcclCall_t, int, int, ncclUniqueId*, ncclComm_t, int) { return ncclSuccess; }
ncclResult_t Recorder::record(rcclCall_t, ncclComm_t) { return ncclSuccess; }
void Recorder::record(rcclCall_t, int, int, ncclUniqueId*, ncclConfig_t*, ncclComm_t) {}
}  // namespace rccl

ncclResult_t ncclGroupStartInternal() { return ncclSuccess; }
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t*) { return ncclSuccess; }
ncclResult_t ncclGetUniqueId(ncclUniqueId* id) {
  if (id) std::memset(id, 0, sizeof(*id));
  return ncclSuccess;
}

ncclResult_t ncclGroupJobAbort(struct ncclGroupJob*) { return ncclSuccess; }
ncclResult_t ncclGroupJobComplete(struct ncclGroupJob*) { return ncclSuccess; }

bool g_ginHasError = false;
ncclResult_t ncclGinQueryLastError(struct ncclGinState*, bool* hasError) {
  if (hasError) *hasError = g_ginHasError;
  return ncclSuccess;
}

// TODO: the real impls live in rccl_wrap.cc, which pulls in ce_coll/dda/sym_kernels/dev_runtime/strongstream.
void rcclSetDefaultBuffSizes(struct ncclComm*, int* defaults) {
  defaults[0] = 1 << 18;  // LL
  defaults[1] = 1 << 18;  // LL128
  defaults[2] = 1 << 22;  // SIMPLE
}
void rcclSetP2pNetChunkSize(struct ncclComm*, int& sz) { sz = 1 << 17; }

bool g_validHsaScratch = true;
int g_firmwareVersion = 0;
bool validHsaScratchEnvSetting(const char* /*hsaScratchEnv*/, int /*hipRuntimeVersion*/,
                               int /*firmwareVersion*/, const char* /*gcnArchName*/) {
  return g_validHsaScratch;
}
int getFirmwareVersion() { return g_firmwareVersion; }

struct amdsmiFabricDeviceInfo;
ncclResult_t amd_smi_getDeviceIndexByPciBusId(const char*, uint32_t* deviceIndex) {
  if (deviceIndex) *deviceIndex = static_cast<uint32_t>(-1);  // -1 -> skip fabric block
  return ncclSuccess;
}
ncclResult_t amd_smi_getFabricDeviceInfo(uint32_t, struct amdsmiFabricDeviceInfo*) {
  return ncclSuccess;
}
ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported) {
  if (supported) *supported = false;
  return ncclSuccess;
}
int g_gdrSupportValue = 0;
int g_gdrSupportCalls = 0;
ncclResult_t ncclGpuGdrSupport(struct ncclComm*, int* gdrSupport) {
  ++g_gdrSupportCalls;
  if (gdrSupport) *gdrSupport = g_gdrSupportValue;
  return ncclSuccess;
}
ncclResult_t rocmLibraryInit(void) { return ncclSuccess; }
uint64_t ncclOsGetPid() { return 4321; }
// dmaBufSupported gate: NULL -> unsupported.
PFN_hsa_amd_portable_export_dmabuf pfn_hsa_amd_portable_export_dmabuf = nullptr;

// The NCCL_API dispatch symbol is emitted by the api-trace layer, which is not linked here.
extern ncclResult_t ncclCommGetAsyncError_impl(ncclComm_t comm, ncclResult_t* asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  return ncclCommGetAsyncError_impl(comm, asyncError);
}

bool g_bootstrapNetInitFail = false;
ncclResult_t bootstrapNetInit() { return g_bootstrapNetInitFail ? ncclSystemError : ncclSuccess; }
void initEnv() {}
ncclResult_t ncclOsInitialize() { return ncclSuccess; }
void initNvtxRegisteredEnums() {}
ncclResult_t ncclEnvPluginInit(void) { return ncclSuccess; }
bool ncclIommuPassthroughOk(const char*) { return true; }
// ncclInit() strtok_r()s the /proc version read, so it must have >= 3 whitespace tokens.
ncclResult_t ncclTopoGetStrFromSys(const char* /*path*/, const char* fileName, char* strValue) {
  if (!strValue) return ncclSuccess;
  if (fileName && std::strcmp(fileName, "version") == 0)
    std::strcpy(strValue, "Linux version 6.8.0-microtest");
  else if (fileName && std::strcmp(fileName, "numa_balancing") == 0)
    std::strcpy(strValue, "0");
  else
    std::strcpy(strValue, "microtest");
  return ncclSuccess;
}

// ncclNetInit/ncclNetInitFromParent live in init-test.cc: they need the full ncclComm/ncclNet_t layout.
ncclResult_t g_ncclNetInitResult        = ncclSuccess;
ncclResult_t g_ncclGinInitResult        = ncclSuccess;
ncclResult_t g_ncclStrongStreamResult   = ncclSuccess;
ncclResult_t g_ncclMemManagerInitResult = ncclSuccess;
ncclResult_t g_amdSmiInitResult         = ncclSuccess;
// Defaults to failure so the bootstrapAllGather call sites no test reaches stay fail-fast.
std::function<ncclResult_t(void*, void*, int)> g_bootstrapAllGather =
    [](void*, void*, int) { return ncclInternalError; };

ncclResult_t g_bootstrapGetUniqueIdResult = ncclSuccess;
ncclResult_t g_bcastGrowHandleResult      = ncclSuccess;
uint64_t g_bootstrapHandleMagic           = 0xB007ULL;
int g_bcastGrowHandleCalls                = 0;
bool g_bcastGrowHandleIsRoot              = false;

ncclResult_t g_initChannelResult        = ncclSuccess;
int g_initChannelLastId                 = -1;
// Default 1 (!= VerSuccess) means "version unknown".
int g_getROCmVersionResult = 1;
unsigned int g_rocmVersionMajor = 0;
unsigned int g_rocmVersionMinor = 0;
unsigned int g_rocmVersionPatch = 0;

ncclResult_t ncclGinInit(struct ncclComm*) { return g_ncclGinInitResult; }
ncclResult_t ncclGinInitFromParent(struct ncclComm*, struct ncclComm*) { return g_ncclGinInitResult; }
ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream*) { return g_ncclStrongStreamResult; }
ncclResult_t amd_smi_init() { return g_amdSmiInitResult; }
size_t ncclOsGetPageSize() { return 4096; }
extern "C" ncclResult_t ncclMemManagerInit(struct ncclComm*) { return g_ncclMemManagerInitResult; }

ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream*) { return g_ncclStrongStreamResult; }

void InstallCommAllocSuccess() {
  g_ncclNetInitResult = ncclSuccess;
  g_ncclGinInitResult = ncclSuccess;
  g_ncclStrongStreamResult = ncclSuccess;
  g_ncclMemManagerInitResult = ncclSuccess;
  g_amdSmiInitResult = ncclSuccess;
  g_hipDeviceGetAttributeResult = hipSuccess;
  g_hipDeviceGetPCIBusIdResult  = hipSuccess;
  g_hipEventCreateResult        = hipSuccess;
  g_hipMemPoolResult            = hipSuccess;
  g_hipStreamCreateResult       = hipSuccess;
}

void InstallDevCommSetupSuccess() {
  InstallCommAllocSuccess();
  g_hipAsyncOpsResult = hipSuccess;
}

void ResetInitFakes() {
  ResetHipFakes();
  ResetNcclFakes();
  ClearMicroEnv();
  g_ginHasError = false;
  g_bootstrapNetInitFail = false;
  g_validHsaScratch = true;
  g_firmwareVersion = 0;
  g_gdrSupportValue = 0;
  g_gdrSupportCalls = 0;
  pfn_hsa_amd_portable_export_dmabuf = nullptr;
  g_ncclNetInitResult = ncclSuccess;
  g_ncclGinInitResult = ncclSuccess;
  g_ncclStrongStreamResult = ncclSuccess;
  g_ncclMemManagerInitResult = ncclSuccess;
  g_amdSmiInitResult = ncclSuccess;
  g_initChannelResult = ncclSuccess;
  g_initChannelLastId = -1;
  g_bootstrapGetUniqueIdResult = ncclSuccess;
  g_bcastGrowHandleResult = ncclSuccess;
  g_bootstrapHandleMagic = 0xB007ULL;
  g_bcastGrowHandleCalls = 0;
  g_bcastGrowHandleIsRoot = false;
  g_bootstrapAllGather = [](void*, void*, int) { return ncclInternalError; };
  g_gethostnameFail = false;
  g_dladdrFail = false;
  g_lastGethostnameLen = 0;
  g_getROCmVersionResult = 1;
  g_rocmVersionMajor = 0;
  g_rocmVersionMinor = 0;
  g_rocmVersionPatch = 0;
}
