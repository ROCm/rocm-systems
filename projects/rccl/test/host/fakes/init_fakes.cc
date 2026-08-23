/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the init-only fake seams. See init_fakes.h.
//
// This TU interposes libc getenv() (see the extern "C" getenv below) so BOTH
// getenv() and std::getenv() in the unit-under-test route through the
// controllable microEnvMap; real_getenv() reaches the actual libc getenv via
// RTLD_NEXT for anything not scripted.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // RTLD_NEXT
#endif
#include <dlfcn.h>

#include "init_fakes.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <unistd.h>
#include <unordered_map>

#include "recorder.h"

namespace {
// Scripted environment overrides. When a name is present, its value (which may
// be an explicit "absent" -> nullptr) is returned; otherwise fall through to
// the real getenv so unrelated reads keep working. A nullopt entry is that
// explicit absence: it masks whatever the real environment holds, which is what
// lets a test cover an `if (getenv(...) == NULL)` arm hermetically.
std::unordered_map<std::string, std::optional<std::string>>& microEnvMap() {
  static std::unordered_map<std::string, std::optional<std::string>> m;
  return m;
}
// The real libc getenv, resolved past our interposing definition below so the
// map-miss fallback doesn't recurse into ourselves.
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

// Interpose libc getenv for the whole init binary. init.cc reads a few env vars
// via getenv()/std::getenv() directly; a link-level override (vs a scoped macro)
// catches both spellings -- including std::getenv under ENABLE_ROCSHMEM -- with
// no per-call-site macro. Unmapped names fall through to the real libc getenv,
// so gtest and other harness reads are unaffected. Tests script via SetMicroEnv().
extern "C" char* getenv(const char* name) {
  return const_cast<char*>(micro_getenv(name));
}

// A null value means "absent", the same as SetMicroEnvAbsent -- NOT "leave
// unmapped". Silently ignoring it would fall through to the real libc getenv,
// i.e. the exact opposite of what the caller asked for on a host that exports
// the name.
void SetMicroEnv(const char* name, const char* value) {
  if (name == nullptr) return;
  if (value == nullptr) microEnvMap()[name] = std::nullopt;
  else microEnvMap()[name] = value;
}

void SetMicroEnvAbsent(const char* name) { SetMicroEnv(name, nullptr); }

void ClearMicroEnv() { microEnvMap().clear(); }

// -------------------------------------------------------------------------
// gethostname / dladdr failure seams. showVersion() (init.cc:1012, :1016) falls
// back to "Unknown" when either fails, and both succeed in practice -- so the
// fallbacks are only reachable by interposing, the same extern "C" +
// dlsym(RTLD_NEXT) trick used for getenv above. Default is pure pass-through;
// the real symbol is resolved on first call so the lookup itself never runs
// while a failure is armed. ResetInitFakes() disarms both.
// -------------------------------------------------------------------------
namespace {
bool g_gethostnameFail = false;
bool g_dladdrFail = false;
}  // namespace

void SetGethostnameFail(bool fail) { g_gethostnameFail = fail; }
void SetDladdrFail(bool fail) { g_dladdrFail = fail; }

extern "C" int gethostname(char* name, size_t len) {
  using Fn = int (*)(char*, size_t);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "gethostname"));
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

// -------------------------------------------------------------------------
// Environment read: init.cc calls ncclGetEnv() for NCCL_* lookups. Route it
// through the same controllable map as micro_getenv (SetMicroEnv controls both).
// -------------------------------------------------------------------------
const char* ncclGetEnv(const char* name) { return micro_getenv(name); }

// -------------------------------------------------------------------------
// External ncclParam* referenced by init.cc but NOT defined via NCCL_PARAM in
// the UUT (the redirected NCCL_PARAM only covers params declared inside init.cc).
// Route through g_loadParam so tests can flip them per-case; distinct env keys.
// Defaults mirror the production NCCL_PARAM defaults.
// -------------------------------------------------------------------------
int64_t ncclParamLaunchOrderImplicit() { return g_loadParam("LAUNCH_ORDER_IMPLICIT", 0); }
int64_t ncclParamNvlsEnable() { return g_loadParam("NVLS_ENABLE", 2); }
int64_t ncclParamNvtxDisable() { return g_loadParam("NVTX_DISABLE", 0); }
int64_t ncclParamPatEnable() { return g_loadParam("PAT_ENABLE", 2); }
int64_t ncclParamSingleProcMemRegEnable() { return g_loadParam("SINGLE_PROC_MEM_REG_ENABLE", 1); }

// -------------------------------------------------------------------------
// Recorder: pure instrumentation -> no-op fake. Only the overloads reached by
// the currently-tested init.cc paths are defined (record(const char*) covers
// the getters / version / async-error). More overloads are added as deeper
// (InitAll/Destroy/InitRank) paths come under test.
// -------------------------------------------------------------------------
namespace rccl {
Recorder::Recorder() {}
Recorder::~Recorder() {}
Recorder& Recorder::instance() {
  static Recorder inst;
  return inst;
}
void Recorder::record(const char*) {}
void Recorder::record(ncclComm_t*, int, const int*) {}  // CommInitAll
ncclResult_t Recorder::record(rcclCall_t, int, int, ncclUniqueId*, ncclComm_t, int) { return ncclSuccess; }  // InitRankDev
ncclResult_t Recorder::record(rcclCall_t, ncclComm_t) { return ncclSuccess; }  // finalize/destroy
void Recorder::record(rcclCall_t, int, int, ncclUniqueId*, ncclConfig_t*, ncclComm_t) {}  // RankConfig
}  // namespace rccl

// Group boundary + unique-id seams reached by ncclCommInitAll / rank wrappers.
// No-op success (balanced start/end); ncclGetUniqueId zeroes the id.
ncclResult_t ncclGroupStartInternal() { return ncclSuccess; }
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t*) { return ncclSuccess; }
ncclResult_t ncclGetUniqueId(ncclUniqueId* id) {
  if (id) std::memset(id, 0, sizeof(*id));
  return ncclSuccess;
}

// -------------------------------------------------------------------------
// Group-job + GIN seams reached via ncclCommEnsureReady / ncclCommGetAsyncError.
// Success/no-op defaults; ncclGinQueryLastError reports "no error".
// -------------------------------------------------------------------------
ncclResult_t ncclGroupJobAbort(struct ncclGroupJob*) { return ncclSuccess; }
ncclResult_t ncclGroupJobComplete(struct ncclGroupJob*) { return ncclSuccess; }

bool g_ginHasError = false;
ncclResult_t ncclGinQueryLastError(struct ncclGinState*, bool* hasError) {
  if (hasError) *hasError = g_ginHasError;
  return ncclSuccess;
}

// computeBuffSizes seams: rcclSetDefaultBuffSizes fills the per-protocol default
// buffer sizes; rcclSetP2pNetChunkSize fills the multi-node net chunk size.
// Deterministic values let tests assert the assignment paths.
// TODO: real impls live in rccl_wrap.cc, which pulls in ce_coll/dda/sym_kernels/
// dev_runtime/strongstream (all unstubbed here). Swap these for the real thing
// once rccl_wrap.cc gets its own microtests and stub floor.
void rcclSetDefaultBuffSizes(struct ncclComm*, int* defaults) {
  defaults[0] = 1 << 18;  // LL
  defaults[1] = 1 << 18;  // LL128
  defaults[2] = 1 << 22;  // SIMPLE
}
void rcclSetP2pNetChunkSize(struct ncclComm*, int& sz) { sz = 1 << 17; }

// checkHsaEnvSetting seams (controllable). Defaults: valid setting, firmware 0.
bool g_validHsaScratch = true;
int g_firmwareVersion = 0;
bool validHsaScratchEnvSetting(const char* /*hsaScratchEnv*/, int /*hipRuntimeVersion*/,
                               int /*firmwareVersion*/, const char* /*gcnArchName*/) {
  return g_validHsaScratch;
}
int getFirmwareVersion() { return g_firmwareVersion; }
// Note: ncclCuMemEnable() is provided by nccl_fakes.cc (g_cuMemEnable seam);
// getHostHash/getPidHash come from the real utils.cc oracle -- not faked here.

// fillInfo downstream seams. gc-sections retains the whole fillInfo function
// (so these must LINK even when a test returns early); defaults keep the happy
// path benign: no fabric device, no cross-nic, no GDR.
struct amdsmiFabricDeviceInfo;  // opaque; only a pointer is used here
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
// DMA-BUF export function pointer (dmaBufSupported gate): NULL -> unsupported.
// The typedef is visible via the transitively-included rocmwrap.h, so define it
// with the real type.
PFN_hsa_amd_portable_export_dmabuf pfn_hsa_amd_portable_export_dmabuf = nullptr;

// The NCCL_API dispatch symbol ncclCommGetAsyncError is emitted outside init.cc
// (the api-trace layer, not linked here); ncclCommEnsureReady calls it. Route it
// to the in-TU _impl (defined in the init-test.cc object via the UUT include).
// nccl.h (via nccl_fakes.h) supplies the public declaration + its linkage.
extern ncclResult_t ncclCommGetAsyncError_impl(ncclComm_t comm, ncclResult_t* asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  return ncclCommGetAsyncError_impl(comm, asyncError);
}

// ncclInit()-tree seams -- controllable SUCCESS so real ncclInit() runs
// host-only (moved from the fail-loud floor). bootstrapNetInit failure is
// injectable to drive ncclInit's error arm (process-isolated tests).
bool g_bootstrapNetInitFail = false;
ncclResult_t bootstrapNetInit() { return g_bootstrapNetInitFail ? ncclSystemError : ncclSuccess; }
void initEnv() {}
ncclResult_t ncclOsInitialize() { return ncclSuccess; }
void initNvtxRegisteredEnums() {}
ncclResult_t ncclEnvPluginInit(void) { return ncclSuccess; }
bool ncclIommuPassthroughOk(const char*) { return true; }
// Canned /proc,/sys strings so ncclInit()'s tokenization stays well-formed
// (>=3 whitespace tokens for /proc version).
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

// -------------------------------------------------------------------------
// commAlloc() deep seams (controllable). Defaults succeed so real
// commAlloc() runs host-only; a test injects one failure to cover a specific
// early-return arm. ncclNetInit/ncclNetInitFromParent live in init-test.cc --
// they set comm->ncclNet, which needs the full ncclComm/ncclNet_t layout that
// only the UUT TU has. ncclCudaCompCap and the static-inline ncclCreateSideStream
// are the real code (utils.cc oracle / alloc.h), driven via the HIP device model.
// -------------------------------------------------------------------------
ncclResult_t g_ncclNetInitResult        = ncclSuccess;
ncclResult_t g_ncclGinInitResult        = ncclSuccess;
ncclResult_t g_ncclStrongStreamResult   = ncclSuccess;
ncclResult_t g_ncclMemManagerInitResult = ncclSuccess;
ncclResult_t g_amdSmiInitResult         = ncclSuccess;
// commGetSplitInfo() seam; the stub itself lives in bootstrap_stubs.cc. Default
// FAILS -- see init_fakes.h for why this one differs from g_initChannelResult.
std::function<ncclResult_t(void*, void*, int)> g_bootstrapAllGather =
    [](void*, void*, int) { return ncclInternalError; };

// ncclCommGetUniqueId() seams; the stubs live in bootstrap_stubs.cc.
ncclResult_t g_bootstrapGetUniqueIdResult = ncclSuccess;
ncclResult_t g_bcastGrowHandleResult      = ncclSuccess;
uint64_t g_bootstrapHandleMagic           = 0xB007ULL;
int g_bcastGrowHandleCalls                = 0;
bool g_bcastGrowHandleIsRoot              = false;

// setupChannel() seam; the stub itself lives in nccl_stubs.cc.
ncclResult_t g_initChannelResult        = ncclSuccess;
// showVersion() seam; the getROCmVersion stub also lives in nccl_stubs.cc.
// Default 1 (!= VerSuccess) preserves the stub's original "version unknown".
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

// devCommSetup() deep seam: the strong-stream sync on the exit path succeeds.
// (ncclCommPushCuda*Free are defined by init.cc itself -- real host code that
// pushes destructors onto comm->destructorHead, so they are not faked here.)
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
  g_bootstrapGetUniqueIdResult = ncclSuccess;
  g_bcastGrowHandleResult = ncclSuccess;
  g_bootstrapHandleMagic = 0xB007ULL;
  g_bcastGrowHandleCalls = 0;
  g_bcastGrowHandleIsRoot = false;
  g_bootstrapAllGather = [](void*, void*, int) { return ncclInternalError; };
  g_gethostnameFail = false;
  g_dladdrFail = false;
  g_getROCmVersionResult = 1;  // != VerSuccess
  g_rocmVersionMajor = 0;
  g_rocmVersionMinor = 0;
  g_rocmVersionPatch = 0;
}
