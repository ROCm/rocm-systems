/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the init-only fake seams. See init_fakes.h. The macro
// `getenv` is NOT active in this translation unit, so micro_getenv() can call
// the real libc getenv() as its default.

#include "init_fakes.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include "recorder.h"

namespace {
// Scripted environment overrides. When a name is present, its value (which may
// be an explicit "absent" -> nullptr) is returned; otherwise fall through to
// the real getenv so unrelated reads keep working.
std::unordered_map<std::string, std::string>& microEnvMap() {
  static std::unordered_map<std::string, std::string> m;
  return m;
}
}  // namespace

const char* micro_getenv(const char* name) {
  if (name != nullptr) {
    auto& m = microEnvMap();
    auto it = m.find(name);
    if (it != m.end()) {
      return it->second.c_str();
    }
  }
  return std::getenv(name);
}

void SetMicroEnv(const char* name, const char* value) {
  if (name != nullptr && value != nullptr) {
    microEnvMap()[name] = value;
  }
}

void ClearMicroEnv() { microEnvMap().clear(); }

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
// the getters / version / async-error). More overloads are added as Tier-D
// (InitAll/Destroy/InitRank) lands.
// -------------------------------------------------------------------------
namespace rccl {
Recorder::Recorder() {}
Recorder::~Recorder() {}
Recorder& Recorder::instance() {
  static Recorder inst;
  return inst;
}
void Recorder::record(const char*) {}
}  // namespace rccl

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
ncclResult_t ncclGpuGdrSupport(struct ncclComm*, int* gdrSupport) {
  if (gdrSupport) *gdrSupport = 0;
  return ncclSuccess;
}
ncclResult_t rocmLibraryInit(void) { return ncclSuccess; }
uint64_t ncclOsGetPid() { return 4321; }
// DMA-BUF export function pointer (dmaBufSupported gate): NULL -> unsupported.
// Global-scope name is unmangled, so an untyped definition satisfies the ref.
void* pfn_hsa_amd_portable_export_dmabuf = nullptr;

// The NCCL_API dispatch symbol ncclCommGetAsyncError is emitted outside init.cc
// (the api-trace layer, not linked here); ncclCommEnsureReady calls it. Route it
// to the in-TU _impl (defined in the init-test.cc object via the UUT include).
// nccl.h (via nccl_fakes.h) supplies the public declaration + its linkage.
extern ncclResult_t ncclCommGetAsyncError_impl(ncclComm_t comm, ncclResult_t* asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  return ncclCommGetAsyncError_impl(comm, asyncError);
}

void ResetInitFakes() {
  ResetHipFakes();
  ResetNcclFakes();
  ClearMicroEnv();
  g_ginHasError = false;
  g_validHsaScratch = true;
  g_firmwareVersion = 0;
}
