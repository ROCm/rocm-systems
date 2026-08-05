/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the init-only fake seams. See init_fakes.h. The macro
// `getenv` is NOT active in this translation unit, so micro_getenv() can call
// the real libc getenv() as its default.

#include "init_fakes.h"

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
ncclResult_t ncclGinQueryLastError(struct ncclGinState*, bool* hasError) {
  if (hasError) *hasError = false;
  return ncclSuccess;
}

void ResetInitFakes() {
  ResetHipFakes();
  ResetNcclFakes();
  ClearMicroEnv();
}
