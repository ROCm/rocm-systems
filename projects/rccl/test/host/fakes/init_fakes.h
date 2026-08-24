/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Init-specific fake layer for the host-only `rccl-UnitTestsMicroInit` binary
// (AICOMRCCL-1685). This binary compiles the hipified src/init.cc directly
// (via INIT_CC_PATH) and links NONE of librccl/HIP; every external symbol is
// satisfied by the fake layers (hip_fakes / nccl_fakes / init_fakes) or by the
// real host-only oracle TUs (argcheck.cc / archinfo.cc / utils.cc).
//
// This header is intentionally thin at bring-up: it re-exports the shared HIP
// and NCCL fake seams and adds the init-only controllable seams. It grows as
// the GPU decision inventory + per-test matrix are completed (see
// init_coverage_plan.md).

#ifndef RCCL_TEST_HOST_INIT_FAKES_H_
#define RCCL_TEST_HOST_INIT_FAKES_H_

#include "hip_fakes.h"
#include "nccl_fakes.h"

// -------------------------------------------------------------------------
// getenv seam. init.cc reads a couple of environment variables via
// libc getenv() directly (HSA_NO_SCRATCH_RECLAIM, HSA_FORCE_FINE_GRAIN_PCIE),
// which ncclGetEnv()/g_getEnv cannot control. init-test.cc activates a
// `#define getenv(n) micro_getenv(n)` ONLY around `#include INIT_CC_PATH`;
// micro_getenv lives here (macro inactive in this TU) and calls the real
// getenv by default. Tests script values with SetMicroEnv().
// -------------------------------------------------------------------------
const char* micro_getenv(const char* name);
void SetMicroEnv(const char* name, const char* value);  // scripts one var
void ClearMicroEnv();                                    // back to real getenv

// Controllable GIN error state: ncclGinQueryLastError() reports this. Tests set
// it to drive the ncclRemoteError precedence branch in ncclCommGetAsyncError.
extern bool g_ginHasError;

// checkHsaEnvSetting seams: validHsaScratchEnvSetting()'s verdict (true = OK,
// no WARN) and getFirmwareVersion()'s value.
extern bool g_validHsaScratch;
extern int g_firmwareVersion;

// fillInfo GDR fallback seam: ncclGpuGdrSupport() writes g_gdrSupportValue and
// bumps g_gdrSupportCalls, so tests can assert the fallback path was taken.
extern int g_gdrSupportValue;
extern int g_gdrSupportCalls;

// ncclInit()-tree seams run real ncclInit() host-only. bootstrapNetInit
// success is injectable so a (process-isolated) test can drive ncclInit failure.
extern bool g_bootstrapNetInitFail;

// -------------------------------------------------------------------------
// commAlloc() deep seams. These were fail-loud stubs; they are now
// controllable so commAlloc() runs host-only. Each returns its g_*Result
// (default ncclSuccess) so a test can inject a failure at exactly one check to
// cover that early-return arm. ncclNetInit installs a fake comm->ncclNet
// (name "microfake") on success. (ncclCreateSideStream is a static-inline in
// alloc.h and ncclCudaCompCap comes from the real utils.cc oracle -- both real,
// driven via the HIP device model, not faked here.)
// -------------------------------------------------------------------------
extern ncclResult_t g_ncclNetInitResult;
extern ncclResult_t g_ncclGinInitResult;
extern ncclResult_t g_ncclStrongStreamResult;
extern ncclResult_t g_ncclMemManagerInitResult;
extern ncclResult_t g_amdSmiInitResult;

// Enable the full commAlloc() happy path in one call: flips the HIP deep-path
// seams (attribute/PCIBusId/event/mempool/stream) to success and resets the
// nccl seams above to their success defaults. Call at the top of a commAlloc
// test, then inject a single failure to exercise a specific arm.
void InstallCommAllocSuccess();

// Enable the full devCommSetup() happy path: InstallCommAllocSuccess() plus the
// HIP async stream ops (thread-exchange / memsetAsync / memcpyAsync) that its
// alloc/copy templates drive. Call after building a comm via commAlloc().
void InstallDevCommSetupSuccess();

// Reset every init-layer fake to defaults. Cascades to ResetHipFakes() and
// ResetNcclFakes(). Called from the fixture TearDown().
void ResetInitFakes();

#endif  // RCCL_TEST_HOST_INIT_FAKES_H_
