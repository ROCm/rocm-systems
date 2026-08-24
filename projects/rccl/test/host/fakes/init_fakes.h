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
// It re-exports the shared HIP and NCCL fake seams and adds the init-only
// controllable seams, growing as more of init.cc comes under test. The workflow
// for adding one is in test/host/MICROTEST_README.md ("Adding more controllable
// seams").

#ifndef RCCL_TEST_HOST_INIT_FAKES_H_
#define RCCL_TEST_HOST_INIT_FAKES_H_

#include <functional>
#include <string>
#include <vector>

#include "hip_fakes.h"
#include "nccl_fakes.h"
#include "os.h"  // ncclAffinity, for the initTransportsRank affinity seams below

struct ncclTopoSystem;

// -------------------------------------------------------------------------
// getenv seam. init.cc reads a couple of environment variables via
// libc getenv() directly (HSA_NO_SCRATCH_RECLAIM, HSA_FORCE_FINE_GRAIN_PCIE),
// which ncclGetEnv()/g_getEnv cannot control. init-test.cc activates a
// `#define getenv(n) micro_getenv(n)` ONLY around `#include INIT_CC_PATH`;
// micro_getenv lives here (macro inactive in this TU) and calls the real
// getenv by default. Tests script values with SetMicroEnv().
//
// Lifetime: micro_getenv returns a pointer into the map's own std::string, so
// re-scripting a name invalidates a pointer an earlier caller may still hold --
// getEnvCtaPolicyOnce holds its `env` across the whole parse. Do not re-script mid-call.
// -------------------------------------------------------------------------
const char* micro_getenv(const char* name);
void SetMicroEnv(const char* name, const char* value);  // nullptr value == absent
void SetMicroEnvAbsent(const char* name);                // readable alias for the above
void ClearMicroEnv();                                    // back to real getenv

// -------------------------------------------------------------------------
// gethostname / dladdr seams. showVersion() (init.cc:1012, :1016) falls back to
// "Unknown" when either fails; both succeed in practice, so those arms need the
// same extern "C" + dlsym(RTLD_NEXT) interposition as the getenv seam above.
// Default is pass-through; ResetInitFakes() disarms. LastGethostnameLen()
// exposes the `len` argument so a test can pin showVersion's sizeof(buf)-1.
// -------------------------------------------------------------------------
void SetGethostnameFail(bool fail);
void SetDladdrFail(bool fail);
size_t LastGethostnameLen();

// -------------------------------------------------------------------------
// showVersion() ROCm-version seam. The stub (nccl_stubs.cc) returns
// g_getROCmVersionResult and writes the three g_rocmVersion* values; the default
// of 1 is != VerSuccess(0), so the runtime-ROCm block is skipped as before.
// -------------------------------------------------------------------------
extern int g_getROCmVersionResult;
extern unsigned int g_rocmVersionMajor;
extern unsigned int g_rocmVersionMinor;
extern unsigned int g_rocmVersionPatch;

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

// -------------------------------------------------------------------------
// commGetSplitInfo() seam. Unlike the result-only seams below this is a
// std::function, because the ALLGATHERED TABLE IS THE ALGORITHM'S INPUT: a test
// has to write (color, key) pairs into allData, not merely pick a return code.
//
// Defaults to FAILURE, deliberately unlike g_initChannelResult. bootstrapAllGather
// has four call sites in init.cc (1466 and 1909 inside initTransportsRank, plus
// 2496); no test reaches the first two, and a failing default keeps them
// fail-fast per MICROTEST_README's "return failure loudly" rule. Tests that want
// success install a scripting lambda.
// -------------------------------------------------------------------------
extern std::function<ncclResult_t(void* commState, void* allData, int size)>
    g_bootstrapAllGather;

// ncclCommGetUniqueId() seams. bootstrapGetUniqueId was a fail-loud abort;
// bcastGrowHandle had no fake at all (it lives in src/bootstrap.cc, which this
// target does not compile) and only linked because --gc-sections dropped the
// whole function. Both default to success; on success bootstrapGetUniqueId
// stamps g_bootstrapHandleMagic into handle->magic so a test can prove the
// handle is what ends up memcpy'd into the caller's ncclUniqueId.
extern ncclResult_t g_bootstrapGetUniqueIdResult;
extern ncclResult_t g_bcastGrowHandleResult;
extern uint64_t g_bootstrapHandleMagic;
extern int g_bcastGrowHandleCalls;   // lets a test assert it was reached
extern bool g_bcastGrowHandleIsRoot; // and with which role

// setupChannel() seam. initChannel() (src/channel.cc:16) is unreachable host-only
// -- strong streams, memory stacks, ncclCudaCallocAsync -- so it stays faked, but
// injectable so a test can cover both the NCCLCHECK early-return arm and the
// ncclInProgress fall-through. NOTE: the real initChannel allocates
// ring->userRanks/rankToIndex (channel.cc:61-62); the fake does not, so a test
// calling setupChannel must point the ring at storage it owns.
extern ncclResult_t g_initChannelResult;
// Records the channelId setupChannel forwarded. Without this the fake ignores
// the argument entirely, so `initChannel(comm, channelId) -> initChannel(comm, 0)`
// is unobservable.
extern int g_initChannelLastId;

// -------------------------------------------------------------------------
// initTransportsRank() seams (init.cc:1386). All five were fail-loud stubs.
// ncclOsCpuCount is load-bearing: exit::2403 calls it on EVERY path, so nothing in the function was
// testable until it was seamed, and its counter is the only way to see that :1488 skips exit:.
// ncclTopoGetSystem stays defaulted to FAILURE on purpose -- it is the first call after the
// MNNVL/intra-proc block, so that default is what terminates the ladder and makes :1462-1565
// reachable. Its dumpXmlFile argument passes through so a test can tell :1573 from :1576.
// -------------------------------------------------------------------------
extern int g_ncclOsCpuCountValue;
extern int g_ncclOsCpuCountCalls;
// Every mask ncclOsCpuCount was handed, in call order: [0] is :1608, the last is exit::2403.
extern std::vector<ncclAffinity> g_ncclOsCpuCountMasks;
extern ncclResult_t g_ncclOsSetAffinityResult;
// Every mask handed to ncclOsSetAffinity, in call order: [0] is :1610, [1] is exit::2404. A single
// "last" slot is not enough -- the exit: write masks whatever :1610 forwarded.
extern std::vector<ncclAffinity> g_ncclOsSetAffinityMasks;
extern ncclResult_t g_ncclMnnvlCheckResult;
extern int g_ncclMnnvlCheckCalls;  // the oracle for the :1503-1509 enable/auto/disable logic
extern std::function<ncclResult_t(int*)> g_ncclGetUserP2pLevel;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTopoSystem**, const char*)> g_ncclTopoGetSystem;

// -------------------------------------------------------------------------
// Topology-detection / CPU-affinity seams (init.cc:1576-1648), rung 2 of the ladder.
// All default to success so a test can walk :1576-1648 and inject exactly one failure. ncclTopoCompute
// is the exception -- it defaults to FAILURE because it is now the terminator, the same role
// ncclTopoGetSystem played for rung 1. ncclTopoComputePaths gets a FailAt index rather than a result
// because :1591 and :1596 call it twice and a single knob cannot separate them.
// -------------------------------------------------------------------------
extern int g_tuningIndexValue;
extern std::string g_tuningIndexLastArch;  // :1577 forwards comm->archName; without this that is untested
extern int g_ncclTopoComputePathsCalls;
extern int g_ncclTopoComputePathsFailAt;   // -1 = never fail; 0 = the :1591 call, 1 = the :1596 one
extern ncclResult_t g_ncclTopoTrimSystemResult;
extern ncclResult_t g_ncclTopoSearchInitResult;
extern ncclResult_t g_ncclTopoComputeCommCPUResult;
extern ncclResult_t g_ncclTopoPrintResult;
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, ncclAffinity*)> g_ncclTopoGetCpuAffinity;
extern int g_ncclTopoGetCpuAffinityLastRank;
extern std::function<ncclResult_t(ncclAffinity*)> g_ncclOsGetAffinity;
extern ncclResult_t g_ncclNvlsInitResult;
extern int g_ncclNvlsInitCalls;
extern ncclResult_t g_ncclTopoComputeResult;
extern int g_ncclTopoComputeCalls;

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
