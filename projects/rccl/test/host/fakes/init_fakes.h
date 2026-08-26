/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Init-only fake seams for the host-only `rccl-UnitTestsMicroInit` binary. See test/host/MICROTEST_README.md.

#ifndef RCCL_TEST_HOST_INIT_FAKES_H_
#define RCCL_TEST_HOST_INIT_FAKES_H_

#include "hip_fakes.h"
#include "nccl_fakes.h"

// Returns a pointer into the map's own std::string: re-scripting a name invalidates a pointer a caller may still hold.
const char* micro_getenv(const char* name);
void SetMicroEnv(const char* name, const char* value);
void SetMicroEnvAbsent(const char* name);
void ClearMicroEnv();

void SetGethostnameFail(bool fail);
void SetDladdrFail(bool fail);
size_t LastGethostnameLen();

// Default 1 is != VerSuccess(0), so showVersion()'s runtime-ROCm block is skipped.
extern int g_getROCmVersionResult;
extern unsigned int g_rocmVersionMajor;
extern unsigned int g_rocmVersionMinor;
extern unsigned int g_rocmVersionPatch;

extern bool g_ginHasError;

extern bool g_validHsaScratch;
extern const char* g_lastHsaScratchEnv;  // hsaScratchEnv as passed to validHsaScratchEnvSetting
extern int g_firmwareVersion;

extern int g_gdrSupportValue;
extern int g_gdrSupportCalls;

extern bool g_bootstrapNetInitFail;

extern ncclResult_t g_ncclNetInitResult;
extern ncclResult_t g_ncclGinInitResult;
extern ncclResult_t g_ncclStrongStreamResult;
extern ncclResult_t g_ncclMemManagerInitResult;
extern ncclResult_t g_amdSmiInitResult;

// A std::function, not a result code: tests must write the allgathered (color, key) table into allData.
extern std::function<ncclResult_t(void* commState, void* allData, int size)>
    g_bootstrapAllGather;

extern ncclResult_t g_bootstrapGetUniqueIdResult;
extern ncclResult_t g_bcastGrowHandleResult;
extern uint64_t g_bootstrapHandleMagic;
extern int g_bcastGrowHandleCalls;
extern bool g_bcastGrowHandleIsRoot;

// The fake initChannel does NOT allocate ring->userRanks/rankToIndex like the real one; callers must supply storage.
extern ncclResult_t g_initChannelResult;
extern int g_initChannelLastId;

void InstallCommAllocSuccess();

void InstallDevCommSetupSuccess();

void ResetInitFakes();

#endif  // RCCL_TEST_HOST_INIT_FAKES_H_
