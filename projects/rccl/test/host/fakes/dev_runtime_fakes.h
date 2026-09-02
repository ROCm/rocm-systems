/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Controllable seams exposed by DevRuntimeTestsStubs.cc.
 *
 * Each hook defaults to the success behaviour the rest of the suite relies on.
 * A test that needs a HIP VMM call to fail installs its own via ScopedHook
 * (test/host/ScopedHook.h), which restores the previous one on scope exit.
 *************************************************************************/

#ifndef RCCL_TEST_HOST_FAKES_DEV_RUNTIME_FAKES_H_
#define RCCL_TEST_HOST_FAKES_DEV_RUNTIME_FAKES_H_

#include "nccl.h"          // ncclResult_t, for the proxy seam below
#include "gin/gin_host.h"  // NCCL_GIN_MAX_CONNECTIONS, ncclGinWindow_t

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>

extern std::function<hipError_t(hipMemAllocationProp*, hipMemGenericAllocationHandle_t)>
    g_devrHipMemGetAllocationPropertiesFromHandle;

extern std::function<hipError_t(void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
                                unsigned long long)>
    g_devrHipMemExportToShareableHandle;

extern std::function<hipError_t(void*, size_t, const hipMemAccessDesc*, size_t)> g_devrHipMemSetAccess;

extern std::function<hipError_t(size_t*, const hipMemAllocationProp*, hipMemAllocationGranularity_flags)>
    g_devrHipMemGetAllocationGranularity;

extern std::function<hipError_t(hipStream_t*, unsigned int)> g_devrHipStreamCreateWithFlags;
extern std::function<hipError_t(hipStream_t)> g_devrHipStreamSynchronize;
extern std::function<hipError_t(hipStream_t)> g_devrHipStreamDestroy;
extern std::function<hipError_t(hipStreamCaptureMode*)> g_devrHipThreadExchangeStreamCaptureMode;

extern std::function<hipError_t(hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType)>
    g_devrHipMemImportFromShareableHandle;
extern std::function<hipError_t(void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long)>
    g_devrHipMemMap;
extern std::function<hipError_t(hipMemGenericAllocationHandle_t)> g_devrHipMemRelease;

extern std::function<hipError_t(void**, size_t, size_t, void*, unsigned long long)> g_devrHipMemAddressReserve;
extern std::function<hipError_t(void*, size_t)> g_devrHipMemAddressFree;
extern std::function<hipError_t(void*, size_t)> g_devrHipMemUnmap;
extern std::function<hipError_t(void*, const void*, size_t, hipMemcpyKind, hipStream_t)> g_devrHipMemcpyAsync;
extern std::function<hipError_t(void*, int, size_t, hipStream_t)> g_devrHipMemsetAsync;

extern std::function<hipError_t(hipIpcMemHandle_t*, void*)> g_devrHipIpcGetMemHandle;
extern std::function<hipError_t(void**, hipIpcMemHandle_t, unsigned int)> g_devrHipIpcOpenMemHandle;
extern std::function<hipError_t(void*)> g_devrHipIpcCloseMemHandle;
extern std::function<hipError_t(int*)> g_devrHipGetDevice;
extern std::function<hipError_t(int)> g_devrHipSetDevice;
extern std::function<hipError_t(void*, const void*, size_t, hipMemcpyKind)> g_devrHipMemcpy;
extern std::function<hipError_t(hipDeviceptr_t*, size_t*, hipDeviceptr_t)> g_devrHipMemGetAddressRange;
extern std::function<hipError_t(hipMemGenericAllocationHandle_t*, void*)> g_devrHipMemRetainAllocationHandle;

extern std::function<ncclResult_t(const ncclComm_t, void*, size_t, void**)> g_devrNcclCommRegister;
extern std::function<ncclResult_t(const ncclComm_t, void*)> g_devrNcclCommDeregister;
extern std::function<ncclResult_t(ncclComm_t, ncclWindow_t)> g_devrNcclCommWindowDeregister;

extern std::function<ncclResult_t(void*, void*, int)> g_devrBootstrapAllGather;
extern std::function<ncclResult_t(void*, int*, int, int, int)> g_devrBootstrapIntraNodeBarrier;
extern std::function<ncclResult_t(void*, int*, int, int, void*, int)> g_devrBootstrapIntraNodeAllGather;

extern std::function<ncclResult_t(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS],
                                  ncclGinWindow_t[NCCL_GIN_MAX_CONNECTIONS], int, bool, int)>
    g_devrGinRegister;
extern std::function<ncclResult_t(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS])> g_devrGinDeregister;

extern std::function<ncclResult_t(struct ncclSpace*, int64_t, int64_t, int, int64_t*)> g_devrSpaceAlloc;
extern std::function<ncclResult_t(struct ncclSpace*, int64_t, int64_t)> g_devrSpaceFree;
extern std::function<ncclResult_t(struct ncclDevrMemory*, int)> g_devrPopulateSegmentSizes;

extern std::function<ncclResult_t(struct ncclShadowPool*, size_t, void**, void**, hipStream_t)> g_devrShadowPoolAlloc;
extern std::function<ncclResult_t(struct ncclShadowPool*, void*, hipStream_t)> g_devrShadowPoolFree;
extern std::function<ncclResult_t(struct ncclShadowPool*, void*, void**)> g_devrShadowPoolToHost;

// Resolves a device window to its host record; the default finds nothing.
extern std::function<ncclResult_t(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void**)>
    g_devrIntruAddressMapFind;
extern std::function<ncclResult_t(struct ncclDevrState*, struct ncclDevrMemory*, hipStream_t,
                                  struct ncclSegmentWindow**)>
    g_devrAllocAndPopulateSegmentWindows;

extern std::function<ncclResult_t(struct ncclComm*)> g_devrRmaProxyConnectOnce;
extern std::function<ncclResult_t(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS])>
    g_devrRmaProxyRegister;
extern std::function<ncclResult_t(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS])> g_devrRmaProxyDeregister;

// Hands back an fd the caller must close; the default opens /dev/null.
extern std::function<ncclResult_t(struct ncclComm*, int, void*, int*)> g_devrProxyClientGetFdBlocking;

// Backs every NCCL_PARAM in the unit under test. DevRuntimeTests.cpp redefines
// the macro to call this instead of param.h's caching body, so a param's value
// can differ between tests; the default returns the param's own default.
// Takes the bare env name (no "NCCL_" prefix) and that default.
extern std::function<int64_t(const char*, int64_t)> g_devrLoadParam;

// Restore every seam above to its default. Call from a fixture TearDown so a
// test cannot leak behaviour into the next one.
void ResetDevRuntimeFakes();

#endif  // RCCL_TEST_HOST_FAKES_DEV_RUNTIME_FAKES_H_
