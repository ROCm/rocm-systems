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

#ifndef RCCL_TEST_DEVRUNTIMETESTSSTUBS_H_
#define RCCL_TEST_DEVRUNTIMETESTSSTUBS_H_

#include "nccl.h"          // ncclResult_t, for the proxy seam below
#include "gin/gin_host.h"  // NCCL_GIN_MAX_CONNECTIONS, ncclGinWindow_t

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>

extern std::function<hipError_t(hipMemAllocationProp*, hipMemGenericAllocationHandle_t)>
    g_hipMemGetAllocationPropertiesFromHandle;

extern std::function<hipError_t(void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
                                unsigned long long)>
    g_hipMemExportToShareableHandle;

extern std::function<hipError_t(void*, size_t, const hipMemAccessDesc*, size_t)> g_hipMemSetAccess;

extern std::function<hipError_t(size_t*, const hipMemAllocationProp*, hipMemAllocationGranularity_flags)>
    g_hipMemGetAllocationGranularity;

extern std::function<hipError_t(hipStream_t*, unsigned int)> g_hipStreamCreateWithFlags;
extern std::function<hipError_t(hipStream_t)> g_hipStreamSynchronize;
extern std::function<hipError_t(hipStream_t)> g_hipStreamDestroy;
extern std::function<hipError_t(hipStreamCaptureMode*)> g_hipThreadExchangeStreamCaptureMode;

extern std::function<hipError_t(hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType)>
    g_hipMemImportFromShareableHandle;
extern std::function<hipError_t(void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long)>
    g_hipMemMap;
extern std::function<hipError_t(hipMemGenericAllocationHandle_t)> g_hipMemRelease;

extern std::function<hipError_t(void**, size_t, size_t, void*, unsigned long long)> g_hipMemAddressReserve;
extern std::function<hipError_t(void*, size_t)> g_hipMemAddressFree;
extern std::function<hipError_t(void*, size_t)> g_hipMemUnmap;
extern std::function<hipError_t(void*, const void*, size_t, hipMemcpyKind, hipStream_t)> g_hipMemcpyAsync;
extern std::function<hipError_t(void*, int, size_t, hipStream_t)> g_hipMemsetAsync;

extern std::function<hipError_t(hipIpcMemHandle_t*, void*)> g_hipIpcGetMemHandle;
extern std::function<hipError_t(void**, hipIpcMemHandle_t, unsigned int)> g_hipIpcOpenMemHandle;
extern std::function<hipError_t(void*)> g_hipIpcCloseMemHandle;
extern std::function<hipError_t(int*)> g_hipGetDevice;
extern std::function<hipError_t(int)> g_hipSetDevice;
extern std::function<hipError_t(void*, const void*, size_t, hipMemcpyKind)> g_hipMemcpy;
extern std::function<hipError_t(hipDeviceptr_t*, size_t*, hipDeviceptr_t)> g_hipMemGetAddressRange;
extern std::function<hipError_t(hipMemGenericAllocationHandle_t*, void*)> g_hipMemRetainAllocationHandle;

extern std::function<ncclResult_t(const ncclComm_t, void*, size_t, void**)> g_ncclCommRegister;
extern std::function<ncclResult_t(const ncclComm_t, void*)> g_ncclCommDeregister;
extern std::function<ncclResult_t(ncclComm_t, ncclWindow_t)> g_ncclCommWindowDeregister;

extern std::function<ncclResult_t(void*, void*, int)> g_bootstrapAllGather;
extern std::function<ncclResult_t(void*, int*, int, int, int)> g_bootstrapIntraNodeBarrier;
extern std::function<ncclResult_t(void*, int*, int, int, void*, int)> g_bootstrapIntraNodeAllGather;

extern std::function<ncclResult_t(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS],
                                  ncclGinWindow_t[NCCL_GIN_MAX_CONNECTIONS], int, bool, int)>
    g_ginRegister;
extern std::function<ncclResult_t(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS])> g_ginDeregister;

extern std::function<ncclResult_t(struct ncclSpace*, int64_t, int64_t, int, int64_t*)> g_spaceAlloc;
extern std::function<ncclResult_t(struct ncclSpace*, int64_t, int64_t)> g_spaceFree;
extern std::function<ncclResult_t(struct ncclDevrMemory*, int)> g_devrPopulateSegmentSizes;

extern std::function<ncclResult_t(struct ncclShadowPool*, size_t, void**, void**, hipStream_t)> g_shadowPoolAlloc;
extern std::function<ncclResult_t(struct ncclShadowPool*, void*, hipStream_t)> g_shadowPoolFree;
extern std::function<ncclResult_t(struct ncclShadowPool*, void*, void**)> g_shadowPoolToHost;

// Resolves a device window to its host record; the default finds nothing.
extern std::function<ncclResult_t(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void**)>
    g_intruAddressMapFind;
extern std::function<ncclResult_t(struct ncclDevrState*, struct ncclDevrMemory*, hipStream_t,
                                  struct ncclSegmentWindow**)>
    g_devrAllocAndPopulateSegmentWindows;

extern std::function<ncclResult_t(struct ncclComm*)> g_rmaProxyConnectOnce;
extern std::function<ncclResult_t(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS])>
    g_rmaProxyRegister;
extern std::function<ncclResult_t(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS])> g_rmaProxyDeregister;

// Hands back an fd the caller must close; the default opens /dev/null.
extern std::function<ncclResult_t(struct ncclComm*, int, void*, int*)> g_proxyClientGetFdBlocking;

// Backs every NCCL_PARAM in the unit under test. DevRuntimeTests.cpp redefines
// the macro to call this instead of param.h's caching body, so a param's value
// can differ between tests; the default returns the param's own default.
// Takes the bare env name (no "NCCL_" prefix) and that default.
extern std::function<int64_t(const char*, int64_t)> g_loadParam;

// Restore every seam above to its default. Call from a fixture TearDown so a
// test cannot leak behaviour into the next one.
void ResetDevRuntimeFakes();

#endif  // RCCL_TEST_DEVRUNTIMETESTSSTUBS_H_
