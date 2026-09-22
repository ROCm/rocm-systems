/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable HIP seams for the micro-test fakes layer.
//
// The micro-test binary does not link the real HIP runtime (see
// hip_fakes.cc for why). These std::function hooks are the handful of HIP
// calls the unit-under-test drives on its happy path; the macro shims in
// p2p-test.cc route the p2p.cc call sites through them. Tests install
// per-test behaviour by overwriting a hook (see the ScopedHook helper in
// p2p-test.cc) and ResetHipFakes() restores the defaults.
//
// Defaults return hipErrorInvalidValue so any call site a test doesn't
// explicitly opt into surfaces the unexpected call as
// ncclUnhandledCudaError via CUCHECKGOTO.

#ifndef RCCL_TEST_HOST_HIP_FAKES_H_
#define RCCL_TEST_HOST_HIP_FAKES_H_

#include <cstddef>
#include <vector>
#include <functional>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

// hipMemGetAddressRange / hipIpcGetMemHandle
extern std::function<hipError_t(hipDeviceptr_t* /*pbase*/, std::size_t* /*psize*/,
                                hipDeviceptr_t /*dptr*/)>
    g_hipMemGetAddressRange;
extern std::function<hipError_t(hipIpcMemHandle_t* /*handle*/, void* /*devPtr*/)>
    g_hipIpcGetMemHandle;

// hipMemRetainAllocationHandle / hipMemExportToShareableHandle /
// hipMemRelease: the three HIP runtime entry points the cuMem*-export arm
// of ipcRegisterBuffer calls.
//
// Defaults return hipErrorInvalidValue so unexpected call sites surface
// via CUCHECKGOTO; tests that want a happy path install a hook that
// returns hipSuccess (and, for Retain, hands back a sentinel handle).
extern std::function<hipError_t(hipMemGenericAllocationHandle_t* /*handle*/,
                                void* /*addr*/)>
    g_hipMemRetainAllocationHandle;
extern std::function<hipError_t(void* /*shareableHandle*/,
                                hipMemGenericAllocationHandle_t /*handle*/,
                                hipMemAllocationHandleType /*handleType*/,
                                unsigned long long /*flags*/)>
    g_hipMemExportToShareableHandle;
extern std::function<hipError_t(hipMemGenericAllocationHandle_t /*handle*/)>
    g_hipMemRelease;

// hipPointerGetAttribute: on HIP_VERSION >= 71260540 the fresh-registration
// arm of ipcRegisterBuffer queries legacy-IPC capability
// (HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE) through this call
// instead of consulting ncclParamLegacyCudaRegister(). The default returns
// hipSuccess and reports the buffer as NOT legacy-capable (writes 0), which
// keeps the cuMem and nothing-works arms reachable.
extern std::function<hipError_t(void* /*data*/,
                                hipPointer_attribute /*attribute*/,
                                hipDeviceptr_t /*ptr*/)>
    g_hipPointerGetAttribute;

// Device model + inventory seams. Defaults succeed with plausible values.
extern std::function<hipError_t(int* /*version*/)> g_hipRuntimeGetVersion;
extern std::function<hipError_t(hipDeviceProp_t* /*prop*/, int /*device*/)>
    g_hipGetDeviceProperties;
extern std::function<hipError_t(void** /*ptr*/, std::size_t /*size*/,
                                unsigned /*flags*/)>
    g_hipExtMallocWithFlags;
extern std::function<hipError_t(void** /*ptr*/, std::size_t /*size*/,
                                unsigned /*flags*/)>
    g_hipHostMalloc;
extern std::function<hipError_t(void* /*ptr*/)> g_hipFree;
extern std::function<hipError_t(void* /*ptr*/)> g_hipHostFree;
extern int g_deviceCount;
extern int g_currentDevice;
extern std::function<hipError_t(int* /*dev*/)> g_hipGetDevice;
extern std::function<hipError_t(int /*dev*/)> g_hipSetDevice;
extern std::function<hipError_t(int* /*count*/)> g_hipGetDeviceCount;
// Defaults to hipErrorInvalidValue with *canAccessPeer = 0, the fail-loud floor's behaviour.
extern std::function<hipError_t(int* /*canAccessPeer*/, int /*dev1*/, int /*dev2*/)> g_hipDeviceCanAccessPeer;

// Deep-path result seams. Default to hipErrorInvalidValue so any call a test
// hasn't opted into surfaces as an unexpected call; set to hipSuccess to enable
// the happy path, or leave one at the error value to exercise a specific
// CUDACHECK early-return. g_hipWarpSize backs
// hipDeviceGetAttribute(hipDeviceAttributeWarpSize).
extern std::function<hipError_t(int* /*pi*/, hipDeviceAttribute_t /*attr*/, int /*dev*/)> g_hipDeviceGetAttribute;
extern std::function<hipError_t(hipLimit_t /*limit*/, size_t /*value*/)> g_hipDeviceSetLimit;
extern hipError_t g_hipDeviceGetAttributeResult;
extern hipError_t g_hipDeviceGetPCIBusIdResult;
extern hipError_t g_hipEventCreateResult;
extern hipError_t g_hipMemPoolResult;
extern hipError_t g_hipStreamCreateResult;
extern hipError_t g_hipAsyncOpsResult;
extern int g_hipWarpSize;
// Backs hipDeviceGetAttribute(hipDeviceAttributeDirectManagedMemAccessFromHost); 1 is the MI300A answer.
extern int g_hipDirectManagedMemAccess;
// A call count alone cannot tell one device copy's operands from another's, so record them per call.
extern int g_hipMemcpyAsyncCalls;
struct HipMemcpyAsyncRecord {
    void*       dst;
    const void* src;
    size_t      bytes;
};
extern std::vector<HipMemcpyAsyncRecord> g_hipMemcpyAsyncArgs;

// --- VMM / IPC / stream seams -------------------------------------------
// The driver-level surface dev_runtime.cc builds symmetric memory on, plus the
// stream and copy calls its teardown paths use. These were plain stubs until
// the dev_runtime suite needed to drive their failure arms; every default below
// is the stub's old return verbatim, so nothing that relied on the fail-loud
// floor changed. A suite that wants working memory installs
// InstallHipVmmEmulator() instead of hooking them one by one.
extern std::function<hipError_t(void** /*ptr*/, size_t /*size*/, size_t /*alignment*/,
                                void* /*addr*/, unsigned long long /*flags*/)>
    g_hipMemAddressReserve;
extern std::function<hipError_t(void* /*ptr*/, size_t /*size*/)> g_hipMemAddressFree;
extern std::function<hipError_t(hipMemGenericAllocationHandle_t* /*handle*/, size_t /*size*/,
                                const hipMemAllocationProp* /*prop*/, unsigned long long /*flags*/)>
    g_hipMemCreate;
extern std::function<hipError_t(size_t* /*granularity*/, const hipMemAllocationProp* /*prop*/,
                                hipMemAllocationGranularity_flags /*flags*/)>
    g_hipMemGetAllocationGranularity;
extern std::function<hipError_t(hipMemAllocationProp* /*prop*/, hipMemGenericAllocationHandle_t /*handle*/)>
    g_hipMemGetAllocationPropertiesFromHandle;
extern std::function<hipError_t(hipMemGenericAllocationHandle_t* /*handle*/, void* /*shareable*/,
                                hipMemAllocationHandleType /*type*/)>
    g_hipMemImportFromShareableHandle;
extern std::function<hipError_t(void* /*ptr*/, size_t /*size*/, size_t /*offset*/,
                                hipMemGenericAllocationHandle_t /*handle*/, unsigned long long /*flags*/)>
    g_hipMemMap;
extern std::function<hipError_t(void* /*ptr*/, size_t /*size*/, const hipMemAccessDesc* /*desc*/,
                                size_t /*count*/)>
    g_hipMemSetAccess;
extern std::function<hipError_t(void* /*ptr*/, size_t /*size*/)> g_hipMemUnmap;

extern std::function<hipError_t(void** /*ptr*/, hipIpcMemHandle_t /*handle*/, unsigned /*flags*/)>
    g_hipIpcOpenMemHandle;
extern std::function<hipError_t(void* /*ptr*/)> g_hipIpcCloseMemHandle;

extern std::function<hipError_t(void* /*dst*/, const void* /*src*/, size_t /*bytes*/,
                                hipMemcpyKind /*kind*/)>
    g_hipMemcpy;
// The wrapper records into g_hipMemcpyAsyncCalls / g_hipMemcpyAsyncArgs before
// calling this, so installing a hook does not cost a test that recording.
extern std::function<hipError_t(void* /*dst*/, const void* /*src*/, size_t /*bytes*/,
                                hipMemcpyKind /*kind*/, hipStream_t /*stream*/)>
    g_hipMemcpyAsync;
extern std::function<hipError_t(void* /*dst*/, int /*value*/, size_t /*bytes*/, hipStream_t /*stream*/)>
    g_hipMemsetAsync;

extern std::function<hipError_t(hipStream_t* /*stream*/, unsigned /*flags*/)> g_hipStreamCreateWithFlags;
extern std::function<hipError_t(hipStream_t /*stream*/)> g_hipStreamSynchronize;
extern std::function<hipError_t(hipStream_t /*stream*/)> g_hipStreamDestroy;
extern std::function<hipError_t(hipStreamCaptureMode* /*mode*/)> g_hipThreadExchangeStreamCaptureMode;
extern std::function<hipError_t(void)> g_hipGetLastError;

// Install a working host-memory stand-in for the VMM surface: mmap-backed
// reserve/free (honouring the requested alignment), succeeding map/unmap/
// create/import, and copies that actually copy.
//
// Opt-in rather than the default because the fail-loud defaults above are what
// make an unstubbed call surface, and four suites depend on that. A unit whose
// code under test must allocate real memory to run at all -- dev_runtime.cc's
// symmetric-memory paths -- calls this from its fixture SetUp.
// ResetHipFakes() puts the fail-loud defaults back.
void InstallHipVmmEmulator();
// Cross-stream ordering seams, defaulting to hipErrorInvalidValue as the stubs
// they replaced did. std::function, not a result global: rma.cc calls each twice
// around its launch pair, so the arms need per-call control.
extern std::function<hipError_t(hipEvent_t /*event*/, hipStream_t /*stream*/)>
    g_hipEventRecord;
extern std::function<hipError_t(hipStream_t /*stream*/, hipEvent_t /*event*/,
                                unsigned int /*flags*/)>
    g_hipStreamWaitEvent;

// Restore the HIP controllable seams above to their defaults. Called by
// ResetP2pFakes(); exposed for tests that only touch HIP hooks.
void ResetHipFakes();

#endif  // RCCL_TEST_HOST_HIP_FAKES_H_
