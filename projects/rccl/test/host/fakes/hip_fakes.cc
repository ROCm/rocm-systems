/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// HIP runtime fakes for rccl-UnitTestsMicro. The binary does not link the real
// HIP runtime, so this file provides (1) controllable std::function seams
// (g_hip*) the tests drive via the macro shims in p2p-test.cc, and (2) plain
// stubs for every other HIP symbol the object code references (returning
// hipErrorInvalidValue so unexercised paths fail loudly instead of binding the
// real driver).

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

#include <sys/mman.h>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "hip_fakes.h"   // g_hip* hook declarations + ResetHipFakes()

// Compile-time watchdog: each VMM/stream seam must keep the exact signature of
// the real HIP symbol it stands in for (templates + macro in signature-drift.h).
#include "signature-drift.h"
ASSERT_HOOK_MATCHES_PROD(g_hipMemAddressReserve,     hipMemAddressReserve);
ASSERT_HOOK_MATCHES_PROD(g_hipMemAddressFree,        hipMemAddressFree);
ASSERT_HOOK_MATCHES_PROD(g_hipMemCreate,             hipMemCreate);
ASSERT_HOOK_MATCHES_PROD(g_hipMemGetAllocationGranularity,
                         hipMemGetAllocationGranularity);
ASSERT_HOOK_MATCHES_PROD(g_hipMemGetAllocationPropertiesFromHandle,
                         hipMemGetAllocationPropertiesFromHandle);
ASSERT_HOOK_MATCHES_PROD(g_hipMemImportFromShareableHandle,
                         hipMemImportFromShareableHandle);
ASSERT_HOOK_MATCHES_PROD(g_hipMemMap,                hipMemMap);
ASSERT_HOOK_MATCHES_PROD(g_hipMemUnmap,              hipMemUnmap);
ASSERT_HOOK_MATCHES_PROD(g_hipMemSetAccess,          hipMemSetAccess);
ASSERT_HOOK_MATCHES_PROD(g_hipStreamCreateWithFlags, hipStreamCreateWithFlags);
ASSERT_HOOK_MATCHES_PROD(g_hipStreamDestroy,         hipStreamDestroy);
ASSERT_HOOK_MATCHES_PROD(g_hipStreamSynchronize,     hipStreamSynchronize);
ASSERT_HOOK_MATCHES_PROD(g_hipThreadExchangeStreamCaptureMode,
                         hipThreadExchangeStreamCaptureMode);
#undef ASSERT_HOOK_MATCHES_PROD

// ===========================================================================
// Section 1: controllable HIP seams (defaults return hipErrorInvalidValue)
// ===========================================================================

// --- hipMemGetAddressRange / hipIpcGetMemHandle -------------------------
static hipError_t DefaultHipMemGetAddressRange(hipDeviceptr_t*, std::size_t*,
                                               hipDeviceptr_t)
{
    return hipErrorInvalidValue;
}

static hipError_t DefaultHipIpcGetMemHandle(hipIpcMemHandle_t*, void*)
{
    return hipErrorInvalidValue;
}

std::function<hipError_t(hipDeviceptr_t*, std::size_t*, hipDeviceptr_t)>
    g_hipMemGetAddressRange = DefaultHipMemGetAddressRange;
std::function<hipError_t(hipIpcMemHandle_t*, void*)>
    g_hipIpcGetMemHandle = DefaultHipIpcGetMemHandle;

// --- hipMemRetainAllocationHandle / hipMemExportToShareableHandle /
//     hipMemRelease (the cuMem*-export arm) ------------------------------
static hipError_t DefaultHipMemRetainAllocationHandle(
    hipMemGenericAllocationHandle_t*, void*)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemExportToShareableHandle(
    void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
    unsigned long long)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemRelease(hipMemGenericAllocationHandle_t)
{
    return hipErrorInvalidValue;
}

std::function<hipError_t(hipMemGenericAllocationHandle_t*, void*)>
    g_hipMemRetainAllocationHandle = DefaultHipMemRetainAllocationHandle;
std::function<hipError_t(void*, hipMemGenericAllocationHandle_t,
                         hipMemAllocationHandleType, unsigned long long)>
    g_hipMemExportToShareableHandle = DefaultHipMemExportToShareableHandle;
std::function<hipError_t(hipMemGenericAllocationHandle_t)>
    g_hipMemRelease = DefaultHipMemRelease;

// --- hipPointerGetAttribute (legacy-IPC capability query) ---------------
// Default: succeed and report NOT legacy-capable, so the cuMem-export and
// nothing-works arms stay reachable.
static hipError_t DefaultHipPointerGetAttribute(void* data,
                                                hipPointer_attribute attribute,
                                                hipDeviceptr_t)
{
    if (data && attribute == HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE) {
        *static_cast<int*>(data) = 0;   // matches `int legacyIpcCap` in p2p.cc
    }
    return hipSuccess;
}

std::function<hipError_t(void*, hipPointer_attribute, hipDeviceptr_t)>
    g_hipPointerGetAttribute = DefaultHipPointerGetAttribute;

// --- device model: runtime version + device properties ------------------
static hipError_t DefaultHipRuntimeGetVersion(int* version)
{
    if (version) {
        *version = 60443484;  // plausible ROCm 6.x runtime version
    }
    return hipSuccess;
}
std::function<hipError_t(int*)> g_hipRuntimeGetVersion = DefaultHipRuntimeGetVersion;

static hipError_t DefaultHipGetDeviceProperties(hipDeviceProp_t* prop, int)
{
    if (prop) {
        *prop = hipDeviceProp_t{};
        // snprintf null-terminates within the field's fixed size (no manual
        // strncpy + terminator, no heap alloc as fmt::format would incur).
        std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "gfx942:sramecc+:xnack-");
        prop->totalGlobalMem = static_cast<size_t>(64) << 30;
        prop->warpSize = 64;
    }
    return hipSuccess;
}
std::function<hipError_t(hipDeviceProp_t*, int)>
    g_hipGetDeviceProperties = DefaultHipGetDeviceProperties;

static hipError_t DefaultHipExtMallocWithFlags(void** ptr, std::size_t size, unsigned)
{
    if (ptr) {
        *ptr = std::malloc(size);
        return *ptr ? hipSuccess : hipErrorOutOfMemory;
    }
    return hipErrorInvalidValue;
}
std::function<hipError_t(void**, std::size_t, unsigned)>
    g_hipExtMallocWithFlags = DefaultHipExtMallocWithFlags;

static hipError_t DefaultHipHostMalloc(void** ptr, std::size_t size, unsigned)
{
    if (ptr) {
        *ptr = std::malloc(size);
        return *ptr ? hipSuccess : hipErrorOutOfMemory;
    }
    return hipErrorInvalidValue;
}
std::function<hipError_t(void**, std::size_t, unsigned)>
    g_hipHostMalloc = DefaultHipHostMalloc;

static hipError_t DefaultHipFree(void* ptr)
{
    std::free(ptr);
    return hipSuccess;
}
std::function<hipError_t(void*)> g_hipFree = DefaultHipFree;

// --- device inventory + current-device state ----------------------------
int g_deviceCount = 8;
int g_currentDevice = 0;

static hipError_t DefaultHipGetDevice(int* dev)
{
    if (dev) {
        *dev = g_currentDevice;
    }
    return hipSuccess;
}
std::function<hipError_t(int*)> g_hipGetDevice = DefaultHipGetDevice;

static hipError_t DefaultHipSetDevice(int dev)
{
    g_currentDevice = dev;
    return hipSuccess;
}
std::function<hipError_t(int)> g_hipSetDevice = DefaultHipSetDevice;

static hipError_t DefaultHipGetDeviceCount(int* count)
{
    if (count) {
        *count = g_deviceCount;
    }
    return hipSuccess;
}
std::function<hipError_t(int*)> g_hipGetDeviceCount = DefaultHipGetDeviceCount;

// --- deep-path result seams (commAlloc/devCommSetup) --------------------
// Default to failure so any call a test hasn't opted into surfaces as an
// unexpected call; a test sets the relevant seam to hipSuccess to enable the
// happy path.
hipError_t g_hipDeviceGetAttributeResult = hipErrorInvalidValue;
hipError_t g_hipDeviceGetPCIBusIdResult  = hipErrorInvalidValue;
hipError_t g_hipEventCreateResult        = hipErrorInvalidValue;
hipError_t g_hipMemPoolResult            = hipErrorInvalidValue;
hipError_t g_hipStreamCreateResult       = hipErrorInvalidValue;
hipError_t g_hipAsyncOpsResult           = hipErrorInvalidValue;
int        g_hipWarpSize                 = 64;

// --- HIP VMM driver API + stream lifecycle seams ------------------------
// Defaults mirror the old fail-loud stubs: return hipErrorInvalidValue and
// zero any out-param. InstallHostBackedVmm() below overrides them.
static hipError_t DefaultHipMemAddressReserve(void** ptr, std::size_t,
                                              std::size_t, void*,
                                              unsigned long long)
{
    if (ptr) *ptr = nullptr;
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemAddressFree(void*, std::size_t)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemCreate(hipMemGenericAllocationHandle_t* handle,
                                      std::size_t, const hipMemAllocationProp*,
                                      unsigned long long)
{
    if (handle) *handle = nullptr;
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemGetAllocationGranularity(
    std::size_t* granularity, const hipMemAllocationProp*,
    hipMemAllocationGranularity_flags)
{
    if (granularity) *granularity = 0;
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemGetAllocationPropertiesFromHandle(
    hipMemAllocationProp*, hipMemGenericAllocationHandle_t)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemImportFromShareableHandle(
    hipMemGenericAllocationHandle_t* handle, void*, hipMemAllocationHandleType)
{
    if (handle) *handle = nullptr;
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemMap(void*, std::size_t, std::size_t,
                                   hipMemGenericAllocationHandle_t,
                                   unsigned long long)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemUnmap(void*, std::size_t)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemSetAccess(void*, std::size_t,
                                         const hipMemAccessDesc*, std::size_t)
{
    return hipErrorInvalidValue;
}
// The stream-lifecycle defaults delegate to the plain result seams
// (g_hipStreamCreateResult / g_hipAsyncOpsResult) so the init micro-test, which
// opts into the happy path by setting those results to hipSuccess, still drives
// devCommSetup to completion. rccl-UnitTestsDevRuntime overrides these hooks
// wholesale via InstallHostBackedVmm(), so its behaviour is unaffected.
static hipError_t DefaultHipStreamCreateWithFlags(hipStream_t* stream,
                                                  unsigned int)
{
    if (stream) *stream = (g_hipStreamCreateResult == hipSuccess)
                              ? reinterpret_cast<hipStream_t>(0x1) : nullptr;
    return g_hipStreamCreateResult;
}
static hipError_t DefaultHipStreamDestroy(hipStream_t)
{
    return hipSuccess;  // benign teardown (ncclDestroySideStream)
}
static hipError_t DefaultHipStreamSynchronize(hipStream_t)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipThreadExchangeStreamCaptureMode(
    hipStreamCaptureMode*)
{
    return g_hipAsyncOpsResult;
}

std::function<hipError_t(void**, std::size_t, std::size_t, void*,
                         unsigned long long)>
    g_hipMemAddressReserve = DefaultHipMemAddressReserve;
std::function<hipError_t(void*, std::size_t)>
    g_hipMemAddressFree = DefaultHipMemAddressFree;
std::function<hipError_t(hipMemGenericAllocationHandle_t*, std::size_t,
                         const hipMemAllocationProp*, unsigned long long)>
    g_hipMemCreate = DefaultHipMemCreate;
std::function<hipError_t(std::size_t*, const hipMemAllocationProp*,
                         hipMemAllocationGranularity_flags)>
    g_hipMemGetAllocationGranularity = DefaultHipMemGetAllocationGranularity;
std::function<hipError_t(hipMemAllocationProp*,
                         hipMemGenericAllocationHandle_t)>
    g_hipMemGetAllocationPropertiesFromHandle =
        DefaultHipMemGetAllocationPropertiesFromHandle;
std::function<hipError_t(hipMemGenericAllocationHandle_t*, void*,
                         hipMemAllocationHandleType)>
    g_hipMemImportFromShareableHandle = DefaultHipMemImportFromShareableHandle;
std::function<hipError_t(void*, std::size_t, std::size_t,
                         hipMemGenericAllocationHandle_t, unsigned long long)>
    g_hipMemMap = DefaultHipMemMap;
std::function<hipError_t(void*, std::size_t)>
    g_hipMemUnmap = DefaultHipMemUnmap;
std::function<hipError_t(void*, std::size_t, const hipMemAccessDesc*,
                         std::size_t)>
    g_hipMemSetAccess = DefaultHipMemSetAccess;
std::function<hipError_t(hipStream_t*, unsigned int)>
    g_hipStreamCreateWithFlags = DefaultHipStreamCreateWithFlags;
std::function<hipError_t(hipStream_t)>
    g_hipStreamDestroy = DefaultHipStreamDestroy;
std::function<hipError_t(hipStream_t)>
    g_hipStreamSynchronize = DefaultHipStreamSynchronize;
std::function<hipError_t(hipStreamCaptureMode*)>
    g_hipThreadExchangeStreamCaptureMode =
        DefaultHipThreadExchangeStreamCaptureMode;

// Restore every HIP hook to its default.
void ResetHipFakes()
{
    g_hipMemGetAddressRange         = DefaultHipMemGetAddressRange;
    g_hipIpcGetMemHandle            = DefaultHipIpcGetMemHandle;
    g_hipMemRetainAllocationHandle  = DefaultHipMemRetainAllocationHandle;
    g_hipMemExportToShareableHandle = DefaultHipMemExportToShareableHandle;
    g_hipMemRelease                 = DefaultHipMemRelease;
    g_hipPointerGetAttribute        = DefaultHipPointerGetAttribute;
    // init.cc device-model seams
    g_hipRuntimeGetVersion          = DefaultHipRuntimeGetVersion;
    g_hipGetDeviceProperties        = DefaultHipGetDeviceProperties;
    g_hipExtMallocWithFlags         = DefaultHipExtMallocWithFlags;
    g_hipHostMalloc                 = DefaultHipHostMalloc;
    g_hipFree                       = DefaultHipFree;
    g_hipGetDevice                  = DefaultHipGetDevice;
    g_hipSetDevice                  = DefaultHipSetDevice;
    g_hipGetDeviceCount             = DefaultHipGetDeviceCount;
    g_deviceCount                   = 8;
    g_currentDevice                 = 0;
    g_hipDeviceGetAttributeResult   = hipErrorInvalidValue;
    g_hipDeviceGetPCIBusIdResult    = hipErrorInvalidValue;
    g_hipEventCreateResult          = hipErrorInvalidValue;
    g_hipMemPoolResult              = hipErrorInvalidValue;
    g_hipStreamCreateResult         = hipErrorInvalidValue;
    g_hipAsyncOpsResult             = hipErrorInvalidValue;
    g_hipWarpSize                   = 64;

    g_hipMemAddressReserve          = DefaultHipMemAddressReserve;
    g_hipMemAddressFree             = DefaultHipMemAddressFree;
    g_hipMemCreate                  = DefaultHipMemCreate;
    g_hipMemGetAllocationGranularity = DefaultHipMemGetAllocationGranularity;
    g_hipMemGetAllocationPropertiesFromHandle =
        DefaultHipMemGetAllocationPropertiesFromHandle;
    g_hipMemImportFromShareableHandle = DefaultHipMemImportFromShareableHandle;
    g_hipMemMap                     = DefaultHipMemMap;
    g_hipMemUnmap                   = DefaultHipMemUnmap;
    g_hipMemSetAccess               = DefaultHipMemSetAccess;
    g_hipStreamCreateWithFlags      = DefaultHipStreamCreateWithFlags;
    g_hipStreamDestroy              = DefaultHipStreamDestroy;
    g_hipStreamSynchronize          = DefaultHipStreamSynchronize;
    g_hipThreadExchangeStreamCaptureMode =
        DefaultHipThreadExchangeStreamCaptureMode;
}

// ===========================================================================
// Host-memory-backed HIP VMM/stream profile.
//
// Sets the seams above to behaviour that lets code driving the driver VMM API
// run to completion on a plain CPU (no GPU). Opt in per test via
// InstallHostBackedVmm() (see the DevRuntime host tests).
// ===========================================================================
void InstallHostBackedVmm()
{
    // Reserve VA with an uncommitted anonymous mapping (MAP_NORESERVE): cheap
    // multi-GB flat-VA reservations that are never dereferenced (all
    // map/set-access are no-ops), so no physical memory is committed.
    g_hipMemAddressReserve = [](void** ptr, std::size_t size, std::size_t,
                                void*, unsigned long long) -> hipError_t {
        // The real driver rejects a zero-size reservation; a zero here would
        // be a bug in the code under test, so surface it.
        assert(size != 0);
        void* p = mmap(nullptr, size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) return hipErrorOutOfMemory;
        if (ptr) *ptr = p;
        return hipSuccess;
    };
    g_hipMemAddressFree = [](void* devPtr, std::size_t size) -> hipError_t {
        assert(size != 0);
        munmap(devPtr, size);
        return hipSuccess;
    };
    g_hipMemCreate = [](hipMemGenericAllocationHandle_t* handle, std::size_t,
                        const hipMemAllocationProp*,
                        unsigned long long) -> hipError_t {
        if (handle)
            *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
        return hipSuccess;
    };
    g_hipMemGetAllocationGranularity =
        [](std::size_t* granularity, const hipMemAllocationProp*,
           hipMemAllocationGranularity_flags) -> hipError_t {
        if (granularity) *granularity = 4096;
        return hipSuccess;
    };
    g_hipMemGetAllocationPropertiesFromHandle =
        [](hipMemAllocationProp* prop,
           hipMemGenericAllocationHandle_t) -> hipError_t {
        if (prop) {
            *prop = hipMemAllocationProp{};
            prop->location.type = hipMemLocationTypeDevice;
        }
        return hipSuccess;
    };
    g_hipMemImportFromShareableHandle =
        [](hipMemGenericAllocationHandle_t* handle, void*,
           hipMemAllocationHandleType) -> hipError_t {
        if (handle)
            *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
        return hipSuccess;
    };
    g_hipMemExportToShareableHandle =
        [](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
           unsigned long long) -> hipError_t { return hipSuccess; };
    g_hipMemMap = [](void*, std::size_t, std::size_t,
                     hipMemGenericAllocationHandle_t,
                     unsigned long long) -> hipError_t { return hipSuccess; };
    g_hipMemUnmap = [](void*, std::size_t) -> hipError_t {
        return hipSuccess;
    };
    g_hipMemSetAccess = [](void*, std::size_t, const hipMemAccessDesc*,
                           std::size_t) -> hipError_t { return hipSuccess; };
    g_hipMemRelease = [](hipMemGenericAllocationHandle_t) -> hipError_t {
        return hipSuccess;
    };
    g_hipMemRetainAllocationHandle =
        [](hipMemGenericAllocationHandle_t* handle, void*) -> hipError_t {
        if (handle)
            *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
        return hipSuccess;
    };
    g_hipMemGetAddressRange = [](hipDeviceptr_t* pbase, std::size_t* psize,
                                 hipDeviceptr_t dptr) -> hipError_t {
        if (pbase) *pbase = dptr;
        if (psize) *psize = 0;
        return hipSuccess;
    };

    // Stream lifecycle: ncclDevrFinalize creates/synchronizes/destroys
    // throwaway streams for teardown bookkeeping; none carry real work on the
    // host, so a non-null opaque handle and success returns suffice.
    g_hipStreamCreateWithFlags = [](hipStream_t* stream,
                                    unsigned int) -> hipError_t {
        if (stream) *stream = reinterpret_cast<hipStream_t>(0x1);
        return hipSuccess;
    };
    g_hipStreamSynchronize = [](hipStream_t) -> hipError_t {
        return hipSuccess;
    };
    g_hipStreamDestroy = [](hipStream_t) -> hipError_t { return hipSuccess; };
    g_hipThreadExchangeStreamCaptureMode =
        [](hipStreamCaptureMode* mode) -> hipError_t {
        if (mode) *mode = hipStreamCaptureModeRelaxed;
        return hipSuccess;
    };
}

// ===========================================================================
// Section 2: plain HIP runtime symbol stubs (link without libamdhip64.so).
// Three (hipMemGetAddressRange, hipMemRetainAllocationHandle, hipMemRelease)
// delegate to the seams above so shimmed and non-shimmed call sites agree.
// ===========================================================================

// --- hook-backed real symbols -------------------------------------------
hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize,
                                 hipDeviceptr_t dptr)
{
    return g_hipMemGetAddressRange(pbase, psize, dptr);
}

hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle,
                                        void* addr)
{
    return g_hipMemRetainAllocationHandle(handle, addr);
}

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr)
{
    return g_hipIpcGetMemHandle(handle, devPtr);
}

hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle)
{
    return g_hipMemRelease(handle);
}

hipError_t hipMemExportToShareableHandle(void* shareableHandle,
                                         hipMemGenericAllocationHandle_t handle,
                                         hipMemAllocationHandleType handleType,
                                         unsigned long long flags)
{
    return g_hipMemExportToShareableHandle(shareableHandle, handle, handleType,
                                           flags);
}

// --- plain link-satisfying stubs (unexercised paths) --------------------
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int, int)
{
    if (canAccessPeer) *canAccessPeer = 0;
    return hipErrorInvalidValue;
}

hipError_t hipDeviceEnablePeerAccess(int, unsigned int)
{
    return hipErrorInvalidValue;
}

hipError_t hipDeviceGet(hipDevice_t* device, int)
{
    if (device) *device = 0;
    return hipErrorInvalidValue;
}

hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t)
{
    if (uuid) {
        *uuid = hipUUID{};
    }
    return hipSuccess;
}

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int)
{
    if (!pi) return g_hipDeviceGetAttributeResult;
    switch (attr) {
        case hipDeviceAttributeWarpSize:
            *pi = g_hipWarpSize; break;
        case hipDeviceAttributeDirectManagedMemAccessFromHost:
            *pi = 1; break;   // report managed -> ncclCudaHostCalloc takes the extMalloc arm
        default:
            *pi = 0; break;
    }
    return g_hipDeviceGetAttributeResult;
}

hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int)
{
    if (pciBusId && len > 0) {
        if (g_hipDeviceGetPCIBusIdResult == hipSuccess)
            std::snprintf(pciBusId, len, "0000:00:00.0");
        else
            pciBusId[0] = '\0';
    }
    return g_hipDeviceGetPCIBusIdResult;
}

hipError_t hipEventCreate(hipEvent_t* event)
{
    if (event) *event = nullptr;
    return hipErrorInvalidValue;
}

hipError_t hipEventDestroy(hipEvent_t)      { return hipSuccess; }  // benign teardown (commFree)
hipError_t hipEventQuery(hipEvent_t)        { return hipErrorInvalidValue; }
hipError_t hipEventRecord(hipEvent_t, hipStream_t) { return hipErrorInvalidValue; }

hipError_t hipExtMallocWithFlags(void** ptr, size_t size, unsigned int flags)
{
    return g_hipExtMallocWithFlags(ptr, size, flags);
}

hipError_t hipFree(void* ptr) { return g_hipFree(ptr); }

hipError_t hipGetDevice(int* deviceId) { return g_hipGetDevice(deviceId); }

hipError_t hipGetDeviceCount(int* count) { return g_hipGetDeviceCount(count); }

const char* hipGetErrorString(hipError_t) { return "[hip_fake] stub error"; }

hipError_t hipGetLastError(void) { return hipErrorInvalidValue; }

hipError_t hipHostFree(void* ptr)
{
    std::free(ptr);
    return hipSuccess;
}

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags)
{
    return g_hipHostMalloc(ptr, size, flags);
}

hipError_t hipIpcCloseMemHandle(void*) { return hipErrorInvalidValue; }

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t, unsigned int)
{
    if (devPtr) *devPtr = nullptr;
    return hipErrorInvalidValue;
}

hipError_t hipMemAddressFree(void* devPtr, size_t size)
{
    return g_hipMemAddressFree(devPtr, size);
}

hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment,
                                void* addr, unsigned long long flags)
{
    return g_hipMemAddressReserve(ptr, size, alignment, addr, flags);
}

hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
                        const hipMemAllocationProp* prop, unsigned long long flags)
{
    return g_hipMemCreate(handle, size, prop, flags);
}

hipError_t hipMemGetAllocationGranularity(size_t* granularity,
                                          const hipMemAllocationProp* prop,
                                          hipMemAllocationGranularity_flags option)
{
    return g_hipMemGetAllocationGranularity(granularity, prop, option);
}

hipError_t hipMemGetAllocationPropertiesFromHandle(
    hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle)
{
    return g_hipMemGetAllocationPropertiesFromHandle(prop, handle);
}

hipError_t hipMemImportFromShareableHandle(
    hipMemGenericAllocationHandle_t* handle, void* osHandle,
    hipMemAllocationHandleType shHandleType)
{
    return g_hipMemImportFromShareableHandle(handle, osHandle, shHandleType);
}

hipError_t hipMemMap(void* ptr, size_t size, size_t offset,
                     hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags)
{
    return g_hipMemMap(ptr, size, offset, handle, flags);
}

hipError_t hipMemSetAccess(void* ptr, size_t size,
                           const hipMemAccessDesc* desc, size_t count)
{
    return g_hipMemSetAccess(ptr, size, desc, count);
}

hipError_t hipMemUnmap(void* ptr, size_t size)
{
    return g_hipMemUnmap(ptr, size);
}

hipError_t hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind,
                          hipStream_t)
{
    return g_hipAsyncOpsResult;
}

hipError_t hipMemsetAsync(void*, int, size_t, hipStream_t)
{
    return g_hipAsyncOpsResult;
}

hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr)
{
    return g_hipPointerGetAttribute(data, attribute, ptr);
}

hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags)
{
    return g_hipStreamCreateWithFlags(stream, flags);
}

hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int, int)
{
    if (stream) {
        *stream = (g_hipStreamCreateResult == hipSuccess)
                      ? reinterpret_cast<hipStream_t>(0x1) : nullptr;
    }
    return g_hipStreamCreateResult;
}

hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
    if (leastPriority) {
        *leastPriority = 0;
    }
    if (greatestPriority) {
        *greatestPriority = 0;
    }
    return hipSuccess;
}

hipError_t hipStreamDestroy(hipStream_t stream)
{
    return g_hipStreamDestroy(stream);
}
hipError_t hipStreamSynchronize(hipStream_t stream)
{
    return g_hipStreamSynchronize(stream);
}

hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode)
{
    return g_hipThreadExchangeStreamCaptureMode(mode);
}

hipError_t hipSetDevice(int deviceId) { return g_hipSetDevice(deviceId); }
hipError_t hipMalloc(void** p, size_t) { if (p) *p = nullptr; return hipErrorInvalidValue; }
hipError_t hipMemcpy(void*, const void*, size_t, hipMemcpyKind) { return hipErrorInvalidValue; }
hipError_t hipMemset(void*, int, size_t) { return hipErrorInvalidValue; }
hipError_t hipDeviceSynchronize(void) { return hipErrorInvalidValue; }
// init.cc's hipGetDeviceProperties call binds to hipGetDevicePropertiesR0600
// after hipify; the ROCm header remaps the unversioned name to this symbol.
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device)
{
    return g_hipGetDeviceProperties(prop, device);
}
hipError_t hipDriverGetVersion(int* v) { if (v) *v = 70002000; return hipSuccess; }
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int) { return hipErrorInvalidValue; }
hipError_t hipStreamCreate(hipStream_t*) { return hipErrorInvalidValue; }
// hipStreamCreateWithPriority / hipDeviceGetStreamPriorityRange are defined
// above (seam-routed) -- the develop merge added plainer duplicates here.
hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*) { return hipErrorInvalidValue; }
hipError_t hipHostGetDevicePointer(void**, void*, unsigned int) { return hipErrorInvalidValue; }
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t*, hipEvent_t) { return hipErrorInvalidValue; }
hipError_t hipEventSynchronize(hipEvent_t) { return hipErrorInvalidValue; }

// --- init.cc deep-path HIP stubs (commAlloc/devCommSetup) ---------------
hipError_t hipRuntimeGetVersion(int* version) { return g_hipRuntimeGetVersion(version); }
hipError_t hipDeviceSetLimit(hipLimit_t, size_t) { return hipErrorInvalidValue; }
hipError_t hipEventCreateWithFlags(hipEvent_t* e, unsigned int) {
    if (e) *e = (g_hipEventCreateResult == hipSuccess) ? reinterpret_cast<hipEvent_t>(0x1) : nullptr;
    return g_hipEventCreateResult;
}
hipError_t hipMemPoolCreate(hipMemPool_t* p, const hipMemPoolProps*) {
    if (p) *p = (g_hipMemPoolResult == hipSuccess) ? reinterpret_cast<hipMemPool_t>(0x1) : nullptr;
    return g_hipMemPoolResult;
}
hipError_t hipMemPoolDestroy(hipMemPool_t) { return hipSuccess; }  // benign teardown (commFree)
hipError_t hipMemPoolSetAttribute(hipMemPool_t, hipMemPoolAttr, void*) { return g_hipMemPoolResult; }
