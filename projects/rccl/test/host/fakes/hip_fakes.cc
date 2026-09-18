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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

#include <cassert>
#include <cstdint>
#include <sys/mman.h>  // InstallHipVmmEmulator's mmap-backed VA

#include <hip/hip_ext.h>  // hipExtModuleLaunchKernel, for the profile-interceptor stubs
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "fail_loud.h"   // FailLoud: the one abort-on-unreachable spelling
#include "hip_fakes.h"   // g_hip* hook declarations + ResetHipFakes()
#include "hip_profile_interceptor_fakes.h"  // X-list of the profile-runtime HIP entry points

// ---------------------------------------------------------------------------
// Signature-drift watchdog: anchor each controllable HIP seam to its
// production declaration so a hook whose signature drifts from the real
// symbol becomes a compile error rather than a silent std::function coercion
// (templates + macro live in fakes/signature-drift.h).
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_hipMemGetAddressRange,         hipMemGetAddressRange);
ASSERT_HOOK_MATCHES_PROD(g_hipIpcGetMemHandle,            hipIpcGetMemHandle);
ASSERT_HOOK_MATCHES_PROD(g_hipMemRetainAllocationHandle,  hipMemRetainAllocationHandle);
ASSERT_HOOK_MATCHES_PROD(g_hipMemExportToShareableHandle, hipMemExportToShareableHandle);
ASSERT_HOOK_MATCHES_PROD(g_hipMemRelease,                 hipMemRelease);
ASSERT_HOOK_MATCHES_PROD(g_hipPointerGetAttribute,        hipPointerGetAttribute);
ASSERT_HOOK_MATCHES_PROD(g_hipEventRecord,                hipEventRecord);
ASSERT_HOOK_MATCHES_PROD(g_hipStreamWaitEvent,            hipStreamWaitEvent);

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

static hipError_t DefaultHipHostFree(void* ptr)
{
    std::free(ptr);
    return hipSuccess;
}
std::function<hipError_t(void*)> g_hipHostFree = DefaultHipHostFree;

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

// Defined with the plain HIP stubs below, where the attribute switch lives.
static hipError_t DefaultHipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device);
static hipError_t DefaultHipDeviceSetLimit(hipLimit_t limit, size_t value);

static hipError_t DefaultHipDeviceCanAccessPeer(int* canAccessPeer, int, int)
{
    if (canAccessPeer) *canAccessPeer = 0;
    return hipErrorInvalidValue;
}
std::function<hipError_t(int*, int, int)> g_hipDeviceCanAccessPeer = DefaultHipDeviceCanAccessPeer;

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
int        g_hipDirectManagedMemAccess   = 1;
int        g_hipMemcpyAsyncCalls         = 0;
std::vector<HipMemcpyAsyncRecord> g_hipMemcpyAsyncArgs;


// --- VMM / IPC / stream seams -------------------------------------------
// Every default here reproduces exactly what the plain stub in section 2
// returned before it became a seam, so the four suites that predate the
// dev_runtime one see no change. InstallHipVmmEmulator() below swaps in a
// working host-memory implementation for the suites that need one.
static hipError_t DefaultHipMemAddressReserve(void** ptr, size_t, size_t, void*,
                                              unsigned long long)
{
    if (ptr) *ptr = nullptr;
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemAddressFree(void*, size_t) { return hipErrorInvalidValue; }
static hipError_t DefaultHipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t,
                                      const hipMemAllocationProp*, unsigned long long)
{
    if (handle) *handle = nullptr;
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemGetAllocationGranularity(size_t* granularity,
                                                        const hipMemAllocationProp*,
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
static hipError_t DefaultHipMemMap(void*, size_t, size_t, hipMemGenericAllocationHandle_t,
                                   unsigned long long)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemSetAccess(void*, size_t, const hipMemAccessDesc*, size_t)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemUnmap(void*, size_t) { return hipErrorInvalidValue; }

static hipError_t DefaultHipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t, unsigned)
{
    if (devPtr) *devPtr = nullptr;
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipIpcCloseMemHandle(void*) { return hipErrorInvalidValue; }

static hipError_t DefaultHipMemcpy(void*, const void*, size_t, hipMemcpyKind)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t)
{
    return g_hipAsyncOpsResult;
}
static hipError_t DefaultHipMemsetAsync(void*, int, size_t, hipStream_t)
{
    return g_hipAsyncOpsResult;
}

static hipError_t DefaultHipStreamCreateWithFlags(hipStream_t* stream, unsigned)
{
    if (stream) *stream = (g_hipStreamCreateResult == hipSuccess)
                              ? reinterpret_cast<hipStream_t>(0x1) : nullptr;
    return g_hipStreamCreateResult;
}
// Benign teardown (ncclDestroySideStream) -- succeeded as a plain stub.
static hipError_t DefaultHipStreamDestroy(hipStream_t) { return hipSuccess; }
static hipError_t DefaultHipStreamSynchronize(hipStream_t) { return hipErrorInvalidValue; }
static hipError_t DefaultHipThreadExchangeStreamCaptureMode(hipStreamCaptureMode*)
{
    return g_hipAsyncOpsResult;
}
static hipError_t DefaultHipGetLastError(void) { return hipErrorInvalidValue; }

std::function<hipError_t(void**, size_t, size_t, void*, unsigned long long)>
    g_hipMemAddressReserve = DefaultHipMemAddressReserve;
std::function<hipError_t(void*, size_t)> g_hipMemAddressFree = DefaultHipMemAddressFree;
std::function<hipError_t(hipMemGenericAllocationHandle_t*, size_t,
                         const hipMemAllocationProp*, unsigned long long)>
    g_hipMemCreate = DefaultHipMemCreate;
std::function<hipError_t(size_t*, const hipMemAllocationProp*, hipMemAllocationGranularity_flags)>
    g_hipMemGetAllocationGranularity = DefaultHipMemGetAllocationGranularity;
std::function<hipError_t(hipMemAllocationProp*, hipMemGenericAllocationHandle_t)>
    g_hipMemGetAllocationPropertiesFromHandle = DefaultHipMemGetAllocationPropertiesFromHandle;
std::function<hipError_t(hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType)>
    g_hipMemImportFromShareableHandle = DefaultHipMemImportFromShareableHandle;
std::function<hipError_t(void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long)>
    g_hipMemMap = DefaultHipMemMap;
std::function<hipError_t(void*, size_t, const hipMemAccessDesc*, size_t)>
    g_hipMemSetAccess = DefaultHipMemSetAccess;
std::function<hipError_t(void*, size_t)> g_hipMemUnmap = DefaultHipMemUnmap;
std::function<hipError_t(void**, hipIpcMemHandle_t, unsigned)>
    g_hipIpcOpenMemHandle = DefaultHipIpcOpenMemHandle;
std::function<hipError_t(void*)> g_hipIpcCloseMemHandle = DefaultHipIpcCloseMemHandle;
std::function<hipError_t(void*, const void*, size_t, hipMemcpyKind)> g_hipMemcpy = DefaultHipMemcpy;
std::function<hipError_t(void*, const void*, size_t, hipMemcpyKind, hipStream_t)>
    g_hipMemcpyAsync = DefaultHipMemcpyAsync;
std::function<hipError_t(void*, int, size_t, hipStream_t)> g_hipMemsetAsync = DefaultHipMemsetAsync;
std::function<hipError_t(hipStream_t*, unsigned)> g_hipStreamCreateWithFlags = DefaultHipStreamCreateWithFlags;
std::function<hipError_t(hipStream_t)> g_hipStreamSynchronize = DefaultHipStreamSynchronize;
std::function<hipError_t(hipStream_t)> g_hipStreamDestroy = DefaultHipStreamDestroy;
std::function<hipError_t(hipStreamCaptureMode*)>
    g_hipThreadExchangeStreamCaptureMode = DefaultHipThreadExchangeStreamCaptureMode;
std::function<hipError_t(void)> g_hipGetLastError = DefaultHipGetLastError;

// --- the working host-memory VMM stand-in (opt-in) -----------------------
// A reserved VA range mirrors cuMemAddressReserve with an uncommitted anonymous
// mapping (MAP_NORESERVE), so multi-GB flat-VA reservations stay cheap and
// commit no physical memory until touched.
//
// PROT_READ | PROT_WRITE, not PROT_NONE: the mapping stands in for the flat VA
// that windows are carved out of, and a test that writes through win->userPtr --
// which the resource-window path does, via hipMemsetAsync -- faults on a
// PROT_NONE range.
static hipError_t EmulatedMemAddressReserve(void** ptr, size_t size, size_t alignment, void*,
                                            unsigned long long)
{
    // The real driver rejects a zero-size reservation; a zero here would be a
    // bug in the code under test, so surface it rather than substituting one.
    assert(size != 0);
    // Honour the requested alignment rather than ignoring it: production asks
    // for NCCL_MAX_PAGE_SIZE and mmap only guarantees page alignment, so a base
    // that happened to be page- but not NCCL_MAX_PAGE_SIZE-aligned would let an
    // alignment regression through unseen. Over-map, trim, return the slack.
    if (alignment <= 1) alignment = 1;
    size_t over = size + alignment;
    char* raw = static_cast<char*>(
        mmap(nullptr, over, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    if (raw == MAP_FAILED) return hipErrorOutOfMemory;
    uintptr_t base = (reinterpret_cast<uintptr_t>(raw) + alignment - 1) & ~(uintptr_t)(alignment - 1);
    size_t head = base - reinterpret_cast<uintptr_t>(raw);
    if (head != 0) munmap(raw, head);
    size_t tail = over - head - size;
    if (tail != 0) munmap(reinterpret_cast<void*>(base + size), tail);
    *ptr = reinterpret_cast<void*>(base);
    return hipSuccess;
}

static hipError_t EmulatedMemAddressFree(void* devPtr, size_t size)
{
    // Not assert(): NDEBUG strips it, and this binary is built Release as well
    // as Debug. There munmap(ptr, 0) fails with EINVAL, the result is discarded,
    // and a zero-size free would report success instead of surfacing.
    if (size == 0) return hipErrorInvalidValue;
    munmap(devPtr, size);
    return hipSuccess;
}

void InstallHipVmmEmulator()
{
    g_hipMemAddressReserve = EmulatedMemAddressReserve;
    g_hipMemAddressFree    = EmulatedMemAddressFree;
    g_hipMemCreate = [](hipMemGenericAllocationHandle_t* handle, size_t,
                        const hipMemAllocationProp*, unsigned long long) {
        if (handle) *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
        return hipSuccess;
    };
    g_hipMemGetAllocationGranularity = [](size_t* granularity, const hipMemAllocationProp*,
                                          hipMemAllocationGranularity_flags) {
        if (granularity) *granularity = 4096;
        return hipSuccess;
    };
    g_hipMemGetAllocationPropertiesFromHandle = [](hipMemAllocationProp* prop,
                                                   hipMemGenericAllocationHandle_t) {
        if (prop) {
            *prop = hipMemAllocationProp{};
            prop->location.type = hipMemLocationTypeDevice;
        }
        return hipSuccess;
    };
    g_hipMemImportFromShareableHandle = [](hipMemGenericAllocationHandle_t* handle, void*,
                                           hipMemAllocationHandleType) {
        if (handle) *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
        return hipSuccess;
    };
    g_hipMemExportToShareableHandle = [](void*, hipMemGenericAllocationHandle_t,
                                         hipMemAllocationHandleType, unsigned long long) {
        return hipSuccess;
    };
    g_hipMemMap = [](void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long) {
        return hipSuccess;
    };
    g_hipMemSetAccess = [](void*, size_t, const hipMemAccessDesc*, size_t) { return hipSuccess; };
    g_hipMemUnmap     = [](void*, size_t) { return hipSuccess; };
    g_hipMemRelease   = [](hipMemGenericAllocationHandle_t) { return hipSuccess; };
    g_hipMemRetainAllocationHandle = [](hipMemGenericAllocationHandle_t* handle, void*) {
        if (handle) *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
        return hipSuccess;
    };
    g_hipMemGetAddressRange = [](hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
        if (pbase) *pbase = dptr;
        if (psize) *psize = 0;
        return hipSuccess;
    };

    g_hipIpcGetMemHandle = [](hipIpcMemHandle_t* handle, void*) {
        if (handle) *handle = hipIpcMemHandle_t{};
        return hipSuccess;
    };
    g_hipIpcOpenMemHandle = [](void** ptr, hipIpcMemHandle_t, unsigned) {
        if (ptr) *ptr = reinterpret_cast<void*>(0x9000);
        return hipSuccess;
    };
    g_hipIpcCloseMemHandle = [](void*) { return hipSuccess; };

    // The shadow-pool fake hands back the same buffer for device and host, so a
    // self-copy is skipped rather than being undefined behaviour.
    g_hipMemcpyAsync = [](void* dst, const void* src, size_t n, hipMemcpyKind, hipStream_t) {
        if (dst != nullptr && src != nullptr && dst != src) std::memcpy(dst, src, n);
        return hipSuccess;
    };
    g_hipMemsetAsync = [](void* dst, int value, size_t n, hipStream_t) {
        if (dst != nullptr) std::memset(dst, value, n);
        return hipSuccess;
    };

    g_hipStreamCreateWithFlags = [](hipStream_t* stream, unsigned) {
        if (stream) *stream = reinterpret_cast<hipStream_t>(0x1);
        return hipSuccess;
    };
    g_hipStreamSynchronize = [](hipStream_t) { return hipSuccess; };
    g_hipStreamDestroy     = [](hipStream_t) { return hipSuccess; };
    g_hipThreadExchangeStreamCaptureMode = [](hipStreamCaptureMode* mode) {
        if (mode) *mode = hipStreamCaptureModeRelaxed;
        return hipSuccess;
    };
    g_hipGetLastError = []() { return hipSuccess; };
    g_hipSetDevice    = [](int) { return hipSuccess; };
    g_hipGetDevice    = [](int* dev) {
        if (dev) *dev = 0;
        return hipSuccess;
    };
    // alloc.h's rcclSkipCuMemFreeIfArch queries device properties and skips
    // parts of the cuMem free path on gfx950/gfx1250. gfx900 makes both arch
    // predicates false, so the ordinary free path stays exercised.
    g_hipGetDeviceProperties = [](hipDeviceProp_t* prop, int) {
        if (prop == nullptr) return hipErrorInvalidValue;
        *prop = hipDeviceProp_t{};
        std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "%s", "gfx900");
        return hipSuccess;
    };
}
// Cross-stream ordering seams; defaults preserve the replaced stubs' behaviour.
static hipError_t DefaultHipEventRecord(hipEvent_t, hipStream_t)
{
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int)
{
    return hipErrorInvalidValue;
}
std::function<hipError_t(hipEvent_t, hipStream_t)> g_hipEventRecord = DefaultHipEventRecord;
std::function<hipError_t(hipStream_t, hipEvent_t, unsigned int)> g_hipStreamWaitEvent =
    DefaultHipStreamWaitEvent;

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
    g_hipHostFree                   = DefaultHipHostFree;
    g_hipGetDevice                  = DefaultHipGetDevice;
    g_hipSetDevice                  = DefaultHipSetDevice;
    g_hipGetDeviceCount             = DefaultHipGetDeviceCount;
    g_hipDeviceCanAccessPeer        = DefaultHipDeviceCanAccessPeer;
    g_deviceCount                   = 8;
    g_currentDevice                 = 0;
    g_hipDeviceGetAttribute         = DefaultHipDeviceGetAttribute;
    g_hipDeviceSetLimit             = DefaultHipDeviceSetLimit;
    g_hipDeviceGetAttributeResult   = hipErrorInvalidValue;
    g_hipDeviceGetPCIBusIdResult    = hipErrorInvalidValue;
    g_hipEventCreateResult          = hipErrorInvalidValue;
    g_hipMemPoolResult              = hipErrorInvalidValue;
    g_hipStreamCreateResult         = hipErrorInvalidValue;
    g_hipAsyncOpsResult             = hipErrorInvalidValue;
    g_hipWarpSize                   = 64;
    g_hipDirectManagedMemAccess     = 1;
    g_hipMemcpyAsyncCalls           = 0;
    g_hipMemcpyAsyncArgs.clear();
    // VMM / IPC / stream seams (undoes InstallHipVmmEmulator too)
    g_hipMemAddressReserve          = DefaultHipMemAddressReserve;
    g_hipMemAddressFree             = DefaultHipMemAddressFree;
    g_hipMemCreate                  = DefaultHipMemCreate;
    g_hipMemGetAllocationGranularity = DefaultHipMemGetAllocationGranularity;
    g_hipMemGetAllocationPropertiesFromHandle = DefaultHipMemGetAllocationPropertiesFromHandle;
    g_hipMemImportFromShareableHandle = DefaultHipMemImportFromShareableHandle;
    g_hipMemMap                     = DefaultHipMemMap;
    g_hipMemSetAccess               = DefaultHipMemSetAccess;
    g_hipMemUnmap                   = DefaultHipMemUnmap;
    g_hipIpcOpenMemHandle           = DefaultHipIpcOpenMemHandle;
    g_hipIpcCloseMemHandle          = DefaultHipIpcCloseMemHandle;
    g_hipMemcpy                     = DefaultHipMemcpy;
    g_hipMemcpyAsync                = DefaultHipMemcpyAsync;
    g_hipMemsetAsync                = DefaultHipMemsetAsync;
    g_hipStreamCreateWithFlags      = DefaultHipStreamCreateWithFlags;
    g_hipStreamSynchronize          = DefaultHipStreamSynchronize;
    g_hipStreamDestroy              = DefaultHipStreamDestroy;
    g_hipThreadExchangeStreamCaptureMode = DefaultHipThreadExchangeStreamCaptureMode;
    g_hipGetLastError               = DefaultHipGetLastError;
    g_hipEventRecord                = DefaultHipEventRecord;
    g_hipStreamWaitEvent            = DefaultHipStreamWaitEvent;
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

hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle)
{
    return g_hipMemRelease(handle);
}

// --- plain link-satisfying stubs (unexercised paths) --------------------
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int dev1, int dev2)
{
    return g_hipDeviceCanAccessPeer(canAccessPeer, dev1, dev2);
}

hipError_t hipDeviceEnablePeerAccess(int, unsigned int)
{
    return hipErrorInvalidValue;
}

hipError_t hipDeviceGet(hipDevice_t* device, int)
{
    // fillInfo() CUCHECKs cuDeviceGet (hipDeviceGet) on every rank before it
    // probes the fabric-handle attribute, so this must succeed on the default
    // TransportsRankComm path or initTransportsRank never reaches the postset.
    if (device) *device = 0;
    return hipSuccess;
}

hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t)
{
    if (uuid) {
        *uuid = hipUUID{};
    }
    return hipSuccess;
}

static hipError_t DefaultHipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int)
{
    if (!pi) return g_hipDeviceGetAttributeResult;
    switch (attr) {
        case hipDeviceAttributeWarpSize:
            *pi = g_hipWarpSize; break;
        case hipDeviceAttributeDirectManagedMemAccessFromHost:
            *pi = g_hipDirectManagedMemAccess; break;   // 1 -> ncclCudaHostCalloc takes the extMalloc arm
        default:
            *pi = 0; break;
    }
    return g_hipDeviceGetAttributeResult;
}
std::function<hipError_t(int*, hipDeviceAttribute_t, int)>
    g_hipDeviceGetAttribute = DefaultHipDeviceGetAttribute;

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device)
{
    return g_hipDeviceGetAttribute(pi, attr, device);
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
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)
{
    return g_hipEventRecord(event, stream);
}

hipError_t hipExtMallocWithFlags(void** ptr, size_t size, unsigned int flags)
{
    return g_hipExtMallocWithFlags(ptr, size, flags);
}

hipError_t hipFree(void* ptr) { return g_hipFree(ptr); }

hipError_t hipGetDevice(int* deviceId) { return g_hipGetDevice(deviceId); }

hipError_t hipGetDeviceCount(int* count) { return g_hipGetDeviceCount(count); }

const char* hipGetErrorString(hipError_t) { return "[hip_fake] stub error"; }

hipError_t hipGetLastError(void) { return g_hipGetLastError(); }

hipError_t hipHostFree(void* ptr) { return g_hipHostFree(ptr); }

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags)
{
    return g_hipHostMalloc(ptr, size, flags);
}

hipError_t hipIpcCloseMemHandle(void* ptr) { return g_hipIpcCloseMemHandle(ptr); }

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags)
{
    return g_hipIpcOpenMemHandle(devPtr, handle, flags);
}

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr)
{
    return g_hipIpcGetMemHandle(handle, devPtr);
}

hipError_t hipMemAddressFree(void* ptr, size_t size)
{
    return g_hipMemAddressFree(ptr, size);
}

hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags)
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
                                          hipMemAllocationGranularity_flags flags)
{
    return g_hipMemGetAllocationGranularity(granularity, prop, flags);
}

hipError_t hipMemGetAllocationPropertiesFromHandle(
    hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle)
{
    return g_hipMemGetAllocationPropertiesFromHandle(prop, handle);
}

hipError_t hipMemExportToShareableHandle(void* shareable,
                                         hipMemGenericAllocationHandle_t handle,
                                         hipMemAllocationHandleType type,
                                         unsigned long long flags)
{
    return g_hipMemExportToShareableHandle(shareable, handle, type, flags);
}

hipError_t hipMemImportFromShareableHandle(
    hipMemGenericAllocationHandle_t* handle, void* shareable, hipMemAllocationHandleType type)
{
    return g_hipMemImportFromShareableHandle(handle, shareable, type);
}

hipError_t hipMemMap(void* ptr, size_t size, size_t offset,
                     hipMemGenericAllocationHandle_t handle, unsigned long long flags)
{
    return g_hipMemMap(ptr, size, offset, handle, flags);
}

hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count)
{
    return g_hipMemSetAccess(ptr, size, desc, count);
}

hipError_t hipMemUnmap(void* ptr, size_t size) { return g_hipMemUnmap(ptr, size); }

// The recording stays in the wrapper rather than the default hook, so a test
// that installs its own copy behaviour still gets the call log p2p-test asserts on.
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind,
                          hipStream_t stream)
{
    g_hipMemcpyAsyncCalls++;
    g_hipMemcpyAsyncArgs.push_back({dst, src, bytes});
    return g_hipMemcpyAsync(dst, src, bytes, kind, stream);
}

hipError_t hipMemsetAsync(void* dst, int value, size_t bytes, hipStream_t stream)
{
    return g_hipMemsetAsync(dst, value, bytes, stream);
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

hipError_t hipStreamDestroy(hipStream_t s)     { return g_hipStreamDestroy(s); }
hipError_t hipStreamSynchronize(hipStream_t s) { return g_hipStreamSynchronize(s); }

hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode)
{
    return g_hipThreadExchangeStreamCaptureMode(mode);
}

hipError_t hipSetDevice(int deviceId) { return g_hipSetDevice(deviceId); }
hipError_t hipMalloc(void** p, size_t) { if (p) *p = nullptr; return hipErrorInvalidValue; }
hipError_t hipMemcpy(void* d, const void* s, size_t n, hipMemcpyKind k) { return g_hipMemcpy(d, s, n, k); }
hipError_t hipMemset(void*, int, size_t) { return hipErrorInvalidValue; }
hipError_t hipDeviceSynchronize(void) { return hipErrorInvalidValue; }
// We define hipGetDevicePropertiesR0600, the versioned public HIP ABI symbol,
// rather than the unversioned hipGetDeviceProperties. Callers write
// hipGetDeviceProperties in source, but the HIP header (post rocm-systems
// #10358) provides a `static inline hipGetDeviceProperties` wrapper that
// delegates to hipGetDevicePropertiesR0600, so every call site inlines down to
// a reference to that R0600 symbol -- which is what our fake must supply.
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_t* prop, int device)
{
    return g_hipGetDeviceProperties(prop, device);
}
hipError_t hipDriverGetVersion(int* v) { if (v) *v = 70002000; return hipSuccess; }
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)
{
    return g_hipStreamWaitEvent(stream, event, flags);
}
hipError_t hipStreamCreate(hipStream_t*) { return hipErrorInvalidValue; }
// hipStreamCreateWithPriority / hipDeviceGetStreamPriorityRange are defined
// above (seam-routed) -- the develop merge added plainer duplicates here.
hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*) { return hipErrorInvalidValue; }
hipError_t hipHostGetDevicePointer(void**, void*, unsigned int) { return hipErrorInvalidValue; }
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t*, hipEvent_t) { return hipErrorInvalidValue; }
hipError_t hipEventSynchronize(hipEvent_t) { return hipErrorInvalidValue; }

// --- init.cc deep-path HIP stubs (commAlloc/devCommSetup) ---------------
hipError_t hipRuntimeGetVersion(int* version) { return g_hipRuntimeGetVersion(version); }
static hipError_t DefaultHipDeviceSetLimit(hipLimit_t, size_t) { return hipErrorInvalidValue; }
std::function<hipError_t(hipLimit_t, size_t)> g_hipDeviceSetLimit = DefaultHipDeviceSetLimit;
hipError_t hipDeviceSetLimit(hipLimit_t limit, size_t value) { return g_hipDeviceSetLimit(limit, value); }
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

// ---- HIP entry points ROCm's profile runtime also defines -------------------
// These MUST stay defined even though no microtest calls them. Left undefined,
// -fprofile-instr-generate lets the linker satisfy them from the INTERCEPTOR in
// libclang_rt.profile, which drags in __interception/__sanitizer deps that ship
// in no archive. See hip_profile_interceptor_fakes.h for the full chain.
//
// Aborting rather than returning hipErrorInvalidValue: a host-only microtest
// that reaches a kernel launch or module load is broken, not merely unexercised.

// extern "C" is stated rather than inherited from the HIP declaration: if a
// future SDK drops one of these, the definition still emits the C symbol the
// linker needs instead of silently emitting a mangled one.
extern "C" {
#define RCCL_DEFINE_HIP_STUB(name, params) \
  hipError_t name params { FailLoudUnfaked("hip_fakes", #name); }
RCCL_HIP_PROFILE_INTERCEPTORS(RCCL_DEFINE_HIP_STUB)
#undef RCCL_DEFINE_HIP_STUB
}  // extern "C"
