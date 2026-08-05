/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// HIP runtime fakes for rccl-UnitTestsMicro.
//
// The micro-test binary deliberately does NOT link the real HIP runtime
// (libamdhip64.so): test/CMakeLists.txt pulls hip::host / hip::device in
// via $<COMPILE_ONLY:...> so the HIP *compile* requirements (amdgcn
// device intrinsics, gfx target codegen, headers) are satisfied while the
// runtime stays off the link line. That means every HIP host-API entry
// point the linked object code references must be provided here -- a
// missing one is now a link-time error instead of a silent bind to the
// real driver (which, on a box with a newer HIP than the tests were
// written against, is exactly how an un-shimmed hipPointerGetAttribute
// slipped through and aborted a test at runtime).
//
// Two kinds of definitions live here:
//
//   1. Controllable HIP seams (g_hip*) -- the handful of HIP calls the
//      unit-under-test drives on its happy path. Tests install per-test
//      behaviour by overwriting these std::function hooks; the macro shims
//      in p2p-test.cc route the p2p.cc call sites through them.
//
//   2. Plain link-satisfying stubs for every other real HIP symbol the
//      binary references. These are code paths the micro-tests don't
//      exercise; the stubs zero any output params and return
//      hipErrorInvalidValue so an unexpected call surfaces loudly rather
//      than pretending to succeed.

#include <cstring>
#include <cstdlib>
#include <functional>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "hip_fakes.h"   // g_hip* hook declarations + ResetHipFakes()

// ===========================================================================
// Section 1: controllable HIP seams
//
// Moved verbatim from p2p_fakes.cc. Defaults return hipErrorInvalidValue so
// any test that doesn't opt in surfaces the unexpected call as
// ncclUnhandledCudaError via CUCHECKGOTO.
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
// nothing-works arms stay reachable. Tests force the legacy arm by
// installing ForceLegacyIpcCapable() (p2p-test.cc).
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

// --- Device model (init.cc Tier-C): runtime version + device properties -----
static hipError_t DefaultHipRuntimeGetVersion(int* version)
{
    if (version) *version = 60443484;  // plausible ROCm 6.x runtime version
    return hipSuccess;
}
static hipError_t DefaultHipGetDeviceProperties(hipDeviceProp_t* prop, int)
{
    if (prop) {
        *prop = hipDeviceProp_t{};
        const char* arch = "gfx942:sramecc+:xnack-";
        std::strncpy(prop->gcnArchName, arch, sizeof(prop->gcnArchName) - 1);
        prop->gcnArchName[sizeof(prop->gcnArchName) - 1] = '\0';
        prop->totalGlobalMem = static_cast<size_t>(64) << 30;
        prop->warpSize = 64;
    }
    return hipSuccess;
}
std::function<hipError_t(int*)> g_hipRuntimeGetVersion = DefaultHipRuntimeGetVersion;
std::function<hipError_t(hipDeviceProp_t*, int)>
    g_hipGetDeviceProperties = DefaultHipGetDeviceProperties;

static hipError_t DefaultHipExtMallocWithFlags(void** ptr, std::size_t size, unsigned)
{
    if (ptr) { *ptr = std::malloc(size ? size : 1); return *ptr ? hipSuccess : hipErrorOutOfMemory; }
    return hipErrorInvalidValue;
}
static hipError_t DefaultHipFree(void* ptr) { std::free(ptr); return hipSuccess; }
std::function<hipError_t(void**, std::size_t, unsigned)>
    g_hipExtMallocWithFlags = DefaultHipExtMallocWithFlags;
std::function<hipError_t(void*)> g_hipFree = DefaultHipFree;

int g_deviceCount = 8;
int g_currentDevice = 0;
static hipError_t DefaultHipGetDevice(int* dev) { if (dev) *dev = g_currentDevice; return hipSuccess; }
static hipError_t DefaultHipSetDevice(int dev) { g_currentDevice = dev; return hipSuccess; }
static hipError_t DefaultHipGetDeviceCount(int* count) { if (count) *count = g_deviceCount; return hipSuccess; }
std::function<hipError_t(int*)> g_hipGetDevice = DefaultHipGetDevice;
std::function<hipError_t(int)> g_hipSetDevice = DefaultHipSetDevice;
std::function<hipError_t(int*)> g_hipGetDeviceCount = DefaultHipGetDeviceCount;

// Restore every HIP hook to its default. Called from ResetP2pFakes().
void ResetHipFakes()
{
    g_hipMemGetAddressRange         = DefaultHipMemGetAddressRange;
    g_hipIpcGetMemHandle            = DefaultHipIpcGetMemHandle;
    g_hipMemRetainAllocationHandle  = DefaultHipMemRetainAllocationHandle;
    g_hipMemExportToShareableHandle = DefaultHipMemExportToShareableHandle;
    g_hipMemRelease                 = DefaultHipMemRelease;
    g_hipPointerGetAttribute        = DefaultHipPointerGetAttribute;
    g_hipRuntimeGetVersion          = DefaultHipRuntimeGetVersion;
    g_hipGetDeviceProperties        = DefaultHipGetDeviceProperties;
    g_hipExtMallocWithFlags         = DefaultHipExtMallocWithFlags;
    g_hipFree                       = DefaultHipFree;
    g_hipGetDevice                  = DefaultHipGetDevice;
    g_hipSetDevice                  = DefaultHipSetDevice;
    g_hipGetDeviceCount             = DefaultHipGetDeviceCount;
    g_deviceCount                   = 8;
    g_currentDevice                 = 0;
}

// ===========================================================================
// Section 2: real HIP runtime symbol stubs
//
// These provide the exact extern-"C" symbols the linked object code
// references so the binary links without libamdhip64.so. The signatures
// match hip_runtime_api.h (included above), so C linkage is inherited from
// the prior declarations.
//
// Three of these (hipMemGetAddressRange, hipMemRetainAllocationHandle,
// hipMemRelease) are also driven through the controllable seams above from
// macro-shimmed call sites in p2p-test.cc; here the *real* symbol delegates
// to the same hook so any non-shimmed call site behaves identically and
// there is a single behavioural source of truth per HIP function. All other
// stubs are for paths the micro-tests don't exercise.
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

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t, int)
{
    if (pi) *pi = 0;
    return hipErrorInvalidValue;
}

hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int)
{
    if (pciBusId && len > 0) pciBusId[0] = '\0';
    return hipErrorInvalidValue;
}

hipError_t hipEventCreate(hipEvent_t* event)
{
    if (event) *event = nullptr;
    return hipErrorInvalidValue;
}

hipError_t hipEventDestroy(hipEvent_t)      { return hipErrorInvalidValue; }
hipError_t hipEventQuery(hipEvent_t)        { return hipErrorInvalidValue; }
hipError_t hipEventRecord(hipEvent_t, hipStream_t) { return hipErrorInvalidValue; }

hipError_t hipExtMallocWithFlags(void** ptr, size_t size, unsigned int flags)
{
    return g_hipExtMallocWithFlags(ptr, size, flags);
}

hipError_t hipFree(void* ptr) { return g_hipFree(ptr); }

hipError_t hipGetDevice(int* deviceId) { return g_hipGetDevice(deviceId); }
hipError_t hipSetDevice(int deviceId) { return g_hipSetDevice(deviceId); }

hipError_t hipGetDeviceCount(int* count) { return g_hipGetDeviceCount(count); }

// Deep-path HIP stubs (commAlloc/devCommSetup); not exercised by Tier A-D tests.
hipError_t hipDeviceSetLimit(hipLimit_t, size_t) { return hipErrorInvalidValue; }
hipError_t hipEventCreateWithFlags(hipEvent_t*, unsigned int) { return hipErrorInvalidValue; }
hipError_t hipEventSynchronize(hipEvent_t) { return hipErrorInvalidValue; }
hipError_t hipMemPoolCreate(hipMemPool_t*, const hipMemPoolProps*) { return hipErrorInvalidValue; }
hipError_t hipMemPoolDestroy(hipMemPool_t) { return hipErrorInvalidValue; }
hipError_t hipMemPoolSetAttribute(hipMemPool_t, hipMemPoolAttr, void*) { return hipErrorInvalidValue; }
hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*) { return hipErrorInvalidValue; }

// Device-model symbols routed through the controllable hooks above. After
// hipify, init.cc's hipGetDeviceProperties call binds to hipGetDevicePropertiesR0600.
hipError_t hipRuntimeGetVersion(int* version) { return g_hipRuntimeGetVersion(version); }
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_t* prop, int device)
{
    return g_hipGetDeviceProperties(prop, device);
}

const char* hipGetErrorString(hipError_t) { return "[hip_fake] stub error"; }

hipError_t hipGetLastError(void) { return hipErrorInvalidValue; }

hipError_t hipHostFree(void*) { return hipErrorInvalidValue; }

hipError_t hipHostMalloc(void** ptr, size_t, unsigned int)
{
    if (ptr) *ptr = nullptr;
    return hipErrorInvalidValue;
}

hipError_t hipIpcCloseMemHandle(void*) { return hipErrorInvalidValue; }

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t, unsigned int)
{
    if (devPtr) *devPtr = nullptr;
    return hipErrorInvalidValue;
}

hipError_t hipMemAddressFree(void*, size_t) { return hipErrorInvalidValue; }

hipError_t hipMemAddressReserve(void** ptr, size_t, size_t, void*,
                                unsigned long long)
{
    if (ptr) *ptr = nullptr;
    return hipErrorInvalidValue;
}

hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t,
                        const hipMemAllocationProp*, unsigned long long)
{
    if (handle) *handle = nullptr;
    return hipErrorInvalidValue;
}

hipError_t hipMemGetAllocationGranularity(size_t* granularity,
                                          const hipMemAllocationProp*,
                                          hipMemAllocationGranularity_flags)
{
    if (granularity) *granularity = 0;
    return hipErrorInvalidValue;
}

hipError_t hipMemGetAllocationPropertiesFromHandle(
    hipMemAllocationProp*, hipMemGenericAllocationHandle_t)
{
    return hipErrorInvalidValue;
}

hipError_t hipMemImportFromShareableHandle(
    hipMemGenericAllocationHandle_t* handle, void*, hipMemAllocationHandleType)
{
    if (handle) *handle = nullptr;
    return hipErrorInvalidValue;
}

hipError_t hipMemMap(void*, size_t, size_t, hipMemGenericAllocationHandle_t,
                     unsigned long long)
{
    return hipErrorInvalidValue;
}

hipError_t hipMemSetAccess(void*, size_t, const hipMemAccessDesc*, size_t)
{
    return hipErrorInvalidValue;
}

hipError_t hipMemUnmap(void*, size_t) { return hipErrorInvalidValue; }

hipError_t hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind,
                          hipStream_t)
{
    return hipErrorInvalidValue;
}

hipError_t hipMemsetAsync(void*, int, size_t, hipStream_t)
{
    return hipErrorInvalidValue;
}

hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr)
{
    return g_hipPointerGetAttribute(data, attribute, ptr);
}

hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int)
{
    if (stream) *stream = nullptr;
    return hipErrorInvalidValue;
}

hipError_t hipStreamDestroy(hipStream_t)     { return hipErrorInvalidValue; }
hipError_t hipStreamSynchronize(hipStream_t) { return hipErrorInvalidValue; }

hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode*)
{
    return hipErrorInvalidValue;
}
