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

#include <cstring>
#include <functional>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "hip_fakes.h"   // g_hip* hook declarations + ResetHipFakes()

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

// Restore every HIP hook to its default.
void ResetHipFakes()
{
    g_hipMemGetAddressRange         = DefaultHipMemGetAddressRange;
    g_hipIpcGetMemHandle            = DefaultHipIpcGetMemHandle;
    g_hipMemRetainAllocationHandle  = DefaultHipMemRetainAllocationHandle;
    g_hipMemExportToShareableHandle = DefaultHipMemExportToShareableHandle;
    g_hipMemRelease                 = DefaultHipMemRelease;
    g_hipPointerGetAttribute        = DefaultHipPointerGetAttribute;
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

hipError_t hipExtMallocWithFlags(void** ptr, size_t, unsigned int)
{
    if (ptr) *ptr = nullptr;
    return hipErrorInvalidValue;
}

hipError_t hipFree(void*) { return hipErrorInvalidValue; }

hipError_t hipGetDevice(int* deviceId)
{
    if (deviceId) *deviceId = 0;
    return hipErrorInvalidValue;
}

hipError_t hipGetDeviceCount(int* count)
{
    if (count) *count = 0;
    return hipErrorInvalidValue;
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

hipError_t hipSetDevice(int) { return hipErrorInvalidValue; }
hipError_t hipMalloc(void** p, size_t) { if (p) *p = nullptr; return hipErrorInvalidValue; }
hipError_t hipMemcpy(void*, const void*, size_t, hipMemcpyKind) { return hipErrorInvalidValue; }
hipError_t hipMemset(void*, int, size_t) { return hipErrorInvalidValue; }
hipError_t hipDeviceSynchronize(void) { return hipErrorInvalidValue; }
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int) { return hipErrorInvalidValue; }
hipError_t hipDriverGetVersion(int* v) { if (v) *v = 70002000; return hipSuccess; }
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int) { return hipErrorInvalidValue; }
hipError_t hipStreamCreate(hipStream_t*) { return hipErrorInvalidValue; }
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int, int) { if (stream) *stream = nullptr; return hipErrorInvalidValue; }
hipError_t hipDeviceGetStreamPriorityRange(int* least, int* greatest) { if (least) *least = 0; if (greatest) *greatest = 0; return hipSuccess; }
hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*) { return hipErrorInvalidValue; }
hipError_t hipHostGetDevicePointer(void**, void*, unsigned int) { return hipErrorInvalidValue; }
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t*, hipEvent_t) { return hipErrorInvalidValue; }
hipError_t hipEventSynchronize(hipEvent_t) { return hipErrorInvalidValue; }
