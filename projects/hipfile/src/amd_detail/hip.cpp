/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "hip.h"
#include "static.h"

#include <cstdlib>
#include <system_error>

namespace hipFile {

static void *
hipGetProcAddressHelper(const char *symbol)
try {
    auto hip_version{Context<Hip>::get()->hipRuntimeGetVersion()};
    return Context<Hip>::get()->hipGetProcAddress(symbol, hip_version, 0, nullptr);
}
catch (...) {
    return nullptr;
}

hipAmdFileRead_t
getHipAmdFileReadPtr()
{
    HIPFILE_STATIC hipAmdFileRead_t hipAmdFileReadPtr{
        reinterpret_cast<hipAmdFileRead_t>(hipGetProcAddressHelper("hipAmdFileRead"))};
    return hipAmdFileReadPtr;
}

hipAmdFileWrite_t
getHipAmdFileWritePtr()
{
    HIPFILE_STATIC hipAmdFileWrite_t hipAmdFileWritePtr{
        reinterpret_cast<hipAmdFileWrite_t>(hipGetProcAddressHelper("hipAmdFileWrite"))};
    return hipAmdFileWritePtr;
}

/// Throws Hip::RuntimeError(error) if error != hipSuccess
template <typename ExceptionType>
static inline hipError_t
throwOnHipError(hipError_t error)
{
    if (hipSuccess != error) {
        throw ExceptionType(error);
    }
    return error;
}

hipPointerAttribute_t
Hip::hipPointerGetAttributes(const void *ptr) const
{
    hipPointerAttribute_t attributes;
    (void)throwOnHipError<Hip::RuntimeError>(::hipPointerGetAttributes(&attributes, ptr));
    return attributes;
}

void
Hip::hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) const
{
    (void)throwOnHipError<Hip::RuntimeError>(::hipMemcpy(dst, src, sizeBytes, kind));
}

void
Hip::hipStreamSynchronize(hipStream_t stream) const
{
    (void)throwOnHipError<Hip::RuntimeError>(::hipStreamSynchronize(stream));
}

void *
Hip::hipHostMalloc(size_t size, unsigned int flags) const
{
    void *host_ptr;
    (void)throwOnHipError<Hip::RuntimeError>(::hipHostMalloc(&host_ptr, size, flags));
    return host_ptr;
}

void
Hip::hipHostFree(void *ptr) const
{
    (void)throwOnHipError<Hip::RuntimeError>(::hipHostFree(ptr));
}

void *
Hip::hipHostGetDevicePointer(void *hstPtr, unsigned int flags) const
{
    void *dev_ptr;
    (void)throwOnHipError<Hip::RuntimeError>(::hipHostGetDevicePointer(&dev_ptr, hstPtr, flags));
    return dev_ptr;
}

HipMemAddressRange
Hip::hipMemGetAddressRange(hipDeviceptr_t dptr) const
{
    HipMemAddressRange range;
    (void)throwOnHipError<Hip::RuntimeError>(::hipMemGetAddressRange(&range.base, &range.size, dptr));
    return range;
}

int
Hip::hipRuntimeGetVersion() const
{
    int version;
    (void)throwOnHipError<Hip::RuntimeError>(::hipRuntimeGetVersion(&version));
    return version;
}

void *
Hip::hipGetProcAddress(const char *symbol, int hipVersion, uint64_t flags,
                       hipDriverProcAddressQueryResult *symbolStatus) const
{
    void *ptr;
    (void)throwOnHipError<Hip::RuntimeError>(
        ::hipGetProcAddress(symbol, &ptr, hipVersion, flags, symbolStatus));
    return ptr;
}

uint64_t
Hip::hipAmdFileRead(hipAmdFileHandle_t handle, void *devicePtr, uint64_t size, int64_t file_offset) const
{
    static const hipAmdFileRead_t hipAmdFileReadPtr{getHipAmdFileReadPtr()};
    uint64_t                      bytes_read{};
    int                           status{};

    if (!hipAmdFileReadPtr) {
        throw Hip::SymbolNotFound("Could not find hipAmdFileRead()");
    }

    auto hip_error{(*hipAmdFileReadPtr)(handle, devicePtr, size, file_offset, &bytes_read, &status)};

    if (status) {
        throw std::system_error(abs(status), std::generic_category());
    }
    else if (hip_error) {
        throw Hip::RuntimeError(hip_error);
    }

    return bytes_read;
}

uint64_t
Hip::hipAmdFileWrite(hipAmdFileHandle_t handle, void *devicePtr, uint64_t size, int64_t file_offset) const
{
    static const hipAmdFileWrite_t hipAmdFileWritePtr{getHipAmdFileWritePtr()};
    uint64_t                       bytes_written{};
    int32_t                        status{};

    if (!hipAmdFileWritePtr) {
        throw Hip::SymbolNotFound("Could not find hipAmdFileWrite()");
    }

    auto hip_error{(*hipAmdFileWritePtr)(handle, devicePtr, size, file_offset, &bytes_written, &status)};

    if (status) {
        throw std::system_error(abs(status), std::generic_category());
    }
    else if (hip_error) {
        throw Hip::RuntimeError(hip_error);
    }

    return bytes_written;
}

void
Hip::hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void *user_data) const
{
    (void)throwOnHipError<Hip::RuntimeError>(::hipLaunchHostFunc(stream, fn, user_data));
}

void
Hip::hipLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args,
                     size_t sharedMemBytes, hipStream_t stream) const
{
    (void)throwOnHipError<Hip::RuntimeError>(
        ::hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream));
}

int
Hip::hipDeviceGetAttribute(hipDeviceAttribute_t attr, int device_id) const
{
    int attr_value;
    (void)throwOnHipError<Hip::RuntimeError>(::hipDeviceGetAttribute(&attr_value, attr, device_id));
    return attr_value;
}

hipDevice_t
Hip::hipStreamGetDevice(hipStream_t stream) const
{
    hipDevice_t device_id;
    (void)throwOnHipError<Hip::RuntimeError>(::hipStreamGetDevice(stream, &device_id));
    return device_id;
}

void
Hip::hipInit() const
{
    (void)throwOnHipError<Hip::RuntimeError>(::hipInit(0));
}

int
Hip::hipGetDevice() const
{
    int device_id;
    (void)throwOnHipError<Hip::RuntimeError>(::hipGetDevice(&device_id));
    return device_id;
}

int
Hip::hipGetDeviceCount() const
{
    int device_count;
    (void)throwOnHipError<Hip::RuntimeError>(::hipGetDeviceCount(&device_count));
    return device_count;
}
}
