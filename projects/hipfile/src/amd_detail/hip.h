/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <hip/driver_types.h>
#include <stdexcept>
#include <string>

/* hipxx (hip++)
 *
 * hipxx wraps HIP APIs used by the hipFile library. This enables unit
 * tests to mock HIP APIs.
 *
 * The wrapper methods should
 *   - Throw a RuntimeError if the wrapped function fails instead of returning
 *     a hipError_t value
 *   - Use return values instead of out arguments when possible
 */

namespace hipFile {

struct HipMemAddressRange {
    hipDeviceptr_t base;
    size_t         size;
};

/// @brief Platform independent handle to an open file
///
/// This definition must stay in sync with the one defined in
/// rocm-systems:projects/clr/hipamd/include/hip/amd_detail/hip_storage.h
struct hipAmdFileHandle_t {
    union {
        void   *handle;
        int     fd;
        uint8_t pad[8];
    };
};

typedef hipError_t (*hipAmdFileRead_t)(hipAmdFileHandle_t, void *, uint64_t, int64_t, uint64_t *, int32_t *);
typedef hipError_t (*hipAmdFileWrite_t)(hipAmdFileHandle_t, void *, uint64_t, int64_t, uint64_t *, int32_t *);

/// @brief Lookup the address of hipAmdFileRead() in the hip runtime library
/// @return The address of hipAmdFileRead if the function was found, null otherwise
hipAmdFileRead_t getHipAmdFileReadPtr();

/// @brief Lookup the address of hipAmdFileWrite() in the hip runtime library
/// @return The address of hipAmdFileWrite if the function was found, null otherwise
hipAmdFileWrite_t getHipAmdFileWritePtr();

struct Hip {
    virtual ~Hip() = default;
    virtual hipPointerAttribute_t hipPointerGetAttributes(const void *ptr) const;
    virtual void     hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) const;
    virtual void     hipStreamSynchronize(hipStream_t stream) const;
    virtual void    *hipHostMalloc(size_t size, unsigned int flags) const;
    virtual void     hipHostFree(void *ptr) const;
    virtual void    *hipHostGetDevicePointer(void *hstPtr, unsigned int flags) const;
    virtual int      hipRuntimeGetVersion() const;
    virtual void    *hipGetProcAddress(const char *symbol, int hipVersion, uint64_t flags,
                                       hipDriverProcAddressQueryResult *symbolStatus) const;
    virtual uint64_t hipAmdFileRead(hipAmdFileHandle_t handle, void *devicePtr, uint64_t size,
                                    int64_t file_offset) const;
    virtual uint64_t hipAmdFileWrite(hipAmdFileHandle_t handle, void *devicePtr, uint64_t size,
                                     int64_t file_offset) const;
    virtual HipMemAddressRange hipMemGetAddressRange(hipDeviceptr_t dptr) const;
    virtual void               hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void *user_data) const;
    virtual void hipLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args,
                                 size_t sharedMemBytes, hipStream_t stream) const;
    virtual int  hipDeviceGetAttribute(hipDeviceAttribute_t attr, int device_id) const;
    virtual hipDevice_t hipStreamGetDevice(hipStream_t stream) const;
    virtual void        hipInit() const;
    virtual int         hipGetDevice() const;
    virtual int         hipGetDeviceCount() const;

    struct RuntimeError : public std::runtime_error {
        hipError_t error;

        RuntimeError(hipError_t _error) : std::runtime_error("HIP Runtime Error"), error(_error)
        {
        }
    };

    struct SymbolNotFound : public std::runtime_error {
        SymbolNotFound(const std::string &msg) : std::runtime_error(msg)
        {
        }
    };
};

}
