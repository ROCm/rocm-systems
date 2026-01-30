/*
 * Stub for onerank.cu when DEVICE_LINKER is enabled.
 * Onerank is excluded to maintain a single fatbin.
 */

#include "nccl.h"
#include "device.h"
#include "collectives.h"
#include <hip/hip_runtime.h>

ncclResult_t ncclLaunchOneRank(void* dst, void const* src, size_t nElts, 
                               struct ncclDevRedOpFull redOp, ncclDataType_t eltType, 
                               hipStream_t stream) {
    // Onerank optimization is disabled - fall back to memcpy
    if (dst != src) {
        size_t eltSize = ncclTypeSize(eltType);
        hipError_t err = hipMemcpyAsync(dst, src, nElts * eltSize, hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) return ncclUnhandledCudaError;
    }
    return ncclSuccess;
}
