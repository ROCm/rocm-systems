/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Minimal CMake-consumer smoke test for the cmake install + find_package
// support added in NCCL 2.29.7. The build system is the actual unit under
// test: if this file compiles and links against an installed RCCL using
// only find_package(rccl), the integration works.
//
// At runtime we exercise the smallest pair of API calls that don't
// require a working CUDA/HIP runtime: query the library version (when
// available) and request a unique-id. The latter does require a working
// network stack, so we treat its failure as a soft skip rather than a
// hard error - the build-system aspect is what we care about here.

#include <rccl/rccl.h>
#include <cstdio>
#include <cstdlib>

int main() {
#if defined(NCCL_VERSION_CODE)
    std::printf("cmake_consumer: linked against RCCL with NCCL_VERSION_CODE=%d\n",
                (int)NCCL_VERSION_CODE);
#else
    std::printf("cmake_consumer: linked against RCCL (no NCCL_VERSION_CODE macro)\n");
#endif

    int runtimeVersion = 0;
    ncclResult_t r = ncclGetVersion(&runtimeVersion);
    if (r == ncclSuccess) {
        std::printf("cmake_consumer: ncclGetVersion() -> %d\n", runtimeVersion);
    } else {
        std::printf("cmake_consumer: ncclGetVersion() returned %d (%s)\n",
                    (int)r, ncclGetErrorString(r));
    }

    ncclUniqueId id;
    r = ncclGetUniqueId(&id);
    if (r != ncclSuccess) {
        std::printf("cmake_consumer: ncclGetUniqueId() returned %d (%s); "
                    "linkage OK, runtime call soft-skip.\n",
                    (int)r, ncclGetErrorString(r));
    } else {
        std::printf("cmake_consumer: ncclGetUniqueId() OK\n");
    }

    // Returning success on linkage success is the whole point of this test.
    return 0;
}
