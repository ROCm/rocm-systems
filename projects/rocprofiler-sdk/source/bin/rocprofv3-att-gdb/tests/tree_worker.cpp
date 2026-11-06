// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include <dlfcn.h>
#include <sys/prctl.h>
#include "tree_ipc.hpp"

#if defined(ATT_GPU_FIXTURE)
#    include <hip/hip_runtime.h>
#    define HIP_CHECK(call) require((call) == hipSuccess)
__global__ void
tree_kernel(int* result)
{
    for(int i = 0; i < 100; ++i)
        atomicAdd(result, 1);
}
#endif

int
main(int argc, char** argv)
{
    require(argc == 3);
    int input = std::atoi(argv[1]), output = std::atoi(argv[2]);
#if defined(ATT_GPU_FIXTURE)
    // Permit the sibling attach test process under Linux Yama ptrace_scope=1.
    require(prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY) == 0);
    using marker_fn = int (*)(uint64_t);
    auto resume     = reinterpret_cast<marker_fn>(dlsym(RTLD_DEFAULT, "roctxProfilerResume"));
    auto pause      = reinterpret_cast<marker_fn>(dlsym(RTLD_DEFAULT, "roctxProfilerPause"));
    bool markers    = std::getenv("ATT_TEST_MARKERS");
    if(markers)
    {
        require(resume && pause);
        require(resume(0) == 0);
    }
    int* result;
    HIP_CHECK(hipMalloc(&result, sizeof(int)));
    HIP_CHECK(hipMemset(result, 0, sizeof(int)));
    tree_kernel<<<1, 64>>>(result);
    HIP_CHECK(hipDeviceSynchronize());
#endif
    put(output, 'R');
    while(get(input) == 'W')
    {
#if defined(ATT_GPU_FIXTURE)
        if(markers) require(pause(0) == 0);
        tree_kernel<<<64, 128>>>(result);
        HIP_CHECK(hipDeviceSynchronize());
        if(markers) require(resume(0) == 0);
#endif
        put(output, 'D');
    }
#if defined(ATT_GPU_FIXTURE)
    int host = 0;
    HIP_CHECK(hipMemcpy(&host, result, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(result));
    std::printf("TREE_RESULT=%d\n", host);
#endif
    return 0;
}
