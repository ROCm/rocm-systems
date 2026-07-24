// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hipfile.h>

#include <cstdio>
#include <cstdlib>

namespace
{
bool
is_success(hipFileError_t err)
{
    return err.err == hipFileSuccess && err.hip_drv_err == hipSuccess;
}
}  // namespace

// Minimal hipFILE workload used to exercise rocprofiler-systems hipFILE API
// tracing (ROCPROFSYS_ROCM_DOMAINS=hipfile_api). It only calls host-side driver
// APIs so it can run without a valid GPU file mapping.
int
main()
{
    unsigned major = 0;
    unsigned minor = 0;
    unsigned patch = 0;

    auto version = hipFileGetVersion(&major, &minor, &patch);
    if(!is_success(version))
    {
        std::fprintf(stderr, "hipFileGetVersion failed: %d\n",
                     static_cast<int>(version.err));
        return EXIT_FAILURE;
    }
    std::printf("hipFILE version: %u.%u.%u\n", major, minor, patch);

    const auto* err_string = hipFileGetOpErrorString(hipFileSuccess);
    if(err_string == nullptr) return EXIT_FAILURE;

    (void) hipFileUseCount();

    size_t size_value = 0;
    (void) hipFileGetParameterSizeT(hipFileParamExecutionMaxIOThreads, &size_value);
    (void) hipFileSetParameterSizeT(hipFileParamExecutionMaxIOThreads, size_value);

    bool bool_value = false;
    (void) hipFileGetParameterBool(hipFileParamPropertiesUsePollMode, &bool_value);
    (void) hipFileSetParameterBool(hipFileParamPropertiesUsePollMode, bool_value);

    auto open = hipFileDriverOpen();
    (void) hipFileUseCount();

    if(open.err == hipFileSuccess)
    {
        hipFileDriverProps_t props = {};
        (void) hipFileDriverGetProperties(&props);

        while(hipFileUseCount() > 0)
            (void) hipFileDriverClose();
    }

    return EXIT_SUCCESS;
}
