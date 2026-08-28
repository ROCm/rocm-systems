// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

int
main()
{
    unsigned major = 0;
    unsigned minor = 0;
    unsigned patch = 0;

    auto version = hipFileGetVersion(&major, &minor, &patch);
    if(!is_success(version))
    {
        std::fprintf(stderr, "hipFileGetVersion failed: %d\n", static_cast<int>(version.err));
        return EXIT_FAILURE;
    }

    auto* err_string = hipFileGetOpErrorString(hipFileSuccess);
    if(err_string == nullptr) return EXIT_FAILURE;

    // Exercise APIs that do not require creating a valid file mapping.
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
