// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/gpu.hpp"

#if defined(OMNITRACE_USE_ROCM_SMI)
#    include "library/components/rocm_smi.hpp"
#elif defined(OMNITRACE_USE_HIP)
#    if !defined(TIMEMORY_USE_HIP)
#        define TIMEMORY_USE_HIP 1
#    endif
#    include "timemory/components/hip/backends.hpp"
#endif

namespace omnitrace
{
namespace gpu
{
int
device_count()
{
#if defined(OMNITRACE_USE_ROCM_SMI)
    // store as static since calls after rsmi_shutdown will return zero
    static auto _v = rocm_smi::device_count();
    return _v;
#elif defined(OMNITRACE_USE_HIP)
    return ::tim::hip::device_count();
#else
    return 0;
#endif
}
}  // namespace gpu
}  // namespace omnitrace
