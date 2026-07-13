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

/**
 * @file tests/hip-buffered-dangling-arg/test_hip_buffered_dangling_arg.cpp
 *
 * @brief Workload that calls a HIP API taking a `const char*` whose storage is
 * freed before the buffered tracing record is flushed.
 *
 * hipDeviceGetByPCIBusId takes a `const char* pciBusId` argument. The
 * rocprofiler-sdk HIP tracer records this argument into a buffered (`*_EXT`)
 * record and only stringifies it later, when the buffer is flushed on the
 * rocprofiler callback thread. This test deliberately passes the string via a
 * short-lived std::string temporary whose backing buffer is released as soon as
 * the enclosing scope exits -- i.e. long before the flush. Unless the tracer
 * interns the argument (copies it into stable storage), the flush dereferences a
 * dangling pointer and segfaults.
 *
 * Paired with client.cpp, which enables buffered HIP API tracing and iterates
 * each record's arguments (forcing the stringification / dereference).
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <string>

#define HIP_CHECK(expr)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (expr);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            std::cerr << "HIP error " << hipGetErrorString(_err) << " at " << __FILE__ << ":"      \
                      << __LINE__ << "\n";                                                         \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
// Issue one traced hipDeviceGetByPCIBusId call, passing the PCI bus-id via a
// std::string temporary that is destroyed the instant this function returns. The
// const char* handed to the HIP API is therefore invalid by the time the
// buffered record is flushed. The call itself is expected to fail (the bus id is
// synthetic) -- the argument is still recorded on the enter phase, which is all
// this test needs.
//
// The string is deliberately padded well past the small-string-optimization
// threshold so its characters live on the heap, not inline in the std::string.
// A heap block that is freed on return (and reused by later iterations) makes
// the dangling read observable -- as a segfault, or deterministically under the
// sanitizer/memcheck test variants -- which an SSO stack buffer would not.
void
query_device_with_transient_busid(int index)
{
    int         device = -1;
    std::string bus_id =
        std::string("0000:0000:0000:00:00.0-transient-busid-") + std::to_string(index);
    (void) hipDeviceGetByPCIBusId(&device, bus_id.c_str());
}
}  // namespace

int
main()
{
    std::cout << "HIP buffered dangling-arg test\n";

    int num_gpus = 0;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    if(num_gpus == 0)
    {
        std::cerr << "No GPUs found. Test requires at least one GPU.\n";
        return 1;
    }

    // Repeat a few times so the buffer is likely to flush mid-run (not only at
    // finalize), broadening coverage of the deferred-stringify path.
    constexpr int iterations = 8;
    for(int i = 0; i < iterations; ++i)
    {
        query_device_with_transient_busid(i);
    }

    std::cout << "Test completed successfully\n";
    return 0;
}
