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

// Models a rocprofiler-sdk client tool library: it exports rocprofiler_configure, so it is
// discovered by rocprofiler-sdk's link-map symbol walk during OMPT tool bring-up. It depends
// on the blas-stub library (see CMakeLists.txt), so blas-stub's DT_INIT runs first and this
// library's DT_INIT is still pending when discovery occurs -- the exact condition under which
// a dlopen of this library would re-run the constructor below and deadlock libomp. The
// constructor calls omp_get_num_places() (which re-enters libomp's init path) to model that.

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <omp.h>

#include <cstdio>

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    // a no-op client: returning nullptr means "do not activate", which is sufficient for
    // exercising the discovery path that previously deadlocked
    return nullptr;
}

namespace
{
__attribute__((constructor)) void
configure_client_init()
{
    fprintf(stderr, "[client] DT_INIT (still pending during discovery): omp_get_num_places()\n");
    fflush(stderr);
    int n = omp_get_num_places();
    fprintf(stderr, "[client] omp_get_num_places() returned %d\n", n);
    fflush(stderr);
}
}  // namespace
