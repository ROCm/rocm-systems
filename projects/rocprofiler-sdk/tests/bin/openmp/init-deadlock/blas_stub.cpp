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

// Models an OpenBLAS-style library built with OpenMP support (USE_OPENMP=1). The deadlock
// originates in OpenBLAS's DT_INIT constructor, which queries the CPU/thread count through
// OpenMP:
//
//     gotoblas_init -> blas_get_cpu_number -> get_num_procs -> omp_get_num_places
//
// The call into libomp (omp_get_num_places) forces libomp's first-touch initialization, which
// acquires libomp's non-recursive init lock and performs OMPT tool discovery while this
// constructor is still on the stack. This stub reproduces that exact libomp entry point rather
// than a generic parallel region so the call chain matches the real failure.

#include <omp.h>

#include <cstdio>

namespace
{
__attribute__((constructor)) void
blas_stub_init()
{
    fprintf(stderr, "[blas-stub] DT_INIT: omp_get_num_places() (models gotoblas_init CPU query)\n");
    fflush(stderr);
    int n = omp_get_num_places();
    fprintf(stderr, "[blas-stub] omp_get_num_places() returned %d\n", n);
    fflush(stderr);
}
}  // namespace
