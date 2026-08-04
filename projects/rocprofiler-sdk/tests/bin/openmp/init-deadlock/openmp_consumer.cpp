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

// Models an ordinary OpenMP-using application library (distinct from the rocprofiler-sdk tool
// library that exports rocprofiler_configure). Its DT_INIT constructor calls into libomp, which
// is the entry point that must be able to run without deadlocking when rocprofiler-sdk is
// discovering client tools during OMPT bring-up. The captured value is exposed via
// lib_get_num_places() so the executable can confirm the constructor actually ran.

#include <omp.h>

#include <cstdio>

namespace
{
int g_lib_num_places = 0;

__attribute__((constructor)) void
openmp_consumer_init()
{
    fprintf(stderr, "[consumer] DT_INIT (still pending during discovery): omp_get_num_places()\n");
    fflush(stderr);
    g_lib_num_places = omp_get_num_places();
    fprintf(stderr, "[consumer] omp_get_num_places() returned %d\n", g_lib_num_places);
    fflush(stderr);
}
}  // namespace

extern "C" int
lib_get_num_places() __attribute__((visibility("default")));

extern "C" int
lib_get_num_places()
{
    return g_lib_num_places;
}
