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

// Regression test for an OMPT-init deadlock (see tests/openmp-init-deadlock):
//
// When rocprofiler-sdk is loaded as an OMPT tool, libomp invokes ompt_start_tool while
// holding its non-recursive __kmp_initz_lock. rocprofiler-sdk client discovery must not
// dlopen a library during this window: dlopen re-runs the pending DT_INIT of an
// already-mapped-but-not-yet-initialized library, and if that DT_INIT calls back into
// OpenMP it self-deadlocks on the ticket lock.
//
// This executable links libopenmp-init-deadlock-blas-stub.so (whose DT_INIT calls into
// OpenMP, triggering OMPT bring-up) and libopenmp-init-deadlock-client.so (which exports
// rocprofiler_configure so it is discovered, and whose DT_INIT is still pending when
// discovery runs). All work happens in the linked libraries' constructors; reaching main
// means no deadlock occurred.

#include <cstdio>
#include <cstdlib>

#include <omp.h>

extern "C" int
lib_get_num_places();

int
main()
{
    printf("reached main: no OMPT-init deadlock\n");

    int _omp_num_places = omp_get_num_places();
    int _lib_num_places = lib_get_num_places();
    printf("Number of places via OpenMP call:  %d\n", _omp_num_places);
    printf("Number of places via library call: %d\n", _lib_num_places);

    return (_omp_num_places == _lib_num_places) ? EXIT_SUCCESS : EXIT_FAILURE;
}
