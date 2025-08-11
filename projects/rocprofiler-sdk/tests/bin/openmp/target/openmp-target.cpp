// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <rocprofiler-sdk-roctx/roctx.h>

#include <math.h>
#include <stdio.h>

constexpr float  EPS_FLOAT  = 1.0e-7f;
constexpr double EPS_DOUBLE = 1.0e-15;

#pragma omp declare target
template <typename T>
T
mul(T a, T b)
{
    volatile T c = a * b;
    return c;
}
#pragma omp end declare target

template <typename T>
void
vmul(T* a, T* b, T* c, int N)
{
#pragma omp target           map(to : a [0:N], b [0:N]) map(from : c [0:N])
#pragma omp teams distribute parallel for
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < 100000; ++j)
            c[i] = mul(a[i], b[i]);
    }
}

int
main()
{
    auto range_id = roctxRangeStart("main");

    constexpr int N = 100000;
    int           a_i[N], b_i[N], c_i[N], validate_i[N];
    float         a_f[N], b_f[N], c_f[N], validate_f[N];
    double        a_d[N], b_d[N], c_d[N], validate_d[N];
    int           N_errors = 0;
    bool          flag     = false;

    roctxMark("initialization");

#pragma omp parallel for
    for(int i = 0; i < N; ++i)
    {
        a_f[i] = a_i[i] = i + 1;
        b_f[i] = b_i[i] = i + 2;
        a_d[i]          = a_i[i];
        b_d[i]          = b_i[i];
        validate_i[i]   = a_i[i] * b_i[i];
        validate_f[i]   = a_f[i] * b_f[i];
        validate_d[i]   = a_d[i] * b_d[i];
    }

    vmul(a_i, b_i, c_i, N);
    vmul(a_f, b_f, c_f, N);

    auto tid = roctx_thread_id_t{};
    // get the thread id recognized by rocprofiler-sdk from roctx
    roctxGetThreadId(&tid);
    // pause API tracing
    roctxProfilerPause(tid);

    // we don't expect to see the third vmul
    vmul(a_d, b_d, c_d, N);

    // resume API tracing
    roctxProfilerResume(tid);

    for(int i = 0; i < N; i++)
    {
        if(c_i[i] != validate_i[i])
        {
            ++N_errors;
            //       print 1st bad index
            if(!flag)
            {
                printf(
                    "First fail: c_i[%d](%d) != validate_i[%d](%d)\n", i, c_i[i], i, validate_i[i]);
                flag = true;
            }
        }
    }
    flag = false;
    for(int i = 0; i < N; i++)
    {
        if(fabs(c_f[i] - validate_f[i]) > EPS_FLOAT)
        {
            ++N_errors;
            //      print 1st bad index
            if(!flag)
            {
                printf("First fail: c_f[%d](%f) != validate_f[%d](%f)\n",
                       i,
                       static_cast<double>(c_f[i]),
                       i,
                       static_cast<double>(validate_f[i]));
                flag = true;
            }
        }
    }
    flag = false;
    for(int i = 0; i < N; i++)
    {
        if(fabs(c_d[i] - validate_d[i]) > EPS_DOUBLE)
        {
            ++N_errors;
            //      print 1st bad index
            if(!flag)
            {
                printf(
                    "First fail: c_d[%d](%f) != validate_d[%d](%f)\n", i, c_d[i], i, validate_d[i]);
                flag = true;
            }
        }
    }
    if(N_errors == 0)
    {
        printf("Success\n");
        return 0;
    }
    else
    {
        printf("Total %d failures\n", N_errors);
        printf("Fail\n");
        return 1;
    }

    roctxRangeStop(range_id);
}
