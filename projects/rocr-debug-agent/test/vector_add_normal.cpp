/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimers in
      the documentation and/or other materials provided with the distribution.
    - Neither the names of Advanced Micro Devices, Inc,
      nor the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior written
      permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS WITH THE SOFTWARE.  */

#include "util.h"

#include <cstdlib>
#include <string.h>
#include <string>

#include <hip/hip_runtime.h>

#define M_ORDER 16
#define M_GET(M, I, J) M[I * M_ORDER + J]
#define M_SET(M, I, J, V) M[I * M_ORDER + J] = V

__global__ void
vector_add_normal (int *a, int *b, int *c)
{
  int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  c[gid] = a[gid] + b[gid];
}

void
VectorAddNormalTest ()
{
  int *M_IN0 = nullptr;
  int *M_IN1 = nullptr;
  int *M_RESULT_DEVICE = nullptr;
  int M_RESULT_HOST[M_ORDER * M_ORDER];
  hipError_t err;

  // allocate input and output kernel arguments
  err = hipMalloc (&M_IN0, M_ORDER * M_ORDER * sizeof (int));
  TEST_ASSERT (err == hipSuccess, "hipMalloc");

  err = hipMalloc (&M_IN1, M_ORDER * M_ORDER * sizeof (int));
  TEST_ASSERT (err == hipSuccess, "hipMalloc");

  err = hipMalloc (&M_RESULT_DEVICE, M_ORDER * M_ORDER * sizeof (int));
  TEST_ASSERT (err == hipSuccess, "hipMalloc");

  memset (M_RESULT_HOST, 0, M_ORDER * M_ORDER * sizeof (int));
  err = hipMemset (M_RESULT_DEVICE, 0, M_ORDER * M_ORDER * sizeof (int));
  TEST_ASSERT (err == hipSuccess, "hipMemset");

  int *M_IN0_HOST = (int *)malloc (M_ORDER * M_ORDER * sizeof (int));
  int *M_IN1_HOST = (int *)malloc (M_ORDER * M_ORDER * sizeof (int));

  // initialize input and run on host
  srand (time (nullptr));
  for (int i = 0; i < M_ORDER; ++i)
    {
      for (int j = 0; j < M_ORDER; ++j)
        {
          M_SET (M_IN0_HOST, i, j, (1 + rand () % 10));
          M_SET (M_IN1_HOST, i, j, (1 + rand () % 10));
        }
    }

  for (int i = 0; i < M_ORDER; ++i)
    {
      for (int j = 0; j < M_ORDER; ++j)
        {
          int s = M_GET (M_IN0_HOST, i, j) + M_GET (M_IN1_HOST, i, j);
          M_SET (M_RESULT_HOST, i, j, s);
        }
    }

  err = hipMemcpy (M_IN0, M_IN0_HOST, M_ORDER * M_ORDER * sizeof (int),
                   hipMemcpyHostToDevice);
  TEST_ASSERT (err == hipSuccess, "hipMemcpy");

  err = hipMemcpy (M_IN1, M_IN1_HOST, M_ORDER * M_ORDER * sizeof (int),
                   hipMemcpyHostToDevice);
  TEST_ASSERT (err == hipSuccess, "hipMemcpy");

  const unsigned blocks = M_ORDER * M_ORDER / 64;
  const unsigned threadsPerBlock = 64;
  hipLaunchKernelGGL (vector_add_normal, dim3 (blocks), dim3 (threadsPerBlock),
                      0, 0, M_IN0, M_IN1, M_RESULT_DEVICE);
  err = hipDeviceSynchronize ();
  TEST_ASSERT (err == hipSuccess, "hipDeviceSynchronize");

  err = hipFree (M_IN0);
  TEST_ASSERT (err == hipSuccess, "hipFree");
  err = hipFree (M_IN1);
  TEST_ASSERT (err == hipSuccess, "hipFree");
  err = hipFree (M_RESULT_DEVICE);
  TEST_ASSERT (err == hipSuccess, "hipFree");
  free (M_IN0_HOST);
  free (M_IN1_HOST);
}
