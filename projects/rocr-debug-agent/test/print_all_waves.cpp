/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip/hip_runtime.h>

__global__ void
print_all_waves_kern ()
{
  __syncthreads ();
  if (threadIdx.x == 0)
    {
      volatile int *k = nullptr;
      *k = 8;
    }
  /* Make sure that by the time the debug agent prints all the waves following
     the abort, other waves in the workgroup are still present.  */
  __syncthreads ();
}

void
PrintAllWavesTest ()
{
  hipError_t err;
  int deviceId;
  err = hipGetDevice (&deviceId);
  TEST_ASSERT (err == hipSuccess, "get device ID");

  hipDeviceProp_t props;
  err = hipGetDeviceProperties (&props, deviceId);
  TEST_ASSERT (err == hipSuccess, "get device properties");

  print_all_waves_kern<<<2, props.warpSize * 4>>> ();

  err = hipDeviceSynchronize ();
  TEST_ASSERT (err == hipSuccess, "kernel error");
}
