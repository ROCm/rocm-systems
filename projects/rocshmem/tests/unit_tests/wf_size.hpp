/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef ROCSHMEM_WF_SIZE_HPP
#define ROCSHMEM_WF_SIZE_HPP

#include <hip/hip_runtime.h>
#include "mpi.h"

#define CHECK_HIP_MPI(cond) {                                      \
  if(cond != hipSuccess){                                          \
     fprintf(stderr,"HIP error: %d line: %d\n", cond,  __LINE__);  \
     MPI_Abort(MPI_COMM_WORLD, 1);  	     	  	 	   \
  }                                                                \
}

static int get_wf_size() {
  int deviceId;
  hipDeviceProp_t prop;
  CHECK_HIP_MPI(hipGetDevice(&deviceId));
  CHECK_HIP_MPI(hipGetDeviceProperties(&prop, deviceId));
  return prop.warpSize;
}

#endif
