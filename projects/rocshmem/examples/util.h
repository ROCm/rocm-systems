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

#ifndef __ROCSHMEM_EXAMPLES_UTIL_H__
#define __ROCSHMEM_EXAMPLES_UTIL_H__

#include <iostream>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

#define CHECK_HIP(condition) {                                            \
    hipError_t error = condition;                                         \
    if(error != hipSuccess){                                              \
        fprintf(stderr, "HIP error: %d line: %s:%d\n",                    \
                error, __FILE__,  __LINE__);                              \
        exit(error);                                                      \
    }                                                                     \
}

#define ASSERT(condition) {                                               \
  if (!(condition)) {                                                     \
    fprintf(stderr, "Assertion failed: [%s:%d], condition:  %s\n",        \
            __FILE__, __LINE__, #condition);                              \
    exit(EXIT_FAILURE);                                                   \
  }                                                                       \
}

#define DEVICE_ASSERT(condition) {                                        \
  if (!(condition)) {                                                     \
    printf("Assertion failed: [%s:%d], condition:  %s\n",                 \
            __FILE__, __LINE__, #condition);                              \
    abort();                                                              \
  }                                                                       \
}

void comm_init() {

  // Initialize MPI
  int mpi_rank {0}, mpi_size {0};
  int ret {0};
  int provided {0};
  MPI_Init_thread (nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cerr << "MPI_THREAD_MULTIPLE support disabled.\n";
  }
  MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);

  // Set GPU id
  int device_count {0};
  CHECK_HIP(hipGetDeviceCount(&device_count));
  CHECK_HIP(hipSetDevice(mpi_rank % device_count));

  // Initialize rocSHMEM with unique ID
  rocshmem_uniqueid_t uid;
  rocshmem_init_attr_t attr;
  if (mpi_rank == 0) {
    ret = rocshmem_get_uniqueid (&uid);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << mpi_rank
      << ": Error in rocshmem_get_uniqueid. Aborting." << std::endl;
      MPI_Abort (MPI_COMM_WORLD, ret);
    }
  }

  // Broadcast the unique ID to all ranks
  MPI_Bcast (&uid, sizeof(rocshmem_uniqueid_t), MPI_BYTE, 0, MPI_COMM_WORLD);
  ret = rocshmem_set_attr_uniqueid_args(mpi_rank, mpi_size, &uid, &attr);
  if (ret != ROCSHMEM_SUCCESS) {
    std::cout << mpi_rank
              << ": Error in rocshmem_set_attr_uniqueid_args. Aborting"
              << std::endl;
    MPI_Abort (MPI_COMM_WORLD, ret);
  }

  ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
  if (ret != ROCSHMEM_SUCCESS) {
    std::cout << mpi_rank << ": Error in rocshmem_init_attr. Aborting."
              << std::endl;
    MPI_Abort (MPI_COMM_WORLD, ret);
  }
}

void comm_finalize() {
  rocshmem_finalize();
  MPI_Finalize();
}


#endif /* __ROCSHMEM_EXAMPLES_UTIL_H__ */
