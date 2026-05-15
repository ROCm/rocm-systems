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

#ifndef ROCSHMEM_REMOTE_HEAP_INFO_GTEST_HPP
#define ROCSHMEM_REMOTE_HEAP_INFO_GTEST_HPP

#include "gtest/gtest.h"

#include <mpi.h>

#include "../src/memory/heap_memory.hpp"
#include "../src/memory/hip_allocator.hpp"
#include "../src/memory/remote_heap_info.hpp"

namespace rocshmem {

class RemoteHeapInfoTestFixture : public ::testing::Test
{
    /**
     * @brief Helper type for heap memory
     */
    using HEAP_T = HeapMemory<HIPAllocator>;

    /**
     * @brief Helper type for RemoteHeapInfo with MPI
     */
    using MPI_T = RemoteHeapInfo<CommunicatorMPI>;

  protected:
    /**
     * @brief Heap memory object
     */
    HEAP_T heap_mem_ {};

    /**
     * @brief Remote heap info with MPI Communicator
     */
    MPI_T mpi_ {heap_mem_.get_ptr(),
                heap_mem_.get_size(),
                MPI_COMM_WORLD};
};

} // namespace rocshmem

#endif // ROCSHMEM_REMOTE_HEAP_INFO_GTEST_HPP
