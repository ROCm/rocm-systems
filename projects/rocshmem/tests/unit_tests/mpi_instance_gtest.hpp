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

#ifndef ROCSHMEM_MPI_INSTANCE_GTEST_HPP
#define ROCSHMEM_MPI_INSTANCE_GTEST_HPP

#include "gtest/gtest.h"

#include "../src/mpi_instance.hpp"

namespace rocshmem {

class MPIInstanceTestFixture : public ::testing::Test
{
  public:
    MPIInstanceTestFixture() {
        s_ptr_ = new MPIInstance(MPI_COMM_WORLD);
    }

    ~MPIInstanceTestFixture() {
        delete s_ptr_;
    }

  protected:
    /**
     * @brief A MPI instance object used to initialize MPI
     */
    MPIInstance* s_ptr_ {nullptr};
};

} // namespace rocshmem

#endif  // ROCSHMEM_MPI_INSTANCE_GTEST_HPP
