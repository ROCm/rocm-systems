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

#ifndef ROCSHMEM_BACKEND_BC_BUFFER_REGISTER_GTEST_HPP
#define ROCSHMEM_BACKEND_BC_BUFFER_REGISTER_GTEST_HPP

#include <mpi.h>

#include "gtest/gtest.h"

#include "../src/backend_bc.hpp"
#include "../src/mpi_instance.hpp"
#include "util_unit_test.hpp"

namespace rocshmem {

/**
 * @brief Minimal mock backend for testing buffer_register/buffer_unregister.
 */
class MockBackend : public Backend {
  public:
    MockBackend(MPI_Comm comm) : Backend(comm) {}

    void create_new_team(Team* parent_team,
                         const TeamInfo& team_info_wrt_parent,
                         const TeamInfo& team_info_wrt_world,
                         int num_pes, int my_pe_in_new_team,
                         MPI_Comm team_comm,
                         rocshmem_team_t* new_team) override {}

    void team_destroy(rocshmem_team_t team) override {}

    void global_exit(int status) override {}

    void ctx_create(int64_t options, void** ctx) override {}

    void ctx_destroy(Context* ctx) override {}

  protected:
    void dump_backend_stats() override {}
    void reset_backend_stats() override {}
};

class BufferRegisterTestFixture : public ::testing::Test {
  protected:
    MockBackend* backend_;

    void SetUp() override {
      set_gpu_ordinal();
      MPIInstance::mpilib_dl_init();
      backend_ = new MockBackend(MPI_COMM_WORLD);
    }

    void TearDown() override {
      delete backend_;
      MPIInstance::mpilib_dl_close();
    }
};

}  // namespace rocshmem

#endif  // ROCSHMEM_BACKEND_BC_BUFFER_REGISTER_GTEST_HPP
