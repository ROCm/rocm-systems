// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

// Tests for comm_data in the context of SHMEM tracing. The SHMEM gotcha uses
// comm_data::start() as part of its lifecycle; it does not call comm_data::audit()
// for individual OpenSHMEM operations. This file exercises the comm_data
// lifecycle and ensures it is safe for use by the SHMEM implementation.

#include "core/state.hpp"
#include "rocprof-sys/library/components/comm_data.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>

namespace rocprofsys
{
namespace component
{
namespace testing
{

class comm_data_test : public ::testing::Test
{
protected:
    void SetUp() override { test_gotcha_data.tool_id = "test_shmem_comm"; }

    void TearDown() override {}

    tim::component::gotcha_data test_gotcha_data;
};

TEST_F(comm_data_test, component_lifecycle)
{
    // SHMEM gotcha calls comm_data::start() in its start(); lifecycle must not throw.
    EXPECT_NO_THROW(comm_data::preinit());
    EXPECT_NO_THROW(comm_data::configure());
    EXPECT_NO_THROW(comm_data::start());
    EXPECT_NO_THROW(comm_data::stop());
    EXPECT_NO_THROW(comm_data::global_finalize());
}

TEST_F(comm_data_test, start_safe_for_shmem)
{
    // After preinit/configure, start() is what shmem_gotcha uses; must be callable.
    EXPECT_NO_THROW(comm_data::preinit());
    EXPECT_NO_THROW(comm_data::configure());
    EXPECT_NO_THROW(comm_data::start());
    EXPECT_NO_THROW(comm_data::stop());
}

TEST_F(comm_data_test, comm_data_value_label)
{
    // comm_data is shared; ensure basic value/label exist for tooling.
    EXPECT_STREQ(comm_data::mpi_send::value, "comm_data");
    EXPECT_STREQ(comm_data::mpi_recv::value, "comm_data");
}

}  // namespace testing
}  // namespace component
}  // namespace rocprofsys
