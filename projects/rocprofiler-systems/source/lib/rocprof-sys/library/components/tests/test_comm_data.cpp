// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
