// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/guarded_read.hpp"
#include "core/collection/tests/log_capture.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <exception>
#include <stdexcept>

namespace rocprofsys::collection
{
namespace
{
using testing::log_capture;

constexpr std::size_t k_device = 3;
constexpr int         k_thrown = 7;
}  // namespace

TEST(guarded_read, successful_read_runs_and_keeps_the_device)
{
    bool failed = false;
    int  reads  = 0;

    EXPECT_TRUE(guarded_read(failed, "cpu", k_device, [&reads] { ++reads; }));

    EXPECT_EQ(reads, 1);
    EXPECT_FALSE(failed);
}

TEST(guarded_read, std_exception_disables_the_device_and_is_logged_once)
{
    const log_capture logs;
    bool              failed = false;
    int               reads  = 0;
    const auto        read   = [&reads] {
        ++reads;
        throw std::runtime_error("device lost");
    };

    EXPECT_FALSE(guarded_read(failed, "gpu_smi", k_device, read));
    EXPECT_FALSE(guarded_read(failed, "gpu_smi", k_device, read));

    EXPECT_TRUE(failed);
    EXPECT_EQ(reads, 1);
    EXPECT_EQ(logs.count("error", "gpu_smi device 3 disabled: device lost"), 1U);
}

TEST(guarded_read, non_std_exception_disables_the_device)
{
    const log_capture logs;
    bool              failed = false;

    EXPECT_FALSE(guarded_read(failed, "nic_smi", k_device, [] {
        std::rethrow_exception(std::make_exception_ptr(k_thrown));
    }));

    EXPECT_TRUE(failed);
    EXPECT_EQ(logs.count("error", "nic_smi device 3 disabled: unknown exception"), 1U);
}

TEST(guarded_read, failed_device_is_not_read)
{
    bool failed = true;
    int  reads  = 0;

    EXPECT_FALSE(guarded_read(failed, "hipfile", k_device, [&reads] { ++reads; }));

    EXPECT_EQ(reads, 0);
}
}  // namespace rocprofsys::collection
