// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <string>

using namespace rocprofiler;

TEST(pc_sampling_kfd_unavailable, kfd_fd_matches_kfd_device_availability)
{
    EXPECT_EQ(pc_sampling::ioctl::get_kfd_fd() >= 0, agent::kfd_device_available());
}

// Exercised on hosts without /dev/kfd (e.g. WSL2/DXG): PC sampling must report
// NOT_AVAILABLE without aborting or issuing ioctls on an invalid descriptor.
TEST(pc_sampling_kfd_unavailable, pc_sampling_reports_not_available)
{
    if(agent::kfd_device_available()) GTEST_SKIP() << "/dev/kfd is available";

    rocprofiler_agent_t agent{};
    agent.name   = "gfx1100";
    agent.gpu_id = 1;

    ::testing::internal::CaptureStdout();

    EXPECT_EQ(pc_sampling::ioctl::get_kfd_fd(), -1);

    pc_sampling::ioctl::rocp_pcs_cfgs_vec_t configs{};
    EXPECT_EQ(pc_sampling::ioctl::ioctl_query_pcs_configs(&agent, configs),
              ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE);
    EXPECT_TRUE(configs.empty());

    uint32_t ioctl_pcs_id = 0;
    EXPECT_EQ(pc_sampling::ioctl::ioctl_pcs_create(&agent,
                                                   ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                                   ROCPROFILER_PC_SAMPLING_UNIT_TIME,
                                                   1000,
                                                   &ioctl_pcs_id),
              ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE);

    EXPECT_EQ(::testing::internal::GetCapturedStdout(), std::string{});
}
