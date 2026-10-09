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
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"
#include "lib/rocprofiler-sdk/platform/wsl/agent.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <cerrno>
#include <string>

using namespace rocprofiler;

// An unavailable KFD descriptor must be rejected before ::ioctl is reached, so the wrapper's
// "Invalid KFD descriptor" message for a real EBADF is not printed.
TEST(pc_sampling_kfd_unavailable, ioctl_rejects_invalid_fd)
{
    kfd_ioctl_get_version_args args = {};

    ::testing::internal::CaptureStdout();
    errno    = 0;
    auto ret = pc_sampling::ioctl::ioctl(-1, AMDKFD_IOC_GET_VERSION, &args);
    auto err = errno;

    EXPECT_EQ(ret, -EBADF);
    EXPECT_EQ(err, EBADF);
    EXPECT_EQ(::testing::internal::GetCapturedStdout(), std::string{});
}

// Outside WSL2/DXG a /dev/kfd that cannot be opened is reported through ROCP_CI_LOG, which is
// fatal in CI builds, so only WSL2/DXG and hosts with a usable /dev/kfd are checked here.
TEST(pc_sampling_kfd_unavailable, kfd_fd_matches_kfd_device_availability)
{
    if(!platform::wsl::is_available() && !agent::kfd_device_available())
        GTEST_SKIP() << "/dev/kfd cannot be opened and this is not WSL2/DXG";

    EXPECT_EQ(pc_sampling::ioctl::get_kfd_fd() >= 0, agent::kfd_device_available());
}

// WSL2/DXG does not expose KFD: PC sampling must report NOT_AVAILABLE without aborting or
// issuing ioctls on an invalid descriptor.
TEST(pc_sampling_kfd_unavailable, pc_sampling_reports_not_available)
{
    if(!platform::wsl::is_available()) GTEST_SKIP() << "not running under WSL2/DXG";

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
