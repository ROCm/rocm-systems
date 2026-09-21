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

#include "lib/common/defines.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

using namespace rocprofiler;

namespace
{
#define PCS_VERSION(MAJOR, MINOR) ROCPROFILER_COMPUTE_VERSION(MAJOR, MINOR, 0)

rocprofiler_agent_t
make_agent(const char* name)
{
    rocprofiler_agent_t agent{};
    agent.name = name;
    return agent;
}
}  // namespace

TEST(pc_sampling_versioning, host_trap_version_thresholds)
{
    struct
    {
        const char* name;
        uint32_t    min_supported_version;
    } cases[] = {{"gfx90a", PCS_VERSION(0, 1)},
                 {"gfx940", PCS_VERSION(0, 3)},
                 {"gfx941", PCS_VERSION(0, 3)},
                 {"gfx942", PCS_VERSION(0, 3)},
                 {"gfx950", PCS_VERSION(1, 2)},
                 {"gfx1200", PCS_VERSION(1, 5)},
                 {"gfx1201", PCS_VERSION(1, 5)}};

    for(const auto& c : cases)
    {
        auto agent = make_agent(c.name);

        EXPECT_EQ(pc_sampling::ioctl::is_pc_sampling_method_supported(
                      ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP, &agent, c.min_supported_version),
                  ROCPROFILER_STATUS_SUCCESS)
            << c.name;

        EXPECT_EQ(
            pc_sampling::ioctl::is_pc_sampling_method_supported(
                ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP, &agent, c.min_supported_version - 1),
            ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL)
            << c.name;
    }
}

TEST(pc_sampling_versioning, host_trap_unsupported_architecture)
{
    auto agent = make_agent("gfx803");

    EXPECT_EQ(pc_sampling::ioctl::is_pc_sampling_method_supported(
                  ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP, &agent, PCS_VERSION(99, 0)),
              ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE);
}

TEST(pc_sampling_versioning, stochastic_version_thresholds)
{
    struct
    {
        const char* name;
        uint32_t    min_supported_version;
    } cases[] = {{"gfx940", PCS_VERSION(1, 3)},
                 {"gfx941", PCS_VERSION(1, 3)},
                 {"gfx942", PCS_VERSION(1, 3)},
                 {"gfx950", PCS_VERSION(1, 4)},
                 {"gfx1250", PCS_VERSION(1, 7)}};

    for(const auto& c : cases)
    {
        auto agent = make_agent(c.name);

        EXPECT_EQ(pc_sampling::ioctl::is_pc_sampling_method_supported(
                      ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC, &agent, c.min_supported_version),
                  ROCPROFILER_STATUS_SUCCESS)
            << c.name;

        EXPECT_EQ(
            pc_sampling::ioctl::is_pc_sampling_method_supported(
                ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC, &agent, c.min_supported_version - 1),
            ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL)
            << c.name;
    }
}

TEST(pc_sampling_versioning, stochastic_unsupported_on_gfx90a)
{
    auto agent = make_agent("gfx90a");

    // gfx90a never supports stochastic PC sampling, regardless of the IOCTL version.
    EXPECT_EQ(pc_sampling::ioctl::is_pc_sampling_method_supported(
                  ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC, &agent, PCS_VERSION(99, 0)),
              ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE);
}

// gfx1250 stochastic support is intentionally gated on an *exact* name match (unlike the
// prefix matching used for gfx94/gfx95/gfx12 families).
TEST(pc_sampling_versioning, stochastic_gfx1250_requires_exact_name_match)
{
    // "gfx1250a" shares the "gfx1250" prefix but is not an exact match; it must be rejected
    // even though a naive prefix check (agent_name.find("gfx1250") == 0) would accept it.
    auto agent = make_agent("gfx1250a");

    EXPECT_EQ(pc_sampling::ioctl::is_pc_sampling_method_supported(
                  ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC, &agent, PCS_VERSION(99, 0)),
              ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE);
}

TEST(pc_sampling_versioning, invalid_method_is_rejected)
{
    auto agent = make_agent("gfx1250");

    EXPECT_EQ(pc_sampling::ioctl::is_pc_sampling_method_supported(
                  ROCPROFILER_PC_SAMPLING_METHOD_NONE, &agent, PCS_VERSION(99, 0)),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

    EXPECT_EQ(pc_sampling::ioctl::is_pc_sampling_method_supported(
                  ROCPROFILER_PC_SAMPLING_METHOD_LAST, &agent, PCS_VERSION(99, 0)),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST(pc_sampling_versioning, kfd_method_kind_overload_maps_to_rocp_method)
{
    auto agent = make_agent("gfx950");

    EXPECT_EQ(pc_sampling::ioctl::is_pc_sampling_method_supported(
                  pc_sampling::ioctl::ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1,
                  &agent,
                  PCS_VERSION(1, 2)),
              ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(pc_sampling::ioctl::is_pc_sampling_method_supported(
                  pc_sampling::ioctl::ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1,
                  &agent,
                  PCS_VERSION(1, 4)),
              ROCPROFILER_STATUS_SUCCESS);

    // KFD kind NONE maps to ROCPROFILER_PC_SAMPLING_METHOD_NONE, which is not a valid method
    // to query support for.
    EXPECT_EQ(pc_sampling::ioctl::is_pc_sampling_method_supported(
                  pc_sampling::ioctl::ROCPROFILER_IOCTL_PC_SAMPLING_METHOD_KIND_NONE,
                  &agent,
                  PCS_VERSION(1, 4)),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
}
