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

// Unit tests for building the packet a kernel replay pass submits when a tool replaces its
// dispatch (replay_set_dispatch). Pure packet manipulation: no GPU, no HSA runtime.

#include "lib/rocprofiler-sdk/kernel_replay/dispatch_override.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/replay_callbacks.hpp"

#include <rocprofiler-sdk/experimental/kernel_replay.h>

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>

namespace kr  = rocprofiler::kernel_replay;
namespace hsa = rocprofiler::hsa;

namespace
{
constexpr uint16_t
dispatch_header(uint16_t type)
{
    return static_cast<uint16_t>(
        (type << HSA_PACKET_HEADER_TYPE) | (1u << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
}

int app_kernarg  = 0;
int tool_kernarg = 0;

hsa::rocprofiler_packet
plain_app_packet()
{
    auto pkt                 = hsa_kernel_dispatch_packet_t{};
    pkt.header               = dispatch_header(HSA_PACKET_TYPE_KERNEL_DISPATCH);
    pkt.setup                = 1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    pkt.workgroup_size_x     = 256;
    pkt.workgroup_size_y     = 1;
    pkt.workgroup_size_z     = 1;
    pkt.grid_size_x          = 256 * 64;
    pkt.grid_size_y          = 1;
    pkt.grid_size_z          = 1;
    pkt.private_segment_size = 16;
    pkt.group_segment_size   = 1024;
    pkt.kernel_object        = 0x1000;
    pkt.kernarg_address      = &app_kernarg;
    pkt.completion_signal    = hsa_signal_t{.handle = 0x77};
    return hsa::rocprofiler_packet{pkt};
}

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
hsa::rocprofiler_packet
cluster_app_packet()
{
    auto pkt                 = hsa_amd_ext_kernel_dispatch_packet_t{};
    pkt.header               = dispatch_header(HSA_PACKET_TYPE_VENDOR_SPECIFIC);
    pkt.amd_format           = HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
    pkt.setup                = 2u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    pkt.workgroup_size_x     = 64;
    pkt.workgroup_size_y     = 2;
    pkt.workgroup_size_z     = 1;
    pkt.cluster_count_x      = 8;
    pkt.cluster_count_y      = 4;
    pkt.cluster_count_z      = 1;
    pkt.cluster_size_x       = 2;
    pkt.cluster_size_y       = 2;
    pkt.cluster_size_z       = 1;
    pkt.private_segment_size = 0;
    pkt.group_segment_size   = 512;
    pkt.kernel_object        = 0x2000;
    pkt.kernarg_address      = &app_kernarg;
    pkt.dep_signal           = hsa_signal_t{.handle = 0x55};
    pkt.completion_signal    = hsa_signal_t{.handle = 0x77};
    return hsa::rocprofiler_packet{pkt};
}
#endif

rocprofiler_kernel_replay_dispatch_t
described(const hsa::rocprofiler_packet& pkt)
{
    auto dispatch = kr::describe_dispatch(pkt);
    EXPECT_TRUE(dispatch.has_value());
    return dispatch.value_or(rocprofiler_kernel_replay_dispatch_t{});
}

uint16_t
dimensions(uint16_t setup)
{
    return (setup >> HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS) &
           ((1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS) - 1);
}
}  // namespace

TEST(kernel_replay_dispatch_override, describes_a_plain_dispatch)
{
    const auto dispatch = described(plain_app_packet());
    EXPECT_EQ(dispatch.size, sizeof(rocprofiler_kernel_replay_dispatch_t));
    EXPECT_EQ(dispatch.kernel_object, 0x1000u);
    EXPECT_EQ(dispatch.kernarg_address, &app_kernarg);
    EXPECT_EQ(dispatch.workgroup_size.x, 256u);
    EXPECT_EQ(dispatch.grid_size.x, 256u * 64);
    EXPECT_EQ(dispatch.cluster_size.x, 0u);
    EXPECT_EQ(dispatch.group_segment_size, 1024u);
    EXPECT_EQ(dispatch.private_segment_size, 16u);
}

TEST(kernel_replay_dispatch_override, non_dispatch_packets_have_no_description)
{
    auto barrier   = hsa_barrier_and_packet_t{};
    barrier.header = dispatch_header(HSA_PACKET_TYPE_BARRIER_AND);
    EXPECT_FALSE(kr::describe_dispatch(hsa::rocprofiler_packet{barrier}).has_value());
}

// The replacement changes only the dispatch's own fields: the header (fences, barrier bit) and
// completion signal of the application's packet survive.
TEST(kernel_replay_dispatch_override, plain_replacement_keeps_the_app_header)
{
    const auto app              = plain_app_packet();
    auto       dispatch         = described(app);
    dispatch.kernel_object      = 0x3000;
    dispatch.kernarg_address    = &tool_kernarg;
    dispatch.grid_size          = {256 * 128, 1, 1};
    dispatch.group_segment_size = 2048;

    auto out = hsa::rocprofiler_packet{};
    ASSERT_EQ(kr::build_dispatch_packet(app, dispatch, out), ROCPROFILER_STATUS_SUCCESS);
    const auto& s = out.kernel_dispatch;
    EXPECT_EQ(s.header, app.kernel_dispatch.header);
    EXPECT_EQ(s.completion_signal.handle, 0x77u);
    EXPECT_EQ(s.kernel_object, 0x3000u);
    EXPECT_EQ(s.kernarg_address, &tool_kernarg);
    EXPECT_EQ(s.grid_size_x, 256u * 128);
    EXPECT_EQ(s.workgroup_size_x, 256);
    EXPECT_EQ(s.group_segment_size, 2048u);
    EXPECT_EQ(s.private_segment_size, 16u);
    EXPECT_EQ(dimensions(s.setup), 1);
}

// The dimension count grows to fit a taller grid but never drops below what the application
// declared.
TEST(kernel_replay_dispatch_override, setup_dimensions_fit_the_grid)
{
    const auto app      = plain_app_packet();
    auto       dispatch = described(app);
    dispatch.grid_size  = {256, 4, 2};

    auto out = hsa::rocprofiler_packet{};
    ASSERT_EQ(kr::build_dispatch_packet(app, dispatch, out), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(dimensions(out.kernel_dispatch.setup), 3);

    auto three_d                  = app;
    three_d.kernel_dispatch.setup = 3u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    dispatch.grid_size            = {256, 1, 1};
    ASSERT_EQ(kr::build_dispatch_packet(three_d, dispatch, out), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(dimensions(out.kernel_dispatch.setup), 3);
}

TEST(kernel_replay_dispatch_override, rejects_dispatches_the_packet_cannot_hold)
{
    const auto app                    = plain_app_packet();
    const auto base                   = described(app);
    auto       out                    = hsa::rocprofiler_packet{};
    out.kernel_dispatch.kernel_object = 0xdead;

    auto expect_rejected = [&](rocprofiler_kernel_replay_dispatch_t dispatch, const char* why) {
        EXPECT_EQ(kr::build_dispatch_packet(app, dispatch, out),
                  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT)
            << why;
        EXPECT_EQ(out.kernel_dispatch.kernel_object, 0xdeadu) << why << ": out must be untouched";
    };

    auto d = base;
    d.size = 8;
    expect_rejected(d, "size does not cover the struct");
    d               = base;
    d.kernel_object = 0;
    expect_rejected(d, "no kernel");
    d                  = base;
    d.workgroup_size.y = 0;
    expect_rejected(d, "zero workgroup dimension");
    d             = base;
    d.grid_size.z = 0;
    expect_rejected(d, "zero grid dimension");
    d                  = base;
    d.workgroup_size.x = 70000;
    expect_rejected(d, "workgroup wider than the packet field");
    d              = base;
    d.cluster_size = {2, 1, 1};
    expect_rejected(d, "clusters on a plain dispatch");

    auto barrier   = hsa_barrier_and_packet_t{};
    barrier.header = dispatch_header(HSA_PACKET_TYPE_BARRIER_AND);
    EXPECT_EQ(kr::build_dispatch_packet(hsa::rocprofiler_packet{barrier}, base, out),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
}

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
TEST(kernel_replay_dispatch_override, describes_a_cluster_dispatch)
{
    const auto dispatch = described(cluster_app_packet());
    EXPECT_EQ(dispatch.workgroup_size.y, 2u);
    EXPECT_EQ(dispatch.cluster_size.x, 2u);
    EXPECT_EQ(dispatch.cluster_size.y, 2u);
    EXPECT_EQ(dispatch.cluster_size.z, 1u);
    EXPECT_EQ(dispatch.grid_size.x, 8u * 2 * 64);
    EXPECT_EQ(dispatch.grid_size.y, 4u * 2 * 2);
    EXPECT_EQ(dispatch.grid_size.z, 1u);
}

// A cluster dispatch stays one: cluster counts are recomputed from the new grid, and the extended
// packet's own fields (format, dependency signal) are kept.
TEST(kernel_replay_dispatch_override, cluster_replacement_recomputes_cluster_counts)
{
    const auto app           = cluster_app_packet();
    auto       dispatch      = described(app);
    dispatch.kernel_object   = 0x4000;
    dispatch.kernarg_address = &tool_kernarg;
    dispatch.grid_size       = {16 * 2 * 64, 4 * 2 * 2, 1};

    auto out = hsa::rocprofiler_packet{};
    ASSERT_EQ(kr::build_dispatch_packet(app, dispatch, out), ROCPROFILER_STATUS_SUCCESS);
    const auto& e = out.ext_kernel_dispatch;
    EXPECT_EQ(e.header, app.ext_kernel_dispatch.header);
    EXPECT_EQ(e.amd_format, HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH);
    EXPECT_EQ(e.dep_signal.handle, 0x55u);
    EXPECT_EQ(e.cluster_count_x, 16u);
    EXPECT_EQ(e.cluster_count_y, 4);
    EXPECT_EQ(e.cluster_count_z, 1);
    EXPECT_EQ(e.cluster_size_x, 2);
    EXPECT_EQ(e.kernel_object, 0x4000u);
    EXPECT_EQ(e.kernarg_address, &tool_kernarg);
    EXPECT_EQ(dimensions(e.setup), 2);
}

TEST(kernel_replay_dispatch_override, rejects_cluster_dispatches_the_packet_cannot_hold)
{
    const auto app  = cluster_app_packet();
    const auto base = described(app);
    auto       out  = hsa::rocprofiler_packet{};

    auto d         = base;
    d.cluster_size = {0, 0, 0};
    EXPECT_EQ(kr::build_dispatch_packet(app, d, out), ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT)
        << "a cluster dispatch needs a cluster size";
    d             = base;
    d.grid_size.x = 2 * 64 * 8 + 64;
    EXPECT_EQ(kr::build_dispatch_packet(app, d, out), ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT)
        << "the grid must be a whole number of clusters";
    d                = base;
    d.cluster_size.x = 300;
    d.grid_size.x    = 300 * 64;
    EXPECT_EQ(kr::build_dispatch_packet(app, d, out), ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT)
        << "cluster size wider than the packet field";
    d             = base;
    d.grid_size.y = 70000 * 2 * 2;
    EXPECT_EQ(kr::build_dispatch_packet(app, d, out), ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT)
        << "too many clusters in y for the packet field";
}
#endif

// The setter only works while a PASS PHASE_ENTER callback runs.
TEST(kernel_replay_dispatch_override, setter_outside_pass_enter_is_rejected)
{
    auto dispatch = described(plain_app_packet());
    EXPECT_EQ(kr::replay_set_dispatch(&dispatch), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
    EXPECT_EQ(kr::replay_set_dispatch(nullptr), ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
}
