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

// How a tool's replay_options choose where the snapshot lives. No GPU or HSA is involved.

#include "lib/rocprofiler-sdk/kernel_replay/replay_callbacks.hpp"

#include <rocprofiler-sdk/experimental/kernel_replay.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace kr   = rocprofiler::kernel_replay;
namespace msnp = rocprofiler::kernel_replay::memory_snapshot;

namespace
{
rocprofiler_kernel_replay_options_t
options(uint64_t flags)
{
    return rocprofiler_kernel_replay_options_t{sizeof(rocprofiler_kernel_replay_options_t), flags};
}

constexpr auto device_flag   = uint64_t{ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_DEVICE};
constexpr auto pinned_flag   = uint64_t{ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_HOST_PINNED};
constexpr auto pageable_flag = uint64_t{ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_HOST_PAGEABLE};
constexpr auto disk_flag     = uint64_t{ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_DISK};
}  // namespace

TEST(kernel_replay_options, no_options_keep_the_default_placement)
{
    EXPECT_EQ(kr::capture_mode_for(nullptr), msnp::capture_mode::automatic);

    const auto none = options(ROCPROFILER_KERNEL_REPLAY_OPTION_NONE);
    EXPECT_EQ(kr::capture_mode_for(&none), msnp::capture_mode::automatic);
}

TEST(kernel_replay_options, one_snapshot_flag_forces_that_memory)
{
    const auto device = options(device_flag);
    const auto pinned = options(pinned_flag);
    EXPECT_EQ(kr::capture_mode_for(&device), msnp::capture_mode::force_gpu);
    EXPECT_EQ(kr::capture_mode_for(&pinned), msnp::capture_mode::force_pinned);
}

TEST(kernel_replay_options, both_snapshot_flags_leave_the_choice_to_the_sdk)
{
    const auto both = options(device_flag | pinned_flag);
    EXPECT_EQ(kr::capture_mode_for(&both), msnp::capture_mode::automatic);
}

TEST(kernel_replay_options, honored_options_have_no_rejection)
{
    EXPECT_EQ(kr::options_rejection(nullptr), nullptr);
    for(auto flags : {uint64_t{ROCPROFILER_KERNEL_REPLAY_OPTION_NONE},
                      device_flag,
                      pinned_flag,
                      device_flag | pinned_flag})
    {
        const auto honored = options(flags);
        EXPECT_EQ(kr::options_rejection(&honored), nullptr) << "flags=" << flags;
    }
}

TEST(kernel_replay_options, an_unknown_flag_is_rejected)
{
    const auto unknown = options(uint64_t{1} << 4);
    const auto mixed   = options(pinned_flag | (uint64_t{1} << 40));
    EXPECT_FALSE(kr::capture_mode_for(&unknown).has_value());
    EXPECT_FALSE(kr::capture_mode_for(&mixed).has_value());
    EXPECT_NE(kr::options_rejection(&unknown), nullptr);
}

TEST(kernel_replay_options, pageable_and_disk_are_not_supported_yet)
{
    const auto unknown = options(uint64_t{1} << 4);
    for(auto flags :
        {pageable_flag, disk_flag, device_flag | disk_flag, pinned_flag | pageable_flag})
    {
        const auto unsupported = options(flags);
        EXPECT_FALSE(kr::capture_mode_for(&unsupported).has_value()) << "flags=" << flags;

        const auto* reason = kr::options_rejection(&unsupported);
        ASSERT_NE(reason, nullptr) << "flags=" << flags;
        EXPECT_NE(std::string{reason}, std::string{kr::options_rejection(&unknown)})
            << "flags=" << flags;
    }
}

TEST(kernel_replay_options, a_size_that_does_not_cover_flags_is_rejected)
{
    auto unsized = options(pinned_flag);
    unsized.size = 0;
    EXPECT_FALSE(kr::capture_mode_for(&unsized).has_value());

    unsized.size = sizeof(unsized.size);
    EXPECT_FALSE(kr::capture_mode_for(&unsized).has_value());
}

TEST(kernel_replay_options, a_larger_size_from_a_newer_tool_is_accepted)
{
    auto newer = options(pinned_flag);
    newer.size = sizeof(rocprofiler_kernel_replay_options_t) + sizeof(uint64_t);
    EXPECT_EQ(kr::capture_mode_for(&newer), msnp::capture_mode::force_pinned);
}
