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

#include "lib/rocprofiler-sdk/kernel_replay/restore_policy.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace
{
namespace policy = rocprofiler::kernel_replay::restore_policy;
using policy::restore_mode;

constexpr auto mib = size_t{1} << 20;

policy::inputs_t
inputs(uint64_t          passes,
       size_t            gpu_bytes,
       size_t            pinned_bytes = 0,
       policy::history_t history      = {},
       restore_mode      requested    = restore_mode::automatic)
{
    auto value         = policy::inputs_t{};
    value.requested    = requested;
    value.total_passes = passes;
    value.history      = history;
    value.gpu_bytes    = gpu_bytes;
    value.pinned_bytes = pinned_bytes;
    return value;
}

TEST(kernel_replay_restore_policy, parses_the_environment_values)
{
    EXPECT_EQ(policy::parse_mode(""), restore_mode::automatic);
    EXPECT_EQ(policy::parse_mode("auto"), restore_mode::automatic);
    EXPECT_EQ(policy::parse_mode("full"), restore_mode::full);
    EXPECT_EQ(policy::parse_mode("hash-every-pass"), restore_mode::hash_every_pass);
    EXPECT_EQ(policy::parse_mode("hash-once"), restore_mode::hash_once);
    EXPECT_EQ(policy::parse_mode("hash"), std::nullopt);
    for(const auto mode : {restore_mode::automatic,
                           restore_mode::full,
                           restore_mode::hash_every_pass,
                           restore_mode::hash_once})
        EXPECT_EQ(policy::parse_mode(policy::name_of(mode)), mode);
}

TEST(kernel_replay_restore_policy, small_device_snapshots_restore_everything)
{
    EXPECT_EQ(policy::choose(inputs(4, 64 * mib - 1)), restore_mode::full);
    EXPECT_EQ(policy::choose(inputs(4, 64 * mib)), restore_mode::hash_once);
}

TEST(kernel_replay_restore_policy, pinned_memory_is_hashed_from_a_mebibyte)
{
    EXPECT_EQ(policy::choose(inputs(4, 0, mib - 1)), restore_mode::full);
    EXPECT_EQ(policy::choose(inputs(4, 0, mib)), restore_mode::hash_once);
}

TEST(kernel_replay_restore_policy, two_or_three_passes_hash_before_each)
{
    EXPECT_EQ(policy::choose(inputs(2, 64 * mib)), restore_mode::hash_every_pass);
    EXPECT_EQ(policy::choose(inputs(3, 64 * mib)), restore_mode::hash_every_pass);
    EXPECT_EQ(policy::choose(inputs(4, 64 * mib)), restore_mode::hash_once);
    // A tool that decides as the replay goes may run any number of passes.
    EXPECT_EQ(policy::choose(inputs(0, 64 * mib)), restore_mode::hash_once);
}

TEST(kernel_replay_restore_policy, one_pass_never_hashes)
{
    EXPECT_EQ(policy::choose(inputs(1, 1024 * mib, 0, {}, restore_mode::hash_once)),
              restore_mode::full);
    EXPECT_FALSE(policy::may_hash(inputs(1, 0)));
}

TEST(kernel_replay_restore_policy, history_turns_hashing_off_once_full_restores_move_less)
{
    // Hashing before every pass moves less until half the tiles change.
    EXPECT_EQ(policy::choose(inputs(3, 1024 * mib, 0, policy::history_t{100, 49})),
              restore_mode::hash_every_pass);
    EXPECT_EQ(policy::choose(inputs(3, 1024 * mib, 0, policy::history_t{100, 50})),
              restore_mode::full);
    // Hashing once moves less until 1 - 1/(2N - 3) of them change: 4/5 with 4 passes, 6/7 with 5.
    EXPECT_EQ(policy::choose(inputs(4, 1024 * mib, 0, policy::history_t{100, 79})),
              restore_mode::hash_once);
    EXPECT_EQ(policy::choose(inputs(4, 1024 * mib, 0, policy::history_t{100, 80})),
              restore_mode::full);
    EXPECT_EQ(policy::choose(inputs(5, 1024 * mib, 0, policy::history_t{7, 5})),
              restore_mode::hash_once);
    EXPECT_EQ(policy::choose(inputs(5, 1024 * mib, 0, policy::history_t{7, 6})),
              restore_mode::full);
    // Without a pass count, hashing once never moves more.
    EXPECT_EQ(policy::choose(inputs(0, 1024 * mib, 0, policy::history_t{100, 100})),
              restore_mode::hash_once);
    // Pinned memory is hashed whatever changed.
    EXPECT_EQ(policy::choose(inputs(4, 1024 * mib, 4 * mib, policy::history_t{100, 100})),
              restore_mode::hash_once);

    const auto unlisted = policy::history_t{100, 1, true};
    EXPECT_EQ(policy::choose(inputs(4, 1024 * mib, 4 * mib, unlisted)), restore_mode::full);
    EXPECT_FALSE(policy::may_hash(inputs(4, 0, 0, unlisted)));
}

TEST(kernel_replay_restore_policy, requests_override_size_and_history)
{
    const auto history = policy::history_t{100, 100, true};
    EXPECT_EQ(policy::choose(inputs(4, 16, 0, history, restore_mode::hash_once)),
              restore_mode::hash_once);
    EXPECT_EQ(policy::choose(inputs(4, 16, 0, history, restore_mode::hash_every_pass)),
              restore_mode::hash_every_pass);
    EXPECT_EQ(policy::choose(inputs(4, 1024 * mib, 1024 * mib, {}, restore_mode::full)),
              restore_mode::full);
    EXPECT_FALSE(policy::may_hash(inputs(4, 0, 0, {}, restore_mode::full)));
}

TEST(kernel_replay_restore_policy, captures_keep_digests_before_the_size_is_known)
{
    EXPECT_TRUE(policy::may_hash(inputs(4, 0)));
    EXPECT_TRUE(policy::may_hash(inputs(2, 0, 0, policy::history_t{100, 100})));
}

TEST(kernel_replay_restore_policy, records_what_each_kernel_changed)
{
    const auto kernel = policy::kernel_key_t{1, 0x1000};
    const auto other  = policy::kernel_key_t{2, 0x1000};
    EXPECT_EQ(policy::history_of(kernel).tiles, 0U);

    policy::record(kernel, 100, 50, 0);
    EXPECT_EQ(policy::history_of(kernel).tiles, 100U);
    EXPECT_EQ(policy::history_of(kernel).changed, 50U);
    EXPECT_EQ(policy::history_of(other).tiles, 0U);
    policy::record(kernel, 100, 49, 0);
    EXPECT_EQ(policy::history_of(kernel).changed, 49U);

    // Once tiles changed off a reused list, that kernel keeps restoring everything.
    policy::record(kernel, 100, 1, 2);
    EXPECT_TRUE(policy::history_of(kernel).unlisted);
    policy::record(kernel, 100, 1, 0);
    EXPECT_TRUE(policy::history_of(kernel).unlisted);
}
}  // namespace
