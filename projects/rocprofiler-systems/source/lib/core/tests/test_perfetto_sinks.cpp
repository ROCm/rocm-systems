// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output_file_registry.hpp"
#include "core/perfetto/sinks.hpp"

#include <vector>

TEST(recording_sink, default_state_is_empty_and_unfinalized)
{
    rocprofsys::core::recording_sink sink;
    EXPECT_TRUE(sink.records().empty());
    EXPECT_FALSE(sink.finalized());
}

TEST(recording_sink, on_source_drained_captures_records_in_order)
{
    rocprofsys::core::recording_sink sink;

    sink.on_source_drained(11, std::vector<char>{ 'a', 'b', 'c' });
    sink.on_source_drained(22, std::vector<char>{ 'x', 'y' });

    ASSERT_EQ(sink.records().size(), 2u);
    EXPECT_EQ(sink.records()[0].first, 11);
    EXPECT_EQ(sink.records()[0].second,
              (std::vector<char>{ 'a', 'b', 'c' }));
    EXPECT_EQ(sink.records()[1].first, 22);
    EXPECT_EQ(sink.records()[1].second, (std::vector<char>{ 'x', 'y' }));

    EXPECT_FALSE(sink.finalized()) << "finalize must not auto-fire on drain";
}

TEST(recording_sink, finalize_sets_flag)
{
    rocprofsys::core::recording_sink sink;
    sink.on_source_drained(0, std::vector<char>{ 'z' });
    EXPECT_FALSE(sink.finalized());

    sink.finalize();
    EXPECT_TRUE(sink.finalized());
}

TEST(recording_sink, finalize_without_drain_is_safe)
{
    // Drain contract allows finalize-after-zero-sources (e.g. empty pid set).
    rocprofsys::core::recording_sink sink;
    EXPECT_NO_THROW(sink.finalize());
    EXPECT_TRUE(sink.finalized());
    EXPECT_TRUE(sink.records().empty());
}

// ----------------------------------------------------------------------------
// per_pid_file_sink (slice C1)
// ----------------------------------------------------------------------------

TEST(per_pid_file_sink, empty_bytes_is_early_return)
{
    // Empty drains must not touch the filesystem or the registry —
    // per_pid_file_sink::on_source_drained returns early on empty bytes
    // so the (uninitialised in unit tests) config singleton is never
    // queried for the output filename.
    rocprofsys::output_file_registry registry;
    rocprofsys::core::per_pid_file_sink sink{ static_cast<pid_t>(1), registry };

    EXPECT_NO_THROW(sink.on_source_drained(1, std::vector<char>{}));
    EXPECT_NO_THROW(sink.finalize());
}

// NOTE: file-IO and RF5 (per-pid open-failure isolation) coverage for
// per_pid_file_sink is deferred to slice C2's integration tests, which
// exercise the sink through cache_manager + the live config singleton.
// Unit-level mocking of config::get_perfetto_output_filename(...) here
// would require library-init machinery this test binary does not own.
