// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/storage.hpp"
#include "profiler-hub/cpp/writer.hpp"
#include "profiler-hub/cpp/writer_types.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace profiler_hub::test
{

/** Identifier fragment embedded into table names, so it must not contain hyphens. */
inline constexpr std::string_view trace_uuid = "testuuid0000";

[[nodiscard]] inline std::string
temp_trace_path(std::string_view prefix)
{
    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    return (std::filesystem::temp_directory_path() /
            (std::string{ prefix } + "_" + test_info->name() + ".db"))
        .string();
}

[[nodiscard]] inline std::unique_ptr<writer_t>
make_trace_writer(const std::string& path)
{
    return std::make_unique<writer_t>(
        std::make_unique<storage_t>(path, std::string{ trace_uuid }));
}

/** Registers one node, process, thread and the matching track. */
inline void
register_thread_track(writer_t& writer)
{
    writer.register_node_info(writer_types::node_info_t{ 1, 42, "machine-1" });

    writer_types::process_info_t process_info;
    process_info.pid     = 100;
    process_info.node_id = 1;
    writer.register_process_info(process_info);

    writer_types::thread_info_t thread_info;
    thread_info.thread_id  = 200;
    thread_info.node_id    = 1;
    thread_info.process_id = 100;
    writer.register_thread_info(thread_info);

    writer_types::track_info_t track_info;
    track_info.node_id    = 1;
    track_info.process_id = 100;
    track_info.thread_id  = 200;
    writer.register_track_info(track_info);
}

inline void
insert_region(writer_t&   writer,
              const char* name,
              uint64_t    start,
              uint64_t    end,
              uint64_t    thread_id = 200)
{
    writer_types::trace_environment_t trace_environment;
    trace_environment.node_id    = 1;
    trace_environment.process_id = 100;
    trace_environment.thread_id  = thread_id;

    writer_types::event_data_t event_data;
    event_data.stack_id = 1;

    writer_types::region_data_t region_data;
    region_data.name            = name;
    region_data.start_timestamp = start;
    region_data.end_timestamp   = end;
    region_data.event           = event_data;
    writer.insert_region_data(region_data, trace_environment);
}

/** Writes a trace with one thread track holding regions a [1000,2000] and b [5000,6000].
 */
inline void
write_thread_track_with_two_regions(const std::string& path)
{
    std::filesystem::remove(path);
    auto writer = make_trace_writer(path);
    register_thread_track(*writer);
    insert_region(*writer, "region-a", 1000, 2000);
    insert_region(*writer, "region-b", 5000, 6000);
    writer->flush_in_memory_data_to_disk();
}

/** Writes a trace with two thread tracks, each holding two regions. */
inline void
write_two_thread_tracks(const std::string& path)
{
    std::filesystem::remove(path);
    auto writer = make_trace_writer(path);
    register_thread_track(*writer);

    writer_types::thread_info_t thread_info;
    thread_info.thread_id  = 201;
    thread_info.node_id    = 1;
    thread_info.process_id = 100;
    writer->register_thread_info(thread_info);

    insert_region(*writer, "region-a", 1000, 2000);
    insert_region(*writer, "region-b", 5000, 6000);
    insert_region(*writer, "region-c", 1000, 2000, 201);
    insert_region(*writer, "region-d", 5000, 6000, 201);
    writer->flush_in_memory_data_to_disk();
}

/** Writes a trace that only has a node, so it contains no tracks. */
inline void
write_node_only(const std::string& path)
{
    std::filesystem::remove(path);
    auto writer = make_trace_writer(path);
    writer->register_node_info(writer_types::node_info_t{ 1, 42, "machine-1" });
    writer->flush_in_memory_data_to_disk();
}

}  // namespace profiler_hub::test
