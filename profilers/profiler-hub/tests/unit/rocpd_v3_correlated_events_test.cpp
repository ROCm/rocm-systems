// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "event_query.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/storage.hpp"
#include "trace_fixtures.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace profiler_hub;
using reader_types::event_type_t;

/** One event written by the fixture, identified by its type and start time. */
using event_key_t = std::pair<event_type_t, uint64_t>;

enum class stack_group_t
{
    one,
    two,
    zero,
    unset,
};

constexpr uint64_t group_stride = 10000;

uint64_t
group_base(stack_group_t group)
{
    return group_stride * (static_cast<uint64_t>(group) + 1);
}

/** Start time of the @p index-th (0 or 1) event of @p type in @p group. */
uint64_t
start_of(stack_group_t group, event_type_t type, size_t index)
{
    uint64_t offset = 0;
    switch(type)
    {
        case event_type_t::region: offset = 0; break;
        case event_type_t::kernel_dispatch: offset = 100; break;
        case event_type_t::memory_copy: offset = 200; break;
        case event_type_t::memory_allocate: offset = 300; break;
        default: break;
    }
    return group_base(group) + offset + 10 * index;
}

constexpr event_type_t all_types[] = { event_type_t::region,
                                       event_type_t::kernel_dispatch,
                                       event_type_t::memory_copy,
                                       event_type_t::memory_allocate };

/** The types an event of @p source reaches through a shared stack id. */
std::vector<event_type_t>
reachable_types(event_type_t source)
{
    switch(source)
    {
        case event_type_t::region: return { all_types, all_types + 4 };
        case event_type_t::kernel_dispatch:
            return { event_type_t::region, event_type_t::kernel_dispatch };
        case event_type_t::memory_copy:
            return { event_type_t::region, event_type_t::memory_copy };
        case event_type_t::memory_allocate: return { event_type_t::region };
        default: return {};
    }
}

uint32_t
table_type_of(event_type_t type)
{
    switch(type)
    {
        case event_type_t::region: return PH_EVENT_TYPE_REGION;
        case event_type_t::kernel_dispatch: return PH_EVENT_TYPE_KERNEL_DISPATCH;
        case event_type_t::memory_copy: return PH_EVENT_TYPE_MEMORY_COPY;
        case event_type_t::memory_allocate: return PH_EVENT_TYPE_MEMORY_ALLOCATE;
        default: return PH_EVENT_TYPE_REGION;
    }
}

class rocpd_v3_correlated_events_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_path =
            std::filesystem::temp_directory_path() / "rocpd_v3_correlated_events_test.db";
        std::filesystem::remove(m_path);
        write_trace();
        m_reader =
            std::make_unique<reader_t>(std::make_unique<storage_t>(m_path.string(), ""));
    }

    void TearDown() override
    {
        m_reader.reset();
        std::filesystem::remove(m_path);
    }

    /** Writes two events of every type with @p stack_id, or no stack id if empty. */
    void write_group(writer_t&                                writer,
                     const writer_types::trace_environment_t& environment,
                     stack_group_t                            group,
                     std::optional<size_t>                    stack_id) const
    {
        writer_types::event_data_t event;
        event.stack_id = stack_id;

        for(size_t index = 0; index < 2; ++index)
        {
            // The writer's name fields are string_views, so the names must outlive
            // the inserts below.
            const std::string suffix =
                std::to_string(static_cast<int>(group)) + "-" + std::to_string(index);
            const std::string region_name = "region-" + suffix;
            const std::string kernel_name = "kernel-" + suffix;
            const std::string copy_name   = "copy-" + suffix;

            const uint64_t region_start = start_of(group, event_type_t::region, index);
            writer_types::region_data_t region;
            region.name            = region_name;
            region.start_timestamp = region_start;
            region.end_timestamp   = region_start + 5;
            region.event           = event;
            writer.insert_region_data(region, environment);

            const uint64_t kernel_start =
                start_of(group, event_type_t::kernel_dispatch, index);
            writer_types::kernel_dispatch_data_t kernel;
            kernel.kernel_symbol_id = 20;
            kernel.start_timestamp  = kernel_start;
            kernel.end_timestamp    = kernel_start + 5;
            kernel.name             = kernel_name;
            kernel.event            = event;
            writer.insert_kernel_dispatch_data(kernel, environment);

            const uint64_t copy_start = start_of(group, event_type_t::memory_copy, index);
            writer_types::memory_copy_data_t copy;
            copy.name            = copy_name;
            copy.region_name     = copy_name;
            copy.start_timestamp = copy_start;
            copy.end_timestamp   = copy_start + 5;
            copy.size            = 64;
            copy.event           = event;
            writer.insert_memory_copy_data(copy, environment);

            const uint64_t alloc_start =
                start_of(group, event_type_t::memory_allocate, index);
            writer_types::memory_alloc_data_t alloc;
            alloc.type            = "ALLOC";
            alloc.level           = "REAL";
            alloc.start_timestamp = alloc_start;
            alloc.end_timestamp   = alloc_start + 5;
            alloc.size            = 64;
            alloc.event           = event;
            writer.insert_memory_alloc_data(alloc, environment);
        }
    }

    void write_trace() const
    {
        auto writer = test::make_trace_writer(m_path.string());
        test::register_thread_track(*writer);

        writer_types::agent_info_t agent;
        agent.unique_id.agent_type = "GPU";
        agent.unique_id.type_index = 0;
        agent.node_id              = 1;
        agent.process_id           = 100;
        writer->register_agent_info(agent);

        writer_types::queue_info_t queue;
        queue.queue_id   = 30;
        queue.node_id    = 1;
        queue.process_id = 100;
        writer->register_queue_info(queue);

        writer_types::stream_info_t stream;
        stream.stream_id  = 40;
        stream.node_id    = 1;
        stream.process_id = 100;
        writer->register_stream_info(stream);

        writer_types::code_object_info_t code_object;
        code_object.id         = 10;
        code_object.node_id    = 1;
        code_object.process_id = 100;
        writer->register_code_object_info(code_object);

        writer_types::kernel_symbol_info_t symbol;
        symbol.id          = 20;
        symbol.node_id     = 1;
        symbol.process_id  = 100;
        symbol.code_obj_id = code_object.id;
        writer->register_kernel_symbol_info(symbol);

        writer_types::trace_environment_t environment;
        environment.node_id    = 1;
        environment.process_id = 100;
        environment.thread_id  = 200;
        environment.agent_id   = agent.unique_id;
        environment.queue_id   = queue.queue_id;
        environment.stream_id  = stream.stream_id;

        write_group(*writer, environment, stack_group_t::one, 1);
        write_group(*writer, environment, stack_group_t::two, 2);
        write_group(*writer, environment, stack_group_t::zero, 0);
        write_group(*writer, environment, stack_group_t::unset, std::nullopt);
        writer->flush_in_memory_data_to_disk();
    }

    /** Looks up the stored event written at @p start and returns it as a source. */
    [[nodiscard]] reader_types::timeline_event_t source_event(event_type_t type,
                                                              uint64_t     start) const
    {
        std::vector<ph_table_event> rows(64);
        const uint32_t              count = m_reader->read_table_events(
            resolved_event_query{}, table_type_of(type), 0, UINT64_MAX, rows.data(), 64);
        rows.resize(count);
        const auto found = std::find_if(rows.begin(), rows.end(), [&](const auto& row) {
            return row.start == start;
        });
        EXPECT_NE(found, rows.end()) << "no event of the type starts at " << start;

        reader_types::timeline_event_t source{};
        source.unique_identifier.type = type;
        source.unique_identifier.id   = found == rows.end() ? 0 : found->id;
        return source;
    }

    /** Correlated events of the event written at @p start, as sorted keys. */
    [[nodiscard]] std::vector<event_key_t> correlated(event_type_t type,
                                                      uint64_t     start) const
    {
        const auto events = m_reader->get_correlated_events(source_event(type, start));
        std::vector<event_key_t> keys;
        for(const auto& event : events)
        {
            keys.emplace_back(event.unique_identifier.type, event.start_timestamp);
        }
        std::sort(keys.begin(), keys.end());
        return keys;
    }

    /** Every event of @p group in a type @p source reaches, except @p excluded. */
    [[nodiscard]] static std::vector<event_key_t> expected_keys(stack_group_t group,
                                                                event_type_t  source,
                                                                uint64_t      excluded)
    {
        std::vector<event_key_t> keys;
        for(const event_type_t type : reachable_types(source))
        {
            for(size_t index = 0; index < 2; ++index)
            {
                const uint64_t start = start_of(group, type, index);
                if(start != excluded) keys.emplace_back(type, start);
            }
        }
        std::sort(keys.begin(), keys.end());
        return keys;
    }

    std::filesystem::path     m_path;
    std::unique_ptr<reader_t> m_reader;
};

}  // namespace

TEST_F(rocpd_v3_correlated_events_test, a_zero_or_unset_stack_id_correlates_nothing)
{
    // Each of these groups holds two events of every type on the same stack, so a
    // stack id of 0 (or none) that were matched like any other would return rows.
    for(const stack_group_t group : { stack_group_t::zero, stack_group_t::unset })
    {
        for(const event_type_t type : all_types)
        {
            for(size_t index = 0; index < 2; ++index)
            {
                EXPECT_TRUE(correlated(type, start_of(group, type, index)).empty())
                    << "group " << static_cast<int>(group) << " type "
                    << static_cast<int>(type) << " index " << index;
            }
        }
    }
}

TEST_F(rocpd_v3_correlated_events_test, each_event_type_reaches_only_its_listed_types)
{
    // The expected set is every other event of the reachable types on the stack, so
    // this also checks that the source event is excluded.
    for(const stack_group_t group : { stack_group_t::one, stack_group_t::two })
    {
        for(const event_type_t source : all_types)
        {
            for(size_t index = 0; index < 2; ++index)
            {
                const uint64_t start = start_of(group, source, index);
                EXPECT_EQ(correlated(source, start), expected_keys(group, source, start))
                    << "group " << static_cast<int>(group) << " source type "
                    << static_cast<int>(source) << " index " << index;
            }
        }
    }
}

TEST_F(rocpd_v3_correlated_events_test, events_on_another_stack_are_not_returned)
{
    for(const stack_group_t group : { stack_group_t::one, stack_group_t::two })
    {
        const uint64_t group_start = group_base(group);
        const uint64_t group_end   = group_base(group) + group_stride;
        for(const event_type_t source : all_types)
        {
            const auto keys = correlated(source, start_of(group, source, 0));
            ASSERT_FALSE(keys.empty());
            for(const auto& [type, start] : keys)
            {
                EXPECT_GE(start, group_start);
                EXPECT_LT(start, group_end);
            }
        }
    }
}
