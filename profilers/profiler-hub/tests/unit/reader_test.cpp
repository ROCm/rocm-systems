// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <string>

namespace
{

class reader_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_test, create_reader_instance) { ASSERT_NE(m_reader, nullptr); }

TEST_F(reader_test, get_node_list_returns_correct_value)
{
    auto node_list = m_reader->get_all_nodes();
    ASSERT_EQ(node_list.size(), 1);

    ASSERT_EQ(node_list[0]->node_id, 9162464413581981795);
    ASSERT_EQ(node_list[0]->hash, 9162464413581981795);
    ASSERT_EQ(node_list[0]->machine_id, "7cd7e017ddf442f5b7ce8428af366498");
    ASSERT_EQ(node_list[0]->system_name, "Linux");
    ASSERT_EQ(node_list[0]->hostname, "smci350-zts-gtu-c14-05");
    ASSERT_EQ(node_list[0]->release, "5.15.0-70-generic");
    ASSERT_EQ(node_list[0]->version, "#77-Ubuntu SMP Tue Mar 21 14:02:37 UTC 2023");
    ASSERT_EQ(node_list[0]->hardware_name, "x86_64");
    ASSERT_EQ(node_list[0]->domain_name, "(none)");
}

TEST_F(reader_test, get_process_list_returns_correct_value)
{
    auto process_list = m_reader->get_all_processes();
    ASSERT_EQ(process_list.size(), 1);

    ASSERT_EQ(process_list[0]->pid, 67979);
    ASSERT_EQ(process_list[0]->ppid, 67166);
    ASSERT_EQ(process_list[0]->command, "./bit_extract");
    ASSERT_EQ(process_list[0]->node_info->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_thread_list_returns_correct_value)
{
    auto thread_list = m_reader->get_all_threads();
    ASSERT_EQ(thread_list.size(), 4);

    // First thread
    ASSERT_EQ(thread_list[0]->thread_id, 67979);
    ASSERT_EQ(thread_list[0]->parent_process_id, 67166);
    ASSERT_EQ(thread_list[0]->name, "Thread 67979");
    ASSERT_EQ(thread_list[0]->start, 1702525691);
    ASSERT_EQ(thread_list[0]->process_info->pid, 67979);
    ASSERT_EQ(thread_list[0]->node_info->node_id, 9162464413581981795);

    // Second thread
    ASSERT_EQ(thread_list[1]->thread_id, 67991);
    ASSERT_EQ(thread_list[1]->name, "Thread 67991");
}

TEST_F(reader_test, get_agent_list_returns_correct_value)
{
    auto agent_list = m_reader->get_all_agents();
    ASSERT_EQ(agent_list.size(), 10);

    ASSERT_EQ(agent_list[0]->agent_type, "CPU");
    ASSERT_EQ(agent_list[0]->type_index, 0);
    ASSERT_EQ(agent_list[0]->absolute_index, 0);
    ASSERT_EQ(agent_list[0]->logical_index, 0);
    ASSERT_EQ(agent_list[0]->name, "AMD EPYC 9575F 64-Core Processor");
    ASSERT_EQ(agent_list[0]->model_name, "");
    ASSERT_EQ(agent_list[0]->vendor_name, "CPU");
    ASSERT_EQ(agent_list[0]->product_name, "AMD EPYC 9575F 64-Core Processor");
    ASSERT_EQ(agent_list[0]->process_info->pid, 67979);
    ASSERT_EQ(agent_list[0]->node_info->node_id, 9162464413581981795);

    ASSERT_EQ(agent_list[2]->agent_type, "GPU");
    ASSERT_EQ(agent_list[2]->type_index, 0);
    ASSERT_EQ(agent_list[2]->absolute_index, 2);
    ASSERT_EQ(agent_list[2]->name, "gfx950");
    ASSERT_EQ(agent_list[2]->model_name, "ip discovery");
    ASSERT_EQ(agent_list[2]->vendor_name, "AMD");
    ASSERT_EQ(agent_list[2]->product_name, "AMD Instinct MI350X");
}

TEST_F(reader_test, get_stream_list_returns_correct_value)
{
    auto stream_list = m_reader->get_all_streams();
    ASSERT_EQ(stream_list.size(), 1);

    ASSERT_EQ(stream_list[0]->stream_id, 0);
    ASSERT_EQ(stream_list[0]->name, "Stream 0");
    ASSERT_EQ(stream_list[0]->process_info->pid, 67979);
    ASSERT_EQ(stream_list[0]->node_info->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_queue_list_returns_correct_value)
{
    auto queue_list = m_reader->get_all_queues();
    ASSERT_EQ(queue_list.size(), 2);

    ASSERT_EQ(queue_list[0]->queue_id, 0);
    ASSERT_EQ(queue_list[0]->name, "Queue 0");
    ASSERT_EQ(queue_list[0]->process_info->pid, 67979);
    ASSERT_EQ(queue_list[0]->node_info->node_id, 9162464413581981795);

    ASSERT_EQ(queue_list[1]->queue_id, 1);
    ASSERT_EQ(queue_list[1]->name, "Queue 1");
}

TEST_F(reader_test, get_kernel_symbol_list_returns_correct_value)
{
    auto kernel_symbol_list = m_reader->get_all_kernel_symbols();
    ASSERT_EQ(kernel_symbol_list.size(), 11);

    // First kernel symbol
    ASSERT_EQ(kernel_symbol_list[0]->id, 1);
    ASSERT_EQ(kernel_symbol_list[0]->name, "__amd_rocclr_initHeap.kd");
    ASSERT_EQ(kernel_symbol_list[0]->display_name, "__amd_rocclr_initHeap.kd");
    ASSERT_EQ(kernel_symbol_list[0]->kernel_object, 2953328576);
    ASSERT_EQ(kernel_symbol_list[0]->kernarg_segment_size, 24);
    ASSERT_EQ(kernel_symbol_list[0]->kernarg_segment_alignment, 16);
    ASSERT_EQ(kernel_symbol_list[0]->sgpr_count, 32);
    ASSERT_EQ(kernel_symbol_list[0]->arch_vgpr_count, 8);
    ASSERT_EQ(kernel_symbol_list[0]->code_object_info->id, 1);
    ASSERT_EQ(kernel_symbol_list[0]->process_info->pid, 67979);
    ASSERT_EQ(kernel_symbol_list[0]->node_info->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_code_object_list_returns_correct_value)
{
    auto code_object_list = m_reader->get_all_code_objects();
    ASSERT_EQ(code_object_list.size(), 2);

    // First code object
    ASSERT_EQ(code_object_list[0]->id, 1);
    ASSERT_EQ(code_object_list[0]->uri, "memory://67979#offset=0x4608f10&size=32640");
    ASSERT_EQ(code_object_list[0]->load_base, 140018887163904);
    ASSERT_EQ(code_object_list[0]->load_size, 36864);
    ASSERT_EQ(code_object_list[0]->load_delta, 140018887163904);
    ASSERT_EQ(code_object_list[0]->storage_type, "MEMORY");
    ASSERT_EQ(code_object_list[0]->process_info->pid, 67979);
    ASSERT_EQ(code_object_list[0]->node_info->node_id, 9162464413581981795);
    ASSERT_EQ(code_object_list[0]->agent_info->agent_type, "GPU");
    ASSERT_EQ(code_object_list[0]->agent_info->type_index, 0);
}

TEST_F(reader_test, get_track_list_returns_correct_count)
{
    auto track_list = m_reader->get_all_tracks();
    ASSERT_EQ(track_list.size(), 2369);
}

TEST_F(reader_test, get_track_list_first_track_has_correct_values)
{
    auto track_list = m_reader->get_all_tracks();
    ASSERT_GE(track_list.size(), 1);

    // First track has name_id=9 which maps to "GPU Kernel Dispatch [0] Queue 1"
    ASSERT_EQ(track_list[0]->name, "GPU Kernel Dispatch [0] Queue 1");
    ASSERT_EQ(track_list[0]->node_info->node_id, 9162464413581981795);
    ASSERT_EQ(track_list[0]->process_info->pid, 67979);
}

TEST_F(reader_test, get_pmc_info_list_returns_correct_count)
{
    auto pmc_list = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_list.size(), 2358);
}

TEST_F(reader_test, get_pmc_info_list_first_item_has_correct_values)
{
    auto pmc_list = m_reader->get_all_pmc_info();
    ASSERT_GE(pmc_list.size(), 1);

    // First PMC info
    ASSERT_EQ(pmc_list[0]->name, "device_jpeg_activity_5_28");
    ASSERT_EQ(pmc_list[0]->agent_info->agent_type, "GPU");
    ASSERT_EQ(pmc_list[0]->target_arch, "GPU");
    ASSERT_EQ(pmc_list[0]->symbol, "JpegAct_5_28");
    ASSERT_EQ(pmc_list[0]->description, "JPEG Activity of a GPU device");
    ASSERT_EQ(pmc_list[0]->units, "%");
    ASSERT_EQ(pmc_list[0]->value_type, "ABS");
    ASSERT_EQ(pmc_list[0]->is_constant, 0);
    ASSERT_EQ(pmc_list[0]->is_derived, 0);
    ASSERT_EQ(pmc_list[0]->process_info->pid, 67979);
    ASSERT_EQ(pmc_list[0]->node_info->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_events_returns_non_empty_list)
{
    auto events = m_reader->get_events();
    ASSERT_GT(events.size(), 0);
}

TEST_F(reader_test, get_events_with_type_filter_region)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types = { profiler_hub::reader_types::event_type_t::region };
    auto events  = m_reader->get_events(filter);
    ASSERT_GT(events.size(), 0);

    for(const auto& event : events)
    {
        ASSERT_EQ(event.unique_identifier.type,
                  profiler_hub::reader_types::event_type_t::region);
    }
}

TEST_F(reader_test, get_events_region_has_correct_fields)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    const auto& event = events[0];
    ASSERT_EQ(event.unique_identifier.type,
              profiler_hub::reader_types::event_type_t::region);
    ASSERT_GT(event.unique_identifier.id, 0);
    ASSERT_GT(event.start_timestamp, 0);
    ASSERT_GE(event.end_timestamp, event.start_timestamp);
    ASSERT_FALSE(event.display_name.empty());
}

TEST_F(reader_test, get_events_with_pagination_limit)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.pagination = { 5, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_LE(events.size(), 5);
    ASSERT_GT(events.size(), 0);
}

TEST_F(reader_test, get_events_with_pagination_offset)
{
    auto all_events = m_reader->get_events();

    profiler_hub::reader_types::event_filter_t filter;
    filter.pagination  = { std::nullopt, 2 };
    auto offset_events = m_reader->get_events(filter);

    ASSERT_EQ(offset_events.size(), all_events.size() - 2);
}

TEST_F(reader_test, get_events_for_track_returns_events)
{
    auto tracks = m_reader->get_all_tracks();
    ASSERT_GT(tracks.size(), 0);

    bool found_events = false;
    for(const auto& track : tracks)
    {
        auto events = m_reader->get_events_for_track(track);
        if(!events.empty())
        {
            found_events = true;
            for(const auto& event : events)
            {
                ASSERT_NE(event.track, nullptr);
            }
            break;
        }
    }
    ASSERT_TRUE(found_events);
}

TEST_F(reader_test, get_event_count_matches_events_size)
{
    auto count  = m_reader->get_event_count();
    auto events = m_reader->get_events();
    ASSERT_EQ(count, events.size());
}

TEST_F(reader_test, get_event_count_ignores_pagination)
{
    const auto total = m_reader->get_event_count();
    ASSERT_GT(total, 1U);

    profiler_hub::reader_types::event_filter_t paged_filter;
    paged_filter.pagination.limit  = 1;
    paged_filter.pagination.offset = 0;

    ASSERT_EQ(m_reader->get_event_count(paged_filter), total);
}

TEST_F(reader_test, get_event_count_respects_types_filter)
{
    const auto counts = m_reader->get_event_counts({});

    profiler_hub::reader_types::event_filter_t region_filter;
    region_filter.types = { profiler_hub::reader_types::event_type_t::region };
    ASSERT_EQ(m_reader->get_event_count(region_filter),
              counts.at(profiler_hub::reader_types::event_type_t::region));

    profiler_hub::reader_types::event_filter_t dispatch_filter;
    dispatch_filter.types = { profiler_hub::reader_types::event_type_t::kernel_dispatch };
    ASSERT_EQ(m_reader->get_event_count(dispatch_filter),
              counts.at(profiler_hub::reader_types::event_type_t::kernel_dispatch));
}

TEST_F(reader_test, get_event_count_with_time_window_matches_filtered_events)
{
    const auto unfiltered = m_reader->get_events();
    ASSERT_FALSE(unfiltered.empty());

    auto first_start = unfiltered.front().start_timestamp;
    auto last_start  = unfiltered.back().start_timestamp;
    if(last_start < first_start) std::swap(first_start, last_start);

    const auto mid = first_start + (last_start - first_start) / 2;

    profiler_hub::reader_types::event_filter_t windowed;
    windowed.time_window.start = first_start;
    windowed.time_window.end   = mid;

    const auto windowed_events = m_reader->get_events(windowed);
    ASSERT_EQ(m_reader->get_event_count(windowed), windowed_events.size());
}

// ============================================================================
// Event detail tests
// ============================================================================

TEST_F(reader_test, get_region_details_first_region_has_correct_values)
{
    // First region in DB: id=1, start=23040314699996, end=23040314726875, name="mbind"
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_region_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->start_timestamp, 23040314699996);
    ASSERT_EQ(details->end_timestamp, 23040314726875);
    ASSERT_EQ(details->name, "mbind");
}

TEST_F(reader_test, get_region_details_has_event_metadata)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_region_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_NE(details->event, nullptr);
    // First region event_id=28 has category_id=3302 -> "numa"
    ASSERT_EQ(details->event->event_category, "numa");
}

TEST_F(reader_test, get_region_details_returns_nullopt_for_wrong_type)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::kernel_dispatch };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_region_details(events[0]);
    ASSERT_FALSE(details.has_value());
}

TEST_F(reader_test, get_kernel_dispatch_details_has_correct_values)
{
    // DB has 1 kernel dispatch: id=1, dispatch_id=1, wg=256x1x1, grid=131072x1x1
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::kernel_dispatch };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_kernel_dispatch_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->dispatch_id, 1);
    ASSERT_EQ(details->start_timestamp, 23040497580868);
    ASSERT_EQ(details->end_timestamp, 23040497591788);
    ASSERT_EQ(details->workgroup_size_x, 256);
    ASSERT_EQ(details->workgroup_size_y, 1);
    ASSERT_EQ(details->workgroup_size_z, 1);
    ASSERT_EQ(details->grid_size_x, 131072);
    ASSERT_EQ(details->grid_size_y, 1);
    ASSERT_EQ(details->grid_size_z, 1);
}

TEST_F(reader_test, get_kernel_dispatch_details_resolves_kernel_symbol)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::kernel_dispatch };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_kernel_dispatch_details(events[0]);
    ASSERT_TRUE(details.has_value());
    // kernel_id=11 -> display_name contains "bit_extract_kernel"
    ASSERT_NE(details->kernel_symbol_info, nullptr);
    EXPECT_NE(details->kernel_symbol_info->display_name.find("bit_extract_kernel"),
              std::string::npos);
}

TEST_F(reader_test, get_kernel_dispatch_details_resolves_node_and_process)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::kernel_dispatch };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_kernel_dispatch_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_NE(details->node_info, nullptr);
    ASSERT_EQ(details->node_info->node_id, 9162464413581981795);
    ASSERT_NE(details->process_info, nullptr);
    ASSERT_EQ(details->process_info->pid, 67979);
}

TEST_F(reader_test, get_memory_copy_details_has_correct_values)
{
    // DB has 2 memory copies. First: id=1, size=4000000, name=MEMORY_COPY_HOST_TO_DEVICE
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_copy };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_memory_copy_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->start_timestamp, 23040496787705);
    ASSERT_EQ(details->end_timestamp, 23040496865705);
    ASSERT_EQ(details->size, 4000000);
    ASSERT_EQ(details->name, "MEMORY_COPY_HOST_TO_DEVICE");
}

TEST_F(reader_test, get_memory_copy_details_resolves_agents)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_copy };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_memory_copy_details(events[0]);
    ASSERT_TRUE(details.has_value());
    // dst_agent_id=3, src_agent_id=1
    ASSERT_NE(details->dst_agent_id, nullptr);
    ASSERT_NE(details->src_agent_id, nullptr);
}

TEST_F(reader_test, get_memory_alloc_details_has_correct_values)
{
    // Inserted test data: id=1, type=ALLOC, level=REAL, size=4096, address=1048576
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_allocate };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_memory_alloc_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->type, "ALLOC");
    ASSERT_EQ(details->level, "REAL");
    ASSERT_EQ(details->start_timestamp, 23040314700000);
    ASSERT_EQ(details->end_timestamp, 23040314710000);
    ASSERT_TRUE(details->address.has_value());
    ASSERT_EQ(details->address.value(), 1048576);
    ASSERT_EQ(details->size, 4096);
}

TEST_F(reader_test, get_memory_alloc_details_has_event_with_call_stack)
{
    // The inserted memory_allocate event has call_stack JSON with hipMalloc
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_allocate };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_memory_alloc_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_NE(details->event, nullptr);

    // call_stack should be deserialized from JSON
    ASSERT_FALSE(details->event->call_stack.empty());
    ASSERT_TRUE(details->event->call_stack.front().program_counter.has_value());
    ASSERT_EQ(details->event->call_stack.front().program_counter->function, "hipMalloc");
    ASSERT_EQ(details->event->call_stack.front().program_counter->filename,
              "/opt/rocm/hip/src/hip_memory.cpp");
    ASSERT_TRUE(
        details->event->call_stack.front().program_counter->line_number.has_value());
    ASSERT_EQ(details->event->call_stack.front().program_counter->line_number.value(),
              123);

    // line_info should also be deserialized
    ASSERT_FALSE(details->event->line_info_list.empty());
    ASSERT_TRUE(details->event->line_info_list.front().program_counter.has_value());
    ASSERT_EQ(details->event->line_info_list.front().program_counter->function,
              "hipMalloc");
}

// ============================================================================
// Event property tests
// ============================================================================

TEST_F(reader_test, get_call_stack_for_memory_alloc_returns_hipMalloc)
{
    // The inserted memory_allocate event has call_stack with hipMalloc
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_allocate };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto stack = m_reader->get_call_stack(events[0]);
    ASSERT_EQ(stack.size(), 1);
    ASSERT_TRUE(stack.front().program_counter.has_value());
    ASSERT_EQ(stack.front().program_counter->function, "hipMalloc");

    ASSERT_TRUE(stack.front().address_range.has_value());
    ASSERT_EQ(stack.front().address_range->address_base, 4096);
    ASSERT_EQ(stack.front().address_range->address_high, 8192);
}

TEST_F(reader_test, get_source_context_for_memory_alloc_returns_entry)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_allocate };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto context = m_reader->get_source_context(events[0]);
    ASSERT_EQ(context.size(), 1);
    ASSERT_TRUE(context.front().program_counter.has_value());
    ASSERT_EQ(context.front().program_counter->function, "hipMalloc");
}

TEST_F(reader_test, get_call_stack_returns_empty_for_no_call_stack)
{
    // Region events in this DB have empty call_stack JSON
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto stack = m_reader->get_call_stack(events[0]);
    ASSERT_TRUE(stack.empty());
}

TEST_F(reader_test, get_arguments_for_hipGetDevice_has_correct_values)
{
    // Region id=23 (hipGetDevice, event_id=86) has 1 arg: pos=0, type=int*, name=deviceId
    // Region id=22 (hipGetDevice, event_id=85) has 0 args
    // Find the hipGetDevice instance that has args and verify values
    profiler_hub::reader_types::event_filter_t filter;
    filter.types = { profiler_hub::reader_types::event_type_t::region };
    auto events  = m_reader->get_events(filter);
    ASSERT_GT(events.size(), 0);

    bool found = false;
    for(const auto& event : events)
    {
        if(event.display_name != "hipGetDevice") continue;

        auto args = m_reader->get_arguments(event);
        if(args.empty()) continue;

        ASSERT_EQ(args.size(), 1);
        ASSERT_EQ(args[0]->position, 0);
        ASSERT_EQ(args[0]->type, "int*");
        ASSERT_EQ(args[0]->name, "deviceId");
        ASSERT_EQ(args[0]->value, "0");
        found = true;
        break;
    }
    ASSERT_TRUE(found) << "No hipGetDevice region with arguments found";
}

TEST_F(reader_test, get_arguments_returns_empty_for_event_without_args)
{
    // First region (mbind) has event_id=28 with 0 args
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto args = m_reader->get_arguments(events[0]);
    ASSERT_TRUE(args.empty());
}

TEST_F(reader_test, get_correlated_events_finds_related_events)
{
    // stack_id=7 has 2 events (event_id 182 and 203).
    // event_id=203 is in memory_copy (MC id=1)
    // We need to find the memory_copy event, then check its correlated events
    profiler_hub::reader_types::event_filter_t filter;
    filter.types = { profiler_hub::reader_types::event_type_t::memory_copy };
    auto events  = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto correlated = m_reader->get_correlated_events(events[0]);
    // Should find at least 1 correlated event (the region with the same stack_id)
    ASSERT_GE(correlated.size(), 1);
    // Correlated events should have valid IDs and not be the same event
    for(const auto& ce : correlated)
    {
        ASSERT_GT(ce.unique_identifier.id, 0);
    }
}

// ============================================================================
// Database metadata tests
// ============================================================================

TEST_F(reader_test, get_data_time_range_has_correct_values)
{
    auto range = m_reader->get_data_time_range();
    ASSERT_TRUE(range.start.has_value());
    ASSERT_TRUE(range.end.has_value());
    // min across all tables: 23040260707644, max: 23040498732102
    ASSERT_EQ(range.start.value(), 23040260707644);
    ASSERT_EQ(range.end.value(), 23040498732102);
}

TEST_F(reader_test, get_event_counts_has_correct_values)
{
    auto counts = m_reader->get_event_counts();

    // DB: 59 regions, 1 kernel dispatch, 2 memory copies, 1 memory allocate
    auto region_it = counts.find(profiler_hub::reader_types::event_type_t::region);
    ASSERT_NE(region_it, counts.end());
    ASSERT_EQ(region_it->second, 59);

    auto kd_it = counts.find(profiler_hub::reader_types::event_type_t::kernel_dispatch);
    ASSERT_NE(kd_it, counts.end());
    ASSERT_EQ(kd_it->second, 1);

    auto mc_it = counts.find(profiler_hub::reader_types::event_type_t::memory_copy);
    ASSERT_NE(mc_it, counts.end());
    ASSERT_EQ(mc_it->second, 2);

    auto ma_it = counts.find(profiler_hub::reader_types::event_type_t::memory_allocate);
    ASSERT_NE(ma_it, counts.end());
    ASSERT_EQ(ma_it->second, 1);
}

TEST_F(reader_test, get_event_counts_total_matches_get_events)
{
    auto counts = m_reader->get_event_counts();
    auto events = m_reader->get_events();

    size_t total = 0;
    for(const auto& [type, count] : counts)
    {
        total += count;
    }
    ASSERT_EQ(total, events.size());
}

}  // namespace
