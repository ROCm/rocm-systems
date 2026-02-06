// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "rocstorage/reader.hpp"
#include "rocstorage/storage.hpp"

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
        m_storage = std::make_unique<rocstorage::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<rocstorage::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                            m_database_path{ ROCPD_DB_PATH };
    std::unique_ptr<rocstorage::storage_t> m_storage;
    std::shared_ptr<rocstorage::reader_t>  m_reader;
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
    rocstorage::reader_types::event_filter_t filter;
    filter.types = { rocstorage::reader_types::event_type_t::region };
    auto events  = m_reader->get_events(filter);
    ASSERT_GT(events.size(), 0);

    for(const auto& event : events)
    {
        ASSERT_EQ(event.unique_identifier.type,
                  rocstorage::reader_types::event_type_t::region);
    }
}

TEST_F(reader_test, get_events_region_has_correct_fields)
{
    rocstorage::reader_types::event_filter_t filter;
    filter.types      = { rocstorage::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    const auto& event = events[0];
    ASSERT_EQ(event.unique_identifier.type,
              rocstorage::reader_types::event_type_t::region);
    ASSERT_GT(event.unique_identifier.id, 0);
    ASSERT_GT(event.start_timestamp, 0);
    ASSERT_GE(event.end_timestamp, event.start_timestamp);
    ASSERT_FALSE(event.display_name.empty());
}

TEST_F(reader_test, get_events_with_pagination_limit)
{
    rocstorage::reader_types::event_filter_t filter;
    filter.pagination = { 5, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_LE(events.size(), 5);
    ASSERT_GT(events.size(), 0);
}

TEST_F(reader_test, get_events_with_pagination_offset)
{
    auto all_events = m_reader->get_events();

    rocstorage::reader_types::event_filter_t filter;
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

}  // namespace
