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
    ASSERT_STREQ(node_list[0]->machine_id, "7cd7e017ddf442f5b7ce8428af366498");
    ASSERT_STREQ(node_list[0]->system_name, "Linux");
    ASSERT_STREQ(node_list[0]->hostname, "smci350-zts-gtu-c14-05");
    ASSERT_STREQ(node_list[0]->release, "5.15.0-70-generic");
    ASSERT_STREQ(node_list[0]->version, "#77-Ubuntu SMP Tue Mar 21 14:02:37 UTC 2023");
    ASSERT_STREQ(node_list[0]->hardware_name, "x86_64");
    ASSERT_STREQ(node_list[0]->domain_name, "(none)");
}

TEST_F(reader_test, get_process_list_returns_correct_value)
{
    auto process_list = m_reader->get_all_processes();
    ASSERT_EQ(process_list.size(), 1);

    ASSERT_EQ(process_list[0]->pid, 67979);
    ASSERT_EQ(process_list[0]->ppid, 67166);
    ASSERT_STREQ(process_list[0]->command, "./bit_extract");
    ASSERT_EQ(process_list[0]->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_thread_list_returns_correct_value)
{
    auto thread_list = m_reader->get_all_threads();
    ASSERT_EQ(thread_list.size(), 4);

    // First thread
    ASSERT_EQ(thread_list[0]->thread_id, 67979);
    ASSERT_EQ(thread_list[0]->parent_process_id, 67166);
    ASSERT_STREQ(thread_list[0]->name, "Thread 67979");
    ASSERT_EQ(thread_list[0]->start, 1702525691);
    ASSERT_EQ(thread_list[0]->process_id, 67979);
    ASSERT_EQ(thread_list[0]->node_id, 9162464413581981795);

    // Second thread
    ASSERT_EQ(thread_list[1]->thread_id, 67991);
    ASSERT_STREQ(thread_list[1]->name, "Thread 67991");
}

TEST_F(reader_test, get_agent_list_returns_correct_value)
{
    auto agent_list = m_reader->get_all_agents();
    ASSERT_EQ(agent_list.size(), 10);

    // First agent (CPU) - model_name is empty, vendor_name is "CPU"
    ASSERT_STREQ(agent_list[0]->unique_id.agent_type, "CPU");
    ASSERT_EQ(agent_list[0]->unique_id.type_index, 0);
    ASSERT_EQ(agent_list[0]->absolute_index, 0);
    ASSERT_EQ(agent_list[0]->logical_index, 0);
    ASSERT_STREQ(agent_list[0]->name, "AMD EPYC 9575F 64-Core Processor");
    ASSERT_STREQ(agent_list[0]->model_name, "");  // CPU has empty model_name
    ASSERT_STREQ(agent_list[0]->vendor_name, "CPU");
    ASSERT_STREQ(agent_list[0]->product_name, "AMD EPYC 9575F 64-Core Processor");
    ASSERT_EQ(agent_list[0]->process_id, 67979);
    ASSERT_EQ(agent_list[0]->node_id, 9162464413581981795);

    // Third agent (GPU)
    ASSERT_STREQ(agent_list[2]->unique_id.agent_type, "GPU");
    ASSERT_EQ(agent_list[2]->unique_id.type_index, 0);
    ASSERT_EQ(agent_list[2]->absolute_index, 2);
    ASSERT_STREQ(agent_list[2]->name, "gfx950");
    ASSERT_STREQ(agent_list[2]->model_name, "ip discovery");
    ASSERT_STREQ(agent_list[2]->vendor_name, "AMD");
    ASSERT_STREQ(agent_list[2]->product_name, "AMD Instinct MI350X");
}

TEST_F(reader_test, get_stream_list_returns_correct_value)
{
    auto stream_list = m_reader->get_all_streams();
    ASSERT_EQ(stream_list.size(), 1);

    ASSERT_EQ(stream_list[0]->stream_id, 0);
    ASSERT_STREQ(stream_list[0]->name, "Stream 0");
    ASSERT_EQ(stream_list[0]->process_id, 67979);
    ASSERT_EQ(stream_list[0]->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_queue_list_returns_correct_value)
{
    auto queue_list = m_reader->get_all_queues();
    ASSERT_EQ(queue_list.size(), 2);

    ASSERT_EQ(queue_list[0]->queue_id, 0);
    ASSERT_STREQ(queue_list[0]->name, "Queue 0");
    ASSERT_EQ(queue_list[0]->process_id, 67979);
    ASSERT_EQ(queue_list[0]->node_id, 9162464413581981795);

    ASSERT_EQ(queue_list[1]->queue_id, 1);
    ASSERT_STREQ(queue_list[1]->name, "Queue 1");
}

TEST_F(reader_test, get_kernel_symbol_list_returns_correct_value)
{
    auto kernel_symbol_list = m_reader->get_all_kernel_symbols();
    ASSERT_EQ(kernel_symbol_list.size(), 11);

    // First kernel symbol
    ASSERT_EQ(kernel_symbol_list[0]->id, 1);
    ASSERT_STREQ(kernel_symbol_list[0]->name, "__amd_rocclr_initHeap.kd");
    ASSERT_STREQ(kernel_symbol_list[0]->display_name, "__amd_rocclr_initHeap.kd");
    ASSERT_EQ(kernel_symbol_list[0]->kernel_object, 2953328576);
    ASSERT_EQ(kernel_symbol_list[0]->kernarg_segment_size, 24);
    ASSERT_EQ(kernel_symbol_list[0]->kernarg_segment_alignment, 16);
    ASSERT_EQ(kernel_symbol_list[0]->sgpr_count, 32);
    ASSERT_EQ(kernel_symbol_list[0]->arch_vgpr_count, 8);
    ASSERT_EQ(kernel_symbol_list[0]->code_obj_id, 1);
    ASSERT_EQ(kernel_symbol_list[0]->process_id, 67979);
    ASSERT_EQ(kernel_symbol_list[0]->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_code_object_list_returns_correct_value)
{
    auto code_object_list = m_reader->get_all_code_objects();
    ASSERT_EQ(code_object_list.size(), 2);

    // First code object
    ASSERT_EQ(code_object_list[0]->id, 1);
    ASSERT_STREQ(code_object_list[0]->uri, "memory://67979#offset=0x4608f10&size=32640");
    ASSERT_EQ(code_object_list[0]->load_base, 140018887163904);
    ASSERT_EQ(code_object_list[0]->load_size, 36864);
    ASSERT_EQ(code_object_list[0]->load_delta, 140018887163904);
    ASSERT_STREQ(code_object_list[0]->storage_type, "MEMORY");
    ASSERT_EQ(code_object_list[0]->process_id, 67979);
    ASSERT_EQ(code_object_list[0]->node_id, 9162464413581981795);
    // Agent ID should be resolved to GPU agent
    ASSERT_TRUE(code_object_list[0]->agent_id.has_value());
    ASSERT_STREQ(code_object_list[0]->agent_id->agent_type, "GPU");
    ASSERT_EQ(code_object_list[0]->agent_id->type_index, 0);
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
    ASSERT_TRUE(track_list[0]->name.has_value());
    ASSERT_STREQ(track_list[0]->name.value(), "GPU Kernel Dispatch [0] Queue 1");
    ASSERT_EQ(track_list[0]->node_id, 9162464413581981795);
    ASSERT_TRUE(track_list[0]->process_id.has_value());
    ASSERT_EQ(track_list[0]->process_id.value(), 67979);
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
    ASSERT_STREQ(pmc_list[0]->unique_id.name, "device_jpeg_activity_5_28");
    ASSERT_TRUE(pmc_list[0]->unique_id.agent_id.has_value());
    ASSERT_STREQ(pmc_list[0]->unique_id.agent_id->agent_type, "GPU");
    ASSERT_STREQ(pmc_list[0]->target_arch, "GPU");
    ASSERT_STREQ(pmc_list[0]->symbol, "JpegAct_5_28");
    ASSERT_STREQ(pmc_list[0]->description, "JPEG Activity of a GPU device");
    ASSERT_STREQ(pmc_list[0]->units, "%");
    ASSERT_STREQ(pmc_list[0]->value_type, "ABS");
    ASSERT_EQ(pmc_list[0]->is_constant, 0);
    ASSERT_EQ(pmc_list[0]->is_derived, 0);
    ASSERT_EQ(pmc_list[0]->process_id, 67979);
    ASSERT_EQ(pmc_list[0]->node_id, 9162464413581981795);
}

}  // namespace
