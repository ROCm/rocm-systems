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

#include "core/trace_cache/sample_type.hpp"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

using namespace rocprofsys::trace_cache;

class sample_type_test : public ::testing::Test
{
protected:
    void SetUp() override { buffer.fill(0); }

    std::array<uint8_t, 4096> buffer;
};

TEST_F(sample_type_test, kernel_dispatch_sample_serialize_deserialize)
{
    kernel_dispatch_sample original(1000, 2000, 42, 100, 200, 300, 400, 500, 600, 1024,
                                    2048, 64, 32, 16, 256, 128, 64, 0xABCD);

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<kernel_dispatch_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.start_timestamp, original.start_timestamp);
    EXPECT_EQ(deserialized.end_timestamp, original.end_timestamp);
    EXPECT_EQ(deserialized.thread_id, original.thread_id);
    EXPECT_EQ(deserialized.agent_id_handle, original.agent_id_handle);
    EXPECT_EQ(deserialized.kernel_id, original.kernel_id);
    EXPECT_EQ(deserialized.dispatch_id, original.dispatch_id);
    EXPECT_EQ(deserialized.queue_id_handle, original.queue_id_handle);
    EXPECT_EQ(deserialized.correlation_id_internal, original.correlation_id_internal);
    EXPECT_EQ(deserialized.correlation_id_ancestor, original.correlation_id_ancestor);
    EXPECT_EQ(deserialized.private_segment_size, original.private_segment_size);
    EXPECT_EQ(deserialized.group_segment_size, original.group_segment_size);
    EXPECT_EQ(deserialized.workgroup_size_x, original.workgroup_size_x);
    EXPECT_EQ(deserialized.workgroup_size_y, original.workgroup_size_y);
    EXPECT_EQ(deserialized.workgroup_size_z, original.workgroup_size_z);
    EXPECT_EQ(deserialized.grid_size_x, original.grid_size_x);
    EXPECT_EQ(deserialized.grid_size_y, original.grid_size_y);
    EXPECT_EQ(deserialized.grid_size_z, original.grid_size_z);
    EXPECT_EQ(deserialized.stream_handle, original.stream_handle);
}

TEST_F(sample_type_test, kernel_dispatch_sample_get_size)
{
    kernel_dispatch_sample sample(1000, 2000, 42, 100, 200, 300, 400, 500, 600, 1024,
                                  2048, 64, 32, 16, 256, 128, 64, 0xABCD);

    size_t expected_size = sizeof(uint64_t) * 9 + sizeof(uint32_t) * 8 + sizeof(uint64_t);

    EXPECT_EQ(get_size(sample), expected_size);
}

TEST_F(sample_type_test, kernel_dispatch_sample_type_identifier)
{
    EXPECT_EQ(kernel_dispatch_sample::type_identifier,
              type_identifier_t::kernel_dispatch);
}

TEST_F(sample_type_test, memory_copy_sample_serialize_deserialize)
{
    memory_copy_sample original(5000, 6000, 123, 200, 201, 1, 2, 4096, 700, 800, 0x1000,
                                0x2000, 0xDEAD);

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<memory_copy_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.start_timestamp, original.start_timestamp);
    EXPECT_EQ(deserialized.end_timestamp, original.end_timestamp);
    EXPECT_EQ(deserialized.thread_id, original.thread_id);
    EXPECT_EQ(deserialized.dst_agent_id_handle, original.dst_agent_id_handle);
    EXPECT_EQ(deserialized.src_agent_id_handle, original.src_agent_id_handle);
    EXPECT_EQ(deserialized.kind, original.kind);
    EXPECT_EQ(deserialized.operation, original.operation);
    EXPECT_EQ(deserialized.bytes, original.bytes);
    EXPECT_EQ(deserialized.correlation_id_internal, original.correlation_id_internal);
    EXPECT_EQ(deserialized.correlation_id_ancestor, original.correlation_id_ancestor);
    EXPECT_EQ(deserialized.dst_address_value, original.dst_address_value);
    EXPECT_EQ(deserialized.src_address_value, original.src_address_value);
    EXPECT_EQ(deserialized.stream_handle, original.stream_handle);
}

TEST_F(sample_type_test, memory_copy_sample_get_size)
{
    memory_copy_sample sample(5000, 6000, 123, 200, 201, 1, 2, 4096, 700, 800, 0x1000,
                              0x2000, 0xDEAD);

    size_t expected_size = sizeof(uint64_t) * 11 + sizeof(int32_t) * 2;

    EXPECT_EQ(get_size(sample), expected_size);
}

TEST_F(sample_type_test, memory_copy_sample_type_identifier)
{
    EXPECT_EQ(memory_copy_sample::type_identifier, type_identifier_t::memory_copy);
}

TEST_F(sample_type_test, memory_allocate_sample_serialize_deserialize)
{
    memory_allocate_sample original(7000, 8000, 456, 300, 3, 4, 8192, 900, 1000, 0x3000,
                                    0xBEEF);

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<memory_allocate_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.start_timestamp, original.start_timestamp);
    EXPECT_EQ(deserialized.end_timestamp, original.end_timestamp);
    EXPECT_EQ(deserialized.thread_id, original.thread_id);
    EXPECT_EQ(deserialized.agent_id_handle, original.agent_id_handle);
    EXPECT_EQ(deserialized.kind, original.kind);
    EXPECT_EQ(deserialized.operation, original.operation);
    EXPECT_EQ(deserialized.allocation_size, original.allocation_size);
    EXPECT_EQ(deserialized.correlation_id_internal, original.correlation_id_internal);
    EXPECT_EQ(deserialized.correlation_id_ancestor, original.correlation_id_ancestor);
    EXPECT_EQ(deserialized.address_value, original.address_value);
    EXPECT_EQ(deserialized.stream_handle, original.stream_handle);
}

TEST_F(sample_type_test, memory_allocate_sample_get_size)
{
    memory_allocate_sample sample(7000, 8000, 456, 300, 3, 4, 8192, 900, 1000, 0x3000,
                                  0xBEEF);

    size_t expected_size = sizeof(uint64_t) * 9 + sizeof(int32_t) * 2;

    EXPECT_EQ(get_size(sample), expected_size);
}

TEST_F(sample_type_test, memory_allocate_sample_type_identifier)
{
    EXPECT_EQ(memory_allocate_sample::type_identifier, type_identifier_t::memory_alloc);
}

TEST_F(sample_type_test, region_sample_serialize_deserialize)
{
    region_sample original(789, "test_function", 1100, 1200, 10000, 20000,
                           "frame1\nframe2", "arg1=1, arg2=hello", "hip");

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<region_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.thread_id, original.thread_id);
    EXPECT_EQ(deserialized.name, original.name);
    EXPECT_EQ(deserialized.correlation_id_internal, original.correlation_id_internal);
    EXPECT_EQ(deserialized.correlation_id_ancestor, original.correlation_id_ancestor);
    EXPECT_EQ(deserialized.start_timestamp, original.start_timestamp);
    EXPECT_EQ(deserialized.end_timestamp, original.end_timestamp);
    EXPECT_EQ(deserialized.call_stack, original.call_stack);
    EXPECT_EQ(deserialized.args_str, original.args_str);
    EXPECT_EQ(deserialized.category, original.category);
}

TEST_F(sample_type_test, region_sample_get_size)
{
    region_sample sample(789, "test_function", 1100, 1200, 10000, 20000, "frame1\nframe2",
                         "arg1=1, arg2=hello", "hip");

    size_t expected_size = sizeof(uint64_t) * 5 + sizeof(size_t) * 4 + 13 + 13 + 18 + 3;

    EXPECT_EQ(get_size(sample), expected_size);
}

TEST_F(sample_type_test, region_sample_type_identifier)
{
    EXPECT_EQ(region_sample::type_identifier, type_identifier_t::region);
}

TEST_F(sample_type_test, region_sample_empty_strings)
{
    region_sample original(123, "", 0, 0, 0, 0, "", "", "");

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<region_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.thread_id, original.thread_id);
    EXPECT_EQ(deserialized.name, "");
    EXPECT_EQ(deserialized.call_stack, "");
    EXPECT_EQ(deserialized.args_str, "");
    EXPECT_EQ(deserialized.category, "");
}

TEST_F(sample_type_test, in_time_sample_serialize_deserialize)
{
    in_time_sample original(42, "GPU:0", 50000, "kernel_launch", 100, 99, 1500,
                            "main\nfoo\nbar", "file.cpp:42");

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<in_time_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.category_enum_id, original.category_enum_id);
    EXPECT_EQ(deserialized.track_name, original.track_name);
    EXPECT_EQ(deserialized.timestamp_ns, original.timestamp_ns);
    EXPECT_EQ(deserialized.event_metadata, original.event_metadata);
    EXPECT_EQ(deserialized.stack_id, original.stack_id);
    EXPECT_EQ(deserialized.parent_stack_id, original.parent_stack_id);
    EXPECT_EQ(deserialized.correlation_id, original.correlation_id);
    EXPECT_EQ(deserialized.call_stack, original.call_stack);
    EXPECT_EQ(deserialized.line_info, original.line_info);
}

TEST_F(sample_type_test, in_time_sample_get_size)
{
    in_time_sample sample(42, "GPU:0", 50000, "kernel_launch", 100, 99, 1500,
                          "main\nfoo\nbar", "file.cpp:42");

    size_t expected_size = sizeof(size_t) + sizeof(size_t) + 5 + sizeof(uint64_t) +
                           sizeof(size_t) + 13 + sizeof(uint64_t) * 3 + sizeof(size_t) +
                           12 + sizeof(size_t) + 11;

    EXPECT_EQ(get_size(sample), expected_size);
}

TEST_F(sample_type_test, in_time_sample_type_identifier)
{
    EXPECT_EQ(in_time_sample::type_identifier, type_identifier_t::in_time_sample);
}

TEST_F(sample_type_test, pmc_event_with_sample_serialize_deserialize)
{
    pmc_event_with_sample original(42, "CPU:0", 60000, "counter_sample", 200, 199, 1600,
                                   "entry\nexit", "counter.cpp:100", 5, 1,
                                   "PERF_COUNT_HW_CPU_CYCLES", 12345.67);

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<pmc_event_with_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.category_enum_id, original.category_enum_id);
    EXPECT_EQ(deserialized.track_name, original.track_name);
    EXPECT_EQ(deserialized.timestamp_ns, original.timestamp_ns);
    EXPECT_EQ(deserialized.event_metadata, original.event_metadata);
    EXPECT_EQ(deserialized.stack_id, original.stack_id);
    EXPECT_EQ(deserialized.parent_stack_id, original.parent_stack_id);
    EXPECT_EQ(deserialized.correlation_id, original.correlation_id);
    EXPECT_EQ(deserialized.call_stack, original.call_stack);
    EXPECT_EQ(deserialized.line_info, original.line_info);
    EXPECT_EQ(deserialized.device_id, original.device_id);
    EXPECT_EQ(deserialized.device_type, original.device_type);
    EXPECT_EQ(deserialized.pmc_info_name, original.pmc_info_name);
    EXPECT_DOUBLE_EQ(deserialized.value, original.value);
}

TEST_F(sample_type_test, pmc_event_with_sample_get_size)
{
    pmc_event_with_sample sample(42, "CPU:0", 60000, "counter_sample", 200, 199, 1600,
                                 "entry\nexit", "counter.cpp:100", 5, 1,
                                 "PERF_COUNT_HW_CPU_CYCLES", 12345.67);

    size_t expected_size = sizeof(size_t) * 6 + 5 + 14 + 10 + 15 + 24 +
                           sizeof(uint64_t) * 4 + sizeof(uint32_t) + sizeof(uint8_t) +
                           sizeof(double);

    EXPECT_EQ(get_size(sample), expected_size);
}

TEST_F(sample_type_test, pmc_event_with_sample_type_identifier)
{
    EXPECT_EQ(pmc_event_with_sample::type_identifier,
              type_identifier_t::pmc_event_with_sample);
}

TEST_F(sample_type_test, amd_smi_sample_serialize_deserialize)
{
    std::vector<uint8_t> gpu_activity_data = { 10, 20, 30, 40, 50 };
    amd_smi_sample       original(0xFF, 2, 70000, 80, 60, 40, 250, 75, 1024 * 1024 * 512,
                                  gpu_activity_data);

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<amd_smi_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.settings, original.settings);
    EXPECT_EQ(deserialized.device_id, original.device_id);
    EXPECT_EQ(deserialized.timestamp, original.timestamp);
    EXPECT_EQ(deserialized.gfx_activity, original.gfx_activity);
    EXPECT_EQ(deserialized.umc_activity, original.umc_activity);
    EXPECT_EQ(deserialized.mm_activity, original.mm_activity);
    EXPECT_EQ(deserialized.power, original.power);
    EXPECT_EQ(deserialized.temperature, original.temperature);
    EXPECT_EQ(deserialized.mem_usage, original.mem_usage);
    EXPECT_EQ(deserialized.gpu_activity.size(), original.gpu_activity.size());
    EXPECT_EQ(deserialized.gpu_activity, original.gpu_activity);
}

TEST_F(sample_type_test, amd_smi_sample_get_size)
{
    std::vector<uint8_t> gpu_activity_data = { 10, 20, 30, 40, 50 };
    amd_smi_sample       sample(0xFF, 2, 70000, 80, 60, 40, 250, 75, 1024 * 1024 * 512,
                                gpu_activity_data);

    size_t expected_size = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t) +
                           sizeof(uint32_t) * 4 + sizeof(int64_t) + sizeof(uint64_t) +
                           sizeof(size_t) + 5;

    EXPECT_EQ(get_size(sample), expected_size);
}

TEST_F(sample_type_test, amd_smi_sample_type_identifier)
{
    EXPECT_EQ(amd_smi_sample::type_identifier, type_identifier_t::amd_smi_sample);
}

TEST_F(sample_type_test, amd_smi_sample_empty_gpu_activity)
{
    std::vector<uint8_t> empty_activity;
    amd_smi_sample       original(0, 0, 0, 0, 0, 0, 0, 0, 0, empty_activity);

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<amd_smi_sample>(buffer_ptr);

    EXPECT_TRUE(deserialized.gpu_activity.empty());
}

TEST_F(sample_type_test, cpu_freq_sample_serialize_deserialize)
{
    std::vector<uint8_t> freqs_data = { 100, 150, 200, 180, 190, 195, 185, 170 };
    cpu_freq_sample      original(80000, 1024, 2048, 4096, 500, 100, 1000000, 500000,
                                  freqs_data);

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<cpu_freq_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.timestamp, original.timestamp);
    EXPECT_EQ(deserialized.page_rss, original.page_rss);
    EXPECT_EQ(deserialized.virt_mem_usage, original.virt_mem_usage);
    EXPECT_EQ(deserialized.peak_rss, original.peak_rss);
    EXPECT_EQ(deserialized.context_switch_count, original.context_switch_count);
    EXPECT_EQ(deserialized.page_faults, original.page_faults);
    EXPECT_EQ(deserialized.user_mode_time, original.user_mode_time);
    EXPECT_EQ(deserialized.kernel_mode_time, original.kernel_mode_time);
    EXPECT_EQ(deserialized.freqs.size(), original.freqs.size());
    EXPECT_EQ(deserialized.freqs, original.freqs);
}

TEST_F(sample_type_test, cpu_freq_sample_get_size)
{
    std::vector<uint8_t> freqs_data = { 100, 150, 200, 180, 190, 195, 185, 170 };
    cpu_freq_sample      sample(80000, 1024, 2048, 4096, 500, 100, 1000000, 500000,
                                freqs_data);

    size_t expected_size = sizeof(uint64_t) + sizeof(int64_t) * 7 + sizeof(size_t) + 8;

    EXPECT_EQ(get_size(sample), expected_size);
}

TEST_F(sample_type_test, cpu_freq_sample_type_identifier)
{
    EXPECT_EQ(cpu_freq_sample::type_identifier, type_identifier_t::cpu_freq_sample);
}

TEST_F(sample_type_test, cpu_freq_sample_empty_freqs)
{
    std::vector<uint8_t> empty_freqs;
    cpu_freq_sample      original(0, 0, 0, 0, 0, 0, 0, 0, empty_freqs);

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<cpu_freq_sample>(buffer_ptr);

    EXPECT_TRUE(deserialized.freqs.empty());
}

TEST_F(sample_type_test, backtrace_region_sample_serialize_deserialize)
{
    backtrace_region_sample original(1, 999, "Thread:999", "my_function", 90000, 95000,
                                     "rocm", "main\nworker\nfunc", "worker.cpp:256",
                                     "{\"extra\":\"data\"}");

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<backtrace_region_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.type, original.type);
    EXPECT_EQ(deserialized.thread_id, original.thread_id);
    EXPECT_EQ(deserialized.track_name, original.track_name);
    EXPECT_EQ(deserialized.name, original.name);
    EXPECT_EQ(deserialized.start_timestamp, original.start_timestamp);
    EXPECT_EQ(deserialized.end_timestamp, original.end_timestamp);
    EXPECT_EQ(deserialized.category, original.category);
    EXPECT_EQ(deserialized.call_stack, original.call_stack);
    EXPECT_EQ(deserialized.line_info, original.line_info);
    EXPECT_EQ(deserialized.extdata, original.extdata);
}

TEST_F(sample_type_test, backtrace_region_sample_get_size)
{
    backtrace_region_sample sample(1, 999, "Thread:999", "my_function", 90000, 95000,
                                   "rocm", "main\nworker\nfunc", "worker.cpp:256",
                                   "{\"extra\":\"data\"}");

    size_t expected_size = sizeof(uint32_t) + sizeof(uint64_t) * 3 + sizeof(size_t) * 6 +
                           10 + 11 + 4 + 16 + 14 + 16;

    EXPECT_EQ(get_size(sample), expected_size);
}

TEST_F(sample_type_test, backtrace_region_sample_type_identifier)
{
    EXPECT_EQ(backtrace_region_sample::type_identifier,
              type_identifier_t::backtrace_region_sample);
}

TEST_F(sample_type_test, backtrace_region_sample_empty_strings)
{
    backtrace_region_sample original(0, 0, "", "", 0, 0, "", "", "", "");

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<backtrace_region_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.track_name, "");
    EXPECT_EQ(deserialized.name, "");
    EXPECT_EQ(deserialized.category, "");
    EXPECT_EQ(deserialized.call_stack, "");
    EXPECT_EQ(deserialized.line_info, "");
    EXPECT_EQ(deserialized.extdata, "");
}

TEST_F(sample_type_test, type_identifier_enum_values)
{
    EXPECT_EQ(static_cast<uint32_t>(type_identifier_t::in_time_sample), 0x0000);
    EXPECT_EQ(static_cast<uint32_t>(type_identifier_t::pmc_event_with_sample), 0x0001);
    EXPECT_EQ(static_cast<uint32_t>(type_identifier_t::region), 0x0002);
    EXPECT_EQ(static_cast<uint32_t>(type_identifier_t::kernel_dispatch), 0x0003);
    EXPECT_EQ(static_cast<uint32_t>(type_identifier_t::memory_copy), 0x0004);
    EXPECT_EQ(static_cast<uint32_t>(type_identifier_t::memory_alloc), 0x0005);
    EXPECT_EQ(static_cast<uint32_t>(type_identifier_t::amd_smi_sample), 0x0006);
    EXPECT_EQ(static_cast<uint32_t>(type_identifier_t::cpu_freq_sample), 0x0007);
    EXPECT_EQ(static_cast<uint32_t>(type_identifier_t::backtrace_region_sample), 0x0008);
    EXPECT_EQ(static_cast<uint32_t>(type_identifier_t::fragmented_space), 0xFFFF);
}

TEST_F(sample_type_test, kernel_dispatch_sample_default_constructor)
{
    kernel_dispatch_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::kernel_dispatch);
}

TEST_F(sample_type_test, memory_copy_sample_default_constructor)
{
    memory_copy_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::memory_copy);
}

TEST_F(sample_type_test, memory_allocate_sample_default_constructor)
{
    memory_allocate_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::memory_alloc);
}

TEST_F(sample_type_test, region_sample_default_constructor)
{
    region_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::region);
}

TEST_F(sample_type_test, in_time_sample_default_constructor)
{
    in_time_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::in_time_sample);
}

TEST_F(sample_type_test, pmc_event_with_sample_default_constructor)
{
    pmc_event_with_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::pmc_event_with_sample);
}

TEST_F(sample_type_test, amd_smi_sample_default_constructor)
{
    amd_smi_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::amd_smi_sample);
}

TEST_F(sample_type_test, cpu_freq_sample_default_constructor)
{
    cpu_freq_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::cpu_freq_sample);
}

TEST_F(sample_type_test, backtrace_region_sample_default_constructor)
{
    backtrace_region_sample sample;
    EXPECT_EQ(sample.type_identifier, type_identifier_t::backtrace_region_sample);
}

TEST_F(sample_type_test, kernel_dispatch_sample_large_values)
{
    kernel_dispatch_sample original(
        UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX,
        UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
        UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, SIZE_MAX);

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<kernel_dispatch_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.start_timestamp, UINT64_MAX);
    EXPECT_EQ(deserialized.end_timestamp, UINT64_MAX);
    EXPECT_EQ(deserialized.thread_id, UINT64_MAX);
    EXPECT_EQ(deserialized.private_segment_size, UINT32_MAX);
    EXPECT_EQ(deserialized.grid_size_z, UINT32_MAX);
}

TEST_F(sample_type_test, amd_smi_sample_large_gpu_activity)
{
    std::vector<uint8_t> large_activity(256);
    for(size_t i = 0; i < large_activity.size(); ++i)
    {
        large_activity[i] = static_cast<uint8_t>(i);
    }

    amd_smi_sample original(0xFF, 0, 0, 0, 0, 0, 0, 0, 0, large_activity);

    serialize(buffer.data(), original);

    uint8_t* buffer_ptr   = buffer.data();
    auto     deserialized = deserialize<amd_smi_sample>(buffer_ptr);

    EXPECT_EQ(deserialized.gpu_activity.size(), 256);
    EXPECT_EQ(deserialized.gpu_activity, large_activity);
}
