// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// rocpd_processor_t::handle() for runtime/GPU samples against a StrictMock profiler-hub
// writer.

#include "rocpd_processor_mock_fixture.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::trace_cache::test
{
namespace
{

using ::testing::Eq;
using ::testing::Throw;

constexpr std::uint64_t k_start_ts = 5000;
constexpr std::uint64_t k_end_ts   = 6000;

class rocpd_processor_handle_test : public rocpd_processor_mock_test
{
protected:
    void add_gpu_and_cpu_agents()
    {
        add_agent(agent_type::gpu, 0, k_gpu_handle);
        add_agent(agent_type::cpu, 0, k_cpu_handle);
    }
};

// ---------------------------------------------------------------------------
// kernel_dispatch_sample
// ---------------------------------------------------------------------------

struct expected_dispatch
{
    std::size_t      dispatch_id;
    std::size_t      kernel_id;
    std::size_t      private_bytes;
    std::size_t      group_bytes;
    std::size_t      workgroup[3];
    std::size_t      grid[3];
    std::string_view name;
    expected_event   event;
};

MATCHER_P(IsKernelDispatch, expected, "kernel_dispatch_sample mapped to a dispatch")
{
    return arg.dispatch_id == expected.dispatch_id && arg.start_timestamp == k_start_ts &&
           arg.end_timestamp == k_end_ts && arg.kernel_symbol_id == expected.kernel_id &&
           arg.private_segment_size == expected.private_bytes &&
           arg.group_segment_size == expected.group_bytes &&
           arg.workgroup_size_x == expected.workgroup[0] &&
           arg.workgroup_size_y == expected.workgroup[1] &&
           arg.workgroup_size_z == expected.workgroup[2] &&
           arg.grid_size_x == expected.grid[0] && arg.grid_size_y == expected.grid[1] &&
           arg.grid_size_z == expected.grid[2] && arg.name == expected.name &&
           event_equals(arg.event, expected.event);
}

[[nodiscard]] expected_env
gpu_queue_stream_env()
{
    return { .agent_id  = make_uid("GPU", 0),
             .stream_id = k_stream_id,
             .queue_id  = k_queue_id };
}

TEST_F(rocpd_processor_handle_test,
       handle_kernel_dispatch_inserts_full_grid_with_correlation_ids)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);
    seed_kernel_symbol(9, "matmul_kernel");

    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        insert_kernel_dispatch_data(
            IsKernelDispatch(expected_dispatch{
                .dispatch_id   = 77,
                .kernel_id     = 9,
                .private_bytes = 2048,
                .group_bytes   = 16384,
                .workgroup     = { 64, 4, 2 },
                .grid          = { 256, 16, 8 },
                .name          = "matmul_kernel",
                .event         = { .stack_id        = 5,
                                   .parent_stack_id = 512,
                                   .correlation_id  = 0,
                                   .category =
                                       category_name<category::rocm_kernel_dispatch>() } }),
            IsEnv(gpu_queue_stream_env())))
        .Times(1);

    make_processor()->handle(kernel_dispatch_sample{
        k_start_ts, k_end_ts, k_thread_id, k_gpu_handle, 9, 77, k_queue_id, 5, 512, 2048,
        16384, 64, 4, 2, 256, 16, 8, k_stream_id });
}

TEST_F(rocpd_processor_handle_test, handle_kernel_dispatch_propagates_writer_failure)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);
    seed_kernel_symbol(9, "my_test_kernel");

    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_kernel_dispatch_data(::testing::_, ::testing::_))
        .WillOnce(Throw(std::runtime_error{ "Queue not registered" }));

    auto processor = make_processor();
    EXPECT_THAT(
        [&] {
            processor->handle(kernel_dispatch_sample{ k_start_ts, k_end_ts, k_thread_id,
                                                      k_gpu_handle, 9, 1, 999, 1, 0, 0, 0,
                                                      1, 1, 1, 1, 1, 1, k_stream_id });
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Queue not registered")));
}

// ---------------------------------------------------------------------------
// region_sample
// ---------------------------------------------------------------------------

struct expected_arg
{
    std::size_t      position;
    std::string_view type;
    std::string_view name;
    std::string_view value;
};

MATCHER_P3(IsRegion, name, event, args, "region_sample mapped to a region")
{
    if(arg.name != name || arg.start_timestamp != k_start_ts ||
       arg.end_timestamp != k_end_ts || !event_equals(arg.event, event) ||
       arg.args.size() != args.size())
    {
        return false;
    }
    for(std::size_t i = 0; i < args.size(); ++i)
    {
        const auto& got      = arg.args[i];
        const auto& expected = args[i];
        if(got.position != expected.position || got.type != expected.type ||
           got.name != expected.name || got.value != expected.value)
        {
            return false;
        }
    }
    return true;
}

[[nodiscard]] expected_env
thread_env()
{
    return {};
}

TEST_F(rocpd_processor_handle_test, handle_region_inserts_region_with_parsed_arguments)
{
    const auto region_args =
        get_args_string(function_args_t{ { .arg_number = 0U,
                                           .arg_type   = "void*",
                                           .arg_name   = "dst",
                                           .arg_value  = "0x7f0000000000" },
                                         { .arg_number = 1U,
                                           .arg_type   = "size_t",
                                           .arg_name   = "sizeBytes",
                                           .arg_value  = "4096" } });

    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_region_data(IsRegion(std::string_view{ "hipMemcpy" },
                                            expected_event{ .stack_id        = 1,
                                                            .parent_stack_id = 0,
                                                            .correlation_id  = 0,
                                                            .category        = "HIP_API",
                                                            .extdata         = "" },
                                            std::vector<expected_arg>{
                                                { 0, "void*", "dst", "0x7f0000000000" },
                                                { 1, "size_t", "sizeBytes", "4096" } }),
                                   IsEnv(thread_env())))
        .Times(1);

    make_processor()->handle(region_sample{ k_thread_id, "hipMemcpy", 1, 0, k_start_ts,
                                            k_end_ts, "", region_args, "HIP_API" });
}

TEST_F(rocpd_processor_handle_test, handle_region_forwards_call_stack_as_event_extdata)
{
    constexpr std::string_view k_call_stack = R"([{"function": "main"}])";

    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_region_data(IsRegion(std::string_view{ "hsa_signal_wait" },
                                            expected_event{ .stack_id        = 1,
                                                            .parent_stack_id = 0,
                                                            .correlation_id  = 0,
                                                            .category        = "HSA_API",
                                                            .extdata = k_call_stack },
                                            std::vector<expected_arg>{}),
                                   IsEnv(thread_env())))
        .Times(1);

    make_processor()->handle(region_sample{ k_thread_id, "hsa_signal_wait", 1, 0,
                                            k_start_ts, k_end_ts, k_call_stack, "",
                                            "HSA_API" });
}

TEST_F(rocpd_processor_handle_test, handle_region_propagates_writer_failure)
{
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_region_data(::testing::_, ::testing::_))
        .WillOnce(Throw(std::runtime_error{ "Thread not registered" }));

    auto processor = make_processor();
    EXPECT_THAT(
        [&] {
            processor->handle(region_sample{ 999, "hipMemcpy", 1, 0, k_start_ts, k_end_ts,
                                             "", "", "HIP_API" });
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Thread not registered")));
}

TEST_F(rocpd_processor_handle_test, handle_two_regions_then_dispatch_inserts_each_once)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);
    seed_kernel_symbol(1, "count_test_kernel");

    for(std::uint64_t idx = 0; idx < 2; ++idx)
    {
        EXPECT_CALL(*g_mock_profiler_hub_writer,
                    insert_region_data(IsRegion(idx == 0 ? std::string_view{ "region_0" }
                                                         : std::string_view{ "region_1" },
                                                expected_event{ .stack_id        = idx,
                                                                .parent_stack_id = 0,
                                                                .correlation_id  = 0,
                                                                .category = "HIP_API",
                                                                .extdata  = "" },
                                                std::vector<expected_arg>{}),
                                       IsEnv(thread_env())))
            .Times(1);
    }
    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        insert_kernel_dispatch_data(
            IsKernelDispatch(expected_dispatch{
                .dispatch_id   = 1,
                .kernel_id     = 1,
                .private_bytes = 0,
                .group_bytes   = 0,
                .workgroup     = { 64, 1, 1 },
                .grid          = { 256, 1, 1 },
                .name          = "count_test_kernel",
                .event         = { .stack_id        = 10,
                                   .parent_stack_id = 0,
                                   .correlation_id  = 0,
                                   .category =
                                       category_name<category::rocm_kernel_dispatch>() } }),
            IsEnv(gpu_queue_stream_env())))
        .Times(1);

    auto processor = make_processor();
    processor->handle(region_sample{ k_thread_id, "region_0", 0, 0, k_start_ts, k_end_ts,
                                     "", "", "HIP_API" });
    processor->handle(region_sample{ k_thread_id, "region_1", 1, 0, k_start_ts, k_end_ts,
                                     "", "", "HIP_API" });
    processor->handle(kernel_dispatch_sample{ k_start_ts, k_end_ts, k_thread_id,
                                              k_gpu_handle, 1, 1, k_queue_id, 10, 0, 0, 0,
                                              64, 1, 1, 256, 1, 1, k_stream_id });
}

// ---------------------------------------------------------------------------
// memory_copy_sample
// ---------------------------------------------------------------------------

constexpr std::size_t k_copy_bytes  = 4096;
constexpr std::size_t k_dst_address = 0x100000;
constexpr std::size_t k_src_address = 0x7F0000000000;

MATCHER_P(IsMemoryCopy, expected_name, "memory_copy_sample mapped to a copy")
{
    return arg.start_timestamp == k_start_ts && arg.end_timestamp == k_end_ts &&
           arg.dst_agent_id == make_uid("GPU", 0) &&
           arg.src_agent_id == make_uid("CPU", 0) && arg.dst_address == k_dst_address &&
           arg.src_address == k_src_address && arg.size == k_copy_bytes &&
           arg.name == expected_name && arg.region_name == expected_name &&
           event_equals(arg.event,
                        { .stack_id        = 1,
                          .parent_stack_id = 0,
                          .correlation_id  = 0,
                          .category = category_name<category::rocm_memory_copy>() });
}

[[nodiscard]] expected_env
copy_env()
{
    return { .stream_id = k_stream_id, .queue_id = 0 };
}

TEST_F(rocpd_processor_handle_test,
       handle_memory_copy_host_to_device_uses_buffer_name_and_both_agents)
{
    add_gpu_and_cpu_agents();

    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_memory_copy_data(
                    IsMemoryCopy(std::string_view{ "MEMORY_COPY_HOST_TO_DEVICE" }),
                    IsEnv(copy_env())))
        .Times(1);

    make_processor()->handle(
        memory_copy_sample{ k_start_ts, k_end_ts, k_thread_id, k_gpu_handle, k_cpu_handle,
                            "MEMORY_COPY_HOST_TO_DEVICE", k_copy_bytes, 1, 0,
                            k_dst_address, k_src_address, k_stream_id });
}

TEST_F(rocpd_processor_handle_test, handle_memory_copy_device_to_host_uses_its_own_name)
{
    add_gpu_and_cpu_agents();

    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_memory_copy_data(
                    IsMemoryCopy(std::string_view{ "MEMORY_COPY_DEVICE_TO_HOST" }),
                    IsEnv(copy_env())))
        .Times(1);

    make_processor()->handle(
        memory_copy_sample{ k_start_ts, k_end_ts, k_thread_id, k_gpu_handle, k_cpu_handle,
                            "MEMORY_COPY_DEVICE_TO_HOST", k_copy_bytes, 1, 0,
                            k_dst_address, k_src_address, k_stream_id });
}

TEST_F(rocpd_processor_handle_test, handle_memory_copy_propagates_writer_failure)
{
    add_gpu_and_cpu_agents();

    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_memory_copy_data(::testing::_, ::testing::_))
        .WillOnce(Throw(std::runtime_error{ "Stream not registered" }));

    auto processor = make_processor();
    EXPECT_THAT(
        [&] {
            processor->handle(memory_copy_sample{
                k_start_ts, k_end_ts, k_thread_id, k_gpu_handle, k_cpu_handle,
                "MEMORY_COPY_HOST_TO_DEVICE", k_copy_bytes, 1, 0, k_dst_address,
                k_src_address, 999 });
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Stream not registered")));
}

// ---------------------------------------------------------------------------
// scratch_memory_sample / memory_allocate_sample
// ---------------------------------------------------------------------------

constexpr std::size_t k_alloc_bytes   = 131072;
constexpr std::size_t k_scratch_flags = 7;

MATCHER(IsScratchAlloc, "scratch_memory_sample mapped to an allocation")
{
    return arg.type == "ALLOC" && arg.level == "SCRATCH" &&
           arg.start_timestamp == k_start_ts && arg.end_timestamp == k_end_ts &&
           arg.address == 0U && arg.size == k_alloc_bytes &&
           arg.extdata == R"({"flags": 7})" &&
           event_equals(arg.event,
                        { .stack_id        = 100,
                          .parent_stack_id = 50,
                          .correlation_id  = 0,
                          .category = category_name<category::rocm_scratch_memory>() });
}

TEST_F(rocpd_processor_handle_test, handle_scratch_memory_inserts_scratch_allocation)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);

    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_memory_alloc_data(IsScratchAlloc(), IsEnv(gpu_queue_stream_env())))
        .Times(1);

    make_processor()->handle(scratch_memory_sample{
        k_start_ts, k_end_ts, k_thread_id, k_gpu_handle, k_queue_id,
        "SCRATCH_MEMORY_ALLOC",
        static_cast<std::int32_t>(ROCPROFILER_SCRATCH_MEMORY_ALLOC), k_scratch_flags,
        k_alloc_bytes, 100, 50, k_stream_id });
}

TEST_F(rocpd_processor_handle_test, handle_scratch_memory_propagates_writer_failure)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);

    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_memory_alloc_data(::testing::_, ::testing::_))
        .WillOnce(Throw(std::runtime_error{ "Queue not registered" }));

    auto processor = make_processor();
    EXPECT_THAT(
        [&] {
            processor->handle(scratch_memory_sample{
                k_start_ts, k_end_ts, k_thread_id, k_gpu_handle, 999,
                "SCRATCH_MEMORY_ALLOC",
                static_cast<std::int32_t>(ROCPROFILER_SCRATCH_MEMORY_ALLOC),
                k_scratch_flags, k_alloc_bytes, 100, 50, k_stream_id });
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Queue not registered")));
}

#if(ROCPROFILER_VERSION >= 600)
constexpr std::size_t k_allocate_address = 0x7F0000100000;
constexpr std::size_t k_allocate_bytes   = 8192;

MATCHER(IsMemoryAllocate, "memory_allocate_sample mapped to an allocation")
{
    return arg.type == "ALLOC" && arg.level == "REAL" &&
           arg.start_timestamp == k_start_ts && arg.end_timestamp == k_end_ts &&
           arg.address == k_allocate_address && arg.size == k_allocate_bytes &&
           event_equals(arg.event,
                        { .stack_id        = 200,
                          .parent_stack_id = 100,
                          .correlation_id  = 0,
                          .category = category_name<category::rocm_memory_allocate>() });
}

TEST_F(rocpd_processor_handle_test, handle_memory_allocate_inserts_real_allocation)
{
    add_agent(agent_type::gpu, 0, k_gpu_handle);

    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        insert_memory_alloc_data(IsMemoryAllocate(),
                                 IsEnv(expected_env{ .agent_id  = make_uid("GPU", 0),
                                                     .stream_id = k_stream_id,
                                                     .queue_id  = 0 })))
        .Times(1);

    make_processor()->handle(memory_allocate_sample{
        k_start_ts, k_end_ts, k_thread_id, k_gpu_handle, "MEMORY_ALLOCATION_ALLOCATE",
        static_cast<std::int32_t>(ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE),
        k_allocate_bytes, 200, 100, k_allocate_address, k_stream_id });
}
#endif

// ---------------------------------------------------------------------------
// backtrace_region_sample
// ---------------------------------------------------------------------------

TEST_F(rocpd_processor_handle_test, handle_backtrace_region_uses_track_name_environment)
{
    constexpr std::string_view k_call_stack = R"({"backtrace": true})";

    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        insert_region_data(IsRegion(std::string_view{ "backtrace_sample_func" },
                                    expected_event{ .stack_id        = 0,
                                                    .parent_stack_id = 0,
                                                    .correlation_id  = 0,
                                                    .category        = "sampling",
                                                    .extdata         = k_call_stack },
                                    std::vector<expected_arg>{}),
                           IsEnv(expected_env{ .track_name = "Sampling [CPU 0]" })))
        .Times(1);

    make_processor()->handle(backtrace_region_sample{
        0, k_thread_id, "Sampling [CPU 0]", "backtrace_sample_func", k_start_ts, k_end_ts,
        "sampling", k_call_stack, "", "" });
}

// ---------------------------------------------------------------------------
// several handlers on one processor
// ---------------------------------------------------------------------------

TEST_F(rocpd_processor_handle_test, handle_mixed_samples_inserts_one_record_per_sample)
{
    add_gpu_and_cpu_agents();
    seed_kernel_symbol(1, "test_kernel");

    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_region_data(::testing::Field(&writer_types::region_data_t::name,
                                                    Eq("hipLaunchKernel")),
                                   IsEnv(thread_env())))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_region_data(
                    ::testing::Field(&writer_types::region_data_t::name, Eq("bt_func")),
                    IsEnv(expected_env{ .track_name = "Sampling" })))
        .Times(1);
    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        insert_kernel_dispatch_data(
            ::testing::Field(&writer_types::kernel_dispatch_data_t::name,
                             Eq(std::optional<std::string_view>{ "test_kernel" })),
            IsEnv(gpu_queue_stream_env())))
        .Times(1);
    EXPECT_CALL(*g_mock_profiler_hub_writer,
                insert_memory_copy_data(
                    ::testing::Field(&writer_types::memory_copy_data_t::size, Eq(2048U)),
                    IsEnv(copy_env())))
        .Times(1);
    EXPECT_CALL(
        *g_mock_profiler_hub_writer,
        insert_memory_alloc_data(
            ::testing::Field(&writer_types::memory_alloc_data_t::size, Eq(32768U)),
            IsEnv(gpu_queue_stream_env())))
        .Times(1);

    auto processor = make_processor();
    processor->handle(region_sample{ k_thread_id, "hipLaunchKernel", 1, 0, 1000, 1200, "",
                                     "", "HIP_API" });
    processor->handle(kernel_dispatch_sample{ 1500, 2000, k_thread_id, k_gpu_handle, 1, 1,
                                              k_queue_id, 2, 1, 0, 0, 128, 1, 1, 512, 1,
                                              1, k_stream_id });
    processor->handle(memory_copy_sample{ 2500, 3000, k_thread_id, k_gpu_handle,
                                          k_cpu_handle, "MEMORY_COPY_HOST_TO_DEVICE",
                                          2048, 3, 0, 0, 0, k_stream_id });
    processor->handle(scratch_memory_sample{
        3500, 3600, k_thread_id, k_gpu_handle, k_queue_id, "SCRATCH_MEMORY_ALLOC",
        static_cast<std::int32_t>(ROCPROFILER_SCRATCH_MEMORY_ALLOC), 0, 32768, 4, 0,
        k_stream_id });
    processor->handle(backtrace_region_sample{ 0, k_thread_id, "Sampling", "bt_func",
                                               4000, 4100, "sampling", "", "", "" });
}

}  // namespace
}  // namespace rocprofsys::trace_cache::test
