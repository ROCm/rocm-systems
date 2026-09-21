// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered/kernel_dispatch.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"
#include <cstdint>

#include <fmt/format.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

namespace rocprofsys::domains::buffered
{
namespace
{

using ::testing::Eq;
using ::testing::StrictMock;

using test_support::externals;
using test_support::g_buffer_storage_mock;
using test_support::g_metadata_registry_mock;
using test_support::gmock_buffer_storage;
using test_support::gmock_metadata_registry;
using test_support::kernel_dispatch_sample_data_t;
using test_support::mock_sdk;
using test_support::thread_info_data_t;
using test_support::track_data_t;

}  // namespace

TEST(kernel_dispatch_test, descriptor_reports_correct_metadata)
{
    using mock_dispatcher =
        buffered_callback_dispatcher<mock_sdk, mock_sdk::kernel_dispatch_record_t,
                                     on_kernel_dispatch<mock_sdk, externals>>;
    constexpr const auto& k_domain = k_kernel_dispatch<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "kernel_dispatch");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::BUFFER_TRACING_KERNEL_DISPATCH);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::buffered);
    EXPECT_FALSE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.on_records, &mock_dispatcher::callback);
}

TEST(kernel_dispatch_test, descriptor_uses_default_buffer_properties)
{
    constexpr const auto& k_domain = k_kernel_dispatch<mock_sdk, externals>;

    EXPECT_EQ(k_domain.buffer.buffer_size, k_default_buffer_properties.buffer_size);
    EXPECT_EQ(k_domain.buffer.buffer_watermark,
              k_default_buffer_properties.buffer_watermark);
}

TEST(kernel_dispatch_test, on_kernel_dispatch_handles_null_record_without_crashing)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();
    g_buffer_storage_mock    = std::make_unique<StrictMock<gmock_buffer_storage>>();

    EXPECT_CALL(*g_metadata_registry_mock, add_thread_info).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_track).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_queue).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_stream).Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, store).Times(1);

    mock_sdk::kernel_dispatch_record_t record{};

    on_kernel_dispatch<mock_sdk, externals>(&record, nullptr);

    g_metadata_registry_mock.reset();
    g_buffer_storage_mock.reset();
}

TEST(kernel_dispatch_test, on_kernel_dispatch_forwards_record_fields_to_dependencies)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();
    g_buffer_storage_mock    = std::make_unique<StrictMock<gmock_buffer_storage>>();

    mock_sdk::kernel_dispatch_record_t record{};
    record.thread_id                          = 111;
    record.start_timestamp                    = 1000;
    record.end_timestamp                      = 2000;
    record.correlation_id.internal            = 222;
    record.dispatch_info.agent_id.handle      = 333;
    record.dispatch_info.kernel_id            = 444;
    record.dispatch_info.dispatch_id          = 555;
    record.dispatch_info.queue_id.handle      = 666;
    record.dispatch_info.private_segment_size = 7;
    record.dispatch_info.group_segment_size   = 8;
    record.dispatch_info.workgroup_size       = { 1, 2, 3 };
    record.dispatch_info.grid_size            = { 4, 5, 6 };

    // Every derived value the mocks can't be steered to produce
    // (device_id/stream_id/parent_stack_id) is fixed at 0 by the test doubles: see
    // externals::agent_manager_t::get_agent_by_handle, mock_sdk::get_stream_id, and
    // mock_sdk::get_parent_stack_id in mock_domain_service.hpp.
    constexpr std::uint32_t k_mock_device_id       = 0;
    constexpr std::uint64_t k_mock_stream_id       = 0;
    constexpr std::uint64_t k_mock_parent_stack_id = 0;

    const auto expected_thread_info =
        thread_info_data_t{ 0, 0, record.thread_id, 0, 0, "{}" };
    const auto expected_track =
        track_data_t{ fmt::format("GPU Kernel Dispatch [{}] Queue {}", k_mock_device_id,
                                  record.dispatch_info.queue_id.handle),
                      record.thread_id, "{}" };
    const auto expected_sample =
        kernel_dispatch_sample_data_t{ record.start_timestamp,
                                       record.end_timestamp,
                                       record.thread_id,
                                       record.dispatch_info.agent_id.handle,
                                       record.dispatch_info.kernel_id,
                                       record.dispatch_info.dispatch_id,
                                       record.dispatch_info.queue_id.handle,
                                       record.correlation_id.internal,
                                       k_mock_parent_stack_id,
                                       record.dispatch_info.private_segment_size,
                                       record.dispatch_info.group_segment_size,
                                       record.dispatch_info.workgroup_size.x,
                                       record.dispatch_info.workgroup_size.y,
                                       record.dispatch_info.workgroup_size.z,
                                       record.dispatch_info.grid_size.x,
                                       record.dispatch_info.grid_size.y,
                                       record.dispatch_info.grid_size.z,
                                       k_mock_stream_id };

    EXPECT_CALL(*g_metadata_registry_mock, add_thread_info(Eq(expected_thread_info)))
        .Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_track(Eq(expected_track))).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock,
                add_queue(Eq(record.dispatch_info.queue_id.handle)))
        .Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_stream(Eq(k_mock_stream_id))).Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, store(Eq(expected_sample))).Times(1);

    on_kernel_dispatch<mock_sdk, externals>(&record, nullptr);

    g_metadata_registry_mock.reset();
    g_buffer_storage_mock.reset();
}

}  // namespace rocprofsys::domains::buffered
