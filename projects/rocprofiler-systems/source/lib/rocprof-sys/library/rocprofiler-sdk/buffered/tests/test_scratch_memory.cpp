// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered/scratch_memory.hpp"
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
using test_support::mock_sdk;
using test_support::scratch_memory_sample_data_t;
using test_support::thread_info_data_t;
using test_support::track_data_t;

}  // namespace

TEST(scratch_memory_test, descriptor_reports_correct_metadata)
{
    using mock_dispatcher =
        buffered_callback_dispatcher<mock_sdk, mock_sdk::scratch_memory_record_t,
                                     on_scratch_memory<mock_sdk, externals>>;
    constexpr const auto& k_domain = k_scratch_memory<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "scratch_memory");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::BUFFER_TRACING_SCRATCH_MEMORY);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::buffered);
    EXPECT_FALSE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.on_records, &mock_dispatcher::callback);
}

TEST(scratch_memory_test, descriptor_uses_default_buffer_properties)
{
    constexpr const auto& k_domain = k_scratch_memory<mock_sdk, externals>;

    EXPECT_EQ(k_domain.buffer.buffer_size, k_default_buffer_properties.buffer_size);
    EXPECT_EQ(k_domain.buffer.buffer_watermark,
              k_default_buffer_properties.buffer_watermark);
}

TEST(scratch_memory_test, on_scratch_memory_configure_adds_category_string)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();

    EXPECT_CALL(*g_metadata_registry_mock,
                add_string(Eq(externals::scratch_memory_category_name)))
        .Times(1);

    on_scratch_memory_configure<externals>();

    g_metadata_registry_mock.reset();
}

TEST(scratch_memory_test, on_scratch_memory_handles_null_record_without_crashing)
{
    on_scratch_memory<mock_sdk, externals>(nullptr, nullptr);
}

TEST(scratch_memory_test, on_scratch_memory_forwards_record_fields_to_dependencies)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();
    g_buffer_storage_mock    = std::make_unique<StrictMock<gmock_buffer_storage>>();

    mock_sdk::scratch_memory_record_t record{};
    record.thread_id               = 111;
    record.start_timestamp         = 1000;
    record.end_timestamp           = 2000;
    record.correlation_id.internal = 222;
    record.agent_id.handle         = 333;
    record.queue_id.handle         = 444;
    record.kind                    = 1;
    record.operation               = 2;
    record.flags                   = 3;
    record.allocation_size         = 4096;

    // The stream_id/device_id are fixed at 0 by the test doubles: see
    // mock_sdk::get_stream_id and agent_manager_t::get_agent_by_handle in
    // mock_domain_service.hpp.
    constexpr std::uint64_t k_mock_stream_id = 0;
    constexpr std::uint32_t k_mock_device_id = 0;

    const auto expected_thread_info =
        thread_info_data_t{ 0, 0, record.thread_id, 0, 0, "{}" };
    const auto expected_track =
        track_data_t{ fmt::format("GPU Scratch Memory [{}] Thread {}", k_mock_device_id,
                                  record.thread_id),
                      record.thread_id, "{}" };
    const auto expected_sample =
        scratch_memory_sample_data_t{ record.start_timestamp,
                                      record.end_timestamp,
                                      record.thread_id,
                                      record.agent_id.handle,
                                      record.queue_id.handle,
                                      static_cast<std::int32_t>(record.kind),
                                      static_cast<std::int32_t>(record.operation),
                                      static_cast<std::int32_t>(record.flags),
                                      record.allocation_size,
                                      record.correlation_id.internal,
                                      std::uint64_t{ 0 },
                                      k_mock_stream_id };

    EXPECT_CALL(*g_metadata_registry_mock, add_thread_info(Eq(expected_thread_info)))
        .Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_track(Eq(expected_track))).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_queue(Eq(record.queue_id.handle)))
        .Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_stream(Eq(k_mock_stream_id))).Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, store_scratch_memory(Eq(expected_sample)))
        .Times(1);

    on_scratch_memory<mock_sdk, externals>(&record, nullptr);

    g_metadata_registry_mock.reset();
    g_buffer_storage_mock.reset();
}

}  // namespace rocprofsys::domains::buffered
