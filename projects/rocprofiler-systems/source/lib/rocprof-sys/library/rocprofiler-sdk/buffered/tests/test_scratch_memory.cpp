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
#include <string_view>

namespace rocprofsys::domains::buffered
{
namespace
{

using ::testing::Eq;
using ::testing::Return;
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

    const auto expected_thread_info = thread_info_data_t{ .parent_process_id = 0,
                                                          .process_id        = 0,
                                                          .thread_id = record.thread_id,
                                                          .start     = 0,
                                                          .end       = 0,
                                                          .extdata   = "{}" };
    const auto expected_track =
        track_data_t{ .track_name = fmt::format("GPU Scratch Memory [{}] Thread {}",
                                                k_mock_device_id, record.thread_id),
                      .thread_id  = record.thread_id,
                      .extdata    = "{}" };
    const auto expected_sample = scratch_memory_sample_data_t{
        .start_timestamp         = record.start_timestamp,
        .end_timestamp           = record.end_timestamp,
        .thread_id               = record.thread_id,
        .agent_id_handle         = record.agent_id.handle,
        .queue_id_handle         = record.queue_id.handle,
        .kind                    = static_cast<std::int32_t>(record.kind),
        .operation               = static_cast<std::int32_t>(record.operation),
        .flags                   = static_cast<std::int32_t>(record.flags),
        .allocation_size         = record.allocation_size,
        .correlation_id_internal = record.correlation_id.internal,
        .correlation_id_ancestor = std::uint64_t{ 0 },
        .stream_handle           = k_mock_stream_id
    };

    EXPECT_CALL(*g_metadata_registry_mock, add_thread_info(Eq(expected_thread_info)))
        .Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_track(Eq(expected_track))).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_queue(Eq(record.queue_id.handle)))
        .Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_stream(Eq(k_mock_stream_id))).Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, store_scratch_memory(Eq(expected_sample)))
        .Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, get_use_timemory).WillOnce(Return(false));

    on_scratch_memory<mock_sdk, externals>(&record, nullptr);

    g_metadata_registry_mock.reset();
    g_buffer_storage_mock.reset();
}

TEST(scratch_memory_test, on_scratch_memory_drops_record_with_unknown_agent)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();
    g_buffer_storage_mock    = std::make_unique<StrictMock<gmock_buffer_storage>>();

    mock_sdk::scratch_memory_record_t record{};
    record.agent_id.handle = externals::agent_manager_t::k_unknown_agent_handle;

    EXPECT_NO_THROW((on_scratch_memory<mock_sdk, externals>(&record, nullptr)));

    g_metadata_registry_mock.reset();
    g_buffer_storage_mock.reset();
}

TEST(scratch_memory_test, on_scratch_memory_writes_timemory_bundle_when_enabled)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();
    g_buffer_storage_mock    = std::make_unique<StrictMock<gmock_buffer_storage>>();

    mock_sdk::scratch_memory_record_t record{};
    record.thread_id       = 111;
    record.start_timestamp = 1000;
    record.end_timestamp   = 2500;

    // The tracing-name table and get_thread_info_sequent_tid are fixed by the test
    // doubles: see tracing_names_t::at and externals::get_thread_info_sequent_tid.
    constexpr std::string_view k_mock_name        = "operation";
    constexpr std::uint64_t    k_mock_sequent_tid = 0;
    const std::uint64_t        expected_elapsed_ns =
        record.end_timestamp - record.start_timestamp;

    EXPECT_CALL(*g_metadata_registry_mock, add_thread_info).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_track).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_queue).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_stream).Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, store_scratch_memory).Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, get_use_timemory).WillOnce(Return(true));
    EXPECT_CALL(*g_buffer_storage_mock,
                write_timemory_bundle(Eq(k_mock_name), Eq(k_mock_sequent_tid),
                                      Eq(expected_elapsed_ns)))
        .Times(1);

    on_scratch_memory<mock_sdk, externals>(&record, nullptr);

    g_metadata_registry_mock.reset();
    g_buffer_storage_mock.reset();
}

TEST(scratch_memory_test, on_scratch_memory_skips_timemory_bundle_for_unknown_thread)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();
    g_buffer_storage_mock    = std::make_unique<StrictMock<gmock_buffer_storage>>();

    mock_sdk::scratch_memory_record_t record{};
    record.thread_id = externals::k_unknown_tid;

    EXPECT_CALL(*g_metadata_registry_mock, add_thread_info).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_track).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_queue).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_stream).Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, store_scratch_memory).Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, get_use_timemory).WillOnce(Return(true));

    EXPECT_NO_THROW((on_scratch_memory<mock_sdk, externals>(&record, nullptr)));

    g_metadata_registry_mock.reset();
    g_buffer_storage_mock.reset();
}

}  // namespace rocprofsys::domains::buffered
