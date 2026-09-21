// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered/memory_copy.hpp"
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
using test_support::memory_copy_sample_data_t;
using test_support::mock_sdk;
using test_support::thread_info_data_t;
using test_support::track_data_t;

}  // namespace

TEST(memory_copy_test, descriptor_reports_correct_metadata)
{
    using mock_dispatcher =
        buffered_callback_dispatcher<mock_sdk, mock_sdk::memory_copy_record_t,
                                     on_memory_copy<mock_sdk, externals>>;
    constexpr const auto& k_domain = k_memory_copy<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "memory_copy");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::BUFFER_TRACING_MEMORY_COPY);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::buffered);
    EXPECT_FALSE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.on_records, &mock_dispatcher::callback);
}

TEST(memory_copy_test, descriptor_uses_default_buffer_properties)
{
    constexpr const auto& k_domain = k_memory_copy<mock_sdk, externals>;

    EXPECT_EQ(k_domain.buffer.buffer_size, k_default_buffer_properties.buffer_size);
    EXPECT_EQ(k_domain.buffer.buffer_watermark,
              k_default_buffer_properties.buffer_watermark);
}

TEST(memory_copy_test, on_memory_copy_handles_null_record_without_crashing)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();
    g_buffer_storage_mock    = std::make_unique<StrictMock<gmock_buffer_storage>>();

    EXPECT_CALL(*g_metadata_registry_mock, add_thread_info).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_track).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_stream).Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, store_memory_copy).Times(1);

    mock_sdk::memory_copy_record_t record{};

    on_memory_copy<mock_sdk, externals>(&record, nullptr);

    g_metadata_registry_mock.reset();
    g_buffer_storage_mock.reset();
}

TEST(memory_copy_test, on_memory_copy_forwards_record_fields_to_dependencies)
{
    g_metadata_registry_mock = std::make_unique<StrictMock<gmock_metadata_registry>>();
    g_buffer_storage_mock    = std::make_unique<StrictMock<gmock_buffer_storage>>();

    mock_sdk::memory_copy_record_t record{};
    record.thread_id               = 111;
    record.start_timestamp         = 1000;
    record.end_timestamp           = 2000;
    record.correlation_id.internal = 222;
    record.dst_agent_id.handle     = 333;
    record.src_agent_id.handle     = 444;
    record.kind                    = 1;
    record.operation               = 2;
    record.bytes                   = 4096;

    // Every derived value the mocks can't be steered to produce (logical_node_id,
    // stream_id, parent_stack_id, dst/src address) is fixed at 0 by the test doubles:
    // see externals::agent_manager_t::get_agent_by_handle, mock_sdk::get_stream_id,
    // mock_sdk::get_parent_stack_id, and mock_sdk::get_memory_copy_{dst,src}_address
    // in mock_domain_service.hpp.
    constexpr std::int32_t  k_mock_logical_node_id = 0;
    constexpr std::uint64_t k_mock_stream_id       = 0;
    constexpr std::uint64_t k_mock_parent_stack_id = 0;
    constexpr std::uint64_t k_mock_address         = 0;

    const auto expected_thread_info =
        thread_info_data_t{ 0, 0, record.thread_id, 0, 0, "{}" };
    const auto expected_track =
        track_data_t{ fmt::format("GPU Memory Copy to Agent [{}] Thread {}",
                                  k_mock_logical_node_id, record.thread_id),
                      record.thread_id, "{}" };
    const auto expected_sample =
        memory_copy_sample_data_t{ record.start_timestamp,
                                   record.end_timestamp,
                                   record.thread_id,
                                   record.dst_agent_id.handle,
                                   record.src_agent_id.handle,
                                   static_cast<std::int32_t>(record.kind),
                                   static_cast<std::int32_t>(record.operation),
                                   record.bytes,
                                   record.correlation_id.internal,
                                   k_mock_parent_stack_id,
                                   k_mock_address,
                                   k_mock_address,
                                   k_mock_stream_id };

    EXPECT_CALL(*g_metadata_registry_mock, add_thread_info(Eq(expected_thread_info)))
        .Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_track(Eq(expected_track))).Times(1);
    EXPECT_CALL(*g_metadata_registry_mock, add_stream(Eq(k_mock_stream_id))).Times(1);
    EXPECT_CALL(*g_buffer_storage_mock, store_memory_copy(Eq(expected_sample))).Times(1);

    on_memory_copy<mock_sdk, externals>(&record, nullptr);

    g_metadata_registry_mock.reset();
    g_buffer_storage_mock.reset();
}

}  // namespace rocprofsys::domains::buffered
