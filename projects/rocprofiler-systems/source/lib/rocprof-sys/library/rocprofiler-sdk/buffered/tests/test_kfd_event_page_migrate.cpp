// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered/kfd_event_page_migrate.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gtest/gtest.h>

namespace rocprofsys::domains::buffered
{
namespace
{

using test_support::externals;
using test_support::mock_sdk;

using mock_dispatcher =
    buffered_callback_dispatcher<mock_sdk, mock_sdk::kfd_event_page_migrate_record,
                                 on_kfd_event_page_migrate<mock_sdk, externals>>;

}  // namespace

TEST(kfd_event_page_migrate_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& k_domain = k_kfd_event_page_migrate<mock_sdk, externals>;

    EXPECT_EQ(k_domain.meta.name, "kfd_event_page_migrate");
    EXPECT_EQ(k_domain.meta.id, mock_sdk::BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE);
    EXPECT_EQ(k_domain.meta.mode, collection_mode::buffered);
    ASSERT_TRUE(k_domain.meta.group.has_value());
    EXPECT_EQ(k_domain.meta.group.value().name, "kfd_events");
    EXPECT_EQ(k_domain.on_records, &mock_dispatcher::callback);
}

TEST(kfd_event_page_migrate_test, descriptor_uses_default_buffer_properties)
{
    constexpr const auto& k_domain = k_kfd_event_page_migrate<mock_sdk, externals>;

    EXPECT_EQ(k_domain.buffer.buffer_size, k_default_buffer_properties.buffer_size);
    EXPECT_EQ(k_domain.buffer.buffer_watermark,
              k_default_buffer_properties.buffer_watermark);
}

TEST(kfd_event_page_migrate_test,
     on_kfd_event_page_migrate_handles_empty_record_batch_without_crashing)
{
    mock_sdk::kfd_event_page_migrate_record record{};

    on_kfd_event_page_migrate<mock_sdk, externals>(&record, nullptr);
}

}  // namespace rocprofsys::domains::buffered
