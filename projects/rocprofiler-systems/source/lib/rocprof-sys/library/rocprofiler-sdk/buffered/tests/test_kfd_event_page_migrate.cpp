// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered/kfd_event_page_migrate.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace rocprofsys::domains::buffered
{
namespace
{

// Self-contained stand-in for SdkBackend: kfd_event_page_migrate<SdkBackend, Externals>
// and on_kfd_event_page_migrate<SdkBackend, Externals> only ever touch these members.
struct mock_sdk
{
    struct context_id_t
    {
        std::uint64_t handle = 0;
    };
    struct buffer_id_t
    {
        std::uint64_t handle = 0;
    };
    struct record_header_t
    {
        std::uint32_t category = 0;
        std::uint32_t kind     = 0;
        void*         payload  = nullptr;
    };

    static constexpr std::size_t BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE = 23;

    struct kfd_event_page_migrate_record
    {};
};

// Externals is unused by this domain today; any type satisfies the template parameter.
struct externals
{};

using mock_dispatcher =
    buffered_callback_dispatcher<mock_sdk, mock_sdk::kfd_event_page_migrate_record,
                                 on_kfd_event_page_migrate<mock_sdk, externals>>;

}  // namespace

TEST(kfd_event_page_migrate_test, descriptor_reports_correct_metadata)
{
    constexpr const auto& domain = k_kfd_event_page_migrate<mock_sdk, externals>;

    EXPECT_EQ(domain.meta.name, "kfd_event_page_migrate");
    EXPECT_EQ(domain.meta.id, mock_sdk::BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE);
    EXPECT_EQ(domain.meta.mode, collection_mode::buffered);
    ASSERT_TRUE(domain.meta.group.has_value());
    EXPECT_EQ(domain.meta.group->name, "kfd_events");
    EXPECT_EQ(domain.on_records, &mock_dispatcher::callback);
}

TEST(kfd_event_page_migrate_test, descriptor_uses_default_buffer_properties)
{
    constexpr const auto& domain = k_kfd_event_page_migrate<mock_sdk, externals>;

    EXPECT_EQ(domain.buffer.buffer_size, k_default_buffer_properties.buffer_size);
    EXPECT_EQ(domain.buffer.buffer_watermark,
              k_default_buffer_properties.buffer_watermark);
}

TEST(kfd_event_page_migrate_test,
     on_kfd_event_page_migrate_handles_empty_record_batch_without_crashing)
{
    mock_sdk::kfd_event_page_migrate_record record{};

    on_kfd_event_page_migrate<mock_sdk, externals>(&record, nullptr);
}

}  // namespace rocprofsys::domains::buffered
