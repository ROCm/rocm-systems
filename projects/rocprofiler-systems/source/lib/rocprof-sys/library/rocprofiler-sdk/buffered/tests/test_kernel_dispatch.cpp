// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/buffered/kernel_dispatch.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <gtest/gtest.h>

namespace rocprofsys::domains::buffered
{
namespace
{

using test_support::externals;
using test_support::mock_sdk;

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
    mock_sdk::kernel_dispatch_record_t record{};

    on_kernel_dispatch<mock_sdk, externals>(&record, nullptr);
}

}  // namespace rocprofsys::domains::buffered
