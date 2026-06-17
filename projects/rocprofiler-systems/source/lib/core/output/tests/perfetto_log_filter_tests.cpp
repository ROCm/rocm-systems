// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output/perfetto_log_filter.hpp"
#include "core/output/perfetto_log_filter_detail.hpp"

#include <perfetto.h>

namespace pblog  = ::perfetto::base;
namespace filter = rocprofsys::output::perfetto_log_filter;

TEST(perfetto_log_filter, classify_drops_debug_and_info)
{
    EXPECT_EQ(filter::classify(pblog::kLogDebug), filter::filter_action::drop);
    EXPECT_EQ(filter::classify(pblog::kLogInfo), filter::filter_action::drop);
}

TEST(perfetto_log_filter, classify_warning_for_important)
{
    EXPECT_EQ(filter::classify(pblog::kLogImportant), filter::filter_action::warning);
}

TEST(perfetto_log_filter, classify_error_for_error)
{
    EXPECT_EQ(filter::classify(pblog::kLogError), filter::filter_action::error);
}

TEST(perfetto_log_filter, classify_unknown_for_out_of_range)
{
    // Cast through the enum's underlying type to construct a value
    // outside the defined cases — exercises the fallback path so
    // future SDK enum additions render with "unknown severity" rather
    // than being silently dropped.
    const auto unknown_level = static_cast<pblog::LogLev>(255);
    EXPECT_EQ(filter::classify(unknown_level), filter::filter_action::unknown);
}

TEST(perfetto_log_filter, register_with_perfetto_logger_is_idempotent)
{
    // Double-register must not crash; std::call_once guards the
    // underlying SetLogMessageCallback so the second call is a no-op.
    filter::register_with_perfetto_logger();
    filter::register_with_perfetto_logger();
    SUCCEED();
}

TEST(perfetto_log_filter, unregister_is_safe_with_and_without_prior_registration)
{
    // unregister clears perfetto's callback so a late worker-thread log
    // cannot reach the destroyed spdlog logger. It must be safe to call
    // repeatedly and without a preceding registration.
    filter::unregister_from_perfetto_logger();
    filter::register_with_perfetto_logger();
    filter::unregister_from_perfetto_logger();
    filter::unregister_from_perfetto_logger();
    SUCCEED();
}
