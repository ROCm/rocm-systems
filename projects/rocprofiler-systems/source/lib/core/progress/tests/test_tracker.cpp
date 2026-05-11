// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/progress/tracker.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

namespace rocprofsys::progress
{

TEST(tracker_test, begin_with_factory_invokes_factory_with_label_and_total)
{
    std::string   recorded_label;
    std::uint64_t recorded_total = 0;

    tracker t{ [&recorded_label, &recorded_total](std::string   label,
                                                  std::uint64_t total) {
        recorded_label = std::move(label);
        recorded_total = total;
        return progress_callback{};
    } };

    (void) t.begin("perfetto: pid_1234.bin", 4096);

    EXPECT_EQ(recorded_label, "perfetto: pid_1234.bin");
    EXPECT_EQ(recorded_total, 4096U);
}

TEST(tracker_test, begin_returned_callback_advances_captured_renderer)
{
    struct recording_renderer
    {
        std::uint64_t total_advanced = 0;
        void on_advance(std::uint64_t delta) noexcept { total_advanced += delta; }
    };

    auto    renderer = std::make_shared<recording_renderer>();
    tracker t{ [renderer](std::string, std::uint64_t) {
        return progress_callback{ [renderer](std::uint64_t delta) {
            renderer->on_advance(delta);
        } };
    } };

    auto cb = t.begin("rocpd", 1000);
    cb(100);
    cb(250);
    cb(50);

    EXPECT_EQ(renderer->total_advanced, 400U);
}

TEST(tracker_test, begin_with_empty_factory_returns_empty_callback)
{
    tracker t{ tracker::factory_t{} };

    auto cb = t.begin("anything", 1234);

    EXPECT_FALSE(static_cast<bool>(cb));
}

TEST(tracker_test, callback_outlives_call_site_and_still_advances_renderer)
{
    struct recording_renderer
    {
        std::uint64_t total_advanced = 0;
        void on_advance(std::uint64_t delta) noexcept { total_advanced += delta; }
    };

    auto              renderer = std::make_shared<recording_renderer>();
    progress_callback escaped_cb;

    {
        tracker t{ [renderer](std::string, std::uint64_t) {
            return progress_callback{ [renderer](std::uint64_t delta) {
                renderer->on_advance(delta);
            } };
        } };

        escaped_cb = t.begin("scoped", 1000);
        // tracker goes out of scope here; the renderer must still be reachable
        // through escaped_cb's captured shared_ptr.
    }

    escaped_cb(42);
    escaped_cb(58);

    EXPECT_EQ(renderer->total_advanced, 100U);
}

}  // namespace rocprofsys::progress
