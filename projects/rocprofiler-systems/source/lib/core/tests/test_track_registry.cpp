// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <cstdint>

#include "core/track_registry.hpp"

#include <atomic>
#include <thread>
#include <vector>

namespace
{
// RAII helper: records the previous active registry on construction and
// restores it on destruction. Prevents test-to-test pollution of the
// thread-local active pointer.
class scoped_active_registry
{
public:
    explicit scoped_active_registry(rocprofsys::track_registry* registry) noexcept
    : m_previous(rocprofsys::get_active_track_registry())
    {
        rocprofsys::set_active_track_registry(registry);
    }

    ~scoped_active_registry() noexcept
    {
        rocprofsys::set_active_track_registry(m_previous);
    }

    scoped_active_registry(const scoped_active_registry&)            = delete;
    scoped_active_registry& operator=(const scoped_active_registry&) = delete;

private:
    rocprofsys::track_registry* m_previous;
};
}  // namespace

TEST(track_registry, default_state_is_empty)
{
    rocprofsys::track_registry reg;
    EXPECT_TRUE(reg.map().empty());
}

TEST(track_registry, two_instances_no_shared_state)
{
    rocprofsys::track_registry first;
    rocprofsys::track_registry second;

    {
        std::lock_guard<std::mutex> _lk{ first.mutex() };
        first.map().emplace(std::uint64_t{ 0xABC }, std::string{ "first-track" });
    }

    EXPECT_EQ(first.map().size(), 1u);
    EXPECT_TRUE(second.map().empty()) << "second registry must not see first's mutation";
}

TEST(track_registry, set_active_round_trip)
{
    rocprofsys::track_registry reg;
    scoped_active_registry     guard{ &reg };

    EXPECT_EQ(rocprofsys::get_active_track_registry(), &reg);
}

TEST(track_registry, set_active_null_clears)
{
    rocprofsys::track_registry reg;
    scoped_active_registry     guard{ &reg };
    EXPECT_EQ(rocprofsys::get_active_track_registry(), &reg);

    rocprofsys::set_active_track_registry(nullptr);
    EXPECT_EQ(rocprofsys::get_active_track_registry(), nullptr);
}

TEST(track_registry, active_is_thread_local)
{
    rocprofsys::track_registry main_reg;
    scoped_active_registry     guard{ &main_reg };
    EXPECT_EQ(rocprofsys::get_active_track_registry(), &main_reg);

    rocprofsys::track_registry* other_thread_view = nullptr;
    std::thread                 other{ [&other_thread_view]() {
        // Active pointer is thread-local; a fresh thread starts with nullptr.
        other_thread_view = rocprofsys::get_active_track_registry();
    } };
    other.join();

    EXPECT_EQ(other_thread_view, nullptr)
        << "active registry must not leak across threads";
    EXPECT_EQ(rocprofsys::get_active_track_registry(), &main_reg)
        << "main thread's active registry must be unchanged by other thread";
}

TEST(track_registry, concurrent_emplace_under_mutex_is_race_free)
{
    rocprofsys::track_registry reg;

    constexpr int            thread_count = 8;
    constexpr int            inserts_each = 256;
    std::atomic<int>         ready_counter{ 0 };
    std::atomic<bool>        go{ false };
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for(int t = 0; t < thread_count; ++t)
    {
        threads.emplace_back([&, t]() {
            ++ready_counter;
            while(!go.load(std::memory_order_acquire))
            {
            }
            for(int i = 0; i < inserts_each; ++i)
            {
                const auto uuid = static_cast<std::uint64_t>(t) * inserts_each +
                                  static_cast<std::uint64_t>(i);
                std::lock_guard<std::mutex> _lk{ reg.mutex() };
                reg.map().emplace(uuid, std::string{ "track-" } + std::to_string(uuid));
            }
        });
    }

    while(ready_counter.load() < thread_count)
    {
    }
    go.store(true, std::memory_order_release);
    for(auto& th : threads)
        th.join();

    EXPECT_EQ(reg.map().size(), static_cast<size_t>(thread_count * inserts_each));
}

TEST(track_registry, scoped_guard_restores_previous_active)
{
    rocprofsys::track_registry outer;
    rocprofsys::track_registry inner;

    rocprofsys::set_active_track_registry(&outer);
    ASSERT_EQ(rocprofsys::get_active_track_registry(), &outer);

    {
        scoped_active_registry guard{ &inner };
        EXPECT_EQ(rocprofsys::get_active_track_registry(), &inner);
    }

    EXPECT_EQ(rocprofsys::get_active_track_registry(), &outer);

    // Tear down so the test does not pollute subsequent test cases on this
    // thread.
    rocprofsys::set_active_track_registry(nullptr);
}
