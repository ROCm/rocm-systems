// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Concurrency stress test for sampling_service operations that are safe
// to call from multiple threads: block_samples/unblock_samples (atomic),
// pause/resume (mutex-protected), and is_blocked/is_paused (read-only).
//
// setup(tid) and shutdown(tid) are per-thread operations that set TLS
// pointers — they must be called from the thread they configure. These
// are tested sequentially in the full_pipeline and fork tests.

#include <gtest/gtest.h>

#include "doubles/test_sampling_policies.hpp"
#include "sampling/sampling_service.hpp"

#include <atomic>
#include <thread>
#include <vector>

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;

TEST(concurrent_lifecycle, rapid_pause_resume)
{
    constexpr int num_threads = 4;
    constexpr int iterations  = 100;
    test_service  svc{ make_test_config(), make_test_callbacks() };

    std::atomic<bool>        go{ false };
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for(int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([&svc, &go]() {
            while(!go.load(std::memory_order_acquire))
                std::this_thread::yield();
            for(int j = 0; j < iterations; ++j)
            {
                svc.pause();
                svc.resume();
            }
        });
    }

    go.store(true, std::memory_order_release);

    for(auto& thr : threads)
        thr.join();

    SUCCEED();
}

TEST(concurrent_lifecycle, block_unblock_under_contention)
{
    constexpr int num_threads = 4;
    constexpr int iterations  = 200;
    test_service  svc{ make_test_config(), make_test_callbacks() };

    std::atomic<bool>        go{ false };
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for(int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([&svc, &go]() {
            while(!go.load(std::memory_order_acquire))
                std::this_thread::yield();
            for(int j = 0; j < iterations; ++j)
            {
                svc.block_samples();
                svc.unblock_samples();
            }
        });
    }

    go.store(true, std::memory_order_release);

    for(auto& thr : threads)
        thr.join();

    EXPECT_FALSE(svc.is_blocked());
}

TEST(concurrent_lifecycle, mixed_block_and_pause_contention)
{
    constexpr int iterations = 100;
    test_service  svc{ make_test_config(), make_test_callbacks() };

    std::atomic<bool> go{ false };

    std::thread blocker([&svc, &go]() {
        while(!go.load(std::memory_order_acquire))
            std::this_thread::yield();
        for(int j = 0; j < iterations; ++j)
        {
            svc.block_samples();
            svc.unblock_samples();
        }
    });

    std::thread pauser([&svc, &go]() {
        while(!go.load(std::memory_order_acquire))
            std::this_thread::yield();
        for(int j = 0; j < iterations; ++j)
        {
            svc.pause();
            svc.resume();
        }
    });

    go.store(true, std::memory_order_release);

    blocker.join();
    pauser.join();

    EXPECT_FALSE(svc.is_blocked());
}
