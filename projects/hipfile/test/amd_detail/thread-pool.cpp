/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-warnings.h"
#include "thread-pool.h"

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

using namespace hipFile;
using namespace std::chrono_literals;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

TEST(HipFileThreadPool, RunExecutesWork)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> ran{false};

    group->run([&ran]() { ran = true; });
    group->wait();

    ASSERT_TRUE(ran);
}

TEST(HipFileThreadPool, WaitBlocksUntilRunningWorkCompletes)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> gate{false};
    std::atomic<bool> started{false};

    group->run([&gate, &started]() {
        started.store(true);
        started.notify_all();
        gate.wait(false);
    });
    started.wait(false);

    auto wait_result = std::async(std::launch::async, [&group]() { group->wait(); });
    ASSERT_EQ(wait_result.wait_for(20ms), std::future_status::timeout);

    gate.store(true);
    gate.notify_all();
    ASSERT_EQ(wait_result.wait_for(1s), std::future_status::ready);
}

TEST(HipFileThreadPool, CancelSkipsPendingWork)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> gate{false};
    std::atomic<bool> started{false};
    std::atomic<bool> pending_ran{false};

    group->run([&gate, &started]() {
        started.store(true);
        started.notify_all();
        gate.wait(false);
    });
    started.wait(false);

    group->run([&pending_ran]() { pending_ran = true; });
    group->cancel();
    gate.store(true);
    gate.notify_all();
    group->wait();

    ASSERT_FALSE(pending_ran);
}

TEST(HipFileThreadPool, CancelDoesNotStopRunningWork)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> gate{false};
    std::atomic<bool> started{false};
    std::atomic<bool> completed{false};

    group->run([&gate, &started, &completed]() {
        started.store(true);
        started.notify_all();
        gate.wait(false);
        completed = true;
    });
    started.wait(false);

    group->cancel();
    gate.store(true);
    gate.notify_all();
    group->wait();

    ASSERT_TRUE(completed);
}

TEST(HipFileThreadPool, CanSubmitAfterCancel)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> ran{false};

    group->cancel();
    group->run([&ran]() { ran = true; });
    group->wait();

    ASSERT_TRUE(ran);
}

TEST(HipFileThreadPool, DestructorCancelsAndWaits)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> gate{false};
    std::atomic<bool> started{false};
    std::atomic<bool> pending_ran{false};

    group->run([&gate, &started]() {
        started.store(true);
        started.notify_all();
        gate.wait(false);
    });
    started.wait(false);

    group->run([&pending_ran]() { pending_ran = true; });

    auto destroy_result =
        std::async(std::launch::async, [owned_group = std::move(group)]() mutable { owned_group.reset(); });
    ASSERT_EQ(destroy_result.wait_for(20ms), std::future_status::timeout);

    gate.store(true);
    gate.notify_all();
    ASSERT_EQ(destroy_result.wait_for(1s), std::future_status::ready);
    ASSERT_FALSE(pending_ran);
}

// Repeatedly publish captures to active workers and wait before reading their
// results. This also exercises task-group state teardown immediately after the
// last completion, which must finish notifying waiters before releasing state.
TEST(HipFileThreadPool, RepeatedPublicationAndWait)
{
    ThreadPool pool{4};
    for (int iteration = 0; iteration < 64; ++iteration) {
        SCOPED_TRACE(iteration);
        std::array<int, 32> results{};
        auto                group = pool.makeTaskGroup();
        for (size_t index = 0; index < results.size(); ++index) {
            const auto expected = static_cast<int>(index) + iteration + 1;
            group->run([&results, index, expected]() { results[index] = expected; });
        }
        group->wait();
        group.reset();
        for (size_t index = 0; index < results.size(); ++index) {
            EXPECT_EQ(results[index], static_cast<int>(index) + iteration + 1);
        }
    }
}

// Keep one task running while cancelling queued work, then destroy the group
// as the running task completes. Reuse the pool to exercise task-node reuse.
TEST(HipFileThreadPool, RepeatedCancelAndDestroy)
{
    ThreadPool pool{1};
    for (int iteration = 0; iteration < 64; ++iteration) {
        SCOPED_TRACE(iteration);
        std::atomic<bool> started{false};
        std::atomic<bool> release{false};
        std::atomic<bool> completed{false};
        std::atomic<int>  pending_count{0};
        auto              group = pool.makeTaskGroup();
        group->run([&]() {
            started.store(true);
            started.notify_all();
            release.wait(false);
            completed.store(true);
        });
        started.wait(false);
        for (int index = 0; index < 32; ++index) {
            group->run([&]() { pending_count.fetch_add(1); });
        }
        group->cancel();
        release.store(true);
        release.notify_all();
        group.reset();
        EXPECT_TRUE(completed.load());
        EXPECT_EQ(pending_count.load(), 0);
    }
}

// Submissions from a Taskflow worker use a different queue from submissions
// from external threads. Keep the submitting worker occupied until all children
// finish so another worker must acquire their published captures.
TEST(HipFileThreadPool, WorkerPublishesToOtherWorkers)
{
    ThreadPool pool{4};
    for (int iteration = 0; iteration < 32; ++iteration) {
        SCOPED_TRACE(iteration);
        std::array<int, 32> results{};
        std::atomic<int>    remaining{32};
        std::atomic<bool>   other_worker{false};
        auto                group = pool.makeTaskGroup();
        group->run([&]() {
            const auto publisher = std::this_thread::get_id();
            for (size_t index = 0; index < results.size(); ++index) {
                const int expected = iteration + static_cast<int>(index) + 1;
                group->run([&, index, expected, publisher]() {
                    results[index] = expected;
                    if (std::this_thread::get_id() != publisher) {
                        other_worker.store(true);
                    }
                    if (remaining.fetch_sub(1) == 1) {
                        remaining.notify_all();
                    }
                });
            }
            for (int count = remaining.load(); count != 0; count = remaining.load()) {
                remaining.wait(count);
            }
        });
        group->wait();
        EXPECT_TRUE(other_worker.load());
        for (size_t index = 0; index < results.size(); ++index) {
            EXPECT_EQ(results[index], iteration + static_cast<int>(index) + 1);
        }
    }
}

// Different external submitters and task groups share an executor's spill
// queues. Every task writes a distinct result and waits finish before reads.
TEST(HipFileThreadPool, ConcurrentExternalSubmitters)
{
    ThreadPool pool{4};
    for (int iteration = 0; iteration < 16; ++iteration) {
        SCOPED_TRACE(iteration);
        std::array<std::array<int, 64>, 4>         results{};
        std::array<std::unique_ptr<ITaskGroup>, 4> groups;
        std::array<std::thread, 4>                 producers;
        std::atomic<int>                           ready{0};
        for (size_t producer = 0; producer < producers.size(); ++producer) {
            groups[producer]    = pool.makeTaskGroup();
            producers[producer] = std::thread([&, producer]() {
                ready.fetch_add(1);
                while (ready.load() != static_cast<int>(producers.size())) {
                    std::this_thread::yield();
                }
                for (size_t index = 0; index < results[producer].size(); ++index) {
                    const int expected = iteration + static_cast<int>(producer + index) + 1;
                    groups[producer]->run(
                        [&, producer, index, expected]() { results[producer][index] = expected; });
                }
            });
        }
        for (auto &producer : producers) {
            producer.join();
        }
        for (size_t producer = 0; producer < groups.size(); ++producer) {
            groups[producer]->wait();
            for (size_t index = 0; index < results[producer].size(); ++index) {
                EXPECT_EQ(results[producer][index], iteration + static_cast<int>(producer + index) + 1);
            }
        }
    }
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
