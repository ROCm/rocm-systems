// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/collection/fork_gate.hpp"
#include "core/collection/tests/wait_helpers.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <latch>
#include <optional>
#include <thread>
#include <utility>

namespace rocprofsys::collection
{
namespace
{
using testing::let_other_threads_run;
using testing::pass_granted_elsewhere;
using testing::wait_for_fork_in_progress;
}  // namespace

TEST(fork_gate, pass_is_granted_without_a_fork)
{
    fork_gate gate;

    const fork_gate::pass pass{ gate };

    EXPECT_TRUE(pass);
}

TEST(fork_gate, pass_is_refused_while_blocked_and_granted_after_unblock)
{
    fork_gate gate;

    gate.block();
    EXPECT_FALSE(pass_granted_elsewhere(gate));
    EXPECT_FALSE(fork_gate::pass{ gate });

    gate.unblock();
    EXPECT_TRUE(pass_granted_elsewhere(gate));
}

TEST(fork_gate, block_waits_until_a_held_pass_is_released)
{
    fork_gate         gate;
    std::latch        pass_held{ 1 };
    std::latch        release_pass{ 1 };
    std::atomic<bool> released{ false };
    std::atomic<bool> released_before_block_returned{ false };

    const std::jthread poller{ [&] {
        const fork_gate::pass pass{ gate };
        pass_held.count_down();
        release_pass.wait();
        released = true;
    } };
    pass_held.wait();

    std::jthread forker{ [&] {
        gate.block();
        released_before_block_returned = released.load();
        gate.unblock();
    } };
    EXPECT_TRUE(wait_for_fork_in_progress(gate)) << "block() never counted the fork";
    release_pass.count_down();
    forker.join();

    EXPECT_TRUE(released_before_block_returned);
}

TEST(fork_gate, concurrent_forks_keep_passes_refused_until_the_last_unblock)
{
    fork_gate  gate;
    std::latch both_blocked{ 2 };
    std::latch first_unblock{ 1 };
    std::latch second_unblock{ 1 };
    std::latch first_done{ 1 };

    const std::jthread first{ [&] {
        gate.block();
        both_blocked.count_down();
        first_unblock.wait();
        gate.unblock();
        first_done.count_down();
    } };
    std::jthread second{ [&] {
        gate.block();
        both_blocked.count_down();
        second_unblock.wait();
        gate.unblock();
    } };
    both_blocked.wait();
    EXPECT_FALSE(pass_granted_elsewhere(gate));

    first_unblock.count_down();
    first_done.wait();
    EXPECT_FALSE(pass_granted_elsewhere(gate));

    second_unblock.count_down();
    second.join();
    EXPECT_TRUE(pass_granted_elsewhere(gate));
}

TEST(fork_gate, nested_block_keeps_passes_refused_until_the_outer_unblock)
{
    fork_gate gate;

    gate.block();
    gate.block();
    gate.unblock();
    EXPECT_FALSE(pass_granted_elsewhere(gate));

    gate.unblock();
    EXPECT_TRUE(pass_granted_elsewhere(gate));
}

TEST(fork_gate, poll_that_forks_does_not_wait_for_its_own_pass)
{
    fork_gate gate;
    {
        const fork_gate::pass pass{ gate };
        ASSERT_TRUE(pass);

        gate.block();
        EXPECT_FALSE(pass_granted_elsewhere(gate));
        gate.unblock();
        EXPECT_TRUE(pass_granted_elsewhere(gate));
    }

    // the pass count is balanced again: a fork elsewhere does not wait forever
    std::jthread{ [&gate] {
        gate.block();
        gate.unblock();
    } }.join();
    EXPECT_TRUE(pass_granted_elsewhere(gate));
}

TEST(fork_gate, nested_fork_inside_a_poll_leaves_the_gate_balanced)
{
    fork_gate gate;
    {
        const fork_gate::pass pass{ gate };
        gate.block();
        gate.block();
        gate.unblock();
        gate.unblock();
    }

    std::jthread{ [&gate] {
        gate.block();
        gate.unblock();
    } }.join();
    EXPECT_TRUE(pass_granted_elsewhere(gate));
}

TEST(fork_gate, fork_from_another_thread_waits_for_a_poll_that_forked)
{
    fork_gate         gate;
    std::latch        poll_forked{ 1 };
    std::latch        finish_poll{ 1 };
    std::atomic<bool> poll_done{ false };
    std::atomic<bool> poll_done_before_block_returned{ false };

    const std::jthread poller{ [&] {
        {
            const fork_gate::pass pass{ gate };
            gate.block();
            gate.unblock();
            poll_forked.count_down();
            finish_poll.wait();
            poll_done = true;
        }
    } };
    poll_forked.wait();

    std::jthread forker{ [&] {
        gate.block();
        poll_done_before_block_returned = poll_done.load();
        gate.unblock();
    } };
    EXPECT_TRUE(wait_for_fork_in_progress(gate)) << "block() never counted the fork";
    finish_poll.count_down();
    forker.join();

    EXPECT_TRUE(poll_done_before_block_returned);
}

TEST(fork_gate, waiting_pass_enters_only_after_the_unblock)
{
    fork_gate         gate;
    std::atomic<bool> unblocked{ false };
    std::atomic<bool> entered_after_unblock{ false };

    gate.block();
    std::jthread poller{ [&] {
        const fork_gate::pass pass{ gate, fork_gate::pass::k_wait };
        entered_after_unblock = static_cast<bool>(pass) && unblocked.load();
    } };
    let_other_threads_run();
    unblocked = true;
    gate.unblock();
    poller.join();

    EXPECT_TRUE(entered_after_unblock);
}

TEST(fork_gate, waiting_pass_inside_a_held_pass_does_not_wait_for_a_fork)
{
    fork_gate                      gate;
    std::optional<fork_gate::pass> outer{ std::in_place, gate };
    std::jthread                   forker{ [&gate] {
        gate.block();
        gate.unblock();
    } };
    EXPECT_TRUE(wait_for_fork_in_progress(gate)) << "block() never counted the fork";

    {
        const fork_gate::pass inner{ gate, fork_gate::pass::k_wait };
        EXPECT_TRUE(inner);
    }
    outer.reset();
    forker.join();

    EXPECT_TRUE(pass_granted_elsewhere(gate));
}

TEST(fork_gate, fork_after_a_nested_pass_left_does_not_wait_for_the_outer_pass)
{
    fork_gate gate;
    {
        const fork_gate::pass outer{ gate };
        {
            const fork_gate::pass inner{ gate };
            ASSERT_TRUE(inner);
        }
        gate.block();
        EXPECT_FALSE(pass_granted_elsewhere(gate));
        gate.unblock();
    }

    std::jthread{ [&gate] {
        gate.block();
        gate.unblock();
    } }.join();
    EXPECT_TRUE(pass_granted_elsewhere(gate));
}

TEST(fork_gate, fork_inside_nested_passes_does_not_wait_for_them)
{
    fork_gate gate;
    {
        const fork_gate::pass outer{ gate };
        const fork_gate::pass inner{ gate, fork_gate::pass::k_wait };
        gate.block();
        gate.unblock();
    }

    std::jthread{ [&gate] {
        gate.block();
        gate.unblock();
    } }.join();
    EXPECT_TRUE(pass_granted_elsewhere(gate));
}

TEST(fork_gate, pass_held_on_one_gate_does_not_affect_another_gate)
{
    fork_gate held;
    fork_gate other;
    {
        const fork_gate::pass pass{ held };
        ASSERT_TRUE(pass);

        other.block();
        other.unblock();
    }

    // other's pass count is still balanced: a fork elsewhere does not wait forever
    std::jthread{ [&other] {
        other.block();
        other.unblock();
    } }.join();
    EXPECT_TRUE(pass_granted_elsewhere(other));
}
}  // namespace rocprofsys::collection
