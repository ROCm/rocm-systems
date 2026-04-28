// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for sampling_service::pause() / resume() — AC-12, NFR-T-8.

#include <gtest/gtest.h>

#include "doubles/fake_clock.hpp"
#include "doubles/test_sampling_policies.hpp"
#include "sampling/sampling_service.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;
using test_service = sampling_service<test_sampling_policies>;

// ─── AC-12: pause() sets the paused flag ─────────────────────────────────────

TEST(sampling_service_pause_resume, pause_sets_paused_flag)
{
    test_service svc;
    fake_clock::reset(1000);

    EXPECT_FALSE(svc.is_paused());
    svc.pause();
    EXPECT_TRUE(svc.is_paused());
}

TEST(sampling_service_pause_resume, resume_clears_paused_flag)
{
    test_service svc;
    fake_clock::reset(1000);
    svc.pause();

    fake_clock::advance_ns(500);
    svc.resume();

    EXPECT_FALSE(svc.is_paused());
}

// ─── AC-12: pause calls block_samples ────────────────────────────────────────

TEST(sampling_service_pause_resume, pause_sets_blocked_flag)
{
    test_service svc;
    fake_clock::reset(0);

    EXPECT_FALSE(svc.is_blocked());
    svc.pause();
    EXPECT_TRUE(svc.is_blocked());
}

// ─── AC-12: resume calls unblock_samples ─────────────────────────────────────

TEST(sampling_service_pause_resume, resume_clears_blocked_flag)
{
    test_service svc;
    fake_clock::reset(0);

    svc.pause();
    svc.resume();

    EXPECT_FALSE(svc.is_blocked());
}

// ─── NFR-T-8: double-pause is a no-op ────────────────────────────────────────

TEST(sampling_service_pause_resume, double_pause_is_noop_after_first_pause)
{
    test_service svc;
    fake_clock::reset(0);

    svc.pause();
    EXPECT_TRUE(svc.is_paused());

    svc.pause();
    EXPECT_TRUE(svc.is_paused()) << "Double-pause must leave paused flag as true";

    svc.resume();
    EXPECT_FALSE(svc.is_paused());
}

// ─── NFR-T-8: double-resume is a no-op ───────────────────────────────────────

TEST(sampling_service_pause_resume, double_resume_is_noop)
{
    test_service svc;
    fake_clock::reset(0);

    svc.pause();
    svc.resume();
    EXPECT_FALSE(svc.is_paused());

    svc.resume();
    EXPECT_FALSE(svc.is_paused()) << "Double-resume must not change the paused flag";
}
