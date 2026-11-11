/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Linked against the real test main and parameter listener, and run by CTest
// with a rejected HIP_TEST_LEVEL. The listener must abort the run before any
// test case starts, so this test must never run.
#include <catch2/catch_test_macros.hpp>

TEST_CASE("LevelAbort_MustNotRun") { FAIL("LevelAbort_MustNotRun ran"); }
