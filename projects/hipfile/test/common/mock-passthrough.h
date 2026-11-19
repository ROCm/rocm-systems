/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// MOCK_PASSTHROUGH belongs in test-common.h. However, the unit tests
// (hipfile-test.h) and the system tests (test-common.h) each maintain their
// own copies of the same common helpers. Including test-common.h from a mock
// pulls both sets into the unit tests and causes redefinition errors. Until
// the two sets of helpers are consolidated, this macro lives in its own header.

#include <utility>

// Set the default action of a mocked method to call the real base class
// implementation. Prefer NiceMock, so calls without an EXPECT_CALL pass through
// silently. An EXPECT_CALL without an action also passes through.
//
// To alter only some calls and pass the rest through, chain the altered calls
// and DoDefault() into one expectation. Once a method has an EXPECT_CALL, every
// call to it must match an expectation, even on a NiceMock, so a plain
// `.WillOnce(...)` makes the next call fail:
// ```
// NiceMock<MHip> mhip;
// mhip.enable_passthrough();
// EXPECT_CALL(mhip, <func>).WillOnce(...).WillRepeatedly(DoDefault());
// ```
// See "Understanding Uninteresting vs Unexpected Calls" in the gMock cookbook.
#define MOCK_PASSTHROUGH(base_class, func)                                                                   \
    ON_CALL(*this, func).WillByDefault([this](auto &&...args) {                                              \
        return this->base_class::func(std::forward<decltype(args)>(args)...);                                \
    })
