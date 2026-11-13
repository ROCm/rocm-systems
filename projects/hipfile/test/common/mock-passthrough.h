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
// implementation. Prefer using NiceMock when using passthrough. Otherwise,
// every called function must have an EXPECT_CALL set or an "uninteresting call"
// failure will occur. EXPECT_CALL without an action uses passthrough. If setting
// up an expectation to alter behaviour, add `.RetiresOnSaturation()` to re-enable
// passthrough after the function has been called.
//
// StrictMock example:
// ```
// StrictMock<MHip> mhip;
// mhip.enable_passthrough();
// EXPECT_CALL(mhip, <passthrough_func>).Times(AnyNumber());
// EXPECT_CALL(mhip, <modified_func>).WillOnce(...).RetiresOnSaturation();
// ```
#define MOCK_PASSTHROUGH(base_class, func)                                                                   \
    ON_CALL(*this, func).WillByDefault([this](auto &&...args) {                                              \
        return this->base_class::func(std::forward<decltype(args)>(args)...);                                \
    })
