/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "async.h"
#include "context.h"

#include <gmock/gmock.h>

namespace hipFile {

struct MAsyncResourcePool : AsyncResourcePool {
    ContextOverride<AsyncResourcePool> co;
    MAsyncResourcePool() : co{this}
    {
    }
    MOCK_METHOD(void *, acquireOp, (size_t size), (override));
    MOCK_METHOD(void, releaseOp, (void *ptr), (noexcept, override));
    MOCK_METHOD(uint64_t *, acquireSignal, (), (override));
    MOCK_METHOD(void, releaseSignal, (uint64_t * slot), (noexcept, override));
    MOCK_METHOD(void, drain, (), (override));
};

}
