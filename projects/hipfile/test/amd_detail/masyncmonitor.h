/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "async.h"
#include "context.h"

#include <gmock/gmock.h>

namespace hipFile {

struct MAsyncMonitor : AsyncMonitor {
    ContextOverride<AsyncMonitor> co;
    MAsyncMonitor() : co{this}
    {
    }
    MOCK_METHOD(void, addOp, (std::shared_ptr<AsyncOp> op), (override));
    MOCK_METHOD(void, completeOp, (AsyncOp * op), (override));
};

}
