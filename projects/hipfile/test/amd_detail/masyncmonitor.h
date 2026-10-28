/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "async.h"
#include "context.h"

#include <gmock/gmock.h>
#include <memory>

namespace hipFile {

struct MAsyncMonitor : IAsyncMonitor {
    ContextOverride<IAsyncMonitor> co;
    MAsyncMonitor() : co{this}
    {
    }
    MOCK_METHOD(void, submitIo, (std::shared_ptr<AsyncOp> op), (override));
};

}
