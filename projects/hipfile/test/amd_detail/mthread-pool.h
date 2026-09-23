/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"
#include "thread-pool.h"

#include <functional>
#include <memory>

#include <gmock/gmock.h>

namespace hipFile {

class MTaskGroup : public ITaskGroup {
public:
    MOCK_METHOD(void, run, (std::function<void()> work), (override));
    MOCK_METHOD(void, cancel, (), (override));
    MOCK_METHOD(void, wait, (), (override));
};

class MThreadPool : public IThreadPool {
public:
    ContextOverride<IThreadPool> o_co;

    MThreadPool() : o_co{this}
    {
    }

    MOCK_METHOD(std::unique_ptr<ITaskGroup>, makeTaskGroup, (), (override));
};

}
