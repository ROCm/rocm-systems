/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"
#include "thread-pool.h"

#include <cstddef>
#include <functional>
#include <gmock/gmock.h>
#include <memory>

namespace hipFile {

class MTaskGroup : public ITaskGroup {
public:
    MOCK_METHOD(void, run, (std::function<void()> work), (override));
    MOCK_METHOD(void, cancel, (), (override));
    MOCK_METHOD(void, wait, (), (override));
};

class MThreadPool : public IThreadPool {
public:
    ContextOverride<IThreadPool> co;

    MThreadPool() : co{this}
    {
    }

    MOCK_METHOD(std::unique_ptr<ITaskGroup>, makeTaskGroup, (), (override));
    MOCK_METHOD(std::size_t, threadCount, (), (const, noexcept, override));
};

}
