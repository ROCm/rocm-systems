/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "batch/batch.h"

#include <gmock/gmock.h>

/*
 * A place to create mocks for the batch module.
 */

namespace hipFile {

class MBatchContext : public IBatchContext {
public:
    MOCK_METHOD(unsigned, getCapacity, (), (const, noexcept, override));
    MOCK_METHOD(void, submitOperations, (BatchOperations ops), (override));
};

}
