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

class MBatchOperation : public IBatchOperation {
public:
    MOCK_METHOD(void, markPending, (), (override));
    MOCK_METHOD(bool, tryCancel, (), (override));
    MOCK_METHOD(void, run, (), (noexcept, override));
    MOCK_METHOD(void, recordInternalError, (), (override));
    MOCK_METHOD(hipFileIOEvents_t, event, (), (const, override));
};

class MBatchContext : public IBatchContext {
public:
    MOCK_METHOD(unsigned, getCapacity, (), (const, noexcept, override));
    MOCK_METHOD(void, submitOperations, (BatchOperations ops), (override));
    MOCK_METHOD(void, getStatus,
                (unsigned min_nr, unsigned *nr, hipFileIOEvents_t *iocbp, BatchDeadline deadline),
                (override));
    MOCK_METHOD(void, cancelOperations, (), (override));
    MOCK_METHOD(void, cancelOperationsAndWait, (), (override));
};

}
