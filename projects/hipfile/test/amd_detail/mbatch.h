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
    MOCK_METHOD(void, tryCancel, (), (override));
    MOCK_METHOD(void, run, (), (noexcept, override));
    MOCK_METHOD(void, recordInternalError, (), (override));
    MOCK_METHOD(hipFileIOEvents_t, event, (), (const, override));
    MOCK_METHOD(bool, isTerminal, (), (const, override));
};

class MBatchOperationFactory : public IBatchOperationFactory {
public:
    MOCK_METHOD(std::shared_ptr<IBatchOperation>, create,
                (std::unique_ptr<const hipFileIOParams_t> params, std::shared_ptr<IBuffer> buffer,
                 std::shared_ptr<IFile> file),
                (override));
};

class MBatchContext : public IBatchContext {
public:
    MOCK_METHOD(unsigned, getCapacity, (), (const, noexcept, override));
    MOCK_METHOD(void, submitOperations,
                (const hipFileIOParams_t *params, const unsigned num_params,
                 IBatchOperationFactory *operation_factory),
                (override));
    MOCK_METHOD(void, getStatus,
                (unsigned min_nr, unsigned *nr, hipFileIOEvents_t *iocbp, struct timespec *timeout),
                (override));
    MOCK_METHOD(void, cancelOperations, (), (override));
};

}
