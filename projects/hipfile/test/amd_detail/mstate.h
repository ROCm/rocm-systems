/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"
#include "state.h"

#include "mbuffer.h"
#include "mfile.h"

#include <gmock/gmock.h>

/*
 * A place to create mocks for the DriverState singleton.
 */

namespace hipFile {

class MDriverState : public DriverState {
public:
    ContextOverride<DriverState> o_co;

    MDriverState() : o_co{this}
    {
    }

    MOCK_METHOD(hipFileBatchHandle_t, createBatchContext, (unsigned capacity), (override));
    MOCK_METHOD(void, destroyBatchContext, (hipFileBatchHandle_t handle), (override));
    MOCK_METHOD(std::shared_ptr<IBatchContext>, getBatchContext, (hipFileBatchHandle_t handle), (override));
    MOCK_METHOD(void, registerBuffer, (const void *buf, size_t length, int flags), (override));
    MOCK_METHOD(void, deregisterBuffer, (const void *buf), (override));
    MOCK_METHOD(std::shared_ptr<IBuffer>, getRegisteredBuffer, (const void *buf), (override));
    MOCK_METHOD(std::shared_ptr<IBuffer>, getBuffer, (const void *buf), (override));
    MOCK_METHOD(hipFileHandle_t, registerFile, (UnregisteredFile && uf), (override));
    MOCK_METHOD(void, deregisterFile, (hipFileHandle_t fh), (override));
    MOCK_METHOD(std::shared_ptr<IFile>, getFile, (hipFileHandle_t fh), (override));
    MOCK_METHOD(file_buffer_pair, getFileAndBuffer, (hipFileHandle_t fh, const void *buf), (override));
    MOCK_METHOD(void, incrRefCount, (), (override));
    MOCK_METHOD(void, decrRefCount, (), (override));
    MOCK_METHOD(int64_t, getRefCount, (), (override, const));
    MOCK_METHOD(std::vector<std::shared_ptr<Backend>>, getBackends, (), (const, override));
};

}
