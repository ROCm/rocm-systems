/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "state.h"

#include "mbuffer.h"
#include "mfile.h"

#include <gmock/gmock.h>

/*
 * A place to create mocks for IDriverState.
 */

namespace hipFile {

class MDriverState : public IDriverState {
public:
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
    MOCK_METHOD(void, registerStream, (const hipStream_t hip_stream, uint32_t flags), (override));
    MOCK_METHOD(void, deregisterStream, (const hipStream_t hip_stream), (override));
    MOCK_METHOD(std::shared_ptr<IStream>, getStream, (hipStream_t hip_stream), (override));
    MOCK_METHOD(file_buffer_pair, getFileAndBuffer, (hipFileHandle_t fh, const void *buf), (override));
    MOCK_METHOD(file_buffer_stream_tuple, getFileBufferAndStream,
                (hipFileHandle_t fh, const void *buf, hipStream_t hipStream), (override));
    MOCK_METHOD(void, incrRefCount, (), (override));
    MOCK_METHOD(void, decrRefCount, (), (override));
    MOCK_METHOD(int64_t, getRefCount, (), (override, const));
    MOCK_METHOD(void, ensureInitialized, (), (override));
};

}
