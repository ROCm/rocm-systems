/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "buffer.h"

#include <gmock/gmock.h>

/*
 * A place to create mocks for the buffer module.
 */

namespace hipFile {

class MBuffer : public IBuffer {
public:
    MOCK_METHOD(void *, getBuffer, (), (const, override));
    MOCK_METHOD(size_t, getLength, (), (const, override));
    MOCK_METHOD(int, getFlags, (), (const, override));
    MOCK_METHOD(hipMemoryType, getType, (), (const, override));
    MOCK_METHOD(int, getGpuId, (), (const, override));
};

class MBufferMap : public BufferMap {
public:
    MBufferMap()
    {
    }
    MOCK_METHOD(void, registerBuffer, (const void *bufptr, size_t length, int flags), (override));
    MOCK_METHOD(void, deregisterBuffer, (const void *bufptr), (override));
    MOCK_METHOD(std::shared_ptr<IBuffer>, getRegisteredBuffer, (const void *bufptr), (override));
    MOCK_METHOD(std::shared_ptr<IBuffer>, getBuffer, (const void *bufptr), (override));
    MOCK_METHOD(void, clear, (), (override));
};

}
