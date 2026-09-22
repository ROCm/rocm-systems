/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "file.h"
#include "hipfile.h"

#include <gmock/gmock.h>

/*
 * A place to create mocks for the file module.
 */

namespace hipFile {

class MFile : public IFile {
public:
    MOCK_METHOD(hipFileHandle_t, handle, (), (const, noexcept, override));
    MOCK_METHOD(int, clientFd, (), (const, noexcept, override));
    MOCK_METHOD(int, bufferedFd, (), (const, noexcept, override));
    MOCK_METHOD(std::optional<int>, unbufferedFd, (), (const, noexcept, override));
    MOCK_METHOD(uint32_t, dioMemAlign, (), (const, noexcept, override));
    MOCK_METHOD(uint32_t, dioOffsetAlign, (), (const, noexcept, override));
    MOCK_METHOD(bool, isBlockDevice, (), (const, noexcept, override));
    MOCK_METHOD(bool, isRegularFile, (), (const, noexcept, override));
    MOCK_METHOD(bool, onExt4Ordered, (), (const, noexcept, override));
    MOCK_METHOD(bool, onXfs, (), (const, noexcept, override));
};

class MFileMap : public FileMap {
public:
    MFileMap()
    {
    }
    MOCK_METHOD(hipFileHandle_t, registerFile, (UnregisteredFile && uf), (override));
    MOCK_METHOD(void, deregisterFile, (hipFileHandle_t fh), (override));
    MOCK_METHOD(std::shared_ptr<IFile>, getFile, (hipFileHandle_t), (override));
    MOCK_METHOD(void, clear, (), (override));
};

}
