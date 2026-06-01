/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "stream.h"

#include <mutex>

namespace hipFile {

class MStream : public IStream {
public:
    MOCK_METHOD(hipStream_t, getHipStream, (), (const, override));
    MOCK_METHOD(hipDevice_t, getHipDevice, (), (const, override));
    MOCK_METHOD(bool, fixedBufferOffset, (), (const, override));
    MOCK_METHOD(bool, fixedFileOffset, (), (const, override));
    MOCK_METHOD(bool, fixedIOSize, (), (const, override));
    MOCK_METHOD(bool, pageAligned, (), (const, override));
    MOCK_METHOD(std::unique_lock<std::mutex>, getLock, (), (override));
};

}
