/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "buffer.h"
#include "io.h"

namespace hipFile {

bool
paramsValid(const std::shared_ptr<IBuffer> &buffer, size_t size, hoff_t file_offset, hoff_t buffer_offset)
{
    size_t buffer_length = buffer->getLength();
    if (file_offset < 0) {
        return false;
    }
    if (buffer_offset < 0) {
        return false;
    }
    if (buffer_length <= static_cast<size_t>(buffer_offset)) {
        return false;
    }
    if (buffer_length - static_cast<size_t>(buffer_offset) < size) {
        return false;
    }
    return true;
}

}
