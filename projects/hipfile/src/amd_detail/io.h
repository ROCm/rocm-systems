/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "hipfile.h"

#include <cstddef>
#include <memory>

namespace hipFile {

class IBuffer;

enum class IoType {
    Read,
    Write,
};

bool paramsValid(const std::shared_ptr<IBuffer> &buffer, size_t size, hoff_t file_offset,
                 hoff_t buffer_offset);

}
