/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

#include <memory>
#include <sys/types.h>
#include <vector>

namespace hipFile {
enum class IoType;
struct Backend;
}

ssize_t hipFileIo(hipFile::IoType type, hipFileHandle_t fh, const void *buffer_base, size_t size,
                  hoff_t file_offset, hoff_t buffer_offset,
                  const std::vector<std::shared_ptr<hipFile::Backend>> &backends);
