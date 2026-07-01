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
class IDriverState;

ssize_t hipFileIo(hipFile::IoType type, hipFileHandle_t fh, const void *buffer_base, size_t size,
                  hoff_t file_offset, hoff_t buffer_offset, hipFile::IDriverState &state,
                  const std::vector<std::shared_ptr<hipFile::Backend>> &backends);

hipFileError_t hipFileIOAsync(hipFile::IoType io_type, hipFileHandle_t fh, void *buffer_base, size_t *size_p,
                              hoff_t *file_offset_p, hoff_t *buffer_offset_p, ssize_t *bytes_transferred_p,
                              hipStream_t hipStream, hipFile::IDriverState &state);

hipFileError_t hipFileBatchSetUp(hipFileBatchHandle_t *batch_idp, unsigned max_nr,
                                 hipFile::IDriverState &state);

hipFileError_t hipFileBatchSubmit(hipFileBatchHandle_t batch_idp, unsigned nr, hipFileIOParams_t *iocbp,
                                  unsigned flags, hipFile::IDriverState &state);
}
