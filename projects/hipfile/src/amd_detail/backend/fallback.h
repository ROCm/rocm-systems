/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "async.h"
#include "backend/asyncop-fallback.h"
#include "backend.h"
#include "hipfile.h"

#include <memory>
#include <sys/types.h>

namespace hipFile {
class IBuffer;
}
namespace hipFile {
class IFile;
}
namespace hipFile {
enum class IoType;
}

namespace hipFile {

struct Fallback : public Backend {
    using Backend::io;
    virtual ~Fallback() override = default;

    int score(std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size, hoff_t file_offset,
              hoff_t buffer_offset) const override;

    void async_io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t *size_p,
                  hoff_t *file_offset_p, hoff_t *buffer_offset_p, ssize_t *bytes_transferred_p,
                  std::shared_ptr<IStream> stream);

    // Once we can import gtest.h and make test suites or test friends everything
    // below here should be made protected.
    ssize_t io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
               hoff_t file_offset, hoff_t buffer_offset, size_t chunk_size);

protected:
    ssize_t _io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                     hoff_t file_offset, hoff_t buffer_offset) override;
    ssize_t _io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                     hoff_t file_offset, hoff_t buffer_offset, size_t chunk_size);
};

}

extern "C" {
void async_io_bind_params(void *userargs);
void async_io_cleanup(void *userargs);
void async_io_cpu_copy(void *userargs);
}
