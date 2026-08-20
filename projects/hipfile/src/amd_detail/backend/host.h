/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "async.h"
#include "backend.h"
#include "hipfile.h"

#include <memory>
#include <sys/types.h>

namespace hipFile {

class IBuffer;
class IFile;
class IStream;
enum class IoType;

class Host final : public Backend {
public:
    using Backend::io;

    Host()                   = default;
    virtual ~Host() override = default;

    int score(const std::shared_ptr<IFile> &file, const std::shared_ptr<IBuffer> &buffer, size_t size,
              hoff_t file_offset, hoff_t buffer_offset) const override;

    void async_io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t *size_p,
                  hoff_t *file_offset_p, hoff_t *buffer_offset_p, ssize_t *bytes_transferred_p,
                  std::shared_ptr<IStream> stream) override;

protected:
    ssize_t _io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                     hoff_t file_offset, hoff_t buffer_offset) override;
};

class AsyncOpHost final : public AsyncOp {
public:
    size_t submitted_size;

    AsyncOpHost(IoType ioType, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer,
                std::shared_ptr<IStream> stream, size_t *size, hoff_t *fileOffset, hoff_t *bufferOffset,
                ssize_t *bytesTransferred);

    virtual ~AsyncOpHost() override;
    void  operator delete(void *ptr) noexcept;
    void *operator new(size_t size);
};

} // namespace hipFile

extern "C" {
void async_host_do(void *userargs);
}
