/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "async.h"
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
class IStream;
}
namespace hipFile {
enum class IoType;
}

namespace hipFile {

struct AsyncOpFallback : AsyncOp {
    size_t      submitted_size;
    size_t      chunk_bytes_copied;
    void *const gpu_buffer;
    void       *bounce_buffer_host_ptr;
    void       *bounce_buffer_dev_ptr;
    size_t      bounce_buffer_size;

    AsyncOpFallback(IoType ioType, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer,
                    std::shared_ptr<IStream> stream, size_t *size, hoff_t *fileOffset, hoff_t *bufferOffset,
                    ssize_t *bytesTransferred);

    void *devPtr();

    virtual ~AsyncOpFallback() override;
    void  operator delete(void *ptr) noexcept;
    void *operator new(size_t size);
};

}
