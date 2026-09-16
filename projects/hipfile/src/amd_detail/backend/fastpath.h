/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "backend.h"
#include "buffer.h"
#include "file.h"
#include "hipfile.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
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

struct Fastpath : public BackendWithFallback {
    virtual ~Fastpath() override = default;

    int score(const std::shared_ptr<IFile> &file, const std::shared_ptr<IBuffer> &buffer, size_t size,
              hoff_t file_offset, hoff_t buffer_offset) const override;

    void enqueueAsyncIo(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer,
                        size_t *size_p, hoff_t *file_offset_p, hoff_t *buffer_offset_p,
                        ssize_t *bytes_transferred_p, std::shared_ptr<IStream> stream,
                        std::shared_ptr<AsyncFailoverState> failover) override;

protected:
    ssize_t _io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                     hoff_t file_offset, hoff_t buffer_offset) override;
    bool    is_fallback_eligible(std::exception_ptr e_ptr, ssize_t nbytes) const override;
};

}

extern "C" {
void async_fastpath_copy(void *userArgs);
}
