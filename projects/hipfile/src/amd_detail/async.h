/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "hipfile.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <sys/types.h>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

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

class AsyncOp {
public:
    const IoType                         io_type;
    std::shared_ptr<IFile>               file;
    std::shared_ptr<IBuffer>             buffer;
    std::shared_ptr<IStream>             stream;
    std::variant<size_t, size_t *>       size;
    std::variant<const hoff_t, hoff_t *> file_offset;
    std::variant<const hoff_t, hoff_t *> buffer_offset;
    ssize_t *const                       bytes_transferred;

    AsyncOp(const AsyncOp &)            = delete;
    AsyncOp &operator=(const AsyncOp &) = delete;
    AsyncOp(AsyncOp &&)                 = delete;
    AsyncOp &&operator=(AsyncOp &&)     = delete;
    virtual ~AsyncOp();

    AsyncOp(IoType ioType, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer,
            std::shared_ptr<IStream> stream, size_t *size, hoff_t *file_offset, hoff_t *buffer_offset,
            ssize_t *bytes_transferred);
};

class AsyncMonitor {
public:
    virtual ~AsyncMonitor();
    AsyncMonitor();

    virtual void addOp(std::shared_ptr<AsyncOp> op);
    virtual void completeOp(AsyncOp *op);

private:
    void                                                 completion_thread();
    std::unordered_map<void *, std::shared_ptr<AsyncOp>> submitted_ops;
    std::vector<AsyncOp *>                               completed_ops;
    std::thread                                          thread;
    std::mutex                                           mutex;
    std::condition_variable                              cv;
    bool                                                 is_finished;
};
}
