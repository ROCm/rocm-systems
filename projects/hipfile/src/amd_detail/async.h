/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "hipfile.h"
#include "thread-pool.h"

#include <condition_variable>
#include <cstdint>
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
struct Backend;
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
    void (*io_fn)(void *){nullptr};
    uint64_t                 wait_target{0};
    std::shared_ptr<Backend> backend{};

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
    virtual void spawnDrain(std::shared_ptr<IStream> stream);

private:
    void                                                 completion_thread();
    std::unordered_map<void *, std::shared_ptr<AsyncOp>> submitted_ops;
    std::vector<AsyncOp *>                               completed_ops;
    std::unique_ptr<ITaskGroup>                          task_group;
    std::thread                                          thread;
    std::mutex                                           mutex;
    std::condition_variable                              cv;
    bool                                                 is_finished;
};

void async_run_io(void *userargs);

void enqueueAsync(std::shared_ptr<Backend> backend, IoType type, std::shared_ptr<IFile> file,
                  std::shared_ptr<IBuffer> buffer, size_t *size_p, hoff_t *file_offset_p,
                  hoff_t *buffer_offset_p, ssize_t *bytes_transferred_p, std::shared_ptr<IStream> stream);
}

extern "C" {
void async_run_inline(void *userargs);
}
