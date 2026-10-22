/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "context.h"
#include "hipfile.h"
#include "io.h"
#include "thread-pool.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <sys/types.h>
#include <variant>

namespace hipFile {
class IBuffer;
class IFile;
class IStream;
struct Backend;

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
    std::function<void(AsyncOp *)>       io_fn;
    std::shared_ptr<Backend>             backend;

    // false only after enqueueAsync arms a stream wait value for this op; otherwise the dispatch host
    // function self-blocks until the IO completes. Read and written under the stream's lock.
    bool block{true};

    AsyncOp(const AsyncOp &)            = delete;
    AsyncOp &operator=(const AsyncOp &) = delete;
    AsyncOp(AsyncOp &&)                 = delete;
    AsyncOp &operator=(AsyncOp &&)      = delete;
    ~AsyncOp()                          = default;

    AsyncOp(IoType io_type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer,
            std::shared_ptr<IStream> stream, size_t *size, hoff_t *file_offset, hoff_t *buffer_offset,
            ssize_t *bytes_transferred, std::function<void(AsyncOp *)> io_fn,
            std::shared_ptr<Backend> backend);
};

class IAsyncMonitor {
public:
    virtual ~IAsyncMonitor() = default;

    /// @brief Run an op's IO on the pool without waiting (completion is observed via the stream signal).
    virtual void submitIo(std::shared_ptr<AsyncOp> op) = 0;

    /// @brief Run an op's IO on the pool and block the caller until it finishes.
    virtual void submitIoAndWait(std::shared_ptr<AsyncOp> op) = 0;
};

class AsyncMonitor : public IAsyncMonitor {
public:
    explicit AsyncMonitor(std::unique_ptr<IThreadPool> pool = std::make_unique<ThreadPool>());
    ~AsyncMonitor() override;

    void submitIo(std::shared_ptr<AsyncOp> op) override;
    void submitIoAndWait(std::shared_ptr<AsyncOp> op) override;

private:
    std::unique_ptr<IThreadPool> pool;       // Dedicated async pool; never the batch Context<IThreadPool>.
    std::unique_ptr<ITaskGroup>  task_group; // From pool; waited and destroyed before pool joins.
};

HIPFILE_CONTEXT_DEFAULT_IMPL(IAsyncMonitor, AsyncMonitor);

/// @brief The IO worker run on the pool: performs a synchronous backend IO and records the result.
void async_run_io(AsyncOp *op);

/// @brief Validate and dispatch an async IO on its stream. Throws before any work is committed.
void enqueueAsync(std::shared_ptr<Backend> backend, IoType io_type, std::shared_ptr<IFile> file,
                  std::shared_ptr<IBuffer> buffer, size_t *size, hoff_t *file_offset, hoff_t *buffer_offset,
                  ssize_t *bytes_transferred, std::shared_ptr<IStream> stream);

}

extern "C" {
/// @brief Stream host function that hands a committed op to the async monitor.
void async_dispatch(void *userargs);
}
