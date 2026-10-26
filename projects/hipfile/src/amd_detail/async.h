/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "context.h"
#include "hipfile.h"
#include "io.h"
#include "thread-pool.h"

#include <atomic>
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

enum class AsyncOpState {
    Pending,   // Queued on the stream; the pool hasn't picked it up.
    Started,   // The pool is running the IO.
    Done,      // The IO finished and its result is published.
    Abandoned, // The submitter couldn't gate the stream before the pool started it; the IO is skipped.
};

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

    // Set before the dispatch host function is queued and never changed: true when the stream can't be gated
    // on a wait value, so the host function blocks until the IO finishes.
    bool block{false};

    // Pending -> Started (pool) or Pending -> Abandoned (submitter); Started -> Done (pool).
    std::atomic<AsyncOpState> state{AsyncOpState::Pending};

    // Keeps the op alive between queuing the dispatch host function and that function taking ownership.
    std::shared_ptr<AsyncOp> self;

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

    /// @brief Run an op on the pool without waiting (completion is observed via the stream signal).
    virtual void submitIo(std::shared_ptr<AsyncOp> op) = 0;

    /// @brief Run an op on the pool and block the caller until it finishes.
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

/// @brief Validate and dispatch an async IO on its stream. Throws if nothing was left in flight.
void enqueueAsync(std::shared_ptr<Backend> backend, IoType io_type, std::shared_ptr<IFile> file,
                  std::shared_ptr<IBuffer> buffer, size_t *size, hoff_t *file_offset, hoff_t *buffer_offset,
                  ssize_t *bytes_transferred, std::shared_ptr<IStream> stream);

}

extern "C" {
/// @brief Stream host function that hands a queued op to the async monitor. Takes ownership of the op
///        through AsyncOp::self. Never takes a lock and only blocks for ops that set AsyncOp::block.
void async_dispatch(void *userargs);
}
