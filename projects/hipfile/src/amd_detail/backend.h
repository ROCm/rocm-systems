/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "buffer.h"
#include "file.h"
#include "hip.h"
#include "io.h"
#include "sys.h"

#include <climits>
#include <cstddef>
#include <exception>
#include <memory>
#include <sys/types.h>
#include <unistd.h>

namespace hipFile {

[[nodiscard]] inline size_t
getPageSize()
{
    static const size_t value = [] {
        const long v = sysconf(_SC_PAGESIZE);
        if (v == -1) {
            throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
        }
        return static_cast<size_t>(v);
    }();
    return value;
}

[[nodiscard]] inline size_t
getPageMask()
{
    static const size_t value = ~(getPageSize() - 1);
    return value;
}

// The maximum number of bytes that can be transferred in a single read() or
// write() system call. Calculation is same as kernel's MAX_RW_COUNT.
[[nodiscard]] inline size_t
getMaxRwCount()
{
    static const size_t value = static_cast<size_t>(INT_MAX) & getPageMask();
    return value;
}

/// @brief Backend is not enabled
struct BackendDisabled : public std::runtime_error {
    BackendDisabled() : std::runtime_error("Backend is disabled")
    {
    }
};

struct Backend {
    virtual ~Backend() = default;

    /// @brief Indicate this backend's eagerness to service the IO request.
    ///
    /// Indicates the willingness of the backend to perform service an IO
    /// request. The higher the number the more eager the backend is to service
    /// the request. If a negative number is returned, the backend is refusing to
    /// perform the operation. If the operation is submitted to the backend, the
    /// operation would likely fail.
    ///
    /// @param file           The IO request's file
    /// @param buffer         The IO request's buffer
    /// @param size           The IO request's size
    /// @param file_offset    Offset from the start of the file
    /// @param buffer_offset  Offset from the start of the buffer
    /// @return The eagerness "score"
    virtual int score(const std::shared_ptr<IFile> &file, const std::shared_ptr<IBuffer> &buffer, size_t size,
                      hoff_t file_offset, hoff_t buffer_offset) const = 0;

    /// @brief Perform a read or write operation
    ///
    /// @param type          IO type (read/write)
    /// @param file          File to read from or write to
    /// @param buffer        Buffer to write to or read from
    /// @param size          Number of bytes to transfer
    /// @param file_offset   Offset from the start of the file
    /// @param buffer_offset Offset from the start of the buffer
    ///
    /// @return Number of bytes transferred
    ///
    /// @throws Hip::RuntimeError Sys::RuntimeError
    virtual ssize_t io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                       hoff_t file_offset, hoff_t buffer_offset);

protected:
    /// @brief Perform a read or write operation
    ///
    /// @note Provides a common target across all Backends that provides the
    ///       implementation for running IO.
    /// @param type          IO type (read/write)
    /// @param file          File to read from or write to
    /// @param buffer        Buffer to write to or read from
    /// @param size          Number of bytes to transfer
    /// @param file_offset   Offset from the start of the file
    /// @param buffer_offset Offset from the start of the buffer
    ///
    /// @return Number of bytes transferred
    ///
    /// @throws Hip::RuntimeError Sys::RuntimeError
    virtual ssize_t _io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer,
                             size_t size, hoff_t file_offset, hoff_t buffer_offset) = 0;
};

// BackendWithFallback allows for an IO to be retried automatically with a
// different Backend in the event of an error.
struct BackendWithFallback : public Backend {
    ssize_t io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
               hoff_t file_offset, hoff_t buffer_offset) override final;

    /// @brief Register a Backend to retry a failed IO operation.
    ///
    /// @param backend Backend to retry a failed IO operation.
    ///
    /// @throws std::invalid_argument If a nullptr or self reference is passed.
    void register_fallback_backend(std::shared_ptr<Backend> backend);

protected:
    /// @brief Check if a failed IO operation can be re-issued to the fallback Backend.
    ///
    /// @param e_ptr         exception_ptr to the thrown exception from the failed IO
    /// @param nbytes        Return value from `_io_impl`, or 0 if an exception was thrown.
    ///
    /// @return True if this BackendWithFallback can retry the IO, else False.
    virtual bool is_fallback_eligible(std::exception_ptr e_ptr, ssize_t nbytes) const = 0;

private:
    std::shared_ptr<Backend> fallback_backend;
};

}
