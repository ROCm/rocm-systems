/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend/bounce-buffer.h"
#include "context.h"
#include "hip.h"
#include "sys.h"

#include <cstddef>
#include <hip/hip_runtime_api.h>
#include <syslog.h>

namespace hipFile {

namespace {

    /// @brief A host buffer owned by, and cached for, a single thread
    class ThreadBounceBuffer {
    public:
        ThreadBounceBuffer()                                      = default;
        ThreadBounceBuffer(const ThreadBounceBuffer &)            = delete;
        ThreadBounceBuffer(ThreadBounceBuffer &&)                 = delete;
        ThreadBounceBuffer &operator=(const ThreadBounceBuffer &) = delete;
        ThreadBounceBuffer &operator=(ThreadBounceBuffer &&)      = delete;

        ~ThreadBounceBuffer()
        {
            release();
        }

        /// @brief Grow the buffer to at least size bytes
        /// @return A pointer to the buffer
        void *reserve(size_t size)
        {
            if (m_ptr && m_size >= size) {
                return m_ptr;
            }

            release();

            // Portable so that the buffer stays usable if the calling thread
            // switches devices between operations.
            m_ptr  = Context<Hip>::get()->hipHostMalloc(size, hipHostMallocPortable);
            m_size = size;

            return m_ptr;
        }

        /// @brief Free the buffer, if there is one
        void release() noexcept
        {
            if (!m_ptr) {
                return;
            }

            // Nothing useful can be done if the free fails, and this runs from a
            // destructor at thread exit, so swallow any error. Losing the mapping
            // is preferable to terminating the process.
            try {
                Context<Hip>::get()->hipHostFree(m_ptr);
            }
            catch (...) {
                logQuietly("Unable to free the thread's IO bounce buffer.");
            }

            m_ptr  = nullptr;
            m_size = 0;
        }

        size_t size() const noexcept
        {
            return m_ptr ? m_size : 0;
        }

    private:
        static void logQuietly(const char *msg) noexcept
        {
            try {
                Context<Sys>::get()->syslog(LOG_WARNING, msg);
            }
            catch (...) {
            }
        }

        void  *m_ptr{nullptr};
        size_t m_size{0};
    };

    ThreadBounceBuffer &threadBounceBuffer()
    {
        thread_local ThreadBounceBuffer buffer{};
        return buffer;
    }

}

void *
getThreadBounceBuffer(size_t size)
{
    return threadBounceBuffer().reserve(size);
}

void
releaseThreadBounceBuffer() noexcept
{
    threadBounceBuffer().release();
}

size_t
getThreadBounceBufferSize() noexcept
{
    return threadBounceBuffer().size();
}

}
