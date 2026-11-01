// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"

#include <fmt/format.h>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>

#include <sys/mman.h>

namespace rocprofiler
{
namespace kfd
{
// Raw page mapping for one chunk. The production policy; unit tests substitute
// their own (the mapping policy is the pool's template parameter).
struct chunk_mapping
{
    static void* map(size_t bytes) noexcept
    {
        void* const _p =
            ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return (_p == MAP_FAILED) ? nullptr : _p;
    }

    // munmap failure is reported, not swallowed: the address space leaks, and a
    // silent failure would make cleanup look successful.
    static void unmap(void* p, size_t bytes) noexcept
    {
        if(::munmap(p, bytes) != 0)
            ROCP_WARNING << fmt::format(
                "KFD dispatch-log: chunk pool munmap({}, {}) failed (errno={})", p, bytes, errno);
    }
};

// Spare chunks for one GPU's record stream. SPSC: the reader pops, the processor
// pushes (setup preallocates before publication). Lifecycle: preallocate at
// setup, clear() after both workers are joined, abandon() in a fork child.
//
// MEMORY ORDERING: each index has exactly one writer. The release on m_push_idx
// publishes the chunk's ownership and its reset header to the reader's acquire;
// the release on m_pop_idx publishes the slot for reuse, so the producer cannot
// overwrite a chunk the consumer still holds.
template <typename Mapping = chunk_mapping>
class basic_record_chunk_pool
{
public:
    struct chunk;
    struct chunk_deleter;

    using chunk_ptr = std::unique_ptr<chunk, chunk_deleter>;

    // Power of two: the spare slots are addressed by mask.
    static constexpr size_t kCapacity = 256;

    basic_record_chunk_pool() = default;
    ~basic_record_chunk_pool() { clear(); }

    basic_record_chunk_pool(const basic_record_chunk_pool&) = delete;
    basic_record_chunk_pool& operator=(const basic_record_chunk_pool&) = delete;

    // CONSUMER. The oldest spare, or nullptr when empty (may miss a concurrent push).
    chunk_ptr pop() noexcept
    {
        const auto _pop = m_pop_idx.load(std::memory_order_relaxed);
        if(m_push_idx.load(std::memory_order_acquire) == _pop) return nullptr;
        chunk_ptr _c{m_slots[_pop & (kCapacity - 1)]};
        m_pop_idx.store(_pop + 1, std::memory_order_release);
        return _c;
    }

    // PRODUCER. false when full; `c` is moved from only on success.
    bool push(chunk_ptr&& c) noexcept
    {
        const auto _push = m_push_idx.load(std::memory_order_relaxed);
        if(_push - m_pop_idx.load(std::memory_order_acquire) == kCapacity) return false;
        // Ownership passes into the slot BEFORE publication: the release store
        // of m_push_idx below is what makes the pointer visible to pop().
        m_slots[_push & (kCapacity - 1)] = c.release();
        m_push_idx.store(_push + 1, std::memory_order_release);
        return true;
    }

    // CONSUMER only: lower bound on spares (the push-index sample may be stale).
    size_t available() const noexcept
    {
        return m_push_idx.load(std::memory_order_acquire) -
               m_pop_idx.load(std::memory_order_relaxed);
    }

    // PRODUCER only: upper bound on spares (the pop-index sample may be stale-low).
    size_t occupancy() const noexcept
    {
        return m_push_idx.load(std::memory_order_relaxed) -
               m_pop_idx.load(std::memory_order_acquire);
    }

    // Free every pooled chunk. No concurrent pop/push is allowed here.
    void clear() noexcept
    {
        const auto _pop  = m_pop_idx.load(std::memory_order_relaxed);
        const auto _push = m_push_idx.load(std::memory_order_relaxed);
        for(size_t _i = _pop; _i != _push; ++_i)
            chunk_deleter{}(m_slots[_i & (kCapacity - 1)]);
        m_pop_idx.store(0, std::memory_order_relaxed);
        m_push_idx.store(0, std::memory_order_relaxed);
    }

    // Fork child: forget the pooled chunks without munmap or any chunk access.
    void abandon() noexcept
    {
        m_pop_idx.store(0, std::memory_order_relaxed);
        m_push_idx.store(0, std::memory_order_relaxed);
    }

    // Mapping::map, then a volatile store to byte 0 of every page on the CALLING
    // thread (a read would only map the shared zero page), then `new (p) chunk`:
    // the header is initialized and the slots are left untouched. nullptr on failure.
    static chunk_ptr allocate() noexcept
    {
        void* const _p = Mapping::map(map_bytes());
        if(_p == nullptr) return nullptr;
        auto* const _bytes = static_cast<volatile uint8_t*>(_p);
        for(size_t _off = 0; _off < map_bytes(); _off += kPageBytes)
            _bytes[_off] = 0;
        return chunk_ptr{new(_p) chunk};
    }

    // sizeof(chunk) rounded up to kPageBytes.
    static size_t map_bytes() noexcept
    {
        return (sizeof(chunk) + kPageBytes - 1) / kPageBytes * kPageBytes;
    }

    struct chunk
    {
        static constexpr size_t kRecords = 16 * 1024;

        // User-provided on purpose: even `chunk()` never zeroes the slots, so
        // value-initialization cannot hide a missing explicit page touch.
        chunk() noexcept {}

        uint64_t            base_pos = 0;          // stream position of slots[0]
        std::atomic<chunk*> next     = {nullptr};  // non-owning; stored once by the reader
        // Uninitialized slots are only sound for a trivial record type.
        static_assert(std::is_trivially_default_constructible_v<fw_record>);
        alignas(64) fw_record slots[kRecords];  // read only below the published position
    };

    // The chunk IS the mapping (placement-new at offset 0), so unmapping the chunk
    // address releases the whole mapping.
    struct chunk_deleter
    {
        void operator()(chunk* c) const noexcept
        {
            c->~chunk();
            Mapping::unmap(c, map_bytes());
        }
    };

private:
    // A 4 KiB stride touches every page (page size >= 4 KiB), and mmap rounds the length up.
    static constexpr size_t kPageBytes = 4096;

    // Owned while the index is in [m_pop_idx, m_push_idx); slots are addressed by
    // mask.
    std::array<chunk*, kCapacity> m_slots = {};
    // Written only by the consumer / only by the producer respectively.
    alignas(64) std::atomic<size_t> m_pop_idx  = {0};
    alignas(64) std::atomic<size_t> m_push_idx = {0};
};

using record_chunk_pool = basic_record_chunk_pool<>;

}  // namespace kfd
}  // namespace rocprofiler
