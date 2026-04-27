// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/common/scope_destructor.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace rocprofiler
{
namespace common
{
namespace container
{
/// @brief Experimental per-producer record buffer.
///
/// This backend is a user-space POC for the logical-buffer-to-sharded-ring transport. Each
/// producer thread lazily gets a single-producer shard and writes record headers and payloads
/// in-band. The drain side walks the registered shards and forms the same
/// rocprofiler_record_header_t pointer array used by the existing buffered callback ABI.
struct user_ring_record_header_buffer
{
    using record_ptr_vec_t = std::vector<rocprofiler_record_header_t*>;

    user_ring_record_header_buffer();
    explicit user_ring_record_header_buffer(size_t nbytes);
    ~user_ring_record_header_buffer();

    user_ring_record_header_buffer(const user_ring_record_header_buffer&) = delete;
    user_ring_record_header_buffer(user_ring_record_header_buffer&&) noexcept;

    user_ring_record_header_buffer& operator=(const user_ring_record_header_buffer&) = delete;
    user_ring_record_header_buffer& operator=(user_ring_record_header_buffer&&) noexcept;

    bool allocate(size_t nbytes);
    bool is_allocated() const;

    template <typename Tp>
    bool emplace(Tp&);

    template <typename Tp>
    bool emplace(uint64_t, Tp&);

    template <typename Tp>
    bool emplace(uint32_t, uint32_t, Tp&);

    size_t get_num_record_headers();

    template <typename ClearRecordsT, typename FuncT, typename... Args>
    size_t process_record_headers(ClearRecordsT, FuncT&& functor, Args&&... args);

    void lock();
    void unlock();
    void read_lock();
    void read_unlock();
    bool is_locked() const;

    size_t clear();
    void   save(std::fstream&);
    void   load(std::fstream&);
    size_t reset();

    auto size() const;
    auto capacity() const;
    auto count() const;
    auto free() const;
    auto is_empty() const;
    auto is_full() const;

private:
    enum class storage_mode
    {
        private_memory,
        shared_mmap,
    };

    struct slot_header
    {
        uint32_t total_size     = 0;
        uint32_t header_offset  = 0;
        uint32_t payload_offset = 0;
        uint32_t payload_size   = 0;
        uint32_t flags          = 0;
        uint32_t reserved       = 0;
        uint64_t sequence       = 0;
    };

    struct shard
    {
        void*               ptr          = nullptr;
        size_t              capacity     = 0;
        size_t              write_count  = 0;
        size_t              record_count = 0;
        std::atomic<size_t> published    = {0};
        storage_mode        storage      = storage_mode::private_memory;

        bool allocate(size_t, storage_mode);
        void destroy();
        void reset();

        void* reserve_slot(size_t, size_t, rocprofiler_record_header_t**);
    };

    static size_t align_up(size_t value, size_t alignment);
    static size_t page_aligned_size(size_t value);
    static size_t slot_size(size_t payload_size, size_t payload_align);
    static bool   use_shared_mmap();

    shard* get_thread_shard();
    shard* create_shard_locked();
    shard* create_shard_for_load(size_t);
    void   destroy_storage();
    void   move_from(user_ring_record_header_buffer&&) noexcept;
    size_t aggregate_count() const;

    void write_lock();
    void write_unlock();

private:
    std::atomic<int64_t> m_requested = {0};
    std::atomic<int64_t> m_locked    = {0};
    std::shared_mutex    m_shared    = {};
    mutable std::mutex   m_mutex     = {};

    std::vector<std::unique_ptr<shard>> m_shards       = {};
    std::atomic<size_t>                 m_record_count = {0};
    size_t                              m_capacity     = 0;
    storage_mode                        m_storage      = storage_mode::private_memory;
    bool                                m_init         = false;
    uint64_t                            m_generation   = 0;
};

inline bool
user_ring_record_header_buffer::is_locked() const
{
    return m_locked.load(std::memory_order_acquire) > 0;
}

inline void
user_ring_record_header_buffer::lock()
{
    auto n = m_locked.fetch_add(1, std::memory_order_release);
    if(n == 0) write_lock();
}

inline void
user_ring_record_header_buffer::unlock()
{
    auto n = m_locked.fetch_sub(1, std::memory_order_release);
    if(n <= 1) write_unlock();
}

inline void
user_ring_record_header_buffer::read_lock()
{
    m_shared.lock_shared();
}

inline void
user_ring_record_header_buffer::read_unlock()
{
    m_shared.unlock_shared();
}

inline void
user_ring_record_header_buffer::write_lock()
{
    m_shared.lock();
}

inline void
user_ring_record_header_buffer::write_unlock()
{
    m_shared.unlock();
}

inline bool
user_ring_record_header_buffer::is_allocated() const
{
    return m_init && m_capacity > 0;
}

inline auto
user_ring_record_header_buffer::size() const
{
    return m_record_count.load(std::memory_order_acquire);
}

inline auto
user_ring_record_header_buffer::capacity() const
{
    auto _lk = std::lock_guard<std::mutex>{m_mutex};
    return m_shards.empty() ? m_capacity : (m_capacity * m_shards.size());
}

inline auto
user_ring_record_header_buffer::count() const
{
    return aggregate_count();
}

inline auto
user_ring_record_header_buffer::free() const
{
    auto _lk = std::lock_guard<std::mutex>{m_mutex};
    auto used = size_t{0};
    for(const auto& itr : m_shards)
    {
        used += itr->published.load(std::memory_order_acquire);
    }
    auto cap = m_shards.empty() ? m_capacity : (m_capacity * m_shards.size());
    return (used < cap) ? (cap - used) : size_t{0};
}

inline auto
user_ring_record_header_buffer::is_empty() const
{
    return size() == 0 && m_requested.load(std::memory_order_acquire) == 0;
}

inline auto
user_ring_record_header_buffer::is_full() const
{
    if(!is_allocated()) return true;
    auto _lk = std::lock_guard<std::mutex>{m_mutex};
    if(m_shards.empty()) return false;
    for(const auto& itr : m_shards)
    {
        if(itr->published.load(std::memory_order_acquire) < itr->capacity) return false;
    }
    return true;
}

template <typename Tp>
bool
user_ring_record_header_buffer::emplace(uint64_t hash, Tp& value)
{
    if(!is_allocated()) return false;

    m_requested.fetch_add(1, std::memory_order_acq_rel);
    auto _request_scope =
        scope_destructor{[&]() { m_requested.fetch_sub(1, std::memory_order_acq_rel); }};

    read_lock();
    auto _read_scope = scope_destructor{[&]() { read_unlock(); }};

    auto* _shard = get_thread_shard();
    if(!_shard) return false;

    auto* record = static_cast<rocprofiler_record_header_t*>(nullptr);
    auto* addr   = _shard->reserve_slot(sizeof(Tp), alignof(Tp), &record);
    if(!addr) return false;

    new(addr) Tp{value};
    record->hash    = hash;
    record->payload = addr;
    m_record_count.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

template <typename Tp>
bool
user_ring_record_header_buffer::emplace(uint32_t category, uint32_t kind, Tp& value)
{
    if(!is_allocated()) return false;

    m_requested.fetch_add(1, std::memory_order_acq_rel);
    auto _request_scope =
        scope_destructor{[&]() { m_requested.fetch_sub(1, std::memory_order_acq_rel); }};

    read_lock();
    auto _read_scope = scope_destructor{[&]() { read_unlock(); }};

    auto* _shard = get_thread_shard();
    if(!_shard) return false;

    auto* record = static_cast<rocprofiler_record_header_t*>(nullptr);
    auto* addr   = _shard->reserve_slot(sizeof(Tp), alignof(Tp), &record);
    if(!addr) return false;

    new(addr) Tp{value};
    record->category = category;
    record->kind     = kind;
    record->payload  = addr;
    m_record_count.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

template <typename Tp>
bool
user_ring_record_header_buffer::emplace(Tp& value)
{
    return emplace(typeid(Tp).hash_code(), value);
}

template <typename ClearRecordsT, typename FuncT, typename... Args>
size_t
user_ring_record_header_buffer::process_record_headers(ClearRecordsT, FuncT&& functor, Args&&... args)
{
    auto _lk = scope_destructor{[&]() { unlock(); }, [&]() { lock(); }};

    auto records = record_ptr_vec_t{};
    records.reserve(m_record_count.load(std::memory_order_acquire));

    {
        auto _shards_lk = std::lock_guard<std::mutex>{m_mutex};
        for(auto& shard_v : m_shards)
        {
            auto*  base       = static_cast<std::byte*>(shard_v->ptr);
            size_t offset     = 0;
            auto   write_size = shard_v->published.load(std::memory_order_acquire);

            while(base && offset < write_size)
            {
                auto* slot = reinterpret_cast<slot_header*>(base + offset);
                if(slot->total_size == 0 || offset + slot->total_size > write_size) break;

                auto* record = reinterpret_cast<rocprofiler_record_header_t*>(
                    base + offset + slot->header_offset);
                if(slot->flags > 0 && record->payload != nullptr) records.emplace_back(record);

                offset += slot->total_size;
            }
        }
    }

    auto num_records = records.size();
    std::forward<FuncT>(functor)(std::move(records), std::forward<Args>(args)...);

    if constexpr(ClearRecordsT::value) clear();

    return num_records;
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
