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
/// @brief Experimental BPF-style record buffer.
///
/// This is a user-space transport that borrows the BPF ring buffer reserve/commit layout:
/// producers reserve a variable-size slot, write the public record header and payload in-band,
/// and the consumer drains committed slots in allocation order. It intentionally does not create
/// a kernel BPF map because rocprofiler-sdk producers and tool callbacks run in the same process.
struct bpf_record_header_buffer
{
    using record_ptr_vec_t = std::vector<rocprofiler_record_header_t*>;

    bpf_record_header_buffer() = default;
    explicit bpf_record_header_buffer(size_t nbytes);
    ~bpf_record_header_buffer();

    bpf_record_header_buffer(const bpf_record_header_buffer&) = delete;
    bpf_record_header_buffer(bpf_record_header_buffer&&) noexcept;

    bpf_record_header_buffer& operator=(const bpf_record_header_buffer&) = delete;
    bpf_record_header_buffer& operator=(bpf_record_header_buffer&&) noexcept;

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
    struct slot_header
    {
        uint32_t total_size     = 0;
        uint32_t header_offset  = 0;
        uint32_t payload_offset = 0;
        uint32_t payload_size   = 0;
    };

    static size_t align_up(size_t value, size_t alignment);
    static size_t page_aligned_size(size_t value);
    static size_t slot_size(size_t payload_size, size_t payload_align);

    void* reserve_slot(size_t                        payload_size,
                       size_t                        payload_align,
                       rocprofiler_record_header_t** record);
    void  write_lock();
    void  write_unlock();
    void  destroy_storage();
    void  move_from(bpf_record_header_buffer&&) noexcept;

private:
    std::atomic<int64_t> m_requested    = {0};
    std::atomic<int64_t> m_locked       = {0};
    std::atomic<size_t>  m_write_count  = {0};
    std::atomic<size_t>  m_record_count = {0};
    std::shared_mutex    m_shared       = {};
    void*                m_ptr          = nullptr;
    size_t               m_capacity     = 0;
    bool                 m_init         = false;
};

inline bool
bpf_record_header_buffer::is_locked() const
{
    return m_locked.load(std::memory_order_acquire) > 0;
}

inline void
bpf_record_header_buffer::lock()
{
    auto n = m_locked.fetch_add(1, std::memory_order_release);
    if(n == 0) write_lock();
}

inline void
bpf_record_header_buffer::unlock()
{
    auto n = m_locked.fetch_sub(1, std::memory_order_release);
    if(n <= 1) write_unlock();
}

inline void
bpf_record_header_buffer::read_lock()
{
    m_shared.lock_shared();
}

inline void
bpf_record_header_buffer::read_unlock()
{
    m_shared.unlock_shared();
}

inline void
bpf_record_header_buffer::write_lock()
{
    m_shared.lock();
}

inline void
bpf_record_header_buffer::write_unlock()
{
    m_shared.unlock();
}

inline bool
bpf_record_header_buffer::is_allocated() const
{
    return m_init && m_ptr != nullptr;
}

inline auto
bpf_record_header_buffer::size() const
{
    return m_record_count.load(std::memory_order_acquire);
}

inline auto
bpf_record_header_buffer::capacity() const
{
    return m_capacity;
}

inline auto
bpf_record_header_buffer::count() const
{
    return m_write_count.load(std::memory_order_acquire);
}

inline auto
bpf_record_header_buffer::free() const
{
    auto _count = count();
    return (_count < m_capacity) ? (m_capacity - _count) : size_t{0};
}

inline auto
bpf_record_header_buffer::is_empty() const
{
    return (count() == 0 && m_requested.load(std::memory_order_acquire) == 0);
}

inline auto
bpf_record_header_buffer::is_full() const
{
    return free() == 0;
}

template <typename Tp>
bool
bpf_record_header_buffer::emplace(uint64_t hash, Tp& value)
{
    if(!is_allocated()) return false;

    m_requested.fetch_add(1, std::memory_order_acq_rel);
    auto _request_scope =
        scope_destructor{[&]() { m_requested.fetch_sub(1, std::memory_order_acq_rel); }};

    read_lock();
    auto _read_scope = scope_destructor{[&]() { read_unlock(); }};

    auto* record = static_cast<rocprofiler_record_header_t*>(nullptr);
    auto* addr   = reserve_slot(sizeof(Tp), alignof(Tp), &record);
    if(!addr) return false;

    new(addr) Tp{value};
    record->hash    = hash;
    record->payload = addr;
    m_record_count.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

template <typename Tp>
bool
bpf_record_header_buffer::emplace(uint32_t category, uint32_t kind, Tp& value)
{
    if(!is_allocated()) return false;

    m_requested.fetch_add(1, std::memory_order_acq_rel);
    auto _request_scope =
        scope_destructor{[&]() { m_requested.fetch_sub(1, std::memory_order_acq_rel); }};

    read_lock();
    auto _read_scope = scope_destructor{[&]() { read_unlock(); }};

    auto* record = static_cast<rocprofiler_record_header_t*>(nullptr);
    auto* addr   = reserve_slot(sizeof(Tp), alignof(Tp), &record);
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
bpf_record_header_buffer::emplace(Tp& value)
{
    return emplace(typeid(Tp).hash_code(), value);
}

template <typename ClearRecordsT, typename FuncT, typename... Args>
size_t
bpf_record_header_buffer::process_record_headers(ClearRecordsT, FuncT&& functor, Args&&... args)
{
    auto _lk = scope_destructor{[&]() { unlock(); }, [&]() { lock(); }};

    auto records = record_ptr_vec_t{};
    records.reserve(m_record_count.load(std::memory_order_acquire));

    auto*  base       = static_cast<std::byte*>(m_ptr);
    size_t offset     = 0;
    auto   write_size = m_write_count.load(std::memory_order_acquire);

    while(base && offset < write_size)
    {
        auto* slot = reinterpret_cast<slot_header*>(base + offset);
        if(slot->total_size == 0 || offset + slot->total_size > write_size) break;

        auto* record =
            reinterpret_cast<rocprofiler_record_header_t*>(base + offset + slot->header_offset);
        if(record->payload != nullptr) records.emplace_back(record);

        offset += slot->total_size;
    }

    auto num_records = records.size();
    std::forward<FuncT>(functor)(std::move(records), std::forward<Args>(args)...);

    if constexpr(ClearRecordsT::value) clear();

    return num_records;
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
