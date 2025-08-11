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

#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/scope_destructor.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <atomic>
#include <functional>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <vector>

namespace rocprofiler
{
namespace common
{
namespace container
{
/// @brief this struct stores all the record information in an ring_buffer.
/// It is thread-safe to have multiple threads emplace records into the buffer.
struct record_header_buffer
{
    using base_buffer_t    = base::ring_buffer;
    using record_vec_t     = std::vector<rocprofiler_record_header_t>;
    using record_ptr_vec_t = std::vector<rocprofiler_record_header_t*>;

    record_header_buffer() = default;
    explicit record_header_buffer(size_t nbytes);
    ~record_header_buffer() = default;

    record_header_buffer(const record_header_buffer&) = delete;
    record_header_buffer(record_header_buffer&&) noexcept;

    record_header_buffer& operator=(const record_header_buffer&) = delete;
    record_header_buffer& operator                               =(record_header_buffer&&) noexcept;

    // allocate the buffer if it is not already allocated. Will return false if buffer is already
    // allocated
    bool allocate(size_t nbytes);

    // return whether the buffer has been allocated
    bool is_allocated() const;

    /// place an object in the buffer using its typeid hash code
    template <typename Tp>
    bool emplace(Tp&);

    /// place an object in the buffer using the specified numerical identifier
    template <typename Tp>
    bool emplace(uint64_t, Tp&);

    /// place an object in the buffer using the specified numerical identifier
    template <typename Tp>
    bool emplace(uint32_t, uint32_t, Tp&);

    /// this function will return the number of record headers
    size_t get_num_record_headers();

    /// this function will invoke functor with a vector of pointers to the record headers.
    /// if ClearRecordsV is true, the container will be cleared after invoking the functor
    template <typename ClearRecordsT, typename FuncT, typename... Args>
    size_t process_record_headers(ClearRecordsT, FuncT&& functor, Args&&... args);

    /// record_header_buffer is a multiple writer, single reader data structure so
    /// this function prevents writing via emplace
    void lock();

    /// potentially re-enable emplace if no other readers have locked
    void unlock();

    /// record_header_buffer is a multiple writer, single reader data structure so
    /// this function prevents reading while emplacing
    void read_lock();

    /// potentially allow reading after writing via emplace
    void read_unlock();

    /// check if writing is available
    bool is_locked() const;

    /// restores to original empty state
    size_t clear();

    /// binary save to file
    void save(std::fstream& _fs);

    /// binary load from file
    void load(std::fstream& _fs);

    /// full deallocation
    size_t reset();

    /// the number of header entries
    auto size() const;

    /// the number of bytes in the buffer
    auto capacity() const;

    /// the number of used bytes in the buffer
    auto count() const;

    /// the number of free bytes in the buffer
    auto free() const;

    /// true if no bytes are used in the buffer
    auto is_empty() const;

    /// true if all the bytes are used in the buffer or there is no buffer allocation
    auto is_full() const;

private:
    /// this is an explicit write lock that does not guard against deadlocking like lock()
    void write_lock();

    /// this is an explicit write unlock that does not guard against deadlocking like unlock()
    void write_unlock();

private:
    std::atomic<int64_t> m_requested = {0};
    std::atomic<int64_t> m_locked    = {0};
    std::atomic<size_t>  m_index     = {};
    std::shared_mutex    m_shared    = {};
    base_buffer_t        m_buffer    = {};
    record_vec_t         m_headers   = {};
};

inline bool
record_header_buffer::is_locked() const
{
    return m_locked.load(std::memory_order_acquire) > 0;
}

inline void
record_header_buffer::lock()
{
    auto n = m_locked.fetch_add(1, std::memory_order_release);
    if(n == 0) write_lock();
}

inline void
record_header_buffer::unlock()
{
    auto n = m_locked.fetch_sub(1, std::memory_order_release);
    if(n <= 1) write_unlock();
}

inline void
record_header_buffer::read_lock()
{
    m_shared.lock_shared();
}

inline void
record_header_buffer::read_unlock()
{
    m_shared.unlock_shared();
}

inline void
record_header_buffer::write_lock()
{
    m_shared.lock();
}

inline void
record_header_buffer::write_unlock()
{
    m_shared.unlock();
}

inline bool
record_header_buffer::is_allocated() const
{
    return m_buffer.is_initialized();
}

inline auto
record_header_buffer::size() const
{
    return m_index.load(std::memory_order_acquire);
}

inline auto
record_header_buffer::capacity() const
{
    return std::min<size_t>(m_headers.size(), m_buffer.capacity());
}

inline auto
record_header_buffer::count() const
{
    return m_buffer.count();
}

inline auto
record_header_buffer::free() const
{
    return m_buffer.free();
}

inline auto
record_header_buffer::is_empty() const
{
    return (m_buffer.is_empty() && m_requested.load() == 0) || m_headers.empty();
}

inline auto
record_header_buffer::is_full() const
{
    return m_buffer.is_full() || size() == m_headers.size();
}

template <typename Tp>
bool
record_header_buffer::emplace(uint64_t _hash, Tp& _v)
{
    if(m_headers.empty()) return false;

    constexpr auto request_size = sizeof(Tp);
    constexpr auto align_size   = alignof(Tp);

    // notify there was a request
    m_requested.fetch_add(1);

    // in theory, we shouldn't need to lock here but the thread sanitizer says there is a race.
    // the lock will be short-lived so hopefully, it will scale fine
    write_lock();
    auto* _addr = m_buffer.request(request_size, align_size, false);
    write_unlock();

    read_lock();
    if(_addr)
    {
        // if there is space in the buffer, atomically get an index
        // for where the header record should be placed.
        // NOTE: m_headers was resized to be large enough to accomodate
        // sizeof(Tp) == 1 for every entry in buffer
        auto idx = m_index.fetch_add(1, std::memory_order_release);

        // placement new
        new(_addr) Tp{_v};

        auto record       = rocprofiler_record_header_t{};
        record.hash       = _hash;
        record.payload    = _addr;
        m_headers.at(idx) = record;
    }
    read_unlock();

    // remove notification of request
    m_requested.fetch_sub(1);

    return (_addr != nullptr);
}

template <typename Tp>
bool
record_header_buffer::emplace(uint32_t _category, uint32_t _kind, Tp& _v)
{
    if(m_headers.empty()) return false;

    constexpr auto request_size = sizeof(Tp);
    constexpr auto align_size   = alignof(Tp);

    // notify there was a request
    m_requested.fetch_add(1);

    // in theory, we shouldn't need to lock here but the thread sanitizer says there is a race.
    // the lock will be short-lived so hopefully, it will scale fine
    write_lock();
    auto* _addr = m_buffer.request(request_size, align_size, false);
    write_unlock();

    read_lock();
    if(_addr)
    {
        // if there is space in the buffer, atomically get an index
        // for where the header record should be placed.
        // NOTE: m_headers was resized to be large enough to accomodate
        // sizeof(Tp) == 1 for every entry in buffer
        auto idx = m_index.fetch_add(1, std::memory_order_release);

        // placement new
        new(_addr) Tp{_v};

        auto record       = rocprofiler_record_header_t{};
        record.category   = _category;
        record.kind       = _kind;
        record.payload    = _addr;
        m_headers.at(idx) = record;
    }
    read_unlock();

    // remove notification of request
    m_requested.fetch_sub(1);

    return (_addr != nullptr);
}

template <typename Tp>
bool
record_header_buffer::emplace(Tp& _v)
{
    // if enumerations are not used, use the typeid hash code
    return emplace(typeid(Tp).hash_code(), _v);
}

template <typename ClearRecordsT, typename FuncT, typename... Args>
size_t
record_header_buffer::process_record_headers(ClearRecordsT, FuncT&& _functor, Args&&... _args)
{
    // RAII for lock/unlock
    auto _lk = scope_destructor{[&]() { unlock(); }, [&]() { lock(); }};

    auto _n       = m_index.load(std::memory_order_acquire);
    auto _records = record_ptr_vec_t{};
    _records.reserve(_n);
    for(size_t i = 0; i < _n; ++i)
    {
        if(auto& itr = m_headers.at(i); itr.hash > 0 && itr.payload != nullptr)
            _records.emplace_back(&itr);
    }

    // get number of records before vector is moved
    auto _num_records = _records.size();

    // invoke the callback
    std::forward<FuncT>(_functor)(std::move(_records), std::forward<Args>(_args)...);

    // clear the container
    if constexpr(ClearRecordsT::value) clear();

    return _num_records;
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
