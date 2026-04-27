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

#include "lib/common/container/lttng_record_header_buffer_tracepoints.h"
#include "lib/common/container/record_header_buffer.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <typeinfo>
#include <utility>

namespace rocprofiler
{
namespace common
{
namespace container
{
/// @brief Experimental LTTng-UST shadow-emission backend for rocprofiler buffers.
///
/// The class keeps the existing in-process record_header_buffer semantics so the current
/// buffered callback ABI remains valid, while also emitting a typed LTTng-UST record envelope
/// for out-of-process subscription experiments.
struct lttng_record_header_buffer
{
    using base_buffer_t    = record_header_buffer::base_buffer_t;
    using record_vec_t     = record_header_buffer::record_vec_t;
    using record_ptr_vec_t = record_header_buffer::record_ptr_vec_t;

    lttng_record_header_buffer() = default;
    explicit lttng_record_header_buffer(size_t nbytes);
    ~lttng_record_header_buffer() = default;

    lttng_record_header_buffer(const lttng_record_header_buffer&) = delete;
    lttng_record_header_buffer(lttng_record_header_buffer&&) noexcept;

    lttng_record_header_buffer& operator=(const lttng_record_header_buffer&) = delete;
    lttng_record_header_buffer& operator=(lttng_record_header_buffer&&) noexcept;

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
    void emit(uint32_t category, uint32_t kind, uint64_t hash, size_t size, size_t alignment);

private:
    record_header_buffer m_buffer   = {};
    std::atomic<uint64_t> m_sequence = {0};
};

inline bool
lttng_record_header_buffer::allocate(size_t nbytes)
{
    return m_buffer.allocate(nbytes);
}

inline bool
lttng_record_header_buffer::is_allocated() const
{
    return m_buffer.is_allocated();
}

inline size_t
lttng_record_header_buffer::get_num_record_headers()
{
    return m_buffer.get_num_record_headers();
}

inline void
lttng_record_header_buffer::lock()
{
    m_buffer.lock();
}

inline void
lttng_record_header_buffer::unlock()
{
    m_buffer.unlock();
}

inline void
lttng_record_header_buffer::read_lock()
{
    m_buffer.read_lock();
}

inline void
lttng_record_header_buffer::read_unlock()
{
    m_buffer.read_unlock();
}

inline bool
lttng_record_header_buffer::is_locked() const
{
    return m_buffer.is_locked();
}

inline size_t
lttng_record_header_buffer::clear()
{
    return m_buffer.clear();
}

inline void
lttng_record_header_buffer::save(std::fstream& fs)
{
    m_buffer.save(fs);
}

inline void
lttng_record_header_buffer::load(std::fstream& fs)
{
    m_buffer.load(fs);
}

inline size_t
lttng_record_header_buffer::reset()
{
    m_sequence.store(0, std::memory_order_release);
    return m_buffer.reset();
}

inline auto
lttng_record_header_buffer::size() const
{
    return m_buffer.size();
}

inline auto
lttng_record_header_buffer::capacity() const
{
    return m_buffer.capacity();
}

inline auto
lttng_record_header_buffer::count() const
{
    return m_buffer.count();
}

inline auto
lttng_record_header_buffer::free() const
{
    return m_buffer.free();
}

inline auto
lttng_record_header_buffer::is_empty() const
{
    return m_buffer.is_empty();
}

inline auto
lttng_record_header_buffer::is_full() const
{
    return m_buffer.is_full();
}

template <typename Tp>
bool
lttng_record_header_buffer::emplace(uint64_t hash, Tp& value)
{
    auto ret = m_buffer.emplace(hash, value);
    if(ret) emit(0, 0, hash, sizeof(Tp), alignof(Tp));
    return ret;
}

template <typename Tp>
bool
lttng_record_header_buffer::emplace(uint32_t category, uint32_t kind, Tp& value)
{
    auto ret = m_buffer.emplace(category, kind, value);
    if(ret) emit(category, kind, typeid(Tp).hash_code(), sizeof(Tp), alignof(Tp));
    return ret;
}

template <typename Tp>
bool
lttng_record_header_buffer::emplace(Tp& value)
{
    return emplace(typeid(Tp).hash_code(), value);
}

template <typename ClearRecordsT, typename FuncT, typename... Args>
size_t
lttng_record_header_buffer::process_record_headers(ClearRecordsT, FuncT&& functor, Args&&... args)
{
    return m_buffer.process_record_headers(
        ClearRecordsT{}, std::forward<FuncT>(functor), std::forward<Args>(args)...);
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
