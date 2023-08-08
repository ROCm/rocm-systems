// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "lib/common/units.hpp"
#include "lib/common/environment.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cstddef>
#include <cstdlib>

namespace rocprofiler
{
namespace common
{
namespace container
{
template <typename Tp>
struct atomic_ring_buffer;
//
namespace base
{
/// \struct tim::base::atomic_ring_buffer
/// \brief Ring buffer implementation, with support for mmap as backend (Linux only).
struct atomic_ring_buffer
{
    template <typename Tp>
    friend struct container::atomic_ring_buffer;

    atomic_ring_buffer() = default;
    explicit atomic_ring_buffer(bool _use_mmap) { set_use_mmap(_use_mmap); }
    explicit atomic_ring_buffer(size_t _size) { init(_size); }
    atomic_ring_buffer(size_t _size, bool _use_mmap);

    ~atomic_ring_buffer();

    atomic_ring_buffer(const atomic_ring_buffer&);
    atomic_ring_buffer& operator=(const atomic_ring_buffer&);

    atomic_ring_buffer(atomic_ring_buffer&&) noexcept;
    atomic_ring_buffer& operator=(atomic_ring_buffer&&) noexcept;

    /// Returns whether the buffer has been allocated
    bool is_initialized() const { return m_init; }

    /// Get the total number of bytes supported
    size_t capacity() const { return m_size; }

    /// Creates new ring buffer.
    void init(size_t size);

    /// Destroy ring buffer.
    void destroy();

    /// Request a pointer for writing at least \param n bytes.
    void* request(size_t n);

    /// Retrieve a pointer for reading at least \param n bytes.
    void* retrieve(size_t n) const;

    /// Write class-type data to buffer (uses placement new).
    template <typename Tp>
    std::pair<size_t, Tp*> write(Tp* in, std::enable_if_t<std::is_class<Tp>::value, int> = 0);

    /// Write non-class-type data to buffer (uses memcpy).
    template <typename Tp>
    std::pair<size_t, Tp*> write(Tp* in, std::enable_if_t<!std::is_class<Tp>::value, int> = 0);

    /// Request a pointer to an allocation. This is similar to a "write" except the
    /// memory is uninitialized. Typically used by allocators. If Tp is a class type,
    /// be sure to use a placement new instead of a memcpy.
    template <typename Tp>
    Tp* request();

    /// Read class-type data from buffer (uses placement new).
    template <typename Tp>
    std::pair<size_t, Tp*> read(Tp* _dest,
                                std::enable_if_t<std::is_class<Tp>::value, int> = 0) const;

    /// Read non-class-type data from buffer (uses memcpy).
    template <typename Tp>
    std::pair<size_t, Tp*> read(Tp* _dest,
                                std::enable_if_t<!std::is_class<Tp>::value, int> = 0) const;

    /// Retrieve a pointer to the head allocation (read).
    template <typename Tp>
    Tp* retrieve() const;

    /// Returns number of bytes currently held by the buffer.
    size_t count() const { return (m_write_count - m_read_count); }

    /// Returns how many bytes are availiable in the buffer.
    size_t free() const { return (m_size - count()); }

    /// Returns if the buffer is empty.
    bool is_empty() const { return (count() == 0); }

    /// Returns if the buffer is full.
    bool is_full() const { return (count() == m_size); }

    /// explicitly configure to use mmap if avail
    void set_use_mmap(bool);

    /// query whether using mmap
    bool get_use_mmap() const { return m_use_mmap; }

    std::string as_string() const;

    void save(std::fstream& _fs);
    void load(std::fstream& _fs);

private:
    /// Returns the current write pointer.
    void* write_ptr(size_t _write_count) const
    {
        return static_cast<char*>(m_ptr) + (_write_count % m_size);
    }

    /// Returns the current read pointer.
    void* read_ptr(size_t _read_count) const
    {
        return static_cast<char*>(m_ptr) + (_read_count % m_size);
    }

    void reset();

private:
    bool                        m_init              = false;
    bool                        m_use_mmap          = true;
    bool                        m_use_mmap_explicit = false;
    void*                       m_ptr               = nullptr;
    size_t                      m_size              = 0;
    mutable std::atomic<size_t> m_read_count        = 0;
    std::atomic<size_t>         m_write_count       = 0;
};
//
template <typename Tp>
std::pair<size_t, Tp*>
atomic_ring_buffer::write(Tp* in, std::enable_if_t<std::is_class<Tp>::value, int>)
{
    if(in == nullptr || m_ptr == nullptr) return {0, nullptr};

    auto  _length = sizeof(Tp);
    void* _out_p  = request(_length);

    if(_out_p == nullptr) return {0, nullptr};

    // Copy in.
    new(_out_p) Tp{std::move(*in)};

    // pointer in buffer
    Tp* _out = reinterpret_cast<Tp*>(_out_p);

    return {_length, _out};
}
//
template <typename Tp>
std::pair<size_t, Tp*>
atomic_ring_buffer::write(Tp* in, std::enable_if_t<!std::is_class<Tp>::value, int>)
{
    if(in == nullptr || m_ptr == nullptr) return {0, nullptr};

    auto  _length = sizeof(Tp);
    void* _out_p  = request(_length);

    if(_out_p == nullptr) return {0, nullptr};

    // Copy in.
    memcpy(_out_p, in, _length);

    // pointer in buffer
    Tp* _out = reinterpret_cast<Tp*>(_out_p);

    return {_length, _out};
}
//
template <typename Tp>
Tp*
atomic_ring_buffer::request()
{
    if(m_ptr == nullptr) return nullptr;

    return request(sizeof(Tp));
}
//
template <typename Tp>
std::pair<size_t, Tp*>
atomic_ring_buffer::read(Tp* _dest, std::enable_if_t<std::is_class<Tp>::value, int>) const
{
    if(is_empty() || _dest == nullptr) return {0, nullptr};

    auto  _length = sizeof(Tp);
    void* _out_p  = retrieve(_length);

    if(_out_p == nullptr) return {0, nullptr};

    // pointer in buffer
    Tp* in = reinterpret_cast<Tp*>(_out_p);

    // Copy out for BYTE, nothing magic here.
    *_dest = *in;

    return {_length, in};
}
//
template <typename Tp>
std::pair<size_t, Tp*>
atomic_ring_buffer::read(Tp* _dest, std::enable_if_t<!std::is_class<Tp>::value, int>) const
{
    if(is_empty() || _dest == nullptr) return {0, nullptr};

    auto  _length = sizeof(Tp);
    void* _out_p  = retrieve(_length);

    if(_out_p == nullptr) return {0, nullptr};

    // pointer in buffer
    Tp* in = reinterpret_cast<Tp*>(_out_p);

    using Up = typename std::remove_const<Tp>::type;

    // Copy out for BYTE, nothing magic here.
    Up* _out = const_cast<Up*>(_dest);
    memcpy(_out, in, _length);

    return {_length, in};
}
//
template <typename Tp>
Tp*
atomic_ring_buffer::retrieve() const
{
    if(m_ptr == nullptr) return nullptr;

    return retrieve(sizeof(Tp));
}
//
}  // namespace base
//
/// \struct tim::data_storage::atomic_ring_buffer
/// \brief Ring buffer wrapper around \ref tim::base::atomic_ring_buffer for data of type
/// Tp. If the data object size is larger than the page size (typically 4KB), behavior is
/// undefined. During initialization, one requests a minimum number of objects and the
/// buffer will support that number of object + the remainder of the page, e.g. if a page
/// is 1000 bytes, the object is 1 byte, and the buffer is requested to support 1500
/// objects, then an allocation supporting 2000 objects (i.e. 2 pages) will be created.
template <typename Tp>
struct atomic_ring_buffer : private base::atomic_ring_buffer
{
    using base_type = base::atomic_ring_buffer;

    static size_t get_items_per_page();

    atomic_ring_buffer()  = default;
    ~atomic_ring_buffer() = default;

    explicit atomic_ring_buffer(bool _use_mmap)
    : base_type{_use_mmap}
    {}

    explicit atomic_ring_buffer(size_t _size)
    : base_type{_size * sizeof(Tp)}
    {}

    atomic_ring_buffer(size_t _size, bool _use_mmap)
    : base_type{_size * sizeof(Tp), _use_mmap}
    {}

    atomic_ring_buffer(const atomic_ring_buffer&);
    atomic_ring_buffer(atomic_ring_buffer&&) noexcept = default;

    atomic_ring_buffer& operator=(const atomic_ring_buffer&);
    atomic_ring_buffer& operator=(atomic_ring_buffer&&) noexcept = default;

    /// Returns whether the buffer has been allocated
    bool is_initialized() const { return base_type::is_initialized(); }

    /// Get the total number of Tp instances supported
    size_t capacity() const { return (base_type::capacity()) / sizeof(Tp); }

    /// Creates new ring buffer.
    void init(size_t _size) { base_type::init(_size * sizeof(Tp)); }

    /// Destroy ring buffer.
    void destroy() { base_type::destroy(); }

    /// Write data to buffer.
    size_t data_size() const { return sizeof(Tp); }

    /// Write data to buffer. Return pointer to location of write
    Tp* write(Tp* in) { return base_type::write<Tp>(in).second; }

    /// Read data from buffer. Return pointer to location of read
    Tp* read(Tp* _dest) const { return base_type::read<Tp>(_dest).second; }

    /// Get an uninitialized address at tail of buffer.
    Tp* request() { return base_type::request<Tp>(); }

    /// Read data from head of buffer.
    Tp* retrieve() { return base_type::retrieve<Tp>(); }

    /// Returns number of Tp instances currently held by the buffer.
    size_t count() const { return (base_type::count()) / sizeof(Tp); }

    /// Returns how many Tp instances are availiable in the buffer.
    size_t free() const { return (base_type::free()) / sizeof(Tp); }

    /// Returns if the buffer is empty.
    bool is_empty() const { return base_type::is_empty(); }

    /// Returns if the buffer is full.
    bool is_full() const { return (base_type::free() < sizeof(Tp)); }

    template <typename... Args>
    auto emplace(Args&&... args)
    {
        Tp _obj{std::forward<Args>(args)...};
        return write(&_obj);
    }

    using base_type::get_use_mmap;
    using base_type::load;
    using base_type::save;
    using base_type::set_use_mmap;

    std::string as_string() const
    {
        std::ostringstream ss{};
        size_t             _w = std::log10(base_type::capacity()) + 1;
        ss << std::boolalpha << std::right << "data size: " << std::setw(_w) << data_size()
           << " B, is_initialized: " << std::setw(5) << is_initialized()
           << ", is_empty: " << std::setw(5) << is_empty() << ", is_full: " << std::setw(5)
           << is_full() << ", capacity: " << std::setw(_w) << capacity()
           << ", count: " << std::setw(_w) << count() << ", free: " << std::setw(_w) << free()
           << ", raw capacity: " << std::setw(_w) << base_type::capacity()
           << " B, raw count: " << std::setw(_w) << base_type::count()
           << " B, raw free: " << std::setw(_w) << base_type::free()
           << " B, pointer: " << std::setw(15) << base_type::m_ptr
           << ", raw read count: " << std::setw(_w) << base_type::m_read_count
           << ", raw write count: " << std::setw(_w) << base_type::m_write_count;
        return ss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const atomic_ring_buffer& obj)
    {
        return os << obj.as_string();
    }
};
//
template <typename Tp>
size_t
atomic_ring_buffer<Tp>::get_items_per_page()
{
    return std::max<size_t>(units::get_page_size() / sizeof(Tp), 1);
}
//
template <typename Tp>
atomic_ring_buffer<Tp>::atomic_ring_buffer(const atomic_ring_buffer<Tp>& rhs)
: base_type{rhs}
{
    size_t _n   = rhs.count();
    char*  _end = static_cast<char*>(rhs.m_ptr) + rhs.m_size;
    for(size_t i = 0; i < _n; ++i)
    {
        char* _addr = static_cast<char*>(rhs.read_ptr(m_read_count)) + (i * sizeof(Tp));
        if((_addr + sizeof(Tp)) > _end) _addr = static_cast<char*>(rhs.m_ptr);
        Tp* _in = static_cast<Tp*>(static_cast<void*>(_addr));
        write(_in);
    }
}
//
template <typename Tp>
atomic_ring_buffer<Tp>&
atomic_ring_buffer<Tp>::operator=(const atomic_ring_buffer<Tp>& rhs)
{
    if(this == &rhs) return *this;

    base_type::operator=(rhs);
    size_t     _n      = rhs.count();
    char*      _end    = static_cast<char*>(rhs.m_ptr) + rhs.m_size;
    for(size_t i = 0; i < _n; ++i)
    {
        char* _addr = static_cast<char*>(rhs.read_ptr(m_read_count)) + (i * sizeof(Tp));
        if((_addr + sizeof(Tp)) > _end) _addr = static_cast<char*>(rhs.m_ptr);
        Tp* _in = static_cast<Tp*>(static_cast<void*>(_addr));
        write(_in);
    }

    return *this;
}
//
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
