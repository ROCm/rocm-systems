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

#include "lib/common/environment.hpp"
#include "lib/common/units.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace common
{
namespace container
{
template <typename Tp>
struct ring_buffer;
//
namespace base
{
/// \struct tim::base::ring_buffer
/// \brief Ring buffer implementation, with support for mmap as backend (Linux only).
struct ring_buffer
{
    template <typename Tp>
    friend struct container::ring_buffer;

    ring_buffer() = default;
    explicit ring_buffer(bool _use_mmap) { set_use_mmap(_use_mmap); }
    explicit ring_buffer(size_t _size) { init(_size); }
    ring_buffer(size_t _size, bool _use_mmap);

    ~ring_buffer();

    ring_buffer(const ring_buffer&);
    ring_buffer& operator=(const ring_buffer&);

    ring_buffer(ring_buffer&&) noexcept;
    ring_buffer& operator=(ring_buffer&&) noexcept;

    /// Returns whether the buffer has been allocated
    bool is_initialized() const { return m_init; }

    /// Get the total number of bytes supported
    size_t capacity() const { return m_size; }

    /// Creates new ring buffer.
    void init(size_t size);

    /// Destroy ring buffer.
    void destroy();

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

    /// Request a pointer to an allocation for at least \param n bytes.
    void* request(size_t n);

    /// Read class-type data from buffer (uses placement new).
    template <typename Tp>
    std::pair<size_t, Tp*> read(Tp* out, std::enable_if_t<std::is_class<Tp>::value, int> = 0) const;

    /// Read non-class-type data from buffer (uses memcpy).
    template <typename Tp>
    std::pair<size_t, Tp*> read(Tp* out,
                                std::enable_if_t<!std::is_class<Tp>::value, int> = 0) const;

    /// Retrieve a pointer to the head allocation (read).
    template <typename Tp>
    Tp* retrieve();

    /// Retrieve a pointer to the head allocation of at least \param n bytes (read).
    void* retrieve(size_t n);

    /// Returns number of bytes currently held by the buffer.
    size_t count() const { return (m_write_count - m_read_count); }

    /// Returns how many bytes are availiable in the buffer.
    size_t free() const { return (m_size - count()); }

    /// Returns if the buffer is empty.
    bool is_empty() const { return (count() == 0); }

    /// Returns if the buffer is full.
    bool is_full() const { return (count() == m_size); }

    /// Rewind the read position n bytes
    size_t rewind(size_t n) const;

    /// explicitly configure to use mmap if avail
    void set_use_mmap(bool);

    /// query whether using mmap
    bool get_use_mmap() const { return m_use_mmap; }

    std::string as_string() const;

    void save(std::fstream& _fs);
    void load(std::fstream& _fs);

    friend std::ostream& operator<<(std::ostream& os, const ring_buffer& obj)
    {
        return os << obj.as_string();
    }

private:
    /// Returns the current write pointer.
    void* write_ptr() const { return static_cast<char*>(m_ptr) + (m_write_count % m_size); }

    /// Returns the current read pointer.
    void* read_ptr() const { return static_cast<char*>(m_ptr) + (m_read_count % m_size); }

    void reset();

private:
    bool           m_init              = false;
    bool           m_use_mmap          = true;
    bool           m_use_mmap_explicit = false;
    void*          m_ptr               = nullptr;
    size_t         m_size              = 0;
    mutable size_t m_read_count        = 0;
    size_t         m_write_count       = 0;
};
//
template <typename Tp>
std::pair<size_t, Tp*>
ring_buffer::write(Tp* in, std::enable_if_t<std::is_class<Tp>::value, int>)
{
    if(in == nullptr || m_ptr == nullptr) return {0, nullptr};

    auto _length = sizeof(Tp);

    // Make sure we don't put in more than there's room for, by writing no
    // more than there is free.
    if(_length > free())
        throw std::runtime_error("heap-buffer-overflow :: ring buffer is full. read data "
                                 "to avoid data corruption");

    // if write count is at the tail of buffer, bump to the end of buffer
    auto _modulo = m_size - (m_write_count % m_size);
    if(_modulo < _length) m_write_count += _modulo;

    // pointer in buffer
    Tp* out = reinterpret_cast<Tp*>(write_ptr());

    // Copy in.
    new((void*) out) Tp{std::move(*in)};

    // Update write count
    m_write_count += _length;

    return {_length, out};
}
//
template <typename Tp>
std::pair<size_t, Tp*>
ring_buffer::write(Tp* in, std::enable_if_t<!std::is_class<Tp>::value, int>)
{
    if(in == nullptr || m_ptr == nullptr) return {0, nullptr};

    auto _length = sizeof(Tp);

    // Make sure we don't put in more than there's room for, by writing no
    // more than there is free.
    if(_length > free())
        throw std::runtime_error("heap-buffer-overflow :: ring buffer is full. read data "
                                 "to avoid data corruption");

    // if write count is at the tail of buffer, bump to the end of buffer
    auto _modulo = m_size - (m_write_count % m_size);
    if(_modulo < _length) m_write_count += _modulo;

    // pointer in buffer
    Tp* out = reinterpret_cast<Tp*>(write_ptr());

    // Copy in.
    memcpy((void*) out, in, _length);

    // Update write count
    m_write_count += _length;

    return {_length, out};
}
//
template <typename Tp>
Tp*
ring_buffer::request()
{
    if(m_ptr == nullptr) return nullptr;

    auto _length = sizeof(Tp);

    // Make sure we don't put in more than there's room for, by writing no
    // more than there is free.
    if(_length > free())
        throw std::runtime_error("heap-buffer-overflow :: ring buffer is full. read data "
                                 "to avoid data corruption");

    // if write count is at the tail of buffer, bump to the end of buffer
    auto _modulo = m_size - (m_write_count % m_size);
    if(_modulo < _length) m_write_count += _modulo;

    // pointer in buffer
    Tp* _out = reinterpret_cast<Tp*>(write_ptr());

    // Update write count
    m_write_count += _length;

    return _out;
}
//
template <typename Tp>
std::pair<size_t, Tp*>
ring_buffer::read(Tp* out, std::enable_if_t<std::is_class<Tp>::value, int>) const
{
    if(is_empty() || out == nullptr) return {0, nullptr};

    auto _length = sizeof(Tp);

    // Make sure we do not read out more than there is actually in the buffer.
    if(_length > count()) throw std::runtime_error("ring buffer is empty");

    // if read count is at the tail of buffer, bump to the end of buffer
    auto _modulo = m_size - (m_read_count % m_size);
    if(_modulo < _length) m_read_count += _modulo;

    // pointer in buffer
    Tp* in = reinterpret_cast<Tp*>(read_ptr());

    // Copy out for BYTE, nothing magic here.
    *out = *in;

    // Update read count.
    m_read_count += _length;

    return {_length, in};
}
//
template <typename Tp>
std::pair<size_t, Tp*>
ring_buffer::read(Tp* out, std::enable_if_t<!std::is_class<Tp>::value, int>) const
{
    if(is_empty() || out == nullptr) return {0, nullptr};

    auto _length = sizeof(Tp);

    using Up = typename std::remove_const<Tp>::type;

    // Make sure we do not read out more than there is actually in the buffer.
    if(_length > count()) throw std::runtime_error("ring buffer is empty");

    // if read count is at the tail of buffer, bump to the end of buffer
    auto _modulo = m_size - (m_read_count % m_size);
    if(_modulo < _length) m_read_count += _modulo;

    // pointer in buffer
    Tp* in = reinterpret_cast<Tp*>(read_ptr());

    // Copy out for BYTE, nothing magic here.
    Up* _out = const_cast<Up*>(out);
    memcpy(_out, in, _length);

    // Update read count.
    m_read_count += _length;

    return {_length, in};
}
//
template <typename Tp>
Tp*
ring_buffer::retrieve()
{
    if(m_ptr == nullptr) return nullptr;

    auto _length = sizeof(Tp);

    // Make sure we don't put in more than there's room for, by writing no
    // more than there is free.
    if(_length > count()) throw std::runtime_error("ring buffer is empty");

    // if read count is at the tail of buffer, bump to the end of buffer
    auto _modulo = m_size - (m_read_count % m_size);
    if(_modulo < _length) m_read_count += _modulo;

    // pointer in buffer
    Tp* _out = reinterpret_cast<Tp*>(read_ptr());

    // Update write count
    m_read_count += _length;

    return _out;
}
//
}  // namespace base
///
/// \struct rocprofiler::container::ring_buffer
/// \brief Ring buffer wrapper around \ref tim::base::ring_buffer for data of type Tp. If
/// the data object size is larger than the page size (typically 4KB), behavior is
/// undefined. During initialization, one requests a minimum number of objects and the
/// buffer will support that number of object + the remainder of the page, e.g. if a page
/// is 1000 bytes, the object is 1 byte, and the buffer is requested to support 1500
/// objects, then an allocation supporting 2000 objects (i.e. 2 pages) will be created.
template <typename Tp>
struct ring_buffer : private base::ring_buffer
{
    using base_type = base::ring_buffer;

    static size_t get_items_per_page();

    ring_buffer()  = default;
    ~ring_buffer() = default;

    explicit ring_buffer(bool _use_mmap)
    : base_type{_use_mmap}
    {}

    explicit ring_buffer(size_t _size)
    : base_type{_size * sizeof(Tp)}
    {}

    ring_buffer(size_t _size, bool _use_mmap)
    : base_type{_size * sizeof(Tp), _use_mmap}
    {}

    ring_buffer(const ring_buffer&);
    ring_buffer(ring_buffer&&) noexcept = default;

    ring_buffer& operator=(const ring_buffer&);
    ring_buffer& operator=(ring_buffer&&) noexcept = default;

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
    Tp* read(Tp* out) const { return base_type::read<Tp>(out).second; }

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

    /// Rewinds the read pointer
    size_t rewind(size_t n) const { return base_type::rewind(n); }

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

    friend std::ostream& operator<<(std::ostream& os, const ring_buffer& obj)
    {
        return os << obj.as_string();
    }
};
//
template <typename Tp>
size_t
ring_buffer<Tp>::get_items_per_page()
{
    return std::max<size_t>(units::get_page_size() / sizeof(Tp), 1);
}
//
template <typename Tp>
ring_buffer<Tp>::ring_buffer(const ring_buffer<Tp>& rhs)
: base_type{rhs}
{
    size_t _n   = rhs.count();
    char*  _end = static_cast<char*>(rhs.m_ptr) + rhs.m_size;
    for(size_t i = 0; i < _n; ++i)
    {
        char* _addr = static_cast<char*>(rhs.read_ptr()) + (i * sizeof(Tp));
        if((_addr + sizeof(Tp)) > _end) _addr = static_cast<char*>(rhs.m_ptr);
        Tp* _in = static_cast<Tp*>(static_cast<void*>(_addr));
        write(_in);
    }
}
//
template <typename Tp>
ring_buffer<Tp>&
ring_buffer<Tp>::operator=(const ring_buffer<Tp>& rhs)
{
    if(this == &rhs) return *this;

    base_type::operator=(rhs);
    size_t     _n      = rhs.count();
    char*      _end    = static_cast<char*>(rhs.m_ptr) + rhs.m_size;
    for(size_t i = 0; i < _n; ++i)
    {
        char* _addr = static_cast<char*>(rhs.read_ptr()) + (i * sizeof(Tp));
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
