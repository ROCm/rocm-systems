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

#include "ring_buffer.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/units.hpp"

#include <fmt/format.h>

#include <sys/mman.h>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace rocprofiler
{
namespace common
{
namespace container
{
namespace base
{
ring_buffer::~ring_buffer() { destroy(); }

ring_buffer::ring_buffer(ring_buffer&& rhs) noexcept
: m_init{rhs.m_init}
, m_ptr{rhs.m_ptr}
, m_size{rhs.m_size}
, m_read_count{rhs.m_read_count.load()}
, m_write_count{rhs.m_write_count.load()}
{
    rhs.reset();
}

ring_buffer&
ring_buffer::operator=(ring_buffer&& rhs) noexcept
{
    if(this == &rhs) return *this;
    destroy();
    m_init        = rhs.m_init;
    m_ptr         = rhs.m_ptr;
    m_size        = rhs.m_size;
    m_read_count  = rhs.m_read_count.load();
    m_write_count = rhs.m_write_count.load();
    rhs.reset();
    return *this;
}

void
ring_buffer::init(size_t _size)
{
    ROCP_FATAL_IF(m_init)
        << "rocprofiler::common::container::base::ring_buffer::init(size_t) :: already initialized";

    m_init = true;

    // Round up to multiple of page size.
    _size += units::get_page_size() - ((_size % units::get_page_size() > 0)
                                           ? (_size % units::get_page_size())
                                           : units::get_page_size());

    if((_size % units::get_page_size()) > 0)
    {
        std::ostringstream _oss{};
        ROCP_FATAL << fmt::format("Error! size is not a multiple of page size: {} % {} = {}",
                                  _size,
                                  units::get_page_size(),
                                  (_size % units::get_page_size()));
    }

    m_size        = _size;
    m_read_count  = 0;
    m_write_count = 0;

    // Map twice the buffer size.
    if((m_ptr =
            mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) ==
       MAP_FAILED)
    {
        destroy();
        auto _err = errno;
        ROCP_FATAL << fmt::format("mmap failed with errno {} :: {}", _err, strerror(_err));
    }
}

void
ring_buffer::destroy()
{
    if(m_ptr && m_init)
    {
        // Unmap the mapped virtual memmory.
        auto ret = munmap(m_ptr, m_size);
        if(ret != 0) perror("ring_buffer: munmap failed");
    }
    m_init        = false;
    m_size        = 0;
    m_read_count  = 0;
    m_write_count = 0;
    m_ptr         = nullptr;
}

std::string
ring_buffer::as_string() const
{
    std::ostringstream ss{};
    ss << std::boolalpha << "is_initialized: " << is_initialized() << ", capacity: " << capacity()
       << ", count: " << count() << ", free: " << free() << ", is_empty: " << is_empty()
       << ", is_full: " << is_full() << ", pointer: " << m_ptr << ", read count: " << m_read_count
       << ", write count: " << m_write_count;
    return ss.str();
}
//

void*
ring_buffer::request(size_t _length, size_t _align, bool _wrap)
{
    if(m_ptr == nullptr || m_size == 0) return nullptr;

    if(is_full()) return (_wrap) ? retrieve(_length, _align) : nullptr;

    LOG_IF(FATAL, _align == 0) << "alignment must be non-zero";

    // if write count is at the tail of buffer, bump to the end of buffer
    size_t _write_count = 0;
    size_t _offset      = 0;
    size_t _write_pos   = 0;
    do
    {
        // Make sure we don't put in more than there's room for, by writing no
        // more than there is free.
        if(_length > free()) return nullptr;
        _offset      = 0;
        _write_count = m_write_count.load(std::memory_order_acquire);
        auto _modulo = m_size - (_write_count % m_size);
        if(_modulo < _length) _offset = _modulo;
        auto _align_modulo = (_write_count % _align);
        auto _align_offset = (_align_modulo > 0) ? (_align - _align_modulo) : 0;
        _write_pos         = _write_count + _align_offset;
    } while(!m_write_count.compare_exchange_strong(
        _write_count, _write_pos + _length + _offset, std::memory_order_seq_cst));

    // pointer in buffer
    void* _out = write_ptr(_write_pos);

    return _out;
}
//

void*
ring_buffer::retrieve(size_t _length, size_t _align) const
{
    if(m_ptr == nullptr || m_size == 0) return nullptr;

    // Make sure we don't put in more than there's room for, by writing no
    // more than there is free.

    // if read count is at the tail of buffer, bump to the end of buffer
    size_t _read_count = 0;
    size_t _offset     = 0;
    size_t _read_pos   = 0;
    do
    {
        if(_length > count()) return nullptr;
        _offset      = 0;
        _read_count  = m_read_count.load(std::memory_order_acquire);
        auto _modulo = m_size - (_read_count % m_size);
        if(_modulo < _length) _offset = _modulo;
        auto _align_modulo = (_read_count % _align);
        auto _align_offset = (_align_modulo > 0) ? (_align - _align_modulo) : 0;
        _read_pos          = _read_count + _align_offset;
    } while(!m_read_count.compare_exchange_strong(
        _read_count, _read_pos + _length + _offset, std::memory_order_seq_cst));

    // pointer in buffer
    void* _out = read_ptr(_read_pos);

    return _out;
}
//

void
ring_buffer::reset()
{
    m_init = false;
    m_size = 0;
    m_ptr  = nullptr;
    m_read_count.store(0);
    m_write_count.store(0);
}
//

void
ring_buffer::save(std::fstream& _fs)
{
    auto _read_count  = m_read_count.load();
    auto _write_count = m_write_count.load();
    _fs.write(reinterpret_cast<char*>(&m_size), sizeof(m_size));
    _fs.write(reinterpret_cast<char*>(&_read_count), sizeof(_read_count));
    _fs.write(reinterpret_cast<char*>(&_write_count), sizeof(_write_count));
    _fs.write(reinterpret_cast<char*>(m_ptr), m_size * sizeof(char));
}
//

void
ring_buffer::load(std::fstream& _fs)
{
    destroy();

    size_t _read_count  = 0;
    size_t _write_count = 0;
    size_t _size        = 0;

    _fs.read(reinterpret_cast<char*>(&_size), sizeof(_size));

    init(_size);

    if(!m_ptr) throw std::bad_alloc{};

    _fs.read(reinterpret_cast<char*>(&_read_count), sizeof(_read_count));
    _fs.read(reinterpret_cast<char*>(&_write_count), sizeof(_write_count));
    _fs.read(reinterpret_cast<char*>(m_ptr), m_size * sizeof(char));

    m_read_count.store(_read_count, std::memory_order_release);
    m_write_count.store(_write_count, std::memory_order_release);
}

bool
ring_buffer::can_clear() const
{
    auto _read_count = m_read_count.load(std::memory_order_acquire);
    return (_read_count == 0);
}

bool
ring_buffer::clear()
{
    ROCP_CI_LOG_IF(WARNING, !can_clear())
        << "ring_buffer does not permit invoking clear() member function when the read pointer is "
           "non-zero because this introduces thread-safety issues";

    m_write_count.store(0, std::memory_order_release);
    return true;
}

bool ring_buffer::clear(std::nothrow_t)
{
    if(!can_clear()) return false;

    m_write_count.store(0, std::memory_order_release);
    return true;
}
}  // namespace base
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
