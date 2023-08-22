// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include "atomic_ring_buffer.hpp"
#include "lib/common/units.hpp"
#include "lib/common/environment.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <sys/mman.h>

namespace rocprofiler
{
namespace common
{
namespace container
{
namespace base
{
atomic_ring_buffer::atomic_ring_buffer(size_t _size, bool _use_mmap)
{
    set_use_mmap(_use_mmap);
    init(_size);
}

atomic_ring_buffer::~atomic_ring_buffer() { destroy(); }

atomic_ring_buffer::atomic_ring_buffer(const atomic_ring_buffer& rhs)
: m_use_mmap{rhs.m_use_mmap}
, m_use_mmap_explicit{rhs.m_use_mmap_explicit}
{
    init(rhs.m_size);
}

atomic_ring_buffer::atomic_ring_buffer(atomic_ring_buffer&& rhs) noexcept
: m_init{rhs.m_init}
, m_use_mmap{rhs.m_use_mmap}
, m_use_mmap_explicit{rhs.m_use_mmap_explicit}
, m_ptr{rhs.m_ptr}
, m_size{rhs.m_size}
, m_read_count{rhs.m_read_count.load()}
, m_write_count{rhs.m_write_count.load()}
{
    rhs.reset();
}

atomic_ring_buffer&
atomic_ring_buffer::operator=(const atomic_ring_buffer& rhs)
{
    if(this == &rhs) return *this;
    destroy();
    m_use_mmap          = rhs.m_use_mmap;
    m_use_mmap_explicit = rhs.m_use_mmap_explicit;
    init(rhs.m_size);
    return *this;
}

atomic_ring_buffer&
atomic_ring_buffer::operator=(atomic_ring_buffer&& rhs) noexcept
{
    if(this == &rhs) return *this;
    destroy();
    m_init              = rhs.m_init;
    m_use_mmap          = rhs.m_use_mmap;
    m_use_mmap_explicit = rhs.m_use_mmap_explicit;
    m_ptr               = rhs.m_ptr;
    m_size              = rhs.m_size;
    m_read_count        = rhs.m_read_count.load();
    m_write_count       = rhs.m_write_count.load();
    rhs.reset();
    return *this;
}

void
atomic_ring_buffer::init(size_t _size)
{
    if(m_init)
        throw std::runtime_error(
            "tim::base::atomic_ring_buffer::init(size_t) :: already initialized");

    m_init = true;

    // Round up to multiple of page size.
    _size += units::get_page_size() - ((_size % units::get_page_size() > 0)
                                           ? (_size % units::get_page_size())
                                           : units::get_page_size());

    if((_size % units::get_page_size()) > 0)
    {
        std::ostringstream _oss{};
        _oss << "Error! size is not a multiple of page size: " << _size << " % "
             << units::get_page_size() << " = " << (_size % units::get_page_size());
        throw std::runtime_error(_oss.str());
    }

    m_size        = _size;
    m_read_count  = 0;
    m_write_count = 0;

    if(!m_use_mmap_explicit) m_use_mmap = get_env("ROCPROFILER_USE_MMAP", m_use_mmap);

    if(!m_use_mmap)
    {
        m_ptr = malloc(m_size * sizeof(char));
        return;
    }

    // Map twice the buffer size.
    if((m_ptr =
            mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) ==
       MAP_FAILED)
    {
        destroy();
        auto _err = errno;
        // TIMEMORY_PRINTF_FATAL(stderr, "Error using mmap: %s\n", strerror(_err));
        throw std::runtime_error(strerror(_err));
    }
}

void
atomic_ring_buffer::destroy()
{
    if(m_ptr && m_init)
    {
        if(!m_use_mmap)
        {
            ::free(m_ptr);
        }
        else
        {
            // Unmap the mapped virtual memmory.
            auto ret = munmap(m_ptr, m_size);
            if(ret != 0) perror("munmap");
        }
    }
    m_init        = false;
    m_size        = 0;
    m_read_count  = 0;
    m_write_count = 0;
    m_ptr         = nullptr;
}

void
atomic_ring_buffer::set_use_mmap(bool _v)
{
    if(m_init)
        throw std::runtime_error("tim::base::atomic_ring_buffer::set_use_mmap(bool) cannot be "
                                 "called after initialization");
    m_use_mmap          = _v;
    m_use_mmap_explicit = true;
}

std::string
atomic_ring_buffer::as_string() const
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
atomic_ring_buffer::request(size_t _length)
{
    if(m_ptr == nullptr || m_size == 0) return nullptr;

    if(is_full()) return retrieve(_length);

    // if write count is at the tail of buffer, bump to the end of buffer
    size_t _write_count = 0;
    size_t _offset      = 0;
    do
    {
        // Make sure we don't put in more than there's room for, by writing no
        // more than there is free.
        if(_length > free()) return nullptr;

        _offset      = 0;
        _write_count = m_write_count.load();
        auto _modulo = m_size - (_write_count % m_size);
        if(_modulo < _length) _offset = _modulo;
    } while(!m_write_count.compare_exchange_strong(
        _write_count, _write_count + _length + _offset, std::memory_order_seq_cst));

    // pointer in buffer
    void* _out = write_ptr(_write_count);

    return _out;
}
//

void*
atomic_ring_buffer::retrieve(size_t _length) const
{
    if(m_ptr == nullptr || m_size == 0) return nullptr;

    // Make sure we don't put in more than there's room for, by writing no
    // more than there is free.

    // if read count is at the tail of buffer, bump to the end of buffer
    size_t _read_count = 0;
    size_t _offset     = 0;
    do
    {
        if(_length > count()) return nullptr;
        _offset      = 0;
        _read_count  = m_read_count.load();
        auto _modulo = m_size - (_read_count % m_size);
        if(_modulo < _length) _offset = _modulo;
    } while(!m_read_count.compare_exchange_strong(
        _read_count, _read_count + _length + _offset, std::memory_order_seq_cst));

    // pointer in buffer
    void* _out = read_ptr(_read_count);

    return _out;
}
//

void
atomic_ring_buffer::reset()
{
    m_init = false;
    m_size = 0;
    m_ptr  = nullptr;
    m_read_count.store(0);
    m_write_count.store(0);
}
//

void
atomic_ring_buffer::save(std::fstream& _fs)
{
    auto _read_count  = m_read_count.load();
    auto _write_count = m_write_count.load();
    _fs.write(reinterpret_cast<char*>(&m_use_mmap), sizeof(m_use_mmap));
    _fs.write(reinterpret_cast<char*>(&m_use_mmap_explicit), sizeof(m_use_mmap_explicit));
    _fs.write(reinterpret_cast<char*>(&m_size), sizeof(m_size));
    _fs.write(reinterpret_cast<char*>(&_read_count), sizeof(_read_count));
    _fs.write(reinterpret_cast<char*>(&_write_count), sizeof(_write_count));
    _fs.write(reinterpret_cast<char*>(m_ptr), m_size * sizeof(char));
}
//

void
atomic_ring_buffer::load(std::fstream& _fs)
{
    destroy();
    size_t _read_count  = 0;
    size_t _write_count = 0;

    _fs.read(reinterpret_cast<char*>(&m_use_mmap), sizeof(m_use_mmap));
    _fs.read(reinterpret_cast<char*>(&m_use_mmap_explicit), sizeof(m_use_mmap_explicit));
    _fs.read(reinterpret_cast<char*>(&m_size), sizeof(m_size));

    init(m_size);
    if(!m_ptr) m_ptr = malloc(m_size);

    _fs.read(reinterpret_cast<char*>(&_read_count), sizeof(_read_count));
    _fs.read(reinterpret_cast<char*>(&_write_count), sizeof(_write_count));
    _fs.read(reinterpret_cast<char*>(m_ptr), m_size * sizeof(char));

    m_read_count.store(_read_count);
    m_write_count.store(_write_count);
}
}  // namespace base
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
