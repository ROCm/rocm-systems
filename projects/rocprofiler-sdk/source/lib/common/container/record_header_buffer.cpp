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

#include "lib/common/container/record_header_buffer.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <algorithm>
#include <atomic>
#include <new>

namespace rocprofiler::common::container
{
namespace
{
// record_header_buffer RAII locker
struct rhb_raii_lock
{
    explicit rhb_raii_lock(record_header_buffer& _rhb)
    : m_rhb{_rhb}
    {
        m_rhb.lock();
    }

    ~rhb_raii_lock() { m_rhb.unlock(); }

    record_header_buffer& m_rhb;
};
}  // namespace

record_header_buffer::record_header_buffer(size_t num_bytes) { allocate(num_bytes); }

record_header_buffer::record_header_buffer(record_header_buffer&& _rhs) noexcept
{
    this->operator=(std::move(_rhs));
}

record_header_buffer&
record_header_buffer::operator=(record_header_buffer&& _rhs) noexcept
{
    if(this != &_rhs)
    {
        auto _lk  = rhb_raii_lock{_rhs};
        m_index   = _rhs.m_index.load(std::memory_order_acquire);
        m_buffer  = std::move(_rhs.m_buffer);
        m_headers = std::move(_rhs.m_headers);
        _rhs.reset();
    }
    return *this;
}

bool
record_header_buffer::allocate(size_t num_bytes)
{
    if(m_buffer.is_initialized()) return false;

    auto _lk = rhb_raii_lock{*this};
    m_buffer.init(num_bytes);
    rocprofiler_record_header_t record = {};
    record.hash                        = 0;
    record.payload                     = nullptr;
    m_headers.resize(m_buffer.capacity(), record);
    return true;
}

size_t
record_header_buffer::get_num_record_headers()
{
    auto _lk = rhb_raii_lock{*this};

    auto   _size = m_index.load(std::memory_order_acquire);
    size_t _ret  = 0;
    for(size_t i = 0; i < _size; ++i)
    {
        if(auto& itr = m_headers.at(i); itr.hash > 0 && itr.payload != nullptr) ++_ret;
    }

    return _ret;
}

size_t
record_header_buffer::clear()
{
    auto _lk = rhb_raii_lock{*this};

    auto _n = m_index.load(std::memory_order_acquire);
    {
        auto _sz = m_buffer.capacity();
        if(!m_buffer.clear(std::nothrow_t{})) return 0;
        std::for_each(m_headers.begin(), m_headers.end(), [](auto& itr) {
            rocprofiler_record_header_t record = {};
            record.hash                        = 0;
            record.payload                     = nullptr;
            itr                                = record;
        });
        rocprofiler_record_header_t record = {};
        record.hash                        = 0;
        record.payload                     = nullptr;
        m_headers.resize(_sz, record);
        m_index.store(0, std::memory_order_release);
    }

    return _n;
}

size_t
record_header_buffer::reset()
{
    auto _lk = rhb_raii_lock{*this};

    auto _n = m_index.load(std::memory_order_acquire);
    m_buffer.destroy();
    m_buffer.clear();
    m_headers.clear();
    m_index.store(0, std::memory_order_release);

    return _n;
}

void
record_header_buffer::save(std::fstream& _fs)
{
    auto _lk = rhb_raii_lock{*this};

    auto _idx = m_index.load(std::memory_order_acquire);
    auto _sz  = m_headers.size();
    _fs.write(reinterpret_cast<char*>(&_idx), sizeof(_idx));
    _fs.write(reinterpret_cast<char*>(&_sz), sizeof(_sz));
    _fs.write(reinterpret_cast<char*>(m_headers.data()), sizeof(rocprofiler_record_header_t) * _sz);
    m_buffer.save(_fs);
}

void
record_header_buffer::load(std::fstream& _fs)
{
    auto _lk = rhb_raii_lock{*this};

    {
        auto _idx = size_t{0};
        _fs.read(reinterpret_cast<char*>(&_idx), sizeof(_idx));
        m_index.store(_idx, std::memory_order_release);
    }

    {
        auto _sz = size_t{0};
        _fs.read(reinterpret_cast<char*>(&_sz), sizeof(_sz));
        m_headers.resize(_sz);
        _fs.read(reinterpret_cast<char*>(m_headers.data()),
                 sizeof(rocprofiler_record_header_t) * _sz);
    }

    m_buffer.load(_fs);
}
}  // namespace rocprofiler::common::container
