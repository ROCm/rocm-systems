// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "lib/common/defines.hpp"
#include "lib/common/logging.hpp"
#include "rocprofiler-sdk/cxx/utility.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rocprofiler
{
namespace common
{
namespace container
{
template <typename Tp>
struct pool;

template <typename Tp>
struct pool_object
{
    using pool_type = pool<Tp>;

    pool_object(size_t idx, bool in_use, pool_type* pool)
    : m_in_use{in_use}
    , m_index{idx}
    , m_pool{pool}
    {}

    pool_object()                       = default;
    ~pool_object()                      = default;
    pool_object(pool_object&&) noexcept = default;
    pool_object& operator=(pool_object&&) noexcept = default;

    // pool_object(const pool_object& rhs) = delete;
    // pool_object& operator=(const pool_object& rhs) = delete;

    pool_object(const pool_object& rhs)
    : m_object{rhs.m_object}
    , m_in_use{rhs.m_in_use.load(std::memory_order_relaxed)}
    , m_index{rhs.m_index}
    , m_pool{rhs.m_pool}
    {}

    pool_object& operator=(const pool_object& rhs)
    {
        if(this != &rhs)
        {
            m_object = rhs.m_object;
            m_in_use.store(rhs.m_in_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
            m_index = rhs.m_index;
            m_pool  = rhs.m_pool;
        }
        return *this;
    }

    bool acquire();
    bool release();
    bool in_use() const { return m_in_use.load(std::memory_order_relaxed); }

    Tp&       get() { return m_object; }
    const Tp& get() const { return m_object; }

    auto index() const { return m_index; }
    auto index(size_t index) { m_index = index; }

private:
    Tp                m_object = {};
    std::atomic<bool> m_in_use = false;
    size_t            m_index  = 0;
    pool_type*        m_pool   = nullptr;
};

template <typename Tp>
bool
pool_object<Tp>::acquire()
{
    bool expected = false;
    return m_in_use.compare_exchange_strong(expected, true);
}

template <typename Tp>
bool
pool_object<Tp>::release()
{
    bool expected = true;
    auto val      = m_in_use.compare_exchange_strong(expected, false);

    if(val && m_pool) m_pool->release(m_index);

    return val;
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
