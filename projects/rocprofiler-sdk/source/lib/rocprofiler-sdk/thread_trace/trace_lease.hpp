// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <memory>
#include <utility>

namespace rocprofiler
{
namespace thread_trace
{
struct trace_usage_t;

class trace_lease_t
{
public:
    trace_lease_t() = default;
    trace_lease_t(std::shared_ptr<trace_usage_t> usage, bool is_device)
    : m_usage(std::move(usage))
    , m_is_device(is_device)
    {}

    ~trace_lease_t() { reset(); }

    trace_lease_t(trace_lease_t&& other) noexcept
    : m_usage(std::move(other.m_usage))
    , m_is_device(std::exchange(other.m_is_device, false))
    {}

    trace_lease_t& operator=(trace_lease_t&& other) noexcept
    {
        if(this != &other)
        {
            reset();
            m_usage     = std::move(other.m_usage);
            m_is_device = std::exchange(other.m_is_device, false);
        }
        return *this;
    }

    trace_lease_t(const trace_lease_t&) = delete;
    trace_lease_t& operator=(const trace_lease_t&) = delete;

    explicit operator bool() const noexcept { return m_usage != nullptr; }

    friend bool operator==(const trace_lease_t& lhs, std::nullptr_t) noexcept
    {
        return lhs.m_usage == nullptr;
    }
    friend bool operator!=(const trace_lease_t& lhs, std::nullptr_t) noexcept
    {
        return lhs.m_usage != nullptr;
    }

    void reset();

private:
    std::shared_ptr<trace_usage_t> m_usage     = {};
    bool                           m_is_device = false;
};

}  // namespace thread_trace
}  // namespace rocprofiler
