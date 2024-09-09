// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "helper.hpp"
#include "statistics.hpp"
#include "tmp_file_buffer.hpp"

#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

namespace rocprofiler
{
namespace tool
{
using float_type   = double;
using stats_data_t = statistics<uint64_t, float_type>;

template <typename Tp, domain_type DomainT>
struct buffered_output
{
    using ring_buffer_type              = rocprofiler::common::container::ring_buffer<Tp>;
    static constexpr auto buffer_type_v = DomainT;

    explicit buffered_output(bool _enabled);
    ~buffered_output()                          = default;
    buffered_output(const buffered_output&)     = delete;
    buffered_output(buffered_output&&) noexcept = delete;
    buffered_output& operator=(const buffered_output&) = default;
    buffered_output& operator=(buffered_output&&) noexcept = default;

    void flush();
    void read();
    void clear();
    void destroy();

    operator bool() const { return enabled; }

    std::deque<Tp> element_data = {};
    stats_entry_t  stats        = {};

private:
    bool enabled = false;
};

template <typename Tp, domain_type DomainT>
buffered_output<Tp, DomainT>::buffered_output(bool _enabled)
: enabled{_enabled}
{}

template <typename Tp, domain_type DomainT>
void
buffered_output<Tp, DomainT>::flush()
{
    if(!enabled) return;

    flush_tmp_buffer<ring_buffer_type>(buffer_type_v);
}

template <typename Tp, domain_type DomainT>
void
buffered_output<Tp, DomainT>::read()
{
    if(!enabled) return;

    flush();

    element_data = get_buffer_elements(read_tmp_file<ring_buffer_type>(buffer_type_v));
}

template <typename Tp, domain_type DomainT>
void
buffered_output<Tp, DomainT>::clear()
{
    if(!enabled) return;

    element_data.clear();
}

template <typename Tp, domain_type DomainT>
void
buffered_output<Tp, DomainT>::destroy()
{
    if(!enabled) return;

    clear();
    auto [_tmp_buf, _tmp_file] = get_tmp_file_buffer<ring_buffer_type>(buffer_type_v);
    _tmp_buf->destroy();
    delete _tmp_buf;
    delete _tmp_file;
}
}  // namespace tool
}  // namespace rocprofiler
