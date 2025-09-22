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

#include "domain_type.hpp"
#include "output_config.hpp"
#include "tmp_file.hpp"

#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/units.hpp"

#include <fmt/format.h>

#include <deque>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace tool
{
template <typename Tp>
using ring_buffer_t = rocprofiler::common::container::ring_buffer<Tp>;

using tmp_file_name_callback_t = std::function<std::string(domain_type)>;

std::string
compose_tmp_file_name(const output_config& cfg, domain_type buffer_type);

tmp_file_name_callback_t&
get_tmp_file_name_callback();

template <typename Tp>
struct file_buffer
{
    file_buffer() = delete;
    file_buffer(domain_type _domain)
    : domain{_domain}
    , buffer{16 * static_cast<uint64_t>(::rocprofiler::common::units::get_page_size())}
    , file{get_tmp_file_name_callback()(_domain)}
    {}

    ~file_buffer()                      = default;
    file_buffer(const file_buffer&)     = delete;
    file_buffer(file_buffer&&) noexcept = default;
    file_buffer& operator=(const file_buffer&) = delete;
    file_buffer& operator=(file_buffer&&) noexcept = default;

    domain_type       domain = {};
    uint64_t          nbytes = 0;
    ring_buffer_t<Tp> buffer = {};
    tmp_file          file;
};

template <typename Tp>
struct file_buffer<ring_buffer_t<Tp>>
{
    static_assert(std::is_void<Tp>::value && std::is_empty<Tp>::value,
                  "error! instantiated with ring_buffer_t<Tp> instead of Tp");
};

template <typename Tp>
file_buffer<Tp>*&
get_tmp_file_buffer(domain_type type)
{
    static file_buffer<Tp>* val = new file_buffer<Tp>{type};
    return val;
}

template <typename Tp>
void
offload_buffer(domain_type type)
{
    auto* filebuf = get_tmp_file_buffer<Tp>(type);

    if(!filebuf)
    {
        ROCP_CI_LOG(WARNING) << "rocprofv3 cannot offload buffer for "
                             << get_domain_column_name(type) << ". Buffer has been destroyed.";
        return;
    }

    auto                         _lk      = std::lock_guard<std::mutex>(filebuf->file.file_mutex);
    [[maybe_unused]] static auto _success = filebuf->file.open();
    auto&                        _fs      = filebuf->file.stream;

    ROCP_CI_LOG_IF(WARNING, _fs.tellg() != _fs.tellp())  // this should always be true
        << "tellg=" << _fs.tellg() << ", tellp=" << _fs.tellp();

    auto _nbytes = (filebuf->buffer.count() * filebuf->buffer.data_size());

    ROCP_TRACE << fmt::format(
        "offloading {} B from {} buffer to tmp file", _nbytes, get_domain_column_name(type));

    filebuf->file.file_pos.emplace(_fs.tellp());
    filebuf->nbytes += _nbytes;
    filebuf->buffer.save(_fs);
    filebuf->buffer.clear();

    ROCP_CI_LOG_IF(ERROR, !filebuf->buffer.is_empty())
        << "buffer is not empty after offload: count=" << filebuf->buffer.count();
}

template <typename Tp>
void
write_ring_buffer(Tp _v, domain_type type)
{
    auto* filebuf = get_tmp_file_buffer<Tp>(type);

    if(!filebuf)
    {
        ROCP_CI_LOG(WARNING) << "rocprofv3 is dropping record from domain "
                             << get_domain_column_name(type) << ". Buffer has been destroyed.";
        return;
    }
    else if(filebuf->buffer.capacity() == 0)
    {
        ROCP_CI_LOG(WARNING) << "rocprofv3 is dropping record from domain "
                             << get_domain_column_name(type) << ". Buffer has a capacity of zero.";
        return;
    }

    auto* ptr = filebuf->buffer.request(false);
    if(ptr == nullptr)
    {
        offload_buffer<Tp>(type);
        ptr = filebuf->buffer.request(false);

        // if failed, try again
        if(!ptr) ptr = filebuf->buffer.request(false);

        // after second failure, emit warning message
        ROCP_CI_LOG_IF(WARNING, !ptr)
            << "rocprofv3 is dropping record from domain " << get_domain_column_name(type)
            << ". No space in buffer: "
            << fmt::format(
                   "capacity={}, record_size={}, used_count={}, free_count={} | raw_info=[{}]",
                   filebuf->buffer.capacity(),
                   filebuf->buffer.data_size(),
                   filebuf->buffer.count(),
                   filebuf->buffer.free(),
                   filebuf->buffer.as_string());
    }

    if(ptr)
    {
        if constexpr(std::is_move_constructible<Tp>::value)
        {
            new(ptr) Tp{std::move(_v)};
        }
        else if constexpr(std::is_move_assignable<Tp>::value)
        {
            *ptr = std::move(_v);
        }
        else if constexpr(std::is_copy_constructible<Tp>::value)
        {
            new(ptr) Tp{_v};
        }
        else if constexpr(std::is_copy_assignable<Tp>::value)
        {
            *ptr = _v;
        }
        else
        {
            static_assert(std::is_void<Tp>::value,
                          "data type is neither move/copy constructible nor move/copy assignable");
        }
    }
}

template <typename Tp>
void
flush_tmp_buffer(domain_type type)
{
    auto* filebuf = get_tmp_file_buffer<Tp>(type);
    if(filebuf && !filebuf->buffer.is_empty()) offload_buffer<Tp>(type);
}

template <typename Tp>
void
read_tmp_file(domain_type type)
{
    auto* filebuf = get_tmp_file_buffer<Tp>(type);

    if(!filebuf)
    {
        ROCP_CI_LOG(WARNING) << "rocprofv3 cannot read tmp file for "
                             << get_domain_column_name(type) << ". Buffer has been destroyed.";
        return;
    }

    auto _lk = std::lock_guard<std::mutex>{filebuf->file.file_mutex};
    if(filebuf->file.exists())
    {
        auto& _fs = filebuf->file.stream;
        if(_fs.is_open()) _fs.close();
        filebuf->file.open(std::ios::binary | std::ios::in);
    }
}
}  // namespace tool
}  // namespace rocprofiler
