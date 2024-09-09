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

#include "domain_type.hpp"
#include "helper.hpp"
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

template <typename Tp>
using ring_buffer_t = rocprofiler::common::container::ring_buffer<Tp>;

std::string
compose_tmp_file_name(domain_type buffer_type);

template <typename Tp>
std::tuple<Tp*, tmp_file*>
get_tmp_file_buffer(domain_type type)
{
    static Tp*       _buffer   = new Tp(rocprofiler::common::units::get_page_size());
    static tmp_file* _tmp_file = new tmp_file(compose_tmp_file_name(type));
    return std::tuple(_buffer, _tmp_file);
}

template <typename Tp>
void
offload_buffer(domain_type type)
{
    auto [_tmp_buf, _tmp_file]            = get_tmp_file_buffer<Tp>(type);
    auto                         _lk      = std::lock_guard<std::mutex>(_tmp_file->file_mutex);
    [[maybe_unused]] static auto _success = _tmp_file->open();
    auto&                        _fs      = _tmp_file->stream;
    _tmp_file->file_pos.emplace(_fs.tellg());
    _tmp_buf->save(_fs);
    _tmp_buf->clear();
    CHECK(_tmp_buf->is_empty() == true);
}

template <typename Tp>
void
write_ring_buffer(Tp _v, domain_type type)
{
    auto [_tmp_buf, _tmp_file] = get_tmp_file_buffer<ring_buffer_t<Tp>>(type);

    if(_tmp_buf->capacity() == 0)
    {
        ROCP_INFO << "rocprofv3 is dropping record from domain " << get_domain_column_name(type)
                  << ". Buffer has a capacity of zero.";
        return;
    }

    auto* ptr = _tmp_buf->request(false);
    if(ptr == nullptr)
    {
        offload_buffer<ring_buffer_t<Tp>>(type);
        ptr = _tmp_buf->request(false);

        // if failed, try again
        if(!ptr) ptr = _tmp_buf->request(false);

        // after second failure, emit warning message
        ROCP_CI_LOG_IF(WARNING, !ptr)
            << "rocprofv3 is dropping record from domain " << get_domain_column_name(type)
            << ". No space in buffer: "
            << fmt::format(
                   "capacity={}, record_size={}, used_count={}, free_count={} | raw_info=[{}]",
                   _tmp_buf->capacity(),
                   _tmp_buf->data_size(),
                   _tmp_buf->count(),
                   _tmp_buf->free(),
                   _tmp_buf->as_string());
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
    auto [_tmp_buf, _tmp_file] = get_tmp_file_buffer<Tp>(type);
    if(!_tmp_buf->is_empty()) offload_buffer<Tp>(type);
}

template <typename Tp>
std::deque<Tp>
read_tmp_file(domain_type type)
{
    auto _data = std::deque<Tp>{};

    auto [_tmp_buf, _tmp_file] = get_tmp_file_buffer<Tp>(type);
    auto  _lk                  = std::lock_guard<std::mutex>{_tmp_file->file_mutex};
    auto& _fs                  = _tmp_file->stream;
    if(_fs.is_open()) _fs.close();
    _tmp_file->open(std::ios::binary | std::ios::in);
    for(auto itr : _tmp_file->file_pos)
    {
        _fs.seekg(itr);  // set to the absolute position
        if(_fs.eof()) break;
        Tp _buffer;
        _buffer.load(_fs);
        _data.emplace_back(std::move(_buffer));
    }

    return _data;
}
