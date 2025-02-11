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

#include "tmp_file_buffer.hpp"

#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <iosfwd>
#include <mutex>
#include <set>
#include <vector>

namespace rocprofiler
{
namespace tool
{
/// converts a container of ring buffers of element Tp into a single container of elements
template <typename Tp, template <typename, typename...> class ContainerT, typename... ParamsT>
ContainerT<Tp>
get_buffer_elements(ContainerT<common::container::ring_buffer<Tp>, ParamsT...>&& data)
{
    auto ret = ContainerT<Tp>{};
    for(auto& buf : data)
    {
        Tp* record = nullptr;
        do
        {
            record = buf.retrieve();
            if(record) ret.emplace_back(*record);
        } while(record != nullptr);
    }

    return ret;
}

template <typename Tp>
std::vector<Tp>
get_buffer_elements(common::container::ring_buffer<Tp>&& buf)
{
    auto ret    = std::vector<Tp>{};
    Tp*  record = nullptr;
    ret.reserve(buf.count());
    do
    {
        record = buf.retrieve();
        if(record) ret.emplace_back(*record);
    } while(record != nullptr);

    return ret;
}

template <typename Tp, domain_type DomainT>
struct buffered_output;

template <typename Tp>
struct generator
{
    template <typename Up, domain_type DomainT>
    friend struct buffered_output;

    generator()  = delete;
    ~generator() = default;

    generator(const generator&) = delete;
    generator(generator&&)      = delete;
    generator& operator=(const generator&) = delete;
    generator& operator=(generator&&) = delete;

    auto begin() { return file_pos.begin(); }
    auto begin() const { return file_pos.begin(); }
    auto cbegin() const { return file_pos.cbegin(); }

    auto end() { return file_pos.end(); }
    auto end() const { return file_pos.end(); }
    auto cend() const { return file_pos.cend(); }

    auto size() const { return file_pos.size(); }
    auto empty() const { return file_pos.empty(); }

    std::vector<Tp> get(std::streampos itr) const;

private:
    generator(file_buffer<Tp>* fbuf);

    file_buffer<Tp>*            filebuf = nullptr;
    std::lock_guard<std::mutex> lk_guard;
    std::set<std::streampos>    file_pos = {};
};

template <typename Tp>
generator<Tp>::generator(file_buffer<Tp>* fbuf)
: filebuf{fbuf}
, lk_guard{filebuf->file.file_mutex}
, file_pos{filebuf->file.file_pos}
{}

template <typename Tp>
std::vector<Tp>
generator<Tp>::get(std::streampos itr) const
{
    auto  _data = std::vector<Tp>{};
    auto& _fs   = filebuf->file.stream;
    _fs.seekg(itr);  // set to the absolute position
    if(!_fs.eof())
    {
        auto _buffer = ring_buffer_t<Tp>{};
        _buffer.load(_fs);
        _data = get_buffer_elements(std::move(_buffer));
    }
    return _data;
}
}  // namespace tool
}  // namespace rocprofiler

namespace cereal
{
template <typename ArchiveT, typename Tp>
void
save(ArchiveT& ar, const rocprofiler::tool::generator<Tp>& data)
{
    ar.makeArray();
    for(auto itr : data)
    {
        auto dat = data.get(itr);
        for(const auto& ditr : dat)
            ar(ditr);
    }
}
}  // namespace cereal
