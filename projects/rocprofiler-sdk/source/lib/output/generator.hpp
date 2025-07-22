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
#include "lib/common/demangle.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <iosfwd>
#include <mutex>
#include <numeric>
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

struct defer_size
{};

template <typename Tp>
struct generator
{
    generator()          = delete;
    virtual ~generator() = default;

    generator(const generator&)     = delete;
    generator(generator&&) noexcept = delete;
    generator& operator=(const generator&) = delete;
    generator& operator=(generator&&) noexcept = delete;

    auto begin() { return m_pos.begin(); }
    auto begin() const { return m_pos.begin(); }
    auto cbegin() const { return m_pos.cbegin(); }

    auto end() { return m_pos.end(); }
    auto end() const { return m_pos.end(); }
    auto cend() const { return m_pos.cend(); }

    auto size() const { return m_pos.size(); }
    auto empty() const { return m_pos.empty(); }

    virtual std::vector<Tp> get(size_t) const = 0;

protected:
    explicit generator(defer_size);
    explicit generator(size_t sz);

    void resize(size_t sz);

    std::vector<size_t> m_pos = {};
};

template <typename Tp>
generator<Tp>::generator(defer_size)
{}

template <typename Tp>
generator<Tp>::generator(size_t sz)
{
    resize(sz);
}

template <typename Tp>
void
generator<Tp>::resize(size_t sz)
{
    m_pos.resize(sz, 0);
    std::iota(m_pos.begin(), m_pos.end(), 0);
}

template <typename Tp>
struct file_generator : public generator<Tp>
{
    template <typename Up, domain_type DomainT>
    friend struct buffered_output;

    file_generator()           = delete;
    ~file_generator() override = default;

    file_generator(const file_generator&)     = delete;
    file_generator(file_generator&&) noexcept = delete;
    file_generator& operator=(const file_generator&) = delete;
    file_generator& operator=(file_generator&&) noexcept = delete;

    std::vector<Tp> get(size_t itr) const override;

private:
    explicit file_generator(file_buffer<Tp>* fbuf);

    file_buffer<Tp>*            filebuf = nullptr;
    std::lock_guard<std::mutex> lk_guard;
    std::set<std::streampos>    file_pos = {};
};

template <typename Tp>
file_generator<Tp>::file_generator(file_buffer<Tp>* fbuf)
: generator<Tp>{fbuf->file.file_pos.size()}
, filebuf{fbuf}
, lk_guard{filebuf->file.file_mutex}
, file_pos{filebuf->file.file_pos}
{}

template <typename Tp>
std::vector<Tp>
file_generator<Tp>::get(size_t idx) const
{
    auto  _data = std::vector<Tp>{};
    auto& _fs   = filebuf->file.stream;

    if(idx >= file_pos.size())
    {
        ROCP_ERROR << fmt::format("file_generator has no file position at index {}", idx);
        return _data;
    }

    auto itr = file_pos.begin();
    std::advance(itr, idx);

    if(itr == file_pos.end())
    {
        ROCP_ERROR << fmt::format("file_generator at index {} == end of file_pos set", idx);
        return _data;
    }

    ROCP_TRACE << fmt::format("file_generator file position at index={} :: ", idx) << *itr;

    _fs.seekg(*itr);  // set to the absolute position
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
namespace tool = ::rocprofiler::tool;

template <typename ArchiveT, typename Tp>
void
save(ArchiveT& ar, const tool::file_generator<Tp>& data)
{
    ar.makeArray();
    for(auto itr : data)
    {
        auto dat = data.get(itr);
        for(const auto& ditr : dat)
            ar(ditr);
    }
}

template <typename ArchiveT, typename Tp>
void
save(ArchiveT& ar, const tool::generator<Tp>& data)
{
    if(const auto* fdata = dynamic_cast<const tool::file_generator<Tp>*>(&data); fdata != nullptr)
    {
        save(ar, *fdata);
    }
    else
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "dynamic_cast failed for {}",
            rocprofiler::common::cxx_demangle(typeid(tool::generator<Tp>).name()));
    }
}
}  // namespace cereal
