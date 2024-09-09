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

#include "config.hpp"
#include "csv.hpp"

#include "lib/common/filesystem.hpp"

#include <array>
#include <iostream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace tool
{
using ostream_dtor_t = void (*)(std::ostream*&);

using output_stream_pair_t = std::pair<std::ostream*, ostream_dtor_t>;

struct output_stream_t
{
    output_stream_t() = default;
    output_stream_t(std::ostream* _os, ostream_dtor_t _dtor)
    : stream{_os}
    , dtor{_dtor}
    {}

    ~output_stream_t() { close(); }
    output_stream_t(const output_stream_t&)     = delete;
    output_stream_t(output_stream_t&&) noexcept = default;
    output_stream_t& operator=(const output_stream_t&) = delete;
    output_stream_t& operator=(output_stream_t&&) noexcept = default;

    explicit operator bool() const { return stream != nullptr; }

    template <typename Tp>
    std::ostream& operator<<(Tp&& value)
    {
        return ((stream) ? *stream : std::cerr) << std::forward<Tp>(value) << std::flush;
    }

    void close()
    {
        if(stream) (*stream) << std::flush;
        if(dtor) dtor(stream);
    }

    bool writes_to_file() const { return (dynamic_cast<std::ofstream*>(stream) != nullptr); }

    std::ostream*  stream = nullptr;
    ostream_dtor_t dtor   = nullptr;
};

std::string
get_output_filename(std::string_view fname, std::string_view ext);

output_stream_t
get_output_stream(std::string_view fname, std::string_view ext);

struct output_file
{
    template <size_t N>
    output_file(std::string name, csv::csv_encoder<N>, std::array<std::string_view, N>&& header);

    ~output_file();

    output_file(const output_file&) = delete;
    output_file& operator=(const output_file&) = delete;

    std::string name() const { return m_name; }

    template <typename T>
    std::ostream& operator<<(T&& value)
    {
        auto _lk = std::unique_lock<std::mutex>{m_mutex};
        return ((m_os.stream) ? *m_os.stream : std::cerr) << std::forward<T>(value) << std::flush;
    }

    operator bool() const { return m_os.stream != nullptr; }

private:
    const std::string m_name  = {};
    std::mutex        m_mutex = {};
    output_stream_t   m_os    = {};
};

template <size_t N>
output_file::output_file(std::string                       name,
                         csv::csv_encoder<N>               encoder,
                         std::array<std::string_view, N>&& header)
: m_name{std::move(name)}
, m_os{get_output_stream(m_name, ".csv")}
{
    for(auto& itr : header)
    {
        ROCP_FATAL_IF(itr.empty())
            << "CSV file for " << m_name << " was not provided the correct number of headers";
    }

    // write the csv header
    if(m_os.stream) encoder.write_row(*m_os.stream, header);
}
}  // namespace tool
}  // namespace rocprofiler
