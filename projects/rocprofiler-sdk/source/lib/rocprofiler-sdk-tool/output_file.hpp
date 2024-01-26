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
#include "lib/rocprofiler-sdk-tool/csv.hpp"

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
std::pair<std::ostream*, void (*)(std::ostream*&)>
get_output_stream(const std::string& fname, const std::string& ext = ".csv");

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
        return ((m_stream) ? *m_stream : std::cerr) << std::forward<T>(value) << std::flush;
    }

    operator bool() const { return m_stream != nullptr; }

private:
    using stream_dtor_t = void (*)(std::ostream*&);

    const std::string m_name   = {};
    std::mutex        m_mutex  = {};
    std::ostream*     m_stream = nullptr;
    stream_dtor_t     m_dtor   = [](std::ostream*&) {};
};

template <size_t N>
output_file::output_file(std::string                       name,
                         csv::csv_encoder<N>               encoder,
                         std::array<std::string_view, N>&& header)
: m_name{std::move(name)}
{
    std::tie(m_stream, m_dtor) = get_output_stream(m_name);

    for(auto& itr : header)
    {
        LOG_IF(FATAL, itr.empty())
            << "CSV file for " << m_name << " was not provided the correct number of headers";
    }

    // write the csv header
    if(m_stream) encoder.write_row(*m_stream, header);
}
}  // namespace tool
}  // namespace rocprofiler
