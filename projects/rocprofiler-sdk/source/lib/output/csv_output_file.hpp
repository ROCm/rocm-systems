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

#include "csv.hpp"
#include "domain_type.hpp"
#include "output_stream.hpp"

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
struct csv_output_file
{
    template <size_t N>
    csv_output_file(const output_config& cfg,
                    std::string_view     name,
                    csv::csv_encoder<N>,
                    std::array<std::string_view, N>&& header);

    template <size_t N>
    csv_output_file(const output_config& cfg,
                    domain_type          domain,
                    csv::csv_encoder<N>,
                    std::array<std::string_view, N>&& header);

    ~csv_output_file();

    csv_output_file(const csv_output_file&) = delete;
    csv_output_file& operator=(const csv_output_file&) = delete;

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
    output_stream     m_os    = {};
};

template <size_t N>
csv_output_file::csv_output_file(const output_config&              cfg,
                                 std::string_view                  name,
                                 csv::csv_encoder<N>               encoder,
                                 std::array<std::string_view, N>&& header)
: m_name{std::string{name}}
, m_os{get_output_stream(cfg, m_name, ".csv")}
{
    for(auto& itr : header)
    {
        ROCP_FATAL_IF(itr.empty())
            << "CSV file for " << m_name << " was not provided the correct number of headers";
    }

    // write the csv header
    if(m_os.stream) encoder.write_row(*m_os.stream, header);
}

template <size_t N>
csv_output_file::csv_output_file(const output_config&              cfg,
                                 domain_type                       domain,
                                 csv::csv_encoder<N>               encoder,
                                 std::array<std::string_view, N>&& header)
: csv_output_file{cfg, get_domain_trace_file_name(domain), encoder, std::move(header)}
{}
}  // namespace tool
}  // namespace rocprofiler
