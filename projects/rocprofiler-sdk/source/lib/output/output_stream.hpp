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

#include "lib/common/filesystem.hpp"
#include "output_config.hpp"

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

struct output_stream
{
    output_stream() = default;
    output_stream(std::ostream* _os, ostream_dtor_t _dtor)
    : stream{_os}
    , dtor{_dtor}
    {}

    ~output_stream() { close(); }
    output_stream(const output_stream&)     = delete;
    output_stream(output_stream&&) noexcept = default;
    output_stream& operator=(const output_stream&) = delete;
    output_stream& operator=(output_stream&&) noexcept = default;

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
get_output_filename(const output_config& cfg, std::string_view fname, std::string_view ext);

output_stream
get_output_stream(const output_config& cfg,
                  std::string_view     fname,
                  std::string_view     ext,
                  std::ios::openmode   mode = {});
}  // namespace tool
}  // namespace rocprofiler
