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

#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <fstream>
#include <ios>
#include <mutex>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

struct tmp_file
{
    tmp_file(std::string _filename);
    ~tmp_file();

    bool open(std::ios::openmode = std::ios::binary | std::ios::in | std::ios::out);
    bool fopen(const char* = "r+");
    bool flush();
    bool close();
    bool remove();
    bool exists() const;

    explicit operator bool() const;

    template <typename Tp>
    std::streampos write(const Tp* data, size_t num_records);

    template <typename Tp>
    std::streampos write(const Tp& data);

    template <typename Tp>
    std::vector<Tp> read(std::streampos seekpos);

    std::string              filename     = {};
    std::string              subdirectory = {};
    std::fstream             stream       = {};
    FILE*                    file         = nullptr;
    int                      fd           = -1;
    std::set<std::streampos> file_pos     = {};
    std::mutex               file_mutex   = {};
};

template <typename Tp>
std::streampos
tmp_file::write(const Tp* data, size_t num_records)
{
    auto lk = std::unique_lock<std::mutex>{file_mutex};

    if(!stream.is_open()) open();
    ROCP_CI_LOG_IF(WARNING, stream.tellg() != stream.tellp())  // this should always be true
        << "tellg=" << stream.tellg() << ", tellp=" << stream.tellp();

    auto pos = stream.tellp();
    stream.write(reinterpret_cast<const char*>(&num_records), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(data), num_records * sizeof(Tp));
    return pos;
}

template <typename Tp>
std::streampos
tmp_file::write(const Tp& data)
{
    static_assert(std::is_standard_layout<Tp>::value, "only supports standard layout types");
    static_assert(!std::is_pointer<Tp>::value, "only supports non-pointer types");

    auto lk = std::unique_lock<std::mutex>{file_mutex};

    if(!stream.is_open()) open();
    ROCP_CI_LOG_IF(WARNING, stream.tellg() != stream.tellp())
        << fmt::format("tellg={}, tellp={}", stream.tellg(), stream.tellp());

    auto   pos         = stream.tellp();
    size_t num_records = 1;
    stream.write(reinterpret_cast<const char*>(&num_records), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(&data), num_records * sizeof(Tp));
    return pos;
}

template <typename Tp>
std::vector<Tp>
tmp_file::read(std::streampos seekpos)
{
    auto lk = std::unique_lock<std::mutex>{file_mutex};
    if(!stream.is_open()) open();

    stream.seekg(seekpos);
    size_t num_elements = 0;
    stream.read(reinterpret_cast<char*>(&num_elements), sizeof(size_t));

    auto ret = std::vector<Tp>{};
    ret.resize(num_elements, Tp{});
    stream.read(reinterpret_cast<char*>(ret.data()), num_elements * sizeof(Tp));

    return ret;
}
