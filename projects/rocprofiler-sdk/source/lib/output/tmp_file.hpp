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

#include <atomic>
#include <fstream>
#include <ios>
#include <mutex>
#include <set>
#include <string>
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

    explicit operator bool() const;

    template <typename Type>
    size_t write(const Type* data, size_t num_records)
    {
        // Assert we are not mixing types with tool_counter_value_t
        static_assert(sizeof(Type) == 16);
        size_t allocated = offset.fetch_add(num_records);

        std::unique_lock<std::mutex> lk(file_mutex);
        if(!stream.is_open()) open();
        stream.seekp(allocated * sizeof(Type));
        stream.write((char*) data, num_records * sizeof(Type));
        return allocated;
    };

    template <typename Type>
    std::vector<Type> read(size_t seekpos, size_t num_elements)
    {
        // Assert we are not mixing types with tool_counter_value_t
        static_assert(sizeof(Type) == 16);

        std::vector<Type> ret;
        ret.resize(num_elements);

        std::unique_lock<std::mutex> lk(file_mutex);
        if(!stream.is_open()) open();

        stream.seekg(seekpos * sizeof(Type));
        stream.read((char*) ret.data(), num_elements * sizeof(Type));
        return ret;
    }

    std::atomic<size_t> offset{0};

    std::string              filename     = {};
    std::string              subdirectory = {};
    std::fstream             stream       = {};
    FILE*                    file         = nullptr;
    int                      fd           = -1;
    std::set<std::streampos> file_pos     = {};
    std::mutex               file_mutex   = {};
};
