// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "buffer_storage.hpp"
#include "sample_type.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <rocprofiler-systems/categories.h>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{
using postprocessing_callback = std::function<void(const storage_parsed_type_base&)>;
class cache_manager;
class storage_parser
{
public:
    void register_type_callback(const entry_type&              type,
                                const postprocessing_callback& callback);

    void consume_storage();
    void register_on_finished_callback(std::unique_ptr<std::function<void()>> callback);

private:
    friend class cache_manager;
    storage_parser(std::string _filename);

    template <typename T>
    static void process_arg(const uint8_t*& data_pos, T& arg)
    {
        if constexpr(std::is_same_v<T, std::string>)
        {
            arg = std::string((const char*) data_pos);
            data_pos += arg.size() + 1;
        }
        else if constexpr(std::is_same_v<T, std::vector<uint8_t>>)
        {
            size_t vector_size = *reinterpret_cast<const size_t*>(data_pos);
            data_pos += sizeof(size_t);
            arg.reserve(vector_size);
            std::copy_n(data_pos, vector_size, std::back_inserter(arg));
            data_pos += vector_size;
        }
        else
        {
            arg = *reinterpret_cast<const T*>(data_pos);
            data_pos += sizeof(T);
        }
    }

    template <typename... Args>
    static void parse_data(const uint8_t* data_pos, Args&... args)
    {
        (process_arg(data_pos, args), ...);
    }

    void invoke_callbacks(entry_type type, const storage_parsed_type_base& parsed);

    std::string                                                m_filename;
    std::map<entry_type, std::vector<postprocessing_callback>> m_callbacks;
    std::unique_ptr<std::function<void()>> m_on_finished_callback{ nullptr };
};

}  // namespace trace_cache
}  // namespace rocprofsys
